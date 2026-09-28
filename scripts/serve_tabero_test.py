"""CPU-only AFT checkpoint contract tests; parameter restoration is mocked."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from openpi.shared import normalize
from openpi.training import config as configs
from openpi.training.aft_assets import model_contract
from scripts import serve_tabero

NAME = "aft_pi0_whiteboard_20260922_v2_next_state_full_30k"


def checkpoint_fixture(tmp_path, monkeypatch):
    config = configs.get_config(NAME)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    assets = checkpoint / "assets/aft"
    assets.mkdir(parents=True)
    stats = {
        key: normalize.NormStats(mean=np.zeros(dim), std=np.ones(dim))
        for key, dim in (("state", 7), ("actions", 7), ("tactile_prefix", 396), ("shear", 396), ("wrench", 6))
    }
    normalize.save(assets, stats)
    (assets / "adjacency.json").write_text("{}")
    conversion = tmp_path / "conversion.json"
    conversion.write_text(
        json.dumps(
            {
                "output_contract": "tabero_action_only_lerobot_v2.1",
                "marker_field": {"shape": [9, 198, 2], "history_length": 8, "side_order": ["left", "right"]},
            }
        )
    )
    manifest = {
        "config": NAME,
        "model_contract": model_contract(config.model),
        "source_root": config.data.base_config.root,
        "action_source_mode": "next-state",
        "train_episodes": list(config.data.base_config.episodes),
        "validation_episodes": list(config.data.base_config.validation_episodes),
        "norm_stats_sha256": hashlib.sha256((assets / "norm_stats.json").read_bytes()).hexdigest(),
        "adjacency_sha256": hashlib.sha256((assets / "adjacency.json").read_bytes()).hexdigest(),
        "conversion_sha256": hashlib.sha256(conversion.read_bytes()).hexdigest(),
    }
    (assets / "manifest.json").write_text(json.dumps(manifest))
    calls = []

    def fake_restore(cfg, path, **kwargs):
        calls.append((cfg, path, kwargs))
        return object()

    monkeypatch.setattr("openpi.policies.policy_config.create_trained_policy", fake_restore)
    monkeypatch.setattr(type(config.data), "create", lambda *_: pytest.fail("AFT serving opened training data"))
    return checkpoint, conversion, assets, calls


def test_aft_serving_uses_checkpoint_assets_without_training_mount(tmp_path, monkeypatch):
    checkpoint, conversion, _, calls = checkpoint_fixture(tmp_path, monkeypatch)
    policy, meta = serve_tabero.load_policy(NAME, checkpoint, conversion, 10)
    assert policy is not None
    assert len(calls) == 1
    assert calls[0][1] == checkpoint
    assert calls[0][2] == {"sample_kwargs": {"num_steps": 10}, "strict_params": True}
    assert meta["architecture"] == "aft"
    assert meta["prediction_layout"] == "separate_aft_actions_shear_wrench"
    assert (meta["predicts_wrench"], meta["predicts_tactile"], meta["use_tactile"]) == (True, True, True)
    assert (meta["action_horizon"], meta["force_history_frames"]) == (50, 8)
    assert meta["task_prompt"] == "Pick up the yellow whiteboard eraser and erase the X-shaped mark on the whiteboard."
    assert meta["wrist_wrench_frame"] == "K"
    assert meta["tactile_shear_shape"] == [50, 198, 2]
    assert meta["wrist_wrench_shape"] == [50, 6]


def test_aft_serving_rejects_conversion_manifest_and_stats_mismatch(tmp_path, monkeypatch):
    checkpoint, conversion, assets, _ = checkpoint_fixture(tmp_path, monkeypatch)
    conversion.write_text(conversion.read_text() + " ")
    with pytest.raises(ValueError, match="conversion"):
        serve_tabero.load_policy(NAME, checkpoint, conversion, 10)
    manifest_path = assets / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["conversion_sha256"] = hashlib.sha256(conversion.read_bytes()).hexdigest()
    manifest["config"] = "old-assembly"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="config"):
        serve_tabero.load_policy(NAME, checkpoint, conversion, 10)
    manifest["config"] = NAME
    manifest_path.write_text(json.dumps(manifest))
    stats = normalize.load(assets)
    del stats["shear"]
    normalize.save(assets, stats)
    manifest["norm_stats_sha256"] = hashlib.sha256((assets / "norm_stats.json").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shear"):
        serve_tabero.load_policy(NAME, checkpoint, conversion, 10)


def test_diagnostic_configuration_rejects_legacy_and_advertises_aft():
    from openpi.policies.aft_diagnostics import DiagnosticsOptions

    class Policy:
        def configure_diagnostics(self, options, *, output_dir=None, metadata=None):
            self.options = options
    policy = Policy()
    with pytest.raises(ValueError, match="AFT"):
        serve_tabero.configure_aft_diagnostics(policy, {}, DiagnosticsOptions())
    metadata = {"architecture": "aft"}
    serve_tabero.configure_aft_diagnostics(policy, metadata, DiagnosticsOptions(every=5, ablations=True))
    assert metadata["diagnostics"]["schema"] == "aft_diagnostics_v1"
    assert metadata["diagnostics"]["every"] == 5
    assert metadata["diagnostics"]["ablations"] is True


@pytest.mark.parametrize(
    ("name", "use_tactile", "predicts_wrench", "target_dim", "asset_id"),
    [
        ("pi0_lora_tacfield_local_tactile_lora_smoke", True, False, 7, "local/tabero_lerobot_compact_v1"),
        ("pi0_lora_tabero_v3_touch_20k", True, False, 7, "local/tabero_lerobot_compact_v3"),
        ("pi0_lora_tabero_rgb_state", False, False, 7, "local/tabero_lerobot_compact_v1"),
        (
            "pi0_lora_tabero_whiteboard_next_state_force_20k",
            True,
            True,
            13,
            "local/test2_tabero_next_state_compact",
        ),
    ],
)
def test_server_uses_moved_checkpoint_assets_and_strict_restore(
    monkeypatch, tmp_path, name, use_tactile, predicts_wrench, target_dim, asset_id
):
    from openpi import transforms
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config as configs

    checkpoint = tmp_path / "moved" / "2999"
    (checkpoint / "params").mkdir(parents=True)
    stats = {
        "state": transforms.NormStats(mean=np.zeros(7), std=np.ones(7)),
        "actions": transforms.NormStats(mean=np.zeros(target_dim), std=np.ones(target_dim)),
        "tactile_prefix": transforms.NormStats(mean=np.zeros(396), std=np.ones(396)),
    }
    normalize.save(checkpoint / "assets" / asset_id, stats)
    conversion = Path(__file__).parents[1] / "examples/fr3_deploy/tabero_conversion.json"
    monkeypatch.setattr(configs.ModelTransformFactory, "__call__", lambda *_: transforms.Group())
    calls = {}

    def create(config, directory, **kwargs):
        calls.update(config=config, directory=directory, kwargs=kwargs)
        return "mock_policy"

    monkeypatch.setattr(policy_config, "create_trained_policy", create)
    result, metadata = serve_tabero.load_policy(name, checkpoint, conversion, 10)
    assert result == "mock_policy"
    assert calls["directory"] == checkpoint
    assert calls["config"].data.assets.assets_dir == str(checkpoint / "assets")
    assert calls["kwargs"] == {"sample_kwargs": {"num_steps": 10}, "strict_params": True}
    assert metadata["use_tactile"] is use_tactile
    assert metadata["predicts_wrench"] is predicts_wrench
    assert metadata["asset_id"] == asset_id
    assert metadata["action_horizon"] == 50
    assert metadata["conversion_sha256"] == hashlib.sha256(conversion.read_bytes()).hexdigest()
    if use_tactile:
        assert metadata["tactile_marker_shape"] == [9, 198, 2]
        assert metadata["tactile_marker_dtype"] == "float32"
        assert metadata["tactile_marker_layout"] == "reference_then_8_history_frames_left_then_right"


def test_wrong_simulation_config_rejected_before_loading(monkeypatch, tmp_path):
    (tmp_path / "params").mkdir()
    with pytest.raises(ValueError, match="real-FR3"):
        serve_tabero.load_policy("pi0_lora_tacfield_tabero", tmp_path, tmp_path / "not-read.json", 10)


def test_server_reports_config_checkpoint_asset_mismatch(monkeypatch, tmp_path):
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training import config as configs

    checkpoint = tmp_path / "20000"
    (checkpoint / "params").mkdir(parents=True)
    stats = {
        "state": transforms.NormStats(mean=np.zeros(7), std=np.ones(7)),
        "actions": transforms.NormStats(mean=np.zeros(7), std=np.ones(7)),
        "tactile_prefix": transforms.NormStats(mean=np.zeros(396), std=np.ones(396)),
    }
    normalize.save(checkpoint / "assets/local/tabero_lerobot_compact_v3", stats)
    monkeypatch.setattr(configs.ModelTransformFactory, "__call__", lambda *_: transforms.Group())

    with pytest.raises(ValueError, match="compact_v1.*compact_v3.*exact training config"):
        serve_tabero.load_policy(
            "pi0_lora_tacfield_local_tactile_lora_smoke",
            checkpoint,
            Path(__file__).parents[1] / "examples/fr3_deploy/tabero_conversion.json",
            10,
        )
