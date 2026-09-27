"""CPU-only AFT checkpoint contract tests; parameter restoration is mocked."""

import hashlib
import json

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
