"""Tests of attention aggregation and physically meaningful paired differences."""

import numpy as np
import pytest


def test_attention_groups_use_sums_valid_counts_and_safe_zero_groups():
    from openpi.policies.aft_diagnostics import summarize_attention
    trace = dict(
        attention=np.array([[[0.1, 0.2, 0.3, 0.4, 0.0]]], np.float32),
        key_group_id=np.array([0, 0, 6, 8, 3]),
        key_valid=np.array([True, True, True, True, False]),
    )
    result = summarize_attention(trace)
    np.testing.assert_allclose(result["modality_mass"][0, 0, [0, 6, 8]], [0.3, 0.3, 0.4])
    np.testing.assert_allclose(result["modality_per_token"][0, 0, [0, 6, 8]], [0.15, 0.3, 0.4])
    assert result["valid_key_counts"][3] == 0
    assert result["modality_mass"][0, 0, 3] == 0
    assert np.isfinite(result["modality_per_token"]).all()


def test_action_difference_uses_so3_not_rotvec_subtraction():
    from openpi.policies.aft_diagnostics import prediction_difference
    base = dict(actions=np.array([[0, 0, 0, 0, 0, np.pi - 0.01, 0.02]]),
                tactile_shear=np.zeros((1, 198, 2)), wrist_wrench=np.zeros((1, 6)))
    altered = dict(actions=np.array([[0.003, 0.004, 0, 0, 0, -np.pi + 0.01, 0.021]]),
                   tactile_shear=np.ones((1, 198, 2)), wrist_wrench=np.array([[3, 4, 0, 0, 0, 2]]))
    delta = prediction_difference(base, altered)
    np.testing.assert_allclose(delta["position_delta_mm"], [5.0])
    np.testing.assert_allclose(delta["rotation_delta_deg"], [1.1459155902616465])
    np.testing.assert_allclose(delta["gripper_delta_mm"], [1.0])
    np.testing.assert_allclose(delta["force_delta_norm_N"], [5.0])
    np.testing.assert_allclose(delta["torque_delta_norm_Nm"], [2.0])
    np.testing.assert_allclose(delta["tactile_delta_rms"], [1.0])


def test_diagnostic_options_reject_invalid_sampling():
    from openpi.policies.aft_diagnostics import DiagnosticsOptions
    for kwargs in ({"every": 0}, {"query_stride": 0}, {"denoise_step": -2}, {"layers": (-1,)},
                   {"layers": (1, 1)}):
        with pytest.raises(ValueError):
            DiagnosticsOptions(**kwargs)


def test_bundle_roundtrip_and_no_overwrite(tmp_path):
    from openpi.policies.aft_diagnostics import load_bundle, save_bundle
    response = dict(actions=np.zeros((2, 7), np.float32),
                    diagnostics=dict(schema="aft_diagnostics_v1", attention=np.ones((1, 6, 1), np.float32),
                                     layer_indices=[3], ablations={}))
    prefix = save_bundle(response, tmp_path, "sample_001", context={"plan_id": "p1", "capture_time": 123.4})
    loaded, context = load_bundle(prefix)
    np.testing.assert_array_equal(loaded["diagnostics"]["attention"], response["diagnostics"]["attention"])
    assert loaded["diagnostics"]["layer_indices"] == [3]
    assert context == {"plan_id": "p1", "capture_time": 123.4}
    with pytest.raises(FileExistsError):
        save_bundle(response, tmp_path, "sample_001")
    with pytest.raises(ValueError):
        save_bundle(response, tmp_path, "../escape")


def test_bundle_failure_does_not_publish_orphan_and_allows_retry(tmp_path, monkeypatch):
    from pathlib import Path
    from openpi.policies.aft_diagnostics import save_bundle, load_bundle
    original_open = Path.open

    def fail_json(path, *args, **kwargs):
        if path.suffix == ".json" and args and "x" in args[0]:
            raise OSError("simulated full disk")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_json)
    with pytest.raises(OSError):
        save_bundle(dict(diagnostics={"schema": "aft_diagnostics_v1", "attention": np.ones((1, 1, 1))}),
                    tmp_path, "retry")
    assert not (tmp_path / "retry.npz").exists()
    assert not (tmp_path / "retry.json").exists()
    monkeypatch.setattr(Path, "open", original_open)
    prefix = save_bundle(dict(diagnostics={"schema": "aft_diagnostics_v1"}), tmp_path, "retry")
    assert load_bundle(prefix)[0]["diagnostics"]["schema"] == "aft_diagnostics_v1"
