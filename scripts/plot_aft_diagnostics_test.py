import numpy as np


def test_three_diagnostic_figures_from_safe_bundle(tmp_path):
    from openpi.policies.aft_diagnostics import save_bundle, summarize_attention
    from scripts.plot_aft_diagnostics import plot_bundle
    trace = dict(schema="aft_diagnostics_v1", attention=np.full((1, 6, 10), 0.1, np.float32),
                 key_group_id=np.arange(10), key_valid=np.ones(10, bool),
                 key_group_names=["front", "wrist", "other", "text", "state", "A", "Th", "T", "Fh", "F"],
                 query_stream_names=["action", "tactile", "force"],
                 query_stream_id=np.repeat(np.arange(3), 2), query_horizon_index=np.tile([0, 1], 3),
                 layer_indices=[17], denoise_step=9, num_denoise_steps=10, ablations={})
    trace.update(summarize_attention(trace))
    trace["ablations"]["mean_force_history"] = dict(position_delta_mm=np.array([1., 2.]),
                                                   rotation_delta_deg=np.array([0.1, 0.2]),
                                                   gripper_delta_mm=np.array([0., 0.]))
    prefix = save_bundle(dict(actions=np.zeros((2, 7)), diagnostics=trace), tmp_path / "data", "one")
    files = plot_bundle(prefix, tmp_path / "figures")
    assert len(files) == 3
    for path in files:
        assert path.read_bytes().startswith(b"\x89PNG")
