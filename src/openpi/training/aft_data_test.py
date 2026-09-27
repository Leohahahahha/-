import numpy as np
import pytest
import json


def rows(n=10):
    marker = np.zeros((n, 9, 198, 2), np.float32)
    marker[:, -1, :, :] = np.arange(n)[:, None, None]
    return dict(
        state=np.zeros((n, 7), np.float32),
        actions=np.zeros((n, 7), np.float32),
        tactile_marker_motion=marker,
        wrist_wrench=np.repeat(np.arange(n, dtype=np.float32)[:, None], 6, axis=1),
        episode_index=np.zeros(n, np.int64),
    )


def test_history_future_and_terminal_masks():
    from openpi.training.aft_data import build_episode_windows

    w = build_episode_windows(rows(), 2, contiguous_edges=np.ones(9, bool))
    np.testing.assert_array_equal(w["force_history"][:, 0], [0, 0, 0, 0, 0, 0, 1, 2])
    assert w["targets"].wrench[0, 0] == 3
    assert w["targets"].shear[0, 0] == 3
    assert w["targets"].sensor_mask.sum() == 7
    assert w["targets"].action_mask.sum() == 8
    assert not w["targets"].sensor_mask[-1]


def test_timing_audit_refuses_unknown_gap_locations():
    from openpi.training.aft_data import audit_edges

    report = {"timing_reports": [{"output_episode_index": 0, "frames": 10, "missing_candidate_steps": 1}]}
    with pytest.raises(ValueError, match="adjacency"):
        audit_edges(report, 0, 10)
    edges = np.ones(9, bool)
    edges[3] = False
    np.testing.assert_array_equal(audit_edges(report, 0, 10, {"0": edges.tolist()}), edges)


def test_timing_report_counts_source_terminal_observation():
    from openpi.training.aft_data import audit_edges

    report = {
        "terminal_frame_policy": "omit final source observation; use it only as the previous frame action target",
        "timing_reports": [{"output_episode_index": 0, "frames": 11, "missing_candidate_steps": 0}],
    }
    assert audit_edges(report, 0, 10).shape == (9,)


def test_gap_masks_all_targets_across_gap():
    from openpi.training.aft_data import build_episode_windows

    edges = np.ones(9, bool)
    edges[3] = False
    w = build_episode_windows(rows(), 2, contiguous_edges=edges)
    np.testing.assert_array_equal(w["targets"].sensor_mask[:4], [True, False, False, False])


def test_reject_unknown_timing_cross_episode_and_bad_anchor():
    from openpi.training.aft_data import build_episode_windows

    with pytest.raises(ValueError, match="contiguous"):
        build_episode_windows(rows(), 2)
    r = rows()
    r["episode_index"][-1] = 1
    with pytest.raises(ValueError, match="episode"):
        build_episode_windows(r, 2, contiguous_edges=np.ones(9, bool))
    with pytest.raises(ValueError, match="anchor"):
        build_episode_windows(rows(), -1, contiguous_edges=np.ones(9, bool))


def test_nonfinite_valid_sensor_rejected():
    from openpi.training.aft_data import build_episode_windows

    r = rows()
    r["wrist_wrench"][1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        build_episode_windows(r, 2, contiguous_edges=np.ones(9, bool))


def filtered_mapping(tmp_path, *, source_frames=(0, 1, 3, 4), timestamps=(0.0, 0.1, 0.3, 0.4)):
    mapping_dir = tmp_path / "meta/source_mapping"
    mapping_dir.mkdir(parents=True)
    mapping = {
        "source_episode_index": 0,
        "rows": [
            dict(
                output_frame_index=i,
                source_row_index=frame,
                source_frame_index=frame,
                source_timestamp=time,
                action_source_row_index=frame + 1,
                action_source_frame_index=frame + 1,
                action_source_timestamp=time + 0.1,
            )
            for i, (frame, time) in enumerate(zip(source_frames, timestamps, strict=True))
        ],
    }
    (mapping_dir / "episode_000000.json").write_text(json.dumps(mapping))
    return mapping


def filtered_conversion(*, missing_steps=0, frames=6):
    return {
        "version": 4,
        "terminal_frame_policy": "omit final source observation; use it only as the previous frame action target",
        "timing_reports": [
            dict(
                output_episode_index=0,
                source_episode_index=0,
                frames=frames,
                removed_supervised_samples=1,
                missing_candidate_steps=missing_steps,
            )
        ],
    }


def test_filtered_mapping_breaks_removed_source_row_and_masks_windows(tmp_path):
    from openpi.training.aft_data import audit_episode_edges, build_episode_windows

    filtered_mapping(tmp_path)
    edges = audit_episode_edges(tmp_path, filtered_conversion(), 0, 4)
    np.testing.assert_array_equal(edges, [True, False, True])
    window = build_episode_windows(rows(4), 1, contiguous_edges=edges, action_state_step_offset=1)
    np.testing.assert_array_equal(window["targets"].action_mask[:3], [False, False, False])
    np.testing.assert_array_equal(window["targets"].sensor_mask[:3], [False, False, False])
    after_gap = build_episode_windows(rows(4), 2, contiguous_edges=edges, action_state_step_offset=1)
    np.testing.assert_array_equal(after_gap["force_history"][:, 0], [2] * 8)


def test_filtered_mapping_breaks_compacted_time_gap(tmp_path):
    from openpi.training.aft_data import audit_episode_edges

    filtered_mapping(tmp_path, source_frames=(0, 1, 2, 3))
    np.testing.assert_array_equal(
        audit_episode_edges(tmp_path, filtered_conversion(missing_steps=1), 0, 4),
        [True, False, True],
    )


def test_filtered_mapping_requires_valid_provenance(tmp_path):
    from openpi.training.aft_data import audit_episode_edges

    conversion = filtered_conversion()
    with pytest.raises(ValueError, match="source.mapping|mapping"):
        audit_episode_edges(tmp_path, conversion, 0, 4)
    older_filtered_report = {**conversion, "version": 3}
    with pytest.raises(ValueError, match="source.mapping|mapping"):
        audit_episode_edges(tmp_path, older_filtered_report, 0, 4)
    mapping = filtered_mapping(tmp_path)
    with pytest.raises(ValueError, match="report|frames"):
        audit_episode_edges(tmp_path, filtered_conversion(frames=7), 0, 4)
    mapping["rows"][1]["output_frame_index"] = 0
    (tmp_path / "meta/source_mapping/episode_000000.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="output_frame_index"):
        audit_episode_edges(tmp_path, conversion, 0, 4)
    mapping["rows"][1]["output_frame_index"] = 1
    mapping["rows"][1]["action_source_row_index"] = 0
    (tmp_path / "meta/source_mapping/episode_000000.json").write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match="action_source"):
        audit_episode_edges(tmp_path, conversion, 0, 4)


def test_filtered_mapping_rejects_untrue_supplied_adjacency(tmp_path):
    from openpi.training.aft_data import audit_episode_edges

    filtered_mapping(tmp_path)
    with pytest.raises(ValueError, match="adjacency"):
        audit_episode_edges(tmp_path, filtered_conversion(), 0, 4, {"0": [True, True, True]})
