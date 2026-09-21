import numpy as np
import pytest


def rows(n=10):
    marker = np.zeros((n, 9, 198, 2), np.float32)
    marker[:, -1, :, :] = np.arange(n)[:, None, None]
    return dict(state=np.zeros((n, 7), np.float32), actions=np.zeros((n, 7), np.float32),
                tactile_marker_motion=marker, wrist_wrench=np.repeat(np.arange(n, dtype=np.float32)[:, None], 6, axis=1),
                episode_index=np.zeros(n, np.int64))


def test_history_future_and_terminal_masks():
    from openpi.training.aft_data import build_episode_windows
    w = build_episode_windows(rows(), 2, contiguous_edges=np.ones(9, bool))
    np.testing.assert_array_equal(w['force_history'][:, 0], [0, 0, 0, 0, 0, 0, 1, 2])
    assert w['targets'].wrench[0, 0] == 3
    assert w['targets'].shear[0, 0] == 3
    assert w['targets'].sensor_mask.sum() == 7
    assert w['targets'].action_mask.sum() == 8
    assert not w['targets'].sensor_mask[-1]


def test_gap_masks_all_targets_across_gap():
    from openpi.training.aft_data import build_episode_windows
    edges = np.ones(9, bool)
    edges[3] = False
    w = build_episode_windows(rows(), 2, contiguous_edges=edges)
    np.testing.assert_array_equal(w['targets'].sensor_mask[:4], [True, False, False, False])


def test_reject_unknown_timing_cross_episode_and_bad_anchor():
    from openpi.training.aft_data import build_episode_windows
    with pytest.raises(ValueError, match='contiguous'):
        build_episode_windows(rows(), 2)
    r = rows()
    r['episode_index'][-1] = 1
    with pytest.raises(ValueError, match='episode'):
        build_episode_windows(r, 2, contiguous_edges=np.ones(9, bool))
    with pytest.raises(ValueError, match='anchor'):
        build_episode_windows(rows(), -1, contiguous_edges=np.ones(9, bool))


def test_nonfinite_valid_sensor_rejected():
    from openpi.training.aft_data import build_episode_windows
    r = rows()
    r['wrist_wrench'][1, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        build_episode_windows(r, 2, contiguous_edges=np.ones(9, bool))
