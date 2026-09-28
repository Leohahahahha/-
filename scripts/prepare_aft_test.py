import numpy as np
import pytest

from scripts.prepare_aft import collect_statistics
from openpi.training.aft_data_test import rows


def test_statistics_exclude_validation_and_padding():
    data = rows(4)
    stats = collect_statistics({0: data}, {0: np.ones(3, bool)}, horizon=3, action_offset=1)
    assert stats["wrench"].mean[0] == 1.5
    assert stats["shear"].mean[0] == 1.5
    assert stats["actions"].mean.shape == (7,)


def test_empty_train_rejected():
    with pytest.raises(ValueError):
        collect_statistics({}, {}, horizon=3, action_offset=1)


def test_preparation_does_not_overwrite_assets(tmp_path):
    from scripts.prepare_aft import main

    (tmp_path / "sentinel").write_text("keep")
    with pytest.raises(FileExistsError):
        main("aft_pi0_next_state_full", str(tmp_path))
    assert (tmp_path / "sentinel").read_text() == "keep"


def test_audit_target_counts_do_not_cross_segment_boundaries():
    from scripts.prepare_aft import window_target_counts

    edges = np.array([True, False, True])
    assert window_target_counts(edges, horizon=50, action_offset=1) == (2, 2)
    assert window_target_counts(edges, horizon=50, action_offset=0) == (6, 2)
    assert window_target_counts(np.ones(3, bool), horizon=2, action_offset=1) == (5, 5)
