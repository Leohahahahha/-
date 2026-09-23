import numpy as np

from scripts.eval_aft_offline import physical_errors


def test_rotation_branches_and_masks():
    a = np.zeros((2, 7))
    b = a.copy()
    a[:, 3] = np.pi
    b[:, 3] = -np.pi
    result = physical_errors(
        a,
        b,
        np.ones((2, 396)),
        np.zeros((2, 396)),
        np.ones((2, 6)),
        np.zeros((2, 6)),
        np.array([True, False]),
        np.array([True, False]),
    )
    assert result["rotation_deg"][0] < 1e-10
    assert result["shear_rmse"][0] == 1.0
    assert np.isnan(result["shear_rmse"][1])
