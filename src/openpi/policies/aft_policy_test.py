import numpy as np
import pytest
import jax

from openpi.models import model
from openpi.models.aft_types import AFTTargets
from openpi.policies import aft_policy
from openpi.policies.libero_policy import _tabero_marker_reference_grid
from openpi.shared.normalize import NormStats


def sample():
    return dict(
        image=np.zeros((16, 16, 3), np.uint8),
        wrist_image=np.zeros((16, 16, 3), np.uint8),
        state=np.zeros(7, np.float32),
        tactile_marker_motion=np.tile(_tabero_marker_reference_grid(), (9, 1, 1)),
        force_history=np.ones((8, 6), np.float32),
        force_history_mask=np.ones(8, bool),
        targets=AFTTargets(
            np.zeros((50, 7), np.float32),
            np.ones((50, 396), np.float32),
            np.ones((50, 6), np.float32),
            np.ones(50, bool),
            np.ones(50, bool),
        ),
    )


def stats():
    return {k: NormStats(mean=np.ones(d), std=np.full(d, 2.0)) for k, d in [("shear", 396), ("wrench", 6)]}


def test_inputs_keep_future_out_of_observation():
    data = aft_policy.AFTInputs(model.ModelType.PI0)(sample())
    batch = jax.tree.map(lambda x: np.stack([x]), data)
    obs = model.Observation.from_dict(batch)
    assert obs.force_history.shape == (1, 8, 6)
    assert obs.tactile_prefix.shape == (1, 9, 396)
    assert "targets" not in obs.to_dict()
    assert "shear" not in obs.to_dict()
    assert data["actions"].shape == (50, 7)


def test_sensor_normalization_shared_and_roundtrip():
    data = aft_policy.AFTInputs(model.ModelType.PI0)(sample())
    normalized = aft_policy.NormalizeSensors(stats())(data)
    np.testing.assert_array_equal(normalized["force_history"][0], normalized["wrench"][0])
    output = aft_policy.AFTOutputs(stats())(normalized)
    np.testing.assert_allclose(output["wrist_wrench"], 1.0)
    np.testing.assert_allclose(output["tactile_shear"], 1.0)
    assert output["actions"].shape == (50, 7)
    assert output["tactile_shear"].shape == (50, 198, 2)


def test_sensor_stats_only_selected_training_episodes():
    episodes = {
        0: dict(shear=np.zeros((2, 396)), wrench=np.ones((2, 6))),
        1: dict(shear=np.full((2, 396), 100.0), wrench=np.full((2, 6), 100.0)),
    }
    result = aft_policy.fit_sensor_stats(episodes, train_episodes=(0,))
    np.testing.assert_array_equal(result["wrench"].mean, np.ones(6))
    np.testing.assert_array_equal(result["shear"].mean, np.zeros(396))


def test_pi05_keeps_tcn_history_zscore():
    from openpi.shared.normalize import NormStats

    common = {
        key: NormStats(mean=np.zeros(dim), std=np.full(dim, 2.0), q01=np.full(dim, -10.0), q99=np.full(dim, 10.0))
        for key, dim in (("state", 7), ("actions", 7), ("tactile_prefix", 396))
    }
    normalized = aft_policy.NormalizeAFTCommon(common, use_quantiles=True)(
        {"state": np.full(7, 2.0), "actions": np.full((2, 7), 2.0), "tactile_prefix": np.full((9, 396), 2.0)}
    )
    np.testing.assert_allclose(normalized["state"], 0.2, atol=1e-6)
    np.testing.assert_allclose(normalized["tactile_prefix"], 1.0, atol=1e-6)


def test_invalid_input_and_output():
    data = sample()
    data["state"][6] = 1.0
    with pytest.raises(ValueError, match="gripper"):
        aft_policy.AFTInputs(model.ModelType.PI0)(data)
    data = aft_policy.AFTInputs(model.ModelType.PI0)(sample())
    data["wrench"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        aft_policy.AFTOutputs(stats())(data)


def test_synchronous_policy_returns_all_physical_modalities():
    import flax.nnx as nnx
    import jax.numpy as jnp
    from openpi import transforms

    class EchoSensors(nnx.Module):
        def sample_predictions(self, rng, obs, num_steps=1):
            return dict(
                actions=jnp.zeros((1, 50, 32)),
                shear=jnp.zeros((1, 50, 396)),
                wrench=jnp.broadcast_to(obs.force_history[:, -1:], (1, 50, 6)),
            )

    encode = transforms.compose(
        [
            aft_policy.AFTInputs(model.ModelType.PI0),
            aft_policy.NormalizeSensors(stats()),
            transforms.PadStatesAndActions(32),
        ]
    )
    policy = aft_policy.AFTPolicy(EchoSensors(), encode, aft_policy.AFTOutputs(stats()), sample_kwargs={"num_steps": 1})
    raw = sample()
    del raw["targets"]
    result = policy.infer(raw)
    assert result["actions"].shape == (50, 7)
    assert result["tactile_shear"].shape == (50, 198, 2)
    np.testing.assert_allclose(result["wrist_wrench"], 1.0)
