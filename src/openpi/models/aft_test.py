import jax
import jax.numpy as jnp
import numpy as np
import pytest
import flax.nnx as nnx
from openpi.shared.nnx_utils import PathRegex

from openpi.models.aft import make_aft_attention_mask, masked_mean
from openpi.models.aft_config import AFTConfig


def test_mask_and_mean():
    groups = jnp.array([0, 1, 2, 1, 2, 1, 2])
    mask = make_aft_attention_mask(jnp.ones((1, 7), bool), groups)
    assert not mask[0, 0, -1]
    assert mask[0, -1, 0] and mask[0, -1, 2]
    assert not mask[0, 1, 2]
    assert masked_mean(jnp.array([1.0, 9.0]), jnp.array([True, False])) == 1.0
    assert masked_mean(jnp.array([1.0, 9.0]), jnp.array([False, False])) == 0.0


@pytest.mark.parametrize("pi05", [False, True])
def test_small_model_three_predictions_and_loss(pi05):
    config = AFTConfig(
        pi05=pi05,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        tactile_width=32,
        force_width=16,
        vision_variant="mu/16",
        dtype="float32",
        action_horizon=3,
        max_token_len=2,
        image_resolution=(32, 32),
    )
    model = config.create(jax.random.key(0))
    obs, targets = config.fake_obs(), config.fake_act()
    loss, components = model.compute_loss(jax.random.key(1), obs, targets, return_components=True)
    assert np.isfinite(loss).all()
    assert set(components) == {"action_loss", "tactile_loss", "wrench_loss"}
    predictions = model.sample_predictions(jax.random.key(2), obs, num_steps=1)
    assert predictions["actions"].shape == (1, 3, 32)
    assert predictions["shear"].shape == (1, 3, 396)
    assert predictions["wrench"].shape == (1, 3, 6)
    masked = targets.replace(action_mask=jnp.zeros((1, 3), bool), sensor_mask=jnp.zeros((1, 3), bool))
    assert model.compute_loss(jax.random.key(1), obs, masked) == 0.0
    partial = targets.replace(
        action_mask=jnp.array([[True, False, False]]), sensor_mask=jnp.array([[True, False, False]])
    )
    changed = partial.replace(
        actions=partial.actions.at[:, 1:].set(1000.0),
        shear=partial.shear.at[:, 1:].set(1000.0),
        wrench=partial.wrench.at[:, 1:].set(1000.0),
    )
    np.testing.assert_allclose(
        model.compute_loss(jax.random.key(3), obs, partial),
        model.compute_loss(jax.random.key(3), obs, changed),
        rtol=1e-6,
        atol=1e-6,
    )
    _, grads = nnx.value_and_grad(
        lambda m: m.compute_loss(jax.random.key(1), obs, targets),
        argnums=nnx.DiffState(0, PathRegex(r"(tactile_expert|force_expert)/.*")),
    )(model)
    for branch in ("tactile_expert", "force_expert"):
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(grads[branch]))
