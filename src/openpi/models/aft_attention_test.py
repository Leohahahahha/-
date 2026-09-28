"""Opt-in attention must not alter predictions or checkpoint parameter leaves."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.aft_config import AFTConfig
from openpi.shared.nnx_utils import module_jit


@pytest.mark.parametrize("pi05", [False, True])
def test_attention_replay_is_normalized_aligned_and_parameter_preserving(pi05):
    config = AFTConfig(
        pi05=pi05, paligemma_variant="dummy", action_expert_variant="dummy",
        tactile_width=32, force_width=16, vision_variant="mu/16", dtype="float32",
        action_horizon=3, max_token_len=2, image_resolution=(32, 32),
    )
    network = config.create(jax.random.key(0))
    obs = config.fake_obs()
    obs = obs.replace(tokenized_prompt_mask=jnp.array([[True, False]]))
    params_before = nnx.state(network, nnx.Param).flat_state()
    rng = jax.random.key(2)
    expected = network.sample_predictions(rng, obs, num_steps=2)
    inspect = module_jit(network.sample_attention, static_argnames=("num_steps", "denoise_step", "query_stride"))
    trace = inspect(rng, obs, num_steps=2, denoise_step=1, query_stride=2)
    attention = np.asarray(trace["attention"])
    assert attention.shape[:3] == (4, 1, 6)  # dummy depth=4, batch=1, 3 streams x 2 queries
    np.testing.assert_allclose(attention.sum(-1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(trace["query_stream_id"], [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(trace["query_horizon_index"], [0, 2, 0, 2, 0, 2])
    key_ids = np.asarray(trace["key_group_id"])
    assert np.count_nonzero(key_ids == 6) == 1  # compressed tactile history
    assert np.count_nonzero(key_ids == 8) == 1  # compressed force history
    assert np.count_nonzero(key_ids == 5) == 3  # action future
    invalid = ~np.asarray(trace["key_valid"])[0]
    assert invalid.any()
    np.testing.assert_array_equal(attention[..., invalid], 0.0)
    np.testing.assert_allclose(trace["velocity"]["actions"],
                               (trace["noisy"]["actions"] - expected["actions"]) * 2, atol=2e-5)
    params_after = nnx.state(network, nnx.Param).flat_state()
    assert params_before.keys() == params_after.keys()
    for path in params_before:
        np.testing.assert_array_equal(params_before[path].value, params_after[path].value)
    again = network.sample_predictions(rng, obs, num_steps=2)
    for key in expected:
        np.testing.assert_array_equal(again[key], expected[key])


def test_attention_replay_rejects_invalid_steps_before_forward():
    # An uninitialized instance suffices: validation must precede parameter access.
    from openpi.models.aft import AFTModel
    network = object.__new__(AFTModel)
    for kwargs in ({"num_steps": 0}, {"num_steps": 2, "denoise_step": 2}, {"query_stride": 0}):
        with pytest.raises(ValueError):
            network.sample_attention(None, None, **kwargs)
