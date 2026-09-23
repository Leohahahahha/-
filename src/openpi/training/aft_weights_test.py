import numpy as np
import pytest
import jax.numpy as jnp

from openpi.training.weight_loaders import merge_aft_sources


def test_sources_and_strict_shapes():
    init = {
        "PaliGemma": {"w": np.zeros(2)},
        "tactile_prefix_encoder": {"w": np.zeros(2)},
        "force_history_in": {"w": np.zeros(2)},
    }
    backbone = {"PaliGemma": {"w": np.ones(2)}}
    tactile = {"tactile_prefix_encoder": {"w": np.full(2, 2.0)}}
    result = merge_aft_sources(init, backbone, tactile)
    np.testing.assert_array_equal(result["PaliGemma"]["w"], 1.0)
    np.testing.assert_array_equal(result["tactile_prefix_encoder"]["w"], 2.0)
    np.testing.assert_array_equal(result["force_history_in"]["w"], 0.0)
    with pytest.raises(ValueError, match="shape"):
        merge_aft_sources(init, {"PaliGemma": {"w": np.zeros(3)}}, tactile)
    with pytest.raises(ValueError, match="missing"):
        merge_aft_sources(init, {}, tactile)


def test_partial_lora_never_silently_filled():
    init = {"PaliGemma": {"w": np.ones(2), "lora_a": np.zeros(2), "lora_b": np.zeros(2)}}
    with pytest.raises(ValueError, match="missing"):
        merge_aft_sources(init, {"PaliGemma": {"w": np.ones(2), "lora_a": np.ones(2)}}, {})
    loaded = merge_aft_sources(init, {"PaliGemma": {"w": np.ones(2)}}, {}, allow_new_lora=True)
    np.testing.assert_array_equal(loaded["PaliGemma"]["lora_b"], 0.0)


def test_restored_storage_dtype_matches_reference():
    result = merge_aft_sources(
        {"PaliGemma": {"w": jnp.zeros(2, jnp.bfloat16)}}, {"PaliGemma": {"w": np.ones(2, np.float32)}}, {}
    )
    assert result["PaliGemma"]["w"].dtype == jnp.bfloat16
