import re
import jax.numpy as jnp
import flax.nnx as nnx

from openpi.training import config


def test_eight_registered_configs_and_sensor_precision():
    for backbone in ("pi0", "pi05"):
        for data in ("next_state", "sent_command"):
            for mode in ("full", "lora"):
                c = config.get_config(f"aft_{backbone}_{data}_{mode}")
                assert c.model.action_horizon == 50
                assert c.model.force_history_frames == 8
                assert c.model.tactile_prefix_diff_from_reference is False
                assert c.data.base_config.validation_episodes == (1, 7, 17)
                assert data in c.data.base_config.root
                assert c.ema_decay is None
                assert c.parameter_dtype_policy.gradient_dtype == "float32"
                for path in (
                    "PaliGemma/llm/layers/attn/q_einsum_2/w",
                    "PaliGemma/llm/layers/mlp_3/gating_einsum",
                    "tactile_prefix_encoder/blocks/block_0/kernels/kernel_0/kernel",
                    "force_expert/in_proj/kernel",
                ):
                    matched = [r.dtype for r in c.parameter_dtype_policy.overrides if re.fullmatch(r.path_regex, path)]
                    assert matched and matched[0] == "float32"


def test_frozen_norms_and_embedding_stay_fp32():
    from scripts.train import apply_parameter_dtype_policy

    c = config.get_config("aft_pi0_next_state_lora")
    state = nnx.State(
        {
            "PaliGemma": {
                "llm": {
                    "embedder": {"input_embedding": nnx.Param(jnp.ones(2, jnp.bfloat16)).to_state()},
                    "layers": {"pre_attention_norm": {"scale": nnx.Param(jnp.ones(2, jnp.bfloat16)).to_state()}},
                }
            }
        }
    )
    actual = apply_parameter_dtype_policy(state, c)
    assert actual["PaliGemma"]["llm"]["embedder"]["input_embedding"].value.dtype == jnp.float32
    assert actual["PaliGemma"]["llm"]["layers"]["pre_attention_norm"]["scale"].value.dtype == jnp.float32


def test_lora_freezes_pretrained_vision_not_new_experts():
    c = config.get_config("aft_pi0_next_state_lora")
    leaf = nnx.Param(jnp.ones(2)).to_state()
    assert c.freeze_filter(("PaliGemma", "img", "Transformer", "kernel"), leaf)
    assert not c.freeze_filter(("PaliGemma", "llm", "layers", "mlp_2", "gating_einsum"), leaf)
    assert not c.freeze_filter(("tactile_prefix_encoder", "out_proj", "kernel"), leaf)
