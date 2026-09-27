import re
import jax.numpy as jnp
import flax.nnx as nnx

from openpi.training import config
from openpi.models.aft_config import AFTConfig


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


def test_whiteboard_v2_aft_full_30k_config():
    name = "aft_pi0_whiteboard_20260922_v2_next_state_full_30k"
    c = config.get_config(name)
    assert isinstance(c.model, AFTConfig)
    assert (c.model.pi05, c.model.action_horizon, c.model.force_history_frames) == (False, 50, 8)
    assert c.data.base_config.root == "/data/yanghaojun/datasets/whiteboard_20260922_v2_tabero_next_state_filtered"
    assert c.data.base_config.episodes == tuple(range(1, 39))
    assert c.data.base_config.validation_episodes == (0,)
    assert c.data.base_config.aft_action_state_step_offset == 1
    assert c.data.base_config.prompt_from_task
    assert c.weight_loader.backbone_params_path.endswith("/pi0_lora_tacfield_tabero/49999/params")
    assert c.weight_loader.tactile_params_path == c.weight_loader.backbone_params_path
    assert c.freeze_filter is nnx.Nothing
    assert (c.batch_size, c.fsdp_devices, c.num_workers, c.seed) == (2, 2, 4, 42)
    assert (c.num_train_steps, c.save_interval, c.keep_period) == (30_000, 6_000, 6_000)
    assert (c.eval_interval, c.eval_num_batches) == (1_000, 174)
    assert (c.lr_schedule.warmup_steps, c.lr_schedule.peak_lr, c.lr_schedule.decay_lr) == (500, 2e-5, 2e-6)
    assert c.lr_schedule.decay_steps == 30_000
    assert c.optimizer.moment_dtype == c.parameter_dtype_policy.gradient_dtype == "float32"
    assert (c.optimizer.b1, c.optimizer.b2, c.optimizer.eps, c.optimizer.clip_gradient_norm) == (0.9, 0.95, 1e-8, 1.0)
    assert (c.wandb_enabled, c.wandb_log_images, c.ema_decay) == (True, False, None)
    assert c.policy_metadata["task_prompt"] == (
        "Pick up the yellow whiteboard eraser and erase the X-shaped mark on the whiteboard."
    )
    assert str(c.assets_dirs) == f"/data/yanghaojun/outputs/assets/{name}"
    assert c.checkpoint_base_dir == "/data/yanghaojun/outputs/checkpoints"
    for backbone in ("pi0", "pi05"):
        for dataset in ("next_state", "sent_command"):
            for tuning in ("full", "lora"):
                assert config.get_config(f"aft_{backbone}_{dataset}_{tuning}").name
