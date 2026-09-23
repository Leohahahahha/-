"""Eight explicit AFT experiment configurations; import registration at config end."""

import flax.nnx as nnx

from openpi.models.aft_config import AFTConfig
from openpi.training import optimizer, weight_loaders


def make_aft_configs():
    from openpi.training import config as c

    tabero = "/data/yanghaojun/checkpoints/tabero-pretrained/checkpoints/pi0_lora_tacfield_tabero/pi0_lora_tacfield_tabero/49999/params"
    policy = c.ParameterDtypePolicy(
        name="aft_mixed_fp32_sensors",
        default_trainable_dtype="bfloat16",
        gradient_dtype="float32",
        optimizer_state_dtype="float32",
        include_frozen=True,
        overrides=(
            c.ParameterDtypeRule(r"(?:tactile.*|force.*)/.*", "float32"),
            c.ParameterDtypeRule(r"PaliGemma/llm/.*(?:_2|_3)(?:/.*)?", "float32"),
            c.ParameterDtypeRule(r"PaliGemma/llm/embedder/.*", "float32"),
            *c._TABERO_MIXED_BF16_PARAMETER_POLICY.overrides,
        ),
    )
    result = []
    for backbone in ("pi0", "pi05"):
        for dataset in ("next_state", "sent_command"):
            for tuning in ("full", "lora"):
                name = f"aft_{backbone}_{dataset}_{tuning}"
                # π0 source already has LoRA leaves even when all leaves train.
                lora = backbone == "pi0" or tuning == "lora"
                model = AFTConfig(
                    pi05=backbone == "pi05",
                    paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
                    action_expert_variant="gemma_300m_lora" if lora else "gemma_300m",
                )
                root = f"/data/yanghaojun/datasets/test2_tabero_{dataset}_compact"
                data = c.AFTDataConfig(
                    repo_id=f"local/test2_tabero_{dataset}_compact",
                    assets=c.AssetsConfig(asset_id="aft"),
                    base_config=c.DataConfig(
                        root=root,
                        episodes=tuple(i for i in range(39) if i not in (1, 7, 17)),
                        validation_episodes=(1, 7, 17),
                        video_backend="pyav",
                        prompt_from_task=True,
                        columns=(
                            "state",
                            "actions",
                            "wrist_wrench",
                            "tactile_marker_motion",
                            "timestamp",
                            "frame_index",
                            "episode_index",
                            "index",
                            "task_index",
                        ),
                        action_sequence_keys=(),
                        aft_enabled=True,
                        aft_action_state_step_offset=int(dataset == "next_state"),
                    ),
                )
                result.append(
                    c.TrainConfig(
                        name=name,
                        model=model,
                        data=data,
                        weight_loader=weight_loaders.AFTWeightLoader(
                            tabero if backbone == "pi0" else "gs://openpi-assets/checkpoints/pi05_base/params",
                            tabero,
                            allow_new_lora=backbone == "pi05" and tuning == "lora",
                        ),
                        freeze_filter=model.get_freeze_filter() if tuning == "lora" else nnx.Nothing,
                        parameter_dtype_policy=policy,
                        optimizer=optimizer.AdamW(moment_dtype="float32"),
                        ema_decay=None,
                        batch_size=2,
                        fsdp_devices=2,
                        num_workers=2,
                        num_train_steps=20_000,
                        save_interval=4000,
                        keep_period=4000,
                        eval_interval=1000,
                        eval_num_batches=50,
                        lr_schedule=optimizer.CosineDecaySchedule(
                            warmup_steps=500,
                            peak_lr=2e-6 if tuning == "full" else 1e-5,
                            decay_steps=20_000,
                            decay_lr=2e-7 if tuning == "full" else 1e-6,
                        ),
                        assets_base_dir="/data/yanghaojun/outputs/assets",
                        checkpoint_base_dir="/data/yanghaojun/outputs/checkpoints",
                        project_name="tabero-aft",
                        wandb_enabled=True,
                        wandb_log_images=False,
                        checkpoint_step_is_update_count=True,
                        wait_for_checkpoint_on_save=True,
                        policy_metadata=dict(
                            architecture="aft",
                            horizon=50,
                            wrench_frame="K",
                            wrench_units=["N"] * 3 + ["N_m"] * 3,
                            sensor_offset=1,
                            dataset=dataset,
                        ),
                    )
                )
    return result
