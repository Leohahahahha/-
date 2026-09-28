"""Four-stream joint-attention slow model with direct physical A/T/F flow targets.

Only VLM and action paths reuse Pi0 parameter names. T/F have independent
projections, FFNs, normalizations and output heads; Q/K/V meet in Gemma attention.
Parameter storage precision is controlled by the training parameter policy, not
by casting parameter leaves during forward. `dtype` is the shared compute dtype.
"""

import dataclasses

import flax.nnx as nnx
import flax.nnx.bridge as bridge
import jax
import jax.numpy as jnp

from openpi.models import gemma, model, siglip, tactile_encoder
from openpi.models.pi0 import Pi0, posemb_sincos
from openpi.shared.tactile_type import TactileType


def make_aft_attention_mask(valid, groups):
    groups = jnp.broadcast_to(groups, valid.shape)
    return (groups[:, None, :] <= groups[:, :, None]) & valid[:, None, :] & valid[:, :, None]


def masked_mean(error, mask):
    mask = jnp.broadcast_to(mask, error.shape)
    return jnp.sum(jnp.where(mask, error, 0.0)) / jnp.maximum(jnp.sum(mask), 1)


class SensorStream(nnx.Module):
    def __init__(self, dim, width, pi05, rngs):
        self.pi05 = pi05
        self.in_proj = nnx.Linear(dim, width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(width if pi05 else 2 * width, width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(width, width, rngs=rngs)
        self.out_proj = nnx.Linear(width, dim, rngs=rngs)

    def encode(self, value, time):
        x = self.in_proj(value)
        t = posemb_sincos(time, x.shape[-1], min_period=4e-3, max_period=4.0)
        if self.pi05:
            return x, nnx.swish(self.time_mlp_out(nnx.swish(self.time_mlp_in(t))))
        t = jnp.broadcast_to(t[:, None], x.shape)
        return self.time_mlp_out(nnx.swish(self.time_mlp_in(jnp.concatenate([x, t], -1)))), None


class AFTModel(Pi0):
    def __init__(self, config, rngs):
        # Do not instantiate/discard a full legacy model just to replace its LLM.
        model.BaseModel.__init__(self, config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.tactile_type = TactileType.EXPERT_HIS_C_FUT
        self.image_resolution = config.image_resolution
        self.tactile_loss_weight = config.tactile_loss_weight
        self.wrench_loss_weight = config.wrench_loss_weight
        vlm = gemma.get_config(config.paligemma_variant)
        self.attention_depth = vlm.depth
        act = gemma.get_config(config.action_expert_variant)
        tac = dataclasses.replace(act, width=config.tactile_width, mlp_dim=4 * config.tactile_width, lora_configs={})
        force = dataclasses.replace(act, width=config.force_width, mlp_dim=4 * config.force_width, lora_configs={})
        llm = bridge.ToNNX(gemma.Module(configs=[vlm, act, tac, force], embed_dtype=config.dtype, adarms=config.pi05))
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, config.pi05, config.pi05, config.pi05])
        img = bridge.ToNNX(
            siglip.Module(
                num_classes=vlm.width, variant=config.vision_variant, pool_type="none", scan=True, dtype_mm=config.dtype
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, act.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(act.width, config.action_dim, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(act.width, act.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(act.width, act.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, act.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * act.width, act.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(act.width, act.width, rngs=rngs)
        self.tactile_prefix_encoder = tactile_encoder.create_tactile_encoder(
            encoder_type="tcn",
            tactile_dim_in=9 * 396,
            tactile_history=8,
            has_reference_frame=True,
            diff_from_reference=config.tactile_prefix_diff_from_reference,
            expert_width=vlm.width,
            rngs=rngs,
            lora_rank=config.tactile_prefix_lora_rank,
            lora_alpha=config.tactile_prefix_lora_alpha,
        )
        self.tactile_history_proj = nnx.Linear(vlm.width, tac.width, rngs=rngs)
        self.force_history_in = nnx.Linear(48, 512, rngs=rngs)
        self.force_history_out = nnx.Linear(512, force.width, rngs=rngs)
        self.tactile_expert = SensorStream(396, tac.width, config.pi05, rngs)
        self.force_expert = SensorStream(6, force.width, config.pi05, rngs)
        self.deterministic = True

    def _process_tactile_tokens(self, obs, mode):
        # Physical history belongs to its expert, never to the VLM prefix.
        return [], [], []

    def flow(self, observation, noisy, time, target_masks=None, *, attention_stride=None):
        prefix, prefix_valid, _ = self.embed_prefix(observation)
        action, action_valid, _, action_cond = self.embed_suffix(observation, noisy["actions"], time)
        tactile, tactile_cond = self.tactile_expert.encode(noisy["shear"], time)
        force, force_cond = self.force_expert.encode(noisy["wrench"], time)
        tactile_history = self.tactile_history_proj(self.tactile_prefix_encoder(observation.tactile_prefix))[:, None]
        force_history = self.force_history_out(
            nnx.swish(self.force_history_in(observation.force_history.reshape(-1, 48)))
        )[:, None]
        tactile = jnp.concatenate([tactile_history, tactile], axis=1)
        force = jnp.concatenate([force_history, force], axis=1)
        b, h = noisy["actions"].shape[:2]
        action_history = 0 if self.pi05 else 1
        groups = jnp.concatenate(
            [
                jnp.zeros(prefix.shape[1], jnp.int32),
                jnp.ones(action_history, jnp.int32),
                jnp.full((h,), 2),
                jnp.array([1]),
                jnp.full((h,), 2),
                jnp.array([1]),
                jnp.full((h,), 2),
            ]
        )
        amask, smask = (jnp.ones((b, h), bool), jnp.ones((b, h), bool)) if target_masks is None else target_masks
        action_valid = action_valid.at[:, action_history:].set(amask)
        sensor_valid = jnp.concatenate([jnp.ones((b, 1), bool), smask], axis=1)
        valid = jnp.concatenate([prefix_valid, action_valid, sensor_valid, sensor_valid], axis=1)
        # Repeat physical slot indices across all three future streams. Sequence
        # concatenation offsets must not masquerade as different physical times.
        start = jnp.sum(prefix_valid, axis=1, keepdims=True)
        future_pos = start + 1 + jnp.arange(h)[None]
        positions = jnp.concatenate(
            [
                jnp.cumsum(prefix_valid, axis=1) - 1,
                start[:, :action_history],
                future_pos,
                start,
                future_pos,
                start,
                future_pos,
            ],
            axis=1,
        )
        diagnostic_queries = None
        if attention_stride is not None:
            slots = jnp.arange(0, h, attention_stride)
            starts = (prefix.shape[1] + action_history,
                      prefix.shape[1] + action.shape[1] + 1,
                      prefix.shape[1] + action.shape[1] + tactile.shape[1] + 1)
            diagnostic_queries = jnp.concatenate([start + slots for start in starts])
        llm_result = self.PaliGemma.llm(
            [prefix, action, tactile, force],
            positions=positions,
            mask=make_aft_attention_mask(valid, groups),
            adarms_cond=[None, action_cond, tactile_cond, force_cond],
            attention_queries=diagnostic_queries,
        )
        outputs = llm_result[0]
        velocity = dict(
            actions=self.action_out_proj(outputs[1][:, action_history : action_history + h]),
            shear=self.tactile_expert.out_proj(outputs[2][:, 1 : 1 + h]),
            wrench=self.force_expert.out_proj(outputs[3][:, 1 : 1 + h]),
        )
        if attention_stride is None:
            return velocity
        text_length = 0 if observation.tokenized_prompt is None else observation.tokenized_prompt.shape[1]
        image_length = (prefix.shape[1] - text_length) // len(observation.images)
        key_groups, camera_ids, patch_indices = [], [], []
        for camera_id, name in enumerate(observation.images):
            group_id = 0 if name == "base_0_rgb" else (1 if name == "left_wrist_0_rgb" else 2)
            key_groups.append(jnp.full((image_length,), group_id))
            camera_ids.append(jnp.full((image_length,), camera_id))
            patch_indices.append(jnp.arange(image_length))
        image_tokens = prefix.shape[1] - text_length
        key_groups += [jnp.full((text_length,), 3), jnp.full((action_history,), 4),
                       jnp.full((h,), 5), jnp.array([6]), jnp.full((h,), 7),
                       jnp.array([8]), jnp.full((h,), 9)]
        non_image_length = valid.shape[1] - image_tokens
        attention = llm_result[2]
        query_valid = jnp.take(valid, diagnostic_queries, axis=1)
        attention = jnp.where(query_valid[None, :, :, None], attention, 0.0)
        return velocity, dict(
            attention=attention, key_group_id=jnp.concatenate(key_groups), key_valid=valid,
            key_camera_id=jnp.concatenate([*camera_ids, jnp.full((non_image_length,), -1)]),
            key_patch_index=jnp.concatenate([*patch_indices, jnp.full((non_image_length,), -1)]),
            query_stream_id=jnp.repeat(jnp.arange(3), len(slots)),
            query_horizon_index=jnp.tile(slots, 3),
        )

    def compute_loss(self, rng, observation, targets, *, train=False, return_components=False):
        prep, time_rng, *noise_rngs = jax.random.split(rng, 5)
        observation = model.preprocess_observation(
            prep, observation, train=train, tactile_type=self.tactile_type, image_resolution=self.image_resolution
        )
        target = dict(actions=targets.actions, shear=targets.shear, wrench=targets.wrench)
        time = jax.random.beta(time_rng, 1.5, 1.0, (targets.actions.shape[0],)) * 0.999 + 0.001
        noise = {k: jax.random.normal(r, v.shape) for (k, v), r in zip(target.items(), noise_rngs, strict=True)}
        noisy = {k: time[:, None, None] * noise[k] + (1 - time[:, None, None]) * v for k, v in target.items()}
        prediction = self.flow(observation, noisy, time, (targets.action_mask, targets.sensor_mask))
        losses = {}
        for key, logkey, mask, dim in [
            ("actions", "action_loss", targets.action_mask, 7),
            ("shear", "tactile_loss", targets.sensor_mask, 396),
            ("wrench", "wrench_loss", targets.sensor_mask, 6),
        ]:
            error = (prediction[key][..., :dim] - (noise[key] - target[key])[..., :dim]) ** 2
            losses[logkey] = masked_mean(error, mask[..., None])
        total = (
            losses["action_loss"]
            + self.tactile_loss_weight * losses["tactile_loss"]
            + self.wrench_loss_weight * losses["wrench_loss"]
        )
        return (total, losses) if return_components else total

    def sample_predictions(self, rng, observation, *, num_steps=10):
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        observation = model.preprocess_observation(
            None, observation, tactile_type=self.tactile_type, image_resolution=self.image_resolution
        )
        b, h = observation.state.shape[0], self.action_horizon
        keys = jax.random.split(rng, 3)
        x = {
            k: jax.random.normal(r, (b, h, d))
            for k, d, r in zip(("actions", "shear", "wrench"), (self.action_dim, 396, 6), keys, strict=True)
        }

        def step(i, current):
            velocity = self.flow(observation, current, jnp.full((b,), 1.0 - i / num_steps))
            return jax.tree.map(lambda value, v: value - v / num_steps, current, velocity)

        return jax.lax.fori_loop(0, num_steps, step, x)

    def sample_actions(self, rng, observation, **kwargs):
        return self.sample_predictions(rng, observation, **kwargs)["actions"]

    def sample_attention(self, rng, observation, *, num_steps=10, denoise_step=-1, query_stride=1):
        """Replay the same noise to one Euler evaluation and export head means.

        Separate opt-in pass: the standard inference/training graphs and returned
        predictions remain untouched. The selected step is zero-based; -1 is last.
        """
        if num_steps <= 0 or query_stride <= 0:
            raise ValueError("num_steps and query_stride must be positive")
        denoise_step = num_steps - 1 if denoise_step == -1 else denoise_step
        if not 0 <= denoise_step < num_steps:
            raise ValueError("denoise_step must be -1 or within num_steps")
        observation = model.preprocess_observation(
            None, observation, tactile_type=self.tactile_type, image_resolution=self.image_resolution
        )
        b, h = observation.state.shape[0], self.action_horizon
        x = {key: jax.random.normal(r, (b, h, dim)) for key, dim, r in zip(
            ("actions", "shear", "wrench"), (self.action_dim, 396, 6), jax.random.split(rng, 3), strict=True
        )}

        def step(i, current):
            v = self.flow(observation, current, jnp.full((b,), 1.0 - i / num_steps))
            return jax.tree.map(lambda value, delta: value - delta / num_steps, current, v)

        noisy = jax.lax.fori_loop(0, denoise_step, step, x)
        velocity, trace = self.flow(
            observation, noisy, jnp.full((b,), 1.0 - denoise_step / num_steps), attention_stride=query_stride
        )
        # Values make replay independently verifiable; serving omits these large
        # physical-state tensors and exports only the attention/layout fields.
        return dict(trace, noisy=noisy, velocity=velocity)
