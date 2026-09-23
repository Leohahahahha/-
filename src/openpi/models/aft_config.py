"""Opt-in slow action/tactile/wrench model; legacy Pi0 configs are unchanged."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.aft_types import AFTTargets
from openpi.models.pi0_config import Pi0Config
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils


@dataclasses.dataclass(frozen=True)
class AFTConfig(Pi0Config):
    tactile_width: int = 512
    force_width: int = 256
    force_history_frames: int = 8
    wrench_loss_weight: float = 0.1
    vision_variant: str = "So400m/14"
    image_resolution: tuple[int, int] = (224, 224)
    tactile_prefix_diff_from_reference: bool = False

    def __post_init__(self):
        super().__post_init__()
        if self.force_history_frames != 8 or self.action_dim < 7:
            raise ValueError("AFT requires eight force frames and >=7 action dimensions")
        if self.tactile_width % 2 or self.force_width % 2:
            raise ValueError("Expert widths must be even for sinusoidal time embeddings")

    def create(self, rng):
        from openpi.models.aft import AFTModel

        return AFTModel(self, nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        obs, actions = super().inputs_spec(batch_size=batch_size)
        b, h = batch_size, self.action_horizon
        with at.disable_typechecking():
            obs = dataclasses.replace(
                obs,
                images={k: jax.ShapeDtypeStruct((b, *self.image_resolution, 3), jnp.float32) for k in obs.images},
                tactile_prefix=jax.ShapeDtypeStruct((b, 9, 396), jnp.float32),
                force_history=jax.ShapeDtypeStruct((b, 8, 6), jnp.float32),
                force_history_mask=jax.ShapeDtypeStruct((b, 8), jnp.bool_),
            )
        return obs, AFTTargets(
            actions,
            jax.ShapeDtypeStruct((b, h, 396), jnp.float32),
            jax.ShapeDtypeStruct((b, h, 6), jnp.float32),
            jax.ShapeDtypeStruct((b, h), jnp.bool_),
            jax.ShapeDtypeStruct((b, h), jnp.bool_),
        )

    def get_freeze_filter(self):
        # Never freeze newly initialized experts because the VLM uses LoRA.
        language = nnx.All(super().get_freeze_filter(), nnx.Not(nnx_utils.PathRegex(r".*(?:_2|_3)(?:/|$).*")))
        if "lora" in self.paligemma_variant or "lora" in self.action_expert_variant:
            return nnx.Any(language, nnx_utils.PathRegex(r"PaliGemma/img/.*"))
        return language
