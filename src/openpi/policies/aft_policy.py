"""AFT physical contracts, independent sensor statistics and output decoding.

Action SO(3) conversion and action normalization remain separate existing
transforms. AFTOutputs expects actions already unnormalized and made absolute.
Historical marker positions retain the pretrained TCN convention unchanged.
"""

import dataclasses

import numpy as np
import jax
from openpi_client import base_policy

from openpi import transforms
from openpi.models import model
from openpi.policies.libero_policy import TaberoActionOnlyInputs
from openpi.shared.normalize import NormStats


def _finite(value, shape, name):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite {shape}, got {value.shape}")
    return value


@dataclasses.dataclass(frozen=True)
class AFTInputs:
    model_type: model.ModelType

    def __call__(self, data):
        raw = dict(data)
        targets = raw.pop("targets", None)
        if targets is not None:
            raw["actions"] = targets.actions
        result = TaberoActionOnlyInputs(self.model_type)(raw)
        result["force_history"] = _finite(raw["force_history"], (8, 6), "force_history")
        history_mask = np.asarray(raw["force_history_mask"])
        if history_mask.shape != (8,) or history_mask.dtype != bool or not history_mask[-1]:
            raise ValueError("force_history_mask must be boolean [8] with valid current frame")
        result["force_history_mask"] = history_mask
        if targets is not None:
            horizon = len(targets.actions)
            result["shear"] = _finite(targets.shear, (horizon, 396), "shear")
            result["wrench"] = _finite(targets.wrench, (horizon, 6), "wrench")
            for key in ("action_mask", "sensor_mask"):
                mask = np.asarray(getattr(targets, key))
                if mask.shape != (horizon,) or mask.dtype != bool:
                    raise ValueError(f"{key} must be boolean [horizon]")
                result[key] = mask
        return result


def _sensor_affine(value, stats, dimension, inverse=False):
    mean = np.asarray(stats.mean, dtype=np.float32)
    std = np.asarray(stats.std, dtype=np.float32)
    if mean.shape != (dimension,) or std.shape != (dimension,):
        raise ValueError("Sensor statistics dimension mismatch")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError("Sensor statistics must be finite with nonnegative std")
    scale = np.maximum(std, 1e-6)
    value = np.asarray(value, dtype=np.float32)
    if value.shape[-1] != dimension or not np.isfinite(value).all():
        raise ValueError("Sensor values must be finite and match statistics dimension")
    return value * scale + mean if inverse else (value - mean) / scale


@dataclasses.dataclass(frozen=True)
class NormalizeSensors:
    stats: dict[str, NormStats]

    def __call__(self, data):
        result = dict(data)
        for key, stat_key, dim in [("force_history", "wrench", 6), ("wrench", "wrench", 6), ("shear", "shear", 396)]:
            if key in result:
                result[key] = _sensor_affine(result[key], self.stats[stat_key], dim)
        return result


@dataclasses.dataclass(frozen=True)
class NormalizeAFTCommon:
    """Preserve z-score inputs for the transferred TCN under either backbone."""

    stats: dict[str, NormStats]
    use_quantiles: bool = False

    def __call__(self, data):
        result = transforms.Normalize(
            {key: self.stats[key] for key in ("state", "actions") if key in self.stats},
            use_quantiles=self.use_quantiles,
        )(data)
        return transforms.Normalize({"tactile_prefix": self.stats["tactile_prefix"]}, use_quantiles=False)(result)


@dataclasses.dataclass(frozen=True)
class AFTOutputs:
    stats: dict[str, NormStats]

    def __call__(self, data):
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] < 7 or not np.isfinite(actions).all():
            raise ValueError("actions must be finite [horizon,>=7]")
        h = len(actions)
        shear = _finite(data["shear"], (h, 396), "shear")
        wrench = _finite(data["wrench"], (h, 6), "wrench")
        return dict(
            actions=actions[:, :7],
            tactile_shear=_sensor_affine(shear, self.stats["shear"], 396, True).reshape(h, 198, 2),
            wrist_wrench=_sensor_affine(wrench, self.stats["wrench"], 6, True),
        )


def fit_sensor_stats(episodes, train_episodes):
    """One physical sample per row; validation rows and overlapping windows excluded."""
    if not train_episodes or len(set(train_episodes)) != len(train_episodes):
        raise ValueError("Require nonempty unique training episode IDs")
    result = {}
    for key, dim in [("shear", 396), ("wrench", 6)]:
        values = []
        for episode in train_episodes:
            value = np.asarray(episodes[episode][key], dtype=np.float64)
            if value.ndim != 2 or value.shape[1] != dim or not np.isfinite(value).all():
                raise ValueError(f"{key} must be finite [frames,{dim}]")
            values.append(value)
        values = np.concatenate(values)
        if len(values) < 2:
            raise ValueError("At least two training rows required")
        result[key] = NormStats(mean=values.mean(0), std=values.std(0))
    return result


class AFTPolicy(base_policy.BasePolicy):
    """Synchronous three-output inference; intentionally no robot control hooks."""

    def __init__(self, network, encode, decode, *, rng=None, sample_kwargs=None, metadata=None):
        from openpi.shared.nnx_utils import module_jit

        self._predict = module_jit(network.sample_predictions, static_argnames=("num_steps",))
        self._encode, self._decode = encode, decode
        self._rng = jax.random.key(0) if rng is None else rng
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @property
    def metadata(self):
        return self._metadata

    def infer(self, obs, *, seed=None):
        # Remove supervision before transformation, even if called on a dataset sample.
        inputs = self._encode(
            {
                k: v
                for k, v in obs.items()
                if k not in ("targets", "actions", "shear", "wrench", "action_mask", "sensor_mask")
            }
        )
        batched = jax.tree.map(lambda x: np.asarray(x)[None], inputs)
        observation = model.Observation.from_dict(batched)
        self._rng, rng = jax.random.split(self._rng)
        if seed is not None:
            rng = jax.random.key(seed)
        result = self._predict(rng, observation, **self._sample_kwargs)
        result = jax.tree.map(lambda x: np.asarray(x[0]), result)
        return self._decode(dict(result, state=inputs["state"]))


def create_aft_policy(train_config, checkpoint, *, repack_transforms=None, sample_kwargs=None, default_prompt=None):
    import pathlib
    from openpi import transforms
    from openpi.shared import normalize
    from openpi.training.aft_assets import validate_assets
    from openpi.training.config import ModelTransformFactory

    checkpoint = pathlib.Path(checkpoint)
    assets = checkpoint / "assets/aft"
    validate_assets(assets, train_config.model, train_config.data.base_config, check_source=False)
    stats = normalize.load(assets)
    # Preserve stored FP32 sensor/normalization parameters; no blanket BF16 restore.
    network = train_config.model.load(model.restore_params(checkpoint / "params"), remove_extra_params=False)
    network.eval()
    repack = repack_transforms or transforms.Group()
    common_stats = {k: stats[k] for k in ("state", "actions", "tactile_prefix")}
    encode = transforms.compose(
        [
            *repack.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            AFTInputs(train_config.model.model_type),
            transforms.RelativePoseActions(),
            NormalizeAFTCommon(common_stats, use_quantiles=train_config.model.pi05),
            NormalizeSensors(stats),
            *ModelTransformFactory()(train_config.model).inputs,
        ]
    )
    decode = transforms.compose(
        [
            transforms.Unnormalize({k: stats[k] for k in ("state", "actions")}, use_quantiles=train_config.model.pi05),
            transforms.AbsolutePoseActions(),
            AFTOutputs(stats),
            *repack.outputs,
        ]
    )
    return AFTPolicy(network, encode, decode, sample_kwargs=sample_kwargs, metadata=train_config.policy_metadata)
