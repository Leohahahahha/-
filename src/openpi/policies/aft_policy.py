"""AFT physical contracts, independent sensor statistics and output decoding.

Action SO(3) conversion and action normalization remain separate existing
transforms. AFTOutputs expects actions already unnormalized and made absolute.
Historical marker positions retain the pretrained TCN convention unchanged.
"""

import dataclasses
import logging
import time

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
        self._network = network
        self._diagnostics = None
        self._infer_count = 0
        self._diagnostics_output_dir = None

    def configure_diagnostics(self, options=None, *, output_dir=None, metadata=None):
        """Enable optional replay/perturbation passes; leave ordinary infer fast."""
        from openpi.shared.nnx_utils import module_jit
        if options is not None:
            steps = self._sample_kwargs.get("num_steps", 10)
            if options.denoise_step >= steps:
                raise ValueError("diagnostic denoise_step is outside num_steps")
            depth = self._network.attention_depth
            if options.layers and max(options.layers) >= depth:
                raise ValueError("diagnostic layer index is outside model depth")
            self._inspect = module_jit(
                self._network.sample_attention, static_argnames=("num_steps", "denoise_step", "query_stride")
            )
        self._diagnostics = options
        self._diagnostics_output_dir = output_dir
        if metadata is not None:
            self._metadata = {**self._metadata, **metadata}
        self._infer_count = 0

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
        physical = self._decode(dict(result, state=inputs["state"]))
        options = self._diagnostics
        count = self._infer_count
        self._infer_count += 1
        if options is not None and count % options.every == 0:
            from openpi.policies.aft_diagnostics import (
                KEY_GROUPS, QUERY_STREAMS, prediction_difference, summarize_attention,
            )
            diagnostic_start = time.perf_counter()
            steps = self._sample_kwargs.get("num_steps", 10)
            trace = self._inspect(rng, observation, num_steps=steps,
                                  denoise_step=options.denoise_step, query_stride=options.query_stride)
            layers = options.layers or (trace["attention"].shape[0] - 1,)
            if max(layers) >= trace["attention"].shape[0]:
                raise ValueError("diagnostic layer index is outside model depth")
            d = {key: np.asarray(value) for key, value in trace.items() if key not in ("velocity", "noisy")}
            d["attention"] = d["attention"][list(layers), 0]
            d["key_valid"] = d["key_valid"][0]
            d.update(summarize_attention(d))
            d.update(schema="aft_diagnostics_v1", layer_indices=list(layers),
                     key_group_names=list(KEY_GROUPS), query_stream_names=list(QUERY_STREAMS),
                     image_camera_names=sorted(observation.images),
                     denoise_step=steps - 1 if options.denoise_step == -1 else options.denoise_step,
                     flow_time=1.0 - (steps - 1 if options.denoise_step == -1 else options.denoise_step) / steps,
                     num_denoise_steps=steps, inference_index=count, rng_key=np.asarray(jax.random.key_data(rng)),
                     interpretation="head_mean_attention_and_normalized_history_sensitivity_not_causal_importance")
            d["ablations"] = {}
            if options.ablations:
                for name, tactile, force in (("mean_tactile_history", True, False),
                                             ("mean_force_history", False, True), ("mean_both_histories", True, True)):
                    perturbed = observation.replace(
                        tactile_prefix=(jax.numpy.zeros_like(observation.tactile_prefix)
                                        if tactile else observation.tactile_prefix),
                        force_history=(jax.numpy.zeros_like(observation.force_history)
                                       if force else observation.force_history),
                    )
                    prediction = self._predict(rng, perturbed, **self._sample_kwargs)
                    prediction = jax.tree.map(lambda x: np.asarray(x[0]), prediction)
                    decoded = self._decode(dict(prediction, state=inputs["state"]))
                    d["ablations"][name] = prediction_difference(physical, decoded)
            physical["diagnostics"] = d
            d["diagnostics_latency_ms"] = (time.perf_counter() - diagnostic_start) * 1000
            d["record_id"] = f"inference_{time.time_ns()}_{count:06d}"
            action_rows = d["query_stream_id"] == 0
            mass = d["modality_mass"][:, action_rows].mean(axis=(0, 1))
            logging.info("AFT attention mass (%s): %s; diagnostic extra %.1f ms", d["record_id"],
                         dict(zip(KEY_GROUPS, np.round(mass, 5).tolist(), strict=True)), d["diagnostics_latency_ms"])
            d["capture_status"] = "not_requested"
            if self._diagnostics_output_dir is not None:
                from openpi.policies.aft_diagnostics import save_bundle
                try:
                    d["capture_status"] = "saved"
                    save_bundle(physical, self._diagnostics_output_dir, d["record_id"],
                                context={"server_wall_time": time.time(), "policy_metadata": self._metadata})
                except (OSError, ValueError, TypeError) as error:
                    d["capture_status"] = "error"
                    d["capture_error"] = f"{type(error).__name__}: {error}"
                    logging.exception("AFT diagnostic capture failed; preserving baseline predictions")
        return physical


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
