"""Inference-only explanation data. Attention is not a causal importance score."""

import dataclasses
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

KEY_GROUPS = (
    "front_rgb", "wrist_rgb", "other_rgb", "language", "state",
    "action_future", "tactile_history", "tactile_future", "force_history", "force_future",
)
QUERY_STREAMS = ("action", "tactile", "force")


@dataclasses.dataclass(frozen=True)
class DiagnosticsOptions:
    every: int = 1
    denoise_step: int = -1
    query_stride: int = 1
    layers: tuple[int, ...] = ()  # empty: last layer only; zero-based indices
    ablations: bool = False

    def __post_init__(self):
        if self.every <= 0 or self.query_stride <= 0 or self.denoise_step < -1:
            raise ValueError("Invalid diagnostics sampling options")
        if any(layer < 0 for layer in self.layers) or len(set(self.layers)) != len(self.layers):
            raise ValueError("Diagnostic layers must be unique nonnegative indices")


def summarize_attention(trace):
    """Aggregate [layer,query,key] by valid key type, retaining token counts."""
    attention = np.asarray(trace["attention"], np.float32)
    ids, valid = np.asarray(trace["key_group_id"]), np.asarray(trace["key_valid"], bool)
    if (attention.ndim != 3 or ids.shape != (attention.shape[-1],) or valid.shape != ids.shape
            or not np.isfinite(attention).all() or (attention < 0).any()
            or ((ids < 0) | (ids >= len(KEY_GROUPS))).any()):
        raise ValueError("Invalid attention/layout for diagnostics")
    mass = np.stack([attention[..., (ids == group) & valid].sum(-1)
                     for group in range(len(KEY_GROUPS))], axis=-1)
    counts = np.array([np.count_nonzero((ids == group) & valid) for group in range(len(KEY_GROUPS))])
    return dict(modality_mass=mass, valid_key_counts=counts,
                modality_per_token=mass / np.maximum(counts, 1))


def prediction_difference(baseline, altered):
    """Physical differences for a paired *history perturbation*, not an error."""
    a, b = np.asarray(baseline["actions"]), np.asarray(altered["actions"])
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 7 or not np.isfinite([a, b]).all():
        raise ValueError("Paired predictions must be finite [H,7]")
    relative = Rotation.from_rotvec(a[:, 3:6]).inv() * Rotation.from_rotvec(b[:, 3:6])
    sensor_delta = np.asarray(altered["wrist_wrench"]) - np.asarray(baseline["wrist_wrench"])
    shear_delta = np.asarray(altered["tactile_shear"]) - np.asarray(baseline["tactile_shear"])
    return dict(
        actions=b.astype(np.float32),
        position_delta_mm=np.linalg.norm(b[:, :3] - a[:, :3], axis=-1) * 1000,
        rotation_delta_deg=np.rad2deg(relative.magnitude()),
        gripper_delta_mm=np.abs(b[:, 6] - a[:, 6]) * 1000,
        tactile_delta_rms=np.sqrt(np.mean(shear_delta**2, axis=(1, 2))),
        force_delta_norm_N=np.linalg.norm(sensor_delta[:, :3], axis=-1),
        torque_delta_norm_Nm=np.linalg.norm(sensor_delta[:, 3:], axis=-1),
    )


def save_bundle(response, output_dir, record_id, *, context=None):
    """Portable JSON + numeric NPZ, no pickle; refuse collisions/unsafe IDs."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", record_id):
        raise ValueError("record_id must be a safe filename component")
    arrays, values = {}, {}

    def collect(value, path):
        if isinstance(value, dict) and value:
            for key, item in value.items():
                if not isinstance(key, str) or "/" in key:
                    raise ValueError("Bundle keys must be strings without slashes")
                collect(item, f"{path}/{key}" if path else key)
        elif isinstance(value, np.ndarray):
            if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
                raise ValueError("Bundle arrays must be finite numeric data")
            arrays[path] = value
        else:
            values[path] = value.item() if isinstance(value, np.generic) else value

    collect(response, "")
    manifest = json.dumps(dict(format="aft_diagnostic_bundle_v1", values=values, context=context or {}),
                          ensure_ascii=False, indent=2, allow_nan=False)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / record_id
    json_path, npz_path = directory / f"{record_id}.json", directory / f"{record_id}.npz"
    if json_path.exists() or npz_path.exists():
        raise FileExistsError(f"Refusing to overwrite {prefix}")
    # Stage both complete files on the same filesystem. Publish the manifest
    # last; hard links give exclusive creation without replacing an old bundle.
    published = []
    with tempfile.TemporaryDirectory(prefix=".aft-diagnostic-", dir=directory) as staging:
        staged_npz, staged_json = Path(staging) / "data.npz", Path(staging) / "data.json"
        with staged_npz.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        with staged_json.open("x") as handle:
            handle.write(manifest)
        try:
            for staged, final in ((staged_npz, npz_path), (staged_json, json_path)):
                os.link(staged, final)
                published.append(final)
        except BaseException:
            for path in published:
                path.unlink(missing_ok=True)
            raise
    return prefix


def load_bundle(prefix):
    prefix = Path(prefix)
    manifest = json.loads(Path(str(prefix) + ".json").read_text())
    if manifest.get("format") != "aft_diagnostic_bundle_v1":
        raise ValueError("Unsupported diagnostic bundle format")
    flat = dict(manifest["values"])
    with np.load(Path(str(prefix) + ".npz"), allow_pickle=False) as arrays:
        flat.update({key: arrays[key] for key in arrays.files})
    result = {}
    for path, value in flat.items():
        keys = path.split("/")
        node = result
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    return result, manifest["context"]
