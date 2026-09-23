"""Offline AFT prediction metrics; no robot interface or commands are imported."""

import argparse
import dataclasses
import json
import pathlib

import jax
import numpy as np
from scipy.spatial.transform import Rotation

from openpi import transforms
from openpi.models import model
from openpi.policies.aft_policy import AFTInputs, AFTOutputs, NormalizeAFTCommon, NormalizeSensors
from openpi.shared import normalize, nnx_utils


def physical_errors(actions, target_actions, shear, target_shear, wrench, target_wrench, action_mask, sensor_mask):
    rotation = (Rotation.from_rotvec(actions[:, 3:6]).inv() * Rotation.from_rotvec(target_actions[:, 3:6])).magnitude()
    return dict(
        position_mm=np.where(
            action_mask, 1000 * np.linalg.norm(actions[:, :3] - target_actions[:, :3], axis=-1), np.nan
        ),
        rotation_deg=np.where(action_mask, np.rad2deg(rotation), np.nan),
        gripper_mm=np.where(action_mask, 1000 * np.abs(actions[:, 6] - target_actions[:, 6]), np.nan),
        shear_rmse=np.where(sensor_mask, np.sqrt(np.mean((shear - target_shear) ** 2, -1)), np.nan),
        wrench_abs=np.where(sensor_mask[:, None], np.abs(wrench - target_wrench), np.nan),
    )


def main(config, checkpoint, output_dir, edges_json=None, max_anchors=100, stride=10, seed=42):
    from openpi.training import config as configs, data_loader

    c = configs.get_config(config)
    checkpoint = pathlib.Path(checkpoint)
    output = pathlib.Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if max_anchors < 1 or stride < 1:
        raise ValueError("max_anchors and stride must be positive")
    # The checkpoint carries its normalization; external training stats are not substituted.
    stats = normalize.load(checkpoint / "assets/aft")
    assets = checkpoint / "assets/aft"
    from openpi.training.aft_assets import validate_assets, file_hash

    manifest = validate_assets(assets, c.model, c.data.base_config)
    if edges_json and file_hash(edges_json) != manifest["adjacency_sha256"]:
        raise ValueError("Override adjacency hash mismatch")
    if set(stats) != {"state", "actions", "tactile_prefix", "shear", "wrench"}:
        raise ValueError("Checkpoint does not contain the AFT statistics contract")
    data = dataclasses.replace(
        c.data.base_config,
        repo_id=c.data.repo_id,
        episodes=c.data.base_config.validation_episodes,
        validation_episodes=(),
        aft_edges_path=edges_json or str(assets / "adjacency.json"),
    )
    dataset = data_loader.create_torch_dataset(data, c.model.action_horizon, c.model)
    network = c.model.load(model.restore_params(checkpoint / "params"))
    predict = nnx_utils.module_jit(network.sample_predictions)
    state_stats = {k: v for k, v in stats.items() if k in ("state", "actions", "tactile_prefix")}
    use_quantiles = c.model.pi05
    encode = transforms.compose(
        [
            AFTInputs(c.model.model_type),
            transforms.RelativePoseActions(),
            NormalizeAFTCommon(state_stats, use_quantiles=use_quantiles),
            NormalizeSensors(stats),
            *configs.ModelTransformFactory()(c.model).inputs,
        ]
    )
    unnormalize = transforms.Unnormalize({k: stats[k] for k in ("state", "actions")}, use_quantiles=use_quantiles)
    records, anchors = [], []
    for index in range(0, len(dataset), stride):
        if len(records) >= max_anchors:
            break
        raw = dataset[index]
        target = raw["targets"]
        inputs = encode(raw)
        obs = model.Observation.from_dict(jax.tree.map(lambda x: np.asarray(x)[None], inputs))
        prediction = jax.tree.map(
            lambda x: np.asarray(x[0]), predict(jax.random.fold_in(jax.random.key(seed), index), obs)
        )
        physical = unnormalize(dict(prediction, state=inputs["state"]))
        physical = transforms.AbsolutePoseActions()(physical)
        physical = AFTOutputs(stats)(physical)
        records.append(
            physical_errors(
                physical["actions"],
                target.actions,
                physical["tactile_shear"].reshape(-1, 396),
                target.shear,
                physical["wrist_wrench"],
                target.wrench,
                target.action_mask,
                target.sensor_mask,
            )
        )
        anchors.append(dict(episode=int(raw["episode_index"]), frame=int(raw["frame_index"])))
    if not records:
        raise ValueError("No evaluation anchors")
    output.mkdir(parents=True)
    arrays = {k: np.stack([r[k] for r in records]) for k in records[0]}
    np.savez_compressed(output / "errors.npz", **arrays)
    summary = {}
    for key, values in arrays.items():
        count = np.isfinite(values).sum(axis=0)
        mean = np.divide(np.nansum(values, axis=0), count, out=np.full(count.shape, np.nan), where=count > 0)
        summary[key] = dict(mean=float(np.nanmean(values)), per_horizon_mean=mean.tolist(), valid_count=count.tolist())
    (output / "metrics.json").write_text(
        json.dumps(
            dict(config=config, checkpoint=str(checkpoint), anchors=anchors, seed=seed, metrics=summary), indent=2
        )
    )
    print(f"Offline metrics saved: {output}; no robot commands issued.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("config", "checkpoint", "output-dir"):
        parser.add_argument("--" + option, required=True)
    parser.add_argument("--edges-json")
    parser.add_argument("--max-anchors", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    main(**vars(parser.parse_args()))
