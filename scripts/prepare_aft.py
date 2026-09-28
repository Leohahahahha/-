"""Audit AFT timing and write train-only statistics; never starts training."""

import argparse
import hashlib
import json
import pathlib

import numpy as np


from openpi import transforms
from openpi.policies.aft_policy import fit_sensor_stats
from openpi.shared import normalize
from openpi.training.aft_data import (
    audit_episode_edges, build_episode_windows, read_episode, source_mapping_path, validate_next_state_labels,
)
from openpi.training.aft_assets import model_contract


def collect_statistics(episodes, edges, *, horizon, action_offset):
    sensors = {
        ep: dict(
            shear=(r["tactile_marker_motion"][:, -1] - r["tactile_marker_motion"][:, 0]).reshape(-1, 396),
            wrench=r["wrist_wrench"],
        )
        for ep, r in episodes.items()
    }
    stats = fit_sensor_stats(sensors, tuple(episodes))
    running = {key: normalize.RunningStats() for key in ("state", "actions", "tactile_prefix")}
    for ep, rows in episodes.items():
        running["state"].update(rows["state"])
        running["tactile_prefix"].update(rows["tactile_marker_motion"].reshape(-1, 396))
        for anchor in range(len(rows["state"])):
            target = build_episode_windows(
                rows, anchor, horizon, contiguous_edges=edges[ep], action_state_step_offset=action_offset
            )["targets"]
            valid = target.actions[target.action_mask]
            if len(valid):
                relative = transforms.RelativePoseActions()(dict(state=rows["state"][anchor], actions=valid))
                running["actions"].update(relative["actions"])
    stats.update({k: value.get_statistics() for k, value in running.items()})
    return stats


def window_target_counts(edges, *, horizon, action_offset):
    """Count valid horizon slots by continuous segments, without reading tensor targets."""
    boundaries = np.r_[0, np.flatnonzero(~edges) + 1, len(edges) + 1]
    lengths = np.diff(boundaries)
    sensors = sum(min(horizon, remaining) for length in lengths for remaining in range(length))
    actions = sum(min(horizon, remaining + 1 - action_offset) for length in lengths for remaining in range(length))
    return int(actions), int(sensors)


def main(config, output_dir, edges_json=None):
    from openpi.training.config import get_config

    c = get_config(config)
    data = c.data.base_config
    if not data.aft_enabled:
        raise ValueError("Requires an AFT configuration")
    output = pathlib.Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    root = pathlib.Path(data.root)
    conversion_path = root / "meta/tabero_conversion.json"
    conversion = json.loads(conversion_path.read_text())
    expected_mode = "next-state" if data.aft_action_state_step_offset else "sent-command"
    if conversion.get("action_source_mode") != expected_mode:
        raise ValueError(f"Expected action_source_mode={expected_mode}")
    supplied = json.loads(pathlib.Path(edges_json).read_text()) if edges_json else None
    # Audit both splits before producing anything that could be mistaken for ready assets.
    all_ids = (*data.episodes, *data.validation_episodes)
    all_rows = {ep: read_episode(str(root), ep) for ep in all_ids}
    edges = {
        ep: audit_episode_edges(root, conversion, ep, len(all_rows[ep]["state"]), supplied)
        for ep in all_ids
    }
    if data.aft_action_state_step_offset:
        for ep in all_ids:
            validate_next_state_labels(all_rows[ep], edges[ep], ep)
    true_edges = sum(int(e.sum()) for e in edges.values())
    false_edges = sum(int((~e).sum()) for e in edges.values())
    print(f"Audited {len(edges)} episodes: {true_edges} contiguous and {false_edges} broken source edges")
    counts = [window_target_counts(e, horizon=c.model.action_horizon,
                                  action_offset=data.aft_action_state_step_offset) for e in edges.values()]
    reports = [r for r in conversion["timing_reports"] if r["output_episode_index"] in all_ids]
    audit_summary = {
        "episodes": len(all_ids),
        "retained_frames": sum(len(r["state"]) for r in all_rows.values()),
        "removed_supervised_samples": sum(r.get("removed_supervised_samples", 0) for r in reports),
        "missing_candidate_steps": sum(r.get("missing_candidate_steps", 0) for r in reports),
        "continuous_edges": true_edges,
        "broken_edges": false_edges,
        "valid_action_targets": sum(a for a, _ in counts),
        "valid_sensor_targets": sum(s for _, s in counts),
    }
    print(f"AFT audit summary: {json.dumps(audit_summary)}")
    train = {ep: all_rows[ep] for ep in data.episodes}
    stats = collect_statistics(
        train, edges, horizon=c.model.action_horizon, action_offset=data.aft_action_state_step_offset
    )
    output.mkdir(parents=True)
    normalize.save(output, stats)
    (output / "adjacency.json").write_text(json.dumps({str(k): v.tolist() for k, v in edges.items()}))
    manifest = dict(
        config=config,
        horizon=c.model.action_horizon,
        force_history=8,
        train_episodes=data.episodes,
        validation_episodes=data.validation_episodes,
        source_root=str(root),
        action_source_mode=expected_mode,
        sensor_offset=1,
        conversion_sha256=hashlib.sha256(conversion_path.read_bytes()).hexdigest(),
        norm_stats_sha256=hashlib.sha256((output / "norm_stats.json").read_bytes()).hexdigest(),
        units={"wrench": ["N"] * 3 + ["N_m"] * 3, "frame": "K", "shear": "marker_coordinate_displacement"},
        initialization=dict(backbone=c.weight_loader.backbone_params_path, tactile=c.weight_loader.tactile_params_path),
        model_contract=model_contract(c.model),
        adjacency_sha256=hashlib.sha256((output / "adjacency.json").read_bytes()).hexdigest(),
        source_mapping_sha256={
            str(ep): hashlib.sha256(source_mapping_path(root, ep).read_bytes()).hexdigest()
            for ep in all_ids
            if source_mapping_path(root, ep).exists()
        },
        audit_summary=audit_summary,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"AFT train-only assets saved: {output}; training was not started.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--edges-json")
    main(**vars(parser.parse_args()))
