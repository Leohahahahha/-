"""Explicit, episode-local physical target windows for the A/F/T slow model.

Compact timestamps are not evidence that the original trajectory was continuous.
Callers must supply audited adjacency flags for original sample intervals.
"""

import numpy as np
import json
import pathlib
import functools
import math

from openpi.models.aft_types import AFTTargets


def audit_edges(conversion, episode, length, supplied=None):
    reports = conversion.get("timing_reports", [])
    report = next((r for r in reports if r.get("output_episode_index") == episode), None)
    omitted_terminal = conversion.get("terminal_frame_policy") in (
        "omit final source observation; use it only as the previous frame action target",
        "final source observation omitted to keep rows identical to next-state exports",
    )
    if report is None or report.get("frames") != length + int(omitted_terminal):
        raise ValueError(f"episode {episode}: missing/mismatched conversion timing report")
    if supplied is not None and str(episode) in supplied:
        edges = np.asarray(supplied[str(episode)])
        if edges.shape != (length - 1,) or edges.dtype != bool:
            raise ValueError("audited adjacency must be boolean [frames-1]")
        if report.get("missing_candidate_steps", 0) > 0 and edges.all():
            raise ValueError("audited adjacency contradicts reported missing steps")
        return edges
    if report.get("missing_candidate_steps") == 0:
        return np.ones(length - 1, bool)
    raise ValueError(f"episode {episode}: compact gap positions unknown; require audited adjacency JSON")


def source_mapping_path(root, episode):
    return pathlib.Path(root) / "meta/source_mapping" / f"episode_{episode:06d}.json"


def _is_contiguous(current, following):
    """Require both source-frame adjacency and the action's actual next observation."""
    return (
        following["source_row_index"] == current["source_row_index"] + 1
        and following["source_frame_index"] == current["source_frame_index"] + 1
        and current["action_source_row_index"] == following["source_row_index"]
        and current["action_source_frame_index"] == following["source_frame_index"]
        and math.isclose(following["source_timestamp"] - current["source_timestamp"], 0.1, abs_tol=0.005)
        and math.isclose(current["action_source_timestamp"], following["source_timestamp"], abs_tol=0.005)
    )


def audit_episode_edges(root, conversion, episode, length, supplied=None):
    """Audit output adjacency against original source provenance, not compact output time."""
    mapping_path = source_mapping_path(root, episode)
    report_has_removed_rows = any(
        r.get("output_episode_index") == episode and r.get("removed_supervised_samples", 0) > 0
        for r in conversion.get("timing_reports", [])
    )
    needs_mapping = (
        report_has_removed_rows or bool(conversion.get("frame_exclusion")) or conversion.get("version", 0) >= 4
    )
    if not mapping_path.exists():
        if needs_mapping:
            raise ValueError(f"episode {episode}: source mapping is required for filtered data")
        return audit_edges(conversion, episode, length, supplied)

    reports = [r for r in conversion.get("timing_reports", []) if r.get("output_episode_index") == episode]
    if len(reports) != 1:
        raise ValueError(f"episode {episode}: missing or duplicate conversion timing report")
    report = reports[0]
    omitted_terminal = conversion.get("terminal_frame_policy") in (
        "omit final source observation; use it only as the previous frame action target",
        "final source observation omitted to keep rows identical to next-state exports",
    )
    if report.get("frames") != length + int(omitted_terminal) + report.get("removed_supervised_samples", 0):
        raise ValueError(f"episode {episode}: conversion timing report frames mismatch")
    mapping = json.loads(mapping_path.read_text())
    rows = mapping.get("rows")
    if mapping.get("source_episode_index") != report.get("source_episode_index") or not isinstance(rows, list):
        raise ValueError(f"episode {episode}: source mapping does not match timing report")
    if len(rows) != length:
        raise ValueError(f"episode {episode}: source mapping length mismatch")
    required_ints = (
        "output_frame_index",
        "source_row_index",
        "source_frame_index",
        "action_source_row_index",
        "action_source_frame_index",
    )
    required_times = ("source_timestamp", "action_source_timestamp")
    for i, row in enumerate(rows):
        if any(type(row.get(k)) is not int or row[k] < 0 for k in required_ints):
            raise ValueError(f"episode {episode}: invalid source/action_source index at row {i}")
        if row["output_frame_index"] != i:
            raise ValueError(f"episode {episode}: output_frame_index is not dense at row {i}")
        if any(not isinstance(row.get(k), (float, int)) or not math.isfinite(row[k]) for k in required_times):
            raise ValueError(f"episode {episode}: invalid source/action_source timestamp at row {i}")
        if (
            row["action_source_row_index"] <= row["source_row_index"]
            or row["action_source_frame_index"] <= row["source_frame_index"]
            or row["action_source_timestamp"] <= row["source_timestamp"]
        ):
            raise ValueError(f"episode {episode}: action_source must follow source row {i}")
        if i and (
            row["source_row_index"] <= rows[i - 1]["source_row_index"]
            or row["source_frame_index"] <= rows[i - 1]["source_frame_index"]
            or row["source_timestamp"] <= rows[i - 1]["source_timestamp"]
        ):
            raise ValueError(f"episode {episode}: source mapping must increase strictly")
    edges = np.asarray([_is_contiguous(a, b) for a, b in zip(rows, rows[1:])], dtype=bool)
    if supplied is not None and str(episode) in supplied:
        provided = np.asarray(supplied[str(episode)])
        if provided.shape != edges.shape or provided.dtype != bool or not np.array_equal(provided, edges):
            raise ValueError(f"episode {episode}: supplied adjacency contradicts source mapping")
    return edges


@functools.lru_cache(maxsize=2)
def read_episode(root, episode):
    import pyarrow.parquet as pq

    root = pathlib.Path(root)
    info = json.loads((root / "meta/info.json").read_text())
    path = info["data_path"].format(episode_chunk=episode // info.get("chunks_size", 1000), episode_index=episode)
    table = pq.read_table(
        root / path, columns=["state", "actions", "wrist_wrench", "tactile_marker_motion", "episode_index"]
    )
    rows = {key: np.asarray(table[key].to_pylist()) for key in table.column_names}
    rows["tactile_marker_motion"] = rows["tactile_marker_motion"].reshape(-1, 9, 198, 2)
    for key in ("state", "actions", "wrist_wrench", "tactile_marker_motion"):
        rows[key] = rows[key].astype(np.float32)
    return rows


class AFTDataset:
    """Use LeRobot for current RGB only, explicit episode-local arrays for targets."""

    def __init__(self, dataset, data_config, horizon):
        self.dataset, self.config, self.horizon = dataset, data_config, horizon
        self.conversion = json.loads((pathlib.Path(data_config.root) / "meta/tabero_conversion.json").read_text())
        expected = "next-state" if data_config.aft_action_state_step_offset else "sent-command"
        if self.conversion.get("action_source_mode") != expected:
            raise ValueError(f"Action contract mismatch: expected {expected}")
        self.supplied = (
            json.loads(pathlib.Path(data_config.aft_edges_path).read_text()) if data_config.aft_edges_path else None
        )
        self.edges = {}
        self.rows = {}
        for ep in data_config.episodes:
            rows = read_episode(data_config.root, ep)
            self.rows[ep] = rows
            self.edges[ep] = audit_episode_edges(
                data_config.root, self.conversion, ep, len(rows["state"]), self.supplied
            )
            if data_config.aft_action_state_step_offset:
                # Validate only known-contiguous transitions; compare physical SO(3), not rotvec branches.
                from scipy.spatial.transform import Rotation

                valid = self.edges[ep]
                a, s = rows["actions"][:-1][valid], rows["state"][1:][valid]
                if not np.allclose(a[:, [0, 1, 2, 6]], s[:, [0, 1, 2, 6]], atol=1e-5):
                    raise ValueError(f"episode {ep}: next_state actions disagree with next state")
                if (
                    len(a)
                    and np.max((Rotation.from_rotvec(a[:, 3:6]).inv() * Rotation.from_rotvec(s[:, 3:6])).magnitude())
                    > 1e-4
                ):
                    raise ValueError(f"episode {ep}: next_state rotations disagree")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = dict(self.dataset[index])
        ep, anchor = int(item["episode_index"]), int(item["frame_index"])
        window = build_episode_windows(
            self.rows[ep],
            anchor,
            self.horizon,
            contiguous_edges=self.edges[ep],
            action_state_step_offset=self.config.aft_action_state_step_offset,
        )
        item.update(window)
        item["tactile_marker_motion"] = window["marker_history"]
        return item


def build_episode_windows(
    rows, anchor, horizon=50, force_history=8, sensor_offset=1, contiguous_edges=None, action_state_step_offset=0
):
    if horizon < 1 or force_history < 1 or sensor_offset != 1:
        raise ValueError("Positive horizon/history and next-observation sensor_offset=1 required")
    state = np.asarray(rows["state"])
    n = len(state)
    if not 0 <= anchor < n:
        raise ValueError("anchor outside episode")
    if contiguous_edges is None:
        raise ValueError("Audited contiguous_edges required; compact timestamps are insufficient")
    edges = np.asarray(contiguous_edges)
    if edges.shape != (n - 1,) or edges.dtype != bool:
        raise ValueError("contiguous_edges must be boolean [episode_length-1]")
    if action_state_step_offset not in (0, 1):
        raise ValueError("action_state_step_offset must be 0 or 1")
    episodes = np.asarray(rows["episode_index"])
    if episodes.shape != (n,) or np.unique(episodes).size != 1:
        raise ValueError("Exactly one episode required")
    arrays = {key: np.asarray(rows[key]) for key in ("state", "actions", "wrist_wrench", "tactile_marker_motion")}
    shapes = {"state": (n, 7), "actions": (n, 7), "wrist_wrench": (n, 6), "tactile_marker_motion": (n, 9, 198, 2)}
    for key, value in arrays.items():
        if value.shape != shapes[key] or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f"{key} must be floating {shapes[key]}, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite")
    past = anchor + np.arange(1 - force_history, 1)
    # Do not pool force measurements from before a known temporal discontinuity.
    gaps_before = np.flatnonzero(~edges[:anchor])
    segment_start = int(gaps_before[-1] + 1) if gaps_before.size else 0
    past_valid = past >= segment_start
    past = np.clip(past, segment_start, anchor)
    action_indices = anchor + np.arange(horizon)
    future = action_indices + sensor_offset

    def valid_through(indices):
        result = (indices >= anchor) & (indices < n)
        for i, end in enumerate(indices):
            result[i] &= bool(edges[anchor : min(int(end), n - 1)].all())
        return result

    sensor_mask = valid_through(future)
    action_mask = valid_through(action_indices + action_state_step_offset)
    future_clipped = np.minimum(future, n - 1)
    motion = arrays["tactile_marker_motion"][future_clipped]
    return dict(
        force_history=arrays["wrist_wrench"][past].astype(np.float32),
        force_history_mask=past_valid,
        marker_history=arrays["tactile_marker_motion"][anchor].astype(np.float32),
        targets=AFTTargets(
            actions=arrays["actions"][np.minimum(action_indices, n - 1)].astype(np.float32),
            shear=(motion[:, -1] - motion[:, 0]).reshape(horizon, 396).astype(np.float32),
            wrench=arrays["wrist_wrench"][future_clipped].astype(np.float32),
            action_mask=action_mask,
            sensor_mask=sensor_mask,
        ),
    )
