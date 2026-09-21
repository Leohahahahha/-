#!/usr/bin/env python3
"""Overlay two whiteboard wrench predictions against the same recorded targets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-ft", type=Path, required=True, help="Full-FT predictions.npz")
    parser.add_argument("--lora", type=Path, required=True, help="LoRA predictions.npz")
    parser.add_argument("--output", type=Path, required=True, help="Output path without suffix")
    return parser.parse_args()


def load_and_align(full_path: Path, lora_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    full_npz = np.load(full_path)
    lora_npz = np.load(lora_path)
    required = (
        "episode",
        "frame",
        "valid",
        "state",
        "target",
        "wrench_prediction",
        "wrench_target",
    )
    for key in required:
        if key not in full_npz.files or key not in lora_npz.files:
            raise KeyError(f"Missing required array: {key}")
    full = {key: full_npz[key] for key in required}
    lora = {key: lora_npz[key] for key in required}

    for key in ("episode", "frame", "valid"):
        if not np.array_equal(full[key], lora[key]):
            raise ValueError(f"Evaluations are not aligned: {key} differs")
    for key in ("state", "target", "wrench_target"):
        if not np.array_equal(full[key], lora[key]):
            difference = float(np.max(np.abs(full[key] - lora[key])))
            raise ValueError(f"Evaluations are not aligned: {key} max difference={difference}")
    if full["wrench_prediction"].shape != lora["wrench_prediction"].shape:
        raise ValueError("Wrench prediction shapes differ")
    return full, lora


def plot(full: dict[str, np.ndarray], lora: dict[str, np.ndarray], output: Path) -> None:
    episodes = np.unique(full["episode"])
    target = full["wrench_target"][:, 0]
    full_prediction = full["wrench_prediction"][:, 0]
    lora_prediction = lora["wrench_prediction"][:, 0]
    if not np.isfinite((target, full_prediction, lora_prediction)).all():
        raise ValueError("Non-finite wrench value")

    colors = {"target": "#111111", "full": "#D55E00", "lora": "#0072B2"}
    figure, axes = plt.subplots(2 * len(episodes), 4, figsize=(21, 5.6 * len(episodes)), squeeze=False)

    for episode_index, episode in enumerate(episodes):
        mask = full["episode"] == episode
        frames = full["frame"][mask]
        episode_target = target[mask]
        episode_full = full_prediction[mask]
        episode_lora = lora_prediction[mask]
        force_row = 2 * episode_index
        torque_row = force_row + 1

        for component, name in enumerate(("Fx", "Fy", "Fz")):
            axis = axes[force_row, component]
            axis.plot(frames, episode_target[:, component], color=colors["target"], linewidth=2.0,
                      label="Recorded target")
            axis.plot(frames, episode_full[:, component], color=colors["full"], linewidth=1.2,
                      alpha=0.9, label="Mixed Full-FT 20k")
            axis.plot(frames, episode_lora[:, component], color=colors["lora"], linewidth=1.2,
                      alpha=0.9, label="LoRA32 12k")
            axis.set_ylabel(f"{name} (N)")

        for offset, name in enumerate(("Tx", "Ty", "Tz"), start=3):
            axis = axes[torque_row, offset - 3]
            axis.plot(frames, episode_target[:, offset], color=colors["target"], linewidth=2.0,
                      label="Recorded target")
            axis.plot(frames, episode_full[:, offset], color=colors["full"], linewidth=1.2,
                      alpha=0.9, label="Mixed Full-FT 20k")
            axis.plot(frames, episode_lora[:, offset], color=colors["lora"], linewidth=1.2,
                      alpha=0.9, label="LoRA32 12k")
            axis.set_ylabel(f"{name} (N m)")

        full_force_error = np.linalg.norm(episode_full[:, :3] - episode_target[:, :3], axis=1)
        lora_force_error = np.linalg.norm(episode_lora[:, :3] - episode_target[:, :3], axis=1)
        force_error_axis = axes[force_row, 3]
        force_error_axis.plot(frames, full_force_error, color=colors["full"], linewidth=1.2,
                              label="Mixed Full-FT 20k")
        force_error_axis.plot(frames, lora_force_error, color=colors["lora"], linewidth=1.2,
                              label="LoRA32 12k")
        force_error_axis.set_ylabel("Force L2 error (N)")
        force_error_axis.set_title(
            f"Mean: {full_force_error.mean():.3f} vs {lora_force_error.mean():.3f} N",
            fontsize=10,
        )

        full_torque_error = np.linalg.norm(episode_full[:, 3:] - episode_target[:, 3:], axis=1)
        lora_torque_error = np.linalg.norm(episode_lora[:, 3:] - episode_target[:, 3:], axis=1)
        torque_error_axis = axes[torque_row, 3]
        torque_error_axis.plot(frames, full_torque_error, color=colors["full"], linewidth=1.2,
                               label="Mixed Full-FT 20k")
        torque_error_axis.plot(frames, lora_torque_error, color=colors["lora"], linewidth=1.2,
                               label="LoRA32 12k")
        torque_error_axis.set_ylabel("Torque L2 error (N m)")
        torque_error_axis.set_title(
            f"Mean: {full_torque_error.mean():.3f} vs {lora_torque_error.mean():.3f} N m",
            fontsize=10,
        )

        axes[force_row, 0].text(
            -0.23, 0.5, f"Episode {int(episode)}\nForce", transform=axes[force_row, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=12, fontweight="bold"
        )
        axes[torque_row, 0].text(
            -0.23, 0.5, f"Episode {int(episode)}\nTorque", transform=axes[torque_row, 0].transAxes,
            rotation=90, va="center", ha="center", fontsize=12, fontweight="bold"
        )

        for row in (force_row, torque_row):
            for axis in axes[row]:
                axis.set_xlabel("Recorded frame (10 Hz compact timeline)")
                axis.grid(alpha=0.22)
                axis.margins(x=0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    figure.suptitle(
        "K-frame wrist-wrench comparison on identical held-out trajectories\n"
        "Open-loop first prediction at each recorded observation (not force-control execution)",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.07, right=0.99, bottom=0.04, top=0.92, wspace=0.23, hspace=0.43)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output.with_suffix(".png"), dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    full, lora = load_and_align(args.full_ft, args.lora)
    plot(full, lora, args.output)
    print(f"Saved {args.output.with_suffix('.png')}")
    print(f"Saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
