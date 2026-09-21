#!/usr/bin/env python3
"""Overlay two whiteboard offline evaluations against the same recorded actions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-ft", type=Path, required=True, help="Full-FT predictions.npz")
    parser.add_argument("--lora", type=Path, required=True, help="LoRA predictions.npz")
    parser.add_argument("--output", type=Path, required=True, help="Output path without suffix")
    return parser.parse_args()


def rotation_error_deg(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = Rotation.from_rotvec(prediction.copy()).inv() * Rotation.from_rotvec(target.copy())
    return np.rad2deg(relative.magnitude())


def load_and_align(full_path: Path, lora_path: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    full_npz = np.load(full_path)
    lora_npz = np.load(lora_path)
    required = ("episode", "frame", "prediction", "target", "valid", "state")
    for key in required:
        if key not in full_npz.files or key not in lora_npz.files:
            raise KeyError(f"Missing required array: {key}")
    full = {key: full_npz[key] for key in required}
    lora = {key: lora_npz[key] for key in required}

    for key in ("episode", "frame", "valid"):
        if not np.array_equal(full[key], lora[key]):
            raise ValueError(f"Evaluations are not aligned: {key} differs")
    for key in ("target", "state"):
        if not np.array_equal(full[key], lora[key]):
            difference = float(np.max(np.abs(full[key] - lora[key])))
            raise ValueError(f"Evaluations are not aligned: {key} max difference={difference}")
    if full["prediction"].shape != lora["prediction"].shape:
        raise ValueError("Prediction shapes differ")
    return full, lora


def plot(full: dict[str, np.ndarray], lora: dict[str, np.ndarray], output: Path) -> None:
    episodes = np.unique(full["episode"])
    if episodes.size == 0:
        raise ValueError("No evaluation records")

    target = full["target"][:, 0]
    full_prediction = full["prediction"][:, 0]
    lora_prediction = lora["prediction"][:, 0]
    if not np.isfinite((target, full_prediction, lora_prediction)).all():
        raise ValueError("Non-finite action value")

    colors = {"target": "#111111", "full": "#D55E00", "lora": "#0072B2"}
    figure, axes = plt.subplots(
        len(episodes),
        6,
        figsize=(29, 4.4 * len(episodes)),
        squeeze=False,
    )
    component_specs = ((0, "X (mm)"), (1, "Y (mm)"), (2, "Z (mm)"), (6, "Single finger (mm)"))

    for row, episode in enumerate(episodes):
        mask = full["episode"] == episode
        frames = full["frame"][mask]
        episode_target = target[mask]
        episode_full = full_prediction[mask]
        episode_lora = lora_prediction[mask]
        full_position_error = 1000.0 * np.linalg.norm(episode_full[:, :3] - episode_target[:, :3], axis=1)
        lora_position_error = 1000.0 * np.linalg.norm(episode_lora[:, :3] - episode_target[:, :3], axis=1)
        full_rotation_error = rotation_error_deg(episode_full[:, 3:6], episode_target[:, 3:6])
        lora_rotation_error = rotation_error_deg(episode_lora[:, 3:6], episode_target[:, 3:6])

        for column, (dimension, ylabel) in enumerate(component_specs):
            axis = axes[row, column]
            axis.plot(frames, episode_target[:, dimension] * 1000.0, color=colors["target"], linewidth=2.0,
                      label="Recorded target")
            axis.plot(frames, episode_full[:, dimension] * 1000.0, color=colors["full"], linewidth=1.25,
                      alpha=0.9, label="Mixed Full-FT 20k")
            axis.plot(frames, episode_lora[:, dimension] * 1000.0, color=colors["lora"], linewidth=1.25,
                      alpha=0.9, label="LoRA32 12k")
            axis.set_ylabel(ylabel)

        position_axis = axes[row, 4]
        position_axis.plot(frames, full_position_error, color=colors["full"], linewidth=1.25,
                           label="Mixed Full-FT 20k")
        position_axis.plot(frames, lora_position_error, color=colors["lora"], linewidth=1.25,
                           label="LoRA32 12k")
        position_axis.set_ylabel("Position L2 error (mm)")
        position_axis.set_title(
            f"Mean: {full_position_error.mean():.2f} vs {lora_position_error.mean():.2f} mm",
            fontsize=10,
        )

        rotation_axis = axes[row, 5]
        rotation_axis.plot(frames, full_rotation_error, color=colors["full"], linewidth=1.25,
                           label="Mixed Full-FT 20k")
        rotation_axis.plot(frames, lora_rotation_error, color=colors["lora"], linewidth=1.25,
                           label="LoRA32 12k")
        rotation_axis.set_ylabel("SO(3) rotation error (deg)")
        rotation_axis.set_title(
            f"Mean: {full_rotation_error.mean():.3f} vs {lora_rotation_error.mean():.3f} deg",
            fontsize=10,
        )

        axes[row, 0].text(
            -0.30,
            0.5,
            f"Episode {int(episode)}",
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=13,
            fontweight="bold",
        )
        for axis in axes[row]:
            axis.set_xlabel("Recorded frame (10 Hz compact timeline)")
            axis.grid(alpha=0.22)
            axis.margins(x=0)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    figure.suptitle(
        "Next-state action comparison on identical held-out trajectories\n"
        "Open-loop first action at each recorded observation (not a rollout)",
        y=0.995,
        fontsize=16,
        fontweight="bold",
    )
    figure.subplots_adjust(left=0.055, right=0.992, bottom=0.055, top=0.885, wspace=0.24, hspace=0.36)
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
