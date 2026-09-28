"""Draw saved AFT diagnostic bundles without a model, GPU or robot connection."""

import argparse
from pathlib import Path

import numpy as np

from openpi.policies.aft_diagnostics import load_bundle


def plot_bundle(prefix, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    response, _ = load_bundle(prefix)
    d = response["diagnostics"]
    if d.get("schema") != "aft_diagnostics_v1":
        raise ValueError("Expected aft_diagnostics_v1")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = []

    def save(fig, name):
        path = output / name
        if path.exists():
            plt.close(fig)
            raise FileExistsError(f"Refusing to overwrite {path}")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        files.append(path)

    attention = np.asarray(d["attention"])
    query_ids, slots = np.asarray(d["query_stream_id"]), np.asarray(d["query_horizon_index"])
    key_ids = np.asarray(d["key_group_id"])
    names = d["key_group_names"]
    streams = d["query_stream_names"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), layout="constrained")
    for stream, ax in enumerate(axes):
        rows = query_ids == stream
        matrix = attention[:, rows].mean(0)
        im = ax.imshow(matrix, aspect="auto", origin="lower", vmin=0,
                       vmax=max(float(attention.max()), 1e-6), interpolation="nearest")
        ticks, labels = [], []
        for group, name in enumerate(names):
            keys = np.flatnonzero(key_ids == group)
            if len(keys):
                ticks.append(float(keys.mean()))
                labels.append(name)
                ax.axvline(keys[0] - 0.5, color="white", linewidth=0.5)
        ax.set_xticks(ticks, labels, rotation=30, ha="right")
        idx = np.linspace(0, len(matrix) - 1, min(6, len(matrix))).astype(int)
        ax.set_yticks(idx, slots[rows][idx])
        ax.set_ylabel(f"{streams[stream]} query slot")
        fig.colorbar(im, ax=ax, label="head/layer mean attention")
    fig.suptitle(f"Selected layers {d['layer_indices']}, denoise step {d['denoise_step']} — not causal importance")
    save(fig, "attention_heatmap.png")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), layout="constrained")
    for stream in range(3):
        rows = query_ids == stream
        x = slots[rows]
        for column, key in enumerate(("modality_mass", "modality_per_token")):
            values = np.asarray(d[key])[:, rows].mean(0)
            for group, name in enumerate(names):
                if d["valid_key_counts"][group]:
                    axes[stream, column].plot(x, values[:, group], label=f"{name} (n={d['valid_key_counts'][group]})")
            kind = "total mass" if column == 0 else "mean per valid key"
            axes[stream, column].set_title(f"{streams[stream]}: {kind}")
            axes[stream, column].set_xlabel("Prediction slot (zero-based)")
            axes[stream, column].grid(alpha=0.2)
    axes[0, 1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "modality_attention.png")

    if d["ablations"]:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout="constrained")
        for name, delta in d["ablations"].items():
            for ax, key in zip(axes, ("position_delta_mm", "rotation_delta_deg", "gripper_delta_mm"), strict=True):
                ax.plot(np.arange(len(delta[key])), delta[key], label=name)
                ax.set_ylabel(key)
                ax.grid(alpha=0.2)
        axes[0].legend()
        axes[-1].set_xlabel("Prediction slot (zero-based)")
        fig.suptitle("Fixed-noise normalized-history perturbation — sensitivity, not prediction error")
        save(fig, "history_sensitivity.png")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-prefix", required=True, help="Shared filename prefix, without .json/.npz")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    for path in plot_bundle(args.bundle_prefix, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
