"""Plot saved rollout metrics without loading checkpoints or running inference."""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(root=None, source=None, *, three_models=False):
    root = root or Path(__file__).resolve().parent
    source = source or root.parent / "gkl_differentiable_vs_detached_reuse_20261119"
    metrics = {
        "mse": "Actual MSE",
        "mean_u": "Mean predicted uncertainty",
        "calibration": "Mean run-level u/MSE",
        "spatial_spearman": "Within-frame Spearman",
    }
    protocols = (
        ("sliding_summary.csv", "axis", "step", "index", "Sliding-window update"),
        ("repeated_reuse_summary.csv", "task", "spatial_grid", "step", "Fixed spatial grid pass"),
        ("repeated_reuse_summary.csv", "task", "temporal_block", "step", "Fixed temporal block pass"),
    )
    models = (("detached", "Detached reuse", "o"), ("differentiable", "Fully differentiable reuse", "s"))
    if three_models:
        models = (models[0], ("mean_only", "Mean-only differentiable reuse", "^"), models[1])
        protocols += (
            ("sliding_summary.csv", "axis", "age", "index", "Frames since last genuine probe"),
            ("forecast_plot.csv", "axis", "forecast", "index", "Forecast horizon"),
        )
    for file, selector, selection, step_key, title in protocols:
        with (source / file).open() as stream:
            rows = [row for row in csv.DictReader(stream) if row[selector] == selection]
        fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex="col", layout="constrained")
        for col, channel in enumerate(("Magnetic", "Density")):
            axes[0, col].set_title(channel)
            for ax, (metric, ylabel) in zip(axes[:, col], metrics.items()):
                series = sorted(
                    (row for row in rows if row["channel"] == channel and row["metric"] == metric),
                    key=lambda row: int(row[step_key]),
                )
                steps = [int(row[step_key]) for row in series]
                for key, label, marker in models:
                    values = [float(row[key]) for row in series]
                    ax.plot(steps, values, marker=marker, label=label)
                    if selection == "step" and not three_models:
                        print(channel, metric, key, *(f"{value:.4f}" for value in values))
                if metric == "calibration":
                    ax.axhline(1, color="0.5", linestyle=":", linewidth=1)
                if metric == "spatial_spearman":
                    ax.axhline(0, color="0.5", linestyle=":", linewidth=1)
                    ax.set_xlabel(title)
                ax.set_xticks(steps)
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.savefig(root / f"{selection}_rollout.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
