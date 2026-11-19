#!/usr/bin/env python3
"""Combine the paired Gaussian inference with the archived Student-t evaluation."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

PROJECT = Path("/pscratch/sd/b/binxia/ai4plasma")
OUT = PROJECT / "evaluations/gaussian_u_continuation_20ep_analysis_20261116"
REF = PROJECT / "evaluations/three_phase_256ep_analysis_20261116"
STUDENT_RUN = PROJECT / "runs/multi_beta_curriculum_direct_v3_three_phase_globalcosine_2e4_1e5_256ep_20261115"
GAUSSIAN_RUN = PROJECT / "runs/multi_beta_three_phase_gaussian_u_continuation_20ep_20261116"
MODELS = ("Student-t epoch 256", "Gaussian epoch 276")
HIGHLIGHTS = (1, 6, 12, 18, 23)


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


student = read_csv(REF / "forecast_horizon_metrics.csv")
gaussian = read_csv(OUT / "forecast_horizon_metrics_gaussian.csv")
student_by = {(row["channel"], int(row["horizon"])): row for row in student}
gaussian_by = {(row["channel"], int(row["horizon"])): row for row in gaussian}
if student_by.keys() != gaussian_by.keys():
    raise RuntimeError("Forecast metric coordinates do not match")

combined = []
for label, rows in zip(MODELS, (student, gaussian)):
    for row in rows:
        key = (row["channel"], int(row["horizon"]))
        s, g = student_by[key], gaussian_by[key]
        combined.append({
            "model": label, **row,
            "gaussian_u_over_student_t_u": float(g["mean_predicted_u"]) / float(s["mean_predicted_u"]),
            "gaussian_mse_over_student_t_mse": float(g["mean_mse"]) / float(s["mean_mse"]),
        })
write_csv(OUT / "forecast_horizon_metrics.csv", combined)

summary = {}
for channel in ("Bx", "By", "Bz", "Magnetic", "Density", "All"):
    summary[channel] = {}
    inflation = []
    for label, values in zip(MODELS, (student_by, gaussian_by)):
        base = values[(channel, 1)]
        horizons = np.arange(1, 24)
        mse = np.array([float(values[(channel, int(h))]["mean_mse"]) for h in horizons])
        variance = np.array([float(values[(channel, int(h))]["mean_predicted_u"]) for h in horizons])
        summary[channel][label] = {
            "ratios": {
                str(h): {
                    "mse_over_h1": float(values[(channel, h)]["mean_mse"]) / float(base["mean_mse"]),
                    "u_over_h1": float(values[(channel, h)]["mean_predicted_u"]) / float(base["mean_predicted_u"]),
                    "u_over_mse": float(values[(channel, h)]["predicted_u_over_actual_mse"]),
                } for h in (6, 12, 18, 23)
            },
            "u_over_mse_h1": float(base["predicted_u_over_actual_mse"]),
            "spearman_h_mse": float(stats.spearmanr(horizons, mse).statistic),
            "spearman_h_u": float(stats.spearmanr(horizons, variance).statistic),
            "spearman_horizonlevel_mse_u": float(stats.spearmanr(mse, variance).statistic),
            "pearson_horizonlevel_mse_u": float(stats.pearsonr(mse, variance).statistic),
        }
    inflation = np.array([
        float(gaussian_by[(channel, h)]["mean_predicted_u"])
        / float(student_by[(channel, h)]["mean_predicted_u"])
        for h in range(1, 24)
    ])
    summary[channel]["gaussian_over_student_t"] = {
        "by_horizon": {str(h): float(inflation[h - 1]) for h in range(1, 24)},
        "spearman_horizon_ratio": float(stats.spearmanr(np.arange(1, 24), inflation).statistic),
    }

# Directly comparable fixed validation masks from the final training histories.
student_history = json.loads((STUDENT_RUN / "history.json").read_text())[-1]
gaussian_history = json.loads((GAUSSIAN_RUN / "history.json").read_text())[-1]
standard_rows = []
for pattern in student_history["val_per_pattern"]:
    s = student_history["val_per_pattern"][pattern]["mse"]
    g = gaussian_history["val_per_pattern"][pattern]["mse"]
    standard_rows.extend([
        {"model": MODELS[0], "pattern": pattern, "reconstruction_mse": s,
         "gaussian_vs_student_t_percent": 100 * (g / s - 1)},
        {"model": MODELS[1], "pattern": pattern, "reconstruction_mse": g,
         "gaussian_vs_student_t_percent": 100 * (g / s - 1)},
    ])
standard_rows.extend([
    {"model": MODELS[0], "pattern": "aggregate", "reconstruction_mse": student_history["val_mse"],
     "gaussian_vs_student_t_percent": 100 * (gaussian_history["val_mse"] / student_history["val_mse"] - 1)},
    {"model": MODELS[1], "pattern": "aggregate", "reconstruction_mse": gaussian_history["val_mse"],
     "gaussian_vs_student_t_percent": 100 * (gaussian_history["val_mse"] / student_history["val_mse"] - 1)},
])
write_csv(OUT / "standard_mask_metrics.csv", standard_rows)

# Validation predicted-u distribution logged over the complete fixed-mask sweep.
distribution = []
for channel in ("Bx", "By", "Bz", "Density"):
    for label, history in zip(MODELS, (student_history, gaussian_history)):
        values = history["val_variance_statistics"][channel]
        distribution.append({"model": label, "channel": channel, **values})
write_csv(OUT / "predicted_u_distribution.csv", distribution)

# Same matched binary pass-0 versus continuous-u pass-1 test used previously.
student_standard = read_csv(REF / "standard_mask_metrics.csv")
gaussian_standard = read_csv(OUT / "standard_mask_detailed_gaussian.csv")
pass_rows = []
for label, rows in zip(MODELS, (student_standard, gaussian_standard)):
    selected = {
        (row["source_task"], row["pass"]): row for row in rows
        if row["role"] == "test" and row["channel"] == "All"
    }
    for source in ("spatial_random", "spatial_grid", "spatial_block", "temporal_random", "temporal_block"):
        pass1 = selected[(source, "pass1")]
        effective = pass1["effective_observation_task"]
        pass0 = selected[(effective, "pass0")]
        pass_rows.append({
            "model": label, "source_task": source, "effective_observation_task": effective,
            "matched_pass0_mse": pass0["mean_mse"], "continuous_reuse_pass1_mse": pass1["mean_mse"],
            "pass1_over_pass0_mse": float(pass1["mean_mse"]) / float(pass0["mean_mse"]),
            "matched_pass0_u_over_mse": pass0["predicted_u_over_actual_mse"],
            "continuous_reuse_pass1_u_over_mse": pass1["predicted_u_over_actual_mse"],
        })
write_csv(OUT / "pass0_pass1_metrics.csv", pass_rows)

# Bounded residual samples from identical windows/horizons.
residual_rows = []
for label, path in (
    (MODELS[0], REF / "forecast_residual_likelihood.csv"),
    (MODELS[1], OUT / "forecast_residual_likelihood_gaussian.csv"),
):
    residual_rows.extend({"model": label, **row} for row in read_csv(path))
write_csv(OUT / "forecast_residual_likelihood.csv", residual_rows)

# Horizon response plot.
fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
for row, channel in enumerate(("Density", "Magnetic")):
    for label, values, style in zip(MODELS, (student_by, gaussian_by), ("o-", "s--")):
        h = np.arange(1, 24)
        axes[row, 0].plot(h, [float(values[(channel, int(x))]["mean_mse"]) for x in h], style, ms=3, label=label)
        axes[row, 1].plot(h, [float(values[(channel, int(x))]["mean_predicted_u"]) for x in h], style, ms=3, label=label)
    axes[row, 0].set(ylabel=f"{channel} actual MSE", yscale="log")
    axes[row, 1].set(ylabel=f"{channel} mean predicted u", yscale="log")
for axis in axes.flat:
    axis.set(xlabel="hidden forecast horizon h")
    axis.legend(frameon=False)
fig.savefig(OUT / "gaussian_vs_studentt_horizon.png", dpi=180)
plt.close(fig)

# Calibration and explicit Gaussian/Student-t inflation ratio.
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
for axis, channel in zip(axes[:2], ("Density", "Magnetic")):
    for label, values, style in zip(MODELS, (student_by, gaussian_by), ("o-", "s--")):
        h = np.arange(1, 24)
        axis.plot(h, [float(values[(channel, int(x))]["predicted_u_over_actual_mse"]) for x in h], style, ms=3, label=label)
    axis.axhline(1, color="black", ls=":", lw=1)
    axis.set(xlabel="hidden forecast horizon h", ylabel=f"{channel} u / MSE", yscale="log")
    axis.legend(frameon=False)
for channel in ("Density", "Magnetic", "All"):
    h = np.arange(1, 24)
    axes[2].plot(h, [summary[channel]["gaussian_over_student_t"]["by_horizon"][str(x)] for x in h], marker="o", ms=3, label=channel)
axes[2].axhline(1, color="black", ls=":", lw=1)
axes[2].set(xlabel="hidden forecast horizon h", ylabel="Gaussian u / Student-t u")
axes[2].legend(frameon=False)
fig.savefig(OUT / "gaussian_vs_studentt_calibration.png", dpi=180)
plt.close(fig)

patterns = ["spatial_random", "spatial_grid", "spatial_block", "temporal_random", "temporal_block", "aggregate"]
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
x = np.arange(len(patterns)); width = .38
for offset, label in zip((-.5, .5), MODELS):
    lookup = {row["pattern"]: float(row["reconstruction_mse"]) for row in standard_rows if row["model"] == label}
    ax.bar(x + offset * width, [lookup[p] for p in patterns], width, label=label)
ax.set(yscale="log", ylabel="fixed-validation reconstruction MSE", xticks=x, xticklabels=patterns)
ax.tick_params(axis="x", rotation=25); ax.legend(frameon=False)
fig.savefig(OUT / "gaussian_vs_studentt_standard_masks.png", dpi=180)
plt.close(fig)

(OUT / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
print("Combined analysis artifacts complete")
