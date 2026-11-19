#!/usr/bin/env python3
"""Read-only, paired evaluation of the completed three-phase epoch-256 model."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

PROJECT = Path("/pscratch/sd/b/binxia/ai4plasma")
sys.path.insert(0, str(PROJECT))

from data.masking import MASK_PATTERNS, make_fixed_validation_mask, make_uncertain_input
from data.vpic_hdf5_dataset import VPICWindowDataset, load_fixed_split
from models.unet3d import UNet3D
from train_masked_unet3d import make_phase_c_reuse_input, rollout_validation_reuse_pattern
from visualize_mask_patterns_unet3d import select_validation_statistics_indices

RUN_DIR = PROJECT / "runs/multi_beta_curriculum_direct_v3_three_phase_globalcosine_2e4_1e5_256ep_20261115"
CHECKPOINT = RUN_DIR / "final.pt"
OLD_RUN = PROJECT / "runs/multi_beta_curriculum_direct_v3_fromscratch_globalcosine_2e4_1e5_1000ep_48h_20261027"
OUTPUT = PROJECT / "evaluations/three_phase_256ep_analysis_20261116"
CHANNELS = ("Bx", "By", "Bz", "Density")
AGGREGATES = ("Magnetic", "All")
SAMPLE_LIMIT = 256
SEED = 1234


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2, allow_nan=True))


def new_bucket() -> dict:
    return {
        "sum_e2": 0.0,
        "sum_u": 0.0,
        "count": 0,
        "window_mse": [],
        "window_u": [],
        "residual_samples": [],
        "u_samples": [],
    }


def channel_groups() -> dict[str, tuple[int, ...]]:
    return {
        "Bx": (0,), "By": (1,), "Bz": (2,), "Density": (3,),
        "Magnetic": (0, 1, 2), "All": (0, 1, 2, 3),
    }


def add_metrics(bucket: dict, residual: torch.Tensor, variance: torch.Tensor,
                unknown: torch.Tensor, rng: np.random.Generator) -> None:
    """Accumulate exact means and bounded paired samples, one window at a time."""
    for batch_index in range(residual.shape[0]):
        choose = unknown[batch_index]
        r = residual[batch_index][choose].float().cpu().numpy()
        u = variance[batch_index][choose].float().cpu().numpy()
        keep = np.isfinite(r) & np.isfinite(u) & (u > 0)
        r, u = r[keep], u[keep]
        if not r.size:
            continue
        e2 = np.square(r, dtype=np.float64)
        bucket["sum_e2"] += float(e2.sum())
        bucket["sum_u"] += float(u.astype(np.float64).sum())
        bucket["count"] += int(r.size)
        bucket["window_mse"].append(float(e2.mean()))
        bucket["window_u"].append(float(u.mean()))
        if r.size > SAMPLE_LIMIT:
            take = rng.choice(r.size, SAMPLE_LIMIT, replace=False)
            r, u = r[take], u[take]
        bucket["residual_samples"].append(r.astype(np.float32))
        bucket["u_samples"].append(u.astype(np.float32))


def finish_bucket(bucket: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    residual = np.concatenate(bucket["residual_samples"]) if bucket["residual_samples"] else np.empty(0)
    variance = np.concatenate(bucket["u_samples"]) if bucket["u_samples"] else np.empty(0)
    mean_mse = bucket["sum_e2"] / bucket["count"]
    mean_u = bucket["sum_u"] / bucket["count"]
    e2_sample = np.square(residual)
    return {
        "voxel_count": bucket["count"],
        "window_count": len(bucket["window_mse"]),
        "mean_mse": mean_mse,
        "median_window_mse": float(np.median(bucket["window_mse"])),
        "mean_predicted_u": mean_u,
        "median_window_mean_u": float(np.median(bucket["window_u"])),
        "median_predicted_u_sampled": float(np.median(variance)),
        "predicted_u_over_actual_mse": mean_u / mean_mse,
        "actual_mse_over_predicted_u": mean_mse / mean_u,
        "sampled_pearson_e2_u": float(stats.pearsonr(e2_sample, variance).statistic),
        "sampled_spearman_e2_u": float(stats.spearmanr(e2_sample, variance).statistic),
    }, residual, variance


def model_context(device: torch.device):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    if int(checkpoint["epoch"]) != 256:
        raise RuntimeError(f"Expected epoch 256, got {checkpoint['epoch']}")
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=int(args.get("base_channels", 16)),
        channel_mults=args.get("channel_mults", [1, 2, 4]),
        architecture=args["model_version"],
        uncertainty_aware=bool(args["uncertainty_aware"]),
        use_attention=bool(args.get("use_attention", False)),
        spatial_only_pooling=bool(args.get("spatial_only_pooling", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    stats_payload = checkpoint["stats"]
    mean = torch.tensor(stats_payload["mean"], device=device).view(1, 4, 1, 1, 1)
    std = torch.tensor(stats_payload["std"], device=device).view(1, 4, 1, 1, 1)
    return checkpoint, args, model, mean, std


@torch.no_grad()
def infer(model, model_input):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=model_input.is_cuda):
        prediction, variance = model(model_input, detach_variance_features=True)
    return prediction.float(), variance.float()


def batches(dataset, indices: list[int], batch_size: int = 2):
    for start in range(0, len(indices), batch_size):
        chosen = indices[start:start + batch_size]
        samples = [dataset[index] for index in chosen]
        yield chosen, torch.stack([sample["block"] for sample in samples]), [sample["metadata"] for sample in samples]


@torch.no_grad()
def evaluate_standard(model, mean, std, dataset, role_indices, device):
    rows, samples = [], {}
    rng = np.random.default_rng(SEED)
    for role, indices in role_indices.items():
        buckets = defaultdict(new_bucket)
        for batch_number, (_indices, block, _metadata) in enumerate(batches(dataset, indices)):
            target = ((block.to(device) - mean) / (std + 1e-8)).float()
            noise_gen = torch.Generator(device=device).manual_seed(SEED + 100000 * (role == "test") + batch_number)
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)
            for task in MASK_PATTERNS:
                mask, _ = make_fixed_validation_mask(task, target.shape, device=device, dtype=target.dtype)
                input_variance = 1.0 - mask
                model_input = torch.cat([make_uncertain_input(target, input_variance, noise), input_variance], dim=1)
                pred0, var0 = infer(model, model_input)
                residual0 = pred0 - target
                pass_data = [("pass0", task, mask, residual0, var0)]
                reuse_task = rollout_validation_reuse_pattern(task)
                reuse_mask, _ = make_fixed_validation_mask(reuse_task, target.shape, device=device, dtype=target.dtype)
                reuse_input, exact_mask = make_phase_c_reuse_input(target, pred0, var0, reuse_mask)
                pred1, var1 = infer(model, reuse_input)
                pass_data.append(("pass1", reuse_task, exact_mask, pred1 - target, var1))
                for pass_name, effective_task, exact, residual, variance in pass_data:
                    unknown_all = exact < 0.5
                    for channel, channel_indices in channel_groups().items():
                        channel_indices = list(channel_indices)
                        add_metrics(
                            buckets[(pass_name, task, effective_task, channel)],
                            residual[:, channel_indices], variance[:, channel_indices],
                            unknown_all[:, channel_indices], rng,
                        )
        for key, bucket in buckets.items():
            pass_name, source_task, effective_task, channel = key
            summary, residual, variance = finish_bucket(bucket)
            row = {"role": role, "pass": pass_name, "source_task": source_task,
                   "effective_observation_task": effective_task, "channel": channel, **summary}
            rows.append(row)
            samples[(role, pass_name, source_task, channel)] = (residual, variance)
        print(f"standard {role}: {len(indices)} paired windows complete", flush=True)
    write_csv(OUTPUT / "standard_mask_metrics.csv", rows)
    return rows, samples


@torch.no_grad()
def evaluate_forecast(model, mean, std, dataset, indices, device):
    buckets = defaultdict(new_bucket)
    rng = np.random.default_rng(SEED + 9)
    for batch_number, (_indices, block, metadata) in enumerate(batches(dataset, indices)):
        target = ((block.to(device) - mean) / (std + 1e-8)).float()
        noise_gen = torch.Generator(device=device).manual_seed(SEED + 200000 + batch_number)
        # Reuse identical noise across every horizon for paired comparisons.
        noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)
        for horizon in range(1, 24):
            prefix = 24 - horizon
            mask = torch.ones_like(target)
            mask[:, :, prefix:] = 0.0
            input_variance = 1.0 - mask
            model_input = torch.cat([make_uncertain_input(target, input_variance, noise), input_variance], dim=1)
            prediction, variance = infer(model, model_input)
            residual = prediction - target
            unknown_all = mask < 0.5
            for channel, channel_indices in channel_groups().items():
                channel_indices = list(channel_indices)
                add_metrics(
                    buckets[(horizon, channel)], residual[:, channel_indices], variance[:, channel_indices],
                    unknown_all[:, channel_indices], rng,
                )
        if batch_number == 0 or (batch_number + 1) % 10 == 0:
            print(f"forecast: {min((batch_number + 1) * 2, len(indices))}/{len(indices)} paired windows", flush=True)
    rows, samples = [], {}
    for (horizon, channel), bucket in sorted(buckets.items()):
        summary, residual, variance = finish_bucket(bucket)
        rows.append({"horizon": horizon, "visible_prefix": 24 - horizon, "channel": channel, **summary})
        samples[(horizon, channel)] = (residual, variance)
    write_csv(OUTPUT / "forecast_horizon_metrics.csv", rows)
    return rows, samples


def reliability_rows(samples, source: str):
    rows = []
    grouped = defaultdict(lambda: [[], []])
    for key, (residual, variance) in samples.items():
        channel = key[-1]
        if channel not in CHANNELS:
            continue
        grouped[channel][0].append(residual)
        grouped[channel][1].append(variance)
    for channel, (residual_parts, variance_parts) in grouped.items():
        residual = np.concatenate(residual_parts)
        variance = np.concatenate(variance_parts)
        e2 = np.square(residual)
        order = np.argsort(variance)
        for bin_index, selected in enumerate(np.array_split(order, 10), 1):
            rows.append({
                "source": source, "channel": channel, "bin": bin_index,
                "count": len(selected), "mean_predicted_u": float(np.mean(variance[selected])),
                "mean_actual_e2": float(np.mean(e2[selected])),
                "median_actual_e2": float(np.median(e2[selected])),
                "predicted_u_over_actual_e2": float(np.mean(variance[selected]) / np.mean(e2[selected])),
            })
    return rows


def residual_and_likelihood_rows(forecast_samples):
    rows = []
    for channel in CHANNELS:
        residual = np.concatenate([forecast_samples[(h, channel)][0] for h in range(1, 24)])
        variance = np.concatenate([forecast_samples[(h, channel)][1] for h in range(1, 24)])
        e2 = np.square(residual.astype(np.float64))
        u = np.maximum(variance.astype(np.float64), 1e-6)
        z = residual / np.sqrt(u)
        nu = 3.0
        scale2 = u * ((nu - 2.0) / nu)
        gaussian_nll = 0.5 * (np.log(2.0 * np.pi * u) + e2 / u)
        student_constant = (
            math.lgamma(nu / 2.0) - math.lgamma((nu + 1.0) / 2.0)
            + 0.5 * math.log(nu * math.pi)
        )
        student_nll = student_constant + 0.5 * np.log(scale2) + 0.5 * (nu + 1.0) * np.log1p(e2 / (nu * scale2))
        rows.append({
            "channel": channel, "sample_count": residual.size,
            "z_mean": float(np.mean(z)), "z_std": float(np.std(z)),
            "z_excess_kurtosis": float(stats.kurtosis(z, fisher=True, bias=False)),
            "z_abs_p95": float(np.quantile(np.abs(z), .95)),
            "z_abs_p99": float(np.quantile(np.abs(z), .99)),
            "z_abs_p999": float(np.quantile(np.abs(z), .999)),
            "mean_gaussian_nll_full": float(np.mean(gaussian_nll)),
            "mean_student_t_df3_nll_full": float(np.mean(student_nll)),
            "student_minus_gaussian_nll": float(np.mean(student_nll - gaussian_nll)),
        })
    write_csv(OUTPUT / "forecast_residual_likelihood.csv", rows)
    return rows


def horizon_summary(rows):
    by_channel = defaultdict(dict)
    for row in rows:
        by_channel[row["channel"]][int(row["horizon"])] = row
    result = {}
    for channel, values in by_channel.items():
        horizons = np.array(sorted(values), dtype=float)
        mse = np.array([values[int(h)]["mean_mse"] for h in horizons])
        u = np.array([values[int(h)]["mean_predicted_u"] for h in horizons])
        ratios = {}
        for horizon in (6, 12, 18, 23):
            ratios[str(horizon)] = {
                "actual_mse_over_h1": values[horizon]["mean_mse"] / values[1]["mean_mse"],
                "predicted_u_over_h1": values[horizon]["mean_predicted_u"] / values[1]["mean_predicted_u"],
            }
        result[channel] = {
            "ratios": ratios,
            "spearman_horizon_actual_mse": float(stats.spearmanr(horizons, mse).statistic),
            "spearman_horizon_predicted_u": float(stats.spearmanr(horizons, u).statistic),
            "pearson_horizonlevel_mse_u": float(stats.pearsonr(mse, u).statistic),
            "spearman_horizonlevel_mse_u": float(stats.spearmanr(mse, u).statistic),
            "adjacent_mse_monotonicity_violations": int(np.sum(np.diff(mse) < 0)),
            "adjacent_u_monotonicity_violations": int(np.sum(np.diff(u) < 0)),
        }
    dump_json(OUTPUT / "forecast_horizon_summary.json", result)
    return result


def training_summary():
    history = json.loads((RUN_DIR / "history.json").read_text())
    old_history = json.loads((OLD_RUN / "history.json").read_text())
    selected_epochs = [65, 75, 76, 105, 115, 116, 128, 129, 148, 168, 169, 216, 236, 256]
    rows = []
    for epoch in selected_epochs:
        row = history[epoch - 1]
        curriculum = row["curriculum"]
        rows.append({
            "epoch": epoch, "phase": curriculum["phase_name"],
            "learning_rate": row["learning_rate"], "train_mse": row["train_mse"],
            "val_mse": row["val_mse"], "train_student_t_nll": row["train_student_t_nll"],
            "val_student_t_nll": row["val_student_t_nll"], "normalized_total_loss": curriculum["total_loss"],
            "pass0_mse": curriculum["L_mu_0"], "pass1_mse": curriculum["L_mu_1"],
            "pass0_nll": curriculum["L_var_0"], "pass1_nll": curriculum["L_var_1"],
            "gradient_norm_mean": row["gradient_norm_mean"], "gradient_norm_max": row["gradient_norm_max"],
            "lambda_v_end": curriculum["lambda_v_end"], "lambda_r_end": curriculum["lambda_r_end"],
        })
    write_csv(OUTPUT / "training_selected_epochs.csv", rows)
    epoch = np.array([row["epoch"] for row in history])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(epoch, [r["train_mse"] for r in history], label="train")
    axes[0, 0].plot(epoch, [r["val_mse"] for r in history], label="validation")
    axes[0, 0].set(yscale="log", ylabel="reconstruction MSE", xlabel="epoch")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].plot(epoch, [r["train_student_t_nll"] for r in history], label="train")
    axes[0, 1].plot(epoch, [r["val_student_t_nll"] for r in history], label="validation")
    axes[0, 1].set(ylabel="Student-t NLL", xlabel="epoch")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(epoch, [r["curriculum"]["L_mu_0"] for r in history], label="pass 0")
    axes[1, 0].plot(epoch, [r["curriculum"]["L_mu_1"] or np.nan for r in history], label="pass 1")
    axes[1, 0].set(yscale="log", ylabel="training MSE", xlabel="epoch")
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(epoch, [r["learning_rate"] for r in history], color="black", label="LR")
    axes[1, 1].plot(epoch, [r["curriculum"]["lambda_v_end"] for r in history], label=r"$\lambda_v$")
    axes[1, 1].plot(epoch, [r["curriculum"]["lambda_r_end"] for r in history], label=r"$\lambda_r$")
    axes[1, 1].set(ylabel="LR / curriculum weight", xlabel="epoch")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        for boundary in (75, 115, 128, 168):
            axis.axvline(boundary, color="0.75", lw=.8, ls="--")
    fig.suptitle("Three-phase training trajectory (vertical lines: A end, variance full, C start, reuse full)")
    fig.savefig(OUTPUT / "training_phase_summary.png", dpi=180)
    plt.close(fig)
    return history, old_history


def plot_forecast(rows):
    by_channel = defaultdict(dict)
    for row in rows:
        by_channel[row["channel"]][int(row["horizon"])] = row
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for channel in ("Magnetic", "Density"):
        values = by_channel[channel]
        h = np.array(sorted(values))
        axes[0].plot(h, [values[x]["mean_mse"] for x in h], marker="o", ms=3, label=f"{channel} actual MSE")
        axes[0].plot(h, [values[x]["mean_predicted_u"] for x in h], marker="s", ms=3, ls="--", label=f"{channel} predicted u")
        axes[1].plot(h, [values[x]["predicted_u_over_actual_mse"] for x in h], marker="o", ms=3, label=channel)
    axes[0].set(xlabel="hidden forecast horizon h", ylabel="normalized squared units", yscale="log")
    axes[0].legend(frameon=False)
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1].set(xlabel="hidden forecast horizon h", ylabel="mean predicted u / actual MSE", yscale="log")
    axes[1].legend(frameon=False)
    fig.savefig(OUTPUT / "forecast_horizon_mse_vs_u.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for channel in CHANNELS:
        values = by_channel[channel]
        h = np.array(sorted(values))
        ax.plot(h, [values[x]["predicted_u_over_actual_mse"] for x in h], marker="o", ms=3, label=channel)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.set(xlabel="hidden forecast horizon h", ylabel="mean predicted u / actual MSE", yscale="log")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(OUTPUT / "forecast_horizon_calibration_ratio.png", dpi=180)
    plt.close(fig)


def plot_passes(rows):
    test = [row for row in rows if row["role"] == "test" and row["channel"] == "All"]
    values = {(row["source_task"], row["pass"]): row for row in test}
    matched_rows = []
    for source_task in MASK_PATTERNS:
        pass1 = values[(source_task, "pass1")]
        effective_task = pass1["effective_observation_task"]
        matched_pass0 = values[(effective_task, "pass0")]
        matched_rows.append({
            "source_task": source_task,
            "effective_observation_task": effective_task,
            "matched_binary_pass0_mse": matched_pass0["mean_mse"],
            "continuous_reuse_pass1_mse": pass1["mean_mse"],
            "pass1_over_matched_pass0_mse": pass1["mean_mse"] / matched_pass0["mean_mse"],
            "matched_binary_pass0_u_over_mse": matched_pass0["predicted_u_over_actual_mse"],
            "continuous_reuse_pass1_u_over_mse": pass1["predicted_u_over_actual_mse"],
        })
    write_csv(OUTPUT / "pass0_vs_pass1_matched.csv", matched_rows)
    x = np.arange(len(MASK_PATTERNS)); width = .36
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].bar(x - width / 2, [row["matched_binary_pass0_mse"] for row in matched_rows], width, label="matched binary pass 0")
    axes[0].bar(x + width / 2, [values[(p, "pass1")]["mean_mse"] for p in MASK_PATTERNS], width, label="pass 1")
    axes[0].set(ylabel="normalized MSE", xticks=x, xticklabels=MASK_PATTERNS)
    axes[0].tick_params(axis="x", rotation=30)
    axes[0].legend(frameon=False)
    axes[1].bar(x - width / 2, [row["matched_binary_pass0_u_over_mse"] for row in matched_rows], width, label="matched binary pass 0")
    axes[1].bar(x + width / 2, [values[(p, "pass1")]["predicted_u_over_actual_mse"] for p in MASK_PATTERNS], width, label="pass 1")
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1].set(ylabel="mean predicted u / actual MSE", xticks=x, xticklabels=MASK_PATTERNS)
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].legend(frameon=False)
    fig.savefig(OUTPUT / "pass0_vs_pass1.png", dpi=180)
    plt.close(fig)


def plot_reliability(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for channel in CHANNELS:
        selected = [r for r in rows if r["source"] == "forecast" and r["channel"] == channel]
        axes[0].plot([r["mean_predicted_u"] for r in selected], [r["mean_actual_e2"] for r in selected], marker="o", label=channel)
        axes[1].plot([r["bin"] for r in selected], [r["predicted_u_over_actual_e2"] for r in selected], marker="o", label=channel)
    lo, hi = axes[0].get_xlim()
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set(xlabel="mean predicted u", ylabel="mean actual squared error", xscale="log", yscale="log")
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1].set(xlabel="predicted-u quantile bin", ylabel="predicted u / actual squared error", yscale="log")
    axes[0].legend(frameon=False); axes[1].legend(frameon=False)
    fig.savefig(OUTPUT / "uncertainty_reliability.png", dpi=180)
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This bounded evaluation requires the available CUDA device")
    checkpoint, args, model, mean, std = model_context(device)
    split = load_fixed_split()
    dataset = VPICWindowDataset(
        h5_dir=args["h5_dir"], betas=args["betas"], delta_t=int(args["delta_t"]),
        stride_t=1, layout="C T X Z", return_metadata=True,
    )
    role_indices = {
        role: select_validation_statistics_indices(dataset, set(split[f"{role}_runs"]), 24, 1)
        for role in ("val", "test")
    }
    if len(role_indices["val"]) != 125 or len(role_indices["test"]) != 125:
        raise RuntimeError(f"Expected 125 paired windows per role, got { {k: len(v) for k,v in role_indices.items()} }")
    manifest = {
        "checkpoint": str(CHECKPOINT), "checkpoint_epoch": checkpoint["epoch"],
        "split_sha256": checkpoint["split_sha256"],
        "split_counts": {key: len(split[key]) for key in ("train_runs", "val_runs", "test_runs")},
        "device": str(device), "paired_windows": {k: len(v) for k, v in role_indices.items()},
        "forecast_geometry": "all four channels observed for prefix 24-h and unknown for suffix h",
        "paired_noise": True, "sampled_pairs_per_window_channel": SAMPLE_LIMIT,
    }
    dump_json(OUTPUT / "evaluation_manifest.json", manifest)
    history, old_history = training_summary()
    standard_rows, standard_samples = evaluate_standard(model, mean, std, dataset, role_indices, device)
    forecast_rows, forecast_samples = evaluate_forecast(model, mean, std, dataset, role_indices["test"], device)
    forecast_summary = horizon_summary(forecast_rows)
    reliability = reliability_rows(forecast_samples, "forecast") + reliability_rows(standard_samples, "standard")
    write_csv(OUTPUT / "uncertainty_reliability_bins.csv", reliability)
    likelihood = residual_and_likelihood_rows(forecast_samples)
    plot_forecast(forecast_rows)
    plot_passes(standard_rows)
    plot_reliability(reliability)
    comparison = {
        "old_final_epoch": old_history[-1]["epoch"],
        "new_final_epoch": history[-1]["epoch"],
        "old_final_fixed_validation": old_history[-1]["val_per_pattern"],
        "new_final_fixed_validation": history[-1]["val_per_pattern"],
        "note": "Raw fixed-validation MSE is directly comparable; old Gaussian NLL and new Student-t NLL are not.",
    }
    dump_json(OUTPUT / "old_baseline_comparison.json", comparison)
    dump_json(OUTPUT / "analysis_results.json", {
        "manifest": manifest, "forecast_summary": forecast_summary,
        "residual_likelihood": likelihood, "comparison": comparison,
    })
    dataset.close()
    print(f"Analysis complete: {OUTPUT}")


if __name__ == "__main__":
    main()
