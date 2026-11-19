#!/usr/bin/env python3
"""Read-only scientific evaluation of baseline vs pure temporal-extrapolation continuation."""

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

BASELINE_CKPT = PROJECT / "runs/multi_beta_curriculum_direct_v3_three_phase_globalcosine_2e4_1e5_256ep_20261115/final.pt"
NEW_CKPT = PROJECT / "runs/multi_beta_three_phase_temporal_extrapolation_continuation_20ep_20261116/final.pt"
BASELINE_ANALYSIS_DIR = PROJECT / "evaluations/three_phase_256ep_analysis_20261116"
OUTPUT = PROJECT / "evaluations/temporal_extrapolation_continuation_20ep_analysis_20261117"

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
        "sampled_pearson_e2_u": float(stats.pearsonr(e2_sample, variance).statistic) if len(residual) > 1 else float("nan"),
        "sampled_spearman_e2_u": float(stats.spearmanr(e2_sample, variance).statistic) if len(residual) > 1 else float("nan"),
    }, residual, variance


def load_model(ckpt_path: Path, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
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
    return checkpoint, model, mean, std


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
def evaluate_standard_all(models: dict[str, tuple], dataset, role_indices, device):
    all_rows = []
    all_samples = {}
    for model_name, (_ckpt, model, mean, std) in models.items():
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
                row = {
                    "model": model_name, "role": role, "pass": pass_name, "source_task": source_task,
                    "effective_observation_task": effective_task, "channel": channel, **summary,
                }
                all_rows.append(row)
                all_samples[(model_name, role, pass_name, source_task, channel)] = (residual, variance)
            print(f"[{model_name}] standard {role}: {len(indices)} paired windows complete", flush=True)

    write_csv(OUTPUT / "standard_mask_metrics.csv", all_rows)
    return all_rows, all_samples


@torch.no_grad()
def evaluate_forecast_all(models: dict[str, tuple], dataset, indices, device):
    all_buckets = {m: defaultdict(new_bucket) for m in models}
    rngs = {m: np.random.default_rng(SEED + 9) for m in models}

    for batch_number, (_indices, block, metadata) in enumerate(batches(dataset, indices)):
        # Normalize target with standard stats (identical for both models)
        _ckpt, _m, mean, std = next(iter(models.values()))
        target = ((block.to(device) - mean) / (std + 1e-8)).float()
        noise_gen = torch.Generator(device=device).manual_seed(SEED + 200000 + batch_number)
        noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

        for horizon in range(1, 24):
            prefix = 24 - horizon
            mask = torch.ones_like(target)
            mask[:, :, prefix:] = 0.0
            input_variance = 1.0 - mask
            model_input = torch.cat([make_uncertain_input(target, input_variance, noise), input_variance], dim=1)
            unknown_all = mask < 0.5

            for model_name, (_ckpt, model, _mean, _std) in models.items():
                prediction, variance = infer(model, model_input)
                residual = prediction - target
                for channel, channel_indices in channel_groups().items():
                    channel_indices = list(channel_indices)
                    add_metrics(
                        all_buckets[model_name][(horizon, channel)],
                        residual[:, channel_indices], variance[:, channel_indices],
                        unknown_all[:, channel_indices], rngs[model_name],
                    )

        if batch_number == 0 or (batch_number + 1) % 10 == 0:
            print(f"forecast: {min((batch_number + 1) * 2, len(indices))}/{len(indices)} paired windows", flush=True)

    all_rows = []
    all_samples = {}
    for model_name, buckets in all_buckets.items():
        for (horizon, channel), bucket in sorted(buckets.items()):
            summary, residual, variance = finish_bucket(bucket)
            all_rows.append({
                "model": model_name, "horizon": horizon, "visible_prefix": 24 - horizon,
                "channel": channel, **summary,
            })
            all_samples[(model_name, horizon, channel)] = (residual, variance)

    write_csv(OUTPUT / "forecast_horizon_metrics.csv", all_rows)
    return all_rows, all_samples


def compute_horizon_summaries(rows):
    by_model_channel = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        by_model_channel[row["model"]][row["channel"]][int(row["horizon"])] = row

    result = {}
    for model_name, channels_data in by_model_channel.items():
        result[model_name] = {}
        for channel, values in channels_data.items():
            horizons = np.array(sorted(values), dtype=float)
            mse = np.array([values[int(h)]["mean_mse"] for h in horizons])
            u = np.array([values[int(h)]["mean_predicted_u"] for h in horizons])
            ratios = {}
            for horizon in (6, 12, 18, 23):
                ratios[str(horizon)] = {
                    "actual_mse_over_h1": values[horizon]["mean_mse"] / values[1]["mean_mse"],
                    "predicted_u_over_h1": values[horizon]["mean_predicted_u"] / values[1]["mean_predicted_u"],
                    "u_over_mse": values[horizon]["predicted_u_over_actual_mse"],
                }
            result[model_name][channel] = {
                "ratios": ratios,
                "h1_u_over_mse": values[1]["predicted_u_over_actual_mse"],
                "spearman_horizon_actual_mse": float(stats.spearmanr(horizons, mse).statistic),
                "spearman_horizon_predicted_u": float(stats.spearmanr(horizons, u).statistic),
                "pearson_horizonlevel_mse_u": float(stats.pearsonr(mse, u).statistic),
                "spearman_horizonlevel_mse_u": float(stats.spearmanr(mse, u).statistic),
                "adjacent_mse_monotonicity_violations": int(np.sum(np.diff(mse) < 0)),
                "adjacent_u_monotonicity_violations": int(np.sum(np.diff(u) < 0)),
            }
    dump_json(OUTPUT / "forecast_horizon_summary.json", result)
    return result


def generate_comparison_table(rows):
    by_key = {}
    for row in rows:
        by_key[(row["model"], int(row["horizon"]), row["channel"])] = row

    comparison_rows = []
    for horizon in range(1, 24):
        for channel in ["All", "Magnetic", "Density", "Bx", "By", "Bz"]:
            base = by_key.get(("baseline", horizon, channel))
            extrap = by_key.get(("extrapolation_continuation", horizon, channel))
            if base and extrap:
                u_ratio = extrap["mean_predicted_u"] / base["mean_predicted_u"]
                mse_ratio = extrap["mean_mse"] / base["mean_mse"]
                comparison_rows.append({
                    "horizon": horizon,
                    "visible_prefix": 24 - horizon,
                    "channel": channel,
                    "baseline_mse": base["mean_mse"],
                    "continuation_mse": extrap["mean_mse"],
                    "mse_ratio_continuation_over_base": mse_ratio,
                    "baseline_mean_u": base["mean_predicted_u"],
                    "continuation_mean_u": extrap["mean_predicted_u"],
                    "u_ratio_continuation_over_base": u_ratio,
                    "baseline_u_over_mse": base["predicted_u_over_actual_mse"],
                    "continuation_u_over_mse": extrap["predicted_u_over_actual_mse"],
                })
    write_csv(OUTPUT / "forecast_horizon_comparison.csv", comparison_rows)
    return comparison_rows


def plot_horizon_comparison(comparison_rows):
    by_channel = defaultdict(dict)
    for row in comparison_rows:
        by_channel[row["channel"]][int(row["horizon"])] = row

    # Figure 1: baseline_vs_extrapolation_horizon.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for channel, color in [("Density", "blue"), ("Magnetic", "red")]:
        values = by_channel[channel]
        h = np.array(sorted(values))
        base_mse = [values[x]["baseline_mse"] for x in h]
        new_mse = [values[x]["continuation_mse"] for x in h]
        base_u = [values[x]["baseline_mean_u"] for x in h]
        new_u = [values[x]["continuation_mean_u"] for x in h]

        # Top-left: Actual MSE baseline vs new
        axes[0, 0].plot(h, base_mse, marker="o", ms=3, ls="--", color=color, alpha=0.6, label=f"{channel} baseline MSE")
        axes[0, 0].plot(h, new_mse, marker="o", ms=4, color=color, label=f"{channel} extrap-trained MSE")

        # Top-right: Predicted u baseline vs new
        axes[0, 1].plot(h, base_u, marker="s", ms=3, ls="--", color=color, alpha=0.6, label=f"{channel} baseline u")
        axes[0, 1].plot(h, new_u, marker="s", ms=4, color=color, label=f"{channel} extrap-trained u")

        # Bottom-left: u_new / u_baseline ratio
        u_ratio = [values[x]["u_ratio_continuation_over_base"] for x in h]
        axes[1, 0].plot(h, u_ratio, marker="^", ms=4, color=color, label=f"{channel} u_new / u_base")

    axes[0, 0].set(xlabel="hidden forecast horizon h", ylabel="actual MSE", yscale="log", title="Actual MSE: Baseline vs Extrapolation")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set(xlabel="hidden forecast horizon h", ylabel="predicted uncertainty u", yscale="log", title="Predicted Uncertainty: Baseline vs Extrapolation")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].axhline(1.0, color="black", ls="--", lw=1)
    axes[1, 0].set(xlabel="hidden forecast horizon h", ylabel="u_new / u_baseline ratio", title="Horizon Conditioning vs Global Inflation (u_new / u_baseline)")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: All channels u_new / u_base
    for ch in CHANNELS:
        values = by_channel[ch]
        h = np.array(sorted(values))
        u_ratio = [values[x]["u_ratio_continuation_over_base"] for x in h]
        axes[1, 1].plot(h, u_ratio, marker="o", ms=3, label=f"{ch}")
    axes[1, 1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1, 1].set(xlabel="hidden forecast horizon h", ylabel="u_new / u_baseline ratio", title="Per-channel u_new / u_baseline ratio")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(True, alpha=0.3)

    fig.savefig(OUTPUT / "baseline_vs_extrapolation_horizon.png", dpi=180)
    plt.close(fig)

    # Figure 2: baseline_vs_extrapolation_calibration.png
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for channel, color in [("Density", "blue"), ("Magnetic", "red")]:
        values = by_channel[channel]
        h = np.array(sorted(values))
        base_cal = [values[x]["baseline_u_over_mse"] for x in h]
        new_cal = [values[x]["continuation_u_over_mse"] for x in h]

        axes[0].plot(h, base_cal, marker="o", ms=3, ls="--", color=color, alpha=0.6, label=f"{channel} baseline u/MSE")
        axes[0].plot(h, new_cal, marker="o", ms=4, color=color, label=f"{channel} extrap-trained u/MSE")

    axes[0].axhline(1.0, color="black", ls="--", lw=1, label="ideal calibration (u/MSE = 1)")
    axes[0].set(xlabel="hidden forecast horizon h", ylabel="predicted u / actual MSE", yscale="log", title="Calibration: Density and Magnetic")
    axes[0].legend(frameon=False)
    axes[0].grid(True, alpha=0.3)

    for ch in CHANNELS:
        values = by_channel[ch]
        h = np.array(sorted(values))
        new_cal = [values[x]["continuation_u_over_mse"] for x in h]
        axes[1].plot(h, new_cal, marker="o", ms=3, label=f"{ch} extrap-trained")
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1].set(xlabel="hidden forecast horizon h", ylabel="predicted u / actual MSE", yscale="log", title="Calibration across Horizons (Per-Channel)")
    axes[1].legend(frameon=False)
    axes[1].grid(True, alpha=0.3)

    fig.savefig(OUTPUT / "baseline_vs_extrapolation_calibration.png", dpi=180)
    plt.close(fig)


def pass0_vs_pass1_comparison(standard_rows):
    matched_rows = []
    for model_name in ["baseline", "extrapolation_continuation"]:
        test = [r for r in standard_rows if r["model"] == model_name and r["role"] == "test" and r["channel"] == "All"]
        values = {(r["source_task"], r["pass"]): r for r in test}
        for source_task in MASK_PATTERNS:
            if (source_task, "pass1") not in values:
                continue
            pass1 = values[(source_task, "pass1")]
            effective_task = pass1["effective_observation_task"]
            matched_pass0 = values.get((effective_task, "pass0"))
            if matched_pass0:
                matched_rows.append({
                    "model": model_name,
                    "source_task": source_task,
                    "effective_observation_task": effective_task,
                    "matched_binary_pass0_mse": matched_pass0["mean_mse"],
                    "continuous_reuse_pass1_mse": pass1["mean_mse"],
                    "pass1_over_matched_pass0_mse": pass1["mean_mse"] / matched_pass0["mean_mse"],
                    "matched_binary_pass0_u_over_mse": matched_pass0["predicted_u_over_actual_mse"],
                    "continuous_reuse_pass1_u_over_mse": pass1["predicted_u_over_actual_mse"],
                })
    write_csv(OUTPUT / "pass0_vs_pass1_matched.csv", matched_rows)
    return matched_rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Evaluation requires CUDA device")

    print(f"Loading models onto {device}...")
    base_ckpt, base_model, base_mean, base_std = load_model(BASELINE_CKPT, device)
    new_ckpt, new_model, new_mean, new_std = load_model(NEW_CKPT, device)

    models = {
        "baseline": (base_ckpt, base_model, base_mean, base_std),
        "extrapolation_continuation": (new_ckpt, new_model, new_mean, new_std),
    }

    split = load_fixed_split()
    dataset = VPICWindowDataset(
        h5_dir=base_ckpt["args"]["h5_dir"], betas=base_ckpt["args"]["betas"],
        delta_t=int(base_ckpt["args"]["delta_t"]), stride_t=1,
        layout="C T X Z", return_metadata=True,
    )

    role_indices = {
        role: select_validation_statistics_indices(dataset, set(split[f"{role}_runs"]), 24, 1)
        for role in ("val", "test")
    }
    print(f"Paired windows: { {k: len(v) for k, v in role_indices.items()} }")

    manifest = {
        "baseline_checkpoint": str(BASELINE_CKPT),
        "new_checkpoint": str(NEW_CKPT),
        "baseline_epoch": base_ckpt["epoch"],
        "continuation_epoch": new_ckpt["epoch"],
        "split_sha256": base_ckpt["split_sha256"],
        "device": str(device),
        "paired_windows": {k: len(v) for k, v in role_indices.items()},
        "forecast_geometry": "all four channels observed for prefix 24-h and unknown for suffix h",
        "paired_noise": True,
        "sampled_pairs_per_window_channel": SAMPLE_LIMIT,
    }
    dump_json(OUTPUT / "evaluation_manifest.json", manifest)

    print("Evaluating standard masks...")
    standard_rows, standard_samples = evaluate_standard_all(models, dataset, role_indices, device)

    print("Evaluating forecast horizons 1..23 on test split...")
    forecast_rows, forecast_samples = evaluate_forecast_all(models, dataset, role_indices["test"], device)

    print("Computing summaries and comparisons...")
    summaries = compute_horizon_summaries(forecast_rows)
    comparison_rows = generate_comparison_table(forecast_rows)
    plot_horizon_comparison(comparison_rows)
    matched_passes = pass0_vs_pass1_comparison(standard_rows)

    dataset.close()
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
