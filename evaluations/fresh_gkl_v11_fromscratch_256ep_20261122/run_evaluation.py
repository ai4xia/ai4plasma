#!/usr/bin/env python3
"""Comprehensive scientific evaluation of the fresh from-scratch 256-epoch GKL v11 model:
  Run: multi_beta_gkl_v11_fromscratch_256ep_20261122
  Checkpoint: runs/multi_beta_gkl_v11_fromscratch_256ep_20261122/final.pt

Evaluates:
  1. Standard 5-mask validation and test benchmarks (Bx, By, Bz, Density, Magnetic, All).
  2. Performance across 10 plasma beta values (0.01 to 0.20).
  3. Forward forecast horizon sweep h=1..23 (especially extreme horizons h=20..23).
  4. Reverse temporal extrapolation & sparse temporal regimes.
  5. Uncertainty quantification:
     - u / MSE ratio
     - Standardized residuals (z = e / sqrt(u))
     - Empirical 1-sigma (68.3%) and 2-sigma (95.4%) coverage
     - Voxel-level Pearson & Spearman correlation between u and e^2
     - Calibration across beta values and mask families
     - Observed vs unobserved voxel comparison
  6. Modality comparison: Magnetic vs Density.
  7. Paired comparison and bootstrap 95% CIs against the fresh 256ep GKL v9 baseline.
  8. Publication-quality figures.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import rankdata, spearmanr
import torch

PROJECT = Path("/pscratch/sd/b/binxia/ai4plasma")
sys.path.insert(0, str(PROJECT))

from data.masking import (
    MASK_PATTERNS,
    make_fixed_validation_mask,
    make_uncertain_input,
)
from data.vpic_hdf5_dataset import (
    VPICWindowDataset,
    load_fixed_split,
    validate_checkpoint_split,
)
from models.unet3d import UNet3D
from visualize_mask_patterns_unet3d import select_validation_statistics_indices

OUTPUT = PROJECT / "evaluations/fresh_gkl_v11_fromscratch_256ep_20261122"
RUN_DIR = PROJECT / "runs/multi_beta_gkl_v11_fromscratch_256ep_20261122"
FINAL_CKPT = RUN_DIR / "final.pt"

# Historical baselines for direct comparison
V9_EVAL_DIR = PROJECT / "evaluations/fresh_gkl_fully_diff_256ep_20261120"
V9_STD_CSV = V9_EVAL_DIR / "standard_mask_metrics.csv"
V9_FORECAST_CSV = V9_EVAL_DIR / "forecast_horizon_metrics.csv"

SEED = 1234
FORWARD_HORIZONS = list(range(1, 24))
FOCUS_HORIZONS = [18, 20, 21, 22, 23]
REVERSE_HORIZONS = [1, 4, 8, 12, 16, 20, 22, 23]
SAMPLE_LIMIT = 4096

GROUPS = {
    "Bx": (0,),
    "By": (1,),
    "Bz": (2,),
    "Density": (3,),
    "Magnetic": (0, 1, 2),
    "All": (0, 1, 2, 3),
}

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=int(args.get("base_channels", 24)),
        channel_mults=args.get("channel_mults", [1, 2, 4, 8]),
        architecture=args["model_version"],
        uncertainty_aware=bool(args["uncertainty_aware"]),
        use_attention=bool(args.get("use_attention", False)),
        spatial_only_pooling=bool(args.get("spatial_only_pooling", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    stats_payload = ckpt["stats"]
    mean = torch.tensor(stats_payload["mean"], device=device).view(1, 4, 1, 1, 1)
    std = torch.tensor(stats_payload["std"], device=device).view(1, 4, 1, 1, 1)
    return ckpt, model, mean, std

@torch.no_grad()
def infer(model, model_input, device):
    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred, var = model(model_input, detach_variance_features=True)
    else:
        pred, var = model(model_input, detach_variance_features=True)
    return pred.float(), var.float()

def batches(dataset, indices: list[int], batch_size: int = 4):
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        samples = [dataset[index] for index in chosen]
        yield chosen, torch.stack([s["block"] for s in samples]), [s["metadata"] for s in samples]

def new_bucket():
    return {
        "sum_e2": 0.0,
        "sum_ae": 0.0,
        "sum_u": 0.0,
        "count": 0,
        "window_mse": [],
        "window_mae": [],
        "window_u": [],
        "residual_samples": [],
        "u_samples": [],
    }

def add_metrics(
    bucket: dict,
    residual: torch.Tensor,
    variance: torch.Tensor,
    unknown: torch.Tensor,
    rng: np.random.Generator,
):
    for b in range(residual.shape[0]):
        choose = unknown[b]
        r = residual[b][choose].float().cpu().numpy()
        u = variance[b][choose].float().cpu().numpy()
        keep = np.isfinite(r) & np.isfinite(u) & (u > 0)
        r, u = r[keep], u[keep]
        if not r.size:
            continue
        e2 = np.square(r, dtype=np.float64)
        ae = np.abs(r, dtype=np.float64)
        bucket["sum_e2"] += float(e2.sum())
        bucket["sum_ae"] += float(ae.sum())
        bucket["sum_u"] += float(u.astype(np.float64).sum())
        bucket["count"] += int(r.size)
        bucket["window_mse"].append(float(e2.mean()))
        bucket["window_mae"].append(float(ae.mean()))
        bucket["window_u"].append(float(u.mean()))
        if r.size > SAMPLE_LIMIT:
            take = rng.choice(r.size, SAMPLE_LIMIT, replace=False)
            r, u = r[take], u[take]
        bucket["residual_samples"].append(r.astype(np.float32))
        bucket["u_samples"].append(u.astype(np.float32))

def finish_bucket(bucket: dict):
    residual = np.concatenate(bucket["residual_samples"]) if bucket["residual_samples"] else np.empty(0)
    variance = np.concatenate(bucket["u_samples"]) if bucket["u_samples"] else np.empty(0)
    mean_mse = bucket["sum_e2"] / bucket["count"] if bucket["count"] > 0 else 0.0
    mean_mae = bucket["sum_ae"] / bucket["count"] if bucket["count"] > 0 else 0.0
    mean_rmse = math.sqrt(mean_mse)
    mean_u = bucket["sum_u"] / bucket["count"] if bucket["count"] > 0 else 0.0
    e2_sample = np.square(residual)

    # Standardized residuals: z = r / sqrt(u)
    if residual.size > 0 and variance.size > 0:
        std_u = np.sqrt(np.maximum(variance, 1e-8))
        z = residual / std_u
        cov_1sigma = float(np.mean(np.abs(z) <= 1.0))
        cov_2sigma = float(np.mean(np.abs(z) <= 2.0))
        z_mean = float(np.mean(z))
        z_std = float(np.std(z))
        z_skew = float(stats.skew(z))
        z_kurt = float(stats.kurtosis(z))
    else:
        cov_1sigma, cov_2sigma, z_mean, z_std, z_skew, z_kurt = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    return {
        "voxel_count": bucket["count"],
        "window_count": len(bucket["window_mse"]),
        "mean_mse": mean_mse,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "median_window_mse": float(np.median(bucket["window_mse"])) if bucket["window_mse"] else 0.0,
        "median_window_mae": float(np.median(bucket["window_mae"])) if bucket["window_mae"] else 0.0,
        "mean_predicted_u": mean_u,
        "median_window_mean_u": float(np.median(bucket["window_u"])) if bucket["window_u"] else 0.0,
        "median_predicted_u_sampled": float(np.median(variance)) if variance.size else 0.0,
        "predicted_u_over_actual_mse": (mean_u / mean_mse) if mean_mse > 0 else 0.0,
        "actual_mse_over_predicted_u": (mean_mse / mean_u) if mean_u > 0 else 0.0,
        "sampled_pearson_e2_u": float(stats.pearsonr(e2_sample, variance).statistic) if len(residual) > 1 else float("nan"),
        "sampled_spearman_e2_u": float(stats.spearmanr(e2_sample, variance).statistic) if len(residual) > 1 else float("nan"),
        "coverage_1sigma_nominal_0.683": cov_1sigma,
        "coverage_2sigma_nominal_0.954": cov_2sigma,
        "std_residual_mean": z_mean,
        "std_residual_std": z_std,
        "std_residual_skewness": z_skew,
        "std_residual_kurtosis": z_kurt,
    }

def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    if n == 0:
        return 0.0, 0.0, 0.0
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diff[indices], axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(np.mean(diff)), float(low), float(high)

def main():
    print("=" * 80)
    print("Fresh GKL v11 From-Scratch 256-Epoch Scientific Evaluation")
    print(f"Run: {RUN_DIR.name}")
    print(f"Checkpoint: {FINAL_CKPT}")
    print("=" * 80, flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(8)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else '8 CPU threads'})", flush=True)

    # 1. Dataset & Split
    print("\n--- Loading Dataset & Fixed Splits ---", flush=True)
    split = load_fixed_split()
    cfg = json.loads((RUN_DIR / "config.json").read_text())
    dataset = VPICWindowDataset(
        h5_dir=cfg["h5_dir"],
        betas=cfg["betas"],
        delta_t=24,
        stride_t=1,
        layout="C T X Z",
        return_metadata=True,
    )
    role_indices = {
        role: select_validation_statistics_indices(dataset, set(split[f"{role}_runs"]), 24, 1)
        for role in ("val", "test")
    }
    assert len(role_indices["val"]) == 125
    assert len(role_indices["test"]) == 125
    print(f"Loaded {len(dataset)} windows. Test: 125 windows, Val: 125 windows.", flush=True)

    # 2. Model Loading
    print("\n--- Loading Model Weights & Channel Stats ---", flush=True)
    ckpt_meta, model, mean_norm, std_norm = load_model(FINAL_CKPT, device)
    validate_checkpoint_split(ckpt_meta)
    print(f"Loaded epoch {ckpt_meta['epoch']} model successfully.", flush=True)

    # Storage for all evaluation tables
    all_standard_rows = []
    all_forecast_rows = []
    all_beta_rows = []
    all_temporal_regime_rows = []
    all_calibration_rows = []

    # Map for window-level paired comparison: (role, task, ch) -> {idx: mse}
    v11_window_standard_mse = defaultdict(dict)
    v11_window_forecast_mse = defaultdict(dict)
    v11_window_forecast_u = defaultdict(dict)

    # Representative slices for qualitative figure
    qualitative_samples = []

    # -------------------------------------------------------------------------
    # 3. Standard Mask Benchmark (Val & Test across 5 tasks)
    # -------------------------------------------------------------------------
    print("\n--- Section 3: Standard Mask Benchmark ---", flush=True)
    for role, indices in role_indices.items():
        print(f"Evaluating {role} split (125 windows)...", flush=True)
        rng = np.random.default_rng(SEED)
        buckets = defaultdict(new_bucket)
        beta_buckets = defaultdict(new_bucket)

        for batch_number, (chosen, block, meta_list) in enumerate(batches(dataset, indices, batch_size=4)):
            target = ((block.to(device) - mean_norm) / (std_norm + 1e-8)).float()
            noise_gen = torch.Generator(device=device).manual_seed(
                SEED + 100000 * (role == "test") + batch_number
            )
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

            for task in MASK_PATTERNS:
                mask, _ = make_fixed_validation_mask(task, target.shape, device=device, dtype=target.dtype)
                input_var = 1.0 - mask
                model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
                pred, var = infer(model, model_input, device)
                res = pred - target
                unknown_all = mask < 0.5

                res_np = res.cpu().numpy()
                unknown_np = unknown_all.cpu().numpy()

                for ch, ci in GROUPS.items():
                    ci = list(ci)
                    for b, idx in enumerate(chosen):
                        u_b = unknown_np[b, ci]
                        r_b = res_np[b, ci][u_b]
                        if r_b.size:
                            w_mse = float(np.mean(np.square(r_b)))
                            v11_window_standard_mse[(role, task, ch)][idx] = w_mse

                    add_metrics(
                        buckets[(task, ch)],
                        res[:, ci],
                        var[:, ci],
                        unknown_all[:, ci],
                        rng,
                    )

                # Beta breakout for All channels
                for b, (idx, meta) in enumerate(zip(chosen, meta_list)):
                    beta_val = meta["beta"]
                    add_metrics(
                        beta_buckets[(task, beta_val)],
                        res[b : b + 1],
                        var[b : b + 1],
                        unknown_all[b : b + 1],
                        rng,
                    )

                # Save a qualitative sample from test set
                if role == "test" and task in ("spatial_block", "temporal_block") and len(qualitative_samples) < 4:
                    if batch_number == 0:
                        qualitative_samples.append({
                            "task": task,
                            "target": target[0].cpu().numpy(),
                            "pred": pred[0].cpu().numpy(),
                            "var": var[0].cpu().numpy(),
                            "mask": mask[0].cpu().numpy(),
                            "beta": meta_list[0]["beta"],
                        })

        for (task, ch), b in buckets.items():
            metrics = finish_bucket(b)
            row = {
                "model": "v11_fromscratch_256ep",
                "role": role,
                "source_task": task,
                "channel": ch,
                **metrics,
            }
            all_standard_rows.append(row)

        if role == "test":
            for (task, beta_val), b in beta_buckets.items():
                metrics = finish_bucket(b)
                all_beta_rows.append({
                    "model": "v11_fromscratch_256ep",
                    "role": "test",
                    "source_task": task,
                    "beta": beta_val,
                    **metrics,
                })

    write_csv(OUTPUT / "standard_mask_metrics.csv", all_standard_rows)
    write_csv(OUTPUT / "beta_breakout_metrics.csv", all_beta_rows)
    print("Wrote standard_mask_metrics.csv and beta_breakout_metrics.csv", flush=True)

    # -------------------------------------------------------------------------
    # 4. Forward Forecast Horizon Sweep (h = 1..23) on Held-Out Test Set
    # -------------------------------------------------------------------------
    print("\n--- Section 4: Forward Forecast Horizon Sweep (h=1..23) ---", flush=True)
    rng = np.random.default_rng(SEED + 2000)
    for h in FORWARD_HORIZONS:
        visible_prefix = 24 - h
        print(f"Evaluating horizon h={h} (visible prefix: {visible_prefix} frames)...", flush=True)
        buckets = defaultdict(new_bucket)
        density_buckets = defaultdict(new_bucket)
        magnetic_buckets = defaultdict(new_bucket)

        for batch_number, (chosen, block, _meta) in enumerate(batches(dataset, role_indices["test"], batch_size=4)):
            target = ((block.to(device) - mean_norm) / (std_norm + 1e-8)).float()
            B, C, T, X, Z = target.shape

            # Observation mask: prefix [0:visible_prefix] is observed (1.0), suffix [visible_prefix:24] missing (0.0)
            mask = torch.zeros((B, 1, T, X, Z), device=device, dtype=target.dtype)
            mask[:, :, :visible_prefix, :, :] = 1.0

            # Target unknown evaluation region is the forecast window: [visible_prefix:24]
            unknown_eval = torch.zeros((B, 1, T, X, Z), device=device, dtype=torch.bool)
            unknown_eval[:, :, visible_prefix:, :, :] = True

            input_var = 1.0 - mask.repeat(1, 4, 1, 1, 1)
            unknown_all = unknown_eval.repeat(1, 4, 1, 1, 1)

            noise_gen = torch.Generator(device=device).manual_seed(SEED + 300000 + h * 1000 + batch_number)
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

            model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
            pred, var = infer(model, model_input, device)
            res = pred - target

            res_np = res.cpu().numpy()
            unknown_np = unknown_all.cpu().numpy()

            for ch, ci in GROUPS.items():
                ci = list(ci)
                for b, idx in enumerate(chosen):
                    u_b = unknown_np[b, ci]
                    r_b = res_np[b, ci][u_b]
                    v_b = var[b, ci].cpu().numpy()[u_b]
                    if r_b.size:
                        w_mse = float(np.mean(np.square(r_b)))
                        w_u = float(np.mean(v_b))
                        v11_window_forecast_mse[(h, ch)][idx] = w_mse
                        v11_window_forecast_u[(h, ch)][idx] = w_u

                add_metrics(
                    buckets[ch],
                    res[:, ci],
                    var[:, ci],
                    unknown_all[:, ci],
                    rng,
                )

        for ch, b in buckets.items():
            metrics = finish_bucket(b)
            all_forecast_rows.append({
                "model": "v11_fromscratch_256ep",
                "horizon": h,
                "visible_prefix": visible_prefix,
                "missing_fraction": h / 24.0,
                "channel": ch,
                **metrics,
            })

    write_csv(OUTPUT / "forecast_horizon_metrics.csv", all_forecast_rows)
    print("Wrote forecast_horizon_metrics.csv", flush=True)

    # -------------------------------------------------------------------------
    # 5. Reverse Extrapolation & Sparse Regimes
    # -------------------------------------------------------------------------
    print("\n--- Section 5: Reverse Extrapolation & Sparse Regimes ---", flush=True)
    rng = np.random.default_rng(SEED + 4000)
    for h in REVERSE_HORIZONS:
        visible_suffix = 24 - h
        buckets = defaultdict(new_bucket)
        for batch_number, (chosen, block, _meta) in enumerate(batches(dataset, role_indices["test"], batch_size=4)):
            target = ((block.to(device) - mean_norm) / (std_norm + 1e-8)).float()
            B, C, T, X, Z = target.shape
            mask = torch.zeros((B, 1, T, X, Z), device=device, dtype=target.dtype)
            mask[:, :, h:, :, :] = 1.0  # observed at the end

            unknown_eval = torch.zeros((B, 1, T, X, Z), device=device, dtype=torch.bool)
            unknown_eval[:, :, :h, :, :] = True  # hindsight past missing

            input_var = 1.0 - mask.repeat(1, 4, 1, 1, 1)
            unknown_all = unknown_eval.repeat(1, 4, 1, 1, 1)
            noise_gen = torch.Generator(device=device).manual_seed(SEED + 500000 + h * 1000 + batch_number)
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

            model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
            pred, var = infer(model, model_input, device)
            res = pred - target

            for ch, ci in GROUPS.items():
                ci = list(ci)
                add_metrics(buckets[ch], res[:, ci], var[:, ci], unknown_all[:, ci], rng)

        for ch, b in buckets.items():
            metrics = finish_bucket(b)
            all_temporal_regime_rows.append({
                "model": "v11_fromscratch_256ep",
                "regime": f"reverse_extrapolation_h{h}",
                "missing_frames": h,
                "visible_frames": visible_suffix,
                "channel": ch,
                **metrics,
            })

    write_csv(OUTPUT / "temporal_extrapolation_and_sparse_metrics.csv", all_temporal_regime_rows)
    print("Wrote temporal_extrapolation_and_sparse_metrics.csv", flush=True)

    # -------------------------------------------------------------------------
    # 6. Paired Comparison vs. Fresh GKL v9 256ep Baseline
    # -------------------------------------------------------------------------
    print("\n--- Section 6: Paired Comparison with GKL v9 Baseline ---", flush=True)
    paired_summary_rows = []

    # Read v9 standard metrics
    if V9_STD_CSV.is_file():
        with open(V9_STD_CSV) as f:
            v9_std = list(csv.DictReader(f))
        
        # Build lookup: (role, task, channel) -> row
        v9_lookup = {(r["role"], r["source_task"], r["channel"]): r for r in v9_std if r.get("pass", "pass0") == "pass0"}
        for r_v11 in all_standard_rows:
            key = (r_v11["role"], r_v11["source_task"], r_v11["channel"])
            if key in v9_lookup:
                r_v9 = v9_lookup[key]
                v9_mse = float(r_v9["mean_mse"])
                v11_mse = float(r_v11["mean_mse"])
                v9_u = float(r_v9["mean_predicted_u"])
                v11_u = float(r_v11["mean_predicted_u"])
                v9_cal = float(r_v9["predicted_u_over_actual_mse"])
                v11_cal = float(r_v11["predicted_u_over_actual_mse"])

                delta_mse = v11_mse - v9_mse
                pct_mse = (v9_mse - v11_mse) / v9_mse * 100.0 if v9_mse > 0 else 0.0

                paired_summary_rows.append({
                    "comparison": "v11_fromscratch_256ep_vs_v9_baseline_256ep",
                    "benchmark": "standard_mask",
                    "role": key[0],
                    "task": key[1],
                    "channel": key[2],
                    "v9_mean_mse": v9_mse,
                    "v11_mean_mse": v11_mse,
                    "delta_mse (v11 - v9)": delta_mse,
                    "improvement_mse_pct": pct_mse,
                    "v9_mean_u": v9_u,
                    "v11_mean_u": v11_u,
                    "v9_u_over_mse": v9_cal,
                    "v11_u_over_mse": v11_cal,
                    "cal_distance_v9_to_ideal": abs(v9_cal - 1.0),
                    "cal_distance_v11_to_ideal": abs(v11_cal - 1.0),
                    "calibration_improved": abs(v11_cal - 1.0) < abs(v9_cal - 1.0),
                })

    # Read v9 forecast metrics
    if V9_FORECAST_CSV.is_file():
        with open(V9_FORECAST_CSV) as f:
            v9_fc = list(csv.DictReader(f))
        v9_fc_lookup = {(int(r["horizon"]), r["channel"]): r for r in v9_fc}
        for r_v11 in all_forecast_rows:
            key = (int(r_v11["horizon"]), r_v11["channel"])
            if key in v9_fc_lookup:
                r_v9 = v9_fc_lookup[key]
                v9_mse = float(r_v9["mean_mse"])
                v11_mse = float(r_v11["mean_mse"])
                v9_u = float(r_v9["mean_predicted_u"])
                v11_u = float(r_v11["mean_predicted_u"])
                v9_cal = float(r_v9["predicted_u_over_actual_mse"])
                v11_cal = float(r_v11["predicted_u_over_actual_mse"])

                delta_mse = v11_mse - v9_mse
                pct_mse = (v9_mse - v11_mse) / v9_mse * 100.0 if v9_mse > 0 else 0.0

                paired_summary_rows.append({
                    "comparison": "v11_fromscratch_256ep_vs_v9_baseline_256ep",
                    "benchmark": "forward_forecast",
                    "role": "test",
                    "task": f"horizon_{key[0]}",
                    "channel": key[1],
                    "v9_mean_mse": v9_mse,
                    "v11_mean_mse": v11_mse,
                    "delta_mse (v11 - v9)": delta_mse,
                    "improvement_mse_pct": pct_mse,
                    "v9_mean_u": v9_u,
                    "v11_mean_u": v11_u,
                    "v9_u_over_mse": v9_cal,
                    "v11_u_over_mse": v11_cal,
                    "cal_distance_v9_to_ideal": abs(v9_cal - 1.0),
                    "cal_distance_v11_to_ideal": abs(v11_cal - 1.0),
                    "calibration_improved": abs(v11_cal - 1.0) < abs(v9_cal - 1.0),
                })

    write_csv(OUTPUT / "paired_summary.csv", paired_summary_rows)
    print("Wrote paired_summary.csv", flush=True)

    # -------------------------------------------------------------------------
    # 7. Generate Publication Figures
    # -------------------------------------------------------------------------
    print("\n--- Section 7: Generating Publication Figures ---", flush=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Standard Mask Comparison (v9 vs v11)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    test_rows_v11 = [r for r in all_standard_rows if r["role"] == "test" and r["channel"] == "All"]
    tasks = [r["source_task"] for r in test_rows_v11]
    mses_v11 = [r["mean_mse"] for r in test_rows_v11]
    u_v11 = [r["mean_predicted_u"] for r in test_rows_v11]

    # Find matching v9
    v9_test_map = {r["source_task"]: r for r in v9_std if r.get("role") == "test" and r.get("channel") == "All"} if V9_STD_CSV.is_file() else {}
    mses_v9 = [float(v9_test_map[t]["mean_mse"]) if t in v9_test_map else 0.0 for t in tasks]
    u_v9 = [float(v9_test_map[t]["mean_predicted_u"]) if t in v9_test_map else 0.0 for t in tasks]

    x = np.arange(len(tasks))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, mses_v9, w, label="v9 Baseline (256ep)", color="#7f7f7f", alpha=0.8)
    ax.bar(x + w/2, mses_v11, w, label="v11 From-Scratch (256ep)", color="#1f77b4")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=10)
    ax.set_ylabel("Mean Squared Error (Standardized Units)")
    ax.set_title("Reconstruction MSE by Mask Family (Test Set)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cal_v9 = [float(v9_test_map[t]["predicted_u_over_actual_mse"]) if t in v9_test_map else 0.0 for t in tasks]
    cal_v11 = [r["predicted_u_over_actual_mse"] for r in test_rows_v11]
    ax.bar(x - w/2, cal_v9, w, label="v9 Baseline", color="#7f7f7f", alpha=0.8)
    ax.bar(x + w/2, cal_v11, w, label="v11 From-Scratch", color="#2ca02c")
    ax.axhline(1.0, color="red", linestyle="--", lw=1.5, label="Ideal Calibration (u/MSE = 1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_", "\n") for t in tasks], fontsize=10)
    ax.set_ylabel("Calibration Ratio (u / MSE)")
    ax.set_title("Uncertainty Calibration Ratio (u / MSE)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "standard_mask_comparison.png", dpi=200)
    plt.close()
    print("Saved standard_mask_comparison.png", flush=True)

    # Figure 2: Forecast Horizon Error and Uncertainty Growth
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 2a: All channels MSE
    h_vals = [r["horizon"] for r in all_forecast_rows if r["channel"] == "All"]
    mse_all = [r["mean_mse"] for r in all_forecast_rows if r["channel"] == "All"]
    u_all = [r["mean_predicted_u"] for r in all_forecast_rows if r["channel"] == "All"]
    cal_all = [r["predicted_u_over_actual_mse"] for r in all_forecast_rows if r["channel"] == "All"]

    v9_fc_all = {int(r["horizon"]): r for r in v9_fc if r["channel"] == "All"} if V9_FORECAST_CSV.is_file() else {}
    v9_mse_all = [float(v9_fc_all[h]["mean_mse"]) if h in v9_fc_all else float("nan") for h in h_vals]
    v9_u_all = [float(v9_fc_all[h]["mean_predicted_u"]) if h in v9_fc_all else float("nan") for h in h_vals]
    v9_cal_all = [float(v9_fc_all[h]["predicted_u_over_actual_mse"]) if h in v9_fc_all else float("nan") for h in h_vals]

    ax = axes[0, 0]
    ax.plot(h_vals, v9_mse_all, label="v9 Baseline MSE", color="gray", linestyle="--", lw=1.5)
    ax.plot(h_vals, mse_all, label="v11 From-Scratch MSE", color="#d62728", lw=2)
    ax.plot(h_vals, v9_u_all, label="v9 Predicted u", color="darkgray", linestyle=":", lw=1.5)
    ax.plot(h_vals, u_all, label="v11 Predicted u", color="#1f77b4", lw=2)
    ax.set_xlabel("Forecast Horizon h (Frames)")
    ax.set_ylabel("Value (Standardized Units)")
    ax.set_title("Error & Uncertainty Growth vs. Horizon (All Channels)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2b: Calibration Ratio (u / MSE) vs Horizon
    ax = axes[0, 1]
    ax.plot(h_vals, v9_cal_all, label="v9 Baseline u/MSE", color="gray", linestyle="--", lw=1.5)
    ax.plot(h_vals, cal_all, label="v11 From-Scratch u/MSE", color="#2ca02c", lw=2)
    ax.axhline(1.0, color="red", linestyle="--", lw=1.2, label="Ideal Calibration (1.0)")
    ax.axvspan(19.5, 23.5, color="orange", alpha=0.15, label="Extreme Horizons h=20..23")
    ax.set_xlabel("Forecast Horizon h (Frames)")
    ax.set_ylabel("Calibration Ratio (u / MSE)")
    ax.set_title("Forecast Uncertainty Calibration vs. Horizon")
    ax.set_ylim(0.2, 1.3)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2c: Modality Breakdown: Magnetic vs Density MSE
    ax = axes[1, 0]
    mag_mse = [r["mean_mse"] for r in all_forecast_rows if r["channel"] == "Magnetic"]
    den_mse = [r["mean_mse"] for r in all_forecast_rows if r["channel"] == "Density"]
    ax.plot(h_vals, mag_mse, label="Magnetic MSE", color="#1f77b4", lw=2)
    ax.plot(h_vals, den_mse, label="Density MSE", color="#ff7f0e", lw=2)
    ax.set_xlabel("Forecast Horizon h (Frames)")
    ax.set_ylabel("MSE")
    ax.set_title("Modality Comparison: Magnetic vs. Density Forecast MSE")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2d: Modality Breakdown: Magnetic vs Density Calibration
    ax = axes[1, 1]
    mag_cal = [r["predicted_u_over_actual_mse"] for r in all_forecast_rows if r["channel"] == "Magnetic"]
    den_cal = [r["predicted_u_over_actual_mse"] for r in all_forecast_rows if r["channel"] == "Density"]
    ax.plot(h_vals, mag_cal, label="Magnetic u/MSE", color="#1f77b4", lw=2)
    ax.plot(h_vals, den_cal, label="Density u/MSE", color="#ff7f0e", lw=2)
    ax.axhline(1.0, color="red", linestyle="--", lw=1.2, label="Ideal Calibration")
    ax.axvspan(19.5, 23.5, color="orange", alpha=0.15)
    ax.set_xlabel("Forecast Horizon h (Frames)")
    ax.set_ylabel("u / MSE")
    ax.set_title("Modality Uncertainty Calibration (Magnetic vs. Density)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "forecast_horizon_comparison.png", dpi=200)
    plt.close()
    print("Saved forecast_horizon_comparison.png", flush=True)

    # Figure 3: Uncertainty Calibration Diagnostic
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 3a: Beta Calibration
    ax = axes[0, 0]
    betas_unique = sorted(list(set(r["beta"] for r in all_beta_rows)))
    for p in ("spatial_random", "spatial_block", "temporal_block"):
        p_rows = {r["beta"]: r["predicted_u_over_actual_mse"] for r in all_beta_rows if r["source_task"] == p}
        vals = [p_rows.get(b, float("nan")) for b in betas_unique]
        ax.plot(betas_unique, vals, marker="o", label=p.replace("_", " "), lw=1.5)
    ax.axhline(1.0, color="red", linestyle="--", lw=1.2, label="Ideal (1.0)")
    ax.set_xlabel("Plasma Beta")
    ax.set_ylabel("u / MSE Ratio")
    ax.set_title("Calibration Ratio across Plasma Beta Values")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3b: Coverage Diagram
    ax = axes[0, 1]
    pats = [r["source_task"] for r in test_rows_v11]
    cov1 = [r["coverage_1sigma_nominal_0.683"] for r in test_rows_v11]
    cov2 = [r["coverage_2sigma_nominal_0.954"] for r in test_rows_v11]
    x_p = np.arange(len(pats))
    ax.bar(x_p - w/2, cov1, w, label=r"Empirical $1\sigma$ Coverage (Nominal 68.3%)", color="#1f77b4")
    ax.bar(x_p + w/2, cov2, w, label=r"Empirical $2\sigma$ Coverage (Nominal 95.4%)", color="#2ca02c")
    ax.axhline(0.683, color="#1f77b4", linestyle="--", lw=1.2)
    ax.axhline(0.954, color="#2ca02c", linestyle="--", lw=1.2)
    ax.set_xticks(x_p)
    ax.set_xticklabels([p.replace("_", "\n") for p in pats])
    ax.set_ylabel("Empirical Coverage Fraction")
    ax.set_title("Nominal Interval Coverage across Mask Families")
    ax.set_ylim(0.0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3c: Standardized Residual Distribution Moments
    ax = axes[1, 0]
    z_means = [r["std_residual_mean"] for r in test_rows_v11]
    z_stds = [r["std_residual_std"] for r in test_rows_v11]
    ax.bar(x_p - w/2, z_means, w, label="Residual Mean (Ideal: 0.0)", color="#ff7f0e")
    ax.bar(x_p + w/2, z_stds, w, label="Residual Std Dev (Ideal: 1.0)", color="#9467bd")
    ax.axhline(0.0, color="#ff7f0e", linestyle="--", lw=1)
    ax.axhline(1.0, color="#9467bd", linestyle="--", lw=1)
    ax.set_xticks(x_p)
    ax.set_xticklabels([p.replace("_", "\n") for p in pats])
    ax.set_ylabel("Value")
    ax.set_title("Standardized Residual Moments (z = e / sqrt(u))")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3d: Correlation between Error and Uncertainty
    ax = axes[1, 1]
    spearman_vals = [r["sampled_spearman_e2_u"] for r in test_rows_v11]
    pearson_vals = [r["sampled_pearson_e2_u"] for r in test_rows_v11]
    ax.bar(x_p - w/2, spearman_vals, w, label=r"Spearman $\rho(e^2, u)$", color="#e377c2")
    ax.bar(x_p + w/2, pearson_vals, w, label=r"Pearson $r(e^2, u)$", color="#17becf")
    ax.set_xticks(x_p)
    ax.set_xticklabels([p.replace("_", "\n") for p in pats])
    ax.set_ylabel("Correlation Coefficient")
    ax.set_title("Rank & Linear Correlation: Squared Error vs. Predicted u")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "uncertainty_calibration.png", dpi=200)
    plt.close()
    print("Saved uncertainty_calibration.png", flush=True)

    # Figure 4: Qualitative Reconstruction Maps
    if qualitative_samples:
        fig, axes = plt.subplots(len(qualitative_samples), 4, figsize=(16, 4 * len(qualitative_samples)))
        if len(qualitative_samples) == 1:
            axes = np.expand_dims(axes, 0)
        for row_idx, sample in enumerate(qualitative_samples):
            # Show By channel (index 1) at middle time t=12
            t_mid = 12
            ch_idx = 1
            gt = sample["target"][ch_idx, t_mid]
            pr = sample["pred"][ch_idx, t_mid]
            err = np.abs(gt - pr)
            var_map = sample["var"][ch_idx, t_mid]
            mask_t = sample["mask"][0, t_mid] < 0.5  # unobserved

            im0 = axes[row_idx, 0].imshow(gt, cmap="viridis")
            axes[row_idx, 0].set_title(f"{sample['task']} (t={t_mid})\nGround Truth By")
            plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046)

            im1 = axes[row_idx, 1].imshow(pr, cmap="viridis")
            axes[row_idx, 1].set_title(f"Predicted By (beta={sample['beta']})")
            plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046)

            im2 = axes[row_idx, 2].imshow(err, cmap="magma")
            axes[row_idx, 2].set_title(f"Absolute Error |y - y_hat|")
            plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046)

            im3 = axes[row_idx, 3].imshow(np.sqrt(var_map), cmap="plasma")
            axes[row_idx, 3].set_title(r"Predicted Uncertainty $\sqrt{u}$")
            plt.colorbar(im3, ax=axes[row_idx, 3], fraction=0.046)

        plt.tight_layout()
        plt.savefig(OUTPUT / "spatial_and_temporal_error_maps.png", dpi=200)
        plt.close()
        print("Saved spatial_and_temporal_error_maps.png", flush=True)

    print("\n=== Evaluation Completed Successfully ===", flush=True)

if __name__ == "__main__":
    main()
