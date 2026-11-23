#!/usr/bin/env python3
"""Comprehensive matched scientific evaluation of 20-epoch GKL continuations:
Control (v9 masking) vs. Experiment (v11 shared temporal + circular temporal block).

Both models continued from the exact same fresh GKL A+B 256-epoch checkpoint:
  runs/multi_beta_three_phase_gkl_joint_grad_globalcosine_256ep_20261119/final.pt

Evaluates:
  1. Training history & convergence integrity (epochs 257-276)
  2. 5 standard validation tasks (val & test sets)
  3. Forward forecast horizons h=1..23 (especially h20..23)
  4. Reverse temporal extrapolation (hindsight forecasting)
  5. Centered & off-center sparse temporal observation regimes
  6. Calibration (predicted u vs actual MSE, u/MSE)
  7. Spatial error ranking (frame-level and space-time Spearman rho)
  8. Modality breakouts (Magnetic vs. Density)
  9. Paired bootstrap 95% confidence intervals (v11 - v9)
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

OUTPUT = PROJECT / "evaluations/matched_gkl_masking_continuation_20ep_20261121"
SRC_CKPT = PROJECT / "runs/multi_beta_three_phase_gkl_joint_grad_globalcosine_256ep_20261119/final.pt"
V9_DIR = PROJECT / "runs/multi_beta_gkl_v9_control_20ep_20261121"
V11_DIR = PROJECT / "runs/multi_beta_gkl_v11_shared_temporal_20ep_20261121"

V9_CKPT = V9_DIR / "final.pt"
V11_CKPT = V11_DIR / "final.pt"

SEED = 1234
FORWARD_HORIZONS = list(range(1, 24))
FOCUS_HORIZONS = [18, 20, 22, 23]
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


def json_ready(v):
    if isinstance(v, dict):
        return {str(k): json_ready(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_ready(val) for val in v]
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    return v


def dump_json(path: Path, v) -> None:
    path.write_text(json.dumps(json_ready(v), indent=2, allow_nan=True))


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
def infer(model, model_input):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
        "sum_u": 0.0,
        "count": 0,
        "window_mse": [],
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


def finish_bucket(bucket: dict):
    residual = np.concatenate(bucket["residual_samples"]) if bucket["residual_samples"] else np.empty(0)
    variance = np.concatenate(bucket["u_samples"]) if bucket["u_samples"] else np.empty(0)
    mean_mse = bucket["sum_e2"] / bucket["count"] if bucket["count"] > 0 else 0.0
    mean_u = bucket["sum_u"] / bucket["count"] if bucket["count"] > 0 else 0.0
    e2_sample = np.square(residual)
    return {
        "voxel_count": bucket["count"],
        "window_count": len(bucket["window_mse"]),
        "mean_mse": mean_mse,
        "median_window_mse": float(np.median(bucket["window_mse"])) if bucket["window_mse"] else 0.0,
        "mean_predicted_u": mean_u,
        "median_window_mean_u": float(np.median(bucket["window_u"])) if bucket["window_u"] else 0.0,
        "median_predicted_u_sampled": float(np.median(variance)) if variance.size else 0.0,
        "predicted_u_over_actual_mse": (mean_u / mean_mse) if mean_mse > 0 else 0.0,
        "actual_mse_over_predicted_u": (mean_mse / mean_u) if mean_u > 0 else 0.0,
        "sampled_pearson_e2_u": float(stats.pearsonr(e2_sample, variance).statistic) if len(residual) > 1 else float("nan"),
        "sampled_spearman_e2_u": float(stats.spearmanr(e2_sample, variance).statistic) if len(residual) > 1 else float("nan"),
    }


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42):
    """Compute 95% bootstrap confidence interval of paired difference (a - b)."""
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
    print("Matched 20-Epoch GKL Masking Continuation Scientific Evaluation")
    print("Control (v9 masking) vs. Experiment (v11 shared temporal masking)")
    print("=" * 80, flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # -------------------------------------------------------------------------
    # 1. Training Completion & Integrity Audit
    # -------------------------------------------------------------------------
    print("\n--- Auditing Training Histories & Checkpoints ---", flush=True)
    src_ckpt = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
    v9_ckpt = torch.load(V9_CKPT, map_location="cpu", weights_only=False)
    v11_ckpt = torch.load(V11_CKPT, map_location="cpu", weights_only=False)

    v9_latest = torch.load(V9_DIR / "latest.pt", map_location="cpu", weights_only=False)
    v11_latest = torch.load(V11_DIR / "latest.pt", map_location="cpu", weights_only=False)

    src_h = json.loads((PROJECT / "runs/multi_beta_three_phase_gkl_joint_grad_globalcosine_256ep_20261119/history.json").read_text())
    v9_cfg = json.loads((V9_DIR / "config.json").read_text())
    v11_cfg = json.loads((V11_DIR / "config.json").read_text())

    v9_h = json.loads((V9_DIR / "history.json").read_text())
    v11_h = json.loads((V11_DIR / "history.json").read_text())

    assert src_ckpt["epoch"] == 256, f"Source epoch was {src_ckpt['epoch']}, expected 256"
    assert v9_ckpt["epoch"] == 276, f"v9 final epoch was {v9_ckpt['epoch']}, expected 276"
    assert v11_ckpt["epoch"] == 276, f"v11 final epoch was {v11_ckpt['epoch']}, expected 276"
    assert v9_latest["epoch"] == 276, f"v9 latest epoch was {v9_latest['epoch']}, expected 276"
    assert v11_latest["epoch"] == 276, f"v11 latest epoch was {v11_latest['epoch']}, expected 276"

    # Dataset split check
    assert v9_ckpt["split_sha256"] == src_ckpt["split_sha256"]
    assert v11_ckpt["split_sha256"] == src_ckpt["split_sha256"]
    validate_checkpoint_split(v9_ckpt)
    validate_checkpoint_split(v11_ckpt)

    # Learning rate check
    v9_opt_lr = v9_ckpt["optimizer"]["param_groups"][0]["lr"]
    v11_opt_lr = v11_ckpt["optimizer"]["param_groups"][0]["lr"]
    assert math.isclose(v9_opt_lr, 1e-5), f"v9 final LR={v9_opt_lr}, expected 1e-5"
    assert math.isclose(v11_opt_lr, 1e-5), f"v11 final LR={v11_opt_lr}, expected 1e-5"

    # Numerical stability check
    v9_nonfinite = sum(r.get("nonfinite_events", 0) for r in v9_h)
    v11_nonfinite = sum(r.get("nonfinite_events", 0) for r in v11_h)
    v9_amp_skips = sum(r.get("amp_skipped_steps", 0.0) for r in v9_h)
    v11_amp_skips = sum(r.get("amp_skipped_steps", 0.0) for r in v11_h)

    assert v9_nonfinite == 0, f"v9 nonfinite events: {v9_nonfinite}"
    assert v11_nonfinite == 0, f"v11 nonfinite events: {v11_nonfinite}"
    assert v9_amp_skips == 0.0, f"v9 AMP skips: {v9_amp_skips}"
    assert v11_amp_skips == 0.0, f"v11 AMP skips: {v11_amp_skips}"

    v9_wall_total = sum(r["epoch_seconds"] for r in v9_h)
    v11_wall_total = sum(r["epoch_seconds"] for r in v11_h)

    training_audit = {
        "experiments": {
            "v9_control": {
                "checkpoint": str(V9_CKPT),
                "source_checkpoint": str(SRC_CKPT),
                "source_epoch": 256,
                "final_epoch": 276,
                "completed_epochs": len(v9_h),
                "status": "COMPLETED",
                "effective_learning_rate": v9_opt_lr,
                "curriculum_phase": "rollout_settled (Phase C)",
                "lambda_v": 1.0,
                "lambda_r": 1.0,
                "uncertainty_loss": "generalized_kl",
                "gradient_routing": {"path_a": "ON", "path_b": "ON", "reuse_grad": "ON"},
                "masking_version": v9_cfg.get("masking_version", "v9"),
                "nonfinite_events": v9_nonfinite,
                "amp_skipped_steps": v9_amp_skips,
                "mean_epoch_wall_seconds": v9_wall_total / len(v9_h),
                "total_wall_hours": v9_wall_total / 3600.0,
                "final_train_mse": v9_h[-1]["train_mse"],
                "final_val_mse": v9_h[-1]["val_mse"],
                "final_val_gkl": v9_h[-1].get("val_generalized_kl", 0.0),
                "final_val_per_pattern": v9_h[-1].get("val_per_pattern", {}),
            },
            "v11_experiment": {
                "checkpoint": str(V11_CKPT),
                "source_checkpoint": str(SRC_CKPT),
                "source_epoch": 256,
                "final_epoch": 276,
                "completed_epochs": len(v11_h),
                "status": "COMPLETED",
                "effective_learning_rate": v11_opt_lr,
                "curriculum_phase": "rollout_settled (Phase C)",
                "lambda_v": 1.0,
                "lambda_r": 1.0,
                "uncertainty_loss": "generalized_kl",
                "gradient_routing": {"path_a": "ON", "path_b": "ON", "reuse_grad": "ON"},
                "masking_version": v11_cfg.get("masking_version", "v11"),
                "nonfinite_events": v11_nonfinite,
                "amp_skipped_steps": v11_amp_skips,
                "mean_epoch_wall_seconds": v11_wall_total / len(v11_h),
                "total_wall_hours": v11_wall_total / 3600.0,
                "final_train_mse": v11_h[-1]["train_mse"],
                "final_val_mse": v11_h[-1]["val_mse"],
                "final_val_gkl": v11_h[-1].get("val_generalized_kl", 0.0),
                "final_val_per_pattern": v11_h[-1].get("val_per_pattern", {}),
            },
        },
        "source_checkpoint": {
            "checkpoint": str(SRC_CKPT),
            "epoch": 256,
            "final_val_mse": src_h[-1]["val_mse"],
        },
        "experimental_integrity_verified": True,
    }
    dump_json(OUTPUT / "training_audit.json", training_audit)
    print("Wrote training_audit.json", flush=True)

    # -------------------------------------------------------------------------
    # 2. Build training_history.csv
    # -------------------------------------------------------------------------
    def get_pat_val(r, pat):
        vbp = r.get("val_per_pattern", {})
        val = vbp.get(pat, {})
        return val.get("mse", "") if isinstance(val, dict) else val

    history_rows = []
    for model_name, h in [("v9_control", v9_h), ("v11_experiment", v11_h)]:
        for r in h:
            c = r["curriculum"]
            history_rows.append({
                "epoch": r["epoch"],
                "model": model_name,
                "learning_rate": r["learning_rate"],
                "train_mse": r["train_mse"],
                "val_mse": r["val_mse"],
                "val_gkl": r.get("val_generalized_kl", 0.0),
                "val_spatial_random": get_pat_val(r, "spatial_random"),
                "val_spatial_grid": get_pat_val(r, "spatial_grid"),
                "val_spatial_block": get_pat_val(r, "spatial_block"),
                "val_temporal_random": get_pat_val(r, "temporal_random"),
                "val_temporal_block": get_pat_val(r, "temporal_block"),
                "L_mu_0": c.get("L_mu_0", r["train_mse"]),
                "L_mu_1": c.get("L_mu_1", 0.0),
                "L_var_0": c.get("L_var_0", 0.0),
                "L_var_1": c.get("L_var_1", 0.0),
                "gradient_norm_mean": r.get("gradient_norm_mean", 0.0),
                "nonfinite_events": r.get("nonfinite_events", 0),
                "amp_skipped_steps": r.get("amp_skipped_steps", 0.0),
                "epoch_seconds": r.get("epoch_seconds", 0.0),
            })
    write_csv(OUTPUT / "training_history.csv", history_rows)
    print("Wrote training_history.csv", flush=True)

    # -------------------------------------------------------------------------
    # 3. Load Dataset & Test Windows for Inference
    # -------------------------------------------------------------------------
    print("\n--- Initializing Dataset & Inference Engine ---", flush=True)
    split = load_fixed_split()
    dataset = VPICWindowDataset(
        h5_dir=v11_cfg["h5_dir"],
        betas=v11_cfg["betas"],
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
    print(f"Loaded dataset: 125 val windows, 125 test windows (Seed {SEED})", flush=True)

    models_to_eval = [
        ("v9_control", V9_CKPT),
        ("v11_experiment", V11_CKPT),
    ]

    all_standard_rows = []
    all_forecast_rows = []
    all_temporal_regime_rows = []
    
    # Store window-level metrics for paired bootstrap CIs
    window_standard_mse = defaultdict(dict)  # (model, role, task, ch) -> {idx: mse}
    window_forecast_mse = defaultdict(dict)  # (model, h, ch) -> {idx: mse}
    window_forecast_u = defaultdict(dict)    # (model, h, ch) -> {idx: u}
    window_regime_mse = defaultdict(dict)    # (model, regime, ch) -> {idx: mse}
    window_spatial_spearman = defaultdict(dict) # (model, h, ch) -> {idx: rho}

    for model_name, ckpt_path in models_to_eval:
        print(f"\n>>> Running Evaluation for {model_name} on GPU <<<", flush=True)
        ckpt_meta, model, mean_norm, std_norm = load_model(ckpt_path, device)

        # ---------------------------------------------------------------------
        # 3a. Standard Mask Benchmark (Val & Test across 5 tasks)
        # ---------------------------------------------------------------------
        print(f"[{model_name}] Evaluating standard mask benchmark...", flush=True)
        rng = np.random.default_rng(SEED)
        for role, indices in role_indices.items():
            buckets = defaultdict(new_bucket)
            for batch_number, (chosen, block, _meta) in enumerate(batches(dataset, indices, batch_size=4)):
                target = ((block.to(device) - mean_norm) / (std_norm + 1e-8)).float()
                noise_gen = torch.Generator(device=device).manual_seed(
                    SEED + 100000 * (role == "test") + batch_number
                )
                noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

                for task in MASK_PATTERNS:
                    mask, _ = make_fixed_validation_mask(task, target.shape, device=device, dtype=target.dtype)
                    input_var = 1.0 - mask
                    model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
                    pred, var = infer(model, model_input)
                    res = pred - target
                    unknown_all = mask < 0.5  # unobserved voxels

                    res_np = res.cpu().numpy()
                    unknown_np = unknown_all.cpu().numpy()
                    for ch, ci in GROUPS.items():
                        ci = list(ci)
                        for b, idx in enumerate(chosen):
                            u_b = unknown_np[b, ci]
                            r_b = res_np[b, ci][u_b]
                            if r_b.size:
                                w_mse = float(np.mean(np.square(r_b)))
                                window_standard_mse[(model_name, role, task, ch)][idx] = w_mse

                        add_metrics(
                            buckets[(task, ch)],
                            res[:, ci],
                            var[:, ci],
                            unknown_all[:, ci],
                            rng,
                        )

            for (task, ch), bucket in buckets.items():
                summary = finish_bucket(bucket)
                all_standard_rows.append({
                    "model": model_name,
                    "role": role,
                    "task": task,
                    "channel": ch,
                    **summary,
                })
            print(f"[{model_name}] Standard {role} complete", flush=True)

        # ---------------------------------------------------------------------
        # 3b. Forward Forecast Horizon Evaluation (h = 1..23 on test set)
        # ---------------------------------------------------------------------
        print(f"[{model_name}] Evaluating forward forecast horizon sweep (h = 1..23)...", flush=True)
        forecast_buckets = defaultdict(new_bucket)
        forecast_rng = np.random.default_rng(SEED + 9)
        test_indices = role_indices["test"]

        for batch_number, (chosen, block, _meta) in enumerate(batches(dataset, test_indices, batch_size=4)):
            target = ((block.to(device) - mean_norm) / (std_norm + 1e-8)).float()
            noise_gen = torch.Generator(device=device).manual_seed(SEED + 200000 + batch_number)
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

            for h in FORWARD_HORIZONS:
                prefix = 24 - h
                mask = torch.ones_like(target)
                mask[:, :, prefix:] = 0.0
                input_var = 1.0 - mask
                model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
                unknown_all = mask < 0.5

                pred, var = infer(model, model_input)
                res = pred - target

                for ch, ci in GROUPS.items():
                    ci = list(ci)
                    add_metrics(
                        forecast_buckets[(h, ch)],
                        res[:, ci],
                        var[:, ci],
                        unknown_all[:, ci],
                        forecast_rng,
                    )

                res_np = res.cpu().numpy()
                var_np = var.cpu().numpy()
                for ch, ci in GROUPS.items():
                    ci = list(ci)
                    for b, idx in enumerate(chosen):
                        e_all = np.square(res_np[b, ci, prefix:])
                        u_all = var_np[b, ci, prefix:]
                        window_forecast_mse[(model_name, h, ch)][idx] = float(np.mean(e_all))
                        window_forecast_u[(model_name, h, ch)][idx] = float(np.mean(u_all))

                # Focus horizons spatial ranking (Spearman rho)
                if h in FOCUS_HORIZONS:
                    for ch, ci in GROUPS.items():
                        ci = list(ci)
                        for b, idx in enumerate(chosen):
                            e_frames = (
                                np.square(res_np[b, ci, prefix:], dtype=np.float64)
                                .transpose(1, 0, 2, 3)
                                .reshape(h, -1)
                            )
                            u_frames = (
                                var_np[b, ci, prefix:]
                                .astype(np.float64)
                                .transpose(1, 0, 2, 3)
                                .reshape(h, -1)
                            )
                            er = rankdata(e_frames, axis=1)
                            ur = rankdata(u_frames, axis=1)
                            er -= er.mean(1, keepdims=True)
                            ur -= ur.mean(1, keepdims=True)
                            denom = np.sqrt((er * er).sum(1) * (ur * ur).sum(1))
                            frame_rho = np.where(denom > 0, (er * ur).sum(1) / denom, 0.0)
                            window_spatial_spearman[(model_name, h, ch)][idx] = float(np.mean(frame_rho))

        for (h, ch), bucket in sorted(forecast_buckets.items()):
            summary = finish_bucket(bucket)
            all_forecast_rows.append({
                "model": model_name,
                "horizon": h,
                "visible_prefix": 24 - h,
                "channel": ch,
                **summary,
            })
        print(f"[{model_name}] Forward forecast sweep complete", flush=True)

        # ---------------------------------------------------------------------
        # 3c. Reverse Extrapolation & Sparse Centered/Off-Center Observations
        # ---------------------------------------------------------------------
        print(f"[{model_name}] Evaluating reverse extrapolation & sparse regimes...", flush=True)
        regime_buckets = defaultdict(new_bucket)
        regime_rng = np.random.default_rng(SEED + 99)

        # Define regimes: name -> mask generation function
        regimes = {}
        for rh in REVERSE_HORIZONS:
            # Reverse: observe last 24-rh frames, predict first rh frames
            regimes[f"reverse_h{rh}"] = lambda t_shape, h=rh: torch.cat([
                torch.zeros((t_shape[0], t_shape[1], h, t_shape[3], t_shape[4]), device=device),
                torch.ones((t_shape[0], t_shape[1], 24 - h, t_shape[3], t_shape[4]), device=device)
            ], dim=2)

        # Centered observations
        # Middle 8 observed (t=8..15), ends hidden (0..7, 16..23)
        regimes["centered_mid8"] = lambda t_shape: torch.cat([
            torch.zeros((t_shape[0], t_shape[1], 8, t_shape[3], t_shape[4]), device=device),
            torch.ones((t_shape[0], t_shape[1], 8, t_shape[3], t_shape[4]), device=device),
            torch.zeros((t_shape[0], t_shape[1], 8, t_shape[3], t_shape[4]), device=device)
        ], dim=2)
        # Middle 16 observed (t=4..19), ends hidden (0..3, 20..23)
        regimes["centered_mid16"] = lambda t_shape: torch.cat([
            torch.zeros((t_shape[0], t_shape[1], 4, t_shape[3], t_shape[4]), device=device),
            torch.ones((t_shape[0], t_shape[1], 16, t_shape[3], t_shape[4]), device=device),
            torch.zeros((t_shape[0], t_shape[1], 4, t_shape[3], t_shape[4]), device=device)
        ], dim=2)

        # Off-center observations
        # Early 8 observed (t=4..11), others hidden
        regimes["offcenter_early8"] = lambda t_shape: torch.cat([
            torch.zeros((t_shape[0], t_shape[1], 4, t_shape[3], t_shape[4]), device=device),
            torch.ones((t_shape[0], t_shape[1], 8, t_shape[3], t_shape[4]), device=device),
            torch.zeros((t_shape[0], t_shape[1], 12, t_shape[3], t_shape[4]), device=device)
        ], dim=2)
        # Late 8 observed (t=12..19), others hidden
        regimes["offcenter_late8"] = lambda t_shape: torch.cat([
            torch.zeros((t_shape[0], t_shape[1], 12, t_shape[3], t_shape[4]), device=device),
            torch.ones((t_shape[0], t_shape[1], 8, t_shape[3], t_shape[4]), device=device),
            torch.zeros((t_shape[0], t_shape[1], 4, t_shape[3], t_shape[4]), device=device)
        ], dim=2)

        # Bridge / Interpolation (endpoints observed, middle hidden)
        regimes["bridge_endpoints8"] = lambda t_shape: torch.cat([
            torch.ones((t_shape[0], t_shape[1], 4, t_shape[3], t_shape[4]), device=device),
            torch.zeros((t_shape[0], t_shape[1], 16, t_shape[3], t_shape[4]), device=device),
            torch.ones((t_shape[0], t_shape[1], 4, t_shape[3], t_shape[4]), device=device)
        ], dim=2)

        for batch_number, (chosen, block, _meta) in enumerate(batches(dataset, test_indices, batch_size=4)):
            target = ((block.to(device) - mean_norm) / (std_norm + 1e-8)).float()
            noise_gen = torch.Generator(device=device).manual_seed(SEED + 300000 + batch_number)
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

            for reg_name, mask_fn in regimes.items():
                mask = mask_fn(target.shape)
                input_var = 1.0 - mask
                model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
                unknown_all = mask < 0.5

                pred, var = infer(model, model_input)
                res = pred - target

                for ch, ci in GROUPS.items():
                    ci = list(ci)
                    add_metrics(
                        regime_buckets[(reg_name, ch)],
                        res[:, ci],
                        var[:, ci],
                        unknown_all[:, ci],
                        regime_rng,
                    )

                res_np = res.cpu().numpy()
                unknown_np = unknown_all.cpu().numpy()
                for ch, ci in GROUPS.items():
                    ci = list(ci)
                    for b, idx in enumerate(chosen):
                        u_b = unknown_np[b, ci]
                        r_b = res_np[b, ci][u_b]
                        if r_b.size:
                            window_regime_mse[(model_name, reg_name, ch)][idx] = float(np.mean(np.square(r_b)))

        for (reg_name, ch), bucket in sorted(regime_buckets.items()):
            summary = finish_bucket(bucket)
            all_temporal_regime_rows.append({
                "model": model_name,
                "regime": reg_name,
                "channel": ch,
                **summary,
            })
        print(f"[{model_name}] Temporal regimes complete", flush=True)

    # -------------------------------------------------------------------------
    # 4. Write Metrics CSVs
    # -------------------------------------------------------------------------
    write_csv(OUTPUT / "standard_mask_metrics.csv", all_standard_rows)
    write_csv(OUTPUT / "forecast_horizon_metrics.csv", all_forecast_rows)
    write_csv(OUTPUT / "temporal_extrapolation_and_sparse_metrics.csv", all_temporal_regime_rows)
    print("Wrote standard_mask_metrics.csv, forecast_horizon_metrics.csv, temporal_extrapolation_and_sparse_metrics.csv", flush=True)

    # -------------------------------------------------------------------------
    # 5. Paired Statistical Comparisons & Bootstrap 95% CIs
    # -------------------------------------------------------------------------
    print("\n--- Computing Paired Differences (v11 - v9) & Bootstrap 95% CIs ---", flush=True)
    paired_rows = []

    # 5a. Standard masks (test set)
    for task in MASK_PATTERNS:
        for ch in ("Magnetic", "Density", "All"):
            v9_dict = window_standard_mse[("v9_control", "test", task, ch)]
            v11_dict = window_standard_mse[("v11_experiment", "test", task, ch)]
            common = sorted(set(v9_dict.keys()) & set(v11_dict.keys()))
            if common:
                a = np.array([v11_dict[i] for i in common])
                b = np.array([v9_dict[i] for i in common])
                diff_mean, ci_low, ci_high = paired_bootstrap_ci(a, b)
                pct_change = (diff_mean / np.mean(b)) * 100.0 if np.mean(b) > 0 else 0.0
                paired_rows.append({
                    "category": "standard_test",
                    "condition": task,
                    "channel": ch,
                    "v9_control_mean": float(np.mean(b)),
                    "v11_experiment_mean": float(np.mean(a)),
                    "paired_diff_mean": diff_mean,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "percent_change": pct_change,
                    "n_windows": len(common),
                    "significant": (ci_low > 0 or ci_high < 0),
                })

    # 5b. Key forecast horizons (h=1, 4, 8, 12, 16, 18, 20, 22, 23)
    key_horizons = [1, 4, 8, 12, 16, 18, 20, 22, 23]
    for h in key_horizons:
        for ch in ("Magnetic", "Density", "All"):
            v9_dict = window_forecast_mse[("v9_control", h, ch)]
            v11_dict = window_forecast_mse[("v11_experiment", h, ch)]
            common = sorted(set(v9_dict.keys()) & set(v11_dict.keys()))
            if common:
                a = np.array([v11_dict[i] for i in common])
                b = np.array([v9_dict[i] for i in common])
                diff_mean, ci_low, ci_high = paired_bootstrap_ci(a, b)
                pct_change = (diff_mean / np.mean(b)) * 100.0 if np.mean(b) > 0 else 0.0
                paired_rows.append({
                    "category": "forecast_horizon",
                    "condition": f"h{h}",
                    "channel": ch,
                    "v9_control_mean": float(np.mean(b)),
                    "v11_experiment_mean": float(np.mean(a)),
                    "paired_diff_mean": diff_mean,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "percent_change": pct_change,
                    "n_windows": len(common),
                    "significant": (ci_low > 0 or ci_high < 0),
                })

    # 5c. Temporal regimes (reverse, centered, offcenter, bridge)
    for reg in regimes.keys():
        for ch in ("Magnetic", "Density", "All"):
            v9_dict = window_regime_mse[("v9_control", reg, ch)]
            v11_dict = window_regime_mse[("v11_experiment", reg, ch)]
            common = sorted(set(v9_dict.keys()) & set(v11_dict.keys()))
            if common:
                a = np.array([v11_dict[i] for i in common])
                b = np.array([v9_dict[i] for i in common])
                diff_mean, ci_low, ci_high = paired_bootstrap_ci(a, b)
                pct_change = (diff_mean / np.mean(b)) * 100.0 if np.mean(b) > 0 else 0.0
                paired_rows.append({
                    "category": "temporal_regime",
                    "condition": reg,
                    "channel": ch,
                    "v9_control_mean": float(np.mean(b)),
                    "v11_experiment_mean": float(np.mean(a)),
                    "paired_diff_mean": diff_mean,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "percent_change": pct_change,
                    "n_windows": len(common),
                    "significant": (ci_low > 0 or ci_high < 0),
                })

    # 5d. Spatial Spearman ranking for focus horizons
    for h in FOCUS_HORIZONS:
        for ch in ("Magnetic", "Density"):
            v9_dict = window_spatial_spearman[("v9_control", h, ch)]
            v11_dict = window_spatial_spearman[("v11_experiment", h, ch)]
            common = sorted(set(v9_dict.keys()) & set(v11_dict.keys()))
            if common:
                a = np.array([v11_dict[i] for i in common])
                b = np.array([v9_dict[i] for i in common])
                diff_mean, ci_low, ci_high = paired_bootstrap_ci(a, b)
                paired_rows.append({
                    "category": "spatial_spearman",
                    "condition": f"h{h}_spearman",
                    "channel": ch,
                    "v9_control_mean": float(np.mean(b)),
                    "v11_experiment_mean": float(np.mean(a)),
                    "paired_diff_mean": diff_mean,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                    "percent_change": (diff_mean / np.mean(b)) * 100.0 if np.mean(b) != 0 else 0.0,
                    "n_windows": len(common),
                    "significant": (ci_low > 0 or ci_high < 0),
                })

    write_csv(OUTPUT / "paired_summary.csv", paired_rows)
    print("Wrote paired_summary.csv", flush=True)

    # -------------------------------------------------------------------------
    # 6. Generate Figures
    # -------------------------------------------------------------------------
    print("\n--- Generating Publication-Quality Figures ---", flush=True)

    # Figure 1: Training History Curves
    epochs = [r["epoch"] for r in v9_h]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Val MSE
    axes[0].plot(epochs, [r["val_mse"] for r in v9_h], "o-", label="v9 Control", color="#1f77b4", linewidth=2)
    axes[0].plot(epochs, [r["val_mse"] for r in v11_h], "s-", label="v11 Shared Temporal", color="#d62728", linewidth=2)
    axes[0].set_title("Validation MSE (Epochs 257–276)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Validation MSE", fontsize=11)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(fontsize=11)

    # Val Uncertainty (GKL)
    axes[1].plot(epochs, [r.get("val_generalized_kl", 0) for r in v9_h], "o-", label="v9 Control", color="#1f77b4", linewidth=2)
    axes[1].plot(epochs, [r.get("val_generalized_kl", 0) for r in v11_h], "s-", label="v11 Shared Temporal", color="#d62728", linewidth=2)
    axes[1].set_title("Validation GKL Loss (Epochs 257–276)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Val GKL", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(fontsize=11)

    # Val temporal_block MSE breakdown
    v9_tb = [get_pat_val(r, "temporal_block") for r in v9_h]
    v11_tb = [get_pat_val(r, "temporal_block") for r in v11_h]
    axes[2].plot(epochs, v9_tb, "o-", label="v9 Control (temporal_block)", color="#1f77b4", linewidth=2)
    axes[2].plot(epochs, v11_tb, "s-", label="v11 Shared Temporal (temporal_block)", color="#d62728", linewidth=2)
    axes[2].set_title("Validation MSE: temporal_block", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("MSE", fontsize=11)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT / "training_curves.png", dpi=200)
    plt.close(fig)

    # Figure 2: Standard Mask Comparison (Test Set)
    tasks = list(MASK_PATTERNS)
    x = np.arange(len(tasks))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    v9_test_mse = [next(r["mean_mse"] for r in all_standard_rows if r["model"] == "v9_control" and r["role"] == "test" and r["task"] == t and r["channel"] == "All") for t in tasks]
    v11_test_mse = [next(r["mean_mse"] for r in all_standard_rows if r["model"] == "v11_experiment" and r["role"] == "test" and r["task"] == t and r["channel"] == "All") for t in tasks]

    axes[0].bar(x - width/2, v9_test_mse, width, label="v9 Control", color="#1f77b4", alpha=0.85)
    axes[0].bar(x + width/2, v11_test_mse, width, label="v11 Shared Temporal", color="#d62728", alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tasks, rotation=20, ha="right", fontsize=10)
    axes[0].set_ylabel("Test MSE (All Channels)", fontsize=11)
    axes[0].set_title("Held-out Test Reconstruction MSE", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[0].legend(fontsize=11)

    v9_test_cal = [next(r["predicted_u_over_actual_mse"] for r in all_standard_rows if r["model"] == "v9_control" and r["role"] == "test" and r["task"] == t and r["channel"] == "All") for t in tasks]
    v11_test_cal = [next(r["predicted_u_over_actual_mse"] for r in all_standard_rows if r["model"] == "v11_experiment" and r["role"] == "test" and r["task"] == t and r["channel"] == "All") for t in tasks]

    axes[1].bar(x - width/2, v9_test_cal, width, label="v9 Control", color="#1f77b4", alpha=0.85)
    axes[1].bar(x + width/2, v11_test_cal, width, label="v11 Shared Temporal", color="#d62728", alpha=0.85)
    axes[1].axhline(1.0, color="black", linestyle="--", alpha=0.7, label="Ideal (u/MSE = 1.0)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tasks, rotation=20, ha="right", fontsize=10)
    axes[1].set_ylabel("Calibration Ratio (u / MSE)", fontsize=11)
    axes[1].set_title("Held-out Calibration Ratio (u / MSE)", fontsize=13, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5, axis="y")
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(OUTPUT / "standard_mask_comparison.png", dpi=200)
    plt.close(fig)

    # Figure 3: Forward Forecast Horizon Curves
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    h_vals = FORWARD_HORIZONS

    for ch, col_idx in [("Magnetic", 0), ("Density", 1)]:
        v9_h_mse = [next(r["mean_mse"] for r in all_forecast_rows if r["model"] == "v9_control" and r["horizon"] == h and r["channel"] == ch) for h in h_vals]
        v11_h_mse = [next(r["mean_mse"] for r in all_forecast_rows if r["model"] == "v11_experiment" and r["horizon"] == h and r["channel"] == ch) for h in h_vals]

        axes[col_idx].plot(h_vals, v9_h_mse, "o-", label="v9 Control", color="#1f77b4", linewidth=2)
        axes[col_idx].plot(h_vals, v11_h_mse, "s-", label="v11 Shared Temporal", color="#d62728", linewidth=2)
        axes[col_idx].set_title(f"Forecast Horizon vs MSE: {ch}", fontsize=13, fontweight="bold")
        axes[col_idx].set_xlabel("Forecast Horizon h (Frames)", fontsize=11)
        axes[col_idx].set_ylabel("Mean MSE", fontsize=11)
        axes[col_idx].grid(True, linestyle="--", alpha=0.6)
        axes[col_idx].legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT / "forecast_horizon_comparison.png", dpi=200)
    plt.close(fig)

    # Figure 4: Temporal Regimes Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    reg_names_display = ["reverse_h4", "reverse_h12", "reverse_h20", "centered_mid8", "centered_mid16", "offcenter_early8", "offcenter_late8", "bridge_endpoints8"]
    x_reg = np.arange(len(reg_names_display))

    for ch, ax_idx in [("Magnetic", 0), ("Density", 1)]:
        v9_reg_mse = [next(r["mean_mse"] for r in all_temporal_regime_rows if r["model"] == "v9_control" and r["regime"] == rn and r["channel"] == ch) for rn in reg_names_display]
        v11_reg_mse = [next(r["mean_mse"] for r in all_temporal_regime_rows if r["model"] == "v11_experiment" and r["regime"] == rn and r["channel"] == ch) for rn in reg_names_display]

        axes[ax_idx].bar(x_reg - width/2, v9_reg_mse, width, label="v9 Control", color="#1f77b4", alpha=0.85)
        axes[ax_idx].bar(x_reg + width/2, v11_reg_mse, width, label="v11 Shared Temporal", color="#d62728", alpha=0.85)
        axes[ax_idx].set_xticks(x_reg)
        axes[ax_idx].set_xticklabels(reg_names_display, rotation=35, ha="right", fontsize=9)
        axes[ax_idx].set_ylabel(f"MSE ({ch})", fontsize=11)
        axes[ax_idx].set_title(f"Sparse Temporal Regimes: {ch}", fontsize=13, fontweight="bold")
        axes[ax_idx].grid(True, linestyle="--", alpha=0.5, axis="y")
        axes[ax_idx].legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(OUTPUT / "temporal_regimes_comparison.png", dpi=200)
    plt.close(fig)

    print("\n=== Evaluation Completed Successfully ===", flush=True)


if __name__ == "__main__":
    main()
