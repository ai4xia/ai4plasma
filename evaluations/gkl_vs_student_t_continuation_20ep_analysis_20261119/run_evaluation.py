#!/usr/bin/env python3
"""Comprehensive paired scientific evaluation of Generalized KL (GKL) vs
matched Student-t unnormalized control 20-epoch continuations from epoch 256 to 276.
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

OUTPUT = PROJECT / "evaluations/gkl_vs_student_t_continuation_20ep_analysis_20261119"
BASE_EVAL_DIR = PROJECT / "evaluations/three_phase_joint_nll_grad_globalcosine_256ep_analysis_20261118"

SRC_CKPT = PROJECT / "runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt"
GKL_DIR = PROJECT / "runs/multi_beta_gkl_continuation_20ep_20261119"
CTRL_DIR = PROJECT / "runs/multi_beta_student_t_unnormalized_control_20ep_20261119"

GKL_CKPT = GKL_DIR / "final.pt"
CTRL_CKPT = CTRL_DIR / "final.pt"

SEED = 1234
HORIZONS = list(range(1, 24))
FOCUS_HORIZONS = [18, 22, 23]
SAMPLE_LIMIT = 256

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


def batches(dataset, indices: list[int], batch_size: int = 2):
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        samples = [dataset[index] for index in chosen]
        yield chosen, torch.stack([s["block"] for s in samples]), [s["metadata"] for s in samples]


def quantiles(e, u, num_bins: int = 10):
    bins = np.minimum((num_bins * (rankdata(u, method="average") - 0.5) / len(u)).astype(int), num_bins - 1)
    return [
        (int(q), float(e[bins == q].sum()), float(u[bins == q].sum()), int((bins == q).sum()))
        for q in np.unique(bins)
    ]


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
    indices = rng.integers(0, n, size=(n_boot, n))
    boot_means = np.mean(diff[indices], axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(np.mean(diff)), float(low), float(high)


def main():
    print("=" * 80)
    print("GKL vs Student-t Matched Control Continuation Scientific Evaluation")
    print("=" * 80, flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # -------------------------------------------------------------------------
    # 1. Training Completion & Integrity Audit
    # -------------------------------------------------------------------------
    print("\n--- Auditing Training Histories & Checkpoints ---", flush=True)
    src_ckpt = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
    gkl_ckpt = torch.load(GKL_CKPT, map_location="cpu", weights_only=False)
    ctrl_ckpt = torch.load(CTRL_CKPT, map_location="cpu", weights_only=False)

    gkl_latest = torch.load(GKL_DIR / "latest.pt", map_location="cpu", weights_only=False)
    ctrl_latest = torch.load(CTRL_DIR / "latest.pt", map_location="cpu", weights_only=False)

    gkl_cfg = json.loads((GKL_DIR / "config.json").read_text())
    ctrl_cfg = json.loads((CTRL_DIR / "config.json").read_text())

    gkl_h = json.loads((GKL_DIR / "history.json").read_text())
    ctrl_h = json.loads((CTRL_DIR / "history.json").read_text())

    # Verification checks
    assert src_ckpt["epoch"] == 256, f"Source epoch was {src_ckpt['epoch']}, expected 256"
    assert gkl_ckpt["epoch"] == 276, f"GKL final epoch was {gkl_ckpt['epoch']}, expected 276"
    assert ctrl_ckpt["epoch"] == 276, f"Control final epoch was {ctrl_ckpt['epoch']}, expected 276"
    assert gkl_latest["epoch"] == 276, f"GKL latest epoch was {gkl_latest['epoch']}, expected 276"
    assert ctrl_latest["epoch"] == 276, f"Control latest epoch was {ctrl_latest['epoch']}, expected 276"

    # Split SHA256 check
    assert gkl_ckpt["split_sha256"] == src_ckpt["split_sha256"]
    assert ctrl_ckpt["split_sha256"] == src_ckpt["split_sha256"]
    validate_checkpoint_split(gkl_ckpt)
    validate_checkpoint_split(ctrl_ckpt)

    # Optimizer state check
    assert "optimizer" in gkl_ckpt and len(gkl_ckpt["optimizer"]["param_groups"]) > 0
    assert "optimizer" in ctrl_ckpt and len(ctrl_ckpt["optimizer"]["param_groups"]) > 0
    gkl_opt_lr = gkl_ckpt["optimizer"]["param_groups"][0]["lr"]
    ctrl_opt_lr = ctrl_ckpt["optimizer"]["param_groups"][0]["lr"]
    assert math.isclose(gkl_opt_lr, 1e-5), f"GKL final LR={gkl_opt_lr}, expected 1e-5"
    assert math.isclose(ctrl_opt_lr, 1e-5), f"Control final LR={ctrl_opt_lr}, expected 1e-5"

    # Nonfinite events & AMP skips
    gkl_nonfinite = sum(r.get("nonfinite_events", 0) for r in gkl_h)
    ctrl_nonfinite = sum(r.get("nonfinite_events", 0) for r in ctrl_h)
    gkl_amp_skips = sum(r.get("amp_skipped_steps", 0.0) for r in gkl_h)
    ctrl_amp_skips = sum(r.get("amp_skipped_steps", 0.0) for r in ctrl_h)

    assert gkl_nonfinite == 0, f"GKL nonfinite events: {gkl_nonfinite}"
    assert ctrl_nonfinite == 0, f"Control nonfinite events: {ctrl_nonfinite}"
    assert gkl_amp_skips == 0.0, f"GKL AMP skips: {gkl_amp_skips}"
    assert ctrl_amp_skips == 0.0, f"Control AMP skips: {ctrl_amp_skips}"

    # Verify GKL transition details
    # Epoch 257: alpha in (0, 1), Epoch 258: alpha in (0, 1), Epoch 259+: alpha == 1.0, student_t_nll == 0.0
    ep257_alpha = gkl_h[0]["curriculum"]["gkl_alpha"]
    ep258_alpha = gkl_h[1]["curriculum"]["gkl_alpha"]
    ep259_alpha = gkl_h[2]["curriculum"]["gkl_alpha"]
    assert 0 < ep257_alpha < 1, f"Epoch 257 alpha: {ep257_alpha}"
    assert 0 < ep258_alpha < 1, f"Epoch 258 alpha: {ep258_alpha}"
    assert ep258_alpha > ep257_alpha, f"Alpha didn't increase: {ep258_alpha} vs {ep257_alpha}"
    assert math.isclose(ep259_alpha, 1.0), f"Epoch 259 alpha: {ep259_alpha}"

    for r in gkl_h[2:]:
        assert math.isclose(r["curriculum"]["gkl_alpha"], 1.0)
        assert math.isclose(r["curriculum"]["L_student_t_0"], 0.0)
        assert math.isclose(r["curriculum"]["L_student_t_1"], 0.0)

    # Wall times
    gkl_wall_total = sum(r["epoch_seconds"] for r in gkl_h)
    ctrl_wall_total = sum(r["epoch_seconds"] for r in ctrl_h)
    gkl_wall_mean = gkl_wall_total / len(gkl_h)
    ctrl_wall_mean = ctrl_wall_total / len(ctrl_h)

    # Dump training_audit.json
    training_audit = {
        "experiments": {
            "gkl_continuation_20ep": {
                "checkpoint": str(GKL_CKPT),
                "source_checkpoint": str(SRC_CKPT),
                "source_epoch": 256,
                "final_epoch": 276,
                "completed_epochs": len(gkl_h),
                "status": "COMPLETED",
                "effective_learning_rate": gkl_opt_lr,
                "curriculum_phase": "rollout_settled (Phase C)",
                "lambda_v": 1.0,
                "lambda_r": 1.0,
                "uncertainty_loss": "generalized_kl",
                "gradient_routing": {"path_a": "OFF (detached)", "path_b": "ON"},
                "transition_completed": True,
                "transition_horizon_epochs": 2,
                "final_alpha": 1.0,
                "nonfinite_events": gkl_nonfinite,
                "amp_skipped_steps": gkl_amp_skips,
                "mean_epoch_wall_seconds": gkl_wall_mean,
                "total_wall_seconds": gkl_wall_total,
                "total_wall_hours": gkl_wall_total / 3600.0,
                "final_train_mse": gkl_h[-1]["train_mse"],
                "final_val_mse": gkl_h[-1]["val_mse"],
                "final_gkl_loss_0": gkl_h[-1]["curriculum"]["L_gkl_0"],
                "final_gkl_loss_1": gkl_h[-1]["curriculum"]["L_gkl_1"],
                "final_student_t_nll": 0.0,
                "final_gradient_norm_mean": gkl_h[-1]["gradient_norm_mean"],
            },
            "student_t_unnormalized_control_20ep": {
                "checkpoint": str(CTRL_CKPT),
                "source_checkpoint": str(SRC_CKPT),
                "source_epoch": 256,
                "final_epoch": 276,
                "completed_epochs": len(ctrl_h),
                "status": "COMPLETED",
                "effective_learning_rate": ctrl_opt_lr,
                "curriculum_phase": "rollout_settled (Phase C)",
                "lambda_v": 1.0,
                "lambda_r": 1.0,
                "uncertainty_loss": "student_t_nll (df=3.0)",
                "gradient_routing": {"path_a": "ON", "path_b": "ON"},
                "transition_completed": "N/A (Control)",
                "nonfinite_events": ctrl_nonfinite,
                "amp_skipped_steps": ctrl_amp_skips,
                "mean_epoch_wall_seconds": ctrl_wall_mean,
                "total_wall_seconds": ctrl_wall_total,
                "total_wall_hours": ctrl_wall_total / 3600.0,
                "final_train_mse": ctrl_h[-1]["train_mse"],
                "final_val_mse": ctrl_h[-1]["val_mse"],
                "final_student_t_nll_0": ctrl_h[-1]["curriculum"]["L_var_0"],
                "final_student_t_nll_1": ctrl_h[-1]["curriculum"]["L_var_1"],
                "final_gradient_norm_mean": ctrl_h[-1]["gradient_norm_mean"],
            },
        },
        "source_checkpoint": {
            "checkpoint": str(SRC_CKPT),
            "epoch": 256,
            "val_mse": 0.017454145196825265,
            "train_mse": 0.014294,
        },
        "experimental_integrity_verified": True,
    }
    dump_json(OUTPUT / "training_audit.json", training_audit)
    print("Wrote training_audit.json", flush=True)

    # -------------------------------------------------------------------------
    # 2. Build training_history.csv
    # -------------------------------------------------------------------------
    print("\n--- Compiling training_history.csv ---", flush=True)
    history_rows = []
    for model_name, h in [("gkl_epoch276", gkl_h), ("student_t_control_epoch276", ctrl_h)]:
        for r in h:
            c = r["curriculum"]
            is_gkl = (model_name == "gkl_epoch276")
            history_rows.append({
                "epoch": r["epoch"],
                "model": model_name,
                "learning_rate": r["learning_rate"],
                "train_mse": r["train_mse"],
                "val_mse": r["val_mse"],
                "L_mu_0": c.get("L_mu_0", r["train_mse"]),
                "L_mu_1": c.get("L_mu_1", 0.0),
                "raw_gkl_loss_0": c.get("L_gkl_0", 0.0) if is_gkl else "",
                "raw_gkl_loss_1": c.get("L_gkl_1", 0.0) if is_gkl else "",
                "raw_student_t_nll_0": c.get("L_student_t_0", c.get("L_var_0", 0.0)) if is_gkl else c.get("L_var_0", 0.0),
                "raw_student_t_nll_1": c.get("L_student_t_1", c.get("L_var_1", 0.0)) if is_gkl else c.get("L_var_1", 0.0),
                "gkl_alpha": c.get("gkl_alpha", 1.0 if is_gkl and r["epoch"] >= 259 else (0.0 if not is_gkl else "")),
                "gradient_norm_mean": r.get("gradient_norm_mean", 0.0),
                "gradient_norm_max": r.get("gradient_norm_max", 0.0),
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
        h5_dir=gkl_cfg["h5_dir"],
        betas=gkl_cfg["betas"],
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
        ("gkl_epoch276", GKL_CKPT),
        ("student_t_control_epoch276", CTRL_CKPT),
    ]

    all_standard_rows = []
    all_forecast_rows = []
    all_frame_rows = []
    all_paired_rows = []
    all_spatial_rows = []

    # Dictionaries to store per-window values for paired statistical testing
    model_window_standard_mse = defaultdict(dict)  # (model, role, task, ch) -> {idx: mse}
    model_window_forecast_mse = defaultdict(dict)  # (model, h, ch) -> {idx: mse}
    model_window_forecast_u = defaultdict(dict)    # (model, h, ch) -> {idx: u}
    model_window_spatial_spearman = defaultdict(dict) # (model, h, ch) -> {idx: spearman}

    # Samples for conditional calibration
    model_calibration_samples = defaultdict(list) # (model, ch) -> list of (e2_sample, u_sample)

    for model_name, ckpt_path in models_to_eval:
        print(f"\n>>> Running Pass-0 Evaluation for {model_name} on GPU <<<", flush=True)
        ckpt_meta, model, mean_norm, std_norm = load_model(ckpt_path, device)

        # 3a. Standard Mask Benchmark (Val & Test across 5 tasks, Pass 0 ONLY)
        print(f"[{model_name}] Evaluating standard mask benchmark (pass 0)...", flush=True)
        rng = np.random.default_rng(SEED)
        for role, indices in role_indices.items():
            buckets = defaultdict(new_bucket)
            for batch_number, (chosen, block, _meta) in enumerate(batches(dataset, indices)):
                target = ((block.to(device) - mean_norm) / (std_norm + 1e-8)).float()
                noise_gen = torch.Generator(device=device).manual_seed(
                    SEED + 100000 * (role == "test") + batch_number
                )
                noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

                for task in MASK_PATTERNS:
                    mask, _ = make_fixed_validation_mask(task, target.shape, device=device, dtype=target.dtype)
                    input_var = 1.0 - mask
                    model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
                    pred0, var0 = infer(model, model_input)
                    res0 = pred0 - target
                    unknown_all = mask < 0.5  # Strictly hidden voxels

                    # Store window MSE for paired comparison
                    res0_np = res0.cpu().numpy()
                    unknown_np = unknown_all.cpu().numpy()
                    for ch, ci in GROUPS.items():
                        ci = list(ci)
                        for b, idx in enumerate(chosen):
                            u_b = unknown_np[b, ci]
                            r_b = res0_np[b, ci][u_b]
                            if r_b.size:
                                w_mse = float(np.mean(np.square(r_b)))
                                model_window_standard_mse[(model_name, role, task, ch)][idx] = w_mse

                        add_metrics(
                            buckets[("pass0", task, task, ch)],
                            res0[:, ci],
                            var0[:, ci],
                            unknown_all[:, ci],
                            rng,
                        )

            for (pass_name, source_task, eff_task, ch), bucket in buckets.items():
                summary = finish_bucket(bucket)
                all_standard_rows.append({
                    "model": model_name,
                    "role": role,
                    "pass": pass_name,
                    "source_task": source_task,
                    "effective_observation_task": eff_task,
                    "channel": ch,
                    **summary,
                })
            print(f"[{model_name}] standard {role} complete", flush=True)

        # 3b. Forecast Horizon Evaluation (h = 1..23 on 125 test windows)
        print(f"[{model_name}] Evaluating forecast horizon sweep (h = 1..23)...", flush=True)
        forecast_buckets = defaultdict(new_bucket)
        forecast_rng = np.random.default_rng(SEED + 9)

        frame_buckets = defaultdict(lambda: np.zeros(3, dtype=np.float64))
        paired_frame_dict = {}

        spatial_window_records = defaultdict(list)
        spatial_pooled = defaultdict(list)

        test_indices = role_indices["test"]
        for batch_number, (chosen, block, _meta) in enumerate(batches(dataset, test_indices)):
            target = ((block.to(device) - mean_norm) / (std_norm + 1e-8)).float()
            noise_gen = torch.Generator(device=device).manual_seed(SEED + 200000 + batch_number)
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

            for h in HORIZONS:
                prefix = 24 - h
                mask = torch.ones_like(target)
                mask[:, :, prefix:] = 0.0
                input_var = 1.0 - mask
                model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)
                unknown_all = mask < 0.5

                pred, var = infer(model, model_input)
                res = pred - target

                # Record metrics across horizons
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
                    # Window-level metrics for h
                    for b, idx in enumerate(chosen):
                        e_all = np.square(res_np[b, ci, prefix:])
                        u_all = var_np[b, ci, prefix:]
                        model_window_forecast_mse[(model_name, h, ch)][idx] = float(np.mean(e_all))
                        model_window_forecast_u[(model_name, h, ch)][idx] = float(np.mean(u_all))

                    for t in range(prefix, 24):
                        e_t = np.square(res_np[:, ci, t], dtype=np.float64)
                        u_t = var_np[:, ci, t].astype(np.float64)
                        frame_buckets[(h, ch, t)] += [e_t.sum(), u_t.sum(), e_t.size]
                        for b, idx in enumerate(chosen):
                            w_mse = float(e_t[b].mean())
                            w_u = float(u_t[b].mean())
                            paired_frame_dict[(h, ch, t, idx)] = (w_mse, w_u)

                # Focus horizons spatial ranking (18, 22, 23)
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
                            e_flat, u_flat = e_frames.ravel(), u_frames.ravel()

                            er = rankdata(e_frames, axis=1)
                            ur = rankdata(u_frames, axis=1)
                            er -= er.mean(1, keepdims=True)
                            ur -= ur.mean(1, keepdims=True)
                            denom = np.sqrt((er * er).sum(1) * (ur * ur).sum(1))
                            frame_rho = np.where(denom > 0, (er * ur).sum(1) / denom, 0.0)
                            mean_frame_spatial_spearman_exact = float(np.mean(frame_rho))

                            model_window_spatial_spearman[(model_name, h, ch)][idx] = mean_frame_spatial_spearman_exact

                            rho_st = (
                                float(spearmanr(e_flat, u_flat).statistic)
                                if e_flat.size > 1
                                else float("nan")
                            )
                            qwindow = quantiles(e_flat, u_flat)
                            low, high = qwindow[0], qwindow[-1]
                            high_over_low = (
                                (high[1] / high[3]) / (low[1] / low[3])
                                if (low[1] > 0 and low[3] > 0)
                                else float("nan")
                            )

                            spatial_window_records[(h, ch)].append({
                                "dataset_index": idx,
                                "mean_frame_spatial_spearman_exact": mean_frame_spatial_spearman_exact,
                                "spearman_space_time_exact": rho_st,
                                "high_u_decile_mse_over_low": high_over_low,
                            })

                            rng_sample = np.random.default_rng(SEED + 300000 + idx)
                            take = rng_sample.choice(e_flat.size, min(4096, e_flat.size), replace=False)
                            spatial_pooled[(h, ch)].append((e_flat[take], u_flat[take]))

                            # Collect samples for conditional calibration across hidden voxels
                            if ch in ("Magnetic", "Density") and h == 18:
                                model_calibration_samples[(model_name, ch)].append((e_flat[take], u_flat[take]))

            if batch_number == 0 or (batch_number + 1) % 15 == 0 or batch_number + 1 == len(test_indices) // 2 + 1:
                print(
                    f"[{model_name}] Forecast: {min((batch_number + 1) * 2, len(test_indices))}/{len(test_indices)} windows complete",
                    flush=True,
                )

        # Record Forecast Horizon Rows
        for (h, ch), bucket in sorted(forecast_buckets.items()):
            summary = finish_bucket(bucket)
            all_forecast_rows.append({
                "model": model_name,
                "horizon": h,
                "visible_prefix": 24 - h,
                "channel": ch,
                **summary,
            })

        # Record Per-Frame Rows
        model_frame_rows = []
        for (h, ch, t), (se, su, n) in sorted(frame_buckets.items()):
            prefix = 24 - h
            mean_mse = se / n
            mean_u = su / n
            row = {
                "model": model_name,
                "pass_name": "pass0",
                "horizon": h,
                "channel": ch,
                "local_frame": t,
                "last_pass0_observed_frame": prefix - 1,
                "forecast_lead_from_pass0": t - prefix + 1,
                "mean_mse": mean_mse,
                "mean_predicted_u": mean_u,
                "u_over_mse": mean_u / mean_mse if mean_mse > 0 else float("nan"),
                "voxel_count": int(n),
                "window_count": len(test_indices),
            }
            model_frame_rows.append(row)
            all_frame_rows.append(row)

        # Record h22 vs h23 Paired Rows
        frame_dict_indexed = {(r["horizon"], r["channel"], r["local_frame"]): r for r in model_frame_rows}
        for ch in ("Bx", "By", "Bz", "Density", "Magnetic", "All"):
            for t in range(2, 24):
                a = frame_dict_indexed[(22, ch, t)]
                b = frame_dict_indexed[(23, ch, t)]
                mse_ratios = [
                    paired_frame_dict[(23, ch, t, idx)][0] / paired_frame_dict[(22, ch, t, idx)][0]
                    for idx in test_indices
                ]
                u_ratios = [
                    paired_frame_dict[(23, ch, t, idx)][1] / paired_frame_dict[(22, ch, t, idx)][1]
                    for idx in test_indices
                ]
                cal_ratios = [
                    (paired_frame_dict[(23, ch, t, idx)][1] / paired_frame_dict[(23, ch, t, idx)][0])
                    / (paired_frame_dict[(22, ch, t, idx)][1] / paired_frame_dict[(22, ch, t, idx)][0])
                    for idx in test_indices
                ]
                all_paired_rows.append({
                    "model": model_name,
                    "channel": ch,
                    "local_frame": t,
                    "lead_h22": t - 1,
                    "lead_h23": t,
                    "mse_h22": a["mean_mse"],
                    "mse_h23": b["mean_mse"],
                    "u_h22": a["mean_predicted_u"],
                    "u_h23": b["mean_predicted_u"],
                    "calibration_h22": a["u_over_mse"],
                    "calibration_h23": b["u_over_mse"],
                    "mse_h23_over_h22": b["mean_mse"] / a["mean_mse"],
                    "mse_h23_over_h22_median_window": float(np.median(mse_ratios)),
                    "mse_h23_over_h22_fraction_windows_below_one": float(np.mean(np.array(mse_ratios) < 1.0)),
                    "u_h23_over_h22": b["mean_predicted_u"] / a["mean_predicted_u"],
                    "u_h23_over_h22_median_window": float(np.median(u_ratios)),
                    "u_h23_over_h22_fraction_windows_below_one": float(np.mean(np.array(u_ratios) < 1.0)),
                    "calibration_h23_over_h22": b["u_over_mse"] / a["u_over_mse"],
                    "calibration_h23_over_h22_median_window": float(np.median(cal_ratios)),
                    "calibration_h23_over_h22_fraction_windows_below_one": float(np.mean(np.array(cal_ratios) < 1.0)),
                })

        # Record Spatial Uncertainty Rows
        for (h, ch), win_list in sorted(spatial_window_records.items()):
            samples = spatial_pooled[(h, ch)]
            e_pooled = np.concatenate([x[0] for x in samples])
            u_pooled = np.concatenate([x[1] for x in samples])
            pooled_sp = float(spearmanr(e_pooled, u_pooled).statistic) if len(e_pooled) > 1 else float("nan")
            mean_frame_sp = float(np.mean([w["mean_frame_spatial_spearman_exact"] for w in win_list]))
            mean_win_sp = float(np.mean([w["spearman_space_time_exact"] for w in win_list]))
            med_win_sp = float(np.median([w["spearman_space_time_exact"] for w in win_list]))
            med_high_low = float(np.median([w["high_u_decile_mse_over_low"] for w in win_list]))

            all_spatial_rows.append({
                "model": model_name,
                "horizon": h,
                "channel": ch,
                "scope": "ranking_summary",
                "quantile": "",
                "mean_frame_spatial_spearman": mean_frame_sp,
                "pooled_sampled_spearman": pooled_sp,
                "pooled_sample_count": len(e_pooled),
                "mean_window_spearman": mean_win_sp,
                "median_window_spearman": med_win_sp,
                "median_window_high_over_low_mse": med_high_low,
            })
            for q, se, su, n in quantiles(e_pooled, u_pooled):
                all_spatial_rows.append({
                    "model": model_name,
                    "horizon": h,
                    "channel": ch,
                    "scope": "pooled_sampled_decile",
                    "quantile": q + 1,
                    "mean_mse": se / n,
                    "mean_predicted_u": su / n,
                    "u_over_mse": su / se if se > 0 else float("nan"),
                    "voxel_count": int(n),
                })

        del model
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Integrate Archived Baseline Results
    # -------------------------------------------------------------------------
    print("\n--- Integrating Primary Baseline (new_joint_256ep) and Historical Reference (original_256ep) ---", flush=True)
    with (BASE_EVAL_DIR / "standard_mask_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in ("new_joint_256ep", "original_256ep"):
                all_standard_rows.append(r)

    with (BASE_EVAL_DIR / "forecast_horizon_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in ("new_joint_256ep", "original_256ep"):
                all_forecast_rows.append(r)

    with (BASE_EVAL_DIR / "per_frame_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in ("new_joint_256ep", "original_256ep"):
                all_frame_rows.append(r)

    with (BASE_EVAL_DIR / "h22_vs_h23_paired.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in ("new_joint_256ep", "original_256ep"):
                all_paired_rows.append(r)

    with (BASE_EVAL_DIR / "spatial_uncertainty_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in ("new_joint_256ep", "original_256ep"):
                all_spatial_rows.append(r)

    # Compute u_over_h1 and mse_over_h1 across all models in forecast rows
    by_model_h_ch = {}
    for r in all_forecast_rows:
        for k in (
            "mean_mse",
            "mean_predicted_u",
            "predicted_u_over_actual_mse",
            "actual_mse_over_predicted_u",
            "median_window_mse",
            "median_window_mean_u",
        ):
            if k in r and r[k] != "":
                r[k] = float(r[k])
        by_model_h_ch[(r["model"], int(r["horizon"]), r["channel"])] = r

    for r in all_forecast_rows:
        m, h, ch = r["model"], int(r["horizon"]), r["channel"]
        h1_row = by_model_h_ch.get((m, 1, ch))
        if h1_row is not None:
            r["u_over_h1"] = (
                r["mean_predicted_u"] / h1_row["mean_predicted_u"]
                if h1_row["mean_predicted_u"] > 0
                else 0.0
            )
            r["mse_over_h1"] = (
                r["mean_mse"] / h1_row["mean_mse"] if h1_row["mean_mse"] > 0 else 0.0
            )

    # Write output CSVs
    write_csv(OUTPUT / "standard_mask_metrics.csv", all_standard_rows)
    print("Wrote standard_mask_metrics.csv", flush=True)

    write_csv(OUTPUT / "forecast_horizon_metrics.csv", all_forecast_rows)
    print("Wrote forecast_horizon_metrics.csv", flush=True)

    write_csv(OUTPUT / "per_frame_metrics.csv", all_frame_rows)
    print("Wrote per_frame_metrics.csv", flush=True)

    write_csv(OUTPUT / "h22_vs_h23_paired.csv", all_paired_rows)
    print("Wrote h22_vs_h23_paired.csv", flush=True)

    write_csv(OUTPUT / "spatial_uncertainty_metrics.csv", all_spatial_rows)
    print("Wrote spatial_uncertainty_metrics.csv", flush=True)

    # -------------------------------------------------------------------------
    # 5. Build forecast_horizon_comparison.csv
    # -------------------------------------------------------------------------
    print("\n--- Compiling forecast_horizon_comparison.csv ---", flush=True)
    comparison_rows = []
    for h in HORIZONS:
        for ch in ["All", "Magnetic", "Density", "Bx", "By", "Bz"]:
            base_row = by_model_h_ch[("new_joint_256ep", h, ch)]
            gkl_row = by_model_h_ch[("gkl_epoch276", h, ch)]
            ctrl_row = by_model_h_ch[("student_t_control_epoch276", h, ch)]
            orig_row = by_model_h_ch.get(("original_256ep", h, ch))

            row = {
                "horizon": h,
                "visible_prefix": 24 - h,
                "channel": ch,
                # Primary baseline (epoch 256)
                "base_256ep_mse": base_row["mean_mse"],
                "base_256ep_mean_u": base_row["mean_predicted_u"],
                "base_256ep_u_over_mse": base_row["predicted_u_over_actual_mse"],
                "base_256ep_u_over_h1": base_row.get("u_over_h1", 1.0),
                "base_256ep_mse_over_h1": base_row.get("mse_over_h1", 1.0),
                # GKL (epoch 276)
                "gkl_276ep_mse": gkl_row["mean_mse"],
                "gkl_276ep_mean_u": gkl_row["mean_predicted_u"],
                "gkl_276ep_u_over_mse": gkl_row["predicted_u_over_actual_mse"],
                "gkl_276ep_u_over_h1": gkl_row.get("u_over_h1", 1.0),
                "gkl_276ep_mse_over_h1": gkl_row.get("mse_over_h1", 1.0),
                # Control (epoch 276)
                "ctrl_276ep_mse": ctrl_row["mean_mse"],
                "ctrl_276ep_mean_u": ctrl_row["mean_predicted_u"],
                "ctrl_276ep_u_over_mse": ctrl_row["predicted_u_over_actual_mse"],
                "ctrl_276ep_u_over_h1": ctrl_row.get("u_over_h1", 1.0),
                "ctrl_276ep_mse_over_h1": ctrl_row.get("mse_over_h1", 1.0),
                # Historical original (epoch 256)
                "orig_256ep_mse": orig_row["mean_mse"] if orig_row else "",
                "orig_256ep_mean_u": orig_row["mean_predicted_u"] if orig_row else "",
                # Key comparisons & ratios
                "mse_ratio_gkl_over_base": gkl_row["mean_mse"] / base_row["mean_mse"],
                "mse_ratio_gkl_over_ctrl": gkl_row["mean_mse"] / ctrl_row["mean_mse"],
                "mse_ratio_ctrl_over_base": ctrl_row["mean_mse"] / base_row["mean_mse"],
                "u_ratio_gkl_over_base": gkl_row["mean_predicted_u"] / base_row["mean_predicted_u"],
                "u_ratio_gkl_over_ctrl": gkl_row["mean_predicted_u"] / ctrl_row["mean_predicted_u"],
            }
            comparison_rows.append(row)
    write_csv(OUTPUT / "forecast_horizon_comparison.csv", comparison_rows)
    print("Wrote forecast_horizon_comparison.csv", flush=True)

    # -------------------------------------------------------------------------
    # 6. Build calibration_metrics.csv (Conditional-MSE Calibration)
    # -------------------------------------------------------------------------
    print("\n--- Compiling calibration_metrics.csv ---", flush=True)
    cal_rows = []

    # First add archived deciles for new_joint_256ep
    for r in all_spatial_rows:
        if r["scope"] == "pooled_sampled_decile" and r["horizon"] in ("18", "22", "23") and r["channel"] in ("Magnetic", "Density"):
            cal_rows.append({
                "model": r["model"],
                "horizon": r["horizon"],
                "channel": r["channel"],
                "bin_type": "decile",
                "bin_index": r["quantile"],
                "mean_predicted_u": r["mean_predicted_u"],
                "mean_empirical_q": r["mean_mse"],
                "calibration_ratio": r["u_over_mse"],
                "absolute_calibration_error": abs(float(r["mean_predicted_u"]) - float(r["mean_mse"])),
                "voxel_count": r["voxel_count"],
            })

    # Next, compute pooled quantile bins across GKL and Control for h18
    for ch in ("Magnetic", "Density"):
        gkl_samples = model_calibration_samples[("gkl_epoch276", ch)]
        ctrl_samples = model_calibration_samples[("student_t_control_epoch276", ch)]

        gkl_e = np.concatenate([x[0] for x in gkl_samples])
        gkl_u = np.concatenate([x[1] for x in gkl_samples])
        ctrl_e = np.concatenate([x[0] for x in ctrl_samples])
        ctrl_u = np.concatenate([x[1] for x in ctrl_samples])

        pooled_u = np.concatenate([gkl_u, ctrl_u])
        bin_edges = np.percentile(pooled_u, np.linspace(0, 100, 11))
        bin_edges[0] = 0.0
        bin_edges[-1] = np.inf

        for model_name, (e_arr, u_arr) in [("gkl_epoch276", (gkl_e, gkl_u)), ("student_t_control_epoch276", (ctrl_e, ctrl_u))]:
            b_indices = np.digitize(u_arr, bin_edges) - 1
            b_indices = np.clip(b_indices, 0, 9)

            total_voxels = len(u_arr)
            binned_abs_errors = []
            binned_rel_devs = []

            for b in range(10):
                mask_b = (b_indices == b)
                n_b = int(np.sum(mask_b))
                if n_b > 0:
                    u_b = float(np.mean(u_arr[mask_b]))
                    q_b = float(np.mean(e_arr[mask_b]))
                    cal_b = u_b / q_b if q_b > 0 else float("nan")
                    abs_err = abs(u_b - q_b)
                    rel_dev = abs(cal_b - 1.0) if not math.isnan(cal_b) else float("nan")
                else:
                    u_b, q_b, cal_b, abs_err, rel_dev = 0.0, 0.0, float("nan"), 0.0, float("nan")

                binned_abs_errors.append((n_b, abs_err))
                if not math.isnan(rel_dev):
                    binned_rel_devs.append(rel_dev)

                cal_rows.append({
                    "model": model_name,
                    "horizon": 18,
                    "channel": ch,
                    "bin_type": "pooled_boundary_decile",
                    "bin_index": b + 1,
                    "mean_predicted_u": u_b,
                    "mean_empirical_q": q_b,
                    "calibration_ratio": cal_b,
                    "absolute_calibration_error": abs_err,
                    "voxel_count": n_b,
                })

            # Add summary row for binned ACE
            ace = sum(n * err for n, err in binned_abs_errors) / total_voxels if total_voxels > 0 else 0.0
            mean_rel_dev = float(np.mean(binned_rel_devs)) if binned_rel_devs else 0.0
            overall_cal = float(np.mean(u_arr)) / float(np.mean(e_arr))

            cal_rows.append({
                "model": model_name,
                "horizon": 18,
                "channel": ch,
                "bin_type": "summary_metric",
                "bin_index": "ALL",
                "mean_predicted_u": float(np.mean(u_arr)),
                "mean_empirical_q": float(np.mean(e_arr)),
                "calibration_ratio": overall_cal,
                "absolute_calibration_error": ace,
                "voxel_count": total_voxels,
            })

    write_csv(OUTPUT / "calibration_metrics.csv", cal_rows)
    print("Wrote calibration_metrics.csv", flush=True)

    # -------------------------------------------------------------------------
    # 7. Build paired_comparison_summary.csv (Statistical Tests & Bootstrap CIs)
    # -------------------------------------------------------------------------
    print("\n--- Compiling paired_comparison_summary.csv ---", flush=True)
    summary_rows = []

    # Paired comparisons across test windows between GKL and Control
    test_idx_list = role_indices["test"]

    # 1. Standard mask MSE paired differences
    for task in MASK_PATTERNS:
        for ch in ("All", "Magnetic", "Density"):
            gkl_vals = np.array([model_window_standard_mse[("gkl_epoch276", "test", task, ch)][i] for i in test_idx_list])
            ctrl_vals = np.array([model_window_standard_mse[("student_t_control_epoch276", "test", task, ch)][i] for i in test_idx_list])
            mean_diff, ci_low, ci_high = paired_bootstrap_ci(gkl_vals, ctrl_vals)
            rel_change = (np.mean(gkl_vals) - np.mean(ctrl_vals)) / np.mean(ctrl_vals)
            w_stat, p_val = stats.wilcoxon(gkl_vals, ctrl_vals) if not np.all(gkl_vals == ctrl_vals) else (0, 1.0)
            summary_rows.append({
                "domain": "standard_mask_test_mse",
                "task_or_horizon": task,
                "channel": ch,
                "metric": "reconstruction_mse",
                "treatment_mean (GKL)": float(np.mean(gkl_vals)),
                "control_mean (Student-t)": float(np.mean(ctrl_vals)),
                "paired_difference (GKL - Ctrl)": mean_diff,
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "relative_change": rel_change,
                "wilcoxon_p_value": float(p_val),
            })

    # Overall test MSE across 5 tasks
    for ch in ("All", "Magnetic", "Density"):
        gkl_all_tasks = np.mean([
            [model_window_standard_mse[("gkl_epoch276", "test", task, ch)][i] for i in test_idx_list]
            for task in MASK_PATTERNS
        ], axis=0)
        ctrl_all_tasks = np.mean([
            [model_window_standard_mse[("student_t_control_epoch276", "test", task, ch)][i] for i in test_idx_list]
            for task in MASK_PATTERNS
        ], axis=0)
        mean_diff, ci_low, ci_high = paired_bootstrap_ci(gkl_all_tasks, ctrl_all_tasks)
        rel_change = (np.mean(gkl_all_tasks) - np.mean(ctrl_all_tasks)) / np.mean(ctrl_all_tasks)
        w_stat, p_val = stats.wilcoxon(gkl_all_tasks, ctrl_all_tasks)
        summary_rows.append({
            "domain": "standard_mask_test_mse",
            "task_or_horizon": "mean_5_tasks",
            "channel": ch,
            "metric": "reconstruction_mse",
            "treatment_mean (GKL)": float(np.mean(gkl_all_tasks)),
            "control_mean (Student-t)": float(np.mean(ctrl_all_tasks)),
            "paired_difference (GKL - Ctrl)": mean_diff,
            "bootstrap_95ci_low": ci_low,
            "bootstrap_95ci_high": ci_high,
            "relative_change": rel_change,
            "wilcoxon_p_value": float(p_val),
        })

    # 2. Forecast horizon MSE, predicted u, calibration ratio, spatial Spearman for h18, h22, h23
    for h in FOCUS_HORIZONS:
        for ch in ("Magnetic", "Density"):
            # MSE
            gkl_mse = np.array([model_window_forecast_mse[("gkl_epoch276", h, ch)][i] for i in test_idx_list])
            ctrl_mse = np.array([model_window_forecast_mse[("student_t_control_epoch276", h, ch)][i] for i in test_idx_list])
            mean_diff, ci_low, ci_high = paired_bootstrap_ci(gkl_mse, ctrl_mse)
            w_stat, p_val = stats.wilcoxon(gkl_mse, ctrl_mse)
            summary_rows.append({
                "domain": "forecast_horizon",
                "task_or_horizon": f"h{h}",
                "channel": ch,
                "metric": "reconstruction_mse",
                "treatment_mean (GKL)": float(np.mean(gkl_mse)),
                "control_mean (Student-t)": float(np.mean(ctrl_mse)),
                "paired_difference (GKL - Ctrl)": mean_diff,
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "relative_change": (np.mean(gkl_mse) - np.mean(ctrl_mse)) / np.mean(ctrl_mse),
                "wilcoxon_p_value": float(p_val),
            })

            # Predicted u
            gkl_u = np.array([model_window_forecast_u[("gkl_epoch276", h, ch)][i] for i in test_idx_list])
            ctrl_u = np.array([model_window_forecast_u[("student_t_control_epoch276", h, ch)][i] for i in test_idx_list])
            mean_diff, ci_low, ci_high = paired_bootstrap_ci(gkl_u, ctrl_u)
            w_stat, p_val = stats.wilcoxon(gkl_u, ctrl_u)
            summary_rows.append({
                "domain": "forecast_horizon",
                "task_or_horizon": f"h{h}",
                "channel": ch,
                "metric": "predicted_uncertainty_u",
                "treatment_mean (GKL)": float(np.mean(gkl_u)),
                "control_mean (Student-t)": float(np.mean(ctrl_u)),
                "paired_difference (GKL - Ctrl)": mean_diff,
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "relative_change": (np.mean(gkl_u) - np.mean(ctrl_u)) / np.mean(ctrl_u),
                "wilcoxon_p_value": float(p_val),
            })

            # Spatial Spearman
            gkl_sp = np.array([model_window_spatial_spearman[("gkl_epoch276", h, ch)][i] for i in test_idx_list])
            ctrl_sp = np.array([model_window_spatial_spearman[("student_t_control_epoch276", h, ch)][i] for i in test_idx_list])
            mean_diff, ci_low, ci_high = paired_bootstrap_ci(gkl_sp, ctrl_sp)
            w_stat, p_val = stats.wilcoxon(gkl_sp, ctrl_sp)
            summary_rows.append({
                "domain": "forecast_horizon",
                "task_or_horizon": f"h{h}",
                "channel": ch,
                "metric": "spatial_spearman",
                "treatment_mean (GKL)": float(np.mean(gkl_sp)),
                "control_mean (Student-t)": float(np.mean(ctrl_sp)),
                "paired_difference (GKL - Ctrl)": mean_diff,
                "bootstrap_95ci_low": ci_low,
                "bootstrap_95ci_high": ci_high,
                "relative_change": (np.mean(gkl_sp) - np.mean(ctrl_sp)) / np.mean(ctrl_sp),
                "wilcoxon_p_value": float(p_val),
            })

    # 3. Extreme-horizon collapse metric (h22 -> h23)
    for ch in ("Magnetic", "Density", "All"):
        gkl_u22 = np.array([model_window_forecast_u[("gkl_epoch276", 22, ch)][i] for i in test_idx_list])
        gkl_u23 = np.array([model_window_forecast_u[("gkl_epoch276", 23, ch)][i] for i in test_idx_list])
        gkl_e22 = np.array([model_window_forecast_mse[("gkl_epoch276", 22, ch)][i] for i in test_idx_list])
        gkl_e23 = np.array([model_window_forecast_mse[("gkl_epoch276", 23, ch)][i] for i in test_idx_list])

        ctrl_u22 = np.array([model_window_forecast_u[("student_t_control_epoch276", 22, ch)][i] for i in test_idx_list])
        ctrl_u23 = np.array([model_window_forecast_u[("student_t_control_epoch276", 23, ch)][i] for i in test_idx_list])
        ctrl_e22 = np.array([model_window_forecast_mse[("student_t_control_epoch276", 22, ch)][i] for i in test_idx_list])
        ctrl_e23 = np.array([model_window_forecast_mse[("student_t_control_epoch276", 23, ch)][i] for i in test_idx_list])

        gkl_collapse = (gkl_e23 > gkl_e22) & (gkl_u23 < gkl_u22)
        ctrl_collapse = (ctrl_e23 > ctrl_e22) & (ctrl_u23 < ctrl_u22)

        summary_rows.append({
            "domain": "extreme_horizon_collapse",
            "task_or_horizon": "h22_to_h23",
            "channel": ch,
            "metric": "collapse_window_fraction",
            "treatment_mean (GKL)": float(np.mean(gkl_collapse)),
            "control_mean (Student-t)": float(np.mean(ctrl_collapse)),
            "paired_difference (GKL - Ctrl)": float(np.mean(gkl_collapse) - np.mean(ctrl_collapse)),
            "bootstrap_95ci_low": "",
            "bootstrap_95ci_high": "",
            "relative_change": (np.mean(gkl_collapse) - np.mean(ctrl_collapse)) / np.mean(ctrl_collapse) if np.mean(ctrl_collapse) > 0 else 0.0,
            "wilcoxon_p_value": "",
        })

    write_csv(OUTPUT / "paired_comparison_summary.csv", summary_rows)
    print("Wrote paired_comparison_summary.csv", flush=True)

    # -------------------------------------------------------------------------
    # 8. Generate All 6 Publication Figures
    # -------------------------------------------------------------------------
    print("\n--- Generating Publication Figures ---", flush=True)

    # Figure 1: training_curves.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    eps = np.arange(257, 277)
    gkl_val_mse = [r["val_mse"] for r in gkl_h]
    ctrl_val_mse = [r["val_mse"] for r in ctrl_h]
    gkl_train_mse = [r["train_mse"] for r in gkl_h]
    ctrl_train_mse = [r["train_mse"] for r in ctrl_h]

    # Val MSE
    axes[0, 0].axhline(0.017454, color="gray", ls="--", label="Epoch 256 Baseline (0.017454)")
    axes[0, 0].axhline(0.014208, color="black", ls=":", label="Original 256ep (0.014208)")
    axes[0, 0].plot(eps, ctrl_val_mse, marker="o", color="blue", label="Student-t Matched Control")
    axes[0, 0].plot(eps, gkl_val_mse, marker="s", color="crimson", lw=2, label="GKL Continuation")
    axes[0, 0].set_title("Validation Reconstruction MSE (Epochs 257-276)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Validation MSE")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Train MSE
    axes[0, 1].plot(eps, ctrl_train_mse, marker="o", color="blue", label="Student-t Control")
    axes[0, 1].plot(eps, gkl_train_mse, marker="s", color="crimson", lw=2, label="GKL Continuation")
    axes[0, 1].set_title("Training Reconstruction MSE (Epochs 257-276)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Train MSE")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Gradient Norm
    gkl_gnorm = [r["gradient_norm_mean"] for r in gkl_h]
    ctrl_gnorm = [r["gradient_norm_mean"] for r in ctrl_h]
    axes[1, 0].plot(eps, ctrl_gnorm, marker="o", color="blue", label="Student-t Control (~180-210)")
    axes[1, 0].plot(eps, gkl_gnorm, marker="s", color="crimson", lw=2, label="GKL Continuation (0.09-0.11)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Mean Gradient Norm (Log Scale)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Gradient L2 Norm")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # GKL Transition Alpha and Loss
    alphas = [r["curriculum"]["gkl_alpha"] for r in gkl_h]
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    l1 = ax1.plot(eps, alphas, color="purple", lw=2, marker="^", label="Transition Alpha")
    l2 = ax2.plot(eps[2:], [r["curriculum"]["L_gkl_0"] for r in gkl_h[2:]], color="darkorange", lw=2, label="Raw GKL Loss (L_gkl_0)")
    ax1.set_title("GKL Transition Dynamics & Objective Value")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("GKL Alpha", color="purple")
    ax2.set_ylabel("GKL Loss (nats/voxel)", color="darkorange")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right")
    ax1.grid(True, alpha=0.3)

    fig.suptitle("Training & Convergence History: GKL vs Matched Student-t Control (Epochs 257-276)", fontsize=13)
    plt.savefig(OUTPUT / "training_curves.png", dpi=200)
    plt.close()
    print("Saved training_curves.png", flush=True)

    # Figure 2: standard_mask_comparison.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    tasks = list(MASK_PATTERNS)
    x = np.arange(len(tasks))
    w = 0.2

    for ax, role in [(axes[0], "val"), (axes[1], "test")]:
        def get_mses(m):
            return [
                float(next(r["mean_mse"] for r in all_standard_rows if r["model"] == m and r["role"] == role and r["source_task"] == t and r["channel"] == "All" and r["pass"] == "pass0"))
                for t in tasks
            ]
        orig_mses = get_mses("original_256ep")
        base_mses = get_mses("new_joint_256ep")
        ctrl_mses = get_mses("student_t_control_epoch276")
        gkl_mses = get_mses("gkl_epoch276")

        ax.bar(x - 1.5 * w, orig_mses, width=w, label="Original 256ep", color="gray")
        ax.bar(x - 0.5 * w, base_mses, width=w, label="Epoch 256 Baseline", color="darkorange")
        ax.bar(x + 0.5 * w, ctrl_mses, width=w, label="Epoch 276 Control", color="blue")
        ax.bar(x + 1.5 * w, gkl_mses, width=w, label="Epoch 276 GKL", color="crimson")

        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=25)
        ax.set_ylabel("Pass-0 Reconstruction MSE")
        ax.set_title(f"Standard Masks ({role.capitalize()} Set)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Standard Mask Reconstruction Comparison Across 5 Pattern Families", fontsize=13)
    plt.savefig(OUTPUT / "standard_mask_comparison.png", dpi=200)
    plt.close()
    print("Saved standard_mask_comparison.png", flush=True)

    # Figure 3: forecast_horizon_comparison.png
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    models_plot = [
        ("original_256ep", "Original 256ep", "gray", ":"),
        ("new_joint_256ep", "Epoch 256 Baseline", "darkorange", "--"),
        ("student_t_control_epoch276", "Epoch 276 Control", "blue", "-"),
        ("gkl_epoch276", "Epoch 276 GKL", "crimson", "-"),
    ]

    for col, ch in enumerate(["Magnetic", "Density", "All"]):
        # Row 0: Reconstruction MSE vs Horizon
        ax = axes[0, col]
        for m, label, color, ls in models_plot:
            mses = [by_model_h_ch[(m, h, ch)]["mean_mse"] for h in HORIZONS]
            ax.plot(HORIZONS, mses, label=label, color=color, ls=ls, lw=1.5)
        ax.set_title(f"{ch} Reconstruction MSE")
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend()

        # Row 1: Calibration Ratio u / MSE vs Horizon
        ax = axes[1, col]
        ax.axhline(1.0, color="black", ls=":", lw=1.5, label="Ideal Calibration (1.0)")
        for m, label, color, ls in models_plot:
            cals = [by_model_h_ch[(m, h, ch)]["predicted_u_over_actual_mse"] for h in HORIZONS]
            ax.plot(HORIZONS, cals, label=label, color=color, ls=ls, lw=1.5)
        ax.set_title(f"{ch} Calibration Ratio (u / MSE)")
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("Calibration Ratio")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend()

    fig.suptitle("Forecast Horizon Benchmark (h = 1..23): Reconstruction & Calibration Dynamics", fontsize=13)
    plt.savefig(OUTPUT / "forecast_horizon_comparison.png", dpi=200)
    plt.close()
    print("Saved forecast_horizon_comparison.png", flush=True)

    # Figure 4: extreme_horizon_and_per_frame.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # (0,0): h22 vs h23 per-frame predicted uncertainty on shared frames (t = 2..23) for Magnetic
    ax = axes[0, 0]
    frames = np.arange(2, 24)
    for m, label, color, ls in [("new_joint_256ep", "Baseline ep256", "darkorange", "--"),
                                ("student_t_control_epoch276", "Control ep276", "blue", "-"),
                                ("gkl_epoch276", "GKL ep276", "crimson", "-")]:
        u22 = [float(next(r["u_h22"] for r in all_paired_rows if r["model"] == m and r["channel"] == "Magnetic" and int(r["local_frame"]) == t)) for t in frames]
        u23 = [float(next(r["u_h23"] for r in all_paired_rows if r["model"] == m and r["channel"] == "Magnetic" and int(r["local_frame"]) == t)) for t in frames]
        ax.plot(frames, u22, color=color, ls="--", alpha=0.7, label=f"{label} h22")
        ax.plot(frames, u23, color=color, ls="-", lw=2, label=f"{label} h23")
    ax.set_title("Magnetic Predicted Uncertainty: h22 vs h23 (Shared Frames t=2..23)")
    ax.set_xlabel("Local Frame (t)")
    ax.set_ylabel("Mean Predicted u")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (0,1): h23 / h22 ratio across shared frames
    ax = axes[0, 1]
    ax.axhline(1.0, color="black", ls=":", label="No Collapse (1.0)")
    for m, label, color in [("new_joint_256ep", "Baseline ep256", "darkorange"),
                            ("student_t_control_epoch276", "Control ep276", "blue"),
                            ("gkl_epoch276", "GKL ep276", "crimson")]:
        ratios = [float(next(r["u_h23_over_h22"] for r in all_paired_rows if r["model"] == m and r["channel"] == "Magnetic" and int(r["local_frame"]) == t)) for t in frames]
        ax.plot(frames, ratios, color=color, lw=2, label=label)
    ax.set_title("Magnetic Uncertainty Ratio: u(h23) / u(h22)")
    ax.set_xlabel("Local Frame (t)")
    ax.set_ylabel("u(h23) / u(h22)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (1,0): Within-window per-frame uncertainty growth at h=18 (Magnetic)
    ax = axes[1, 0]
    h18_frames = np.arange(6, 24)
    for m, label, color in [("original_256ep", "Original 256ep", "gray"),
                            ("new_joint_256ep", "Baseline ep256", "darkorange"),
                            ("student_t_control_epoch276", "Control ep276", "blue"),
                            ("gkl_epoch276", "GKL ep276", "crimson")]:
        u_f = [float(next(r["mean_predicted_u"] for r in all_frame_rows if r["model"] == m and int(r["horizon"]) == 18 and r["channel"] == "Magnetic" and int(r["local_frame"]) == t)) for t in h18_frames]
        ax.plot(h18_frames - 5, u_f, color=color, lw=2, label=label)
    ax.set_title("Within-Window Uncertainty Growth at h=18 (Magnetic)")
    ax.set_xlabel("Forecast Lead Time (Frames into Future)")
    ax.set_ylabel("Predicted Uncertainty u")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (1,1): Within-window per-frame uncertainty growth at h=18 (Density)
    ax = axes[1, 1]
    for m, label, color in [("original_256ep", "Original 256ep", "gray"),
                            ("new_joint_256ep", "Baseline ep256", "darkorange"),
                            ("student_t_control_epoch276", "Control ep276", "blue"),
                            ("gkl_epoch276", "GKL ep276", "crimson")]:
        u_f = [float(next(r["mean_predicted_u"] for r in all_frame_rows if r["model"] == m and int(r["horizon"]) == 18 and r["channel"] == "Density" and int(r["local_frame"]) == t)) for t in h18_frames]
        ax.plot(h18_frames - 5, u_f, color=color, lw=2, label=label)
    ax.set_title("Within-Window Uncertainty Growth at h=18 (Density)")
    ax.set_xlabel("Forecast Lead Time (Frames into Future)")
    ax.set_ylabel("Predicted Uncertainty u")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("Extreme-Horizon Dynamics (h22 -> h23) and Within-Window Uncertainty Scaling", fontsize=13)
    plt.savefig(OUTPUT / "extreme_horizon_and_per_frame.png", dpi=200)
    plt.close()
    print("Saved extreme_horizon_and_per_frame.png", flush=True)

    # Figure 5: spatial_uncertainty_correlation.png
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    channels_sp = ["Bx", "By", "Bz", "Density", "Magnetic"]
    x = np.arange(len(channels_sp))
    w = 0.22

    for idx, h in enumerate(FOCUS_HORIZONS):
        ax = axes[idx]
        def get_sp(m):
            return [
                float(next(r["mean_frame_spatial_spearman"] for r in all_spatial_rows if r["model"] == m and str(r["horizon"]) == str(h) and r["channel"] == ch and r["scope"] == "ranking_summary"))
                for ch in channels_sp
            ]
        base_sp = get_sp("new_joint_256ep")
        ctrl_sp = get_sp("student_t_control_epoch276")
        gkl_sp = get_sp("gkl_epoch276")

        ax.bar(x - w, base_sp, width=w, label="Baseline ep256", color="darkorange")
        ax.bar(x, ctrl_sp, width=w, label="Control ep276", color="blue")
        ax.bar(x + w, gkl_sp, width=w, label="GKL ep276", color="crimson")

        ax.set_xticks(x)
        ax.set_xticklabels(channels_sp)
        ax.set_ylim(0.0, 0.8)
        ax.set_ylabel("Within-Frame Spatial Spearman")
        ax.set_title(f"Spatial Error Ranking (h={h})")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    fig.suptitle("Spatial Uncertainty Quality: Within-Frame Error-Uncertainty Ranking (Spearman rho)", fontsize=13)
    plt.savefig(OUTPUT / "spatial_uncertainty_correlation.png", dpi=200)
    plt.close()
    print("Saved spatial_uncertainty_correlation.png", flush=True)

    # Figure 6: conditional_mse_reliability_diagrams.png
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for idx, ch in enumerate(["Magnetic", "Density"]):
        ax = axes[idx]
        for m, label, color, marker in [("new_joint_256ep", "Baseline ep256", "darkorange", "o"),
                                        ("student_t_control_epoch276", "Control ep276", "blue", "^"),
                                        ("gkl_epoch276", "GKL ep276", "crimson", "s")]:
            decs = [r for r in all_spatial_rows if r["model"] == m and str(r["horizon"]) == "18" and r["channel"] == ch and r["scope"] == "pooled_sampled_decile"]
            decs = sorted(decs, key=lambda r: int(r["quantile"]))
            u_vals = [float(r["mean_predicted_u"]) for r in decs]
            q_vals = [float(r["mean_mse"]) for r in decs]
            ax.plot(u_vals, q_vals, marker=marker, color=color, lw=2, label=label)

        # Plot ideal y = x reference
        all_vals = u_vals + q_vals
        min_v, max_v = min(all_vals) * 0.5, max(all_vals) * 2.0
        ax.plot([min_v, max_v], [min_v, max_v], color="black", ls=":", label="Ideal y = x")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mean Predicted Uncertainty u (Variance)")
        ax.set_ylabel("Mean Realized Squared Error q = (y - mu)^2")
        ax.set_title(f"{ch} Reliability Diagram (h=18 Deciles)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Distribution-Independent Conditional-MSE Calibration (Reliability Diagrams)", fontsize=13)
    plt.savefig(OUTPUT / "conditional_mse_reliability_diagrams.png", dpi=200)
    plt.close()
    print("Saved conditional_mse_reliability_diagrams.png", flush=True)

    print("\nAll evaluation deliverables and figures successfully generated!", flush=True)


if __name__ == "__main__":
    main()
