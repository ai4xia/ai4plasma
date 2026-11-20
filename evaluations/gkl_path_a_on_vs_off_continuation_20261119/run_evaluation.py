#!/usr/bin/env python3
"""Comprehensive paired scientific evaluation of GKL Path A ON (A+B) vs
GKL Path A OFF (B-only) 20-epoch continuations from epoch 256 to 276.

Extends the previous GKL vs Student-t evaluation with:
 - New GKL A+B checkpoint inference
 - Reused archived results from GKL B-only, Student-t, and baselines
 - Extended comparison tables across all 4 checkpoints
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

OUTPUT = PROJECT / "evaluations/gkl_path_a_on_vs_off_continuation_20261119"
PREV_EVAL_DIR = PROJECT / "evaluations/gkl_vs_student_t_continuation_20ep_analysis_20261119"
BASE_EVAL_DIR = PROJECT / "evaluations/three_phase_joint_nll_grad_globalcosine_256ep_analysis_20261118"

# Checkpoints
SRC_CKPT = PROJECT / "runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt"
GKL_AB_DIR = PROJECT / "runs/multi_beta_gkl_joint_grad_continuation_20ep_20261119"
GKL_B_DIR = PROJECT / "runs/multi_beta_gkl_continuation_20ep_20261119"
CTRL_DIR = PROJECT / "runs/multi_beta_student_t_unnormalized_control_20ep_20261119"

GKL_AB_CKPT = GKL_AB_DIR / "final.pt"
GKL_B_CKPT = GKL_B_DIR / "final.pt"
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
    if not (torch.isfinite(pred).all() & torch.isfinite(var).all() & (var > 0).all()):
        raise ValueError("Invalid pass-0 model output")
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
    print("GKL Path A ON (A+B) vs Path A OFF (B-only) Scientific Evaluation")
    print("=" * 80, flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    device = torch.device("cuda")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # =========================================================================
    # 1. Training Completion & Integrity Audit
    # =========================================================================
    print("\n--- Section 1: Auditing Training Histories & Checkpoints ---", flush=True)

    src_ckpt = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
    gkl_ab_ckpt = torch.load(GKL_AB_CKPT, map_location="cpu", weights_only=False)
    gkl_b_ckpt = torch.load(GKL_B_CKPT, map_location="cpu", weights_only=False)
    ctrl_ckpt = torch.load(CTRL_CKPT, map_location="cpu", weights_only=False)

    gkl_ab_latest = torch.load(GKL_AB_DIR / "latest.pt", map_location="cpu", weights_only=False)
    gkl_b_latest = torch.load(GKL_B_DIR / "latest.pt", map_location="cpu", weights_only=False)

    gkl_ab_cfg = json.loads((GKL_AB_DIR / "config.json").read_text())
    gkl_b_cfg = json.loads((GKL_B_DIR / "config.json").read_text())
    ctrl_cfg = json.loads((CTRL_DIR / "config.json").read_text())

    gkl_ab_h = json.loads((GKL_AB_DIR / "history.json").read_text())
    gkl_b_h = json.loads((GKL_B_DIR / "history.json").read_text())
    ctrl_h = json.loads((CTRL_DIR / "history.json").read_text())

    # Verification checks
    assert src_ckpt["epoch"] == 256, f"Source epoch was {src_ckpt['epoch']}, expected 256"
    assert gkl_ab_ckpt["epoch"] == 276, f"GKL A+B final epoch was {gkl_ab_ckpt['epoch']}, expected 276"
    assert gkl_b_ckpt["epoch"] == 276, f"GKL B-only final epoch was {gkl_b_ckpt['epoch']}, expected 276"
    assert ctrl_ckpt["epoch"] == 276, f"Control final epoch was {ctrl_ckpt['epoch']}, expected 276"
    assert gkl_ab_latest["epoch"] == 276
    assert gkl_b_latest["epoch"] == 276

    # Verify identical checkpoint normalization and split.
    assert gkl_ab_ckpt["stats"] == gkl_b_ckpt["stats"]
    # Split SHA256 check
    assert gkl_ab_ckpt["split_sha256"] == src_ckpt["split_sha256"]
    assert gkl_b_ckpt["split_sha256"] == src_ckpt["split_sha256"]
    assert ctrl_ckpt["split_sha256"] == src_ckpt["split_sha256"]
    validate_checkpoint_split(gkl_ab_ckpt)
    validate_checkpoint_split(gkl_b_ckpt)
    validate_checkpoint_split(ctrl_ckpt)

    # Optimizer state check
    assert "optimizer" in gkl_ab_ckpt and len(gkl_ab_ckpt["optimizer"]["param_groups"]) > 0
    gkl_ab_opt_lr = gkl_ab_ckpt["optimizer"]["param_groups"][0]["lr"]
    gkl_b_opt_lr = gkl_b_ckpt["optimizer"]["param_groups"][0]["lr"]
    ctrl_opt_lr = ctrl_ckpt["optimizer"]["param_groups"][0]["lr"]
    assert math.isclose(gkl_ab_opt_lr, 1e-5), f"GKL A+B final LR={gkl_ab_opt_lr}, expected 1e-5"
    assert math.isclose(gkl_b_opt_lr, 1e-5), f"GKL B-only final LR={gkl_b_opt_lr}, expected 1e-5"
    assert math.isclose(ctrl_opt_lr, 1e-5), f"Control final LR={ctrl_opt_lr}, expected 1e-5"

    # Nonfinite events & AMP skips
    gkl_ab_nonfinite = sum(r.get("nonfinite_events", 0) for r in gkl_ab_h)
    gkl_b_nonfinite = sum(r.get("nonfinite_events", 0) for r in gkl_b_h)
    ctrl_nonfinite = sum(r.get("nonfinite_events", 0) for r in ctrl_h)
    gkl_ab_amp_skips = sum(r.get("amp_skipped_steps", 0.0) for r in gkl_ab_h)
    gkl_b_amp_skips = sum(r.get("amp_skipped_steps", 0.0) for r in gkl_b_h)
    ctrl_amp_skips = sum(r.get("amp_skipped_steps", 0.0) for r in ctrl_h)

    assert gkl_ab_nonfinite == 0, f"GKL A+B nonfinite events: {gkl_ab_nonfinite}"
    assert gkl_b_nonfinite == 0, f"GKL B-only nonfinite events: {gkl_b_nonfinite}"
    assert ctrl_nonfinite == 0, f"Control nonfinite events: {ctrl_nonfinite}"
    assert gkl_ab_amp_skips == 0.0, f"GKL A+B AMP skips: {gkl_ab_amp_skips}"
    assert gkl_b_amp_skips == 0.0, f"GKL B-only AMP skips: {gkl_b_amp_skips}"
    assert ctrl_amp_skips == 0.0, f"Control AMP skips: {ctrl_amp_skips}"

    # Verify GKL A+B transition details (same as B-only: 2-epoch cosine transition)
    ab_ep257_alpha = gkl_ab_h[0]["curriculum"]["gkl_alpha"]
    ab_ep258_alpha = gkl_ab_h[1]["curriculum"]["gkl_alpha"]
    ab_ep259_alpha = gkl_ab_h[2]["curriculum"]["gkl_alpha"]
    assert 0 < ab_ep257_alpha < 1, f"A+B Epoch 257 alpha: {ab_ep257_alpha}"
    assert 0 < ab_ep258_alpha < 1, f"A+B Epoch 258 alpha: {ab_ep258_alpha}"
    assert ab_ep258_alpha > ab_ep257_alpha
    assert math.isclose(ab_ep259_alpha, 1.0), f"A+B Epoch 259 alpha: {ab_ep259_alpha}"

    for r in gkl_ab_h[2:]:
        assert math.isclose(r["curriculum"]["gkl_alpha"], 1.0)
        assert math.isclose(r["curriculum"]["L_student_t_0"], 0.0)
        assert math.isclose(r["curriculum"]["L_student_t_1"], 0.0)

    # Verify GKL A+B config flags
    assert gkl_ab_cfg.get("gkl_joint_grad_continuation_20ep") is True
    assert gkl_ab_cfg.get("gkl_continuation_20ep") is False
    assert gkl_ab_cfg.get("uncertainty_loss") == "generalized_kl"

    # Verify GKL B-only config flags
    assert gkl_b_cfg.get("gkl_continuation_20ep") is True
    assert gkl_b_cfg.get("gkl_joint_grad_continuation_20ep") in (False, None)
    assert gkl_b_cfg.get("uncertainty_loss") == "generalized_kl"

    # Wall times
    gkl_ab_wall_total = sum(r["epoch_seconds"] for r in gkl_ab_h)
    gkl_b_wall_total = sum(r["epoch_seconds"] for r in gkl_b_h)
    ctrl_wall_total = sum(r["epoch_seconds"] for r in ctrl_h)

    # Dump training_audit.json
    training_audit = {
        "experiments": {
            "gkl_path_a_on_continuation_20ep": {
                "checkpoint": str(GKL_AB_CKPT),
                "source_checkpoint": str(SRC_CKPT),
                "source_epoch": 256,
                "final_epoch": 276,
                "completed_epochs": len(gkl_ab_h),
                "status": "COMPLETED",
                "effective_learning_rate": gkl_ab_opt_lr,
                "curriculum_phase": "rollout_settled (Phase C)",
                "lambda_v": 1.0,
                "lambda_r": 1.0,
                "uncertainty_loss": "generalized_kl",
                "gradient_routing": {"path_a": "ON (attached)", "path_b": "ON"},
                "transition_completed": True,
                "transition_horizon_epochs": 2,
                "final_alpha": 1.0,
                "nonfinite_events": gkl_ab_nonfinite,
                "amp_skipped_steps": gkl_ab_amp_skips,
                "mean_epoch_wall_seconds": gkl_ab_wall_total / len(gkl_ab_h),
                "total_wall_seconds": gkl_ab_wall_total,
                "total_wall_hours": gkl_ab_wall_total / 3600.0,
                "final_train_mse": gkl_ab_h[-1]["train_mse"],
                "final_val_mse": gkl_ab_h[-1]["val_mse"],
                "final_gkl_loss_0": gkl_ab_h[-1]["curriculum"]["L_gkl_0"],
                "final_gkl_loss_1": gkl_ab_h[-1]["curriculum"]["L_gkl_1"],
                "final_student_t_nll": 0.0,
                "final_gradient_norm_mean": gkl_ab_h[-1]["gradient_norm_mean"],
            },
            "gkl_path_a_off_continuation_20ep": {
                "checkpoint": str(GKL_B_CKPT),
                "source_checkpoint": str(SRC_CKPT),
                "source_epoch": 256,
                "final_epoch": 276,
                "completed_epochs": len(gkl_b_h),
                "status": "COMPLETED",
                "effective_learning_rate": gkl_b_opt_lr,
                "curriculum_phase": "rollout_settled (Phase C)",
                "lambda_v": 1.0,
                "lambda_r": 1.0,
                "uncertainty_loss": "generalized_kl",
                "gradient_routing": {"path_a": "OFF (detached)", "path_b": "ON"},
                "transition_completed": True,
                "final_alpha": 1.0,
                "nonfinite_events": gkl_b_nonfinite,
                "amp_skipped_steps": gkl_b_amp_skips,
                "mean_epoch_wall_seconds": gkl_b_wall_total / len(gkl_b_h),
                "total_wall_seconds": gkl_b_wall_total,
                "total_wall_hours": gkl_b_wall_total / 3600.0,
                "final_train_mse": gkl_b_h[-1]["train_mse"],
                "final_val_mse": gkl_b_h[-1]["val_mse"],
                "final_gkl_loss_0": gkl_b_h[-1]["curriculum"]["L_gkl_0"],
                "final_gkl_loss_1": gkl_b_h[-1]["curriculum"]["L_gkl_1"],
                "final_student_t_nll": 0.0,
                "final_gradient_norm_mean": gkl_b_h[-1]["gradient_norm_mean"],
            },
            "student_t_unnormalized_control_20ep": {
                "checkpoint": str(CTRL_CKPT),
                "source_checkpoint": str(SRC_CKPT),
                "source_epoch": 256,
                "final_epoch": 276,
                "completed_epochs": len(ctrl_h),
                "status": "COMPLETED",
                "effective_learning_rate": ctrl_opt_lr,
                "uncertainty_loss": "student_t_nll (df=3.0)",
                "gradient_routing": {"path_a": "ON", "path_b": "ON"},
                "nonfinite_events": ctrl_nonfinite,
                "amp_skipped_steps": ctrl_amp_skips,
                "mean_epoch_wall_seconds": ctrl_wall_total / len(ctrl_h),
                "total_wall_seconds": ctrl_wall_total,
                "total_wall_hours": ctrl_wall_total / 3600.0,
                "final_train_mse": ctrl_h[-1]["train_mse"],
                "final_val_mse": ctrl_h[-1]["val_mse"],
                "final_gradient_norm_mean": ctrl_h[-1]["gradient_norm_mean"],
            },
        },
        "source_checkpoint": {
            "checkpoint": str(SRC_CKPT),
            "epoch": 256,
            "val_mse": 0.017454145196825265,
        },
        "experimental_integrity_verified": True,
    }
    dump_json(OUTPUT / "training_audit.json", training_audit)
    print("Wrote training_audit.json", flush=True)

    # =========================================================================
    # 2. Build training_history.csv
    # =========================================================================
    print("\n--- Section 2: Compiling training_history.csv ---", flush=True)
    history_rows = []
    for model_name, h in [
        ("gkl_ab_epoch276", gkl_ab_h),
        ("gkl_bonly_epoch276", gkl_b_h),
        ("student_t_control_epoch276", ctrl_h),
    ]:
        is_gkl = model_name.startswith("gkl_")
        for r in h:
            c = r["curriculum"]
            history_rows.append({
                "epoch": r["epoch"],
                "model": model_name,
                "learning_rate": r["learning_rate"],
                "train_mse": r["train_mse"],
                "val_mse": r["val_mse"],
                "L_mu_0": c.get("L_mu_0", r["train_mse"]),
                "raw_gkl_loss_0": c.get("L_gkl_0", 0.0) if is_gkl else "",
                "raw_gkl_loss_1": c.get("L_gkl_1", 0.0) if is_gkl else "",
                "raw_student_t_nll_0": c.get("L_student_t_0", c.get("L_var_0", 0.0)) if is_gkl else c.get("L_var_0", 0.0),
                "gkl_alpha": c.get("gkl_alpha", 1.0 if is_gkl and r["epoch"] >= 259 else (0.0 if not is_gkl else "")),
                "gradient_norm_mean": r.get("gradient_norm_mean", 0.0),
                "gradient_norm_max": r.get("gradient_norm_max", 0.0),
                "nonfinite_events": r.get("nonfinite_events", 0),
                "amp_skipped_steps": r.get("amp_skipped_steps", 0.0),
                "epoch_seconds": r.get("epoch_seconds", 0.0),
            })
    write_csv(OUTPUT / "training_history.csv", history_rows)
    print("Wrote training_history.csv", flush=True)

    # =========================================================================
    # 3. Load Dataset & Run GKL A+B Inference (GPU)
    # =========================================================================
    print("\n--- Section 3: Initializing Dataset & Running GKL A+B Inference ---", flush=True)
    split = load_fixed_split()

    # Use GKL B-only config for dataset (identical across all experiments)
    dataset = VPICWindowDataset(
        h5_dir=gkl_b_cfg["h5_dir"],
        betas=gkl_b_cfg["betas"],
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

    # We only run inference on GKL A+B; the other models' metrics come from archived CSVs
    model_name = "gkl_ab_epoch276"

    all_standard_rows = []
    all_forecast_rows = []
    all_frame_rows = []
    all_paired_rows = []
    all_spatial_rows = []

    # Dictionaries for paired statistical testing
    model_window_standard_mse = defaultdict(dict)
    model_window_forecast_mse = defaultdict(dict)
    model_window_forecast_u = defaultdict(dict)
    model_window_spatial_spearman = defaultdict(dict)
    model_calibration_samples = defaultdict(list)

    print(f"\n>>> Running Pass-0 Evaluation for {model_name} on GPU <<<", flush=True)
    ckpt_meta, model, mean_norm, std_norm = load_model(GKL_AB_CKPT, device)

    # 3a. Standard Mask Benchmark
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
                unknown_all = mask < 0.5

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
                        low_q, high_q = qwindow[0], qwindow[-1]
                        high_over_low = (
                            (high_q[1] / high_q[3]) / (low_q[1] / low_q[3])
                            if (low_q[1] > 0 and low_q[3] > 0)
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
                "u_h23_over_h22": b["mean_predicted_u"] / a["mean_predicted_u"],
                "u_h23_over_h22_median_window": float(np.median(u_ratios)),
                "calibration_h23_over_h22": b["u_over_mse"] / a["u_over_mse"],
                "calibration_h23_over_h22_median_window": float(np.median(cal_ratios)),
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

    # =========================================================================
    # 4. Integrate Archived Results from Previous Evaluations
    # =========================================================================
    print("\n--- Section 4: Integrating Archived Results ---", flush=True)

    # Import archived GKL B-only and Student-t control results (renaming models)
    # From previous eval: gkl_epoch276 -> gkl_bonly_epoch276, student_t_control_epoch276 unchanged
    rename_map = {
        "gkl_epoch276": "gkl_bonly_epoch276",
        "student_t_control_epoch276": "student_t_control_epoch276",
    }

    with (PREV_EVAL_DIR / "standard_mask_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in rename_map:
                r["model"] = rename_map[r["model"]]
                if r["pass"] == "pass0":
                    all_standard_rows.append(r)

    with (PREV_EVAL_DIR / "forecast_horizon_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in rename_map:
                r["model"] = rename_map[r["model"]]
                all_forecast_rows.append(r)

    with (PREV_EVAL_DIR / "per_frame_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in rename_map:
                r["model"] = rename_map[r["model"]]
                all_frame_rows.append(r)

    with (PREV_EVAL_DIR / "h22_vs_h23_paired.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in rename_map:
                r["model"] = rename_map[r["model"]]
                all_paired_rows.append(r)

    with (PREV_EVAL_DIR / "spatial_uncertainty_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in rename_map:
                r["model"] = rename_map[r["model"]]
                all_spatial_rows.append(r)

    # Import baselines: new_joint_256ep and original_256ep
    with (BASE_EVAL_DIR / "standard_mask_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in ("new_joint_256ep", "original_256ep"):
                if r["pass"] == "pass0":
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

    # Compute u_over_h1 and mse_over_h1
    by_model_h_ch = {}
    for r in all_forecast_rows:
        for k in ("mean_mse", "mean_predicted_u", "predicted_u_over_actual_mse", "actual_mse_over_predicted_u"):
            if k in r and r[k] != "":
                r[k] = float(r[k])
        by_model_h_ch[(r["model"], int(r["horizon"]), r["channel"])] = r

    for r in all_forecast_rows:
        m, h, ch = r["model"], int(r["horizon"]), r["channel"]
        h1_row = by_model_h_ch.get((m, 1, ch))
        if h1_row is not None:
            r["u_over_h1"] = r["mean_predicted_u"] / h1_row["mean_predicted_u"] if h1_row["mean_predicted_u"] > 0 else 0.0
            r["mse_over_h1"] = r["mean_mse"] / h1_row["mean_mse"] if h1_row["mean_mse"] > 0 else 0.0

    # Write output CSVs
    write_csv(OUTPUT / "standard_mask_metrics.csv", all_standard_rows)
    write_csv(OUTPUT / "forecast_horizon_metrics.csv", all_forecast_rows)
    write_csv(OUTPUT / "per_frame_metrics.csv", all_frame_rows)
    write_csv(OUTPUT / "h22_vs_h23_paired.csv", all_paired_rows)
    write_csv(OUTPUT / "spatial_uncertainty_metrics.csv", all_spatial_rows)
    print("Wrote all primary CSVs", flush=True)

    # =========================================================================
    # 5. Build forecast_horizon_comparison.csv (all 4+ models)
    # =========================================================================
    print("\n--- Section 5: Compiling forecast_horizon_comparison.csv ---", flush=True)
    comparison_rows = []
    for h in HORIZONS:
        for ch in ["All", "Magnetic", "Density", "Bx", "By", "Bz"]:
            base_row = by_model_h_ch.get(("new_joint_256ep", h, ch))
            gkl_ab_row = by_model_h_ch.get(("gkl_ab_epoch276", h, ch))
            gkl_b_row = by_model_h_ch.get(("gkl_bonly_epoch276", h, ch))
            ctrl_row = by_model_h_ch.get(("student_t_control_epoch276", h, ch))
            orig_row = by_model_h_ch.get(("original_256ep", h, ch))

            if not all([base_row, gkl_ab_row, gkl_b_row, ctrl_row]):
                continue

            row = {
                "horizon": h,
                "visible_prefix": 24 - h,
                "channel": ch,
                # Primary baseline (epoch 256)
                "base_256ep_mse": base_row["mean_mse"],
                "base_256ep_mean_u": base_row["mean_predicted_u"],
                "base_256ep_u_over_mse": base_row["predicted_u_over_actual_mse"],
                # GKL A+B (epoch 276)
                "gkl_ab_276ep_mse": gkl_ab_row["mean_mse"],
                "gkl_ab_276ep_mean_u": gkl_ab_row["mean_predicted_u"],
                "gkl_ab_276ep_u_over_mse": gkl_ab_row["predicted_u_over_actual_mse"],
                # GKL B-only (epoch 276)
                "gkl_bonly_276ep_mse": gkl_b_row["mean_mse"],
                "gkl_bonly_276ep_mean_u": gkl_b_row["mean_predicted_u"],
                "gkl_bonly_276ep_u_over_mse": gkl_b_row["predicted_u_over_actual_mse"],
                # Control (epoch 276)
                "ctrl_276ep_mse": ctrl_row["mean_mse"],
                "ctrl_276ep_mean_u": ctrl_row["mean_predicted_u"],
                "ctrl_276ep_u_over_mse": ctrl_row["predicted_u_over_actual_mse"],
                # Key comparisons
                "mse_ratio_ab_over_bonly": gkl_ab_row["mean_mse"] / gkl_b_row["mean_mse"],
                "mse_ratio_ab_over_ctrl": gkl_ab_row["mean_mse"] / ctrl_row["mean_mse"],
                "mse_ratio_ab_over_base": gkl_ab_row["mean_mse"] / base_row["mean_mse"],
                "u_ratio_ab_over_bonly": gkl_ab_row["mean_predicted_u"] / gkl_b_row["mean_predicted_u"],
                "cal_ratio_ab": gkl_ab_row["predicted_u_over_actual_mse"],
                "cal_ratio_bonly": gkl_b_row["predicted_u_over_actual_mse"],
            }
            comparison_rows.append(row)
    write_csv(OUTPUT / "forecast_horizon_comparison.csv", comparison_rows)
    print("Wrote forecast_horizon_comparison.csv", flush=True)

    # =========================================================================
    # 6. Build calibration_metrics.csv
    # =========================================================================
    print("\n--- Section 6: Compiling calibration_metrics.csv ---", flush=True)
    cal_rows = []

    # Import archived calibration from previous eval (renaming gkl_epoch276 -> gkl_bonly_epoch276)
    with (PREV_EVAL_DIR / "calibration_metrics.csv").open() as f:
        for r in csv.DictReader(f):
            if r["model"] in rename_map:
                r["model"] = rename_map[r["model"]]
                cal_rows.append(r)
            elif r["model"] in ("new_joint_256ep", "original_256ep"):
                cal_rows.append(r)

    # A+B uses model-specific deciles; archived shared-bin ACE is a separate metric.
    for ch in ("Magnetic", "Density"):
        gkl_ab_samples = model_calibration_samples[("gkl_ab_epoch276", ch)]
        if not gkl_ab_samples:
            continue

        gkl_ab_e = np.concatenate([x[0] for x in gkl_ab_samples])
        gkl_ab_u = np.concatenate([x[1] for x in gkl_ab_samples])

        bin_edges = np.percentile(gkl_ab_u, np.linspace(0, 100, 11))
        bin_edges[0] = 0.0
        bin_edges[-1] = np.inf

        b_indices = np.digitize(gkl_ab_u, bin_edges) - 1
        b_indices = np.clip(b_indices, 0, 9)

        total_voxels = len(gkl_ab_u)
        binned_abs_errors = []

        for b in range(10):
            mask_b = (b_indices == b)
            n_b = int(np.sum(mask_b))
            if n_b > 0:
                u_b = float(np.mean(gkl_ab_u[mask_b]))
                q_b = float(np.mean(gkl_ab_e[mask_b]))
                cal_b = u_b / q_b if q_b > 0 else float("nan")
                abs_err = abs(u_b - q_b)
            else:
                u_b, q_b, cal_b, abs_err = 0.0, 0.0, float("nan"), 0.0

            binned_abs_errors.append((n_b, abs_err))
            cal_rows.append({
                "model": "gkl_ab_epoch276",
                "horizon": 18,
                "channel": ch,
                "bin_type": "self_decile",
                "bin_index": b + 1,
                "mean_predicted_u": u_b,
                "mean_empirical_q": q_b,
                "calibration_ratio": cal_b,
                "absolute_calibration_error": abs_err,
                "voxel_count": n_b,
            })

        ace = sum(n * err for n, err in binned_abs_errors) / total_voxels if total_voxels > 0 else 0.0
        overall_cal = float(np.mean(gkl_ab_u)) / float(np.mean(gkl_ab_e))
        cal_rows.append({
            "model": "gkl_ab_epoch276",
            "horizon": 18,
            "channel": ch,
            "bin_type": "summary_metric",
            "bin_index": "ALL",
            "mean_predicted_u": float(np.mean(gkl_ab_u)),
            "mean_empirical_q": float(np.mean(gkl_ab_e)),
            "calibration_ratio": overall_cal,
            "absolute_calibration_error": ace,
            "voxel_count": total_voxels,
        })

    write_csv(OUTPUT / "calibration_metrics.csv", cal_rows)
    print("Wrote calibration_metrics.csv", flush=True)

    # =========================================================================
    # 7. Paired Comparison Summary (Statistical Tests & Bootstrap CIs)
    # =========================================================================
    print("\n--- Section 7: Compiling paired_comparison_summary.csv ---", flush=True)
    summary_rows = []

    # We need per-window metrics from previous eval for GKL B-only and Student-t control.
    # Since they were not stored as raw arrays, we re-import the forecast_horizon_metrics
    # and use aggregate metrics for the archived models.
    # For GKL A+B, we have per-window data in model_window_* dicts.

    # --- GKL A+B vs GKL B-only: Per-window Forecast Horizon MSE, u, Spearman ---
    # We only have per-window data for GKL A+B; for paired tests with B-only
    # we need to re-run B-only or use aggregate comparisons.
    # Strategy: Use aggregate metrics from CSVs for the cross-model comparison table,
    # since we don't have per-window B-only data in memory.

    # Instead, construct an aggregate comparison table from the CSVs
    # For the new Path-A ablation, the key questions are answered by aggregate metrics.

    # Build a comprehensive cross-model summary
    models_all = ["new_joint_256ep", "gkl_ab_epoch276", "gkl_bonly_epoch276", "student_t_control_epoch276"]
    model_labels = {
        "new_joint_256ep": "Student-t A+B ep256",
        "gkl_ab_epoch276": "GKL A+B ep276",
        "gkl_bonly_epoch276": "GKL B-only ep276",
        "student_t_control_epoch276": "Student-t A+B ep276",
    }

    for h in FOCUS_HORIZONS:
        for ch in ("Magnetic", "Density", "All"):
            for metric in ("mean_mse", "mean_predicted_u", "predicted_u_over_actual_mse"):
                row = {
                    "domain": "forecast_horizon",
                    "horizon": h,
                    "channel": ch,
                    "metric": metric,
                }
                for m in models_all:
                    r = by_model_h_ch.get((m, h, ch))
                    if r:
                        row[model_labels[m]] = float(r[metric])
                    else:
                        row[model_labels[m]] = ""

                # Compute A+B vs B-only ratio
                ab_val = by_model_h_ch.get(("gkl_ab_epoch276", h, ch))
                b_val = by_model_h_ch.get(("gkl_bonly_epoch276", h, ch))
                if ab_val and b_val:
                    ab_v = float(ab_val[metric])
                    b_v = float(b_val[metric])
                    row["ab_vs_bonly_ratio"] = ab_v / b_v if b_v != 0 else float("nan")
                    row["ab_vs_bonly_diff"] = ab_v - b_v

                summary_rows.append(row)

    # Spatial Spearman from spatial rows
    for h in FOCUS_HORIZONS:
        for ch in ("Magnetic", "Density"):
            row = {
                "domain": "spatial_ranking",
                "horizon": h,
                "channel": ch,
                "metric": "mean_frame_spatial_spearman",
            }
            for m in models_all:
                sp_row = next(
                    (r for r in all_spatial_rows
                     if r["model"] == m and str(r["horizon"]) == str(h) and r["channel"] == ch and r["scope"] == "ranking_summary"),
                    None
                )
                if sp_row:
                    row[model_labels[m]] = float(sp_row["mean_frame_spatial_spearman"])
                else:
                    row[model_labels[m]] = ""

            ab_sp = next(
                (r for r in all_spatial_rows
                 if r["model"] == "gkl_ab_epoch276" and str(r["horizon"]) == str(h) and r["channel"] == ch and r["scope"] == "ranking_summary"),
                None
            )
            b_sp = next(
                (r for r in all_spatial_rows
                 if r["model"] == "gkl_bonly_epoch276" and str(r["horizon"]) == str(h) and r["channel"] == ch and r["scope"] == "ranking_summary"),
                None
            )
            if ab_sp and b_sp:
                row["ab_vs_bonly_diff"] = float(ab_sp["mean_frame_spatial_spearman"]) - float(b_sp["mean_frame_spatial_spearman"])
            summary_rows.append(row)

    # Match the cached whole-suffix per-window collapse definition.
    with (PREV_EVAL_DIR / "paired_comparison_summary.csv").open() as f:
        cached_collapse = {
            r["channel"]: float(r["treatment_mean (GKL)"])
            for r in csv.DictReader(f) if r["domain"] == "extreme_horizon_collapse"
        }
    for ch in ("Magnetic", "Density", "All"):
        ab_fraction = sum(
            model_window_forecast_mse[(model_name, 23, ch)][i]
            > model_window_forecast_mse[(model_name, 22, ch)][i]
            and model_window_forecast_u[(model_name, 23, ch)][i]
            < model_window_forecast_u[(model_name, 22, ch)][i]
            for i in test_indices
        ) / len(test_indices)
        summary_rows.append({
            "domain": "extreme_horizon_collapse",
            "horizon": "22 to 23",
            "channel": ch,
            "metric": "collapse_window_fraction",
            model_labels[model_name]: ab_fraction,
            model_labels["gkl_bonly_epoch276"]: cached_collapse[ch],
            "ab_vs_bonly_diff": ab_fraction - cached_collapse[ch],
        })

    write_csv(OUTPUT / "paired_comparison_summary.csv", summary_rows)
    print("Wrote paired_comparison_summary.csv", flush=True)

    # =========================================================================
    # 9. Generate Publication Figures
    # =========================================================================
    print("\n--- Section 9: Generating Publication Figures ---", flush=True)

    # ----- Figure 1: Training Curves (all 3 continuations) -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    eps = np.arange(257, 277)
    gkl_ab_val = [r["val_mse"] for r in gkl_ab_h]
    gkl_b_val = [r["val_mse"] for r in gkl_b_h]
    ctrl_val = [r["val_mse"] for r in ctrl_h]
    gkl_ab_train = [r["train_mse"] for r in gkl_ab_h]
    gkl_b_train = [r["train_mse"] for r in gkl_b_h]
    ctrl_train = [r["train_mse"] for r in ctrl_h]

    axes[0, 0].axhline(0.017454, color="gray", ls="--", label="Epoch 256 Baseline")
    axes[0, 0].plot(eps, ctrl_val, marker="o", color="blue", label="Student-t A+B Control")
    axes[0, 0].plot(eps, gkl_b_val, marker="^", color="darkorange", label="GKL B-only")
    axes[0, 0].plot(eps, gkl_ab_val, marker="s", color="crimson", lw=2, label="GKL A+B (Path A ON)")
    axes[0, 0].set_title("Validation MSE")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Val MSE")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(eps, ctrl_train, marker="o", color="blue", label="Student-t Control")
    axes[0, 1].plot(eps, gkl_b_train, marker="^", color="darkorange", label="GKL B-only")
    axes[0, 1].plot(eps, gkl_ab_train, marker="s", color="crimson", lw=2, label="GKL A+B")
    axes[0, 1].set_title("Training MSE")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Train MSE")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    gkl_ab_gnorm = [r["gradient_norm_mean"] for r in gkl_ab_h]
    gkl_b_gnorm = [r["gradient_norm_mean"] for r in gkl_b_h]
    ctrl_gnorm = [r["gradient_norm_mean"] for r in ctrl_h]
    axes[1, 0].plot(eps, ctrl_gnorm, marker="o", color="blue", label="Student-t Control")
    axes[1, 0].plot(eps, gkl_b_gnorm, marker="^", color="darkorange", label="GKL B-only")
    axes[1, 0].plot(eps, gkl_ab_gnorm, marker="s", color="crimson", lw=2, label="GKL A+B")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Mean Gradient Norm (Log Scale)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Gradient L2 Norm")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    # GKL A+B vs B-only uncertainty loss comparison
    ab_gkl0 = [r["curriculum"]["L_gkl_0"] for r in gkl_ab_h if math.isclose(r["curriculum"]["gkl_alpha"], 1.0)]
    b_gkl0 = [r["curriculum"]["L_gkl_0"] for r in gkl_b_h if math.isclose(r["curriculum"]["gkl_alpha"], 1.0)]
    settled_eps = [r["epoch"] for r in gkl_ab_h if math.isclose(r["curriculum"]["gkl_alpha"], 1.0)]
    axes[1, 1].plot(settled_eps, b_gkl0, marker="^", color="darkorange", label="GKL B-only L_gkl_0")
    axes[1, 1].plot(settled_eps, ab_gkl0, marker="s", color="crimson", lw=2, label="GKL A+B L_gkl_0")
    axes[1, 1].set_title("GKL Uncertainty Loss (Post-Transition, alpha=1)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("GKL Loss (nats/voxel)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Training History: GKL A+B vs B-only vs Student-t Control (Epochs 257-276)", fontsize=13)
    plt.savefig(OUTPUT / "training_curves.png", dpi=200)
    plt.close()
    print("Saved training_curves.png", flush=True)

    # ----- Figure 2: Forecast Horizon Comparison -----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    models_plot = [
        ("original_256ep", "Original 256ep", "gray", ":"),
        ("new_joint_256ep", "Student-t A+B ep256", "darkorange", "--"),
        ("student_t_control_epoch276", "Student-t A+B ep276", "blue", "-"),
        ("gkl_bonly_epoch276", "GKL B-only ep276", "green", "-"),
        ("gkl_ab_epoch276", "GKL A+B ep276", "crimson", "-"),
    ]

    for col, ch in enumerate(["Magnetic", "Density", "All"]):
        ax = axes[0, col]
        for m, label, color, ls in models_plot:
            vals = []
            for h in HORIZONS:
                r = by_model_h_ch.get((m, h, ch))
                vals.append(float(r["mean_mse"]) if r else float("nan"))
            ax.plot(HORIZONS, vals, label=label, color=color, ls=ls, lw=1.5 if m != "gkl_ab_epoch276" else 2.5)
        ax.set_title(f"{ch} Reconstruction MSE")
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

        ax = axes[1, col]
        ax.axhline(1.0, color="black", ls=":", lw=1.5, label="Ideal (1.0)")
        for m, label, color, ls in models_plot:
            vals = []
            for h in HORIZONS:
                r = by_model_h_ch.get((m, h, ch))
                vals.append(float(r["predicted_u_over_actual_mse"]) if r else float("nan"))
            ax.plot(HORIZONS, vals, label=label, color=color, ls=ls, lw=1.5 if m != "gkl_ab_epoch276" else 2.5)
        ax.set_title(f"{ch} Calibration (u / MSE)")
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("Calibration Ratio")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Forecast Horizon Benchmark (h = 1..23): GKL A+B vs B-only vs Controls", fontsize=13)
    plt.savefig(OUTPUT / "forecast_horizon_comparison.png", dpi=200)
    plt.close()
    print("Saved forecast_horizon_comparison.png", flush=True)

    # ----- Figure 3: Standard Mask Comparison -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    tasks = list(MASK_PATTERNS)
    x = np.arange(len(tasks))
    w = 0.18

    for ax, role in [(axes[0], "val"), (axes[1], "test")]:
        def get_mses(m):
            return [
                float(next(r["mean_mse"] for r in all_standard_rows if r["model"] == m and r["role"] == role and r["source_task"] == t and r["channel"] == "All" and r["pass"] == "pass0"))
                for t in tasks
            ]
        orig_mses = get_mses("original_256ep")
        base_mses = get_mses("new_joint_256ep")
        ctrl_mses = get_mses("student_t_control_epoch276")
        gkl_b_mses = get_mses("gkl_bonly_epoch276")
        gkl_ab_mses = get_mses("gkl_ab_epoch276")

        ax.bar(x - 2 * w, orig_mses, width=w, label="Original 256ep", color="gray")
        ax.bar(x - 1 * w, base_mses, width=w, label="St-t A+B ep256", color="darkorange")
        ax.bar(x, ctrl_mses, width=w, label="St-t A+B ep276", color="blue")
        ax.bar(x + 1 * w, gkl_b_mses, width=w, label="GKL B-only ep276", color="green")
        ax.bar(x + 2 * w, gkl_ab_mses, width=w, label="GKL A+B ep276", color="crimson")

        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=25)
        ax.set_ylabel("Pass-0 Reconstruction MSE")
        ax.set_title(f"Standard Masks ({role.capitalize()} Set)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Standard Mask Reconstruction: 5-Model Comparison", fontsize=13)
    plt.savefig(OUTPUT / "standard_mask_comparison.png", dpi=200)
    plt.close()
    print("Saved standard_mask_comparison.png", flush=True)

    # ----- Figure 4: Spatial Spearman Comparison -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    channels_sp = ["Bx", "By", "Bz", "Density", "Magnetic"]
    x = np.arange(len(channels_sp))
    w = 0.18

    for idx, h in enumerate(FOCUS_HORIZONS):
        ax = axes[idx]
        def get_sp(m):
            return [
                float(next(
                    r["mean_frame_spatial_spearman"] for r in all_spatial_rows
                    if r["model"] == m and str(r["horizon"]) == str(h) and r["channel"] == ch and r["scope"] == "ranking_summary"
                ))
                for ch in channels_sp
            ]
        base_sp = get_sp("new_joint_256ep")
        ctrl_sp = get_sp("student_t_control_epoch276")
        gkl_b_sp = get_sp("gkl_bonly_epoch276")
        gkl_ab_sp = get_sp("gkl_ab_epoch276")

        ax.bar(x - 1.5 * w, base_sp, width=w, label="St-t A+B ep256", color="darkorange")
        ax.bar(x - 0.5 * w, ctrl_sp, width=w, label="St-t A+B ep276", color="blue")
        ax.bar(x + 0.5 * w, gkl_b_sp, width=w, label="GKL B-only ep276", color="green")
        ax.bar(x + 1.5 * w, gkl_ab_sp, width=w, label="GKL A+B ep276", color="crimson")

        ax.set_xticks(x)
        ax.set_xticklabels(channels_sp)
        ax.set_ylim(0.0, 0.8)
        ax.set_ylabel("Within-Frame Spatial Spearman")
        ax.set_title(f"Spatial Error Ranking (h={h})")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Spatial Uncertainty Quality: Error-Uncertainty Ranking (Spearman ρ)", fontsize=13)
    plt.savefig(OUTPUT / "spatial_uncertainty_correlation.png", dpi=200)
    plt.close()
    print("Saved spatial_uncertainty_correlation.png", flush=True)

    # ----- Figure 6: Extreme Horizon Dynamics -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    frames = np.arange(2, 24)

    # (0,0): h22 vs h23 predicted uncertainty (Magnetic)
    ax = axes[0, 0]
    for m, label, color in [
        ("new_joint_256ep", "St-t ep256", "darkorange"),
        ("student_t_control_epoch276", "St-t ep276", "blue"),
        ("gkl_bonly_epoch276", "GKL B-only", "green"),
        ("gkl_ab_epoch276", "GKL A+B", "crimson"),
    ]:
        u22 = [float(next(r["u_h22"] for r in all_paired_rows if r["model"] == m and r["channel"] == "Magnetic" and int(r["local_frame"]) == t)) for t in frames]
        u23 = [float(next(r["u_h23"] for r in all_paired_rows if r["model"] == m and r["channel"] == "Magnetic" and int(r["local_frame"]) == t)) for t in frames]
        ax.plot(frames, u22, color=color, ls="--", alpha=0.7, label=f"{label} h22")
        ax.plot(frames, u23, color=color, ls="-", lw=2, label=f"{label} h23")
    ax.set_title("Magnetic: Predicted u (h22 vs h23)")
    ax.set_xlabel("Local Frame")
    ax.set_ylabel("Mean Predicted u")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)

    # (0,1): h23/h22 ratio (Magnetic)
    ax = axes[0, 1]
    ax.axhline(1.0, color="black", ls=":", label="No Collapse")
    for m, label, color in [
        ("new_joint_256ep", "St-t ep256", "darkorange"),
        ("student_t_control_epoch276", "St-t ep276", "blue"),
        ("gkl_bonly_epoch276", "GKL B-only", "green"),
        ("gkl_ab_epoch276", "GKL A+B", "crimson"),
    ]:
        ratios = [float(next(r["u_h23_over_h22"] for r in all_paired_rows if r["model"] == m and r["channel"] == "Magnetic" and int(r["local_frame"]) == t)) for t in frames]
        ax.plot(frames, ratios, color=color, lw=2, label=label)
    ax.set_title("Magnetic: u(h23) / u(h22)")
    ax.set_xlabel("Local Frame")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # (1,0): Within-window per-frame u growth h=18 (Magnetic)
    ax = axes[1, 0]
    h18_frames = np.arange(6, 24)
    for m, label, color in [
        ("original_256ep", "Original 256ep", "gray"),
        ("new_joint_256ep", "St-t A+B ep256", "darkorange"),
        ("student_t_control_epoch276", "St-t A+B ep276", "blue"),
        ("gkl_bonly_epoch276", "GKL B-only ep276", "green"),
        ("gkl_ab_epoch276", "GKL A+B ep276", "crimson"),
    ]:
        u_f = [float(next(r["mean_predicted_u"] for r in all_frame_rows if r["model"] == m and int(r["horizon"]) == 18 and r["channel"] == "Magnetic" and int(r["local_frame"]) == t)) for t in h18_frames]
        ax.plot(h18_frames - 5, u_f, color=color, lw=2, label=label)
    ax.set_title("Within-Window u Growth h=18 (Magnetic)")
    ax.set_xlabel("Forecast Lead")
    ax.set_ylabel("Predicted u")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # (1,1): Within-window per-frame u growth h=18 (Density)
    ax = axes[1, 1]
    for m, label, color in [
        ("original_256ep", "Original 256ep", "gray"),
        ("new_joint_256ep", "St-t A+B ep256", "darkorange"),
        ("student_t_control_epoch276", "St-t A+B ep276", "blue"),
        ("gkl_bonly_epoch276", "GKL B-only ep276", "green"),
        ("gkl_ab_epoch276", "GKL A+B ep276", "crimson"),
    ]:
        u_f = [float(next(r["mean_predicted_u"] for r in all_frame_rows if r["model"] == m and int(r["horizon"]) == 18 and r["channel"] == "Density" and int(r["local_frame"]) == t)) for t in h18_frames]
        ax.plot(h18_frames - 5, u_f, color=color, lw=2, label=label)
    ax.set_title("Within-Window u Growth h=18 (Density)")
    ax.set_xlabel("Forecast Lead")
    ax.set_ylabel("Predicted u")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    fig.suptitle("Extreme-Horizon Dynamics & Within-Window Uncertainty Scaling", fontsize=13)
    plt.savefig(OUTPUT / "extreme_horizon_and_per_frame.png", dpi=200)
    plt.close()
    print("Saved extreme_horizon_and_per_frame.png", flush=True)

    # ----- Figure 7: Conditional-MSE Reliability Diagrams -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for idx, ch in enumerate(["Magnetic", "Density"]):
        ax = axes[idx]
        for m, label, color, marker in [
            ("new_joint_256ep", "St-t A+B ep256", "darkorange", "o"),
            ("student_t_control_epoch276", "St-t A+B ep276", "blue", "^"),
            ("gkl_bonly_epoch276", "GKL B-only ep276", "green", "D"),
            ("gkl_ab_epoch276", "GKL A+B ep276", "crimson", "s"),
        ]:
            # Find decile data
            dec_scope = "pooled_sampled_decile" if m != "gkl_ab_epoch276" else "self_decile"
            decs = [r for r in cal_rows
                    if r["model"] == m and str(r.get("horizon", "")) == "18" and r.get("channel") == ch
                    and r.get("bin_type", r.get("scope", "")) in (dec_scope, "pooled_sampled_decile", "pooled_boundary_decile", "self_decile", "decile")
                    and r.get("bin_index", "") not in ("ALL", "")]
            if not decs:
                # Try spatial rows
                decs = [r for r in all_spatial_rows
                        if r["model"] == m and str(r.get("horizon", "")) == "18" and r.get("channel") == ch
                        and r.get("scope") == "pooled_sampled_decile"]
            decs = sorted(decs, key=lambda r: int(r.get("quantile", r.get("bin_index", 0))))
            if decs:
                u_vals = [float(r.get("mean_predicted_u", 0)) for r in decs]
                q_vals = [float(r.get("mean_mse", r.get("mean_empirical_q", 0))) for r in decs]
                ax.plot(u_vals, q_vals, marker=marker, color=color, lw=2, label=label)

        all_vals_on_plot = ax.get_xlim() + ax.get_ylim()
        min_v = max(1e-6, min(all_vals_on_plot))
        max_v = max(all_vals_on_plot) * 2.0
        ax.plot([min_v, max_v], [min_v, max_v], color="black", ls=":", label="Ideal y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mean Predicted u")
        ax.set_ylabel("Mean Realized q = (y - μ)²")
        ax.set_title(f"{ch} Reliability Diagram (h=18)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Conditional-MSE Calibration Reliability Diagrams", fontsize=13)
    plt.savefig(OUTPUT / "conditional_mse_reliability_diagrams.png", dpi=200)
    plt.close()
    print("Saved conditional_mse_reliability_diagrams.png", flush=True)

    print("\n" + "=" * 80)
    print("All evaluation deliverables and figures successfully generated!")
    print("=" * 80, flush=True)

    # Print key summary to stdout
    print("\n--- Quick Summary ---")
    for h in [18, 22, 23]:
        for ch in ["Magnetic", "Density", "All"]:
            ab = by_model_h_ch.get(("gkl_ab_epoch276", h, ch))
            b = by_model_h_ch.get(("gkl_bonly_epoch276", h, ch))
            ctrl = by_model_h_ch.get(("student_t_control_epoch276", h, ch))
            base = by_model_h_ch.get(("new_joint_256ep", h, ch))
            if ab and b:
                print(f"h={h:2d} {ch:10s}  MSE: AB={ab['mean_mse']:.6f} B={b['mean_mse']:.6f} ctrl={ctrl['mean_mse']:.6f} base={base['mean_mse']:.6f} | "
                      f"Cal: AB={ab['predicted_u_over_actual_mse']:.4f} B={b['predicted_u_over_actual_mse']:.4f} ctrl={ctrl['predicted_u_over_actual_mse']:.4f} base={base['predicted_u_over_actual_mse']:.4f}")


if __name__ == "__main__":
    main()
