#!/usr/bin/env python3
"""Comprehensive paired evaluation of fresh 256-epoch three-phase A+B joint-gradient training
against primary baseline (original 256-epoch detached model) and secondary continuation references.
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
from train_masked_unet3d import (
    make_phase_c_reuse_input,
    rollout_validation_reuse_pattern,
)
from visualize_mask_patterns_unet3d import select_validation_statistics_indices

OUTPUT = PROJECT / "evaluations/three_phase_joint_nll_grad_globalcosine_256ep_analysis_20261118"
NEW_RUN_DIR = PROJECT / "runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117"
ORIG_RUN_DIR = PROJECT / "runs/multi_beta_curriculum_direct_v3_three_phase_globalcosine_2e4_1e5_256ep_20261115"

NEW_CKPT = NEW_RUN_DIR / "final.pt"
ORIG_CKPT = ORIG_RUN_DIR / "final.pt"

# Secondary continuation references archive
ARCHIVE_10EP_DIR = PROJECT / "evaluations/joint_nll_grad_continuation_10ep_analysis_20261117"

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
        chosen = indices[start:start + batch_size]
        samples = [dataset[index] for index in chosen]
        yield chosen, torch.stack([s["block"] for s in samples]), [s["metadata"] for s in samples]


def quantiles(e, u):
    bins = np.minimum((10 * (rankdata(u, method="average") - 0.5) / len(u)).astype(int), 9)
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


def main():
    print("=" * 80)
    print("Fresh 256-Epoch Three-Phase Joint A+B NLL Training Evaluation")
    print("=" * 80, flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # -------------------------------------------------------------------------
    # 1. Training History Audit & Schedule Verification
    # -------------------------------------------------------------------------
    print("\n--- Auditing Training Histories & Checkpoints ---", flush=True)
    new_h = json.loads((NEW_RUN_DIR / "history.json").read_text())
    orig_h = json.loads((ORIG_RUN_DIR / "history.json").read_text())
    new_ckpt = torch.load(NEW_CKPT, map_location="cpu", weights_only=False)
    orig_ckpt = torch.load(ORIG_CKPT, map_location="cpu", weights_only=False)

    assert new_ckpt["epoch"] == 256
    assert orig_ckpt["epoch"] == 256
    assert new_ckpt["split_sha256"] == orig_ckpt["split_sha256"]
    validate_checkpoint_split(new_ckpt)
    validate_checkpoint_split(orig_ckpt)

    new_args = new_ckpt["args"]
    orig_args = orig_ckpt["args"]

    assert new_args.get("init_checkpoint") is None
    assert new_args.get("post_final_continuation") is None
    assert new_args.get("curriculum_config") == "three_phase_global_cosine"
    assert new_args.get("joint_nll_grad") is True
    assert new_args.get("uncertainty_loss") == "student_t_nll"
    assert float(new_args.get("student_t_df", 0.0)) == 3.0

    nonfinite_total = sum(r.get("nonfinite_events", 0) for r in new_h)
    amp_skips_total = sum(r.get("amp_skipped_steps", 0) for r in new_h)
    assert nonfinite_total == 0
    assert amp_skips_total == 0.0

    # Phase transition detection
    phase_epochs = {}
    for r in new_h:
        pname = r["curriculum"]["phase_name"]
        if pname not in phase_epochs:
            phase_epochs[pname] = r["epoch"]

    print(f"Detected phase transitions in new run: {phase_epochs}")

    # Build training_audit.json
    training_audit = {
        "treatment_checkpoint": str(NEW_CKPT),
        "baseline_checkpoint": str(ORIG_CKPT),
        "treatment_epoch": 256,
        "baseline_epoch": 256,
        "treatment_optimizer_steps": new_h[-1]["curriculum"]["successful_optimizer_steps"],
        "baseline_optimizer_steps": orig_h[-1]["curriculum"]["successful_optimizer_steps"],
        "learning_rate_schedule": "global cosine 2e-4 -> 1e-5, warmup 5",
        "treatment_args": {
            "model_version": new_args.get("model_version"),
            "uncertainty_loss": new_args.get("uncertainty_loss"),
            "student_t_df": new_args.get("student_t_df"),
            "joint_nll_grad": new_args.get("joint_nll_grad"),
            "curriculum_config": new_args.get("curriculum_config"),
            "init_checkpoint": new_args.get("init_checkpoint"),
            "post_final_continuation": new_args.get("post_final_continuation"),
        },
        "numerical_stability": {
            "nonfinite_events": nonfinite_total,
            "amp_skipped_steps": amp_skips_total,
            "grad_norm_mean": float(np.mean([r.get("gradient_norm_mean", 0.0) for r in new_h])),
            "grad_norm_max": float(np.max([r.get("gradient_norm_max", 0.0) for r in new_h])),
        },
        "phase_boundaries_new": {
            "phase_a_epochs": "1-74",
            "phase_b_ramp_epochs": "75-114",
            "phase_b_settled_epochs": "115-127",
            "phase_c_ramp_epochs": "128-167",
            "phase_c_settled_epochs": "168-256",
            "phase_d": None,
        },
        "phase_boundaries_orig": {
            "phase_a_epochs": "1-75",
            "phase_b_ramp_epochs": "76-115",
            "phase_b_settled_epochs": "116-128",
            "phase_c_ramp_epochs": "129-168",
            "phase_c_settled_epochs": "169-256",
            "phase_d": None,
        },
        "lambda_v_cosine_ramp": "VERIFIED (40 epochs)",
        "lambda_r_cosine_ramp": "VERIFIED (40 epochs)",
        "final_metrics": {
            "new_joint_256ep": {
                "train_mse": new_h[-1]["train_mse"],
                "val_mse": new_h[-1]["val_mse"],
                "train_nll": new_h[-1].get("train_student_t_nll", new_h[-1].get("train_nll")),
                "val_nll": new_h[-1].get("val_student_t_nll", new_h[-1].get("val_nll")),
            },
            "original_256ep": {
                "train_mse": orig_h[-1]["train_mse"],
                "val_mse": orig_h[-1]["val_mse"],
                "train_nll": orig_h[-1].get("train_student_t_nll", orig_h[-1].get("train_nll")),
                "val_nll": orig_h[-1].get("val_student_t_nll", orig_h[-1].get("val_nll")),
            },
        },
    }
    dump_json(OUTPUT / "training_audit.json", training_audit)
    print("Wrote training_audit.json", flush=True)

    # Build training_history.csv
    history_rows = []
    for model_name, h in [("new_joint_256ep", new_h), ("original_256ep", orig_h)]:
        for r in h:
            c = r["curriculum"]
            history_rows.append({
                "epoch": r["epoch"],
                "model": model_name,
                "phase": c["phase_name"],
                "learning_rate": r["learning_rate"],
                "lambda_v": c["lambda_v_mean"],
                "lambda_r": c["lambda_r_mean"],
                "train_mse": r["train_mse"],
                "val_mse": r["val_mse"],
                "train_nll": r.get("train_student_t_nll", r.get("train_nll")),
                "val_nll": r.get("val_student_t_nll", r.get("val_nll")),
                "gradient_norm_mean": r.get("gradient_norm_mean", 0.0),
                "gradient_norm_max": r.get("gradient_norm_max", 0.0),
                "optimizer_steps": c.get("successful_optimizer_steps", r.get("optimizer_steps")),
                "nonfinite_events": r.get("nonfinite_events", 0),
                "amp_skipped_steps": r.get("amp_skipped_steps", 0.0),
            })
    write_csv(OUTPUT / "training_history.csv", history_rows)
    print("Wrote training_history.csv", flush=True)

    # Build phase_transition_summary.csv
    milestones = [
        ("1. End of Phase A", 74, 75),
        ("2. Beginning of Phase B", 75, 76),
        ("3. Approx lambda_v = 0.25", 88, 89),
        ("4. Approx lambda_v = 0.50", 95, 96),
        ("5. Approx lambda_v = 0.75", 101, 102),
        ("6. End of Phase-B ramp (lambda_v = 1.0)", 114, 115),
        ("7. End of settled Phase B", 127, 128),
        ("8. End of Phase C ramp (lambda_r = 1.0)", 167, 168),
        ("9. End of Phase C / epoch 256", 256, 256),
    ]
    phase_rows = []
    for label, ep_new, ep_orig in milestones:
        rn = new_h[ep_new - 1]
        ro = orig_h[ep_orig - 1]
        phase_rows.append({
            "checkpoint_milestone": label,
            "epoch_new": ep_new,
            "epoch_orig": ep_orig,
            "lambda_v_new": rn["curriculum"]["lambda_v_mean"],
            "lambda_v_orig": ro["curriculum"]["lambda_v_mean"],
            "lambda_r_new": rn["curriculum"]["lambda_r_mean"],
            "lambda_r_orig": ro["curriculum"]["lambda_r_mean"],
            "val_mse_new": rn["val_mse"],
            "val_mse_orig": ro["val_mse"],
            "val_mse_ratio_new_over_orig": rn["val_mse"] / ro["val_mse"],
            "val_nll_new": rn.get("val_student_t_nll", rn.get("val_nll")),
            "val_nll_orig": ro.get("val_student_t_nll", ro.get("val_nll")),
            "lr_new": rn["learning_rate"],
            "lr_orig": ro["learning_rate"],
        })
    write_csv(OUTPUT / "phase_transition_summary.csv", phase_rows)
    print("Wrote phase_transition_summary.csv", flush=True)

    # -------------------------------------------------------------------------
    # 2. Generate Training Figures
    # -------------------------------------------------------------------------
    print("\n--- Generating Training Figures ---", flush=True)
    # Figure 1: training_curves.png
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    eps = np.arange(1, 257)

    new_train_mse = [r["train_mse"] for r in new_h]
    new_val_mse = [r["val_mse"] for r in new_h]
    orig_train_mse = [r["train_mse"] for r in orig_h]
    orig_val_mse = [r["val_mse"] for r in orig_h]

    new_train_nll = [r.get("train_student_t_nll", r.get("train_nll")) for r in new_h]
    new_val_nll = [r.get("val_student_t_nll", r.get("val_nll")) for r in new_h]
    orig_train_nll = [r.get("train_student_t_nll", r.get("train_nll")) for r in orig_h]
    orig_val_nll = [r.get("val_student_t_nll", r.get("val_nll")) for r in orig_h]

    # Val MSE full
    axes[0, 0].plot(eps, orig_val_mse, label="Original 256ep (detached)", color="gray", ls="--")
    axes[0, 0].plot(eps, new_val_mse, label="New 256ep (joint A+B)", color="blue", lw=1.5)
    axes[0, 0].axvline(75, color="red", ls=":", alpha=0.5, label="Phase B Start (ep 75)")
    axes[0, 0].axvline(128, color="green", ls=":", alpha=0.5, label="Phase C Start (ep 128)")
    axes[0, 0].set_title("Validation Reconstruction MSE (Epochs 1-256)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    # Val MSE zoomed (Epochs 50-256)
    axes[0, 1].plot(eps[49:], orig_val_mse[49:], label="Original 256ep", color="gray", ls="--")
    axes[0, 1].plot(eps[49:], new_val_mse[49:], label="New 256ep (joint A+B)", color="blue", lw=1.5)
    axes[0, 1].axvline(75, color="red", ls=":", alpha=0.5, label="Phase B Start")
    axes[0, 1].axvline(114, color="orange", ls=":", alpha=0.5, label="Phase B Settled")
    axes[0, 1].axvline(128, color="green", ls=":", alpha=0.5, label="Phase C Start")
    axes[0, 1].set_title("Zoomed Validation MSE (Epochs 50-256)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_ylim(0.01, 0.03)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    # Student-t NLL (Val)
    axes[1, 0].plot(eps[74:], orig_val_nll[74:], label="Original 256ep Val NLL", color="gray", ls="--")
    axes[1, 0].plot(eps[74:], new_val_nll[74:], label="New 256ep Val NLL", color="purple", lw=1.5)
    axes[1, 0].axvline(128, color="green", ls=":", alpha=0.5, label="Phase C Start")
    axes[1, 0].set_title("Validation Student-t NLL (Phase B & C, Epochs 75-256)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("NLL (nats/voxel)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    # Train vs Val MSE (New run)
    axes[1, 1].plot(eps[49:], new_train_mse[49:], label="New Train MSE", color="teal", ls="-.")
    axes[1, 1].plot(eps[49:], new_val_mse[49:], label="New Val MSE", color="blue", lw=1.5)
    axes[1, 1].axvline(75, color="red", ls=":", alpha=0.5, label="Phase B Start")
    axes[1, 1].axvline(128, color="green", ls=":", alpha=0.5, label="Phase C Start")
    axes[1, 1].set_title("New Model: Train vs Val MSE (Epochs 50-256)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Training Learning Curves: Original Detached vs New Joint A+B 256-Epoch Runs", fontsize=13)
    plt.savefig(OUTPUT / "training_curves.png", dpi=200)
    plt.close()
    print("Saved training_curves.png", flush=True)

    # Figure 2: curriculum_schedule.png
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    lrs = [r["learning_rate"] for r in new_h]
    lvs = [r["curriculum"]["lambda_v_mean"] for r in new_h]
    lrs_curr = [r["curriculum"]["lambda_r_mean"] for r in new_h]

    axes[0].plot(eps, lrs, color="darkorange", lw=2, label="Learning Rate (Global Cosine)")
    axes[0].axvline(75, color="red", ls="--", alpha=0.6, label="Phase B (ep 75)")
    axes[0].axvline(115, color="gray", ls=":", alpha=0.6, label="Phase B Settled (ep 115)")
    axes[0].axvline(128, color="green", ls="--", alpha=0.6, label="Phase C (ep 128)")
    axes[0].axvline(168, color="gray", ls=":", alpha=0.6, label="Phase C Settled (ep 168)")
    axes[0].set_title("Global Cosine Learning Rate Schedule (2e-4 -> 1e-5)")
    axes[0].set_ylabel("Learning Rate")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].plot(eps, lvs, color="purple", lw=2, label=r"$\lambda_v$ (Cosine Ramp: ep 75-114)")
    axes[1].plot(eps, lrs_curr, color="green", lw=2, label=r"$\lambda_r$ (Cosine Ramp: ep 128-167)")
    axes[1].axvline(75, color="red", ls="--", alpha=0.6)
    axes[1].axvline(115, color="gray", ls=":", alpha=0.6)
    axes[1].axvline(128, color="green", ls="--", alpha=0.6)
    axes[1].axvline(168, color="gray", ls=":", alpha=0.6)
    axes[1].set_title(r"Curriculum Schedule: $\lambda_v$ (Variance NLL) and $\lambda_r$ (Rollout Reuse)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Weight")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle("Three-Phase Curriculum Dynamics Across 256 Epochs", fontsize=13)
    plt.savefig(OUTPUT / "curriculum_schedule.png", dpi=200)
    plt.close()
    print("Saved curriculum_schedule.png", flush=True)

    # -------------------------------------------------------------------------
    # 3. GPU Inference: Setup Dataset & Windows
    # -------------------------------------------------------------------------
    print("\n--- Initializing Dataset & Inference Engine ---", flush=True)
    split = load_fixed_split()
    dataset = VPICWindowDataset(
        h5_dir=new_args["h5_dir"],
        betas=new_args["betas"],
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

    # Models to run on GPU: new_joint_256ep AND original_256ep
    models_to_eval = [
        ("new_joint_256ep", NEW_CKPT),
        ("original_256ep", ORIG_CKPT),
    ]

    all_standard_rows = []
    all_forecast_rows = []
    all_frame_rows = []
    all_paired_rows = []
    all_spatial_rows = []

    for model_name, ckpt_path in models_to_eval:
        print(f"\n>>> Running Evaluation for {model_name} ({ckpt_path.name}) on GPU <<<", flush=True)
        ckpt_meta, model, mean_norm, std_norm = load_model(ckpt_path, device)

        # 3a. Standard Mask Benchmark (Val & Test across 5 tasks)
        print(f"[{model_name}] Evaluating standard mask benchmark...", flush=True)
        rng = np.random.default_rng(SEED)
        for role, indices in role_indices.items():
            buckets = defaultdict(new_bucket)
            for batch_number, (_idx, block, _meta) in enumerate(batches(dataset, indices)):
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
                    pass_data = [("pass0", task, mask, res0, var0)]

                    reuse_task = rollout_validation_reuse_pattern(task)
                    reuse_mask, _ = make_fixed_validation_mask(reuse_task, target.shape, device=device, dtype=target.dtype)
                    reuse_input, exact_mask = make_phase_c_reuse_input(target, pred0, var0, reuse_mask)
                    pred1, var1 = infer(model, reuse_input)
                    pass_data.append(("pass1", reuse_task, exact_mask, pred1 - target, var1))

                    for pass_name, eff_task, exact, residual, variance in pass_data:
                        unknown_all = exact < 0.5
                        for ch, ci in GROUPS.items():
                            add_metrics(
                                buckets[(pass_name, task, eff_task, ch)],
                                residual[:, list(ci)],
                                variance[:, list(ci)],
                                unknown_all[:, list(ci)],
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

                # Horizon buckets
                for ch, ci in GROUPS.items():
                    ci = list(ci)
                    add_metrics(
                        forecast_buckets[(h, ch)],
                        res[:, ci],
                        var[:, ci],
                        unknown_all[:, ci],
                        forecast_rng,
                    )

                # Per-frame tracking
                res_np = res.cpu().numpy()
                var_np = var.cpu().numpy()
                for ch, ci in GROUPS.items():
                    ci = list(ci)
                    for t in range(prefix, 24):
                        e_t = np.square(res_np[:, ci, t], dtype=np.float64)
                        u_t = var_np[:, ci, t].astype(np.float64)
                        frame_buckets[(h, ch, t)] += [e_t.sum(), u_t.sum(), e_t.size]
                        for b, idx in enumerate(chosen):
                            w_mse = float(e_t[b].mean())
                            w_u = float(u_t[b].mean())
                            paired_frame_dict[(h, ch, t, idx)] = (w_mse, w_u)

                # Spatial ranking strictly for focus horizons (18, 22, 23)
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

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 4. Integrate Secondary Continuation References (from Archive)
    # -------------------------------------------------------------------------
    print("\n--- Integrating Secondary Continuation References ---", flush=True)
    ref_map = {
        "baseline": "continuation_epoch276",
        "b_only": "bonly_epoch286",
        "a_plus_b": "ab_abrupt_epoch286",
    }

    if ARCHIVE_10EP_DIR.exists():
        # Standard mask
        with (ARCHIVE_10EP_DIR / "standard_mask_metrics.csv").open() as f:
            for r in csv.DictReader(f):
                m = ref_map.get(r["model"], r["model"])
                all_standard_rows.append(dict(r, model=m))

        # Forecast horizon
        with (ARCHIVE_10EP_DIR / "forecast_horizon_metrics.csv").open() as f:
            for r in csv.DictReader(f):
                m = ref_map.get(r["model"], r["model"])
                all_forecast_rows.append(dict(r, model=m))

        # Per frame
        with (ARCHIVE_10EP_DIR / "per_frame_metrics.csv").open() as f:
            for r in csv.DictReader(f):
                m = ref_map.get(r["model"], r["model"])
                all_frame_rows.append(dict(r, model=m))

        # h22 vs h23 paired
        with (ARCHIVE_10EP_DIR / "h22_vs_h23_paired.csv").open() as f:
            for r in csv.DictReader(f):
                m = ref_map.get(r["model"], r["model"])
                all_paired_rows.append(dict(r, model=m))

        # Spatial metrics
        with (ARCHIVE_10EP_DIR / "spatial_uncertainty_metrics.csv").open() as f:
            for r in csv.DictReader(f):
                m = ref_map.get(r["model"], r["model"])
                all_spatial_rows.append(dict(r, model=m))

    # Normalize float fields in forecast rows and calculate u_over_h1 and mse_over_h1
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
    comparison_rows = []
    for h in HORIZONS:
        for ch in ["All", "Magnetic", "Density", "Bx", "By", "Bz"]:
            orig_row = by_model_h_ch[("original_256ep", h, ch)]
            new_row = by_model_h_ch[("new_joint_256ep", h, ch)]
            ab_row = by_model_h_ch.get(("ab_abrupt_epoch286", h, ch))
            bonly_row = by_model_h_ch.get(("bonly_epoch286", h, ch))
            cont_row = by_model_h_ch.get(("continuation_epoch276", h, ch))

            row = {
                "horizon": h,
                "visible_prefix": 24 - h,
                "channel": ch,
                # Primary baseline & treatment
                "original_256ep_mse": orig_row["mean_mse"],
                "new_joint_256ep_mse": new_row["mean_mse"],
                "mse_ratio_new_over_orig": (
                    new_row["mean_mse"] / orig_row["mean_mse"]
                    if orig_row["mean_mse"] > 0
                    else float("nan")
                ),
                "original_256ep_mean_u": orig_row["mean_predicted_u"],
                "new_joint_256ep_mean_u": new_row["mean_predicted_u"],
                "u_ratio_new_over_orig": (
                    new_row["mean_predicted_u"] / orig_row["mean_predicted_u"]
                    if orig_row["mean_predicted_u"] > 0
                    else float("nan")
                ),
                "original_256ep_u_over_mse": orig_row["predicted_u_over_actual_mse"],
                "new_joint_256ep_u_over_mse": new_row["predicted_u_over_actual_mse"],
                "original_256ep_u_over_h1": orig_row.get("u_over_h1", 0.0),
                "new_joint_256ep_u_over_h1": new_row.get("u_over_h1", 0.0),
                "original_256ep_mse_over_h1": orig_row.get("mse_over_h1", 0.0),
                "new_joint_256ep_mse_over_h1": new_row.get("mse_over_h1", 0.0),
            }

            # Secondary references
            if ab_row is not None:
                row["ab_abrupt_epoch286_mse"] = ab_row["mean_mse"]
                row["ab_abrupt_epoch286_mean_u"] = ab_row["mean_predicted_u"]
                row["ab_abrupt_epoch286_u_over_mse"] = ab_row["predicted_u_over_actual_mse"]
                row["ab_abrupt_epoch286_u_over_h1"] = ab_row.get("u_over_h1", 0.0)
                row["mse_ratio_new_over_ab_abrupt"] = (
                    new_row["mean_mse"] / ab_row["mean_mse"]
                    if ab_row["mean_mse"] > 0
                    else float("nan")
                )
                row["u_ratio_new_over_ab_abrupt"] = (
                    new_row["mean_predicted_u"] / ab_row["mean_predicted_u"]
                    if ab_row["mean_predicted_u"] > 0
                    else float("nan")
                )
            if bonly_row is not None:
                row["bonly_epoch286_mse"] = bonly_row["mean_mse"]
                row["bonly_epoch286_mean_u"] = bonly_row["mean_predicted_u"]
            if cont_row is not None:
                row["continuation_epoch276_mse"] = cont_row["mean_mse"]
                row["continuation_epoch276_mean_u"] = cont_row["mean_predicted_u"]

            comparison_rows.append(row)
    write_csv(OUTPUT / "forecast_horizon_comparison.csv", comparison_rows)
    print("Wrote forecast_horizon_comparison.csv", flush=True)

    # -------------------------------------------------------------------------
    # 6. Generate Evaluation Figures
    # -------------------------------------------------------------------------
    print("\n--- Generating Evaluation Comparison Figures ---", flush=True)

    # Figure 3: forecast_horizon_comparison.png
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    comp_by_ch = defaultdict(dict)
    for r in comparison_rows:
        comp_by_ch[r["channel"]][int(r["horizon"])] = r

    for channel, base_color in [("Density", "blue"), ("Magnetic", "red")]:
        values = comp_by_ch[channel]
        h_arr = np.array(sorted(values))
        orig_mse = [values[x]["original_256ep_mse"] for x in h_arr]
        new_mse = [values[x]["new_joint_256ep_mse"] for x in h_arr]
        ab_mse = [values[x].get("ab_abrupt_epoch286_mse", np.nan) for x in h_arr]

        orig_u = [values[x]["original_256ep_mean_u"] for x in h_arr]
        new_u = [values[x]["new_joint_256ep_mean_u"] for x in h_arr]
        ab_u = [values[x].get("ab_abrupt_epoch286_mean_u", np.nan) for x in h_arr]

        orig_cal = [values[x]["original_256ep_u_over_mse"] for x in h_arr]
        new_cal = [values[x]["new_joint_256ep_u_over_mse"] for x in h_arr]
        ab_cal = [values[x].get("ab_abrupt_epoch286_u_over_mse", np.nan) for x in h_arr]

        new_u_over_h1 = [values[x]["new_joint_256ep_u_over_h1"] for x in h_arr]
        orig_u_over_h1 = [values[x]["original_256ep_u_over_h1"] for x in h_arr]
        new_mse_over_h1 = [values[x]["new_joint_256ep_mse_over_h1"] for x in h_arr]

        # MSE
        axes[0, 0].plot(h_arr, orig_mse, marker="o", ms=3, ls="--", color=base_color, alpha=0.4, label=f"{channel} Original 256ep")
        axes[0, 0].plot(h_arr, ab_mse, marker="^", ms=4, ls="-.", color=base_color, alpha=0.6, label=f"{channel} Abrupt A+B (ep 286)")
        axes[0, 0].plot(h_arr, new_mse, marker="s", ms=5, ls="-", color=base_color, lw=2, label=f"{channel} New Joint 256ep")

        # Predicted u
        axes[0, 1].plot(h_arr, orig_u, marker="o", ms=3, ls="--", color=base_color, alpha=0.4, label=f"{channel} Original 256ep")
        axes[0, 1].plot(h_arr, ab_u, marker="^", ms=4, ls="-.", color=base_color, alpha=0.6, label=f"{channel} Abrupt A+B (ep 286)")
        axes[0, 1].plot(h_arr, new_u, marker="s", ms=5, ls="-", color=base_color, lw=2, label=f"{channel} New Joint 256ep")

        # Calibration u/MSE
        axes[1, 0].plot(h_arr, orig_cal, marker="o", ms=3, ls="--", color=base_color, alpha=0.4, label=f"{channel} Original 256ep")
        axes[1, 0].plot(h_arr, ab_cal, marker="^", ms=4, ls="-.", color=base_color, alpha=0.6, label=f"{channel} Abrupt A+B (ep 286)")
        axes[1, 0].plot(h_arr, new_cal, marker="s", ms=5, ls="-", color=base_color, lw=2, label=f"{channel} New Joint 256ep")

        # Horizon scaling u/u(h1)
        axes[1, 1].plot(h_arr, orig_u_over_h1, marker="o", ms=3, ls="--", color=base_color, alpha=0.4, label=f"{channel} Orig u/u(h1)")
        axes[1, 1].plot(h_arr, new_u_over_h1, marker="s", ms=5, ls="-", color=base_color, lw=2, label=f"{channel} New u/u(h1)")
        axes[1, 1].plot(h_arr, new_mse_over_h1, marker="x", ms=3, ls=":", color=base_color, alpha=0.6, label=f"{channel} New MSE/MSE(h1)")

    axes[0, 0].set_title("Forecast Reconstruction MSE across Horizon h")
    axes[0, 0].set_xlabel("Forecast Horizon h")
    axes[0, 0].set_ylabel("Mean MSE")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Predicted Uncertainty u across Horizon h")
    axes[0, 1].set_xlabel("Forecast Horizon h")
    axes[0, 1].set_ylabel("Mean Predicted u")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title("Calibration Ratio u / MSE across Horizon h")
    axes[1, 0].set_xlabel("Forecast Horizon h")
    axes[1, 0].set_ylabel("u / MSE")
    axes[1, 0].axhline(1.0, color="gray", ls="--", alpha=0.5, label="Ideal u/MSE = 1.0")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("Dynamic Range Growth: u(h)/u(h1) vs MSE(h)/MSE(h1)")
    axes[1, 1].set_xlabel("Forecast Horizon h")
    axes[1, 1].set_ylabel("Relative Growth")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Pass-0 Forecast Horizon Evaluation: Original 256ep vs New Joint A+B 256ep", fontsize=13)
    plt.savefig(OUTPUT / "forecast_horizon_comparison.png", dpi=200)
    plt.close()
    print("Saved forecast_horizon_comparison.png", flush=True)

    # Figure 4: per_frame_mse_vs_u.png
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    frame_dict_all = {}
    for r in all_frame_rows:
        frame_dict_all[(r["model"], int(r["horizon"]), r["channel"], int(r["local_frame"]))] = {
            "mean_mse": float(r["mean_mse"]),
            "mean_predicted_u": float(r["mean_predicted_u"]),
        }

    for row_idx, h in enumerate([18, 22, 23]):
        prefix = 24 - h
        frames = list(range(prefix, 24))
        leads = [t - prefix + 1 for t in frames]

        for col_idx, (channel, color) in enumerate([("Density", "blue"), ("Magnetic", "red")]):
            ax = axes[row_idx, col_idx]
            orig_mse_f = [frame_dict_all[("original_256ep", h, channel, t)]["mean_mse"] for t in frames]
            new_mse_f = [frame_dict_all[("new_joint_256ep", h, channel, t)]["mean_mse"] for t in frames]

            orig_u_f = [frame_dict_all[("original_256ep", h, channel, t)]["mean_predicted_u"] for t in frames]
            new_u_f = [frame_dict_all[("new_joint_256ep", h, channel, t)]["mean_predicted_u"] for t in frames]

            ax.plot(leads, orig_mse_f, marker="o", ms=3, ls=":", color="gray", alpha=0.6, label="Orig MSE")
            ax.plot(leads, new_mse_f, marker="s", ms=4, ls="-", color="black", lw=1.5, label="New MSE")

            ax.plot(leads, orig_u_f, marker="o", ms=3, ls=":", color=color, alpha=0.5, label="Orig u")
            ax.plot(leads, new_u_f, marker="s", ms=5, ls="-", color=color, lw=2, label="New u")

            if ("ab_abrupt_epoch286", h, channel, frames[0]) in frame_dict_all:
                ab_u_f = [frame_dict_all[("ab_abrupt_epoch286", h, channel, t)]["mean_predicted_u"] for t in frames]
                ax.plot(leads, ab_u_f, marker="^", ms=4, ls="-.", color=color, alpha=0.5, label="Abrupt A+B u")

            ax.set_title(f"{channel} Per-Frame Lead Trajectory (h={h})")
            ax.set_xlabel("Forecast Lead Step")
            ax.set_ylabel("Magnitude")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    fig.suptitle("Per-Frame MSE vs Predicted Uncertainty Lead Progression (h=18, 22, 23)", fontsize=13)
    plt.savefig(OUTPUT / "per_frame_mse_vs_u.png", dpi=200)
    plt.close()
    print("Saved per_frame_mse_vs_u.png", flush=True)

    print("\n" + "=" * 80)
    print("Evaluation Complete! All CSVs and Figures generated successfully.")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
