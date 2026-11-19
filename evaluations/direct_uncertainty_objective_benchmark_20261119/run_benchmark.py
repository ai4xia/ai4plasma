#!/usr/bin/env python3
"""Direct Reconstruction-Error Prediction: Uncertainty Objective Benchmark.

Compares direct error-regression objectives against existing Gaussian and Student-t NLL:
- Candidate A: Absolute-error prediction with ordinary MSE (L_A = E[(sqrt(u) - |e|)^2])
- Candidate B: Squared-error prediction with Smooth L1 (beta=1.0)
- Candidate C: Squared-error prediction with Generalized KL Divergence (I-Divergence)
- Reference 1: Original squared-error MSE (L_MSE = E[(u - q)^2])
- Reference 2: Gaussian NLL (Path B only, and full A+B)
- Reference 3: Student-t NLL (df=3.0, Path B only, and full A+B)

Evaluates on 8 representative held-out windows x 11 mask conditions (88 paired samples)
using the mature epoch-256 frozen checkpoint.
"""

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from data.masking import (
    full_mse_loss,
    make_fixed_validation_mask,
    make_uncertain_input,
    predicted_error_mse_loss,
)
from data.vpic_hdf5_dataset import (
    VPICWindowDataset,
    load_fixed_split,
    parse_run_name,
)
from visualize_mask_patterns_unet3d import select_validation_statistics_indices
from models.unet3d import UNet3D

OUTPUT_DIR = Path("evaluations/direct_uncertainty_objective_benchmark_20261119")
CKPT_PATH = Path("runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt")

SEED = 1234
STUDENT_T_DF = 3.0

MASK_CONDITIONS = [
    ("spatial_random", "standard", None),
    ("spatial_grid", "standard", None),
    ("spatial_block", "standard", None),
    ("temporal_random", "standard", None),
    ("temporal_block", "standard", None),
    ("temporal_extrapolation_h1", "extrapolation", 1),
    ("temporal_extrapolation_h6", "extrapolation", 6),
    ("temporal_extrapolation_h12", "extrapolation", 12),
    ("temporal_extrapolation_h18", "extrapolation", 18),
    ("temporal_extrapolation_h22", "extrapolation", 22),
    ("temporal_extrapolation_h23", "extrapolation", 23),
]


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path} ({len(rows)} rows)")


def json_ready(v):
    if isinstance(v, dict):
        return {str(k): json_ready(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_ready(val) for val in v]
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, Path):
        return str(v)
    return v


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


def partition_parameters(model: UNet3D):
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_tensors = [p for _, p in named_params]

    groups = {
        "shared_unet": [],
        "mean_head": [],
        "var_head": [],
    }

    for i, (name, _) in enumerate(named_params):
        if name.startswith("out."):
            groups["mean_head"].append(i)
        elif name.startswith("variance_out."):
            groups["var_head"].append(i)
        else:
            groups["shared_unet"].append(i)

    return named_params, param_tensors, groups


def compute_grad_vector(loss: torch.Tensor, param_list: List[torch.Tensor], retain: bool = True) -> torch.Tensor:
    raw = torch.autograd.grad(loss, param_list, retain_graph=retain, allow_unused=True)
    flat = []
    for g, p in zip(raw, param_list):
        if g is not None:
            flat.append(g.float().reshape(-1))
        else:
            flat.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
    return torch.cat(flat)


def compute_feature_grad(loss: torch.Tensor, feature_tensor: torch.Tensor, retain: bool = True) -> torch.Tensor:
    raw = torch.autograd.grad(loss, feature_tensor, retain_graph=retain, allow_unused=True)[0]
    if raw is None:
        return torch.zeros_like(feature_tensor, dtype=torch.float32)
    return raw.float()


def compute_cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    n1 = torch.norm(v1).item()
    n2 = torch.norm(v2).item()
    if n1 > 0 and n2 > 0:
        dot = torch.dot(v1, v2).item()
        return float(dot / (n1 * n2))
    return float("nan")


# Loss implementations in FP32
def loss_candidate_a_mae(u: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # Candidate A: Absolute-error prediction with ordinary MSE: L_A = E[(sqrt(u) - |e|)^2]
    # Path A OFF: target is detached
    a_target = torch.abs(pred.detach() - target)
    a_pred = torch.sqrt(torch.clamp_min(u, 1e-8))
    return torch.mean((a_pred - a_target) ** 2)


def loss_candidate_b_smooth_l1(u: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # Candidate B: Squared-error prediction with Smooth L1 (beta=1.0)
    # Path A OFF: target is detached
    q = (pred.detach() - target).square()
    return F.smooth_l1_loss(u, q, beta=1.0, reduction="mean")


def loss_candidate_c_gen_kl(u: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # Candidate C: Generalized KL divergence D(q || u) = q * log(q / u) + u - q
    # Path A OFF: target is detached
    q = (pred.detach() - target).square()
    # xlogy-style: where q > 0, q * (log(q + 1e-30) - log(u)), else 0
    term = torch.where(q > 0, q * (torch.log(q + 1e-30) - torch.log(u)), torch.zeros_like(q))
    return torch.mean(term + u - q)


def loss_ref_direct_mse(u: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # Reference 1: Original direct squared-error MSE: L_MSE = E[(u - q)^2]
    # Path A OFF: target is detached
    q = (pred.detach() - target).square()
    return torch.mean((u - q) ** 2)


def loss_ref_gaussian_path_b(u: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # Reference 2: Gaussian NLL, Path B only (target detached)
    q = (pred.detach() - target).square()
    u_clamped = torch.clamp_min(u, 1e-6)
    return torch.mean(0.5 * (torch.log(u_clamped) + q / u_clamped))


def loss_ref_gaussian_full(u: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # Reference 2: Gaussian NLL, Full A+B
    return predicted_error_mse_loss(u, pred, target, "gaussian_nll", detach_mean=False)


def loss_ref_student_t_path_b(u: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # Reference 3: Student-t NLL, Path B only (target detached)
    return predicted_error_mse_loss(u, pred, target, "student_t_nll", student_t_df=STUDENT_T_DF, detach_mean=True)


def loss_ref_student_t_full(u: torch.Tensor, target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    # Reference 3: Student-t NLL, Full A+B
    return predicted_error_mse_loss(u, pred, target, "student_t_nll", student_t_df=STUDENT_T_DF, detach_mean=False)


def main():
    print("=" * 80)
    print("Direct Reconstruction-Error Prediction: Uncertainty Objective Benchmark")
    print("=" * 80, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ckpt, model, mean_norm, std_norm = load_model(CKPT_PATH, device)
    print(f"Loaded checkpoint: {CKPT_PATH.name} (epoch {ckpt['epoch']})")

    named_params, param_tensors, param_groups = partition_parameters(model)
    shared_params = [param_tensors[i] for i in param_groups["shared_unet"]]
    var_head_params = [param_tensors[i] for i in param_groups["var_head"]]
    mean_head_params = [param_tensors[i] for i in param_groups["mean_head"]]
    print(f"Shared U-Net params: {sum(p.numel() for p in shared_params):,}")
    print(f"Uncertainty head params: {sum(p.numel() for p in var_head_params):,}")
    print(f"Mean head params: {sum(p.numel() for p in mean_head_params):,}")

    # Forward hook on model.up0 to capture shared feature tensor F
    captured = {}
    def up0_hook(module, inp, out):
        captured["F"] = out
    handle_up0 = model.up0.register_forward_hook(up0_hook)

    # Forward hook on model.variance_out to capture raw pre-activation logits z
    def var_out_hook(module, inp, out):
        captured["z"] = out
    handle_var = model.variance_out.register_forward_hook(var_out_hook)

    # Select 8 Representative Windows
    split = load_fixed_split()
    dataset = VPICWindowDataset(
        h5_dir=ckpt["args"]["h5_dir"],
        betas=ckpt["args"]["betas"],
        delta_t=24,
        stride_t=1,
        layout="C T X Z",
        return_metadata=True,
    )
    test_indices = select_validation_statistics_indices(dataset, set(split["test_runs"]), 24, 1)

    idx_by_beta = {}
    for idx in test_indices:
        _fi, run_name, _t0 = dataset.samples[idx]
        info = parse_run_name(run_name)
        beta_str = f"{info['beta']:g}"
        if beta_str not in idx_by_beta:
            idx_by_beta[beta_str] = (idx, run_name, beta_str)

    target_betas = ["0.01", "0.02", "0.04", "0.06", "0.08", "0.1", "0.15", "0.2"]
    selected_windows = [idx_by_beta[b] for b in target_betas]
    print(f"Selected {len(selected_windows)} paired windows from distinct simulation runs:")
    for idx, r, b in selected_windows:
        print(f"  idx={idx:6d} | beta={b:4s} | run={r}")

    # Objective registry
    OBJECTIVES = [
        ("candidate_a_mae", "Candidate A: Absolute-Error MSE", loss_candidate_a_mae, "direct"),
        ("candidate_b_smooth_l1", "Candidate B: Smooth L1 (beta=1.0)", loss_candidate_b_smooth_l1, "direct"),
        ("candidate_c_gen_kl", "Candidate C: Generalized KL", loss_candidate_c_gen_kl, "direct"),
        ("ref1_direct_mse", "Reference 1: Squared-Error MSE", loss_ref_direct_mse, "direct"),
        ("ref2_gauss_path_b", "Reference 2: Gaussian NLL (Path B)", loss_ref_gaussian_path_b, "nll_path_b"),
        ("ref2_gauss_full_ab", "Reference 2: Gaussian NLL (Full A+B)", loss_ref_gaussian_full, "nll_full_ab"),
        ("ref3_student_t_path_b", "Reference 3: Student-t NLL (Path B)", loss_ref_student_t_path_b, "nll_path_b"),
        ("ref3_student_t_full_ab", "Reference 3: Student-t NLL (Full A+B)", loss_ref_student_t_full, "nll_full_ab"),
    ]

    # Data collection containers
    gradient_norm_rows = []
    cosine_sim_rows = []
    tail_stat_rows = []

    print("\nStarting evaluation across 8 windows and 11 mask conditions (88 total)...", flush=True)
    t0_eval = time.time()

    for win_idx_num, (dataset_idx, run_name, beta_str) in enumerate(selected_windows):
        sample = dataset[dataset_idx]
        block = sample["block"].unsqueeze(0).to(device)
        target = ((block - mean_norm) / (std_norm + 1e-8)).float()

        print(f"Evaluating window {win_idx_num + 1}/8 (idx={dataset_idx}, beta={beta_str})...", flush=True)

        for mask_idx, (mask_name, mask_type, horizon) in enumerate(MASK_CONDITIONS):
            noise_gen = torch.Generator(device=device).manual_seed(SEED + 1000 * win_idx_num + mask_idx)
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

            if mask_type == "standard":
                mask, _ = make_fixed_validation_mask(mask_name, target.shape, device=device, dtype=target.dtype)
            else:
                prefix = 24 - horizon
                mask = torch.ones_like(target)
                mask[:, :, prefix:] = 0.0

            input_var = 1.0 - mask
            model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)

            # Forward pass: Pass 0, Path B enabled (detach_variance_features=False)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, var = model(model_input, detach_variance_features=False)
            pred = pred.float()
            var = var.float()
            F_feat = captured["F"]
            z_logits = captured["z"]

            # Reconstruction loss (Production MSE)
            l_mu = full_mse_loss(pred, target, mask)

            # Gradient for reconstruction MSE
            g_mu_full = compute_grad_vector(l_mu, shared_params, retain=True)
            norm_g_mu_full = torch.norm(g_mu_full).item()

            gF_mu = compute_feature_grad(l_mu, F_feat, retain=True)
            norm_gF_mu = torch.norm(gF_mu).item()

            g_mu_head = compute_grad_vector(l_mu, mean_head_params, retain=True)
            norm_g_mu_head = torch.norm(g_mu_head).item()

            # Record baseline row
            gradient_norm_rows.append({
                "window_idx": dataset_idx,
                "run_name": run_name,
                "beta": beta_str,
                "mask_condition": mask_name,
                "mask_type": mask_type,
                "horizon": horizon if horizon is not None else "",
                "objective_id": "reconstruction_mse",
                "objective_name": "Reconstruction MSE (L_mu)",
                "objective_category": "reconstruction",
                "loss_value": l_mu.item(),
                "norm_shared_backbone": norm_g_mu_full,
                "norm_shared_feature": norm_gF_mu,
                "norm_head": norm_g_mu_head,
                "ratio_backbone_to_mu": 1.0,
                "ratio_feature_to_mu": 1.0,
            })

            # Evaluate each uncertainty objective
            for obj_idx, (obj_id, obj_name, loss_fn, obj_cat) in enumerate(OBJECTIVES):
                is_last_obj = (obj_idx == len(OBJECTIVES) - 1) and (horizon not in [18, 22, 23])
                loss_val = loss_fn(var, target, pred)

                # Gradient vectors
                g_full = compute_grad_vector(loss_val, shared_params, retain=True)
                norm_g_full = torch.norm(g_full).item()

                gF = compute_feature_grad(loss_val, F_feat, retain=True)
                norm_gF = torch.norm(gF).item()

                g_head = compute_grad_vector(loss_val, var_head_params, retain=True)
                norm_g_head = torch.norm(g_head).item()

                # Cosine similarity with reconstruction MSE
                cos_sim = compute_cosine_similarity(g_full, g_mu_full)

                gradient_norm_rows.append({
                    "window_idx": dataset_idx,
                    "run_name": run_name,
                    "beta": beta_str,
                    "mask_condition": mask_name,
                    "mask_type": mask_type,
                    "horizon": horizon if horizon is not None else "",
                    "objective_id": obj_id,
                    "objective_name": obj_name,
                    "objective_category": obj_cat,
                    "loss_value": loss_val.item(),
                    "norm_shared_backbone": norm_g_full,
                    "norm_shared_feature": norm_gF,
                    "norm_head": norm_g_head,
                    "ratio_backbone_to_mu": norm_g_full / (norm_g_mu_full + 1e-12),
                    "ratio_feature_to_mu": norm_gF / (norm_gF_mu + 1e-12),
                })

                cosine_sim_rows.append({
                    "window_idx": dataset_idx,
                    "run_name": run_name,
                    "beta": beta_str,
                    "mask_condition": mask_name,
                    "mask_type": mask_type,
                    "horizon": horizon if horizon is not None else "",
                    "objective_id": obj_id,
                    "objective_name": obj_name,
                    "objective_category": obj_cat,
                    "cosine_similarity_with_mu": cos_sim,
                })

            # Section 9: Extreme-error and numerical-stability diagnostics for h18, h22, h23
            if horizon in [18, 22, 23]:
                # Analyze per-voxel errors and gradients
                # Channels: 0..2 = Magnetic, 3 = Density
                e = (pred.detach() - target).float()
                abs_e = torch.abs(e)
                q = e.square()
                u_val = var.float()
                q_over_u = q / torch.clamp_min(u_val, 1e-6)

                # Collect per-voxel gradients for each objective
                for obj_id, obj_name, loss_fn, obj_cat in OBJECTIVES:
                    loss_v = loss_fn(var, target, pred)

                    # Per-voxel gradient w.r.t u
                    grad_u = torch.autograd.grad(loss_v, var, retain_graph=True)[0].detach().float()
                    # Per-voxel gradient w.r.t z (logits)
                    grad_z = torch.autograd.grad(loss_v, z_logits, retain_graph=True)[0].detach().float()
                    # Per-voxel gradient w.r.t log u: u * grad_u
                    grad_log_u = (u_val * grad_u).detach().float()

                    for field_group, ch_slice in [("magnetic", slice(0, 3)), ("density", slice(3, 4))]:
                        fg_abs_e = abs_e[:, ch_slice].reshape(-1)
                        fg_q = q[:, ch_slice].reshape(-1)
                        fg_u = u_val[:, ch_slice].reshape(-1)
                        fg_q_over_u = q_over_u[:, ch_slice].reshape(-1)

                        fg_gu = grad_u[:, ch_slice].reshape(-1)
                        fg_gz = grad_z[:, ch_slice].reshape(-1)
                        fg_glogu = grad_log_u[:, ch_slice].reshape(-1)

                        # Upward gradient magnitude (gradient that increases u: -grad_u or -grad_z or -grad_log_u)
                        up_gz = torch.clamp(-fg_gz, min=0.0)
                        up_glogu = torch.clamp(-fg_glogu, min=0.0)

                        # Top 1% error threshold
                        q_top1_thresh = torch.quantile(fg_q, 0.99).item()
                        top1_mask = fg_q >= q_top1_thresh

                        sum_up_gz = up_gz.sum().item()
                        top1_up_gz = up_gz[top1_mask].sum().item() if top1_mask.any() else 0.0
                        frac_top1_gz = float(top1_up_gz / (sum_up_gz + 1e-12)) if sum_up_gz > 0 else 0.0

                        sum_up_glogu = up_glogu.sum().item()
                        top1_up_glogu = up_glogu[top1_mask].sum().item() if top1_mask.any() else 0.0
                        frac_top1_glogu = float(top1_up_glogu / (sum_up_glogu + 1e-12)) if sum_up_glogu > 0 else 0.0

                        # Percentiles of absolute gradients
                        abs_gu = fg_gu.abs()
                        abs_gz = fg_gz.abs()
                        abs_glogu = fg_glogu.abs()

                        tail_stat_rows.append({
                            "window_idx": dataset_idx,
                            "run_name": run_name,
                            "beta": beta_str,
                            "horizon": horizon,
                            "field_group": field_group,
                            "objective_id": obj_id,
                            "objective_name": obj_name,
                            "objective_category": obj_cat,
                            "nonfinite_count": int((~torch.isfinite(fg_gz)).sum().item()),
                            # Error statistics
                            "abs_e_median": float(torch.median(fg_abs_e).item()),
                            "abs_e_p95": float(torch.quantile(fg_abs_e, 0.95).item()),
                            "abs_e_max": float(torch.max(fg_abs_e).item()),
                            "q_median": float(torch.median(fg_q).item()),
                            "q_p95": float(torch.quantile(fg_q, 0.95).item()),
                            "q_max": float(torch.max(fg_q).item()),
                            "u_median": float(torch.median(fg_u).item()),
                            "u_p95": float(torch.quantile(fg_u, 0.95).item()),
                            "q_over_u_median": float(torch.median(fg_q_over_u).item()),
                            "q_over_u_p95": float(torch.quantile(fg_q_over_u, 0.95).item()),
                            "q_over_u_max": float(torch.max(fg_q_over_u).item()),
                            # Gradient w.r.t u
                            "grad_u_median": float(torch.median(abs_gu).item()),
                            "grad_u_p95": float(torch.quantile(abs_gu, 0.95).item()),
                            "grad_u_p99": float(torch.quantile(abs_gu, 0.99).item()),
                            "grad_u_max": float(torch.max(abs_gu).item()),
                            # Gradient w.r.t z (logits)
                            "grad_z_median": float(torch.median(abs_gz).item()),
                            "grad_z_p95": float(torch.quantile(abs_gz, 0.95).item()),
                            "grad_z_p99": float(torch.quantile(abs_gz, 0.99).item()),
                            "grad_z_max": float(torch.max(abs_gz).item()),
                            # Gradient w.r.t log u
                            "grad_log_u_median": float(torch.median(abs_glogu).item()),
                            "grad_log_u_p95": float(torch.quantile(abs_glogu, 0.95).item()),
                            "grad_log_u_p99": float(torch.quantile(abs_glogu, 0.99).item()),
                            "grad_log_u_max": float(torch.max(abs_glogu).item()),
                            # Tail concentration
                            "frac_upward_grad_z_top1pct": frac_top1_gz,
                            "frac_upward_grad_log_u_top1pct": frac_top1_glogu,
                        })

    handle_up0.remove()
    handle_var.remove()
    eval_time = time.time() - t0_eval
    print(f"\nCompleted 88 paired evaluations in {eval_time:.1f}s.")

    # Save raw CSVs
    write_csv(OUTPUT_DIR / "objective_gradient_norms.csv", gradient_norm_rows)
    write_csv(OUTPUT_DIR / "gradient_cosine_similarity.csv", cosine_sim_rows)
    write_csv(OUTPUT_DIR / "gradient_tail_statistics.csv", tail_stat_rows)

    # -------------------------------------------------------------
    # Statistical Summary and Comparison Table
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Computing Statistical Summary Across Objectives...")
    print("=" * 80, flush=True)

    def summarize_series(vals: List[float]):
        arr = np.array(vals)
        return {
            "median": float(np.median(arr)),
            "p16": float(np.percentile(arr, 16)),
            "p84": float(np.percentile(arr, 84)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    summary_rows = []
    summary_data = {
        "checkpoint": str(CKPT_PATH),
        "total_evaluations": 88,
        "objectives": {},
        "tail_statistics_summary": {},
    }

    for obj_id, obj_name, _, obj_cat in OBJECTIVES:
        obj_norms = [r for r in gradient_norm_rows if r["objective_id"] == obj_id]
        obj_cos = [r for r in cosine_sim_rows if r["objective_id"] == obj_id]

        for subset_name, subset_filter in [
            ("all_conditions", lambda r: True),
            ("standard_masks_only", lambda r: r["mask_type"] == "standard"),
            ("extrapolation_only", lambda r: r["mask_type"] == "extrapolation"),
        ]:
            sub_norms = [r for r in obj_norms if subset_filter(r)]
            sub_cos = [r for r in obj_cos if subset_filter(r)]

            losses = [r["loss_value"] for r in sub_norms]
            backbone_norms = [r["norm_shared_backbone"] for r in sub_norms]
            feature_norms = [r["norm_shared_feature"] for r in sub_norms]
            head_norms = [r["norm_head"] for r in sub_norms]
            ratios_backbone = [r["ratio_backbone_to_mu"] for r in sub_norms]
            ratios_feat = [r["ratio_feature_to_mu"] for r in sub_norms]
            cos_vals = [r["cosine_similarity_with_mu"] for r in sub_cos]

            stat_loss = summarize_series(losses)
            stat_bb = summarize_series(backbone_norms)
            stat_feat = summarize_series(feature_norms)
            stat_head = summarize_series(head_norms)
            stat_ratio_bb = summarize_series(ratios_backbone)
            stat_ratio_feat = summarize_series(ratios_feat)
            stat_cos = summarize_series(cos_vals)

            summary_rows.append({
                "objective_id": obj_id,
                "objective_name": obj_name,
                "category": obj_cat,
                "subset": subset_name,
                "count": len(sub_norms),
                "loss_median": stat_loss["median"],
                "norm_backbone_median": stat_bb["median"],
                "norm_backbone_p16": stat_bb["p16"],
                "norm_backbone_p84": stat_bb["p84"],
                "norm_feature_median": stat_feat["median"],
                "norm_head_median": stat_head["median"],
                "ratio_backbone_median": stat_ratio_bb["median"],
                "ratio_backbone_p16": stat_ratio_bb["p16"],
                "ratio_backbone_p84": stat_ratio_bb["p84"],
                "ratio_feature_median": stat_ratio_feat["median"],
                "cosine_sim_median": stat_cos["median"],
                "cosine_sim_p16": stat_cos["p16"],
                "cosine_sim_p84": stat_cos["p84"],
            })

            if obj_id not in summary_data["objectives"]:
                summary_data["objectives"][obj_id] = {
                    "name": obj_name,
                    "category": obj_cat,
                    "subsets": {},
                }
            summary_data["objectives"][obj_id]["subsets"][subset_name] = {
                "loss": stat_loss,
                "norm_backbone": stat_bb,
                "norm_feature": stat_feat,
                "norm_head": stat_head,
                "ratio_backbone_to_mu": stat_ratio_bb,
                "ratio_feature_to_mu": stat_ratio_feat,
                "cosine_similarity": stat_cos,
            }

    write_csv(OUTPUT_DIR / "objective_comparison_summary.csv", summary_rows)

    # Summarize tail statistics for extreme horizons
    for obj_id, obj_name, _, obj_cat in OBJECTIVES:
        obj_tail = [r for r in tail_stat_rows if r["objective_id"] == obj_id]
        summary_data["tail_statistics_summary"][obj_id] = {
            "name": obj_name,
            "category": obj_cat,
            "magnetic": {
                "grad_z_median": float(np.median([r["grad_z_median"] for r in obj_tail if r["field_group"] == "magnetic"])),
                "grad_z_p95": float(np.median([r["grad_z_p95"] for r in obj_tail if r["field_group"] == "magnetic"])),
                "grad_z_max": float(np.max([r["grad_z_max"] for r in obj_tail if r["field_group"] == "magnetic"])),
                "frac_top1pct_median": float(np.median([r["frac_upward_grad_z_top1pct"] for r in obj_tail if r["field_group"] == "magnetic"])),
            },
            "density": {
                "grad_z_median": float(np.median([r["grad_z_median"] for r in obj_tail if r["field_group"] == "density"])),
                "grad_z_p95": float(np.median([r["grad_z_p95"] for r in obj_tail if r["field_group"] == "density"])),
                "grad_z_max": float(np.max([r["grad_z_max"] for r in obj_tail if r["field_group"] == "density"])),
                "frac_top1pct_median": float(np.median([r["frac_upward_grad_z_top1pct"] for r in obj_tail if r["field_group"] == "density"])),
            },
        }

    summary_json_path = OUTPUT_DIR / "benchmark_summary.json"
    summary_json_path.write_text(json.dumps(json_ready(summary_data), indent=2))
    print(f"Saved: {summary_json_path}")

    # -------------------------------------------------------------
    # Generate Deliverable Figures (at most 3)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Generating Deliverable Figures...")
    print("=" * 80, flush=True)

    # Color scheme
    cat_colors = {
        "candidate_a_mae": "#4a90e2",
        "candidate_b_smooth_l1": "#50e3c2",
        "candidate_c_gen_kl": "#e94e77",
        "ref1_direct_mse": "#f5a623",
        "ref2_gauss_path_b": "#b8e986",
        "ref2_gauss_full_ab": "#7ed321",
        "ref3_student_t_path_b": "#bd10e0",
        "ref3_student_t_full_ab": "#9013fe",
    }

    # Figure 1: Shared-backbone gradient norm ratios across objectives (log scale)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax_idx, (subset_title, subset_key) in enumerate([("Standard Mask Families", "standard_masks_only"), ("Extrapolation Horizons", "extrapolation_only")]):
        ax = axes[ax_idx]
        data_boxes = []
        labels = []
        colors = []

        for obj_id, obj_name, _, _ in OBJECTIVES:
            sub = summary_data["objectives"][obj_id]["subsets"][subset_key]
            # Get raw values
            rows_sub = [r for r in gradient_norm_rows if r["objective_id"] == obj_id and (r["mask_type"] == ("standard" if subset_key == "standard_masks_only" else "extrapolation"))]
            vals = [r["ratio_backbone_to_mu"] for r in rows_sub]
            data_boxes.append(vals)
            # Short label
            short_lbl = obj_name.replace("Candidate ", "Cand ").replace("Reference ", "Ref ")
            labels.append(short_lbl)
            colors.append(cat_colors[obj_id])

        bp = ax.boxplot(data_boxes, patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4})
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.5)

        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Parity with MSE (1.0)")
        ax.set_yscale("log")
        ax.set_title(f"Shared-Backbone Gradient Ratio $R = ||\\nabla_\\theta L_u|| / ||\\nabla_\\theta L_\\mu||$\n({subset_title})", fontsize=11, fontweight="bold")
        ax.set_ylabel("Gradient Norm Ratio (Log Scale)" if ax_idx == 0 else "")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        if ax_idx == 0:
            ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "figure1_shared_backbone_gradient_ratios.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Gradient Alignment (Cosine Similarity with Reconstruction MSE)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    data_cos = []
    labels_cos = []
    colors_cos = []

    for obj_id, obj_name, _, _ in OBJECTIVES:
        rows_obj = [r for r in cosine_sim_rows if r["objective_id"] == obj_id]
        cos_vals = [r["cosine_similarity_with_mu"] for r in rows_obj]
        data_cos.append(cos_vals)
        labels_cos.append(obj_name.replace("Candidate ", "Cand ").replace("Reference ", "Ref "))
        colors_cos.append(cat_colors[obj_id])

    bp = ax.boxplot(data_cos, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5})
    ax.set_xticks(range(1, len(labels_cos) + 1))
    ax.set_xticklabels(labels_cos, rotation=35, ha="right", fontsize=9)

    for patch, col in zip(bp["boxes"], colors_cos):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)

    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.2)
    ax.set_title("Cosine Similarity Between Shared-Backbone Uncertainty and Reconstruction Gradients\n$\\cos(\\nabla_\\theta L_u, \\nabla_\\theta L_\\mu)$ Across 88 Diagnostic Samples", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cosine Similarity", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "figure2_gradient_alignment.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Figure 3: Extreme-Error Supervision Across Uncertainty Objectives (h18, h22, h23)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Median and P95 logit gradient magnitude |dz| on Density
    ax_a = axes[0]
    obj_short = [o[0] for o in OBJECTIVES]
    obj_names_short = [o[1].replace("Candidate ", "Cand ").replace("Reference ", "Ref ") for o in OBJECTIVES]
    p95_density = [summary_data["tail_statistics_summary"][o[0]]["density"]["grad_z_p95"] for o in OBJECTIVES]
    med_density = [summary_data["tail_statistics_summary"][o[0]]["density"]["grad_z_median"] for o in OBJECTIVES]

    x_pos = np.arange(len(OBJECTIVES))
    ax_a.bar(x_pos - 0.2, med_density, width=0.4, label="Median |dz|", color="#4a90e2", alpha=0.85)
    ax_a.bar(x_pos + 0.2, p95_density, width=0.4, label="95th Percentile |dz|", color="#e94e77", alpha=0.85)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(obj_names_short, rotation=45, ha="right", fontsize=8)
    ax_a.set_yscale("log")
    ax_a.set_title("Per-Voxel Uncertainty Logit Gradient $|\\partial L / \\partial z|$\n(Density Channel, Extreme Horizons h18-h23)", fontsize=10.5, fontweight="bold")
    ax_a.set_ylabel("Gradient Magnitude (Log Scale)", fontsize=9.5)
    ax_a.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_a.legend(loc="upper right", fontsize=8.5)

    # Panel B: Fraction of upward gradient contributed by top 1% error voxels
    ax_b = axes[1]
    top1_density = [summary_data["tail_statistics_summary"][o[0]]["density"]["frac_top1pct_median"] * 100.0 for o in OBJECTIVES]
    top1_mag = [summary_data["tail_statistics_summary"][o[0]]["magnetic"]["frac_top1pct_median"] * 100.0 for o in OBJECTIVES]

    ax_b.bar(x_pos - 0.2, top1_mag, width=0.4, label="Magnetic Top 1% Share", color="#50e3c2", alpha=0.85)
    ax_b.bar(x_pos + 0.2, top1_density, width=0.4, label="Density Top 1% Share", color="#f5a623", alpha=0.85)
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(obj_names_short, rotation=45, ha="right", fontsize=8)
    ax_b.set_title("Tail Concentration: % of Upward Gradient\nContributed by Top 1% Error Voxels (Extreme Horizons)", fontsize=10.5, fontweight="bold")
    ax_b.set_ylabel("Upward Gradient Share (%)", fontsize=9.5)
    ax_b.grid(True, linestyle=":", alpha=0.5)
    ax_b.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig3_path = OUTPUT_DIR / "figure3_extreme_error_supervision.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path}")

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
