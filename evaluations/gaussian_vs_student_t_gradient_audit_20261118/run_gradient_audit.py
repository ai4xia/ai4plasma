#!/usr/bin/env python3
"""Frozen-checkpoint gradient scale and alignment audit: Gaussian vs Student-t NLL.
Evaluates joint A+B gradient routing on fresh from-scratch 256-epoch model.
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
import torch
import torch.nn.functional as F

PROJECT = Path("/pscratch/sd/b/binxia/ai4plasma")
sys.path.insert(0, str(PROJECT))

from data.masking import (
    full_mse_loss,
    predicted_error_mse_loss,
    make_fixed_validation_mask,
    make_uncertain_input,
    GAUSSIAN_NLL_VARIANCE_FLOOR,
)
from data.vpic_hdf5_dataset import (
    VPICWindowDataset,
    load_fixed_split,
    parse_run_name,
    validate_checkpoint_split,
)
from models.unet3d import UNet3D
from visualize_mask_patterns_unet3d import select_validation_statistics_indices

OUTPUT = PROJECT / "evaluations/gaussian_vs_student_t_gradient_audit_20261118"
CKPT_PATH = PROJECT / "runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt"

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


def partition_parameters(model: UNet3D):
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    param_tensors = [p for _, p in named_params]

    groups = {
        "shared_unet": [],
        "encoder": [],
        "attention": [],
        "decoder": [],
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
            if any(name.startswith(p) for p in ["enc0.", "enc1.", "enc2.", "enc3."]):
                groups["encoder"].append(i)
            elif any(name.startswith(p) for p in ["attention_", "attn."]):
                groups["attention"].append(i)
            elif any(name.startswith(p) for p in ["mid.", "up0.", "up1.", "up2."]):
                groups["decoder"].append(i)
            else:
                raise ValueError(f"Uncategorized shared parameter: {name}")

    return named_params, param_tensors, groups


def compute_gradient_list(loss: torch.Tensor, param_tensors: list[torch.Tensor], retain: bool = True):
    raw = torch.autograd.grad(loss, param_tensors, retain_graph=retain, allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(raw, param_tensors)]


def subset_norm(grads: list[torch.Tensor], indices: list[int]) -> float:
    if not indices:
        return 0.0
    return float(torch.sqrt(sum((grads[i] ** 2).sum() for i in indices)).item())


def subset_cosine(g1: list[torch.Tensor], g2: list[torch.Tensor], indices: list[int]) -> float:
    if not indices:
        return float("nan")
    dot = sum((g1[i] * g2[i]).sum() for i in indices).item()
    n1 = math.sqrt(sum((g1[i] ** 2).sum().item() for i in indices))
    n2 = math.sqrt(sum((g2[i] ** 2).sum().item() for i in indices))
    if n1 > 0 and n2 > 0:
        return float(dot / (n1 * n2))
    return float("nan")


def max_additivity_error(g_full: list[torch.Tensor], g_a: list[torch.Tensor], g_b: list[torch.Tensor], indices: list[int]) -> float:
    norm_full = subset_norm(g_full, indices)
    if norm_full == 0.0:
        return 0.0
    max_err = max((g_full[i] - (g_a[i] + g_b[i])).abs().max().item() for i in indices)
    return float(max_err / norm_full)


def main():
    print("=" * 80)
    print("Gaussian vs Student-t NLL Gradient-Scale & Directional Audit")
    print("=" * 80, flush=True)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 1. Load Model & Validate Checkpoint
    ckpt, model, mean_norm, std_norm = load_model(CKPT_PATH, device)
    validate_checkpoint_split(ckpt)
    print(f"Loaded checkpoint: {CKPT_PATH.name} (epoch {ckpt['epoch']}, split {ckpt['split_sha256'][:10]}...)")

    named_params, param_tensors, param_groups = partition_parameters(model)
    print("Parameter groups verified:")
    for k, idxs in param_groups.items():
        total_p = sum(param_tensors[i].numel() for i in idxs)
        print(f"  {k:<15s}: {len(idxs)} tensors, {total_p:,} parameters")

    # 2. Select 8 Representative Paired Windows across Betas
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
    print(f"\nSelected {len(selected_windows)} paired windows from distinct simulation runs:")
    for idx, r, b in selected_windows:
        print(f"  idx={idx:6d} | beta={b:4s} | run={r}")

    # Forward hook to capture raw variance logits z
    captured_z = {}
    def variance_hook(module, inp, out):
        captured_z["z"] = out
    hook_handle = model.variance_out.register_forward_hook(variance_hook)

    # Data collection containers
    gradient_norm_rows = []
    gradient_cosine_rows = []
    tail_statistics_rows = []

    print("\nStarting gradient audit across 8 windows and 11 mask conditions (88 total)...", flush=True)

    for win_idx_num, (dataset_idx, run_name, beta_str) in enumerate(selected_windows):
        sample = dataset[dataset_idx]
        block = sample["block"].unsqueeze(0).to(device)
        target = ((block - mean_norm) / (std_norm + 1e-8)).float()

        print(f"Processing window {win_idx_num + 1}/8 (idx={dataset_idx}, beta={beta_str})...", flush=True)

        for mask_idx, (mask_name, mask_type, horizon) in enumerate(MASK_CONDITIONS):
            # Create deterministic noise
            noise_gen = torch.Generator(device=device).manual_seed(SEED + 1000 * win_idx_num + mask_idx)
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

            # Create mask
            if mask_type == "standard":
                mask, _ = make_fixed_validation_mask(mask_name, target.shape, device=device, dtype=target.dtype)
            else:
                prefix = 24 - horizon
                mask = torch.ones_like(target)
                mask[:, :, prefix:] = 0.0

            input_var = 1.0 - mask
            model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)

            # Forward pass (with grad for input/feature path)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, var = model(model_input, detach_variance_features=False)
            pred = pred.float()
            var = var.float()
            z_logits = captured_z["z"]

            # Losses
            l_mu = full_mse_loss(pred, target, mask)

            l_gauss_a = predicted_error_mse_loss(var.detach(), pred, target, "gaussian_nll", detach_mean=False)
            l_gauss_b = predicted_error_mse_loss(var, pred, target, "gaussian_nll", detach_mean=True)
            l_gauss = predicted_error_mse_loss(var, pred, target, "gaussian_nll", detach_mean=False)

            l_stud_a = predicted_error_mse_loss(var.detach(), pred, target, "student_t_nll", student_t_df=STUDENT_T_DF, detach_mean=False)
            l_stud_b = predicted_error_mse_loss(var, pred, target, "student_t_nll", student_t_df=STUDENT_T_DF, detach_mean=True)
            l_stud = predicted_error_mse_loss(var, pred, target, "student_t_nll", student_t_df=STUDENT_T_DF, detach_mean=False)

            # Gradients
            g_mu = compute_gradient_list(l_mu, param_tensors, retain=True)
            g_gauss_a = compute_gradient_list(l_gauss_a, param_tensors, retain=True)
            g_gauss_b = compute_gradient_list(l_gauss_b, param_tensors, retain=True)
            g_gauss = compute_gradient_list(l_gauss, param_tensors, retain=True)
            g_stud_a = compute_gradient_list(l_stud_a, param_tensors, retain=True)
            g_stud_b = compute_gradient_list(l_stud_b, param_tensors, retain=True)
            g_stud = compute_gradient_list(l_stud, param_tensors, retain=True)

            # Logit gradients w.r.t z (for tail analysis)
            gz_g = torch.autograd.grad(l_gauss, z_logits, retain_graph=True)[0].detach()
            gz_t = torch.autograd.grad(l_stud, z_logits, retain_graph=False)[0].detach()

            # Record gradient norms for all parameter groups
            for group_name, group_idxs in param_groups.items():
                n_mu = subset_norm(g_mu, group_idxs)
                n_g_a = subset_norm(g_gauss_a, group_idxs)
                n_g_b = subset_norm(g_gauss_b, group_idxs)
                n_g = subset_norm(g_gauss, group_idxs)
                n_t_a = subset_norm(g_stud_a, group_idxs)
                n_t_b = subset_norm(g_stud_b, group_idxs)
                n_t = subset_norm(g_stud, group_idxs)

                r_g = (n_g / n_mu) if n_mu > 0 else float("nan")
                r_t = (n_t / n_mu) if n_mu > 0 else float("nan")
                r_gt = (n_g / n_t) if n_t > 0 else float("nan")
                r_ab_g = (n_g_a / n_g_b) if n_g_b > 0 else float("nan")
                r_ab_t = (n_t_a / n_t_b) if n_t_b > 0 else float("nan")

                s_g = (n_mu / n_g) if n_g > 0 else float("nan")
                s_t = (n_mu / n_t) if n_t > 0 else float("nan")

                err_add_g = max_additivity_error(g_gauss, g_gauss_a, g_gauss_b, group_idxs)
                err_add_t = max_additivity_error(g_stud, g_stud_a, g_stud_b, group_idxs)

                gradient_norm_rows.append({
                    "window_idx": dataset_idx,
                    "run_name": run_name,
                    "beta": beta_str,
                    "mask_condition": mask_name,
                    "mask_type": mask_type,
                    "horizon": horizon if horizon is not None else "",
                    "param_group": group_name,
                    "norm_g_mu": n_mu,
                    "norm_g_G_A": n_g_a,
                    "norm_g_G_B": n_g_b,
                    "norm_g_G": n_g,
                    "norm_g_T_A": n_t_a,
                    "norm_g_T_B": n_t_b,
                    "norm_g_T": n_t,
                    "ratio_R_G": r_g,
                    "ratio_R_T": r_t,
                    "ratio_G_over_T": r_gt,
                    "ratio_path_A_over_B_G": r_ab_g,
                    "ratio_path_A_over_B_T": r_ab_t,
                    "scale_factor_s_G": s_g,
                    "scale_factor_s_T": s_t,
                    "additivity_error_G": err_add_g,
                    "additivity_error_T": err_add_t,
                })

            # Record cosine similarities for shared U-Net parameters
            shared_idxs = param_groups["shared_unet"]
            gradient_cosine_rows.append({
                "window_idx": dataset_idx,
                "run_name": run_name,
                "beta": beta_str,
                "mask_condition": mask_name,
                "mask_type": mask_type,
                "horizon": horizon if horizon is not None else "",
                "cos_g_mu_vs_g_G_A": subset_cosine(g_mu, g_gauss_a, shared_idxs),
                "cos_g_mu_vs_g_G_B": subset_cosine(g_mu, g_gauss_b, shared_idxs),
                "cos_g_mu_vs_g_G": subset_cosine(g_mu, g_gauss, shared_idxs),
                "cos_g_mu_vs_g_T_A": subset_cosine(g_mu, g_stud_a, shared_idxs),
                "cos_g_mu_vs_g_T_B": subset_cosine(g_mu, g_stud_b, shared_idxs),
                "cos_g_mu_vs_g_T": subset_cosine(g_mu, g_stud, shared_idxs),
                "cos_g_G_A_vs_g_G_B": subset_cosine(g_gauss_a, g_gauss_b, shared_idxs),
                "cos_g_T_A_vs_g_T_B": subset_cosine(g_stud_a, g_stud_b, shared_idxs),
                "cos_g_G_vs_g_T": subset_cosine(g_gauss, g_stud, shared_idxs),
            })

            # Tail error analysis for forecast horizons h in {18, 22, 23}
            if horizon in [18, 22, 23]:
                hidden_mask = (mask < 0.5)
                q = (pred - target) ** 2
                r = q / var

                for ch_group, ch_slice in [("Magnetic", slice(0, 3)), ("Density", slice(3, 4)), ("All", slice(0, 4))]:
                    ch_hidden = hidden_mask[:, ch_slice]
                    r_voxels = r[:, ch_slice][ch_hidden].detach().float().cpu().numpy()
                    gz_g_voxels = gz_g[:, ch_slice][ch_hidden].float().cpu().numpy()
                    gz_t_voxels = gz_t[:, ch_slice][ch_hidden].float().cpu().numpy()

                    # Theoretical log-u gradients
                    grad_logu_g = 0.5 - 0.5 * r_voxels
                    grad_logu_t = 0.5 - 2.0 * r_voxels / (1.0 + r_voxels)

                    # Upward variance gradients (negative gradient pulls logit UP)
                    upward_g = -np.minimum(gz_g_voxels, 0.0)
                    upward_t = -np.minimum(gz_t_voxels, 0.0)
                    total_upward_g = float(upward_g.sum())
                    total_upward_t = float(upward_t.sum())

                    # Top 1% threshold on r
                    p99_r = float(np.percentile(r_voxels, 99))
                    top1_mask = r_voxels >= p99_r
                    share_top1_g = float(upward_g[top1_mask].sum() / total_upward_g) if total_upward_g > 0 else 0.0
                    share_top1_t = float(upward_t[top1_mask].sum() / total_upward_t) if total_upward_t > 0 else 0.0

                    tail_statistics_rows.append({
                        "window_idx": dataset_idx,
                        "run_name": run_name,
                        "beta": beta_str,
                        "horizon": horizon,
                        "channel_group": ch_group,
                        "voxel_count": len(r_voxels),
                        "median_r": float(np.median(r_voxels)),
                        "p90_r": float(np.percentile(r_voxels, 90)),
                        "p99_r": p99_r,
                        "max_r": float(np.max(r_voxels)),
                        "frac_r_gt_1": float(np.mean(r_voxels > 1.0)),
                        "frac_r_gt_10": float(np.mean(r_voxels > 10.0)),
                        "frac_r_gt_100": float(np.mean(r_voxels > 100.0)),
                        "frac_r_gt_1000": float(np.mean(r_voxels > 1000.0)),
                        "theory_grad_logu_G_median": float(np.median(grad_logu_g)),
                        "theory_grad_logu_G_p10": float(np.percentile(grad_logu_g, 10)),
                        "theory_grad_logu_G_min": float(np.min(grad_logu_g)),
                        "theory_grad_logu_T_median": float(np.median(grad_logu_t)),
                        "theory_grad_logu_T_p10": float(np.percentile(grad_logu_t, 10)),
                        "theory_grad_logu_T_min": float(np.min(grad_logu_t)),
                        "logit_grad_G_median": float(np.median(gz_g_voxels)),
                        "logit_grad_G_p10": float(np.percentile(gz_g_voxels, 10)),
                        "logit_grad_G_min": float(np.min(gz_g_voxels)),
                        "logit_grad_T_median": float(np.median(gz_t_voxels)),
                        "logit_grad_T_p10": float(np.percentile(gz_t_voxels, 10)),
                        "logit_grad_T_min": float(np.min(gz_t_voxels)),
                        "top1pct_upward_grad_share_G": share_top1_g,
                        "top1pct_upward_grad_share_T": share_top1_t,
                    })

    hook_handle.remove()
    print("All gradient evaluations complete. Writing output CSVs...", flush=True)

    # 3. Write CSVs
    write_csv(OUTPUT / "gradient_norms.csv", gradient_norm_rows)
    print("Wrote gradient_norms.csv")

    write_csv(OUTPUT / "gradient_cosine_similarity.csv", gradient_cosine_rows)
    print("Wrote gradient_cosine_similarity.csv")

    write_csv(OUTPUT / "residual_tail_statistics.csv", tail_statistics_rows)
    print("Wrote residual_tail_statistics.csv")

    # 4. Normalization Factor Summary CSV
    # Aggregate across mask conditions and overall for shared_unet
    shared_norm_rows = [r for r in gradient_norm_rows if r["param_group"] == "shared_unet"]

    categories = [("All Mask Conditions", shared_norm_rows)]
    for mname, _, _ in MASK_CONDITIONS:
        categories.append((mname, [r for r in shared_norm_rows if r["mask_condition"] == mname]))

    norm_summary_rows = []
    for cat_name, rows in categories:
        s_g_vals = [r["scale_factor_s_G"] for r in rows if not math.isnan(r["scale_factor_s_G"])]
        s_t_vals = [r["scale_factor_s_T"] for r in rows if not math.isnan(r["scale_factor_s_T"])]
        r_g_vals = [r["ratio_R_G"] for r in rows if not math.isnan(r["ratio_R_G"])]
        r_t_vals = [r["ratio_R_T"] for r in rows if not math.isnan(r["ratio_R_T"])]
        gt_vals = [r["ratio_G_over_T"] for r in rows if not math.isnan(r["ratio_G_over_T"])]
        ab_g_vals = [r["ratio_path_A_over_B_G"] for r in rows if not math.isnan(r["ratio_path_A_over_B_G"])]
        ab_t_vals = [r["ratio_path_A_over_B_T"] for r in rows if not math.isnan(r["ratio_path_A_over_B_T"])]

        norm_summary_rows.append({
            "category": cat_name,
            "sample_count": len(rows),
            "median_s_G": float(np.median(s_g_vals)),
            "p16_s_G": float(np.percentile(s_g_vals, 16)),
            "p84_s_G": float(np.percentile(s_g_vals, 84)),
            "median_s_T": float(np.median(s_t_vals)),
            "p16_s_T": float(np.percentile(s_t_vals, 16)),
            "p84_s_T": float(np.percentile(s_t_vals, 84)),
            "median_R_G": float(np.median(r_g_vals)),
            "p16_R_G": float(np.percentile(r_g_vals, 16)),
            "p84_R_G": float(np.percentile(r_g_vals, 84)),
            "median_R_T": float(np.median(r_t_vals)),
            "p16_R_T": float(np.percentile(r_t_vals, 16)),
            "p84_R_T": float(np.percentile(r_t_vals, 84)),
            "median_G_over_T": float(np.median(gt_vals)),
            "p16_G_over_T": float(np.percentile(gt_vals, 16)),
            "p84_G_over_T": float(np.percentile(gt_vals, 84)),
            "median_path_A_over_B_G": float(np.median(ab_g_vals)),
            "median_path_A_over_B_T": float(np.median(ab_t_vals)),
        })

    write_csv(OUTPUT / "normalization_factor_summary.csv", norm_summary_rows)
    print("Wrote normalization_factor_summary.csv")

    # 5. Diagnostic Summary JSON
    # Overall summary metrics for shared_unet
    all_s_g = [r["scale_factor_s_G"] for r in shared_norm_rows]
    all_s_t = [r["scale_factor_s_T"] for r in shared_norm_rows]
    all_r_g = [r["ratio_R_G"] for r in shared_norm_rows]
    all_r_t = [r["ratio_R_T"] for r in shared_norm_rows]
    all_gt = [r["ratio_G_over_T"] for r in shared_norm_rows]
    all_ab_g = [r["ratio_path_A_over_B_G"] for r in shared_norm_rows]
    all_ab_t = [r["ratio_path_A_over_B_T"] for r in shared_norm_rows]

    # Overall cosine similarities
    all_cos_mu_ga = [r["cos_g_mu_vs_g_G_A"] for r in gradient_cosine_rows]
    all_cos_mu_gb = [r["cos_g_mu_vs_g_G_B"] for r in gradient_cosine_rows]
    all_cos_mu_g = [r["cos_g_mu_vs_g_G"] for r in gradient_cosine_rows]
    all_cos_mu_ta = [r["cos_g_mu_vs_g_T_A"] for r in gradient_cosine_rows]
    all_cos_mu_tb = [r["cos_g_mu_vs_g_T_B"] for r in gradient_cosine_rows]
    all_cos_mu_t = [r["cos_g_mu_vs_g_T"] for r in gradient_cosine_rows]
    all_cos_ga_gb = [r["cos_g_G_A_vs_g_G_B"] for r in gradient_cosine_rows]
    all_cos_ta_tb = [r["cos_g_T_A_vs_g_T_B"] for r in gradient_cosine_rows]
    all_cos_g_t = [r["cos_g_G_vs_g_T"] for r in gradient_cosine_rows]

    diagnostic_summary = {
        "checkpoint": str(CKPT_PATH),
        "windows_evaluated": len(selected_windows),
        "mask_conditions": len(MASK_CONDITIONS),
        "total_paired_evaluations": len(shared_norm_rows),
        "shared_unet_metrics": {
            "median_R_G": float(np.median(all_r_g)),
            "p16_R_G": float(np.percentile(all_r_g, 16)),
            "p84_R_G": float(np.percentile(all_r_g, 84)),
            "median_R_T": float(np.median(all_r_t)),
            "p16_R_T": float(np.percentile(all_r_t, 16)),
            "p84_R_T": float(np.percentile(all_r_t, 84)),
            "median_G_over_T": float(np.median(all_gt)),
            "p16_G_over_T": float(np.percentile(all_gt, 16)),
            "p84_G_over_T": float(np.percentile(all_gt, 84)),
            "median_s_G": float(np.median(all_s_g)),
            "median_s_T": float(np.median(all_s_t)),
            "median_path_A_over_B_G": float(np.median(all_ab_g)),
            "median_path_A_over_B_T": float(np.median(all_ab_t)),
        },
        "shared_unet_cosine_similarities": {
            "median_cos_g_mu_vs_g_G_A": float(np.median(all_cos_mu_ga)),
            "median_cos_g_mu_vs_g_G_B": float(np.median(all_cos_mu_gb)),
            "median_cos_g_mu_vs_g_G": float(np.median(all_cos_mu_g)),
            "median_cos_g_mu_vs_g_T_A": float(np.median(all_cos_mu_ta)),
            "median_cos_g_mu_vs_g_T_B": float(np.median(all_cos_mu_tb)),
            "median_cos_g_mu_vs_g_T": float(np.median(all_cos_mu_t)),
            "median_cos_g_G_A_vs_g_G_B": float(np.median(all_cos_ga_gb)),
            "median_cos_g_T_A_vs_g_T_B": float(np.median(all_cos_ta_tb)),
            "median_cos_g_G_vs_g_T": float(np.median(all_cos_g_t)),
        },
        "additivity_verification": {
            "max_error_G": float(max(r["additivity_error_G"] for r in shared_norm_rows)),
            "max_error_T": float(max(r["additivity_error_T"] for r in shared_norm_rows)),
        },
    }
    dump_json(OUTPUT / "diagnostic_summary.json", diagnostic_summary)
    print("Wrote diagnostic_summary.json")

    # 6. Generate Figures (at most three useful figures)
    print("\n--- Generating Comparison Figures ---", flush=True)

    # Figure 1: Gradient Norm Ratios Across Mask Conditions
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    mask_names = [m[0] for m in MASK_CONDITIONS]
    x_pos = np.arange(len(mask_names))

    # Panel A: R_G and R_T across masks on shared U-Net
    med_rg = [next(r["median_R_G"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]
    p16_rg = [next(r["p16_R_G"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]
    p84_rg = [next(r["p84_R_G"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]

    med_rt = [next(r["median_R_T"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]
    p16_rt = [next(r["p16_R_T"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]
    p84_rt = [next(r["p84_R_T"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]

    axes[0, 0].errorbar(x_pos - 0.15, med_rg, yerr=[np.array(med_rg) - np.array(p16_rg), np.array(p84_rg) - np.array(med_rg)],
                        fmt="o-", color="red", label=r"Gaussian $R_G = \|g_G\|/\|g_\mu\|$", capsize=4)
    axes[0, 0].errorbar(x_pos + 0.15, med_rt, yerr=[np.array(med_rt) - np.array(p16_rt), np.array(p84_rt) - np.array(med_rt)],
                        fmt="s-", color="blue", label=r"Student-t $R_T = \|g_T\|/\|g_\mu\|$", capsize=4)
    axes[0, 0].axhline(1.0, color="gray", ls="--", alpha=0.6, label="Parity (1.0)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(mask_names, rotation=45, ha="right", fontsize=8)
    axes[0, 0].set_ylabel(r"Gradient Norm Ratio to $\|g_\mu\|_2$ (Log Scale)")
    axes[0, 0].set_title(r"Shared U-Net Gradient Scale Ratios: $\|g_{\mathrm{NLL}}\| / \|g_\mu\|$")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=9)

    # Panel B: Path A vs Path B Ratio
    med_ab_g = [next(r["median_path_A_over_B_G"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]
    med_ab_t = [next(r["median_path_A_over_B_T"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]
    axes[0, 1].plot(x_pos, med_ab_g, "o-", color="red", label=r"Gaussian $\|g^{(A)}\| / \|g^{(B)}\|$")
    axes[0, 1].plot(x_pos, med_ab_t, "s-", color="blue", label=r"Student-t $\|g^{(A)}\| / \|g^{(B)}\|$")
    axes[0, 1].axhline(1.0, color="gray", ls="--", alpha=0.6, label="Equal Path Scale (1.0)")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(mask_names, rotation=45, ha="right", fontsize=8)
    axes[0, 1].set_ylabel(r"Path A / Path B Norm Ratio")
    axes[0, 1].set_title("Dominance of Path A (Mean) vs Path B (Variance)")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=9)

    # Panel C: Relative scale ||g_G|| / ||g_T||
    med_gt = [next(r["median_G_over_T"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]
    p16_gt = [next(r["p16_G_over_T"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]
    p84_gt = [next(r["p84_G_over_T"] for r in norm_summary_rows if r["category"] == m) for m in mask_names]
    axes[1, 0].errorbar(x_pos, med_gt, yerr=[np.array(med_gt) - np.array(p16_gt), np.array(p84_gt) - np.array(med_gt)],
                        fmt="d-", color="purple", label=r"$\|g_G\| / \|g_T\|$", capsize=4)
    axes[1, 0].axhline(1.0, color="gray", ls="--", alpha=0.6, label="Parity (1.0)")
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(mask_names, rotation=45, ha="right", fontsize=8)
    axes[1, 0].set_ylabel(r"Gaussian / Student-t Norm Ratio")
    axes[1, 0].set_title(r"Relative Likelihood Gradient Magnitude: $\|g_G\| / \|g_T\|$")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=9)

    # Panel D: Norm breakdown across sub-networks (Encoder, Attention, Decoder)
    subgroups = ["encoder", "attention", "decoder"]
    colors = ["teal", "orange", "darkgreen"]
    for sg, col in zip(subgroups, colors):
        rows_sg = [r for r in gradient_norm_rows if r["param_group"] == sg]
        med_rt_sg = [float(np.median([r["ratio_R_T"] for r in rows_sg if r["mask_condition"] == m])) for m in mask_names]
        axes[1, 1].plot(x_pos, med_rt_sg, "o-", color=col, label=f"Student-t {sg.capitalize()}")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(mask_names, rotation=45, ha="right", fontsize=8)
    axes[1, 1].set_ylabel(r"Ratio $\|g_T\| / \|g_\mu\|$")
    axes[1, 1].set_title("Student-t NLL / MSE Gradient Ratio Across Sub-networks")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=9)

    fig.suptitle("Gradient Norm Scale Audit: Gaussian vs Student-t NLL Across Mask Conditions", fontsize=14)
    plt.savefig(OUTPUT / "figure1_gradient_norm_ratios.png", dpi=200)
    plt.close()
    print("Saved figure1_gradient_norm_ratios.png")

    # Figure 2: Gradient Cosine Similarity
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    # Panel A: Alignment of Path A, Path B, and Joint with MSE
    cos_names = [
        ("Path A vs MSE", "cos_g_mu_vs_g_G_A", "cos_g_mu_vs_g_T_A"),
        ("Path B vs MSE", "cos_g_mu_vs_g_G_B", "cos_g_mu_vs_g_T_B"),
        ("Full A+B vs MSE", "cos_g_mu_vs_g_G", "cos_g_mu_vs_g_T"),
    ]
    pair_labels = [p[0] for p in cos_names]
    bar_pos = np.arange(len(pair_labels))
    width = 0.35

    g_cos_med = [float(np.median([r[p[1]] for r in gradient_cosine_rows])) for p in cos_names]
    t_cos_med = [float(np.median([r[p[2]] for r in gradient_cosine_rows])) for p in cos_names]
    g_cos_p16 = [float(np.percentile([r[p[1]] for r in gradient_cosine_rows], 16)) for p in cos_names]
    g_cos_p84 = [float(np.percentile([r[p[1]] for r in gradient_cosine_rows], 84)) for p in cos_names]
    t_cos_p16 = [float(np.percentile([r[p[2]] for r in gradient_cosine_rows], 16)) for p in cos_names]
    t_cos_p84 = [float(np.percentile([r[p[2]] for r in gradient_cosine_rows], 84)) for p in cos_names]

    axes[0].bar(bar_pos - width/2, g_cos_med, width, yerr=[np.array(g_cos_med)-np.array(g_cos_p16), np.array(g_cos_p84)-np.array(g_cos_med)],
                color="red", alpha=0.7, capsize=5, label="Gaussian NLL")
    axes[0].bar(bar_pos + width/2, t_cos_med, width, yerr=[np.array(t_cos_med)-np.array(t_cos_p16), np.array(t_cos_p84)-np.array(t_cos_med)],
                color="blue", alpha=0.7, capsize=5, label="Student-t NLL")
    axes[0].axhline(0.0, color="black", lw=1)
    axes[0].set_xticks(bar_pos)
    axes[0].set_xticklabels(pair_labels, fontsize=10)
    axes[0].set_ylabel("Cosine Similarity with Reconstruction MSE Gradient")
    axes[0].set_title(r"Directional Alignment with Reconstruction $\nabla_\theta L_\mu$")
    axes[0].set_ylim(-0.2, 0.8)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)

    # Panel B: Cosine alignment across mask conditions: g_G vs g_T and Path A vs Path B
    cos_gt_by_mask = [float(np.median([r["cos_g_G_vs_g_T"] for r in gradient_cosine_rows if r["mask_condition"] == m])) for m in mask_names]
    cos_ta_tb_by_mask = [float(np.median([r["cos_g_T_A_vs_g_T_B"] for r in gradient_cosine_rows if r["mask_condition"] == m])) for m in mask_names]
    cos_ga_gb_by_mask = [float(np.median([r["cos_g_G_A_vs_g_G_B"] for r in gradient_cosine_rows if r["mask_condition"] == m])) for m in mask_names]

    axes[1].plot(x_pos, cos_gt_by_mask, "d-", color="purple", lw=2, label=r"Cross-Likelihood $\cos(g_G, g_T)$")
    axes[1].plot(x_pos, cos_ta_tb_by_mask, "s--", color="blue", label=r"Student-t $\cos(g_T^{(A)}, g_T^{(B)})$")
    axes[1].plot(x_pos, cos_ga_gb_by_mask, "o--", color="red", label=r"Gaussian $\cos(g_G^{(A)}, g_G^{(B)})$")
    axes[1].axhline(0.0, color="gray", ls=":", alpha=0.5)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(mask_names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("Cross-Objective & Internal Path Orthogonality")
    axes[1].set_ylim(-0.3, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle("Gradient Directional Alignment: Reconstruction vs Uncertainty Pathways", fontsize=14)
    plt.savefig(OUTPUT / "figure2_gradient_cosine_similarity.png", dpi=200)
    plt.close()
    print("Saved figure2_gradient_cosine_similarity.png")

    # Figure 3: Gaussian vs Student-t Extreme Forecast Error Supervision
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    # Panel A: Theoretical log-u gradient response curve
    r_theoretical = np.logspace(-2, 4, 500)
    theo_grad_g = 0.5 - 0.5 * r_theoretical
    theo_grad_t = 0.5 - 2.0 * r_theoretical / (1.0 + r_theoretical)

    axes[0].plot(r_theoretical, -theo_grad_g, color="red", lw=2, label=r"Gaussian: $-\frac{\partial L_G}{\partial \log u} = \frac{r-1}{2}$ (Linear)")
    axes[0].plot(r_theoretical, -theo_grad_t, color="blue", lw=2, label=r"Student-t ($\nu=3$): $-\frac{\partial L_T}{\partial \log u} \to 1.5$ (Saturating)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel(r"Residual Ratio $r = (y - \mu)^2 / u$")
    axes[0].set_ylabel(r"Upward Variance Drive $-\frac{\partial L}{\partial \log u}$")
    axes[0].set_title(r"Theoretical Upward Variance Gradient vs Residual Error $r$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Panel B: Residual distribution at h=18, 22, 23 (Density & Magnetic)
    horizons = [18, 22, 23]
    h_colors = {"18": "teal", "22": "darkorange", "23": "crimson"}
    for h in horizons:
        h_rows = [r for r in tail_statistics_rows if r["horizon"] == h and r["channel_group"] == "All"]
        med_r = np.median([r["median_r"] for r in h_rows])
        p90_r = np.median([r["p90_r"] for r in h_rows])
        p99_r = np.median([r["p99_r"] for r in h_rows])
        max_r = np.median([r["max_r"] for r in h_rows])
        axes[1].plot([50, 90, 99, 100], [med_r, p90_r, p99_r, max_r], "o-", color=h_colors[str(h)], label=f"Horizon h={h}")

    axes[1].axhline(1.0, color="gray", ls="--", alpha=0.5, label="r = 1 (Calibrated)")
    axes[1].axhline(10.0, color="gray", ls=":", alpha=0.5, label="r = 10")
    axes[1].axhline(100.0, color="gray", ls="-.", alpha=0.5, label="r = 100")
    axes[1].set_yscale("log")
    axes[1].set_xticks([50, 90, 99, 100])
    axes[1].set_xticklabels(["Median (50%)", "p90", "p99", "Max"])
    axes[1].set_ylabel(r"Residual Ratio $r = (y - \mu)^2 / u$")
    axes[1].set_title(r"Empirical Forecast Error Distribution $r$ on Hidden Voxels")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    # Panel C: Top 1% Error Voxels Contribution to Total Upward Gradient
    share_g = [float(np.mean([r["top1pct_upward_grad_share_G"] for r in tail_statistics_rows if r["horizon"] == h and r["channel_group"] == "All"])) for h in horizons]
    share_t = [float(np.mean([r["top1pct_upward_grad_share_T"] for r in tail_statistics_rows if r["horizon"] == h and r["channel_group"] == "All"])) for h in horizons]
    h_pos = np.arange(len(horizons))
    w = 0.35
    axes[2].bar(h_pos - w/2, [s * 100 for s in share_g], w, color="red", alpha=0.7, label="Gaussian NLL")
    axes[2].bar(h_pos + w/2, [s * 100 for s in share_t], w, color="blue", alpha=0.7, label="Student-t NLL")
    axes[2].set_xticks(h_pos)
    axes[2].set_xticklabels([f"h={h}" for h in horizons])
    axes[2].set_ylabel("Share of Total Upward Variance Gradient (%)")
    axes[2].set_title("Gradient Concentration: Top 1% Extreme Error Voxels")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)

    fig.suptitle("Extreme Forecast Error Supervision: Gaussian (Outlier-Sensitive) vs Student-t (Saturating)", fontsize=14)
    plt.savefig(OUTPUT / "figure3_extreme_horizon_gradient_behavior.png", dpi=200)
    plt.close()
    print("Saved figure3_extreme_horizon_gradient_behavior.png")

    print("\n" + "=" * 80)
    print("Diagnostic execution complete! All CSVs and figures saved.")
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
