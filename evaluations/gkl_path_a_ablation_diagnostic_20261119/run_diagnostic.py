#!/usr/bin/env python3
"""Generalized KL Path A ON/OFF Gradient Ablation Diagnostic.

Investigates whether enabling Path A in generalized KL (GKL) uncertainty
supervision offers any scientific benefit or introduces severe gradient conflict /
reversal compared to Path A OFF (B-only).

Evaluates:
1. Primary: runs/multi_beta_gkl_continuation_20ep_20261119/final.pt (Epoch 276 GKL B-only)
2. Secondary: runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt (Epoch 256 Student-t A+B source)

Conditions:
8 held-out windows x 11 mask conditions (5 standard training + 6 extrapolation horizons)
= 88 paired conditions per checkpoint. Pass 0 only.
"""

import csv
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Tuple

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
)
from data.vpic_hdf5_dataset import (
    VPICWindowDataset,
    load_fixed_split,
    parse_run_name,
)
from visualize_mask_patterns_unet3d import select_validation_statistics_indices
from models.unet3d import UNet3D

OUTPUT_DIR = Path("evaluations/gkl_path_a_ablation_diagnostic_20261119")
PRIMARY_CKPT = Path("runs/multi_beta_gkl_continuation_20ep_20261119/final.pt")
SECONDARY_CKPT = Path("runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt")

SEED = 1234
LAMBDAS = [0.1, 0.25, 0.5, 1.0]

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


def dump_json(path: Path, v: Any) -> None:
    path.write_text(json.dumps(json_ready(v), indent=2, allow_nan=True))
    print(f"Saved: {path}")


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


def compute_gradient_list(loss: torch.Tensor, param_tensors: List[torch.Tensor], retain: bool = True) -> List[torch.Tensor]:
    raw = torch.autograd.grad(loss, param_tensors, retain_graph=retain, allow_unused=True)
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(raw, param_tensors)]


def subset_norm(grads: List[torch.Tensor], indices: List[int]) -> float:
    if not indices:
        return 0.0
    return float(torch.sqrt(sum((grads[i] ** 2).sum() for i in indices)).item())


def subset_dot(g1: List[torch.Tensor], g2: List[torch.Tensor], indices: List[int]) -> float:
    if not indices:
        return 0.0
    return float(sum((g1[i] * g2[i]).sum() for i in indices).item())


def subset_cosine(g1: List[torch.Tensor], g2: List[torch.Tensor], indices: List[int]) -> float:
    if not indices:
        return float("nan")
    dot = subset_dot(g1, g2, indices)
    n1 = subset_norm(g1, indices)
    n2 = subset_norm(g2, indices)
    if n1 > 0 and n2 > 0:
        return float(dot / (n1 * n2))
    return float("nan")


def gkl_loss_func(q: torch.Tensor, u: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    q_v = q[valid]
    u_v = u[valid]
    q_safe = torch.where(q_v > 0, q_v, torch.ones_like(q_v))
    loss_voxel = torch.where(q_v > 0, q_v * torch.log(q_safe), torch.zeros_like(q_v)) - q_v * torch.log(u_v) + u_v - q_v
    return loss_voxel.mean()


def run_synthetic_tests(device: torch.device) -> Dict[str, Any]:
    print("Running synthetic verification tests...")
    # 1. Zero-residual test
    mu = torch.tensor([0.0, 1e-25, 0.5, 1.0, 2.0], requires_grad=True, device=device, dtype=torch.float32)
    y = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], device=device, dtype=torch.float32)
    u = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], device=device, dtype=torch.float32)
    valid = torch.ones_like(mu, dtype=torch.bool)

    q = (mu - y) ** 2
    loss = gkl_loss_func(q, u, valid)
    loss.backward()
    grad_at_zero = mu.grad[0].item()
    is_zero_finite = math.isfinite(grad_at_zero) and abs(grad_at_zero) < 1e-12

    # 2. Analytical gradient match
    mu_test = torch.tensor([0.2, 0.8, 1.5, 3.0], requires_grad=True, device=device, dtype=torch.float32)
    y_test = torch.tensor([1.0, 0.5, 1.0, 1.0], device=device, dtype=torch.float32)
    u_test = torch.tensor([0.5, 1.2, 0.8, 2.0], requires_grad=True, device=device, dtype=torch.float32)
    valid_test = torch.ones_like(mu_test, dtype=torch.bool)

    q_test = (mu_test - y_test) ** 2
    loss_test = gkl_loss_func(q_test, u_test, valid_test)
    loss_test.backward()

    N = len(mu_test)
    analytic_grad_mu = (2 * (mu_test.detach() - y_test) * torch.log(q_test.detach() / u_test.detach())) / N
    analytic_grad_u = (1.0 - q_test.detach() / u_test.detach()) / N

    max_err_mu = (mu_test.grad - analytic_grad_mu).abs().max().item()
    max_err_u = (u_test.grad - analytic_grad_u).abs().max().item()

    # 3. Additivity test
    class SimpleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 4)
            self.mu_head = torch.nn.Linear(4, 1)
            self.var_head = torch.nn.Linear(4, 1)
        def forward(self, x):
            h = torch.relu(self.fc(x))
            return self.mu_head(h), F.softplus(self.var_head(h)) + 1e-6

    sm = SimpleModule().to(device)
    x = torch.randn(8, 4, device=device)
    y = torch.randn(8, 1, device=device)
    val = torch.ones_like(y, dtype=torch.bool)
    mu_s, u_s = sm(x)
    q_s = (mu_s - y) ** 2

    l_ab = gkl_loss_func(q_s, u_s, val)
    l_a = gkl_loss_func(q_s, u_s.detach(), val)
    l_b = gkl_loss_func(q_s.detach(), u_s, val)

    p_list = list(sm.parameters())
    g_ab = compute_gradient_list(l_ab, p_list, retain=True)
    g_a = compute_gradient_list(l_a, p_list, retain=True)
    g_b = compute_gradient_list(l_b, p_list, retain=True)

    diffs = [torch.norm(gab - (ga + gb)).item() for gab, ga, gb in zip(g_ab, g_a, g_b)]
    norms = [torch.norm(gab).item() for gab in g_ab]
    additivity_err = max(d / (n + 1e-12) for d, n in zip(diffs, norms))

    print(f"  Zero-residual grad: {grad_at_zero:.4e} (finite={is_zero_finite})")
    print(f"  Max analytic grad diff mu: {max_err_mu:.4e}")
    print(f"  Max analytic grad diff u: {max_err_u:.4e}")
    print(f"  Synthetic additivity residual: {additivity_err:.4e}")

    return {
        "zero_residual_grad": grad_at_zero,
        "zero_residual_finite": bool(is_zero_finite),
        "analytic_grad_diff_mu": max_err_mu,
        "analytic_grad_diff_u": max_err_u,
        "synthetic_additivity_err": additivity_err,
        "synthetic_tests_passed": bool(is_zero_finite and max_err_mu < 1e-5 and max_err_u < 1e-5 and additivity_err < 1e-5),
    }


def compute_quantiles(arr: np.ndarray, qs=[1, 5, 25, 50, 75, 95, 99]) -> Dict[str, float]:
    if len(arr) == 0:
        return {f"p{q:02d}": float("nan") for q in qs} | {"mean": float("nan")}
    res = {f"p{q:02d}": float(np.percentile(arr, q)) for q in qs}
    res["mean"] = float(np.mean(arr))
    return res


def main():
    print("=" * 80)
    print("Generalized KL Path A ON/OFF Gradient Ablation Diagnostic")
    print("=" * 80, flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Diagnostic requires a CUDA device on compute allocation")

    slurm_job_id = os.environ.get("SLURM_JOB_ID", "61582305")
    hostname = socket.gethostname()
    gpu_name = torch.cuda.get_device_name(0)
    free_mem, total_mem = torch.cuda.mem_get_info(0)
    total_mem_gb = total_mem / (1024 ** 3)
    free_mem_gb = free_mem / (1024 ** 3)

    print(f"Slurm Job ID: {slurm_job_id}")
    print(f"Compute Node: {hostname}")
    print(f"CUDA Device:  {gpu_name}")
    print(f"GPU Memory:   {free_mem_gb:.2f} GB free / {total_mem_gb:.2f} GB total")

    # Run Synthetic Tests
    synthetic_res = run_synthetic_tests(device)
    if not synthetic_res["synthetic_tests_passed"]:
        raise RuntimeError(f"Synthetic tests failed: {synthetic_res}")
    print("Synthetic tests PASSED cleanly!\n")

    # Select 8 Representative Windows
    split = load_fixed_split()
    h5_dir = "/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta_official2500_none_compat"
    if not os.path.exists(h5_dir):
        tmp_ckpt = torch.load(PRIMARY_CKPT, map_location="cpu", weights_only=False)
        h5_dir = tmp_ckpt["args"]["h5_dir"]
        del tmp_ckpt

    dataset = VPICWindowDataset(
        h5_dir=h5_dir,
        betas=["0.01", "0.02", "0.04", "0.06", "0.08", "0.1", "0.12", "0.15", "0.18", "0.2"],
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
    print(f"Selected {len(selected_windows)} paired windows across betas:")
    for idx, r, b in selected_windows:
        print(f"  idx={idx:6d} | beta={b:4s} | run={r}")

    checkpoints_to_eval = [
        ("gkl_epoch276", PRIMARY_CKPT),
        ("student_t_epoch256", SECONDARY_CKPT),
    ]

    # Result containers
    voxel_conflict_rows: List[dict] = []
    combined_mean_rows: List[dict] = []
    parameter_norm_rows: List[dict] = []
    cosine_sim_rows: List[dict] = []
    checkpoint_comparison_rows: List[dict] = []

    # Nonfinite event tracking
    nonfinite_events = {
        "gkl_epoch276": {"loss": 0, "g_mu": 0, "g_A": 0, "g_B": 0, "g_AB": 0, "voxel_grads": 0},
        "student_t_epoch256": {"loss": 0, "g_mu": 0, "g_A": 0, "g_B": 0, "g_AB": 0, "voxel_grads": 0},
    }

    # Paired lookup for comparison table
    # key: (window_idx, mask_condition) -> dict of metrics per ckpt
    paired_lookup: Dict[Tuple[int, str], Dict[str, dict]] = {}

    for ckpt_id, ckpt_path in checkpoints_to_eval:
        print("\n" + "=" * 60)
        print(f"Evaluating Checkpoint: {ckpt_id} ({ckpt_path.name})")
        print("=" * 60, flush=True)

        ckpt, model, mean_norm, std_norm = load_model(ckpt_path, device)
        named_params, param_tensors, param_groups = partition_parameters(model)
        shared_indices = param_groups["shared_unet"]

        print("Parameter Groups:")
        for grp_name, idxs in param_groups.items():
            cnt = sum(param_tensors[i].numel() for i in idxs)
            print(f"  {grp_name:<15s}: {len(idxs)} tensors, {cnt:,} params")

        t0_ckpt = time.time()

        for win_num, (dataset_idx, run_name, beta_str) in enumerate(selected_windows):
            sample = dataset[dataset_idx]
            block = sample["block"].unsqueeze(0).to(device)
            target = ((block - mean_norm) / (std_norm + 1e-8)).float()

            for mask_idx, (mask_name, mask_type, horizon) in enumerate(MASK_CONDITIONS):
                noise_gen = torch.Generator(device=device).manual_seed(SEED + 1000 * win_num + mask_idx)
                noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

                if mask_type == "standard":
                    mask, _ = make_fixed_validation_mask(mask_name, target.shape, device=device, dtype=target.dtype)
                else:
                    prefix = 24 - horizon
                    mask = torch.ones_like(target)
                    mask[:, :, prefix:] = 0.0

                input_var = 1.0 - mask
                model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)

                # Forward pass: Pass 0, Path B active (detach_variance_features=False)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred, var = model(model_input, detach_variance_features=False)
                pred = pred.float()
                var = var.float()

                valid = torch.isfinite(target)

                # Check forward finite
                if not (torch.isfinite(pred).all() and torch.isfinite(var).all()):
                    nonfinite_events[ckpt_id]["loss"] += 1

                # 1. Losses
                l_mu = full_mse_loss(pred, target, mask)

                q_ab = (pred - target) ** 2
                u_ab = var
                l_ab = gkl_loss_func(q_ab, u_ab, valid)

                q_a = (pred - target) ** 2
                u_a = var.detach()
                l_a = gkl_loss_func(q_a, u_a, valid)

                q_b = (pred.detach() - target) ** 2
                u_b = var
                l_b = gkl_loss_func(q_b, u_b, valid)

                # 2. Voxel-level gradients wrt pred
                g_mu_voxel = torch.autograd.grad(l_mu, pred, retain_graph=True)[0]
                g_A_voxel = torch.autograd.grad(l_a, pred, retain_graph=True)[0]

                if not (torch.isfinite(g_mu_voxel).all() and torch.isfinite(g_A_voxel).all()):
                    nonfinite_events[ckpt_id]["voxel_grads"] += 1

                # 3. Model parameter gradients
                g_mu = compute_gradient_list(l_mu, param_tensors, retain=True)
                g_A = compute_gradient_list(l_a, param_tensors, retain=True)
                g_B = compute_gradient_list(l_b, param_tensors, retain=True)
                g_AB = compute_gradient_list(l_ab, param_tensors, retain=True)

                # Check finite
                for name_g, g_vec in [("g_mu", g_mu), ("g_A", g_A), ("g_B", g_B), ("g_AB", g_AB)]:
                    if not all(torch.isfinite(p).all() for p in g_vec):
                        nonfinite_events[ckpt_id][name_g] += 1

                # Decomposition residual on shared backbone
                norm_AB_shared = subset_norm(g_AB, shared_indices)
                norm_A_shared = subset_norm(g_A, shared_indices)
                norm_B_shared = subset_norm(g_B, shared_indices)
                diff_shared = math.sqrt(sum((g_AB[i] - (g_A[i] + g_B[i])).square().sum().item() for i in shared_indices))
                decomp_res_shared = diff_shared / (norm_AB_shared + 1e-12)

                # Record parameter norms and ratios
                for grp_name, idxs in param_groups.items():
                    n_mu = subset_norm(g_mu, idxs)
                    n_A = subset_norm(g_A, idxs)
                    n_B = subset_norm(g_B, idxs)
                    n_AB = subset_norm(g_AB, idxs)
                    diff_grp = math.sqrt(sum((g_AB[i] - (g_A[i] + g_B[i])).square().sum().item() for i in idxs))
                    decomp_res_grp = diff_grp / (n_AB + 1e-12)

                    parameter_norm_rows.append({
                        "checkpoint_id": ckpt_id,
                        "window_idx": dataset_idx,
                        "run_name": run_name,
                        "beta": beta_str,
                        "mask_condition": mask_name,
                        "mask_type": mask_type,
                        "horizon": horizon if horizon is not None else "",
                        "param_group": grp_name,
                        "n_params": sum(param_tensors[i].numel() for i in idxs),
                        "norm_g_mu": n_mu,
                        "norm_g_A": n_A,
                        "norm_g_B": n_B,
                        "norm_g_AB": n_AB,
                        "ratio_A_mu": n_A / (n_mu + 1e-12),
                        "ratio_B_mu": n_B / (n_mu + 1e-12),
                        "ratio_AB_mu": n_AB / (n_mu + 1e-12),
                        "decomposition_residual": decomp_res_grp,
                    })

                # Cosine similarities on shared backbone (and other key groups)
                cos_A_mu = subset_cosine(g_A, g_mu, shared_indices)
                cos_B_mu = subset_cosine(g_B, g_mu, shared_indices)
                cos_A_B = subset_cosine(g_A, g_B, shared_indices)
                cos_AB_mu = subset_cosine(g_AB, g_mu, shared_indices)

                # Combined training gradients & projections for shared backbone
                proj_rows_for_paired = {}
                for lam in LAMBDAS:
                    # g_total_B = (g_mu + lam * g_B) / (1 + lam)
                    # dot with g_mu: (||g_mu||^2 + lam * (g_B . g_mu)) / (1 + lam)
                    dot_B_mu = subset_dot(g_B, g_mu, shared_indices)
                    dot_A_mu = subset_dot(g_A, g_mu, shared_indices)
                    dot_AB_mu = dot_A_mu + dot_B_mu

                    n_mu_sq = subset_norm(g_mu, shared_indices) ** 2
                    proj_B = ((n_mu_sq + lam * dot_B_mu) / (1.0 + lam)) / (n_mu_sq + 1e-12)
                    proj_AB = ((n_mu_sq + lam * dot_AB_mu) / (1.0 + lam)) / (n_mu_sq + 1e-12)

                    # Total vector norms & cosines
                    # ||g_total||^2 = (||g_mu||^2 + 2 lam (g_unc . g_mu) + lam^2 ||g_unc||^2) / (1+lam)^2
                    n_B_sq = subset_norm(g_B, shared_indices) ** 2
                    norm_total_B = math.sqrt(max(0.0, (n_mu_sq + 2 * lam * dot_B_mu + (lam ** 2) * n_B_sq))) / (1.0 + lam)
                    cos_total_B = ((n_mu_sq + lam * dot_B_mu) / (1.0 + lam)) / ((norm_total_B * math.sqrt(n_mu_sq)) + 1e-12)

                    n_AB_sq = subset_norm(g_AB, shared_indices) ** 2
                    norm_total_AB = math.sqrt(max(0.0, (n_mu_sq + 2 * lam * dot_AB_mu + (lam ** 2) * n_AB_sq))) / (1.0 + lam)
                    cos_total_AB = ((n_mu_sq + lam * dot_AB_mu) / (1.0 + lam)) / ((norm_total_AB * math.sqrt(n_mu_sq)) + 1e-12)

                    cosine_sim_rows.append({
                        "checkpoint_id": ckpt_id,
                        "window_idx": dataset_idx,
                        "run_name": run_name,
                        "beta": beta_str,
                        "mask_condition": mask_name,
                        "mask_type": mask_type,
                        "horizon": horizon if horizon is not None else "",
                        "param_group": "shared_unet",
                        "lambda_v": lam,
                        "cos_gA_gmu": cos_A_mu,
                        "cos_gB_gmu": cos_B_mu,
                        "cos_gA_gB": cos_A_B,
                        "cos_gAB_gmu": cos_AB_mu,
                        "projection_B": proj_B,
                        "projection_AB": proj_AB,
                        "cos_total_B_gmu": cos_total_B,
                        "cos_total_AB_gmu": cos_total_AB,
                    })
                    proj_rows_for_paired[lam] = {
                        "proj_B": proj_B,
                        "proj_AB": proj_AB,
                        "cos_total_B": cos_total_B,
                        "cos_total_AB": cos_total_AB,
                    }

                # 4. Voxel-Level Gradient Conflict Analysis
                # Detach tensors for voxel inspection
                q_tensor = ((pred.detach() - target.detach()) ** 2).cpu().numpy()
                u_tensor = var.detach().cpu().numpy()
                g_mu_v_np = g_mu_voxel.detach().cpu().numpy()
                g_A_v_np = g_A_voxel.detach().cpu().numpy()
                mask_np = mask.detach().cpu().numpy()
                valid_np = valid.detach().cpu().numpy()

                # Slices to report
                field_slices = [
                    ("all", slice(0, 4)),
                    ("magnetic", slice(0, 3)),
                    ("bx", slice(0, 1)),
                    ("by", slice(1, 2)),
                    ("bz", slice(2, 3)),
                    ("density", slice(3, 4)),
                ]
                regions = ["all", "observed", "hidden"]

                summary_for_paired_voxel = {}

                for f_name, f_sl in field_slices:
                    for r_name in regions:
                        sub_valid = valid_np[:, f_sl]
                        sub_mask = mask_np[:, f_sl]
                        if r_name == "observed":
                            sel = sub_valid & (sub_mask > 0.5)
                        elif r_name == "hidden":
                            sel = sub_valid & (sub_mask <= 0.5)
                        else:
                            sel = sub_valid

                        if not np.any(sel):
                            continue

                        q_sub = q_tensor[:, f_sl][sel]
                        u_sub = u_tensor[:, f_sl][sel]
                        gmu_sub = g_mu_v_np[:, f_sl][sel]
                        gA_sub = g_A_v_np[:, f_sl][sel]

                        n_vox = int(len(q_sub))
                        frac_lt = float(np.mean(q_sub < u_sub - 1e-12))
                        frac_eq = float(np.mean(np.abs(q_sub - u_sub) <= 1e-12))
                        frac_gt = float(np.mean(q_sub > u_sub + 1e-12))

                        nonzero = (np.abs(gmu_sub) > 1e-12) & (np.abs(gA_sub) > 1e-12)
                        if np.any(nonzero):
                            opp = (gmu_sub[nonzero] * gA_sub[nonzero] < 0)
                            frac_opp_nonzero = float(np.mean(opp))
                        else:
                            frac_opp_nonzero = 0.0

                        total_gA_abs = float(np.sum(np.abs(gA_sub)))
                        opp_mask_all = (gmu_sub * gA_sub < 0) & nonzero
                        if total_gA_abs > 0:
                            opp_mag_frac = float(np.sum(np.abs(gA_sub[opp_mask_all])) / total_gA_abs)
                        else:
                            opp_mag_frac = 0.0

                        ratio_qu = q_sub / (u_sub + 1e-12)
                        log_qu = np.log(np.maximum(ratio_qu, 1e-30))

                        qu_stats = compute_quantiles(ratio_qu, [1, 25, 50, 75, 99])
                        log_qu_stats = compute_quantiles(log_qu, [1, 25, 50, 75, 99])

                        voxel_conflict_rows.append({
                            "checkpoint_id": ckpt_id,
                            "window_idx": dataset_idx,
                            "run_name": run_name,
                            "beta": beta_str,
                            "mask_condition": mask_name,
                            "mask_type": mask_type,
                            "horizon": horizon if horizon is not None else "",
                            "field_group": f_name,
                            "region": r_name,
                            "n_voxels": n_vox,
                            "frac_q_lt_u": frac_lt,
                            "frac_q_eq_u": frac_eq,
                            "frac_q_gt_u": frac_gt,
                            "frac_opposing_nonzero": frac_opp_nonzero,
                            "opposing_magnitude_fraction": opp_mag_frac,
                            "q_over_u_p01": qu_stats["p01"],
                            "q_over_u_p25": qu_stats["p25"],
                            "q_over_u_p50": qu_stats["p50"],
                            "q_over_u_p75": qu_stats["p75"],
                            "q_over_u_p99": qu_stats["p99"],
                            "q_over_u_mean": qu_stats["mean"],
                            "log_q_over_u_p01": log_qu_stats["p01"],
                            "log_q_over_u_p25": log_qu_stats["p25"],
                            "log_q_over_u_p50": log_qu_stats["p50"],
                            "log_q_over_u_p75": log_qu_stats["p75"],
                            "log_q_over_u_p99": log_qu_stats["p99"],
                            "log_q_over_u_mean": log_qu_stats["mean"],
                        })

                        if f_name == "all" and r_name == "all":
                            summary_for_paired_voxel = {
                                "frac_opposing_nonzero": frac_opp_nonzero,
                                "opposing_magnitude_fraction": opp_mag_frac,
                                "frac_q_lt_u": frac_lt,
                                "frac_q_gt_u": frac_gt,
                                "median_q_over_u": qu_stats["p50"],
                            }

                # 5. Combined Mean-Gradient Reversals
                # Flattened over all valid voxels
                all_valid = valid_np
                gmu_all = g_mu_v_np[all_valid]
                gA_all = g_A_v_np[all_valid]
                nz_mu = np.abs(gmu_all) > 1e-12
                total_recon_mag = float(np.sum(np.abs(gmu_all[nz_mu])))
                norm_gmu = float(np.linalg.norm(gmu_all))

                summary_for_paired_reversal = {}

                for lam in LAMBDAS:
                    g_comb = (gmu_all + lam * gA_all) / (1.0 + lam)
                    is_reversed = (g_comb * gmu_all < 0) & nz_mu
                    frac_rev = float(np.mean(is_reversed[nz_mu])) if np.any(nz_mu) else 0.0

                    if total_recon_mag > 0:
                        mag_affected = float(np.sum(np.abs(gmu_all[is_reversed])) / total_recon_mag)
                    else:
                        mag_affected = 0.0

                    norm_comb = float(np.linalg.norm(g_comb))
                    if norm_comb > 0 and norm_gmu > 0:
                        cos_comb_recon = float(np.dot(g_comb, gmu_all) / (norm_comb * norm_gmu))
                    else:
                        cos_comb_recon = float("nan")

                    mult = g_comb[nz_mu] / gmu_all[nz_mu]
                    mult_stats = compute_quantiles(mult, [5, 25, 50, 75, 95])

                    combined_mean_rows.append({
                        "checkpoint_id": ckpt_id,
                        "window_idx": dataset_idx,
                        "run_name": run_name,
                        "beta": beta_str,
                        "mask_condition": mask_name,
                        "mask_type": mask_type,
                        "horizon": horizon if horizon is not None else "",
                        "lambda_v": lam,
                        "frac_voxels_reversed": frac_rev,
                        "reversal_recon_mag_frac": mag_affected,
                        "direct_mean_cosine_with_recon": cos_comb_recon,
                        "multiplier_mean": mult_stats["mean"],
                        "multiplier_p05": mult_stats["p05"],
                        "multiplier_p25": mult_stats["p25"],
                        "multiplier_p50": mult_stats["p50"],
                        "multiplier_p75": mult_stats["p75"],
                        "multiplier_p95": mult_stats["p95"],
                    })

                    summary_for_paired_reversal[lam] = {
                        "frac_rev": frac_rev,
                        "mag_affected": mag_affected,
                        "cos_comb_recon": cos_comb_recon,
                    }

                # Store paired metrics
                cond_key = (dataset_idx, mask_name)
                if cond_key not in paired_lookup:
                    paired_lookup[cond_key] = {}
                paired_lookup[cond_key][ckpt_id] = {
                    "beta": beta_str,
                    "mask_type": mask_type,
                    "horizon": horizon,
                    "norm_g_mu": subset_norm(g_mu, shared_indices),
                    "norm_g_A": norm_A_shared,
                    "norm_g_B": norm_B_shared,
                    "norm_g_AB": norm_AB_shared,
                    "ratio_A_mu": norm_A_shared / (subset_norm(g_mu, shared_indices) + 1e-12),
                    "ratio_B_mu": norm_B_shared / (subset_norm(g_mu, shared_indices) + 1e-12),
                    "ratio_AB_mu": norm_AB_shared / (subset_norm(g_mu, shared_indices) + 1e-12),
                    "cos_A_mu": cos_A_mu,
                    "cos_B_mu": cos_B_mu,
                    "cos_A_B": cos_A_B,
                    "cos_AB_mu": cos_AB_mu,
                    "decomp_res": decomp_res_shared,
                    "voxel": summary_for_paired_voxel,
                    "reversal": summary_for_paired_reversal,
                    "proj": proj_rows_for_paired,
                }

            print(f"  [{ckpt_id}] Window {win_num + 1}/8 (idx={dataset_idx}, beta={beta_str}) done.")

        print(f"Completed {ckpt_id} in {time.time() - t0_ckpt:.1f}s")
        # Clean up model to free GPU memory
        del model, ckpt
        torch.cuda.empty_cache()

    # Build Checkpoint Comparison Table
    for (dataset_idx, mask_name), ckpt_dict in paired_lookup.items():
        if "gkl_epoch276" in ckpt_dict and "student_t_epoch256" in ckpt_dict:
            gkl_data = ckpt_dict["gkl_epoch276"]
            st_data = ckpt_dict["student_t_epoch256"]
            checkpoint_comparison_rows.append({
                "window_idx": dataset_idx,
                "beta": gkl_data["beta"],
                "mask_condition": mask_name,
                "mask_type": gkl_data["mask_type"],
                "horizon": gkl_data["horizon"] if gkl_data["horizon"] is not None else "",
                # Voxel conflict metrics
                "gkl_opposing_frac": gkl_data["voxel"]["frac_opposing_nonzero"],
                "st_opposing_frac": st_data["voxel"]["frac_opposing_nonzero"],
                "gkl_opposing_mag_frac": gkl_data["voxel"]["opposing_magnitude_fraction"],
                "st_opposing_mag_frac": st_data["voxel"]["opposing_magnitude_fraction"],
                "gkl_median_q_over_u": gkl_data["voxel"]["median_q_over_u"],
                "st_median_q_over_u": st_data["voxel"]["median_q_over_u"],
                # Mean reversal at lambda=1.0
                "gkl_rev_frac_lam1": gkl_data["reversal"][1.0]["frac_rev"],
                "st_rev_frac_lam1": st_data["reversal"][1.0]["frac_rev"],
                "gkl_rev_mag_lam1": gkl_data["reversal"][1.0]["mag_affected"],
                "st_rev_mag_lam1": st_data["reversal"][1.0]["mag_affected"],
                "gkl_direct_mean_cos_lam1": gkl_data["reversal"][1.0]["cos_comb_recon"],
                "st_direct_mean_cos_lam1": st_data["reversal"][1.0]["cos_comb_recon"],
                # Backbone gradient norms & ratios
                "gkl_ratio_A_mu": gkl_data["ratio_A_mu"],
                "st_ratio_A_mu": st_data["ratio_A_mu"],
                "gkl_ratio_B_mu": gkl_data["ratio_B_mu"],
                "st_ratio_B_mu": st_data["ratio_B_mu"],
                "gkl_ratio_AB_mu": gkl_data["ratio_AB_mu"],
                "st_ratio_AB_mu": st_data["ratio_AB_mu"],
                # Cosine similarities
                "gkl_cos_A_mu": gkl_data["cos_A_mu"],
                "st_cos_A_mu": st_data["cos_A_mu"],
                "gkl_cos_B_mu": gkl_data["cos_B_mu"],
                "st_cos_B_mu": st_data["cos_B_mu"],
                "gkl_cos_AB_mu": gkl_data["cos_AB_mu"],
                "st_cos_AB_mu": st_data["cos_AB_mu"],
                # Projections at lambda=1.0
                "gkl_proj_B_lam1": gkl_data["proj"][1.0]["proj_B"],
                "gkl_proj_AB_lam1": gkl_data["proj"][1.0]["proj_AB"],
                "st_proj_B_lam1": st_data["proj"][1.0]["proj_B"],
                "st_proj_AB_lam1": st_data["proj"][1.0]["proj_AB"],
            })

    # Save CSV Deliverables
    write_csv(OUTPUT_DIR / "voxel_gradient_conflict.csv", voxel_conflict_rows)
    write_csv(OUTPUT_DIR / "combined_mean_gradient_metrics.csv", combined_mean_rows)
    write_csv(OUTPUT_DIR / "parameter_gradient_norms.csv", parameter_norm_rows)
    write_csv(OUTPUT_DIR / "gradient_cosine_similarity.csv", cosine_sim_rows)
    write_csv(OUTPUT_DIR / "checkpoint_comparison.csv", checkpoint_comparison_rows)

    # Compute High-Level Summary Statistics
    summary_stats = compute_summary_json(
        synthetic_res,
        voxel_conflict_rows,
        combined_mean_rows,
        parameter_norm_rows,
        cosine_sim_rows,
        checkpoint_comparison_rows,
        nonfinite_events,
        slurm_job_id,
        hostname,
        gpu_name,
        total_mem_gb,
    )
    dump_json(OUTPUT_DIR / "diagnostic_summary.json", summary_stats)

    # Generate Figures
    generate_figures(OUTPUT_DIR, checkpoint_comparison_rows, combined_mean_rows, cosine_sim_rows)

    print("\n" + "=" * 80)
    print("All diagnostic computations, tables, figures, and summaries completed successfully!")
    print("=" * 80, flush=True)


def compute_summary_json(
    synthetic_res,
    voxel_rows,
    combined_rows,
    norm_rows,
    cos_rows,
    comp_rows,
    nonfinite_events,
    job_id,
    hostname,
    gpu_name,
    gpu_mem,
) -> dict:
    # Filter helper
    def get_vals(rows, key, ckpt=None, mask_type=None, field=None, region=None, lam=None, group=None):
        res = []
        for r in rows:
            if ckpt and r.get("checkpoint_id") != ckpt:
                continue
            if mask_type and r.get("mask_type") != mask_type:
                continue
            if field and r.get("field_group") != field:
                continue
            if region and r.get("region") != region:
                continue
            if lam is not None and r.get("lambda_v") != lam:
                continue
            if group and r.get("param_group") != group:
                continue
            v = r.get(key)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                res.append(float(v))
        return res

    summary = {
        "metadata": {
            "slurm_job_id": job_id,
            "hostname": hostname,
            "cuda_device": gpu_name,
            "gpu_memory_gb": gpu_mem,
            "evaluation_pass": 0,
            "num_windows": 8,
            "num_conditions": 11,
            "total_paired_samples_per_ckpt": 88,
        },
        "synthetic_verification": synthetic_res,
        "nonfinite_events": nonfinite_events,
        "checkpoints": {},
    }

    for ckpt_id in ["gkl_epoch276", "student_t_epoch256"]:
        # Voxel metrics (all fields, all regions)
        opp_all = get_vals(voxel_rows, "frac_opposing_nonzero", ckpt=ckpt_id, field="all", region="all")
        opp_mag_all = get_vals(voxel_rows, "opposing_magnitude_fraction", ckpt=ckpt_id, field="all", region="all")
        q_lt_u_all = get_vals(voxel_rows, "frac_q_lt_u", ckpt=ckpt_id, field="all", region="all")
        q_gt_u_all = get_vals(voxel_rows, "frac_q_gt_u", ckpt=ckpt_id, field="all", region="all")
        med_qu_all = get_vals(voxel_rows, "q_over_u_p50", ckpt=ckpt_id, field="all", region="all")

        # Standard training masks vs extrapolation
        opp_std = get_vals(voxel_rows, "frac_opposing_nonzero", ckpt=ckpt_id, mask_type="standard", field="all", region="all")
        opp_mag_std = get_vals(voxel_rows, "opposing_magnitude_fraction", ckpt=ckpt_id, mask_type="standard", field="all", region="all")
        opp_ext = get_vals(voxel_rows, "frac_opposing_nonzero", ckpt=ckpt_id, mask_type="extrapolation", field="all", region="all")
        opp_mag_ext = get_vals(voxel_rows, "opposing_magnitude_fraction", ckpt=ckpt_id, mask_type="extrapolation", field="all", region="all")

        # Field specific (Magnetic vs Density)
        opp_mag_field = get_vals(voxel_rows, "frac_opposing_nonzero", ckpt=ckpt_id, field="magnetic", region="all")
        opp_dens_field = get_vals(voxel_rows, "frac_opposing_nonzero", ckpt=ckpt_id, field="density", region="all")

        # Region specific (Observed vs Hidden)
        opp_obs = get_vals(voxel_rows, "frac_opposing_nonzero", ckpt=ckpt_id, field="all", region="observed")
        opp_hid = get_vals(voxel_rows, "frac_opposing_nonzero", ckpt=ckpt_id, field="all", region="hidden")

        # Combined mean reversal at lambda=1.0
        rev_all_1 = get_vals(combined_rows, "frac_voxels_reversed", ckpt=ckpt_id, lam=1.0, field=None, region=None, group=None)
        rev_mag_1 = get_vals(combined_rows, "reversal_recon_mag_frac", ckpt=ckpt_id, lam=1.0, field=None, region=None, group=None)
        cos_mean_1 = get_vals(combined_rows, "direct_mean_cosine_with_recon", ckpt=ckpt_id, lam=1.0, field=None, region=None, group=None)

        # Parameter gradient norms on shared_unet
        rA = get_vals(norm_rows, "ratio_A_mu", ckpt=ckpt_id, group="shared_unet", field=None, region=None)
        rB = get_vals(norm_rows, "ratio_B_mu", ckpt=ckpt_id, group="shared_unet", field=None, region=None)
        rAB = get_vals(norm_rows, "ratio_AB_mu", ckpt=ckpt_id, group="shared_unet", field=None, region=None)
        decomp = get_vals(norm_rows, "decomposition_residual", ckpt=ckpt_id, group="shared_unet", field=None, region=None)

        # Cosine similarities on shared_unet
        cosA = get_vals(cos_rows, "cos_gA_gmu", ckpt=ckpt_id, group="shared_unet", field=None, region=None, lam=1.0)
        cosB = get_vals(cos_rows, "cos_gB_gmu", ckpt=ckpt_id, group="shared_unet", field=None, region=None, lam=1.0)
        cosAB = get_vals(cos_rows, "cos_gAB_gmu", ckpt=ckpt_id, group="shared_unet", field=None, region=None, lam=1.0)

        # Projections at lambda=1.0
        pB_1 = get_vals(cos_rows, "projection_B", ckpt=ckpt_id, group="shared_unet", field=None, region=None, lam=1.0)
        pAB_1 = get_vals(cos_rows, "projection_AB", ckpt=ckpt_id, group="shared_unet", field=None, region=None, lam=1.0)

        summary["checkpoints"][ckpt_id] = {
            "voxel_conflict": {
                "overall": {
                    "mean_opposing_fraction": float(np.mean(opp_all)),
                    "median_opposing_fraction": float(np.median(opp_all)),
                    "mean_opposing_magnitude_fraction": float(np.mean(opp_mag_all)),
                    "median_opposing_magnitude_fraction": float(np.median(opp_mag_all)),
                    "mean_frac_q_lt_u": float(np.mean(q_lt_u_all)),
                    "mean_frac_q_gt_u": float(np.mean(q_gt_u_all)),
                    "median_q_over_u": float(np.median(med_qu_all)),
                },
                "standard_masks": {
                    "mean_opposing_fraction": float(np.mean(opp_std)),
                    "mean_opposing_magnitude_fraction": float(np.mean(opp_mag_std)),
                },
                "extrapolation_horizons": {
                    "mean_opposing_fraction": float(np.mean(opp_ext)),
                    "mean_opposing_magnitude_fraction": float(np.mean(opp_mag_ext)),
                },
                "fields": {
                    "magnetic_mean_opposing_frac": float(np.mean(opp_mag_field)),
                    "density_mean_opposing_frac": float(np.mean(opp_dens_field)),
                },
                "regions": {
                    "observed_mean_opposing_frac": float(np.mean(opp_obs)),
                    "hidden_mean_opposing_frac": float(np.mean(opp_hid)),
                },
            },
            "combined_mean_reversal_lam1": {
                "mean_fraction_reversed": float(np.mean(rev_all_1)),
                "median_fraction_reversed": float(np.median(rev_all_1)),
                "mean_recon_magnitude_affected": float(np.mean(rev_mag_1)),
                "mean_cosine_with_recon": float(np.mean(cos_mean_1)),
            },
            "shared_backbone_parameter_gradients": {
                "ratio_A_mu_median": float(np.median(rA)),
                "ratio_B_mu_median": float(np.median(rB)),
                "ratio_AB_mu_median": float(np.median(rAB)),
                "cos_gA_gmu_mean": float(np.mean(cosA)),
                "cos_gB_gmu_mean": float(np.mean(cosB)),
                "cos_gAB_gmu_mean": float(np.mean(cosAB)),
                "projection_B_lam1_mean": float(np.mean(pB_1)),
                "projection_AB_lam1_mean": float(np.mean(pAB_1)),
                "max_decomposition_residual": float(np.max(decomp)),
            },
        }

    return summary


def generate_figures(output_dir: Path, comp_rows: List[dict], combined_rows: List[dict], cos_rows: List[dict]) -> None:
    print("\nGenerating Diagnostic Figures...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Path-A gradient conflict across masking conditions
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Opposing voxel fraction across masks
    mask_order = [m[0] for m in MASK_CONDITIONS]
    gkl_opp_by_mask = [float(np.mean([r["gkl_opposing_frac"] for r in comp_rows if r["mask_condition"] == m])) for m in mask_order]
    st_opp_by_mask = [float(np.mean([r["st_opposing_frac"] for r in comp_rows if r["mask_condition"] == m])) for m in mask_order]

    x = np.arange(len(mask_order))
    width = 0.35
    axes1[0].bar(x - width/2, [v * 100 for v in gkl_opp_by_mask], width, label="GKL Epoch 276", color="#2b5c8f")
    axes1[0].bar(x + width/2, [v * 100 for v in st_opp_by_mask], width, label="Student-t Epoch 256", color="#d95f02")
    axes1[0].set_ylabel("Opposing Nonzero Voxels (%)")
    axes1[0].set_title("(a) Voxel-Level Opposing Gradient Fraction")
    axes1[0].set_xticks(x)
    axes1[0].set_xticklabels([m.replace("temporal_extrapolation_", "") for m in mask_order], rotation=45, ha="right", fontsize=9)
    axes1[0].legend()
    axes1[0].set_ylim(0, 100)

    # Panel B: Opposing gradient magnitude fraction
    gkl_mag_by_mask = [float(np.mean([r["gkl_opposing_mag_frac"] for r in comp_rows if r["mask_condition"] == m])) for m in mask_order]
    st_mag_by_mask = [float(np.mean([r["st_opposing_mag_frac"] for r in comp_rows if r["mask_condition"] == m])) for m in mask_order]

    axes1[1].bar(x - width/2, [v * 100 for v in gkl_mag_by_mask], width, label="GKL Epoch 276", color="#2b5c8f")
    axes1[1].bar(x + width/2, [v * 100 for v in st_mag_by_mask], width, label="Student-t Epoch 256", color="#d95f02")
    axes1[1].set_ylabel("Opposing Magnitude Fraction (%)")
    axes1[1].set_title("(b) Opposing Path-A Gradient Magnitude")
    axes1[1].set_xticks(x)
    axes1[1].set_xticklabels([m.replace("temporal_extrapolation_", "") for m in mask_order], rotation=45, ha="right", fontsize=9)
    axes1[1].legend()
    axes1[1].set_ylim(0, 100)

    # Panel C: Median q/u ratio
    gkl_qu_by_mask = [float(np.median([r["gkl_median_q_over_u"] for r in comp_rows if r["mask_condition"] == m])) for m in mask_order]
    st_qu_by_mask = [float(np.median([r["st_median_q_over_u"] for r in comp_rows if r["mask_condition"] == m])) for m in mask_order]

    axes1[2].plot(x, gkl_qu_by_mask, "o-", label="GKL Epoch 276", color="#2b5c8f", lw=2)
    axes1[2].plot(x, st_qu_by_mask, "s--", label="Student-t Epoch 256", color="#d95f02", lw=2)
    axes1[2].axhline(1.0, color="black", linestyle=":", label="q = u (Equilibrium)")
    axes1[2].axhline(0.368, color="red", linestyle=":", label="Reversal threshold (lam=1)")
    axes1[2].set_ylabel("Median q / u Ratio")
    axes1[2].set_title("(c) Median q/u Ratio Across Conditions")
    axes1[2].set_xticks(x)
    axes1[2].set_xticklabels([m.replace("temporal_extrapolation_", "") for m in mask_order], rotation=45, ha="right", fontsize=9)
    axes1[2].set_yscale("log")
    axes1[2].legend(fontsize=8)

    plt.tight_layout()
    fig1_path = output_dir / "figure1_path_a_voxel_conflict.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Combined mean-gradient reversal versus lambda_v
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    lambdas = [0.1, 0.25, 0.5, 1.0]

    for ckpt_id, col, lbl in [("gkl_epoch276", "#2b5c8f", "GKL Epoch 276"), ("student_t_epoch256", "#d95f02", "Student-t Epoch 256")]:
        # Panel A: Voxel reversal fraction vs lambda
        std_rev = [float(np.mean([r["frac_voxels_reversed"] for r in combined_rows if r["checkpoint_id"] == ckpt_id and r["lambda_v"] == l and r["mask_type"] == "standard"])) * 100 for l in lambdas]
        ext_rev = [float(np.mean([r["frac_voxels_reversed"] for r in combined_rows if r["checkpoint_id"] == ckpt_id and r["lambda_v"] == l and r["mask_type"] == "extrapolation"])) * 100 for l in lambdas]

        axes2[0].plot(lambdas, std_rev, "o-", color=col, lw=2, label=f"{lbl} (Standard)")
        axes2[0].plot(lambdas, ext_rev, "s--", color=col, lw=2, alpha=0.7, label=f"{lbl} (Extrap)")

        # Panel B: Recon magnitude affected vs lambda
        std_mag = [float(np.mean([r["reversal_recon_mag_frac"] for r in combined_rows if r["checkpoint_id"] == ckpt_id and r["lambda_v"] == l and r["mask_type"] == "standard"])) * 100 for l in lambdas]
        ext_mag = [float(np.mean([r["reversal_recon_mag_frac"] for r in combined_rows if r["checkpoint_id"] == ckpt_id and r["lambda_v"] == l and r["mask_type"] == "extrapolation"])) * 100 for l in lambdas]

        axes2[1].plot(lambdas, std_mag, "o-", color=col, lw=2, label=f"{lbl} (Standard)")
        axes2[1].plot(lambdas, ext_mag, "s--", color=col, lw=2, alpha=0.7, label=f"{lbl} (Extrap)")

        # Panel C: Direct mean cosine alignment with recon MSE vs lambda
        std_cos = [float(np.mean([r["direct_mean_cosine_with_recon"] for r in combined_rows if r["checkpoint_id"] == ckpt_id and r["lambda_v"] == l and r["mask_type"] == "standard"])) for l in lambdas]
        ext_cos = [float(np.mean([r["direct_mean_cosine_with_recon"] for r in combined_rows if r["checkpoint_id"] == ckpt_id and r["lambda_v"] == l and r["mask_type"] == "extrapolation"])) for l in lambdas]

        axes2[2].plot(lambdas, std_cos, "o-", color=col, lw=2, label=f"{lbl} (Standard)")
        axes2[2].plot(lambdas, ext_cos, "s--", color=col, lw=2, alpha=0.7, label=f"{lbl} (Extrap)")

    axes2[0].set_xlabel("Uncertainty Loss Weight ($\\lambda_v$)")
    axes2[0].set_ylabel("Reversed Voxels (%)")
    axes2[0].set_title("(a) Combined Mean Gradient Reversal Fraction")
    axes2[0].legend(fontsize=8)
    axes2[0].set_ylim(0, 100)

    axes2[1].set_xlabel("Uncertainty Loss Weight ($\\lambda_v$)")
    axes2[1].set_ylabel("Recon Magnitude Affected (%)")
    axes2[1].set_title("(b) Reconstruction Gradient Magnitude Affected")
    axes2[1].legend(fontsize=8)
    axes2[1].set_ylim(0, 100)

    axes2[2].set_xlabel("Uncertainty Loss Weight ($\\lambda_v$)")
    axes2[2].set_ylabel("Cosine Similarity with Reconstruction MSE")
    axes2[2].set_title("(c) Aggregate Mean Gradient Alignment")
    axes2[2].legend(fontsize=8)
    axes2[2].set_ylim(-0.2, 1.05)

    plt.tight_layout()
    fig2_path = output_dir / "figure2_combined_reversal_vs_lambda.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    # Figure 3: Shared-backbone gradient alignment and magnitude for GKL B-only vs A+B
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Ratio of uncertainty gradient to recon gradient on shared backbone
    groups = ["shared_unet", "encoder", "attention", "decoder"]
    gkl_rB = [float(np.median([r["gkl_ratio_B_mu"] for r in comp_rows]))]  # B-only
    gkl_rAB = [float(np.median([r["gkl_ratio_AB_mu"] for r in comp_rows]))]  # A+B
    st_rB = [float(np.median([r["st_ratio_B_mu"] for r in comp_rows]))]
    st_rAB = [float(np.median([r["st_ratio_AB_mu"] for r in comp_rows]))]

    bars = [gkl_rB[0], gkl_rAB[0], st_rB[0], st_rAB[0]]
    bar_labels = ["GKL B-only", "GKL A+B", "Student-t B-only", "Student-t A+B"]
    bar_colors = ["#2b5c8f", "#7570b3", "#d95f02", "#e7298a"]
    axes3[0].bar(bar_labels, bars, color=bar_colors, width=0.5)
    axes3[0].set_ylabel("Norm Ratio to Recon MSE ($||g_{unc}|| / ||g_\\mu||$)")
    axes3[0].set_title("(a) Shared-Backbone Gradient Norm Ratios")
    axes3[0].tick_params(axis="x", rotation=25)
    for i, v in enumerate(bars):
        axes3[0].text(i, v + 0.05 * max(bars), f"{v:.2f}", ha="center", fontsize=9)

    # Panel B: Cosine similarity with recon MSE on shared backbone
    cos_b_gkl = float(np.mean([r["gkl_cos_B_mu"] for r in comp_rows]))
    cos_ab_gkl = float(np.mean([r["gkl_cos_AB_mu"] for r in comp_rows]))
    cos_b_st = float(np.mean([r["st_cos_B_mu"] for r in comp_rows]))
    cos_ab_st = float(np.mean([r["st_cos_AB_mu"] for r in comp_rows]))

    cos_bars = [cos_b_gkl, cos_ab_gkl, cos_b_st, cos_ab_st]
    axes3[1].bar(bar_labels, cos_bars, color=bar_colors, width=0.5)
    axes3[1].set_ylabel("Cosine Similarity $\\cos(g_{unc}, g_\\mu)$")
    axes3[1].set_title("(b) Shared-Backbone Directional Alignment")
    axes3[1].tick_params(axis="x", rotation=25)
    axes3[1].axhline(0.0, color="black", linestyle="--", lw=1)
    for i, v in enumerate(cos_bars):
        axes3[1].text(i, v + (0.02 if v >= 0 else -0.05), f"{v:+.3f}", ha="center", fontsize=9)

    # Panel C: First-order projection P onto g_mu vs lambda
    for ckpt_id, col, lbl in [("gkl_epoch276", "#2b5c8f", "GKL Epoch 276"), ("student_t_epoch256", "#d95f02", "Student-t Epoch 256")]:
        proj_B_curve = [float(np.mean([r["projection_B"] for r in cos_rows if r["checkpoint_id"] == ckpt_id and r["lambda_v"] == l and r["param_group"] == "shared_unet"])) for l in lambdas]
        proj_AB_curve = [float(np.mean([r["projection_AB"] for r in cos_rows if r["checkpoint_id"] == ckpt_id and r["lambda_v"] == l and r["param_group"] == "shared_unet"])) for l in lambdas]

        axes3[2].plot(lambdas, proj_B_curve, "o-", color=col, lw=2, label=f"{lbl} B-only")
        axes3[2].plot(lambdas, proj_AB_curve, "^--", color=col, lw=2, alpha=0.7, label=f"{lbl} A+B")

    axes3[2].set_xlabel("Uncertainty Loss Weight ($\\lambda_v$)")
    axes3[2].set_ylabel("Reconstruction Projection ($P = g_{total} \\cdot g_\\mu / ||g_\\mu||^2$)")
    axes3[2].set_title("(c) First-Order Reconstruction Projection")
    axes3[2].axhline(1.0, color="gray", linestyle=":", label="$P=1$ (Pure Recon)")
    axes3[2].legend(fontsize=8)

    plt.tight_layout()
    fig3_path = output_dir / "figure3_backbone_alignment.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"Saved: {fig3_path}")


if __name__ == "__main__":
    main()
