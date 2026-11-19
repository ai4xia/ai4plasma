#!/usr/bin/env python3
"""Gradient Normalization Benchmark for Joint Reconstruction and Uncertainty Learning.

Evaluates three gradient normalization methods:
- Method 1: Full shared-backbone parameter gradients (reference)
- Method 2: Last shared decoder layer parameter gradients (model.up0)
- Method 3: Shared feature tensor gradients (output of model.up0 consumed by heads)

Evaluates on 8 representative held-out windows x 11 mask conditions (88 paired evaluations)
for both Student-t NLL (df=3.0) and Gaussian NLL under joint A+B gradient routing.
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

OUTPUT_DIR = Path("evaluations/gradient_normalization_benchmark_20261119")
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
        "last_layer": [],
        "encoder": [],
        "decoder": [],
        "attention": [],
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
            if name.startswith("up0."):
                groups["last_layer"].append(i)
            if any(name.startswith(p) for p in ["enc0.", "enc1.", "enc2.", "enc3."]):
                groups["encoder"].append(i)
            elif any(name.startswith(p) for p in ["attention_", "attn."]):
                groups["attention"].append(i)
            elif any(name.startswith(p) for p in ["mid.", "up0.", "up1.", "up2."]):
                groups["decoder"].append(i)

    return named_params, param_tensors, groups


def compute_grad_norm(loss: torch.Tensor, param_list: List[torch.Tensor], retain: bool = True) -> float:
    raw = torch.autograd.grad(loss, param_list, retain_graph=retain, allow_unused=True)
    sq_sum = 0.0
    for g, p in zip(raw, param_list):
        if g is not None:
            sq_sum += (g.float() ** 2).sum().item()
    return math.sqrt(sq_sum)


def compute_feature_grad_norm(loss: torch.Tensor, feature_tensor: torch.Tensor, retain: bool = True) -> float:
    raw = torch.autograd.grad(loss, feature_tensor, retain_graph=retain, allow_unused=True)[0]
    if raw is None:
        return 0.0
    return float(torch.norm(raw.float()).item())


def main():
    print("=" * 80)
    print("Gradient Normalization Benchmark (Method 1 vs Method 2 vs Method 3)")
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
    last_layer_params = [param_tensors[i] for i in param_groups["last_layer"]]
    print(f"Shared U-Net params: {sum(p.numel() for p in shared_params):,}")
    print(f"Last layer (up0) params: {sum(p.numel() for p in last_layer_params):,}")

    # Forward hook on model.up0 to capture shared feature tensor F
    captured = {}
    def up0_hook(module, inp, out):
        captured["F"] = out
    handle_up0 = model.up0.register_forward_hook(up0_hook)

    # Dataset selection
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

    # Containers for benchmark data
    comparison_rows = []
    effective_ratio_rows = []

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

            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred, var = model(model_input, detach_variance_features=False)
            pred = pred.float()
            var = var.float()
            F_feat = captured["F"]

            # Losses
            l_mu = full_mse_loss(pred, target, mask)
            l_stud = predicted_error_mse_loss(var, pred, target, "student_t_nll", student_t_df=STUDENT_T_DF, detach_mean=False)
            l_gauss = predicted_error_mse_loss(var, pred, target, "gaussian_nll", detach_mean=False)

            # Feature gradients (Method 3)
            gF_mu_norm = compute_feature_grad_norm(l_mu, F_feat, retain=True)
            gF_stud_norm = compute_feature_grad_norm(l_stud, F_feat, retain=True)
            gF_gauss_norm = compute_feature_grad_norm(l_gauss, F_feat, retain=True)

            s_feature_stud = gF_mu_norm / (gF_stud_norm + 1e-12)
            s_feature_gauss = gF_mu_norm / (gF_gauss_norm + 1e-12)

            # Last layer gradients (Method 2)
            glast_mu_norm = compute_grad_norm(l_mu, last_layer_params, retain=True)
            glast_stud_norm = compute_grad_norm(l_stud, last_layer_params, retain=True)
            glast_gauss_norm = compute_grad_norm(l_gauss, last_layer_params, retain=True)

            s_last_stud = glast_mu_norm / (glast_stud_norm + 1e-12)
            s_last_gauss = glast_mu_norm / (glast_gauss_norm + 1e-12)

            # Full shared backbone gradients (Method 1)
            gfull_mu_norm = compute_grad_norm(l_mu, shared_params, retain=True)
            gfull_stud_norm = compute_grad_norm(l_stud, shared_params, retain=True)
            gfull_gauss_norm = compute_grad_norm(l_gauss, shared_params, retain=False)

            s_full_stud = gfull_mu_norm / (gfull_stud_norm + 1e-12)
            s_full_gauss = gfull_mu_norm / (gfull_gauss_norm + 1e-12)

            # Record comparison row for Student-t
            comparison_rows.append({
                "window_idx": dataset_idx,
                "run_name": run_name,
                "beta": beta_str,
                "mask_condition": mask_name,
                "mask_type": mask_type,
                "horizon": horizon if horizon is not None else "",
                "likelihood": "student_t",
                "norm_gF_mu": gF_mu_norm,
                "norm_gF_nll": gF_stud_norm,
                "norm_glast_mu": glast_mu_norm,
                "norm_glast_nll": glast_stud_norm,
                "norm_gfull_mu": gfull_mu_norm,
                "norm_gfull_nll": gfull_stud_norm,
                "s_full": s_full_stud,
                "s_last": s_last_stud,
                "s_feature": s_feature_stud,
                "ratio_s_last_over_s_full": s_last_stud / (s_full_stud + 1e-12),
                "ratio_s_feat_over_s_full": s_feature_stud / (s_full_stud + 1e-12),
            })

            # Record comparison row for Gaussian
            comparison_rows.append({
                "window_idx": dataset_idx,
                "run_name": run_name,
                "beta": beta_str,
                "mask_condition": mask_name,
                "mask_type": mask_type,
                "horizon": horizon if horizon is not None else "",
                "likelihood": "gaussian",
                "norm_gF_mu": gF_mu_norm,
                "norm_gF_nll": gF_gauss_norm,
                "norm_glast_mu": glast_mu_norm,
                "norm_glast_nll": glast_gauss_norm,
                "norm_gfull_mu": gfull_mu_norm,
                "norm_gfull_nll": gfull_gauss_norm,
                "s_full": s_full_gauss,
                "s_last": s_last_gauss,
                "s_feature": s_feature_gauss,
                "ratio_s_last_over_s_full": s_last_gauss / (s_full_gauss + 1e-12),
                "ratio_s_feat_over_s_full": s_feature_gauss / (s_full_gauss + 1e-12),
            })

            # Effective gradient ratio R_effective = s_approx * ||g_full_nll|| / ||g_full_mu|| = s_approx / s_full
            # At lambda_v = 1.0, ideal is 1.0.
            R_eff_full_stud = 1.0
            R_eff_last_stud = s_last_stud / (s_full_stud + 1e-12)
            R_eff_feat_stud = s_feature_stud / (s_full_stud + 1e-12)

            R_eff_full_gauss = 1.0
            R_eff_last_gauss = s_last_gauss / (s_full_gauss + 1e-12)
            R_eff_feat_gauss = s_feature_gauss / (s_full_gauss + 1e-12)

            effective_ratio_rows.append({
                "window_idx": dataset_idx,
                "run_name": run_name,
                "beta": beta_str,
                "mask_condition": mask_name,
                "mask_type": mask_type,
                "horizon": horizon if horizon is not None else "",
                "likelihood": "student_t",
                "R_eff_method1_full": R_eff_full_stud,
                "R_eff_method2_last": R_eff_last_stud,
                "R_eff_method3_feat": R_eff_feat_stud,
            })
            effective_ratio_rows.append({
                "window_idx": dataset_idx,
                "run_name": run_name,
                "beta": beta_str,
                "mask_condition": mask_name,
                "mask_type": mask_type,
                "horizon": horizon if horizon is not None else "",
                "likelihood": "gaussian",
                "R_eff_method1_full": R_eff_full_gauss,
                "R_eff_method2_last": R_eff_last_gauss,
                "R_eff_method3_feat": R_eff_feat_gauss,
            })

    handle_up0.remove()
    eval_time = time.time() - t0_eval
    print(f"\nCompleted 88 paired evaluations in {eval_time:.1f}s.")

    # Save CSVs
    write_csv(OUTPUT_DIR / "normalization_method_comparison.csv", comparison_rows)
    write_csv(OUTPUT_DIR / "effective_gradient_ratios.csv", effective_ratio_rows)

    # -------------------------------------------------------------
    # 2. Computational Overhead Benchmark
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Running Computational Overhead Benchmark on GPU...")
    print("=" * 80, flush=True)

    # Test batch sizes: 1 and 4 (production per-GPU batch size)
    overhead_results = []

    for bs in [1, 4]:
        print(f"\nBenchmarking batch size B = {bs}...")
        dummy_input = torch.randn(bs, 8, 24, 64, 64, device=device, dtype=torch.float32)
        dummy_target = torch.randn(bs, 4, 24, 64, 64, device=device, dtype=torch.float32)
        dummy_mask = torch.ones(bs, 4, 24, 64, 64, device=device, dtype=torch.float32)

        # Warmup iterations
        for _ in range(5):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p, v = model(dummy_input, detach_variance_features=False)
            l = full_mse_loss(p.float(), dummy_target, dummy_mask) + predicted_error_mse_loss(v.float(), p.float(), dummy_target, "student_t_nll", 3.0, detach_mean=False)
            l.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        n_iters = 20

        # Benchmark Baseline: Unnormalized A+B joint loss forward + backward
        torch.cuda.reset_peak_memory_stats()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]

        for i in range(n_iters):
            start_events[i].record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p, v = model(dummy_input, detach_variance_features=False)
            l_mu = full_mse_loss(p.float(), dummy_target, dummy_mask)
            l_nll = predicted_error_mse_loss(v.float(), p.float(), dummy_target, "student_t_nll", 3.0, detach_mean=False)
            l_tot = 0.5 * (l_mu + l_nll)
            l_tot.backward()
            end_events[i].record()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times_baseline = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        t_base_mean = float(np.mean(times_baseline))
        mem_base_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Benchmark Method 1: Full shared-backbone gradients (two full backward passes)
        torch.cuda.reset_peak_memory_stats()
        for i in range(n_iters):
            start_events[i].record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p, v = model(dummy_input, detach_variance_features=False)
            l_mu = full_mse_loss(p.float(), dummy_target, dummy_mask)
            l_nll = predicted_error_mse_loss(v.float(), p.float(), dummy_target, "student_t_nll", 3.0, detach_mean=False)
            # Separate full backward passes for norm
            g_mu = torch.autograd.grad(l_mu, shared_params, retain_graph=True, allow_unused=True)
            n_mu = torch.sqrt(sum((g.float()**2).sum() for g in g_mu if g is not None))
            g_nll = torch.autograd.grad(l_nll, shared_params, retain_graph=True, allow_unused=True)
            n_nll = torch.sqrt(sum((g.float()**2).sum() for g in g_nll if g is not None))
            s_full = (n_mu / (n_nll + 1e-12)).detach()
            # Combined backward pass
            l_tot = (l_mu + s_full * l_nll) / (1.0 + 1.0)
            l_tot.backward()
            end_events[i].record()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times_m1 = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        t_m1_mean = float(np.mean(times_m1))
        mem_m1_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Benchmark Method 2: Last layer parameters (model.up0)
        torch.cuda.reset_peak_memory_stats()
        for i in range(n_iters):
            start_events[i].record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p, v = model(dummy_input, detach_variance_features=False)
            l_mu = full_mse_loss(p.float(), dummy_target, dummy_mask)
            l_nll = predicted_error_mse_loss(v.float(), p.float(), dummy_target, "student_t_nll", 3.0, detach_mean=False)
            # Backward to last layer
            g_mu_last = torch.autograd.grad(l_mu, last_layer_params, retain_graph=True, allow_unused=True)
            n_mu_last = torch.sqrt(sum((g.float()**2).sum() for g in g_mu_last if g is not None))
            g_nll_last = torch.autograd.grad(l_nll, last_layer_params, retain_graph=True, allow_unused=True)
            n_nll_last = torch.sqrt(sum((g.float()**2).sum() for g in g_nll_last if g is not None))
            s_last = (n_mu_last / (n_nll_last + 1e-12)).detach()
            # Combined backward pass
            l_tot = (l_mu + s_last * l_nll) / (1.0 + 1.0)
            l_tot.backward()
            end_events[i].record()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times_m2 = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        t_m2_mean = float(np.mean(times_m2))
        mem_m2_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Benchmark Method 3: Feature gradients (model.up0 output tensor F)
        captured_bench = {}
        def bench_hook(module, inp, out):
            captured_bench["F"] = out
        h_bench = model.up0.register_forward_hook(bench_hook)

        torch.cuda.reset_peak_memory_stats()
        for i in range(n_iters):
            start_events[i].record()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                p, v = model(dummy_input, detach_variance_features=False)
            F_tensor = captured_bench["F"]
            l_mu = full_mse_loss(p.float(), dummy_target, dummy_mask)
            l_nll = predicted_error_mse_loss(v.float(), p.float(), dummy_target, "student_t_nll", 3.0, detach_mean=False)
            # Feature gradients: only backprop through output heads (100 params)
            gF_mu = torch.autograd.grad(l_mu, F_tensor, retain_graph=True)[0]
            n_mu_feat = torch.norm(gF_mu.float())
            gF_nll = torch.autograd.grad(l_nll, F_tensor, retain_graph=True)[0]
            n_nll_feat = torch.norm(gF_nll.float())
            s_feat = (n_mu_feat / (n_nll_feat + 1e-12)).detach()
            # Combined backward pass
            l_tot = (l_mu + s_feat * l_nll) / (1.0 + 1.0)
            l_tot.backward()
            end_events[i].record()
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times_m3 = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
        t_m3_mean = float(np.mean(times_m3))
        mem_m3_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        h_bench.remove()

        print(f"Results for B = {bs}:")
        print(f"  Baseline (unnormalized):     {t_base_mean:6.2f} ms | Peak Mem: {mem_base_mb:6.1f} MB")
        print(f"  Method 1 (Full Backbone):    {t_m1_mean:6.2f} ms | +{(t_m1_mean/t_base_mean - 1)*100:5.1f}% | Peak Mem: {mem_m1_mb:6.1f} MB")
        print(f"  Method 2 (Last Layer up0):   {t_m2_mean:6.2f} ms | +{(t_m2_mean/t_base_mean - 1)*100:5.1f}% | Peak Mem: {mem_m2_mb:6.1f} MB")
        print(f"  Method 3 (Shared Feature F): {t_m3_mean:6.2f} ms | +{(t_m3_mean/t_base_mean - 1)*100:5.1f}% | Peak Mem: {mem_m3_mb:6.1f} MB")

        overhead_results.append({
            "batch_size": bs,
            "method": "Baseline (Unnormalized)",
            "wall_time_ms": t_base_mean,
            "delta_time_ms": 0.0,
            "overhead_percent": 0.0,
            "peak_memory_mb": mem_base_mb,
            "memory_overhead_percent": 0.0,
        })
        overhead_results.append({
            "batch_size": bs,
            "method": "Method 1: Full Shared-Backbone",
            "wall_time_ms": t_m1_mean,
            "delta_time_ms": t_m1_mean - t_base_mean,
            "overhead_percent": (t_m1_mean / t_base_mean - 1.0) * 100.0,
            "peak_memory_mb": mem_m1_mb,
            "memory_overhead_percent": (mem_m1_mb / mem_base_mb - 1.0) * 100.0,
        })
        overhead_results.append({
            "batch_size": bs,
            "method": "Method 2: Last Layer (up0)",
            "wall_time_ms": t_m2_mean,
            "delta_time_ms": t_m2_mean - t_base_mean,
            "overhead_percent": (t_m2_mean / t_base_mean - 1.0) * 100.0,
            "peak_memory_mb": mem_m2_mb,
            "memory_overhead_percent": (mem_m2_mb / mem_base_mb - 1.0) * 100.0,
        })
        overhead_results.append({
            "batch_size": bs,
            "method": "Method 3: Shared Feature (F)",
            "wall_time_ms": t_m3_mean,
            "delta_time_ms": t_m3_mean - t_base_mean,
            "overhead_percent": (t_m3_mean / t_base_mean - 1.0) * 100.0,
            "peak_memory_mb": mem_m3_mb,
            "memory_overhead_percent": (mem_m3_mb / mem_base_mb - 1.0) * 100.0,
        })

    write_csv(OUTPUT_DIR / "computational_overhead.csv", overhead_results)

    # -------------------------------------------------------------
    # 3. Statistical Analysis and Summary JSON
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Computing Statistical Summary...")
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

    summary_data = {
        "checkpoint": str(CKPT_PATH),
        "total_paired_samples": 88,
        "student_t": {
            "all_conditions": {},
            "standard_masks_only": {},
            "extrapolation_only": {},
        },
        "gaussian": {
            "all_conditions": {},
            "standard_masks_only": {},
            "extrapolation_only": {},
        },
        "computational_overhead": overhead_results,
    }

    for l_key, l_name in [("student_t", "student_t"), ("gaussian", "gaussian")]:
        rows_all = [r for r in comparison_rows if r["likelihood"] == l_name]
        rows_std = [r for r in rows_all if r["mask_type"] == "standard"]
        rows_ext = [r for r in rows_all if r["mask_type"] == "extrapolation"]

        for subset_name, subset_rows in [
            ("all_conditions", rows_all),
            ("standard_masks_only", rows_std),
            ("extrapolation_only", rows_ext),
        ]:
            s_full_vals = [r["s_full"] for r in subset_rows]
            s_last_vals = [r["s_last"] for r in subset_rows]
            s_feat_vals = [r["s_feature"] for r in subset_rows]
            ratio_last = [r["ratio_s_last_over_s_full"] for r in subset_rows]
            ratio_feat = [r["ratio_s_feat_over_s_full"] for r in subset_rows]

            # In effective gradient ratio, Method 1 is 1.0 by definition.
            # Method 2 has R_eff = s_last / s_full.
            # Method 3 has R_eff = s_feature / s_full.
            # Calculate fraction within factor of 2: [0.5, 2.0]
            # Calculate fraction within factor of 5: [0.2, 5.0]
            def fractions_within(r_vals, target=1.0):
                arr = np.array(r_vals)
                f2 = float(np.mean((arr >= target / 2.0) & (arr <= target * 2.0)))
                f5 = float(np.mean((arr >= target / 5.0) & (arr <= target * 5.0)))
                return f2, f5

            f2_last, f5_last = fractions_within(ratio_last)
            f2_feat, f5_feat = fractions_within(ratio_feat)

            # Compute empirical multiplicative calibration factor c = median(s_full / s_feat)
            c_calib = float(np.median([1.0 / (rf + 1e-12) for rf in ratio_feat]))
            calib_ratio = [rf * c_calib for rf in ratio_feat]
            f2_calib, f5_calib = fractions_within(calib_ratio)

            summary_data[l_key][subset_name] = {
                "count": len(subset_rows),
                "s_full": summarize_series(s_full_vals),
                "s_last": summarize_series(s_last_vals),
                "s_feature": summarize_series(s_feat_vals),
                "s_last_over_s_full": summarize_series(ratio_last),
                "s_feat_over_s_full": summarize_series(ratio_feat),
                "effective_ratio_method2_last": {
                    "stats": summarize_series(ratio_last),
                    "fraction_within_factor_2": f2_last,
                    "fraction_within_factor_5": f5_last,
                },
                "effective_ratio_method3_feat_raw": {
                    "stats": summarize_series(ratio_feat),
                    "fraction_within_factor_2": f2_feat,
                    "fraction_within_factor_5": f5_feat,
                },
                "calibrated_feature_method": {
                    "empirical_c": c_calib,
                    "stats": summarize_series(calib_ratio),
                    "fraction_within_factor_2": f2_calib,
                    "fraction_within_factor_5": f5_calib,
                },
            }

    # Extreme extrapolation behavior (h22, h23)
    for l_key, l_name in [("student_t", "student_t"), ("gaussian", "gaussian")]:
        h22_rows = [r for r in comparison_rows if r["likelihood"] == l_name and r["horizon"] == 22]
        h23_rows = [r for r in comparison_rows if r["likelihood"] == l_name and r["horizon"] == 23]
        summary_data[l_key]["extreme_horizons"] = {
            "h22": {
                "s_full": summarize_series([r["s_full"] for r in h22_rows]),
                "s_last_over_s_full": summarize_series([r["ratio_s_last_over_s_full"] for r in h22_rows]),
                "s_feat_over_s_full": summarize_series([r["ratio_s_feat_over_s_full"] for r in h22_rows]),
            },
            "h23": {
                "s_full": summarize_series([r["s_full"] for r in h23_rows]),
                "s_last_over_s_full": summarize_series([r["ratio_s_last_over_s_full"] for r in h23_rows]),
                "s_feat_over_s_full": summarize_series([r["ratio_s_feat_over_s_full"] for r in h23_rows]),
            },
        }

    summary_path = OUTPUT_DIR / "benchmark_summary.json"
    summary_path.write_text(json.dumps(json_ready(summary_data), indent=2))
    print(f"Saved: {summary_path}")

    # -------------------------------------------------------------
    # 4. Figures (at most 2 figures)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Generating Benchmark Figures...")
    print("=" * 80, flush=True)

    # Figure 1: Effective Gradient Ratios across Methods and Likelihoods
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax_idx, (l_key, l_label) in enumerate([("student_t", "Student-t (df=3)"), ("gaussian", "Gaussian")]):
        ax = axes[ax_idx]
        rows_l = [r for r in comparison_rows if r["likelihood"] == l_key]

        # Extract data for standard vs extrapolation
        r_last_std = [r["ratio_s_last_over_s_full"] for r in rows_l if r["mask_type"] == "standard"]
        r_last_ext = [r["ratio_s_last_over_s_full"] for r in rows_l if r["mask_type"] == "extrapolation"]
        r_feat_std = [r["ratio_s_feat_over_s_full"] for r in rows_l if r["mask_type"] == "standard"]
        r_feat_ext = [r["ratio_s_feat_over_s_full"] for r in rows_l if r["mask_type"] == "extrapolation"]

        c_calib = summary_data[l_key]["all_conditions"]["calibrated_feature_method"]["empirical_c"]
        r_calib_std = [rf * c_calib for rf in r_feat_std]
        r_calib_ext = [rf * c_calib for rf in r_feat_ext]

        data_box = [
            [1.0] * len(rows_l), # Method 1 (Full)
            r_last_std, r_last_ext,
            r_feat_std, r_feat_ext,
            r_calib_std, r_calib_ext,
        ]
        labels = [
            "Method 1\n(Full Backbone)",
            "Method 2\n(Last, Standard)",
            "Method 2\n(Last, Extrap)",
            "Method 3\n(Feat, Standard)",
            "Method 3\n(Feat, Extrap)",
            "Method 3 (Calib)\n(Standard)",
            "Method 3 (Calib)\n(Extrap)",
        ]

        bp = ax.boxplot(data_box, patch_artist=True, showmeans=True,
                        meanprops={"marker": "D", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 5})
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        colors = ["#4a90e2", "#50e3c2", "#b8e986", "#f5a623", "#f8e71c", "#bd10e0", "#9013fe"]
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.7)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.5, label="Target Ratio (1.0)")
        ax.axhspan(0.5, 2.0, color="gray", alpha=0.15, label="Factor of 2 [0.5, 2.0]")
        ax.set_yscale("log")
        ax.set_title(f"{l_label} Effective Gradient Ratio $R_{{effective}}$", fontsize=12, fontweight="bold")
        ax.set_ylabel("Effective Ratio $R_{effective} = s_{approx} / s_{full}$" if ax_idx == 0 else "")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.tick_params(axis="x", rotation=35)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig1_path = OUTPUT_DIR / "figure1_effective_gradient_ratios.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"Saved: {fig1_path}")

    # Figure 2: Normalization Accuracy vs Computational Overhead
    fig, ax = plt.subplots(figsize=(8.5, 6))

    # Overhead for production batch size B=4
    overhead_b4 = {r["method"]: r for r in overhead_results if r["batch_size"] == 4}

    # Accuracy metric: median error from 1.0 in log space, or median absolute percentage error
    methods_plot = [
        ("Method 1: Full Shared-Backbone", 0.0, 0.0, overhead_b4["Method 1: Full Shared-Backbone"]["overhead_percent"], "#4a90e2", "s", 120),
        ("Method 2: Last Layer (up0)",
         abs(summary_data["student_t"]["all_conditions"]["s_last_over_s_full"]["median"] - 1.0),
         abs(summary_data["gaussian"]["all_conditions"]["s_last_over_s_full"]["median"] - 1.0),
         overhead_b4["Method 2: Last Layer (up0)"]["overhead_percent"], "#50e3c2", "o", 110),
        ("Method 3: Shared Feature (Raw)",
         abs(summary_data["student_t"]["all_conditions"]["s_feat_over_s_full"]["median"] - 1.0),
         abs(summary_data["gaussian"]["all_conditions"]["s_feat_over_s_full"]["median"] - 1.0),
         overhead_b4["Method 3: Shared Feature (F)"]["overhead_percent"], "#f5a623", "^", 110),
        ("Method 3: Shared Feature (Calibrated)",
         abs(summary_data["student_t"]["all_conditions"]["calibrated_feature_method"]["stats"]["median"] - 1.0),
         abs(summary_data["gaussian"]["all_conditions"]["calibrated_feature_method"]["stats"]["median"] - 1.0),
         overhead_b4["Method 3: Shared Feature (F)"]["overhead_percent"] + 0.1, "#bd10e0", "*", 150),
    ]

    for name, err_t, err_g, overhead_pct, color, marker, sz in methods_plot:
        ax.scatter([overhead_pct], [err_t], color=color, marker=marker, s=sz, label=f"{name} (Student-t)", zorder=5)
        if name != "Method 1: Full Shared-Backbone":
            ax.scatter([overhead_pct], [err_g], color=color, marker=marker, s=sz, facecolors="none", edgecolors=color, linewidth=2, label=f"{name} (Gaussian)", zorder=5)

    ax.set_xlabel("Computational Wall-Time Overhead (% over Unnormalized Baseline)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Approximation Error $|\\mathrm{median}(R_{effective}) - 1.0|$", fontsize=11, fontweight="bold")
    ax.set_title("Accuracy vs Computational Overhead Across Normalization Methods (B=4)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    fig2_path = OUTPUT_DIR / "figure2_accuracy_vs_overhead.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2_path}")

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
