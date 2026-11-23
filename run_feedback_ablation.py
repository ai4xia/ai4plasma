#!/usr/bin/env python3
"""Matched-Noise Prediction-Feedback Ablation for ai4plasma v11.

Investigates whether reusing previous Density predictions and their continuous
uncertainty causes the reconstruction degradation observed in online causal filtering:
  - Target frame: Global frame F24 (occupying local temporal index 23).
  - Common physical window: [F1, F2, ..., F24] (24 frames total).
  - Three conditioning strategies:
      * Strategy A (No-feedback baseline):
          Historical unobserved Density (F1-F23): Gaussian noise, u_in=1.0.
          Current unobserved Density (F24): Gaussian noise, u_in=1.0.
      * Strategy B (Prediction-feedback):
          Executes online causal filtering sequentially from F0 through F23.
          Historical unobserved Density (F1-F23): prior filtering estimates rho_hat,
          with continuous input variance u_in = u_hat.
          Current unobserved Density (F24): bitwise identical Gaussian noise to Strategy A, u_in=1.0.
      * Strategy C (Oracle-history diagnostic):
          Historical unobserved Density (F1-F23): ground-truth Density, u_in=0.0.
          Current unobserved Density (F24): bitwise identical Gaussian noise to Strategies A and B, u_in=1.0.
  - Strict noise matching: A single reproducible global noise tensor is generated per run
    and sliced consistently across all three strategies and common unknown inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from journal_statistics import (
    calibration_bias,
    pearson_spearman,
)
from make_paper_figures import (
    find_and_load_run,
    peek_run_n_frames,
    sliding_density_probe_mask,
)
from models.unet3d import UNet3D
from run_online_causal_reconstruction import (
    build_causal_window_input,
    load_model_and_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run matched-noise prediction-feedback ablation for ai4plasma v11."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/multi_beta_gkl_v11_fromscratch_256ep_20261122"),
        help="Path to training run directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="final.pt",
        help="Checkpoint filename relative to run-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/physworld_v11_feedback_ablation_20261123"),
        help="Output directory for ablation results.",
    )
    parser.add_argument(
        "--h5-dir",
        type=Path,
        default=Path("/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta_official2500_none_compat"),
        help="Directory of HDF5 VPIC data.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("split.json"),
        help="Path to split.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.2],
        help="Plasma beta values to evaluate.",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=48,
        help="Total rollout frames per run.",
    )
    parser.add_argument(
        "--target-frame",
        type=int,
        default=24,
        help="Target global frame to reconstruct (default: 24).",
    )
    parser.add_argument(
        "--density-probe-count",
        type=int,
        default=1000,
        help="Number of fixed density probes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum eligible runs to evaluate (0 for all).",
    )
    return parser.parse_args()


@torch.no_grad()
def run_online_history_rollout(
    model: torch.nn.Module,
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    global_noise: torch.Tensor,
    history_steps: int = 24,
    window_size: int = 24,
    hide_magnetic: bool = False,
    amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Executes online causal filtering sequentially from t=0 to history_steps-1.

    For negative historical times tau < 0, noise is sampled reproducibly using
    a deterministic per-step seed so that negative padding matches established online filtering.
    For non-negative global frames tau >= 0, unknown values are strictly matched to global_noise.

    Returns:
      history_pred: [history_steps, H, W]
      history_var:  [history_steps, H, W]
    """
    device = target_normalized.device
    dtype = target_normalized.dtype
    _, _, H, W = target_normalized.shape

    history_pred = torch.zeros((history_steps, H, W), device=device, dtype=dtype)
    history_var = torch.ones((history_steps, H, W), device=device, dtype=dtype)

    current_idx = window_size - 1
    probe_bool = (probe_mask > 0.5)

    for t in range(history_steps):
        # Build 24-frame causal window [t - 23, ..., t]
        values = torch.zeros((1, 4, window_size, H, W), device=device, dtype=dtype)
        input_variance = torch.ones_like(values)

        for l in range(window_size):
            tau = t - window_size + 1 + l
            if tau < 0:
                # Nonexistent historical frame: sample reproducible noise with variance 1.0
                g_pad = torch.Generator(device=device)
                g_pad.manual_seed((10007 * tau + 54321) & 0x7FFFFFFF)
                values[0, :, l] = torch.randn((4, H, W), generator=g_pad, device=device, dtype=dtype)
                input_variance[0, :, l] = 1.0
            else:
                # Valid frame tau in [0, t]
                if not hide_magnetic:
                    values[0, :3, l] = target_normalized[:3, tau]
                    input_variance[0, :3, l] = 0.0
                else:
                    values[0, :3, l] = global_noise[:3, tau]
                    input_variance[0, :3, l] = 1.0

                if tau == t:
                    # Current frame: probes known, unobserved is global_noise with variance 1.0
                    values[0, 3, l] = global_noise[3, tau]
                    input_variance[0, 3, l] = 1.0
                    values[0, 3, l, probe_bool] = target_normalized[3, tau, probe_bool]
                    input_variance[0, 3, l, probe_bool] = 0.0
                else:
                    # Historical frame tau < t: reuse prior prediction and continuous variance
                    values[0, 3, l] = history_pred[tau]
                    input_variance[0, 3, l] = history_var[tau]
                    # Genuine probes override
                    values[0, 3, l, probe_bool] = target_normalized[3, tau, probe_bool]
                    input_variance[0, 3, l, probe_bool] = 0.0

        model_input = torch.cat([values, input_variance], dim=1)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            enabled=(amp and device.type == "cuda"),
        ):
            prediction = model(model_input)

        if getattr(model, "uncertainty_aware", True):
            mean_pred, var_pred = prediction
        else:
            mean_pred = prediction
            var_pred = torch.zeros_like(mean_pred)

        density_pred_t = mean_pred[0, 3, current_idx].float()
        density_var_t = var_pred[0, 3, current_idx].float()

        density_pred_t[probe_bool] = target_normalized[3, t, probe_bool]
        density_var_t[probe_bool] = 0.0

        history_pred[t] = density_pred_t
        history_var[t] = density_var_t

    return history_pred, history_var


def build_feedback_ablation_input(
    strategy: str,
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    global_noise: torch.Tensor,
    history_pred: Optional[torch.Tensor],
    history_var: Optional[torch.Tensor],
    target_frame: int = 24,
    hide_magnetic: bool = False,
) -> torch.Tensor:
    """Constructs the 24-frame model input for [target_frame-23, ..., target_frame].

    For target_frame=24, window spans [F1, F2, ..., F24], so start=1, end=25.
    Local index 23 is always target_frame.

    Strategies for historical non-probe Density (l in 0..22, global frame tau = 1..23):
      - Strategy A (No-feedback): values = global_noise[3, tau], u_in = 1.0
      - Strategy B (Prediction-feedback): values = history_pred[tau], u_in = history_var[tau]
      - Strategy C (Oracle-history): values = target_normalized[3, tau], u_in = 0.0

    Current frame F24 (local index 23):
      - Probes: exact target_normalized[3, 24], u_in = 0.0
      - Unobserved: global_noise[3, 24], u_in = 1.0 (EXACTLY MATCHED ACROSS A, B, C)
    """
    device = target_normalized.device
    dtype = target_normalized.dtype
    _, _, H, W = target_normalized.shape
    window_size = 24
    start = target_frame - window_size + 1
    end = target_frame + 1

    window_target = target_normalized[:, start:end]
    window_noise = global_noise[:, start:end]

    values = window_noise.clone()
    input_variance = torch.ones_like(values)

    # 1. Magnetic components
    if not hide_magnetic:
        values[:3] = window_target[:3]
        input_variance[:3] = 0.0

    # 2. Density component (channel 3)
    probe_bool = (probe_mask > 0.5)

    # Historical frames l in 0..22 (tau in start..target_frame-1)
    for l in range(window_size - 1):
        tau = start + l
        if strategy == "A":
            # No-feedback: unobserved Density remains window_noise[3, l] with variance 1.0
            values[3, l] = window_noise[3, l]
            input_variance[3, l] = 1.0
        elif strategy == "B":
            # Prediction-feedback: reuse prior filtering predictions and continuous variance
            assert history_pred is not None and history_var is not None
            values[3, l] = history_pred[tau]
            input_variance[3, l] = history_var[tau]
        elif strategy == "C":
            # Oracle-history: ground truth Density with variance 0.0
            values[3, l] = window_target[3, l]
            input_variance[3, l] = 0.0
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # Genuine probes always win with zero variance
        values[3, l, probe_bool] = window_target[3, l, probe_bool]
        input_variance[3, l, probe_bool] = 0.0

    # Current target frame F24 at local index 23:
    # Probes take exact target with zero variance; unobserved remains window_noise with variance 1.0
    l_curr = window_size - 1
    values[3, l_curr] = window_noise[3, l_curr]
    input_variance[3, l_curr] = 1.0
    values[3, l_curr, probe_bool] = window_target[3, l_curr, probe_bool]
    input_variance[3, l_curr, probe_bool] = 0.0

    model_input = torch.cat([values.unsqueeze(0), input_variance.unsqueeze(0)], dim=1)
    return model_input


@torch.no_grad()
def evaluate_feedback_strategy(
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    global_noise: torch.Tensor,
    history_pred: Optional[torch.Tensor],
    history_var: Optional[torch.Tensor],
    strategy: str,
    target_frame: int = 24,
    hide_magnetic: bool = False,
    amp: bool = True,
) -> Dict:
    device = target_normalized.device
    local_idx = 23

    model_input = build_feedback_ablation_input(
        strategy=strategy,
        target_normalized=target_normalized,
        probe_mask=probe_mask,
        global_noise=global_noise,
        history_pred=history_pred,
        history_var=history_var,
        target_frame=target_frame,
        hide_magnetic=hide_magnetic,
    )

    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        enabled=(amp and device.type == "cuda"),
    ):
        prediction = model(model_input)

    if getattr(model, "uncertainty_aware", True):
        mean_pred, var_pred = prediction
    else:
        mean_pred = prediction
        var_pred = torch.zeros_like(mean_pred)

    density_pred_norm = mean_pred[0, 3, local_idx].float()
    density_var_norm = var_pred[0, 3, local_idx].float()
    target_norm = target_normalized[3, target_frame].float()

    hidden_mask = ~(probe_mask > 0.5)

    diff_all = density_pred_norm - target_norm
    e2_all = diff_all.square()
    full_field_rmse = float(e2_all.mean().sqrt().item())

    e2_hidden = e2_all[hidden_mask]
    var_hidden = density_var_norm[hidden_mask]

    mean_e2_hidden = float(e2_hidden.mean().item())
    hidden_rmse = float(math.sqrt(mean_e2_hidden))
    mean_var_hidden = float(var_hidden.mean().item())
    u_ratio = float(mean_var_hidden / (mean_e2_hidden + 1e-12))

    e2_np = e2_hidden.detach().cpu().numpy().ravel()
    var_np = var_hidden.detach().cpu().numpy().ravel()
    corr_dict = pearson_spearman(var_np, e2_np)
    pearson_r = float(corr_dict["pearson"])
    spearman_r = float(corr_dict["spearman"])

    pred_phys = (density_pred_norm * float(std[3].item()) + float(mean[3].item())).detach().cpu().numpy()
    target_phys = (target_norm * float(std[3].item()) + float(mean[3].item())).detach().cpu().numpy()

    return {
        "strategy": strategy,
        "target_frame": target_frame,
        "density_mse_hidden": mean_e2_hidden,
        "density_rmse_hidden": hidden_rmse,
        "full_field_rmse": full_field_rmse,
        "mean_var_hidden": mean_var_hidden,
        "u_ratio_hidden": u_ratio,
        "pearson_r": pearson_r,
        "spearman_r": spearman_r,
        "pred_phys": pred_phys,
        "target_phys": target_phys,
        "pred_norm": density_pred_norm.detach().cpu().numpy(),
        "target_norm": target_norm.detach().cpu().numpy(),
        "hidden_mask": hidden_mask.detach().cpu().numpy(),
    }


def plot_feedback_figures(
    df_results: pd.DataFrame,
    qualitative_cache: Optional[Dict],
    figures_dir: Path,
    target_frame: int,
):
    colors = {"B_full": "#1f77b4", "B_hidden": "#d62728"}
    strat_order = ["Strategy A (No-feedback)", "Strategy B (Prediction-feedback)", "Strategy C (Oracle-history)"]
    strat_labels = {"A": strat_order[0], "B": strat_order[1], "C": strat_order[2]}

    # Map strategy code to full label
    df_results = df_results.copy()
    df_results["strategy_label"] = df_results["strategy"].map(strat_labels)

    # -------------------------------------------------------------
    # Figure 1: Hidden-voxel MSE across A/B/C
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub = df_results[df_results["b_condition"] == cond]

        runs = sub["run_name"].unique()
        for r in runs:
            r_sub = sub[sub["run_name"] == r].set_index("strategy_label").reindex(strat_order)
            ax.plot(strat_order, r_sub["density_mse_hidden"], color="gray", alpha=0.4, lw=1.2, marker="o", markersize=4)

        mean_vals = [sub[sub["strategy_label"] == s]["density_mse_hidden"].mean() for s in strat_order]
        ax.plot(strat_order, mean_vals, color=colors[cond], lw=2.8, marker="s", markersize=8, label=f"Mean ({cond})")

        ax.set_title(f"{cond}: Hidden MSE Across Conditioning Strategies (F{target_frame})", fontsize=10)
        ax.set_ylabel("Standardized MSE (e² on Hidden Voxels)")
        ax.set_xticks(range(len(strat_order)))
        ax.set_xticklabels(["A (No-fb)", "B (Pred-fb)", "C (Oracle)"], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"Figure 1: Matched-Noise Prediction-Feedback Ablation (Target Frame F{target_frame})\n"
        "(Identical Window F1–F24, Target Position idx 23, and F24 Noise Realization)",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_mse_across_feedback_strategies.png", dpi=200)
    fig.savefig(figures_dir / "figure1_mse_across_feedback_strategies.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Predicted Variance vs. Actual MSE (u/MSE)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub = df_results[df_results["b_condition"] == cond]

        mean_mse = [sub[sub["strategy_label"] == s]["density_mse_hidden"].mean() for s in strat_order]
        mean_var = [sub[sub["strategy_label"] == s]["mean_var_hidden"].mean() for s in strat_order]

        x = np.arange(len(strat_order))
        width = 0.35
        ax.bar(x - width / 2, mean_mse, width, label="Actual MSE (e²)", color="firebrick", alpha=0.85)
        ax.bar(x + width / 2, mean_var, width, label="Predicted Var (u)", color="navy", alpha=0.85)

        for i in range(len(strat_order)):
            ratio = mean_var[i] / (mean_mse[i] + 1e-12)
            ax.text(x[i], max(mean_mse[i], mean_var[i]) + 0.0002, f"u/MSE={ratio:.2f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(["A (No-fb)", "B (Pred-fb)", "C (Oracle)"], fontsize=9)
        ax.set_title(f"{cond}: Uncertainty Calibration (F{target_frame})", fontsize=11)
        ax.set_ylabel("Standardized squared units")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"Figure 2: Predicted Uncertainty vs. Reconstruction Error Across Strategies (F{target_frame})",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_predicted_variance_and_calibration.png", dpi=200)
    fig.savefig(figures_dir / "figure2_predicted_variance_and_calibration.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: Representative Reconstruction Comparison (A/B/C)
    # -------------------------------------------------------------
    if qualitative_cache is not None:
        target_p = qualitative_cache["target_phys"]
        preds = qualitative_cache["preds"]

        fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
        vmin, vmax = np.nanpercentile(target_p, [2, 98])

        res_limits = []
        for s in ["A", "B", "C"]:
            res = preds[s] - target_p
            res_limits.append(max(abs(np.nanpercentile(res, 2)), abs(np.nanpercentile(res, 98))))
        max_res = max(res_limits)

        # Row 0: Target
        im_tgt = axes[0, 0].imshow(target_p, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f"Ground Truth Density (F{target_frame})", fontsize=10)
        axes[0, 0].set_ylabel("Z index")
        fig.colorbar(im_tgt, ax=axes[0, 0], fraction=0.046, pad=0.04)
        axes[0, 1].axis("off")

        strat_names = {
            "A": "Strategy A: No-feedback (u_hist=1)",
            "B": "Strategy B: Pred-feedback (rho_hat, u_hat)",
            "C": "Strategy C: Oracle-history (rho_true, u_hist=0)",
        }

        for r_idx, s in enumerate(["A", "B", "C"], start=1):
            pred = preds[s]
            res = pred - target_p

            im_p = axes[r_idx, 0].imshow(pred, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            axes[r_idx, 0].set_title(f"Reconstruction: {strat_names[s]}", fontsize=9)
            axes[r_idx, 0].set_ylabel("Z index")
            fig.colorbar(im_p, ax=axes[r_idx, 0], fraction=0.046, pad=0.04)

            im_r = axes[r_idx, 1].imshow(res, origin="lower", aspect="auto", cmap="coolwarm", vmin=-max_res, vmax=max_res)
            axes[r_idx, 1].set_title(f"Residual: {strat_names[s]}", fontsize=9)
            fig.colorbar(im_r, ax=axes[r_idx, 1], fraction=0.046, pad=0.04)

        axes[3, 0].set_xlabel("X index")
        axes[3, 1].set_xlabel("X index")

        run_lbl = qualitative_cache.get("run_name", "Test Run")
        fig.suptitle(
            f"Figure 3: Representative Target F{target_frame} Reconstruction Across Strategies\n"
            f"Run: {run_lbl} ({qualitative_cache.get('b_condition', 'B_full')}) — Matched F24 Noise Realization",
            fontsize=11,
            y=0.98,
        )
        fig.tight_layout()
        fig.savefig(figures_dir / "figure3_representative_reconstruction_comparison.png", dpi=200)
        fig.savefig(figures_dir / "figure3_representative_reconstruction_comparison.pdf")
        plt.close(fig)


def main():
    args = parse_args()
    output_dir = args.output_dir
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model and stats
    model, mean, std, ckpt_args = load_model_and_stats(args.run_dir, args.checkpoint, device)

    # Load split and filter 10 eligible runs
    with open(args.split) as f:
        split_data = json.load(f)

    test_runs = sorted(split_data.get("test_runs", []))
    beta_prefix = f"beta{args.betas[0]}_" if len(args.betas) == 1 else "beta0.2_"
    candidate_runs = [r for r in test_runs if r.startswith(beta_prefix)]

    eligible_runs = []
    for r in candidate_runs:
        n_avail = peek_run_n_frames(args.h5_dir, args.betas, r)
        if n_avail is not None and int(n_avail) >= args.n_frames:
            eligible_runs.append(r)

    if args.max_runs > 0:
        eligible_runs = eligible_runs[: args.max_runs]

    print(f"Evaluating feedback ablation on F{args.target_frame} across {len(eligible_runs)} eligible runs: {eligible_runs}")

    b_conditions = ["B_full", "B_hidden"]
    strategies = ["A", "B", "C"]

    results_rows = []
    qualitative_cache = None

    for run_idx, run_name in enumerate(eligible_runs):
        print(f"\n--- [{run_idx + 1}/{len(eligible_runs)}] Run: {run_name} ---")
        fields, _frame_ids, _h5_path = find_and_load_run(args.h5_dir, args.betas, run_name)
        fields = np.asarray(fields)[: args.n_frames]
        run_tensor = torch.from_numpy(np.transpose(fields, (1, 0, 2, 3))).to(
            device=device, dtype=torch.float32
        )
        run_tensor_norm = (run_tensor - mean) / (std + 1e-8)

        probe_mask_2d, _probe_info = sliding_density_probe_mask(
            run_tensor_norm.unsqueeze(0),
            int(args.density_probe_count),
        )

        # Generate single reproducible global noise tensor [4, n_frames, H, W] for this run
        g = torch.Generator(device=device)
        g.manual_seed(args.seed + run_idx * 1000)
        global_noise = torch.randn(
            (4, args.n_frames, run_tensor.shape[2], run_tensor.shape[3]),
            generator=g,
            device=device,
            dtype=torch.float32,
        )

        for b_cond in b_conditions:
            hide_magnetic = (b_cond == "B_hidden")
            run_preds = {}

            # Execute preceding online causal rollout for Strategy B (frames 0..23)
            print(f"  [{b_cond}] Computing online causal history rollout (F0–F23)...")
            history_pred, history_var = run_online_history_rollout(
                model=model,
                target_normalized=run_tensor_norm,
                probe_mask=probe_mask_2d,
                global_noise=global_noise,
                history_steps=args.target_frame,  # evaluates 0..23
                window_size=24,
                hide_magnetic=hide_magnetic,
                amp=True,
            )

            for strat in strategies:
                res = evaluate_feedback_strategy(
                    model=model,
                    mean=mean,
                    std=std,
                    target_normalized=run_tensor_norm,
                    probe_mask=probe_mask_2d,
                    global_noise=global_noise,
                    history_pred=history_pred if strat == "B" else None,
                    history_var=history_var if strat == "B" else None,
                    strategy=strat,
                    target_frame=args.target_frame,
                    hide_magnetic=hide_magnetic,
                    amp=True,
                )

                strat_label = {"A": "No-fb", "B": "Pred-fb", "C": "Oracle"}[strat]
                print(
                    f"    [Strategy {strat} ({strat_label})] Hidden MSE: {res['density_mse_hidden']:.5f} | "
                    f"RMSE: {res['density_rmse_hidden']:.4f} | "
                    f"Var u: {res['mean_var_hidden']:.5f} | "
                    f"u/MSE: {res['u_ratio_hidden']:.3f} | "
                    f"Pearson r: {res['pearson_r']:.3f}"
                )

                results_rows.append({
                    "run_name": run_name,
                    "target_frame": args.target_frame,
                    "b_condition": b_cond,
                    "strategy": strat,
                    "density_mse_hidden": res["density_mse_hidden"],
                    "density_rmse_hidden": res["density_rmse_hidden"],
                    "full_field_rmse": res["full_field_rmse"],
                    "mean_var_hidden": res["mean_var_hidden"],
                    "u_ratio_hidden": res["u_ratio_hidden"],
                    "pearson_r": res["pearson_r"],
                    "spearman_r": res["spearman_r"],
                })

                run_preds[strat] = res["pred_phys"]

            if qualitative_cache is None and b_cond == "B_full":
                qualitative_cache = {
                    "run_name": run_name,
                    "b_condition": b_cond,
                    "target_phys": res["target_phys"],
                    "preds": run_preds,
                }

    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(data_dir / "feedback_results.csv", index=False)
    print(f"\nSaved results: {data_dir / 'feedback_results.csv'}")

    # Generate paired comparison summary: B - A, C - A, B - C
    paired_rows = []
    for b_cond in b_conditions:
        sub = df_results[df_results["b_condition"] == b_cond]
        p_mse = sub.pivot(index="run_name", columns="strategy", values="density_mse_hidden")
        p_rmse = sub.pivot(index="run_name", columns="strategy", values="density_rmse_hidden")

        diff_mse_B_minus_A = p_mse["B"] - p_mse["A"]
        diff_mse_C_minus_A = p_mse["C"] - p_mse["A"]
        diff_mse_B_minus_C = p_mse["B"] - p_mse["C"]

        diff_rmse_B_minus_A = p_rmse["B"] - p_rmse["A"]
        diff_rmse_C_minus_A = p_rmse["C"] - p_rmse["A"]
        diff_rmse_B_minus_C = p_rmse["B"] - p_rmse["C"]

        for r in p_mse.index:
            paired_rows.append({
                "run_name": r,
                "b_condition": b_cond,
                "target_frame": args.target_frame,
                "mse_A": p_mse.loc[r, "A"],
                "mse_B": p_mse.loc[r, "B"],
                "mse_C": p_mse.loc[r, "C"],
                "diff_mse_B_minus_A": diff_mse_B_minus_A[r],
                "diff_mse_C_minus_A": diff_mse_C_minus_A[r],
                "diff_mse_B_minus_C": diff_mse_B_minus_C[r],
                "diff_rmse_B_minus_A": diff_rmse_B_minus_A[r],
                "diff_rmse_C_minus_A": diff_rmse_C_minus_A[r],
                "diff_rmse_B_minus_C": diff_rmse_B_minus_C[r],
            })

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(data_dir / "feedback_paired_comparisons.csv", index=False)
    print(f"Saved paired comparisons: {data_dir / 'feedback_paired_comparisons.csv'}")

    # Generate Figures
    plot_feedback_figures(
        df_results=df_results,
        qualitative_cache=qualitative_cache,
        figures_dir=figures_dir,
        target_frame=args.target_frame,
    )

    # Save Manifest
    manifest = {
        "experiment": "matched_noise_prediction_feedback_ablation",
        "checkpoint": str(args.run_dir / args.checkpoint),
        "split": str(args.split),
        "target_frame": args.target_frame,
        "eligible_runs": eligible_runs,
        "n_runs": len(eligible_runs),
        "b_conditions": b_conditions,
        "strategies": {
            "A": "No-feedback baseline (historical unobserved Density = Gaussian noise, u_in=1.0)",
            "B": "Prediction-feedback (historical unobserved Density = prior online filtering rho_hat, u_in=u_hat)",
            "C": "Oracle-history diagnostic (historical unobserved Density = true Density, u_in=0.0)",
        },
        "output_files": [
            str(data_dir / "feedback_results.csv"),
            str(data_dir / "feedback_paired_comparisons.csv"),
            str(figures_dir / "figure1_mse_across_feedback_strategies.png"),
            str(figures_dir / "figure2_predicted_variance_and_calibration.png"),
            str(figures_dir / "figure3_representative_reconstruction_comparison.png"),
        ],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
