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
        default=Path("evaluations/physworld_v11_history_value_uncertainty_ablation_20261123"),
        help="Output directory for ablation results.",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        nargs="+",
        default=["A", "B", "C", "D", "E"],
        choices=["A", "B", "C", "D", "E"],
        help="Conditioning strategies to evaluate.",
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
        elif strategy == "D":
            # Diagnostic control D: Ground truth Density with variance 1.0
            values[3, l] = window_target[3, l]
            input_variance[3, l] = 1.0
        elif strategy == "E":
            # Diagnostic control E: Gaussian noise with variance 0.0
            values[3, l] = window_noise[3, l]
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
    all_strats = ["A", "B", "C", "D", "E"]
    active_strats = [s for s in all_strats if s in df_results["strategy"].unique()]

    strat_labels = {
        "A": "Strategy A: Noise (u=1)",
        "B": "Strategy B: Preds (u_hat)",
        "C": "Strategy C: GT (u=0)",
        "D": "Strategy D: GT (u=1)",
        "E": "Strategy E: Noise (u=0)",
    }
    short_labels = {
        "A": "A\n(Noise, u=1)",
        "B": "B\n(Pred, u)",
        "C": "C\n(GT, u=0)",
        "D": "D\n(GT, u=1)",
        "E": "E\n(Noise, u=0)",
    }
    strat_order = [strat_labels[s] for s in active_strats]
    df_results = df_results.copy()
    df_results["strategy_label"] = df_results["strategy"].map(strat_labels)

    # -------------------------------------------------------------
    # Figure 1: Hidden-voxel MSE across Strategies
    # -------------------------------------------------------------
    n_strats = len(active_strats)
    fig_w = max(10, 2.2 * n_strats)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5), sharey=True)
    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub = df_results[df_results["b_condition"] == cond]

        runs = sub["run_name"].unique()
        for r in runs:
            r_sub = sub[sub["run_name"] == r].set_index("strategy_label").reindex(strat_order)
            ax.plot(strat_order, r_sub["density_mse_hidden"], color="gray", alpha=0.35, lw=1.2, marker="o", markersize=4)

        mean_vals = [sub[sub["strategy_label"] == s]["density_mse_hidden"].mean() for s in strat_order]
        ax.plot(strat_order, mean_vals, color=colors[cond], lw=2.8, marker="s", markersize=8, label=f"Mean ({cond})")

        ax.set_title(f"{cond}: Hidden MSE Across Conditioning Strategies (F{target_frame})", fontsize=10)
        ax.set_ylabel("Standardized MSE (e² on Hidden Voxels)")
        ax.set_xticks(range(len(strat_order)))
        ax.set_xticklabels([short_labels[s] for s in active_strats], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"Figure 1: Historical Density vs. Uncertainty Ablation (Target Frame F{target_frame})\n"
        "(Identical Window F1–F24, Target Position idx 23, and Matched F24 Noise Realization)",
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
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5))
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
        ax.set_xticklabels([short_labels[s] for s in active_strats], fontsize=9)
        ax.set_title(f"{cond}: Uncertainty Calibration (F{target_frame})", fontsize=11)
        ax.set_ylabel("Standardized squared units")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"Figure 2: Predicted Uncertainty vs. Actual MSE Across Strategies (F{target_frame})",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_predicted_variance_and_calibration.png", dpi=200)
    fig.savefig(figures_dir / "figure2_predicted_variance_and_calibration.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: Representative Reconstruction Comparison
    # -------------------------------------------------------------
    if qualitative_cache is not None:
        target_p = qualitative_cache["target_phys"]
        preds = qualitative_cache["preds"]

        n_rows = 1 + len(active_strats)
        fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.4 * n_rows), sharex=True, sharey=True)
        vmin, vmax = np.nanpercentile(target_p, [2, 98])

        res_limits = []
        for s in active_strats:
            if s in preds:
                res = preds[s] - target_p
                res_limits.append(max(abs(np.nanpercentile(res, 2)), abs(np.nanpercentile(res, 98))))
        max_res = max(res_limits) if res_limits else 1.0

        # Row 0: Target
        im_tgt = axes[0, 0].imshow(target_p, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f"Ground Truth Density (F{target_frame})", fontsize=10)
        axes[0, 0].set_ylabel("Z index")
        fig.colorbar(im_tgt, ax=axes[0, 0], fraction=0.046, pad=0.04)
        axes[0, 1].axis("off")

        for r_idx, s in enumerate(active_strats, start=1):
            pred = preds.get(s, target_p)
            res = pred - target_p

            im_p = axes[r_idx, 0].imshow(pred, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            axes[r_idx, 0].set_title(f"Reconstruction: {strat_labels[s]}", fontsize=9)
            axes[r_idx, 0].set_ylabel("Z index")
            fig.colorbar(im_p, ax=axes[r_idx, 0], fraction=0.046, pad=0.04)

            im_r = axes[r_idx, 1].imshow(res, origin="lower", aspect="auto", cmap="coolwarm", vmin=-max_res, vmax=max_res)
            axes[r_idx, 1].set_title(f"Residual: {strat_labels[s]}", fontsize=9)
            fig.colorbar(im_r, ax=axes[r_idx, 1], fraction=0.046, pad=0.04)

        axes[-1, 0].set_xlabel("X index")
        axes[-1, 1].set_xlabel("X index")

        run_lbl = qualitative_cache.get("run_name", "Test Run")
        fig.suptitle(
            f"Figure 3: Representative Target F{target_frame} Reconstruction Across Strategies\n"
            f"Run: {run_lbl} ({qualitative_cache.get('b_condition', 'B_full')}) — Matched F24 Noise Realization",
            fontsize=11,
            y=0.99,
        )
        fig.tight_layout()
        fig.savefig(figures_dir / "figure3_representative_reconstruction_comparison.png", dpi=200)
        fig.savefig(figures_dir / "figure3_representative_reconstruction_comparison.pdf")
        plt.close(fig)
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
    strategies = [s for s in ["A", "B", "C", "D", "E"] if s in args.strategies]

    strat_labels = {
        "A": "Noise, u=1",
        "B": "Pred-fb",
        "C": "Oracle (GT, u=0)",
        "D": "Diag-D (GT, u=1)",
        "E": "Diag-E (Noise, u=0)",
    }

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

            history_pred, history_var = None, None
            if "B" in strategies:
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

                label = strat_labels.get(strat, strat)
                print(
                    f"    [Strategy {strat} ({label})] Hidden MSE: {res['density_mse_hidden']:.5f} | "
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

    # Generate paired comparisons across key scientific contrasts
    # Specified pairs: A vs D, A vs E, C vs D, C vs E, B vs C, B vs A, C vs A
    candidate_pairs = [
        ("D", "A", "GT_vs_Noise_u1 (D - A)"),
        ("E", "A", "u0_vs_u1_Noise (E - A)"),
        ("D", "C", "u1_vs_u0_GT (D - C)"),
        ("E", "C", "Noise_vs_GT_u0 (E - C)"),
        ("B", "C", "Pred_vs_GT (B - C)"),
        ("B", "A", "Pred_vs_Noise (B - A)"),
        ("C", "A", "GT_vs_Noise (C - A)"),
    ]

    paired_rows = []
    summary_rows = []

    for b_cond in b_conditions:
        sub = df_results[df_results["b_condition"] == b_cond]
        p_mse = sub.pivot(index="run_name", columns="strategy", values="density_mse_hidden")
        p_rmse = sub.pivot(index="run_name", columns="strategy", values="density_rmse_hidden")

        # Per-strategy summary
        for strat in strategies:
            strat_mse = p_mse[strat]
            strat_sub = sub[sub["strategy"] == strat]
            summary_rows.append({
                "b_condition": b_cond,
                "strategy": strat,
                "label": strat_labels.get(strat, strat),
                "mean_hidden_mse": float(strat_mse.mean()),
                "median_hidden_mse": float(strat_mse.median()),
                "std_hidden_mse": float(strat_mse.std()),
                "mean_predicted_var": float(strat_sub["mean_var_hidden"].mean()),
                "mean_u_ratio": float(strat_sub["u_ratio_hidden"].mean()),
            })

        for r in p_mse.index:
            row = {
                "run_name": r,
                "b_condition": b_cond,
                "target_frame": args.target_frame,
            }
            for strat in strategies:
                row[f"mse_{strat}"] = p_mse.loc[r, strat]
                row[f"rmse_{strat}"] = p_rmse.loc[r, strat]

            for s1, s2, pair_key in candidate_pairs:
                if s1 in p_mse.columns and s2 in p_mse.columns:
                    diff_m = p_mse.loc[r, s1] - p_mse.loc[r, s2]
                    ratio_m = p_mse.loc[r, s1] / (p_mse.loc[r, s2] + 1e-12)
                    diff_r = p_rmse.loc[r, s1] - p_rmse.loc[r, s2]
                    row[f"diff_mse_{s1}_minus_{s2}"] = diff_m
                    row[f"ratio_mse_{s1}_to_{s2}"] = ratio_m
                    row[f"diff_rmse_{s1}_minus_{s2}"] = diff_r

            paired_rows.append(row)

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(data_dir / "feedback_paired_comparisons.csv", index=False)
    print(f"Saved paired comparisons: {data_dir / 'feedback_paired_comparisons.csv'}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(data_dir / "feedback_summary_table.csv", index=False)
    print(f"Saved summary table: {data_dir / 'feedback_summary_table.csv'}")

    # Print summary table to console
    print("\n=== HISTORICAL DENSITY VS. UNCERTAINTY ABLATION SUMMARY ===")
    for b_cond in b_conditions:
        print(f"\nCondition: {b_cond}")
        sub_sum = df_summary[df_summary["b_condition"] == b_cond]
        for _, r in sub_sum.iterrows():
            print(f"  Strategy {r['strategy']} ({r['label']}): "
                  f"MSE Mean={r['mean_hidden_mse']:.5f}, Median={r['median_hidden_mse']:.5f} | "
                  f"Var u={r['mean_predicted_var']:.5f} | u/MSE={r['mean_u_ratio']:.3f}")

        # Pairwise comparison stats
        sub_paired = df_paired[df_paired["b_condition"] == b_cond]
        print(f"  Critical Pairwise Comparisons ({b_cond}):")
        for s1, s2, desc in candidate_pairs:
            col_diff = f"diff_mse_{s1}_minus_{s2}"
            col_ratio = f"ratio_mse_{s1}_to_{s2}"
            if col_diff in sub_paired.columns:
                diffs = sub_paired[col_diff]
                ratios = sub_paired[col_ratio]
                n_better = int((diffs < 0).sum())
                n_worse = int((diffs > 0).sum())
                n_eq = int((diffs == 0).sum())
                print(f"    {desc}: Mean Diff={diffs.mean():+.5f}, Median Diff={diffs.median():+.5f}, "
                      f"Mean Ratio={ratios.mean():.3f}x | {s1}<{s2} (improves): {n_better}/{len(diffs)}, "
                      f"{s1}>{s2} (deteriorates): {n_worse}/{len(diffs)}")

    # Generate Figures
    plot_feedback_figures(
        df_results=df_results,
        qualitative_cache=qualitative_cache,
        figures_dir=figures_dir,
        target_frame=args.target_frame,
    )

    # Save Manifest
    manifest = {
        "experiment": "historical_density_vs_uncertainty_ablation",
        "checkpoint": str(args.run_dir / args.checkpoint),
        "split": str(args.split),
        "target_frame": args.target_frame,
        "eligible_runs": eligible_runs,
        "n_runs": len(eligible_runs),
        "b_conditions": b_conditions,
        "strategies": {
            "A": "Noise history with u=1.0",
            "B": "Online prediction feedback with continuous u=u_hat",
            "C": "Ground truth Density history with u=0.0",
            "D": "Diagnostic D: Ground truth Density history with u=1.0",
            "E": "Diagnostic E: Noise history with u=0.0",
        },
        "critical_comparisons": [
            "A vs D (Same u=1.0, different Density: Noise vs GT)",
            "A vs E (Same Noise values, different u: 1.0 vs 0.0)",
            "C vs D (Same GT values, different u: 0.0 vs 1.0)",
            "C vs E (Same u=0.0, different Density: GT vs Noise)",
            "B vs C (Prediction feedback vs Oracle history)",
        ],
        "output_files": [
            str(data_dir / "feedback_results.csv"),
            str(data_dir / "feedback_paired_comparisons.csv"),
            str(data_dir / "feedback_summary_table.csv"),
            str(figures_dir / "figure1_mse_across_feedback_strategies.png"),
            str(figures_dir / "figure2_predicted_variance_and_calibration.png"),
            str(figures_dir / "figure3_representative_reconstruction_comparison.png"),
        ],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
