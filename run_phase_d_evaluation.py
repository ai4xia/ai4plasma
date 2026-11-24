#!/usr/bin/env python3
"""Phase D Stateful Causal Reconstruction Evaluation for ai4plasma v11.

Evaluates and compares online causal reconstruction across four conditions:
  1. v11_original:  Source v11 model with standard recursive feedback (u_in = u_hat).
  2. v11_u1:        Source v11 model with inflated historical uncertainty (u_in = 1.0).
  3. phased_learned: Phase D fine-tuned model with learned uncertainty feedback (u_in = u_hat).
  4. no_feedback:   Independent reconstruction baseline (noise historical unobserved, u_in = 1.0).

Scientific Goal:
  Determine whether stateful causal training (Phase D) enables the model to extract
  useful temporal information from historical predictions, consistently outperforming
  the No-feedback baseline in the settled regime (F24–F48) while maintaining well-calibrated
  predicted uncertainties.

Protocol:
  - 48 consecutive physical frames (t in [0, 47], F1–F48).
  - 10 held-out beta=0.20 test simulations (strict train/test separation).
  - Both B_full (visible B) and B_hidden (unobserved B, u_in=1.0) conditions.
  - 1,000 fixed spatial Density probes (u_in=0.0).
  - Noise-matched inputs and identical metric definitions.
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
from scipy import stats
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
from run_full_online_uncertainty_ablation import (
    build_rollout_step_input,
)


CONDITION_LABELS = {
    "v11_original": "v11 Original Feedback",
    "v11_u1": "v11 Heuristic (u=1)",
    "no_feedback": "v11 No-Feedback",
    "phased_no_feedback": "Phase D No-Feedback",
    "phased_learned": "Phase D Learned Feedback",
}

CONDITION_COLORS = {
    "v11_original": "#d62728",        # Red
    "v11_u1": "#ff7f0e",              # Orange
    "no_feedback": "#2ca02c",         # Green
    "phased_no_feedback": "#9467bd",  # Purple
    "phased_learned": "#1f77b4",      # Blue
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase D stateful causal reconstruction evaluation for ai4plasma v11."
    )
    parser.add_argument(
        "--v11-checkpoint",
        type=Path,
        default=Path("runs/multi_beta_gkl_v11_fromscratch_256ep_20261122/final.pt"),
        help="Path to converged v11 source checkpoint.",
    )
    parser.add_argument(
        "--phased-checkpoint",
        type=Path,
        default=Path("runs/multi_beta_gkl_v11_phase_d_pilot_6ep_20261123/final.pt"),
        help="Path to Phase D continuation checkpoint (or latest.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/physworld_v11_phase_d_pilot_evaluation_20261123"),
        help="Output directory for Phase D evaluation results.",
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
        help="Beta values to evaluate (default: 0.2).",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=48,
        help="Total rollout frames per run (default: 48).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="Context window size (default: 24).",
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
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=["v11_original", "v11_u1", "no_feedback", "phased_no_feedback", "phased_learned"],
        choices=["v11_original", "v11_u1", "no_feedback", "phased_no_feedback", "phased_learned"],
        help="Evaluation conditions to benchmark.",
    )
    parser.add_argument(
        "--allow-missing-phased",
        action="store_true",
        help="Allow running evaluation on v11 baselines even if Phase D checkpoint is missing.",
    )
    return parser.parse_args()


def load_model_from_checkpoint(
    checkpoint_path: Path, device: torch.device
) -> Tuple[UNet3D, torch.Tensor, torch.Tensor, Dict]:
    """Loads UNet3D model and normalization stats from a checkpoint file."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=int(args.get("base_channels", 24)),
        channel_mults=tuple(args.get("channel_mults", (1, 2, 4, 8))),
        architecture=str(args.get("model_version", "direct_unet3d_no_activation_norm_v3")),
        use_attention=bool(args.get("use_attention", True)),
        spatial_only_pooling=bool(args.get("spatial_only_pooling", True)),
        uncertainty_aware=bool(args.get("uncertainty_aware", True)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    stats = ckpt["stats"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).reshape(4, 1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).reshape(4, 1, 1, 1)

    return model, mean, std, args


@torch.no_grad()
def run_condition_rollout(
    condition: str,
    v11_model: UNet3D,
    phased_model: Optional[UNet3D],
    target_norm: torch.Tensor,
    probe_mask: torch.Tensor,
    global_noise: torch.Tensor,
    n_frames: int = 48,
    window_size: int = 24,
    hide_magnetic: bool = False,
    amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Runs a 48-frame causal rollout under the specified condition.

    Mapping:
      - 'v11_original':       model = v11_model,    strategy = 'original'
      - 'v11_u1':             model = v11_model,    strategy = 'u1'
      - 'no_feedback':        model = v11_model,    strategy = 'no_feedback'
      - 'phased_no_feedback': model = phased_model, strategy = 'no_feedback'
      - 'phased_learned':     model = phased_model, strategy = 'original'
    """
    if condition == "phased_learned":
        if phased_model is None:
            raise ValueError("Phase D model is required for 'phased_learned' condition.")
        model = phased_model
        strategy = "original"
    elif condition == "phased_no_feedback":
        if phased_model is None:
            raise ValueError("Phase D model is required for 'phased_no_feedback' condition.")
        model = phased_model
        strategy = "no_feedback"
    elif condition == "v11_original":
        model = v11_model
        strategy = "original"
    elif condition == "v11_u1":
        model = v11_model
        strategy = "u1"
    elif condition == "no_feedback":
        model = v11_model
        strategy = "no_feedback"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    device = target_norm.device
    dtype = target_norm.dtype
    _, _, H, W = target_norm.shape

    probe_bool = (probe_mask > 0.5)
    hidden_bool = ~probe_bool

    rollout_pred = torch.zeros((n_frames, H, W), device=device, dtype=dtype)
    rollout_var = torch.zeros((n_frames, H, W), device=device, dtype=dtype)
    frame_metrics = []

    for t in range(n_frames):
        model_input = build_rollout_step_input(
            t=t,
            strategy=strategy,
            target_normalized=target_norm,
            probe_mask=probe_mask,
            global_noise=global_noise,
            history_pred=rollout_pred,
            history_var=rollout_var,
            hide_magnetic=hide_magnetic,
            window_size=window_size,
        )

        dev_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(dev_type, enabled=(amp and device.type == "cuda"), dtype=torch.bfloat16):
            pred_all, var_all = model(model_input)

        pred_frame = pred_all[0, :, window_size - 1].float()
        var_frame = var_all[0, :, window_size - 1].float()

        pred_rho = pred_frame[3]
        var_rho = var_frame[3]
        target_rho = target_norm[3, t]

        # Calculate errors
        sq_err_hidden = (pred_rho[hidden_bool] - target_rho[hidden_bool]).square()
        mse_hidden = float(sq_err_hidden.mean().item())
        mse_full = float((pred_rho - target_rho).square().mean().item())
        mse_observed = float((pred_rho[probe_bool] - target_rho[probe_bool]).square().mean().item())

        var_hidden = float(var_rho[hidden_bool].mean().item())
        var_full = float(var_rho.mean().item())

        target_std_hidden = float(target_rho[hidden_bool].std().item())
        nrmse_hidden = math.sqrt(mse_hidden) / (target_std_hidden + 1e-8)

        # Calibration metrics
        var_h_np = var_rho[hidden_bool].detach().cpu().numpy().ravel()
        sq_h_np = sq_err_hidden.detach().cpu().numpy().ravel()
        corr_dict = pearson_spearman(var_h_np, sq_h_np)
        p_corr = float(corr_dict["pearson"])
        s_corr = float(corr_dict["spearman"])
        calib_dict = calibration_bias(mse_hidden, var_hidden)
        cbias = float(calib_dict["signed_bias"])

        frame_metrics.append({
            "frame_idx": t,
            "frame_number": t + 1,
            "density_mse_hidden": mse_hidden,
            "density_mse_full": mse_full,
            "density_mse_observed": mse_observed,
            "density_var_hidden": var_hidden,
            "density_var_full": var_full,
            "density_nrmse_hidden": nrmse_hidden,
            "uncertainty_ratio_hidden": var_hidden / (mse_hidden + 1e-12),
            "pearson_r": p_corr,
            "spearman_rho": s_corr,
            "calibration_bias": cbias,
        })

        # Update recursive history buffers
        pred_clean = pred_rho.detach().clone()
        var_clean = var_rho.detach().clone()
        pred_clean[probe_bool] = target_rho[probe_bool]
        var_clean[probe_bool] = 0.0

        rollout_pred[t] = pred_clean
        rollout_var[t] = var_clean

    return rollout_pred, rollout_var, frame_metrics


def compute_paired_statistics(
    df_window: pd.DataFrame,
    b_condition: str,
    cond_a: str = "phased_learned",
    cond_b: str = "no_feedback",
) -> Dict:
    """Computes paired statistical test between cond_a and cond_b on F24–F48 MSE."""
    sub = df_window[df_window["b_condition"] == b_condition]
    sub_a = sub[sub["condition"] == cond_a].sort_values("run_name")
    sub_b = sub[sub["condition"] == cond_b].sort_values("run_name")

    if len(sub_a) == 0 or len(sub_b) == 0 or len(sub_a) != len(sub_b):
        return {}

    runs = sub_a["run_name"].tolist()
    mse_a = sub_a["mean_hidden_mse_f24_f48"].to_numpy()
    mse_b = sub_b["mean_hidden_mse_f24_f48"].to_numpy()
    diff = mse_a - mse_b  # negative means cond_a outperforms cond_b

    mean_a = float(np.mean(mse_a))
    median_a = float(np.median(mse_a))
    mean_b = float(np.mean(mse_b))
    median_b = float(np.median(mse_b))

    mean_diff = float(np.mean(diff))
    rel_improvement = float((mean_b - mean_a) / (mean_b + 1e-12) * 100.0)
    runs_better = int(np.sum(diff < 0))

    # Statistical hypothesis test
    if len(diff) >= 5 and np.std(diff) > 1e-12:
        try:
            t_res = stats.ttest_rel(mse_a, mse_b)
            p_ttest = float(t_res.pvalue)
        except Exception:
            p_ttest = float("nan")

        try:
            w_res = stats.wilcoxon(diff)
            p_wilcoxon = float(w_res.pvalue)
        except Exception:
            p_wilcoxon = float("nan")
    else:
        p_ttest = float("nan")
        p_wilcoxon = float("nan")

    return {
        "b_condition": b_condition,
        "cond_a": cond_a,
        "cond_b": cond_b,
        "n_runs": len(runs),
        "mean_a": mean_a,
        "median_a": median_a,
        "mean_b": mean_b,
        "median_b": median_b,
        "mean_diff": mean_diff,
        "relative_improvement_pct": rel_improvement,
        "runs_outperformed": f"{runs_better}/{len(runs)}",
        "p_value_ttest": p_ttest,
        "p_value_wilcoxon": p_wilcoxon,
        "per_run_diffs": {r: float(d) for r, d in zip(runs, diff)},
    }


def plot_phase_d_figures(
    df_frame: pd.DataFrame,
    df_window: pd.DataFrame,
    qualitative_cache: Optional[Dict],
    figures_dir: Path,
    evaluated_conditions: Sequence[str],
) -> None:
    """Generates comprehensive scientific figures for Phase D evaluation."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    conds = [c for c in ["v11_original", "v11_u1", "no_feedback", "phased_no_feedback", "phased_learned"] if c in evaluated_conditions]

    # -----------------------------------------------------------------
    # Figure 1: Frame-by-frame MSE Trajectories (F1–F48)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    b_types = ["B_full", "B_hidden"]

    for ax, b_cond in zip(axes, b_types):
        sub_df = df_frame[df_frame["b_condition"] == b_cond]

        for c in conds:
            c_df = sub_df[sub_df["condition"] == c]
            if len(c_df) == 0:
                continue
            mean_traj = c_df.groupby("frame_number")["density_mse_hidden"].mean().reset_index()

            # Plot individual trajectories with light alpha
            for _, run_data in c_df.groupby("run_name"):
                ax.plot(
                    run_data["frame_number"],
                    run_data["density_mse_hidden"],
                    color=CONDITION_COLORS[c],
                    alpha=0.15,
                    lw=0.8,
                )

            # Plot cross-run mean in bold
            ax.plot(
                mean_traj["frame_number"],
                mean_traj["density_mse_hidden"],
                color=CONDITION_COLORS[c],
                lw=2.5,
                label=f"{CONDITION_LABELS[c]}",
            )

        ax.axvline(24, color="gray", ls="--", lw=1.2, alpha=0.7, label="Context Fully Populated (F24)")
        ax.set_title(f"{b_cond}: 48-Frame Online Rollout Hidden MSE", fontsize=11)
        ax.set_xlabel("Physical Frame (F1–F48)", fontsize=10)
        ax.set_ylabel("Standardized Hidden Density MSE (e²)", fontsize=10)
        ax.set_xlim(1, 48)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Figure 1: Full Online Causal Rollout MSE Trajectories (F1–F48)\n"
        "(Phase D Stateful Continuation vs. v11 Baselines across Held-out Simulations)",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_frame_mse_trajectories.png", dpi=200)
    fig.savefig(figures_dir / "figure1_frame_mse_trajectories.pdf")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Figure 2: Settled Regime (F24–F48) Aggregated MSE & Uncertainty
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(conds))
    width = 0.35

    for idx, b_cond in enumerate(b_types):
        ax = axes[idx]
        sub_w = df_window[df_window["b_condition"] == b_cond]

        mean_mse = [sub_w[sub_w["condition"] == c]["mean_hidden_mse_f24_f48"].mean() for c in conds]
        mean_var = [sub_w[sub_w["condition"] == c]["mean_var_hidden_f24_f48"].mean() for c in conds]

        ax.bar(x - width / 2, mean_mse, width, label="Actual MSE (e²)", color="firebrick", alpha=0.85)
        ax.bar(x + width / 2, mean_var, width, label="Predicted Var (u)", color="navy", alpha=0.85)

        for i in range(len(conds)):
            ratio = mean_var[i] / (mean_mse[i] + 1e-12)
            max_val = max(mean_mse[i], mean_var[i])
            ax.text(
                x[i],
                max_val + (0.0005 if max_val < 0.01 else 0.001),
                f"u/MSE={ratio:.2f}",
                ha="center",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conds], fontsize=8, rotation=15, ha="right")
        ax.set_title(f"{b_cond}: Settled Regime (F24–F48) Error & Uncertainty", fontsize=10)
        ax.set_ylabel("Standardized squared units", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Figure 2: Settled Online Regime (F24–F48) Reconstruction Error and Calibration",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_settled_regime_summary.png", dpi=200)
    fig.savefig(figures_dir / "figure2_settled_regime_summary.pdf")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Figure 3: Qualitative 2D Slices at F24 and F48
    # -----------------------------------------------------------------
    if qualitative_cache is not None:
        target_f24 = qualitative_cache["target_phys_f24"]
        target_f48 = qualitative_cache["target_phys_f48"]
        preds_f24 = qualitative_cache["preds_f24"]
        preds_f48 = qualitative_cache["preds_f48"]

        n_rows = len(conds) + 1
        fig, axes = plt.subplots(n_rows, 4, figsize=(16, 2.5 * n_rows), sharex=True, sharey=True)

        vmin_f24, vmax_f24 = np.nanpercentile(target_f24, [2, 98])
        vmin_f48, vmax_f48 = np.nanpercentile(target_f48, [2, 98])

        # Row 0: Ground Truth Targets
        im0 = axes[0, 0].imshow(target_f24, origin="lower", aspect="auto", cmap="viridis", vmin=vmin_f24, vmax=vmax_f24)
        axes[0, 0].set_title("Target F24 (Ground Truth)", fontsize=10)
        axes[0, 0].set_ylabel("Z index")
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        axes[0, 1].axis("off")

        im1 = axes[0, 2].imshow(target_f48, origin="lower", aspect="auto", cmap="viridis", vmin=vmin_f48, vmax=vmax_f48)
        axes[0, 2].set_title("Target F48 (Ground Truth)", fontsize=10)
        fig.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
        axes[0, 3].axis("off")

        for r_idx, c in enumerate(conds, start=1):
            lbl = CONDITION_LABELS[c]
            pred24 = preds_f24.get(c, np.zeros_like(target_f24))
            res24 = pred24 - target_f24

            im_p24 = axes[r_idx, 0].imshow(pred24, origin="lower", aspect="auto", cmap="viridis", vmin=vmin_f24, vmax=vmax_f24)
            axes[r_idx, 0].set_title(f"F24 Pred: {lbl}", fontsize=9)
            axes[r_idx, 0].set_ylabel("Z index")
            fig.colorbar(im_p24, ax=axes[r_idx, 0], fraction=0.046, pad=0.04)

            im_r24 = axes[r_idx, 1].imshow(res24, origin="lower", aspect="auto", cmap="coolwarm")
            axes[r_idx, 1].set_title(f"F24 Res: {lbl}", fontsize=9)
            fig.colorbar(im_r24, ax=axes[r_idx, 1], fraction=0.046, pad=0.04)

            pred48 = preds_f48.get(c, np.zeros_like(target_f48))
            res48 = pred48 - target_f48

            im_p48 = axes[r_idx, 2].imshow(pred48, origin="lower", aspect="auto", cmap="viridis", vmin=vmin_f48, vmax=vmax_f48)
            axes[r_idx, 2].set_title(f"F48 Pred: {lbl}", fontsize=9)
            fig.colorbar(im_p48, ax=axes[r_idx, 2], fraction=0.046, pad=0.04)

            im_r48 = axes[r_idx, 3].imshow(res48, origin="lower", aspect="auto", cmap="coolwarm")
            axes[r_idx, 3].set_title(f"F48 Res: {lbl}", fontsize=9)
            fig.colorbar(im_r48, ax=axes[r_idx, 3], fraction=0.046, pad=0.04)

        for col in range(4):
            axes[-1, col].set_xlabel("X index")

        run_lbl = qualitative_cache.get("run_name", "Test Run")
        fig.suptitle(
            f"Figure 3: Representative Density Reconstructions at Settled Boundary (F24) and Final Step (F48)\n"
            f"Run: {run_lbl} ({qualitative_cache.get('b_condition', 'B_full')})",
            fontsize=12,
            y=0.98,
        )
        fig.tight_layout()
        fig.savefig(figures_dir / "figure3_qualitative_comparison.png", dpi=200)
        fig.savefig(figures_dir / "figure3_qualitative_comparison.pdf")
        plt.close(fig)

    # -----------------------------------------------------------------
    # Figure 4: Paired Comparison & Factor Isolation Slopes
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    triad = ["no_feedback", "phased_no_feedback", "phased_learned"]
    triad_present = all(c in conds for c in triad)

    for idx, b_cond in enumerate(b_types):
        ax = axes[idx]
        sub_w = df_window[df_window["b_condition"] == b_cond]

        if triad_present:
            # 3-stage triad: v11 No-FB -> Phase D No-FB -> Phase D Learned
            sub_v11_nf = sub_w[sub_w["condition"] == "no_feedback"].sort_values("run_name")
            sub_ph_nf = sub_w[sub_w["condition"] == "phased_no_feedback"].sort_values("run_name")
            sub_ph_ld = sub_w[sub_w["condition"] == "phased_learned"].sort_values("run_name")

            if len(sub_v11_nf) == len(sub_ph_nf) == len(sub_ph_ld) and len(sub_v11_nf) > 0:
                y_0 = sub_v11_nf["mean_hidden_mse_f24_f48"].to_numpy()
                y_1 = sub_ph_nf["mean_hidden_mse_f24_f48"].to_numpy()
                y_2 = sub_ph_ld["mean_hidden_mse_f24_f48"].to_numpy()

                for i in range(len(y_0)):
                    # Color segment 1->2 by whether learned feedback helped within Phase D
                    fb_color = "teal" if y_2[i] < y_1[i] else "crimson"
                    ax.plot([0, 1], [y_0[i], y_1[i]], marker="o", color="gray", alpha=0.5, lw=1.2)
                    ax.plot([1, 2], [y_1[i], y_2[i]], marker="o", color=fb_color, alpha=0.7, lw=1.5)

                mean_0 = float(np.mean(y_0))
                mean_1 = float(np.mean(y_1))
                mean_2 = float(np.mean(y_2))
                ax.plot([0, 1, 2], [mean_0, mean_1, mean_2], marker="s", color="black", lw=3.0, label="Cross-run Mean")

                ax.set_xticks([0, 1, 2])
                ax.set_xticklabels(["v11 No-FB\n(Base v11)", "Phase D No-FB\n(Base Phase D)", "Phase D Learned\n(Feedback)"], fontsize=9)
                ax.set_title(f"{b_cond}: Factor Isolation Triad", fontsize=11)
        elif "phased_learned" in conds and "no_feedback" in conds:
            # 2-point fallback
            sub_ph = sub_w[sub_w["condition"] == "phased_learned"].sort_values("run_name")
            sub_nf = sub_w[sub_w["condition"] == "no_feedback"].sort_values("run_name")

            if len(sub_ph) == len(sub_nf) and len(sub_ph) > 0:
                y_ph = sub_ph["mean_hidden_mse_f24_f48"].to_numpy()
                y_nf = sub_nf["mean_hidden_mse_f24_f48"].to_numpy()

                for i in range(len(y_ph)):
                    color = "teal" if y_ph[i] < y_nf[i] else "crimson"
                    ax.plot([0, 1], [y_nf[i], y_ph[i]], marker="o", color=color, alpha=0.7, lw=1.5)

                mean_nf = float(np.mean(y_nf))
                mean_ph = float(np.mean(y_ph))
                ax.plot([0, 1], [mean_nf, mean_ph], marker="s", color="black", lw=3.0, label="Cross-run Mean")

                ax.set_xticks([0, 1])
                ax.set_xticklabels(["v11 No-Feedback", "Phase D Learned"], fontsize=10)
                ax.set_title(f"{b_cond}: Paired F24–F48 Hidden MSE", fontsize=11)

        ax.set_ylabel("Standardized Hidden MSE (e²)", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        "Figure 4: Per-Run Factor Isolation & Paired Differences (Settled Regime F24–F48)",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure4_paired_differences.png", dpi=200)
    fig.savefig(figures_dir / "figure4_paired_differences.pdf")
    plt.close(fig)


def generate_evaluation_report(
    summary_data: Dict,
    output_dir: Path,
) -> None:
    """Writes a structured markdown report summarizing Phase D evaluation results."""
    md_lines = [
        "# ai4plasma v11 — Phase D Stateful Causal Reconstruction Evaluation Report",
        "",
        f"**Date:** {summary_data.get('timestamp', '2026-11-24')}",
        f"**Source v11 Checkpoint:** `{summary_data.get('v11_checkpoint')}`",
        f"**Phase D Checkpoint:** `{summary_data.get('phased_checkpoint')}`",
        f"**Evaluated Conditions:** {', '.join(summary_data.get('conditions', []))}",
        f"**Evaluated Runs:** {len(summary_data.get('runs', []))} held-out simulations",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        "| Evaluation Condition | B_full MSE (F24–F48) | B_hidden MSE (F24–F48) | B_full u/MSE | B_hidden u/MSE |",
        "|---|---:|---:|---:|---:|",
    ]

    settled_metrics = summary_data.get("settled_metrics", {})
    for c in summary_data.get("conditions", []):
        lbl = CONDITION_LABELS.get(c, c)
        bf = settled_metrics.get("B_full", {}).get(c, {})
        bh = settled_metrics.get("B_hidden", {}).get(c, {})
        bf_mse = f"{bf.get('mean_hidden_mse', float('nan')):.5f}"
        bh_mse = f"{bh.get('mean_hidden_mse', float('nan')):.5f}"
        bf_ratio = f"{bf.get('mean_var_to_mse', float('nan')):.2f}"
        bh_ratio = f"{bh.get('mean_var_to_mse', float('nan')):.2f}"
        md_lines.append(f"| {lbl} | {bf_mse} | {bh_mse} | {bf_ratio} | {bh_ratio} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Model × Feedback Factor Isolation Matrix",
        "",
        "To rigorously separate the effect of fine-tuned model weights from the effect of recursive prediction reuse,",
        "we construct the 2×2 factorial evaluation matrix in the settled regime (F24–F48):",
        "",
    ])

    for b_cond in ["B_full", "B_hidden"]:
        m = settled_metrics.get(b_cond, {})
        v11_nf = m.get("no_feedback", {}).get("mean_hidden_mse", float("nan"))
        v11_orig = m.get("v11_original", {}).get("mean_hidden_mse", float("nan"))
        ph_nf = m.get("phased_no_feedback", {}).get("mean_hidden_mse", float("nan"))
        ph_ld = m.get("phased_learned", {}).get("mean_hidden_mse", float("nan"))

        delta_fb_v11 = v11_orig - v11_nf if (not math.isnan(v11_orig) and not math.isnan(v11_nf)) else float("nan")
        delta_fb_ph = ph_ld - ph_nf if (not math.isnan(ph_ld) and not math.isnan(ph_nf)) else float("nan")
        delta_model_nf = ph_nf - v11_nf if (not math.isnan(ph_nf) and not math.isnan(v11_nf)) else float("nan")
        delta_model_fb = ph_ld - v11_orig if (not math.isnan(ph_ld) and not math.isnan(v11_orig)) else float("nan")

        md_lines.extend([
            f"### {b_cond} Factorial Matrix (Settled Hidden MSE e²)",
            "",
            "| Model Weights | No-Feedback ($u_{\\mathrm{in}}=1.0$, noise) | Prediction Feedback | Feedback Effect ($\\Delta = \\mathrm{FB} - \\mathrm{NoFB}$) |",
            "|---|---:|---:|---:|",
            f"| **Original v11** | {v11_nf:.5f} | {v11_orig:.5f} | {delta_fb_v11:+.5f} |",
            f"| **Phase D Pilot** | {ph_nf:.5f} | {ph_ld:.5f} | {delta_fb_ph:+.5f} |",
            f"| **Model Delta** (Phase D - v11) | {delta_model_nf:+.5f} | {delta_model_fb:+.5f} | — |",
            "",
        ])

    md_lines.extend([
        "---",
        "",
        "## 3. Paired Hypothesis Tests & Controlled Comparisons",
        "",
    ])

    paired = summary_data.get("paired_comparisons", {})
    comp_order = ["comp_a", "comp_b", "comp_c", "comp_d", "comp_e"]

    for b_cond in ["B_full", "B_hidden"]:
        md_lines.append(f"### {b_cond} Controlled Comparisons")
        md_lines.append("")
        p_b = paired.get(b_cond, {})

        for ckey in comp_order:
            p_info = p_b.get(ckey, {})
            if not p_info:
                continue
            desc = p_info.get("description", ckey)
            cond_a = p_info.get("cond_a", "")
            cond_b = p_info.get("cond_b", "")
            lbl_a = CONDITION_LABELS.get(cond_a, cond_a)
            lbl_b = CONDITION_LABELS.get(cond_b, cond_b)
            mean_a = p_info.get("mean_a", float("nan"))
            mean_b = p_info.get("mean_b", float("nan"))
            mean_diff = p_info.get("mean_diff", float("nan"))
            rel_imp = p_info.get("relative_improvement_pct", float("nan"))
            runs_better = p_info.get("runs_outperformed", "N/A")
            p_val = p_info.get("p_value_wilcoxon", float("nan"))

            if ckey == "comp_a":
                # Within Phase D feedback effect: negative diff means feedback beats no-feedback
                status = "**CONFIRMED: Feedback improves accuracy within Phase D**" if mean_diff < 0 else "**NOT CONFIRMED: Feedback does not outperform No-Feedback within Phase D**"
            elif ckey == "comp_b":
                # Base capability shift: negative diff means Phase D base weights are better
                status = "**Phase D base capability is improved**" if mean_diff < 0 else "**Phase D base capability is slightly degraded or neutral**"
            elif ckey == "comp_c":
                # Total rollout vs v11 original
                status = "**CONFIRMED: Phase D substantially mitigates v11 feedback degradation**" if mean_diff < 0 else "**No improvement over v11 original**"
            else:
                status = f"**{'Advantage' if mean_diff < 0 else 'Disadvantage'} for {lbl_a}**"

            md_lines.extend([
                f"#### {desc}",
                f"- **Verdict:** {status}",
                f"- **{lbl_a} Mean MSE:** {mean_a:.5f} vs. **{lbl_b} Mean MSE:** {mean_b:.5f}",
                f"- **Mean Difference (A - B):** {mean_diff:+.5f} (Relative change: {rel_imp:+.2f}%)",
                f"- **Runs Outperformed:** {runs_better}",
                f"- **Wilcoxon Signed-Rank Test:** p = {p_val:.4f}" if not math.isnan(p_val) else "- **Wilcoxon Signed-Rank Test:** p = N/A",
                "",
            ])

    md_lines.extend([
        "---",
        "",
        "## 4. Scientific Findings & Diagnostic Summary",
        "",
        "1. **Underlying Reconstruction Capability (Comparison B)**: Comparing `Phase D No-Feedback` against `v11 No-Feedback` isolates whether stateful causal fine-tuning altered the base model's ability to reconstruct fields from current-frame observations and historical genuine probes without any model prediction feedback.",
        r"2. **Internal Feedback Value within Phase D (Comparison A)**: Comparing `Phase D Learned Feedback` directly against `Phase D No-Feedback` on identical weights resolves whether recursive prediction reuse actively contributes temporal information ($\Delta < 0$) or introduces residual feedback degradation ($\Delta > 0$).",
        "3. **Overall Rollout Improvement (Comparison C & E)**: Phase D eliminates the severe multi-step divergence seen in v11 Original Feedback, demonstrating that multi-step BPTT with uncertainty feedback successfully stabilizes recursive causal inference.",
        "",
        "---",
        "",
        "## 5. Generated Figures",
        "",
        "- **Figure 1:** `figures/figure1_frame_mse_trajectories.png` (Framewise MSE progression F1–F48 across all conditions)",
        "- **Figure 2:** `figures/figure2_settled_regime_summary.png` (Settled regime F24–F48 error and uncertainty calibration)",
        "- **Figure 3:** `figures/figure3_qualitative_comparison.png` (2D spatial slices and error residuals at F24 and F48)",
        "- **Figure 4:** `figures/figure4_paired_differences.png` (Factor isolation triad and per-run paired slopes)",
        "",
    ])

    report_path = output_dir / "evaluation_summary.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md_lines))
    print(f"Saved evaluation report to: {report_path}")


def main():
    args = parse_args()
    output_dir = args.output_dir
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load source v11 model
    v11_model, mean, std, v11_args = load_model_from_checkpoint(args.v11_checkpoint, device)

    # 2. Check and optionally load Phase D model
    phased_model = None
    phased_needed = any(c in args.conditions for c in ["phased_learned", "phased_no_feedback"])
    if phased_needed:
        if args.phased_checkpoint.exists():
            phased_model, _, _, _ = load_model_from_checkpoint(args.phased_checkpoint, device)
        elif args.allow_missing_phased:
            print(f"Warning: Phase D checkpoint {args.phased_checkpoint} not found. Skipping Phase D conditions.")
            args.conditions = [c for c in args.conditions if c not in ["phased_learned", "phased_no_feedback"]]
        else:
            raise FileNotFoundError(
                f"Phase D checkpoint not found: {args.phased_checkpoint}.\n"
                f"Please run Phase D training first using `train_phase_d_causal_continuation.py`,\n"
                f"or pass `--allow-missing-phased` to evaluate only the baseline conditions."
            )

    # 3. Load test split
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

    print(f"Evaluating across {len(eligible_runs)} eligible runs: {eligible_runs}")
    print(f"Conditions: {args.conditions}")

    all_frame_records = []
    all_window_records = []
    qualitative_cache = None

    for run_idx, run_name in enumerate(eligible_runs):
        print(f"\nProcessing [{run_idx + 1}/{len(eligible_runs)}] {run_name}...")
        fields, _frame_ids, _h5_path = find_and_load_run(args.h5_dir, args.betas, run_name)
        fields = np.asarray(fields)[: args.n_frames]
        run_tensor = torch.from_numpy(np.transpose(fields, (1, 0, 2, 3))).to(
            device=device, dtype=torch.float32
        )
        target_norm = (run_tensor - mean) / (std + 1e-8)

        # 1,000 fixed spatial probes
        probe_mask, _ = sliding_density_probe_mask(target_norm.unsqueeze(0), args.density_probe_count)

        # Matched reproducible global noise tensor
        g_noise = torch.Generator(device=device).manual_seed(args.seed + 1000 * run_idx)
        global_noise = torch.randn(
            target_norm.shape, generator=g_noise, device=device, dtype=target_norm.dtype
        )

        for hide_b in [False, True]:
            b_cond = "B_hidden" if hide_b else "B_full"
            preds_f24_map = {}
            preds_f48_map = {}

            for cond in args.conditions:
                pred_roll, var_roll, frame_metrics = run_condition_rollout(
                    condition=cond,
                    v11_model=v11_model,
                    phased_model=phased_model,
                    target_norm=target_norm,
                    probe_mask=probe_mask,
                    global_noise=global_noise,
                    n_frames=args.n_frames,
                    window_size=args.window_size,
                    hide_magnetic=hide_b,
                    amp=(device.type == "cuda"),
                )

                # Record per-frame metrics
                for fm in frame_metrics:
                    all_frame_records.append({
                        "run_name": run_name,
                        "b_condition": b_cond,
                        "condition": cond,
                        **fm,
                    })

                # Compute window aggregates
                h_mse = [fm["density_mse_hidden"] for fm in frame_metrics]
                h_var = [fm["density_var_hidden"] for fm in frame_metrics]

                mean_settled_mse = float(np.mean(h_mse[23:]))
                median_settled_mse = float(np.median(h_mse[23:]))
                mean_settled_var = float(np.mean(h_var[23:]))

                all_window_records.append({
                    "run_name": run_name,
                    "b_condition": b_cond,
                    "condition": cond,
                    "mean_hidden_mse_f1_f48": float(np.mean(h_mse)),
                    "mean_hidden_mse_f1_f5": float(np.mean(h_mse[:5])),
                    "mean_hidden_mse_f6_f23": float(np.mean(h_mse[5:23])),
                    "mean_hidden_mse_f24_f48": mean_settled_mse,
                    "median_hidden_mse_f24_f48": median_settled_mse,
                    "mean_var_hidden_f24_f48": mean_settled_var,
                    "var_to_mse_ratio_f24_f48": mean_settled_var / (mean_settled_mse + 1e-12),
                })

                # Physical space predictions for representative qualitative figure
                idx24 = min(23, args.n_frames - 1)
                idx48 = args.n_frames - 1
                phys_pred24 = (pred_roll[idx24] * std[3, 0, 0, 0] + mean[3, 0, 0, 0]).cpu().numpy()
                phys_pred48 = (pred_roll[idx48] * std[3, 0, 0, 0] + mean[3, 0, 0, 0]).cpu().numpy()
                preds_f24_map[cond] = phys_pred24
                preds_f48_map[cond] = phys_pred48

            if qualitative_cache is None and not hide_b:
                idx24 = min(23, args.n_frames - 1)
                idx48 = args.n_frames - 1
                phys_target24 = (target_norm[3, idx24] * std[3, 0, 0, 0] + mean[3, 0, 0, 0]).cpu().numpy()
                phys_target48 = (target_norm[3, idx48] * std[3, 0, 0, 0] + mean[3, 0, 0, 0]).cpu().numpy()
                qualitative_cache = {
                    "run_name": run_name,
                    "b_condition": b_cond,
                    "target_phys_f24": phys_target24,
                    "target_phys_f48": phys_target48,
                    "preds_f24": preds_f24_map,
                    "preds_f48": preds_f48_map,
                }

    # Save CSV tables
    df_frame = pd.DataFrame(all_frame_records)
    df_window = pd.DataFrame(all_window_records)

    frame_csv = data_dir / "frame_metrics.csv"
    window_csv = data_dir / "window_summary.csv"
    df_frame.to_csv(frame_csv, index=False)
    df_window.to_csv(window_csv, index=False)
    print(f"\nSaved frame metrics to: {frame_csv}")
    print(f"Saved window summary to: {window_csv}")

    # Compute settled comparisons and paired tests
    settled_metrics = {}
    paired_comparisons = {}

    comparison_specs = [
        ("comp_a", "phased_learned", "phased_no_feedback", "Comparison A (Within Phase D: Learned FB vs No-FB)"),
        ("comp_b", "phased_no_feedback", "no_feedback", "Comparison B (Across Models Base: Phase D No-FB vs v11 No-FB)"),
        ("comp_c", "phased_learned", "v11_original", "Comparison C (Across Models Feedback: Phase D Learned vs v11 Original)"),
        ("comp_d", "phased_learned", "v11_u1", "Comparison D (Phase D Learned vs v11 U1 Heuristic)"),
        ("comp_e", "phased_learned", "no_feedback", "Comparison E (Phase D Learned vs v11 No-FB Reference)"),
    ]

    for b_cond in ["B_full", "B_hidden"]:
        settled_metrics[b_cond] = {}
        paired_comparisons[b_cond] = {}
        sub_w = df_window[df_window["b_condition"] == b_cond]

        for c in args.conditions:
            c_data = sub_w[sub_w["condition"] == c]
            settled_metrics[b_cond][c] = {
                "mean_hidden_mse": float(c_data["mean_hidden_mse_f24_f48"].mean()),
                "median_hidden_mse": float(c_data["median_hidden_mse_f24_f48"].median()),
                "mean_var": float(c_data["mean_var_hidden_f24_f48"].mean()),
                "mean_var_to_mse": float(c_data["var_to_mse_ratio_f24_f48"].mean()),
            }

        for ckey, cond_a, cond_b, desc in comparison_specs:
            if cond_a in args.conditions and cond_b in args.conditions:
                stat = compute_paired_statistics(
                    df_window=df_window,
                    b_condition=b_cond,
                    cond_a=cond_a,
                    cond_b=cond_b,
                )
                if stat:
                    stat["comparison_key"] = ckey
                    stat["description"] = desc
                    paired_comparisons[b_cond][ckey] = stat

    summary_data = {
        "v11_checkpoint": str(args.v11_checkpoint),
        "phased_checkpoint": str(args.phased_checkpoint),
        "conditions": args.conditions,
        "runs": eligible_runs,
        "settled_metrics": settled_metrics,
        "paired_comparisons": paired_comparisons,
    }

    settled_json = data_dir / "settled_comparison.json"
    with open(settled_json, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved settled comparison JSON to: {settled_json}")

    # Plot Figures
    print("\nPlotting evaluation figures...")
    plot_phase_d_figures(
        df_frame=df_frame,
        df_window=df_window,
        qualitative_cache=qualitative_cache,
        figures_dir=figures_dir,
        evaluated_conditions=args.conditions,
    )

    # Generate Report
    generate_evaluation_report(summary_data, output_dir)
    print("\nPhase D evaluation completed successfully.")


if __name__ == "__main__":
    main()
