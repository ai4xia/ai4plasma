#!/usr/bin/env python3
"""Full Online Causal Rollout Uncertainty Ablation for ai4plasma v11.

Evaluates whether assigning u_in=1.0 to reused historical predictions (U1 strategy)
consistently improves reconstruction throughout a full 48-frame online causal rollout,
compared to the standard recursive rollout (Original) and independent reconstruction (No-feedback).

Three online strategies evaluated across all 48 frames:
  1. Original: Standard recursive online rollout reusing prior Density predictions
     rho_hat and predicted uncertainties u_hat without modification.
  2. U1: Recursive rollout where prior predictions rho_hat are reused with u_in=1.0
     at unobserved locations. Crucially, U1 recomputes its own predictions at every step.
  3. No-feedback: Independent reconstruction at each step. Historical unobserved Density
     is filled with Gaussian noise and u_in=1.0 (no prediction reuse).

Invariants:
  - All 48 frames evaluated (T=24 context window, causal padding for t < 23).
  - 10 held-out beta=0.20 simulations, 1,000 fixed Density probes, seed 42.
  - Both B_full and B_hidden evaluated.
  - Strict noise matching using reproducible global noise tensor per run.
  - Genuine probes remain exact with u_in=0.0 across all strategies and frames.
  - Current-frame inputs at local index 23 are bitwise identical across strategies.
  - No future data leakage or target exposure.
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
from run_feedback_ablation import (
    load_model_and_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full online causal rollout uncertainty ablation for ai4plasma v11."
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
        default=Path("evaluations/physworld_v11_full_online_uncertainty_ablation_20261123"),
        help="Output directory for full online rollout results.",
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
        "--strategies",
        type=str,
        nargs="+",
        default=["original", "u1", "no_feedback"],
        choices=["original", "u1", "no_feedback"],
        help="Online rollout strategies to evaluate.",
    )
    return parser.parse_args()


def build_rollout_step_input(
    t: int,
    strategy: str,
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    global_noise: torch.Tensor,
    history_pred: torch.Tensor,
    history_var: torch.Tensor,
    hide_magnetic: bool = False,
    window_size: int = 24,
) -> torch.Tensor:
    """Constructs the [1, 8, window_size, H, W] input for causal frame t.

    Window spans [t - window_size + 1, ..., t].
    Local index (window_size - 1) is always target frame t.

    Conditioning policies:
      - tau < 0: Deterministic reproducible Gaussian noise with u_in=1.0.
      - tau <= t (valid frame):
          Magnetic (channels 0..2):
            * Not hide_magnetic (B_full): target[:3, tau], u_in = 0.0
            * Hide_magnetic (B_hidden): global_noise[:3, tau], u_in = 1.0
          Density (channel 3):
            * Current frame tau == t:
                Probes: target[3, tau], u_in = 0.0
                Unobserved: global_noise[3, tau], u_in = 1.0 (BITWISE IDENTICAL across strategies)
            * Historical frame tau < t:
                Probes: target[3, tau], u_in = 0.0
                Unobserved:
                  - 'original':    history_pred[tau], u_in = history_var[tau]
                  - 'u1':          history_pred[tau], u_in = 1.0
                  - 'no_feedback': global_noise[3, tau], u_in = 1.0
    """
    device = target_normalized.device
    dtype = target_normalized.dtype
    _, _, H, W = target_normalized.shape

    values = torch.zeros((1, 4, window_size, H, W), device=device, dtype=dtype)
    input_variance = torch.ones_like(values)
    probe_bool = (probe_mask > 0.5)

    for l in range(window_size):
        tau = t - window_size + 1 + l
        if tau < 0:
            # Nonexistent historical frame: sample reproducible noise with variance 1.0
            g_pad = torch.Generator(device=device)
            g_pad.manual_seed((10007 * tau + 54321) & 0x7FFFFFFF)
            values[0, :, l] = torch.randn((4, H, W), generator=g_pad, device=device, dtype=dtype)
            input_variance[0, :, l] = 1.0
        else:
            # 1. Magnetic components
            if not hide_magnetic:
                values[0, :3, l] = target_normalized[:3, tau]
                input_variance[0, :3, l] = 0.0
            else:
                values[0, :3, l] = global_noise[:3, tau]
                input_variance[0, :3, l] = 1.0

            # 2. Density component
            if tau == t:
                # Current frame: probes known, unobserved is global_noise with variance 1.0
                values[0, 3, l] = global_noise[3, tau]
                input_variance[0, 3, l] = 1.0
                values[0, 3, l, probe_bool] = target_normalized[3, tau, probe_bool]
                input_variance[0, 3, l, probe_bool] = 0.0
            else:
                # Historical frame tau < t
                if strategy == "original":
                    values[0, 3, l] = history_pred[tau]
                    input_variance[0, 3, l] = history_var[tau]
                elif strategy == "u1":
                    values[0, 3, l] = history_pred[tau]
                    input_variance[0, 3, l] = 1.0
                elif strategy == "no_feedback":
                    values[0, 3, l] = global_noise[3, tau]
                    input_variance[0, 3, l] = 1.0
                else:
                    raise ValueError(f"Unknown rollout strategy: {strategy}")

                # Genuine probes always override with zero variance
                values[0, 3, l, probe_bool] = target_normalized[3, tau, probe_bool]
                input_variance[0, 3, l, probe_bool] = 0.0

    return torch.cat([values, input_variance], dim=1)


@torch.no_grad()
def run_online_causal_rollout(
    strategy: str,
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    global_noise: torch.Tensor,
    n_frames: int = 48,
    window_size: int = 24,
    hide_magnetic: bool = False,
    amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Executes a full n_frames causal rollout under the specified strategy.

    Returns:
      rollout_pred: [n_frames, H, W] in normalized space
      rollout_var:  [n_frames, H, W] in normalized space
      frame_metrics: List of dicts with metrics per frame t
    """
    device = target_normalized.device
    dtype = target_normalized.dtype
    _, _, H, W = target_normalized.shape

    rollout_pred = torch.zeros((n_frames, H, W), device=device, dtype=dtype)
    rollout_var = torch.ones((n_frames, H, W), device=device, dtype=dtype)

    current_idx = window_size - 1
    probe_bool = (probe_mask > 0.5)
    hidden_mask = ~probe_bool

    frame_metrics = []

    for t in range(n_frames):
        model_input = build_rollout_step_input(
            t=t,
            strategy=strategy,
            target_normalized=target_normalized,
            probe_mask=probe_mask,
            global_noise=global_noise,
            history_pred=rollout_pred,
            history_var=rollout_var,
            hide_magnetic=hide_magnetic,
            window_size=window_size,
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

        density_pred_t = mean_pred[0, 3, current_idx].float()
        density_var_t = var_pred[0, 3, current_idx].float()
        target_t = target_normalized[3, t].float()

        # Compute metrics before overwriting probes
        diff_all = density_pred_t - target_t
        e2_all = diff_all.square()
        full_field_rmse = float(e2_all.mean().sqrt().item())

        e2_hidden = e2_all[hidden_mask]
        var_hidden = density_var_t[hidden_mask]

        mean_e2_hidden = float(e2_hidden.mean().item())
        hidden_rmse = float(math.sqrt(mean_e2_hidden))
        mean_var_hidden = float(var_hidden.mean().item())
        u_ratio = float(mean_var_hidden / (mean_e2_hidden + 1e-12))

        e2_np = e2_hidden.detach().cpu().numpy().ravel()
        var_np = var_hidden.detach().cpu().numpy().ravel()
        corr_dict = pearson_spearman(var_np, e2_np)
        pearson_r = float(corr_dict["pearson"])
        spearman_r = float(corr_dict["spearman"])

        # Genuine probes override prediction for future rollout steps
        density_pred_t[probe_bool] = target_t[probe_bool]
        density_var_t[probe_bool] = 0.0

        rollout_pred[t] = density_pred_t
        rollout_var[t] = density_var_t

        frame_metrics.append({
            "frame_idx": t,
            "frame_number": t + 1,
            "strategy": strategy,
            "density_mse_hidden": mean_e2_hidden,
            "density_rmse_hidden": hidden_rmse,
            "full_field_rmse": full_field_rmse,
            "mean_var_hidden": mean_var_hidden,
            "u_ratio_hidden": u_ratio,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
        })

    return rollout_pred, rollout_var, frame_metrics


def plot_rollout_figures(
    df_frames: pd.DataFrame,
    df_window: pd.DataFrame,
    qualitative_cache: Optional[Dict],
    figures_dir: Path,
):
    """Generates the required figures for the full rollout uncertainty ablation."""
    colors = {
        "original": "#1f77b4",     # Blue
        "u1": "#2ca02c",           # Green
        "no_feedback": "#d62728",  # Red
    }
    strat_labels = {
        "original": "Original (u=u_hat)",
        "u1": "U1 (u=1.0)",
        "no_feedback": "No-feedback (Noise, u=1)",
    }
    strats = [s for s in ["original", "u1", "no_feedback"] if s in df_frames["strategy"].unique()]

    # -----------------------------------------------------------------
    # Figure 1: Frame-by-Frame MSE Trajectories (F1–F48)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub_c = df_frames[df_frames["b_condition"] == cond]

        for s in strats:
            sub_s = sub_c[sub_c["strategy"] == s]
            # Thin per-run trajectories
            runs = sub_s["run_name"].unique()
            for r in runs:
                sub_r = sub_s[sub_s["run_name"] == r].sort_values("frame_number")
                ax.plot(
                    sub_r["frame_number"],
                    sub_r["density_mse_hidden"],
                    color=colors[s],
                    alpha=0.18,
                    lw=0.8,
                )

            # Bold mean trajectory
            mean_traj = sub_s.groupby("frame_number")["density_mse_hidden"].mean().reset_index()
            ax.plot(
                mean_traj["frame_number"],
                mean_traj["density_mse_hidden"],
                color=colors[s],
                lw=2.5,
                label=f"{strat_labels[s]} Mean",
            )

        # Mark settled regime boundary (F24)
        ax.axvline(24, color="gray", ls="--", lw=1.2, alpha=0.7, label="Context Fully Populated (F24)")

        ax.set_title(f"{cond}: Full 48-Frame Online Rollout MSE", fontsize=11)
        ax.set_xlabel("Physical Frame (F1–F48)", fontsize=10)
        ax.set_ylabel("Standardized Hidden-Voxel MSE (e²)", fontsize=10)
        ax.set_xlim(1, 48)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Figure 1: Full Online Causal Rollout MSE Trajectories (F1–F48)\n"
        "(Original vs. U1 vs. No-Feedback across 10 Held-out Simulations)",
        fontsize=12,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_frame_mse_trajectories.png", dpi=200)
    fig.savefig(figures_dir / "figure1_frame_mse_trajectories.pdf")
    plt.close(fig)

    # -----------------------------------------------------------------
    # Figure 2: Representative Reconstruction Comparison at F24 and F48
    # -----------------------------------------------------------------
    if qualitative_cache is not None:
        target_f24 = qualitative_cache["target_phys_f24"]
        target_f48 = qualitative_cache["target_phys_f48"]
        preds_f24 = qualitative_cache["preds_f24"]
        preds_f48 = qualitative_cache["preds_f48"]

        fig, axes = plt.subplots(4, 4, figsize=(16, 10), sharex=True, sharey=True)

        vmin_f24, vmax_f24 = np.nanpercentile(target_f24, [2, 98])
        vmin_f48, vmax_f48 = np.nanpercentile(target_f48, [2, 98])

        # F24 Row 0: Target
        im0 = axes[0, 0].imshow(target_f24, origin="lower", aspect="auto", cmap="viridis", vmin=vmin_f24, vmax=vmax_f24)
        axes[0, 0].set_title("Target F24 (Ground Truth)", fontsize=10)
        axes[0, 0].set_ylabel("Z index")
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
        axes[0, 1].axis("off")

        # F48 Row 0: Target
        im1 = axes[0, 2].imshow(target_f48, origin="lower", aspect="auto", cmap="viridis", vmin=vmin_f48, vmax=vmax_f48)
        axes[0, 2].set_title("Target F48 (Ground Truth)", fontsize=10)
        fig.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
        axes[0, 3].axis("off")

        row_map = {"original": 1, "u1": 2, "no_feedback": 3}

        for s in strats:
            r = row_map[s]
            lbl = strat_labels[s]

            # F24 Pred & Residual
            pred24 = preds_f24[s]
            res24 = pred24 - target_f24
            im_p24 = axes[r, 0].imshow(pred24, origin="lower", aspect="auto", cmap="viridis", vmin=vmin_f24, vmax=vmax_f24)
            axes[r, 0].set_title(f"F24 Pred: {lbl}", fontsize=9)
            axes[r, 0].set_ylabel("Z index")
            fig.colorbar(im_p24, ax=axes[r, 0], fraction=0.046, pad=0.04)

            im_r24 = axes[r, 1].imshow(res24, origin="lower", aspect="auto", cmap="coolwarm")
            axes[r, 1].set_title(f"F24 Res: {lbl}", fontsize=9)
            fig.colorbar(im_r24, ax=axes[r, 1], fraction=0.046, pad=0.04)

            # F48 Pred & Residual
            pred48 = preds_f48[s]
            res48 = pred48 - target_f48
            im_p48 = axes[r, 2].imshow(pred48, origin="lower", aspect="auto", cmap="viridis", vmin=vmin_f48, vmax=vmax_f48)
            axes[r, 2].set_title(f"F48 Pred: {lbl}", fontsize=9)
            fig.colorbar(im_p48, ax=axes[r, 2], fraction=0.046, pad=0.04)

            im_r48 = axes[r, 3].imshow(res48, origin="lower", aspect="auto", cmap="coolwarm")
            axes[r, 3].set_title(f"F48 Res: {lbl}", fontsize=9)
            fig.colorbar(im_r48, ax=axes[r, 3], fraction=0.046, pad=0.04)

        for c in range(4):
            axes[3, c].set_xlabel("X index")

        run_lbl = qualitative_cache.get("run_name", "Test Run")
        fig.suptitle(
            f"Figure 2: Representative Reconstructions at Settled Boundary (F24) and Final Step (F48)\n"
            f"Run: {run_lbl} ({qualitative_cache.get('b_condition', 'B_full')})",
            fontsize=12,
            y=0.98,
        )
        fig.tight_layout()
        fig.savefig(figures_dir / "figure2_reconstruction_comparison_f24_f48.png", dpi=200)
        fig.savefig(figures_dir / "figure2_reconstruction_comparison_f24_f48.pdf")
        plt.close(fig)

    # -----------------------------------------------------------------
    # Figure 3: Settled Regime (F24–F48) Aggregated MSE & Calibration
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(strats))
    width = 0.35

    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub_w = df_window[df_window["b_condition"] == cond]

        mean_mse = [sub_w[sub_w["strategy"] == s]["mean_hidden_mse_f24_f48"].mean() for s in strats]
        mean_var = [sub_w[sub_w["strategy"] == s]["mean_var_hidden_f24_f48"].mean() for s in strats]

        ax.bar(x - width / 2, mean_mse, width, label="Actual MSE (e²)", color="firebrick", alpha=0.85)
        ax.bar(x + width / 2, mean_var, width, label="Predicted Var (u)", color="navy", alpha=0.85)

        for i in range(len(strats)):
            ratio = mean_var[i] / (mean_mse[i] + 1e-12)
            ax.text(
                x[i],
                max(mean_mse[i], mean_var[i]) + 0.0002,
                f"u/MSE={ratio:.2f}",
                ha="center",
                fontsize=8,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([strat_labels[s] for s in strats], fontsize=8)
        ax.set_title(f"{cond}: Settled Regime (F24–F48) Mean Error & Var", fontsize=10)
        ax.set_ylabel("Standardized squared units", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "Figure 3: Settled Online Regime (F24–F48) Reconstruction Error and Calibration",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure3_settled_regime_summary.png", dpi=200)
    fig.savefig(figures_dir / "figure3_settled_regime_summary.pdf")
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

    # Load model and normalization stats
    model, mean, std, ckpt_args = load_model_and_stats(args.run_dir, args.checkpoint, device)

    # Load split and filter eligible runs
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

    print(f"Evaluating full online rollout across {len(eligible_runs)} eligible runs: {eligible_runs}")

    b_conditions = ["B_full", "B_hidden"]
    strategies = [s for s in ["original", "u1", "no_feedback"] if s in args.strategies]

    all_frame_rows = []
    all_window_rows = []
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
            run_preds_f24 = {}
            run_preds_f48 = {}

            for strat in strategies:
                print(f"  [{b_cond}] Executing {strat} 48-frame rollout...")
                rollout_pred, rollout_var, frame_metrics = run_online_causal_rollout(
                    strategy=strat,
                    model=model,
                    mean=mean,
                    std=std,
                    target_normalized=run_tensor_norm,
                    probe_mask=probe_mask_2d,
                    global_noise=global_noise,
                    n_frames=args.n_frames,
                    window_size=args.window_size,
                    hide_magnetic=hide_magnetic,
                    amp=True,
                )

                # Attach run metadata
                for fm in frame_metrics:
                    fm["run_name"] = run_name
                    fm["b_condition"] = b_cond
                    all_frame_rows.append(fm)

                # Compute settled regime metrics (F24–F48, i.e. frame_idx 23 to 47)
                settled_metrics = [m for m in frame_metrics if m["frame_idx"] >= 23]
                settled_mse = [m["density_mse_hidden"] for m in settled_metrics]
                settled_var = [m["mean_var_hidden"] for m in settled_metrics]

                mean_settled_mse = float(np.mean(settled_mse))
                median_settled_mse = float(np.median(settled_mse))
                mean_settled_var = float(np.mean(settled_var))
                settled_u_ratio = float(mean_settled_var / (mean_settled_mse + 1e-12))

                print(
                    f"    [{strat}] F24-F48 Mean MSE: {mean_settled_mse:.5f} | "
                    f"Median MSE: {median_settled_mse:.5f} | "
                    f"Var u: {mean_settled_var:.5f} | "
                    f"u/MSE: {settled_u_ratio:.3f}"
                )

                all_window_rows.append({
                    "run_name": run_name,
                    "b_condition": b_cond,
                    "strategy": strat,
                    "mean_hidden_mse_f24_f48": mean_settled_mse,
                    "median_hidden_mse_f24_f48": median_settled_mse,
                    "mean_var_hidden_f24_f48": mean_settled_var,
                    "u_ratio_f24_f48": settled_u_ratio,
                    "mse_f24": frame_metrics[23]["density_mse_hidden"] if len(frame_metrics) > 23 else np.nan,
                    "mse_f48": frame_metrics[args.n_frames - 1]["density_mse_hidden"],
                })

                # Cache physical reconstructions for representative plot
                pred_phys_23 = (rollout_pred[23] * float(std[3].item()) + float(mean[3].item())).detach().cpu().numpy()
                pred_phys_47 = (rollout_pred[args.n_frames - 1] * float(std[3].item()) + float(mean[3].item())).detach().cpu().numpy()
                run_preds_f24[strat] = pred_phys_23
                run_preds_f48[strat] = pred_phys_47

            if qualitative_cache is None and b_cond == "B_full":
                tgt_phys_23 = (run_tensor_norm[3, 23] * float(std[3].item()) + float(mean[3].item())).detach().cpu().numpy()
                tgt_phys_47 = (run_tensor_norm[3, args.n_frames - 1] * float(std[3].item()) + float(mean[3].item())).detach().cpu().numpy()
                qualitative_cache = {
                    "run_name": run_name,
                    "b_condition": b_cond,
                    "target_phys_f24": tgt_phys_23,
                    "target_phys_f48": tgt_phys_47,
                    "preds_f24": run_preds_f24,
                    "preds_f48": run_preds_f48,
                }

    df_frames = pd.DataFrame(all_frame_rows)
    df_frames.to_csv(data_dir / "frame_results.csv", index=False)
    print(f"\nSaved frame results: {data_dir / 'frame_results.csv'}")

    df_window = pd.DataFrame(all_window_rows)
    df_window.to_csv(data_dir / "window_summary.csv", index=False)
    print(f"Saved window summary: {data_dir / 'window_summary.csv'}")

    # Generate paired comparisons over settled regime (F24–F48)
    paired_rows = []
    comparisons = [
        ("u1", "original", "U1_minus_Original"),
        ("no_feedback", "original", "NoFeedback_minus_Original"),
        ("u1", "no_feedback", "U1_minus_NoFeedback"),
    ]

    for b_cond in b_conditions:
        sub_w = df_window[df_window["b_condition"] == b_cond]
        p_mse = sub_w.pivot(index="run_name", columns="strategy", values="mean_hidden_mse_f24_f48")

        for r in p_mse.index:
            row = {
                "run_name": r,
                "b_condition": b_cond,
            }
            for s in strategies:
                row[f"mse_{s}"] = p_mse.loc[r, s]

            for s1, s2, key in comparisons:
                if s1 in p_mse.columns and s2 in p_mse.columns:
                    diff_m = p_mse.loc[r, s1] - p_mse.loc[r, s2]
                    ratio_m = p_mse.loc[r, s1] / (p_mse.loc[r, s2] + 1e-12)
                    row[f"diff_mse_{key}"] = diff_m
                    row[f"ratio_mse_{key}"] = ratio_m
            paired_rows.append(row)

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(data_dir / "paired_comparisons.csv", index=False)
    print(f"Saved paired comparisons: {data_dir / 'paired_comparisons.csv'}")

    # Generate Overall Summary Table
    summary_rows = []
    for b_cond in b_conditions:
        sub_w = df_window[df_window["b_condition"] == b_cond]
        for s in strategies:
            sub_ws = sub_w[sub_w["strategy"] == s]
            summary_rows.append({
                "b_condition": b_cond,
                "strategy": s,
                "mean_hidden_mse_f24_f48": float(sub_ws["mean_hidden_mse_f24_f48"].mean()),
                "median_hidden_mse_f24_f48": float(sub_ws["mean_hidden_mse_f24_f48"].median()),
                "std_hidden_mse_f24_f48": float(sub_ws["mean_hidden_mse_f24_f48"].std()),
                "mean_var_hidden_f24_f48": float(sub_ws["mean_var_hidden_f24_f48"].mean()),
                "u_ratio_f24_f48": float(sub_ws["u_ratio_f24_f48"].mean()),
                "mean_mse_f24": float(sub_ws["mse_f24"].mean()),
                "mean_mse_f48": float(sub_ws["mse_f48"].mean()),
            })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(data_dir / "overall_summary_table.csv", index=False)
    print(f"Saved overall summary table: {data_dir / 'overall_summary_table.csv'}")

    # Print summary to console
    print("\n=== FULL ONLINE CAUSAL ROLLOUT UNCERTAINTY ABLATION SUMMARY ===")
    for b_cond in b_conditions:
        print(f"\nCondition: {b_cond}")
        sub_sum = df_summary[df_summary["b_condition"] == b_cond]
        for _, r in sub_sum.iterrows():
            print(f"  Strategy {r['strategy']:12s}: "
                  f"F24-F48 MSE Mean={r['mean_hidden_mse_f24_f48']:.5f}, "
                  f"Median={r['median_hidden_mse_f24_f48']:.5f} | "
                  f"Var u={r['mean_var_hidden_f24_f48']:.5f} | "
                  f"u/MSE={r['u_ratio_f24_f48']:.3f} | "
                  f"F24={r['mean_mse_f24']:.5f}, F48={r['mean_mse_f48']:.5f}")

        sub_paired = df_paired[df_paired["b_condition"] == b_cond]
        print(f"  Settled Regime Paired Differences ({b_cond}):")
        for s1, s2, key in comparisons:
            col_diff = f"diff_mse_{key}"
            col_ratio = f"ratio_mse_{key}"
            if col_diff in sub_paired.columns:
                diffs = sub_paired[col_diff]
                ratios = sub_paired[col_ratio]
                n_better = int((diffs < 0).sum())
                n_worse = int((diffs > 0).sum())
                print(f"    {s1:12s} vs. {s2:12s}: Mean Diff={diffs.mean():+.5f}, "
                      f"Median Diff={diffs.median():+.5f}, Mean Ratio={ratios.mean():.3f}x | "
                      f"{s1}<{s2} (improves): {n_better}/{len(diffs)}, "
                      f"{s1}>{s2} (deteriorates): {n_worse}/{len(diffs)}")

    # Plot Figures
    plot_rollout_figures(
        df_frames=df_frames,
        df_window=df_window,
        qualitative_cache=qualitative_cache,
        figures_dir=figures_dir,
    )

    # Save Manifest
    manifest = {
        "experiment": "full_online_causal_rollout_uncertainty_ablation",
        "checkpoint": str(args.run_dir / args.checkpoint),
        "split": str(args.split),
        "n_frames": args.n_frames,
        "window_size": args.window_size,
        "eligible_runs": eligible_runs,
        "n_runs": len(eligible_runs),
        "b_conditions": b_conditions,
        "strategies": {
            "original": "Standard recursive rollout reusing previous predictions and predicted variance u_hat",
            "u1": "Recursive rollout reusing previous predictions with input uncertainty inflated to u_in=1.0",
            "no_feedback": "Non-recursive rollout reconstructing each frame independently without prediction reuse",
        },
        "output_files": [
            str(data_dir / "frame_results.csv"),
            str(data_dir / "window_summary.csv"),
            str(data_dir / "paired_comparisons.csv"),
            str(data_dir / "overall_summary_table.csv"),
            str(figures_dir / "figure1_frame_mse_trajectories.png"),
            str(figures_dir / "figure2_reconstruction_comparison_f24_f48.png"),
            str(figures_dir / "figure3_settled_regime_summary.png"),
        ],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
