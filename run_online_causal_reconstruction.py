#!/usr/bin/env python3
"""Online, Causal Sliding-Window Reconstruction & Filtering for ai4plasma v11.

Evaluates sequential, causal plasma-state reconstruction where frames arrive one by one:
  - At global time t, the 24-frame model context corresponds to [t-23, ..., t].
  - Current frame t always occupies local temporal index 23.
  - Slots with global frame tau < 0 represent nonexistent historical frames:
      * Input values: Gaussian noise sampled in normalized space.
      * Input variance: exactly 1.0.
      * No ground-truth observations and no prediction feedback.
  - Slots with global frame 0 <= tau <= t represent valid frames:
      * Genuine observations (B_full: all B components, Density at 1,000 probes):
          value = y_obs, u_in = 0.0.
      * Non-probe Density at historical frames tau < t:
          reused prior filtering prediction y_hat[tau], u_in = u_hat[tau].
          No added Gaussian noise to reused predictions.
      * Unobserved slots (e.g., B components under B_hidden, unobserved Density):
          value = Gaussian noise, u_in = 1.0.
  - Model forward pass extracts prediction and predicted variance at local temporal index 23.
  - This single-pass prediction represents the real-time online filtering estimate:
      rho_hat[t | observations up to t].
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
    _bounded_pair_sample,
    find_and_load_run,
    peek_run_n_frames,
    sliding_density_probe_mask,
)
from models.unet3d import UNet3D
from visualize_sliding_density_reconstruction import (
    convert_density_units,
    framewise_rmse_mae,
    nrmse_nmae,
    project_time_z,
    rmse_mae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run online causal sliding reconstruction for ai4plasma v11."
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
        default=Path("evaluations/physworld_v11_online_causal_20261123"),
        help="Output directory for online causal evaluation results.",
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
        help="Total sequential rollout frames per run.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="Context window size (T=24).",
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
        "--x-index",
        type=int,
        default=77,
        help="X index for time-z projections.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum eligible runs to evaluate (0 for all).",
    )
    return parser.parse_args()


def load_model_and_stats(
    run_dir: Path, checkpoint_name: str, device: torch.device
) -> Tuple[torch.nn.Module, torch.Tensor, torch.Tensor, Dict]:
    ckpt_path = run_dir / checkpoint_name if not Path(checkpoint_name).is_absolute() else Path(checkpoint_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    base_channels = int(args.get("base_channels", 24))
    use_attention = bool(args.get("use_attention", True))
    channel_mults = tuple(args.get("channel_mults", (1, 2, 4, 8)))
    architecture = str(args.get("model_version", "direct_unet3d_no_activation_norm_v3"))
    spatial_only_pooling = bool(args.get("spatial_only_pooling", True))
    uncertainty_aware = bool(args.get("uncertainty_aware", True))

    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=base_channels,
        channel_mults=channel_mults,
        architecture=architecture,
        use_attention=use_attention,
        spatial_only_pooling=spatial_only_pooling,
        uncertainty_aware=uncertainty_aware,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()

    stats = ckpt["stats"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).reshape(4, 1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).reshape(4, 1, 1, 1)
    return model, mean, std, args


def build_causal_window_input(
    t: int,
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    history_pred: torch.Tensor,
    history_var: torch.Tensor,
    hide_magnetic: bool,
    window_size: int = 24,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Constructs the 8-channel input for global time step t under strict causality.

    Window corresponds to global frame indices: [t - window_size + 1, ..., t].
    Current frame t is always placed at local temporal index (window_size - 1).

    For each local index l in 0 .. (window_size - 1):
      global frame tau = t - window_size + 1 + l.
      - If tau < 0 (nonexistent historical frame):
          All 4 channels: Gaussian noise in normalized space, u_in = 1.0.
      - If 0 <= tau <= t (valid frame):
          Magnetic (channels 0..2):
            * If not hide_magnetic: exact target[:3, tau], u_in = 0.0.
            * If hide_magnetic: Gaussian noise, u_in = 1.0.
          Density (channel 3):
            * At probe locations (probe_mask == True):
                exact target[3, tau], u_in = 0.0.
            * At unobserved locations:
                - If tau == t (current frame): Gaussian noise, u_in = 1.0.
                - If tau < t (historical frame): history_pred[tau], u_in = history_var[tau].

    Returns:
      model_input: [1, 8, window_size, H, W]
      current_frame_target: [4, H, W] target at global time t
    """
    device = target_normalized.device
    dtype = target_normalized.dtype
    _, _, H, W = target_normalized.shape

    # Initialize full window as unknown Gaussian noise with u_in = 1.0
    if generator is not None:
        noise = torch.randn((1, 4, window_size, H, W), generator=generator, device=device, dtype=dtype)
    else:
        noise = torch.randn((1, 4, window_size, H, W), device=device, dtype=dtype)

    values = noise.clone()
    input_variance = torch.ones_like(values)

    probe_bool = (probe_mask > 0.5)

    for l in range(window_size):
        tau = t - window_size + 1 + l
        if tau < 0:
            # Nonexistent historical frame: retain Gaussian noise and variance = 1.0
            continue
        elif tau <= t:
            # Valid frame up to current time t
            # 1. Magnetic components (channels 0..2)
            if not hide_magnetic:
                values[0, :3, l] = target_normalized[:3, tau]
                input_variance[0, :3, l] = 0.0
            else:
                # hide_magnetic: already noise and variance 1.0
                pass

            # 2. Density component (channel 3)
            if tau == t:
                # Current frame: genuine probes available, non-probes are unknown (noise)
                # Keep noise where unobserved
                values[0, 3, l, probe_bool] = target_normalized[3, tau, probe_bool]
                input_variance[0, 3, l, probe_bool] = 0.0
                # Non-probe slots remain noise and input_variance = 1.0
            else:
                # Historical frame tau < t:
                # Real probes always win and have zero variance
                values[0, 3, l] = history_pred[tau]
                input_variance[0, 3, l] = history_var[tau]
                values[0, 3, l, probe_bool] = target_normalized[3, tau, probe_bool]
                input_variance[0, 3, l, probe_bool] = 0.0

    model_input = torch.cat([values, input_variance], dim=1)
    return model_input, target_normalized[:, t]


@torch.no_grad()
def reconstruct_online_causal(
    model: torch.nn.Module,
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    n_frames: int = 48,
    window_size: int = 24,
    hide_magnetic: bool = False,
    seed: Optional[int] = None,
    amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs sequential online causal filtering for global frames t = 0 .. n_frames - 1.

    At each step t:
      - Window is [t-23, ..., t].
      - Forward pass through model.
      - Extract Density prediction and variance at local temporal index 23.
      - Save into history_pred[t] and history_var[t].

    Returns:
      pred_density_norm: [n_frames, H, W] normalized density predictions
      pred_var_norm: [n_frames, H, W] predicted normalized variance (squared error)
    """
    device = target_normalized.device
    dtype = target_normalized.dtype
    _, _, H, W = target_normalized.shape

    history_pred = torch.zeros((n_frames, H, W), device=device, dtype=dtype)
    history_var = torch.ones((n_frames, H, W), device=device, dtype=dtype)

    current_idx = window_size - 1

    for t in range(n_frames):
        if seed is not None:
            g = torch.Generator(device=device)
            # Deterministic, unique seed per global step t
            g.manual_seed((int(seed) + int(t) * 10007) & 0x7FFFFFFF)
        else:
            g = None

        model_input, _ = build_causal_window_input(
            t=t,
            target_normalized=target_normalized,
            probe_mask=probe_mask,
            history_pred=history_pred,
            history_var=history_var,
            hide_magnetic=hide_magnetic,
            window_size=window_size,
            generator=g,
        )

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(amp and device.type == "cuda")):
            prediction = model(model_input)

        if getattr(model, "uncertainty_aware", True):
            mean_pred, var_pred = prediction
        else:
            mean_pred = prediction
            var_pred = torch.zeros_like(mean_pred)

        # Extract Density (channel 3) at current local time (index 23)
        density_pred_t = mean_pred[0, 3, current_idx].float()
        density_var_t = var_pred[0, 3, current_idx].float()

        # Real probes always win with zero variance in state
        probe_bool = (probe_mask > 0.5)
        density_pred_t[probe_bool] = target_normalized[3, t, probe_bool]
        density_var_t[probe_bool] = 0.0

        history_pred[t] = density_pred_t
        history_var[t] = density_var_t

    pred_density_np = history_pred.detach().cpu().numpy()
    pred_var_np = history_var.detach().cpu().numpy()
    return pred_density_np, pred_var_np


def evaluate_run_online(
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    target: torch.Tensor,
    probe_mask: torch.Tensor,
    window_size: int,
    b_condition: str,
    run_seed: int,
    n_frames: int = 48,
    amp: bool = True,
) -> Dict:
    hide_magnetic = (b_condition == "B_hidden")
    target_normalized = (target - mean) / (std + 1e-8)
    target_density_plot = target[3, :n_frames].detach().cpu().numpy()
    target_density_norm = target_normalized[3, :n_frames].detach().cpu().numpy()
    hidden_mask = ~(probe_mask.detach().cpu().numpy() > 0.5)

    pred_norm, var_norm = reconstruct_online_causal(
        model=model,
        target_normalized=target_normalized[:, :n_frames],
        probe_mask=probe_mask,
        n_frames=n_frames,
        window_size=window_size,
        hide_magnetic=hide_magnetic,
        seed=run_seed,
        amp=amp,
    )

    pred_phys = pred_norm * float(std[3].item()) + float(mean[3].item())

    # Overall metrics on hidden voxels
    full_nrmse, _ = nrmse_nmae(pred_phys, target_density_plot, mask=hidden_mask)
    e2_all = np.square(pred_norm - target_density_norm)
    mean_e2 = float(np.mean(e2_all[:, hidden_mask]))
    mean_var = float(np.mean(var_norm[:, hidden_mask]))
    u_ratio = float(mean_var / (mean_e2 + 1e-12))

    # Correlation on hidden voxels across entire rollout
    e2_flat = e2_all[:, hidden_mask].ravel()
    var_flat = var_norm[:, hidden_mask].ravel()
    pear_r, spear_r = pearson_spearman(var_flat, e2_flat)

    # Framewise metrics
    framewise_e2_hidden = []
    framewise_var_hidden = []
    framewise_nrmse = []
    framewise_u_ratio = []
    for t in range(n_frames):
        fnrmse, _ = nrmse_nmae(pred_phys[t : t + 1], target_density_plot[t : t + 1], mask=hidden_mask)
        e2_t = float(np.mean(e2_all[t, hidden_mask]))
        var_t = float(np.mean(var_norm[t, hidden_mask]))
        u_rat_t = float(var_t / (e2_t + 1e-12))
        framewise_nrmse.append(float(fnrmse))
        framewise_e2_hidden.append(e2_t)
        framewise_var_hidden.append(var_t)
        framewise_u_ratio.append(u_rat_t)

    # Partition into temporal regimes:
    # Cold start: frames 0-5
    # Early history accumulation: frames 6-22
    # Full history availability: frames 23-47
    regimes = {
        "cold_start_0_5": (0, 6),
        "early_history_6_22": (6, 23),
        "full_history_23_47": (23, n_frames),
    }
    regime_metrics = {}
    for reg_name, (r_start, r_end) in regimes.items():
        reg_nrmse, _ = nrmse_nmae(pred_phys[r_start:r_end], target_density_plot[r_start:r_end], mask=hidden_mask)
        reg_e2 = float(np.mean(e2_all[r_start:r_end, hidden_mask]))
        reg_var = float(np.mean(var_norm[r_start:r_end, hidden_mask]))
        reg_urat = float(reg_var / (reg_e2 + 1e-12))
        regime_metrics[reg_name] = {
            "nrmse": float(reg_nrmse),
            "mse_hidden": reg_e2,
            "var_hidden": reg_var,
            "u_ratio": reg_urat,
        }

    return {
        "density_nrmse": float(full_nrmse),
        "mean_e2_hidden": mean_e2,
        "mean_var_hidden": mean_var,
        "u_ratio_hidden": u_ratio,
        "pearson_r": float(pear_r),
        "spearman_r": float(spear_r),
        "framewise_nrmse": framewise_nrmse,
        "framewise_e2_hidden": framewise_e2_hidden,
        "framewise_var_hidden": framewise_var_hidden,
        "framewise_u_ratio": framewise_u_ratio,
        "regime_metrics": regime_metrics,
        "pred_phys": pred_phys,
        "pred_norm": pred_norm,
        "var_norm": var_norm,
        "target_density_plot": target_density_plot,
        "hidden_mask": hidden_mask,
    }


def generate_online_figures(
    df_framewise: pd.DataFrame,
    df_summary: pd.DataFrame,
    offline_summary_path: Optional[Path],
    qualitative_cache: Optional[Dict],
    figures_dir: Path,
):
    colors = {"B_full": "#1f77b4", "B_hidden": "#d62728"}

    # Figure 1: Online Framewise Error and Predicted Variance vs. Global Time
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for row_idx, cond in enumerate(["B_full", "B_hidden"]):
        sub = df_framewise[df_framewise["b_condition"] == cond]
        grp = sub.groupby("global_frame")[["e2_hidden", "var_hidden", "density_nrmse"]].mean()

        # Left: Framewise MSE (hidden)
        ax_err = axes[row_idx, 0]
        ax_err.plot(grp.index, grp["e2_hidden"], color=colors[cond], lw=2.2, label=f"Online Causal {cond}")
        ax_err.axvline(5.5, color="gray", linestyle=":", alpha=0.7, label="Cold start boundary (t=5)")
        ax_err.axvline(22.5, color="black", linestyle="--", alpha=0.7, label="Full history boundary (t=22)")
        ax_err.set_title(f"{cond}: Framewise MSE (e² on Hidden Voxels)", fontsize=10)
        ax_err.set_ylabel("Standardized MSE (e²)")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(fontsize=8, loc="upper right")

        # Right: Framewise Predicted Variance (u)
        ax_var = axes[row_idx, 1]
        ax_var.plot(grp.index, grp["var_hidden"], color="purple" if cond == "B_full" else "darkorange", lw=2.2, label=f"Predicted Var u ({cond})")
        ax_var.axvline(5.5, color="gray", linestyle=":", alpha=0.7)
        ax_var.axvline(22.5, color="black", linestyle="--", alpha=0.7)
        ax_var.set_title(f"{cond}: Framewise Predicted Variance (u)", fontsize=10)
        ax_var.set_ylabel("Standardized Variance (u)")
        ax_var.grid(True, alpha=0.3)
        ax_var.legend(fontsize=8, loc="upper right")

    axes[1, 0].set_xlabel("Global Time Step t (0 .. 47)")
    axes[1, 1].set_xlabel("Global Time Step t (0 .. 47)")
    fig.suptitle("Figure 1: Online Causal Filtering Framewise Error & Uncertainty Calibration", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_online_framewise_error_variance.png", dpi=200)
    fig.savefig(figures_dir / "figure1_online_framewise_error_variance.pdf")
    plt.close(fig)

    # Figure 2: Online Causal vs Offline Sliding Reconstruction
    # (Explicitly distinguished information-availability protocol)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    offline_data = None
    if offline_summary_path is not None and offline_summary_path.exists():
        try:
            offline_df = pd.read_csv(offline_summary_path)
            # Find step=24 Variant A (authoritative offline baseline)
            offline_data = offline_df[offline_df["step"] == 24]
        except Exception as e:
            print(f"Notice: Could not parse offline summary: {e}")

    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        online_sub = df_summary[df_summary["b_condition"] == cond]
        online_nrmse = online_sub["density_nrmse"].values

        x = np.arange(len(online_sub))
        width = 0.35
        ax.bar(x - width/2, online_nrmse, width, label="Online Causal (1 frame at a time)", color="#2ca02c", alpha=0.85)

        if offline_data is not None:
            off_sub = offline_data[offline_data["b_condition"] == cond]
            if len(off_sub) == len(online_sub):
                ax.bar(x + width/2, off_sub["density_nrmse"].values, width, label="Offline Window (24 frames initial context)", color="#1f77b4", alpha=0.85)

        ax.set_title(f"{cond}: Online Causal vs. Offline Protocol", fontsize=11)
        ax.set_xlabel("Held-Out Simulation Run (10 runs)")
        ax.set_ylabel("Density NRMSE")
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{i+1}" for i in range(len(online_sub))], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Figure 2: Online Causal Filtering vs. Offline Sliding Protocol (Different Information Availability)", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_online_vs_offline_comparison.png", dpi=200)
    fig.savefig(figures_dir / "figure2_online_vs_offline_comparison.pdf")
    plt.close(fig)

    # Figure 3: Cold-Start Behavior (Frames 0-23)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub = df_framewise[(df_framewise["b_condition"] == cond) & (df_framewise["global_frame"] < 24)]
        grp = sub.groupby("global_frame")[["e2_hidden", "var_hidden"]].mean()
        ax.plot(grp.index, grp["e2_hidden"], marker="o", color="firebrick", lw=2, label="Actual MSE (e²)")
        ax.plot(grp.index, grp["var_hidden"], marker="s", color="navy", lw=2, label="Predicted Var (u)")
        ax.axvspan(0, 5, color="yellow", alpha=0.15, label="Cold start (t=0..5)")
        ax.set_title(f"{cond}: Cold-Start Trajectory (Frames 0–23)", fontsize=11)
        ax.set_xlabel("Global Time Step t")
        ax.set_ylabel("Standardized squared units")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Figure 3: Online Cold-Start Calibration and Error Dynamics", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure3_cold_start_dynamics.png", dpi=200)
    fig.savefig(figures_dir / "figure3_cold_start_dynamics.pdf")
    plt.close(fig)

    # Figure 4: B_full vs. B_hidden Online Reconstruction Direct Comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    grp_bf = df_framewise[df_framewise["b_condition"] == "B_full"].groupby("global_frame")[["e2_hidden", "density_nrmse"]].mean()
    grp_bh = df_framewise[df_framewise["b_condition"] == "B_hidden"].groupby("global_frame")[["e2_hidden", "density_nrmse"]].mean()

    # Left: MSE
    axes[0].plot(grp_bf.index, grp_bf["e2_hidden"], color=colors["B_full"], lw=2.2, label="B_full Online")
    axes[0].plot(grp_bh.index, grp_bh["e2_hidden"], color=colors["B_hidden"], lw=2.2, label="B_hidden Online")
    axes[0].set_title("Framewise MSE (Hidden Voxels)", fontsize=11)
    axes[0].set_xlabel("Global Time Step t")
    axes[0].set_ylabel("Standardized MSE (e²)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    # Right: NRMSE
    axes[1].plot(grp_bf.index, grp_bf["density_nrmse"], color=colors["B_full"], lw=2.2, label="B_full Online")
    axes[1].plot(grp_bh.index, grp_bh["density_nrmse"], color=colors["B_hidden"], lw=2.2, label="B_hidden Online")
    axes[1].set_title("Framewise Density NRMSE", fontsize=11)
    axes[1].set_xlabel("Global Time Step t")
    axes[1].set_ylabel("Density NRMSE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle("Figure 4: B_full vs. B_hidden Online Causal Filtering Performance", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure4_bfull_vs_bhidden_online.png", dpi=200)
    fig.savefig(figures_dir / "figure4_bfull_vs_bhidden_online.pdf")
    plt.close(fig)

    # Figure 5: Qualitative Time-Z Slice
    if qualitative_cache is not None:
        target_p = qualitative_cache["target_plot"]
        pred_p = qualitative_cache["pred_phys"]
        x_idx = qualitative_cache.get("x_index", 77)

        proj_tgt = project_time_z(target_p, x_index=x_idx)
        proj_prd = project_time_z(pred_p, x_index=x_idx)
        proj_res = proj_prd - proj_tgt

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        vmin, vmax = np.nanpercentile(proj_tgt, [2, 98])
        res_limit = max(abs(np.nanpercentile(proj_res, 2)), abs(np.nanpercentile(proj_res, 98)))

        im0 = axes[0].imshow(proj_tgt.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0].set_title("Ground Truth Density (Time-Z Projection)", fontsize=10)
        axes[0].set_ylabel("Z index")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(proj_prd.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Online Causal Filtered Density ({qualitative_cache.get('b_condition', 'B_full')})", fontsize=10)
        axes[1].set_ylabel("Z index")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        im2 = axes[2].imshow(proj_res.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-res_limit, vmax=res_limit)
        axes[2].set_title("Online Reconstruction Residual (Prediction − Target)", fontsize=10)
        axes[2].set_xlabel("Global Time Step t")
        axes[2].set_ylabel("Z index")
        axes[2].axvline(5.5, color="k", linestyle=":", alpha=0.6)
        axes[2].axvline(22.5, color="k", linestyle="--", alpha=0.6)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        fig.suptitle("Figure 5: Representative Online Causal Filtering Rollout", fontsize=12, y=0.98)
        fig.tight_layout()
        fig.savefig(figures_dir / "figure5_representative_qualitative_online.png", dpi=200)
        fig.savefig(figures_dir / "figure5_representative_qualitative_online.pdf")
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

    print(f"Evaluating {len(eligible_runs)} eligible runs: {eligible_runs}")

    b_conditions = ["B_full", "B_hidden"]

    summary_rows = []
    framewise_rows = []
    regime_rows = []
    qualitative_cache = None

    for run_idx, run_name in enumerate(eligible_runs):
        print(f"\n--- [{run_idx + 1}/{len(eligible_runs)}] Run: {run_name} ---")
        run_data, _ = find_and_load_run(args.h5_dir, args.betas, run_name, n_frames=args.n_frames)
        run_tensor = torch.tensor(run_data, dtype=torch.float32, device=device)

        probe_mask_2d = sliding_density_probe_mask(
            spatial_shape=(run_tensor.shape[2], run_tensor.shape[3]),
            probe_count=args.density_probe_count,
            seed=args.seed,
            device=device,
        )

        for b_cond in b_conditions:
            run_seed = args.seed + run_idx * 1000 + (0 if b_cond == "B_full" else 500)
            res = evaluate_run_online(
                model=model,
                mean=mean,
                std=std,
                target=run_tensor,
                probe_mask=probe_mask_2d,
                window_size=args.window_size,
                b_condition=b_cond,
                run_seed=run_seed,
                n_frames=args.n_frames,
                amp=True,
            )

            print(
                f"  [{b_cond}] Density NRMSE: {res['density_nrmse']:.4f} | "
                f"MSE e²: {res['mean_e2_hidden']:.5f} | "
                f"Var u: {res['mean_var_hidden']:.5f} | "
                f"u/MSE: {res['u_ratio_hidden']:.4f} | "
                f"Pearson r: {res['pearson_r']:.3f}"
            )

            summary_rows.append({
                "run_name": run_name,
                "b_condition": b_cond,
                "density_nrmse": res["density_nrmse"],
                "mean_e2_hidden": res["mean_e2_hidden"],
                "mean_var_hidden": res["mean_var_hidden"],
                "u_ratio_hidden": res["u_ratio_hidden"],
                "pearson_r": res["pearson_r"],
                "spearman_r": res["spearman_r"],
            })

            for t in range(args.n_frames):
                framewise_rows.append({
                    "run_name": run_name,
                    "b_condition": b_cond,
                    "global_frame": t,
                    "density_nrmse": res["framewise_nrmse"][t],
                    "e2_hidden": res["framewise_e2_hidden"][t],
                    "var_hidden": res["framewise_var_hidden"][t],
                    "u_ratio_hidden": res["framewise_u_ratio"][t],
                })

            for reg_name, reg_data in res["regime_metrics"].items():
                regime_rows.append({
                    "run_name": run_name,
                    "b_condition": b_cond,
                    "regime": reg_name,
                    **reg_data,
                })

            if qualitative_cache is None and b_cond == "B_full":
                qualitative_cache = {
                    "run_name": run_name,
                    "b_condition": b_cond,
                    "target_plot": res["target_density_plot"],
                    "pred_phys": res["pred_phys"],
                    "x_index": args.x_index,
                }

    # Save CSV outputs
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(data_dir / "online_summary.csv", index=False)
    print(f"Saved: {data_dir / 'online_summary.csv'}")

    df_framewise = pd.DataFrame(framewise_rows)
    df_framewise.to_csv(data_dir / "online_framewise_metrics.csv", index=False)
    print(f"Saved: {data_dir / 'online_framewise_metrics.csv'}")

    df_regimes = pd.DataFrame(regime_rows)
    df_regimes.to_csv(data_dir / "online_regime_metrics.csv", index=False)
    print(f"Saved: {data_dir / 'online_regime_metrics.csv'}")

    # Generate Figures
    offline_summary_path = Path("evaluations/physworld_v11_sliding_overwrite_ablation_20261123/data/ablation_summary.csv")
    generate_online_figures(
        df_framewise=df_framewise,
        df_summary=df_summary,
        offline_summary_path=offline_summary_path,
        qualitative_cache=qualitative_cache,
        figures_dir=figures_dir,
    )

    # Save Manifest
    manifest = {
        "experiment": "online_causal_sliding_reconstruction",
        "checkpoint": str(args.run_dir / args.checkpoint),
        "split": str(args.split),
        "eligible_runs": eligible_runs,
        "n_runs": len(eligible_runs),
        "n_frames": args.n_frames,
        "window_size": args.window_size,
        "b_conditions": b_conditions,
        "protocol": {
            "type": "online_causal_filtering",
            "frame_arrival": "sequential (t advances 0 .. 47)",
            "context_window": "[t-23, ..., t]",
            "current_frame_local_index": args.window_size - 1,
            "padding_negative_time": "Gaussian noise (mean=0, std=1 in normalized space, u_in=1.0)",
            "observation_semantics": {
                "genuine_observations": "x = y_obs, u_in = 0.0",
                "historical_predictions": "x = y_hat, u_in = u_hat (continuous variance)",
                "unobserved_locations": "x = Gaussian noise, u_in = 1.0",
            },
        },
        "output_files": [
            str(data_dir / "online_summary.csv"),
            str(data_dir / "online_framewise_metrics.csv"),
            str(data_dir / "online_regime_metrics.csv"),
            str(figures_dir / "figure1_online_framewise_error_variance.png"),
            str(figures_dir / "figure2_online_vs_offline_comparison.png"),
            str(figures_dir / "figure3_cold_start_dynamics.png"),
            str(figures_dir / "figure4_bfull_vs_bhidden_online.png"),
            str(figures_dir / "figure5_representative_qualitative_online.png"),
        ],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
