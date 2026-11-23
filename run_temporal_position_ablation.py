#!/usr/bin/env python3
"""Fixed-Target Temporal-Position Ablation for ai4plasma v11.

Investigates how the local temporal position of a fixed physical frame within
the 24-frame UNet3D input window affects its reconstruction accuracy and predicted uncertainty:
  - Default target: Global frame F24 (with optional F23 support).
  - Three distinct 24-frame windows for F24:
      * Left edge:  Window F24-F47 (target F24 at local temporal index 0)
      * Interior:   Window F13-F36 (target F24 at local temporal index 11)
      * Right edge: Window F1-F24  (target F24 at local temporal index 23)
  - Every window contains 24 valid VPIC simulation frames (no negative-time padding).
  - Reconstructs exactly the same target frame F24 across all three configurations.
  - Zero prediction reuse, zero recursive feedback, zero historical overwriting.
  - Consistent global noise slicing: Gaussian noise is generated over the full 48-frame
    rollout and sliced consistently so that any given global frame, physical channel,
    and spatial voxel receives the exact same noise realization across all windows.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-target temporal-position ablation for ai4plasma v11."
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
        default=Path("evaluations/physworld_v11_temporal_position_ablation_20261123"),
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
        help="Total rollout frames per run (at least 48).",
    )
    parser.add_argument(
        "--target-frame",
        type=int,
        default=24,
        choices=[23, 24],
        help="Target global frame to reconstruct (default: 24, optional: 23).",
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


def get_window_configurations(target_frame: int = 24) -> Dict[str, Dict]:
    """Returns the window slicing parameters for the three temporal configurations.

    For target_frame=24:
      Left edge:  Window F24-F47 (start=24, end=48, target local index 0)
      Interior:   Window F13-F36 (start=13, end=37, target local index 11)
      Right edge: Window F1-F24  (start=1,  end=25, target local index 23)

    For target_frame=23:
      Left edge:  Window F23-F46 (start=23, end=47, target local index 0)
      Interior:   Window F12-F35 (start=12, end=36, target local index 11)
      Right edge: Window F0-F23  (start=0,  end=24, target local index 23)
    """
    if target_frame == 24:
        return {
            "left_edge": {
                "name": "Left edge (index 0)",
                "start": 24,
                "end": 48,
                "local_index": 0,
                "label": "Left (idx 0)",
            },
            "interior": {
                "name": "Interior (index 11)",
                "start": 13,
                "end": 37,
                "local_index": 11,
                "label": "Interior (idx 11)",
            },
            "right_edge": {
                "name": "Right edge (index 23)",
                "start": 1,
                "end": 25,
                "local_index": 23,
                "label": "Right (idx 23)",
            },
        }
    elif target_frame == 23:
        return {
            "left_edge": {
                "name": "Left edge (index 0)",
                "start": 23,
                "end": 47,
                "local_index": 0,
                "label": "Left (idx 0)",
            },
            "interior": {
                "name": "Interior (index 11)",
                "start": 12,
                "end": 36,
                "local_index": 11,
                "label": "Interior (idx 11)",
            },
            "right_edge": {
                "name": "Right edge (index 23)",
                "start": 0,
                "end": 24,
                "local_index": 23,
                "label": "Right (idx 23)",
            },
        }
    else:
        raise ValueError(f"Unsupported target_frame: {target_frame}. Expected 23 or 24.")


def build_controlled_window_input(
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    global_noise: torch.Tensor,
    start: int,
    end: int,
    hide_magnetic: bool,
) -> torch.Tensor:
    """Builds the 8-channel UNet input for a 24-frame slice [start:end].

    Ensures that for any global frame tau in [start:end]:
      - B components:
          * If not hide_magnetic: exact target[:3, tau], variance = 0.0
          * If hide_magnetic: global_noise[:3, tau], variance = 1.0
      - Density (channel 3):
          * At probe locations: exact target[3, tau], variance = 0.0
          * At unobserved locations: global_noise[3, tau], variance = 1.0

    Input noise is consistently sliced from global_noise [4, n_frames, H, W]
    so matching global frames receive the exact same noise in all window positions.

    Returns:
      model_input: [1, 8, 24, H, W]
    """
    device = target_normalized.device
    dtype = target_normalized.dtype
    _, _, H, W = target_normalized.shape
    window_length = end - start
    assert window_length == 24, f"Window length must be 24, got {window_length}"

    # Slice target and noise for this window
    window_target = target_normalized[:, start:end]
    window_noise = global_noise[:, start:end]

    # Initialize values with noise and variance with 1.0
    values = window_noise.clone()
    input_variance = torch.ones_like(values)

    # 1. Magnetic components (channels 0..2)
    if not hide_magnetic:
        values[:3] = window_target[:3]
        input_variance[:3] = 0.0

    # 2. Density component (channel 3)
    probe_bool = (probe_mask > 0.5)
    # Density probes take exact target with zero variance
    values[3, :, probe_bool] = window_target[3, :, probe_bool]
    input_variance[3, :, probe_bool] = 0.0

    # Combine into 8-channel format: [1, 8, 24, H, W]
    model_input = torch.cat([values.unsqueeze(0), input_variance.unsqueeze(0)], dim=1)
    return model_input


@torch.no_grad()
def evaluate_position_configuration(
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    global_noise: torch.Tensor,
    target_frame: int,
    config: Dict,
    hide_magnetic: bool,
    amp: bool = True,
) -> Dict:
    """Executes a single forward pass for a position configuration and extracts

    predictions at the target frame's specified local index.
    """
    device = target_normalized.device
    start = config["start"]
    end = config["end"]
    local_idx = config["local_index"]

    # Invariant assertion: the local index in [start:end] must map to target_frame
    assert start + local_idx == target_frame, (
        f"Position mismatch: start={start} + local_idx={local_idx} != target_frame={target_frame}"
    )

    model_input = build_controlled_window_input(
        target_normalized=target_normalized,
        probe_mask=probe_mask,
        global_noise=global_noise,
        start=start,
        end=end,
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

    # Extract Density (channel 3) at the target's specified local index
    density_pred_norm = mean_pred[0, 3, local_idx].float()
    density_var_norm = var_pred[0, 3, local_idx].float()

    # Ground truth normalized Density at target_frame
    target_norm = target_normalized[3, target_frame].float()

    # Compute errors on unobserved voxels
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

    # Pearson & Spearman correlations on hidden voxels
    e2_np = e2_hidden.detach().cpu().numpy().ravel()
    var_np = var_hidden.detach().cpu().numpy().ravel()
    corr_dict = pearson_spearman(var_np, e2_np)
    pearson_r = float(corr_dict["pearson"])
    spearman_r = float(corr_dict["spearman"])

    pred_phys = (density_pred_norm * float(std[3].item()) + float(mean[3].item())).detach().cpu().numpy()
    target_phys = (target_norm * float(std[3].item()) + float(mean[3].item())).detach().cpu().numpy()

    return {
        "config_key": config["label"],
        "local_index": local_idx,
        "window_start": start,
        "window_end": end,
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


def plot_position_figures(
    df_results: pd.DataFrame,
    qualitative_cache: Optional[Dict],
    figures_dir: Path,
    target_frame: int,
):
    colors = {"B_full": "#1f77b4", "B_hidden": "#d62728"}
    pos_order = ["Left (idx 0)", "Interior (idx 11)", "Right (idx 23)"]

    # -------------------------------------------------------------
    # Figure 1: Reconstruction Error vs. Local Temporal Position
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub = df_results[df_results["b_condition"] == cond]

        # Plot individual run trajectories
        runs = sub["run_name"].unique()
        for r in runs:
            r_sub = sub[sub["run_name"] == r].set_index("config_key").reindex(pos_order)
            ax.plot(pos_order, r_sub["density_mse_hidden"], color="gray", alpha=0.4, lw=1.2, marker="o", markersize=4)

        # Plot cross-run mean
        mean_vals = [sub[sub["config_key"] == p]["density_mse_hidden"].mean() for p in pos_order]
        ax.plot(pos_order, mean_vals, color=colors[cond], lw=2.8, marker="s", markersize=8, label=f"Mean ({cond})")

        ax.set_title(f"{cond}: Hidden MSE vs. Local Position (Target F{target_frame})", fontsize=11)
        ax.set_ylabel("Standardized MSE (e² on Hidden Voxels)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"Figure 1: Fixed-Target (F{target_frame}) Reconstruction Error vs. Local Temporal Position\n"
        "(Identical Target Frame & Spatial Probes Across All Three Configurations)",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_reconstruction_error_vs_temporal_position.png", dpi=200)
    fig.savefig(figures_dir / "figure1_reconstruction_error_vs_temporal_position.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 2: Predicted Variance & Calibration Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub = df_results[df_results["b_condition"] == cond]

        mean_mse = [sub[sub["config_key"] == p]["density_mse_hidden"].mean() for p in pos_order]
        mean_var = [sub[sub["config_key"] == p]["mean_var_hidden"].mean() for p in pos_order]

        x = np.arange(len(pos_order))
        width = 0.35
        ax.bar(x - width / 2, mean_mse, width, label="Actual MSE (e²)", color="firebrick", alpha=0.85)
        ax.bar(x + width / 2, mean_var, width, label="Predicted Variance (u)", color="navy", alpha=0.85)

        for i in range(len(pos_order)):
            ratio = mean_var[i] / (mean_mse[i] + 1e-12)
            ax.text(x[i], max(mean_mse[i], mean_var[i]) + 0.0003, f"u/MSE={ratio:.2f}", ha="center", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(pos_order, fontsize=9)
        ax.set_title(f"{cond}: Calibration across Positions (Target F{target_frame})", fontsize=11)
        ax.set_ylabel("Standardized squared units")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        f"Figure 2: Fixed-Target (F{target_frame}) Predicted Variance vs. Actual Error\n"
        "(Assessing Temporal Uncertainty Calibration at Boundaries)",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_predicted_variance_and_calibration.png", dpi=200)
    fig.savefig(figures_dir / "figure2_predicted_variance_and_calibration.pdf")
    plt.close(fig)

    # -------------------------------------------------------------
    # Figure 3: Representative Fixed-Frame Reconstruction & Residuals
    # -------------------------------------------------------------
    if qualitative_cache is not None:
        target_p = qualitative_cache["target_phys"]
        preds = qualitative_cache["preds"]  # dict of config_key -> pred_phys

        fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, sharey=True)
        vmin, vmax = np.nanpercentile(target_p, [2, 98])

        # Compute max residual limit across all three positions for identical colorbar
        res_limits = []
        for p in pos_order:
            res = preds[p] - target_p
            res_limits.append(max(abs(np.nanpercentile(res, 2)), abs(np.nanpercentile(res, 98))))
        max_res = max(res_limits)

        # Row 0: Target
        im_tgt = axes[0, 0].imshow(target_p, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title(f"Ground Truth Density (F{target_frame})", fontsize=10)
        axes[0, 0].set_ylabel("Z index")
        fig.colorbar(im_tgt, ax=axes[0, 0], fraction=0.046, pad=0.04)
        axes[0, 1].axis("off")

        # Rows 1-3: Left, Interior, Right
        for r_idx, p in enumerate(pos_order, start=1):
            pred = preds[p]
            res = pred - target_p

            im_p = axes[r_idx, 0].imshow(pred, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            axes[r_idx, 0].set_title(f"Reconstruction: {p}", fontsize=10)
            axes[r_idx, 0].set_ylabel("Z index")
            fig.colorbar(im_p, ax=axes[r_idx, 0], fraction=0.046, pad=0.04)

            im_r = axes[r_idx, 1].imshow(res, origin="lower", aspect="auto", cmap="coolwarm", vmin=-max_res, vmax=max_res)
            axes[r_idx, 1].set_title(f"Residual (Pred − Truth): {p}", fontsize=10)
            fig.colorbar(im_r, ax=axes[r_idx, 1], fraction=0.046, pad=0.04)

        axes[3, 0].set_xlabel("X index")
        axes[3, 1].set_xlabel("X index")

        run_lbl = qualitative_cache.get("run_name", "Test Run")
        fig.suptitle(
            f"Figure 3: Representative Fixed-Target (F{target_frame}) Reconstruction Comparison\n"
            f"Run: {run_lbl} ({qualitative_cache.get('b_condition', 'B_full')}) — Identical Physical Time",
            fontsize=11,
            y=0.98,
        )
        fig.tight_layout()
        fig.savefig(figures_dir / "figure3_representative_fixed_frame_reconstruction.png", dpi=200)
        fig.savefig(figures_dir / "figure3_representative_fixed_frame_reconstruction.pdf")
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

    print(f"Evaluating target frame F{args.target_frame} across {len(eligible_runs)} eligible runs: {eligible_runs}")

    configs = get_window_configurations(target_frame=args.target_frame)
    b_conditions = ["B_full", "B_hidden"]

    results_rows = []
    qualitative_cache = None

    for run_idx, run_name in enumerate(eligible_runs):
        print(f"\n--- [{run_idx + 1}/{len(eligible_runs)}] Run: {run_name} ---")
        fields, _frame_ids, _h5_path = find_and_load_run(args.h5_dir, args.betas, run_name)
        fields = np.asarray(fields)[: args.n_frames]
        # Transpose from (T, 4, X, Z) to (4, T, X, Z)
        run_tensor = torch.from_numpy(np.transpose(fields, (1, 0, 2, 3))).to(
            device=device, dtype=torch.float32
        )
        run_tensor_norm = (run_tensor - mean) / (std + 1e-8)

        # Standard sliding probe mask matching offline sliding evaluation
        probe_mask_2d, _probe_info = sliding_density_probe_mask(
            run_tensor_norm.unsqueeze(0),
            int(args.density_probe_count),
        )

        # Generate reproducible global noise tensor [4, n_frames, H, W] for this run
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

            for cfg_key, cfg in configs.items():
                res = evaluate_position_configuration(
                    model=model,
                    mean=mean,
                    std=std,
                    target_normalized=run_tensor_norm,
                    probe_mask=probe_mask_2d,
                    global_noise=global_noise,
                    target_frame=args.target_frame,
                    config=cfg,
                    hide_magnetic=hide_magnetic,
                    amp=True,
                )

                print(
                    f"  [{b_cond} | {cfg['label']}] Hidden MSE: {res['density_mse_hidden']:.5f} | "
                    f"RMSE: {res['density_rmse_hidden']:.4f} | "
                    f"Var u: {res['mean_var_hidden']:.5f} | "
                    f"u/MSE: {res['u_ratio_hidden']:.3f} | "
                    f"Pearson r: {res['pearson_r']:.3f}"
                )

                results_rows.append({
                    "run_name": run_name,
                    "target_frame": args.target_frame,
                    "b_condition": b_cond,
                    "config_key": cfg["label"],
                    "local_index": cfg["local_index"],
                    "window_start": cfg["start"],
                    "window_end": cfg["end"],
                    "density_mse_hidden": res["density_mse_hidden"],
                    "density_rmse_hidden": res["density_rmse_hidden"],
                    "full_field_rmse": res["full_field_rmse"],
                    "mean_var_hidden": res["mean_var_hidden"],
                    "u_ratio_hidden": res["u_ratio_hidden"],
                    "pearson_r": res["pearson_r"],
                    "spearman_r": res["spearman_r"],
                })

                run_preds[cfg["label"]] = res["pred_phys"]

            if qualitative_cache is None and b_cond == "B_full":
                qualitative_cache = {
                    "run_name": run_name,
                    "b_condition": b_cond,
                    "target_phys": res["target_phys"],
                    "preds": run_preds,
                }

    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(data_dir / "temporal_position_results.csv", index=False)
    print(f"\nSaved results: {data_dir / 'temporal_position_results.csv'}")

    # Generate paired comparison summary
    paired_rows = []
    for b_cond in b_conditions:
        sub = df_results[df_results["b_condition"] == b_cond]
        p_df = sub.pivot(index="run_name", columns="config_key", values="density_mse_hidden")

        # Left vs Interior
        diff_left_interior = p_df["Left (idx 0)"] - p_df["Interior (idx 11)"]
        # Right vs Interior
        diff_right_interior = p_df["Right (idx 23)"] - p_df["Interior (idx 11)"]
        # Left vs Right
        diff_left_right = p_df["Left (idx 0)"] - p_df["Right (idx 23)"]

        for r in p_df.index:
            paired_rows.append({
                "run_name": r,
                "b_condition": b_cond,
                "target_frame": args.target_frame,
                "mse_left": p_df.loc[r, "Left (idx 0)"],
                "mse_interior": p_df.loc[r, "Interior (idx 11)"],
                "mse_right": p_df.loc[r, "Right (idx 23)"],
                "diff_left_minus_interior": diff_left_interior[r],
                "diff_right_minus_interior": diff_right_interior[r],
                "diff_left_minus_right": diff_left_right[r],
            })

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(data_dir / "temporal_position_paired_comparisons.csv", index=False)
    print(f"Saved paired comparisons: {data_dir / 'temporal_position_paired_comparisons.csv'}")

    # Generate Figures
    plot_position_figures(
        df_results=df_results,
        qualitative_cache=qualitative_cache,
        figures_dir=figures_dir,
        target_frame=args.target_frame,
    )

    # Save Manifest
    manifest = {
        "experiment": "fixed_target_temporal_position_ablation",
        "checkpoint": str(args.run_dir / args.checkpoint),
        "split": str(args.split),
        "target_frame": args.target_frame,
        "eligible_runs": eligible_runs,
        "n_runs": len(eligible_runs),
        "b_conditions": b_conditions,
        "configurations": configs,
        "output_files": [
            str(data_dir / "temporal_position_results.csv"),
            str(data_dir / "temporal_position_paired_comparisons.csv"),
            str(figures_dir / "figure1_reconstruction_error_vs_temporal_position.png"),
            str(figures_dir / "figure2_predicted_variance_and_calibration.png"),
            str(figures_dir / "figure3_representative_fixed_frame_reconstruction.png"),
        ],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
