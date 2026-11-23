#!/usr/bin/env python3
"""Historical Prediction Uncertainty Sweep for ai4plasma v11.

Investigates whether increasing the input uncertainty assigned to historical predictions
can prevent the severe degradation observed during online causal reconstruction:
  - Target frame: Global frame F24 (local index 23 in window F1-F24).
  - Common physical window: [F1, F2, ..., F24] (24 frames total).
  - Historical Density values: exactly the same online causal predictions (rho_hat)
    reused across all uncertainty settings.
  - Historical input uncertainty settings for unobserved Density in F1-F23:
      * Baseline: original predicted variance u_hat
      * U001:     0.01
      * U01:      0.1
      * U1:       1.0
  - Invariants:
      * Current F24 observations, unknown noise values, and uncertainties remain identical.
      * Historical genuine probes remain exact with uncertainty 0.0.
      * Magnetic observations and uncertainties remain unchanged.
      * Strict noise matching using reproducible global noise per run.
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
    run_online_history_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run historical prediction uncertainty sweep for ai4plasma v11."
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
        default=Path("evaluations/physworld_v11_historical_uncertainty_sweep_20261123"),
        help="Output directory for sweep results.",
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
    parser.add_argument(
        "--settings",
        type=str,
        nargs="+",
        default=["baseline", "U001", "U01", "U1", "no_feedback_ref"],
        choices=["baseline", "U001", "U01", "U1", "no_feedback_ref"],
        help="Uncertainty sweep settings to evaluate.",
    )
    return parser.parse_args()


def build_uncertainty_sweep_input(
    setting: str,
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    global_noise: torch.Tensor,
    history_pred: torch.Tensor,
    history_var: torch.Tensor,
    target_frame: int = 24,
    hide_magnetic: bool = False,
) -> torch.Tensor:
    """Constructs the 24-frame model input for [target_frame-23, ..., target_frame].

    Target frame F24 occupies local index 23.

    Historical frames l in 0..22 (tau in start..target_frame-1):
      - Probes: exact target_normalized[3, tau], u_in = 0.0
      - Unobserved locations:
          * 'baseline': values = history_pred[tau], u_in = history_var[tau]
          * 'U001':     values = history_pred[tau], u_in = 0.01
          * 'U01':      values = history_pred[tau], u_in = 0.1
          * 'U1':       values = history_pred[tau], u_in = 1.0
          * 'no_feedback_ref': values = global_noise[3, tau], u_in = 1.0

    Current target frame F24 (local index 23):
      - Probes: exact target_normalized[3, 24], u_in = 0.0
      - Unobserved: global_noise[3, 24], u_in = 1.0 (BITWISE IDENTICAL across all settings)
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

    # Map setting to historical unobserved uncertainty value
    fixed_u_map = {
        "U001": 0.01,
        "U01": 0.1,
        "U1": 1.0,
    }

    # Historical frames l in 0..22 (tau in start..target_frame-1)
    for l in range(window_size - 1):
        tau = start + l
        if setting == "no_feedback_ref":
            values[3, l] = window_noise[3, l]
            input_variance[3, l] = 1.0
        elif setting == "baseline":
            values[3, l] = history_pred[tau]
            input_variance[3, l] = history_var[tau]
        elif setting in fixed_u_map:
            values[3, l] = history_pred[tau]
            input_variance[3, l] = fixed_u_map[setting]
        else:
            raise ValueError(f"Unknown sweep setting: {setting}")

        # Genuine probes always win with zero variance
        values[3, l, probe_bool] = window_target[3, l, probe_bool]
        input_variance[3, l, probe_bool] = 0.0

    # Current target frame F24 at local index 23:
    l_curr = window_size - 1
    values[3, l_curr] = window_noise[3, l_curr]
    input_variance[3, l_curr] = 1.0
    values[3, l_curr, probe_bool] = window_target[3, l_curr, probe_bool]
    input_variance[3, l_curr, probe_bool] = 0.0

    model_input = torch.cat([values.unsqueeze(0), input_variance.unsqueeze(0)], dim=1)
    return model_input


@torch.no_grad()
def evaluate_uncertainty_sweep_setting(
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    target_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    global_noise: torch.Tensor,
    history_pred: torch.Tensor,
    history_var: torch.Tensor,
    setting: str,
    target_frame: int = 24,
    hide_magnetic: bool = False,
    amp: bool = True,
) -> Dict:
    device = target_normalized.device
    local_idx = 23

    model_input = build_uncertainty_sweep_input(
        setting=setting,
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
        "setting": setting,
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


def plot_uncertainty_sweep_figure(
    df_results: pd.DataFrame,
    figures_dir: Path,
    target_frame: int,
):
    """Generates the concise comparison figure showing MSE against historical input uncertainty."""
    colors = {"B_full": "#1f77b4", "B_hidden": "#d62728"}
    sweep_settings = ["baseline", "U001", "U01", "U1"]
    active_settings = [s for s in sweep_settings if s in df_results["setting"].unique()]

    setting_labels = {
        "baseline": "Baseline\n(u=u_hat)",
        "U001": "U001\n(u=0.01)",
        "U01": "U01\n(u=0.1)",
        "U1": "U1\n(u=1.0)",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub = df_results[df_results["b_condition"] == cond]

        # Individual run lines
        runs = sub["run_name"].unique()
        for r in runs:
            r_sub = sub[sub["run_name"] == r].set_index("setting")
            y_vals = [r_sub.loc[s, "density_mse_hidden"] if s in r_sub.index else np.nan for s in active_settings]
            ax.plot(range(len(active_settings)), y_vals, color="gray", alpha=0.35, lw=1.2, marker="o", markersize=4)

        # Cross-run Mean
        mean_vals = [sub[sub["setting"] == s]["density_mse_hidden"].mean() for s in active_settings]
        ax.plot(
            range(len(active_settings)),
            mean_vals,
            color=colors[cond],
            lw=2.8,
            marker="s",
            markersize=8,
            label=f"Mean MSE ({cond})",
        )

        # Cross-run Median
        median_vals = [sub[sub["setting"] == s]["density_mse_hidden"].median() for s in active_settings]
        ax.plot(
            range(len(active_settings)),
            median_vals,
            color=colors[cond],
            lw=1.8,
            ls="--",
            marker="^",
            markersize=6,
            label=f"Median MSE ({cond})",
        )

        # Reference line for no_feedback_ref if present
        if "no_feedback_ref" in sub["setting"].unique():
            ref_mean = sub[sub["setting"] == "no_feedback_ref"]["density_mse_hidden"].mean()
            ax.axhline(
                ref_mean,
                color="black",
                ls=":",
                lw=1.5,
                label=f"No-Feedback Ref (Noise, u=1): {ref_mean:.5f}",
            )

        # Annotate mean values on points
        for i, val in enumerate(mean_vals):
            sub_s = sub[sub["setting"] == active_settings[i]]
            u_ratio = sub_s["u_ratio_hidden"].mean()
            ax.text(
                i,
                val + 0.0004,
                f"MSE={val:.4f}\nu/MSE={u_ratio:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
            )

        ax.set_title(f"{cond}: Reconstruction Error vs. Historical Uncertainty (F{target_frame})", fontsize=10)
        ax.set_ylabel("Standardized Hidden-Voxel MSE (e²)")
        ax.set_xticks(range(len(active_settings)))
        ax.set_xticklabels([setting_labels[s] for s in active_settings], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        f"Figure 1: Historical Prediction Uncertainty Sweep (Target Frame F{target_frame})\n"
        "(Fixed Historical Predictions rho_hat, Varied Historical Input Uncertainty u_in)",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_historical_uncertainty_sweep_mse.png", dpi=200)
    fig.savefig(figures_dir / "figure1_historical_uncertainty_sweep_mse.pdf")
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

    print(f"Evaluating uncertainty sweep on F{args.target_frame} across {len(eligible_runs)} eligible runs: {eligible_runs}")

    b_conditions = ["B_full", "B_hidden"]
    settings = [s for s in ["baseline", "U001", "U01", "U1", "no_feedback_ref"] if s in args.settings]

    results_rows = []

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

            # Compute online causal history rollout ONCE per run and magnetic condition
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

            for setting in settings:
                res = evaluate_uncertainty_sweep_setting(
                    model=model,
                    mean=mean,
                    std=std,
                    target_normalized=run_tensor_norm,
                    probe_mask=probe_mask_2d,
                    global_noise=global_noise,
                    history_pred=history_pred,
                    history_var=history_var,
                    setting=setting,
                    target_frame=args.target_frame,
                    hide_magnetic=hide_magnetic,
                    amp=True,
                )

                print(
                    f"    [{setting}] Hidden MSE: {res['density_mse_hidden']:.5f} | "
                    f"RMSE: {res['density_rmse_hidden']:.4f} | "
                    f"Var u: {res['mean_var_hidden']:.5f} | "
                    f"u/MSE: {res['u_ratio_hidden']:.3f} | "
                    f"Pearson r: {res['pearson_r']:.3f}"
                )

                results_rows.append({
                    "run_name": run_name,
                    "target_frame": args.target_frame,
                    "b_condition": b_cond,
                    "setting": setting,
                    "density_mse_hidden": res["density_mse_hidden"],
                    "density_rmse_hidden": res["density_rmse_hidden"],
                    "full_field_rmse": res["full_field_rmse"],
                    "mean_var_hidden": res["mean_var_hidden"],
                    "u_ratio_hidden": res["u_ratio_hidden"],
                    "pearson_r": res["pearson_r"],
                    "spearman_r": res["spearman_r"],
                })

    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(data_dir / "sweep_results.csv", index=False)
    print(f"\nSaved results: {data_dir / 'sweep_results.csv'}")

    # Generate paired comparisons relative to baseline
    paired_rows = []
    summary_rows = []
    comparisons = [s for s in settings if s != "baseline"]

    for b_cond in b_conditions:
        sub = df_results[df_results["b_condition"] == b_cond]
        p_mse = sub.pivot(index="run_name", columns="setting", values="density_mse_hidden")
        p_rmse = sub.pivot(index="run_name", columns="setting", values="density_rmse_hidden")

        # Per-setting aggregated metrics
        for s in settings:
            s_mse = p_mse[s]
            s_sub = sub[sub["setting"] == s]
            summary_rows.append({
                "b_condition": b_cond,
                "setting": s,
                "mean_hidden_mse": float(s_mse.mean()),
                "median_hidden_mse": float(s_mse.median()),
                "std_hidden_mse": float(s_mse.std()),
                "mean_predicted_var": float(s_sub["mean_var_hidden"].mean()),
                "mean_u_ratio": float(s_sub["u_ratio_hidden"].mean()),
            })

        for r in p_mse.index:
            row = {
                "run_name": r,
                "b_condition": b_cond,
                "target_frame": args.target_frame,
                "mse_baseline": p_mse.loc[r, "baseline"],
                "rmse_baseline": p_rmse.loc[r, "baseline"],
            }
            for comp in comparisons:
                if comp in p_mse.columns:
                    diff_m = p_mse.loc[r, comp] - p_mse.loc[r, "baseline"]
                    ratio_m = p_mse.loc[r, comp] / (p_mse.loc[r, "baseline"] + 1e-12)
                    diff_r = p_rmse.loc[r, comp] - p_rmse.loc[r, "baseline"]
                    row[f"mse_{comp}"] = p_mse.loc[r, comp]
                    row[f"diff_mse_{comp}_minus_baseline"] = diff_m
                    row[f"ratio_mse_{comp}_to_baseline"] = ratio_m
                    row[f"diff_rmse_{comp}_minus_baseline"] = diff_r
            paired_rows.append(row)

    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(data_dir / "sweep_paired_comparisons.csv", index=False)
    print(f"Saved paired comparisons: {data_dir / 'sweep_paired_comparisons.csv'}")

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(data_dir / "sweep_summary_table.csv", index=False)
    print(f"Saved summary table: {data_dir / 'sweep_summary_table.csv'}")

    # Print summary to console
    print("\n=== HISTORICAL PREDICTION UNCERTAINTY SWEEP SUMMARY ===")
    for b_cond in b_conditions:
        print(f"\nCondition: {b_cond}")
        sub_sum = df_summary[df_summary["b_condition"] == b_cond]
        for _, r in sub_sum.iterrows():
            print(f"  Setting {r['setting']:16s}: "
                  f"MSE Mean={r['mean_hidden_mse']:.5f}, Median={r['median_hidden_mse']:.5f} | "
                  f"Var u={r['mean_predicted_var']:.5f} | u/MSE={r['mean_u_ratio']:.3f}")

        sub_paired = df_paired[df_paired["b_condition"] == b_cond]
        print(f"  Paired Differences vs. Baseline ({b_cond}):")
        for comp in comparisons:
            col_diff = f"diff_mse_{comp}_minus_baseline"
            col_ratio = f"ratio_mse_{comp}_to_baseline"
            if col_diff in sub_paired.columns:
                diffs = sub_paired[col_diff]
                ratios = sub_paired[col_ratio]
                n_better = int((diffs < 0).sum())
                n_worse = int((diffs > 0).sum())
                print(f"    {comp:16s}: Mean Diff={diffs.mean():+.5f}, Median Diff={diffs.median():+.5f}, "
                      f"Mean Ratio={ratios.mean():.3f}x | Improved (<baseline): {n_better}/{len(diffs)}, "
                      f"Deteriorated (>baseline): {n_worse}/{len(diffs)}")

    # Plot comparison figure
    plot_uncertainty_sweep_figure(
        df_results=df_results,
        figures_dir=figures_dir,
        target_frame=args.target_frame,
    )

    # Manifest
    manifest = {
        "experiment": "historical_prediction_uncertainty_sweep",
        "checkpoint": str(args.run_dir / args.checkpoint),
        "split": str(args.split),
        "target_frame": args.target_frame,
        "eligible_runs": eligible_runs,
        "n_runs": len(eligible_runs),
        "b_conditions": b_conditions,
        "settings": {
            "baseline": "Original predicted variance u_hat from online filtering",
            "U001": "Fixed historical input uncertainty u_in = 0.01",
            "U01": "Fixed historical input uncertainty u_in = 0.1",
            "U1": "Fixed historical input uncertainty u_in = 1.0",
            "no_feedback_ref": "No-feedback baseline reference (noise history, u_in = 1.0)",
        },
        "output_files": [
            str(data_dir / "sweep_results.csv"),
            str(data_dir / "sweep_paired_comparisons.csv"),
            str(data_dir / "sweep_summary_table.csv"),
            str(figures_dir / "figure1_historical_uncertainty_sweep_mse.png"),
            str(figures_dir / "figure1_historical_uncertainty_sweep_mse.pdf"),
        ],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
