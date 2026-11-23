#!/usr/bin/env python3
"""Targeted Sliding-Window Overwriting & Uncertainty Ablation for ai4plasma v11.

Compares three controlled variants:
  Variant A: Existing last-write-wins baseline. Overwrites predictions & variances.
  Variant B: First-write-wins output evaluation (extracted from Trajectory A).
  Variant C: Frozen-state recursive reconstruction (retains first write in conditioning).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    reconstruct_with_slide_step,
    rmse_mae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sliding-window overwriting ablation for ai4plasma v11."
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
        default=Path("evaluations/physworld_v11_sliding_overwrite_ablation_20261123"),
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
        "--window-size",
        type=int,
        default=24,
        help="Context window size.",
    )
    parser.add_argument(
        "--slide-steps",
        type=int,
        nargs="+",
        default=[1, 12, 24],
        help="Sliding update strides to evaluate.",
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


def evaluate_run_ablation(
    model: torch.nn.Module,
    mean: torch.Tensor,
    std: torch.Tensor,
    target: torch.Tensor,
    probe_mask: torch.Tensor,
    window_size: int,
    step: int,
    b_condition: str,
    run_seed: int,
    x_index: int,
) -> Dict:
    hide_magnetic = (b_condition == "B_hidden")
    target_normalized = (target - mean) / (std + 1e-8)
    target_density_plot = target[3].detach().cpu().numpy()
    target_density_norm = target_normalized[3].detach().cpu().numpy()
    hidden_mask = ~(probe_mask.detach().cpu().numpy() > 0.5)

    # 1. Trajectory 1: Variant A (last_write) & Variant B (first_write)
    res_ab = reconstruct_with_slide_step(
        model=model,
        target_normalized=target_normalized,
        target_density_plot=target_density_plot,
        density_mean=mean[3],
        density_std=std[3],
        probe_mask=probe_mask,
        window_size=window_size,
        slide_step=step,
        plot_units="physical",
        x_index=x_index,
        amp=True,
        hide_magnetic=hide_magnetic,
        noise_seed=run_seed,
        overwrite_policy="last_write",
        track_overwrites=True,
    )

    # 2. Trajectory 2: Variant C (frozen)
    res_c = reconstruct_with_slide_step(
        model=model,
        target_normalized=target_normalized,
        target_density_plot=target_density_plot,
        density_mean=mean[3],
        density_std=std[3],
        probe_mask=probe_mask,
        window_size=window_size,
        slide_step=step,
        plot_units="physical",
        x_index=x_index,
        amp=True,
        hide_magnetic=hide_magnetic,
        noise_seed=run_seed,
        overwrite_policy="frozen",
        track_overwrites=True,
    )

    # Extract Variant A arrays (last write)
    pred_a_phys = np.asarray(res_ab["final_reconstruction"])
    pred_a_norm = (pred_a_phys - float(mean[3].item())) / (float(std[3].item()) + 1e-8)
    var_a_norm = np.asarray(res_ab["predicted_squared_error_normalized"])

    # Extract Variant B arrays (first write)
    pred_b_phys = np.asarray(res_ab["first_write_reconstruction"])
    pred_b_norm = (pred_b_phys - float(mean[3].item())) / (float(std[3].item()) + 1e-8)
    var_b_norm = np.asarray(res_ab["first_write_squared_error_normalized"])

    # Extract Variant C arrays (frozen)
    pred_c_phys = np.asarray(res_c["final_reconstruction"])
    pred_c_norm = (pred_c_phys - float(mean[3].item())) / (float(std[3].item()) + 1e-8)
    var_c_norm = np.asarray(res_c["predicted_squared_error_normalized"])

    write_counts = np.asarray(res_ab["write_counts"])
    overwrite_events = res_ab["overwrite_events"]

    variants_data = {
        "A": {"pred_phys": pred_a_phys, "pred_norm": pred_a_norm, "var_norm": var_a_norm},
        "B": {"pred_phys": pred_b_phys, "pred_norm": pred_b_norm, "var_norm": var_b_norm},
        "C": {"pred_phys": pred_c_phys, "pred_norm": pred_c_norm, "var_norm": var_c_norm},
    }

    n_frames = target.shape[1]
    metrics = {}
    for var_key, vdata in variants_data.items():
        pn = vdata["pred_norm"]
        vn = vdata["var_norm"]
        diff = pn - target_density_norm
        e2_all = np.square(diff)
        full_nrmse = float(np.sqrt(np.mean(e2_all)))

        sel = np.broadcast_to(hidden_mask, pn.shape)
        e2_hid = e2_all[sel]
        v_hid = vn[sel]
        mean_e2 = float(np.mean(e2_hid))
        mean_var = float(np.mean(v_hid))
        u_ratio = mean_var / (mean_e2 + 1e-12)

        # Framewise metrics
        framewise_e2_hidden = []
        framewise_var_hidden = []
        framewise_e2_full = []
        framewise_var_full = []
        for t in range(n_frames):
            e2_t_hid = np.mean(e2_all[t][hidden_mask])
            var_t_hid = np.mean(vn[t][hidden_mask])
            e2_t_full = np.mean(e2_all[t])
            var_t_full = np.mean(vn[t])
            framewise_e2_hidden.append(float(e2_t_hid))
            framewise_var_hidden.append(float(var_t_hid))
            framewise_e2_full.append(float(e2_t_full))
            framewise_var_full.append(float(var_t_full))

        metrics[var_key] = {
            "density_nrmse": full_nrmse,
            "mean_e2_hidden": mean_e2,
            "mean_var_hidden": mean_var,
            "u_ratio_hidden": u_ratio,
            "framewise_e2_hidden": framewise_e2_hidden,
            "framewise_var_hidden": framewise_var_hidden,
            "framewise_e2_full": framewise_e2_full,
            "framewise_var_full": framewise_var_full,
        }

    return {
        "metrics": metrics,
        "write_counts": write_counts,
        "overwrite_events": overwrite_events,
        "variants_data": variants_data,
        "target_density_plot": target_density_plot,
        "target_density_norm": target_density_norm,
        "hidden_mask": hidden_mask,
    }


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

    print(f"Eligible held-out test runs ({len(eligible_runs)}): {eligible_runs}")

    # Results collections
    summary_rows = []
    framewise_rows = []
    per_run_paired_rows = []
    overwrite_event_rows = []
    qualitative_cache = None

    slide_steps = args.slide_steps
    b_conditions = ["B_full", "B_hidden"]

    for r_idx, run_name in enumerate(eligible_runs):
        fields, _frame_ids, _h5_path = find_and_load_run(args.h5_dir, args.betas, run_name)
        fields = np.asarray(fields)[: args.n_frames]
        target = torch.from_numpy(np.transpose(fields, (1, 0, 2, 3))).to(
            device=device, dtype=torch.float32
        )
        target_normalized = (target - mean) / (std + 1e-8)
        probe_mask, _ = sliding_density_probe_mask(
            target_normalized.unsqueeze(0),
            args.density_probe_count,
        )

        for b_cond in b_conditions:
            for step in slide_steps:
                run_seed = args.seed + r_idx * 1000 + step * 31
                res = evaluate_run_ablation(
                    model=model,
                    mean=mean,
                    std=std,
                    target=target,
                    probe_mask=probe_mask,
                    window_size=args.window_size,
                    step=step,
                    b_condition=b_cond,
                    run_seed=run_seed,
                    x_index=args.x_index,
                )

                # Save qualitative example for step 1, B_full, canonical run
                if qualitative_cache is None and run_name == "beta0.2_nu2_Bz0_dt2_tau70" and step == 1 and b_cond == "B_full":
                    qualitative_cache = {
                        "run_name": run_name,
                        "step": step,
                        "b_condition": b_cond,
                        "target_plot": res["target_density_plot"],
                        "pred_a": res["variants_data"]["A"]["pred_phys"],
                        "pred_b": res["variants_data"]["B"]["pred_phys"],
                        "pred_c": res["variants_data"]["C"]["pred_phys"],
                        "write_counts": res["write_counts"],
                    }

                # Record per-variant metrics
                m = res["metrics"]
                for var_key in ["A", "B", "C"]:
                    vm = m[var_key]
                    summary_rows.append({
                        "run_name": run_name,
                        "b_condition": b_cond,
                        "step": step,
                        "variant": var_key,
                        "density_nrmse": vm["density_nrmse"],
                        "mean_e2_hidden": vm["mean_e2_hidden"],
                        "mean_var_hidden": vm["mean_var_hidden"],
                        "u_ratio_hidden": vm["u_ratio_hidden"],
                    })

                    # Record framewise metrics
                    for t in range(args.n_frames):
                        framewise_rows.append({
                            "run_name": run_name,
                            "b_condition": b_cond,
                            "step": step,
                            "variant": var_key,
                            "global_frame": t,
                            "write_count": int(res["write_counts"][t]),
                            "e2_hidden": vm["framewise_e2_hidden"][t],
                            "var_hidden": vm["framewise_var_hidden"][t],
                            "e2_full": vm["framewise_e2_full"][t],
                            "var_full": vm["framewise_var_full"][t],
                        })

                # Paired differences: B - A and C - A
                per_run_paired_rows.append({
                    "run_name": run_name,
                    "b_condition": b_cond,
                    "step": step,
                    "nrmse_A": m["A"]["density_nrmse"],
                    "nrmse_B": m["B"]["density_nrmse"],
                    "nrmse_C": m["C"]["density_nrmse"],
                    "diff_nrmse_B_minus_A": m["B"]["density_nrmse"] - m["A"]["density_nrmse"],
                    "diff_nrmse_C_minus_A": m["C"]["density_nrmse"] - m["A"]["density_nrmse"],
                    "e2_hidden_A": m["A"]["mean_e2_hidden"],
                    "e2_hidden_B": m["B"]["mean_e2_hidden"],
                    "e2_hidden_C": m["C"]["mean_e2_hidden"],
                    "diff_e2_B_minus_A": m["B"]["mean_e2_hidden"] - m["A"]["mean_e2_hidden"],
                    "diff_e2_C_minus_A": m["C"]["mean_e2_hidden"] - m["A"]["mean_e2_hidden"],
                    "u_ratio_A": m["A"]["u_ratio_hidden"],
                    "u_ratio_B": m["B"]["u_ratio_hidden"],
                    "u_ratio_C": m["C"]["u_ratio_hidden"],
                })

                # Overwrite events
                for ev in res["overwrite_events"]:
                    overwrite_event_rows.append({
                        "run_name": run_name,
                        "b_condition": b_cond,
                        "step": step,
                        **ev,
                    })

    # Save DataFrames
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(data_dir / "ablation_summary.csv", index=False)
    print(f"Saved: {data_dir / 'ablation_summary.csv'}")

    df_framewise = pd.DataFrame(framewise_rows)
    df_framewise.to_csv(data_dir / "framewise_metrics.csv", index=False)
    print(f"Saved: {data_dir / 'framewise_metrics.csv'}")

    df_paired = pd.DataFrame(per_run_paired_rows)
    df_paired.to_csv(data_dir / "per_run_paired_metrics.csv", index=False)
    print(f"Saved: {data_dir / 'per_run_paired_metrics.csv'}")

    df_events = pd.DataFrame(overwrite_event_rows)
    df_events.to_csv(data_dir / "overwrite_events.csv", index=False)
    print(f"Saved: {data_dir / 'overwrite_events.csv'}")

    # Generate Figures
    generate_ablation_figures(
        df_framewise=df_framewise,
        df_paired=df_paired,
        df_events=df_events,
        qualitative_cache=qualitative_cache,
        figures_dir=figures_dir,
    )

    # Write Manifest
    manifest = {
        "experiment": "sliding_window_overwriting_ablation",
        "checkpoint": str(args.run_dir / args.checkpoint),
        "split": str(args.split),
        "eligible_runs": eligible_runs,
        "n_runs": len(eligible_runs),
        "slide_steps": slide_steps,
        "b_conditions": b_conditions,
        "variants": {
            "A": "Existing last-write-wins baseline",
            "B": "First-write-wins output evaluation (Trajectory A forward passes)",
            "C": "Frozen-state recursive reconstruction (retained predictions in conditioning)",
        },
        "output_files": [
            str(data_dir / "ablation_summary.csv"),
            str(data_dir / "framewise_metrics.csv"),
            str(data_dir / "per_run_paired_metrics.csv"),
            str(data_dir / "overwrite_events.csv"),
            str(figures_dir / "figure1_framewise_mse_variance_abc.png"),
            str(figures_dir / "figure2_first_vs_last_write_error.png"),
            str(figures_dir / "figure3_error_vs_overwrite_count.png"),
            str(figures_dir / "figure4_per_run_paired_comparisons.png"),
            str(figures_dir / "figure5_representative_qualitative_abc.png"),
        ],
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {output_dir / 'manifest.json'}")


def generate_ablation_figures(
    df_framewise: pd.DataFrame,
    df_paired: pd.DataFrame,
    df_events: pd.DataFrame,
    qualitative_cache: Dict | None,
    figures_dir: Path,
):
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    labels = {
        "A": "Variant A: Last-write (Baseline)",
        "B": "Variant B: First-write (No overwrite eval)",
        "C": "Variant C: Frozen-state (No overwrite state)",
    }

    # Figure 1: Framewise MSE vs predicted variance across global frames
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for row_idx, cond in enumerate(["B_full", "B_hidden"]):
        # Step 1
        ax_err = axes[row_idx, 0]
        ax_var = axes[row_idx, 1]
        for v in ["A", "B", "C"]:
            sub = df_framewise[(df_framewise["b_condition"] == cond) & (df_framewise["step"] == 1) & (df_framewise["variant"] == v)]
            grp = sub.groupby("global_frame")[["e2_hidden", "var_hidden"]].mean()
            ax_err.plot(grp.index, grp["e2_hidden"], label=labels[v], color=colors[v], lw=2)
            ax_var.plot(grp.index, grp["var_hidden"], label=labels[v], color=colors[v], lw=2)

        ax_err.set_title(f"{cond} (step=1): Framewise MSE (Hidden voxels)", fontsize=10)
        ax_err.set_ylabel("Standardized MSE (e²)")
        ax_err.grid(True, alpha=0.3)
        ax_err.legend(fontsize=8)

        ax_var.set_title(f"{cond} (step=1): Framewise Predicted Variance", fontsize=10)
        ax_var.set_ylabel("Standardized Variance (u)")
        ax_var.grid(True, alpha=0.3)
        ax_var.legend(fontsize=8)

    axes[1, 0].set_xlabel("Global Frame (0..47)")
    axes[1, 1].set_xlabel("Global Frame (0..47)")
    fig.suptitle("Figure 1: Framewise Squared Error vs. Predicted Variance (step=1)", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_framewise_mse_variance_abc.png", dpi=200)
    fig.savefig(figures_dir / "figure1_framewise_mse_variance_abc.pdf")
    plt.close(fig)

    # Figure 2: First-write versus last-write error across global frames
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        for step, ls in zip([1, 12, 24], ["-", "--", ":"]):
            sub_a = df_framewise[(df_framewise["b_condition"] == cond) & (df_framewise["step"] == step) & (df_framewise["variant"] == "A")].groupby("global_frame")["e2_hidden"].mean()
            sub_b = df_framewise[(df_framewise["b_condition"] == cond) & (df_framewise["step"] == step) & (df_framewise["variant"] == "B")].groupby("global_frame")["e2_hidden"].mean()
            ax.plot(sub_a.index, sub_a.values, color=colors["A"], linestyle=ls, label=f"Last-write (s={step})")
            ax.plot(sub_b.index, sub_b.values, color=colors["B"], linestyle=ls, label=f"First-write (s={step})")

        ax.axvline(23.5, color="gray", linestyle="--", alpha=0.7, label="Initial context boundary (frame 23)")
        ax.set_title(f"{cond}: First-write vs. Last-write MSE", fontsize=11)
        ax.set_xlabel("Global Frame (0..47)")
        ax.set_ylabel("Standardized MSE (e²)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Figure 2: First-Write vs. Last-Write Error Across Rollout Trajectory", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_first_vs_last_write_error.png", dpi=200)
    fig.savefig(figures_dir / "figure2_first_vs_last_write_error.pdf")
    plt.close(fig)

    # Figure 3: Reconstruction error versus number of times each frame was overwritten
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub = df_events[(df_events["b_condition"] == cond) & (df_events["step"] == 1)]
        # Group by write_count
        grp = sub.groupby("write_count")[["e2_hidden", "var_hidden"]].mean()
        ax.plot(grp.index, grp["e2_hidden"], marker="o", color="firebrick", lw=2, label="Actual MSE (e²)")
        ax.plot(grp.index, grp["var_hidden"], marker="s", color="navy", lw=2, label="Predicted Variance (u)")
        ax.set_title(f"{cond} (step=1): Error & Variance vs. Overwrite Count", fontsize=11)
        ax.set_xlabel("Number of times frame was predicted / updated")
        ax.set_ylabel("Standardized squared units")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle("Figure 3: Degradation of Accuracy & Uncertainty vs. Cumulative Overwrites", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure3_error_vs_overwrite_count.png", dpi=200)
    fig.savefig(figures_dir / "figure3_error_vs_overwrite_count.pdf")
    plt.close(fig)

    # Figure 4: Per-run paired comparisons for step=1 and step=24
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for idx, cond in enumerate(["B_full", "B_hidden"]):
        ax = axes[idx]
        sub = df_paired[(df_paired["b_condition"] == cond) & (df_paired["step"] == 1)]
        x = np.arange(len(sub))
        width = 0.35
        ax.bar(x - width/2, sub["diff_nrmse_B_minus_A"], width, label="Variant B − A (First − Last)", color="#ff7f0e", alpha=0.8)
        ax.bar(x + width/2, sub["diff_nrmse_C_minus_A"], width, label="Variant C − A (Frozen − Last)", color="#2ca02c", alpha=0.8)
        ax.axhline(0, color="k", linestyle="--", lw=1)
        ax.set_title(f"{cond} (step=1): Paired NRMSE Difference vs. Baseline A", fontsize=10)
        ax.set_xlabel("Test Run Index (10 independent runs)")
        ax.set_ylabel("Δ NRMSE (< 0 indicates improvement over A)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"R{i+1}" for i in range(len(sub))], fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Figure 4: Independent-Run Paired Differences (step=1)", fontsize=12, y=0.98)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure4_per_run_paired_comparisons.png", dpi=200)
    fig.savefig(figures_dir / "figure4_per_run_paired_comparisons.pdf")
    plt.close(fig)

    # Figure 5: Representative qualitative reconstruction comparing A, B, C
    if qualitative_cache is not None:
        target_p = qualitative_cache["target_plot"]
        pred_a = qualitative_cache["pred_a"]
        pred_b = qualitative_cache["pred_b"]
        pred_c = qualitative_cache["pred_c"]

        proj_tgt = project_time_z(target_p, x_index=77)
        proj_a = project_time_z(pred_a, x_index=77)
        proj_b = project_time_z(pred_b, x_index=77)
        proj_c = project_time_z(pred_c, x_index=77)

        fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True, sharey=True)
        # Left: projections. Right: residuals
        res_a = proj_a - proj_tgt
        res_b = proj_b - proj_tgt
        res_c = proj_c - proj_tgt

        vmin, vmax = np.nanpercentile(proj_tgt, [2, 98])
        res_limit = max(abs(np.nanpercentile(res_a, 2)), abs(np.nanpercentile(res_a, 98)))

        # Target
        axes[0, 0].imshow(proj_tgt.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Ground Truth Target (Density)", fontsize=9)
        axes[0, 1].axis("off")

        # Variant A
        axes[1, 0].imshow(proj_a.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[1, 0].set_title("Variant A: Last-Write Baseline", fontsize=9)
        im_ra = axes[1, 1].imshow(res_a.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-res_limit, vmax=res_limit)
        axes[1, 1].set_title("Variant A Residual (A − Target)", fontsize=9)

        # Variant B
        axes[2, 0].imshow(proj_b.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[2, 0].set_title("Variant B: First-Write Output", fontsize=9)
        axes[2, 1].imshow(res_b.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-res_limit, vmax=res_limit)
        axes[2, 1].set_title("Variant B Residual (B − Target)", fontsize=9)

        # Variant C
        axes[3, 0].imshow(proj_c.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        axes[3, 0].set_title("Variant C: Frozen-State Conditioning", fontsize=9)
        axes[3, 1].imshow(res_c.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-res_limit, vmax=res_limit)
        axes[3, 1].set_title("Variant C Residual (C − Target)", fontsize=9)

        for ax in axes[:, 0]:
            ax.set_ylabel("z index")
        axes[3, 0].set_xlabel("Global Frame (0..47)")
        axes[3, 1].set_xlabel("Global Frame (0..47)")

        fig.suptitle(f"Figure 5: Qualitative Time-Z Comparison ({qualitative_cache['run_name']}, step=1, B_full)", fontsize=11, y=0.98)
        fig.tight_layout()
        fig.savefig(figures_dir / "figure5_representative_qualitative_abc.png", dpi=200)
        fig.savefig(figures_dir / "figure5_representative_qualitative_abc.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
