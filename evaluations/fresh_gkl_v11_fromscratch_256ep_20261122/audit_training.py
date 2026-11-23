#!/usr/bin/env python3
"""Audit 256-epoch training history, allocation boundaries, and curriculum evolution
for multi_beta_gkl_v11_fromscratch_256ep_20261122.
"""

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path("/pscratch/sd/b/binxia/ai4plasma")
RUN_DIR = PROJECT / "runs/multi_beta_gkl_v11_fromscratch_256ep_20261122"
EVAL_DIR = PROJECT / "evaluations/fresh_gkl_v11_fromscratch_256ep_20261122"
V9_DIR = PROJECT / "runs/multi_beta_three_phase_gkl_joint_grad_globalcosine_256ep_20261119"

def audit_training():
    history_file = RUN_DIR / "history.json"
    with open(history_file) as f:
        history = json.load(f)

    # Load v9 baseline history for comparison if available
    v9_history = None
    v9_history_file = V9_DIR / "history.json"
    if v9_history_file.is_file():
        with open(v9_history_file) as f:
            v9_history = json.load(f)

    print(f"Loaded {len(history)} epochs from {history_file}")

    # Prepare tabular data
    csv_rows = []
    epochs = []
    lrs = []
    train_mses = []
    val_mses = []
    train_gkls = []
    val_gkls = []
    lambda_vs = []
    lambda_rs = []
    grad_means = []
    grad_maxs = []
    patterns = ["spatial_random", "spatial_grid", "spatial_block", "temporal_random", "temporal_block"]
    pattern_val_mses = {p: [] for p in patterns}

    phase_transitions = {}
    prev_phase = None

    for i, entry in enumerate(history):
        ep = entry["epoch"]
        epochs.append(ep)
        lr = entry["learning_rate"]
        lrs.append(lr)
        train_mse = entry.get("train_mse", float("nan"))
        val_mse = entry.get("val_mse", float("nan"))
        train_mses.append(train_mse)
        val_mses.append(val_mse)

        train_gkl = entry.get("train_generalized_kl", float("nan"))
        val_gkl = entry.get("val_generalized_kl", float("nan"))
        train_gkls.append(train_gkl)
        val_gkls.append(val_gkl)

        curr = entry.get("curriculum", {})
        lv = curr.get("lambda_v_mean", 0.0)
        lr_w = curr.get("lambda_r_mean", 0.0)
        lambda_vs.append(lv)
        lambda_rs.append(lr_w)

        phase_name = curr.get("phase_name", "unknown")
        if phase_name != prev_phase:
            phase_transitions[ep] = phase_name
            prev_phase = phase_name

        gmean = entry.get("gradient_norm_mean", float("nan"))
        gmax = entry.get("gradient_norm_max", float("nan"))
        grad_means.append(gmean)
        grad_maxs.append(gmax)

        val_pat = entry.get("val_per_pattern", {})
        row = {
            "epoch": ep,
            "learning_rate": lr,
            "train_mse": train_mse,
            "val_mse": val_mse,
            "train_generalized_kl": train_gkl,
            "val_generalized_kl": val_gkl,
            "phase": phase_name,
            "lambda_v": lv,
            "lambda_r": lr_w,
            "gradient_norm_mean": gmean,
            "gradient_norm_max": gmax,
        }
        for p in patterns:
            p_mse = val_pat.get(p, {}).get("mse", float("nan"))
            pattern_val_mses[p].append(p_mse)
            row[f"val_mse_{p}"] = p_mse
            row[f"val_gkl_{p}"] = val_pat.get(p, {}).get("generalized_kl", float("nan"))

        csv_rows.append(row)

    # Save training_history.csv
    csv_path = EVAL_DIR / "training_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote {csv_path}")

    # Boundary analysis
    # Allocation 1: 0-117, Alloc 2: 117-181, Alloc 3: 181-245, Alloc 4: 245-256
    allocation_boundaries = [117, 181, 245]
    boundary_audit = {}
    for b in allocation_boundaries:
        before = history[b - 1]
        after = history[b]
        boundary_audit[f"epoch_{b}_handoff"] = {
            "lr_before": before["learning_rate"],
            "lr_after": after["learning_rate"],
            "lr_ratio": after["learning_rate"] / before["learning_rate"] if before["learning_rate"] > 0 else 1.0,
            "val_mse_before": before["val_mse"],
            "val_mse_after": after["val_mse"],
            "train_mse_before": before["train_mse"],
            "train_mse_after": after["train_mse"],
            "grad_norm_mean_before": before["gradient_norm_mean"],
            "grad_norm_mean_after": after["gradient_norm_mean"],
            "phase_before": before.get("curriculum", {}).get("phase_name"),
            "phase_after": after.get("curriculum", {}).get("phase_name"),
            "lambda_v_before": before.get("curriculum", {}).get("lambda_v_mean"),
            "lambda_v_after": after.get("curriculum", {}).get("lambda_v_mean"),
            "lambda_r_before": before.get("curriculum", {}).get("lambda_r_mean"),
            "lambda_r_after": after.get("curriculum", {}).get("lambda_r_mean"),
        }

    # Best epoch & summary
    best_epoch_idx = int(np.nanargmin(val_mses))
    best_epoch = epochs[best_epoch_idx]
    best_val_mse = val_mses[best_epoch_idx]
    final_epoch = epochs[-1]
    final_val_mse = val_mses[-1]

    audit_summary = {
        "run_name": "multi_beta_gkl_v11_fromscratch_256ep_20261122",
        "total_epochs": len(epochs),
        "target_epochs": 256,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "final_epoch": final_epoch,
        "final_val_mse": final_val_mse,
        "final_train_mse": train_mses[-1],
        "final_val_gkl": val_gkls[-1],
        "final_train_gkl": train_gkls[-1],
        "final_learning_rate": lrs[-1],
        "phase_transitions": phase_transitions,
        "allocation_boundaries": allocation_boundaries,
        "boundary_continuity_audit": boundary_audit,
        "final_val_per_pattern": {p: pattern_val_mses[p][-1] for p in patterns},
    }

    if v9_history is not None:
        v9_final_val_mse = v9_history[-1]["val_mse"]
        v9_best_val_mse = min(e["val_mse"] for e in v9_history)
        v9_pat = {p: v9_history[-1]["val_per_pattern"][p]["mse"] for p in patterns}
        audit_summary["comparison_with_v9_baseline"] = {
            "v9_final_val_mse": v9_final_val_mse,
            "v9_best_val_mse": v9_best_val_mse,
            "delta_final_val_mse": final_val_mse - v9_final_val_mse,
            "pct_improvement_final_val_mse": (v9_final_val_mse - final_val_mse) / v9_final_val_mse * 100,
            "v9_val_per_pattern": v9_pat,
            "pattern_improvements_pct": {
                p: (v9_pat[p] - pattern_val_mses[p][-1]) / v9_pat[p] * 100
                for p in patterns
            }
        }

    audit_path = EVAL_DIR / "training_audit.json"
    with open(audit_path, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"Wrote {audit_path}")

    # Plot 6-panel training curves
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(3, 2, figsize=(16, 15), sharex=True)

    # 1. MSE
    ax = axes[0, 0]
    ax.plot(epochs, train_mses, label="Train MSE", color="#1f77b4", alpha=0.8, lw=1.5)
    ax.plot(epochs, val_mses, label="Validation MSE", color="#d62728", lw=2)
    if v9_history is not None:
        v9_eps = [e["epoch"] for e in v9_history]
        v9_vals = [e["val_mse"] for e in v9_history]
        ax.plot(v9_eps, v9_vals, label="v9 Baseline Val MSE", color="gray", lw=1.2, linestyle="--", alpha=0.7)
    for b in allocation_boundaries:
        ax.axvline(b, color="black", linestyle=":", alpha=0.6, lw=1.2)
    ax.axvline(best_epoch, color="green", linestyle="-.", alpha=0.7, label=f"Best Val (ep {best_epoch})")
    ax.set_ylabel("MSE (Physical / Standardized Units)")
    ax.set_title("Reconstruction Loss (Train & Validation)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 2. Generalized KL Loss
    ax = axes[0, 1]
    ax.plot(epochs, train_gkls, label="Train GKL", color="#2ca02c", alpha=0.8, lw=1.5)
    ax.plot(epochs, val_gkls, label="Validation GKL", color="#9467bd", lw=2)
    for b in allocation_boundaries:
        ax.axvline(b, color="black", linestyle=":", alpha=0.6, lw=1.2)
    ax.set_ylabel("Generalized KL Loss")
    ax.set_title("Uncertainty Loss: Generalized KL")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 3. Curriculum Weights
    ax = axes[1, 0]
    ax.plot(epochs, lambda_vs, label=r"$\lambda_v$ (Variance loss weight)", color="#ff7f0e", lw=2)
    ax.plot(epochs, lambda_rs, label=r"$\lambda_r$ (Rollout reuse weight)", color="#e377c2", lw=2)
    for ep, ph in phase_transitions.items():
        ax.axvline(ep, color="blue", linestyle="--", alpha=0.5, lw=1)
        ax.text(ep + 1, 0.1, ph.replace("_", " "), rotation=90, fontsize=8, color="blue")
    for b in allocation_boundaries:
        ax.axvline(b, color="black", linestyle=":", alpha=0.6, lw=1.2)
    ax.set_ylabel("Curriculum Weight")
    ax.set_title(r"Curriculum Transition Dynamics ($\lambda_v, \lambda_r$)")
    ax.legend(loc="center right")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 4. Learning Rate Schedule
    ax = axes[1, 1]
    ax.plot(epochs, lrs, label="Global Cosine LR", color="#8c564b", lw=2)
    for b in allocation_boundaries:
        ax.axvline(b, color="black", linestyle=":", alpha=0.6, lw=1.2, label="Allocation Boundary" if b == 117 else "")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Evolution (Cosine Decay to 1e-5)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 5. Gradient Norms
    ax = axes[2, 0]
    ax.plot(epochs, grad_means, label="Mean Grad Norm", color="#17becf", lw=1.5)
    ax.plot(epochs, grad_maxs, label="Max Grad Norm", color="#bcbd22", alpha=0.6, lw=1)
    for b in allocation_boundaries:
        ax.axvline(b, color="black", linestyle=":", alpha=0.6, lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient L2 Norm")
    ax.set_title("Gradient Norms & Optimization Stability")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # 6. Validation MSE by Mask Pattern
    ax = axes[2, 1]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for p, c in zip(patterns, colors):
        ax.plot(epochs, pattern_val_mses[p], label=p, color=c, lw=1.5)
        if v9_history is not None:
            v9_pat_p = [e["val_per_pattern"][p]["mse"] for e in v9_history]
            ax.plot(epochs, v9_pat_p, color=c, lw=1, linestyle="--", alpha=0.4)
    for b in allocation_boundaries:
        ax.axvline(b, color="black", linestyle=":", alpha=0.6, lw=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation MSE")
    ax.set_title("Validation MSE by Mask Family (Solid: v11, Dashed: v9)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = EVAL_DIR / "training_curves.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved {plot_path}")

if __name__ == "__main__":
    audit_training()
