# tests/test_backward_compatibility_audit.py

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from compare_phase_d_backward_compatibility import (
    PHASE_A_PATTERNS,
    compare_phase_a,
    compare_phase_b,
    compare_phase_c,
    compute_paired_metric_stats,
    generate_markdown_report,
)
from data.vpic_hdf5_dataset import fixed_split_sha256, validate_final_checkpoint
from prepare_phase_d_audited_checkpoint import build_audited_checkpoint


def test_build_audited_checkpoint_preserves_provenance(tmp_path: Path):
    """Verify final_audited.pt inherits v11 arch metadata while preserving Phase D weights & args."""
    v11_path = tmp_path / "v11_final.pt"
    phd_path = tmp_path / "phd_final.pt"
    out_path = tmp_path / "phd_out" / "final_audited.pt"

    state = {"conv.weight": torch.ones((4, 8, 3, 3, 3))}
    stats = {"mean": [0.0, 0.0, 0.0, 0.5], "std": [1.0, 1.0, 1.0, 0.5]}

    torch.save(
        {
            "epoch": 256,
            "model": state,
            "stats": stats,
            "args": {
                "base_channels": 24,
                "channel_mults": [1, 2, 4, 8],
                "model_version": "direct_unet3d_no_activation_norm_v3",
                "uncertainty_aware": True,
                "use_attention": True,
                "spatial_only_pooling": True,
                "delta_t": 24,
                "stride_t": 2,
                "final_epoch": 256,
            },
            "split_sha256": fixed_split_sha256(),
        },
        v11_path,
    )

    phd_state = {"conv.weight": torch.full((4, 8, 3, 3, 3), 2.0)}
    torch.save(
        {
            "epoch": 6,
            "global_step": 3000,
            "model": phd_state,
            "stats": stats,
            "args": {
                "source_checkpoint": Path("runs/v11/final.pt"),
                "epochs": 6,
                "lr": 2e-6,
                "rollout_depth": 6,
            },
        },
        phd_path,
    )

    v11_jy = tmp_path / "jy_stats.json"
    v11_jy.write_text(json.dumps({"channel_mean": stats["mean"], "channel_std": stats["std"], "jy_std_train": 1.0}))

    info = build_audited_checkpoint(v11_path, phd_path, out_path, copy_jy_stats=True)
    assert info["epoch"] == 6
    assert info["jy_stats_staged"] is True
    assert (out_path.parent / "jy_stats.json").exists()

    audited = torch.load(out_path, map_location="cpu", weights_only=False)
    validate_final_checkpoint(audited)
    assert torch.equal(audited["model"]["conv.weight"], phd_state["conv.weight"])
    assert audited["args"]["base_channels"] == 24
    assert audited["args"]["uncertainty_aware"] is True
    assert audited["args"]["final_epoch"] == 6
    assert audited["phase_d_args"]["lr"] == 2e-6
    assert isinstance(audited["phase_d_args"]["source_checkpoint"], str)


def test_paired_metric_stats_sign_convention():
    """Verify negative rel_change_pct means error reduction and positive means degradation."""
    v11 = [0.10, 0.20, 0.30, 0.40, 0.50]
    phd = [0.11, 0.22, 0.33, 0.44, 0.55]  # 10% worse error
    res = compute_paired_metric_stats(v11, phd, higher_is_better=False)
    assert res["n_runs"] == 5
    assert math.isclose(res["rel_change_pct"], 10.0, rel_tol=1e-6)
    assert res["runs_improved"] == 0
    assert res["runs_degraded"] == 5
    assert math.isfinite(res["p_value_wilcoxon"])


def test_compare_phases_abc_and_report(tmp_path: Path):
    """End-to-end synthetic verification of Phase A, B, C comparison and markdown output."""
    runs = [f"beta0.2_run_{i}" for i in range(6)]

    # Phase A synthetic JSONs
    for tag, mult in [("v11", 1.0), ("phd", 1.05)]:
        d = tmp_path / f"pa_{tag}"
        d.mkdir()
        per_run = {}
        for pat in PHASE_A_PATTERNS:
            per_run[pat] = {
                r: {
                    "overall_mse": (0.01 + 0.001 * i) * mult,
                    "density_nrmse": (0.05 + 0.002 * i) * mult,
                    "jy_nrmse": (0.08 + 0.003 * i) * mult,
                }
                for i, r in enumerate(runs)
            }
        (d / "standardized_validation.json").write_text(
            json.dumps({"n_windows": 60, "n_runs": 6, "per_run": per_run})
        )

    # Phase B synthetic JSON + NPZ
    rng = np.random.default_rng(0)
    for tag, scale in [("v11", 1.0), ("phd", 1.2)]:
        d = tmp_path / f"pb_{tag}"
        d.mkdir()
        (d / "density_calibration.json").write_text(
            json.dumps(
                {
                    "mean_predicted": 0.01 * scale,
                    "mean_actual": 0.01 * scale,
                    "mean_bias_predicted_minus_actual": 0.0,
                    "relative_mean_bias": 0.0,
                    "increasing_mean_transitions": 9,
                    "total_mean_transitions": 9,
                    "upper_prediction_bin_actual_over_predicted": 1.0,
                }
            )
        )
        n_vox = len(runs) * len(PHASE_A_PATTERNS) * 20
        run_arr = np.repeat(runs, len(PHASE_A_PATTERNS) * 20)
        pat_arr = np.tile(np.repeat(list(PHASE_A_PATTERNS), 20), len(runs))
        beta_arr = np.full(n_vox, 0.2)
        pred_err = rng.uniform(0.001, 0.02, size=n_vox) * scale
        act_err = pred_err * rng.uniform(0.8, 1.2, size=n_vox)
        np.savez(
            d / "density_calibration_voxels.npz",
            run_name=run_arr,
            pattern=pat_arr,
            beta=beta_arr,
            predicted_squared_error=pred_err,
            actual_squared_error=act_err,
        )

    # Phase C synthetic CSVs
    for tag, mse_a, mse_b in [("v11", 0.002, 0.015), ("phd", 0.004, 0.002)]:
        d = tmp_path / f"pc_{tag}" / "data"
        d.mkdir(parents=True)
        rows = []
        for b_cond in ["B_full", "B_hidden"]:
            for strat, base in [("A", mse_a), ("B", mse_b), ("C", 0.005)]:
                for i, r in enumerate(runs):
                    rows.append(
                        {
                            "run_name": r,
                            "target_frame": 24,
                            "b_condition": b_cond,
                            "strategy": strat,
                            "density_mse_hidden": base + 0.0001 * i,
                            "mean_var_hidden": base,
                            "u_ratio_hidden": 1.0,
                            "pearson_r": 0.7,
                            "spearman_r": 0.65,
                        }
                    )
        pd.DataFrame(rows).to_csv(d / "feedback_results.csv", index=False)

    res = {
        "v11_checkpoint": "v11.pt",
        "phased_checkpoint": "phd.pt",
        "phase_a": compare_phase_a(tmp_path / "pa_v11", tmp_path / "pa_phd"),
        "phase_b": compare_phase_b(tmp_path / "pb_v11", tmp_path / "pb_phd"),
        "phase_c": compare_phase_c(tmp_path / "pc_v11", tmp_path / "pc_phd"),
    }
    assert res["phase_a"]["available"] is True
    assert res["phase_b"]["available"] is True
    assert res["phase_c"]["available"] is True
    assert math.isclose(res["phase_a"]["aggregate_across_patterns"]["overall_mse"]["rel_change_pct"], 5.0, rel_tol=1e-5)

    md_path = tmp_path / "backward_compatibility_summary.md"
    generate_markdown_report(res, md_path)
    text = md_path.read_text()
    assert "## A. Phase A Reconstruction" in text
    assert "## B. Phase B Uncertainty Calibration" in text
    assert "## C. Phase C Historical Conditioning" in text
    assert "## D. Overall Assessment" in text
