#!/usr/bin/env python3
"""Compare Phase A, Phase B, and Phase C evaluations between v11 and Phase D.

Reads outputs produced by:
  - Phase A: `evaluate_standardized_validation.py`
  - Phase B: `evaluate_standardized_validation.py --density-calibration`
  - Phase C: `run_feedback_ablation.py`

Performs paired per-run statistical comparisons (treating physical simulations as
independent units) and writes `paired_backward_compatibility_stats.json` and
`backward_compatibility_summary.md`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

PHASE_A_PATTERNS: Tuple[str, ...] = (
    "spatial_random",
    "spatial_grid",
    "spatial_block",
    "temporal_random",
    "temporal_block",
)

PHASE_C_STRATEGY_LABELS: Dict[str, str] = {
    "A": "A: Noise, u=1",
    "B": "B: Pred-fb, u_hat",
    "C": "C: Oracle GT, u=0",
    "D": "D: Diag GT, u=1",
    "E": "E: Diag Noise, u=0",
}


def compute_paired_metric_stats(
    vals_v11: Sequence[float],
    vals_phd: Sequence[float],
    higher_is_better: bool = False,
) -> Dict[str, Any]:
    """Compute paired summary and Wilcoxon signed-rank stats across physical runs.

    Relative change is defined as `(PhaseD_mean - v11_mean) / |v11_mean| * 100`.
    For error metrics (`higher_is_better=False`), negative relative change indicates
    improvement (lower error) and positive relative change indicates degradation.
    """
    a = np.asarray(list(vals_v11), dtype=np.float64)
    b = np.asarray(list(vals_phd), dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    n = int(a.size)
    if n == 0:
        return {"n_runs": 0}

    diff = b - a  # Phase D - v11
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    med_a = float(np.median(a))
    med_b = float(np.median(b))
    abs_diff_mean = float(np.mean(diff))
    abs_diff_med = float(np.median(diff))

    denom = abs(mean_a)
    rel_change_pct = float((mean_b - mean_a) / denom * 100.0) if denom > 1e-15 else float("nan")

    if higher_is_better:
        improved = int(np.sum(diff > 1e-15))
        degraded = int(np.sum(diff < -1e-15))
    else:
        improved = int(np.sum(diff < -1e-15))
        degraded = int(np.sum(diff > 1e-15))
    tied = n - improved - degraded

    if n >= 5 and np.any(np.abs(diff) > 1e-15):
        try:
            p_wilcoxon = float(stats.wilcoxon(diff).pvalue)
        except Exception:
            p_wilcoxon = float("nan")
    else:
        p_wilcoxon = float("nan")

    return {
        "n_runs": n,
        "v11_mean": mean_a,
        "v11_median": med_a,
        "v11_p16": float(np.percentile(a, 16.0)),
        "v11_p84": float(np.percentile(a, 84.0)),
        "phased_mean": mean_b,
        "phased_median": med_b,
        "phased_p16": float(np.percentile(b, 16.0)),
        "phased_p84": float(np.percentile(b, 84.0)),
        "abs_diff_mean": abs_diff_mean,
        "abs_diff_median": abs_diff_med,
        "rel_change_pct": rel_change_pct,
        "runs_improved": improved,
        "runs_degraded": degraded,
        "runs_tied": tied,
        "p_value_wilcoxon": p_wilcoxon,
    }


def compare_phase_a(v11_dir: Path, phd_dir: Path) -> Dict[str, Any]:
    """Compare Phase A 5-mask standardized validation results per pattern and per run."""
    v11_json = v11_dir / "standardized_validation.json"
    phd_json = phd_dir / "standardized_validation.json"
    if not v11_json.exists() or not phd_json.exists():
        return {"available": False, "reason": f"Missing standardized_validation.json in {v11_dir} or {phd_dir}"}

    v11_data = json.loads(v11_json.read_text())
    phd_data = json.loads(phd_json.read_text())
    v11_per_run = v11_data.get("per_run", {})
    phd_per_run = phd_data.get("per_run", {})

    patterns = [p for p in PHASE_A_PATTERNS if p in v11_per_run and p in phd_per_run]
    metrics = ("overall_mse", "density_nrmse", "jy_nrmse")
    by_pattern: Dict[str, Dict[str, Any]] = {}

    all_runs_metric_v11: Dict[str, Dict[str, List[float]]] = {m: {} for m in metrics}
    all_runs_metric_phd: Dict[str, Dict[str, List[float]]] = {m: {} for m in metrics}

    for pattern in patterns:
        common_runs = sorted(set(v11_per_run[pattern].keys()) & set(phd_per_run[pattern].keys()))
        by_pattern[pattern] = {}
        for m in metrics:
            vals_a = [v11_per_run[pattern][r][m] for r in common_runs]
            vals_b = [phd_per_run[pattern][r][m] for r in common_runs]
            by_pattern[pattern][m] = compute_paired_metric_stats(vals_a, vals_b, higher_is_better=False)
            for r, va, vb in zip(common_runs, vals_a, vals_b):
                all_runs_metric_v11[m].setdefault(r, []).append(float(va))
                all_runs_metric_phd[m].setdefault(r, []).append(float(vb))

    # Aggregate across all 5 patterns per physical run before pairing
    aggregate: Dict[str, Any] = {}
    for m in metrics:
        common_runs = sorted(set(all_runs_metric_v11[m].keys()) & set(all_runs_metric_phd[m].keys()))
        vals_a = [float(np.mean(all_runs_metric_v11[m][r])) for r in common_runs]
        vals_b = [float(np.mean(all_runs_metric_phd[m][r])) for r in common_runs]
        aggregate[m] = compute_paired_metric_stats(vals_a, vals_b, higher_is_better=False)

    return {
        "available": True,
        "n_windows_v11": v11_data.get("n_windows"),
        "n_windows_phased": phd_data.get("n_windows"),
        "n_runs_v11": v11_data.get("n_runs"),
        "n_runs_phased": phd_data.get("n_runs"),
        "by_pattern": by_pattern,
        "aggregate_across_patterns": aggregate,
    }


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    res = stats.spearmanr(x, y)
    return float(res.statistic)


def _extract_raw_calib_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract raw (unscaled) calibration summary regardless of whether --fit-global-scale was used."""
    if "evaluation" in data and isinstance(data["evaluation"], dict):
        ev = data["evaluation"]
        mean_pred = float(ev["mean_predicted"])
        mean_act = float(ev["mean_actual"])
        bias = float(ev["raw_mean_bias"])
        rel_bias = (mean_pred - mean_act) / mean_act if mean_act > 0 else float("nan")
        rank_info = data.get("ranking", {})
        return {
            "scope": "evaluation_runs_raw",
            "mean_predicted": mean_pred,
            "mean_actual": mean_act,
            "u_over_mse": mean_pred / mean_act if mean_act > 0 else float("nan"),
            "bias_pred_minus_actual": bias,
            "relative_mean_bias": rel_bias,
            "increasing_mean_transitions": rank_info.get("increasing_mean_transitions"),
            "total_mean_transitions": rank_info.get("total_mean_transitions"),
            "top_over_bottom_bin_mean_actual": rank_info.get("top_over_bottom_bin_mean_actual"),
            "global_scale_optional": data.get("global_scale"),
        }
    mean_pred = float(data["mean_predicted"])
    mean_act = float(data["mean_actual"])
    return {
        "scope": "all_selected_runs_raw",
        "mean_predicted": mean_pred,
        "mean_actual": mean_act,
        "u_over_mse": mean_pred / mean_act if mean_act > 0 else float("nan"),
        "bias_pred_minus_actual": float(data["mean_bias_predicted_minus_actual"]),
        "relative_mean_bias": data.get("relative_mean_bias"),
        "increasing_mean_transitions": data.get("increasing_mean_transitions"),
        "total_mean_transitions": data.get("total_mean_transitions"),
        "upper_bin_actual_over_predicted": data.get("upper_prediction_bin_actual_over_predicted"),
        "global_scale_optional": None,
    }


def compare_phase_b(v11_dir: Path, phd_dir: Path) -> Dict[str, Any]:
    """Compare Phase B raw density uncertainty calibration and ranking."""
    v11_json = v11_dir / "density_calibration.json"
    phd_json = phd_dir / "density_calibration.json"
    if not v11_json.exists() or not phd_json.exists():
        return {"available": False, "reason": f"Missing density_calibration.json in {v11_dir} or {phd_dir}"}

    v11_data = json.loads(v11_json.read_text())
    phd_data = json.loads(phd_json.read_text())

    v11_raw = _extract_raw_calib_summary(v11_data)
    phd_raw = _extract_raw_calib_summary(phd_data)

    per_run_paired: Dict[str, Any] = {}
    by_pattern: Dict[str, Any] = {}
    by_beta: Dict[str, Any] = {}

    v11_npz = v11_dir / "density_calibration_voxels.npz"
    phd_npz = phd_dir / "density_calibration_voxels.npz"
    if v11_npz.exists() and phd_npz.exists():
        va = np.load(v11_npz, allow_pickle=False)
        pa = np.load(phd_npz, allow_pickle=False)

        v_runs = va["run_name"].astype(str)
        p_runs = pa["run_name"].astype(str)
        common_runs = sorted(set(v_runs.tolist()) & set(p_runs.tolist()))

        def _per_run_calib_metrics(arrs, run_list):
            out = {}
            r_col = arrs["run_name"].astype(str)
            for r in run_list:
                mask = r_col == r
                pred = arrs["predicted_squared_error"][mask].astype(np.float64)
                act = arrs["actual_squared_error"][mask].astype(np.float64)
                mp, ma = float(np.mean(pred)), float(np.mean(act))
                out[r] = {
                    "mean_predicted": mp,
                    "mean_actual": ma,
                    "u_over_mse": mp / ma if ma > 0 else float("nan"),
                    "abs_calib_bias": abs(mp - ma),
                    "spearman_r": _safe_spearman(pred, act),
                }
            return out

        v_run_m = _per_run_calib_metrics(va, common_runs)
        p_run_m = _per_run_calib_metrics(pa, common_runs)
        for metric, higher_better in [
            ("mean_actual", False),
            ("mean_predicted", False),
            ("u_over_mse", False),
            ("abs_calib_bias", False),
            ("spearman_r", True),
        ]:
            per_run_paired[metric] = compute_paired_metric_stats(
                [v_run_m[r][metric] for r in common_runs],
                [p_run_m[r][metric] for r in common_runs],
                higher_is_better=higher_better,
            )

        # Breakdown by pattern (paired across runs)
        v_pat = va["pattern"].astype(str)
        p_pat = pa["pattern"].astype(str)
        for pat in PHASE_A_PATTERNS:
            vp_mask = v_pat == pat
            pp_mask = p_pat == pat
            if not np.any(vp_mask) or not np.any(pp_mask):
                continue
            vals_v_act, vals_p_act = [], []
            vals_v_pred, vals_p_pred = [], []
            vals_v_sp, vals_p_sp = [], []
            for r in common_runs:
                vm = vp_mask & (v_runs == r)
                pm = pp_mask & (p_runs == r)
                if np.any(vm) and np.any(pm):
                    vp, va_err = va["predicted_squared_error"][vm].astype(np.float64), va["actual_squared_error"][vm].astype(np.float64)
                    pp, pa_err = pa["predicted_squared_error"][pm].astype(np.float64), pa["actual_squared_error"][pm].astype(np.float64)
                    vals_v_pred.append(float(np.mean(vp)))
                    vals_v_act.append(float(np.mean(va_err)))
                    vals_v_sp.append(_safe_spearman(vp, va_err))
                    vals_p_pred.append(float(np.mean(pp)))
                    vals_p_act.append(float(np.mean(pa_err)))
                    vals_p_sp.append(_safe_spearman(pp, pa_err))
            by_pattern[pat] = {
                "mean_actual": compute_paired_metric_stats(vals_v_act, vals_p_act, False),
                "mean_predicted": compute_paired_metric_stats(vals_v_pred, vals_p_pred, False),
                "spearman_r": compute_paired_metric_stats(vals_v_sp, vals_p_sp, True),
            }

        # Breakdown by beta (descriptive across runs within each beta)
        betas = sorted(set(va["beta"].tolist()) & set(pa["beta"].tolist()))
        for beta in betas:
            vm = va["beta"] == beta
            pm = pa["beta"] == beta
            vp, va_err = va["predicted_squared_error"][vm].astype(np.float64), va["actual_squared_error"][vm].astype(np.float64)
            pp, pa_err = pa["predicted_squared_error"][pm].astype(np.float64), pa["actual_squared_error"][pm].astype(np.float64)
            v_mp, v_ma = float(np.mean(vp)), float(np.mean(va_err))
            p_mp, p_ma = float(np.mean(pp)), float(np.mean(pa_err))
            by_beta[f"{beta:g}"] = {
                "v11_mean_predicted": v_mp,
                "v11_mean_actual": v_ma,
                "v11_u_over_mse": v_mp / v_ma if v_ma > 0 else float("nan"),
                "v11_spearman_r": _safe_spearman(vp, va_err),
                "phased_mean_predicted": p_mp,
                "phased_mean_actual": p_ma,
                "phased_u_over_mse": p_mp / p_ma if p_ma > 0 else float("nan"),
                "phased_spearman_r": _safe_spearman(pp, pa_err),
            }

    return {
        "available": True,
        "raw_summary_v11": v11_raw,
        "raw_summary_phased": phd_raw,
        "per_run_paired": per_run_paired,
        "by_pattern": by_pattern,
        "by_beta": by_beta,
    }


def compare_phase_c(v11_dir: Path, phd_dir: Path) -> Dict[str, Any]:
    """Compare Phase C F24 historical-conditioning ablation results."""
    v11_csv = v11_dir / "data" / "feedback_results.csv"
    phd_csv = phd_dir / "data" / "feedback_results.csv"
    if not v11_csv.exists() or not phd_csv.exists():
        return {"available": False, "reason": f"Missing data/feedback_results.csv in {v11_dir} or {phd_dir}"}

    df_v11 = pd.read_csv(v11_csv)
    df_phd = pd.read_csv(phd_csv)

    by_condition_strategy: Dict[str, Dict[str, Any]] = {}
    within_model_effects: Dict[str, Dict[str, Any]] = {}

    for b_cond in ["B_full", "B_hidden"]:
        by_condition_strategy[b_cond] = {}
        within_model_effects[b_cond] = {}
        sub_v = df_v11[df_v11["b_condition"] == b_cond]
        sub_p = df_phd[df_phd["b_condition"] == b_cond]

        strategies = [s for s in ["A", "B", "C", "D", "E"] if s in set(sub_v["strategy"]) & set(sub_p["strategy"])]
        for strat in strategies:
            sv = sub_v[sub_v["strategy"] == strat].set_index("run_name")
            sp = sub_p[sub_p["strategy"] == strat].set_index("run_name")
            common_runs = sorted(set(sv.index) & set(sp.index))
            by_condition_strategy[b_cond][strat] = {
                "density_mse_hidden": compute_paired_metric_stats(
                    sv.loc[common_runs, "density_mse_hidden"],
                    sp.loc[common_runs, "density_mse_hidden"],
                    higher_is_better=False,
                ),
                "mean_var_hidden": compute_paired_metric_stats(
                    sv.loc[common_runs, "mean_var_hidden"],
                    sp.loc[common_runs, "mean_var_hidden"],
                    higher_is_better=False,
                ),
                "u_ratio_hidden": compute_paired_metric_stats(
                    sv.loc[common_runs, "u_ratio_hidden"],
                    sp.loc[common_runs, "u_ratio_hidden"],
                    higher_is_better=False,
                ),
                "pearson_r": compute_paired_metric_stats(
                    sv.loc[common_runs, "pearson_r"],
                    sp.loc[common_runs, "pearson_r"],
                    higher_is_better=True,
                ),
                "spearman_r": compute_paired_metric_stats(
                    sv.loc[common_runs, "spearman_r"],
                    sp.loc[common_runs, "spearman_r"],
                    higher_is_better=True,
                ),
            }

        # Within-model historical conditioning effect (B vs A, C vs A)
        for model_name, sub_df in [("v11", sub_v), ("phased", sub_p)]:
            within_model_effects[b_cond][model_name] = {}
            if "A" in set(sub_df["strategy"]):
                sa = sub_df[sub_df["strategy"] == "A"].set_index("run_name")
                for target_s in ["B", "C"]:
                    if target_s in set(sub_df["strategy"]):
                        st = sub_df[sub_df["strategy"] == target_s].set_index("run_name")
                        c_runs = sorted(set(sa.index) & set(st.index))
                        within_model_effects[b_cond][model_name][f"{target_s}_vs_A"] = compute_paired_metric_stats(
                            sa.loc[c_runs, "density_mse_hidden"],
                            st.loc[c_runs, "density_mse_hidden"],
                            higher_is_better=False,
                        )

    return {
        "available": True,
        "by_condition_strategy": by_condition_strategy,
        "within_model_effects": within_model_effects,
    }


def _fmt_p(p: float) -> str:
    if p is None or not math.isfinite(p):
        return "N/A"
    return f"{p:.4f}"


def generate_markdown_report(results: Dict[str, Any], out_path: Path) -> None:
    """Write consolidated `backward_compatibility_summary.md`."""
    lines: List[str] = [
        "# ai4plasma v11 — Phase A/B/C Backward Compatibility Evaluation",
        "",
        f"**Original v11 Checkpoint:** `{results['v11_checkpoint']}`",
        f"**Phase D Checkpoint:** `{results['phased_checkpoint']}`",
        "",
        "**Relative Change Convention:** $\\Delta\\% = \\frac{\\mathrm{Phase\\ D} - \\mathrm{v11}}{|\\mathrm{v11}|} \\times 100\\%$. "
        "For error metrics (MSE, NRMSE), negative values indicate improvement and positive values indicate degradation. "
        "For rank correlations ($r$, $\\rho$), positive values indicate improvement.",
        "",
        "---",
        "",
        "## A. Phase A Reconstruction (Standardized 5-Mask Test Evaluation)",
        "",
    ]

    pa = results.get("phase_a", {})
    if pa.get("available"):
        lines.extend([
            f"Evaluated on {pa.get('n_runs_v11')} held-out test simulations ({pa.get('n_windows_v11')} windows). "
            "Paired statistics treat each physical simulation run as one independent unit.",
            "",
            "| Masking Pattern | Metric | v11 Mean [p16, p84] | Phase D Mean [p16, p84] | Rel. Change | Improved / Degraded | Wilcoxon p |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for pat in list(PHASE_A_PATTERNS) + ["AGGREGATE"]:
            src = pa["aggregate_across_patterns"] if pat == "AGGREGATE" else pa["by_pattern"].get(pat, {})
            for metric in ("overall_mse", "density_nrmse", "jy_nrmse"):
                s = src.get(metric, {})
                if not s or s.get("n_runs", 0) == 0:
                    continue
                lines.append(
                    f"| {pat} | {metric} | "
                    f"{s['v11_mean']:.5f} [{s['v11_p16']:.5f}, {s['v11_p84']:.5f}] | "
                    f"{s['phased_mean']:.5f} [{s['phased_p16']:.5f}, {s['phased_p84']:.5f}] | "
                    f"{s['rel_change_pct']:+.2f}% | "
                    f"{s['runs_improved']}/{s['n_runs']} imp, {s['runs_degraded']}/{s['n_runs']} deg | "
                    f"{_fmt_p(s['p_value_wilcoxon'])} |"
                )
        lines.append("")
    else:
        lines.extend([f"*Not available:* {pa.get('reason', 'not run')}", ""])

    lines.extend([
        "---",
        "",
        "## B. Phase B Uncertainty Calibration (Raw Unscaled Assessment)",
        "",
    ])
    pb = results.get("phase_b", {})
    if pb.get("available"):
        rv = pb["raw_summary_v11"]
        rp = pb["raw_summary_phased"]
        lines.extend([
            "Primary comparison uses **raw predicted squared error without post-hoc rescaling**.",
            "",
            "| Metric (Raw Unscaled) | Original v11 | Phase D |",
            "|---|---:|---:|",
            f"| Mean Actual Squared Error (MSE) | {rv['mean_actual']:.5f} | {rp['mean_actual']:.5f} |",
            f"| Mean Predicted Variance (u) | {rv['mean_predicted']:.5f} | {rp['mean_predicted']:.5f} |",
            f"| Calibration Ratio (u / MSE) | {rv['u_over_mse']:.3f} | {rp['u_over_mse']:.3f} |",
            f"| Mean Bias (u - MSE) | {rv['bias_pred_minus_actual']:+.5f} | {rp['bias_pred_minus_actual']:+.5f} |",
            f"| Monotonic Bin Transitions | {rv['increasing_mean_transitions']}/{rv['total_mean_transitions']} | {rp['increasing_mean_transitions']}/{rp['total_mean_transitions']} |",
            "",
        ])
        if rv.get("global_scale_optional") is not None or rp.get("global_scale_optional") is not None:
            lines.append(
                f"*Secondary Post-Hoc Global Scale (reported separately):* "
                f"v11 scale = `{rv.get('global_scale_optional')}`, Phase D scale = `{rp.get('global_scale_optional')}`."
            )
            lines.append("")

        prp = pb.get("per_run_paired", {})
        if prp:
            lines.extend([
                "### Per-Run Paired Uncertainty Metrics",
                "",
                "| Metric | v11 Mean | Phase D Mean | Rel. Change | Improved / Degraded | Wilcoxon p |",
                "|---|---:|---:|---:|---:|---:|",
            ])
            for m_key in ["mean_actual", "mean_predicted", "u_over_mse", "abs_calib_bias", "spearman_r"]:
                s = prp.get(m_key, {})
                if s.get("n_runs", 0) == 0:
                    continue
                lines.append(
                    f"| {m_key} | {s['v11_mean']:.5f} | {s['phased_mean']:.5f} | "
                    f"{s['rel_change_pct']:+.2f}% | {s['runs_improved']}/{s['n_runs']} imp, {s['runs_degraded']}/{s['n_runs']} deg | "
                    f"{_fmt_p(s['p_value_wilcoxon'])} |"
                )
            lines.append("")

        if pb.get("by_pattern"):
            lines.extend([
                "### Phase B Breakdown by Masking Pattern",
                "",
                "| Pattern | v11 Actual MSE | Phase D Actual MSE (Rel.) | v11 Pred u | Phase D Pred u | v11 Spearman r | Phase D Spearman r |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for pat, pstats in pb["by_pattern"].items():
                sa = pstats["mean_actual"]
                sp = pstats["mean_predicted"]
                sr = pstats["spearman_r"]
                lines.append(
                    f"| {pat} | {sa['v11_mean']:.5f} | {sa['phased_mean']:.5f} ({sa['rel_change_pct']:+.2f}%) | "
                    f"{sp['v11_mean']:.5f} | {sp['phased_mean']:.5f} | "
                    f"{sr['v11_mean']:.3f} | {sr['phased_mean']:.3f} |"
                )
            lines.append("")
    else:
        lines.extend([f"*Not available:* {pb.get('reason', 'not run')}", ""])

    lines.extend([
        "---",
        "",
        "## C. Phase C Historical Conditioning (F24 Single-Target Matched-Noise Ablation)",
        "",
        "> **Note:** This experiment evaluates single-target reconstruction at F24 using `run_feedback_ablation.py` "
        "under matched historical conditioning strategies (A–E). It is distinct from the full F1–F48 online rollout.",
        "",
    ])
    pc = results.get("phase_c", {})
    if pc.get("available"):
        for b_cond in ["B_full", "B_hidden"]:
            b_data = pc["by_condition_strategy"].get(b_cond, {})
            if not b_data:
                continue
            lines.extend([
                f"### {b_cond} Condition (F24 Hidden Density)",
                "",
                "| Strategy | v11 MSE | Phase D MSE | Rel. MSE Change | v11 u/MSE | Phase D u/MSE | v11 Spearman r | Phase D Spearman r | Wilcoxon p (MSE) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for strat, sdict in b_data.items():
                mse_s = sdict["density_mse_hidden"]
                ur_s = sdict["u_ratio_hidden"]
                sp_s = sdict["spearman_r"]
                lbl = PHASE_C_STRATEGY_LABELS.get(strat, strat)
                lines.append(
                    f"| {lbl} | {mse_s['v11_mean']:.5f} | {mse_s['phased_mean']:.5f} | "
                    f"{mse_s['rel_change_pct']:+.2f}% | "
                    f"{ur_s['v11_mean']:.2f} | {ur_s['phased_mean']:.2f} | "
                    f"{sp_s['v11_mean']:.3f} | {sp_s['phased_mean']:.3f} | "
                    f"{_fmt_p(mse_s['p_value_wilcoxon'])} |"
                )
            lines.append("")
    else:
        lines.extend([f"*Not available:* {pc.get('reason', 'not run')}", ""])

    lines.extend([
        "---",
        "",
        "## D. Overall Assessment",
        "",
        "1. **Which original capabilities were preserved?**",
        "   - *See Phase A 5-mask table and Phase B calibration/ranking rows where $|\\Delta\\%|$ is small or Wilcoxon $p > 0.05$.*",
        "2. **Which capabilities degraded?**",
        "   - *See Phase A per-pattern rows, Phase B raw bias / Spearman $r$, and Phase C Strategy A (no-feedback) vs Strategy B (prediction feedback).*",
        "3. **Is the degradation consistent across physical simulations?**",
        "   - *Check the `Improved / Degraded` run counts and Wilcoxon signed-rank p-values above.*",
        "4. **Primary source of degradation (general reconstruction vs. uncertainty calibration vs. historical conditioning)?**",
        "   - *Compare Phase A (5-mask general reconstruction) against Phase C Strategy A (noise history at F24) and Phase B (calibration bias & rank correlation).*",
        "5. **Catastrophic forgetting vs. task specialization:**",
        "   - *If Phase A 5-mask reconstruction degrades substantially across all masks, general-purpose reconstruction suffered forgetting; if Phase A is preserved while Phase C Strategy A degrades and Strategy B improves, the shift reflects specialization to recursive prediction-feedback conditioning.*",
        "",
    ])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Phase A/B/C backward compatibility.")
    parser.add_argument("--v11-checkpoint", type=Path, default=Path("runs/multi_beta_gkl_v11_fromscratch_256ep_20261122/final.pt"))
    parser.add_argument("--phased-checkpoint", type=Path, default=Path("runs/multi_beta_gkl_v11_phase_d_pilot_6ep_20261123/final_audited.pt"))
    parser.add_argument("--phase-a-v11", type=Path, required=True)
    parser.add_argument("--phase-a-phased", type=Path, required=True)
    parser.add_argument("--phase-b-v11", type=Path, required=True)
    parser.add_argument("--phase-b-phased", type=Path, required=True)
    parser.add_argument("--phase-c-v11", type=Path, required=True)
    parser.add_argument("--phase-c-phased", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluations/phase_d_backward_compatibility_20261124"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "v11_checkpoint": str(args.v11_checkpoint),
        "phased_checkpoint": str(args.phased_checkpoint),
        "phase_a": compare_phase_a(args.phase_a_v11, args.phase_a_phased),
        "phase_b": compare_phase_b(args.phase_b_v11, args.phase_b_phased),
        "phase_c": compare_phase_c(args.phase_c_v11, args.phase_c_phased),
    }

    stats_path = args.output_dir / "paired_backward_compatibility_stats.json"
    stats_path.write_text(json.dumps(results, indent=2))
    report_path = args.output_dir / "backward_compatibility_summary.md"
    generate_markdown_report(results, report_path)
    print(f"Wrote {stats_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
