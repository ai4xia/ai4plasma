#!/usr/bin/env python3
"""Aggregate extreme horizon and standard mask calibration metrics from per-run CSV."""

import csv
import json
import math
from pathlib import Path
import numpy as np

OUTPUT = Path("/pscratch/sd/b/binxia/ai4plasma/evaluations/matched_gkl_masking_continuation_20ep_20261121")
PER_RUN_CSV = OUTPUT / "extreme_horizon_per_run_audit.csv"
SEED = 1234


def run_paired_bootstrap(diff_array: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(diff_array)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.mean(diff_array[idx], axis=1)
    low, high = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(diff_array)), float(low), float(high)


def run_ratio_of_means_bootstrap(
    e2_v9: np.ndarray, u_v9: np.ndarray, e2_v11: np.ndarray, u_v11: np.ndarray,
    n_boot: int = 10000, seed: int = 42
):
    rng = np.random.default_rng(seed)
    n = len(e2_v9)
    idx = rng.integers(0, n, size=(n_boot, n))
    
    boot_diffs = []
    for b in range(n_boot):
        sample_idx = idx[b]
        r_v9 = np.sum(u_v9[sample_idx]) / max(np.sum(e2_v9[sample_idx]), 1e-12)
        r_v11 = np.sum(u_v11[sample_idx]) / max(np.sum(e2_v11[sample_idx]), 1e-12)
        boot_diffs.append(r_v11 - r_v9)
    
    diff_arr = np.array(boot_diffs)
    actual_r_v9 = np.sum(u_v9) / max(np.sum(e2_v9), 1e-12)
    actual_r_v11 = np.sum(u_v11) / max(np.sum(e2_v11), 1e-12)
    actual_diff = actual_r_v11 - actual_r_v9
    low, high = np.percentile(diff_arr, [2.5, 97.5])
    return float(actual_diff), float(low), float(high)


def main():
    with open(PER_RUN_CSV) as f:
        per_run_rows = list(csv.DictReader(f))

    summary_results = []
    
    categories = [("forecast", ["h20", "h21", "h22", "h23"]),
                  ("standard", ["spatial_random", "spatial_grid", "spatial_block", "temporal_random", "temporal_block"])]

    for cond_type, cond_list in categories:
        for cond_val in cond_list:
            for ch in ["Magnetic", "Density", "All"]:
                rows = [r for r in per_run_rows if r["category"] == cond_type and r["condition"] == cond_val and r["channel"] == ch]
                assert len(rows) == 125, f"Expected 125 rows, got {len(rows)}"

                v9_mse_arr = np.array([float(r["v9_mse"]) for r in rows])
                v11_mse_arr = np.array([float(r["v11_mse"]) for r in rows])
                diff_mse_arr = v11_mse_arr - v9_mse_arr
                
                v9_u_arr = np.array([float(r["v9_u"]) for r in rows])
                v11_u_arr = np.array([float(r["v11_u"]) for r in rows])
                diff_u_arr = v11_u_arr - v9_u_arr

                v9_cal_arr = np.array([float(r["v9_cal_ratio"]) for r in rows])
                v11_cal_arr = np.array([float(r["v11_cal_ratio"]) for r in rows])
                diff_cal_arr = v11_cal_arr - v9_cal_arr

                sum_e2_v9_arr = np.array([float(r["sum_e2_v9"]) for r in rows])
                sum_u_v9_arr = np.array([float(r["sum_u_v9"]) for r in rows])
                sum_e2_v11_arr = np.array([float(r["sum_e2_v11"]) for r in rows])
                sum_u_v11_arr = np.array([float(r["sum_u_v11"]) for r in rows])
                n_voxels_arr = np.array([float(r["n_voxels"]) for r in rows])

                # 1. Ratio of aggregated means
                agg_mse_v9 = np.sum(sum_e2_v9_arr) / np.sum(n_voxels_arr)
                agg_u_v9 = np.sum(sum_u_v9_arr) / np.sum(n_voxels_arr)
                agg_cal_v9 = agg_u_v9 / agg_mse_v9 if agg_mse_v9 > 0 else float("nan")

                agg_mse_v11 = np.sum(sum_e2_v11_arr) / np.sum(n_voxels_arr)
                agg_u_v11 = np.sum(sum_u_v11_arr) / np.sum(n_voxels_arr)
                agg_cal_v11 = agg_u_v11 / agg_mse_v11 if agg_mse_v11 > 0 else float("nan")

                diff_agg_cal, ci_agg_cal_low, ci_agg_cal_high = run_ratio_of_means_bootstrap(
                    sum_e2_v9_arr, sum_u_v9_arr, sum_e2_v11_arr, sum_u_v11_arr, seed=SEED + 1
                )

                # 2. Mean & median of per-run ratios
                mean_run_cal_v9 = float(np.mean(v9_cal_arr))
                mean_run_cal_v11 = float(np.mean(v11_cal_arr))
                median_run_cal_v9 = float(np.median(v9_cal_arr))
                median_run_cal_v11 = float(np.median(v11_cal_arr))

                diff_mean_cal, ci_mean_cal_low, ci_mean_cal_high = run_paired_bootstrap(diff_cal_arr, seed=SEED + 2)

                # Bootstrap CI for MSE and u
                diff_mse_mean, ci_mse_low, ci_mse_high = run_paired_bootstrap(diff_mse_arr, seed=SEED + 3)
                diff_u_mean, ci_u_low, ci_u_high = run_paired_bootstrap(diff_u_arr, seed=SEED + 4)

                pct_change_mse = (diff_mse_mean / np.mean(v9_mse_arr)) * 100.0 if np.mean(v9_mse_arr) > 0 else 0.0
                pct_change_u = (diff_u_mean / np.mean(v9_u_arr)) * 100.0 if np.mean(v9_u_arr) > 0 else 0.0

                # Spearman
                if rows[0]["v9_spearman"] != "":
                    v9_sp_arr = np.array([float(r["v9_spearman"]) for r in rows])
                    v11_sp_arr = np.array([float(r["v11_spearman"]) for r in rows])
                    diff_sp_arr = v11_sp_arr - v9_sp_arr
                    mean_sp_v9 = float(np.mean(v9_sp_arr))
                    mean_sp_v11 = float(np.mean(v11_sp_arr))
                    diff_sp_mean, ci_sp_low, ci_sp_high = run_paired_bootstrap(diff_sp_arr, seed=SEED + 5)
                else:
                    mean_sp_v9, mean_sp_v11, diff_sp_mean, ci_sp_low, ci_sp_high = [float("nan")] * 5

                summary_results.append({
                    "category": cond_type,
                    "condition": cond_val,
                    "channel": ch,
                    "n_runs": len(rows),
                    # MSE
                    "v9_mse": float(np.mean(v9_mse_arr)),
                    "v11_mse": float(np.mean(v11_mse_arr)),
                    "diff_mse": diff_mse_mean,
                    "ci_mse_low": ci_mse_low,
                    "ci_mse_high": ci_mse_high,
                    "pct_change_mse": pct_change_mse,
                    "sig_mse": (ci_mse_low > 0 or ci_mse_high < 0),
                    # Uncertainty u
                    "v9_u": float(np.mean(v9_u_arr)),
                    "v11_u": float(np.mean(v11_u_arr)),
                    "diff_u": diff_u_mean,
                    "ci_u_low": ci_u_low,
                    "ci_u_high": ci_u_high,
                    "pct_change_u": pct_change_u,
                    "sig_u": (ci_u_low > 0 or ci_u_high < 0),
                    # Aggregated calibration ratio
                    "v9_agg_cal_ratio": agg_cal_v9,
                    "v11_agg_cal_ratio": agg_cal_v11,
                    "diff_agg_cal_ratio": diff_agg_cal,
                    "ci_agg_cal_low": ci_agg_cal_low,
                    "ci_agg_cal_high": ci_agg_cal_high,
                    "sig_agg_cal": (ci_agg_cal_low > 0 or ci_agg_cal_high < 0),
                    # Per-run calibration ratio
                    "v9_mean_run_cal_ratio": mean_run_cal_v9,
                    "v11_mean_run_cal_ratio": mean_run_cal_v11,
                    "v9_median_run_cal_ratio": median_run_cal_v9,
                    "v11_median_run_cal_ratio": median_run_cal_v11,
                    "diff_mean_run_cal_ratio": diff_mean_cal,
                    "ci_mean_run_cal_low": ci_mean_cal_low,
                    "ci_mean_run_cal_high": ci_mean_cal_high,
                    "sig_mean_run_cal": (ci_mean_cal_low > 0 or ci_mean_cal_high < 0),
                    # Spearman
                    "v9_spearman": mean_sp_v9,
                    "v11_spearman": mean_sp_v11,
                    "diff_spearman": diff_sp_mean,
                    "ci_spearman_low": ci_sp_low,
                    "ci_spearman_high": ci_sp_high,
                    "sig_spearman": (ci_sp_low > 0 or ci_sp_high < 0) if not math.isnan(ci_sp_low) else False,
                })

    summary_json_path = OUTPUT / "extreme_horizon_calibration_audit_summary.json"
    summary_json_path.write_text(json.dumps(summary_results, indent=2))
    print(f"Wrote summary JSON to {summary_json_path.name}")

    summary_csv_path = OUTPUT / "extreme_horizon_calibration_audit_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_results[0].keys()))
        writer.writeheader()
        writer.writerows(summary_results)
    print(f"Wrote summary CSV to {summary_csv_path.name}")


if __name__ == "__main__":
    main()
