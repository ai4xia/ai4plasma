#!/usr/bin/env python3
"""Audit Extreme-Horizon Uncertainty Calibration: GKL v9 vs. v11.

Evaluates test windows for:
  - Horizons h in [20, 21, 22, 23]
  - Standard masks: spatial_random, spatial_grid, spatial_block, temporal_random, temporal_block
Computes:
  1. Ratio of aggregated means: sum(u) / sum(MSE)
  2. Mean of per-run ratios: mean_i(u_i / MSE_i)
  3. Paired run-level bootstrap 95% CIs (10,000 resamples) for MSE, u, both calibration ratios, and Spearman rho
  4. Modality breakouts: Magnetic, Density, All
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import rankdata, spearmanr
import torch

PROJECT = Path("/pscratch/sd/b/binxia/ai4plasma")
sys.path.insert(0, str(PROJECT))

from data.masking import (
    MASK_PATTERNS,
    make_fixed_validation_mask,
    make_uncertain_input,
)
from data.vpic_hdf5_dataset import (
    VPICWindowDataset,
    load_fixed_split,
)
from models.unet3d import UNet3D
from visualize_mask_patterns_unet3d import select_validation_statistics_indices

OUTPUT = PROJECT / "evaluations/matched_gkl_masking_continuation_20ep_20261121"
V9_CKPT = PROJECT / "runs/multi_beta_gkl_v9_control_20ep_20261121/final.pt"
V11_CKPT = PROJECT / "runs/multi_beta_gkl_v11_shared_temporal_20ep_20261121/final.pt"

SEED = 1234
AUDIT_HORIZONS = [20, 21, 22, 23]
GROUPS = {
    "Bx": (0,),
    "By": (1,),
    "Bz": (2,),
    "Density": (3,),
    "Magnetic": (0, 1, 2),
    "All": (0, 1, 2, 3),
}


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=int(args.get("base_channels", 24)),
        channel_mults=args.get("channel_mults", [1, 2, 4, 8]),
        architecture=args["model_version"],
        uncertainty_aware=bool(args["uncertainty_aware"]),
        use_attention=bool(args.get("use_attention", False)),
        spatial_only_pooling=bool(args.get("spatial_only_pooling", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    stats_payload = ckpt["stats"]
    mean = torch.tensor(stats_payload["mean"], device=device).view(1, 4, 1, 1, 1)
    std = torch.tensor(stats_payload["std"], device=device).view(1, 4, 1, 1, 1)
    return model, mean, std


@torch.no_grad()
def infer(model, model_input):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred, var = model(model_input, detach_variance_features=True)
    return pred.float(), var.float()


def batches(dataset, indices: list[int], batch_size: int = 4):
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        samples = [dataset[index] for index in chosen]
        yield chosen, torch.stack([s["block"] for s in samples]), [s["metadata"] for s in samples]


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
    print("=" * 80)
    print("Extreme-Horizon Uncertainty Calibration Audit: GKL v9 vs v11")
    print("=" * 80, flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

    cfg = json.loads((PROJECT / "runs/multi_beta_gkl_v11_shared_temporal_20ep_20261121/config.json").read_text())
    split = load_fixed_split()
    dataset = VPICWindowDataset(
        h5_dir=cfg["h5_dir"],
        betas=cfg["betas"],
        delta_t=24,
        stride_t=1,
        layout="C T X Z",
        return_metadata=True,
    )
    test_indices = select_validation_statistics_indices(dataset, set(split["test_runs"]), 24, 1)
    n_runs = len(test_indices)
    assert n_runs == 125, f"Expected 125 test windows, got {n_runs}"

    # Verify run mapping
    run_names = [dataset[idx]["metadata"]["run_name"] for idx in test_indices]
    unique_runs = set(run_names)
    print(f"Loaded {n_runs} test windows corresponding to {len(unique_runs)} distinct simulation runs.")
    assert len(unique_runs) == n_runs, "Windows are not 1-to-1 with simulation runs!"

    models = {}
    for name, path in [("v9_control", V9_CKPT), ("v11_experiment", V11_CKPT)]:
        m, mn, st = load_model(path, device)
        models[name] = {"model": m, "mean": mn, "std": st}
        print(f"Loaded model {name} from {path.name}", flush=True)

    # Storage: (model, cond_type, cond_val, ch, run_idx) -> metrics
    # cond_type: "forecast" (cond_val = h) or "standard" (cond_val = task)
    per_run_data = defaultdict(dict)

    # 1. Forward Forecasting Horizons h in [20, 21, 22, 23]
    print("\n--- Evaluating Forecast Horizons h in [20, 21, 22, 23] ---", flush=True)
    for batch_number, (chosen, block, metas) in enumerate(batches(dataset, test_indices, batch_size=4)):
        noise_gen = torch.Generator(device=device).manual_seed(SEED + 200000 + batch_number)
        
        for model_name in ["v9_control", "v11_experiment"]:
            m_dict = models[model_name]
            target = ((block.to(device) - m_dict["mean"]) / (m_dict["std"] + 1e-8)).float()
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

            for h in AUDIT_HORIZONS:
                prefix = 24 - h
                mask = torch.ones_like(target)
                mask[:, :, prefix:] = 0.0
                input_var = 1.0 - mask
                model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)

                pred, var = infer(m_dict["model"], model_input)
                res = pred - target

                res_np = res.cpu().numpy()
                var_np = var.cpu().numpy()

                for ch, ci in [("Magnetic", [0, 1, 2]), ("Density", [3]), ("All", [0, 1, 2, 3])]:
                    for b_idx, d_idx in enumerate(chosen):
                        run_name = metas[b_idx]["run_name"]
                        e_vol = np.square(res_np[b_idx, ci, prefix:], dtype=np.float64)
                        u_vol = var_np[b_idx, ci, prefix:].astype(np.float64)
                        
                        run_mse = float(np.mean(e_vol))
                        run_u = float(np.mean(u_vol))
                        cal_ratio = run_u / run_mse if run_mse > 0 else float("nan")

                        # Frame-level spatial Spearman rho
                        e_frames = e_vol.transpose(1, 0, 2, 3).reshape(h, -1)
                        u_frames = u_vol.transpose(1, 0, 2, 3).reshape(h, -1)
                        
                        er = rankdata(e_frames, axis=1)
                        ur = rankdata(u_frames, axis=1)
                        er -= er.mean(1, keepdims=True)
                        ur -= ur.mean(1, keepdims=True)
                        denom = np.sqrt((er * er).sum(1) * (ur * ur).sum(1))
                        frame_rhos = np.where(denom > 0, (er * ur).sum(1) / denom, 0.0)
                        mean_spearman = float(np.mean(frame_rhos))

                        per_run_data[(model_name, "forecast", f"h{h}", ch, run_name)] = {
                            "run_name": run_name,
                            "dataset_index": d_idx,
                            "mse": run_mse,
                            "u": run_u,
                            "cal_ratio": cal_ratio,
                            "spearman": mean_spearman,
                            "sum_e2": float(np.sum(e_vol)),
                            "sum_u": float(np.sum(u_vol)),
                            "n_voxels": int(e_vol.size),
                        }

    # 2. Standard Masks
    print("\n--- Evaluating Standard Masks (Test Windows) ---", flush=True)
    for batch_number, (chosen, block, metas) in enumerate(batches(dataset, test_indices, batch_size=4)):
        noise_gen = torch.Generator(device=device).manual_seed(SEED + 100000 + batch_number)

        for model_name in ["v9_control", "v11_experiment"]:
            m_dict = models[model_name]
            target = ((block.to(device) - m_dict["mean"]) / (m_dict["std"] + 1e-8)).float()
            noise = torch.randn(target.shape, generator=noise_gen, device=device, dtype=target.dtype)

            for task in MASK_PATTERNS:
                mask, _ = make_fixed_validation_mask(task, target.shape, device=device, dtype=target.dtype)
                input_var = 1.0 - mask
                model_input = torch.cat([make_uncertain_input(target, input_var, noise), input_var], dim=1)

                pred, var = infer(m_dict["model"], model_input)
                res = pred - target

                res_np = res.cpu().numpy()
                var_np = var.cpu().numpy()
                unknown_np = (mask < 0.5).cpu().numpy()

                for ch, ci in [("Magnetic", [0, 1, 2]), ("Density", [3]), ("All", [0, 1, 2, 3])]:
                    for b_idx, d_idx in enumerate(chosen):
                        run_name = metas[b_idx]["run_name"]
                        u_b = unknown_np[b_idx, ci]
                        r_b = res_np[b_idx, ci][u_b]
                        v_b = var_np[b_idx, ci][u_b]

                        if r_b.size:
                            e2_vals = np.square(r_b, dtype=np.float64)
                            u_vals = v_b.astype(np.float64)
                            run_mse = float(np.mean(e2_vals))
                            run_u = float(np.mean(u_vals))
                            cal_ratio = run_u / run_mse if run_mse > 0 else float("nan")
                            sum_e2 = float(np.sum(e2_vals))
                            sum_u = float(np.sum(u_vals))
                            n_vox = int(r_b.size)
                        else:
                            run_mse, run_u, cal_ratio, sum_e2, sum_u, n_vox = 0.0, 0.0, float("nan"), 0.0, 0.0, 0

                        per_run_data[(model_name, "standard", task, ch, run_name)] = {
                            "run_name": run_name,
                            "dataset_index": d_idx,
                            "mse": run_mse,
                            "u": run_u,
                            "cal_ratio": cal_ratio,
                            "sum_e2": sum_e2,
                            "sum_u": sum_u,
                            "n_voxels": n_vox,
                        }

    # 3. Export Per-Run CSV
    print("\n--- Writing Per-Run Audit CSV ---", flush=True)
    per_run_rows = []
    for cond_type, cond_list in [("forecast", [f"h{h}" for h in AUDIT_HORIZONS]), ("standard", list(MASK_PATTERNS))]:
        for cond_val in cond_list:
            for ch in ["Magnetic", "Density", "All"]:
                for run_name in run_names:
                    d_v9 = per_run_data[("v9_control", cond_type, cond_val, ch, run_name)]
                    d_v11 = per_run_data[("v11_experiment", cond_type, cond_val, ch, run_name)]
                    per_run_rows.append({
                        "category": cond_type,
                        "condition": cond_val,
                        "channel": ch,
                        "run_name": run_name,
                        "v9_mse": d_v9["mse"],
                        "v11_mse": d_v11["mse"],
                        "diff_mse": d_v11["mse"] - d_v9["mse"],
                        "v9_u": d_v9["u"],
                        "v11_u": d_v11["u"],
                        "diff_u": d_v11["u"] - d_v9["u"],
                        "v9_cal_ratio": d_v9["cal_ratio"],
                        "v11_cal_ratio": d_v11["cal_ratio"],
                        "diff_cal_ratio": d_v11["cal_ratio"] - d_v9["cal_ratio"],
                        "v9_spearman": d_v9.get("spearman", ""),
                        "v11_spearman": d_v11.get("spearman", ""),
                        "diff_spearman": (d_v11["spearman"] - d_v9["spearman"]) if "spearman" in d_v9 else "",
                        "sum_e2_v9": d_v9["sum_e2"],
                        "sum_u_v9": d_v9["sum_u"],
                        "sum_e2_v11": d_v11["sum_e2"],
                        "sum_u_v11": d_v11["sum_u"],
                        "n_voxels": d_v9["n_voxels"],
                    })

    csv_path = OUTPUT / "extreme_horizon_per_run_audit.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_run_rows)
    print(f"Wrote {len(per_run_rows)} rows to {csv_path.name}")

    # 4. Statistical Aggregation & Bootstrap CIs
    print("\n--- Computing Aggregated Statistics & Paired 95% Bootstrap CIs ---", flush=True)
    summary_results = []

    for cond_type, cond_list in [("forecast", [f"h{h}" for h in AUDIT_HORIZONS]), ("standard", list(MASK_PATTERNS))]:
        for cond_val in cond_list:
            for ch in ["Magnetic", "Density", "All"]:
                rows = [r for r in per_run_rows if r["category"] == cond_type and r["condition"] == cond_val and r["channel"] == ch]
                
                v9_mse_arr = np.array([r["v9_mse"] for r in rows])
                v11_mse_arr = np.array([r["v11_mse"] for r in rows])
                diff_mse_arr = v11_mse_arr - v9_mse_arr
                
                v9_u_arr = np.array([r["v9_u"] for r in rows])
                v11_u_arr = np.array([r["v11_u"] for r in rows])
                diff_u_arr = v11_u_arr - v9_u_arr

                v9_cal_arr = np.array([r["v9_cal_ratio"] for r in rows])
                v11_cal_arr = np.array([r["v11_cal_ratio"] for r in rows])
                diff_cal_arr = v11_cal_arr - v9_cal_arr

                sum_e2_v9_arr = np.array([r["sum_e2_v9"] for r in rows])
                sum_u_v9_arr = np.array([r["sum_u_v9"] for r in rows])
                sum_e2_v11_arr = np.array([r["sum_e2_v11"] for r in rows])
                sum_u_v11_arr = np.array([r["sum_u_v11"] for r in rows])

                # 1. Ratio of aggregated means
                agg_mse_v9 = np.sum(sum_e2_v9_arr) / np.sum([r["n_voxels"] for r in rows])
                agg_u_v9 = np.sum(sum_u_v9_arr) / np.sum([r["n_voxels"] for r in rows])
                agg_cal_v9 = agg_u_v9 / agg_mse_v9 if agg_mse_v9 > 0 else float("nan")

                agg_mse_v11 = np.sum(sum_e2_v11_arr) / np.sum([r["n_voxels"] for r in rows])
                agg_u_v11 = np.sum(sum_u_v11_arr) / np.sum([r["n_voxels"] for r in rows])
                agg_cal_v11 = agg_u_v11 / agg_mse_v11 if agg_mse_v11 > 0 else float("nan")

                diff_agg_cal, ci_agg_cal_low, ci_agg_cal_high = run_ratio_of_means_bootstrap(
                    sum_e2_v9_arr, sum_u_v9_arr, sum_e2_v11_arr, sum_u_v11_arr, seed=SEED + 1
                )

                # 2. Mean of per-run ratios
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
    print(f"Wrote audit summary JSON to {summary_json_path.name}")

    summary_csv_path = OUTPUT / "extreme_horizon_calibration_audit_summary.csv"
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_results[0].keys()))
        writer.writeheader()
        writer.writerows(summary_results)
    print(f"Wrote audit summary CSV to {summary_csv_path.name}")

    print("\n=== Calibration Audit Finished Successfully ===", flush=True)


if __name__ == "__main__":
    main()
