# evaluate_standardized_validation.py
"""Post-hoc test-set evaluation on the five standardized ~50% masks."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from data.masking import (
    FIXED_VALIDATION_PATTERNS,
    FIXED_VALIDATION_SPATIAL_RANDOM_SEED,
    MASKING_VERSION,
    full_mse_loss,
    make_fixed_validation_mask,
)
from data.vpic_hdf5_dataset import (
    FIXED_SPLIT_PATH,
    VPICWindowDataset,
    fixed_split_metadata,
    load_fixed_split,
    validate_final_checkpoint,
)
from models.unet3d import LEGACY_MODEL_VERSION, UNet3D
from visualize_mask_patterns_unet3d import (
    DEFAULT_RUN_DIR,
    JY_NRMSE_DEFINITION,
    JY_STATS_DEFINITION,
    checkpoint_cache_signature,
    compute_jy,
    denormalize,
    expand_path,
    get_train_runs,
    get_test_runs,
    jy_metric_metadata,
    load_checkpoint,
    load_or_compute_jy_training_stats,
    normalize,
    predict_masked_fields,
    resolve_checkpoint_path,
)


CACHE_VERSION = 3
OUTPUT_DIRNAME = "paper_figures_v1/standardized_validation"
CACHE_STEM = "standardized_validation"
PATTERN_LABELS = {
    "spatial_random": "Spatial random",
    "spatial_grid": "Spatial grid",
    "spatial_block": "Spatial block",
    "temporal_random": "Temporal random",
    "temporal_block": "Temporal block",
}
OVERALL_MSE_DEFINITION = (
    "mean((prediction_normalized - target_normalized)^2) over all four channels "
    "and all T,X,Z locations"
)
DENSITY_NRMSE_DEFINITION = (
    "sqrt(mean((prediction_normalized - target_normalized)^2)) on the Density channel"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate final.pt on the fixed test runs with five standardized 50% masks "
            "and write a paper numerical table, or a bounded density calibration plot."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=[-21.0, 21.0, -50.0, 50.0],
    )
    parser.add_argument("--h5-dir", default=None)
    parser.add_argument("--density-calibration", action="store_true",
                        help="Run bounded density calibration instead of the standardized paper table.")
    parser.add_argument("--calibration-runs-per-beta", type=int, default=2)
    parser.add_argument("--calibration-windows-per-run", type=int, default=1)
    parser.add_argument("--calibration-voxels-per-pattern", type=int, default=2048)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--calibration-seed", type=int, default=1234)
    parser.add_argument("--fit-global-scale", action="store_true",
                        help="Fit one scale on half the selected runs per beta; assess the other half.")
    args = parser.parse_args()
    if args.calibration_runs_per_beta < 2 or min(
        args.calibration_windows_per_run, args.calibration_voxels_per_pattern,
        args.calibration_bins,
    ) < 1:
        parser.error("Calibration needs at least two runs per beta and positive window/voxel/bin limits.")
    if args.fit_global_scale and args.calibration_runs_per_beta % 2:
        parser.error("Global scaling requires an even number of selected runs per beta.")
    return args


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def build_standardized_masks(
    shape: Sequence[int],
    device=None,
    dtype=torch.float32,
) -> Tuple[List[torch.Tensor], List[Dict]]:
    """Call make_fixed_validation_mask for each standardized pattern, in order."""
    masks = []
    infos = []
    for pattern in FIXED_VALIDATION_PATTERNS:
        mask, info = make_fixed_validation_mask(
            pattern,
            shape,
            device=device,
            dtype=dtype,
        )
        masks.append(mask)
        infos.append(info)
    return masks, infos


def window_overall_mse(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return float(np.mean(np.square(residual)))


def window_density_nrmse(prediction: np.ndarray, target: np.ndarray) -> float:
    residual = np.asarray(prediction[3], dtype=np.float64) - np.asarray(
        target[3], dtype=np.float64
    )
    return float(np.sqrt(np.mean(np.square(residual))))


def window_jy_nrmse(
    prediction: np.ndarray,
    target: np.ndarray,
    extent: Sequence[float],
    jy_std_train: float,
) -> float:
    """Jy NRMSE on already-physical (HDF5-stored) Bx/Bz fields."""
    scale = float(jy_std_train)
    if scale <= 0.0:
        raise ValueError(f"jy_std_train must be positive, got {scale}.")
    residual = compute_jy(prediction, extent) - compute_jy(target, extent)
    return float(np.sqrt(np.mean(np.square(residual))) / scale)


def per_run_means(records: Sequence[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Mean over windows inside each physical run, for each pattern."""
    grouped: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    metrics = ("overall_mse", "density_nrmse", "jy_nrmse")
    for record in records:
        pattern = record["pattern"]
        run_name = record["run_name"]
        by_run = grouped.setdefault(pattern, {}).setdefault(
            run_name,
            {name: [] for name in metrics},
        )
        for name in metrics:
            by_run[name].append(float(record[name]))
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pattern, runs in grouped.items():
        aggregated[pattern] = {}
        for run_name, values in runs.items():
            aggregated[pattern][run_name] = {
                name: float(np.mean(values[name])) for name in metrics
            }
            aggregated[pattern][run_name]["n_windows"] = float(len(values["overall_mse"]))
    return aggregated


def cross_run_stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty run list.")
    return {
        "median": float(np.median(array)),
        "p16": float(np.percentile(array, 16.0)),
        "p84": float(np.percentile(array, 84.0)),
        "n_runs": int(array.size),
    }


def format_median_interval(stats: Dict[str, float]) -> str:
    return f"{stats['median']:.4f} [{stats['p16']:.4f}, {stats['p84']:.4f}]"


def split_identity(split: Dict) -> Dict:
    return {
        "data_role": "test",
        "split_path": str(FIXED_SPLIT_PATH),
        "split_sha256": fixed_split_metadata()["sha256"],
        "test_runs": sorted(split.get("test_runs", [])),
        "train_runs": sorted(split.get("train_runs", [])),
        "n_test_runs": len(split.get("test_runs", [])),
        "n_train_runs": len(split.get("train_runs", [])),
    }


def cache_signature(
    checkpoint_path: Path,
    checkpoint_epoch: int,
    split: Dict,
    mean: Sequence[float],
    std: Sequence[float],
    jy_stats: Dict,
    extent: Sequence[float],
) -> Dict:
    jy_path = checkpoint_path.parent / "jy_stats.json"
    jy_file = None
    if jy_path.exists():
        stat = jy_path.stat()
        jy_file = {
            "path": str(jy_path),
            "mtime_ns": int(stat.st_mtime_ns),
            "size": int(stat.st_size),
        }
    return {
        "cache_version": CACHE_VERSION,
        "checkpoint": checkpoint_cache_signature(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_epoch),
        "split": split_identity(split),
        "normalization": {
            "mean": [float(value) for value in mean],
            "std": [float(value) for value in std],
        },
        "patterns": list(FIXED_VALIDATION_PATTERNS),
        "masking_version": MASKING_VERSION,
        "fixed_validation_spatial_random_seed": int(FIXED_VALIDATION_SPATIAL_RANDOM_SEED),
        "shared_channel_mask": True,
        "jy_stats": {
            "jy_definition": jy_stats.get("jy_definition", JY_STATS_DEFINITION),
            "jy_preprocessing": jy_stats.get("jy_preprocessing")
            or jy_stats.get("preprocessing"),
            "jy_stats_cache_version": jy_stats.get("jy_stats_cache_version"),
            "jy_includes_mu0": jy_stats.get("jy_includes_mu0"),
            "B_unit": jy_stats.get("B_unit", jy_stats.get("jy_b_units")),
            "source_coordinate_unit": jy_stats.get("source_coordinate_unit"),
            "derivative_coordinate_unit": jy_stats.get("derivative_coordinate_unit"),
            "jy_unit": jy_stats.get("jy_unit", jy_stats.get("jy_physical_units")),
            "jy_b_units": jy_stats.get("jy_b_units"),
            "jy_coordinate_units": jy_stats.get("jy_coordinate_units"),
            "jy_physical_units": jy_stats.get("jy_physical_units"),
            "jy_std_train": float(jy_stats["jy_std_train"]),
            "jy_mean_train": float(jy_stats["jy_mean_train"]),
            "n_runs": jy_stats.get("n_runs"),
            "count": jy_stats.get("count"),
            "file": jy_file,
        },
        "extent": [float(value) for value in extent],
    }


def load_cached_payload(out_dir: Path, signature: Dict) -> Dict | None:
    json_path = out_dir / f"{CACHE_STEM}.json"
    npz_path = out_dir / f"{CACHE_STEM}.npz"
    if not json_path.exists() or not npz_path.exists():
        return None
    payload = json.loads(json_path.read_text())
    if payload.get("signature") != signature:
        print(f"Cache signature mismatch at {json_path}; will recompute.")
        return None
    print(f"Reusing standardized test cache: {json_path}")
    return payload


def write_csv(records: Sequence[Dict], path: Path) -> None:
    fieldnames = [
        "run_name",
        "t0",
        "pattern",
        "overall_mse",
        "density_nrmse",
        "jy_nrmse",
        "validate_style_mse",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in fieldnames})


def write_markdown_table(summary: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    lines = [
        "| Task | Overall MSE | Density NRMSE | Jy NRMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for pattern in FIXED_VALIDATION_PATTERNS:
        stats = summary[pattern]
        lines.append(
            "| {label} | {overall} | {density} | {jy} |".format(
                label=PATTERN_LABELS[pattern],
                overall=format_median_interval(stats["overall_mse"]),
                density=format_median_interval(stats["density_nrmse"]),
                jy=format_median_interval(stats["jy_nrmse"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def write_tex_table(summary: Dict[str, Dict[str, Dict[str, float]]], path: Path) -> None:
    rows = [
        r"\begin{tabular}{lrrr}",
        r"\hline",
        r"Task & Overall MSE & Density NRMSE & Jy NRMSE \\",
        r"\hline",
    ]
    for pattern in FIXED_VALIDATION_PATTERNS:
        stats = summary[pattern]
        rows.append(
            r"{label} & {overall} & {density} & {jy} \\".format(
                label=PATTERN_LABELS[pattern],
                overall=format_median_interval(stats["overall_mse"]),
                density=format_median_interval(stats["density_nrmse"]),
                jy=format_median_interval(stats["jy_nrmse"]),
            )
        )
    rows.extend([r"\hline", r"\end{tabular}", ""])
    path.write_text("\n".join(rows))


def summarize_from_records(records: Sequence[Dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    per_run = per_run_means(records)
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for pattern in FIXED_VALIDATION_PATTERNS:
        run_values = per_run[pattern]
        summary[pattern] = {
            metric: cross_run_stats(
                [run_values[run_name][metric] for run_name in sorted(run_values)]
            )
            for metric in ("overall_mse", "density_nrmse", "jy_nrmse")
        }
    return summary


def global_window_weighted_overall_mse(records: Sequence[Dict]) -> Dict[str, float]:
    """Equal weight per validation window, matching validate() averaging."""
    by_pattern: Dict[str, List[float]] = {pattern: [] for pattern in FIXED_VALIDATION_PATTERNS}
    for record in records:
        by_pattern[record["pattern"]].append(float(record["validate_style_mse"]))
    metrics = {
        pattern: float(np.mean(values)) for pattern, values in by_pattern.items()
    }
    metrics["mean_over_patterns"] = float(
        np.mean([metrics[pattern] for pattern in FIXED_VALIDATION_PATTERNS])
    )
    return metrics


def write_tables_from_payload(payload: Dict, out_dir: Path) -> None:
    records = payload["records"]
    summary = payload["summary"]
    write_csv(records, out_dir / f"{CACHE_STEM}.csv")
    write_markdown_table(summary, out_dir / f"{CACHE_STEM}_table.md")
    write_tex_table(summary, out_dir / f"{CACHE_STEM}_table.tex")
    print(f"Wrote {out_dir / f'{CACHE_STEM}.csv'}")
    print(f"Wrote {out_dir / f'{CACHE_STEM}_table.md'}")
    print(f"Wrote {out_dir / f'{CACHE_STEM}_table.tex'}")


def save_payload(payload: Dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = payload["records"]
    np.savez_compressed(
        out_dir / f"{CACHE_STEM}.npz",
        overall_mse=np.asarray([row["overall_mse"] for row in records], dtype=np.float64),
        density_nrmse=np.asarray([row["density_nrmse"] for row in records], dtype=np.float64),
        jy_nrmse=np.asarray([row["jy_nrmse"] for row in records], dtype=np.float64),
        validate_style_mse=np.asarray(
            [row["validate_style_mse"] for row in records], dtype=np.float64
        ),
        t0=np.asarray([row["t0"] for row in records], dtype=np.int32),
        run_name=np.asarray([row["run_name"] for row in records]),
        pattern=np.asarray([row["pattern"] for row in records]),
    )
    (out_dir / f"{CACHE_STEM}.json").write_text(json.dumps(json_ready(payload), indent=2))
    print(f"Saved {out_dir / f'{CACHE_STEM}.json'}")
    print(f"Saved {out_dir / f'{CACHE_STEM}.npz'}")


def select_val_window_indices(dataset: VPICWindowDataset, val_runs: set[str]) -> List[int]:
    indices = [
        index
        for index, (_file_index, run_name, _t0) in enumerate(dataset.samples)
        if run_name in val_runs
    ]
    if not indices:
        raise RuntimeError(f"No test windows found for {FIXED_SPLIT_PATH} test_runs.")
    return indices


def select_calibration_indices(dataset, val_runs, betas, runs_per_beta, windows_per_run, seed):
    """Read only the existing window index; sample independent held-out runs by beta."""
    grouped = {float(beta): {} for beta in betas}
    for index, (file_index, run_name, _t0) in enumerate(dataset.samples):
        beta = float(dataset.file_infos[file_index]["beta_file"])
        if run_name in val_runs and beta in grouped:
            grouped[beta].setdefault(run_name, []).append(index)
    rng = np.random.default_rng(seed)
    selected = []
    for beta, runs in sorted(grouped.items()):
        eligible = sorted(name for name, indices in runs.items() if len(indices) >= windows_per_run)
        if len(eligible) < runs_per_beta:
            raise ValueError(f"Beta {beta}: need {runs_per_beta} eligible test runs, got {len(eligible)}")
        for name in sorted(rng.choice(eligible, runs_per_beta, replace=False)):
            selected.extend(sorted(rng.choice(runs[name], windows_per_run, replace=False).tolist()))
    return selected


def split_calibration_runs(dataset, indices, seed):
    """Assign whole simulations to disjoint, beta-balanced fit/evaluation subsets."""
    grouped = {}
    for index in indices:
        file_index, run_name, _t0 = dataset.samples[index]
        grouped.setdefault(float(dataset.file_infos[file_index]["beta_file"]), set()).add(run_name)
    rng = np.random.default_rng(seed)
    split = dict(seed=seed, by_beta={}, calibration_runs=[], evaluation_runs=[])
    for beta, names in sorted(grouped.items()):
        names = np.array(sorted(names))
        if len(names) < 2 or len(names) % 2:
            raise ValueError(f"Beta {beta}: global calibration needs an even number of independent runs.")
        rng.shuffle(names)
        fit, evaluation = sorted(names[:len(names) // 2].tolist()), sorted(names[len(names) // 2:].tolist())
        split["by_beta"][str(beta)] = dict(calibration=fit, evaluation=evaluation)
        split["calibration_runs"].extend(fit)
        split["evaluation_runs"].extend(evaluation)
    return split


def summarize_density_calibration(predicted, actual, n_bins=10):
    """Mean calibration and descriptive medians; do not clip or discard extremes."""
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    actual = np.asarray(actual, dtype=np.float64).ravel()
    if (not predicted.size or predicted.shape != actual.shape
            or not np.isfinite(predicted).all() or not np.isfinite(actual).all()
            or (predicted < 0).any() or (actual < 0).any() or n_bins < 1):
        raise ValueError("Calibration requires matching, finite, nonnegative squared-error arrays.")
    edges = np.unique(np.quantile(predicted, np.linspace(0, 1, n_bins + 1)))
    groups = np.searchsorted(edges[1:-1], predicted, side="right")
    bins = []
    for group in np.unique(groups):
        keep = groups == group
        bins.append(dict(bin=len(bins) + 1, count=int(keep.sum()),
                         predicted_min=float(predicted[keep].min()),
                         predicted_max=float(predicted[keep].max()),
                         mean_predicted=float(predicted[keep].mean()),
                         mean_actual=float(actual[keep].mean()),
                         median_actual=float(np.median(actual[keep]))))
    p_mean, a_mean = float(predicted.mean()), float(actual.mean())
    tail_threshold = float(np.quantile(actual, 0.99))
    tail = actual >= tail_threshold
    upper = bins[-1]
    return dict(
        bins=bins, quantile_edges=edges.tolist(), count=int(actual.size),
        mean_predicted=p_mean, mean_actual=a_mean, median_actual=float(np.median(actual)),
        mean_bias_predicted_minus_actual=p_mean - a_mean,
        relative_mean_bias=(p_mean - a_mean) / a_mean if a_mean > 0 else None,
        increasing_mean_transitions=int(np.sum(np.diff([b["mean_actual"] for b in bins]) > 0)),
        total_mean_transitions=len(bins) - 1,
        upper_prediction_bin_actual_over_predicted=(
            upper["mean_actual"] / upper["mean_predicted"] if upper["mean_predicted"] > 0 else None),
        actual_error_top_one_percent=dict(
            threshold=tail_threshold, count=int(tail.sum()),
            mean_actual=float(actual[tail].mean()), mean_predicted=float(predicted[tail].mean()),
            fraction_underpredicted=float(np.mean(actual[tail] > predicted[tail])),
            note="Selection on actual extremes alone is not a test of conditional mean calibration."),
    )


def fit_global_calibration_scale(predicted, actual):
    """Fit only calibration-run means; a zero predictor cannot fit positive error."""
    stats = summarize_density_calibration(predicted, actual, 1)
    if stats["mean_predicted"] == 0:
        if stats["mean_actual"] == 0:
            return 1.0
        raise ValueError("Cannot fit a multiplicative scale: zero predicted mean with positive actual error.")
    factor = stats["mean_actual"] / stats["mean_predicted"]
    if not np.isfinite(factor):
        raise ValueError("Nonfinite global calibration factor.")
    return factor


def mean_calibration_comparison(predicted, actual, factor):
    p, y = np.asarray(predicted, dtype=float), np.asarray(actual, dtype=float)
    p_mean, y_mean = float(p.mean()), float(y.mean())
    calibrated_mean = factor * p_mean
    return dict(mean_predicted=p_mean, mean_actual=y_mean, median_actual=float(np.median(y)),
                mean_calibrated_predicted=calibrated_mean,
                raw_mean_bias=p_mean - y_mean, calibrated_mean_bias=calibrated_mean - y_mean,
                raw_actual_over_predicted=y_mean / p_mean if p_mean > 0 else None,
                calibrated_actual_over_predicted=y_mean / calibrated_mean if calibrated_mean > 0 else None)


def posthoc_calibration_summary(arrays, run_split, n_bins):
    fit_names, eval_names = set(run_split["calibration_runs"]), set(run_split["evaluation_runs"])
    if (not fit_names or not eval_names or fit_names & eval_names
            or fit_names | eval_names != set(arrays["run_name"])):
        raise ValueError("Calibration/evaluation runs must be disjoint and cover the selected simulations.")
    fit = np.isin(arrays["run_name"], list(fit_names))
    evaluation = np.isin(arrays["run_name"], list(eval_names))
    # No evaluation errors enter fitting. Raw model predictions remain untouched.
    factor = fit_global_calibration_scale(
        arrays["predicted_squared_error"][fit], arrays["actual_squared_error"][fit])
    p = arrays["predicted_squared_error"][evaluation].astype(float)
    y = arrays["actual_squared_error"][evaluation].astype(float)
    summary = summarize_density_calibration(p, y, n_bins)
    summary.pop("actual_error_top_one_percent")
    for row in summary["bins"]:
        row["mean_calibrated_predicted"] = factor * row["mean_predicted"]
    groups = []
    for beta in [None, *sorted(np.unique(arrays["beta"]))]:
        for pattern in [None, *FIXED_VALIDATION_PATTERNS]:
            if beta is None and pattern is None:
                continue
            keep = evaluation.copy()
            if beta is not None:
                keep &= arrays["beta"] == beta
            if pattern is not None:
                keep &= arrays["pattern"] == pattern
            if not keep.any():
                continue
            groups.append(dict(beta=beta, pattern=pattern,
                               n_runs=len(np.unique(arrays["run_name"][keep])), n_voxels=int(keep.sum()),
                               **mean_calibration_comparison(arrays["predicted_squared_error"][keep],
                                                             arrays["actual_squared_error"][keep], factor)))
    summary.update(
        global_scale=factor, scale_formula="mean(actual)/mean(predicted), calibration runs only",
        summary_scope="Evaluation runs only; bins defined using raw predicted uncertainty",
        collected_voxel_count=len(fit),
        calibration=dict(n_runs=len(fit_names), n_voxels=int(fit.sum()),
                         **mean_calibration_comparison(arrays["predicted_squared_error"][fit],
                                                       arrays["actual_squared_error"][fit], factor)),
        evaluation=dict(n_runs=len(eval_names), n_voxels=len(p), **mean_calibration_comparison(p, y, factor)),
        ranking=dict(positive_scale=factor > 0,
                     unchanged=bool(factor > 0 and np.array_equal(np.argsort(p, kind="stable"),
                                                                np.argsort(factor * p, kind="stable"))),
                     increasing_mean_transitions=summary["increasing_mean_transitions"],
                     total_mean_transitions=summary["total_mean_transitions"],
                     top_over_bottom_bin_mean_actual=(summary["bins"][-1]["mean_actual"] /
                         summary["bins"][0]["mean_actual"] if summary["bins"][0]["mean_actual"] > 0 else None)),
        groups=groups,
    )
    return summary


def save_calibration_heatmap(groups, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    betas = sorted({row["beta"] for row in groups if row["beta"] is not None})
    cells = {(row["beta"], row["pattern"]): row for row in groups
             if row["beta"] is not None and row["pattern"] is not None}
    keys = ["raw_actual_over_predicted", "calibrated_actual_over_predicted"]
    matrices = [np.array([[cells[b, p][key] if cells[b, p][key] is not None else np.nan
                          for p in FIXED_VALIDATION_PATTERNS] for b in betas]) for key in keys]
    maximum = max([2.0, *[float(v) for m in matrices for v in m.ravel() if np.isfinite(v)]])
    norm = TwoSlopeNorm(vmin=0, vcenter=1, vmax=maximum)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True, layout="constrained")
    for ax, matrix, title in zip(axes, matrices, ["Raw", "Global multiplicative calibration"]):
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap="coolwarm", norm=norm, aspect="auto")
        ax.set(xticks=range(5), xticklabels=[p.replace("_", "\n") for p in FIXED_VALIDATION_PATTERNS],
               yticks=range(len(betas)), yticklabels=[f"{b:g}" for b in betas], title=title, xlabel="Mask pattern")
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, f"{matrix[i, j]:.2f}" if np.isfinite(matrix[i, j]) else "N/A",
                    ha="center", va="center", fontsize=9)
    axes[0].set_ylabel("Plasma beta")
    fig.colorbar(im, ax=axes, label="Mean actual / mean predicted squared error (1 = agreement)")
    fig.suptitle("Evaluation simulations only: density mean-error ratios")
    fig.savefig(out_dir / "density_calibration_heatmap.png", dpi=160)
    plt.close(fig)


def save_density_calibration(samples, records, metadata, out_dir, n_bins, run_split=None):
    """Save sampled voxels, bin statistics and one reliability diagram in this evaluator."""
    import matplotlib.pyplot as plt

    arrays = {key: np.concatenate([part[key] for part in samples]) for key in samples[0]}
    summary = (posthoc_calibration_summary(arrays, run_split, n_bins) if run_split is not None
               else summarize_density_calibration(
                   arrays["predicted_squared_error"], arrays["actual_squared_error"], n_bins))
    if run_split is not None:
        arrays["subset"] = np.where(np.isin(arrays["run_name"], run_split["calibration_runs"]),
                                    "calibration", "evaluation")
        (out_dir / "calibration_run_split.json").write_text(json.dumps(run_split, indent=2))
        with (out_dir / "density_calibration_groups.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary["groups"][0]))
            writer.writeheader()
            writer.writerows(summary["groups"])
        save_calibration_heatmap(summary["groups"], out_dir)
    summary.update(metadata)
    summary["per_run_pattern"] = records
    summary["units"] = "squared standardized Density units"
    summary["interpretation"] = (
        "Only unobserved voxels; equal capped samples per selected window/pattern. "
        "Voxel errors are correlated within simulations. The median is descriptive; "
        "y=x tests mean calibration, not median agreement. This pilot cannot establish calibration."
    )
    if run_split is not None:
        summary["interpretation"] += (
            " Held-out-run post-hoc calibration evaluation, not an independent test set: "
            "the final checkpoint was predetermined before this test population was evaluated. Runs are the independent "
            "sampling units; no voxel-level uncertainty intervals are reported. Positive scaling "
            "preserves discrimination but global mean correction does not guarantee conditional "
            "calibration. "
        )
        summary["interpretation"] += (
            "Gaussian NLL with reconstruction detached targets conditional expected squared residual; "
            "finite-sample training does not guarantee calibrated variance."
            if summary.get("uncertainty_objective") == "gaussian_nll" else
            "Smooth L1 targets a robust conditional location of squared error, "
            "which need not equal its conditional mean."
        )
    np.savez_compressed(out_dir / "density_calibration_voxels.npz", **arrays)
    (out_dir / "density_calibration.json").write_text(json.dumps(json_ready(summary), indent=2))
    with (out_dir / "density_calibration_bins.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary["bins"][0]))
        writer.writeheader()
        writer.writerows(summary["bins"])
    bins = summary["bins"]
    means, medians = [b["mean_actual"] for b in bins], [b["median_actual"] for b in bins]
    keys = ["mean_predicted"] + (["mean_calibrated_predicted"] if run_split is not None else [])
    limit = max([1e-12, *means, *medians, *[b[key] for key in keys for b in bins]]) * 1.05
    fig, axes = plt.subplots(1, len(keys), figsize=(7.5 if len(keys) == 1 else 12, 6),
                             squeeze=False, sharex=True, sharey=True)
    for ax, key in zip(axes[0], keys):
        x = [b[key] for b in bins]
        ax.plot(x, means, "o-", label="Mean actual squared error")
        ax.plot(x, medians, "s--", label="Median actual squared error (descriptive)")
        ax.plot([0, limit], [0, limit], "k:", label="y=x: mean calibration")
        ax.set(xlabel="Mean predicted density squared error (standardized units²)",
               ylabel="Actual density squared error (standardized units²)",
               xlim=(0, limit), ylim=(0, limit),
               title=("Held-out density reliability: unobserved voxels" if run_split is None else
                      "Raw" if key == "mean_predicted" else f"Scaled: a={summary['global_scale']:.4g}"))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    if run_split is not None:
        fig.suptitle("Held-out-run post-hoc evaluation: same raw-uncertainty bins in both panels")
    fig.tight_layout()
    fig.savefig(out_dir / "density_calibration.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in {"bins", "per_run_pattern", "split", "groups"}}, indent=2))


@torch.no_grad()
def evaluate_windows(
    model: torch.nn.Module,
    dataset: VPICWindowDataset,
    sample_indices: Sequence[int],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    extent: Sequence[float],
    jy_std_train: float | None,
    calibration_samples: List[Dict] | None = None,
    calibration_voxels_per_pattern: int = 2048,
    calibration_seed: int = 1234,
) -> Tuple[List[Dict], List[Dict]]:
    records: List[Dict] = []
    pattern_infos: List[Dict] | None = None
    rng = np.random.default_rng(calibration_seed)
    model.eval()
    for sample_number, dataset_index in enumerate(sample_indices, start=1):
        sample = dataset[dataset_index]
        target = sample["block"].unsqueeze(0).to(device)
        target_norm = normalize(target, mean, std)
        masks, infos = build_standardized_masks(
            target_norm.shape,
            device=device,
            dtype=target_norm.dtype,
        )
        if pattern_infos is None:
            pattern_infos = infos
        stacked_mask = torch.cat(masks, dim=0)
        stacked_target = target_norm.expand(len(FIXED_VALIDATION_PATTERNS), -1, -1, -1, -1)
        prediction, output_variance = predict_masked_fields(model, stacked_target, stacked_mask)
        prediction = prediction.float()
        target_np = stacked_target[0].detach().cpu().numpy()
        pred_np = prediction.detach().cpu().numpy()
        target_phys_np = target[0].detach().cpu().numpy()
        pred_phys_np = (denormalize(prediction, mean, std).detach().cpu().numpy()
                        if jy_std_train is not None else None)
        metadata = sample["metadata"]
        if calibration_samples is not None:
            if output_variance is None:
                raise ValueError("Density calibration requires an uncertainty-aware checkpoint.")
            if (not torch.isfinite(prediction).all() or not torch.isfinite(output_variance).all()
                    or (output_variance < 0).any()):
                raise ValueError("Nonfinite fields or invalid predicted squared errors; calibration stopped.")
            predicted_error = output_variance[:, 3].detach().cpu().numpy()
            actual_error = (prediction[:, 3] - stacked_target[:, 3]).square().detach().cpu().numpy()
        for pattern_index, pattern in enumerate(FIXED_VALIDATION_PATTERNS):
            pred_i = pred_np[pattern_index]
            mask_i = stacked_mask[pattern_index : pattern_index + 1]
            target_i = stacked_target[pattern_index : pattern_index + 1]
            pred_i_t = prediction[pattern_index : pattern_index + 1]
            records.append(
                {
                    "run_name": metadata["run_name"],
                    "t0": int(metadata["t0"]),
                    "pattern": pattern,
                    "overall_mse": window_overall_mse(pred_i, target_np),
                    "density_nrmse": window_density_nrmse(pred_i, target_np),
                    "jy_nrmse": window_jy_nrmse(
                        pred_phys_np[pattern_index],
                        target_phys_np,
                        extent,
                        jy_std_train,
                    ) if jy_std_train is not None else None,
                    "validate_style_mse": float(
                        full_mse_loss(pred_i_t, target_i, mask_i).detach().cpu()
                    ),
                }
            )
            if calibration_samples is not None:
                observation_mask = mask_i[0, 3].detach().cpu().numpy().ravel()
                hidden = np.flatnonzero(observation_mask == 0)
                chosen = rng.choice(hidden, min(len(hidden), calibration_voxels_per_pattern), replace=False)
                calibration_samples.append(dict(
                    predicted_squared_error=predicted_error[pattern_index].ravel()[chosen],
                    actual_squared_error=actual_error[pattern_index].ravel()[chosen],
                    observed=observation_mask[chosen].astype(bool), voxel_index=chosen,
                    beta=np.full(len(chosen), float(metadata["beta"])),
                    pattern=np.full(len(chosen), pattern), run_name=np.full(len(chosen), metadata["run_name"]),
                    t0=np.full(len(chosen), int(metadata["t0"])),
                ))
                records[-1]["beta"] = float(metadata["beta"])
        if (
            sample_number == 1
            or sample_number % 10 == 0
            or sample_number == len(sample_indices)
        ):
            print(
                f"Standardized test evaluation: {sample_number}/{len(sample_indices)} windows"
            )
    return records, pattern_infos or []


def main() -> None:
    args = parse_args()
    run_dir = expand_path(args.run_dir)
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    split = load_fixed_split()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_final_checkpoint(ckpt)
    ckpt_args = ckpt["args"]
    checkpoint_epoch = int(ckpt["epoch"])
    stats = ckpt["stats"]
    mean_list = [float(value) for value in stats["mean"]]
    std_list = [float(value) for value in stats["std"]]
    uncertainty_aware = bool(ckpt_args.get("uncertainty_aware", False))
    calibration = args.density_calibration or args.fit_global_scale
    if calibration and not uncertainty_aware:
        raise ValueError("Density calibration requires an uncertainty-aware checkpoint.")
    out_dir = (expand_path(args.out_dir) if args.out_dir is not None else run_dir /
               ("density_calibration" if calibration else OUTPUT_DIRNAME))
    out_dir.mkdir(parents=True, exist_ok=True)
    if calibration and (out_dir / "density_calibration.json").exists() and not args.force_recompute:
        raise FileExistsError("Calibration output already exists; choose --out-dir or explicitly --force-recompute.")

    payload = None
    if not calibration:
        jy_path = run_dir / "jy_stats.json"
        if not jy_path.exists():
            raise FileNotFoundError(
                f"Missing {jy_path}; standardized evaluation reuses existing Jy stats."
            )
        jy_stats = json.loads(jy_path.read_text())
        signature = cache_signature(
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=checkpoint_epoch,
            split=split,
            mean=mean_list,
            std=std_list,
            jy_stats=jy_stats,
            extent=args.extent,
        )
        payload = None if args.force_recompute else load_cached_payload(out_dir, signature)

    if payload is None:
        device = torch.device(
            args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        ckpt, ckpt_path = load_checkpoint(run_dir, args.checkpoint, device=device)
        h5_dir = expand_path(args.h5_dir or ckpt_args["h5_dir"])
        delta_t = int(ckpt_args.get("delta_t", 24))
        dataset = VPICWindowDataset(
            h5_dir=h5_dir,
            betas=ckpt_args.get("betas", [0.2]),
            delta_t=delta_t,
            stride_t=int(ckpt_args.get("stride_t", 2)),
            layout="C T X Z",
            return_metadata=True,
        )
        test_runs = get_test_runs()
        train_runs = get_train_runs()
        mean = torch.tensor(mean_list, dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
        std = torch.tensor(std_list, dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
        if not calibration:
            jy_stats = load_or_compute_jy_training_stats(
                run_dir=run_dir,
                dataset=dataset,
                train_runs=train_runs,
                mean=mean,
                std=std,
                extent=args.extent,
                checkpoint_path=ckpt_path,
                checkpoint_epoch=checkpoint_epoch,
            )
        model = UNet3D(
            in_channels=8,
            out_channels=4,
            base_channels=int(ckpt_args.get("base_channels", 16)),
            channel_mults=ckpt_args.get("channel_mults", [1, 2, 4]),
            architecture=ckpt_args.get("model_version", LEGACY_MODEL_VERSION),
            uncertainty_aware=uncertainty_aware,
            use_attention=bool(ckpt_args.get("use_attention", False)),
            spatial_only_pooling=bool(ckpt_args.get("spatial_only_pooling", False)),
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()
        if calibration:
            if test_runs & train_runs:
                raise ValueError("Training and test simulation sets overlap.")
            sample_indices = select_calibration_indices(
                dataset, test_runs, ckpt_args["betas"], args.calibration_runs_per_beta,
                args.calibration_windows_per_run, args.calibration_seed)
            run_split = (split_calibration_runs(dataset, sample_indices, args.calibration_seed)
                         if args.fit_global_scale else None)
            torch.manual_seed(args.calibration_seed)
            samples = []
            started = time.perf_counter()
            records, pattern_infos = evaluate_windows(
                model, dataset, sample_indices, mean, std, device, args.extent, None,
                calibration_samples=samples,
                calibration_voxels_per_pattern=args.calibration_voxels_per_pattern,
                calibration_seed=args.calibration_seed,
            )
            save_density_calibration(samples, records, dict(
                checkpoint=checkpoint_cache_signature(ckpt_path), checkpoint_epoch=checkpoint_epoch,
                uncertainty_objective=ckpt_args.get("uncertainty_loss", "smooth_l1"),
                normalization=stats, split=split_identity(split),
                patterns=list(FIXED_VALIDATION_PATTERNS), pattern_infos=pattern_infos,
                n_windows=len(sample_indices), n_runs=len({r["run_name"] for r in records}),
                betas=ckpt_args["betas"], seed=args.calibration_seed,
                runs_per_beta=args.calibration_runs_per_beta, windows_per_run=args.calibration_windows_per_run,
                voxels_per_window_pattern=args.calibration_voxels_per_pattern,
                device=str(device), inference_dtype="bfloat16 autocast" if device.type == "cuda" else "float32",
                elapsed_seconds=time.perf_counter() - started,
            ), out_dir, args.calibration_bins, run_split=run_split)
            dataset.close()
            return
        sample_indices = select_val_window_indices(dataset, test_runs)
        records, pattern_infos = evaluate_windows(
            model=model,
            dataset=dataset,
            sample_indices=sample_indices,
            mean=mean,
            std=std,
            device=device,
            extent=args.extent,
            jy_std_train=float(jy_stats["jy_std_train"]),
        )
        signature = cache_signature(
            checkpoint_path=ckpt_path,
            checkpoint_epoch=checkpoint_epoch,
            split=split,
            mean=mean_list,
            std=std_list,
            jy_stats=jy_stats,
            extent=args.extent,
        )
        summary = summarize_from_records(records)
        per_run = per_run_means(records)
        payload = {
            "signature": signature,
            "checkpoint": str(ckpt_path),
            "checkpoint_epoch": checkpoint_epoch,
            "patterns": list(FIXED_VALIDATION_PATTERNS),
            "pattern_labels": PATTERN_LABELS,
            "pattern_infos": pattern_infos,
            "n_windows": len(sample_indices),
            "n_runs": len(test_runs),
            "records": records,
            "per_run": per_run,
            "summary": summary,
            "global_window_weighted_overall_mse": global_window_weighted_overall_mse(
                records
            ),
            "aggregation": (
                "mean over windows within each test run, then cross-run "
                "median and 16th-84th percentiles"
            ),
            "overall_mse_definition": OVERALL_MSE_DEFINITION,
            "density_nrmse_definition": DENSITY_NRMSE_DEFINITION,
            "jy_nrmse_definition": JY_NRMSE_DEFINITION,
            "jy_definition": JY_STATS_DEFINITION,
            "note": (
                "global_window_weighted_overall_mse uses data.masking.full_mse_loss "
                "averaged with equal window weight, like train_masked_unet3d.validate(); "
                "it is metadata only and is not the paper table statistic"
            ),
        }
        payload.update(
            jy_metric_metadata(
                jy_stats=jy_stats,
                checkpoint_path=ckpt_path,
                checkpoint_epoch=checkpoint_epoch,
            )
        )
        save_payload(payload, out_dir)
        dataset.close()

    write_tables_from_payload(payload, out_dir)
    manifest = {
        "out_dir": str(out_dir),
        "checkpoint": payload.get("checkpoint", str(checkpoint_path)),
        "checkpoint_epoch": payload.get("checkpoint_epoch", checkpoint_epoch),
        "cache_version": CACHE_VERSION,
        "force_recompute": bool(args.force_recompute),
        "files": [
            f"{CACHE_STEM}.npz",
            f"{CACHE_STEM}.json",
            f"{CACHE_STEM}.csv",
            f"{CACHE_STEM}_table.md",
            f"{CACHE_STEM}_table.tex",
            "manifest.json",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
