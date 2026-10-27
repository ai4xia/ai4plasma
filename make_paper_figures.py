# make_paper_figures.py
"""Paper-specific figures with a dedicated cache, separate from GIF tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
from matplotlib.lines import Line2D

from data.vpic_hdf5_dataset import (
    VPICWindowDataset,
    find_h5_files,
    fixed_split_metadata,
    load_fixed_split,
    validate_checkpoint_split,
    validate_final_checkpoint,
)
from models.unet3d import LEGACY_MODEL_VERSION, UNet3D
from visualize_mask_patterns_unet3d import (
    DEFAULT_MAGNETIC_ABLATION_VISIBLE_FRACTIONS,
    DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS,
    DEFAULT_RESIDUAL_VMAX,
    DEFAULT_RUN_DIR,
    DEFAULT_RUN_NAME,
    DEFAULT_T0,
    JY_INCLUDES_MU0,
    JY_NRMSE_DEFINITION,
    JY_STATS_CACHE_VERSION,
    JY_STATS_DEFINITION,
    JY_STATS_PREPROCESSING,
    RESIDUAL_CMAP,
    _density_probe_count_grid,
    build_density_forecast_rows,
    build_density_only_multifunction_rows,
    build_density_superres_rows,
    build_magnetic_ablation_rows,
    checkpoint_cache_signature,
    collect_validation_statistics,
    compute_normalized_metrics,
    compute_jy,
    compute_physical_jy_from_normalized,
    default_density_forecast_visible_frames,
    denormalize,
    expand_path,
    get_train_runs,
    get_test_runs,
    jy_metric_metadata,
    jy_nrmse_from_residual,
    load_checkpoint,
    load_or_compute_jy_training_stats,
    make_nan_cmap,
    predict_masked_fields,
    normalize,
    normalized_residual,
    resolve_checkpoint_path,
    robust_limits,
    select_run_t0_index,
    select_validation_statistics_indices,
    _style_comparison_axis,
)
from visualize_sliding_density_reconstruction import (
    find_and_load_run,
    framewise_rmse_mae,
    project_time_z,
    reconstruct_with_slide_step,
    validate_test_run,
)
from journal_statistics import (
    calibration_bias,
    group_run_records,
    pearson_spearman,
    predicted_voxel_mask,
    quantile_calibration,
    summarize,
    summarize_run_records,
)


PAPER_CACHE_VERSION = 14
COMPATIBLE_PAPER_CACHE_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
PAPER_DIRNAME = "paper_figures_v1"
DEFAULT_GLOBAL_FRAME = 45
DEFAULT_SLIDE_STEPS = (1, 2, 4, 6, 8, 12, 24)
DEFAULT_PROBE_COUNTS = (0, 10, 100, 1000)
DEFAULT_X_INDEX = 130
DEFAULT_DENSITY_VISIBLE_FRACTION = 0.08
DEFAULT_SLIDING_MAX_RUNS = 16
DEFAULT_SLIDING_DENSITY_PROBE_COUNT = 1000
SLIDING_AGGREGATE_N_FRAMES = 48
LOG_Y_FLOOR = 1e-6
PNG_DPI = 300

SPATIAL_ROW_ORDER = ("spatial_random", "spatial_grid", "spatial_block_inpainting")
SPATIAL_ROW_LABELS = {
    "spatial_random": "Spatial random",
    "spatial_grid": "Spatial grid",
    "spatial_block_inpainting": "Spatial block\ninpainting",
    "spatial_block_outpainting": "Spatial block\noutpainting",
}
VAL_JSON_EXPERIMENT = {
    "forecast": "density_forecast",
    "superres": "density_superres",
    "magnetic_ablation": "magnetic_ablation",
    "density_forecast": "density_forecast",
    "density_superres": "density_superres",
}
FIGURE_STEMS = {
    "spatial": "spatial_qualitative",
    "magnetic_ablation": "magnetic_ablation_summary",
    "forecast": "density_forecast_summary",
    "superres": "density_superres_summary",
    "sliding": "sliding_window_appendix",
    "parameters": "physical_parameter_dependence",
    "global_uq": "global_uncertainty_quality",
}
ALL_FIGURES = tuple(FIGURE_STEMS)
INFO_DIR_WITH_B = "figures_information_suite_plasmoid_merger"
INFO_DIR_NO_B = "figures_information_suite_plasmoid_merger_no_magnetic"
SLIDING_DIR_NO_B = "figures_sliding_density_reconstruction_plasmoid_merger_no_magnetic"
EXPECTED_FORECAST_HISTORIES = tuple(default_density_forecast_visible_frames(24))
EXPECTED_FORECAST_ROW_NAMES = tuple(
    f"density_forecast_history_{history}" for history in EXPECTED_FORECAST_HISTORIES
)
EXPECTED_SUPERRES_PROBE_COUNTS = tuple(DEFAULT_PROBE_COUNTS)
EXPECTED_SUPERRES_ROW_NAMES = tuple(
    f"density_superres_{count}" for count in EXPECTED_SUPERRES_PROBE_COUNTS
)
EXPECTED_MAGNETIC_ROW_NAMES = tuple(
    f"magnetic_ablation_{percent / 100.0:g}"
    for percent in DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS
)
MAGNETIC_DENSITY_CONDITIONS = (
    ("density_hidden", "Density fully hidden"),
    ("density_full", "Density visible 100%"),
)
SUPERRES_B_CONDITION_STYLES = {
    "B_full": dict(color="C0", linestyle="-", marker="o", label="B visible 100%"),
    "B_hidden": dict(color="C1", linestyle="--", marker="^", label="B visible 0%"),
}

DENSITY_NRMSE_DEFINITION = (
    "RMS of (prediction_normalized - target_normalized) on the Density channel"
)


def _jy_signature_fields() -> Dict:
    return {
        "jy_preprocessing": JY_STATS_PREPROCESSING,
        "jy_stats_cache_version": JY_STATS_CACHE_VERSION,
        "jy_includes_mu0": JY_INCLUDES_MU0,
        "B_unit": "T",
        "jy_unit": "A/m^2",
    }


JY_PAPER_PANEL_TITLE = "Jy NRMSE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render paper figures from cached arrays, recomputing only if needed."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="final.pt")
    parser.add_argument("--allow-incomplete-checkpoint", action="store_true",
        help="Allow an explicit latest.pt diagnostic before the predetermined final epoch.")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--paper-dir", default=None)
    parser.add_argument(
        "--output-dir", default=None,
        help="Explicit output directory (alias for --paper-dir; takes precedence).",
    )
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument(
        "--figure",
        nargs="+",
        default=["all"],
        choices=[*ALL_FIGURES, "all"],
    )
    parser.add_argument("--t0", type=int, default=DEFAULT_T0)
    parser.add_argument("--global-frame", type=int, default=DEFAULT_GLOBAL_FRAME)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--h5-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=[-21.0, 21.0, -50.0, 50.0],
    )
    parser.add_argument("--residual-vmax", type=float, default=DEFAULT_RESIDUAL_VMAX)
    parser.add_argument("--field-q", type=float, default=99.0)
    parser.add_argument("--dpi", type=int, default=PNG_DPI)
    parser.add_argument("--x-index", type=int, default=DEFAULT_X_INDEX)
    parser.add_argument(
        "--density-visible-fraction",
        type=float,
        default=DEFAULT_DENSITY_VISIBLE_FRACTION,
        help=(
            "Unused by paper sliding; kept for CLI compatibility. "
            "Sliding Density probes use --density-probe-count."
        ),
    )
    parser.add_argument(
        "--density-probe-count",
        type=int,
        default=DEFAULT_SLIDING_DENSITY_PROBE_COUNT,
        help=(
            "Exact Density probe count for sliding appendix, using the same "
            "exact-count grid as density super-resolution. Default: 1000."
        ),
    )
    parser.add_argument(
        "--density-probe-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROBE_COUNTS),
    )
    parser.add_argument(
        "--slide-steps",
        type=int,
        nargs="+",
        default=list(DEFAULT_SLIDE_STEPS),
    )
    parser.add_argument(
        "--sliding-max-runs",
        type=int,
        default=0,
        help=(
            "Maximum held-out test runs (0 means every eligible test run) for "
            "RMSE stats, taken in authoritative test-split order. Runs shorter than 48 frames "
            "are skipped; remaining runs use only the first 48 frames."
        ),
    )
    parser.add_argument("--statistics-max-windows-per-run", type=int, default=None)
    parser.add_argument("--uq-sample-size", "--uq-voxels-per-window", dest="uq_voxels_per_window",
        type=int, default=20000,
        help="Deterministic bounded sample cap for per-run/task correlations (default: 20000).")
    return parser.parse_args()


def selected_figures(figure_args: Sequence[str]) -> List[str]:
    if "all" in figure_args:
        return list(ALL_FIGURES)
    return list(dict.fromkeys(figure_args))


def local_index_for_global_frame(t0: int, global_frame: int, delta_t: int) -> int:
    local = int(global_frame) - int(t0)
    if not (0 <= local < int(delta_t)):
        raise ValueError(
            f"global frame {global_frame} is outside window t0={t0} "
            f"with T={delta_t} (local index would be {local})."
        )
    return local


def density_visible_ratio(probe_count: int, size_x: int, size_z: int) -> float:
    sites = int(size_x) * int(size_z)
    if sites <= 0:
        raise ValueError(f"Invalid spatial size {(size_x, size_z)}")
    return float(probe_count) / float(sites)


def format_visible_ratio_percent(ratio_percent: float) -> str:
    if abs(ratio_percent) < 1e-12:
        return "0"
    if ratio_percent < 0.2:
        return f"{ratio_percent:.3f}".rstrip("0").rstrip(".")
    return f"{ratio_percent:.2f}".rstrip("0").rstrip(".")


def clip_positive_for_log(
    values: np.ndarray,
    floor: float = LOG_Y_FLOOR,
) -> np.ndarray:
    """Clip non-positive values at plot time so a log axis stays defined."""
    return np.maximum(np.asarray(values, dtype=np.float64), float(floor))


def density_nrmse_from_residual(residual: np.ndarray) -> float:
    nrmse, _nmae = compute_normalized_metrics(residual)
    return float(nrmse)


def qualitative_density_nrmse_key(step: int) -> str:
    return f"qualitative_density_nrmse_step{int(step)}"


def compute_sliding_qualitative_density_nrmse(arrays: Dict, step: int) -> float:
    """Canonical-run Density NRMSE from cached sliding arrays, no inference.

    Uses RMS(pred_norm - target_norm) on Density, same as the paper pipeline.
    Prefers full (T,X,Z) prediction/target; falls back to the plotted (T,Z)
    normalized residual. Scope is this canonical example only.
    """
    step = int(step)
    pred = arrays.get(f"step_{step}_prediction")
    target = arrays.get("target_density")
    tz_residual = arrays.get(f"step_{step}_tz_residual")
    tz_pred = arrays.get(f"step_{step}_tz")
    tz_target = arrays.get("target_tz")
    if (
        pred is not None
        and target is not None
        and tz_residual is not None
        and tz_pred is not None
        and tz_target is not None
    ):
        slice_nrmse = density_nrmse_from_residual(tz_residual)
        slice_rmse = density_nrmse_from_residual(
            np.asarray(tz_pred, dtype=np.float64)
            - np.asarray(tz_target, dtype=np.float64)
        )
        if (
            slice_nrmse > 0
            and np.isfinite(slice_nrmse)
            and np.isfinite(slice_rmse)
        ):
            std = slice_rmse / slice_nrmse
            if std > 0:
                return density_nrmse_from_residual(
                    (
                        np.asarray(pred, dtype=np.float64)
                        - np.asarray(target, dtype=np.float64)
                    )
                    / std
                )
    if tz_residual is None:
        raise KeyError(f"missing Density residual arrays for sliding step={step}")
    return density_nrmse_from_residual(tz_residual)


def ensure_sliding_qualitative_nrmse_metadata(
    arrays: Dict,
    metadata: Dict,
) -> bool:
    """Fill canonical-run NRMSE scalars in metadata. Does not change signature."""
    changed = False
    for step in metadata.get("slide_steps") or []:
        key = qualitative_density_nrmse_key(step)
        nrmse = float(compute_sliding_qualitative_density_nrmse(arrays, int(step)))
        if metadata.get(key) != nrmse:
            metadata[key] = nrmse
            changed = True
    if metadata.get("qualitative_density_nrmse_definition") != DENSITY_NRMSE_DEFINITION:
        metadata["qualitative_density_nrmse_definition"] = DENSITY_NRMSE_DEFINITION
        metadata["qualitative_density_nrmse_scope"] = (
            "canonical run, first 48 frames, B fully hidden, 1000 Density probes; "
            "one aggregate over the qualitative reconstruction, not cross-run"
        )
        changed = True
    return changed


def format_density_nrmse_label(nrmse: float) -> str:
    return f"NRMSE = {nrmse:.3g}"


def annotate_density_nrmse(ax, nrmse: float) -> None:
    ax.text(
        0.03,
        0.97,
        format_density_nrmse_label(nrmse),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "0.75",
            "linewidth": 0.4,
            "alpha": 0.88,
        },
    )


def validation_run_order(run_dir: Path) -> List[str]:
    del run_dir
    return [str(name) for name in load_fixed_split()["test_runs"]]


def peek_run_n_frames(
    h5_dir: Path,
    betas: Sequence[float],
    run_name: str,
) -> int | None:
    for h5_path in find_h5_files(h5_dir, betas=betas):
        with h5py.File(h5_path, "r") as h5_file:
            runs = h5_file.get("runs")
            if runs is None or run_name not in runs:
                continue
            return int(runs[run_name]["fields"].shape[0])
    return None


def select_sliding_aggregate_runs(
    val_run_order: Sequence[str],
    max_runs: int,
) -> List[str]:
    """Take the first ``max_runs`` test runs in authoritative split order."""
    if int(max_runs) < 0:
        raise ValueError(f"sliding max runs must be nonnegative, got {max_runs}")
    runs = [str(name) for name in val_run_order]
    return runs if int(max_runs) == 0 else runs[: int(max_runs)]


def sliding_density_probe_mask(
    block: torch.Tensor,
    probe_count: int,
) -> Tuple[torch.Tensor, Dict]:
    """Exact-count Density grid, same helper as density super-resolution."""
    mask_full, info = _density_probe_count_grid(block, int(probe_count))
    return mask_full[0, 0, 0], info


def stack_rmse_by_global_frame(
    per_run_frame_ids: Sequence[np.ndarray],
    per_run_rmse: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    frames = np.array(
        sorted(
            {
                int(value)
                for ids in per_run_frame_ids
                for value in np.asarray(ids).ravel()
            }
        ),
        dtype=np.int32,
    )
    stacked = np.full((len(per_run_rmse), len(frames)), np.nan, dtype=np.float64)
    index = {int(frame): col for col, frame in enumerate(frames)}
    for row, (ids, rmse) in enumerate(zip(per_run_frame_ids, per_run_rmse)):
        for frame_id, value in zip(np.asarray(ids).ravel(), np.asarray(rmse).ravel()):
            stacked[row, index[int(frame_id)]] = float(value)
    return frames, stacked


def density_std_from_run_dir(run_dir: Path) -> float | None:
    stats_path = run_dir / "stats.json"
    if not stats_path.exists():
        return None
    payload = json.loads(stats_path.read_text())
    std = payload.get("std") or []
    if len(std) < 4:
        return None
    return float(std[3])


def apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


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


def save_figure_cache(data_dir: Path, stem: str, arrays: Dict, metadata: Dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    npz_path = data_dir / f"{stem}.npz"
    json_path = data_dir / f"{stem}.json"
    np.savez_compressed(npz_path, **arrays)
    json_path.write_text(json.dumps(json_ready(metadata), indent=2))
    print(f"Saved paper cache: {json_path}")


def load_figure_cache(data_dir: Path, stem: str) -> Tuple[Dict, Dict] | None:
    npz_path = data_dir / f"{stem}.npz"
    json_path = data_dir / f"{stem}.json"
    if not npz_path.exists() or not json_path.exists():
        return None
    try:
        metadata = json.loads(json_path.read_text())
        with np.load(npz_path, allow_pickle=False) as handle:
            arrays = {key: np.asarray(handle[key]) for key in handle.files}
        return arrays, metadata
    except Exception as exc:
        print(f"Could not load paper cache {json_path}: {exc}")
        return None


def load_or_build_figure_cache(
    data_dir: Path,
    stem: str,
    signature: Dict,
    builder: Callable[[], Tuple[Dict, Dict]],
    force: bool = False,
    provenance: Dict | None = None,
) -> Tuple[Dict, Dict]:
    if not force:
        loaded = load_figure_cache(data_dir, stem)
        if loaded is not None:
            arrays, metadata = loaded
            if paper_signatures_match(metadata.get("signature"), signature):
                metadata = apply_provenance_metadata(metadata, provenance)
                metadata["signature"] = signature
                if metadata.get("signature") != loaded[1].get("signature") or any(
                    metadata.get(key) != loaded[1].get(key)
                    for key in (
                        "source_checkpoint",
                        "source_checkpoint_epoch",
                        "source_files",
                    )
                ):
                    save_figure_cache(data_dir, stem, arrays, metadata)
                    print(f"Updated paper cache provenance: {data_dir / (stem + '.json')}")
                else:
                    print(f"Reusing paper cache: {data_dir / (stem + '.json')}")
                return arrays, metadata
            print(f"Paper cache signature mismatch for {stem}; recomputing")
    arrays, metadata = builder()
    metadata = apply_provenance_metadata(dict(metadata), provenance)
    metadata["signature"] = signature
    save_figure_cache(data_dir, stem, arrays, metadata)
    return arrays, metadata


def save_png_pdf(fig, figures_dir: Path, stem: str, dpi: int) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    png_path = figures_dir / f"{stem}.png"
    pdf_path = figures_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02,)
    fig.savefig(pdf_path,bbox_inches="tight", pad_inches=0.02,transparent=True)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def base_signature(args: argparse.Namespace, run_dir: Path, checkpoint_path: Path) -> Dict:
    split = load_fixed_split()
    return {
        "cache_version": PAPER_CACHE_VERSION,
        "checkpoint": checkpoint_cache_signature(checkpoint_path),
        "run_name": args.run_name,
        "t0": int(args.t0),
        "global_frame": int(args.global_frame),
        "seed": int(args.seed),
        "extent": [float(value) for value in args.extent],
        "data_role": "test",
        "data_split": fixed_split_metadata(),
        "test_runs": sorted(split["test_runs"]),
        "train_run_count": len(split.get("train_runs", [])),
    }


def summarize_run_scalars(per_run_profiles: Dict[str, Sequence[float]]) -> Dict:
    """Mean over local frames of each run profile, then cross-run percentiles."""
    run_names = sorted(per_run_profiles)
    values = np.asarray(
        [np.mean(np.asarray(per_run_profiles[name], dtype=np.float64)) for name in run_names],
        dtype=np.float64,
    )
    return {
        "run_names": run_names,
        "per_run": values.tolist(),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "p16": float(np.percentile(values, 16.0)),
        "p84": float(np.percentile(values, 84.0)),
        "n": int(values.size),
        "aggregation": (
            "existing per-run median over windows, then mean over local frames, "
            "then cross-run median and 16th-84th percentiles"
        ),
    }


def parse_trailing_number(name: str) -> float:
    return float(str(name).rsplit("_", 1)[1])


def paper_signatures_match(stored: Dict | None, current: Dict) -> bool:
    if stored == current:
        return True
    if not isinstance(stored, dict):
        return False
    stored_norm = dict(stored)
    current_norm = dict(current)
    if stored_norm.get("cache_version") not in COMPATIBLE_PAPER_CACHE_VERSIONS:
        return False
    stored_norm["cache_version"] = current_norm.get("cache_version")
    return stored_norm == current_norm


def infer_source_files(metadata: Dict) -> List[str]:
    if metadata.get("source_files"):
        return [str(path) for path in metadata["source_files"]]
    source = metadata.get("source")
    if isinstance(source, str) and source.endswith((".json", ".npz")):
        return [source]
    if isinstance(source, dict):
        files = []
        for value in source.values():
            if isinstance(value, str) and value.endswith((".json", ".npz")):
                files.append(value)
        return files
    if (
        metadata.get("figure") == "magnetic_ablation_summary"
        and source == "validation_json"
    ):
        checkpoint = metadata.get("source_checkpoint") or (
            (metadata.get("signature") or {}).get("checkpoint") or {}
        ).get("path")
        if checkpoint:
            run_dir = Path(checkpoint).parent
            return [
                str(
                    run_dir
                    / INFO_DIR_WITH_B
                    / "validation-runs_stride-24_experiment-magnetic_ablation_error_vs_local_frame.json"
                )
            ]
    return []


def apply_provenance_metadata(metadata: Dict, provenance: Dict | None) -> Dict:
    metadata = dict(metadata)
    if not provenance:
        return metadata
    metadata["source_checkpoint"] = provenance["source_checkpoint"]
    metadata["source_checkpoint_epoch"] = int(provenance["source_checkpoint_epoch"])
    files = provenance.get("source_files")
    if files is None:
        files = infer_source_files(metadata)
    if files:
        metadata["source_files"] = [str(path) for path in files]
    return metadata


def manifest_figure_entry(
    figure: str,
    stem: str,
    data_dir: Path,
    metadata: Dict,
) -> Dict:
    return {
        "figure": figure,
        "source_checkpoint": metadata.get("source_checkpoint"),
        "source_checkpoint_epoch": metadata.get("source_checkpoint_epoch"),
        "source_data": str(data_dir / f"{stem}.json"),
        "source_files": list(metadata.get("source_files") or infer_source_files(metadata)),
    }


def read_checkpoint_epoch(checkpoint_path: Path, require_final: bool = True) -> int:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    (validate_final_checkpoint if require_final else validate_checkpoint_split)(ckpt)
    epoch = ckpt.get("epoch")
    if epoch is None:
        raise RuntimeError(f"Checkpoint {checkpoint_path} has no epoch metadata.")
    return int(epoch)


def provenance_for_checkpoint(checkpoint_path: Path, checkpoint_epoch: int) -> Dict:
    return {
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_epoch": int(checkpoint_epoch),
    }


def expected_validation_row_names(experiment: str) -> Tuple[str, ...]:
    json_name = VAL_JSON_EXPERIMENT.get(experiment, experiment)
    if json_name == "magnetic_ablation":
        return EXPECTED_MAGNETIC_ROW_NAMES
    if json_name == "density_forecast":
        return EXPECTED_FORECAST_ROW_NAMES
    if json_name == "density_superres":
        return EXPECTED_SUPERRES_ROW_NAMES
    return ()


def source_dir_matches_b_condition(path: Path, hide_magnetic: bool) -> bool:
    parent = path.parent.name
    if hide_magnetic:
        return parent == INFO_DIR_NO_B
    return parent == INFO_DIR_WITH_B


def recorded_checkpoint_identity_error(
    recorded_checkpoint,
    recorded_epoch,
    checkpoint_path: Path,
    checkpoint_epoch: int,
    *,
    require_epoch: bool,
) -> str | None:
    recorded_path = Path(str(recorded_checkpoint or ""))
    if recorded_path.name != checkpoint_path.name:
        return (
            f"checkpoint name {recorded_path.name or '<missing>'} "
            f"!= {checkpoint_path.name}"
        )
    if recorded_epoch is None:
        if require_epoch:
            return "provenance insufficient (missing checkpoint_epoch)"
        try:
            resolved = recorded_path.expanduser().resolve()
        except Exception:
            resolved = recorded_path
        if resolved != checkpoint_path.resolve():
            return (
                "missing checkpoint_epoch and recorded checkpoint path "
                f"{resolved} != {checkpoint_path.resolve()}"
            )
        return None
    if int(recorded_epoch) != int(checkpoint_epoch):
        return f"checkpoint_epoch {recorded_epoch} != {checkpoint_epoch}"
    return None


def sliding_provenance_error(
    metrics: Dict,
    checkpoint_path: Path,
    checkpoint_epoch: int,
) -> str | None:
    return recorded_checkpoint_identity_error(
        metrics.get("checkpoint"),
        metrics.get("checkpoint_epoch"),
        checkpoint_path,
        checkpoint_epoch,
        require_epoch=False,
    )


def _val_json_path(run_dir: Path, hide_magnetic: bool, experiment: str) -> Path:
    json_experiment = VAL_JSON_EXPERIMENT.get(experiment, experiment)
    folder = INFO_DIR_NO_B if hide_magnetic else INFO_DIR_WITH_B
    return (
        run_dir
        / folder
        / f"validation-runs_stride-24_experiment-{json_experiment}_error_vs_local_frame.json"
    )


def try_load_validation_json(
    run_dir: Path,
    checkpoint_path: Path,
    experiment: str,
    hide_magnetic: bool = False,
    checkpoint_epoch: int | None = None,
) -> Dict | None:
    path = _val_json_path(run_dir, hide_magnetic, experiment)
    if not path.exists():
        return None
    if not source_dir_matches_b_condition(path, hide_magnetic):
        print(
            f"Ignoring {path}: source directory does not match "
            f"{'B hidden' if hide_magnetic else 'B visible'} condition"
        )
        return None
    if path.stat().st_mtime_ns < checkpoint_path.stat().st_mtime_ns:
        print(f"Ignoring {path}: older than checkpoint {checkpoint_path.name}")
        return None
    payload = json.loads(path.read_text())
    if "hide_magnetic" in payload and bool(payload["hide_magnetic"]) != bool(hide_magnetic):
        print(
            f"Ignoring {path}: hide_magnetic={payload['hide_magnetic']} "
            f"!= expected {hide_magnetic}"
        )
        return None
    if checkpoint_epoch is None:
        print(f"Ignoring {path}: provenance insufficient (no current checkpoint epoch)")
        return None
    identity_error = recorded_checkpoint_identity_error(
        payload.get("checkpoint"),
        payload.get("checkpoint_epoch"),
        checkpoint_path,
        checkpoint_epoch,
        require_epoch=True,
    )
    if identity_error is not None:
        print(f"Ignoring {path}: {identity_error}")
        return None
    expected_rows = expected_validation_row_names(experiment)
    row_names = [str(row.get("name")) for row in payload.get("rows", [])]
    if expected_rows and set(row_names) != set(expected_rows):
        print(
            f"Ignoring {path}: row names {row_names} != expected {list(expected_rows)}"
        )
        return None
    json_experiment = VAL_JSON_EXPERIMENT.get(experiment, experiment)
    recorded_experiment = payload.get("experiment")
    if recorded_experiment not in (None, json_experiment):
        print(
            f"Ignoring {path}: experiment {recorded_experiment} != {json_experiment}"
        )
        return None
    recorded_preprocessing = payload.get(
        "jy_preprocessing", payload.get("preprocessing")
    )
    if recorded_preprocessing != JY_STATS_PREPROCESSING:
        print(
            f"Ignoring {path}: Jy preprocessing {recorded_preprocessing!r} != "
            f"{JY_STATS_PREPROCESSING!r}"
        )
        return None
    payload["_source_json"] = str(path)
    return payload


def _profiles_from_val_row(row: Dict, metric: str) -> Dict[str, List[float]]:
    key = "density_nrmse" if metric == "density" else "jy_nrmse"
    return {
        run_name: [float(value) for value in payload[key]]
        for run_name, payload in row["per_run"].items()
    }


# ---------------------------------------------------------------------------
# Inference context (loaded only when a builder needs the model)
# ---------------------------------------------------------------------------


def load_inference_context(args: argparse.Namespace) -> Dict:
    run_dir = expand_path(args.run_dir)
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    ckpt, ckpt_path = load_checkpoint(
        run_dir, args.checkpoint, device=device,
        require_final=not bool(getattr(args, "allow_incomplete_checkpoint", False)),
    )
    ckpt_args = ckpt["args"]
    uncertainty_aware = bool(ckpt_args.get("uncertainty_aware", False))
    stats = ckpt["stats"]
    h5_dir = expand_path(args.h5_dir or ckpt_args["h5_dir"])
    delta_t = int(ckpt_args.get("delta_t", ckpt_args.get("delta-t", 24)))
    dataset = VPICWindowDataset(
        h5_dir=h5_dir,
        betas=ckpt_args.get("betas", [0.2]),
        delta_t=delta_t,
        stride_t=int(ckpt_args.get("stride_t", ckpt_args.get("stride-t", 2))),
        layout="C T X Z",
        return_metadata=True,
    )
    val_runs = get_test_runs()
    train_runs = get_train_runs()
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
    jy_stats = load_or_compute_jy_training_stats(
        run_dir=run_dir,
        dataset=dataset,
        train_runs=train_runs,
        mean=mean,
        std=std,
        extent=args.extent,
        checkpoint_path=ckpt_path,
        checkpoint_epoch=ckpt.get("epoch"),
    )
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=int(ckpt_args.get("base_channels", 16)),
        channel_mults=ckpt_args.get("channel_mults", [1, 2, 4]),
        architecture=ckpt_args.get("model_version", LEGACY_MODEL_VERSION),
        use_attention=bool(ckpt_args.get("use_attention", False)),
        spatial_only_pooling=bool(ckpt_args.get("spatial_only_pooling", False)),
        uncertainty_aware=uncertainty_aware,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return {
        "run_dir": run_dir,
        "device": device,
        "ckpt": ckpt,
        "ckpt_path": ckpt_path,
        "ckpt_args": ckpt_args,
        "dataset": dataset,
        "val_runs": val_runs,
        "mean": mean,
        "std": std,
        "jy_stats": jy_stats,
        "model": model,
        "delta_t": delta_t,
        "h5_dir": h5_dir,
    }


def select_named_window(ctx: Dict, args: argparse.Namespace):
    dataset_idx = select_run_t0_index(
        dataset=ctx["dataset"],
        val_runs=ctx["val_runs"],
        run_name=args.run_name,
        t0=args.t0,
    )
    sample = ctx["dataset"][dataset_idx]
    y = sample["block"].unsqueeze(0).to(ctx["device"])
    return sample, y


def collect_experiment_statistics(
    ctx: Dict,
    args: argparse.Namespace,
    canonical_rows: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    statistics_dataset = VPICWindowDataset(
        h5_dir=ctx["h5_dir"],
        betas=ctx["ckpt_args"].get("betas", [0.2]),
        delta_t=ctx["delta_t"],
        stride_t=1,
        layout="C T X Z",
        return_metadata=True,
    )
    indices = select_validation_statistics_indices(
        dataset=statistics_dataset,
        val_runs=ctx["val_runs"],
        window_stride=ctx["delta_t"],
        max_windows_per_run=None,
    )
    collect_args = SimpleNamespace(extent=args.extent)
    stats = collect_validation_statistics(
        model=ctx["model"],
        dataset=statistics_dataset,
        sample_indices=indices,
        args=collect_args,
        mean=ctx["mean"],
        std=ctx["std"],
        device=ctx["device"],
        canonical_rows=canonical_rows,
        jy_std_train=float(ctx["jy_stats"]["jy_std_train"]),
    )
    statistics_dataset.close()
    return stats


def row_dicts_from_masks(mask_rows: Sequence[Tuple[str, str, torch.Tensor]]) -> List[Dict]:
    return [
        {
            "name": name,
            "label": label,
            "mask": mask[0].detach().cpu().numpy(),
        }
        for name, label, mask in mask_rows
    ]


# ---------------------------------------------------------------------------
# Figure A: spatial qualitative
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_spatial_qualitative_cache(ctx: Dict, args: argparse.Namespace) -> Tuple[Dict, Dict]:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    sample, y = select_named_window(ctx, args)
    metadata = sample["metadata"]
    local_time = local_index_for_global_frame(args.t0, args.global_frame, ctx["delta_t"])
    mean, std, device = ctx["mean"], ctx["std"], ctx["device"]
    y_norm = normalize(y, mean, std)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    all_rows = build_density_only_multifunction_rows(
        block=y_norm,
        patterns=[
            "spatial_random",
            "spatial_grid",
            "spatial_block",
            "temporal_random",
            "temporal_block",
        ],
        mask_fraction=0.8,
        block_fraction=0.5,
        grid_stride=4,
        magnetic_grid_stride=2,
        generator=generator,
        magnetic_visible=True,
    )
    by_name = {name: (label, mask) for name, label, mask in all_rows}
    target_plot = y[0, 3].detach().cpu().numpy()
    target_norm = y_norm[0, 3].detach().cpu().numpy()
    arrays: Dict[str, np.ndarray] = {
        "target_density": np.asarray(target_plot[local_time], dtype=np.float32),
    }
    row_meta = []
    for name in SPATIAL_ROW_ORDER:
        _label, mask = by_name[name]
        pred_norm, pred_variance = predict_masked_fields(ctx["model"], y_norm, mask)
        if pred_variance is None:
            raise RuntimeError("Journal uncertainty panels require an uncertainty-aware checkpoint.")
        pred_plot = denormalize(pred_norm, mean, std)
        density_mask = mask[0, 3].detach().cpu().numpy()
        visible = np.array(target_plot[local_time], copy=True)
        visible[density_mask[local_time] < 0.5] = np.nan
        # Raw model output over the full field, including visible Density sites.
        pred = pred_plot[0, 3, local_time].detach().cpu().numpy()
        residual = normalized_residual(
            pred_norm[0, 3, local_time].detach().cpu().numpy(),
            target_norm[local_time],
        )
        squared_error = np.square(residual, dtype=np.float32)
        variance = pred_variance[0, 3, local_time].float().detach().cpu().numpy()
        unknown = density_mask[local_time] < 0.5
        corr = pearson_spearman(variance[unknown], squared_error[unknown])
        bins = quantile_calibration(variance[unknown], squared_error[unknown], 10)
        bias = calibration_bias(float(np.mean(squared_error[unknown])), float(np.mean(variance[unknown])))
        arrays[f"{name}_visible"] = np.asarray(visible, dtype=np.float32)
        arrays[f"{name}_prediction"] = np.asarray(pred, dtype=np.float32)
        arrays[f"{name}_actual_squared_error"] = squared_error
        arrays[f"{name}_predicted_variance"] = np.asarray(variance, dtype=np.float32)
        arrays[f"{name}_mask"] = np.asarray(density_mask[local_time], dtype=np.float32)
        row_meta.append(
            {
                "name": name,
                "density_masked_fraction": float(1.0 - density_mask.mean()),
                "density_nrmse": density_nrmse_from_residual(residual),
                **corr,
                "signed_bias": bias["signed_bias"],
                "calibration_gap": bias["calibration_gap"],
                "calibration_bins": bins,
                "uncertainty_scope": "unobserved Density voxels at displayed frame",
            }
        )
    meta = {
        "figure": "spatial_qualitative",
        "run_name": args.run_name,
        "t0": int(args.t0),
        "global_frame": int(args.global_frame),
        "local_time": int(local_time),
        "delta_t": int(ctx["delta_t"]),
        "row_names": list(SPATIAL_ROW_ORDER),
        "rows": row_meta,
        "b_condition": "B fully observed",
        "density_policy": "standardized ~50% multifunction geometries",
        "prediction_definition": (
            "raw model Density prediction over the full field, including visible sites"
        ),
        "residual_definition": DENSITY_NRMSE_DEFINITION,
        "residual_scope": (
            "full-field prediction_normalized - target_normalized, including visible sites"
        ),
        "sample_metadata": dict(metadata),
        "layout": {"rows": 3, "columns": 5},
        "uncertainty_units": "checkpoint-standardized Density squared units",
    }
    return arrays, meta


def plot_spatial_qualitative(
    arrays: Dict,
    metadata: Dict,
    figures_dir: Path,
    extent: Sequence[float],
    residual_vmax: float,
    field_q: float,
    dpi: int,
) -> Tuple[int, int]:
    row_names = [str(name) for name in metadata["row_names"]]
    n_rows, n_cols = 3, 5
    if len(row_names) != n_rows:
        raise ValueError(f"Expected {n_rows} spatial rows, got {row_names}")
    field_arrays = [arrays["target_density"]]
    error_arrays = []
    for name in row_names:
        field_arrays.extend([arrays[f"{name}_visible"], arrays[f"{name}_prediction"]])
        error_arrays.extend([arrays[f"{name}_actual_squared_error"], arrays[f"{name}_predicted_variance"]])
    vmin, vmax = robust_limits(field_arrays, channel=3, q=field_q, symmetric=False)
    error_vmax = float(np.nanpercentile(np.concatenate([a.ravel() for a in error_arrays]), 99.0))
    field_cmap = make_nan_cmap("magma")
    error_cmap = make_nan_cmap("viridis")

    fig = plt.figure(figsize=(10.2, 6.3))
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        wspace=0.16,
        hspace=0.08,
        left=0.22,
        right=0.84,
        top=0.92,
        bottom=0.11,
    )
    col_titles = ["Target Density", "Visible Density", "Prediction", "Actual squared error", "Predicted variance"]
    nrmse_by_name = {
        str(row.get("name")): row.get("density_nrmse")
        for row in metadata.get("rows") or []
        if isinstance(row, dict)
    }
    field_im = None
    residual_im = None
    base_ax = None
    for r, name in enumerate(row_names):
        panels = [
            arrays["target_density"],
            arrays[f"{name}_visible"],
            arrays[f"{name}_prediction"],
            arrays[f"{name}_actual_squared_error"],
            arrays[f"{name}_predicted_variance"],
        ]
        for c, panel in enumerate(panels):
            if base_ax is None:
                ax = fig.add_subplot(gs[r, c])
                base_ax = ax
            else:
                ax = fig.add_subplot(gs[r, c], sharex=base_ax, sharey=base_ax)
            if c < 3:
                im = ax.imshow(
                    panel,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap=field_cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
                field_im = im
            else:
                im = ax.imshow(
                    panel,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap=error_cmap,
                    vmin=0.0,
                    vmax=error_vmax,
                    interpolation="nearest",
                )
                residual_im = im
                if c == 3:
                    nrmse = nrmse_by_name.get(name)
                    if nrmse is None:
                        nrmse = float(np.sqrt(np.nanmean(panel)))
                    annotate_density_nrmse(ax, float(nrmse))
            if r == 0:
                ax.set_title(col_titles[c], pad=4)
            if c == 0:
                ax.text(
                    -0.3,
                    0.5,
                    SPATIAL_ROW_LABELS[name].replace("\n", " "),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=8,
                    linespacing=1.05,
                )
            _style_comparison_axis(ax, r, c, n_rows)
            ax.tick_params(axis="both", which="both", labelsize=6)

    fig.text(0.53, 0.08, "z [cm]",
         ha="center", va="center", fontsize=9)
    fig.text(0.15, 0.52, "x [cm]",
            ha="center", va="center", rotation=90, fontsize=9)

    cax_field = fig.add_axes([0.86, 0.56, 0.018, 0.36])
    cax_res = fig.add_axes([0.86, 0.12, 0.018, 0.36])
    cb_field = fig.colorbar(field_im, cax=cax_field)
    cb_field.set_label("Density", fontsize=8)
    cb_res = fig.colorbar(residual_im, cax=cax_res)
    cb_res.set_label("Standardized squared units", fontsize=8)
    fig.suptitle(
        "B fully observed  ·  Density ~50% masked  ·  "
        f"global frame {metadata['global_frame']}",
        fontsize=10,
        y=0.965,
    )
    save_png_pdf(fig, figures_dir, FIGURE_STEMS["spatial"], dpi)
    return n_rows, n_cols


def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def plot_spatial_calibration(metadata: Dict, figures_dir: Path, data_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    csv_rows = []
    for row in metadata["rows"]:
        bins = row["calibration_bins"]
        x = [item["mean_predicted_variance"] for item in bins]
        y = [item["mean_actual_squared_error"] for item in bins]
        ax.plot(x, y, marker="o", label=SPATIAL_ROW_LABELS[row["name"]].replace("\n", " "))
        csv_rows.extend({"task": row["name"], **item} for item in bins)
    limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, limit], [0, limit], "k--", linewidth=1, label="y=x")
    ax.set(xlabel="Mean predicted variance (standardized²)", ylabel="Mean actual squared error (standardized²)")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(alpha=.25)
    save_png_pdf(fig, figures_dir, "spatial_uncertainty_calibration", dpi)
    write_csv(data_dir / "spatial_uncertainty_calibration.csv", csv_rows)


_RUN_PARAMETERS = re.compile(
    r"^beta(?P<beta>[^_]+)_nu(?P<nu>[^_]+)_Bz(?P<Bz0>[^_]+)_dt(?P<dt>[^_]+)_tau(?P<tau>[^_]+)$"
)


def parse_run_parameters(run_name: str) -> Dict[str, float]:
    match = _RUN_PARAMETERS.match(str(run_name))
    if match is None:
        raise ValueError(f"Cannot parse physical parameters from run name {run_name!r}")
    return {key: float(value) for key, value in match.groupdict().items()}


def _bounded_pair_sample(variance: np.ndarray, e2: np.ndarray, limit: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    variance, e2 = np.asarray(variance).ravel(), np.asarray(e2).ravel()
    keep = np.isfinite(variance) & np.isfinite(e2)
    variance, e2 = variance[keep], e2[keep]
    if variance.size > limit:
        indices = np.random.default_rng(seed).choice(variance.size, limit, replace=False)
        variance, e2 = variance[indices], e2[indices]
    return variance.astype(np.float32), e2.astype(np.float32)


def _journal_mask_rows(ctx: Dict, args: argparse.Namespace) -> List[Dict]:
    _sample, y = select_named_window(ctx, args)
    generator = torch.Generator().manual_seed(args.seed)
    rows = []
    spatial = build_density_only_multifunction_rows(
        y, ["spatial_random", "spatial_grid", "spatial_block", "temporal_random", "temporal_block"],
        .8, .5, 4, 2, generator, magnetic_visible=True,
    )
    for name, label, mask in spatial:
        if name in {*SPATIAL_ROW_ORDER, "temporal_random", "temporal_block"}:
            rows.append({"family": "standard", "name": name, "label": label, "mask": mask[0].cpu().numpy()})
    for b_hidden in (False, True):
        suffix = "B_hidden" if b_hidden else "B_full"
        for name, label, mask in build_density_forecast_rows(y, default_density_forecast_visible_frames(ctx["delta_t"]), not b_hidden):
            rows.append({"family": "forecast", "name": f"{name}_{suffix}", "label": label, "mask": mask[0].cpu().numpy()})
        for name, label, mask in build_density_superres_rows(y, args.density_probe_counts, not b_hidden):
            rows.append({"family": "superres", "name": f"{name}_{suffix}", "label": label, "mask": mask[0].cpu().numpy()})
    magnetic = build_magnetic_ablation_rows(y, DEFAULT_MAGNETIC_ABLATION_VISIBLE_FRACTIONS, generator)
    for density_full in (False, True):
        condition = "density_full" if density_full else "density_hidden"
        for name, label, mask in magnetic:
            mask = mask.clone()
            mask[:, 3] = 1.0 if density_full else 0.0
            rows.append({"family": "magnetic_ablation", "name": f"{name}_{condition}", "label": label, "mask": mask[0].cpu().numpy()})
    return rows


@torch.no_grad()
def build_journal_statistics_cache(ctx: Dict, args: argparse.Namespace) -> Tuple[Dict, Dict]:
    """Shared test-set pass for Figures 2/3/4/6/7; statistical unit is a run."""
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    if not bool(ctx["ckpt_args"].get("uncertainty_aware", False)):
        raise RuntimeError("Journal uncertainty evaluation requires an uncertainty-aware checkpoint.")
    dataset = VPICWindowDataset(h5_dir=ctx["h5_dir"], betas=ctx["ckpt_args"].get("betas", [.2]),
        delta_t=ctx["delta_t"], stride_t=1, layout="C T X Z", return_metadata=True)
    indices = select_validation_statistics_indices(dataset, set(ctx["val_runs"]), ctx["delta_t"], args.statistics_max_windows_per_run)
    canonical = _journal_mask_rows(ctx, args)
    by_run: Dict[str, Dict[str, Dict]] = {}
    sample_limit = max(32, int(args.uq_voxels_per_window))
    for ordinal, index in enumerate(indices):
        sample = dataset[index]; run_name = sample["metadata"]["run_name"]
        target = sample["block"].unsqueeze(0).to(ctx["device"])
        target_norm = normalize(target, ctx["mean"], ctx["std"])
        target_np = target_norm[0].cpu().numpy(); target_phys = target[0].cpu().numpy()
        target_jy = compute_jy(target_phys, args.extent)
        run = by_run.setdefault(run_name, {})
        for row_index, row in enumerate(canonical):
            mask = torch.as_tensor(row["mask"], device=ctx["device"], dtype=target_norm.dtype).unsqueeze(0)
            prediction, variance = predict_masked_fields(ctx["model"], target_norm, mask)
            pred = prediction[0].float().cpu().numpy(); var = variance[0].float().cpu().numpy()
            residual = pred - target_np; unknown = np.asarray(row["mask"]) < .5
            density_nrmse = float(np.sqrt(np.mean(residual[3] ** 2)))
            prediction_jy = compute_physical_jy_from_normalized(pred, ctx["mean"], ctx["std"], args.extent)
            jy_nrmse = float(jy_nrmse_from_residual(prediction_jy - target_jy, float(ctx["jy_stats"]["jy_std_train"])).mean())
            entry = run.setdefault(row["name"], {"family": row["family"], "density_nrmse": [], "jy_nrmse": []})
            entry["density_nrmse"].append(density_nrmse); entry["jy_nrmse"].append(jy_nrmse)
            for prefix, channels in (("density", [3]), ("bx", [0]), ("by", [1]), ("bz", [2])):
                choose = predicted_voxel_mask(row["mask"], channels)
                if not np.any(choose):
                    continue
                entry.setdefault(f"{prefix}_e2", [])
                entry.setdefault(f"{prefix}_var", [])
                entry.setdefault(f"{prefix}_sum_e2", 0.0)
                entry.setdefault(f"{prefix}_sum_var", 0.0)
                entry.setdefault(f"{prefix}_count", 0)
                exact_v = var[channels][choose]; exact_e = residual[channels][choose] ** 2
                entry[f"{prefix}_sum_var"] += float(np.sum(exact_v, dtype=np.float64))
                entry[f"{prefix}_sum_e2"] += float(np.sum(exact_e, dtype=np.float64))
                entry[f"{prefix}_count"] += int(exact_v.size)
                v, e = _bounded_pair_sample(exact_v, exact_e,
                    min(256, sample_limit), args.seed + ordinal * 1009 + row_index * 31 + channels[0])
                entry[f"{prefix}_var"].append(v); entry[f"{prefix}_e2"].append(e)
        if ordinal == 0 or (ordinal + 1) % 25 == 0 or ordinal + 1 == len(indices):
            print(f"Journal statistics: {ordinal + 1}/{len(indices)} windows")
    dataset.close()
    records = []
    prefixes = ("density", "bx", "by", "bz")
    global_samples = {
        f"{prefix}_{kind}": []
        for prefix in (*prefixes, "b")
        for kind in ("variance", "e2")
    }
    for run_name, tasks in sorted(by_run.items()):
        params = parse_run_parameters(run_name)
        for task, entry in tasks.items():
            record = {"run_name": run_name, "task": task, "family": entry["family"], **params,
                "density_nrmse": float(np.mean(entry["density_nrmse"])), "jy_nrmse": float(np.mean(entry["jy_nrmse"]))}
            samples = {}
            for prefix_index, prefix in enumerate(prefixes):
                count = int(entry.get(f"{prefix}_count", 0))
                if count:
                    variance_sample, e2_sample = _bounded_pair_sample(
                        np.concatenate(entry.pop(f"{prefix}_var")), np.concatenate(entry.pop(f"{prefix}_e2")),
                        sample_limit, args.seed + len(records) * 101 + prefix_index,
                    )
                    corr = pearson_spearman(variance_sample, e2_sample)
                    mean_e2 = entry[f"{prefix}_sum_e2"] / count
                    mean_variance = entry[f"{prefix}_sum_var"] / count
                    bias = calibration_bias(mean_e2, mean_variance)
                else:
                    variance_sample = e2_sample = np.empty(0, dtype=np.float32)
                    corr = {"pearson": float("nan"), "spearman": float("nan")}
                    mean_e2 = mean_variance = float("nan")
                    bias = {"signed_bias": float("nan"), "calibration_gap": float("nan")}
                record.update({
                    f"{prefix}_mean_e2": mean_e2, f"{prefix}_mean_variance": mean_variance,
                    f"{prefix}_signed_bias": bias["signed_bias"], f"{prefix}_calibration_gap": bias["calibration_gap"],
                    f"{prefix}_pearson": corr["pearson"], f"{prefix}_spearman": corr["spearman"],
                    f"{prefix}_predicted_voxel_count": count, f"sampled_{prefix}_voxels": int(variance_sample.size),
                })
                samples[prefix] = (variance_sample, e2_sample)
            available_b = [samples[prefix] for prefix in ("bx", "by", "bz") if samples[prefix][0].size]
            if available_b:
                b_variance = np.concatenate([pair[0] for pair in available_b])
                b_e2 = np.concatenate([pair[1] for pair in available_b])
                b_corr = pearson_spearman(b_variance, b_e2)
                b_count = sum(int(record[f"{prefix}_predicted_voxel_count"]) for prefix in ("bx", "by", "bz"))
                b_mean_e2 = sum(entry.get(f"{prefix}_sum_e2", 0.0) for prefix in ("bx", "by", "bz")) / b_count
                b_mean_variance = sum(entry.get(f"{prefix}_sum_var", 0.0) for prefix in ("bx", "by", "bz")) / b_count
                b_bias = calibration_bias(b_mean_e2, b_mean_variance)
            else:
                b_variance = b_e2 = np.empty(0, dtype=np.float32); b_count = 0
                b_corr = {"pearson": float("nan"), "spearman": float("nan")}
                b_mean_e2 = b_mean_variance = float("nan")
                b_bias = {"signed_bias": float("nan"), "calibration_gap": float("nan")}
            record.update({"b_mean_e2": b_mean_e2, "b_mean_variance": b_mean_variance,
                "b_signed_bias": b_bias["signed_bias"], "b_calibration_gap": b_bias["calibration_gap"],
                "b_pearson": b_corr["pearson"], "b_spearman": b_corr["spearman"],
                "b_predicted_voxel_count": b_count, "sampled_b_voxels": int(b_variance.size)})
            records.append(record)
            for prefix, (variance_sample, e2_sample) in {**samples, "b": (b_variance, b_e2)}.items():
                if not variance_sample.size:
                    continue
                vg, eg = _bounded_pair_sample(variance_sample, e2_sample, min(512, sample_limit), args.seed + len(records) * 29 + prefixes.index(prefix) if prefix in prefixes else args.seed + len(records) * 29 + 4)
                global_samples[f"{prefix}_variance"].append(vg); global_samples[f"{prefix}_e2"].append(eg)
    sample_arrays = {
        f"global::{key}": np.concatenate(values)
        for key, values in global_samples.items() if values
    }
    metadata = {"figure": "journal_statistics", "records": records, "test_run_count": len(by_run),
        "window_count": len(indices), "sampling_seed": int(args.seed), "max_sampled_voxels_per_run_task_channel_group": sample_limit,
        "per_window_sampling_cap": min(256, sample_limit), "global_calibration_sample_per_run_task_cap": min(512, sample_limit),
        "units": "checkpoint-standardized squared units", "aggregation": "windows within run, then runs",
        "uncertainty_domain": "only mask==0 predicted/unknown voxels; exact observations excluded",
        "magnetic_uncertainty": "Bx, By, Bz separately plus optional macro aggregate; never Jy",
        "model_version": ctx["ckpt_args"].get("model_version"), "split": fixed_split_metadata()}
    return sample_arrays, metadata


def _summary_rows(records: Sequence[Dict], group_key: str, metrics: Sequence[str]) -> List[Dict]:
    grouped = {}
    for record in records:
        grouped.setdefault(record[group_key], []).append(record)
    rows = []
    for group, members in sorted(grouped.items()):
        for metric, stats in summarize_run_records(members, metrics).items():
            rows.append({group_key: group, "metric": metric, **stats})
    return rows


def uncertainty_condition_label(family: str, task: str) -> str:
    if family == "magnetic_ablation":
        return "Density fully visible" if task.endswith("density_full") else "Density fully hidden"
    if family in {"superres", "forecast"}:
        return "B hidden" if task.endswith("B_hidden") else "B visible"
    raise ValueError(f"No uncertainty condition label for family={family!r}, task={task!r}")


def plot_uncertainty_companions(records: Sequence[Dict], figures_dir: Path, data_dir: Path, dpi: int) -> None:
    specifications = (
        ("magnetic_ablation", "magnetic_ablation_uncertainty", "B visible level", lambda r: float(r["task"].split("_")[2]),
         lambda r: uncertainty_condition_label("magnetic_ablation", r["task"])),
        ("superres", "density_superres_uncertainty", "Density probe count", lambda r: float(r["task"].split("_")[2]),
         lambda r: uncertainty_condition_label("superres", r["task"])),
        ("forecast", "conditional_temporal_uncertainty", "Missing future frames", lambda r: 24 - int(r["task"].split("_")[3]),
         lambda r: uncertainty_condition_label("forecast", r["task"])),
    )
    for family, stem, xlabel, x_fn, condition_fn in specifications:
        selected = [dict(r, x=x_fn(r), condition=condition_fn(r)) for r in records if r["family"] == family]
        if not selected:
            continue
        combos = sorted({(r["x"], r["condition"]) for r in selected}, key=lambda item: (item[1], item[0]))
        summary_rows = []
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), sharey=False)
        for condition in sorted({r["condition"] for r in selected}):
            x_values = sorted({r["x"] for r in selected if r["condition"] == condition})
            for ax, prefix, title in ((axes[0], "density", "Density"), (axes[1], "b", "Direct B macro-average")):
                available_for_condition = any(
                    np.isfinite(r.get(f"{prefix}_mean_e2", np.nan))
                    for r in selected if r["condition"] == condition
                )
                if not available_for_condition:
                    continue
                for field, linestyle, label in (("mean_e2", "-", "actual e²"), ("mean_variance", "--", "predicted variance")):
                    mids=[]; lows=[]; highs=[]
                    for x in x_values:
                        members=[r for r in selected if r["condition"] == condition and r["x"] == x]
                        stats=summarize(r[f"{prefix}_{field}"] for r in members)
                        mids.append(stats["median"]); lows.append(stats["p16"]); highs.append(stats["p84"])
                        summary_rows.append({"family":family,"condition":condition,"x":x,"channel_group":prefix,"quantity":field,**stats})
                    line=ax.plot(x_values,mids,linestyle=linestyle,marker="o",label=f"{condition}: {label}")[0]
                    ax.fill_between(x_values,lows,highs,color=line.get_color(),alpha=.12)
                for metric in ("signed_bias", "calibration_gap", "pearson", "spearman"):
                    for x in x_values:
                        members=[r for r in selected if r["condition"] == condition and r["x"] == x]
                        summary_rows.append({"family":family,"condition":condition,"x":x,"channel_group":prefix,
                            "quantity":metric, **summarize(r.get(f"{prefix}_{metric}", np.nan) for r in members)})
                ax.set_title(title); ax.set_xlabel(xlabel); ax.grid(alpha=.25)
        axes[0].set_ylabel("Standardized squared units")
        axes[1].legend(frameon=False,fontsize=6)
        fig.tight_layout(); save_png_pdf(fig, figures_dir, stem, dpi)
        write_csv(data_dir / f"{stem}.csv", summary_rows)


def build_parameter_summary(records: Sequence[Dict]) -> List[Dict]:
    spatial = [r for r in records if r["task"] in SPATIAL_ROW_ORDER]
    per_run = {}
    for record in spatial:
        item = per_run.setdefault(record["run_name"], {k: record[k] for k in ("run_name", "beta", "nu", "Bz0", "tau")})
        item.setdefault("density_values", []).append(record["density_nrmse"])
        item.setdefault("jy_values", []).append(record["jy_nrmse"])
    run_records=[]
    for item in per_run.values():
        item["density_nrmse"] = float(np.mean(item.pop("density_values")))
        item["jy_nrmse"] = float(np.mean(item.pop("jy_values")))
        run_records.append(item)
    return [group for parameter in ("beta", "nu", "Bz0", "tau") for group in group_run_records(run_records, parameter, ("density_nrmse", "jy_nrmse"))]


def plot_parameter_summary(groups: Sequence[Dict], figures_dir: Path, data_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8)); csv_rows=[]
    for ax, parameter in zip(axes.ravel(), ("beta", "nu", "Bz0", "tau")):
        selected=[g for g in groups if g["parameter"] == parameter]
        x=np.arange(len(selected)); labels=[f"{g['value']:g}" for g in selected]
        for metric, marker, label in (("density_nrmse","o","Density NRMSE"),("jy_nrmse","s","Jy NRMSE")):
            mid=np.array([g["metrics"][metric]["median"] for g in selected]); lo=np.array([g["metrics"][metric]["p16"] for g in selected]); hi=np.array([g["metrics"][metric]["p84"] for g in selected])
            ax.errorbar(x,mid,yerr=[mid-lo,hi-mid],marker=marker,capsize=2,label=label)
            for g in selected: csv_rows.append({"parameter":parameter,"value":g["value"],"metric":metric,**g["metrics"][metric]})
        ax.set_xticks(x,labels); ax.set_xlabel(parameter); ax.set_ylabel("NRMSE"); ax.grid(alpha=.25)
    axes[0,0].legend(frameon=False); fig.suptitle("Marginal grouped performance summaries (three spatial masks)")
    fig.tight_layout(); save_png_pdf(fig, figures_dir, FIGURE_STEMS["parameters"], dpi)
    write_csv(data_dir / f"{FIGURE_STEMS['parameters']}.csv", csv_rows)


def _global_task_records(records: Sequence[Dict]) -> List[Dict]:
    wanted = set(SPATIAL_ROW_ORDER) | {"temporal_random", "temporal_block"}
    direct = [dict(r, task_family=r["task"]) for r in records if r["task"] in wanted]
    forecasts = [r for r in records if r["family"] == "forecast"]
    for run_name in sorted({r["run_name"] for r in forecasts}):
        members=[r for r in forecasts if r["run_name"] == run_name]
        base={"run_name":run_name,"task_family":"conditional_temporal_completion"}
        for prefix in ("density", "bx", "by", "bz", "b"):
            for suffix in ("pearson", "spearman", "mean_e2", "mean_variance", "signed_bias", "calibration_gap"):
                key=f"{prefix}_{suffix}"
                values=np.asarray([r.get(key, np.nan) for r in members], dtype=np.float64)
                finite=values[np.isfinite(values)]
                base[key]=float(np.mean(finite)) if finite.size else float("nan")
        direct.append(base)
    return direct


def plot_global_uncertainty(arrays: Dict, records: Sequence[Dict], figures_dir: Path, data_dir: Path, dpi: int) -> Dict:
    # Equal-cap sampling was already performed deterministically per run/window/task.
    labels={"density":"Density","bx":"Bx","by":"By","bz":"Bz","b":"B macro"}
    calibrations={}
    for prefix,label in labels.items():
        variance=arrays.get(f"global::{prefix}_variance");e2=arrays.get(f"global::{prefix}_e2")
        if variance is not None and e2 is not None:
            calibrations[label]=quantile_calibration(variance,e2,10)
    fig,ax=plt.subplots(figsize=(4.3,3.8)); calibration_rows=[]
    for label,bins in calibrations.items():
        x=[b["mean_predicted_variance"] for b in bins]; y=[b["mean_actual_squared_error"] for b in bins]
        ax.plot(x,y,marker="o",label=label); calibration_rows.extend({"channel_group":label,**b} for b in bins)
    limit=max(ax.get_xlim()[1],ax.get_ylim()[1]); ax.plot([0,limit],[0,limit],"k--",label="y=x")
    ax.set(xlabel="Mean predicted variance (standardized²)",ylabel="Mean actual squared error (standardized²)"); ax.grid(alpha=.25); ax.legend(frameon=False)
    save_png_pdf(fig,figures_dir,"global_uncertainty_calibration",dpi); write_csv(data_dir/"global_uncertainty_calibration.csv",calibration_rows)
    task_records=_global_task_records(records); tasks=sorted({r["task_family"] for r in task_records}); x=np.arange(len(tasks))
    detailed_rows=[]
    for task in sorted({r["task"] for r in records}):
        members=[r for r in records if r["task"]==task]
        for prefix,label in (("density","Density"),("bx","Bx"),("by","By"),("bz","Bz"),("b","B macro")):
            if not any(int(r.get(f"{prefix}_predicted_voxel_count",0)) > 0 for r in members):
                continue
            for metric in ("pearson","spearman","mean_e2","mean_variance","signed_bias","calibration_gap"):
                detailed_rows.append({"task":task,"family":members[0]["family"],"channel":label,"metric":metric,
                    **summarize(r.get(f"{prefix}_{metric}",np.nan) for r in members)})
    write_csv(data_dir/"uq_task_channel_summary.csv",detailed_rows)
    summary_rows=[]; fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,suffix,title in ((axes[0],"pearson","Pearson"),(axes[1],"spearman","Spearman"),(axes[2],"calibration_gap","Calibration gap")):
        for offset,(prefix,label,marker) in zip((-.08,.08),(('density','Density','o'),('b','B macro','s'))):
            metric=f"{prefix}_{suffix}";mids=[];lo=[];hi=[]
            for task in tasks:
                stats=summarize(r[metric] for r in task_records if r["task_family"]==task);mids.append(stats["median"]);lo.append(stats["p16"]);hi.append(stats["p84"]);summary_rows.append({"task":task,"channel_group":label,"metric":metric,**stats})
            mids=np.array(mids);lo=np.array(lo);hi=np.array(hi);finite=np.isfinite(mids)
            if np.any(finite):
                ax.errorbar(x[finite]+offset,mids[finite],yerr=[mids[finite]-lo[finite],hi[finite]-mids[finite]],fmt=marker,capsize=2,label=label)
        ax.set_title(title);ax.set_xticks(x,[t.replace("_","\n") for t in tasks],rotation=25,ha="right");ax.grid(alpha=.25)
    axes[0].legend(frameon=False,fontsize=7)
    fig.tight_layout();save_png_pdf(fig,figures_dir,"global_uncertainty_task_summary",dpi);write_csv(data_dir/"global_uncertainty_task_summary.csv",summary_rows)

    channel_rows=[]; fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,suffix,title in ((axes[0],"pearson","Pearson"),(axes[1],"spearman","Spearman"),(axes[2],"signed_bias","Signed bias (e² − variance)")):
        for offset,(prefix,label,marker) in zip((-.16,0,.16),(('bx','Bx','o'),('by','By','s'),('bz','Bz','^'))):
            mids=[];los=[];his=[]
            for task in tasks:
                stats=summarize(r.get(f"{prefix}_{suffix}",np.nan) for r in task_records if r["task_family"]==task)
                mids.append(stats["median"]);los.append(stats["p16"]);his.append(stats["p84"])
                channel_rows.append({"task":task,"channel":label,"metric":suffix,**stats})
            mids=np.asarray(mids);los=np.asarray(los);his=np.asarray(his)
            finite=np.isfinite(mids)
            if np.any(finite):
                ax.errorbar(x[finite]+offset,mids[finite],yerr=[mids[finite]-los[finite],his[finite]-mids[finite]],fmt=marker,capsize=2,label=label)
        ax.set_title(title);ax.set_xticks(x,[t.replace("_","\n") for t in tasks],rotation=25,ha="right");ax.grid(alpha=.25)
    axes[0].legend(frameon=False,fontsize=7)
    fig.tight_layout();save_png_pdf(fig,figures_dir,"magnetic_channel_uncertainty_summary",dpi)
    write_csv(data_dir/"magnetic_channel_uncertainty_summary.csv",channel_rows)

    bias_rows=[];fig,ax=plt.subplots(figsize=(7.2,3.8))
    for offset,(prefix,label,marker) in zip((-.08,.08),(('density','Density','o'),('b','B macro','s'))):
        mids=[];los=[];his=[]
        for task in tasks:
            stats=summarize(r.get(f"{prefix}_signed_bias",np.nan) for r in task_records if r["task_family"]==task)
            mids.append(stats["median"]);los.append(stats["p16"]);his.append(stats["p84"])
            bias_rows.append({"task":task,"channel_group":label,"metric":"signed_bias",**stats})
        mids=np.asarray(mids);los=np.asarray(los);his=np.asarray(his);finite=np.isfinite(mids)
        ax.errorbar(x[finite]+offset,mids[finite],yerr=[mids[finite]-los[finite],his[finite]-mids[finite]],fmt=marker,capsize=2,label=label)
    ax.axhline(0,color="k",linestyle="--",linewidth=1);ax.set_ylabel("Mean e² − mean predicted variance");ax.set_xticks(x,[t.replace("_","\n") for t in tasks],rotation=25,ha="right");ax.grid(alpha=.25);ax.legend(frameon=False)
    fig.tight_layout();save_png_pdf(fig,figures_dir,"uncertainty_signed_bias_summary",dpi)
    write_csv(data_dir/"uncertainty_signed_bias_summary.csv",bias_rows)
    return {"calibration":calibrations,"task_summary":summary_rows,"task_channel_summary":detailed_rows,
        "magnetic_channel_summary":channel_rows,"signed_bias_summary":bias_rows}


# ---------------------------------------------------------------------------
# Quantitative summary figures from validation JSON or fresh aggregation
# ---------------------------------------------------------------------------


def _magnetic_ablation_rows_from_stats(stats: Sequence[Dict]) -> List[Dict]:
    rows = []
    for row in stats:
        rows.append(
            {
                "name": row["name"],
                "visible_fraction": parse_trailing_number(row["name"]),
                "density": summarize_run_scalars(
                    {
                        name: profile.tolist()
                        for name, profile in zip(
                            row["density"]["run_names"],
                            row["density"]["run_profiles"],
                        )
                    }
                ),
                "jy": summarize_run_scalars(
                    {
                        name: profile.tolist()
                        for name, profile in zip(
                            row["jy"]["run_names"],
                            row["jy"]["run_profiles"],
                        )
                    }
                ),
            }
        )
    rows.sort(key=lambda item: item["visible_fraction"])
    return rows


def _magnetic_ablation_rows_from_payload(payload: Dict) -> List[Dict]:
    rows = []
    for row in payload["rows"]:
        rows.append(
            {
                "name": row["name"],
                "visible_fraction": parse_trailing_number(row["name"]),
                "density": summarize_run_scalars(_profiles_from_val_row(row, "density")),
                "jy": summarize_run_scalars(_profiles_from_val_row(row, "jy")),
            }
        )
    rows.sort(key=lambda item: item["visible_fraction"])
    return rows


def _pack_magnetic_condition(rows: Sequence[Dict], prefix: str) -> Dict[str, np.ndarray]:
    return {
        f"{prefix}_density_median": np.asarray([row["density"]["median"] for row in rows]),
        f"{prefix}_density_p16": np.asarray([row["density"]["p16"] for row in rows]),
        f"{prefix}_density_p84": np.asarray([row["density"]["p84"] for row in rows]),
        f"{prefix}_jy_median": np.asarray([row["jy"]["median"] for row in rows]),
        f"{prefix}_jy_p16": np.asarray([row["jy"]["p16"] for row in rows]),
        f"{prefix}_jy_p84": np.asarray([row["jy"]["p84"] for row in rows]),
        f"{prefix}_density_per_run": np.asarray([row["density"]["per_run"] for row in rows]),
        f"{prefix}_jy_per_run": np.asarray([row["jy"]["per_run"] for row in rows]),
    }


def _magnetic_ablation_masks_with_density_full(
    mask_rows: Sequence[Tuple[str, str, torch.Tensor]],
) -> List[Tuple[str, str, torch.Tensor]]:
    filled = []
    for name, label, mask in mask_rows:
        mask_full = mask.clone()
        mask_full[:, 3:4] = 1.0
        filled.append((name, label, mask_full))
    return filled


def build_magnetic_ablation_cache(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    ctx: Dict | None,
    checkpoint_epoch: int,
) -> Tuple[Dict, Dict]:
    if ctx is None:
        raise RuntimeError(
            "Magnetic ablation paired density conditions need inference context."
        )
    _sample, y = select_named_window(ctx, args)
    generator = torch.Generator().manual_seed(args.seed)
    base_rows = build_magnetic_ablation_rows(
        block=y,
        magnetic_visible_fractions=list(DEFAULT_MAGNETIC_ABLATION_VISIBLE_FRACTIONS),
        generator=generator,
    )
    full_rows_masks = _magnetic_ablation_masks_with_density_full(base_rows)
    stats = collect_experiment_statistics(
        ctx,
        args,
        {
            "magnetic_ablation": row_dicts_from_masks(base_rows),
            "magnetic_ablation_density_full": row_dicts_from_masks(full_rows_masks),
        },
    )
    hidden_rows = _magnetic_ablation_rows_from_stats(stats["magnetic_ablation"])
    full_rows = _magnetic_ablation_rows_from_stats(stats["magnetic_ablation_density_full"])
    percents = np.asarray(
        [100.0 * float(row["visible_fraction"]) for row in hidden_rows],
        dtype=np.float64,
    )
    full_percents = np.asarray(
        [100.0 * float(row["visible_fraction"]) for row in full_rows],
        dtype=np.float64,
    )
    if not np.allclose(percents, full_percents):
        raise RuntimeError("Magnetic ablation B-visible levels differ across density conditions.")
    arrays = {"b_visible_percent": percents}
    arrays.update(_pack_magnetic_condition(hidden_rows, "density_hidden"))
    arrays.update(_pack_magnetic_condition(full_rows, "density_full"))
    run_count = len(hidden_rows[0]["density"]["run_names"]) if hidden_rows else 0
    meta = {
        "figure": "magnetic_ablation_summary",
        "source": {
            "density_hidden": "collect_validation_statistics",
            "density_full": "collect_validation_statistics",
        },
        "density_conditions": ["fully_hidden", "visible_100"],
        "shared_b_probe_layout": True,
        "b_probe_layout": (
            "single nested spatial ranking cloned across Density hidden and "
            "Density 100%; same ranking applied to every validation window"
        ),
        "b_visible_percents": percents.tolist(),
        "n_levels": len(hidden_rows),
        "run_count": run_count,
        "rows": {"density_hidden": hidden_rows, "density_full": full_rows},
        "density_nrmse_definition": DENSITY_NRMSE_DEFINITION,
        "jy_nrmse_definition": JY_NRMSE_DEFINITION,
        "jy_definition": JY_STATS_DEFINITION,
        "interval": "median and 16th-84th percentile range",
        "source_files": [],
    }
    meta.update(
        jy_metric_metadata(
            jy_stats=ctx.get("jy_stats"),
            checkpoint_path=ctx.get("ckpt_path") or checkpoint_path,
            checkpoint_epoch=checkpoint_epoch,
        )
    )
    return arrays, meta


def plot_magnetic_ablation_summary(arrays: Dict, metadata: Dict, figures_dir: Path, dpi: int) -> Dict:
    percents = np.asarray(arrays["b_visible_percent"])
    x = np.arange(len(percents))
    labels = [f"{value:g}" for value in percents]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    styles = {
        "density_hidden": dict(
            color="C0", linestyle="-", marker="o", label="Density fully hidden"
        ),
        "density_full": dict(
            color="C1", linestyle="--", marker="^", label="Density visible 100%"
        ),
    }
    plotted = []
    for prefix, style in styles.items():
        plot_style = {k: v for k, v in style.items() if k != "label"}
        for ax, metric, labeled in (
            (axes[0], "density", True),
            (axes[1], "jy", False),
        ):
            med = clip_positive_for_log(arrays[f"{prefix}_{metric}_median"])
            lo = clip_positive_for_log(arrays[f"{prefix}_{metric}_p16"])
            hi = clip_positive_for_log(arrays[f"{prefix}_{metric}_p84"])
            plotted.extend([med, lo, hi])
            ax.fill_between(
                x,
                lo,
                hi,
                color=style["color"],
                alpha=0.18,
                linewidth=0.4,
                edgecolor="black",
            )
            ax.plot(
                x,
                med,
                linewidth=1.8,
                markersize=5,
                **plot_style,
                label=style["label"] if labeled else None,
            )
    stacked = np.concatenate([np.asarray(values, dtype=np.float64).ravel() for values in plotted])
    stacked = stacked[np.isfinite(stacked) & (stacked > 0)]
    if stacked.size == 0:
        ymin, ymax = LOG_Y_FLOOR, 1.0
    else:
        ymin = max(LOG_Y_FLOOR, float(np.min(stacked))) / 1.2
        ymax = float(np.max(stacked)) * 1.2
    titles = ("Density NRMSE", JY_PAPER_PANEL_TITLE)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("B visible (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("NRMSE")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    ylims = (axes[0].get_ylim(), axes[1].get_ylim())
    save_png_pdf(fig, figures_dir, FIGURE_STEMS["magnetic_ablation"], dpi)
    return {
        "layout": (1, 2),
        "titles": list(titles),
        "sharey": True,
        "ylims": ylims,
        "density_conditions": [label for _prefix, label in MAGNETIC_DENSITY_CONDITIONS],
    }


def _forecast_or_superres_from_stats_rows(stats_rows: List[Dict], kind: str) -> List[Dict]:
    converted = []
    for row in stats_rows:
        number = parse_trailing_number(row["name"])
        if kind == "forecast":
            extra = {"history": int(number)}
        else:
            extra = {"probe_count": int(number)}
        if "density" in row and "run_profiles" in row["density"]:
            density_profiles = {
                name: profile.tolist()
                for name, profile in zip(row["density"]["run_names"], row["density"]["run_profiles"])
            }
            jy_profiles = {
                name: profile.tolist()
                for name, profile in zip(row["jy"]["run_names"], row["jy"]["run_profiles"])
            }
            density_median = np.asarray(row["density"]["median"], dtype=np.float64)
            jy_median = np.asarray(row["jy"]["median"], dtype=np.float64)
            density_p16 = np.asarray(row["density"]["p16"], dtype=np.float64)
            jy_p16 = np.asarray(row["jy"]["p16"], dtype=np.float64)
            density_p84 = np.asarray(row["density"]["p84"], dtype=np.float64)
            jy_p84 = np.asarray(row["jy"]["p84"], dtype=np.float64)
        else:
            density_profiles = _profiles_from_val_row(row, "density")
            jy_profiles = _profiles_from_val_row(row, "jy")
            stacked_d = np.stack([density_profiles[name] for name in sorted(density_profiles)])
            stacked_j = np.stack([jy_profiles[name] for name in sorted(jy_profiles)])
            density_median = np.median(stacked_d, axis=0)
            jy_median = np.median(stacked_j, axis=0)
            density_p16 = np.percentile(stacked_d, 16.0, axis=0)
            jy_p16 = np.percentile(stacked_j, 16.0, axis=0)
            density_p84 = np.percentile(stacked_d, 84.0, axis=0)
            jy_p84 = np.percentile(stacked_j, 84.0, axis=0)
        converted.append(
            {
                "name": row["name"],
                **extra,
                "density_profiles": density_profiles,
                "jy_profiles": jy_profiles,
                "density_median": density_median.tolist(),
                "density_p16": density_p16.tolist(),
                "density_p84": density_p84.tolist(),
                "jy_median": jy_median.tolist(),
                "jy_p16": jy_p16.tolist(),
                "jy_p84": jy_p84.tolist(),
                "density_summary": summarize_run_scalars(density_profiles),
                "jy_summary": summarize_run_scalars(jy_profiles),
            }
        )
    return converted


def build_paired_b_condition_cache(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    ctx: Dict | None,
    experiment: str,
    checkpoint_epoch: int,
) -> Tuple[Dict, Dict]:
    conditions: Dict[str, List[Dict]] = {}
    source: Dict[str, str] = {}
    source_files: List[str] = []
    local_frames = None
    context_length = None
    run_count = None
    jy_payload_stats = None
    for hide_magnetic, key in ((False, "B_full"), (True, "B_hidden")):
        payload = None if args.force_recompute else try_load_validation_json(
            run_dir,
            checkpoint_path,
            experiment,
            hide_magnetic=hide_magnetic,
            checkpoint_epoch=checkpoint_epoch,
        )
        if payload is not None:
            if jy_payload_stats is None:
                jy_payload_stats = payload
            conditions[key] = _forecast_or_superres_from_stats_rows(payload["rows"], experiment)
            source[key] = payload.get("_source_json")
            source_files.append(str(payload.get("_source_json")))
            local_frames = payload["local_frames"]
            context_length = payload["context_length"]
            run_count = payload["run_count"]
            continue
        if ctx is None:
            raise RuntimeError(f"{experiment} cache needs inference context.")
        sample, y = select_named_window(ctx, args)
        if experiment == "forecast":
            histories = default_density_forecast_visible_frames(ctx["delta_t"])
            mask_rows = build_density_forecast_rows(
                block=y,
                visible_frame_counts=histories,
                magnetic_visible=not hide_magnetic,
            )
        else:
            mask_rows = build_density_superres_rows(
                block=y,
                probe_counts=list(args.density_probe_counts),
                magnetic_visible=not hide_magnetic,
            )
        stats = collect_experiment_statistics(
            ctx,
            args,
            {experiment: row_dicts_from_masks(mask_rows)},
        )[experiment]
        conditions[key] = _forecast_or_superres_from_stats_rows(stats, experiment)
        source[key] = "collect_validation_statistics"
        local_frames = list(range(ctx["delta_t"]))
        context_length = ctx["delta_t"]
        run_count = len(next(iter(conditions[key][0]["density_profiles"])))

    first = conditions["B_full"][0]
    n_frames = len(first["density_median"])
    arrays = {"local_frames": np.asarray(local_frames, dtype=np.int32)}
    meta_rows = {"B_full": [], "B_hidden": []}
    for key, rows in conditions.items():
        for row in rows:
            tag = f"{key}_{row['name']}"
            arrays[f"{tag}_density_median"] = np.asarray(row["density_median"])
            arrays[f"{tag}_density_p16"] = np.asarray(row["density_p16"])
            arrays[f"{tag}_density_p84"] = np.asarray(row["density_p84"])
            arrays[f"{tag}_jy_median"] = np.asarray(row["jy_median"])
            arrays[f"{tag}_jy_p16"] = np.asarray(row["jy_p16"])
            arrays[f"{tag}_jy_p84"] = np.asarray(row["jy_p84"])
            meta_rows[key].append(
                {k: v for k, v in row.items() if k not in {"density_profiles", "jy_profiles"}}
                | {
                    "density_profiles": row["density_profiles"],
                    "jy_profiles": row["jy_profiles"],
                }
            )
    meta = {
        "figure": f"{experiment}_summary",
        "source": source,
        "b_conditions": ["B_full", "B_hidden"],
        "local_frames": list(local_frames),
        "context_length": context_length,
        "n_frames": n_frames,
        "run_count": run_count,
        "rows": meta_rows,
        "density_nrmse_definition": DENSITY_NRMSE_DEFINITION,
        "jy_nrmse_definition": JY_NRMSE_DEFINITION,
        "interval": "median and 16th-84th percentile range",
        "source_files": source_files,
    }
    if experiment == "forecast":
        meta["histories"] = [int(row["history"]) for row in conditions["B_full"]]
        meta["horizons"] = [
            int(context_length) - int(row["history"]) for row in conditions["B_full"]
        ]
    if experiment == "superres":
        size_x, size_z = infer_spatial_size(run_dir, ctx=ctx, args=args)
        probe_counts = [int(row["probe_count"]) for row in conditions["B_full"]]
        meta["size_x"] = size_x
        meta["size_z"] = size_z
        meta["probe_counts"] = probe_counts
        meta["visible_ratio_percent"] = [
            100.0 * density_visible_ratio(count, size_x, size_z) for count in probe_counts
        ]
    meta.update(
        jy_metric_metadata(
            jy_stats=(ctx or {}).get("jy_stats") or jy_payload_stats,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=checkpoint_epoch,
        )
    )
    return arrays, meta


def plot_density_forecast_summary(arrays: Dict, metadata: Dict, figures_dir: Path, dpi: int) -> Dict:
    frames = np.asarray(arrays["local_frames"])
    full_rows = metadata["rows"]["B_full"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    plotted = []
    for index, row in enumerate(full_rows):
        color = colors[index % len(colors)]
        history = int(row["history"])
        horizon = int(metadata["context_length"]) - history
        for key, linestyle, alpha in (("B_full", "-", 0.18), ("B_hidden", "--", 0.08)):
            tag = f"{key}_{row['name']}"
            for ax, metric in ((axes[0], "density"), (axes[1], "jy")):
                lo = clip_positive_for_log(arrays[f"{tag}_{metric}_p16"])
                hi = clip_positive_for_log(arrays[f"{tag}_{metric}_p84"])
                mid = clip_positive_for_log(arrays[f"{tag}_{metric}_median"])
                plotted.extend([lo, hi, mid])
                ax.fill_between(
                    frames,
                    lo,
                    hi,
                    color=color,
                    alpha=alpha,
                    linewidth=0.4,
                    edgecolor="black",
                )
                ax.plot(
                    frames,
                    mid,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=f"horizon={horizon}" if key == "B_full" else None,
                )
    stacked = np.concatenate([np.asarray(values, dtype=np.float64).ravel() for values in plotted])
    stacked = stacked[np.isfinite(stacked) & (stacked > 0)]
    if stacked.size == 0:
        ymin, ymax = LOG_Y_FLOOR, 1.0
    else:
        ymin = max(LOG_Y_FLOOR, float(np.min(stacked))) / 1.2
        ymax = float(np.max(stacked)) * 1.2
    xlabel = f"Local frame in {int(metadata['context_length'])}-frame context window"
    titles = ("Density NRMSE", JY_PAPER_PANEL_TITLE)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("NRMSE")
    color_handles = [
        Line2D([0], [0], color=colors[i % len(colors)], linewidth=1.8, label=f"horizon={int(metadata['context_length']) - int(row['history'])}")
        for i, row in enumerate(full_rows)
    ]
    style_handles = [
        Line2D([0], [0], color="0.3", linestyle="-", linewidth=1.8, label="B visible 100%"),
        Line2D([0], [0], color="0.3", linestyle="--", linewidth=1.8, label="B visible 0%"),
    ]
    axes[0].legend(handles=color_handles, frameon=False, fontsize=7, loc="upper left")
    axes[1].legend(handles=style_handles, frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    ylims = (axes[0].get_ylim(), axes[1].get_ylim())
    save_png_pdf(fig, figures_dir, FIGURE_STEMS["forecast"], dpi)
    return {
        "layout": (1, 2),
        "titles": list(titles),
        "sharey": True,
        "ylims": ylims,
    }


def plot_density_superres_summary(
    arrays: Dict,
    metadata: Dict,
    figures_dir: Path,
    dpi: int,
    size_x: int,
    size_z: int,
) -> Dict:
    full_rows = sorted(metadata["rows"]["B_full"], key=lambda row: int(row["probe_count"]))
    probe_counts = [int(row["probe_count"]) for row in full_rows]
    ratios = [100.0 * density_visible_ratio(count, size_x, size_z) for count in probe_counts]
    metadata["size_x"] = int(size_x)
    metadata["size_z"] = int(size_z)
    metadata["visible_ratio_percent"] = ratios
    x = np.arange(len(probe_counts))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    styles = SUPERRES_B_CONDITION_STYLES
    plotted = []
    for key, style in styles.items():
        rows_by_count = {
            int(row["probe_count"]): row for row in metadata["rows"][key]
        }
        density = clip_positive_for_log(
            [rows_by_count[count]["density_summary"]["median"] for count in probe_counts]
        )
        density_p16 = clip_positive_for_log(
            [rows_by_count[count]["density_summary"]["p16"] for count in probe_counts]
        )
        density_p84 = clip_positive_for_log(
            [rows_by_count[count]["density_summary"]["p84"] for count in probe_counts]
        )
        jy = clip_positive_for_log(
            [rows_by_count[count]["jy_summary"]["median"] for count in probe_counts]
        )
        jy_p16 = clip_positive_for_log(
            [rows_by_count[count]["jy_summary"]["p16"] for count in probe_counts]
        )
        jy_p84 = clip_positive_for_log(
            [rows_by_count[count]["jy_summary"]["p84"] for count in probe_counts]
        )
        plotted.extend([density, density_p16, density_p84, jy, jy_p16, jy_p84])
        plot_style = {k: v for k, v in style.items() if k != "label"}
        for ax, lo, hi, mid, labeled in (
            (axes[0], density_p16, density_p84, density, True),
            (axes[1], jy_p16, jy_p84, jy, False),
        ):
            ax.fill_between(
                x,
                lo,
                hi,
                color=style["color"],
                alpha=0.18,
                linewidth=0.4,
                edgecolor="black",
            )
            ax.plot(
                x,
                mid,
                linewidth=1.8,
                markersize=5,
                **plot_style,
                label=style["label"] if labeled else None,
            )
    stacked = np.concatenate([np.asarray(values, dtype=np.float64).ravel() for values in plotted])
    stacked = stacked[np.isfinite(stacked) & (stacked > 0)]
    if stacked.size == 0:
        ymin, ymax = LOG_Y_FLOOR, 1.0
    else:
        ymin = max(LOG_Y_FLOOR, float(np.min(stacked))) / 1.2
        ymax = float(np.max(stacked)) * 1.2
    titles = ("Density NRMSE", JY_PAPER_PANEL_TITLE)
    xticklabels = [format_visible_ratio_percent(ratio) for ratio in ratios]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("Density visible ratio (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("NRMSE")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    ylims = (axes[0].get_ylim(), axes[1].get_ylim())
    save_png_pdf(fig, figures_dir, FIGURE_STEMS["superres"], dpi)
    return {
        "layout": (1, 2),
        "titles": list(titles),
        "sharey": True,
        "ylims": ylims,
        "b_condition_styles": {
            key: {
                "color": style["color"],
                "linestyle": style["linestyle"],
                "marker": style["marker"],
                "label": style["label"],
            }
            for key, style in styles.items()
        },
    }


# ---------------------------------------------------------------------------
# Appendix: sliding window from existing arrays when possible
# ---------------------------------------------------------------------------


def find_sliding_step_files(run_dir: Path) -> Tuple[Path, Path] | None:
    folder = run_dir / SLIDING_DIR_NO_B
    if not folder.exists():
        return None
    npz_matches = sorted(folder.glob("*_steps-*_B-hidden_*_final_reconstructions.npz"))
    json_matches = sorted(folder.glob("*_steps-*_B-hidden_*_metrics.json"))
    if not npz_matches or not json_matches:
        return None
    return npz_matches[0], json_matches[0]


def compute_sliding_aggregate_rmse(
    ctx: Dict,
    args: argparse.Namespace,
    candidate_runs: Sequence[str],
    slide_steps: Sequence[int],
    n_frames: int = SLIDING_AGGREGATE_N_FRAMES,
) -> Tuple[
    np.ndarray,
    Dict[str, Dict[int, np.ndarray]],
    Dict[str, Dict[int, np.ndarray]],
    Dict[str, Dict[int, np.ndarray]],
    Dict[str, Dict[int, np.ndarray]],
    List[str],
    List[Dict],
    List[str],
]:
    device = ctx["device"]
    mean = ctx["mean"].reshape(4, 1, 1, 1)
    std = ctx["std"].reshape(4, 1, 1, 1)
    betas = ctx["ckpt_args"].get("betas", [0.2])
    window_size = int(ctx["delta_t"])
    frame_index = np.arange(int(n_frames), dtype=np.int32)
    b_keys = ("B_full", "B_hidden")
    per_rmse: Dict[str, Dict[int, List[np.ndarray]]] = {
        key: {int(step): [] for step in slide_steps} for key in b_keys
    }
    uq_records: List[Dict] = []
    loaded: List[str] = []
    skipped: List[str] = []
    for run_name in candidate_runs:
        n_available = peek_run_n_frames(ctx["h5_dir"], betas, run_name)
        if n_available is None:
            print(f"Skipping {run_name}: not found in HDF5")
            skipped.append(str(run_name))
            continue
        if int(n_available) < int(n_frames):
            print(
                f"Skipping {run_name}: {n_available} frames < {n_frames}, no padding"
            )
            skipped.append(str(run_name))
            continue
        fields, _frame_ids, h5_path = find_and_load_run(ctx["h5_dir"], betas, run_name)
        fields = np.asarray(fields)[: int(n_frames)]
        print(
            f"Sliding aggregate RMSE: {run_name} "
            f"({n_available} frames, using first {n_frames}) from {h5_path}"
        )
        target = torch.from_numpy(np.transpose(fields, (1, 0, 2, 3))).to(
            device=device, dtype=torch.float32
        )
        target_normalized = (target - mean) / (std + 1e-8)
        target_density_plot = target[3].detach().cpu().numpy()
        probe_mask, _probe_info = sliding_density_probe_mask(
            target_normalized.unsqueeze(0),
            int(args.density_probe_count),
        )
        for hide_magnetic, b_key in ((False, "B_full"), (True, "B_hidden")):
            for step in slide_steps:
                result = reconstruct_with_slide_step(
                    model=ctx["model"],
                    target_normalized=target_normalized,
                    target_density_plot=target_density_plot,
                    density_mean=mean[3],
                    density_std=std[3],
                    probe_mask=probe_mask,
                    window_size=window_size,
                    slide_step=int(step),
                    plot_units="physical",
                    x_index=int(args.x_index),
                    amp=True,
                    hide_magnetic=hide_magnetic,
                )
                rmse, _mae = framewise_rmse_mae(
                    result["final_reconstruction"], target_density_plot
                )
                if np.asarray(rmse).shape[0] != int(n_frames):
                    raise RuntimeError(
                        f"{run_name} step={step} RMSE length {len(rmse)} != {n_frames}"
                    )
                per_rmse[b_key][int(step)].append(np.asarray(rmse, dtype=np.float64))
                if "predicted_squared_error_normalized" in result:
                    pred_norm = (np.asarray(result["final_reconstruction"]) - float(mean[3].item())) / (float(std[3].item()) + 1e-8)
                    target_norm = target_normalized[3].detach().cpu().numpy()
                    variance = np.asarray(result["predicted_squared_error_normalized"])
                    hidden = ~(probe_mask.detach().cpu().numpy() > .5)
                    selection = np.broadcast_to(hidden, pred_norm.shape)
                    e2 = np.square(pred_norm - target_norm)
                    sampled_v, sampled_e2 = _bounded_pair_sample(
                        variance[selection], e2[selection], int(args.uq_voxels_per_window),
                        int(args.seed) + len(uq_records) * 101,
                    )
                    corr = pearson_spearman(sampled_v, sampled_e2)
                    mean_e2 = float(np.mean(e2[selection])); mean_variance = float(np.mean(variance[selection]))
                    bias = calibration_bias(mean_e2, mean_variance)
                    uq_records.append({
                        "run_name": str(run_name), "b_condition": b_key, "step": int(step),
                        "density_nrmse": float(np.sqrt(np.mean(e2[selection]))),
                        "mean_e2": mean_e2, "mean_variance": mean_variance,
                        "signed_bias": bias["signed_bias"], "calibration_gap": bias["calibration_gap"], **corr,
                        "frame_e2": np.mean(e2[:, hidden], axis=1).tolist(),
                        "frame_variance": np.mean(variance[:, hidden], axis=1).tolist(),
                    })
        loaded.append(str(run_name))
    if not loaded:
        raise RuntimeError(
            "No validation runs with at least "
            f"{n_frames} frames were available for sliding aggregate RMSE."
        )
    medians: Dict[str, Dict[int, np.ndarray]] = {key: {} for key in b_keys}
    p16s: Dict[str, Dict[int, np.ndarray]] = {key: {} for key in b_keys}
    p84s: Dict[str, Dict[int, np.ndarray]] = {key: {} for key in b_keys}
    stacked_by: Dict[str, Dict[int, np.ndarray]] = {key: {} for key in b_keys}
    for b_key in b_keys:
        for step in slide_steps:
            stacked = np.stack(per_rmse[b_key][int(step)], axis=0)
            medians[b_key][int(step)] = np.median(stacked, axis=0)
            p16s[b_key][int(step)] = np.percentile(stacked, 16.0, axis=0)
            p84s[b_key][int(step)] = np.percentile(stacked, 84.0, axis=0)
            stacked_by[b_key][int(step)] = stacked
    return frame_index, medians, p16s, p84s, stacked_by, loaded, skipped, uq_records


def reconstruct_sliding_qualitative(
    ctx: Dict,
    args: argparse.Namespace,
    slide_steps: Sequence[int],
    n_frames: int = SLIDING_AGGREGATE_N_FRAMES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, np.ndarray], Dict]:
    """Canonical-run B-hidden reconstructions for the right-hand qualitative panel."""
    betas = ctx["ckpt_args"].get("betas", [0.2])
    fields, frame_ids, h5_path = find_and_load_run(
        ctx["h5_dir"], betas, args.run_name
    )
    fields = np.asarray(fields)
    frame_ids = np.asarray(frame_ids)
    if fields.shape[0] < int(n_frames):
        raise RuntimeError(
            f"Qualitative run {args.run_name} has {fields.shape[0]} frames < "
            f"{n_frames}; no padding."
        )
    fields = fields[: int(n_frames)]
    frame_ids = frame_ids[: int(n_frames)]
    print(
        f"Sliding qualitative: {args.run_name} "
        f"(first {n_frames} frames) from {h5_path}"
    )
    mean = ctx["mean"].reshape(4, 1, 1, 1)
    std = ctx["std"].reshape(4, 1, 1, 1)
    target = torch.from_numpy(np.transpose(fields, (1, 0, 2, 3))).to(
        device=ctx["device"], dtype=torch.float32
    )
    target_normalized = (target - mean) / (std + 1e-8)
    target_density_plot = target[3].detach().cpu().numpy()
    probe_mask, probe_info = sliding_density_probe_mask(
        target_normalized.unsqueeze(0),
        int(args.density_probe_count),
    )
    predictions = {}
    for step in slide_steps:
        result = reconstruct_with_slide_step(
            model=ctx["model"],
            target_normalized=target_normalized,
            target_density_plot=target_density_plot,
            density_mean=mean[3],
            density_std=std[3],
            probe_mask=probe_mask,
            window_size=int(ctx["delta_t"]),
            slide_step=int(step),
            plot_units="physical",
            x_index=int(args.x_index),
            amp=True,
            hide_magnetic=True,
        )
        predictions[int(step)] = np.asarray(result["final_reconstruction"])
    return (
        np.asarray(frame_ids),
        target_density_plot,
        probe_mask.detach().cpu().numpy(),
        predictions,
        dict(probe_info),
    )


def build_sliding_appendix_cache(
    args: argparse.Namespace,
    run_dir: Path,
    ctx: Dict | None,
    checkpoint_path: Path,
    checkpoint_epoch: int,
) -> Tuple[Dict, Dict]:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    if ctx is None:
        raise RuntimeError("Sliding multi-run RMSE needs inference context.")
    slide_steps = [int(step) for step in args.slide_steps]
    candidate_runs = validation_run_order(run_dir)
    if not candidate_runs:
        raise RuntimeError(f"No test runs in the authoritative project split")
    aggregate_runs = select_sliding_aggregate_runs(
        candidate_runs,
        int(args.sliding_max_runs),
    )
    (
        aggregate_frames,
        medians,
        p16s,
        p84s,
        stacked_by,
        loaded_runs,
        skipped_runs,
        uq_records,
    ) = compute_sliding_aggregate_rmse(
        ctx,
        args,
        aggregate_runs,
        slide_steps,
        n_frames=SLIDING_AGGREGATE_N_FRAMES,
    )
    representative_step = int(min(slide_steps))
    (
        frame_ids,
        target,
        probe_mask,
        predictions,
        probe_info,
    ) = reconstruct_sliding_qualitative(
        ctx, args, slide_steps, n_frames=SLIDING_AGGREGATE_N_FRAMES
    )
    std_density = float(ctx["std"].detach().cpu().numpy().reshape(-1)[3])
    size_x = int(target.shape[-2])
    size_z = int(target.shape[-1])
    probe_count = int(args.density_probe_count)
    visible_ratio = density_visible_ratio(probe_count, size_x, size_z)
    arrays = {
        "frame_ids": frame_ids.astype(np.int32),
        "aggregate_frame_ids": np.asarray(aggregate_frames, dtype=np.int32),
        "target_density": target.astype(np.float32),
        "probe_mask": probe_mask.astype(np.float32),
        "target_tz": project_time_z(target, x_index=args.x_index).astype(np.float32),
    }
    for step, prediction in predictions.items():
        arrays[f"step_{step}_prediction"] = prediction.astype(np.float32)
        arrays[f"step_{step}_tz"] = project_time_z(
            prediction, x_index=args.x_index
        ).astype(np.float32)
        if std_density > 0:
            arrays[f"step_{step}_tz_residual"] = (
                (arrays[f"step_{step}_tz"] - arrays["target_tz"]) / std_density
            ).astype(np.float32)
        else:
            arrays[f"step_{step}_tz_residual"] = (
                arrays[f"step_{step}_tz"] - arrays["target_tz"]
            ).astype(np.float32)
    for step in slide_steps:
        for b_key in ("B_full", "B_hidden"):
            arrays[f"{b_key}_step_{step}_rmse_median"] = np.asarray(
                medians[b_key][int(step)], dtype=np.float32
            )
            arrays[f"{b_key}_step_{step}_rmse_p16"] = np.asarray(
                p16s[b_key][int(step)], dtype=np.float32
            )
            arrays[f"{b_key}_step_{step}_rmse_p84"] = np.asarray(
                p84s[b_key][int(step)], dtype=np.float32
            )
            arrays[f"{b_key}_step_{step}_rmse_per_run"] = np.asarray(
                stacked_by[b_key][int(step)], dtype=np.float32
            )
    visible_ratio_percent = 100.0 * visible_ratio
    meta = {
        "figure": "sliding_window_appendix",
        "source": "sliding reconstruction",
        "b_condition": "B fully hidden",
        "slide_steps": list(slide_steps),
        "representative_step": representative_step,
        "x_index": int(args.x_index),
        "run_name": args.run_name,
        "plot_units": "physical",
        "left_panel": "target_and_multi_run_framewise_rmse",
        "right_panel": "single_run_qualitative_by_step",
        "layout": "target_rmse_predictions_residuals",
        "qualitative_steps": list(slide_steps),
        "qualitative_run": args.run_name,
        "b_conditions": ["B_full", "B_hidden"],
        "aggregate_n_runs": len(loaded_runs),
        "aggregate_runs": loaded_runs,
        "aggregate_skipped_runs": skipped_runs,
        "statistical_unit": "run",
        "frame_selection": f"first {SLIDING_AGGREGATE_N_FRAMES} frames",
        "frame_range": [0, SLIDING_AGGREGATE_N_FRAMES],
        "n_frames": SLIDING_AGGREGATE_N_FRAMES,
        "context_length": int(ctx["delta_t"]),
        "hide_magnetic": True,
        "density_probe_count": probe_count,
        "density_visible_ratio": visible_ratio,
        "density_visible_ratio_percent": visible_ratio_percent,
        "probe_layout_semantics": "same as density_superres",
        "probe_grid": probe_info,
        "size_x": size_x,
        "size_z": size_z,
        "aggregation": (
            f"{len(loaded_runs)} test runs; one framewise RMSE series per run "
            f"from the first {SLIDING_AGGREGATE_N_FRAMES} frames, B condition, "
            "and slide step; then median and 16th-84th percentiles across runs"
        ),
        "sliding_semantics": (
            "Each window overwrites its full valid Density prediction. Overlapping prior predictions are reused "
            "as exact field inputs with their predicted variance; real probes override them. At step=24 windows "
            "do not overlap, so no previous prediction is reused: independent non-overlapping window inference."
        ),
        "uq_records": uq_records,
        "uq_correlation_sampling": {"seed": int(args.seed), "max_voxels_per_run_step_condition": int(args.uq_voxels_per_window)},
        "source_files": [],
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_epoch": int(checkpoint_epoch),
    }
    return arrays, meta


def plot_sliding_window_appendix(
    arrays: Dict,
    metadata: Dict,
    figures_dir: Path,
    extent: Sequence[float],
    dpi: int,
) -> Dict:
    steps = [int(step) for step in metadata["slide_steps"]]
    qualitative_steps = [step for step in (1, 12, 24) if step in steps]
    if len(qualitative_steps) != 3:
        qualitative_steps = steps[:3]
    frame_ids = arrays["frame_ids"]
    aggregate_frames = arrays.get("aggregate_frame_ids", frame_ids)
    n_runs = metadata.get("aggregate_n_runs")
    n_frames = metadata.get("n_frames")
    probe_count = metadata.get("density_probe_count")
    ratio_percent = metadata.get("density_visible_ratio_percent")
    run_phrase = (
        f"{int(n_runs)} test runs, first {int(n_frames)} frames"
        if n_runs and n_frames
        else (f"{int(n_runs)} test runs" if n_runs else "Test runs")
    )
    if probe_count is not None and ratio_percent is not None:
        figure_title = (
            f"{run_phrase}\n"
            f"{int(probe_count)} fixed Density probes "
            f"({format_visible_ratio_percent(float(ratio_percent))}% spatial visibility)"
        )
    else:
        figure_title = run_phrase

    fig = plt.figure(figsize=(11.4, 5.4))
    gs = gridspec.GridSpec(
        3,
        5,
        figure=fig,
        width_ratios=[1.18, 1.0, 0.045, 1.0, 0.045],
        wspace=0.16,
        hspace=0.18,
        left=0.07,
        right=0.95,
        top=0.84,
        bottom=0.10,
    )
    ax_target = fig.add_subplot(gs[0, 0])
    ax_err = fig.add_subplot(gs[1:, 0])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    b_styles = (
        ("B_full", "-", 0.18),
        ("B_hidden", "--", 0.08),
    )
    for index, step in enumerate(steps):
        color = colors[index % len(colors)]
        for b_key, linestyle, alpha in b_styles:
            ax_err.fill_between(
                aggregate_frames,
                arrays[f"{b_key}_step_{step}_rmse_p16"],
                arrays[f"{b_key}_step_{step}_rmse_p84"],
                facecolor=color,
                alpha=alpha,
                linewidth=0.4,
                edgecolor="0.35",
            )
            ax_err.plot(
                aggregate_frames,
                arrays[f"{b_key}_step_{step}_rmse_median"],
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
            )
    ax_err.set_xlabel("Global frame")
    ax_err.set_ylabel("Frame RMSE")
    ax_err.grid(alpha=0.25, linewidth=0.6)
    ax_err.set_axisbelow(True)
    ax_err.set_title("RMSE vs global frame", fontsize=9)
    mean_rmse_by_step = {}
    color_handles = []
    for index, step in enumerate(steps):
        mean_b100 = float(
            np.nanmean(
                np.asarray(
                    arrays[f"B_full_step_{step}_rmse_median"], dtype=np.float64
                )
            )
        )
        mean_b0 = float(
            np.nanmean(
                np.asarray(
                    arrays[f"B_hidden_step_{step}_rmse_median"], dtype=np.float64
                )
            )
        )
        mean_rmse_by_step[int(step)] = {"B100": mean_b100, "B0": mean_b0}
        color_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[index % len(colors)],
                linewidth=1.6,
                label=f"step={step}   {mean_b100:.3f} / {mean_b0:.3f}",
            )
        )
    style_handles = [
        Line2D([0], [0], color="0.3", linestyle="-", linewidth=1.6, label="B visible 100%"),
        Line2D([0], [0], color="0.3", linestyle="--", linewidth=1.6, label="B visible 0%"),
    ]
    color_legend = ax_err.legend(
        handles=color_handles,
        frameon=False,
        fontsize=6.5,
        loc="upper left",
        title="step   mean RMSE (B100 / B0)",
        title_fontsize=6.5,
    )
    ax_err.add_artist(color_legend)
    ax_err.legend(handles=style_handles, frameon=False, fontsize=7, loc="upper right")

    zmin, zmax, _xmin, _xmax = extent
    tmin, tmax = float(frame_ids.min()) - 0.5, float(frame_ids.max()) + 0.5
    field_panels = [arrays["target_tz"]] + [arrays[f"step_{step}_tz"] for step in qualitative_steps]
    residual_panels = [arrays[f"step_{step}_tz_residual"] for step in qualitative_steps]
    field_vmin, field_vmax = robust_limits(
        field_panels, channel=3, q=99.0, symmetric=False
    )
    res_vmin, res_vmax = robust_limits(
        residual_panels, channel=3, q=99.0, symmetric=True
    )
    target_im = ax_target.imshow(
        arrays["target_tz"].T,
        origin="lower",
        aspect="auto",
        extent=[tmin, tmax, zmin, zmax],
        cmap=make_nan_cmap("magma"),
        vmin=field_vmin,
        vmax=field_vmax,
        interpolation="nearest",
    )
    ax_target.set_title("Target", fontsize=9, pad=2)
    ax_target.set_ylabel("z [cm]", fontsize=8)
    ax_target.tick_params(labelbottom=False)
    ensure_sliding_qualitative_nrmse_metadata(arrays, metadata)
    nrmse_by_step = {
        int(step): float(metadata[qualitative_density_nrmse_key(step)])
        for step in qualitative_steps
    }

    pred_axes = []
    res_axes = []
    pred_im = None
    res_im = None
    panel_titles = ["Target", "RMSE vs global frame"]
    for index, step in enumerate(qualitative_steps):
        ax_pred = fig.add_subplot(gs[index, 1])
        pred_im = ax_pred.imshow(
            arrays[f"step_{step}_tz"].T,
            origin="lower",
            aspect="auto",
            extent=[tmin, tmax, zmin, zmax],
            cmap=make_nan_cmap("magma"),
            vmin=field_vmin,
            vmax=field_vmax,
            interpolation="nearest",
        )
        pred_title = f"Prediction (step={step})"
        ax_pred.set_title(pred_title, fontsize=9, pad=2)
        ax_pred.set_ylabel("z [cm]", fontsize=8)
        if index == len(qualitative_steps) - 1:
            ax_pred.set_xlabel("Global frame")
        else:
            ax_pred.tick_params(labelbottom=False)
        pred_axes.append(ax_pred)
        panel_titles.append(pred_title)

        ax_res = fig.add_subplot(gs[index, 3], sharey=ax_pred)
        res_im = ax_res.imshow(
            arrays[f"step_{step}_tz_residual"].T,
            origin="lower",
            aspect="auto",
            extent=[tmin, tmax, zmin, zmax],
            cmap=make_nan_cmap(RESIDUAL_CMAP),
            vmin=res_vmin,
            vmax=res_vmax,
            interpolation="nearest",
        )
        res_title = f"Residual (step={step})"
        ax_res.set_title(res_title, fontsize=9, pad=2)
        ax_res.set_ylabel("z [cm]", fontsize=8)
        ax_res.tick_params(labelleft=False)
        if index == len(qualitative_steps) - 1:
            ax_res.set_xlabel("Global frame")
        else:
            ax_res.tick_params(labelbottom=False)
        annotate_density_nrmse(ax_res, nrmse_by_step[step])
        res_axes.append(ax_res)
        panel_titles.append(res_title)
    cax_pred = fig.add_subplot(gs[:, 2])
    cax_res = fig.add_subplot(gs[:, 4])
    fig.colorbar(pred_im, cax=cax_pred)
    fig.colorbar(res_im, cax=cax_res)
    pred_res_sharey = all(
        getattr(ax_res, "_sharey", None) is ax_pred
        for ax_pred, ax_res in zip(pred_axes, res_axes)
    )
    fig.suptitle(figure_title, fontsize=10, y=0.98)
    save_png_pdf(fig, figures_dir, FIGURE_STEMS["sliding"], dpi)
    return {
        "panel_titles": panel_titles,
        "legend_labels": [handle.get_label() for handle in color_handles + style_handles],
        "legend_title": "step   mean RMSE (B100 / B0)",
        "slide_steps": list(steps),
        "qualitative_steps": list(qualitative_steps),
        "qualitative_density_nrmse": nrmse_by_step,
        "mean_rmse": mean_rmse_by_step,
        "target_has_colorbar": False,
        "n_column_colorbars": 2,
        "pred_res_sharey": bool(pred_res_sharey),
    }


def plot_sliding_journal_summaries(metadata: Dict, figures_dir: Path, data_dir: Path, dpi: int) -> None:
    records = metadata.get("uq_records") or []
    if not records:
        print("No uncertainty-aware sliding records; skipping Figure 5 UQ companions")
        return
    for record in records:
        if "signed_bias" not in record:
            record["signed_bias"] = float(record["mean_e2"]) - float(record["mean_variance"])
    steps=sorted({int(r["step"]) for r in records}); conditions=("B_full","B_hidden")
    rows=[]; fig,axes=plt.subplots(1,2,figsize=(9.0,3.6))
    for condition in conditions:
        for ax,metric,ylabel in ((axes[0],"density_nrmse","Time-averaged Density NRMSE"),(axes[1],"mean_e2","Standardized squared units")):
            stats_by=[summarize(r[metric] for r in records if r["b_condition"]==condition and int(r["step"])==step) for step in steps]
            mid=np.array([s["median"] for s in stats_by]);lo=np.array([s["p16"] for s in stats_by]);hi=np.array([s["p84"] for s in stats_by])
            line=ax.plot(steps,mid,marker="o",label=condition.replace("_"," "))[0];ax.fill_between(steps,lo,hi,color=line.get_color(),alpha=.15);ax.set_ylabel(ylabel)
            for step,stats in zip(steps,stats_by):rows.append({"panel":"error","condition":condition,"step":step,"metric":metric,**stats})
        variance_stats=[summarize(r["mean_variance"] for r in records if r["b_condition"]==condition and int(r["step"])==step) for step in steps]
        vmid=np.array([s["median"] for s in variance_stats]);vlo=np.array([s["p16"] for s in variance_stats]);vhi=np.array([s["p84"] for s in variance_stats])
        vline=axes[1].plot(steps,vmid,marker="s",linestyle="--",label=f"{condition.replace('_',' ')} predicted variance")[0]
        axes[1].fill_between(steps,vlo,vhi,color=vline.get_color(),alpha=.10)
        for step,stats in zip(steps,variance_stats):rows.append({"panel":"uncertainty","condition":condition,"step":step,"metric":"mean_variance",**stats})
        for step in steps:
            members=[r for r in records if r["b_condition"]==condition and int(r["step"])==step]
            for metric in ("pearson","spearman","signed_bias","calibration_gap"):
                rows.append({"panel":"uq_quality_summary","condition":condition,"step":step,"metric":metric,**summarize(r[metric] for r in members)})
    for ax in axes:ax.set_xlabel("Sliding step");ax.set_xticks(steps);ax.grid(alpha=.25);ax.legend(frameon=False,fontsize=7)
    fig.tight_layout();save_png_pdf(fig,figures_dir,"sliding_step_error_uncertainty",dpi)
    # Temporal diagnostic, median across runs at the three requested strides.
    fig,axes=plt.subplots(3,1,figsize=(8.2,7.0),sharex=True)
    for ax,step in zip(axes,(1,12,24)):
        for condition in conditions:
            members=[r for r in records if r["b_condition"]==condition and int(r["step"])==step]
            if not members:continue
            e=np.asarray([r["frame_e2"] for r in members]);v=np.asarray([r["frame_variance"] for r in members]);frames=np.arange(e.shape[1])
            ax.plot(frames,np.median(e,axis=0),label=f"{condition}: actual e²");ax.plot(frames,np.median(v,axis=0),"--",label=f"{condition}: predicted variance")
        ax.set_ylabel(f"step={step}\nstandardized²");ax.grid(alpha=.25)
    axes[-1].set_xlabel("Long-sequence frame");axes[0].legend(frameon=False,fontsize=6,ncol=2)
    fig.tight_layout();save_png_pdf(fig,figures_dir,"sliding_error_accumulation",dpi)
    for record in records:
        rows.append({k:v for k,v in record.items() if k not in {"frame_e2","frame_variance"}})
    write_csv(data_dir/"sliding_uncertainty_statistics.csv",rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def infer_spatial_size(
    run_dir: Path,
    metadata: Dict | None = None,
    ctx: Dict | None = None,
    args: argparse.Namespace | None = None,
    data_dir: Path | None = None,
) -> Tuple[int, int]:
    if metadata:
        size_x = int(metadata.get("size_x") or 0)
        size_z = int(metadata.get("size_z") or 0)
        if size_x > 0 and size_z > 0:
            return size_x, size_z
    found = find_sliding_step_files(run_dir)
    if found is not None:
        with np.load(found[0]) as handle:
            target = np.asarray(handle["target_density"])
        return int(target.shape[-2]), int(target.shape[-1])
    if data_dir is not None:
        loaded = load_figure_cache(data_dir, FIGURE_STEMS["spatial"])
        if loaded is not None:
            target = loaded[0].get("target_density")
            if target is not None and np.asarray(target).ndim == 2:
                return int(target.shape[0]), int(target.shape[1])
    if ctx is not None and args is not None:
        _sample, y = select_named_window(ctx, args)
        return int(y.shape[-2]), int(y.shape[-1])
    raise RuntimeError(
        "Could not infer X,Z from cache, sliding arrays, or model context."
    )


def main() -> None:
    args = parse_args()
    apply_paper_style()
    run_dir = expand_path(args.run_dir)
    checkpoint_path = resolve_checkpoint_path(run_dir, args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    figures = selected_figures(args.figure)
    if "spatial" in figures or "sliding" in figures:
        validate_test_run(run_dir, args.run_name)
    checkpoint_epoch = read_checkpoint_epoch(
        checkpoint_path,
        require_final=not bool(getattr(args, "allow_incomplete_checkpoint", False)),
    )
    checkpoint_header = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_model_version = checkpoint_header.get("args", {}).get("model_version")
    del checkpoint_header
    provenance = provenance_for_checkpoint(checkpoint_path, checkpoint_epoch)
    requested_output = getattr(args, "output_dir", None) or getattr(args, "paper_dir", None)
    paper_dir = (
        expand_path(requested_output)
        if requested_output is not None
        else run_dir / PAPER_DIRNAME
    )
    data_dir = paper_dir / "data"
    figures_dir = paper_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sig_base = base_signature(args, run_dir, checkpoint_path)
    sig_base["model_version"] = checkpoint_model_version
    ctx: Dict | None = None
    figure_manifest: List[Dict] = []
    journal_cache: Tuple[Dict, Dict] | None = None

    def ensure_ctx() -> Dict:
        nonlocal ctx
        if ctx is None:
            ctx = load_inference_context(args)
        return ctx

    def ensure_journal_cache() -> Tuple[Dict, Dict]:
        nonlocal journal_cache
        if journal_cache is None:
            signature = {**sig_base, "figure": "journal_statistics", "test_runs_only": True,
                "uq_domain_version": 2,
                "window_stride": 24, "max_windows_per_run": args.statistics_max_windows_per_run,
                "uq_voxels_per_window": int(args.uq_voxels_per_window)}
            journal_cache = load_or_build_figure_cache(
                data_dir, "journal_statistics", signature,
                builder=lambda: build_journal_statistics_cache(ensure_ctx(), args),
                force=args.force_recompute, provenance=provenance,
            )
            write_csv(data_dir / "journal_run_task_statistics.csv", journal_cache[1]["records"])
        return journal_cache

    def spatial_signature() -> Dict:
        return {
            **sig_base,
            "figure": "spatial",
            "b_visible": 1.0,
            "rows": list(SPATIAL_ROW_ORDER),
            "mask_experiment": "multifunction_standardized_50pct",
            "uq_domain_version": 2,
        }

    def mag_signature() -> Dict:
        return {
            **sig_base,
            "figure": "magnetic_ablation",
            "density_conditions": ["fully_hidden", "visible_100"],
            "shared_b_probe_layout": True,
            "layout": "1x2_density_jy",
            "b_visible_levels": list(DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS),
            **_jy_signature_fields(),
        }

    def forecast_signature() -> Dict:
        return {
            **sig_base,
            "figure": "forecast",
            "histories": default_density_forecast_visible_frames(24),
            "b_conditions": ["B_full", "B_hidden"],
            **_jy_signature_fields(),
        }

    def superres_signature() -> Dict:
        return {
            **sig_base,
            "figure": "superres",
            "probe_counts": [int(value) for value in args.density_probe_counts],
            "b_conditions": ["B_full", "B_hidden"],
            "line_style": "two_color_b_conditions",
            **_jy_signature_fields(),
        }

    def sliding_signature() -> Dict:
        signature = {
            **sig_base,
            "figure": "sliding",
            "slide_steps": [int(step) for step in args.slide_steps],
            "density_probe_count": int(args.density_probe_count),
            "probe_layout_semantics": "same as density_superres",
            "x_index": int(args.x_index),
            "layout": "target_rmse_predictions_residuals",
            "left_panel": "target_and_multi_run_framewise_rmse",
            "right_panel": "single_run_qualitative_by_step",
            "qualitative_steps": [int(step) for step in args.slide_steps],
            "statistical_unit": "run",
            "sliding_max_runs": int(args.sliding_max_runs),
            "selected_runs": select_sliding_aggregate_runs(
                validation_run_order(run_dir),
                int(args.sliding_max_runs),
            ),
            "b_conditions": ["B_full", "B_hidden"],
            "min_run_frames": SLIDING_AGGREGATE_N_FRAMES,
            "frame_selection": f"first {SLIDING_AGGREGATE_N_FRAMES} frames",
            "frame_range": [0, SLIDING_AGGREGATE_N_FRAMES],
            "val_run_order": validation_run_order(run_dir),
            "context_length": 24,
        }
        return signature

    if "spatial" in figures:
        arrays, meta = load_or_build_figure_cache(
            data_dir,
            FIGURE_STEMS["spatial"],
            spatial_signature(),
            builder=lambda: build_spatial_qualitative_cache(ensure_ctx(), args),
            force=args.force_recompute,
            provenance=provenance,
        )
        plot_spatial_qualitative(
            arrays,
            meta,
            figures_dir,
            extent=args.extent,
            residual_vmax=args.residual_vmax,
            field_q=args.field_q,
            dpi=args.dpi,
        )
        plot_spatial_calibration(meta, figures_dir, data_dir, args.dpi)
        figure_manifest.append(
            manifest_figure_entry("spatial", FIGURE_STEMS["spatial"], data_dir, meta)
        )

    if "magnetic_ablation" in figures:
        def build_mag():
            try:
                return build_magnetic_ablation_cache(
                    args, run_dir, checkpoint_path, None, checkpoint_epoch
                )
            except RuntimeError:
                return build_magnetic_ablation_cache(
                    args, run_dir, checkpoint_path, ensure_ctx(), checkpoint_epoch
                )

        arrays, meta = load_or_build_figure_cache(
            data_dir,
            FIGURE_STEMS["magnetic_ablation"],
            mag_signature(),
            builder=build_mag,
            force=args.force_recompute,
            provenance=provenance,
        )
        plot_magnetic_ablation_summary(arrays, meta, figures_dir, dpi=args.dpi)
        figure_manifest.append(
            manifest_figure_entry(
                "magnetic_ablation",
                FIGURE_STEMS["magnetic_ablation"],
                data_dir,
                meta,
            )
        )

    if "forecast" in figures:
        def build_forecast():
            try:
                return build_paired_b_condition_cache(
                    args, run_dir, checkpoint_path, None, "forecast", checkpoint_epoch
                )
            except RuntimeError:
                return build_paired_b_condition_cache(
                    args,
                    run_dir,
                    checkpoint_path,
                    ensure_ctx(),
                    "forecast",
                    checkpoint_epoch,
                )

        arrays, meta = load_or_build_figure_cache(
            data_dir,
            FIGURE_STEMS["forecast"],
            forecast_signature(),
            builder=build_forecast,
            force=args.force_recompute,
            provenance=provenance,
        )
        plot_density_forecast_summary(arrays, meta, figures_dir, dpi=args.dpi)
        figure_manifest.append(
            manifest_figure_entry("forecast", FIGURE_STEMS["forecast"], data_dir, meta)
        )

    if "superres" in figures:
        def build_superres():
            try:
                return build_paired_b_condition_cache(
                    args, run_dir, checkpoint_path, None, "superres", checkpoint_epoch
                )
            except RuntimeError:
                return build_paired_b_condition_cache(
                    args,
                    run_dir,
                    checkpoint_path,
                    ensure_ctx(),
                    "superres",
                    checkpoint_epoch,
                )

        arrays, meta = load_or_build_figure_cache(
            data_dir,
            FIGURE_STEMS["superres"],
            superres_signature(),
            builder=build_superres,
            force=args.force_recompute,
            provenance=provenance,
        )
        size_x, size_z = infer_spatial_size(
            run_dir,
            metadata=meta,
            ctx=ctx,
            args=args,
            data_dir=data_dir,
        )
        meta["size_x"], meta["size_z"] = size_x, size_z
        plot_density_superres_summary(
            arrays, meta, figures_dir, dpi=args.dpi, size_x=size_x, size_z=size_z
        )
        figure_manifest.append(
            manifest_figure_entry("superres", FIGURE_STEMS["superres"], data_dir, meta)
        )

    if "sliding" in figures:
        arrays, meta = load_or_build_figure_cache(
            data_dir,
            FIGURE_STEMS["sliding"],
            sliding_signature(),
            builder=lambda: build_sliding_appendix_cache(
                args, run_dir, ensure_ctx(), checkpoint_path, checkpoint_epoch
            ),
            force=args.force_recompute,
            provenance=provenance,
        )
        if ensure_sliding_qualitative_nrmse_metadata(arrays, meta):
            save_figure_cache(data_dir, FIGURE_STEMS["sliding"], arrays, meta)
            print(
                "Updated sliding qualitative Density NRMSE metadata: "
                f"{data_dir / (FIGURE_STEMS['sliding'] + '.json')}"
            )
        plot_sliding_window_appendix(
            arrays, meta, figures_dir, extent=args.extent, dpi=args.dpi
        )
        plot_sliding_journal_summaries(meta, figures_dir, data_dir, args.dpi)
        figure_manifest.append(
            manifest_figure_entry("sliding", FIGURE_STEMS["sliding"], data_dir, meta)
        )

    if any(name in figures for name in ("magnetic_ablation", "superres", "forecast")):
        _journal_arrays, journal_meta = ensure_journal_cache()
        plot_uncertainty_companions(journal_meta["records"], figures_dir, data_dir, args.dpi)

    if "parameters" in figures:
        _journal_arrays, journal_meta = ensure_journal_cache()
        groups = build_parameter_summary(journal_meta["records"])
        plot_parameter_summary(groups, figures_dir, data_dir, args.dpi)
        (data_dir / f"{FIGURE_STEMS['parameters']}.json").write_text(json.dumps(json_ready({
            "groups": groups, "definition": "marginal grouped performance summaries; macro-average over Figure-1 spatial masks",
            **provenance, "split": fixed_split_metadata(), "model_version": checkpoint_model_version,
        }), indent=2))
        figure_manifest.append(manifest_figure_entry("parameters", FIGURE_STEMS["parameters"], data_dir, {**journal_meta, **provenance}))

    if "global_uq" in figures:
        journal_arrays, journal_meta = ensure_journal_cache()
        plot_uncertainty_companions(journal_meta["records"], figures_dir, data_dir, args.dpi)
        result = plot_global_uncertainty(journal_arrays, journal_meta["records"], figures_dir, data_dir, args.dpi)
        (data_dir / f"{FIGURE_STEMS['global_uq']}.json").write_text(json.dumps(json_ready({
            **result, **provenance, "split": fixed_split_metadata(), "model_version": checkpoint_model_version,
        }), indent=2))
        figure_manifest.append(manifest_figure_entry("global_uq", FIGURE_STEMS["global_uq"], data_dir, {**journal_meta, **provenance}))

    manifest = {
        "paper_dir": str(paper_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "figures": figures,
        "figure_provenance": figure_manifest,
        "cache_version": PAPER_CACHE_VERSION,
        "force_recompute": bool(args.force_recompute),
        "split_path": fixed_split_metadata().get("path"),
        "split_sha256": fixed_split_metadata().get("sha256"),
        "data_role": "test_runs",
        "model_version": checkpoint_model_version,
    }
    (paper_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {paper_dir / 'manifest.json'}")
    if ctx is not None:
        ctx["dataset"].close()


if __name__ == "__main__":
    main()
