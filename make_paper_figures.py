# make_paper_figures.py
"""Paper-specific figures with a dedicated cache, separate from GIF tables."""

from __future__ import annotations

import argparse
import json
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

from data.vpic_hdf5_dataset import VPICWindowDataset, find_h5_files
from models.unet3d import LEGACY_MODEL_VERSION, UNet3D
from visualize_mask_patterns_unet3d import (
    DEFAULT_MAGNETIC_ABLATION_VISIBLE_FRACTIONS,
    DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS,
    DEFAULT_RESIDUAL_VMAX,
    DEFAULT_RUN_DIR,
    DEFAULT_RUN_NAME,
    DEFAULT_T0,
    JY_STATS_DEFINITION,
    RESIDUAL_CMAP,
    _density_probe_count_grid,
    build_density_forecast_rows,
    build_density_only_multifunction_rows,
    build_density_superres_rows,
    build_magnetic_ablation_rows,
    checkpoint_cache_signature,
    collect_validation_statistics,
    compute_normalized_metrics,
    default_density_forecast_visible_frames,
    denormalize,
    expand_path,
    get_train_runs,
    get_val_runs,
    jy_nrmse_from_residual,
    load_checkpoint,
    load_or_compute_jy_training_stats,
    make_nan_cmap,
    make_visible_input,
    normalize,
    normalized_residual,
    resolve_checkpoint_path,
    robust_limits,
    select_run_t0_index,
    select_validation_statistics_indices,
)
from visualize_sliding_density_reconstruction import (
    find_and_load_run,
    framewise_rmse_mae,
    project_time_z,
    reconstruct_with_slide_step,
)


PAPER_CACHE_VERSION = 8
COMPATIBLE_PAPER_CACHE_VERSIONS = (1, 2, 3, 4, 5, 6, 7, 8)
PAPER_DIRNAME = "paper_figures_v1"
DEFAULT_GLOBAL_FRAME = 45
DEFAULT_SLIDE_STEPS = (1, 12, 24)
DEFAULT_PROBE_COUNTS = (0, 10, 100, 1000)
DEFAULT_X_INDEX = 130
DEFAULT_DENSITY_VISIBLE_FRACTION = 0.08
DEFAULT_SLIDING_MAX_RUNS = 16
DEFAULT_SLIDING_DENSITY_PROBE_COUNT = 1000
SLIDING_AGGREGATE_N_FRAMES = 48
LOG_Y_FLOOR = 1e-6
PNG_DPI = 300

SPATIAL_ROW_ORDER = (
    "spatial_random",
    "spatial_grid",
    "spatial_block_inpainting",
    "spatial_block_outpainting",
)
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

DENSITY_NRMSE_DEFINITION = (
    "RMS of (prediction_normalized - target_normalized) on the Density channel"
)
JY_NRMSE_DEFINITION = (
    "RMSE(Jy_pred - Jy_target) / training-set Jy std, with Jy = dBx/dz - dBz/dx "
    "on checkpoint-standardized Bx,Bz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render paper figures from cached arrays, recomputing only if needed."
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default="latest.pt")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--paper-dir", default=None)
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
        default=DEFAULT_SLIDING_MAX_RUNS,
        help=(
            "Maximum held-out validation runs for sliding appendix left-panel "
            "RMSE stats, taken in split.json order. Runs shorter than 48 frames "
            "are skipped; remaining runs use only the first 48 frames."
        ),
    )
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
    split_path = run_dir / "split.json"
    if not split_path.exists():
        return []
    return [str(name) for name in json.loads(split_path.read_text()).get("val_runs", [])]


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
    """Take the first ``max_runs`` validation runs in ``split.json`` order."""
    if int(max_runs) <= 0:
        raise ValueError(f"sliding max runs must be positive, got {max_runs}")
    return [str(name) for name in val_run_order][: int(max_runs)]


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
    fig.savefig(png_path, dpi=dpi)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def base_signature(args: argparse.Namespace, run_dir: Path, checkpoint_path: Path) -> Dict:
    split_path = run_dir / "split.json"
    split = json.loads(split_path.read_text()) if split_path.exists() else {}
    return {
        "cache_version": PAPER_CACHE_VERSION,
        "checkpoint": checkpoint_cache_signature(checkpoint_path),
        "run_name": args.run_name,
        "t0": int(args.t0),
        "global_frame": int(args.global_frame),
        "seed": int(args.seed),
        "extent": [float(value) for value in args.extent],
        "val_runs": sorted(split.get("val_runs", [])),
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
        "median": float(np.median(values)),
        "p16": float(np.percentile(values, 16.0)),
        "p84": float(np.percentile(values, 84.0)),
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


def read_checkpoint_epoch(checkpoint_path: Path) -> int:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    ckpt, ckpt_path = load_checkpoint(run_dir, args.checkpoint, device=device)
    ckpt_args = ckpt["args"]
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
    val_runs = get_val_runs(run_dir)
    train_runs = get_train_runs(run_dir)
    if train_runs is None:
        raise FileNotFoundError("Jy stats require split.json train_runs.")
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
    jy_stats = load_or_compute_jy_training_stats(
        run_dir=run_dir,
        dataset=dataset,
        train_runs=train_runs,
        mean=mean,
        std=std,
        extent=args.extent,
    )
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=int(ckpt_args.get("base_channels", 16)),
        channel_mults=ckpt_args.get("channel_mults", [1, 2, 4]),
        architecture=ckpt_args.get("model_version", LEGACY_MODEL_VERSION),
        use_attention=bool(ckpt_args.get("use_attention", False)),
        spatial_only_pooling=bool(ckpt_args.get("spatial_only_pooling", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
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
        x_visible = make_visible_input(y_norm, mask)
        pred_norm = ctx["model"](torch.cat([x_visible, mask], dim=1))
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
        arrays[f"{name}_visible"] = np.asarray(visible, dtype=np.float32)
        arrays[f"{name}_prediction"] = np.asarray(pred, dtype=np.float32)
        arrays[f"{name}_residual"] = np.asarray(residual, dtype=np.float32)
        arrays[f"{name}_mask"] = np.asarray(density_mask[local_time], dtype=np.float32)
        row_meta.append(
            {
                "name": name,
                "density_masked_fraction": float(1.0 - density_mask.mean()),
                "density_nrmse": density_nrmse_from_residual(residual),
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
        "layout": {"rows": 4, "columns": 4},
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
    n_rows, n_cols = 4, 4
    if len(row_names) != n_rows:
        raise ValueError(f"Expected {n_rows} spatial rows, got {row_names}")
    field_arrays = [arrays["target_density"]]
    residual_arrays = []
    for name in row_names:
        field_arrays.extend([arrays[f"{name}_visible"], arrays[f"{name}_prediction"]])
        residual_arrays.append(arrays[f"{name}_residual"])
    vmin, vmax = robust_limits(field_arrays, channel=3, q=field_q, symmetric=False)
    res_vmin, res_vmax = -float(residual_vmax), float(residual_vmax)
    field_cmap = make_nan_cmap("viridis")
    residual_cmap = make_nan_cmap(RESIDUAL_CMAP)

    fig = plt.figure(figsize=(7.4, 8.0))
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        wspace=0.05,
        hspace=0.08,
        left=0.16,
        right=0.84,
        top=0.90,
        bottom=0.06,
    )
    col_titles = ["Target Density", "Visible Density", "Prediction", "Residual"]
    nrmse_by_name = {
        str(row.get("name")): row.get("density_nrmse")
        for row in metadata.get("rows") or []
        if isinstance(row, dict)
    }
    field_im = None
    residual_im = None
    for r, name in enumerate(row_names):
        panels = [
            arrays["target_density"],
            arrays[f"{name}_visible"],
            arrays[f"{name}_prediction"],
            arrays[f"{name}_residual"],
        ]
        for c, panel in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
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
                    cmap=residual_cmap,
                    vmin=res_vmin,
                    vmax=res_vmax,
                    interpolation="nearest",
                )
                residual_im = im
                nrmse = nrmse_by_name.get(name)
                if nrmse is None:
                    nrmse = density_nrmse_from_residual(panel)
                annotate_density_nrmse(ax, float(nrmse))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], pad=4)
            if c == 0:
                ax.set_ylabel(SPATIAL_ROW_LABELS[name], fontsize=9)
    cax_field = fig.add_axes([0.86, 0.52, 0.018, 0.36])
    cax_res = fig.add_axes([0.86, 0.08, 0.018, 0.36])
    cb_field = fig.colorbar(field_im, cax=cax_field)
    cb_field.set_label("Density", fontsize=8)
    cb_res = fig.colorbar(residual_im, cax=cax_res)
    cb_res.set_label("Normalized residual", fontsize=8)
    fig.suptitle(
        "B fully observed  ·  Density ~50% masked  ·  "
        f"global frame {metadata['global_frame']}",
        fontsize=10,
        y=0.98,
    )
    save_png_pdf(fig, figures_dir, FIGURE_STEMS["spatial"], dpi)
    return n_rows, n_cols


# ---------------------------------------------------------------------------
# Quantitative summary figures from validation JSON or fresh aggregation
# ---------------------------------------------------------------------------


def build_magnetic_ablation_cache(
    args: argparse.Namespace,
    run_dir: Path,
    checkpoint_path: Path,
    ctx: Dict | None,
    checkpoint_epoch: int,
) -> Tuple[Dict, Dict]:
    payload = None if args.force_recompute else try_load_validation_json(
        run_dir,
        checkpoint_path,
        "magnetic_ablation",
        hide_magnetic=False,
        checkpoint_epoch=checkpoint_epoch,
    )
    source = "validation_json"
    source_files: List[str] = []
    if payload is None:
        if ctx is None:
            raise RuntimeError("Magnetic ablation cache needs inference context.")
        sample, y = select_named_window(ctx, args)
        generator = torch.Generator().manual_seed(args.seed)
        mask_rows = build_magnetic_ablation_rows(
            block=y,
            magnetic_visible_fractions=list(DEFAULT_MAGNETIC_ABLATION_VISIBLE_FRACTIONS),
            generator=generator,
        )
        stats = collect_experiment_statistics(
            ctx,
            args,
            {"magnetic_ablation": row_dicts_from_masks(mask_rows)},
        )["magnetic_ablation"]
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
        source = "collect_validation_statistics"
        run_count = len(rows[0]["density"]["run_names"]) if rows else 0
    else:
        rows = []
        for row in payload["rows"]:
            visible_fraction = parse_trailing_number(row["name"])
            rows.append(
                {
                    "name": row["name"],
                    "visible_fraction": visible_fraction,
                    "density": summarize_run_scalars(_profiles_from_val_row(row, "density")),
                    "jy": summarize_run_scalars(_profiles_from_val_row(row, "jy")),
                }
            )
        run_count = int(payload.get("run_count", len(rows[0]["density"]["run_names"])))
        source_files = [str(payload.get("_source_json"))]

    rows.sort(key=lambda item: item["visible_fraction"])
    percents = np.asarray(
        [100.0 * float(row["visible_fraction"]) for row in rows], dtype=np.float64
    )
    arrays = {
        "b_visible_percent": percents,
        "density_median": np.asarray([row["density"]["median"] for row in rows]),
        "density_p16": np.asarray([row["density"]["p16"] for row in rows]),
        "density_p84": np.asarray([row["density"]["p84"] for row in rows]),
        "jy_median": np.asarray([row["jy"]["median"] for row in rows]),
        "jy_p16": np.asarray([row["jy"]["p16"] for row in rows]),
        "jy_p84": np.asarray([row["jy"]["p84"] for row in rows]),
        "density_per_run": np.asarray([row["density"]["per_run"] for row in rows]),
        "jy_per_run": np.asarray([row["jy"]["per_run"] for row in rows]),
    }
    meta = {
        "figure": "magnetic_ablation_summary",
        "source": source,
        "density_visible": 0.0,
        "b_visible_percents": percents.tolist(),
        "n_levels": len(rows),
        "run_count": run_count,
        "rows": rows,
        "density_nrmse_definition": DENSITY_NRMSE_DEFINITION,
        "jy_nrmse_definition": JY_NRMSE_DEFINITION,
        "jy_definition": JY_STATS_DEFINITION,
        "interval": "median and 16th-84th percentile range",
        "source_files": source_files,
    }
    return arrays, meta


def plot_magnetic_ablation_summary(arrays: Dict, metadata: Dict, figures_dir: Path, dpi: int) -> None:
    percents = np.asarray(arrays["b_visible_percent"])
    x = np.arange(len(percents))
    labels = [f"{value:g}" for value in percents]
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.6), sharex=True)
    series = (
        (axes[0], "Density NRMSE", "density_median", "density_p16", "density_p84"),
        (axes[1], "Jy NRMSE", "jy_median", "jy_p16", "jy_p84"),
    )
    for ax, ylabel, med_key, p16_key, p84_key in series:
        ax.fill_between(
            x,
            clip_positive_for_log(arrays[p16_key]),
            clip_positive_for_log(arrays[p84_key]),
            color="C0",
            alpha=0.22,
            linewidth=0,
        )
        ax.plot(
            x,
            clip_positive_for_log(arrays[med_key]),
            color="C0",
            marker="o",
            linewidth=1.8,
            markersize=5,
        )
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("B visible (%)")
    fig.suptitle("Density fully hidden", fontsize=10, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    save_png_pdf(fig, figures_dir, FIGURE_STEMS["magnetic_ablation"], dpi)


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
    for hide_magnetic, key in ((False, "B_full"), (True, "B_hidden")):
        payload = None if args.force_recompute else try_load_validation_json(
            run_dir,
            checkpoint_path,
            experiment,
            hide_magnetic=hide_magnetic,
            checkpoint_epoch=checkpoint_epoch,
        )
        if payload is not None:
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
    return arrays, meta


def plot_density_forecast_summary(arrays: Dict, metadata: Dict, figures_dir: Path, dpi: int) -> None:
    frames = np.asarray(arrays["local_frames"])
    full_rows = metadata["rows"]["B_full"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.2), sharex=True)
    for index, row in enumerate(full_rows):
        color = colors[index % len(colors)]
        history = int(row["history"])
        horizon = int(metadata["context_length"]) - history
        for key, linestyle, alpha in (("B_full", "-", 0.18), ("B_hidden", "--", 0.08)):
            tag = f"{key}_{row['name']}"
            for ax, metric in ((axes[0], "density"), (axes[1], "jy")):
                ax.fill_between(
                    frames,
                    clip_positive_for_log(arrays[f"{tag}_{metric}_p16"]),
                    clip_positive_for_log(arrays[f"{tag}_{metric}_p84"]),
                    color=color,
                    alpha=alpha,
                    linewidth=0.4,
                    edgecolor="black",
                )
                ax.plot(
                    frames,
                    clip_positive_for_log(arrays[f"{tag}_{metric}_median"]),
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=f"horizon={horizon}" if key == "B_full" else None,
                )
    axes[0].set_ylabel("Density NRMSE")
    axes[1].set_ylabel("Jy NRMSE")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(
        f"Local frame in {int(metadata['context_length'])}-frame context window"
    )
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
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
    save_png_pdf(fig, figures_dir, FIGURE_STEMS["forecast"], dpi)


def plot_density_superres_summary(
    arrays: Dict,
    metadata: Dict,
    figures_dir: Path,
    dpi: int,
    size_x: int,
    size_z: int,
) -> None:
    full_rows = sorted(metadata["rows"]["B_full"], key=lambda row: int(row["probe_count"]))
    probe_counts = [int(row["probe_count"]) for row in full_rows]
    ratios = [100.0 * density_visible_ratio(count, size_x, size_z) for count in probe_counts]
    metadata["size_x"] = int(size_x)
    metadata["size_z"] = int(size_z)
    metadata["visible_ratio_percent"] = ratios
    x = np.arange(len(probe_counts))
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.8), sharex=True)
    styles = {
        "B_full": dict(color="C0", linestyle="-", marker="o", label="B visible 100%"),
        "B_hidden": dict(color="C0", linestyle="--", marker="^", label="B visible 0%"),
    }
    for key, style in styles.items():
        rows_by_count = {
            int(row["probe_count"]): row for row in metadata["rows"][key]
        }
        density = [rows_by_count[count]["density_summary"]["median"] for count in probe_counts]
        density_p16 = [rows_by_count[count]["density_summary"]["p16"] for count in probe_counts]
        density_p84 = [rows_by_count[count]["density_summary"]["p84"] for count in probe_counts]
        jy = [rows_by_count[count]["jy_summary"]["median"] for count in probe_counts]
        jy_p16 = [rows_by_count[count]["jy_summary"]["p16"] for count in probe_counts]
        jy_p84 = [rows_by_count[count]["jy_summary"]["p84"] for count in probe_counts]
        axes[0].fill_between(
            x,
            clip_positive_for_log(density_p16),
            clip_positive_for_log(density_p84),
            color=style["color"],
            alpha=0.18,
            linewidth=0,
        )
        axes[1].fill_between(
            x,
            clip_positive_for_log(jy_p16),
            clip_positive_for_log(jy_p84),
            color=style["color"],
            alpha=0.18,
            linewidth=0,
        )
        plot_style = {k: v for k, v in style.items() if k != "label"}
        axes[0].plot(
            x,
            clip_positive_for_log(density),
            linewidth=1.8,
            **plot_style,
            label=style["label"],
        )
        axes[1].plot(
            x,
            clip_positive_for_log(jy),
            linewidth=1.8,
            **plot_style,
        )
    axes[0].set_ylabel("Density NRMSE")
    axes[1].set_ylabel("Jy NRMSE")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Density visible ratio (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([format_visible_ratio_percent(ratio) for ratio in ratios])
    for ax in axes:
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_png_pdf(fig, figures_dir, FIGURE_STEMS["superres"], dpi)


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
    return frame_index, medians, p16s, p84s, stacked_by, loaded, skipped


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
    if ctx is None:
        raise RuntimeError("Sliding multi-run RMSE needs inference context.")
    slide_steps = [int(step) for step in args.slide_steps]
    candidate_runs = validation_run_order(run_dir)
    if not candidate_runs:
        raise RuntimeError(f"No validation runs in {run_dir / 'split.json'}")
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
        ctx, args, [representative_step], n_frames=SLIDING_AGGREGATE_N_FRAMES
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
        "left_panel": "multi_run_framewise_rmse",
        "right_panel": "single_run_qualitative",
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
            f"{len(loaded_runs)} validation runs; one framewise RMSE series per run "
            f"from the first {SLIDING_AGGREGATE_N_FRAMES} frames, B condition, "
            "and slide step; then median and 16th-84th percentiles across runs"
        ),
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
) -> None:
    steps = [int(step) for step in metadata["slide_steps"]]
    frame_ids = arrays["frame_ids"]
    aggregate_frames = arrays.get("aggregate_frame_ids", frame_ids)
    fig = plt.figure(figsize=(8.4, 4.6))
    gs = gridspec.GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.05, 1.15],
        wspace=0.28,
        hspace=0.08,
        left=0.08,
        right=0.92,
        top=0.90,
        bottom=0.12,
    )
    ax_err = fig.add_subplot(gs[:, 0])
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
                color=color,
                alpha=alpha,
                linewidth=0,
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
    color_handles = [
        Line2D(
            [0],
            [0],
            color=colors[index % len(colors)],
            linewidth=1.6,
            label=f"step={step}",
        )
        for index, step in enumerate(steps)
    ]
    style_handles = [
        Line2D([0], [0], color="0.3", linestyle="-", linewidth=1.6, label="B visible 100%"),
        Line2D([0], [0], color="0.3", linestyle="--", linewidth=1.6, label="B visible 0%"),
    ]
    color_legend = ax_err.legend(
        handles=color_handles, frameon=False, fontsize=7, loc="upper left"
    )
    ax_err.add_artist(color_legend)
    ax_err.legend(handles=style_handles, frameon=False, fontsize=7, loc="upper right")
    n_runs = metadata.get("aggregate_n_runs")
    n_frames = metadata.get("n_frames")
    probe_count = metadata.get("density_probe_count")
    ratio_percent = metadata.get("density_visible_ratio_percent")
    run_phrase = (
        f"{int(n_runs)} validation runs, first {int(n_frames)} frames"
        if n_runs and n_frames
        else (f"{int(n_runs)} validation runs" if n_runs else "Validation runs")
    )
    if probe_count is not None and ratio_percent is not None:
        title = (
            f"{run_phrase}\n"
            f"{int(probe_count)} fixed Density probes "
            f"({format_visible_ratio_percent(float(ratio_percent))}% spatial visibility)"
        )
    else:
        title = run_phrase
    ax_err.set_title(title, fontsize=9)

    step = int(metadata["representative_step"])
    panels = [
        ("Target", arrays["target_tz"], "viridis", False),
        (f"Recon. step={step}", arrays[f"step_{step}_tz"], "viridis", False),
        ("Residual", arrays[f"step_{step}_tz_residual"], RESIDUAL_CMAP, True),
    ]
    zmin, zmax, _xmin, _xmax = extent
    tmin, tmax = float(frame_ids.min()) - 0.5, float(frame_ids.max()) + 0.5
    field_vmin, field_vmax = robust_limits(
        [arrays["target_tz"], arrays[f"step_{step}_tz"]],
        channel=3,
        q=99.0,
        symmetric=False,
    )
    res_vmin, res_vmax = robust_limits(
        [arrays[f"step_{step}_tz_residual"]],
        channel=3,
        q=99.0,
        symmetric=True,
    )
    for index, (title, panel, cmap, symmetric) in enumerate(panels):
        ax = fig.add_subplot(gs[index, 1])
        vmin, vmax = (res_vmin, res_vmax) if symmetric else (field_vmin, field_vmax)
        im = ax.imshow(
            panel.T,
            origin="lower",
            aspect="auto",
            extent=[tmin, tmax, zmin, zmax],
            cmap=make_nan_cmap(cmap),
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_ylabel("z [cm]", fontsize=8)
        if index == 2:
            ax.set_xlabel("Global frame")
        else:
            ax.tick_params(labelbottom=False)
        ax.set_title(title, fontsize=9, pad=2)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    save_png_pdf(fig, figures_dir, FIGURE_STEMS["sliding"], dpi)


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
    checkpoint_epoch = read_checkpoint_epoch(checkpoint_path)
    provenance = provenance_for_checkpoint(checkpoint_path, checkpoint_epoch)
    paper_dir = (
        expand_path(args.paper_dir)
        if args.paper_dir is not None
        else run_dir / PAPER_DIRNAME
    )
    data_dir = paper_dir / "data"
    figures_dir = paper_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    figures = selected_figures(args.figure)
    sig_base = base_signature(args, run_dir, checkpoint_path)
    ctx: Dict | None = None
    figure_manifest: List[Dict] = []

    def ensure_ctx() -> Dict:
        nonlocal ctx
        if ctx is None:
            ctx = load_inference_context(args)
        return ctx

    def spatial_signature() -> Dict:
        return {
            **sig_base,
            "figure": "spatial",
            "b_visible": 1.0,
            "rows": list(SPATIAL_ROW_ORDER),
            "mask_experiment": "multifunction_standardized_50pct",
        }

    def mag_signature() -> Dict:
        return {
            **sig_base,
            "figure": "magnetic_ablation",
            "density_visible": 0.0,
            "b_visible_levels": list(DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS),
        }

    def forecast_signature() -> Dict:
        return {
            **sig_base,
            "figure": "forecast",
            "histories": default_density_forecast_visible_frames(24),
            "b_conditions": ["B_full", "B_hidden"],
        }

    def superres_signature() -> Dict:
        return {
            **sig_base,
            "figure": "superres",
            "probe_counts": [int(value) for value in args.density_probe_counts],
            "b_conditions": ["B_full", "B_hidden"],
        }

    def sliding_signature() -> Dict:
        signature = {
            **sig_base,
            "figure": "sliding",
            "slide_steps": [int(step) for step in args.slide_steps],
            "density_probe_count": int(args.density_probe_count),
            "probe_layout_semantics": "same as density_superres",
            "x_index": int(args.x_index),
            "left_panel": "multi_run_framewise_rmse",
            "right_panel": "single_run_qualitative",
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
        plot_sliding_window_appendix(
            arrays, meta, figures_dir, extent=args.extent, dpi=args.dpi
        )
        figure_manifest.append(
            manifest_figure_entry("sliding", FIGURE_STEMS["sliding"], data_dir, meta)
        )

    manifest = {
        "paper_dir": str(paper_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "figures": figures,
        "figure_provenance": figure_manifest,
        "cache_version": PAPER_CACHE_VERSION,
        "force_recompute": bool(args.force_recompute),
    }
    (paper_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {paper_dir / 'manifest.json'}")
    if ctx is not None:
        ctx["dataset"].close()


if __name__ == "__main__":
    main()
