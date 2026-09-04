from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from data.vpic_hdf5_dataset import find_h5_files, parse_run_name
from models.unet3d import LEGACY_MODEL_VERSION, UNet3D
from visualize_mask_patterns_unet3d import (
    DEFAULT_RESIDUAL_VMAX,
    RESIDUAL_CMAP,
    _density_probe_grid,
    compute_normalized_metrics,
    make_nan_cmap,
    normalized_residual,
    robust_limits,
    write_animation,
)


DEFAULT_RUN_NAME = "beta0.2_nu2_Bz0_dt2_tau70"


def framewise_rmse_mae(prediction: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return spatial RMSE and MAE for every time frame."""
    reduce_axes = tuple(range(1, target.ndim))
    error = np.asarray(prediction) - np.asarray(target)
    return (
        np.sqrt(np.mean(np.square(error), axis=reduce_axes)),
        np.mean(np.abs(error), axis=reduce_axes),
    )


def save_framewise_error_plot(
    predictions: Sequence[np.ndarray],
    labels: Sequence[str],
    target: np.ndarray,
    frame_ids: np.ndarray,
    out_path: Path,
    title: str,
    plot_units: str,
) -> List[Dict]:
    """Plot one error curve per GIF row and return JSON-ready values."""
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    payload = []
    for prediction, label in zip(predictions, labels):
        rmse, mae = framewise_rmse_mae(prediction, target)
        axes[0].plot(frame_ids, rmse, linewidth=1.8, label=label)
        axes[1].plot(frame_ids, mae, linewidth=1.8, label=label)
        payload.append({"label": label, "rmse": rmse.tolist(), "mae": mae.tolist()})
    axes[0].set_ylabel(f"Frame RMSE ({plot_units})")
    axes[1].set_ylabel(f"Frame MAE ({plot_units})")
    axes[1].set_xlabel("Global frame")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle(title, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved framewise error plot: {out_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct one complete VPIC run with recursively conditioned "
            "sliding T-frame windows and compare several slide steps."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--h5-dir", default=None)
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=(
            "Complete HDF5 run to reconstruct. The default is the 52-frame "
            "held-out two-plasmoid merger run with beta=0.2."
        ),
    )
    parser.add_argument(
        "--allow-non-validation-run",
        action="store_true",
        help="Allow analysis of a run absent from run-dir/split.json val_runs.",
    )
    parser.add_argument(
        "--slide-steps",
        type=int,
        nargs="+",
        default=[24, 12, 6, 3],
    )
    parser.add_argument(
        "--analysis",
        choices=["both", "slide_steps", "bidirectional"],
        default="both",
        help=(
            "Analyses to render. 'both' preserves the slide-step comparison "
            "and adds the bidirectional repeated-sweep comparison."
        ),
    )
    parser.add_argument(
        "--refinement-step",
        type=int,
        default=12,
        help="Slide step for the 1-to-N-pass bidirectional rows.",
    )
    parser.add_argument(
        "--refinement-passes",
        type=int,
        default=4,
        help="Maximum alternating L-to-R/R-to-L passes (default: 4).",
    )
    parser.add_argument(
        "--refinement-offset",
        type=int,
        default=12,
        help=(
            "Alternating offset for the step=T shifted-window control row "
            "(default: 12)."
        ),
    )
    parser.add_argument(
        "--density-visible-fraction",
        type=float,
        default=0.08,
        help="Target fraction of fixed Density super-resolution probes.",
    )
    parser.add_argument(
        "--hide-magnetic",
        action="store_true",
        help=(
            "Hide all Bx/By/Bz observations so reconstruction uses only the "
            "fixed Density probes."
        ),
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--x-index",
        type=int,
        default=None,
        help=(
            "Plot Density(time,z) at this x index. By default Density is "
            "averaged over x. Full 3D arrays are always used for metrics."
        ),
    )
    parser.add_argument(
        "--plot-units",
        choices=["physical", "normalized"],
        default="physical",
    )
    parser.add_argument("--field-q", type=float, default=99.0)
    parser.add_argument(
        "--residual-q",
        type=float,
        default=99.0,
        help="Percentile used with --auto-residual-range.",
    )
    parser.add_argument(
        "--residual-vmax",
        type=float,
        default=DEFAULT_RESIDUAL_VMAX,
        help=(
            "Fixed symmetric normalized-residual color limit. "
            f"Default: {DEFAULT_RESIDUAL_VMAX}."
        ),
    )
    parser.add_argument(
        "--auto-residual-range",
        action="store_const",
        const=None,
        dest="residual_vmax",
        default=argparse.SUPPRESS,
        help="Use robust percentile-based residual limits instead of a fixed range.",
    )
    parser.add_argument(
        "--animation-format",
        choices=["quicktime", "mp4", "gif", "both"],
        default="gif",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--animation-dpi",
        type=int,
        default=100,
        help="DPI for animation frames; final static PNG continues to use --dpi.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=[-21.0, 21.0, -50.0, 50.0],
        metavar=("ZMIN", "ZMAX", "XMIN", "XMAX"),
    )
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def sliding_window_starts(
    run_length: int,
    window_size: int,
    slide_step: int,
) -> List[int]:
    """
    Start at zero and move forward by slide_step until the run is covered.

    The final window may extend beyond the run. Missing future slots are fully
    masked during inference and discarded from the reconstruction afterward.
    """
    if run_length < window_size:
        raise ValueError(
            f"run length {run_length} is shorter than window size {window_size}."
        )
    if not (1 <= slide_step <= window_size):
        raise ValueError(
            f"slide step must be in [1, {window_size}], got {slide_step}."
        )

    starts = [0]
    while starts[-1] + window_size < run_length:
        starts.append(starts[-1] + slide_step)
    return starts


def shifted_window_starts(
    run_length: int,
    window_size: int,
    slide_step: int,
    offset: int,
    include_boundary_padding: bool = False,
) -> List[int]:
    """Translate the ordinary forward schedule by a non-negative offset."""
    if not (0 <= offset < slide_step):
        raise ValueError(
            f"offset must lie in [0, {slide_step - 1}], got {offset}."
        )
    if not include_boundary_padding or offset == 0:
        return [
            start + offset
            for start in sliding_window_starts(
                run_length, window_size, slide_step
            )
            if start + offset < run_length
        ]

    # Include the translated window immediately left of t=0. Its negative
    # slots are fully masked, while its physical part reconstructs the run's
    # leading boundary. For T=150, W=step=24, offset=12 this yields
    # [-12, 12, 36, 60, 84, 108, 132], matching the seven offset=0 calls.
    first_start = offset - slide_step
    starts = []
    start = first_start
    while start < run_length:
        if start + window_size > 0:
            starts.append(start)
        start += slide_step
    return starts


def project_time_z(density: np.ndarray, x_index: int | None) -> np.ndarray:
    """Project Density(T,X,Z) to Density(T,Z) by x mean or one x slice."""
    if density.ndim != 3:
        raise ValueError(f"Expected Density(T,X,Z), got {density.shape}.")
    if x_index is None:
        # Future time slices are intentionally all-NaN until reconstructed.
        # Compute the finite mean explicitly to avoid all-NaN warnings.
        finite = np.isfinite(density)
        count = finite.sum(axis=1)
        total = np.where(finite, density, 0.0).sum(axis=1)
        projection = np.full(total.shape, np.nan, dtype=np.float64)
        np.divide(total, count, out=projection, where=count > 0)
        return projection
    if not (0 <= x_index < density.shape[1]):
        raise ValueError(
            f"--x-index must be in [0, {density.shape[1] - 1}], got {x_index}."
        )
    return density[:, x_index, :]


def rmse_mae(prediction: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    error = np.asarray(prediction) - np.asarray(target)
    finite = np.isfinite(error)
    if not finite.any():
        return float("nan"), float("nan")
    error = error[finite]
    return float(np.sqrt(np.mean(error**2))), float(np.mean(np.abs(error)))


def nrmse_nmae(
    prediction_normalized: np.ndarray,
    target_normalized: np.ndarray,
) -> Tuple[float, float]:
    residual = normalized_residual(prediction_normalized, target_normalized)
    return compute_normalized_metrics(residual)


def find_and_load_run(
    h5_dir: Path,
    betas: Sequence[float],
    run_name: str,
) -> Tuple[np.ndarray, np.ndarray, Path]:
    for h5_path in find_h5_files(h5_dir, betas=betas):
        with h5py.File(h5_path, "r") as h5_file:
            if run_name not in h5_file["runs"]:
                continue
            group = h5_file["runs"][run_name]
            fields = np.asarray(group["fields"], dtype=np.float32)
            frame_ids = np.asarray(group["frame_ids"], dtype=np.int64)
            return fields, frame_ids, h5_path
    raise KeyError(f"Run {run_name!r} was not found in {h5_dir} for betas={betas}.")


def validate_test_run(run_dir: Path, run_name: str, allow_non_validation: bool) -> None:
    split_path = run_dir / "split.json"
    if not split_path.exists():
        if allow_non_validation:
            return
        raise FileNotFoundError(
            f"Cannot verify a test run because {split_path} does not exist."
        )
    split = json.loads(split_path.read_text())
    if run_name not in set(split["val_runs"]) and not allow_non_validation:
        raise ValueError(
            f"{run_name!r} is not in the checkpoint validation split. Use a "
            "validation run or pass --allow-non-validation-run explicitly."
        )


def convert_density_units(
    density_normalized: torch.Tensor,
    density_mean: torch.Tensor,
    density_std: torch.Tensor,
    plot_units: str,
) -> torch.Tensor:
    if plot_units == "normalized":
        return density_normalized
    return density_normalized * (density_std + 1e-8) + density_mean


@torch.no_grad()
def reconstruct_with_slide_step(
    model: torch.nn.Module,
    target_normalized: torch.Tensor,
    target_density_plot: np.ndarray,
    density_mean: torch.Tensor,
    density_std: torch.Tensor,
    probe_mask: torch.Tensor,
    window_size: int,
    slide_step: int,
    plot_units: str,
    x_index: int | None,
    amp: bool,
    retain_state: bool = False,
    hide_magnetic: bool = False,
) -> Dict:
    """Run recursive reconstruction and retain a compact projection per update."""
    channels, run_length, size_x, size_z = target_normalized.shape
    if channels != 4:
        raise ValueError(f"Expected four target channels, got {channels}.")

    starts = sliding_window_starts(run_length, window_size, slide_step)
    raw_reconstruction = torch.full(
        (run_length, size_x, size_z),
        float("nan"),
        device=target_normalized.device,
        dtype=target_normalized.dtype,
    )
    conditioning_reconstruction = raw_reconstruction.clone()
    probe_bool = probe_mask > 0.5
    target_density_normalized = target_normalized[3].detach().cpu().numpy()
    snapshots = []
    covered_end = 0

    for update_index, start in enumerate(starts):
        valid_end = min(start + window_size, run_length)
        valid_length = valid_end - start
        values = torch.zeros(
            (1, 4, window_size, size_x, size_z),
            device=target_normalized.device,
            dtype=target_normalized.dtype,
        )
        mask = torch.zeros_like(values)

        # By default all physically available magnetic data are true
        # observations. --hide-magnetic leaves Bx/By/Bz invisible so the
        # reconstruction is conditioned only on Density probes. Padded slots
        # beyond the end of the run remain zero with mask=0.
        if not hide_magnetic:
            values[0, :3, :valid_length] = target_normalized[:3, start:valid_end]
            mask[0, :3, :valid_length] = 1.0

        density_target = target_normalized[3, start:valid_end]
        values[0, 3, :valid_length] = density_target * probe_mask
        mask[0, 3, :valid_length] = probe_mask

        overlap_end = min(covered_end, valid_end)
        overlap_length = max(0, overlap_end - start)
        if overlap_length > 0:
            values[0, 3, :overlap_length] = conditioning_reconstruction[
                start:overlap_end
            ]
            mask[0, 3, :overlap_length] = 1.0

        # Real probes always win in the state passed to subsequent windows.
        density_values = values[0, 3, :valid_length]
        density_values[:, probe_bool] = density_target[:, probe_bool]
        values[0, 3, :valid_length] = density_values

        model_input = torch.cat([values * mask, mask], dim=1)
        with torch.autocast(
            device_type=target_normalized.device.type,
            enabled=(amp and target_normalized.device.type == "cuda"),
        ):
            prediction = model(model_input)

        # Preserve the raw prediction everywhere, including at true probes, for
        # figures and residual metrics. Use a separately clamped copy only as
        # pseudo-visible conditioning for later overlapping windows.
        raw_window = prediction[0, 3, :valid_length].float()
        raw_reconstruction[start:valid_end] = raw_window
        conditioned_window = raw_window.clone()
        conditioned_window[:, probe_bool] = density_target[:, probe_bool]
        conditioning_reconstruction[start:valid_end] = conditioned_window
        covered_end = max(covered_end, valid_end)

        raw_plot_tensor = convert_density_units(
            raw_reconstruction,
            density_mean=density_mean,
            density_std=density_std,
            plot_units=plot_units,
        )
        raw_plot = raw_plot_tensor.detach().cpu().numpy()
        raw_normalized = raw_reconstruction.detach().cpu().numpy()
        prediction_projection = project_time_z(raw_plot, x_index=x_index)
        residual_projection = project_time_z(
            normalized_residual(raw_normalized, target_density_normalized),
            x_index=x_index,
        )
        covered_rmse, covered_mae = rmse_mae(
            raw_plot[:covered_end], target_density_plot[:covered_end]
        )
        last_slice_rmse, last_slice_mae = rmse_mae(
            raw_plot[covered_end - 1], target_density_plot[covered_end - 1]
        )
        current_window_rmse, current_window_mae = rmse_mae(
            raw_plot[start:valid_end], target_density_plot[start:valid_end]
        )
        covered_nrmse, covered_nmae = nrmse_nmae(
            raw_normalized[:covered_end], target_density_normalized[:covered_end]
        )
        last_slice_nrmse, last_slice_nmae = nrmse_nmae(
            raw_normalized[covered_end - 1],
            target_density_normalized[covered_end - 1],
        )
        current_window_nrmse, current_window_nmae = nrmse_nmae(
            raw_normalized[start:valid_end],
            target_density_normalized[start:valid_end],
        )
        snapshots.append(
            {
                "update_index": update_index,
                "window_start": start,
                "valid_end": valid_end,
                "covered_end": covered_end,
                "prediction_projection": prediction_projection,
                "residual_projection": residual_projection,
                "covered_rmse": covered_rmse,
                "covered_mae": covered_mae,
                "last_slice_rmse": last_slice_rmse,
                "last_slice_mae": last_slice_mae,
                "current_window_rmse": current_window_rmse,
                "current_window_mae": current_window_mae,
                "covered_nrmse": covered_nrmse,
                "covered_nmae": covered_nmae,
                "last_slice_nrmse": last_slice_nrmse,
                "last_slice_nmae": last_slice_nmae,
                "current_window_nrmse": current_window_nrmse,
                "current_window_nmae": current_window_nmae,
            }
        )

    final_plot = convert_density_units(
        raw_reconstruction,
        density_mean=density_mean,
        density_std=density_std,
        plot_units=plot_units,
    ).detach().cpu().numpy()
    final_window_start = starts[-1]
    final_window_rmse, final_window_mae = rmse_mae(
        final_plot[final_window_start:],
        target_density_plot[final_window_start:],
    )
    final_slice_rmse, final_slice_mae = rmse_mae(
        final_plot[-1], target_density_plot[-1]
    )
    full_rmse, full_mae = rmse_mae(final_plot, target_density_plot)
    final_normalized = raw_reconstruction.detach().cpu().numpy()
    final_window_nrmse, final_window_nmae = nrmse_nmae(
        final_normalized[final_window_start:],
        target_density_normalized[final_window_start:],
    )
    final_slice_nrmse, final_slice_nmae = nrmse_nmae(
        final_normalized[-1], target_density_normalized[-1]
    )
    full_nrmse, full_nmae = nrmse_nmae(
        final_normalized, target_density_normalized
    )

    result = {
        "slide_step": slide_step,
        "window_starts": starts,
        "snapshots": snapshots,
        "final_reconstruction": final_plot,
        "metrics": {
            "full_run_rmse": full_rmse,
            "full_run_mae": full_mae,
            "full_run_nrmse": full_nrmse,
            "full_run_nmae": full_nmae,
            "normalized_metric_space": "preprocessing channel mean/std",
            "final_window_start": final_window_start,
            "final_window_padded_slots": max(
                0, final_window_start + window_size - run_length
            ),
            "final_window_rmse": final_window_rmse,
            "final_window_mae": final_window_mae,
            "final_window_nrmse": final_window_nrmse,
            "final_window_nmae": final_window_nmae,
            "final_slice": run_length - 1,
            "final_slice_rmse": final_slice_rmse,
            "final_slice_mae": final_slice_mae,
            "final_slice_nrmse": final_slice_nrmse,
            "final_slice_nmae": final_slice_nmae,
        },
    }
    if retain_state:
        result["raw_state_normalized"] = raw_reconstruction.detach().clone()
        result["conditioning_state_normalized"] = (
            conditioning_reconstruction.detach().clone()
        )
    return result


def summarize_density_state(
    raw_state_normalized: torch.Tensor,
    target_density_normalized: np.ndarray,
    target_density_plot: np.ndarray,
    density_mean: torch.Tensor,
    density_std: torch.Tensor,
    probe_mask: torch.Tensor,
    plot_units: str,
    x_index: int | None,
) -> Dict:
    """Convert one complete raw state to plot arrays and transparent metrics."""
    raw_plot = convert_density_units(
        raw_state_normalized,
        density_mean=density_mean,
        density_std=density_std,
        plot_units=plot_units,
    ).detach().cpu().numpy()
    raw_normalized = raw_state_normalized.detach().cpu().numpy()
    probe_bool = probe_mask.detach().cpu().numpy() > 0.5
    full_rmse, full_mae = rmse_mae(raw_plot, target_density_plot)
    probe_rmse, probe_mae = rmse_mae(
        raw_plot[:, probe_bool],
        target_density_plot[:, probe_bool],
    )
    hidden_rmse, hidden_mae = rmse_mae(
        raw_plot[:, ~probe_bool],
        target_density_plot[:, ~probe_bool],
    )
    full_nrmse, full_nmae = nrmse_nmae(
        raw_normalized, target_density_normalized
    )
    probe_nrmse, probe_nmae = nrmse_nmae(
        raw_normalized[:, probe_bool],
        target_density_normalized[:, probe_bool],
    )
    hidden_nrmse, hidden_nmae = nrmse_nmae(
        raw_normalized[:, ~probe_bool],
        target_density_normalized[:, ~probe_bool],
    )
    return {
        "final_reconstruction": raw_plot,
        "prediction_projection": project_time_z(raw_plot, x_index=x_index),
        "residual_projection": project_time_z(
            normalized_residual(raw_normalized, target_density_normalized),
            x_index=x_index,
        ),
        "metrics": {
            "full_run_rmse": full_rmse,
            "full_run_mae": full_mae,
            "probe_rmse": probe_rmse,
            "probe_mae": probe_mae,
            "hidden_rmse": hidden_rmse,
            "hidden_mae": hidden_mae,
            "full_run_nrmse": full_nrmse,
            "full_run_nmae": full_nmae,
            "probe_nrmse": probe_nrmse,
            "probe_nmae": probe_nmae,
            "hidden_nrmse": hidden_nrmse,
            "hidden_nmae": hidden_nmae,
            "normalized_metric_space": "preprocessing channel mean/std",
        },
    }


@torch.no_grad()
def refine_reconstruction_pass(
    model: torch.nn.Module,
    target_normalized: torch.Tensor,
    raw_state_normalized: torch.Tensor,
    conditioning_state_normalized: torch.Tensor,
    probe_mask: torch.Tensor,
    window_size: int,
    slide_step: int,
    direction: str,
    offset: int,
    amp: bool,
    snapshot_fn: Callable[[torch.Tensor], Dict] | None = None,
    include_boundary_padding: bool = False,
    max_updates: int | None = None,
    hide_magnetic: bool = False,
) -> Dict:
    """Apply one full-visible repeated sweep and update the state in place."""
    if direction not in {"forward", "reverse"}:
        raise ValueError(f"Unknown direction {direction!r}.")
    channels, run_length, size_x, size_z = target_normalized.shape
    if channels != 4:
        raise ValueError(f"Expected four target channels, got {channels}.")
    if raw_state_normalized.shape != (run_length, size_x, size_z):
        raise ValueError("raw reconstruction state has an incompatible shape.")
    if conditioning_state_normalized.shape != raw_state_normalized.shape:
        raise ValueError("conditioning and raw reconstruction states must match.")
    if not torch.isfinite(conditioning_state_normalized).all():
        raise ValueError("Repeated sweeps require a complete initial reconstruction.")

    starts = shifted_window_starts(
        run_length=run_length,
        window_size=window_size,
        slide_step=slide_step,
        offset=offset,
        include_boundary_padding=include_boundary_padding,
    )
    ordered_starts = starts if direction == "forward" else list(reversed(starts))
    if max_updates is not None:
        if max_updates < 1:
            raise ValueError("max_updates must be positive when provided.")
        ordered_starts = ordered_starts[:max_updates]
    probe_bool = probe_mask > 0.5
    updates = []
    snapshots = []

    for update_index, start in enumerate(ordered_starts):
        valid_start = max(start, 0)
        valid_end = min(start + window_size, run_length)
        valid_length = valid_end - valid_start
        local_start = valid_start - start
        local_end = local_start + valid_length
        values = torch.zeros(
            (1, 4, window_size, size_x, size_z),
            device=target_normalized.device,
            dtype=target_normalized.dtype,
        )
        mask = torch.zeros_like(values)

        # Magnetic fields remain complete true observations unless
        # --hide-magnetic is set. Every valid Density position uses the latest
        # conditioning reconstruction and is marked visible; only padded
        # future slots remain invisible.
        if not hide_magnetic:
            values[0, :3, local_start:local_end] = target_normalized[
                :3, valid_start:valid_end
            ]
            mask[0, :3, local_start:local_end] = 1.0
        values[0, 3, local_start:local_end] = conditioning_state_normalized[
            valid_start:valid_end
        ]
        mask[0, 3, local_start:local_end] = 1.0

        density_target = target_normalized[3, valid_start:valid_end]
        density_values = values[0, 3, local_start:local_end]
        density_values[:, probe_bool] = density_target[:, probe_bool]
        values[0, 3, local_start:local_end] = density_values

        model_input = torch.cat([values * mask, mask], dim=1)
        with torch.autocast(
            device_type=target_normalized.device.type,
            enabled=(amp and target_normalized.device.type == "cuda"),
        ):
            prediction = model(model_input)

        # The raw output replaces the entire valid window, including regions
        # used as fully visible input. Only the separate conditioning state is
        # clamped at real probes before the next update.
        raw_window = prediction[0, 3, local_start:local_end].float()
        raw_state_normalized[valid_start:valid_end] = raw_window
        conditioned_window = raw_window.clone()
        conditioned_window[:, probe_bool] = density_target[:, probe_bool]
        conditioning_state_normalized[valid_start:valid_end] = conditioned_window
        updates.append(
            {
                "update_index": update_index,
                "window_start": start,
                "valid_start": valid_start,
                "valid_end": valid_end,
            }
        )
        if snapshot_fn is not None:
            snapshot = snapshot_fn(raw_state_normalized)
            snapshot.update(updates[-1])
            snapshots.append(snapshot)

    return {
        "direction": direction,
        "offset": offset,
        "ordered_window_starts": ordered_starts,
        "updates": updates,
        "snapshots": snapshots,
    }


def build_bidirectional_rows(
    model: torch.nn.Module,
    target_normalized: torch.Tensor,
    target_density_plot: np.ndarray,
    density_mean: torch.Tensor,
    density_std: torch.Tensor,
    probe_mask: torch.Tensor,
    window_size: int,
    refinement_step: int,
    refinement_passes: int,
    refinement_offset: int,
    plot_units: str,
    x_index: int | None,
    amp: bool,
    initial_results: Dict[int, Dict] | None = None,
    hide_magnetic: bool = False,
) -> List[Dict]:
    """Build the repeated-sweep rows requested for bidirectional comparison."""
    if refinement_passes < 1:
        raise ValueError("--refinement-passes must be at least 1.")
    if not (1 <= refinement_step <= window_size):
        raise ValueError(
            f"--refinement-step must lie in [1, {window_size}]."
        )
    if not (0 <= refinement_offset < window_size):
        raise ValueError(
            f"--refinement-offset must lie in [0, {window_size - 1}]."
        )

    cached = {} if initial_results is None else initial_results
    target_density_normalized = target_normalized[3].detach().cpu().numpy()

    def initial_result(step: int) -> Dict:
        result = cached.get(step)
        if result is not None and "raw_state_normalized" in result:
            return result
        return reconstruct_with_slide_step(
            model=model,
            target_normalized=target_normalized,
            target_density_plot=target_density_plot,
            density_mean=density_mean,
            density_std=density_std,
            probe_mask=probe_mask,
            window_size=window_size,
            slide_step=step,
            plot_units=plot_units,
            x_index=x_index,
            amp=amp,
            retain_state=True,
            hide_magnetic=hide_magnetic,
        )

    def compact_snapshot(raw_state: torch.Tensor) -> Dict:
        summary = summarize_density_state(
            raw_state_normalized=raw_state,
            target_density_normalized=target_density_normalized,
            target_density_plot=target_density_plot,
            density_mean=density_mean,
            density_std=density_std,
            probe_mask=probe_mask,
            plot_units=plot_units,
            x_index=x_index,
        )
        summary.pop("final_reconstruction")
        return summary

    def initial_animation_snapshots(result: Dict) -> List[Dict]:
        snapshots = []
        for snapshot in result["snapshots"]:
            snapshots.append(
                {
                    "prediction_projection": snapshot["prediction_projection"],
                    "residual_projection": snapshot["residual_projection"],
                    "metrics": {
                        "full_run_rmse": snapshot["covered_rmse"],
                        "full_run_mae": snapshot["covered_mae"],
                        "full_run_nrmse": snapshot["covered_nrmse"],
                        "full_run_nmae": snapshot["covered_nmae"],
                    },
                    "pass_index": 0,
                    "pass_count": 1,
                    "direction": "forward",
                    "offset": 0,
                    "update_index": snapshot["update_index"],
                    "cumulative_window_calls": snapshot["update_index"] + 1,
                    "window_start": snapshot["window_start"],
                    "valid_end": snapshot["valid_end"],
                    "covered_end": snapshot["covered_end"],
                }
            )
        return snapshots

    def annotate_pass_snapshots(pass_info: Dict, pass_index: int) -> List[Dict]:
        for snapshot in pass_info["snapshots"]:
            snapshot.update(
                {
                    "pass_index": pass_index,
                    "pass_count": pass_index + 1,
                    "direction": pass_info["direction"],
                    "offset": pass_info["offset"],
                }
            )
        return pass_info["snapshots"]

    def compact_history(pass_info: Dict) -> Dict:
        return {
            key: value
            for key, value in pass_info.items()
            if key != "snapshots"
        }

    def state_summary(
        raw_state: torch.Tensor,
        pass_count: int,
        step: int,
        directions: List[str],
        offsets: List[int],
        history: List[Dict],
        animation_snapshots: List[Dict],
        row_kind: str,
    ) -> Dict:
        summary = summarize_density_state(
            raw_state_normalized=raw_state,
            target_density_normalized=target_density_normalized,
            target_density_plot=target_density_plot,
            density_mean=density_mean,
            density_std=density_std,
            probe_mask=probe_mask,
            plot_units=plot_units,
            x_index=x_index,
        )
        summary.update(
            {
                "row_kind": row_kind,
                "slide_step": step,
                "pass_count": pass_count,
                "directions": list(directions),
                "offsets": list(offsets),
                "history": list(history),
                "animation_snapshots": list(animation_snapshots),
            }
        )
        return summary

    direction_names = [
        "forward" if pass_index % 2 == 0 else "reverse"
        for pass_index in range(refinement_passes)
    ]
    rows = []

    small_initial = initial_result(refinement_step)
    small_raw = small_initial["raw_state_normalized"].clone()
    small_conditioning = small_initial["conditioning_state_normalized"].clone()
    small_history = [
        {
            "direction": "forward",
            "offset": 0,
            "ordered_window_starts": small_initial["window_starts"],
        }
    ]
    small_animation = initial_animation_snapshots(small_initial)
    rows.append(
        state_summary(
            small_raw,
            pass_count=1,
            step=refinement_step,
            directions=direction_names[:1],
            offsets=[0],
            history=small_history,
            animation_snapshots=small_animation,
            row_kind="overlap_refinement",
        )
    )
    for pass_index in range(1, refinement_passes):
        pass_info = refine_reconstruction_pass(
            model=model,
            target_normalized=target_normalized,
            raw_state_normalized=small_raw,
            conditioning_state_normalized=small_conditioning,
            probe_mask=probe_mask,
            window_size=window_size,
            slide_step=refinement_step,
            direction=direction_names[pass_index],
            offset=0,
            amp=amp,
            snapshot_fn=compact_snapshot,
            hide_magnetic=hide_magnetic,
        )
        small_history.append(compact_history(pass_info))
        small_animation.extend(annotate_pass_snapshots(pass_info, pass_index))
        for call_index, snapshot in enumerate(small_animation, start=1):
            snapshot["cumulative_window_calls"] = call_index
        rows.append(
            state_summary(
                small_raw,
                pass_count=pass_index + 1,
                step=refinement_step,
                directions=direction_names[: pass_index + 1],
                offsets=[0] * (pass_index + 1),
                history=small_history,
                animation_snapshots=small_animation,
                row_kind="overlap_refinement",
            )
        )

    independent_initial = initial_result(window_size)
    independent_raw = independent_initial["raw_state_normalized"].clone()
    independent_conditioning = independent_initial[
        "conditioning_state_normalized"
    ].clone()
    independent_history = [
        {
            "direction": "forward",
            "offset": 0,
            "ordered_window_starts": independent_initial["window_starts"],
        }
    ]
    independent_animation = initial_animation_snapshots(independent_initial)
    target_window_calls = len(small_animation)
    independent_directions = ["forward"]
    independent_offsets = [0]
    pass_index = 1
    while len(independent_animation) < target_window_calls:
        direction = "forward" if pass_index % 2 == 0 else "reverse"
        remaining_calls = target_window_calls - len(independent_animation)
        pass_info = refine_reconstruction_pass(
            model=model,
            target_normalized=target_normalized,
            raw_state_normalized=independent_raw,
            conditioning_state_normalized=independent_conditioning,
            probe_mask=probe_mask,
            window_size=window_size,
            slide_step=window_size,
            direction=direction,
            offset=0,
            amp=amp,
            snapshot_fn=compact_snapshot,
            max_updates=remaining_calls,
            hide_magnetic=hide_magnetic,
        )
        independent_directions.append(direction)
        independent_offsets.append(0)
        independent_history.append(compact_history(pass_info))
        independent_animation.extend(
            annotate_pass_snapshots(pass_info, pass_index)
        )
        for call_index, snapshot in enumerate(independent_animation, start=1):
            snapshot["cumulative_window_calls"] = call_index
        pass_index += 1
    rows.append(
        state_summary(
            independent_raw,
            pass_count=len(independent_directions),
            step=window_size,
            directions=independent_directions,
            offsets=independent_offsets,
            history=independent_history,
            animation_snapshots=independent_animation,
            row_kind="independent_control",
        )
    )

    shifted_raw = independent_initial["raw_state_normalized"].clone()
    shifted_conditioning = independent_initial[
        "conditioning_state_normalized"
    ].clone()
    shifted_history = [independent_history[0]]
    shifted_offsets = [0]
    shifted_directions = ["forward"]
    shifted_animation = initial_animation_snapshots(independent_initial)
    pass_index = 1
    while len(shifted_animation) < target_window_calls:
        offset = refinement_offset if pass_index % 2 == 1 else 0
        direction = "forward" if pass_index % 2 == 0 else "reverse"
        remaining_calls = target_window_calls - len(shifted_animation)
        shifted_offsets.append(offset)
        shifted_directions.append(direction)
        pass_info = refine_reconstruction_pass(
            model=model,
            target_normalized=target_normalized,
            raw_state_normalized=shifted_raw,
            conditioning_state_normalized=shifted_conditioning,
            probe_mask=probe_mask,
            window_size=window_size,
            slide_step=window_size,
            direction=direction,
            offset=offset,
            amp=amp,
            snapshot_fn=compact_snapshot,
            include_boundary_padding=True,
            max_updates=remaining_calls,
            hide_magnetic=hide_magnetic,
        )
        shifted_history.append(compact_history(pass_info))
        shifted_animation.extend(annotate_pass_snapshots(pass_info, pass_index))
        for call_index, snapshot in enumerate(shifted_animation, start=1):
            snapshot["cumulative_window_calls"] = call_index
        pass_index += 1
    rows.append(
        state_summary(
            shifted_raw,
            pass_count=len(shifted_directions),
            step=window_size,
            directions=shifted_directions,
            offsets=shifted_offsets,
            history=shifted_history,
            animation_snapshots=shifted_animation,
            row_kind="shifted_control",
        )
    )
    return rows


def snapshot_at_or_before(result: Dict, covered_end: int) -> Dict:
    eligible = [
        snapshot
        for snapshot in result["snapshots"]
        if snapshot["covered_end"] <= covered_end
    ]
    if not eligible:
        return result["snapshots"][0]
    return eligible[-1]


def window_outline_bounds(
    snapshot: Dict | None,
    frame_ids: np.ndarray,
) -> Tuple[float, float] | None:
    """Return visible time-axis edges for one snapshot's active window."""
    if snapshot is None or "window_start" not in snapshot or "valid_end" not in snapshot:
        return None
    if len(frame_ids) == 0:
        return None

    valid_start = int(
        snapshot.get("valid_start", max(int(snapshot["window_start"]), 0))
    )
    valid_end = int(snapshot["valid_end"])
    valid_start = max(0, min(valid_start, len(frame_ids)))
    valid_end = max(0, min(valid_end, len(frame_ids)))
    if valid_end <= valid_start:
        return None

    left = float(frame_ids[valid_start]) - 0.5
    right = float(frame_ids[valid_end - 1]) + 0.5
    return left, right


def add_window_outline(
    ax,
    snapshot: Dict | None,
    frame_ids: np.ndarray,
    zmin: float,
    zmax: float,
) -> None:
    """Overlay the current valid sliding-window footprint on a time-z panel."""
    bounds = window_outline_bounds(snapshot, frame_ids)
    if bounds is None:
        return
    left, right = bounds
    width = right - left
    height = zmax - zmin

    # A faint fill makes the footprint readable without hiding the field. The
    # black under-stroke keeps the cyan edge visible on both plotting colormaps.
    ax.add_patch(
        Rectangle(
            (left, zmin),
            width,
            height,
            facecolor="cyan",
            edgecolor="none",
            alpha=0.055,
            zorder=4,
        )
    )
    ax.add_patch(
        Rectangle(
            (left, zmin),
            width,
            height,
            fill=False,
            edgecolor="black",
            linewidth=2.6,
            zorder=5,
        )
    )
    ax.add_patch(
        Rectangle(
            (left, zmin),
            width,
            height,
            fill=False,
            edgecolor="cyan",
            linewidth=1.35,
            zorder=6,
        )
    )


def plot_progress_frame(
    target_projection: np.ndarray,
    results: Sequence[Dict],
    covered_end: int,
    frame_ids: np.ndarray,
    extent: Sequence[float],
    run_name: str,
    window_size: int,
    probe_actual_fraction: float,
    projection_label: str,
    plot_units: str,
    density_limits: Tuple[float, float],
    residual_limits: Tuple[float, float],
    out_path: Path,
    dpi: int,
    show_window_outline: bool = True,
    magnetic_label: str = "B fully observed",
) -> None:
    n_rows = len(results)
    title_inches = 0.90
    bottom_inches = 0.55
    fig_h = 3.35 * n_rows + title_inches + bottom_inches
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(14.0, fig_h),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    if n_rows == 1:
        axes = np.asarray([axes])

    zmin, zmax = float(extent[0]), float(extent[1])
    image_extent = [
        float(frame_ids[0]) - 0.5,
        float(frame_ids[-1]) + 0.5,
        zmin,
        zmax,
    ]
    density_cmap = make_nan_cmap("plasma", bad_color="black")
    residual_cmap = make_nan_cmap(RESIDUAL_CMAP, bad_color="black")
    density_image = None
    residual_image = None

    for row_index, result in enumerate(results):
        snapshot = snapshot_at_or_before(result, covered_end)
        density_image = axes[row_index, 0].imshow(
            target_projection.T,
            origin="lower",
            aspect="auto",
            extent=image_extent,
            cmap=density_cmap,
            vmin=density_limits[0],
            vmax=density_limits[1],
            interpolation="nearest",
        )
        axes[row_index, 1].imshow(
            snapshot["prediction_projection"].T,
            origin="lower",
            aspect="auto",
            extent=image_extent,
            cmap=density_cmap,
            vmin=density_limits[0],
            vmax=density_limits[1],
            interpolation="nearest",
        )
        residual_image = axes[row_index, 2].imshow(
            snapshot["residual_projection"].T,
            origin="lower",
            aspect="auto",
            extent=image_extent,
            cmap=residual_cmap,
            vmin=residual_limits[0],
            vmax=residual_limits[1],
            interpolation="nearest",
        )

        for column in range(3):
            if show_window_outline:
                add_window_outline(
                    axes[row_index, column],
                    snapshot=snapshot,
                    frame_ids=frame_ids,
                    zmin=zmin,
                    zmax=zmax,
                )
            axes[row_index, column].tick_params(labelsize=8)
        axes[row_index, 0].set_ylabel(
            f"step={result['slide_step']}\n"
            f"updates={snapshot['update_index'] + 1}\n"
            f"window={snapshot['window_start']}:"
            f"{snapshot['window_start'] + window_size}; "
            f"valid to {snapshot['valid_end']}\n"
            f"latest NRMSE={snapshot['last_slice_nrmse']:.3g}\n"
            "z [cm]",
            fontsize=8,
        )

    for column, title in enumerate(
        ["Target Density", "Latest reconstruction", "Normalized residual"]
    ):
        axes[0, column].set_title(title, fontsize=11)
        axes[-1, column].set_xlabel("time slice", fontsize=10)

    axes_top = 1.0 - title_inches / fig_h
    fig.subplots_adjust(
        left=0.10,
        right=0.90,
        bottom=bottom_inches / fig_h,
        top=axes_top,
        wspace=0.08,
        hspace=0.12,
    )
    density_cax = fig.add_axes([0.915, 0.53, 0.012, 0.34])
    residual_cax = fig.add_axes([0.915, 0.12, 0.012, 0.34])
    density_colorbar = fig.colorbar(density_image, cax=density_cax)
    density_colorbar.set_label(f"Density ({plot_units})", fontsize=9)
    residual_colorbar = fig.colorbar(residual_image, cax=residual_cax)
    residual_colorbar.set_label("Normalized Density residual", fontsize=9)

    outline_text = (
        "\ncyan outline = window used for the displayed update"
        if show_window_outline
        else ""
    )
    fig.suptitle(
        f"Sliding-window Density reconstruction: {run_name}\n"
        f"T={window_size}, fixed probes={100.0 * probe_actual_fraction:.2f}%, "
        f"{magnetic_label}, {projection_label}; progress through slice "
        f"{int(frame_ids[min(covered_end, len(frame_ids)) - 1])}"
        f"{outline_text}",
        fontsize=12,
        y=0.99,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_bidirectional_refinement_figure(
    target_projection: np.ndarray,
    rows: Sequence[Dict],
    frame_ids: np.ndarray,
    extent: Sequence[float],
    run_name: str,
    window_size: int,
    probe_actual_fraction: float,
    projection_label: str,
    plot_units: str,
    density_limits: Tuple[float, float],
    residual_limits: Tuple[float, float],
    out_path: Path,
    dpi: int,
    snapshots: Sequence[Dict | None] | None = None,
    progress_label: str | None = None,
    show_window_outline: bool = True,
    magnetic_label: str = "B fully observed",
) -> None:
    """Plot final states after alternating bidirectional reconstruction sweeps."""
    n_rows = len(rows)
    title_inches = 1.05
    bottom_inches = 0.50
    fig_h = 3.15 * n_rows + title_inches + bottom_inches
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(14.0, fig_h),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    if n_rows == 1:
        axes = np.asarray([axes])

    zmin, zmax = float(extent[0]), float(extent[1])
    image_extent = [
        float(frame_ids[0]) - 0.5,
        float(frame_ids[-1]) + 0.5,
        zmin,
        zmax,
    ]
    density_cmap = make_nan_cmap("plasma", bad_color="black")
    residual_cmap = make_nan_cmap(RESIDUAL_CMAP, bad_color="black")
    density_image = None
    residual_image = None

    for row_index, row in enumerate(rows):
        display = row if snapshots is None else snapshots[row_index]
        if show_window_outline:
            outline_snapshot = (
                row["animation_snapshots"][-1]
                if snapshots is None and row["animation_snapshots"]
                else display
            )
        else:
            outline_snapshot = None
        if display is None:
            prediction_projection = np.full_like(target_projection, np.nan)
            residual_projection = np.full_like(target_projection, np.nan)
            display_metrics = {"full_run_nrmse": float("nan")}
        else:
            prediction_projection = display["prediction_projection"]
            residual_projection = display["residual_projection"]
            display_metrics = display["metrics"]
        density_image = axes[row_index, 0].imshow(
            target_projection.T,
            origin="lower",
            aspect="auto",
            extent=image_extent,
            cmap=density_cmap,
            vmin=density_limits[0],
            vmax=density_limits[1],
            interpolation="nearest",
        )
        axes[row_index, 1].imshow(
            prediction_projection.T,
            origin="lower",
            aspect="auto",
            extent=image_extent,
            cmap=density_cmap,
            vmin=density_limits[0],
            vmax=density_limits[1],
            interpolation="nearest",
        )
        residual_image = axes[row_index, 2].imshow(
            residual_projection.T,
            origin="lower",
            aspect="auto",
            extent=image_extent,
            cmap=residual_cmap,
            vmin=residual_limits[0],
            vmax=residual_limits[1],
            interpolation="nearest",
        )
        direction_labels = [
            "L→R" if direction == "forward" else "R→L"
            for direction in row["directions"]
        ]
        if len(direction_labels) <= 4:
            directions = "/".join(direction_labels)
        else:
            directions = f"L→R/R→L alternating ({len(direction_labels)} sweeps)"
        if len(row["offsets"]) <= 4:
            offsets = "/".join(str(offset) for offset in row["offsets"])
        elif len(set(row["offsets"])) == 1:
            offsets = str(row["offsets"][0])
        else:
            offsets = "0/shift alternating"
        if snapshots is None and not show_window_outline:
            current_label = ""
        elif snapshots is None:
            if outline_snapshot is None:
                current_label = ""
            else:
                current_label = (
                    f"\nlast window={outline_snapshot['window_start']}:"
                    f"{outline_snapshot['window_start'] + window_size}; "
                    f"valid to {outline_snapshot['valid_end']}"
                )
        elif display is None:
            current_label = "\ncurrent=pending"
        elif "window_start" in display:
            direction = "L→R" if display["direction"] == "forward" else "R→L"
            current_label = (
                f"\ncurrent=p{display['pass_count']} {direction}, "
                f"window={display['window_start']}:"
                f"{display['window_start'] + window_size} "
                f"(valid to {display['valid_end']}), "
                f"calls={display['cumulative_window_calls']}"
            )
        else:
            completed_calls = len(row["animation_snapshots"])
            current_label = f"\ncurrent=complete, calls={completed_calls}"
        axes[row_index, 0].set_ylabel(
            f"step={row['slide_step']}, passes={row['pass_count']}\n"
            f"dirs={directions}\n"
            f"offsets={offsets}, calls={len(row['animation_snapshots'])}\n"
            f"NRMSE={display_metrics['full_run_nrmse']:.3g}"
            f"{current_label}\n"
            "z [cm]",
            fontsize=7.5,
        )
        for column in range(3):
            if show_window_outline:
                add_window_outline(
                    axes[row_index, column],
                    snapshot=outline_snapshot,
                    frame_ids=frame_ids,
                    zmin=zmin,
                    zmax=zmax,
                )
            axes[row_index, column].tick_params(labelsize=8)

    for column, title in enumerate(
        ["Target Density", "Repeated reconstruction", "Normalized residual"]
    ):
        axes[0, column].set_title(title, fontsize=11)
        axes[-1, column].set_xlabel("time slice", fontsize=10)

    fig.subplots_adjust(
        left=0.115,
        right=0.90,
        bottom=bottom_inches / fig_h,
        top=1.0 - title_inches / fig_h,
        wspace=0.08,
        hspace=0.12,
    )
    density_cax = fig.add_axes([0.915, 0.53, 0.012, 0.34])
    residual_cax = fig.add_axes([0.915, 0.12, 0.012, 0.34])
    density_colorbar = fig.colorbar(density_image, cax=density_cax)
    density_colorbar.set_label(f"Density ({plot_units})", fontsize=9)
    residual_colorbar = fig.colorbar(residual_image, cax=residual_cax)
    residual_colorbar.set_label("Normalized Density residual", fontsize=9)

    progress_text = "" if progress_label is None else f"\n{progress_label}"
    if show_window_outline:
        outline_text = (
            "\ncyan outline = last processed window"
            if snapshots is None
            else "\ncyan outline = current window; completed rows have no outline"
        )
    else:
        outline_text = ""
    fig.suptitle(
        f"Bidirectional repeated Density reconstruction: {run_name}\n"
        f"T={window_size}, fixed probes={100.0 * probe_actual_fraction:.2f}%, "
        f"{magnetic_label}, {projection_label}; raw predictions include probes"
        f"{progress_text}{outline_text}",
        fontsize=12,
        y=0.99,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved bidirectional figure: {out_path}")


def bidirectional_animation_events(rows: Sequence[Dict]) -> List[Dict]:
    """Synchronize all rows by sweep pass and normalized within-pass progress."""
    overlap_rows = [row for row in rows if row["row_kind"] == "overlap_refinement"]
    if not overlap_rows:
        raise ValueError("Bidirectional animation requires overlap-refinement rows.")

    def final_snapshot(row: Dict) -> Dict:
        return {
            "prediction_projection": row["prediction_projection"],
            "residual_projection": row["residual_projection"],
            "metrics": row["metrics"],
        }

    target_window_calls = max(len(row["animation_snapshots"]) for row in rows)
    events = []
    for call_index in range(target_window_calls):
        displays = []
        for row in rows:
            snapshots = row["animation_snapshots"]
            if call_index < len(snapshots):
                displays.append(snapshots[call_index])
            else:
                displays.append(final_snapshot(row))
        events.append(
            {
                "phase": "window-call synchronized reconstruction",
                "label": (
                    f"synchronized window call={call_index + 1}/"
                    f"{target_window_calls}"
                ),
                "snapshots": displays,
            }
        )
    return events


def save_bidirectional_analysis(
    rows: Sequence[Dict],
    target_density_plot: np.ndarray,
    target_projection: np.ndarray,
    probe_mask: torch.Tensor,
    frame_ids: np.ndarray,
    extent: Sequence[float],
    run_name: str,
    window_size: int,
    refinement_step: int,
    refinement_passes: int,
    refinement_offset: int,
    probe_actual_fraction: float,
    probe_info: Dict,
    projection_label: str,
    plot_units: str,
    field_q: float,
    residual_q: float,
    residual_vmax: float | None,
    checkpoint_path: Path,
    checkpoint_epoch: int | None,
    h5_path: Path,
    out_dir: Path,
    dpi: int,
    animation_dpi: int,
    animation_format: str,
    fps: float,
    hide_magnetic: bool = False,
) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    density_limits = robust_limits(
        [target_projection] + [row["prediction_projection"] for row in rows],
        channel=3,
        q=field_q,
        symmetric=False,
    )
    if residual_vmax is None:
        residual_limits = robust_limits(
            [row["residual_projection"] for row in rows],
            channel=3,
            q=residual_q,
            symmetric=True,
        )
    else:
        residual_limits = (-float(residual_vmax), float(residual_vmax))

    magnetic_label = "B fully hidden" if hide_magnetic else "B fully observed"
    magnetic_tag = "_B-hidden" if hide_magnetic else ""
    common_stem = (
        f"{run_name}_T{window_size}_bidirectional-step{refinement_step}-"
        f"passes{refinement_passes}_step{window_size}-offset{refinement_offset}_"
        f"density-visible-{probe_actual_fraction:.4f}{magnetic_tag}_{plot_units}"
    )
    figure_path = images_dir / f"{common_stem}.png"
    plot_bidirectional_refinement_figure(
        target_projection=target_projection,
        rows=rows,
        frame_ids=frame_ids,
        extent=extent,
        run_name=run_name,
        window_size=window_size,
        probe_actual_fraction=probe_actual_fraction,
        projection_label=projection_label,
        plot_units=plot_units,
        density_limits=density_limits,
        residual_limits=residual_limits,
        out_path=figure_path,
        dpi=dpi,
        show_window_outline=False,
        magnetic_label=magnetic_label,
    )

    animation_events = bidirectional_animation_events(rows)
    frame_dir = images_dir / f"{common_stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_index, event in enumerate(animation_events):
        frame_path = frame_dir / f"frame-{frame_index:04d}.png"
        plot_bidirectional_refinement_figure(
            target_projection=target_projection,
            rows=rows,
            frame_ids=frame_ids,
            extent=extent,
            run_name=run_name,
            window_size=window_size,
            probe_actual_fraction=probe_actual_fraction,
            projection_label=projection_label,
            plot_units=plot_units,
            density_limits=density_limits,
            residual_limits=residual_limits,
            out_path=frame_path,
            dpi=animation_dpi,
            snapshots=event["snapshots"],
            progress_label=event["label"],
            magnetic_label=magnetic_label,
        )
        frame_paths.append(frame_path)

    # Repeat the completed state once without an active-window outline. GIFs
    # stop on this clean frame, and video viewers also get an unobstructed end.
    final_event = animation_events[-1]
    clean_frame_path = frame_dir / f"frame-{len(frame_paths):04d}-no-window.png"
    plot_bidirectional_refinement_figure(
        target_projection=target_projection,
        rows=rows,
        frame_ids=frame_ids,
        extent=extent,
        run_name=run_name,
        window_size=window_size,
        probe_actual_fraction=probe_actual_fraction,
        projection_label=projection_label,
        plot_units=plot_units,
        density_limits=density_limits,
        residual_limits=residual_limits,
        out_path=clean_frame_path,
        dpi=animation_dpi,
        snapshots=final_event["snapshots"],
        progress_label=final_event["label"],
        show_window_outline=False,
        magnetic_label=magnetic_label,
    )
    frame_paths.append(clean_frame_path)

    formats = (
        ["quicktime", "gif"] if animation_format == "both" else [animation_format]
    )
    for selected_format in formats:
        extension = "mov" if selected_format == "quicktime" else selected_format
        write_animation(
            frame_paths,
            out_dir / f"{common_stem}.{extension}",
            fps=fps,
        )

    row_metrics = []
    for row_index, row in enumerate(rows, start=1):
        row_metrics.append(
            {
                "row": row_index,
                "row_kind": row["row_kind"],
                "slide_step": row["slide_step"],
                "pass_count": row["pass_count"],
                "directions": row["directions"],
                "offsets": row["offsets"],
                "window_calls": len(row["animation_snapshots"]),
                "history": row["history"],
                **row["metrics"],
            }
        )
    framewise_errors = save_framewise_error_plot(
        predictions=[row["final_reconstruction"] for row in rows],
        labels=[
            f"row {index}: step={row['slide_step']}, passes={row['pass_count']}"
            for index, row in enumerate(rows, start=1)
        ],
        target=target_density_plot,
        frame_ids=frame_ids,
        out_path=out_dir / f"{common_stem}_error_vs_frame.png",
        title="Bidirectional reconstruction error by frame",
        plot_units=plot_units,
    )
    metrics_payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "h5_path": str(h5_path),
        "run_name": run_name,
        "run_length": int(target_density_plot.shape[0]),
        "window_size": window_size,
        "refinement_step": refinement_step,
        "refinement_passes": refinement_passes,
        "refinement_offset": refinement_offset,
        "density_visible_fraction_actual": probe_actual_fraction,
        "probe_grid": probe_info,
        "plot_units": plot_units,
        "projection": projection_label,
        "conditioning_policy": (
            "latest reconstruction fully visible; true probes overwrite input"
        ),
        "metric_policy": "raw model predictions, including probe locations",
        "normalized_residual_policy": (
            "prediction_normalized - target_normalized, using checkpoint "
            "preprocessing mean/std"
        ),
        "animation_frames": len(frame_paths),
        "animation_policy": (
            "all rows synchronized by cumulative model window calls; final "
            "state repeated once without window outlines"
        ),
        "rows": row_metrics,
        "framewise_errors": framewise_errors,
    }
    metrics_path = out_dir / f"{common_stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"Saved bidirectional metrics: {metrics_path}")

    arrays_path = out_dir / f"{common_stem}_reconstructions.npz"
    np.savez_compressed(
        arrays_path,
        target_density=target_density_plot,
        probe_mask=probe_mask.detach().cpu().numpy(),
        frame_ids=frame_ids,
        **{
            f"prediction_row_{row_index}": row["final_reconstruction"]
            for row_index, row in enumerate(rows, start=1)
        },
    )
    print(f"Saved bidirectional arrays: {arrays_path}")


def save_slide_step_analysis(
    results: Sequence[Dict],
    slide_steps: Sequence[int],
    target_density_plot: np.ndarray,
    target_projection: np.ndarray,
    probe_mask: torch.Tensor,
    frame_ids: np.ndarray,
    extent: Sequence[float],
    run_name: str,
    window_size: int,
    density_visible_fraction: float,
    probe_actual_fraction: float,
    probe_info: Dict,
    projection_label: str,
    plot_units: str,
    field_q: float,
    residual_q: float,
    residual_vmax: float | None,
    animation_format: str,
    fps: float,
    checkpoint_path: Path,
    checkpoint_epoch: int | None,
    h5_path: Path,
    out_dir: Path,
    dpi: int,
    hide_magnetic: bool = False,
) -> None:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    density_vmin, density_vmax = robust_limits(
        [target_projection]
        + [result["snapshots"][-1]["prediction_projection"] for result in results],
        channel=3,
        q=field_q,
        symmetric=False,
    )
    if residual_vmax is None:
        residual_vmin, residual_vmax_value = robust_limits(
            [
                snapshot["residual_projection"]
                for result in results
                for snapshot in result["snapshots"]
            ],
            channel=3,
            q=residual_q,
            symmetric=True,
        )
    else:
        residual_vmin = -float(residual_vmax)
        residual_vmax_value = float(residual_vmax)

    magnetic_label = "B fully hidden" if hide_magnetic else "B fully observed"
    progress_points = sorted(
        {
            int(snapshot["covered_end"])
            for result in results
            for snapshot in result["snapshots"]
        }
    )
    frame_paths = []
    common_stem = (
        f"{run_name}_T{window_size}_steps-"
        + "-".join(str(step) for step in slide_steps)
        + f"_density-visible-{density_visible_fraction:g}"
        + ("_B-hidden" if hide_magnetic else "")
        + f"_{plot_units}"
    )
    frame_dir = images_dir / f"{common_stem}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, covered_end in enumerate(progress_points):
        frame_path = frame_dir / (
            f"{common_stem}_progress-{frame_index:03d}_"
            f"covered-through-{covered_end - 1:04d}.png"
        )
        plot_progress_frame(
            target_projection=target_projection,
            results=results,
            covered_end=covered_end,
            frame_ids=frame_ids,
            extent=extent,
            run_name=run_name,
            window_size=window_size,
            probe_actual_fraction=probe_actual_fraction,
            projection_label=projection_label,
            plot_units=plot_units,
            density_limits=(density_vmin, density_vmax),
            residual_limits=(residual_vmin, residual_vmax_value),
            out_path=frame_path,
            dpi=dpi,
            magnetic_label=magnetic_label,
        )
        frame_paths.append(frame_path)

    final_covered_end = progress_points[-1]
    clean_frame_path = frame_dir / (
        f"{common_stem}_progress-{len(frame_paths):03d}_"
        f"covered-through-{final_covered_end - 1:04d}_no-window.png"
    )
    plot_progress_frame(
        target_projection=target_projection,
        results=results,
        covered_end=final_covered_end,
        frame_ids=frame_ids,
        extent=extent,
        run_name=run_name,
        window_size=window_size,
        probe_actual_fraction=probe_actual_fraction,
        projection_label=projection_label,
        plot_units=plot_units,
        density_limits=(density_vmin, density_vmax),
        residual_limits=(residual_vmin, residual_vmax_value),
        out_path=clean_frame_path,
        dpi=dpi,
        show_window_outline=False,
        magnetic_label=magnetic_label,
    )
    frame_paths.append(clean_frame_path)

    final_figure_path = images_dir / f"{common_stem}_final.png"
    shutil.copy2(clean_frame_path, final_figure_path)
    print(f"Saved final figure: {final_figure_path}")

    formats = (
        ["quicktime", "gif"] if animation_format == "both" else [animation_format]
    )
    for selected_format in formats:
        extension = "mov" if selected_format == "quicktime" else selected_format
        write_animation(
            frame_paths,
            out_dir / f"{common_stem}.{extension}",
            fps=fps,
        )

    metrics_payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "h5_path": str(h5_path),
        "run_name": run_name,
        "run_length": int(target_density_plot.shape[0]),
        "window_size": window_size,
        "slide_steps": list(slide_steps),
        "density_visible_fraction_target": density_visible_fraction,
        "density_visible_fraction_actual": probe_actual_fraction,
        "probe_grid": probe_info,
        "plot_units": plot_units,
        "projection": projection_label,
        "probe_metric_policy": "raw model predictions included",
        "normalized_residual_policy": (
            "prediction_normalized - target_normalized, using checkpoint "
            "preprocessing mean/std"
        ),
        "results": {
            str(result["slide_step"]): {
                "window_starts": result["window_starts"],
                "updates": [
                    {
                        key: snapshot[key]
                        for key in (
                            "update_index",
                            "window_start",
                            "valid_end",
                            "covered_end",
                            "covered_rmse",
                            "covered_mae",
                            "last_slice_rmse",
                            "last_slice_mae",
                            "current_window_rmse",
                            "current_window_mae",
                            "covered_nrmse",
                            "covered_nmae",
                            "last_slice_nrmse",
                            "last_slice_nmae",
                            "current_window_nrmse",
                            "current_window_nmae",
                        )
                    }
                    for snapshot in result["snapshots"]
                ],
                **result["metrics"],
            }
            for result in results
        },
    }
    metrics_payload["framewise_errors"] = save_framewise_error_plot(
        predictions=[result["final_reconstruction"] for result in results],
        labels=[
            f"row {index}: step={result['slide_step']}"
            for index, result in enumerate(results, start=1)
        ],
        target=target_density_plot,
        frame_ids=frame_ids,
        out_path=out_dir / f"{common_stem}_error_vs_frame.png",
        title="Sliding-window reconstruction error by frame",
        plot_units=plot_units,
    )
    metrics_path = out_dir / f"{common_stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    print(f"Saved metrics: {metrics_path}")

    arrays_path = out_dir / f"{common_stem}_final_reconstructions.npz"
    np.savez_compressed(
        arrays_path,
        target_density=target_density_plot,
        probe_mask=probe_mask.detach().cpu().numpy(),
        frame_ids=frame_ids,
        **{
            f"prediction_step_{result['slide_step']}": result[
                "final_reconstruction"
            ]
            for result in results
        },
    )
    print(f"Saved arrays: {arrays_path}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.residual_vmax is not None and args.residual_vmax <= 0:
        raise ValueError("--residual-vmax must be positive.")
    run_dir = expand_path(args.run_dir)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = run_dir / checkpoint_path
    out_dir = (
        expand_path(args.out_dir)
        if args.out_dir is not None
        else run_dir / "figures_sliding_density_reconstruction"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint["args"]
    stats = checkpoint["stats"]
    window_size = int(checkpoint_args.get("delta_t", 24))
    base_channels = int(checkpoint_args.get("base_channels", 16))
    channel_mults = checkpoint_args.get("channel_mults", [1, 2, 4])
    h5_dir = expand_path(args.h5_dir or checkpoint_args["h5_dir"])
    betas = checkpoint_args.get("betas", [parse_run_name(args.run_name)["beta"]])

    validate_test_run(
        run_dir=run_dir,
        run_name=args.run_name,
        allow_non_validation=args.allow_non_validation_run,
    )
    fields_time_first, frame_ids, h5_path = find_and_load_run(
        h5_dir=h5_dir,
        betas=betas,
        run_name=args.run_name,
    )
    if fields_time_first.shape[0] < window_size:
        raise ValueError(
            f"Run T={fields_time_first.shape[0]} is shorter than model T={window_size}."
        )
    if not (0.0 < args.density_visible_fraction <= 1.0):
        raise ValueError("--density-visible-fraction must lie in (0, 1].")
    slide_steps = [int(step) for step in args.slide_steps]
    if len(set(slide_steps)) != len(slide_steps):
        raise ValueError(f"--slide-steps values must be unique, got {slide_steps}.")
    if args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}.")
    if args.dpi <= 0 or args.animation_dpi <= 0:
        raise ValueError("--dpi and --animation-dpi must both be positive.")
    target = torch.from_numpy(
        np.transpose(fields_time_first, (1, 0, 2, 3))
    ).to(device=device, dtype=torch.float32)
    mean = torch.tensor(stats["mean"], device=device).view(4, 1, 1, 1)
    std = torch.tensor(stats["std"], device=device).view(4, 1, 1, 1)
    target_normalized = (target - mean) / (std + 1e-8)
    if args.plot_units == "normalized":
        target_density_plot = target_normalized[3].detach().cpu().numpy()
    else:
        target_density_plot = target[3].detach().cpu().numpy()

    generator = torch.Generator().manual_seed(args.seed)
    probe_mask_full, probe_info = _density_probe_grid(
        block=target_normalized.unsqueeze(0),
        target_visible_fraction=args.density_visible_fraction,
        generator=generator,
    )
    probe_mask = probe_mask_full[0, 0, 0]
    probe_actual_fraction = float(probe_info["actual_visible_fraction"])

    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=base_channels,
        channel_mults=channel_mults,
        architecture=checkpoint_args.get("model_version", LEGACY_MODEL_VERSION),
        use_attention=bool(checkpoint_args.get("use_attention", False)),
        spatial_only_pooling=bool(
            checkpoint_args.get("spatial_only_pooling", False)
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    if args.x_index is None:
        projection_label = "x-averaged Density(time,z)"
    else:
        if not (0 <= args.x_index < target.shape[2]):
            raise ValueError(
                f"--x-index must be in [0, {target.shape[2] - 1}], got {args.x_index}."
            )
        projection_label = f"Density(time,z) at x index {args.x_index}"
    target_projection = project_time_z(
        target_density_plot,
        x_index=args.x_index,
    )

    print("Checkpoint:", checkpoint_path)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Device:", device)
    print("HDF5:", h5_path)
    print("Run:", args.run_name)
    print("Run field shape (T,C,X,Z):", fields_time_first.shape)
    print("Model window size:", window_size)
    print("Slide steps:", slide_steps)
    print("Probe target fraction:", args.density_visible_fraction)
    print("Probe actual fraction:", probe_actual_fraction)
    print("Hide magnetic:", args.hide_magnetic)
    print("Probe grid:", probe_info)
    print("Projection:", projection_label)
    print("Probe values are clamped only for recursive conditioning.")
    print("Figures and metrics retain raw model predictions at probe positions.")
    print("Normalized residuals use checkpoint preprocessing mean/std.")

    render_slide_steps = args.analysis in {"both", "slide_steps"}
    render_bidirectional = args.analysis in {"both", "bidirectional"}
    steps_to_compute = []
    if render_slide_steps:
        steps_to_compute.extend(slide_steps)
    if render_bidirectional:
        steps_to_compute.extend([args.refinement_step, window_size])
    steps_to_compute = list(dict.fromkeys(steps_to_compute))

    results_by_step = {}
    for slide_step in steps_to_compute:
        print(f"Reconstructing slide step={slide_step}")
        result = reconstruct_with_slide_step(
            model=model,
            target_normalized=target_normalized,
            target_density_plot=target_density_plot,
            density_mean=mean[3],
            density_std=std[3],
            probe_mask=probe_mask,
            window_size=window_size,
            slide_step=slide_step,
            plot_units=args.plot_units,
            x_index=args.x_index,
            amp=True,
            retain_state=(
                render_bidirectional
                and slide_step in {args.refinement_step, window_size}
            ),
            hide_magnetic=args.hide_magnetic,
        )
        print("  starts:", result["window_starts"])
        print("  metrics:", result["metrics"])
        results_by_step[slide_step] = result

    if render_slide_steps:
        save_slide_step_analysis(
            results=[results_by_step[step] for step in slide_steps],
            slide_steps=slide_steps,
            target_density_plot=target_density_plot,
            target_projection=target_projection,
            probe_mask=probe_mask,
            frame_ids=frame_ids,
            extent=args.extent,
            run_name=args.run_name,
            window_size=window_size,
            density_visible_fraction=args.density_visible_fraction,
            probe_actual_fraction=probe_actual_fraction,
            probe_info=probe_info,
            projection_label=projection_label,
            plot_units=args.plot_units,
            field_q=args.field_q,
            residual_q=args.residual_q,
            residual_vmax=args.residual_vmax,
            animation_format=args.animation_format,
            fps=args.fps,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=checkpoint.get("epoch"),
            h5_path=h5_path,
            out_dir=out_dir,
            dpi=args.dpi,
            hide_magnetic=args.hide_magnetic,
        )

    if render_bidirectional:
        print("Building bidirectional repeated-sweep comparison")
        bidirectional_rows = build_bidirectional_rows(
            model=model,
            target_normalized=target_normalized,
            target_density_plot=target_density_plot,
            density_mean=mean[3],
            density_std=std[3],
            probe_mask=probe_mask,
            window_size=window_size,
            refinement_step=args.refinement_step,
            refinement_passes=args.refinement_passes,
            refinement_offset=args.refinement_offset,
            plot_units=args.plot_units,
            x_index=args.x_index,
            amp=True,
            initial_results=results_by_step,
            hide_magnetic=args.hide_magnetic,
        )
        for row_index, row in enumerate(bidirectional_rows, start=1):
            print(
                f"  row={row_index} step={row['slide_step']} "
                f"passes={row['pass_count']} offsets={row['offsets']} "
                f"metrics={row['metrics']}"
            )
        save_bidirectional_analysis(
            rows=bidirectional_rows,
            target_density_plot=target_density_plot,
            target_projection=target_projection,
            probe_mask=probe_mask,
            frame_ids=frame_ids,
            extent=args.extent,
            run_name=args.run_name,
            window_size=window_size,
            refinement_step=args.refinement_step,
            refinement_passes=args.refinement_passes,
            refinement_offset=args.refinement_offset,
            probe_actual_fraction=probe_actual_fraction,
            probe_info=probe_info,
            projection_label=projection_label,
            plot_units=args.plot_units,
            field_q=args.field_q,
            residual_q=args.residual_q,
            residual_vmax=args.residual_vmax,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=checkpoint.get("epoch"),
            h5_path=h5_path,
            out_dir=out_dir,
            dpi=args.dpi,
            animation_dpi=args.animation_dpi,
            animation_format=args.animation_format,
            fps=args.fps,
            hide_magnetic=args.hide_magnetic,
        )


if __name__ == "__main__":
    main()
