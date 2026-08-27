# visualize_mask_patterns_unet3d.py

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

from data.vpic_hdf5_dataset import VPICWindowDataset
from data.masking import MASK_PATTERNS, make_visible_input, sample_mask
from models.unet3d import LEGACY_MODEL_VERSION, UNet3D


DEFAULT_RUN_NAME = "beta0.2_nu2_Bz0_dt2_tau70"
DEFAULT_T0 = 28


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory containing best.pt, stats.json, split.json.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="best.pt",
        help="Checkpoint filename inside run-dir, or an absolute path.",
    )
    p.add_argument(
        "--h5-dir",
        type=str,
        default=None,
        help="Override HDF5 data directory. If None, use checkpoint args.",
    )

    p.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help=(
            "Validation run to visualize (default: the canonical two-plasmoid "
            f"merger run {DEFAULT_RUN_NAME})."
        ),
    )
    p.add_argument(
        "--t0",
        type=int,
        default=DEFAULT_T0,
        help=(
            "First global frame of the temporal window selected with "
            f"--run-name (default: {DEFAULT_T0})."
        ),
    )
    p.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help=(
            "Legacy zero-based index within validation windows. When given, "
            "it overrides --run-name and --t0."
        ),
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--mask-fraction",
        type=float,
        default=0.8,
        help=(
            "Target fraction of hidden voxels for the spatial_random and "
            "temporal_random rows. Visible fraction is about 1 - this value. "
            "spatial_grid and spatial_block use their own settings below."
        ),
    )
    p.add_argument(
        "--block-fraction",
        type=float,
        default=0.5,
        help=(
            "Spatial area covered by the spatial_block rectangle. Position and "
            "aspect ratio stay random."
        ),
    )
    p.add_argument(
        "--grid-stride",
        type=int,
        default=4,
        help=(
            "Density stride for the spatial_grid row. The grid offset stays "
            "random. Default: 4 (one Density observation per 4x4 cell)."
        ),
    )
    p.add_argument(
        "--magnetic-grid-stride",
        type=int,
        default=2,
        help=(
            "Bx/By/Bz stride for the spatial_grid row. Default: 2, making "
            "magnetic observations denser than Density observations."
        ),
    )

    p.add_argument(
        "--local-time",
        type=int,
        default=4,
        help=(
            "One local time index inside the temporal block. Ignored by plotting "
            "when --all-times is set."
        ),
    )
    p.add_argument(
        "--all-times",
        action="store_true",
        help=(
            "Plot all local time slices from the model's temporal window and "
            "combine each figure family into an animation."
        ),
    )
    p.add_argument(
        "--animation-format",
        choices=["quicktime", "mp4", "gif", "both"],
        default="gif",
        help=(
            "Animation output written with --all-times. quicktime writes a "
            "Motion-JPEG .mov that opens in macOS QuickTime; mp4 may use VP9 "
            "when H.264 is unavailable. both writes QuickTime and GIF."
        ),
    )
    p.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Animation frame rate used with --all-times (default: 2).",
    )
    p.add_argument(
        "--mask-patterns",
        type=str,
        nargs="+",
        default=list(MASK_PATTERNS),
        choices=list(MASK_PATTERNS),
        help="Rows to show in the figure.",
    )
    p.add_argument(
        "--experiment",
        choices=[
            "all",
            "multifunction",
            "density_superres",
            "magnetic_ablation",
            "density_forecast",
            "custom",
        ],
        default="all",
        help=(
            "Visualization experiment to render. The default 'all' renders: "
            "Density-only versions of every mask pattern, a Density "
            "super-resolution probe-count sweep, and a magnetic-information "
            "ablation, and a Density forecast-horizon sweep. 'custom' preserves "
            "the original --mask-patterns behavior."
        ),
    )
    p.add_argument(
        "--density-probe-counts",
        type=int,
        nargs="+",
        default=[30, 20, 10, 0],
        help=(
            "Numbers of Density probes for rows in the super-resolution sweep. "
            "Default: 30 20 10 0."
        ),
    )
    p.add_argument(
        "--fixed-density-visible-fraction",
        type=float,
        default=0.08,
        help=(
            "Target Density grid visibility held fixed during the magnetic "
            "information ablation."
        ),
    )
    p.add_argument(
        "--magnetic-visible-fractions",
        type=float,
        nargs="+",
        default=[1.0, 0.8, 0.6, 0.4],
        help=(
            "Magnetic visible fractions for the nested spatial-random ablation."
        ),
    )
    p.add_argument(
        "--density-forecast-visible-frames",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Numbers of complete leading Density frames made visible in the "
            "density_forecast rows. All later Density frames are hidden while "
            "Bx/By/Bz remain visible for the full temporal window. Default for "
            "T=24: 23 18 12 6; other T values use equivalent relative lengths."
        ),
    )

    p.add_argument(
        "--plot-units",
        type=str,
        default="physical",
        choices=["physical", "normalized"],
        help=(
            "physical: plot original/denormalized field values. "
            "normalized: plot channel-normalized values, matching training loss units."
        ),
    )

    p.add_argument(
        "--field-q",
        type=float,
        default=99.0,
        help="Percentile used for robust field color limits.",
    )
    p.add_argument(
        "--residual-q",
        type=float,
        default=99.0,
        help="Percentile used for robust residual color limits.",
    )
    p.add_argument(
        "--residual-vmax",
        type=float,
        default=None,
        help=(
            "Optional fixed symmetric normalized-residual color limit. "
            "If set, use [-vmax, vmax]."
        ),
    )
    p.add_argument(
        "--relative-error-eps",
        type=float,
        default=0.05,
        help=(
            "Stabilizer for pointwise normalized residuals, expressed as a "
            "fraction of target RMS: error / (abs(target) + eps * RMS(target)). "
            "Default: 0.05."
        ),
    )

    p.add_argument(
        "--ay-levels",
        type=int,
        default=15,
        help="Number of Ay contour levels.",
    )
    p.add_argument(
        "--quiver-step",
        type=int,
        default=20,
        help="Spatial subsampling step for in-plane magnetic-field arrows.",
    )
    p.add_argument(
        "--quiver-scale",
        type=float,
        default=15.0,
        help="Matplotlib quiver scale for the in-plane magnetic field.",
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figures. If None, use run-dir/figures_mask_patterns.",
    )

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dpi", type=int, default=180)

    p.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=[-21.0, 21.0, -50.0, 50.0],
        help="imshow extent: zmin zmax xmin xmax.",
    )

    return p.parse_args()


def expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser().resolve()


def load_checkpoint(run_dir: Path, checkpoint_name: str, device: torch.device):
    ckpt_path = Path(checkpoint_name)
    if not ckpt_path.is_absolute():
        ckpt_path = run_dir / checkpoint_name

    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Best val MSE: {ckpt.get('best_val_mse', 'N/A')}")
    return ckpt, ckpt_path


def normalize(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (y - mean) / (std + 1e-8)


def denormalize(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return y * (std + 1e-8) + mean


def get_val_runs(run_dir: Path) -> set[str] | None:
    split_path = run_dir / "split.json"
    if not split_path.exists():
        print(f"No split.json found at {split_path}. Will use all samples.")
        return None

    with open(split_path, "r") as f:
        split = json.load(f)

    val_runs = set(split["val_runs"])
    print(f"Loaded {len(val_runs)} validation runs from split.json")
    return val_runs


def select_sample_index(dataset: VPICWindowDataset, val_runs: set[str] | None, sample_index: int) -> int:
    if sample_index < 0:
        raise IndexError(f"sample_index must be non-negative, got {sample_index}")
    if val_runs is None:
        if sample_index >= len(dataset):
            raise IndexError(f"sample_index={sample_index} out of range for dataset length {len(dataset)}")
        return sample_index

    val_indices: List[int] = []
    for i, (_, run_name, _) in enumerate(dataset.samples):
        if run_name in val_runs:
            val_indices.append(i)

    if len(val_indices) == 0:
        raise RuntimeError("No validation samples found from split.json.")

    if sample_index >= len(val_indices):
        raise IndexError(
            f"sample_index={sample_index} out of range for val sample count {len(val_indices)}"
        )

    return val_indices[sample_index]


def select_run_t0_index(
    dataset: VPICWindowDataset,
    val_runs: set[str] | None,
    run_name: str,
    t0: int,
) -> int:
    """Resolve an exact physical run/window, independently of sample ordering."""
    if t0 < 0:
        raise ValueError(f"t0 must be non-negative, got {t0}")
    if val_runs is not None and run_name not in val_runs:
        raise ValueError(
            f"run_name={run_name!r} is not in split.json validation runs."
        )

    run_windows = [
        (i, int(sample_t0))
        for i, (_, sample_run_name, sample_t0) in enumerate(dataset.samples)
        if sample_run_name == run_name
    ]
    if not run_windows:
        raise ValueError(f"run_name={run_name!r} was not found in the dataset.")

    matches = [i for i, sample_t0 in run_windows if sample_t0 == t0]
    if not matches:
        available_t0 = [sample_t0 for _, sample_t0 in run_windows]
        raise ValueError(
            f"No window found for run_name={run_name!r}, t0={t0}. "
            f"Available t0 range is {min(available_t0)}-{max(available_t0)} "
            f"with values {available_t0}."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple dataset windows for run_name={run_name!r}, t0={t0}."
        )
    return matches[0]


def make_nan_cmap(name: str, bad_color: str = "lightgray"):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color=bad_color)
    return cmap


def robust_limits(
    arrays: Sequence[np.ndarray],
    channel: int,
    q: float = 99.0,
    symmetric: bool | None = None,
):
    vals = []
    for a in arrays:
        aa = np.asarray(a)
        aa = aa[np.isfinite(aa)]
        if aa.size > 0:
            vals.append(aa)

    if len(vals) == 0:
        return -1.0, 1.0

    vals = np.concatenate(vals)

    if symmetric is True:
        vmax = np.nanpercentile(np.abs(vals), q)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        return float(-vmax), float(vmax)

    if symmetric is False:
        vmin = np.nanpercentile(vals, 100.0 - q)
        vmax = np.nanpercentile(vals, q)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    # Default behavior:
    # Density is positive-ish, use asymmetric scale.
    # Magnetic fields are signed, use symmetric scale.
    if channel == 3:
        vmin = np.nanpercentile(vals, 100.0 - q)
        vmax = np.nanpercentile(vals, q)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    vmax = np.nanpercentile(np.abs(vals), q)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return float(-vmax), float(vmax)


def format_mask_label(info: Dict) -> str:
    """
    Short human readable description of one sampled mask layout.
    """
    pattern = info["pattern"]

    if pattern == "spatial_random":
        detail = "shared Bx/By/Bz mask; independent Density mask"
    elif pattern == "spatial_grid":
        if "magnetic_stride" in info:
            detail = (
                f"B stride={info['magnetic_stride']}×{info['magnetic_stride']}, "
                f"offset=({info['magnetic_offset_x']}, {info['magnetic_offset_z']})\n"
                f"Density stride={info['density_stride']}×{info['density_stride']}, "
                f"offset=({info['density_offset_x']}, {info['density_offset_z']})"
            )
        else:
            detail = (
                f"stride={info['stride']}×{info['stride']}\n"
                f"offset=({info['offset_x']}, {info['offset_z']})"
            )
    elif pattern == "spatial_block":
        detail = (
            f"shared B/Density hole={info['rect_height']}×{info['rect_width']} "
            f"at ({info['rect_x0']}, {info['rect_z0']})"
        )
    elif pattern == "temporal_random":
        detail = f"shared B/Density frames={info['visible_frames']}"
    else:
        detail = ""

    return f"{pattern}\n{detail}"


def share_magnetic_channel_mask(mask: torch.Tensor) -> torch.Tensor:
    """Make Bx, By and Bz use exactly the same observation locations."""
    if mask.ndim != 5 or mask.shape[1] < 3:
        raise ValueError(
            f"Expected mask shaped (B, C>=3, T, X, Z), got {tuple(mask.shape)}"
        )

    mask = mask.clone()
    mask[:, 1:3] = mask[:, 0:1]
    return mask


def build_mask_patterns(
    block: torch.Tensor,
    patterns: Sequence[str],
    mask_fraction: float,
    block_fraction: float,
    grid_stride: int,
    magnetic_grid_stride: int,
    generator: torch.Generator,
) -> List[Tuple[str, str, torch.Tensor]]:
    """
    Return list of:
        (short_name, display_label, visible_mask)

    Masks come from data.masking so that the figure shows the same mask family
    the model was trained on. spatial_grid and spatial_block are pinned to a
    standard, interpretable benchmark geometry instead of following the shared
    mask fraction; their position and grid offset stay random.
    """
    rows = []

    for name in patterns:
        if name == "spatial_grid":
            # Visualization-only benchmark geometry: magnetic diagnostics are
            # sampled more densely than Density, matching the intended probe
            # arrangement. Training mask sampling is deliberately unchanged.
            magnetic_mask, magnetic_info = sample_mask(
                block.shape,
                pattern=name,
                mask_fraction=mask_fraction,
                device=block.device,
                dtype=block.dtype,
                generator=generator,
                grid_stride=magnetic_grid_stride,
            )
            density_mask, density_info = sample_mask(
                block.shape,
                pattern=name,
                mask_fraction=mask_fraction,
                device=block.device,
                dtype=block.dtype,
                generator=generator,
                grid_stride=grid_stride,
            )

            mask = magnetic_mask.clone()
            mask[:, 3:4] = density_mask[:, 3:4]
            mask = share_magnetic_channel_mask(mask)
            info = {
                "pattern": name,
                "target_mask_fraction": mask_fraction,
                "actual_mask_fraction": float(1.0 - mask.mean().item()),
                "magnetic_stride": magnetic_info["stride"],
                "magnetic_offset_x": magnetic_info["offset_x"],
                "magnetic_offset_z": magnetic_info["offset_z"],
                "density_stride": density_info["density_stride"],
                "density_offset_x": density_info["density_offset_x"],
                "density_offset_z": density_info["density_offset_z"],
            }
            rows.append((name, format_mask_label(info), mask))
            continue

        mask, info = sample_mask(
            block.shape,
            pattern=name,
            mask_fraction=block_fraction if name == "spatial_block" else mask_fraction,
            device=block.device,
            dtype=block.dtype,
            generator=generator,
        )
        mask = share_magnetic_channel_mask(mask)
        info["actual_mask_fraction"] = float(1.0 - mask.mean().item())
        rows.append((name, format_mask_label(info), mask))

    return rows


def _validate_visible_fractions(
    values: Sequence[float],
    option_name: str,
    allow_zero: bool = False,
) -> List[float]:
    lower = 0.0 if allow_zero else np.nextafter(0.0, 1.0)
    parsed = [float(value) for value in values]
    if not parsed:
        raise ValueError(f"{option_name} requires at least one value.")
    for value in parsed:
        if not (lower <= value <= 1.0):
            interval = "[0, 1]" if allow_zero else "(0, 1]"
            raise ValueError(
                f"{option_name} values must lie in {interval}, got {value}."
            )
    return parsed


def build_density_only_multifunction_rows(
    block: torch.Tensor,
    patterns: Sequence[str],
    mask_fraction: float,
    block_fraction: float,
    grid_stride: int,
    magnetic_grid_stride: int,
    generator: torch.Generator,
) -> List[Tuple[str, str, torch.Tensor]]:
    """Apply every requested topology to Density while keeping all B visible."""
    sampled_rows = build_mask_patterns(
        block=block,
        patterns=patterns,
        mask_fraction=mask_fraction,
        block_fraction=block_fraction,
        grid_stride=grid_stride,
        magnetic_grid_stride=magnetic_grid_stride,
        generator=generator,
    )
    labels = {
        "spatial_random": "Density spatial_random\nB visible=100%; Density random probes",
        "spatial_grid": (
            f"Density spatial_grid\nB visible=100%; Density stride="
            f"{grid_stride}x{grid_stride}"
        ),
        "spatial_block": "Density spatial_block\nB visible=100%; Density-only hole",
        "temporal_random": (
            "Density temporal_random\nB visible=100%; Density-only hidden frames"
        ),
    }

    rows = []
    for name, _old_label, mask in sampled_rows:
        mask = mask.clone()
        mask[:, :3] = 1.0
        if name == "spatial_block":
            # Pin this visualization-only Density hole to the upper half in x,
            # where the plasmoids occur in the selected merger window.
            mask[:, 3:4] = 1.0
            mask[:, 3:4, :, block.shape[-2] // 2 :, :] = 0.0
        rows.append((name, labels[name], mask))
    return rows


def _density_probe_grid(
    block: torch.Tensor,
    target_visible_fraction: float,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, float | int]]:
    """Create a near-isotropic regular grid close to a requested probe ratio."""
    # Allow mildly rectangular cells when that materially improves the target
    # ratio (notably 8%, for which no square integer stride exists).
    candidates = []
    for stride_x in range(1, 33):
        for stride_z in range(1, 33):
            nominal = 1.0 / float(stride_x * stride_z)
            relative_error = abs(nominal - target_visible_fraction) / target_visible_fraction
            anisotropy = abs(stride_x - stride_z) / max(stride_x, stride_z)
            candidates.append(
                (relative_error + 0.05 * anisotropy, anisotropy, stride_x, stride_z)
            )
    batch, _channels, time, size_x, size_z = block.shape
    _score, _anisotropy, stride_x, stride_z = min(candidates)
    offset_candidates = []
    for candidate_x in range(stride_x):
        count_x = (size_x - 1 - candidate_x) // stride_x + 1
        for candidate_z in range(stride_z):
            count_z = (size_z - 1 - candidate_z) // stride_z + 1
            actual = float(count_x * count_z) / float(size_x * size_z)
            offset_candidates.append(
                (abs(actual - target_visible_fraction), candidate_x, candidate_z)
            )
    best_offset_error = min(item[0] for item in offset_candidates)
    best_offsets = [
        item for item in offset_candidates if np.isclose(item[0], best_offset_error)
    ]
    selected_offset = int(
        torch.randint(len(best_offsets), (1,), generator=generator).item()
    )
    _offset_error, offset_x, offset_z = best_offsets[selected_offset]

    plane = torch.zeros(
        (1, 1, 1, size_x, size_z),
        device=block.device,
        dtype=block.dtype,
    )
    plane[..., offset_x::stride_x, offset_z::stride_z] = 1.0
    mask = plane.expand(batch, 1, time, size_x, size_z).contiguous()
    info = {
        "stride_x": stride_x,
        "stride_z": stride_z,
        "offset_x": offset_x,
        "offset_z": offset_z,
        "actual_visible_fraction": float(mask.mean().item()),
    }
    return mask, info


def _density_probe_count_grid(
    block: torch.Tensor,
    probe_count: int,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """Create an approximately isotropic regular grid with exactly N probes."""
    batch, _channels, time, size_x, size_z = block.shape
    if not (0 <= probe_count <= size_x * size_z):
        raise ValueError(
            f"Density probe count must lie in [0, {size_x * size_z}], "
            f"got {probe_count}."
        )

    plane = torch.zeros(
        (1, 1, 1, size_x, size_z),
        device=block.device,
        dtype=block.dtype,
    )
    if probe_count == 0:
        return plane.expand(batch, 1, time, size_x, size_z).contiguous(), {
            "count_x": 0,
            "count_z": 0,
        }

    aspect = float(size_x) / float(size_z)
    factor_pairs = [
        (probe_count // count_z, count_z)
        for count_z in range(1, probe_count + 1)
        if probe_count % count_z == 0
    ]
    count_x, count_z = min(
        factor_pairs,
        key=lambda pair: abs(np.log((pair[0] / pair[1]) / aspect)),
    )
    x_indices = torch.linspace(
        0, size_x - 1, count_x, device=block.device
    ).round().long()
    z_indices = torch.linspace(
        0, size_z - 1, count_z, device=block.device
    ).round().long()
    plane[..., x_indices[:, None], z_indices[None, :]] = 1.0
    mask = plane.expand(batch, 1, time, size_x, size_z).contiguous()
    return mask, {"count_x": count_x, "count_z": count_z}


def build_density_superres_rows(
    block: torch.Tensor,
    probe_counts: Sequence[int],
) -> List[Tuple[str, str, torch.Tensor]]:
    """Sweep exact Density probe counts while keeping the full B field visible."""
    rows = []
    for probe_count in probe_counts:
        density_mask, info = _density_probe_count_grid(
            block=block,
            probe_count=int(probe_count),
        )
        mask = torch.ones_like(block)
        mask[:, 3:4] = density_mask
        label = (
            "Density super-resolution\n"
            f"B visible=100%; Density probes={probe_count}\n"
            f"probe grid={info['count_x']}x{info['count_z']}"
        )
        rows.append((f"density_superres_{probe_count}", label, mask))
    return rows


def build_magnetic_ablation_rows(
    block: torch.Tensor,
    magnetic_visible_fractions: Sequence[float],
    density_visible_fraction: float,
    generator: torch.Generator,
) -> List[Tuple[str, str, torch.Tensor]]:
    """
    Hold one Density grid fixed and progressively remove nested magnetic probes.

    The same spatial ranking is used for all rows, so every lower-visibility B
    layout is a strict subset of the preceding higher-visibility layout.
    """
    density_plane, density_info = _density_probe_grid(
        block=block,
        target_visible_fraction=density_visible_fraction,
        generator=generator,
    )
    density_actual = float(density_info["actual_visible_fraction"])

    batch, _channels, time, size_x, size_z = block.shape
    num_sites = int(size_x * size_z)
    permutation = torch.randperm(num_sites, generator=generator)
    rank = torch.empty(num_sites, dtype=torch.long)
    rank[permutation] = torch.arange(num_sites)

    rows = []
    for visible_fraction in magnetic_visible_fractions:
        num_visible = int(round(visible_fraction * num_sites))
        magnetic_plane = (rank < num_visible).reshape(1, 1, 1, size_x, size_z)
        magnetic_plane = magnetic_plane.to(device=block.device, dtype=block.dtype)
        magnetic_mask = magnetic_plane.expand(batch, 3, time, size_x, size_z)

        mask = torch.cat([magnetic_mask, density_plane], dim=1).contiguous()
        magnetic_actual = float(magnetic_mask.mean().item())
        label = (
            "Magnetic information ablation\n"
            f"B visible={100.0 * magnetic_actual:.2f}% (nested random)\n"
            f"Density grid target={100.0 * density_visible_fraction:g}%, "
            f"actual={100.0 * density_actual:.2f}%\n"
            f"Density stride={density_info['stride_x']}x{density_info['stride_z']}"
        )
        rows.append((f"magnetic_ablation_{visible_fraction:g}", label, mask))
    return rows


def build_density_forecast_rows(
    block: torch.Tensor,
    visible_frame_counts: Sequence[int],
) -> List[Tuple[str, str, torch.Tensor]]:
    """
    Build causal Density-prefix masks with the magnetic field always visible.

    Every visible Density frame is spatially complete. Density is fully hidden
    after the prefix, so the final slice measures conditional forecast accuracy
    at progressively longer horizons given the complete Bx/By/Bz sequence.
    """
    time = int(block.shape[2])
    counts = [int(count) for count in visible_frame_counts]
    if not counts:
        raise ValueError("--density-forecast-visible-frames requires at least one value.")
    if len(set(counts)) != len(counts):
        raise ValueError(
            "--density-forecast-visible-frames values must be unique, "
            f"got {counts}."
        )

    rows = []
    for count in counts:
        if not (1 <= count < time):
            raise ValueError(
                "--density-forecast-visible-frames values must be in "
                f"[1, T-1]=[1, {time - 1}], got {count}."
            )

        mask = torch.ones_like(block)
        mask[:, 3:4, count:] = 0.0
        horizon = time - count
        label = (
            "Conditional Density forecast\n"
            f"B visible=100% for frames 1-{time}\n"
            f"Density visible=frames 1-{count} (spatially complete)\n"
            f"target=frame {time}; forecast horizon={horizon} step"
            f"{'s' if horizon != 1 else ''}"
        )
        rows.append((f"density_forecast_history_{count}", label, mask))
    return rows


def default_density_forecast_visible_frames(time: int) -> List[int]:
    """Return T=24 -> [23, 18, 12, 6], scaled sensibly for legacy windows."""
    if time < 2:
        raise ValueError(f"density_forecast requires at least two frames, got T={time}.")
    candidates = [time - 1, round(0.75 * time), round(0.50 * time), round(0.25 * time)]
    return list(dict.fromkeys(min(max(int(value), 1), time - 1) for value in candidates))


def build_experiment_mask_rows(
    args: argparse.Namespace,
    block: torch.Tensor,
    generator: torch.Generator,
) -> List[Tuple[str, List[Tuple[str, str, torch.Tensor]]]]:
    """Build the selected table groups in their requested display order."""
    fixed_density_fraction = _validate_visible_fractions(
        [args.fixed_density_visible_fraction],
        "--fixed-density-visible-fraction",
    )[0]
    magnetic_fractions = _validate_visible_fractions(
        args.magnetic_visible_fractions,
        "--magnetic-visible-fractions",
        allow_zero=True,
    )

    selected = (
        [
            "multifunction",
            "density_superres",
            "magnetic_ablation",
            "density_forecast",
        ]
        if args.experiment == "all"
        else [args.experiment]
    )
    experiments = []
    for experiment in selected:
        if experiment == "multifunction":
            rows = build_density_only_multifunction_rows(
                block=block,
                patterns=args.mask_patterns,
                mask_fraction=args.mask_fraction,
                block_fraction=args.block_fraction,
                grid_stride=args.grid_stride,
                magnetic_grid_stride=args.magnetic_grid_stride,
                generator=generator,
            )
        elif experiment == "density_superres":
            rows = build_density_superres_rows(
                block=block,
                probe_counts=args.density_probe_counts,
            )
        elif experiment == "magnetic_ablation":
            rows = build_magnetic_ablation_rows(
                block=block,
                magnetic_visible_fractions=magnetic_fractions,
                density_visible_fraction=fixed_density_fraction,
                generator=generator,
            )
        elif experiment == "density_forecast":
            rows = build_density_forecast_rows(
                block=block,
                visible_frame_counts=args.density_forecast_visible_frames,
            )
        elif experiment == "custom":
            rows = build_mask_patterns(
                block=block,
                patterns=args.mask_patterns,
                mask_fraction=args.mask_fraction,
                block_fraction=args.block_fraction,
                grid_stride=args.grid_stride,
                magnetic_grid_stride=args.magnetic_grid_stride,
                generator=generator,
            )
        else:
            raise ValueError(f"Unhandled experiment: {experiment}")
        experiments.append((experiment, rows))
    return experiments


def relative_error_epsilon(
    targets: Sequence[np.ndarray],
    epsilon_fraction: float,
) -> float:
    """Return an absolute denominator floor tied to the target RMS scale."""
    finite_values = []
    for target in targets:
        values = np.asarray(target, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size:
            finite_values.append(values)
    if not finite_values:
        return float(np.finfo(np.float64).eps)
    values = np.concatenate(finite_values)
    target_rms = float(np.sqrt(np.mean(values**2)))
    return max(
        float(epsilon_fraction) * target_rms,
        float(np.finfo(np.float64).eps),
    )


def normalized_residual(
    pred: np.ndarray,
    target: np.ndarray,
    epsilon_abs: float,
) -> np.ndarray:
    """Return signed, stabilized pointwise relative residuals."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    residual = (pred - target) / (np.abs(target) + float(epsilon_abs))
    residual[~(np.isfinite(pred) & np.isfinite(target))] = np.nan
    return residual


def compute_normalized_metrics(residual: np.ndarray) -> Tuple[float, float]:
    """Return RMS and mean absolute values of a normalized residual array."""
    residual = np.asarray(residual, dtype=np.float64)
    residual = residual[np.isfinite(residual)]

    if residual.size == 0:
        return float("nan"), float("nan")

    nrmse = float(np.sqrt(np.mean(residual**2)))
    nmae = float(np.mean(np.abs(residual)))
    return nrmse, nmae


def physical_coordinates(shape: Sequence[int], extent: Sequence[float]):
    """Return x-z coordinates for an (X, Z) field and imshow extent."""
    X, Z = int(shape[-2]), int(shape[-1])
    zmin, zmax, xmin, xmax = (float(v) for v in extent)
    x = np.linspace(xmin, xmax, X)
    z = np.linspace(zmin, zmax, Z)
    return x, z


def compute_ay_jy(
    field: np.ndarray,
    extent: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive Ay and Jy from a complete (C, T, X, Z) field.

    The sign convention matches visualization.ipynb:
        Bx = -dAy/dz, Bz = dAy/dx
        Jy = dBx/dz - dBz/dx

    Ay is path-integrated exactly as in the notebook and its arbitrary additive
    constant is removed independently for every frame.  This function is only
    called for complete target and prediction fields, never masked input.
    """
    if field.ndim != 4 or field.shape[0] < 3:
        raise ValueError(f"Expected (C, T, X, Z) with Bx/Bz channels, got {field.shape}")

    bx = np.asarray(field[0], dtype=np.float64)
    bz = np.asarray(field[2], dtype=np.float64)
    T, X, Z = bx.shape
    x, z = physical_coordinates((X, Z), extent)

    ay = np.zeros((T, X, Z), dtype=np.float64)
    jy = np.empty((T, X, Z), dtype=np.float64)

    for t in range(T):
        # First integrate Bz along x at the left z boundary, then integrate
        # -Bx along z for every x. This is the same path used in the notebook.
        for ix in range(1, X):
            dx = x[ix] - x[ix - 1]
            ay[t, ix, 0] = ay[t, ix - 1, 0] + 0.5 * (
                bz[t, ix - 1, 0] + bz[t, ix, 0]
            ) * dx

        for iz in range(1, Z):
            dz = z[iz] - z[iz - 1]
            ay[t, :, iz] = ay[t, :, iz - 1] - 0.5 * (
                bx[t, :, iz - 1] + bx[t, :, iz]
            ) * dz

        ay[t] -= np.mean(ay[t])
        d_bx_dz = np.gradient(bx[t], z, axis=1)
        d_bz_dx = np.gradient(bz[t], x, axis=0)
        jy[t] = d_bx_dz - d_bz_dx

    return ay, jy


def add_ay_contours(
    ax,
    ay: np.ndarray,
    extent: Sequence[float],
    levels: int,
    color: str,
) -> None:
    """Overlay Ay contours, skipping degenerate fields cleanly."""
    if levels <= 0 or not np.isfinite(ay).any():
        return
    amin = float(np.nanmin(ay))
    amax = float(np.nanmax(ay))
    if np.isclose(amin, amax):
        return

    x, z = physical_coordinates(ay.shape, extent)
    zz, xx = np.meshgrid(z, x)
    ax.contour(zz, xx, ay, levels=levels, colors=color, linewidths=0.6)


def add_inplane_quiver(
    ax,
    bx: np.ndarray,
    bz: np.ndarray,
    extent: Sequence[float],
    step: int,
    scale: float,
    visible_mask: np.ndarray | None = None,
    color: str = "black",
) -> None:
    """Overlay (Bz, Bx) arrows, optionally only where both fields are visible."""
    step = max(1, int(step))
    x, z = physical_coordinates(bx.shape, extent)

    if visible_mask is None:
        zz, xx = np.meshgrid(z[::step], x[::step])
        u = np.asarray(bz[::step, ::step])
        v = np.asarray(bx[::step, ::step])
    else:
        # Pick at most one actually visible probe from each step x step block.
        # Sampling a fixed [::step, ::step] lattice can miss an offset probe
        # grid completely, which would incorrectly show no observed arrows.
        visible = np.asarray(visible_mask) > 0.5
        selected_x = []
        selected_z = []

        for x0 in range(0, visible.shape[0], step):
            for z0 in range(0, visible.shape[1], step):
                block = visible[
                    x0 : min(x0 + step, visible.shape[0]),
                    z0 : min(z0 + step, visible.shape[1]),
                ]
                candidates = np.argwhere(block)
                if candidates.size == 0:
                    continue

                center = np.array([(block.shape[0] - 1) / 2, (block.shape[1] - 1) / 2])
                local_ix, local_iz = candidates[
                    np.argmin(np.sum((candidates - center) ** 2, axis=1))
                ]
                selected_x.append(x0 + int(local_ix))
                selected_z.append(z0 + int(local_iz))

        if not selected_x:
            return

        selected_x = np.asarray(selected_x, dtype=np.int64)
        selected_z = np.asarray(selected_z, dtype=np.int64)
        xx = x[selected_x]
        zz = z[selected_z]
        u = np.asarray(bz[selected_x, selected_z])
        v = np.asarray(bx[selected_x, selected_z])

    ax.quiver(
        zz,
        xx,
        u,
        v,
        color=color,
        scale=scale,
        width=0.002,
    )


def _make_comparison_axes(n_rows: int):
    """Create Target | Visible/mask | Prediction | Residual comparison axes."""
    fig = plt.figure(
        figsize=(13.0, 5 * n_rows + 2),
        constrained_layout=False,
    )
    gs = gridspec.GridSpec(
        nrows=n_rows,
        ncols=6,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 0.035, 1.0, 0.035],
        wspace=0.055,
        hspace=0.075,
        left=0.16,
        right=0.965,
        bottom=0.075,
        top=0.90,
    )

    axes = np.empty((n_rows, 4), dtype=object)
    base_ax = None
    for r in range(n_rows):
        for c, gcol in enumerate([0, 1, 2, 4]):
            if base_ax is None:
                ax = fig.add_subplot(gs[r, gcol])
                base_ax = ax
            else:
                ax = fig.add_subplot(gs[r, gcol], sharex=base_ax, sharey=base_ax)
            axes[r, c] = ax

    return fig, axes, fig.add_subplot(gs[:, 3]), fig.add_subplot(gs[:, 5])


def _style_comparison_axis(ax, row: int, col: int, n_rows: int) -> None:
    if row < n_rows - 1:
        ax.tick_params(labelbottom=False)
    if col > 0:
        ax.tick_params(labelleft=False)
    ax.tick_params(axis="both", which="both", labelsize=8, length=2.5)


def _figure_context(metadata: Dict, local_time: int) -> str:
    t0 = metadata.get("t0", "unknown")
    global_frame = int(t0) + int(local_time)
    return (
        f"{metadata.get('run_name', 'unknown')}, block t0={t0}, "
        f"local t={local_time}, global frame={global_frame}, "
        f"beta={metadata.get('beta', 'unknown')}, "
        f"nu={metadata.get('nu', 'unknown')}, "
        f"Bz0={metadata.get('Bz0', 'unknown')}, "
        f"tau={metadata.get('tau', 'unknown')}"
    )


def write_animation(
    frame_paths: Sequence[Path],
    out_path: Path,
    fps: float,
) -> None:
    """Encode already-rendered PNG frames without repeating model inference."""
    if not frame_paths:
        raise ValueError("Cannot create an animation without frames.")
    if fps <= 0:
        raise ValueError(f"--fps must be positive, got {fps}")

    suffix = out_path.suffix.lower()
    if suffix in {".mp4", ".mov"}:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "MP4/QuickTime output requires ffmpeg on PATH. Use "
                "`--animation-format gif` when ffmpeg is unavailable."
            )

        encoder_listing = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if suffix == ".mov":
            if "mjpeg" not in encoder_listing:
                raise RuntimeError(
                    "QuickTime output requires ffmpeg's Motion-JPEG encoder, "
                    "but it is not enabled on this system."
                )
            codec_args = ["-c:v", "mjpeg", "-q:v", "2"]
            pixel_format = "yuvj420p"
        elif "libx264" in encoder_listing:
            codec_args = ["-c:v", "libx264", "-crf", "20"]
            pixel_format = "yuv420p"
        elif "libvpx-vp9" in encoder_listing:
            codec_args = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0"]
            pixel_format = "yuv420p"
        else:
            raise RuntimeError(
                "ffmpeg is available, but neither libx264 nor libvpx-vp9 is enabled. "
                "Use `--animation-format gif` on this system."
            )

        # Use sequential links so ffmpeg receives an unambiguous frame order,
        # independent of the long descriptive PNG filenames.
        with tempfile.TemporaryDirectory(
            prefix=".animation-frames-",
            dir=out_path.parent,
        ) as temp_dir:
            temp_dir_path = Path(temp_dir)
            for i, frame_path in enumerate(frame_paths):
                os.symlink(
                    frame_path.resolve(),
                    temp_dir_path / f"frame_{i:03d}.png",
                )

            command = [
                ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(fps),
                "-i",
                str(temp_dir_path / "frame_%03d.png"),
            ]
            command.extend(codec_args)
            command.extend(
                [
                    "-pix_fmt",
                    pixel_format,
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-movflags",
                    "+faststart",
                    str(out_path),
                ]
            )
            subprocess.run(
                command,
                check=True,
            )
    elif suffix == ".gif":
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "GIF output requires Pillow. Install it with `pip install pillow`, "
                "or use `--animation-format mp4`."
            ) from exc

        frames = []
        for frame_path in frame_paths:
            with Image.open(frame_path) as image:
                frames.append(
                    image.convert("RGB").convert("P", palette=Image.ADAPTIVE)
                )

        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=max(1, int(round(1000.0 / fps))),
            optimize=False,
            disposal=2,
        )
    else:
        raise ValueError(f"Unsupported animation extension: {out_path.suffix}")

    print(f"Saved animation: {out_path}")


def combine_table_images(
    magnetic_path: Path,
    density_path: Path,
    out_path: Path,
    gap_pixels: int = 16,
) -> None:
    """Place the magnetic/Jy table left and the Density table right."""
    from PIL import Image

    with Image.open(magnetic_path) as magnetic_image, Image.open(density_path) as density_image:
        magnetic_rgb = magnetic_image.convert("RGB")
        density_rgb = density_image.convert("RGB")
        height = max(magnetic_rgb.height, density_rgb.height)
        canvas = Image.new(
            "RGB",
            (magnetic_rgb.width + gap_pixels + density_rgb.width, height),
            color="white",
        )
        canvas.paste(magnetic_rgb, (0, 0))
        canvas.paste(density_rgb, (magnetic_rgb.width + gap_pixels, 0))
        canvas.save(out_path)
    print(f"Saved combined 1x2 frame: {out_path}")


def plot_jy_ay_by_mask_patterns(
    target_ay: np.ndarray,
    target_jy: np.ndarray,
    target_field: np.ndarray,
    rows: List[Dict],
    metadata: Dict,
    local_time: int,
    out_path: Path,
    extent: Sequence[float],
    plot_units: str,
    ay_levels: int,
    field_q: float,
    residual_q: float,
    residual_vmax: float | None,
    relative_error_eps: float,
    quiver_step: int,
    quiver_scale: float,
    dpi: int,
    limit_times: Sequence[int] | None = None,
) -> None:
    """
    Plot Jy with Ay contours for complete target/prediction fields.

    The masked-input column shows target Jy only where all three B channels are
    observed. Jy is computed from the complete target first and then masked; it
    is never differentiated from incomplete B observations.
    """
    n_rows = len(rows)
    fig, axes, cax_field, cax_residual = _make_comparison_axes(n_rows)

    target = target_jy[local_time]
    pred_arrays = [row["pred_jy"][local_time] for row in rows]
    color_times = [local_time] if limit_times is None else list(limit_times)
    all_target_jy = [target_jy[t] for t in color_times]
    all_pred_jy = [row["pred_jy"][t] for row in rows for t in color_times]
    epsilon_abs = relative_error_epsilon(all_target_jy, relative_error_eps)
    residual_arrays = [
        normalized_residual(pred, target, epsilon_abs) for pred in pred_arrays
    ]
    all_residual_jy = [
        normalized_residual(row["pred_jy"][t], target_jy[t], epsilon_abs)
        for row in rows
        for t in color_times
    ]
    jy_vmin, jy_vmax = robust_limits(
        all_target_jy + all_pred_jy,
        channel=0,
        q=field_q,
        symmetric=True,
    )
    if residual_vmax is None:
        res_vmin, res_vmax = robust_limits(
            all_residual_jy,
            channel=0,
            q=residual_q,
            symmetric=True,
        )
    else:
        res_vmin, res_vmax = -float(residual_vmax), float(residual_vmax)

    jy_cmap = make_nan_cmap("seismic", bad_color="black")
    residual_cmap = make_nan_cmap("PRGn")
    field_im = None
    residual_im = None

    for r, row in enumerate(rows):
        pred = pred_arrays[r]
        residual = residual_arrays[r]
        joint_b_mask = np.minimum.reduce(
            [
                row["mask"][0, local_time],
                row["mask"][1, local_time],
                row["mask"][2, local_time],
            ]
        )
        visible_jy = target.copy()
        visible_jy[joint_b_mask < 0.5] = np.nan
        visible_pct = 100.0 * float(np.mean(joint_b_mask))
        nrmse, nmae = compute_normalized_metrics(residual)
        row_label = (
            f"{row['label']}\n"
            f"joint B visible={visible_pct:.2f}%\n"
            f"Jy NRMSE={nrmse:.3g}\n"
            f"Jy NMAE={nmae:.3g}"
        )

        field_im = axes[r, 0].imshow(
            target,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=jy_cmap,
            vmin=jy_vmin,
            vmax=jy_vmax,
            interpolation="nearest",
        )
        add_ay_contours(
            axes[r, 0], target_ay[local_time], extent, ay_levels, color="black"
        )

        axes[r, 1].imshow(
            visible_jy,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=jy_cmap,
            vmin=jy_vmin,
            vmax=jy_vmax,
            interpolation="nearest",
        )
        if visible_pct == 0.0:
            axes[r, 1].text(
                0.5,
                0.5,
                "Frame hidden",
                transform=axes[r, 1].transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )

        axes[r, 2].imshow(
            pred,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=jy_cmap,
            vmin=jy_vmin,
            vmax=jy_vmax,
            interpolation="nearest",
        )
        add_ay_contours(
            axes[r, 2], row["pred_ay"][local_time], extent, ay_levels, color="black"
        )

        residual_im = axes[r, 3].imshow(
            residual,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=residual_cmap,
            vmin=res_vmin,
            vmax=res_vmax,
            interpolation="nearest",
        )
        axes[r, 0].text(
            -0.33,
            0.5,
            row_label,
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=8,
            linespacing=1.05,
        )

        for c in range(4):
            if r == 0:
                axes[r, c].set_title(
                    [
                        "Target Jy + Ay",
                        "Masked target Jy",
                        "Prediction Jy + Ay",
                        "Normalized Jy residual",
                    ][c],
                    fontsize=11,
                    pad=4,
                )
            _style_comparison_axis(axes[r, c], r, c, n_rows)

    cb_field = fig.colorbar(field_im, cax=cax_field)
    cb_field.set_label(f"Jy ({plot_units})", fontsize=9, labelpad=8)
    cb_field.ax.tick_params(labelsize=8, length=2.5)
    cb_res = fig.colorbar(residual_im, cax=cax_residual)
    cb_res.set_label(
        "(Prediction − target) / "
        f"(|target| + {relative_error_eps:g} × target RMS)",
        fontsize=9,
        labelpad=8,
    )
    cb_res.ax.tick_params(labelsize=8, length=2.5)

    fig.text(0.545, 0.032, "z [cm]", ha="center", va="center", fontsize=10)
    fig.text(0.055, 0.50, "x [cm]", ha="center", va="center", rotation=90, fontsize=10)
    fig.suptitle(
        f"Current density Jy and magnetic-potential Ay contours ({plot_units} units)\n"
        + _figure_context(metadata, local_time),
        fontsize=12,
        y=0.985,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_density_magnetic_field_by_mask_patterns(
    target_field: np.ndarray,
    target_ay: np.ndarray,
    rows: List[Dict],
    metadata: Dict,
    local_time: int,
    out_path: Path,
    extent: Sequence[float],
    plot_units: str,
    ay_levels: int,
    quiver_step: int,
    quiver_scale: float,
    field_q: float,
    residual_q: float,
    residual_vmax: float | None,
    relative_error_eps: float,
    dpi: int,
    limit_times: Sequence[int] | None = None,
) -> None:
    """Plot Density with Ay contours and in-plane (Bz, Bx) arrows."""
    n_rows = len(rows)
    fig, axes, cax_field, cax_residual = _make_comparison_axes(n_rows)

    target_density = target_field[3, local_time]
    pred_arrays = [row["pred_plot"][3, local_time] for row in rows]
    color_times = [local_time] if limit_times is None else list(limit_times)
    all_target_density = [target_field[3, t] for t in color_times]
    all_pred_density = [
        row["pred_plot"][3, t] for row in rows for t in color_times
    ]
    epsilon_abs = relative_error_epsilon(all_target_density, relative_error_eps)
    residual_arrays = [
        normalized_residual(pred, target_density, epsilon_abs)
        for pred in pred_arrays
    ]
    all_residual_density = [
        normalized_residual(
            row["pred_plot"][3, t],
            target_field[3, t],
            epsilon_abs,
        )
        for row in rows
        for t in color_times
    ]
    density_vmin, density_vmax = robust_limits(
        all_target_density + all_pred_density,
        channel=3,
        q=field_q,
        symmetric=False,
    )
    if residual_vmax is None:
        res_vmin, res_vmax = robust_limits(
            all_residual_density,
            channel=3,
            q=residual_q,
            symmetric=True,
        )
    else:
        res_vmin, res_vmax = -float(residual_vmax), float(residual_vmax)

    density_cmap = make_nan_cmap("plasma", bad_color="black")
    residual_cmap = make_nan_cmap("PRGn")
    field_im = None
    residual_im = None

    for r, row in enumerate(rows):
        pred_field = row["pred_plot"]
        pred_density = pred_arrays[r]
        residual = residual_arrays[r]
        density_mask = row["mask"][3, local_time]
        joint_b_mask = np.minimum.reduce(
            [
                row["mask"][0, local_time],
                row["mask"][1, local_time],
                row["mask"][2, local_time],
            ]
        )
        visible_pct = 100.0 * float(np.mean(density_mask))
        nrmse, nmae = compute_normalized_metrics(residual)
        row_label = (
            f"{row['label']}\n"
            f"Density visible={visible_pct:.2f}%\n"
            f"Density NRMSE={nrmse:.3g}\n"
            f"Density NMAE={nmae:.3g}"
        )

        field_im = axes[r, 0].imshow(
            target_density,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=density_cmap,
            vmin=density_vmin,
            vmax=density_vmax,
            interpolation="nearest",
        )
        add_ay_contours(
            axes[r, 0], target_ay[local_time], extent, ay_levels, color="white"
        )
        add_inplane_quiver(
            axes[r, 0],
            target_field[0, local_time],
            target_field[2, local_time],
            extent,
            quiver_step,
            quiver_scale,
        )

        axes[r, 1].imshow(
            row["visible_plot"][3, local_time],
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=density_cmap,
            vmin=density_vmin,
            vmax=density_vmax,
            interpolation="nearest",
        )
        add_inplane_quiver(
            axes[r, 1],
            target_field[0, local_time],
            target_field[2, local_time],
            extent,
            quiver_step,
            quiver_scale,
            visible_mask=joint_b_mask,
            color="cyan",
        )
        if visible_pct == 0.0:
            axes[r, 1].text(
                0.5,
                0.5,
                "Frame hidden",
                transform=axes[r, 1].transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=10,
            )

        axes[r, 2].imshow(
            pred_density,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=density_cmap,
            vmin=density_vmin,
            vmax=density_vmax,
            interpolation="nearest",
        )
        add_ay_contours(
            axes[r, 2], row["pred_ay"][local_time], extent, ay_levels, color="white"
        )
        add_inplane_quiver(
            axes[r, 2],
            pred_field[0, local_time],
            pred_field[2, local_time],
            extent,
            quiver_step,
            quiver_scale,
        )

        residual_im = axes[r, 3].imshow(
            residual,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=residual_cmap,
            vmin=res_vmin,
            vmax=res_vmax,
            interpolation="nearest",
        )

        axes[r, 0].text(
            -0.33,
            0.5,
            row_label,
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=8,
            linespacing=1.05,
        )

        for c in range(4):
            if r == 0:
                axes[r, c].set_title(
                    [
                        "Target Density + Ay/B",
                        "Visible Density + visible B",
                        "Prediction Density + Ay/B",
                        "Normalized Density residual",
                    ][c],
                    fontsize=11,
                    pad=4,
                )
            _style_comparison_axis(axes[r, c], r, c, n_rows)

    cb_field = fig.colorbar(field_im, cax=cax_field)
    cb_field.set_label(f"Density ({plot_units})", fontsize=9, labelpad=8)
    cb_field.ax.tick_params(labelsize=8, length=2.5)
    cb_res = fig.colorbar(residual_im, cax=cax_residual)
    cb_res.set_label(
        "(Prediction − target) / "
        f"(|target| + {relative_error_eps:g} × target RMS)",
        fontsize=9,
        labelpad=8,
    )
    cb_res.ax.tick_params(labelsize=8, length=2.5)

    fig.text(0.545, 0.032, "z [cm]", ha="center", va="center", fontsize=10)
    fig.text(0.055, 0.50, "x [cm]", ha="center", va="center", rotation=90, fontsize=10)
    fig.suptitle(
        f"Density, Ay contours and in-plane magnetic field ({plot_units} units)\n"
        + _figure_context(metadata, local_time),
        fontsize=12,
        y=0.985,
    )
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


@torch.no_grad()
def main():
    args = parse_args()

    run_dir = expand_path(args.run_dir)
    out_dir = (
        expand_path(args.out_dir)
        if args.out_dir is not None
        else run_dir / "figures_mask_patterns"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print("Device:", device)

    ckpt, ckpt_path = load_checkpoint(run_dir, args.checkpoint, device=device)
    ckpt_args = ckpt["args"]
    stats = ckpt["stats"]

    h5_dir = args.h5_dir if args.h5_dir is not None else ckpt_args["h5_dir"]
    h5_dir = expand_path(h5_dir)

    betas = ckpt_args.get("betas", [0.2])
    delta_t = int(ckpt_args.get("delta_t", ckpt_args.get("delta-t", 8)))
    stride_t = int(ckpt_args.get("stride_t", ckpt_args.get("stride-t", 2)))
    base_channels = int(ckpt_args.get("base_channels", ckpt_args.get("base-channels", 16)))
    # Checkpoints created before the depth option used the original three-level
    # architecture. Keep that fallback so their figures remain reproducible.
    channel_mults = ckpt_args.get("channel_mults", [1, 2, 4])
    if args.density_forecast_visible_frames is None:
        args.density_forecast_visible_frames = default_density_forecast_visible_frames(
            delta_t
        )

    if not (0 <= args.local_time < delta_t):
        raise ValueError(f"--local-time must be in [0, {delta_t - 1}], got {args.local_time}")
    if args.all_times and args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")
    if args.relative_error_eps <= 0:
        raise ValueError(
            "--relative-error-eps must be positive, got "
            f"{args.relative_error_eps}"
        )

    print("HDF5 dir:", h5_dir)
    print("Betas:", betas)
    print("delta_t:", delta_t)
    print("stride_t:", stride_t)
    print("base_channels:", base_channels)
    print("channel_mults:", channel_mults)
    print("mask_fraction:", args.mask_fraction)
    print("block_fraction:", args.block_fraction)
    print("density_grid_stride:", args.grid_stride)
    print("magnetic_grid_stride:", args.magnetic_grid_stride)
    print("mask_patterns:", args.mask_patterns)
    print("experiment:", args.experiment)
    if args.sample_index is None:
        print("sample selector: run_name/t0", args.run_name, args.t0)
    else:
        print("sample selector: legacy validation sample_index", args.sample_index)
    print("density_probe_counts:", args.density_probe_counts)
    print("fixed_density_visible_fraction:", args.fixed_density_visible_fraction)
    print("magnetic_visible_fractions:", args.magnetic_visible_fractions)
    print(
        "density_forecast_visible_frames:",
        args.density_forecast_visible_frames,
    )
    print("local_time:", args.local_time)
    print("all_times:", args.all_times)
    if args.all_times:
        print("animation_format:", args.animation_format)
        print("fps:", args.fps)
    print("plot_units:", args.plot_units)
    print("relative_error_eps:", args.relative_error_eps)

    dataset = VPICWindowDataset(
        h5_dir=h5_dir,
        betas=betas,
        delta_t=delta_t,
        stride_t=stride_t,
        layout="C T X Z",
        return_metadata=True,
    )

    val_runs = get_val_runs(run_dir)
    if args.sample_index is None:
        dataset_idx = select_run_t0_index(
            dataset=dataset,
            val_runs=val_runs,
            run_name=args.run_name,
            t0=args.t0,
        )
        selection_stem = "named"
    else:
        dataset_idx = select_sample_index(dataset, val_runs, args.sample_index)
        selection_stem = f"sample{args.sample_index:04d}"

    sample = dataset[dataset_idx]
    y = sample["block"].unsqueeze(0).to(device)  # (1, C, T, X, Z)
    metadata = sample["metadata"]

    print("Selected dataset index:", dataset_idx)
    print("Sample metadata:", metadata)
    print("Block shape:", tuple(y.shape))

    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)

    y_norm = normalize(y, mean, std)

    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=base_channels,
        channel_mults=channel_mults,
        architecture=ckpt_args.get("model_version", LEGACY_MODEL_VERSION),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    if args.plot_units == "normalized":
        y_plot_np = y_norm[0].detach().cpu().numpy()
    elif args.plot_units == "physical":
        y_plot_np = y[0].detach().cpu().numpy()
    else:
        raise ValueError(f"Unknown plot_units: {args.plot_units}")

    # Derived magnetic quantities are computed only from complete fields.
    # In particular, no gradients or path integrations are applied to the
    # masked Visible input arrays.
    target_ay, target_jy = compute_ay_jy(y_plot_np, args.extent)
    experiment_mask_groups = build_experiment_mask_rows(
        args=args,
        block=y_norm,
        generator=generator,
    )
    experiment_rows = []
    for experiment_name, mask_rows in experiment_mask_groups:
        rows_for_plot = []
        print(f"Running experiment: {experiment_name} ({len(mask_rows)} rows)")
        for short_name, label, mask in mask_rows:
            x_visible_norm = make_visible_input(y_norm, mask)
            model_input = torch.cat([x_visible_norm, mask], dim=1)
            pred_norm = model(model_input)

            if args.plot_units == "normalized":
                pred_plot_tensor = pred_norm.detach()
                visible_plot_tensor = y_norm.detach().clone()
            else:
                pred_plot_tensor = denormalize(pred_norm, mean, std).detach()
                visible_plot_tensor = y.detach().clone()
            visible_plot_tensor[mask < 0.5] = float("nan")

            row = {
                "name": short_name,
                "label": label,
                "mask": mask[0].detach().cpu().numpy(),
                "visible_plot": visible_plot_tensor[0].detach().cpu().numpy(),
                "pred_plot": pred_plot_tensor[0].detach().cpu().numpy(),
            }
            row["pred_ay"], row["pred_jy"] = compute_ay_jy(
                row["pred_plot"], args.extent
            )
            rows_for_plot.append(row)
        experiment_rows.append((experiment_name, rows_for_plot))

    sample_stem = (
        f"{selection_stem}_"
        f"{metadata['run_name']}_"
        f"t0-{metadata['t0']}_"
        f"{args.plot_units}"
    )
    for experiment_name, rows_for_plot in experiment_rows:
        experiment_images_dir = images_dir / experiment_name
        experiment_images_dir.mkdir(parents=True, exist_ok=True)
        if args.all_times:
            local_times = list(range(delta_t))
            limit_times = local_times
        elif experiment_name == "density_forecast":
            # A static forecast table is meaningful at the requested endpoint,
            # not at the generic --local-time used by reconstruction tables.
            local_times = [delta_t - 1]
            limit_times = None
            print(
                "density_forecast static table uses final local time:",
                delta_t - 1,
            )
        else:
            local_times = [args.local_time]
            limit_times = None

        experiment_stem = f"{sample_stem}_experiment-{experiment_name}"
        combined_frame_paths = []
        for local_time in local_times:
            frame_stem = (
                f"{experiment_stem}_"
                f"localt-{local_time:03d}_"
                f"globalt-{int(metadata['t0']) + local_time:04d}"
            )
            magnetic_path = experiment_images_dir / f"{frame_stem}_magnetic_table.png"
            density_path = experiment_images_dir / f"{frame_stem}_density_table.png"
            combined_path = experiment_images_dir / f"{frame_stem}_combined_1x2.png"

            plot_jy_ay_by_mask_patterns(
                target_ay=target_ay,
                target_jy=target_jy,
                target_field=y_plot_np,
                rows=rows_for_plot,
                metadata=metadata,
                local_time=local_time,
                out_path=magnetic_path,
                extent=args.extent,
                plot_units=args.plot_units,
                ay_levels=args.ay_levels,
                field_q=args.field_q,
                residual_q=args.residual_q,
                residual_vmax=args.residual_vmax,
                relative_error_eps=args.relative_error_eps,
                quiver_step=args.quiver_step,
                quiver_scale=args.quiver_scale,
                dpi=args.dpi,
                limit_times=limit_times,
            )

            plot_density_magnetic_field_by_mask_patterns(
                target_field=y_plot_np,
                target_ay=target_ay,
                rows=rows_for_plot,
                metadata=metadata,
                local_time=local_time,
                out_path=density_path,
                extent=args.extent,
                plot_units=args.plot_units,
                ay_levels=args.ay_levels,
                quiver_step=args.quiver_step,
                quiver_scale=args.quiver_scale,
                field_q=args.field_q,
                residual_q=args.residual_q,
                residual_vmax=args.residual_vmax,
                relative_error_eps=args.relative_error_eps,
                dpi=args.dpi,
                limit_times=limit_times,
            )
            combine_table_images(
                magnetic_path=magnetic_path,
                density_path=density_path,
                out_path=combined_path,
            )
            combined_frame_paths.append(combined_path)

        if args.all_times:
            animation_stem = (
                f"{experiment_stem}_"
                f"globalt-{int(metadata['t0']):04d}-"
                f"{int(metadata['t0']) + delta_t - 1:04d}_combined_1x2"
            )
            formats = (
                ["quicktime", "gif"]
                if args.animation_format == "both"
                else [args.animation_format]
            )
            for animation_format in formats:
                extension = (
                    "mov" if animation_format == "quicktime" else animation_format
                )
                write_animation(
                    combined_frame_paths,
                    out_dir / f"{animation_stem}.{extension}",
                    fps=args.fps,
                )


if __name__ == "__main__":
    main()
