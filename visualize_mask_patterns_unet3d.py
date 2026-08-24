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
from models.unet3d import UNet3D


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

    p.add_argument("--sample-index", type=int, default=0)
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
        default="quicktime",
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
        help="Optional fixed symmetric residual color limit. If set, use [-vmax, vmax].",
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


def compute_full_metrics(
    pred: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    """
    pred and target are 2D arrays for one channel and one time.

    Return RMSE and MAE over all pixels, including both visible and hidden regions.
    Units follow the plotted arrays: physical or normalized.
    """
    err = pred - target
    err = err[np.isfinite(err)]

    if err.size == 0:
        return float("nan"), float("nan")

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae


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
            loop=0,
            optimize=False,
            disposal=2,
        )
    else:
        raise ValueError(f"Unsupported animation extension: {out_path.suffix}")

    print(f"Saved animation: {out_path}")


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
    quiver_step: int,
    quiver_scale: float,
    dpi: int,
    limit_times: Sequence[int] | None = None,
) -> None:
    """
    Plot Jy with Ay contours for complete target/prediction fields.

    The Visible input column shows observed target |B| values and in-plane
    (Bz, Bx) arrows on a black missing-data background. Jy and Ay are not
    derived from incomplete observations.
    """
    n_rows = len(rows)
    fig, axes, cax_field, cax_residual = _make_comparison_axes(n_rows)

    target = target_jy[local_time]
    pred_arrays = [row["pred_jy"][local_time] for row in rows]
    residual_arrays = [pred - target for pred in pred_arrays]
    color_times = [local_time] if limit_times is None else list(limit_times)
    all_target_jy = [target_jy[t] for t in color_times]
    all_pred_jy = [row["pred_jy"][t] for row in rows for t in color_times]
    all_residual_jy = [
        row["pred_jy"][t] - target_jy[t]
        for row in rows
        for t in color_times
    ]
    target_b_magnitude = np.sqrt(np.sum(target_field[:3] ** 2, axis=0))
    b_vmin, b_vmax = robust_limits(
        [target_b_magnitude[t] for t in color_times],
        channel=3,
        q=field_q,
        symmetric=False,
    )
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

    jy_cmap = make_nan_cmap("seismic")
    residual_cmap = make_nan_cmap("PRGn")
    magnetic_cmap = make_nan_cmap("viridis", bad_color="black")
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
        visible_b_magnitude = target_b_magnitude[local_time].copy()
        visible_b_magnitude[joint_b_mask < 0.5] = np.nan
        visible_pct = 100.0 * float(np.mean(joint_b_mask))
        rmse, mae = compute_full_metrics(pred, target)
        row_label = (
            f"{row['label']}\n"
            f"joint B visible={visible_pct:.2f}%\n"
            f"Jy RMSE={rmse:.3g}\n"
            f"Jy MAE={mae:.3g}"
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
            visible_b_magnitude,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=magnetic_cmap,
            vmin=b_vmin,
            vmax=b_vmax,
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
                        "Observed target |B| + in-plane B",
                        "Prediction Jy + Ay",
                        "Jy residual",
                    ][c],
                    fontsize=11,
                    pad=4,
                )
            _style_comparison_axis(axes[r, c], r, c, n_rows)

    cb_field = fig.colorbar(field_im, cax=cax_field)
    cb_field.set_label(f"Jy ({plot_units})", fontsize=9, labelpad=8)
    cb_field.ax.tick_params(labelsize=8, length=2.5)
    cb_res = fig.colorbar(residual_im, cax=cax_residual)
    cb_res.set_label(f"Prediction − target Jy ({plot_units})", fontsize=9, labelpad=8)
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
    dpi: int,
    limit_times: Sequence[int] | None = None,
) -> None:
    """Plot Density with Ay contours and in-plane (Bz, Bx) arrows."""
    n_rows = len(rows)
    fig, axes, cax_field, cax_residual = _make_comparison_axes(n_rows)

    target_density = target_field[3, local_time]
    pred_arrays = [row["pred_plot"][3, local_time] for row in rows]
    residual_arrays = [pred - target_density for pred in pred_arrays]
    color_times = [local_time] if limit_times is None else list(limit_times)
    all_target_density = [target_field[3, t] for t in color_times]
    all_pred_density = [
        row["pred_plot"][3, t] for row in rows for t in color_times
    ]
    all_residual_density = [
        row["pred_plot"][3, t] - target_field[3, t]
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
        rmse, mae = compute_full_metrics(pred_density, target_density)
        row_label = (
            f"{row['label']}\n"
            f"Density visible={visible_pct:.2f}%\n"
            f"Density RMSE={rmse:.3g}\n"
            f"Density MAE={mae:.3g}"
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
                        "Density residual",
                    ][c],
                    fontsize=11,
                    pad=4,
                )
            _style_comparison_axis(axes[r, c], r, c, n_rows)

    cb_field = fig.colorbar(field_im, cax=cax_field)
    cb_field.set_label(f"Density ({plot_units})", fontsize=9, labelpad=8)
    cb_field.ax.tick_params(labelsize=8, length=2.5)
    cb_res = fig.colorbar(residual_im, cax=cax_residual)
    cb_res.set_label(f"Prediction − target Density ({plot_units})", fontsize=9, labelpad=8)
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

    if not (0 <= args.local_time < delta_t):
        raise ValueError(f"--local-time must be in [0, {delta_t - 1}], got {args.local_time}")
    if args.all_times and args.fps <= 0:
        raise ValueError(f"--fps must be positive, got {args.fps}")

    print("HDF5 dir:", h5_dir)
    print("Betas:", betas)
    print("delta_t:", delta_t)
    print("stride_t:", stride_t)
    print("base_channels:", base_channels)
    print("mask_fraction:", args.mask_fraction)
    print("block_fraction:", args.block_fraction)
    print("density_grid_stride:", args.grid_stride)
    print("magnetic_grid_stride:", args.magnetic_grid_stride)
    print("mask_patterns:", args.mask_patterns)
    print("local_time:", args.local_time)
    print("all_times:", args.all_times)
    if args.all_times:
        print("animation_format:", args.animation_format)
        print("fps:", args.fps)
    print("plot_units:", args.plot_units)

    dataset = VPICWindowDataset(
        h5_dir=h5_dir,
        betas=betas,
        delta_t=delta_t,
        stride_t=stride_t,
        layout="C T X Z",
        return_metadata=True,
    )

    val_runs = get_val_runs(run_dir)
    dataset_idx = select_sample_index(dataset, val_runs, args.sample_index)

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
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    mask_rows = build_mask_patterns(
        block=y_norm,
        patterns=args.mask_patterns,
        mask_fraction=args.mask_fraction,
        block_fraction=args.block_fraction,
        grid_stride=args.grid_stride,
        magnetic_grid_stride=args.magnetic_grid_stride,
        generator=generator,
    )

    rows_for_plot = []

    for short_name, label, mask in mask_rows:
        x_visible_norm = make_visible_input(y_norm, mask)
        model_input = torch.cat([x_visible_norm, mask], dim=1)

        pred_norm = model(model_input)

        if args.plot_units == "normalized":
            y_plot_tensor = y_norm.detach()
            pred_plot_tensor = pred_norm.detach()

            visible_plot_tensor = y_norm.detach().clone()
            visible_plot_tensor[mask < 0.5] = float("nan")

        elif args.plot_units == "physical":
            y_plot_tensor = y.detach()
            pred_plot_tensor = denormalize(pred_norm, mean, std).detach()

            visible_plot_tensor = y.detach().clone()
            visible_plot_tensor[mask < 0.5] = float("nan")

        else:
            raise ValueError(f"Unknown plot_units: {args.plot_units}")

        rows_for_plot.append(
            {
                "name": short_name,
                "label": label,
                "mask": mask[0].detach().cpu().numpy(),
                "visible_plot": visible_plot_tensor[0].detach().cpu().numpy(),
                "pred_plot": pred_plot_tensor[0].detach().cpu().numpy(),
            }
        )

    if args.plot_units == "normalized":
        y_plot_np = y_norm[0].detach().cpu().numpy()
    else:
        y_plot_np = y[0].detach().cpu().numpy()

    # Derived magnetic quantities are computed only from complete fields.
    # In particular, no gradients or path integrations are applied to the
    # masked Visible input arrays.
    target_ay, target_jy = compute_ay_jy(y_plot_np, args.extent)
    for row in rows_for_plot:
        row["pred_ay"], row["pred_jy"] = compute_ay_jy(
            row["pred_plot"], args.extent
        )

    pattern_tag = "-".join(args.mask_patterns)
    common_stem = (
        f"sample{args.sample_index:04d}_"
        f"{metadata['run_name']}_"
        f"t0-{metadata['t0']}_"
        f"{args.plot_units}_"
        f"mf{args.mask_fraction:g}_"
        f"masks_{pattern_tag}"
    )
    local_times = list(range(delta_t)) if args.all_times else [args.local_time]
    limit_times = local_times if args.all_times else None
    jy_frame_paths = []
    density_frame_paths = []

    for local_time in local_times:
        frame_stem = (
            f"{common_stem}_"
            f"localt-{local_time:03d}_"
            f"globalt-{int(metadata['t0']) + local_time:04d}"
        )
        jy_path = out_dir / f"{frame_stem}_Jy_Ay_compact.png"
        density_path = out_dir / f"{frame_stem}_Density_Ay_B_compact.png"

        plot_jy_ay_by_mask_patterns(
            target_ay=target_ay,
            target_jy=target_jy,
            target_field=y_plot_np,
            rows=rows_for_plot,
            metadata=metadata,
            local_time=local_time,
            out_path=jy_path,
            extent=args.extent,
            plot_units=args.plot_units,
            ay_levels=args.ay_levels,
            field_q=args.field_q,
            residual_q=args.residual_q,
            residual_vmax=args.residual_vmax,
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
            dpi=args.dpi,
            limit_times=limit_times,
        )

        jy_frame_paths.append(jy_path)
        density_frame_paths.append(density_path)

    if args.all_times:
        animation_stem = (
            f"{common_stem}_"
            f"globalt-{int(metadata['t0']):04d}-"
            f"{int(metadata['t0']) + delta_t - 1:04d}"
        )
        formats = (
            ["quicktime", "gif"]
            if args.animation_format == "both"
            else [args.animation_format]
        )

        for animation_format in formats:
            extension = "mov" if animation_format == "quicktime" else animation_format
            write_animation(
                jy_frame_paths,
                out_dir / f"{animation_stem}_Jy_Ay.{extension}",
                fps=args.fps,
            )
            write_animation(
                density_frame_paths,
                out_dir / f"{animation_stem}_Density_Ay_B.{extension}",
                fps=args.fps,
            )


if __name__ == "__main__":
    main()
