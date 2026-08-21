# visualize_mask_patterns_unet3d.py

from __future__ import annotations

import argparse
import json
import os
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


CHANNEL_NAMES = ["Bx", "By", "Bz", "Density"]


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
        help="Fixed spatial_grid stride. The grid offset stays random.",
    )

    p.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[0, 2, 3],
        help="Channels to plot. 0=Bx, 1=By, 2=Bz, 3=Density.",
    )
    p.add_argument(
        "--local-time",
        type=int,
        default=4,
        help="One local time index inside the temporal block. Rows are mask patterns.",
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
        detail = "independent voxels"
    elif pattern == "spatial_grid":
        detail = (
            f"stride={info['stride']}×{info['stride']}\n"
            f"offset=({info['offset_x']}, {info['offset_z']})"
        )
    elif pattern == "spatial_block":
        detail = (
            f"hole={100.0 * info['target_mask_fraction']:.0f}%\n"
            f"{info['rect_height']}×{info['rect_width']} "
            f"at ({info['rect_x0']}, {info['rect_z0']})"
        )
    elif pattern == "temporal_random":
        detail = f"frames={info['visible_frames']}"
    else:
        detail = ""

    return f"{pattern}\n{detail}"


def build_mask_patterns(
    block: torch.Tensor,
    patterns: Sequence[str],
    mask_fraction: float,
    block_fraction: float,
    grid_stride: int,
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
        mask, info = sample_mask(
            block.shape,
            pattern=name,
            mask_fraction=block_fraction if name == "spatial_block" else mask_fraction,
            device=block.device,
            dtype=block.dtype,
            generator=generator,
            grid_stride=grid_stride if name == "spatial_grid" else None,
        )
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


def plot_channel_by_mask_patterns(
    y_plot: np.ndarray,
    rows: List[Dict],
    metadata: Dict,
    channel: int,
    local_time: int,
    out_path: Path,
    extent,
    plot_units: str,
    mask_fraction: float,
    field_q: float = 99.0,
    residual_q: float = 99.0,
    residual_vmax: float | None = None,
    dpi: int = 180,
):
    """
    y_plot: (C, T, X, Z)

    rows is list of dicts with:
        name, label, pred_plot, visible_plot, mask
    each array has shape (C, T, X, Z)
    """
    channel_name = CHANNEL_NAMES[channel]

    n_rows = len(rows)
    n_data_cols = 4

    # Layout:
    #   Target | Visible | Prediction | field cbar | Residual | residual cbar
    # This gives one colorbar for the first three field columns and one for residuals.
    fig = plt.figure(
        figsize=(13.0, 2.15 * n_rows + 1.2),
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

    axes = np.empty((n_rows, n_data_cols), dtype=object)
    base_ax = None

    data_cols_to_grid_cols = [0, 1, 2, 4]
    for r in range(n_rows):
        for c, gcol in enumerate(data_cols_to_grid_cols):
            if base_ax is None:
                ax = fig.add_subplot(gs[r, gcol])
                base_ax = ax
            else:
                ax = fig.add_subplot(gs[r, gcol], sharex=base_ax, sharey=base_ax)
            axes[r, c] = ax

    cax_field = fig.add_subplot(gs[:, 3])
    cax_residual = fig.add_subplot(gs[:, 5])

    field_cmap = make_nan_cmap("viridis" if channel == 3 else "PuOr")
    residual_cmap = make_nan_cmap("PRGn")

    target = y_plot[channel, local_time]

    all_field_arrays = [target]
    all_residual_arrays = []

    for row in rows:
        pred = row["pred_plot"][channel, local_time]
        residual = pred - target

        all_field_arrays.append(pred)
        all_residual_arrays.append(residual)

    field_vmin, field_vmax = robust_limits(
        all_field_arrays,
        channel=channel,
        q=field_q,
        symmetric=None,
    )

    if residual_vmax is not None:
        res_vmin, res_vmax = -float(residual_vmax), float(residual_vmax)
    else:
        res_vmin, res_vmax = robust_limits(
            all_residual_arrays,
            channel=channel,
            q=residual_q,
            symmetric=True,
        )

    col_titles = ["Target", "Visible input", "Prediction", "Residual"]

    field_im = None
    residual_im = None

    for r, row in enumerate(rows):
        mask = row["mask"][channel, local_time]
        visible = row["visible_plot"][channel, local_time]
        pred = row["pred_plot"][channel, local_time]

        residual = pred - target

        rmse, mae = compute_full_metrics(pred, target)
        visible_pct = 100.0 * float(np.nanmean(mask))

        row_label = (
            f"{row['label']}\n"
            f"visible={visible_pct:.2f}%\n"
            f"RMSE={rmse:.3g}\n"
            f"MAE={mae:.3g}"
        )

        panels = [
            (target, field_cmap, field_vmin, field_vmax),
            (visible, field_cmap, field_vmin, field_vmax),
            (pred, field_cmap, field_vmin, field_vmax),
            (residual, residual_cmap, res_vmin, res_vmax),
        ]

        for c, (image, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[r, c]

            im = ax.imshow(
                image,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )

            if c < 3:
                field_im = im
            else:
                residual_im = im

            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, pad=4)

            # Only show tick labels on outer axes.
            if r < n_rows - 1:
                ax.tick_params(labelbottom=False)
            if c > 0:
                ax.tick_params(labelleft=False)

            ax.tick_params(axis="both", which="both", labelsize=8, length=2.5)

            # Row label on the far left, not repeated axis units.
            if c == 0:
                ax.text(
                    -0.33,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    rotation=90,
                    fontsize=8,
                    linespacing=1.05,
                )

    # Shared colorbars.
    if field_im is not None:
        cb_field = fig.colorbar(field_im, cax=cax_field)
        cb_field.ax.tick_params(labelsize=8, length=2.5)
        cb_field.set_label(
            f"{channel_name} ({plot_units})",
            fontsize=9,
            rotation=90,
            labelpad=8,
        )

    if residual_im is not None:
        cb_res = fig.colorbar(residual_im, cax=cax_residual)
        cb_res.ax.tick_params(labelsize=8, length=2.5)
        cb_res.set_label(
            f"Prediction − target ({plot_units})",
            fontsize=9,
            rotation=90,
            labelpad=8,
        )

    # Shared axis labels.
    fig.text(0.545, 0.032, "z [cm]", ha="center", va="center", fontsize=10)
    fig.text(0.055, 0.50, "x [cm]", ha="center", va="center", rotation=90, fontsize=10)

    run_name = metadata.get("run_name", "unknown")
    t0 = metadata.get("t0", "unknown")
    beta = metadata.get("beta", "unknown")
    nu = metadata.get("nu", "unknown")
    Bz0 = metadata.get("Bz0", "unknown")
    tau = metadata.get("tau", "unknown")

    global_frame = int(t0) + int(local_time)

    fig.suptitle(
        f"{channel_name} reconstruction under different mask patterns "
        f"({plot_units} units, p={mask_fraction:g} for the spatial_random and temporal_random rows)\n"
        f"{run_name}, block t0={t0}, local t={local_time}, global frame={global_frame}, "
        f"beta={beta}, nu={nu}, Bz0={Bz0}, tau={tau}",
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

    print("HDF5 dir:", h5_dir)
    print("Betas:", betas)
    print("delta_t:", delta_t)
    print("stride_t:", stride_t)
    print("base_channels:", base_channels)
    print("mask_fraction:", args.mask_fraction)
    print("block_fraction:", args.block_fraction)
    print("grid_stride:", args.grid_stride)
    print("mask_patterns:", args.mask_patterns)
    print("local_time:", args.local_time)
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

    for ch in args.channels:
        if ch < 0 or ch >= 4:
            raise ValueError(f"Invalid channel {ch}. Use 0=Bx, 1=By, 2=Bz, 3=Density.")

        pattern_tag = "-".join(args.mask_patterns)
        out_path = out_dir / (
            f"sample{args.sample_index:04d}_"
            f"{metadata['run_name']}_"
            f"t0-{metadata['t0']}_"
            f"localt-{args.local_time}_"
            f"{CHANNEL_NAMES[ch]}_"
            f"{args.plot_units}_"
            f"mf{args.mask_fraction:g}_"
            f"masks_{pattern_tag}_compact.png"
        )

        plot_channel_by_mask_patterns(
            y_plot=y_plot_np,
            rows=rows_for_plot,
            metadata=metadata,
            channel=ch,
            local_time=args.local_time,
            out_path=out_path,
            extent=args.extent,
            plot_units=args.plot_units,
            mask_fraction=args.mask_fraction,
            field_q=args.field_q,
            residual_q=args.residual_q,
            residual_vmax=args.residual_vmax,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()