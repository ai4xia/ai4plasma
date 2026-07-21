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
import numpy as np
import torch

from data.vpic_hdf5_dataset import VPICWindowDataset
from data.masking import random_voxel_mask, make_visible_input
from models.unet3d import UNet3D


CHANNEL_NAMES = ["Bx", "By", "Bz", "Density"]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--run-dir",
        type=str,
        default="runs/masked-unet3d_beta0p2_dt8_kp0.2_d53b9bb1",
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
    p.add_argument("--keep-prob", type=float, default=None)

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
        default=["random", "superres", "inpainting", "extrapolation", "probe"],
        choices=["random", "superres", "inpainting", "interpolation", "extrapolation", "probe"],
        help="Rows to show in the figure.",
    )

    p.add_argument("--superres-factor", type=int, default=4)
    p.add_argument("--interp-keep-every", type=int, default=2)
    p.add_argument("--extrap-context", type=int, default=None)

    p.add_argument(
        "--inpaint-x-frac",
        type=float,
        default=0.45,
        help="Fraction of x dimension to hide for central inpainting mask.",
    )
    p.add_argument(
        "--inpaint-z-frac",
        type=float,
        default=0.45,
        help="Fraction of z dimension to hide for central inpainting mask.",
    )

    p.add_argument("--probe-nx", type=int, default=8)
    p.add_argument("--probe-nz", type=int, default=8)

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


def robust_limits(arrays: Sequence[np.ndarray], channel: int, q: float = 99.0):
    vals = []
    for a in arrays:
        aa = np.asarray(a)
        aa = aa[np.isfinite(aa)]
        if aa.size > 0:
            vals.append(aa)

    if len(vals) == 0:
        return -1.0, 1.0

    vals = np.concatenate(vals)

    if channel == 3:
        vmin = np.nanpercentile(vals, 1.0)
        vmax = np.nanpercentile(vals, q)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    vmax = np.nanpercentile(np.abs(vals), q)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return float(-vmax), float(vmax)


def random_visible_mask(block: torch.Tensor, keep_prob: float) -> torch.Tensor:
    return random_voxel_mask(block, keep_prob=keep_prob)


def superres_visible_mask(block: torch.Tensor, factor: int = 4) -> torch.Tensor:
    """
    Keep every factor-th spatial grid point.
    block shape: (B, C, T, X, Z)
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")

    mask = torch.zeros_like(block)
    mask[:, :, :, ::factor, ::factor] = 1.0
    return mask


def temporal_interpolation_visible_mask(block: torch.Tensor, keep_every: int = 2) -> torch.Tensor:
    """
    Keep every keep_every-th time frame.
    """
    if keep_every < 1:
        raise ValueError(f"keep_every must be >= 1, got {keep_every}")

    mask = torch.zeros_like(block)
    mask[:, :, ::keep_every, :, :] = 1.0
    return mask


def temporal_extrapolation_visible_mask(block: torch.Tensor, num_context_frames: int | None = None) -> torch.Tensor:
    """
    Keep first num_context_frames and hide future frames.
    """
    B, C, T, X, Z = block.shape

    if num_context_frames is None:
        num_context_frames = max(1, T // 2)

    if not (1 <= num_context_frames < T):
        raise ValueError(
            f"num_context_frames must be in [1, T-1], got {num_context_frames}, T={T}"
        )

    mask = torch.zeros_like(block)
    mask[:, :, :num_context_frames, :, :] = 1.0
    return mask


def inpainting_visible_mask(
    block: torch.Tensor,
    x_frac: float = 0.45,
    z_frac: float = 0.45,
) -> torch.Tensor:
    """
    Hide a central spatial rectangle, keep everything else.
    """
    B, C, T, X, Z = block.shape

    if not (0.0 < x_frac < 1.0):
        raise ValueError(f"x_frac must be in (0, 1), got {x_frac}")
    if not (0.0 < z_frac < 1.0):
        raise ValueError(f"z_frac must be in (0, 1), got {z_frac}")

    mask = torch.ones_like(block)

    hx = max(1, int(round(X * x_frac)))
    hz = max(1, int(round(Z * z_frac)))

    x0 = (X - hx) // 2
    z0 = (Z - hz) // 2

    mask[:, :, :, x0 : x0 + hx, z0 : z0 + hz] = 0.0
    return mask


def probe_like_visible_mask(
    block: torch.Tensor,
    probe_nx: int = 8,
    probe_nz: int = 8,
) -> torch.Tensor:
    """
    Keep a sparse fixed grid of probe points for all channels and all times.

    This is not meant to be the final experimental geometry;
    it is a simple probe-like structured sparse mask.
    """
    B, C, T, X, Z = block.shape

    probe_nx = max(1, min(probe_nx, X))
    probe_nz = max(1, min(probe_nz, Z))

    xs = torch.linspace(0, X - 1, probe_nx, device=block.device).round().long()
    zs = torch.linspace(0, Z - 1, probe_nz, device=block.device).round().long()

    mask = torch.zeros_like(block)
    for ix in xs:
        for iz in zs:
            mask[:, :, :, ix, iz] = 1.0

    return mask


def build_mask_patterns(
    block: torch.Tensor,
    patterns: Sequence[str],
    keep_prob: float,
    superres_factor: int,
    interp_keep_every: int,
    extrap_context: int | None,
    inpaint_x_frac: float,
    inpaint_z_frac: float,
    probe_nx: int,
    probe_nz: int,
) -> List[Tuple[str, str, torch.Tensor]]:
    """
    Return list of:
        (short_name, display_label, visible_mask)
    """
    rows = []

    for name in patterns:
        if name == "random":
            mask = random_visible_mask(block, keep_prob=keep_prob)
            label = f"Random voxel\nkeep={keep_prob:.2f}"

        elif name == "superres":
            mask = superres_visible_mask(block, factor=superres_factor)
            label = f"Super-resolution\nspatial stride={superres_factor}"

        elif name == "inpainting":
            mask = inpainting_visible_mask(
                block,
                x_frac=inpaint_x_frac,
                z_frac=inpaint_z_frac,
            )
            label = f"Spatial inpainting\ncentral hole"

        elif name == "interpolation":
            mask = temporal_interpolation_visible_mask(block, keep_every=interp_keep_every)
            label = f"Temporal interpolation\nkeep every {interp_keep_every}"

        elif name == "extrapolation":
            mask = temporal_extrapolation_visible_mask(block, num_context_frames=extrap_context)
            context = extrap_context if extrap_context is not None else block.shape[2] // 2
            label = f"Temporal extrapolation\ncontext={context} frames"

        elif name == "probe":
            mask = probe_like_visible_mask(block, probe_nx=probe_nx, probe_nz=probe_nz)
            label = f"Probe-like sparse\n{probe_nx}×{probe_nz} points"

        else:
            raise ValueError(f"Unknown mask pattern: {name}")

        rows.append((name, label, mask))

    return rows


def compute_hidden_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float]:
    """
    pred, target, mask are 2D arrays for one channel and one time.
    mask: 1 visible, 0 hidden.
    Return physical-unit RMSE and MAE over hidden region.
    """
    hidden = mask < 0.5
    if hidden.sum() == 0:
        return float("nan"), float("nan")

    err = pred[hidden] - target[hidden]
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return rmse, mae


def plot_channel_by_mask_patterns(
    y_phys: np.ndarray,
    rows: List[Dict],
    metadata: Dict,
    channel: int,
    local_time: int,
    out_path: Path,
    extent,
    dpi: int = 180,
):
    """
    y_phys: (C, T, X, Z)

    rows is list of dicts with:
        name, label, pred_phys, visible_phys, mask
    each array has shape (C, T, X, Z)
    """
    channel_name = CHANNEL_NAMES[channel]

    n_rows = len(rows)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols, 3.2 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    field_cmap = make_nan_cmap("viridis" if channel == 3 else "PuOr")
    residual_cmap = make_nan_cmap("PRGn")

    target = y_phys[channel, local_time]

    all_field_arrays = [target]
    all_residual_arrays = []

    for row in rows:
        pred = row["pred_phys"][channel, local_time]
        mask = row["mask"][channel, local_time]
        residual_hidden = pred - target
        residual_hidden = residual_hidden.copy()
        residual_hidden[mask > 0.5] = np.nan

        all_field_arrays.append(pred)
        all_residual_arrays.append(residual_hidden)

    field_vmin, field_vmax = robust_limits(all_field_arrays, channel=channel)
    res_vmin, res_vmax = robust_limits(all_residual_arrays, channel=0)

    col_titles = ["Target", "Visible input", "Prediction", "Residual, hidden only"]

    for r, row in enumerate(rows):
        mask = row["mask"][channel, local_time]
        visible = row["visible_phys"][channel, local_time]
        pred = row["pred_phys"][channel, local_time]

        residual_hidden = pred - target
        residual_hidden = residual_hidden.copy()
        residual_hidden[mask > 0.5] = np.nan

        rmse, mae = compute_hidden_metrics(pred, target, mask)
        visible_pct = 100.0 * float(np.nanmean(mask))

        row_label = (
            f"{row['label']}\n"
            f"visible={visible_pct:.2f}%\n"
            f"RMSE={rmse:.3g}, MAE={mae:.3g}"
        )

        panels = [
            (target, field_cmap, field_vmin, field_vmax),
            (visible, field_cmap, field_vmin, field_vmax),
            (pred, field_cmap, field_vmin, field_vmax),
            (residual_hidden, residual_cmap, res_vmin, res_vmax),
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

            if r == 0:
                ax.set_title(col_titles[c], fontsize=12)

            ax.set_xlabel("z [cm]")

            if c == 0:
                ax.set_ylabel(row_label + "\n\nx [cm]", fontsize=10)
            else:
                ax.set_ylabel("x [cm]")

            fig.colorbar(im, ax=ax, shrink=0.82)

    run_name = metadata.get("run_name", "unknown")
    t0 = metadata.get("t0", "unknown")
    beta = metadata.get("beta", "unknown")
    nu = metadata.get("nu", "unknown")
    Bz0 = metadata.get("Bz0", "unknown")
    tau = metadata.get("tau", "unknown")

    global_frame = int(t0) + int(local_time)

    fig.suptitle(
        f"{channel_name} reconstruction under different mask patterns\n"
        f"{run_name}, block t0={t0}, local t={local_time}, global frame={global_frame}, "
        f"beta={beta}, nu={nu}, Bz0={Bz0}, tau={tau}",
        fontsize=14,
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
    keep_prob = float(args.keep_prob if args.keep_prob is not None else ckpt_args.get("keep_prob", 0.2))

    if not (0 <= args.local_time < delta_t):
        raise ValueError(f"--local-time must be in [0, {delta_t - 1}], got {args.local_time}")

    print("HDF5 dir:", h5_dir)
    print("Betas:", betas)
    print("delta_t:", delta_t)
    print("stride_t:", stride_t)
    print("base_channels:", base_channels)
    print("keep_prob:", keep_prob)
    print("mask_patterns:", args.mask_patterns)
    print("local_time:", args.local_time)

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

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    mask_rows = build_mask_patterns(
        block=y_norm,
        patterns=args.mask_patterns,
        keep_prob=keep_prob,
        superres_factor=args.superres_factor,
        interp_keep_every=args.interp_keep_every,
        extrap_context=args.extrap_context,
        inpaint_x_frac=args.inpaint_x_frac,
        inpaint_z_frac=args.inpaint_z_frac,
        probe_nx=args.probe_nx,
        probe_nz=args.probe_nz,
    )

    rows_for_plot = []

    for short_name, label, mask in mask_rows:
        x_visible_norm = make_visible_input(y_norm, mask)
        model_input = torch.cat([x_visible_norm, mask], dim=1)

        pred_norm = model(model_input)
        pred_phys = denormalize(pred_norm, mean, std)

        visible_phys = y.detach().clone()
        visible_phys[mask < 0.5] = float("nan")

        rows_for_plot.append(
            {
                "name": short_name,
                "label": label,
                "mask": mask[0].detach().cpu().numpy(),
                "visible_phys": visible_phys[0].detach().cpu().numpy(),
                "pred_phys": pred_phys[0].detach().cpu().numpy(),
            }
        )

    y_np = y[0].detach().cpu().numpy()

    for ch in args.channels:
        if ch < 0 or ch >= 4:
            raise ValueError(f"Invalid channel {ch}. Use 0=Bx, 1=By, 2=Bz, 3=Density.")

        pattern_tag = "-".join(args.mask_patterns)
        out_path = out_dir / (
            f"sample{args.sample_index:04d}_"
            f"{metadata['run_name']}_"
            f"t0-{metadata['t0']}_"
            f"localt-{args.local_time}_"
            f"{CHANNEL_NAMES[ch]}_masks_{pattern_tag}.png"
        )

        plot_channel_by_mask_patterns(
            y_phys=y_np,
            rows=rows_for_plot,
            metadata=metadata,
            channel=ch,
            local_time=args.local_time,
            out_path=out_path,
            extent=args.extent,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
