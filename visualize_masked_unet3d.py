# visualize_masked_unet3d.py

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence

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
        "--time-indices",
        type=int,
        nargs="+",
        default=[0, 4, 7],
        help="Time indices inside the temporal block to plot.",
    )

    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for figures. If None, use run-dir/figures.",
    )

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dpi", type=int, default=180)

    # The VPIC cropped physical region is roughly z in [-21, 21] cm, x in [-50, 50] cm.
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

    # Density is nonnegative-ish, use asymmetric scale.
    if channel == 3:
        vmin = np.nanpercentile(vals, 1.0)
        vmax = np.nanpercentile(vals, q)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        return float(vmin), float(vmax)

    # B fields are signed, use symmetric scale.
    vmax = np.nanpercentile(np.abs(vals), q)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return float(-vmax), float(vmax)


def make_nan_cmap(name: str, bad_color: str = "lightgray"):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color=bad_color)
    return cmap


def plot_channel(
    y_phys: np.ndarray,
    visible_phys: np.ndarray,
    pred_phys: np.ndarray,
    mask_np: np.ndarray,
    metadata: Dict,
    channel: int,
    time_indices: Sequence[int],
    out_path: Path,
    extent,
    dpi: int = 180,
):
    """
    Arrays:
        y_phys:       (C, T, X, Z)
        visible_phys: (C, T, X, Z), hidden region = nan
        pred_phys:    (C, T, X, Z)
        mask_np:      (C, T, X, Z), 1 visible, 0 hidden
    """
    channel_name = CHANNEL_NAMES[channel]

    n_rows = len(time_indices)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.3 * n_cols, 3.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )

    field_cmap = "viridis" if channel == 3 else "PuOr"
    residual_cmap = "PRGn"

    field_cmap = make_nan_cmap(field_cmap)
    residual_cmap = make_nan_cmap(residual_cmap)

    for r, t in enumerate(time_indices):
        target = y_phys[channel, t]
        visible = visible_phys[channel, t]
        pred = pred_phys[channel, t]

        residual = pred - target
        residual_hidden = residual.copy()
        residual_hidden[mask_np[channel, t] > 0.5] = np.nan

        field_vmin, field_vmax = robust_limits([target, pred], channel=channel)
        res_vmin, res_vmax = robust_limits([residual_hidden], channel=0)

        panels = [
            ("Target", target, field_cmap, field_vmin, field_vmax),
            ("Visible input", visible, field_cmap, field_vmin, field_vmax),
            ("Prediction", pred, field_cmap, field_vmin, field_vmax),
            ("Residual, hidden only", residual_hidden, residual_cmap, res_vmin, res_vmax),
        ]

        for c, (title, image, cmap, vmin, vmax) in enumerate(panels):
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
            ax.set_title(f"{title}\n{channel_name}, local t={t}")
            ax.set_xlabel("z [cm]")
            ax.set_ylabel("x [cm]")
            fig.colorbar(im, ax=ax, shrink=0.85)

    run_name = metadata.get("run_name", "unknown")
    t0 = metadata.get("t0", "unknown")
    beta = metadata.get("beta", "unknown")
    nu = metadata.get("nu", "unknown")
    Bz0 = metadata.get("Bz0", "unknown")
    tau = metadata.get("tau", "unknown")

    fig.suptitle(
        f"{channel_name} masked reconstruction\n"
        f"{run_name}, t0={t0}, beta={beta}, nu={nu}, Bz0={Bz0}, tau={tau}",
        fontsize=14,
    )

    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


@torch.no_grad()
def main():
    args = parse_args()

    run_dir = expand_path(args.run_dir)
    out_dir = expand_path(args.out_dir) if args.out_dir is not None else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
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

    print("HDF5 dir:", h5_dir)
    print("Betas:", betas)
    print("delta_t:", delta_t)
    print("stride_t:", stride_t)
    print("base_channels:", base_channels)
    print("keep_prob:", keep_prob)

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

    mask = random_voxel_mask(y_norm, keep_prob=keep_prob)
    x_visible_norm = make_visible_input(y_norm, mask)

    model_input = torch.cat([x_visible_norm, mask], dim=1)

    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    pred_norm = model(model_input)

    y_phys = y.detach()
    pred_phys = denormalize(pred_norm, mean, std).detach()

    # For visible input figure: show true physical values only where visible, hidden = NaN.
    visible_phys = y_phys.clone()
    visible_phys[mask < 0.5] = float("nan")

    y_np = y_phys[0].cpu().numpy()
    pred_np = pred_phys[0].cpu().numpy()
    visible_np = visible_phys[0].cpu().numpy()
    mask_np = mask[0].cpu().numpy()

    for ch in args.channels:
        if ch < 0 or ch >= 4:
            raise ValueError(f"Invalid channel {ch}. Use 0=Bx, 1=By, 2=Bz, 3=Density.")

        valid_times = []
        for t in args.time_indices:
            if 0 <= t < delta_t:
                valid_times.append(t)
            else:
                print(f"Skipping invalid time index {t}; delta_t={delta_t}")

        out_path = out_dir / (
            f"sample{args.sample_index:04d}_"
            f"{metadata['run_name']}_"
            f"t0-{metadata['t0']}_"
            f"{CHANNEL_NAMES[ch]}.png"
        )

        plot_channel(
            y_phys=y_np,
            visible_phys=visible_np,
            pred_phys=pred_np,
            mask_np=mask_np,
            metadata=metadata,
            channel=ch,
            time_indices=valid_times,
            out_path=out_path,
            extent=args.extent,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
