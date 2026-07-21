# train_masked_unet3d.py

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.vpic_hdf5_dataset import VPICWindowDataset
from data.masking import random_voxel_mask, make_visible_input, masked_mse_loss, masked_mae_loss
from models.unet3d import UNet3D


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--h5-dir",
        type=str,
        default="$SCRATCH/VPIC_PPPL_HDF5_by_beta_official2500_none_compat",
    )
    p.add_argument("--betas", type=float, nargs="+", default=[0.2])
    p.add_argument("--delta-t", type=int, default=8)
    p.add_argument("--stride-t", type=int, default=2)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--keep-prob", type=float, default=0.2)
    p.add_argument("--base-channels", type=int, default=16)

    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)

    p.add_argument("--stats-batches", type=int, default=50)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--out-dir", type=str, default="runs/masked_unet3d_beta0.2_dt8")
    p.add_argument("--amp", action="store_true")

    return p.parse_args()


def expand_path(path: str) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_by_run(
    dataset: VPICWindowDataset,
    val_frac: float,
    seed: int,
) -> Tuple[List[int], List[int], List[str], List[str]]:
    """
    Split by run_name, not by window, to avoid leakage between train/val.
    """
    run_names = sorted({run_name for _, run_name, _ in dataset.samples})

    rng = random.Random(seed)
    rng.shuffle(run_names)

    n_val = max(1, int(round(len(run_names) * val_frac)))
    val_runs = set(run_names[:n_val])
    train_runs = set(run_names[n_val:])

    train_idx = []
    val_idx = []

    for i, (_, run_name, _) in enumerate(dataset.samples):
        if run_name in val_runs:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return train_idx, val_idx, sorted(train_runs), sorted(val_runs)


@torch.no_grad()
def estimate_channel_stats(
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> Dict[str, List[float]]:
    """
    Estimate per-channel mean/std from training data.

    Input batch["block"]:
        (B, C, T, X, Z)
    """
    sum_c = None
    sumsq_c = None
    count = 0

    for i, batch in enumerate(tqdm(loader, desc="Estimating stats")):
        if i >= max_batches:
            break

        y = batch["block"].to(device, non_blocking=True)  # (B, C, T, X, Z)

        dims = (0, 2, 3, 4)
        batch_sum = y.sum(dim=dims)
        batch_sumsq = (y ** 2).sum(dim=dims)
        batch_count = y.shape[0] * y.shape[2] * y.shape[3] * y.shape[4]

        if sum_c is None:
            sum_c = batch_sum
            sumsq_c = batch_sumsq
        else:
            sum_c += batch_sum
            sumsq_c += batch_sumsq

        count += batch_count

    mean = sum_c / count
    var = sumsq_c / count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    return {
        "mean": mean.detach().cpu().tolist(),
        "std": std.detach().cpu().tolist(),
    }


def normalize(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (y - mean) / (std + 1e-8)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    keep_prob: float,
    amp: bool,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()

    total_mse = 0.0
    total_mae = 0.0
    total_batches = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    pbar = tqdm(loader, desc="Train", leave=False)

    for batch in pbar:
        y = batch["block"].to(device, non_blocking=True)
        y = normalize(y, mean, std)

        mask = random_voxel_mask(y, keep_prob=keep_prob)
        x_visible = make_visible_input(y, mask)

        model_input = torch.cat([x_visible, mask], dim=1)  # (B, 8, T, X, Z)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            pred = model(model_input)
            loss_mse = masked_mse_loss(pred, y, mask)
            loss_mae = masked_mae_loss(pred, y, mask)
            loss = loss_mse

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_mse += float(loss_mse.detach().cpu())
        total_mae += float(loss_mae.detach().cpu())
        total_batches += 1

        pbar.set_postfix(
            mse=total_mse / total_batches,
            mae=total_mae / total_batches,
        )

    return {
        "mse": total_mse / max(total_batches, 1),
        "mae": total_mae / max(total_batches, 1),
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    keep_prob: float,
    amp: bool,
) -> Dict[str, float]:
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_batches = 0

    pbar = tqdm(loader, desc="Val", leave=False)

    for batch in pbar:
        y = batch["block"].to(device, non_blocking=True)
        y = normalize(y, mean, std)

        mask = random_voxel_mask(y, keep_prob=keep_prob)
        x_visible = make_visible_input(y, mask)

        model_input = torch.cat([x_visible, mask], dim=1)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            pred = model(model_input)
            loss_mse = masked_mse_loss(pred, y, mask)
            loss_mae = masked_mae_loss(pred, y, mask)

        total_mse += float(loss_mse.detach().cpu())
        total_mae += float(loss_mae.detach().cpu())
        total_batches += 1

    return {
        "mse": total_mse / max(total_batches, 1),
        "mae": total_mae / max(total_batches, 1),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_mse: float,
    args,
    stats: Dict[str, List[float]],
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_mse": best_val_mse,
        "args": vars(args),
        "stats": stats,
    }
    torch.save(ckpt, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = expand_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_dir = expand_path(args.h5_dir)

    print("HDF5 dir:", h5_dir)
    print("Output dir:", out_dir)
    print("Betas:", args.betas)

    dataset = VPICWindowDataset(
        h5_dir=h5_dir,
        betas=args.betas,
        delta_t=args.delta_t,
        stride_t=args.stride_t,
        layout="C T X Z",
        return_metadata=True,
    )

    train_idx, val_idx, train_runs, val_runs = split_by_run(
        dataset,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    print("Total windows:", len(dataset))
    print("Train windows:", len(train_idx))
    print("Val windows:", len(val_idx))
    print("Train runs:", len(train_runs))
    print("Val runs:", len(val_runs))

    with open(out_dir / "split.json", "w") as f:
        json.dump(
            {
                "train_runs": train_runs,
                "val_runs": val_runs,
                "num_train_windows": len(train_idx),
                "num_val_windows": len(val_idx),
            },
            f,
            indent=2,
        )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    stats = estimate_channel_stats(
        train_loader,
        device=device,
        max_batches=args.stats_batches,
    )

    print("Channel mean:", stats["mean"])
    print("Channel std:", stats["std"])

    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)

    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=args.base_channels,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    best_val_mse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mean=mean,
            std=std,
            keep_prob=args.keep_prob,
            amp=args.amp,
            grad_clip=args.grad_clip,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            mean=mean,
            std=std,
            keep_prob=args.keep_prob,
            amp=args.amp,
        )

        row = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
        }
        history.append(row)

        print(
            f"train_mse={row['train_mse']:.6f} "
            f"train_mae={row['train_mae']:.6f} "
            f"val_mse={row['val_mse']:.6f} "
            f"val_mae={row['val_mae']:.6f}"
        )

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        save_checkpoint(
            out_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_mse=best_val_mse,
            args=args,
            stats=stats,
        )

        if row["val_mse"] < best_val_mse:
            best_val_mse = row["val_mse"]
            save_checkpoint(
                out_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_mse=best_val_mse,
                args=args,
                stats=stats,
            )
            print(f"Saved best checkpoint: val_mse={best_val_mse:.6f}")


if __name__ == "__main__":
    main()
