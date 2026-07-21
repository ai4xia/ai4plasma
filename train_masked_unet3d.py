# train_masked_unet3d.py

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data.vpic_hdf5_dataset import VPICWindowDataset
from data.masking import (
    random_voxel_mask,
    make_visible_input,
    full_mse_loss,
    full_mae_loss,
)
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

    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. By default it is generated from the training parameters.",
    )
    p.add_argument("--amp", action="store_true")

    wandb_group = p.add_mutually_exclusive_group()
    wandb_group.add_argument(
        "--wandb",
        dest="wandb",
        action="store_true",
        help="Enable W&B tracking (default).",
    )
    wandb_group.add_argument(
        "--no-wandb",
        dest="wandb",
        action="store_false",
        help="Disable W&B tracking.",
    )
    p.set_defaults(wandb=True)
    p.add_argument("--wandb-project", type=str, default="ai4plasma")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B operating mode. Use offline on compute nodes without network access.",
    )
    p.add_argument(
        "--wandb-log-model",
        action="store_true",
        help="Upload the best checkpoint as a W&B artifact at the end of training.",
    )
    resume_group = p.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        help="Resume from out-dir/latest.pt when parameters match (default).",
    )
    resume_group.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help="Start a new model even when out-dir/latest.pt exists.",
    )
    p.set_defaults(auto_resume=True)

    return p.parse_args()


def expand_path(path: str) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_run_name(args) -> str:
    beta_text = "-".join(str(beta).replace(".", "p") for beta in args.betas)
    signature_data = {
        key: getattr(args, key)
        for key in RESUME_PARAMETER_KEYS
    }
    signature = hashlib.sha1(
        json.dumps(signature_data, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return (
        f"masked-unet3d_beta{beta_text}_dt{args.delta_t}_"
        f"kp{args.keep_prob:g}_{signature}"
    )


def init_wandb(
    args,
    out_dir: Path,
    resume_run_id: Optional[str] = None,
) -> Optional[Any]:
    """Initialize W&B lazily so it remains an optional dependency."""
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking was requested, but wandb is not installed. "
            "Install it with `pip install wandb`, or pass --no-wandb."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name or default_run_name(args),
        group=args.wandb_group,
        tags=args.wandb_tags,
        mode=args.wandb_mode,
        dir=str(out_dir),
        config=vars(args),
        id=resume_run_id,
        resume="allow" if resume_run_id else None,
    )


RESUME_PARAMETER_KEYS = (
    "h5_dir",
    "betas",
    "delta_t",
    "stride_t",
    "batch_size",
    "lr",
    "weight_decay",
    "keep_prob",
    "base_channels",
    "val_frac",
    "seed",
    "stats_batches",
    "grad_clip",
    "amp",
)


def validate_resume_parameters(args, checkpoint_args: Dict[str, Any]) -> None:
    mismatches = []
    for key in RESUME_PARAMETER_KEYS:
        old_value = checkpoint_args.get(key)
        new_value = getattr(args, key)
        if old_value != new_value:
            mismatches.append(f"{key}: checkpoint={old_value!r}, current={new_value!r}")

    if mismatches:
        details = "\n  ".join(mismatches)
        raise ValueError(
            "Refusing to auto-resume because training parameters changed:\n"
            f"  {details}\n"
            "Use a different --out-dir, restore the original parameters, or pass "
            "--no-auto-resume to intentionally start over."
        )


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
            # loss_mse = masked_mse_loss(pred, y, mask)
            # loss_mae = masked_mae_loss(pred, y, mask)
            # loss = loss_mse
            loss_mse = full_mse_loss(pred, y)
            loss_mae = full_mae_loss(pred, y)
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
            # loss_mse = masked_mse_loss(pred, y, mask)
            # loss_mae = masked_mae_loss(pred, y, mask)
            loss_mse = full_mse_loss(pred, y)
            loss_mae = full_mae_loss(pred, y)

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
    wandb_run_id: Optional[str] = None,
) -> None:
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_mse": best_val_mse,
        "args": vars(args),
        "stats": stats,
        "wandb_run_id": wandb_run_id,
    }
    torch.save(ckpt, path)


def main():
    args = parse_args()
    set_seed(args.seed)

    automatic_out_dir = Path("runs") / default_run_name(args)
    out_dir = expand_path(args.out_dir or str(automatic_out_dir))
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

    resume_checkpoint = None
    latest_path = out_dir / "latest.pt"
    if args.auto_resume and latest_path.exists():
        resume_checkpoint = torch.load(latest_path, map_location=device)
        validate_resume_parameters(args, resume_checkpoint.get("args", {}))
        print(
            f"Auto-resuming from {latest_path} "
            f"(completed epoch {resume_checkpoint['epoch']})"
        )

    if resume_checkpoint is not None and "stats" in resume_checkpoint:
        stats = resume_checkpoint["stats"]
        print("Reusing channel statistics from checkpoint")
    else:
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

    start_epoch = 1
    best_val_mse = float("inf")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best_val_mse = float(resume_checkpoint.get("best_val_mse", float("inf")))

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    if start_epoch > args.epochs:
        print(
            f"Training is already complete at epoch {start_epoch - 1}; "
            f"requested epochs={args.epochs}. Increase --epochs to continue."
        )
        return

    resume_run_id = None
    if resume_checkpoint is not None:
        resume_run_id = resume_checkpoint.get("wandb_run_id")
    wandb_run = init_wandb(args, out_dir, resume_run_id=resume_run_id)
    if wandb_run is not None:
        wandb_run.config.update(
            {
                "device": str(device),
                "model_parameters": n_params,
                "num_total_windows": len(dataset),
                "num_train_windows": len(train_idx),
                "num_val_windows": len(val_idx),
                "num_train_runs": len(train_runs),
                "num_val_runs": len(val_runs),
                "channel_mean": stats["mean"],
                "channel_std": stats["std"],
            },
            allow_val_change=True,
        )
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("train/*", step_metric="epoch")
        wandb_run.define_metric("val/*", step_metric="epoch")
        wandb_run.define_metric("optimization/*", step_metric="epoch")

    history = []
    history_path = out_dir / "history.json"
    if resume_checkpoint is not None and history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        history = [row for row in history if row.get("epoch", 0) < start_epoch]

    for epoch in range(start_epoch, args.epochs + 1):
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

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/mse": row["train_mse"],
                    "train/mae": row["train_mae"],
                    "val/mse": row["val_mse"],
                    "val/mae": row["val_mae"],
                    "optimization/learning_rate": optimizer.param_groups[0]["lr"],
                }
            )

        print(
            f"train_mse={row['train_mse']:.6f} "
            f"train_mae={row['train_mae']:.6f} "
            f"val_mse={row['val_mse']:.6f} "
            f"val_mae={row['val_mae']:.6f}"
        )

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

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
                wandb_run_id=wandb_run.id if wandb_run is not None else None,
            )
            print(f"Saved best checkpoint: val_mse={best_val_mse:.6f}")

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_val_mse=best_val_mse,
            args=args,
            stats=stats,
            wandb_run_id=wandb_run.id if wandb_run is not None else None,
        )

    if wandb_run is not None:
        wandb_run.summary["best_val_mse"] = best_val_mse
        if args.wandb_log_model:
            import wandb

            artifact = wandb.Artifact(
                name=f"{wandb_run.id}-best-model",
                type="model",
                metadata={"best_val_mse": best_val_mse},
            )
            artifact.add_file(str(out_dir / "best.pt"))
            wandb_run.log_artifact(artifact)
        wandb_run.finish()


if __name__ == "__main__":
    main()
