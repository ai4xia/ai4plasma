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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.vpic_hdf5_dataset import VPICWindowDataset
from data.masking import (
    DEFAULT_PATTERN_WEIGHTS,
    MASKING_VERSION,
    MASK_PATTERNS,
    PATTERN_TO_ID,
    full_mae_loss,
    full_mse_loss,
    make_visible_input,
    parse_pattern_weights,
    sample_batch_masks,
    sample_patterns,
)
from models.unet3d import UNet3D


def distributed_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not distributed_is_initialized() or dist.get_rank() == 0


def rank_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def initialize_distributed() -> Tuple[torch.device, int, int, int]:
    """Initialize torchrun-provided DDP state, or fall back to one process."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Multi-process DDP requires CUDA/NCCL GPUs.")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device, rank, world_size, local_rank


class DistributedEvalSampler(Sampler[int]):
    """Shard validation indices without DistributedSampler's padding duplicates."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--h5-dir",
        type=str,
        default="$SCRATCH/VPIC_PPPL_HDF5_by_beta_official2500_none_compat",
    )
    p.add_argument("--betas", type=float, nargs="+", default=[0.2])
    p.add_argument("--delta-t", type=int, default=24)
    p.add_argument("--stride-t", type=int, default=2)

    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument(
        "--channel-mults",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        choices=range(1, 17),
        help=(
            "Channel multipliers for the U-Net resolution levels. Use "
            "`1 2 4 8` for the four-level model or `1 2 4` for legacy "
            "three-level checkpoints."
        ),
    )

    p.add_argument(
        "--mask-patterns",
        type=str,
        nargs="+",
        default=None,
        metavar="NAME WEIGHT",
        help=(
            "Alternating mask pattern names and sampling weights, for example "
            "`--mask-patterns spatial_random 2 spatial_grid 1`. Patterns that are "
            "not listed are never sampled. Available patterns: "
            f"{', '.join(MASK_PATTERNS)}. Default: "
            + " ".join(f"{k} {v:g}" for k, v in DEFAULT_PATTERN_WEIGHTS.items())
        ),
    )

    p.add_argument(
        "--val-patterns",
        type=str,
        nargs="+",
        default=list(MASK_PATTERNS),
        choices=list(MASK_PATTERNS),
        help="Mask patterns evaluated separately during validation.",
    )
    p.add_argument(
        "--val-mask-seed",
        type=int,
        default=4321,
        help=(
            "Seed for validation masks. Masks are keyed by (pattern, batch index), "
            "so every epoch sees the same validation masks."
        ),
    )

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

    args = p.parse_args()

    # Normalized weights are what training actually consumes, and what resume
    # compatibility is checked against. The raw token list is kept for the record.
    if args.mask_patterns is None:
        args.mask_pattern_weights = dict(DEFAULT_PATTERN_WEIGHTS)
    else:
        args.mask_pattern_weights = parse_pattern_weights(args.mask_patterns)
    args.masking_version = MASKING_VERSION

    return args


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
        f"bc{args.base_channels}_fourmask_{signature}"
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
    "masking_version",
    "h5_dir",
    "betas",
    "delta_t",
    "stride_t",
    "batch_size",
    "lr",
    "weight_decay",
    "mask_pattern_weights",
    "val_patterns",
    "val_mask_seed",
    "base_channels",
    "channel_mults",
    "val_frac",
    "seed",
    "stats_batches",
    "grad_clip",
    "amp",
)


def validate_resume_parameters(args, checkpoint_args: Dict[str, Any]) -> None:
    if "mask_pattern_weights" not in checkpoint_args:
        raise ValueError(
            "Refusing to auto-resume: this checkpoint predates multi-pattern masking "
            "and was trained with a single random mask at a fixed keep probability "
            f"(keep_prob={checkpoint_args.get('keep_prob')!r}). Its masking behaviour "
            "cannot be reproduced by the current code. Train into a fresh --out-dir, "
            "or pass --no-auto-resume to start over."
        )

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

    for i, batch in enumerate(
        tqdm(loader, desc="Estimating stats", disable=not is_main_process())
    ):
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
    pattern_weights: Dict[str, float],
    amp: bool,
    grad_clip: float,
    postfix_every: int = 50,
) -> Dict[str, Any]:
    model.train()

    n_patterns = len(MASK_PATTERNS)

    # Accumulate on device and read back once per epoch, so per-pattern logging
    # does not synchronize the GPU on every step.
    sum_mse = torch.zeros((), device=device)
    sum_mae = torch.zeros((), device=device)
    sum_mse_by_pattern = torch.zeros(n_patterns, device=device)
    sum_actual_by_pattern = torch.zeros(n_patterns, device=device)
    count_by_pattern = torch.zeros(n_patterns, device=device)

    sum_target_fraction = 0.0
    count_samples = 0
    total_batches = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    pbar = tqdm(loader, desc="Train", leave=False, disable=not is_main_process())

    for step, batch in enumerate(pbar):
        y = batch["block"].to(device, non_blocking=True)
        y = normalize(y, mean, std)

        batch_size = y.shape[0]

        patterns = sample_patterns(pattern_weights, batch_size)
        fractions = torch.rand(batch_size).tolist()

        mask, _ = sample_batch_masks(
            y.shape,
            patterns=patterns,
            mask_fractions=fractions,
            device=device,
            dtype=y.dtype,
        )

        x_visible = make_visible_input(y, mask)

        model_input = torch.cat([x_visible, mask], dim=1)  # (B, 8, T, X, Z)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            pred = model(model_input)
            loss_mse = full_mse_loss(pred, y)
            loss_mae = full_mae_loss(pred, y)
            loss = loss_mse

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            pattern_ids = torch.tensor(
                [PATTERN_TO_ID[name] for name in patterns],
                device=device,
            )
            per_sample_mse = (pred.detach().float() - y).pow(2).flatten(1).mean(1)
            per_sample_actual = 1.0 - mask.flatten(1).mean(1)

            sum_mse_by_pattern.index_add_(0, pattern_ids, per_sample_mse)
            sum_actual_by_pattern.index_add_(0, pattern_ids, per_sample_actual)
            count_by_pattern.index_add_(0, pattern_ids, torch.ones_like(per_sample_mse))

            sum_mse += loss_mse.detach().float()
            sum_mae += loss_mae.detach().float()

        sum_target_fraction += sum(fractions)
        count_samples += batch_size
        total_batches += 1

        if step % postfix_every == 0:
            pbar.set_postfix(
                mse=float(sum_mse) / total_batches,
                mae=float(sum_mae) / total_batches,
            )

    totals = torch.tensor(
        [
            float(sum_mse),
            float(sum_mae),
            float(total_batches),
            float(sum_target_fraction),
            float(count_samples),
        ],
        dtype=torch.float64,
        device=device,
    )
    if distributed_is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_mse_by_pattern, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_actual_by_pattern, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_by_pattern, op=dist.ReduceOp.SUM)

    global_mse, global_mae, global_batches, global_target, global_samples = (
        totals.tolist()
    )
    counts = count_by_pattern.cpu()
    mse_by_pattern = (sum_mse_by_pattern.cpu() / counts.clamp(min=1.0)).tolist()
    actual_by_pattern = (sum_actual_by_pattern.cpu() / counts.clamp(min=1.0)).tolist()
    counts = counts.tolist()

    per_pattern = {
        name: {
            "mse": mse_by_pattern[i],
            "actual_mask_fraction": actual_by_pattern[i],
            "share": counts[i] / max(global_samples, 1),
        }
        for i, name in enumerate(MASK_PATTERNS)
        if counts[i] > 0
    }

    total_count = max(global_samples, 1)

    return {
        "mse": global_mse / max(global_batches, 1),
        "mae": global_mae / max(global_batches, 1),
        "target_mask_fraction": global_target / total_count,
        "actual_mask_fraction": float(sum_actual_by_pattern.sum().cpu()) / total_count,
        "per_pattern": per_pattern,
    }


def val_mask_seed(base_seed: int, pattern: str, batch_index: int) -> int:
    """
    Deterministic seed per (pattern, validation batch).

    The validation loader is unshuffled, so the same batch gets the same mask in
    every epoch and the validation curves are not dominated by mask noise.
    """
    seed = base_seed * 1_000_003 + PATTERN_TO_ID[pattern] * 9_176 + batch_index
    return seed % (2 ** 31)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    patterns: List[str],
    mask_seed: int,
    amp: bool,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate every mask pattern on the same validation blocks.

    The loader is iterated once and each block is reused for all patterns, so
    the extra cost is forward passes rather than repeated HDF5 reads.
    """
    model.eval()

    sums = torch.zeros(len(patterns), 2, device=device)
    counts = torch.zeros(len(patterns), device=device)

    pbar = tqdm(loader, desc="Val", leave=False, disable=not is_main_process())

    for batch_index, batch in enumerate(pbar):
        y = batch["block"].to(device, non_blocking=True)
        y = normalize(y, mean, std)

        batch_size = y.shape[0]

        for i, pattern in enumerate(patterns):
            generator = torch.Generator()
            generator.manual_seed(val_mask_seed(mask_seed, pattern, batch_index))

            fractions = torch.rand(batch_size, generator=generator).tolist()

            mask, _ = sample_batch_masks(
                y.shape,
                patterns=[pattern] * batch_size,
                mask_fractions=fractions,
                device=device,
                dtype=y.dtype,
                generator=generator,
            )

            model_input = torch.cat([make_visible_input(y, mask), mask], dim=1)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                pred = model(model_input)
                loss_mse = full_mse_loss(pred, y)
                loss_mae = full_mae_loss(pred, y)

            sums[i, 0] += loss_mse.detach().float() * batch_size
            sums[i, 1] += loss_mae.detach().float() * batch_size
            counts[i] += batch_size

    if distributed_is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)

    sums = sums.cpu()
    counts = counts.cpu().clamp(min=1.0)

    metrics = {
        pattern: {
            "mse": float(sums[i, 0] / counts[i]),
            "mae": float(sums[i, 1] / counts[i]),
        }
        for i, pattern in enumerate(patterns)
    }

    metrics["mean"] = {
        "mse": sum(metrics[pattern]["mse"] for pattern in patterns) / len(patterns),
        "mae": sum(metrics[pattern]["mae"] for pattern in patterns) / len(patterns),
    }

    return metrics


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
    model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
    ckpt = {
        "model": model_to_save.state_dict(),
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
    device, rank, world_size, local_rank = initialize_distributed()
    set_seed(args.seed + rank)

    automatic_out_dir = Path("runs") / default_run_name(args)
    out_dir = expand_path(args.out_dir or str(automatic_out_dir))
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed_is_initialized():
        dist.barrier()

    h5_dir = expand_path(args.h5_dir)

    rank_print("HDF5 dir:", h5_dir)
    rank_print("Output dir:", out_dir)
    rank_print("Betas:", args.betas)
    rank_print("Mask pattern weights:", args.mask_pattern_weights)
    rank_print("Validation mask patterns:", args.val_patterns)
    rank_print(
        f"Distributed: world_size={world_size}, rank={rank}, "
        f"local_rank={local_rank}, device={device}"
    )
    rank_print(
        f"Batch size: {args.batch_size} per GPU, "
        f"global={args.batch_size * world_size}"
    )

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

    rank_print("Total windows:", len(dataset))
    rank_print("Train windows:", len(train_idx))
    rank_print("Val windows:", len(val_idx))
    rank_print("Train runs:", len(train_runs))
    rank_print("Val runs:", len(val_runs))

    if is_main_process():
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

    train_sampler = None
    val_sampler = None
    if distributed_is_initialized():
        train_sampler = DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        val_sampler = DistributedEvalSampler(
            val_set,
            rank=rank,
            world_size=world_size,
        )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    rank_print("Device:", device)

    resume_checkpoint = None
    latest_path = out_dir / "latest.pt"
    if args.auto_resume and latest_path.exists():
        resume_checkpoint = torch.load(latest_path, map_location=device)
        validate_resume_parameters(args, resume_checkpoint.get("args", {}))
        rank_print(
            f"Auto-resuming from {latest_path} "
            f"(completed epoch {resume_checkpoint['epoch']})"
        )

    if resume_checkpoint is not None and "stats" in resume_checkpoint:
        stats = resume_checkpoint["stats"]
        rank_print("Reusing channel statistics from checkpoint")
    else:
        stats = None
        if is_main_process():
            stats_loader = DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=False,
                drop_last=True,
                generator=torch.Generator().manual_seed(args.seed),
            )
            stats = estimate_channel_stats(
                stats_loader,
                device=device,
                max_batches=args.stats_batches,
            )
        if distributed_is_initialized():
            stats_object = [stats]
            dist.broadcast_object_list(stats_object, src=0)
            stats = stats_object[0]

    rank_print("Channel mean:", stats["mean"])
    rank_print("Channel std:", stats["std"])

    if is_main_process():
        with open(out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).view(1, 4, 1, 1, 1)

    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=args.base_channels,
        channel_mults=args.channel_mults,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    rank_print(f"Model parameters: {n_params / 1e6:.3f} M")

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

    if distributed_is_initialized():
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    if is_main_process():
        with open(out_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    if start_epoch > args.epochs:
        rank_print(
            f"Training is already complete at epoch {start_epoch - 1}; "
            f"requested epochs={args.epochs}. Increase --epochs to continue."
        )
        if distributed_is_initialized():
            dist.destroy_process_group()
        return

    resume_run_id = None
    if resume_checkpoint is not None:
        resume_run_id = resume_checkpoint.get("wandb_run_id")
    wandb_run = (
        init_wandb(args, out_dir, resume_run_id=resume_run_id)
        if is_main_process()
        else None
    )
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
        wandb_run.define_metric("mask/*", step_metric="epoch")
        wandb_run.define_metric("mask_pattern/*", step_metric="epoch")
        wandb_run.define_metric("optimization/*", step_metric="epoch")

    history = []
    history_path = out_dir / "history.json"
    if is_main_process() and resume_checkpoint is not None and history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        history = [row for row in history if row.get("epoch", 0) < start_epoch]

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        rank_print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mean=mean,
            std=std,
            pattern_weights=args.mask_pattern_weights,
            amp=args.amp,
            grad_clip=args.grad_clip,
        )

        val_metrics = validate(
            model=(model.module if isinstance(model, DistributedDataParallel) else model),
            loader=val_loader,
            device=device,
            mean=mean,
            std=std,
            patterns=args.val_patterns,
            mask_seed=args.val_mask_seed,
            amp=args.amp,
        )

        row = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "val_mse": val_metrics["mean"]["mse"],
            "val_mae": val_metrics["mean"]["mae"],
            "train_mask_target_fraction": train_metrics["target_mask_fraction"],
            "train_mask_actual_fraction": train_metrics["actual_mask_fraction"],
            "train_per_pattern": train_metrics["per_pattern"],
            "val_per_pattern": {
                pattern: val_metrics[pattern] for pattern in args.val_patterns
            },
        }
        if is_main_process():
            history.append(row)

        if wandb_run is not None:
            log_data = {
                "epoch": epoch,
                "train/mse": row["train_mse"],
                "train/mae": row["train_mae"],
                "val/mse": row["val_mse"],
                "val/mae": row["val_mae"],
                "val/mean/mse": row["val_mse"],
                "val/mean/mae": row["val_mae"],
                "mask/target_fraction": row["train_mask_target_fraction"],
                "mask/actual_fraction": row["train_mask_actual_fraction"],
                "optimization/learning_rate": optimizer.param_groups[0]["lr"],
            }

            for pattern, pattern_metrics in train_metrics["per_pattern"].items():
                log_data[f"train/{pattern}/mse"] = pattern_metrics["mse"]
                log_data[f"mask/{pattern}/actual_fraction"] = pattern_metrics[
                    "actual_mask_fraction"
                ]
                log_data[f"mask_pattern/{pattern}"] = pattern_metrics["share"]

            for pattern in args.val_patterns:
                log_data[f"val/{pattern}/mse"] = val_metrics[pattern]["mse"]
                log_data[f"val/{pattern}/mae"] = val_metrics[pattern]["mae"]

            wandb_run.log(log_data)

        rank_print(
            f"train_mse={row['train_mse']:.6f} "
            f"train_mae={row['train_mae']:.6f} "
            f"val_mse={row['val_mse']:.6f} "
            f"val_mae={row['val_mae']:.6f}"
        )
        rank_print(
            "  val by pattern: "
            + "  ".join(
                f"{pattern}={val_metrics[pattern]['mse']:.6f}"
                for pattern in args.val_patterns
            )
        )

        if is_main_process():
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

        if row["val_mse"] < best_val_mse:
            best_val_mse = row["val_mse"]
            if is_main_process():
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
                rank_print(f"Saved best checkpoint: val_mse={best_val_mse:.6f}")

        if is_main_process():
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
        if distributed_is_initialized():
            dist.barrier()

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

    if distributed_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
