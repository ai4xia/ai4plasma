# train_masked_unet3d.py

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from models.unet3d import MODEL_VERSION, UNet3D


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
    p.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Linear LR warmup length before cosine decay (default: 10 epochs).",
    )
    p.add_argument(
        "--min-lr",
        type=float,
        default=2e-6,
        help="Final learning rate reached by cosine decay (default: 2e-6).",
    )
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
        "--use-attention",
        action="store_true",
        help=(
            "Enable unified spatiotemporal self-attention with 3D RoPE at "
            "the low-resolution enc2 and bottleneck feature maps."
        ),
    )
    p.add_argument(
        "--spatial-only-pooling",
        action="store_true",
        help=(
            "Downsample only X/Z with pooling kernel (1, 2, 2), preserving "
            "the temporal resolution at every encoder level."
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
    p.add_argument(
        "--density-probe-min",
        type=int,
        default=0,
        help=(
            "Minimum of the sparse Density-probe interval used by "
            "spatial_grid and spatial_random (default: 0)."
        ),
    )
    p.add_argument(
        "--density-probe-max",
        type=int,
        default=30,
        help=(
            "Maximum of the sparse Density-probe interval used by "
            "spatial_grid and spatial_random (default: 30). Each sample has "
            "50%% probability of using this interval and 50%% probability "
            "of using [maximum + 1, X * Z]."
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
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Initialize model, optimizer, and channel statistics from a checkpoint, "
            "while starting a new epoch/LR schedule. Ignored when auto-resuming from "
            "out-dir/latest.pt."
        ),
    )

    args = p.parse_args()

    # Normalized weights are what training actually consumes, and what resume
    # compatibility is checked against. The raw token list is kept for the record.
    if args.mask_patterns is None:
        args.mask_pattern_weights = dict(DEFAULT_PATTERN_WEIGHTS)
    else:
        args.mask_pattern_weights = parse_pattern_weights(args.mask_patterns)
    args.masking_version = MASKING_VERSION
    args.model_version = MODEL_VERSION

    if args.epochs <= 0:
        p.error(f"--epochs must be positive, got {args.epochs}")
    if not (0 <= args.warmup_epochs < args.epochs):
        p.error(
            "--warmup-epochs must lie in [0, epochs), got "
            f"{args.warmup_epochs} for epochs={args.epochs}"
        )
    if args.lr <= 0:
        p.error(f"--lr must be positive, got {args.lr}")
    if not (0 <= args.min_lr <= args.lr):
        p.error(
            f"--min-lr must lie in [0, lr], got min_lr={args.min_lr}, "
            f"lr={args.lr}"
        )
    if not (0 <= args.density_probe_min <= args.density_probe_max):
        p.error(
            "Expected 0 <= density-probe-min <= density-probe-max, got "
            f"{args.density_probe_min} and {args.density_probe_max}"
        )

    return args


def expand_path(path: str) -> Path:
    return Path(os.path.expandvars(path)).expanduser().resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate_for_epoch(
    epoch: int,
    total_epochs: int,
    base_lr: float,
    warmup_epochs: int,
    min_lr: float,
) -> float:
    """Return the deterministic linear-warmup + cosine-decay LR for one epoch."""
    if not (1 <= epoch <= total_epochs):
        raise ValueError(f"epoch must lie in [1, {total_epochs}], got {epoch}")
    if not (0 <= warmup_epochs < total_epochs):
        raise ValueError(
            f"warmup_epochs must lie in [0, {total_epochs}), got {warmup_epochs}"
        )
    if not (0 <= min_lr <= base_lr):
        raise ValueError(
            f"Expected 0 <= min_lr <= base_lr, got {min_lr} and {base_lr}"
        )

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * float(epoch) / float(warmup_epochs)

    decay_progress = float(epoch - warmup_epochs) / float(
        total_epochs - warmup_epochs
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (base_lr - min_lr) * cosine


def sample_mixed_density_probe_counts(
    patterns: Sequence[str],
    minimum: int,
    maximum: int,
    spatial_sites: int,
    generator: Optional[torch.Generator] = None,
) -> List[Optional[int]]:
    """Draw exact Density counts from a 50/50 sparse/dense mixture."""
    if not (0 <= minimum <= maximum < spatial_sites):
        raise ValueError(
            "Expected 0 <= minimum <= maximum < spatial_sites, got "
            f"{minimum}, {maximum}, and {spatial_sites}."
        )

    counts: List[Optional[int]] = []
    for pattern in patterns:
        if pattern in {"spatial_grid", "spatial_random"}:
            use_sparse_interval = _torch_random_bool(generator)
            low = minimum if use_sparse_interval else maximum + 1
            high = maximum if use_sparse_interval else spatial_sites
            count = int(
                torch.randint(
                    low,
                    high + 1,
                    (1,),
                    generator=generator,
                ).item()
            )
            counts.append(count)
        else:
            counts.append(None)
    return counts


def _torch_random_bool(generator: Optional[torch.Generator] = None) -> bool:
    """Return a reproducible fair coin flip using the mask RNG."""
    return bool(torch.randint(0, 2, (1,), generator=generator).item())


def sample_mixed_magnetic_visible_counts(
    patterns: Sequence[str],
    spatial_sites: int,
    generator: Optional[torch.Generator] = None,
) -> List[Optional[int]]:
    """Draw 50% fully visible B layouts and 50% uniform-count layouts."""
    if spatial_sites <= 0:
        raise ValueError(f"spatial_sites must be positive, got {spatial_sites}.")

    counts: List[Optional[int]] = []
    for pattern in patterns:
        if pattern in {"spatial_grid", "spatial_random"}:
            if _torch_random_bool(generator):
                count = spatial_sites
            else:
                count = int(
                    torch.randint(
                        0,
                        spatial_sites + 1,
                        (1,),
                        generator=generator,
                    ).item()
                )
            counts.append(count)
        else:
            counts.append(None)
    return counts


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
    "model_version",
    "masking_version",
    "h5_dir",
    "betas",
    "delta_t",
    "stride_t",
    "batch_size",
    "epochs",
    "lr",
    "warmup_epochs",
    "min_lr",
    "weight_decay",
    "mask_pattern_weights",
    "val_patterns",
    "val_mask_seed",
    "density_probe_min",
    "density_probe_max",
    "base_channels",
    "channel_mults",
    "use_attention",
    "spatial_only_pooling",
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
        if key in {"use_attention", "spatial_only_pooling"}:
            # Checkpoints created before either optional architecture extension
            # use the original attention-off, full-3D-pooling behavior.
            old_value = bool(checkpoint_args.get(key, False))
        else:
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
    density_probe_min: int,
    density_probe_max: int,
    amp: bool,
    grad_clip: float,
    postfix_every: int = 50,
    epoch_number: Optional[int] = None,
    epoch_progress: Optional[Any] = None,
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
    density_probe_sum = 0
    density_probe_sample_count = 0
    sparse_density_sample_count = 0
    magnetic_visible_sum = 0
    magnetic_sample_count = 0
    fully_visible_magnetic_sample_count = 0
    spatial_sites = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    batches_per_epoch = max(len(loader), 1)

    for step, batch in enumerate(loader):
        y = batch["block"].to(device, non_blocking=True)
        y = normalize(y, mean, std)

        batch_size = y.shape[0]

        patterns = sample_patterns(pattern_weights, batch_size)
        fractions = torch.rand(batch_size).tolist()
        spatial_sites = int(y.shape[-2] * y.shape[-1])
        density_probe_counts = sample_mixed_density_probe_counts(
            patterns,
            minimum=density_probe_min,
            maximum=density_probe_max,
            spatial_sites=spatial_sites,
        )
        magnetic_visible_counts = sample_mixed_magnetic_visible_counts(
            patterns,
            spatial_sites=spatial_sites,
        )
        probe_counts_in_batch = [
            count for count in density_probe_counts if count is not None
        ]
        density_probe_sum += sum(probe_counts_in_batch)
        density_probe_sample_count += len(probe_counts_in_batch)
        sparse_density_sample_count += sum(
            count <= density_probe_max for count in probe_counts_in_batch
        )
        magnetic_counts_in_batch = [
            count for count in magnetic_visible_counts if count is not None
        ]
        magnetic_visible_sum += sum(magnetic_counts_in_batch)
        magnetic_sample_count += len(magnetic_counts_in_batch)
        fully_visible_magnetic_sample_count += sum(
            count == spatial_sites for count in magnetic_counts_in_batch
        )
        for i, (probe_count, magnetic_count) in enumerate(
            zip(density_probe_counts, magnetic_visible_counts)
        ):
            if probe_count is not None:
                if magnetic_count is None:
                    raise RuntimeError(
                        "Probe-pattern sample is missing its magnetic visible count."
                    )
                if probe_count > spatial_sites:
                    raise ValueError(
                        f"Density probe count {probe_count} exceeds the "
                        f"{spatial_sites} spatial sites."
                    )
                # Three B channels share magnetic_count visible sites; Density
                # independently contains probe_count sites.
                visible_values = 3 * magnetic_count + probe_count
                fractions[i] = 1.0 - visible_values / (4.0 * spatial_sites)

        mask, _ = sample_batch_masks(
            y.shape,
            patterns=patterns,
            mask_fractions=fractions,
            device=device,
            dtype=y.dtype,
            density_probe_counts=density_probe_counts,
            magnetic_visible_counts=magnetic_visible_counts,
        )

        x_visible = make_visible_input(y, mask)

        model_input = torch.cat([x_visible, mask], dim=1)  # (B, 8, T, X, Z)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            pred = model(model_input)
            loss_mse = full_mse_loss(pred, y, mask)
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

        if epoch_progress is not None:
            epoch_progress.update(1.0 / batches_per_epoch)

        if epoch_progress is not None and step % postfix_every == 0:
            epoch_progress.set_postfix(
                phase="train",
                mse=float(sum_mse) / total_batches,
                mae=float(sum_mae) / total_batches,
            )

    if epoch_progress is not None and epoch_number is not None:
        # Eliminate floating-point accumulation drift at every epoch boundary.
        epoch_progress.n = float(epoch_number)
        epoch_progress.refresh()

    totals = torch.tensor(
        [
            float(sum_mse),
            float(sum_mae),
            float(total_batches),
            float(sum_target_fraction),
            float(count_samples),
            float(density_probe_sum),
            float(density_probe_sample_count),
            float(sparse_density_sample_count),
            float(magnetic_visible_sum),
            float(magnetic_sample_count),
            float(fully_visible_magnetic_sample_count),
        ],
        dtype=torch.float64,
        device=device,
    )
    if distributed_is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_mse_by_pattern, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_actual_by_pattern, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_by_pattern, op=dist.ReduceOp.SUM)

    (
        global_mse,
        global_mae,
        global_batches,
        global_target,
        global_samples,
        global_probe_sum,
        global_probe_sample_count,
        global_sparse_sample_count,
        global_magnetic_visible_sum,
        global_magnetic_sample_count,
        global_fully_visible_magnetic_sample_count,
    ) = totals.tolist()
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
    probe_count = max(global_probe_sample_count, 1)
    mean_density_probes = global_probe_sum / probe_count
    magnetic_count = max(global_magnetic_sample_count, 1)
    mean_magnetic_visible = global_magnetic_visible_sum / magnetic_count

    return {
        "mse": global_mse / max(global_batches, 1),
        "mae": global_mae / max(global_batches, 1),
        "target_mask_fraction": global_target / total_count,
        "actual_mask_fraction": float(sum_actual_by_pattern.sum().cpu()) / total_count,
        "density_probe_count": mean_density_probes,
        "density_visible_fraction": mean_density_probes / max(spatial_sites, 1),
        "sparse_density_sample_share": global_sparse_sample_count / probe_count,
        "magnetic_visible_count": mean_magnetic_visible,
        "magnetic_visible_fraction": mean_magnetic_visible / max(spatial_sites, 1),
        "fully_visible_magnetic_sample_share": (
            global_fully_visible_magnetic_sample_count / magnetic_count
        ),
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
    density_probe_min: int,
    density_probe_max: int,
    amp: bool,
    epoch_progress: Optional[Any] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate every mask pattern on the same validation blocks.

    The loader is iterated once and each block is reused for all patterns, so
    the extra cost is forward passes rather than repeated HDF5 reads.
    """
    model.eval()

    sums = torch.zeros(len(patterns), 2, device=device)
    counts = torch.zeros(len(patterns), device=device)

    if epoch_progress is not None:
        epoch_progress.set_postfix(phase="val")

    for batch_index, batch in enumerate(loader):
        y = batch["block"].to(device, non_blocking=True)
        y = normalize(y, mean, std)

        batch_size = y.shape[0]

        for i, pattern in enumerate(patterns):
            generator = torch.Generator()
            generator.manual_seed(val_mask_seed(mask_seed, pattern, batch_index))

            fractions = torch.rand(batch_size, generator=generator).tolist()
            spatial_sites = int(y.shape[-2] * y.shape[-1])
            density_probe_counts = sample_mixed_density_probe_counts(
                [pattern] * batch_size,
                minimum=density_probe_min,
                maximum=density_probe_max,
                spatial_sites=spatial_sites,
                generator=generator,
            )
            magnetic_visible_counts = sample_mixed_magnetic_visible_counts(
                [pattern] * batch_size,
                spatial_sites=spatial_sites,
                generator=generator,
            )
            for sample_index, (probe_count, magnetic_count) in enumerate(
                zip(density_probe_counts, magnetic_visible_counts)
            ):
                if probe_count is not None:
                    if magnetic_count is None:
                        raise RuntimeError(
                            "Probe-pattern validation sample is missing its "
                            "magnetic visible count."
                        )
                    if probe_count > spatial_sites:
                        raise ValueError(
                            f"Density probe count {probe_count} exceeds the "
                            f"{spatial_sites} spatial sites."
                        )
                    visible_values = 3 * magnetic_count + probe_count
                    fractions[sample_index] = 1.0 - (
                        visible_values / (4.0 * spatial_sites)
                    )

            mask, _ = sample_batch_masks(
                y.shape,
                patterns=[pattern] * batch_size,
                mask_fractions=fractions,
                device=device,
                dtype=y.dtype,
                generator=generator,
                density_probe_counts=density_probe_counts,
                magnetic_visible_counts=magnetic_visible_counts,
            )

            model_input = torch.cat([make_visible_input(y, mask), mask], dim=1)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
                pred = model(model_input)
                loss_mse = full_mse_loss(pred, y, mask)
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
    rank_print("Model version:", args.model_version)
    rank_print(
        "spatial_grid/spatial_random: B uses a 50/50 mixture of fully visible "
        "and an exact count uniformly drawn from [0, X*Z]; Density uses "
        f"a 50/50 mixture of [{args.density_probe_min}, "
        f"{args.density_probe_max}] and "
        f"[{args.density_probe_max + 1}, X*Z]"
    )
    rank_print(
        "Learning-rate schedule: "
        f"linear warmup {args.warmup_epochs} epochs, "
        f"base_lr={args.lr:g}, cosine min_lr={args.min_lr:g}, "
        f"total_epochs={args.epochs}"
    )
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
    init_checkpoint = None
    latest_path = out_dir / "latest.pt"
    if args.auto_resume and latest_path.exists():
        resume_checkpoint = torch.load(latest_path, map_location=device)
        validate_resume_parameters(args, resume_checkpoint.get("args", {}))
        rank_print(
            f"Auto-resuming from {latest_path} "
            f"(completed epoch {resume_checkpoint['epoch']})"
        )
    elif args.init_checkpoint is not None:
        init_path = expand_path(args.init_checkpoint)
        if not init_path.exists():
            raise FileNotFoundError(f"Initialization checkpoint not found: {init_path}")
        init_checkpoint = torch.load(init_path, map_location=device)
        rank_print(
            f"Initializing from {init_path} "
            f"(source epoch {init_checkpoint['epoch']}); starting a new schedule"
        )

    state_checkpoint = (
        resume_checkpoint if resume_checkpoint is not None else init_checkpoint
    )
    if state_checkpoint is not None and "stats" in state_checkpoint:
        stats = state_checkpoint["stats"]
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
        architecture=args.model_version,
        use_attention=args.use_attention,
        spatial_only_pooling=args.spatial_only_pooling,
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
    elif init_checkpoint is not None:
        source_uses_attention = bool(
            init_checkpoint.get("args", {}).get("use_attention", False)
        )
        if source_uses_attention == args.use_attention:
            model.load_state_dict(init_checkpoint["model"])
            optimizer.load_state_dict(init_checkpoint["optimizer"])
        else:
            source_state = init_checkpoint["model"]
            if args.use_attention:
                incompatible = model.load_state_dict(source_state, strict=False)
                invalid_missing = [
                    key
                    for key in incompatible.missing_keys
                    if not key.startswith(("attention_enc2.", "attention_mid."))
                ]
                invalid_unexpected = list(incompatible.unexpected_keys)
            else:
                convolutional_state = {
                    key: value
                    for key, value in source_state.items()
                    if not key.startswith(("attention_enc2.", "attention_mid."))
                }
                incompatible = model.load_state_dict(
                    convolutional_state,
                    strict=False,
                )
                invalid_missing = list(incompatible.missing_keys)
                invalid_unexpected = list(incompatible.unexpected_keys)

            if invalid_missing or invalid_unexpected:
                raise RuntimeError(
                    "Initialization checkpoint is incompatible beyond its "
                    "attention setting: "
                    f"missing={invalid_missing}, unexpected={invalid_unexpected}"
                )
            rank_print(
                "Attention setting differs from the initialization checkpoint; "
                "loaded compatible convolutional weights and started a fresh "
                "optimizer state."
            )

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

    epoch_progress = None
    if is_main_process():
        epoch_progress = tqdm(
            total=float(args.epochs),
            initial=float(start_epoch - 1),
            desc="Training",
            unit="epoch",
            dynamic_ncols=True,
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| "
                "{n:.2f}/{total:.0f} epochs [{elapsed}<{remaining}{postfix}]"
            ),
        )

    def epoch_log(message: str) -> None:
        if epoch_progress is not None:
            epoch_progress.write(message)
        else:
            rank_print(message)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_lr = learning_rate_for_epoch(
            epoch=epoch,
            total_epochs=args.epochs,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = epoch_lr
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if epoch_progress is not None:
            epoch_progress.set_postfix(phase="train", lr=f"{epoch_lr:.8g}")

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mean=mean,
            std=std,
            pattern_weights=args.mask_pattern_weights,
            density_probe_min=args.density_probe_min,
            density_probe_max=args.density_probe_max,
            amp=args.amp,
            grad_clip=args.grad_clip,
            epoch_number=epoch,
            epoch_progress=epoch_progress,
        )

        val_metrics = validate(
            model=(model.module if isinstance(model, DistributedDataParallel) else model),
            loader=val_loader,
            device=device,
            mean=mean,
            std=std,
            patterns=args.val_patterns,
            mask_seed=args.val_mask_seed,
            density_probe_min=args.density_probe_min,
            density_probe_max=args.density_probe_max,
            amp=args.amp,
            epoch_progress=epoch_progress,
        )

        row = {
            "epoch": epoch,
            "learning_rate": epoch_lr,
            "train_mse": train_metrics["mse"],
            "train_mae": train_metrics["mae"],
            "val_mse": val_metrics["mean"]["mse"],
            "val_mae": val_metrics["mean"]["mae"],
            "train_mask_target_fraction": train_metrics["target_mask_fraction"],
            "train_mask_actual_fraction": train_metrics["actual_mask_fraction"],
            "train_density_probe_count": train_metrics[
                "density_probe_count"
            ],
            "train_density_visible_fraction": train_metrics[
                "density_visible_fraction"
            ],
            "train_sparse_density_sample_share": train_metrics[
                "sparse_density_sample_share"
            ],
            "train_magnetic_visible_count": train_metrics[
                "magnetic_visible_count"
            ],
            "train_magnetic_visible_fraction": train_metrics[
                "magnetic_visible_fraction"
            ],
            "train_fully_visible_magnetic_sample_share": train_metrics[
                "fully_visible_magnetic_sample_share"
            ],
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
                "mask/density_probe_patterns/probe_count": row[
                    "train_density_probe_count"
                ],
                "mask/density_probe_patterns/visible_fraction": row[
                    "train_density_visible_fraction"
                ],
                "mask/density_probe_patterns/sparse_sample_share": row[
                    "train_sparse_density_sample_share"
                ],
                "mask/magnetic_probe_patterns/visible_count": row[
                    "train_magnetic_visible_count"
                ],
                "mask/magnetic_probe_patterns/visible_fraction": row[
                    "train_magnetic_visible_fraction"
                ],
                "mask/magnetic_probe_patterns/fully_visible_sample_share": row[
                    "train_fully_visible_magnetic_sample_share"
                ],
                "optimization/learning_rate": row["learning_rate"],
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

        epoch_log(
            f"Epoch {epoch}/{args.epochs}  lr={epoch_lr:.8g}  "
            f"train_mse={row['train_mse']:.6f} "
            f"train_mae={row['train_mae']:.6f} "
            f"val_mse={row['val_mse']:.6f} "
            f"val_mae={row['val_mae']:.6f} "
            f"density_probes="
            f"{row['train_density_probe_count']:.2f} "
            f"sparse_probe_share="
            f"{row['train_sparse_density_sample_share']:.3f} "
            f"B_visible={row['train_magnetic_visible_fraction']:.3f} "
            f"B_full_share="
            f"{row['train_fully_visible_magnetic_sample_share']:.3f}"
        )
        epoch_log(
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
                epoch_log(f"Saved best checkpoint: val_mse={best_val_mse:.6f}")

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

    if epoch_progress is not None:
        epoch_progress.close()

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
