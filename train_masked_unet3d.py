# train_masked_unet3d.py

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.vpic_hdf5_dataset import (
    FIXED_SPLIT_COUNTS,
    FIXED_SPLIT_PATH,
    VPICWindowDataset,
    fixed_split_metadata,
    fixed_split_sha256,
    h5_run_lengths,
    load_fixed_split,
    parse_run_name,
    validate_checkpoint_split,
)
from data.masking import (
    DEFAULT_PATTERN_WEIGHTS,
    MASKING_VERSION,
    MASK_PATTERNS,
    PATTERN_TO_ID,
    PROBE_COUNT_PATTERNS,
    full_mse_loss,
    make_visible_input,
    make_uncertain_input,
    make_rollout_input,
    predicted_error_mse_loss,
    GAUSSIAN_NLL_VARIANCE_FLOOR,
    parse_pattern_weights,
    sample_independent_batch_masks,
    sample_patterns,
    sample_training_severity_counts,
    make_fixed_validation_mask,
)
from models.unet3d import DIRECT_MODEL_VERSION, MODEL_VERSION, UNet3D


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
    p.add_argument("--epochs", type=int, default=20,
                   help="Cosine LR horizon and default final epoch; curriculum can continue at min LR using --stop-after-epochs.")
    p.add_argument("--stop-after-epochs", type=int, default=None,
                   help="Stop after this completed epoch without changing --epochs LR horizon; may exceed the horizon in curriculum mode.")
    p.add_argument(
        "--final-epoch", type=int, default=None,
        help=("Predetermined official final epoch. Bounded curriculum stages may stop "
              "earlier, but final.pt is written only at this epoch."),
    )
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
    p.add_argument("--model-version", choices=(MODEL_VERSION, DIRECT_MODEL_VERSION), default=MODEL_VERSION,
                   help="Output semantics: residual v2 (default), or explicit direct-output v3 for a new run.")
    p.add_argument(
        "--uncertainty-aware", action="store_true",
        help="Use input variance and predict per-element squared reconstruction error.",
    )
    p.add_argument(
        "--uncertainty-loss", choices=("smooth_l1", "gaussian_nll"), default="smooth_l1",
        help="Variance objective; gaussian_nll enables controlled binary-input Stage-1 checks.",
    )
    p.add_argument(
        "--uncertainty-loss-weight", type=float, default=1.0,
        help="Weight of the selected variance objective (default: 1).",
    )
    p.add_argument("--curriculum-training", action="store_true",
                   help="Opt-in reconstruction / detached variance / two-pass curriculum.")
    p.add_argument("--variance-ramp-start-epoch", type=float, default=None,
                   help="Optional legacy scheduled start (completed epochs); omit to leave the ramp unarmed.")
    p.add_argument("--variance-ramp-epochs", type=float, default=40.0)
    p.add_argument("--rollout-ramp-start-epoch", type=float, default=None,
                   help="Optional legacy scheduled start (completed epochs); omit to leave the ramp unarmed.")
    p.add_argument("--rollout-ramp-epochs", type=float, default=40.0)
    arm_group = p.add_mutually_exclusive_group()
    arm_group.add_argument("--arm-variance-ramp", action="store_true",
                           help="Resume only: arm the unarmed variance ramp at the saved successful update count.")
    arm_group.add_argument("--arm-rollout-ramp", action="store_true",
                           help="Resume only: arm rollout after the variance ramp has completed.")
    p.add_argument("--curriculum-log-every", type=int, default=50,
                   help="Batch interval for curriculum component logs; lambdas update every successful step.")
    p.add_argument("--transition-ema-half-life", type=float, default=5.0,
                   help="Curriculum convergence EMA half-life in epochs (MSE: log space; NLL: direct).")
    p.add_argument("--transition-rate-window", type=int, default=5)
    p.add_argument("--transition-rate-threshold", type=float, default=0.005,
                   help="Automatic curriculum-transition threshold (default: 0.005 = 0.5% per epoch).")
    p.add_argument("--transition-patience", type=int, default=4)
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
            "Unused by current standardized validation, which uses fixed ~50% "
            "layouts for all five patterns. Kept so older launch scripts still parse."
        ),
    )
    p.add_argument(
        "--density-probe-min",
        type=int,
        default=0,
        help=(
            "Unused by current standardized validation. Kept so older launch "
            "scripts still parse."
        ),
    )
    p.add_argument(
        "--density-probe-max",
        type=int,
        default=30,
        help=(
            "Unused by current standardized validation. Kept so older launch "
            "scripts still parse."
        ),
    )
    p.add_argument(
        "--density-visible-p-zero",
        type=float,
        default=0.15,
        help=(
            "Training-only: probability that a Density probe pattern has "
            "zero visible sites (default: 0.15)."
        ),
    )
    p.add_argument(
        "--density-visible-p-full",
        type=float,
        default=0.10,
        help=(
            "Training-only: probability that a Density probe pattern keeps "
            "every spatial site (default: 0.10)."
        ),
    )
    p.add_argument(
        "--magnetic-visible-p-zero",
        type=float,
        default=0.10,
        help=(
            "Training-only: probability that a magnetic probe pattern has "
            "zero visible sites (default: 0.10)."
        ),
    )
    p.add_argument(
        "--magnetic-visible-p-full",
        type=float,
        default=0.30,
        help=(
            "Training-only: probability that a magnetic probe pattern keeps "
            "every spatial site (default: 0.30)."
        ),
    )

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
    p.add_argument(
        "--amp-dtype", choices=("float16", "bfloat16"), default="float16",
        help="AMP precision; float16 preserves existing runs, bfloat16 avoids FP16 range overflow.",
    )

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
        help="Upload final.pt as a W&B artifact after the predetermined final epoch.",
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
            "Initialize model weights and channel statistics from a checkpoint, "
            "with a fresh optimizer and epoch/LR schedule. Masking-version "
            "mismatches are allowed here because only weights are loaded. "
            "Ignored when auto-resuming from out-dir/latest.pt."
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
    split_metadata = fixed_split_metadata()
    args.split_path = split_metadata["path"]
    args.split_sha256 = split_metadata["sha256"]

    if (
        not math.isfinite(args.uncertainty_loss_weight)
        or args.uncertainty_loss_weight < 0
    ):
        p.error("--uncertainty-loss-weight must be finite and nonnegative")

    if args.epochs <= 0:
        p.error(f"--epochs must be positive, got {args.epochs}")
    if not math.isfinite(args.transition_ema_half_life) or args.transition_ema_half_life <= 0:
        p.error("--transition-ema-half-life must be finite and positive")
    if args.transition_rate_window < 1 or args.transition_patience < 1:
        p.error("--transition-rate-window and --transition-patience must be positive")
    if not math.isfinite(args.transition_rate_threshold) or args.transition_rate_threshold < 0:
        p.error("--transition-rate-threshold must be finite and nonnegative")
    if args.stop_after_epochs is not None and (args.stop_after_epochs < 1 or
            (not args.curriculum_training and args.stop_after_epochs > args.epochs)):
        p.error("--stop-after-epochs must be positive; only curriculum runs may exceed --epochs")
    if args.final_epoch is None:
        args.final_epoch = args.stop_after_epochs or args.epochs
    if args.final_epoch < (args.stop_after_epochs or args.epochs):
        p.error("--final-epoch cannot precede the requested stopping epoch")
    if (args.arm_variance_ramp or args.arm_rollout_ramp) and (not args.curriculum_training or not args.auto_resume):
        p.error("Ramp arming requires --curriculum-training and --auto-resume")
    if args.uncertainty_loss == "gaussian_nll" and not args.uncertainty_aware:
        p.error("--uncertainty-loss gaussian_nll requires --uncertainty-aware")
    if args.uncertainty_loss == "gaussian_nll":
        args.variance_floor = GAUSSIAN_NLL_VARIANCE_FLOOR
    if args.curriculum_training:
        if not args.uncertainty_aware or args.uncertainty_loss != "gaussian_nll":
            p.error("--curriculum-training requires --uncertainty-aware --uncertainty-loss gaussian_nll")
        if args.uncertainty_loss_weight != 1.0:
            p.error("Curriculum variance weighting is controlled by lambda_v; --uncertainty-loss-weight must be 1")
        for name in ("variance_ramp_start_epoch", "rollout_ramp_start_epoch",
                     "variance_ramp_epochs", "rollout_ramp_epochs"):
            value = getattr(args, name)
            if value is None and name.endswith("start_epoch"):
                continue
            if value is None or not math.isfinite(value) or value < 0:
                p.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
        if min(args.variance_ramp_epochs, args.rollout_ramp_epochs) <= 0:
            p.error("Curriculum ramp durations must be positive")
        if args.rollout_ramp_start_epoch is not None and (args.variance_ramp_start_epoch is None or
                args.rollout_ramp_start_epoch < args.variance_ramp_start_epoch + args.variance_ramp_epochs):
            p.error("Rollout ramp must start at or after the variance ramp ends; overlapping ramps are not supported")
        if args.curriculum_log_every <= 0:
            p.error("--curriculum-log-every must be positive")
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
    for option_name in (
        "density_visible_p_zero",
        "density_visible_p_full",
        "magnetic_visible_p_zero",
        "magnetic_visible_p_full",
    ):
        value = getattr(args, option_name)
        if not (0.0 <= value <= 1.0):
            p.error(f"--{option_name.replace('_', '-')} must lie in [0, 1], got {value}")
    if args.density_visible_p_zero + args.density_visible_p_full > 1.0:
        p.error(
            "--density-visible-p-zero + --density-visible-p-full must be <= 1, "
            f"got {args.density_visible_p_zero} + {args.density_visible_p_full}"
        )
    if args.magnetic_visible_p_zero + args.magnetic_visible_p_full > 1.0:
        p.error(
            "--magnetic-visible-p-zero + --magnetic-visible-p-full must be <= 1, "
            f"got {args.magnetic_visible_p_zero} + {args.magnetic_visible_p_full}"
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
    if getattr(args, "uncertainty_loss", "smooth_l1") != "smooth_l1":
        signature_data["uncertainty_loss"] = args.uncertainty_loss
    if getattr(args, "curriculum_training", False):
        signature_data.update({key: getattr(args, key) for key in (
            "curriculum_training", "variance_ramp_start_epoch", "variance_ramp_epochs",
            "rollout_ramp_start_epoch", "rollout_ramp_epochs")})
    signature = hashlib.sha1(
        json.dumps(signature_data, sort_keys=True).encode("utf-8")
    ).hexdigest()[:8]
    return (
        f"masked-unet3d_beta{beta_text}_dt{args.delta_t}_"
        f"bc{args.base_channels}_fivemask_{signature}"
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
    "density_visible_p_zero",
    "density_visible_p_full",
    "magnetic_visible_p_zero",
    "magnetic_visible_p_full",
    "base_channels",
    "channel_mults",
    "use_attention",
    "spatial_only_pooling",
    "final_epoch",
    "split_sha256",
    "seed",
    "stats_batches",
    "grad_clip",
    "amp",
)


class CheckpointPlan(NamedTuple):
    """How training obtains starting weights and optimizer state."""

    mode: str
    path: Optional[Path]


def resolve_checkpoint_plan(
    out_dir: Path,
    *,
    auto_resume: bool,
    init_checkpoint: Optional[str] = None,
) -> CheckpointPlan:
    """Prefer an in-run latest checkpoint over a one-shot weight initialization.

    Resume is only considered for ``out_dir/latest.pt``. An ``init_checkpoint``
    never restores optimizer, scheduler, or epoch state, and is ignored once a
    resume file exists. Masking-version compatibility is enforced only for
    resume, not for weight-only initialization.
    """
    latest_path = Path(out_dir) / "latest.pt"
    if auto_resume and latest_path.is_file():
        return CheckpointPlan("resume", latest_path)
    if init_checkpoint:
        return CheckpointPlan("init", expand_path(init_checkpoint))
    return CheckpointPlan("fresh", None)


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
    # Missing flags in deterministic checkpoints retain the original behavior.
    for key, default in (
        ("uncertainty_aware", False), ("uncertainty_loss_weight", 1.0),
        ("uncertainty_loss", "smooth_l1"),
        ("amp_dtype", "float16"),
        ("curriculum_training", False),
    ):
        if key == "amp_dtype" and not getattr(args, "amp", False):
            continue
        if key in {"uncertainty_loss_weight", "uncertainty_loss"} and not getattr(
            args, "uncertainty_aware", False
        ):
            continue
        old_value = checkpoint_args.get(key, default)
        new_value = getattr(args, key, default)
        if old_value != new_value:
            mismatches.append(f"{key}: checkpoint={old_value!r}, current={new_value!r}")
    if getattr(args, "curriculum_training", False):
        for key in ("variance_ramp_start_epoch", "variance_ramp_epochs",
                    "rollout_ramp_start_epoch", "rollout_ramp_epochs",
                    "transition_ema_half_life", "transition_rate_window",
                    "transition_rate_threshold", "transition_patience"):
            if checkpoint_args.get(key) != getattr(args, key):
                mismatches.append(f"{key}: checkpoint={checkpoint_args.get(key)!r}, current={getattr(args, key)!r}")
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
    split: Dict,
) -> Tuple[List[int], List[int], List[str], List[str]]:
    """Select training and validation windows from the authoritative split."""
    train_runs = set(split["train_runs"])
    val_runs = set(split["val_runs"])
    test_runs = set(split["test_runs"])

    train_idx = []
    val_idx = []

    for i, (_, run_name, _) in enumerate(dataset.samples):
        if run_name in train_runs:
            train_idx.append(i)
        elif run_name in val_runs:
            val_idx.append(i)
        elif run_name not in test_runs:
            raise ValueError(f"Dataset run {run_name!r} is absent from {FIXED_SPLIT_PATH}.")

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


def _apply_training_visible_counts(
    patterns: Sequence[str],
    counts: Sequence[Optional[int]],
    fractions: List[float],
    *,
    spatial_sites: int,
    time_steps: int,
    modality: str,
) -> None:
    """Overwrite requested mask fractions from exact training-time counts."""
    for i, (pattern, count) in enumerate(zip(patterns, counts)):
        if count is None:
            continue
        if pattern in PROBE_COUNT_PATTERNS:
            maximum = spatial_sites
            label = "spatial sites"
        else:
            maximum = time_steps
            label = "time steps"
        if not (0 <= count <= maximum):
            raise ValueError(
                f"{modality} visible count {count} is outside [0, {maximum}] "
                f"{label} for pattern {pattern}."
            )
        fractions[i] = 1.0 - count / maximum


def check_stage1(condition: torch.Tensor, message: str, synchronize: bool = True) -> None:
    """Stop all training ranks before an invalid optimizer update."""
    bad = (~condition).to(dtype=torch.int32)
    if synchronize and distributed_is_initialized():
        dist.all_reduce(bad, op=dist.ReduceOp.MAX)
    if bad.item():
        raise FloatingPointError(message)


def cosine_continuation(step: int, start: float, duration: float) -> float:
    """Smooth continuation at successful-optimizer-update resolution."""
    u = min(max((step - start) / duration, 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * u))


def arm_curriculum_ramp(state: Dict[str, Any], prefix: str) -> int:
    """Arm one immutable ramp at the current successful optimizer step."""
    if prefix not in ("variance", "rollout"):
        raise ValueError(f"Unknown curriculum ramp: {prefix}")
    if state[f"{prefix}_ramp_armed"]:
        raise ValueError(f"The {prefix} ramp is already armed; its start is immutable")
    if prefix == "rollout" and (not state["variance_ramp_armed"] or
            state["successful_optimizer_steps"] < state["variance_ramp_start_step"] + state["variance_ramp_steps"] or
            curriculum_weights(state)[0] != 1.0):
        raise ValueError("Rollout arming requires a fully completed variance ramp (lambda_v == 1)")
    start = state["successful_optimizer_steps"]
    state[f"{prefix}_ramp_start_step"] = start
    state[f"{prefix}_ramp_armed"] = True
    return start


def apply_ready_curriculum_transition(
    state: Dict[str, Any], diagnostic: Optional[Dict[str, Any]],
) -> Optional[Tuple[str, int]]:
    """Apply the ramp named by an existing readiness detector result."""
    if diagnostic is None:
        return None
    ready = [prefix for prefix, key in (
        ("variance", "ready_for_variance_ramp"),
        ("rollout", "ready_for_rollout_ramp"),
    ) if diagnostic.get(key, False)]
    if len(ready) > 1:
        raise ValueError("A convergence diagnostic cannot arm both curriculum ramps")
    if not ready:
        return None
    prefix = ready[0]
    return prefix, arm_curriculum_ramp(state, prefix)


def make_curriculum_state(args, updates_per_epoch: int, checkpoint=None) -> Dict[str, Any]:
    """Restore immutable ramps, optionally arming one at the saved update count.

    Missing start epochs mean unarmed. Explicit starts and older checkpoints
    retain the original prescheduled curriculum behavior.
    """
    if updates_per_epoch < 1:
        raise ValueError("Curriculum requires at least one training update per epoch")
    state = {"updates_per_epoch": updates_per_epoch}
    for prefix in ("variance", "rollout"):
        start_epoch = getattr(args, f"{prefix}_ramp_start_epoch", None)
        state[f"{prefix}_ramp_start_step"] = start_epoch * updates_per_epoch if start_epoch is not None else None
        state[f"{prefix}_ramp_armed"] = start_epoch is not None
        state[f"{prefix}_ramp_steps"] = getattr(args, f"{prefix}_ramp_epochs") * updates_per_epoch
        if not math.isfinite(state[f"{prefix}_ramp_steps"]) or state[f"{prefix}_ramp_steps"] <= 0:
            raise ValueError("Curriculum ramp durations must be positive")
    state["successful_optimizer_steps"] = 0
    if checkpoint is not None:
        saved = checkpoint.get("curriculum_state")
        if saved is None:
            raise ValueError("Curriculum resume requires saved schedule and successful optimizer step")
        for key in ("updates_per_epoch", "variance_ramp_steps", "rollout_ramp_steps"):
            if state[key] != saved.get(key):
                raise ValueError(f"Curriculum resume changed {key}: {saved.get(key)!r} -> {state[key]!r}")
        count = saved.get("successful_optimizer_steps")
        if not isinstance(count, int) or count < 0:
            raise ValueError("Invalid saved curriculum successful optimizer step")
        state["successful_optimizer_steps"] = count
        for prefix in ("variance", "rollout"):
            key = f"{prefix}_ramp_start_step"
            start = saved.get(key)
            armed = saved.get(f"{prefix}_ramp_armed", start is not None)
            if not isinstance(armed, bool) or armed != (start is not None):
                raise ValueError(f"Invalid saved {prefix} armed/start state")
            if start is not None and (not math.isfinite(start) or start < 0):
                raise ValueError(f"Invalid saved {prefix} ramp start")
            if state[key] is not None and state[key] != start:
                raise ValueError(f"Curriculum resume changed {key}: {start!r} -> {state[key]!r}")
            state[key], state[f"{prefix}_ramp_armed"] = start, armed

    # Validate the restored state before accepting any manual transition.
    if state["rollout_ramp_armed"] and (not state["variance_ramp_armed"] or
            state["rollout_ramp_start_step"] < state["variance_ramp_start_step"] + state["variance_ramp_steps"]):
        raise ValueError("Curriculum ramps must not overlap; rollout requires an armed variance ramp")
    for prefix in ("variance", "rollout"):
        if not getattr(args, f"arm_{prefix}_ramp", False):
            continue
        if checkpoint is None:
            raise ValueError("Ramp arming requires resuming an existing curriculum checkpoint")
        arm_curriculum_ramp(state, prefix)
    return state


def curriculum_weights(state: Dict[str, Any]) -> Tuple[float, float]:
    return tuple(cosine_continuation(state["successful_optimizer_steps"],
                                    state[f"{prefix}_ramp_start_step"], state[f"{prefix}_ramp_steps"])
                 if state.get(f"{prefix}_ramp_armed", state[f"{prefix}_ramp_start_step"] is not None) else 0.0
                 for prefix in ("variance", "rollout"))


def curriculum_is_complete(state: Dict[str, Any]) -> bool:
    steps = state["successful_optimizer_steps"]
    ramps_complete = all(
        state[f"{prefix}_ramp_armed"]
        and state[f"{prefix}_ramp_start_step"] is not None
        and steps >= state[f"{prefix}_ramp_start_step"] + state[f"{prefix}_ramp_steps"]
        for prefix in ("variance", "rollout")
    )
    return ramps_complete and curriculum_weights(state) == (1.0, 1.0)


def require_complete_curriculum(state: Dict[str, Any], epoch: int) -> None:
    if curriculum_is_complete(state):
        return
    lambda_v, lambda_r = curriculum_weights(state)
    raise RuntimeError(
        f"Predetermined final epoch {epoch} reached before the curriculum completed: "
        f"lambda_v={lambda_v:g}, lambda_r={lambda_r:g}. latest.pt and history.json "
        "were saved, but final.pt was not created."
    )


def positive_metric_convergence(
    history, metric="val_mse", half_life=5.0, window=5, threshold=0.005, patience=4,
    segment_start_epoch=None,
):
    """Reconstruct log-EMA diagnostics; never use with signed NLL.

    Initialize m at the first log-loss in the selected segment. Callers can
    select a new segment after an LR regime change or a completed lambda ramp.
    A slope needs K+1 consecutive valid epochs; gaps and invalid values reset it.
    Readiness burn-in uses elapsed epochs since initialization (first epoch = 0).
    """
    if (not math.isfinite(half_life) or half_life <= 0 or window < 1 or patience < 1
            or not math.isfinite(threshold) or threshold < 0):
        raise ValueError("Invalid positive-metric convergence settings")
    alpha = 1.0 - 2.0 ** (-1.0 / half_life)
    logs, diagnostics = [], []
    stable_count, previous_epoch = 0, None
    for index, row in enumerate(history):
        epoch = row.get("epoch", index + 1)
        value = row.get(metric)
        if previous_epoch is not None and epoch != previous_epoch + 1:
            logs, stable_count = [], 0
        previous_epoch = epoch
        ema, rate = None, None
        if ((segment_start_epoch is not None and epoch < segment_start_epoch)
                or value is None or not math.isfinite(value) or value <= 0):
            logs, stable_count = [], 0
        else:
            m = math.log(value)
            if logs:
                m = alpha * m + (1.0 - alpha) * logs[-1]
            logs.append(m)
            ema = math.exp(m)
            if len(logs) > window:
                delta = (m - logs[-window - 1]) / window
                # An extreme finite loss ratio must not stop training via this monitor.
                try:
                    rate = math.expm1(delta)
                except OverflowError:
                    rate = None
                stable_count = stable_count + 1 if rate is not None and abs(rate) < threshold else 0
        segment_age = len(logs) - 1 if logs else None
        diagnostics.append(dict(epoch=epoch, ema_value=ema, smoothed_rate_per_epoch=rate,
                                stable_count=stable_count, converged=stable_count >= patience,
                                segment_age=segment_age, minimum_segment_age=2.0 * half_life,
                                burn_in_complete=segment_age is not None and segment_age >= 2.0 * half_life))
    return diagnostics


def phase_a_convergence(history, args, curriculum_state):
    """Detect Phase-A readiness in the current contiguous fixed-LR regime."""
    def at_floor(row):
        lr = row.get("learning_rate")
        return bool(row["epoch"] >= args.epochs and lr is not None and math.isfinite(lr)
                    and math.isclose(lr, args.min_lr, rel_tol=1e-9, abs_tol=0.0))

    segment_start = None
    for row in history:
        if not at_floor(row):
            segment_start = None
        elif segment_start is None:
            segment_start = row["epoch"]
    diagnostics = positive_metric_convergence(
        history, half_life=getattr(args, "transition_ema_half_life", 5.0),
        window=getattr(args, "transition_rate_window", 5),
        threshold=getattr(args, "transition_rate_threshold", 0.005),
        patience=getattr(args, "transition_patience", 4),
        segment_start_epoch=segment_start if segment_start is not None else history[-1]["epoch"] + 1)[-1]
    phase_a = bool(getattr(args, "curriculum_training", False) and curriculum_state is not None
                   and not curriculum_state.get("variance_ramp_armed", curriculum_state.get("variance_ramp_start_step") is not None)
                   and not curriculum_state.get("rollout_ramp_armed", curriculum_state.get("rollout_ramp_start_step") is not None)
                   and curriculum_weights(curriculum_state) == (0.0, 0.0))
    floor_reached = at_floor(history[-1])
    diagnostics["ema_val_mse"] = diagnostics.pop("ema_value")
    diagnostics.update(phase_a=phase_a, lr_floor_reached=floor_reached, segment_start_epoch=segment_start,
                       threshold=getattr(args, "transition_rate_threshold", 0.005),
                       patience_target=getattr(args, "transition_patience", 4),
                       ready_for_variance_ramp=phase_a and floor_reached and diagnostics["converged"]
                       and diagnostics["burn_in_complete"])
    return diagnostics


def nll_metric_convergence(history, half_life=5.0, window=5, segment_start_epoch=None):
    """Direct signed-NLL EMA; expm1 of its K-epoch slope measures exp(NLL) change."""
    if not math.isfinite(half_life) or half_life <= 0 or window < 1:
        raise ValueError("Invalid NLL convergence settings")
    alpha = 1.0 - 2.0 ** (-1.0 / half_life)
    values, diagnostics = [], []
    previous_epoch = None
    for index, row in enumerate(history):
        epoch = row.get("epoch", index + 1)
        if previous_epoch is not None and epoch != previous_epoch + 1:
            values = []
        previous_epoch = epoch
        value = row.get("val_gaussian_nll")
        ema, rate = None, None
        if ((segment_start_epoch is not None and epoch < segment_start_epoch)
                or value is None or not math.isfinite(value)):
            values = []
        else:
            ema = alpha * value + (1.0 - alpha) * values[-1] if values else value
            values.append(ema)
            if len(values) > window:
                try:
                    rate = math.expm1((ema - values[-window - 1]) / window)
                except OverflowError:
                    rate = None
        age = len(values) - 1 if values else None
        diagnostics.append(dict(epoch=epoch, ema_value=ema, smoothed_rate_per_epoch=rate,
                                segment_age=age, burn_in_complete=age is not None and age >= 2.0 * half_life))
    return diagnostics


def phase_b_convergence(history, args, curriculum_state):
    """Detect joint Phase-B stability after variance-ramp completion."""
    state = curriculum_state or {}
    start = state.get("variance_ramp_start_step")
    duration = state.get("variance_ramp_steps")
    variance_armed = state.get("variance_ramp_armed", start is not None)
    rollout_armed = state.get("rollout_ramp_armed", state.get("rollout_ramp_start_step") is not None)
    completed = bool(variance_armed and start is not None and duration is not None and duration > 0
                     and state.get("successful_optimizer_steps", -1) >= start + duration)
    candidate = bool(getattr(args, "curriculum_training", False) and completed and not rollout_armed
                     and curriculum_weights(state)[1] == 0.0)
    segment_start = None
    if candidate:
        for row in history:
            count = row.get("curriculum", {}).get("successful_optimizer_steps", -1)
            if count < start + duration:
                segment_start = None
            elif segment_start is None:
                segment_start = row["epoch"]
    half_life = getattr(args, "transition_ema_half_life", 5.0)
    window = getattr(args, "transition_rate_window", 5)
    threshold = getattr(args, "transition_rate_threshold", 0.005)
    patience = getattr(args, "transition_patience", 4)
    cutoff = segment_start if segment_start is not None else history[-1]["epoch"] + 1
    mse_rows = positive_metric_convergence(history, half_life=half_life, window=window,
                                          threshold=threshold, patience=patience, segment_start_epoch=cutoff)
    nll_rows = nll_metric_convergence(history, half_life=half_life, window=window, segment_start_epoch=cutoff)
    stable_count, previous_epoch = 0, None
    for mse, nll in zip(mse_rows, nll_rows):
        if previous_epoch is not None and mse["epoch"] != previous_epoch + 1:
            stable_count = 0
        previous_epoch = mse["epoch"]
        burn_in_complete = mse["burn_in_complete"] and nll["burn_in_complete"]
        rates = (mse["smoothed_rate_per_epoch"], nll["smoothed_rate_per_epoch"])
        stable = burn_in_complete and all(rate is not None and math.isfinite(rate) and abs(rate) < threshold for rate in rates)
        stable_count = stable_count + 1 if stable else 0
    ages = (mse["segment_age"], nll["segment_age"])
    phase_b = candidate and segment_start is not None
    return dict(epoch=history[-1]["epoch"], phase_b=phase_b, variance_ramp_complete=completed,
                segment_start_epoch=segment_start, ema_val_mse=mse["ema_value"], ema_val_nll=nll["ema_value"],
                mse_rate_per_epoch=rates[0], nll_rate_per_epoch=rates[1], stable_count=stable_count,
                segment_age=min(ages) if all(age is not None for age in ages) else None,
                minimum_segment_age=2.0 * half_life, burn_in_complete=burn_in_complete,
                threshold=threshold, patience_target=patience,
                ready_for_rollout_ramp=phase_b and burn_in_complete and stable_count >= patience)


def sample_training_mask(y, pattern_weights, density_p_zero, density_p_full,
                         magnetic_p_zero, magnetic_p_full):
    """Shared existing mask sampling for the ordinary pass and rollout pass."""
    batch_size, _, time_steps, nx, nz = y.shape
    spatial_sites = nx * nz
    magnetic_patterns = sample_patterns(pattern_weights, batch_size)
    density_patterns = sample_patterns(pattern_weights, batch_size)
    magnetic_fractions = torch.rand(batch_size).tolist()
    density_fractions = torch.rand(batch_size).tolist()
    density_counts = sample_training_severity_counts(
        density_patterns, spatial_sites=spatial_sites, time_steps=time_steps,
        p_zero=density_p_zero, p_full=density_p_full)
    magnetic_counts = sample_training_severity_counts(
        magnetic_patterns, spatial_sites=spatial_sites, time_steps=time_steps,
        p_zero=magnetic_p_zero, p_full=magnetic_p_full)
    for patterns, counts, fractions, modality in (
        (density_patterns, density_counts, density_fractions, "Density"),
        (magnetic_patterns, magnetic_counts, magnetic_fractions, "magnetic"),
    ):
        _apply_training_visible_counts(patterns, counts, fractions,
                                       spatial_sites=spatial_sites, time_steps=time_steps, modality=modality)
    mask, infos = sample_independent_batch_masks(
        y.shape, magnetic_patterns=magnetic_patterns, density_patterns=density_patterns,
        magnetic_mask_fractions=magnetic_fractions, density_mask_fractions=density_fractions,
        device=y.device, dtype=y.dtype, density_probe_counts=density_counts,
        magnetic_visible_counts=magnetic_counts)
    return (mask, infos, magnetic_patterns, density_patterns, magnetic_fractions,
            density_fractions, magnetic_counts, density_counts)


def curriculum_forward_loss(model, model_input, target, mask, lambda_v, lambda_r, reuse_mask=None):
    """One/two passes with the original balanced reconstruction MSE and FP32 NLL."""
    pred, variance = model(model_input, detach_variance_features=True)
    mu_loss = full_mse_loss(pred, target, mask)
    var_loss = (predicted_error_mse_loss(variance, pred, target, "gaussian_nll")
                if lambda_v > 0 else mu_loss.new_zeros(()))
    check_stage1(torch.isfinite(pred).all() & torch.isfinite(variance).all() & (variance > 0).all()
                 & torch.isfinite(mu_loss) & (mu_loss <= 100) & torch.isfinite(var_loss),
                 "Invalid curriculum first-pass prediction/variance/loss or catastrophic reconstruction MSE (>100)")
    loss = mu_loss + lambda_v * var_loss
    components = {"L_mu_1": mu_loss, "L_var_1": var_loss, "forwards": 1}
    if lambda_r > 0:
        if reuse_mask is None:
            raise ValueError("Positive rollout weight requires a new mask")
        second_input, exact_mask = make_rollout_input(target, pred, variance, mask, reuse_mask)
        pred2, variance2 = model(second_input, detach_variance_features=True)
        mu_loss2 = full_mse_loss(pred2, target, exact_mask)
        var_loss2 = predicted_error_mse_loss(variance2, pred2, target, "gaussian_nll")
        check_stage1(torch.isfinite(pred2).all() & torch.isfinite(variance2).all() & (variance2 > 0).all()
                     & torch.isfinite(mu_loss2) & (mu_loss2 <= 100) & torch.isfinite(var_loss2),
                     "Invalid curriculum second-pass prediction/variance/loss or catastrophic reconstruction MSE (>100)")
        loss = loss + lambda_r * (mu_loss2 + lambda_v * var_loss2)
        components.update(L_mu_2=mu_loss2, L_var_2=var_loss2, forwards=2)
    components["total_loss"] = loss
    return loss, pred, variance, components


def variance_statistics(totals, samples):
    """Exact streaming means/fractions; quantiles from 64 fixed sites/window/mask."""
    quantiles = np.quantile(samples, [0.5, 0.9, 0.99], axis=1)
    result = {}
    for i, channel in enumerate(("Bx", "By", "Bz", "Density")):
        count = totals[i, 1]
        result[channel] = dict(
            count=int(count), mean=float(totals[i, 0] / count),
            median=float(quantiles[0, i]), p90=float(quantiles[1, i]), p99=float(quantiles[2, i]),
            fraction_near_floor=float(totals[i, 2] / count),
            fraction_gt_one=float(totals[i, 3] / count),
            near_floor_threshold=10 * GAUSSIAN_NLL_VARIANCE_FLOOR,
            quantile_sample_count=int(samples.shape[1]),
            quantile_method="approximate: up to 64 evenly spaced voxels per window and mask",
        )
    return result


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    pattern_weights: Dict[str, float],
    density_visible_p_zero: float,
    density_visible_p_full: float,
    magnetic_visible_p_zero: float,
    magnetic_visible_p_full: float,
    amp: bool,
    grad_clip: float,
    postfix_every: int = 50,
    epoch_number: Optional[int] = None,
    epoch_progress: Optional[Any] = None,
    uncertainty_aware: bool = False,
    uncertainty_loss_weight: float = 1.0,
    scaler: Optional[Any] = None,
    amp_dtype: torch.dtype = torch.float16,
    uncertainty_loss: str = "smooth_l1",
    curriculum_state: Optional[Dict[str, Any]] = None,
    curriculum_log_every: int = 50,
) -> Dict[str, Any]:
    model.train()
    stage1 = uncertainty_aware and uncertainty_loss == "gaussian_nll"
    uncertainty_key = "gaussian_nll" if stage1 else "uncertainty_mse"

    n_patterns = len(MASK_PATTERNS)

    # Accumulate on device and read back once per epoch. Training patterns are
    # independent marginals, so no ambiguous 5x5 combination loss is logged.
    sum_mse = torch.zeros((), device=device)
    sum_uncertainty_mse = torch.zeros((), device=device)
    magnetic_pattern_counts = torch.zeros(n_patterns, device=device)
    density_pattern_counts = torch.zeros(n_patterns, device=device)
    sum_actual_fraction = torch.zeros((), device=device)

    sum_target_fraction = 0.0
    count_samples = 0
    total_batches = 0
    density_probe_sum = 0
    density_probe_sample_count = 0
    density_zero_sample_count = 0
    density_full_sample_count = 0
    magnetic_visible_sum = 0
    magnetic_sample_count = 0
    magnetic_zero_sample_count = 0
    magnetic_full_sample_count = 0
    spatial_sites = 0

    if scaler is None:
        scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda" and amp_dtype == torch.float16))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    epoch_started = time.perf_counter()
    last_batch_end = epoch_started
    data_wait_seconds = 0.0
    applied_steps = 0
    beta_sample_counts = {}
    scale_start = scaler.get_scale()
    gradient_norm_sum = torch.zeros((), device=device, dtype=torch.float64)
    gradient_norm_max = torch.zeros_like(gradient_norm_sum)
    curriculum_totals = (torch.zeros(8, device=device, dtype=torch.float64)
                         if curriculum_state is not None else None)
    lambda_start = lambda_end = None

    def count_optimizer_step(_optimizer, _args, _kwargs):
        nonlocal applied_steps
        applied_steps += 1

    step_counter = optimizer.register_step_post_hook(count_optimizer_step)

    batches_per_epoch = max(len(loader), 1)

    for step, batch in enumerate(loader):
        data_wait_seconds += time.perf_counter() - last_batch_end
        for beta in batch.get("metadata", {}).get("beta", []):
            key = str(float(beta))
            beta_sample_counts[key] = beta_sample_counts.get(key, 0) + 1
        y = batch["block"].to(device, non_blocking=True)
        y = normalize(y, mean, std)

        batch_size = y.shape[0]
        time_steps = int(y.shape[-3])
        spatial_sites = int(y.shape[-2] * y.shape[-1])

        (mask, mask_infos, magnetic_patterns, density_patterns, magnetic_fractions,
         density_fractions, magnetic_visible_counts, density_visible_counts) = sample_training_mask(
            y, pattern_weights, density_visible_p_zero, density_visible_p_full,
            magnetic_visible_p_zero, magnetic_visible_p_full)
        for pattern, count in zip(density_patterns, density_visible_counts):
            if pattern not in PROBE_COUNT_PATTERNS:
                continue
            density_probe_sum += int(count)
            density_probe_sample_count += 1
            density_zero_sample_count += int(count == 0)
            density_full_sample_count += int(count == spatial_sites)
        for pattern, count in zip(magnetic_patterns, magnetic_visible_counts):
            if pattern not in PROBE_COUNT_PATTERNS:
                continue
            magnetic_visible_sum += int(count)
            magnetic_sample_count += 1
            magnetic_zero_sample_count += int(count == 0)
            magnetic_full_sample_count += int(count == spatial_sites)
        if uncertainty_aware:
            input_variance = 1.0 - mask
            if stage1:
                check_stage1(((input_variance == 0) | (input_variance == 1)).all(),
                             f"Nonbinary input variance: epoch {epoch_number}, batch {step + 1}")
            model_input = torch.cat(
                [make_uncertain_input(y, input_variance), input_variance], dim=1
            )
        else:
            x_visible = make_visible_input(y, mask)
            model_input = torch.cat([x_visible, mask], dim=1)  # (B, 8, T, X, Z)

        optimizer.zero_grad(set_to_none=True)
        if stage1:
            check_stage1(torch.isfinite(model_input).all() & torch.isfinite(y).all(),
                         f"Nonfinite input/target: epoch {epoch_number}, batch {step + 1}")

        if curriculum_state is not None:
            lambda_v, lambda_r = curriculum_weights(curriculum_state)
            lambda_start = lambda_start or (lambda_v, lambda_r)
            lambda_end = (lambda_v, lambda_r)
            # Do not even sample a rollout mask until the second pass is active.
            reuse_mask = (sample_training_mask(
                y, pattern_weights, density_visible_p_zero, density_visible_p_full,
                magnetic_visible_p_zero, magnetic_visible_p_full)[0] if lambda_r > 0 else None)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda"), dtype=amp_dtype):
            if curriculum_state is not None:
                loss, pred, output_variance, components = curriculum_forward_loss(
                    model, model_input, y, mask, lambda_v, lambda_r, reuse_mask)
                reconstruction_loss, variance_loss = components["L_mu_1"], components["L_var_1"]
            else:
                pred = model(model_input)
                if uncertainty_aware:
                    pred, output_variance = pred
                reconstruction_loss = full_mse_loss(pred, y, mask)
                loss = reconstruction_loss
                if uncertainty_aware:
                    variance_loss = predicted_error_mse_loss(
                        output_variance, pred, y, objective=uncertainty_loss
                    ) if stage1 else predicted_error_mse_loss(output_variance, pred, y)
                    loss = loss + uncertainty_loss_weight * variance_loss

        if stage1:
            check_stage1(torch.isfinite(pred).all() & torch.isfinite(output_variance).all()
                         & (output_variance > 0).all() & torch.isfinite(loss)
                         & torch.isfinite(reconstruction_loss) & torch.isfinite(variance_loss)
                         & (reconstruction_loss <= 100),
                         f"Invalid prediction/variance/loss or catastrophic reconstruction MSE (>100): "
                         f"epoch {epoch_number}, batch {step + 1}, metadata={batch.get('metadata')}")

        scaler.scale(loss).backward()

        if stage1 or (grad_clip is not None and grad_clip > 0):
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip if grad_clip and grad_clip > 0 else float("inf"))
            if stage1:
                check_stage1(torch.isfinite(gradient_norm),
                             f"Nonfinite synchronized preclip gradient norm: epoch {epoch_number}, batch {step + 1}")
                gradient_norm_sum += gradient_norm.detach().double()
                gradient_norm_max = torch.maximum(gradient_norm_max, gradient_norm.detach().double())

        applied_before = applied_steps
        scaler.step(optimizer)
        scaler.update()

        if curriculum_state is not None:
            applied = applied_steps - applied_before
            if distributed_is_initialized():
                bounds = torch.tensor([applied, -applied], device=device)
                dist.all_reduce(bounds, op=dist.ReduceOp.MAX)
                if bounds[0].item() != -bounds[1].item():
                    raise RuntimeError("Curriculum optimizer updates disagree across ranks; stopping")
            coordinate = curriculum_state["successful_optimizer_steps"]
            curriculum_state["successful_optimizer_steps"] += applied
            with torch.no_grad():
                curriculum_totals += torch.stack([
                    components["L_mu_1"], components["L_var_1"],
                    components.get("L_mu_2", loss.new_zeros(())),
                    components.get("L_var_2", loss.new_zeros(())), loss,
                    loss.new_tensor(lambda_v), loss.new_tensor(lambda_r),
                    loss.new_tensor(components["forwards"] == 2),
                ]).detach().double()
            if step % curriculum_log_every == 0 or step + 1 == len(loader):
                region = "A" if lambda_v == 0 else ("B" if lambda_r == 0 else "C")
                message = (f"Curriculum {region}: successful_step={coordinate} "
                           f"next_successful_step={curriculum_state['successful_optimizer_steps']} "
                           f"lambda_v={lambda_v:.9g} lambda_r={lambda_r:.9g} "
                           f"forwards={components['forwards']} applied={applied} "
                           + " ".join(f"{k}={float(v.detach()):.8g}" for k, v in components.items() if k != "forwards"))
                if epoch_progress is not None:
                    epoch_progress.write(message)
                else:
                    rank_print(message)

        with torch.no_grad():
            magnetic_pattern_ids = torch.tensor(
                [PATTERN_TO_ID[name] for name in magnetic_patterns],
                device=device,
            )
            density_pattern_ids = torch.tensor(
                [PATTERN_TO_ID[name] for name in density_patterns],
                device=device,
            )
            per_sample_actual = 1.0 - mask.flatten(1).mean(1)
            ones = torch.ones(batch_size, device=device)
            magnetic_pattern_counts.index_add_(0, magnetic_pattern_ids, ones)
            density_pattern_counts.index_add_(0, density_pattern_ids, ones)
            sum_actual_fraction += per_sample_actual.sum()

            sum_mse += reconstruction_loss.detach().float()
            if uncertainty_aware:
                sum_uncertainty_mse += variance_loss.detach().float()

        # temporal_block samples two boundaries directly, so its requested
        # random fraction is intentionally ignored; use the realized fraction
        # for that modality when reporting the effective masking target.
        for i, info in enumerate(mask_infos):
            magnetic_target = (
                info["magnetic_actual_mask_fraction"]
                if magnetic_patterns[i] == "temporal_block"
                else magnetic_fractions[i]
            )
            density_target = (
                info["density_actual_mask_fraction"]
                if density_patterns[i] == "temporal_block"
                else density_fractions[i]
            )
            sum_target_fraction += (
                3.0 * magnetic_target + density_target
            ) / 4.0
        count_samples += batch_size
        total_batches += 1

        if epoch_progress is not None:
            epoch_progress.update(1.0 / batches_per_epoch)

        if epoch_progress is not None and step % postfix_every == 0:
            epoch_progress.set_postfix(
                phase="train",
                mse=float(sum_mse) / total_batches,
            )
        last_batch_end = time.perf_counter()

    step_counter.remove()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - epoch_started

    if epoch_progress is not None and epoch_number is not None:
        # Eliminate floating-point accumulation drift at every epoch boundary.
        epoch_progress.n = float(epoch_number)
        epoch_progress.refresh()

    totals = torch.tensor(
        [
            float(sum_mse),
            float(total_batches),
            float(sum_target_fraction),
            float(count_samples),
            float(density_probe_sum),
            float(density_probe_sample_count),
            float(density_zero_sample_count),
            float(density_full_sample_count),
            float(magnetic_visible_sum),
            float(magnetic_sample_count),
            float(magnetic_zero_sample_count),
            float(magnetic_full_sample_count),
            float(sum_uncertainty_mse),
            float(applied_steps),
        ],
        dtype=torch.float64,
        device=device,
    )
    if distributed_is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        dist.all_reduce(magnetic_pattern_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(density_pattern_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_actual_fraction, op=dist.ReduceOp.SUM)

    (
        global_mse,
        global_batches,
        global_target,
        global_samples,
        global_probe_sum,
        global_probe_sample_count,
        global_density_zero_sample_count,
        global_density_full_sample_count,
        global_magnetic_visible_sum,
        global_magnetic_sample_count,
        global_magnetic_zero_sample_count,
        global_magnetic_full_sample_count,
        global_uncertainty_mse,
        global_applied_steps,
    ) = totals.tolist()
    total_count = max(global_samples, 1)
    probe_count = max(global_probe_sample_count, 1)
    mean_density_probes = global_probe_sum / probe_count
    magnetic_count = max(global_magnetic_sample_count, 1)
    mean_magnetic_visible = global_magnetic_visible_sum / magnetic_count

    metrics = {
        "mse": global_mse / max(global_batches, 1),
        "target_mask_fraction": global_target / total_count,
        "actual_mask_fraction": float(sum_actual_fraction.cpu()) / total_count,
        "density_probe_count": mean_density_probes,
        "density_visible_fraction": mean_density_probes / max(spatial_sites, 1),
        "density_zero_sample_share": global_density_zero_sample_count / probe_count,
        "density_full_sample_share": global_density_full_sample_count / probe_count,
        "magnetic_visible_count": mean_magnetic_visible,
        "magnetic_visible_fraction": mean_magnetic_visible / max(spatial_sites, 1),
        "magnetic_zero_sample_share": (
            global_magnetic_zero_sample_count / magnetic_count
        ),
        "magnetic_full_sample_share": (
            global_magnetic_full_sample_count / magnetic_count
        ),
        "magnetic_pattern_share": {
            name: float(magnetic_pattern_counts[i].cpu()) / total_count
            for i, name in enumerate(MASK_PATTERNS)
        },
        "density_pattern_share": {
            name: float(density_pattern_counts[i].cpu()) / total_count
            for i, name in enumerate(MASK_PATTERNS)
        },
        "optimizer_steps": global_applied_steps,
        "amp_skipped_steps": global_batches - global_applied_steps,
        "amp_scale_start": scale_start,
        "amp_scale_end": scaler.get_scale(),
        "train_seconds": train_seconds,
        "data_wait_seconds": data_wait_seconds,
    }
    if uncertainty_aware:
        metrics[uncertainty_key] = global_uncertainty_mse / max(global_batches, 1)
    if curriculum_state is not None:
        if distributed_is_initialized():
            dist.all_reduce(curriculum_totals, op=dist.ReduceOp.SUM)
        names = ("L_mu_1", "L_var_1", "L_mu_2", "L_var_2", "total_loss",
                 "lambda_v_mean", "lambda_r_mean", "two_forward_fraction")
        metrics["curriculum"] = dict(zip(names, (curriculum_totals / max(global_batches, 1)).cpu().tolist()))
        metrics["curriculum"].update(lambda_v_start=lambda_start[0], lambda_r_start=lambda_start[1],
                                     lambda_v_end=lambda_end[0], lambda_r_end=lambda_end[1],
                                     successful_optimizer_steps=curriculum_state["successful_optimizer_steps"])
    if stage1:
        if distributed_is_initialized():
            dist.all_reduce(gradient_norm_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(gradient_norm_max, op=dist.ReduceOp.MAX)
        metrics.update(gradient_norm_mean=float(gradient_norm_sum.cpu()) / max(global_batches, 1),
                       gradient_norm_max=float(gradient_norm_max.cpu()),
                       binary_input_variance_verified=True, nonfinite_events=0,
                       optimizer_updates=global_applied_steps / (dist.get_world_size() if distributed_is_initialized() else 1))
    if distributed_is_initialized():
        rank_counts = [None] * dist.get_world_size()
        dist.all_gather_object(rank_counts, beta_sample_counts)
        beta_sample_counts = {}
        for counts in rank_counts:
            for beta, count in counts.items():
                beta_sample_counts[beta] = beta_sample_counts.get(beta, 0) + count
    metrics["beta_sample_counts"] = beta_sample_counts
    return metrics


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
    uncertainty_aware: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    uncertainty_loss: str = "smooth_l1",
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate every mask pattern on the same validation blocks.

    All five patterns use deterministic ~50% paper-style layouts with an
    identical Bx/By/Bz/Density observation mask. Training samplers are
    unchanged. The loader is iterated once and each block is reused for all
    patterns.
    """
    model.eval()
    stage1 = uncertainty_aware and uncertainty_loss == "gaussian_nll"
    uncertainty_key = "gaussian_nll" if stage1 else "uncertainty_mse"
    beta_details = {}
    _ = (mask_seed, density_probe_min, density_probe_max)

    sums = torch.zeros(len(patterns), device=device)
    uncertainty_sums = torch.zeros_like(sums)
    counts = torch.zeros(len(patterns), device=device)

    if epoch_progress is not None:
        epoch_progress.set_postfix(phase="val")

    for batch in loader:
        y = batch["block"].to(device, non_blocking=True)
        y = normalize(y, mean, std)

        batch_size = y.shape[0]

        for i, pattern in enumerate(patterns):
            mask, _ = make_fixed_validation_mask(
                pattern,
                y.shape,
                device=device,
                dtype=y.dtype,
            )

            if uncertainty_aware:
                input_variance = 1.0 - mask
                if stage1:
                    check_stage1(((input_variance == 0) | (input_variance == 1)).all(),
                                 "Nonbinary validation input variance", synchronize=False)
                model_input = torch.cat(
                    [make_uncertain_input(y, input_variance), input_variance], dim=1
                )
            else:
                model_input = torch.cat([make_visible_input(y, mask), mask], dim=1)

            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda"), dtype=amp_dtype):
                pred = model(model_input)
                if uncertainty_aware:
                    pred, output_variance = pred
                loss_mse = full_mse_loss(pred, y, mask)
                if uncertainty_aware:
                    variance_loss = (predicted_error_mse_loss(output_variance, pred, y, objective=uncertainty_loss)
                                     if stage1 else predicted_error_mse_loss(output_variance, pred, y))

            if stage1:
                # Validation shards have unequal lengths: no per-batch collectives.
                check_stage1(torch.isfinite(model_input).all() & torch.isfinite(y).all()
                             & torch.isfinite(pred).all() & torch.isfinite(output_variance).all()
                             & (output_variance > 0).all() & torch.isfinite(variance_loss)
                             & torch.isfinite(loss_mse) & (loss_mse <= 100),
                             f"Invalid validation values or catastrophic reconstruction MSE (>100): {batch.get('metadata')}",
                             synchronize=False)
                betas = batch.get("metadata", {}).get("beta", ["all"] * batch_size)
                names = [str(float(b)) if b != "all" else "all" for b in betas]
                for beta in sorted(set(names)):
                    keep = torch.tensor([name == beta for name in names], device=device)
                    v = output_variance[keep].detach().float()
                    detail = beta_details.setdefault(beta, dict(
                        totals=torch.zeros(4, 4, device=device, dtype=torch.float64), samples=[],
                        reconstruction=torch.zeros(len(patterns), device=device, dtype=torch.float64),
                        counts=torch.zeros(len(patterns), device=device, dtype=torch.float64)))
                    n = v.shape[0]
                    detail["reconstruction"][i] += full_mse_loss(pred[keep], y[keep], mask[keep]).double() * n
                    detail["counts"][i] += n
                    axes = (0, 2, 3, 4)
                    detail["totals"][:, 0] += v.sum(axes, dtype=torch.float64)
                    detail["totals"][:, 1] += v[:, 0].numel()
                    detail["totals"][:, 2] += (v <= 10 * GAUSSIAN_NLL_VARIANCE_FLOOR).sum(axes)
                    detail["totals"][:, 3] += (v > 1).sum(axes)
                    flat = v.flatten(2)
                    sites = torch.linspace(0, flat.shape[2] - 1, min(64, flat.shape[2]), device=device).long()
                    detail["samples"].append(flat[:, :, sites].transpose(0, 1).reshape(4, -1).cpu().numpy())

            sums[i] += loss_mse.detach().float() * batch_size
            counts[i] += batch_size
            if uncertainty_aware:
                uncertainty_sums[i] += variance_loss.detach().float() * batch_size

    if distributed_is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if uncertainty_aware:
            dist.all_reduce(uncertainty_sums, op=dist.ReduceOp.SUM)

    sums = sums.cpu()
    counts = counts.cpu().clamp(min=1.0)

    metrics = {
        pattern: {
            "mse": float(sums[i] / counts[i]),
        }
        for i, pattern in enumerate(patterns)
    }

    metrics["mean"] = {
        "mse": sum(metrics[pattern]["mse"] for pattern in patterns) / len(patterns),
    }
    if uncertainty_aware:
        uncertainty_sums = uncertainty_sums.cpu()
        for i, pattern in enumerate(patterns):
            metrics[pattern][uncertainty_key] = float(uncertainty_sums[i] / counts[i])
        metrics["mean"][uncertainty_key] = sum(
            metrics[p][uncertainty_key] for p in patterns
        ) / len(patterns)

    if stage1:
        local = {beta: {key: (np.concatenate(value, axis=1) if key == "samples" else value.cpu().numpy())
                        for key, value in detail.items()} for beta, detail in beta_details.items()}
        gathered = [local]
        if distributed_is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
        combined = {}
        for shard in gathered:
            for beta, detail in shard.items():
                if beta not in combined:
                    combined[beta] = detail
                else:
                    for key in detail:
                        combined[beta][key] = (np.concatenate([combined[beta][key], detail[key]], axis=1)
                                               if key == "samples" else combined[beta][key] + detail[key])
        metrics["per_beta"] = {
            beta: dict(mse=float(np.mean(detail["reconstruction"] / detail["counts"])),
                       variance=variance_statistics(detail["totals"], detail["samples"]))
            for beta, detail in combined.items() if beta != "all"}
        metrics["variance_statistics"] = variance_statistics(
            sum(d["totals"] for d in combined.values()),
            np.concatenate([d["samples"] for d in combined.values()], axis=1))

    return metrics


def build_epoch_wandb_log(
    row: Dict[str, Any],
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Dict[str, float]],
    val_patterns: Sequence[str],
) -> Dict[str, Any]:
    """Assemble the per-epoch W&B payload. MSE is the only reconstruction metric."""
    log_data = {
        "epoch": row["epoch"],
        "train/mse": row["train_mse"],
        "val/mse": row["val_mse"],
        "val/mean/mse": row["val_mse"],
        "mask/target_fraction": row["train_mask_target_fraction"],
        "mask/actual_fraction": row["train_mask_actual_fraction"],
        "mask/density_probe_patterns/probe_count": row[
            "train_density_probe_count"
        ],
        "mask/density_probe_patterns/visible_fraction": row[
            "train_density_visible_fraction"
        ],
        "mask/density_probe_patterns/zero_sample_share": row[
            "train_density_zero_sample_share"
        ],
        "mask/density_probe_patterns/full_sample_share": row[
            "train_density_full_sample_share"
        ],
        "mask/magnetic_probe_patterns/visible_count": row[
            "train_magnetic_visible_count"
        ],
        "mask/magnetic_probe_patterns/visible_fraction": row[
            "train_magnetic_visible_fraction"
        ],
        "mask/magnetic_probe_patterns/zero_sample_share": row[
            "train_magnetic_zero_sample_share"
        ],
        "mask/magnetic_probe_patterns/full_sample_share": row[
            "train_magnetic_full_sample_share"
        ],
        "optimization/learning_rate": row["learning_rate"],
    }

    for pattern in MASK_PATTERNS:
        log_data[f"mask/magnetic_pattern_share/{pattern}"] = (
            train_metrics["magnetic_pattern_share"][pattern]
        )
        log_data[f"mask/density_pattern_share/{pattern}"] = (
            train_metrics["density_pattern_share"][pattern]
        )

    for pattern in val_patterns:
        log_data[f"val/{pattern}/mse"] = val_metrics[pattern]["mse"]
    if "train_gaussian_nll" in row:
        log_data["train/gaussian_nll"] = row["train_gaussian_nll"]
        log_data["val/gaussian_nll"] = row["val_gaussian_nll"]
    elif "train_uncertainty_mse" in row:
        log_data["train/uncertainty_smooth_l1"] = row["train_uncertainty_mse"]
        log_data["val/uncertainty_smooth_l1"] = row["val_uncertainty_mse"]
    if "curriculum" in row:
        log_data.update({f"curriculum/{key}": value for key, value in row["curriculum"].items()})
        log_data["curriculum/ready_for_variance_ramp"] = False
        log_data["curriculum/ready_for_rollout_ramp"] = False
    if "phase_convergence" in row:
        log_data.update({f"phase_convergence/{key}": value for key, value in row["phase_convergence"].items()
                         if value is not None})
        diagnostic = row["phase_convergence"]
        if diagnostic["smoothed_rate_per_epoch"] is not None:
            log_data["curriculum/mse_rate_per_epoch"] = diagnostic["smoothed_rate_per_epoch"]
        log_data.update({
            "curriculum/convergence_stable_count": diagnostic["stable_count"],
            "curriculum/convergence_segment_age": diagnostic["segment_age"],
            "curriculum/ready_for_variance_ramp": diagnostic["ready_for_variance_ramp"],
        })
    if "phase_b_convergence" in row:
        log_data.update({f"phase_b_convergence/{key}": value for key, value in row["phase_b_convergence"].items()
                         if value is not None})
        diagnostic = row["phase_b_convergence"]
        if diagnostic["mse_rate_per_epoch"] is not None:
            log_data["curriculum/mse_rate_per_epoch"] = diagnostic["mse_rate_per_epoch"]
        if diagnostic["nll_rate_per_epoch"] is not None:
            log_data["curriculum/nll_rate_per_epoch"] = diagnostic["nll_rate_per_epoch"]
        log_data.update({
            "curriculum/convergence_stable_count": diagnostic["stable_count"],
            "curriculum/convergence_segment_age": diagnostic["segment_age"],
            "curriculum/ready_for_rollout_ramp": diagnostic["ready_for_rollout_ramp"],
        })

    return log_data


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args,
    stats: Dict[str, List[float]],
    wandb_run_id: Optional[str] = None,
    scaler: Optional[Any] = None,
    curriculum_state: Optional[Dict[str, Any]] = None,
) -> None:
    model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
    ckpt = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "stats": stats,
        "wandb_run_id": wandb_run_id,
        "split_path": str(FIXED_SPLIT_PATH),
        "split_sha256": fixed_split_sha256(),
    }
    if scaler is not None and scaler.is_enabled():
        ckpt["scaler"] = scaler.state_dict()
    if curriculum_state is not None:
        ckpt["curriculum_state"] = dict(curriculum_state)
        ckpt["curriculum_state"].update(zip(("lambda_v", "lambda_r"), curriculum_weights(curriculum_state)))
        ckpt["lr_schedule"] = dict(horizon_epochs=args.epochs, warmup_epochs=args.warmup_epochs,
                                   base_lr=args.lr, min_lr=args.min_lr,
                                   current_lr=[group["lr"] for group in optimizer.param_groups])
    torch.save(ckpt, path)


def main():
    args = parse_args()
    device, rank, world_size, local_rank = initialize_distributed()
    set_seed(args.seed + rank)

    automatic_out_dir = Path("runs") / default_run_name(args)
    out_dir = expand_path(args.out_dir or str(automatic_out_dir))
    if (args.arm_variance_ramp or args.arm_rollout_ramp) and not (out_dir / "latest.pt").is_file():
        raise ValueError("Ramp arming requires an existing out-dir/latest.pt curriculum checkpoint")
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)
    if distributed_is_initialized():
        dist.barrier()

    h5_dir = expand_path(args.h5_dir)
    split = load_fixed_split()
    split_metadata = fixed_split_metadata()
    run_lengths = h5_run_lengths(h5_dir)
    split_runs = set().union(*(set(split[key]) for key in FIXED_SPLIT_COUNTS))
    if set(run_lengths) != split_runs:
        raise ValueError(
            f"{FIXED_SPLIT_PATH} does not exactly cover the HDF5 runs: "
            f"missing={sorted(set(run_lengths) - split_runs)[:5]}, "
            f"unknown={sorted(split_runs - set(run_lengths))[:5]}."
        )

    rank_print("HDF5 dir:", h5_dir)
    rank_print("Authoritative split:", split_metadata)
    rank_print("Output dir:", out_dir)
    rank_print("Betas:", args.betas)
    rank_print("Model version:", args.model_version)
    rank_print(
        "Training spatial_grid/spatial_random severity: 0 / log-uniform / full "
        "mixture with independent B and Density endpoint probabilities "
        f"(Density p_zero={args.density_visible_p_zero:g}, "
        f"p_full={args.density_visible_p_full:g}; "
        f"B p_zero={args.magnetic_visible_p_zero:g}, "
        f"p_full={args.magnetic_visible_p_full:g}). "
        "spatial_block keeps Uniform(0, 1) area; temporal patterns draw a "
        "uniform visible-frame count in [0, T]. Controlled validation uses "
        "fixed ~50% layouts for all five patterns, with identical "
        "Bx/By/Bz/Density observation masks."
    )
    rank_print(
        "Learning-rate schedule: "
        f"linear warmup {args.warmup_epochs} epochs, "
        f"base_lr={args.lr:g}, cosine min_lr={args.min_lr:g}, "
        f"total_epochs={args.epochs}"
    )
    rank_print("Mask pattern weights:", args.mask_pattern_weights)
    rank_print("Validation mask patterns:", args.val_patterns)
    if args.uncertainty_aware:
        rank_print("Uncertainty objective:", args.uncertainty_loss)
        if args.uncertainty_loss == "gaussian_nll":
            input_label = "Curriculum first pass" if args.curriculum_training else "Stage-1"
            rank_print(f"{input_label}: binary input variance checked every batch; FP32 NLL floor={GAUSSIAN_NLL_VARIANCE_FLOOR:g}; no upper cap")
    if args.curriculum_training:
        rank_print("Curriculum: binary first pass; detached variance features; direct, unclipped rollout reuse; "
                   "no calibration scale. Validation retains the standard single-pass benchmark.")
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
        include_runs=list(split["train_runs"]) + list(split["val_runs"]),
    )

    train_idx, val_idx, train_runs, val_runs = split_by_run(dataset, split)
    selected_betas = {float(beta) for beta in args.betas}
    selected_train_runs = {
        name for name in train_runs if float(parse_run_name(name)["beta"]) in selected_betas
    }
    usable_train_runs = {dataset.samples[index][1] for index in train_idx}
    expected_usable_train_runs = {
        name for name in selected_train_runs if run_lengths[name] >= args.delta_t
    }
    if usable_train_runs != expected_usable_train_runs:
        raise ValueError("Usable fixed-split training runs do not match the HDF5 window index.")

    rank_print("Total windows:", len(dataset))
    rank_print("Train windows:", len(train_idx))
    rank_print("Val windows:", len(val_idx))
    rank_print("Declared train runs:", len(train_runs))
    rank_print("Usable train runs:", len(usable_train_runs))
    rank_print("Val runs:", len({dataset.samples[index][1] for index in val_idx}))
    rank_print("Test runs loaded by training: 0")

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
    checkpoint_plan = resolve_checkpoint_plan(
        out_dir,
        auto_resume=args.auto_resume,
        init_checkpoint=args.init_checkpoint,
    )
    if checkpoint_plan.mode == "resume":
        resume_checkpoint = torch.load(checkpoint_plan.path, map_location=device)
        validate_checkpoint_split(resume_checkpoint)
        validate_resume_parameters(args, resume_checkpoint.get("args", {}))
        rank_print(
            f"Auto-resuming from {checkpoint_plan.path} "
            f"(completed epoch {resume_checkpoint['epoch']})"
        )
    elif checkpoint_plan.mode == "init":
        if checkpoint_plan.path is None or not checkpoint_plan.path.is_file():
            raise FileNotFoundError(
                f"Initialization checkpoint not found: {checkpoint_plan.path}"
            )
        init_checkpoint = torch.load(checkpoint_plan.path, map_location=device)
        rank_print(
            f"Initializing model weights from {checkpoint_plan.path} "
            f"(source epoch {init_checkpoint['epoch']}); "
            "optimizer, scheduler, and epoch counter start fresh. "
            "Masking-version mismatch is allowed for this weight-only init."
        )

    if resume_checkpoint is not None:
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
        architecture=args.model_version,
        use_attention=args.use_attention,
        spatial_only_pooling=args.spatial_only_pooling,
        uncertainty_aware=args.uncertainty_aware,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    rank_print(f"Model parameters: {n_params / 1e6:.3f} M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda" and args.amp_dtype == "float16"))

    start_epoch = 1
    curriculum_state = (make_curriculum_state(args, len(train_loader), resume_checkpoint)
                        if args.curriculum_training else None)
    if curriculum_state is not None:
        rank_print("Curriculum schedule (successful optimizer updates):", curriculum_state)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if resume_checkpoint.get("scaler"):
            scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
    elif init_checkpoint is not None:
        # Allow deterministic weights to initialize the extended head. Only
        # its new parameters retain their initialization; all old keys stay strict.
        source_state = dict(init_checkpoint["model"])
        if args.uncertainty_aware and not init_checkpoint.get("args", {}).get(
            "uncertainty_aware", False
        ):
            for key, value in model.state_dict().items():
                if key.startswith("variance_out."):
                    source_state[key] = value
        source_uses_attention = bool(
            init_checkpoint.get("args", {}).get("use_attention", False)
        )
        if source_uses_attention == args.use_attention:
            model.load_state_dict(source_state)
        else:
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
                "loaded compatible convolutional weights."
            )
        rank_print("Initialization uses a fresh optimizer and LR schedule.")

    if distributed_is_initialized():
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.curriculum_training,
        )

    if is_main_process():
        with open(out_dir / "config.json", "w") as f:
            json.dump({**vars(args), "data_split": split_metadata}, f, indent=2)

    final_epoch = args.stop_after_epochs or args.epochs
    if (args.arm_variance_ramp or args.arm_rollout_ramp) and start_epoch > final_epoch:
        raise ValueError("Arming requires --stop-after-epochs beyond the completed checkpoint epoch")
    if start_epoch > final_epoch:
        rank_print(
            f"Training is already complete at epoch {start_epoch - 1}; "
            f"requested final epoch={final_epoch}. Extend --stop-after-epochs to continue curriculum training."
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
                "num_test_runs_used_during_training": 0,
                "data_split": split_metadata,
                "channel_mean": stats["mean"],
                "channel_std": stats["std"],
            },
            allow_val_change=True,
        )
        wandb_run.define_metric("epoch")
        wandb_run.define_metric("train/*", step_metric="epoch")
        wandb_run.define_metric("val/*", step_metric="epoch")
        wandb_run.define_metric("mask/*", step_metric="epoch")
        wandb_run.define_metric("optimization/*", step_metric="epoch")
        wandb_run.define_metric("curriculum/*", step_metric="epoch")
        wandb_run.define_metric("phase_convergence/*", step_metric="epoch")
        wandb_run.define_metric("phase_b_convergence/*", step_metric="epoch")

    history = []
    history_path = out_dir / "history.json"
    if is_main_process() and resume_checkpoint is not None and history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        history = [row for row in history if row.get("epoch", 0) < start_epoch]

    epoch_progress = None
    if is_main_process():
        epoch_progress = tqdm(
            total=float(final_epoch),
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

    for epoch in range(start_epoch, final_epoch + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        epoch_lr = learning_rate_for_epoch(
            epoch=min(epoch, args.epochs) if args.curriculum_training else epoch,
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
            density_visible_p_zero=args.density_visible_p_zero,
            density_visible_p_full=args.density_visible_p_full,
            magnetic_visible_p_zero=args.magnetic_visible_p_zero,
            magnetic_visible_p_full=args.magnetic_visible_p_full,
            amp=args.amp,
            grad_clip=args.grad_clip,
            epoch_number=epoch,
            epoch_progress=epoch_progress,
            uncertainty_aware=args.uncertainty_aware,
            uncertainty_loss_weight=args.uncertainty_loss_weight,
            scaler=scaler,
            amp_dtype=getattr(torch, args.amp_dtype),
            uncertainty_loss=args.uncertainty_loss,
            curriculum_state=curriculum_state,
            curriculum_log_every=args.curriculum_log_every,
        )

        validation_started = time.perf_counter()
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
            uncertainty_aware=args.uncertainty_aware,
            amp_dtype=getattr(torch, args.amp_dtype),
            uncertainty_loss=args.uncertainty_loss,
        )
        validation_seconds = time.perf_counter() - validation_started

        row = {
            "epoch": epoch,
            "learning_rate": epoch_lr,
            "train_mse": train_metrics["mse"],
            "val_mse": val_metrics["mean"]["mse"],
            "train_mask_target_fraction": train_metrics["target_mask_fraction"],
            "train_mask_actual_fraction": train_metrics["actual_mask_fraction"],
            "train_density_probe_count": train_metrics[
                "density_probe_count"
            ],
            "train_density_visible_fraction": train_metrics[
                "density_visible_fraction"
            ],
            "train_density_zero_sample_share": train_metrics[
                "density_zero_sample_share"
            ],
            "train_density_full_sample_share": train_metrics[
                "density_full_sample_share"
            ],
            "train_magnetic_visible_count": train_metrics[
                "magnetic_visible_count"
            ],
            "train_magnetic_visible_fraction": train_metrics[
                "magnetic_visible_fraction"
            ],
            "train_magnetic_zero_sample_share": train_metrics[
                "magnetic_zero_sample_share"
            ],
            "train_magnetic_full_sample_share": train_metrics[
                "magnetic_full_sample_share"
            ],
            "train_magnetic_pattern_share": train_metrics[
                "magnetic_pattern_share"
            ],
            "train_density_pattern_share": train_metrics[
                "density_pattern_share"
            ],
            "val_per_pattern": {
                pattern: val_metrics[pattern] for pattern in args.val_patterns
            },
        }
        row.update({key: train_metrics[key] for key in (
            "optimizer_steps", "amp_skipped_steps", "amp_scale_start", "amp_scale_end",
            "beta_sample_counts",
        )})
        if curriculum_state is not None:
            row["curriculum"] = train_metrics["curriculum"]
            lambda_v_end = row["curriculum"]["lambda_v_end"]
            lambda_r_end = row["curriculum"]["lambda_r_end"]
            phase_index = 0 if lambda_v_end == 0 else (1 if lambda_r_end == 0 else 2)
            row["curriculum"].update(
                phase_index=phase_index,
                phase_name=("reconstruction_only", "variance", "rollout")[phase_index],
                convergence_rate_threshold=args.transition_rate_threshold,
                convergence_patience_target=args.transition_patience,
            )
            epoch_log("  curriculum (second-pass losses averaged over all batches, inactive = 0): "
                      + json.dumps(row["curriculum"]))
        if args.uncertainty_aware:
            row["uncertainty_objective"] = args.uncertainty_loss
            key = "gaussian_nll" if args.uncertainty_loss == "gaussian_nll" else "uncertainty_mse"
            row[f"train_{key}"] = train_metrics[key]
            row[f"val_{key}"] = val_metrics["mean"][key]
            if args.uncertainty_loss == "gaussian_nll":
                row.update({key: train_metrics[key] for key in (
                    "gradient_norm_mean", "gradient_norm_max", "binary_input_variance_verified",
                    "nonfinite_events", "optimizer_updates")})
                row["val_variance_statistics"] = val_metrics["variance_statistics"]
                row["val_per_beta"] = val_metrics["per_beta"]
        transition_diagnostic = None
        if is_main_process():
            history.append(row)
            if curriculum_state is not None:
                diagnostic = phase_a_convergence(history, args, curriculum_state)
                if diagnostic["phase_a"]:
                    row["phase_convergence"] = diagnostic
                    transition_diagnostic = diagnostic
                    ema = diagnostic["ema_val_mse"]
                    rate = diagnostic["smoothed_rate_per_epoch"]
                    ema_text = f"{ema:.8g}" if ema is not None else "n/a"
                    rate_text = f"{100 * rate:+.3f}%" if rate is not None else "n/a"
                    epoch_log(f"Phase convergence: ema_val_mse={ema_text} smoothed_rate_per_epoch={rate_text} "
                              f"threshold={100 * args.transition_rate_threshold:.3f}% "
                              f"stable_count={diagnostic['stable_count']}/{args.transition_patience} "
                              f"segment_age={diagnostic['segment_age'] if diagnostic['segment_age'] is not None else 'n/a'}"
                              f"/{diagnostic['minimum_segment_age']:g} "
                              f"lr_floor_reached={'yes' if diagnostic['lr_floor_reached'] else 'no'} "
                              f"ready_for_variance_ramp={'yes' if diagnostic['ready_for_variance_ramp'] else 'no'}")
                    if diagnostic["ready_for_variance_ramp"]:
                        epoch_log("READY_FOR_VARIANCE_RAMP=true")
                else:
                    diagnostic = phase_b_convergence(history, args, curriculum_state)
                    if diagnostic["phase_b"]:
                        row["phase_b_convergence"] = diagnostic
                        transition_diagnostic = diagnostic
                        fields = []
                        for name in ("ema_val_mse", "mse_rate_per_epoch", "ema_val_nll", "nll_rate_per_epoch"):
                            value = diagnostic[name]
                            text = (f"{100 * value:+.3f}%" if "rate" in name else f"{value:.8g}") if value is not None else "n/a"
                            fields.append(f"{name}={text}")
                        epoch_log("Phase B convergence: " + " ".join(fields)
                                  + f" threshold={100 * args.transition_rate_threshold:.3f}% "
                                  f"stable_count={diagnostic['stable_count']}/{args.transition_patience} "
                                  f"segment_age={diagnostic['segment_age'] if diagnostic['segment_age'] is not None else 'n/a'}"
                                  f"/{diagnostic['minimum_segment_age']:g} "
                                  f"ready_for_rollout_ramp={'yes' if diagnostic['ready_for_rollout_ramp'] else 'no'}")
                        if diagnostic["ready_for_rollout_ramp"]:
                            epoch_log("READY_FOR_ROLLOUT_RAMP=true")

        if curriculum_state is not None:
            synchronized = [transition_diagnostic]
            if distributed_is_initialized():
                dist.broadcast_object_list(synchronized, src=0)
            transition_diagnostic = synchronized[0]
            transition = apply_ready_curriculum_transition(curriculum_state, transition_diagnostic)
            if transition is not None:
                ramp, start_step = transition
                if is_main_process():
                    row["automatic_curriculum_transition"] = {
                        "ramp": ramp,
                        "start_step": start_step,
                    }
                    epoch_log(f"Automatically armed {ramp} ramp at successful optimizer step {start_step}")

        if wandb_run is not None:
            wandb_run.log(
                build_epoch_wandb_log(
                    row=row,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    val_patterns=args.val_patterns,
                )
            )

        epoch_log(
            f"Epoch {epoch}/{final_epoch if args.curriculum_training else args.epochs}  lr={epoch_lr:.8g}  "
            f"train_mse={row['train_mse']:.6f} "
            f"val_mse={row['val_mse']:.6f} "
            f"density_probes="
            f"{row['train_density_probe_count']:.2f} "
            f"D_zero_share="
            f"{row['train_density_zero_sample_share']:.3f} "
            f"D_full_share="
            f"{row['train_density_full_sample_share']:.3f} "
            f"B_visible={row['train_magnetic_visible_fraction']:.3f} "
            f"B_zero_share="
            f"{row['train_magnetic_zero_sample_share']:.3f} "
            f"B_full_share="
            f"{row['train_magnetic_full_sample_share']:.3f}"
        )
        epoch_log(
            "  val by pattern: "
            + "  ".join(
                f"{pattern}={val_metrics[pattern]['mse']:.6f}"
                for pattern in args.val_patterns
            )
        )

        if is_main_process():
            save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                stats=stats,
                wandb_run_id=wandb_run.id if wandb_run is not None else None,
                scaler=scaler,
                curriculum_state=curriculum_state,
            )
        if distributed_is_initialized():
            dist.barrier()

        performance = torch.tensor([
            time.perf_counter() - epoch_started, train_metrics["train_seconds"],
            train_metrics["data_wait_seconds"], validation_seconds,
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0,
            torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else 0,
        ], dtype=torch.float64, device=device)
        if distributed_is_initialized():
            dist.all_reduce(performance, op=dist.ReduceOp.MAX)
        row.update(zip(("epoch_seconds", "train_seconds", "data_wait_seconds",
                        "validation_seconds", "peak_allocated_gib", "peak_reserved_gib"),
                       performance.cpu().tolist()))
        if args.uncertainty_aware:
            label = "Gaussian NLL" if args.uncertainty_loss == "gaussian_nll" else "uncertainty Smooth L1"
            epoch_log(f"  {label}: train={row[f'train_{key}']:.6f} val={row[f'val_{key}']:.6f}")
            if args.uncertainty_loss == "gaussian_nll":
                epoch_log(f"  synchronized preclip gradient norm: mean={row['gradient_norm_mean']:.6g} "
                          f"max={row['gradient_norm_max']:.6g}; binary v_in verified; nonfinite_events=0")
                epoch_log("  validation variance statistics: " + json.dumps(row["val_variance_statistics"]))
                epoch_log("  validation per beta: " + json.dumps(row["val_per_beta"]))
        epoch_log(f"  epoch_seconds={row['epoch_seconds']:.2f} "
                  f"AMP skipped rank-steps={row['amp_skipped_steps']:.0f} "
                  f"scale={row['amp_scale_start']:g}->{row['amp_scale_end']:g}")
        if is_main_process():
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
        if epoch == args.final_epoch:
            if distributed_is_initialized():
                dist.barrier()
            if curriculum_state is not None:
                require_complete_curriculum(curriculum_state, epoch)
            if is_main_process():
                save_checkpoint(
                    out_dir / "final.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    args=args,
                    stats=stats,
                    wandb_run_id=wandb_run.id if wandb_run is not None else None,
                    scaler=scaler,
                    curriculum_state=curriculum_state,
                )
                epoch_log(f"Saved predetermined final checkpoint at epoch {epoch}")
            if distributed_is_initialized():
                dist.barrier()

    if epoch_progress is not None:
        epoch_progress.close()
    if args.stop_after_epochs is not None:
        rank_print(f"Completed bounded training through epoch {final_epoch}; "
                   f"LR horizon remains {args.epochs}; no further optimizer updates.")

    if wandb_run is not None:
        wandb_run.summary["final_epoch"] = args.final_epoch
        if args.wandb_log_model and (out_dir / "final.pt").is_file():
            import wandb

            artifact = wandb.Artifact(
                name=f"{wandb_run.id}-final-model",
                type="model",
                metadata={"final_epoch": args.final_epoch, "split_sha256": args.split_sha256},
            )
            artifact.add_file(str(out_dir / "final.pt"))
            wandb_run.log_artifact(artifact)
        wandb_run.finish()

    if distributed_is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
