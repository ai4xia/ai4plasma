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
    TEMPORAL_EXTRAPOLATION_PATTERN,
    TEMPORAL_MASK_PATTERNS,
    full_mse_loss,
    make_visible_input,
    make_uncertain_input,
    predicted_error_mse_loss,
    GAUSSIAN_NLL_VARIANCE_FLOOR,
    parse_pattern_weights,
    sample_independent_batch_masks,
    sample_patterns,
    sample_training_severity_counts,
    temporal_extrapolation_frames,
    make_fixed_validation_mask,
)
try:
    from data.masking import sample_training_patterns
except ImportError:
    sample_training_patterns = None
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


def parse_args(argv=None):
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
        "--uncertainty-loss", choices=(
            "smooth_l1", "gaussian_nll", "student_t_nll", "generalized_kl", "gkl",
        ),
        default="smooth_l1",
        help="Variance objective; NLL and GKL objectives use controlled binary-input checks.",
    )
    p.add_argument(
        "--student-t-df", type=float, default=3.0,
        help="Fixed Student-t degrees of freedom; must be > 2 (default: 3).",
    )
    p.add_argument(
        "--uncertainty-loss-weight", type=float, default=1.0,
        help="Weight of the selected variance objective (default: 1).",
    )
    p.add_argument("--curriculum-training", action="store_true",
                   help="Opt-in reconstruction / detached variance / two-pass curriculum.")
    p.add_argument(
        "--curriculum-config",
        choices=(
            "adaptive", "abcd_symmetric_lr", "global_cosine",
            "three_phase_global_cosine",
        ),
        default="adaptive",
        help="Curriculum control flow; the default preserves the existing adaptive run.",
    )
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
    p.add_argument(
        "--post-final-continuation",
        type=str,
        default=None,
        metavar="FINAL_CHECKPOINT",
        help=(
            "Start a separate fixed-LR full-objective continuation from a "
            "completed global-cosine final checkpoint, restoring "
            "optimizer state and holding LR at its saved floor. An in-run "
            "latest.pt still takes precedence for crash recovery."
        ),
    )
    p.add_argument(
        "--gaussian-u-continuation",
        action="store_true",
        help=(
            "Isolated diagnostic: continue a completed three-phase Student-t "
            "checkpoint for 20 epochs with the existing Gaussian NLL. Requires "
            "--post-final-continuation; all other continuation state is restored."
        ),
    )
    p.add_argument(
        "--temporal-extrapolation-continuation",
        action="store_true",
        help=(
            "Isolated diagnostic: replace the temporal_block training draw "
            "with a uniformly sampled visible-prefix/hidden-future-suffix task "
            "during a 20-epoch Student-t post-final continuation."
        ),
    )
    p.add_argument(
        "--temporal-extrapolation-only-continuation",
        action="store_true",
        help=(
            "Isolated diagnostic: replace all training masks with pure "
            "temporal extrapolation (100% exposure) during a 10-epoch "
            "Student-t post-final continuation from the epoch-276 checkpoint."
        ),
    )
    p.add_argument(
        "--uncertainty-feature-grad-continuation",
        action="store_true",
        help=(
            "Isolated 10-epoch continuation from the epoch-276 20% extrapolation "
            "checkpoint: allow uncertainty gradients into shared features, "
            "while retaining detached mean predictions in the NLL."
        ),
    )
    p.add_argument(
        "--joint-nll-grad-continuation",
        action="store_true",
        help=(
            "Isolated 10-epoch continuation from the epoch-276 20% extrapolation "
            "checkpoint: allow Student-t uncertainty gradients into both the mean head "
            "(Path A) and variance head/shared features (Path B)."
        ),
    )
    p.add_argument(
        "--joint-nll-grad",
        action="store_true",
        help=(
            "Enable joint Path A and Path B uncertainty gradients "
            "during curriculum training (Phase B and Phase C)."
        ),
    )
    p.add_argument(
        "--gkl-continuation-20ep",
        action="store_true",
        help=(
            "Isolated diagnostic: continue the completed three-phase joint A+B "
            "Student-t checkpoint for 20 epochs with Generalized KL divergence direct uncertainty "
            "supervision after a 2-epoch cosine transition from Student-t A+B. "
            "Requires --post-final-continuation; all other continuation state is restored."
        ),
    )
    p.add_argument(
        "--gkl-joint-grad-continuation-20ep",
        action="store_true",
        help=(
            "Isolated diagnostic: continue the completed three-phase joint A+B "
            "Student-t checkpoint for 20 epochs with Generalized KL divergence direct uncertainty "
            "supervision with both Path A and Path B enabled after a 2-epoch cosine transition. "
            "Requires --post-final-continuation; all other continuation state is restored."
        ),
    )
    p.add_argument(
        "--differentiable-phase-c-reuse",
        action="store_true",
        help="Ramp cross-forward prediction gradients with lambda_r for fresh GKL A+B "
             "or with alpha for its continuation.",
    )
    p.add_argument(
        "--mean-only-phase-c-reuse",
        action="store_true",
        help="Limit differentiable Phase-C reuse to mean predictions.",
    )
    p.add_argument(
        "--student-t-unnormalized-control-20ep",
        action="store_true",
        help=(
            "Isolated matched control: continue the completed three-phase joint A+B "
            "Student-t checkpoint for 20 epochs with unnormalized Student-t NLL. "
            "Requires --post-final-continuation; all other continuation state is restored."
        ),
    )
    p.add_argument(
        "--gkl-masking-continuation-20ep",
        action="store_true",
        help=(
            "Matched 20-epoch masking continuation: continue the completed "
            "fresh 256-epoch GKL A+B checkpoint for epochs 257-276 with settled "
            "Phase C (lambda_v=lambda_r=1), fixed LR=min_lr, Paths A and B ON, "
            "and fully differentiable Phase C reuse."
        ),
    )
    p.add_argument(
        "--phase-d-sliding-curriculum",
        action="store_true",
        help=(
            "Unified Phase C -> D bidirectional sliding-window curriculum: "
            "extends settled Phase C (slide_step=0) continuously to slide_step in "
            "{-24..24} via cosine lambda_slide with full differentiable reuse."
        ),
    )
    p.add_argument(
        "--slide-ramp-epochs",
        type=float,
        default=10.0,
        help="Epochs over which lambda_slide ramps from 0 to 1 via cosine schedule.",
    )
    p.add_argument(
        "--val-slide-steps",
        type=int,
        nargs="+",
        default=[0, 1, -1, 12, -12],
        help="Representative sliding steps evaluated on validation runs.",
    )

    args = p.parse_args(argv)
    if args.differentiable_phase_c_reuse and not (
        args.gkl_joint_grad_continuation_20ep
        or args.gkl_masking_continuation_20ep
        or args.phase_d_sliding_curriculum
        or (args.joint_nll_grad and args.uncertainty_loss in ("generalized_kl", "gkl"))
    ):
        p.error("--differentiable-phase-c-reuse requires fresh or continuation GKL A+B")
    if args.mean_only_phase_c_reuse and not (
        args.differentiable_phase_c_reuse and args.gkl_joint_grad_continuation_20ep
    ):
        p.error("--mean-only-phase-c-reuse requires differentiable GKL A+B continuation")

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
    if args.post_final_continuation is not None:
        if args.init_checkpoint is not None:
            p.error("--post-final-continuation and --init-checkpoint are mutually exclusive")
        if (not args.curriculum_training
                or args.curriculum_config not in (
                    "global_cosine", "three_phase_global_cosine",
                )):
            p.error(
                "--post-final-continuation requires the global_cosine or "
                "three_phase_global_cosine curriculum"
            )
        if (args.stop_after_epochs != args.final_epoch
                or args.final_epoch <= args.epochs):
            p.error(
                "--post-final-continuation requires matching --stop-after-epochs "
                "and --final-epoch beyond the completed --epochs horizon"
            )
        if args.arm_variance_ramp or args.arm_rollout_ramp:
            p.error("--post-final-continuation does not permit ramp arming")
    if args.gaussian_u_continuation:
        if args.post_final_continuation is None:
            p.error("--gaussian-u-continuation requires --post-final-continuation")
        if args.curriculum_config != "three_phase_global_cosine":
            p.error(
                "--gaussian-u-continuation requires three_phase_global_cosine"
            )
        if args.uncertainty_loss != "gaussian_nll":
            p.error("--gaussian-u-continuation requires --uncertainty-loss gaussian_nll")
    if args.uncertainty_feature_grad_continuation:
        if (not args.temporal_extrapolation_continuation
                or args.temporal_extrapolation_only_continuation
                or args.gaussian_u_continuation
                or args.joint_nll_grad_continuation
                or args.joint_nll_grad):
            p.error(
                "--uncertainty-feature-grad-continuation requires the existing "
                "--temporal-extrapolation-continuation 20% mask policy only"
            )
    if args.joint_nll_grad_continuation:
        if (not args.temporal_extrapolation_continuation
                or args.temporal_extrapolation_only_continuation
                or args.gaussian_u_continuation
                or args.uncertainty_feature_grad_continuation
                or args.joint_nll_grad):
            p.error(
                "--joint-nll-grad-continuation requires the existing "
                "--temporal-extrapolation-continuation 20% mask policy only"
            )
    if args.joint_nll_grad:
        if (not args.curriculum_training
                or args.curriculum_config != "three_phase_global_cosine"):
            p.error(
                "--joint-nll-grad requires --curriculum-training and "
                "--curriculum-config three_phase_global_cosine"
            )
        if (args.uncertainty_loss not in ("student_t_nll", "generalized_kl", "gkl")
                or (args.uncertainty_loss == "student_t_nll" and args.student_t_df != 3.0)):
            p.error(
                "--joint-nll-grad requires generalized_kl or student_t_nll with df=3"
            )
        if args.post_final_continuation is not None:
            p.error(
                "--joint-nll-grad is for from-scratch training and cannot be "
                "used with --post-final-continuation"
            )
        if args.init_checkpoint is not None and not args.phase_d_sliding_curriculum:
            p.error(
                "--joint-nll-grad is for from-scratch training and cannot be "
                "used with --init-checkpoint"
            )
        if (args.gaussian_u_continuation
                or args.temporal_extrapolation_continuation
                or args.temporal_extrapolation_only_continuation
                or args.uncertainty_feature_grad_continuation
                or args.joint_nll_grad_continuation
                or args.gkl_continuation_20ep
                or args.gkl_joint_grad_continuation_20ep
                or args.gkl_masking_continuation_20ep
                or args.student_t_unnormalized_control_20ep):
            p.error(
                "--joint-nll-grad cannot be combined with continuation diagnostic flags"
            )
    if args.gkl_continuation_20ep:
        if args.post_final_continuation is None:
            p.error("--gkl-continuation-20ep requires --post-final-continuation")
        if args.curriculum_config != "three_phase_global_cosine":
            p.error("--gkl-continuation-20ep requires three_phase_global_cosine")
        if args.uncertainty_loss not in ("generalized_kl", "gkl"):
            p.error("--gkl-continuation-20ep requires --uncertainty-loss generalized_kl")
        if (args.gaussian_u_continuation
                or args.temporal_extrapolation_continuation
                or args.temporal_extrapolation_only_continuation
                or args.uncertainty_feature_grad_continuation
                or args.joint_nll_grad_continuation
                or args.gkl_joint_grad_continuation_20ep
                or args.gkl_masking_continuation_20ep
                or args.student_t_unnormalized_control_20ep):
            p.error("--gkl-continuation-20ep cannot be combined with other continuation flags")
    if args.gkl_joint_grad_continuation_20ep:
        if args.post_final_continuation is None:
            p.error("--gkl-joint-grad-continuation-20ep requires --post-final-continuation")
        if args.curriculum_config != "three_phase_global_cosine":
            p.error("--gkl-joint-grad-continuation-20ep requires three_phase_global_cosine")
        if args.uncertainty_loss not in ("generalized_kl", "gkl"):
            p.error("--gkl-joint-grad-continuation-20ep requires --uncertainty-loss generalized_kl")
        if (args.gaussian_u_continuation
                or args.temporal_extrapolation_continuation
                or args.temporal_extrapolation_only_continuation
                or args.uncertainty_feature_grad_continuation
                or args.joint_nll_grad_continuation
                or args.gkl_continuation_20ep
                or args.gkl_masking_continuation_20ep
                or args.student_t_unnormalized_control_20ep):
            p.error("--gkl-joint-grad-continuation-20ep cannot be combined with other continuation flags")
    if args.gkl_masking_continuation_20ep:
        if args.post_final_continuation is None:
            p.error("--gkl-masking-continuation-20ep requires --post-final-continuation")
        if args.curriculum_config != "three_phase_global_cosine":
            p.error("--gkl-masking-continuation-20ep requires three_phase_global_cosine")
        if args.uncertainty_loss not in ("generalized_kl", "gkl"):
            p.error("--gkl-masking-continuation-20ep requires --uncertainty-loss generalized_kl")
        if not args.differentiable_phase_c_reuse:
            p.error("--gkl-masking-continuation-20ep requires --differentiable-phase-c-reuse")
        if (args.gaussian_u_continuation
                or args.temporal_extrapolation_continuation
                or args.temporal_extrapolation_only_continuation
                or args.uncertainty_feature_grad_continuation
                or args.joint_nll_grad_continuation
                or args.gkl_continuation_20ep
                or args.gkl_joint_grad_continuation_20ep
                or args.student_t_unnormalized_control_20ep):
            p.error("--gkl-masking-continuation-20ep cannot be combined with other continuation flags")
    if args.student_t_unnormalized_control_20ep:
        if args.post_final_continuation is None:
            p.error("--student-t-unnormalized-control-20ep requires --post-final-continuation")
        if args.curriculum_config != "three_phase_global_cosine":
            p.error("--student-t-unnormalized-control-20ep requires three_phase_global_cosine")
        if args.uncertainty_loss != "student_t_nll" or args.student_t_df != 3.0:
            p.error("--student-t-unnormalized-control-20ep requires --uncertainty-loss student_t_nll --student-t-df 3")
        if (args.gaussian_u_continuation
                or args.temporal_extrapolation_continuation
                or args.temporal_extrapolation_only_continuation
                or args.uncertainty_feature_grad_continuation
                or args.joint_nll_grad_continuation
                or args.gkl_continuation_20ep
                or args.gkl_joint_grad_continuation_20ep):
            p.error("--student-t-unnormalized-control-20ep cannot be combined with other continuation flags")
    if args.temporal_extrapolation_continuation:
        if args.post_final_continuation is None:
            p.error(
                "--temporal-extrapolation-continuation requires "
                "--post-final-continuation"
            )
        if args.curriculum_config != "three_phase_global_cosine":
            p.error(
                "--temporal-extrapolation-continuation requires "
                "three_phase_global_cosine"
            )
        if args.uncertainty_loss != "student_t_nll" or args.student_t_df != 3.0:
            p.error(
                "--temporal-extrapolation-continuation requires "
                "--uncertainty-loss student_t_nll --student-t-df 3"
            )
        if args.delta_t != 24:
            p.error("--temporal-extrapolation-continuation requires --delta-t 24")
    if args.temporal_extrapolation_only_continuation:
        if args.post_final_continuation is None:
            p.error(
                "--temporal-extrapolation-only-continuation requires "
                "--post-final-continuation"
            )
        if args.curriculum_config != "three_phase_global_cosine":
            p.error(
                "--temporal-extrapolation-only-continuation requires "
                "three_phase_global_cosine"
            )
        if args.uncertainty_loss != "student_t_nll" or args.student_t_df != 3.0:
            p.error(
                "--temporal-extrapolation-only-continuation requires "
                "--uncertainty-loss student_t_nll --student-t-df 3"
            )
        if args.delta_t != 24:
            p.error("--temporal-extrapolation-only-continuation requires --delta-t 24")
        if args.temporal_extrapolation_continuation:
            p.error(
                "--temporal-extrapolation-only-continuation and "
                "--temporal-extrapolation-continuation are mutually exclusive"
            )
    if (args.arm_variance_ramp or args.arm_rollout_ramp) and (not args.curriculum_training or not args.auto_resume):
        p.error("Ramp arming requires --curriculum-training and --auto-resume")
    if not math.isfinite(args.student_t_df) or args.student_t_df <= 2.0:
        p.error("--student-t-df must be finite and greater than 2")
    if args.uncertainty_loss in ("gaussian_nll", "student_t_nll", "generalized_kl", "gkl") and not args.uncertainty_aware:
        p.error(f"--uncertainty-loss {args.uncertainty_loss} requires --uncertainty-aware")
    if args.uncertainty_loss in ("gaussian_nll", "student_t_nll", "generalized_kl", "gkl"):
        args.variance_floor = GAUSSIAN_NLL_VARIANCE_FLOOR
    if args.curriculum_training:
        if (not args.uncertainty_aware
                or args.uncertainty_loss not in ("gaussian_nll", "student_t_nll", "generalized_kl", "gkl")):
            p.error(
                "--curriculum-training requires --uncertainty-aware and an NLL or GKL uncertainty loss"
            )
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
        if args.curriculum_config == "abcd_symmetric_lr":
            if (args.epochs, args.warmup_epochs) != (45, 5):
                p.error("abcd_symmetric_lr requires --epochs 45 --warmup-epochs 5")
            if not (math.isclose(args.lr, 2e-5) and math.isclose(args.min_lr, 2e-6)):
                p.error("abcd_symmetric_lr requires --lr 2e-5 --min-lr 2e-6")
            if (args.variance_ramp_epochs, args.rollout_ramp_epochs) != (40.0, 40.0):
                p.error("abcd_symmetric_lr requires 40-epoch variance and rollout ramps")
            if args.variance_ramp_start_epoch is not None or args.rollout_ramp_start_epoch is not None:
                p.error("abcd_symmetric_lr uses convergence-triggered ramps; scheduled starts are not supported")
            if args.final_epoch != 240:
                p.error("abcd_symmetric_lr requires --final-epoch 240")
        elif args.curriculum_config == "global_cosine":
            if args.warmup_epochs != 5 or args.epochs not in (240, 1000):
                p.error(
                    "global_cosine requires --warmup-epochs 5 and "
                    "--epochs 240 or 1000"
                )
            supported_lr_pairs = ((2e-4, 1e-6), (2e-4, 1e-5))
            if not any(
                math.isclose(args.lr, peak) and math.isclose(args.min_lr, floor)
                for peak, floor in supported_lr_pairs
            ):
                p.error(
                    "global_cosine requires --lr/--min-lr 2e-4/1e-6 "
                    "or 2e-4/1e-5"
                )
            if args.epochs == 1000 and not (
                math.isclose(args.lr, 2e-4)
                and math.isclose(args.min_lr, 1e-5)
            ):
                p.error(
                    "1000-epoch global_cosine requires --lr 2e-4 --min-lr 1e-5"
                )
            if (args.variance_ramp_epochs, args.rollout_ramp_epochs) != (40.0, 40.0):
                p.error("global_cosine requires 40-epoch variance and rollout ramps")
            if args.variance_ramp_start_epoch is not None or args.rollout_ramp_start_epoch is not None:
                p.error("global_cosine uses convergence-triggered ramps; scheduled starts are not supported")
            if args.post_final_continuation is None and args.final_epoch != args.epochs:
                p.error(
                    f"global_cosine requires --final-epoch {args.epochs}"
                )
        elif args.curriculum_config == "three_phase_global_cosine" and not args.phase_d_sliding_curriculum:
            if (args.epochs, args.warmup_epochs) != (256, 5):
                p.error(
                    "three_phase_global_cosine requires --epochs 256 "
                    "--warmup-epochs 5"
                )
            if not (
                math.isclose(args.lr, 2e-4)
                and math.isclose(args.min_lr, 1e-5)
            ):
                p.error(
                    "three_phase_global_cosine requires "
                    "--lr 2e-4 --min-lr 1e-5"
                )
            if (args.variance_ramp_epochs, args.rollout_ramp_epochs) != (40.0, 40.0):
                p.error(
                    "three_phase_global_cosine requires 40-epoch variance "
                    "and rollout ramps"
                )
            if (args.variance_ramp_start_epoch is not None
                    or args.rollout_ramp_start_epoch is not None):
                p.error(
                    "three_phase_global_cosine uses convergence-triggered ramps; "
                    "scheduled starts are not supported"
                )
            if args.post_final_continuation is None and args.final_epoch != 256:
                p.error("three_phase_global_cosine requires --final-epoch 256")
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
    if args.phase_d_sliding_curriculum:
        if not args.curriculum_training or args.curriculum_config != "three_phase_global_cosine":
            p.error("--phase-d-sliding-curriculum requires --curriculum-training --curriculum-config three_phase_global_cosine")
        if not args.uncertainty_aware or args.uncertainty_loss not in ("generalized_kl", "gkl"):
            p.error("--phase-d-sliding-curriculum requires --uncertainty-aware and --uncertainty-loss generalized_kl")
        if not args.joint_nll_grad or not args.differentiable_phase_c_reuse or args.mean_only_phase_c_reuse:
            p.error("--phase-d-sliding-curriculum requires --joint-nll-grad --differentiable-phase-c-reuse (full mean+var)")
        if args.delta_t != 24:
            p.error("--phase-d-sliding-curriculum requires --delta-t 24")
        if not math.isfinite(args.slide_ramp_epochs) or args.slide_ramp_epochs <= 0:
            p.error("--slide-ramp-epochs must be positive")
        for s_val in args.val_slide_steps:
            if not -args.delta_t <= int(s_val) <= args.delta_t:
                p.error(f"Invalid validation slide step: {s_val}")

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


PHASE_D_REHEAT_EPOCHS = 5
PHASE_D_COSINE_EPOCHS = 40


def symmetric_lr_cycle(
    epoch: int, start_epoch: int, min_lr: float, peak_lr: float,
) -> float:
    """Five inclusive warmup epochs, 40 inclusive cosine epochs, then floor."""
    if epoch < start_epoch:
        raise ValueError(f"LR cycle epoch {epoch} precedes start epoch {start_epoch}")
    if not (0 <= min_lr <= peak_lr):
        raise ValueError(f"Expected 0 <= min_lr <= peak_lr, got {min_lr} and {peak_lr}")
    reheat_end = start_epoch + PHASE_D_REHEAT_EPOCHS - 1
    if epoch <= reheat_end:
        progress = float(epoch - start_epoch) / (PHASE_D_REHEAT_EPOCHS - 1)
        return min_lr + (peak_lr - min_lr) * progress
    cosine_start = reheat_end + 1
    cosine_end = cosine_start + PHASE_D_COSINE_EPOCHS - 1
    if epoch >= cosine_end:
        return min_lr
    progress = float(epoch - cosine_start) / (PHASE_D_COSINE_EPOCHS - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def global_cosine_learning_rate(
    epoch: int, total_epochs: int, warmup_epochs: int,
    min_lr: float, peak_lr: float,
) -> float:
    """Inclusive low-to-peak warmup followed by one inclusive global cosine."""
    if not (1 <= epoch <= total_epochs):
        raise ValueError(f"epoch must lie in [1, {total_epochs}], got {epoch}")
    if not (1 < warmup_epochs < total_epochs - 1):
        raise ValueError("Global cosine requires at least two warmup and cosine epochs")
    if not (0 <= min_lr <= peak_lr):
        raise ValueError(f"Expected 0 <= min_lr <= peak_lr, got {min_lr} and {peak_lr}")
    if epoch <= warmup_epochs:
        progress = float(epoch - 1) / float(warmup_epochs - 1)
        return min_lr + (peak_lr - min_lr) * progress
    cosine_epochs = total_epochs - warmup_epochs
    progress = float(epoch - warmup_epochs - 1) / float(cosine_epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def curriculum_learning_rate_for_epoch(epoch: int, args, state: Dict[str, Any]) -> float:
    """Return the existing LR until Phase D, then its fixed reheat/cosine LR."""
    if getattr(args, "post_final_continuation", None):
        return args.min_lr
    phase_d_start = state.get("phase_d_start_epoch")
    if getattr(args, "curriculum_config", "adaptive") in (
        "global_cosine", "three_phase_global_cosine",
    ):
        return global_cosine_learning_rate(
            epoch, args.epochs, args.warmup_epochs, args.min_lr, args.lr,
        )
    if getattr(args, "curriculum_config", "adaptive") == "abcd_symmetric_lr":
        return symmetric_lr_cycle(
            epoch, phase_d_start if phase_d_start is not None else 1,
            args.min_lr, args.lr,
        )
    if phase_d_start is None or epoch < phase_d_start:
        return learning_rate_for_epoch(
            epoch=min(epoch, args.epochs),
            total_epochs=args.epochs,
            base_lr=args.lr,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
        )

    peak_lr = 0.5 * args.lr
    if not (0 <= args.min_lr <= peak_lr):
        raise ValueError(
            f"Phase-D LR requires 0 <= min_lr <= lr / 2, got {args.min_lr} and {args.lr}"
        )
    reheat_end_epoch = phase_d_start + PHASE_D_REHEAT_EPOCHS - 1
    if epoch <= reheat_end_epoch:
        progress = float(epoch - phase_d_start + 1) / PHASE_D_REHEAT_EPOCHS
        return args.min_lr + (peak_lr - args.min_lr) * progress

    cosine_end_epoch = reheat_end_epoch + PHASE_D_COSINE_EPOCHS
    if epoch >= cosine_end_epoch:
        return args.min_lr
    cosine_start_epoch = reheat_end_epoch + 1
    progress = float(epoch - cosine_start_epoch) / (PHASE_D_COSINE_EPOCHS - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + (peak_lr - args.min_lr) * cosine


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
    if getattr(args, "uncertainty_loss", "smooth_l1") == "student_t_nll":
        signature_data["student_t_df"] = args.student_t_df
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
    post_final_continuation: Optional[str] = None,
) -> CheckpointPlan:
    """Prefer in-run crash recovery, then a continuation source or weight init.

    Resume is only considered for ``out_dir/latest.pt``. An ``init_checkpoint``
    never restores optimizer, scheduler, or epoch state. A post-final source
    restores all training state into a distinct run. Both are ignored once a
    resume file exists. Masking-version compatibility is enforced only for
    stateful resume/continuation, not for weight-only initialization.
    """
    latest_path = Path(out_dir) / "latest.pt"
    if auto_resume and latest_path.is_file():
        return CheckpointPlan("resume", latest_path)
    if post_final_continuation:
        return CheckpointPlan(
            "continuation", expand_path(post_final_continuation)
        )
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
        ("curriculum_config", "adaptive"),
        ("post_final_continuation", None),
        ("gaussian_u_continuation", False),
        ("temporal_extrapolation_continuation", False),
        ("temporal_extrapolation_only_continuation", False),
        ("uncertainty_feature_grad_continuation", False),
        ("joint_nll_grad_continuation", False),
        ("joint_nll_grad", False),
        ("gkl_continuation_20ep", False),
        ("gkl_joint_grad_continuation_20ep", False),
        ("gkl_masking_continuation_20ep", False),
        ("differentiable_phase_c_reuse", False),
        ("mean_only_phase_c_reuse", False),
        ("student_t_unnormalized_control_20ep", False),
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
    if getattr(args, "uncertainty_loss", "smooth_l1") == "student_t_nll":
        old_df = checkpoint_args.get("student_t_df", 3.0)
        if old_df != args.student_t_df:
            mismatches.append(
                f"student_t_df: checkpoint={old_df!r}, current={args.student_t_df!r}"
            )
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
        if (key == "final_epoch" and getattr(args, "curriculum_training", False)
                and old_value == 240 and new_value == 200):
            continue
        if (key in {"channel_mults", "betas", "val_patterns"}
                and isinstance(old_value, (list, tuple))
                and isinstance(new_value, (list, tuple))):
            if list(old_value) != list(new_value):
                mismatches.append(f"{key}: checkpoint={old_value!r}, current={new_value!r}")
            continue
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


def validate_post_final_continuation(
    args, checkpoint: Dict[str, Any], source_path: Path, out_dir: Path,
) -> None:
    """Validate a separate fixed-LR continuation of a completed curriculum."""
    source_path = Path(source_path).resolve()
    if source_path.parent == Path(out_dir).resolve():
        raise ValueError(
            "Post-final continuation must write to a different run directory"
        )
    for key in ("model", "optimizer", "stats", "epoch", "args", "curriculum_state"):
        if key not in checkpoint or checkpoint[key] is None:
            raise ValueError(f"Continuation source is missing checkpoint {key!r}")
    source_args = checkpoint["args"]
    source_epoch = int(checkpoint["epoch"])
    expected_source_epochs = (
        source_args.get("epochs")
        if (getattr(args, "temporal_extrapolation_only_continuation", False)
            or getattr(args, "uncertainty_feature_grad_continuation", False)
            or getattr(args, "joint_nll_grad_continuation", False))
        else source_epoch
    )
    if source_args.get("final_epoch") != source_epoch or args.epochs != expected_source_epochs:
        raise ValueError(
            "Post-final continuation requires --epochs to match the completed "
            "source checkpoint's predetermined final epoch"
        )
    if args.final_epoch <= source_epoch or args.stop_after_epochs != args.final_epoch:
        raise ValueError(
            "Post-final continuation requires a later matching stop/final epoch"
        )

    if getattr(args, "gaussian_u_continuation", False):
        if source_epoch != 256:
            raise ValueError(
                "Gaussian-u continuation requires the completed epoch-256 source"
            )
        if source_args.get("curriculum_config") != "three_phase_global_cosine":
            raise ValueError(
                "Gaussian-u continuation requires a three-phase source checkpoint"
            )
        if source_args.get("uncertainty_loss") != "student_t_nll":
            raise ValueError(
                "Gaussian-u continuation requires a Student-t source checkpoint"
            )
        if not math.isclose(
            float(source_args.get("student_t_df", float("nan"))),
            3.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("Gaussian-u continuation requires Student-t df=3 source")
        if args.uncertainty_loss != "gaussian_nll":
            raise ValueError(
                "Gaussian-u continuation must optimize the existing Gaussian NLL"
            )
        if args.final_epoch != source_epoch + 20:
            raise ValueError(
                "Gaussian-u continuation must run exactly 20 additional epochs"
            )
    if getattr(args, "temporal_extrapolation_continuation", False):
        feature_grad = (
            getattr(args, "uncertainty_feature_grad_continuation", False)
            or getattr(args, "joint_nll_grad_continuation", False)
        )
        expected_epoch = 276 if feature_grad else 256
        additional_epochs = 10 if feature_grad else 20
        if source_epoch != expected_epoch:
            raise ValueError(
                "Temporal-extrapolation continuation requires the completed "
                f"epoch-{expected_epoch} source"
            )
        if feature_grad and not source_args.get("temporal_extrapolation_continuation", False):
            raise ValueError("Feature-gradient continuation requires the 20% extrapolation source")
        if source_args.get("curriculum_config") != "three_phase_global_cosine":
            raise ValueError(
                "Temporal-extrapolation continuation requires a three-phase source"
            )
        if (source_args.get("uncertainty_loss") != "student_t_nll"
                or float(source_args.get("student_t_df", float("nan"))) != 3.0):
            raise ValueError(
                "Temporal-extrapolation continuation requires a Student-t df=3 source"
            )
        if (args.uncertainty_loss != "student_t_nll" or args.student_t_df != 3.0
                or args.mask_pattern_weights != DEFAULT_PATTERN_WEIGHTS):
            raise ValueError(
                "Temporal-extrapolation continuation requires Student-t df=3 "
                "and the equal five-category mask weights"
            )
        if args.final_epoch != source_epoch + additional_epochs:
            raise ValueError(
                f"Temporal-extrapolation continuation must run exactly "
                f"{additional_epochs} additional epochs"
            )
    if getattr(args, "temporal_extrapolation_only_continuation", False):
        if source_epoch != 276:
            raise ValueError(
                "Temporal-extrapolation-only continuation requires the completed "
                "epoch-276 source"
            )
        if source_args.get("curriculum_config") != "three_phase_global_cosine":
            raise ValueError(
                "Temporal-extrapolation-only continuation requires a three-phase source"
            )
        if (source_args.get("uncertainty_loss") != "student_t_nll"
                or float(source_args.get("student_t_df", float("nan"))) != 3.0):
            raise ValueError(
                "Temporal-extrapolation-only continuation requires a Student-t df=3 source"
            )
        if (args.uncertainty_loss != "student_t_nll" or args.student_t_df != 3.0
                or args.mask_pattern_weights != DEFAULT_PATTERN_WEIGHTS):
            raise ValueError(
                "Temporal-extrapolation-only continuation requires Student-t df=3 "
                "and the equal five-category mask weights"
            )
        if args.final_epoch != source_epoch + 10:
            raise ValueError(
                "Temporal-extrapolation-only continuation must run exactly 10 "
                "additional epochs"
            )
    if getattr(args, "gkl_continuation_20ep", False) or getattr(args, "gkl_joint_grad_continuation_20ep", False):
        if source_epoch != 256:
            raise ValueError(
                "GKL continuation requires the completed epoch-256 source"
            )
        if source_args.get("curriculum_config") != "three_phase_global_cosine":
            raise ValueError(
                "GKL continuation requires a three-phase source checkpoint"
            )
        if (source_args.get("uncertainty_loss") != "student_t_nll"
                or float(source_args.get("student_t_df", float("nan"))) != 3.0):
            raise ValueError(
                "GKL continuation requires a Student-t df=3 source checkpoint"
            )
        if not source_args.get("joint_nll_grad", False):
            raise ValueError(
                "GKL continuation requires a joint-NLL-gradient source checkpoint"
            )
        if args.uncertainty_loss not in ("generalized_kl", "gkl"):
            raise ValueError("GKL continuation must optimize generalized_kl")
        if args.mask_pattern_weights != DEFAULT_PATTERN_WEIGHTS:
            raise ValueError(
                "GKL continuation requires the equal five-category mask weights"
            )
        if args.final_epoch != source_epoch + 20:
            raise ValueError(
                "GKL continuation must run exactly 20 additional epochs"
            )
    if getattr(args, "student_t_unnormalized_control_20ep", False):
        if source_epoch != 256:
            raise ValueError(
                "Student-t control continuation requires the completed epoch-256 source"
            )
        if source_args.get("curriculum_config") != "three_phase_global_cosine":
            raise ValueError(
                "Student-t control continuation requires a three-phase source checkpoint"
            )
        if (source_args.get("uncertainty_loss") != "student_t_nll"
                or float(source_args.get("student_t_df", float("nan"))) != 3.0):
            raise ValueError(
                "Student-t control continuation requires a Student-t df=3 source checkpoint"
            )
        if not source_args.get("joint_nll_grad", False):
            raise ValueError(
                "Student-t control continuation requires a joint-NLL-gradient source checkpoint"
            )
        if args.uncertainty_loss != "student_t_nll" or args.student_t_df != 3.0:
            raise ValueError(
                "Student-t control continuation must optimize student_t_nll with df=3"
            )
        if args.mask_pattern_weights != DEFAULT_PATTERN_WEIGHTS:
            raise ValueError(
                "Student-t control continuation requires the equal five-category mask weights"
            )
        if args.final_epoch != source_epoch + 20:
            raise ValueError(
                "Student-t control continuation must run exactly 20 additional epochs"
            )
    if getattr(args, "gkl_masking_continuation_20ep", False):
        if source_epoch != 256:
            raise ValueError(
                "GKL masking continuation requires the completed epoch-256 source"
            )
        if source_args.get("curriculum_config") != "three_phase_global_cosine":
            raise ValueError(
                "GKL masking continuation requires a three-phase source checkpoint"
            )
        if source_args.get("uncertainty_loss") not in ("generalized_kl", "gkl"):
            raise ValueError(
                "GKL masking continuation requires a Generalized KL source checkpoint"
            )
        if not source_args.get("joint_nll_grad", False):
            raise ValueError(
                "GKL masking continuation requires a joint-NLL-gradient source checkpoint"
            )
        if not source_args.get("differentiable_phase_c_reuse", False):
            raise ValueError(
                "GKL masking continuation requires a differentiable Phase-C reuse source checkpoint"
            )
        if args.uncertainty_loss not in ("generalized_kl", "gkl"):
            raise ValueError("GKL masking continuation must optimize generalized_kl")
        if not args.differentiable_phase_c_reuse:
            raise ValueError(
                "GKL masking continuation requires --differentiable-phase-c-reuse"
            )
        if args.mask_pattern_weights != DEFAULT_PATTERN_WEIGHTS:
            raise ValueError(
                "GKL masking continuation requires the equal five-category mask weights"
            )
        if args.final_epoch != source_epoch + 20:
            raise ValueError(
                "GKL masking continuation must run exactly 20 additional epochs"
            )

    compatible_args = argparse.Namespace(**vars(args))
    compatible_args.final_epoch = source_args["final_epoch"]
    compatible_args.post_final_continuation = None
    if getattr(args, "gkl_masking_continuation_20ep", False):
        compatible_args.gkl_masking_continuation_20ep = source_args.get(
            "gkl_masking_continuation_20ep", False,
        )
        compatible_args.joint_nll_grad = source_args.get("joint_nll_grad", False)
        source_masking = source_args.get("masking_version")
        if (
            source_masking == "independentBD_fiveMask_logUniformCounts_orientedSpatialBlock_v9"
            and args.masking_version == "independentBD_fiveMask_logUniformCounts_orientedSpatialBlock_sharedTemporal_v11"
        ):
            compatible_args.masking_version = source_masking
    if getattr(args, "gkl_continuation_20ep", False) or getattr(args, "gkl_joint_grad_continuation_20ep", False):
        compatible_args.gkl_continuation_20ep = source_args.get(
            "gkl_continuation_20ep", False,
        )
        compatible_args.gkl_joint_grad_continuation_20ep = source_args.get(
            "gkl_joint_grad_continuation_20ep", False,
        )
        compatible_args.differentiable_phase_c_reuse = False
        compatible_args.mean_only_phase_c_reuse = False
        compatible_args.uncertainty_loss = source_args["uncertainty_loss"]
        compatible_args.student_t_df = source_args.get("student_t_df", 3.0)
        compatible_args.joint_nll_grad = source_args.get("joint_nll_grad", False)
    if getattr(args, "student_t_unnormalized_control_20ep", False):
        compatible_args.student_t_unnormalized_control_20ep = source_args.get(
            "student_t_unnormalized_control_20ep", False,
        )
        compatible_args.joint_nll_grad = source_args.get("joint_nll_grad", False)
    if getattr(args, "uncertainty_feature_grad_continuation", False):
        compatible_args.post_final_continuation = source_args.get("post_final_continuation")
        compatible_args.uncertainty_feature_grad_continuation = source_args.get(
            "uncertainty_feature_grad_continuation", False,
        )
    if getattr(args, "joint_nll_grad_continuation", False):
        compatible_args.post_final_continuation = source_args.get("post_final_continuation")
        compatible_args.joint_nll_grad_continuation = source_args.get(
            "joint_nll_grad_continuation", False,
        )
    if getattr(args, "gaussian_u_continuation", False):
        # This explicit diagnostic changes only the likelihood. Validate every
        # other source parameter using its original value.
        compatible_args.gaussian_u_continuation = source_args.get(
            "gaussian_u_continuation", False,
        )
        compatible_args.uncertainty_loss = source_args["uncertainty_loss"]
        compatible_args.student_t_df = source_args.get("student_t_df", 3.0)
    if getattr(args, "temporal_extrapolation_continuation", False):
        # The explicit diagnostic changes only how the existing fifth mask
        # category is sampled. Every source training parameter remains strict.
        compatible_args.temporal_extrapolation_continuation = source_args.get(
            "temporal_extrapolation_continuation", False,
        )
    if getattr(args, "temporal_extrapolation_only_continuation", False):
        # This diagnostic changes all training masks to pure temporal extrapolation.
        # Preserve source continuation metadata for strict validation.
        compatible_args.post_final_continuation = source_args.get(
            "post_final_continuation"
        )
        compatible_args.temporal_extrapolation_only_continuation = source_args.get(
            "temporal_extrapolation_only_continuation", False,
        )
        compatible_args.temporal_extrapolation_continuation = source_args.get(
            "temporal_extrapolation_continuation", False,
        )
    validate_resume_parameters(compatible_args, source_args)

    if not curriculum_is_complete(checkpoint["curriculum_state"]):
        raise ValueError("Continuation source curriculum is not fully complete")
    if tuple(checkpoint["curriculum_state"].get(key) for key in ("lambda_v", "lambda_r")) != (1.0, 1.0):
        raise ValueError("Continuation source must record lambda_v=lambda_r=1")
    optimizer = checkpoint["optimizer"]
    if not optimizer.get("state"):
        raise ValueError("Continuation source has no optimizer state to restore")
    group_lrs = [group.get("lr") for group in optimizer.get("param_groups", [])]
    if not group_lrs or any(
        lr is None or not math.isclose(float(lr), args.min_lr, rel_tol=0.0, abs_tol=1e-15)
        for lr in group_lrs
    ):
        raise ValueError(
            f"Continuation source optimizer LR is not the requested floor {args.min_lr:g}"
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
    state["phase_d_start_epoch"] = None
    state["phase_d_trajectory_resume_policy"] = "reset_on_resume"
    state["variance_phase_start_epoch"] = None
    state["rollout_phase_start_epoch"] = None
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
        phase_d_start = saved.get("phase_d_start_epoch")
        if phase_d_start is not None and (
            not isinstance(phase_d_start, int) or isinstance(phase_d_start, bool)
            or phase_d_start < 1
        ):
            raise ValueError("Invalid saved Phase-D start epoch")
        state["phase_d_start_epoch"] = phase_d_start
        policy = saved.get("phase_d_trajectory_resume_policy", "reset_on_resume")
        if policy != "reset_on_resume":
            raise ValueError(f"Unsupported Phase-D trajectory resume policy: {policy!r}")
        state["phase_d_trajectory_resume_policy"] = policy
        for prefix in ("variance", "rollout"):
            phase_key = f"{prefix}_phase_start_epoch"
            phase_start = saved.get(phase_key)
            if phase_start is not None and (
                not isinstance(phase_start, int) or isinstance(phase_start, bool)
                or phase_start < 1
            ):
                raise ValueError(f"Invalid saved {prefix} phase start epoch")
            state[phase_key] = phase_start

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
    if getattr(args, "phase_d_sliding_curriculum", False):
        state["phase_d_sliding_curriculum"] = True
        state["slide_ramp_start_step"] = 0
        state["slide_ramp_steps"] = float(getattr(args, "slide_ramp_epochs", 10.0)) * updates_per_epoch
        if not math.isfinite(state["slide_ramp_steps"]) or state["slide_ramp_steps"] <= 0:
            raise ValueError("Sliding curriculum ramp duration must be positive")
        if checkpoint is not None:
            saved = checkpoint.get("curriculum_state", {})
            if not saved.get("phase_d_sliding_curriculum", False):
                raise ValueError("Cannot resume non-sliding checkpoint with --phase-d-sliding-curriculum")
            if state["slide_ramp_steps"] != saved.get("slide_ramp_steps"):
                raise ValueError(
                    f"Sliding curriculum resume changed slide_ramp_steps: "
                    f"{saved.get('slide_ramp_steps')!r} -> {state['slide_ramp_steps']!r}"
                )
            state["slide_ramp_start_step"] = int(saved.get("slide_ramp_start_step", 0))
    return state


def slide_curriculum_weight(state: Dict[str, Any]) -> float:
    """Cosine curriculum lambda_slide(p) = (1 - cos(pi * p)) / 2 in [0, 1]."""
    if not state.get("phase_d_sliding_curriculum", False):
        return 0.0
    return cosine_continuation(
        state["successful_optimizer_steps"],
        state.get("slide_ramp_start_step", 0),
        state["slide_ramp_steps"],
    )


def sample_slide_step(
    lambda_slide: float,
    window_size: int = 24,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample s in {-T..T} from P(s; p) = (1 - lambda_slide) * 1_{s=0} + lambda_slide / (2T + 1)."""
    lambda_slide = float(lambda_slide)
    if not 0.0 <= lambda_slide <= 1.0:
        raise ValueError(f"lambda_slide must be in [0, 1], got {lambda_slide}")
    if lambda_slide == 0.0:
        return 0
    u = float(torch.rand((), generator=generator).item())
    if u < 1.0 - lambda_slide:
        return 0
    return int(torch.randint(-window_size, window_size + 1, (1,), generator=generator).item())


def curriculum_weights(state: Dict[str, Any]) -> Tuple[float, float]:
    if state.get("phase_d_sliding_curriculum", False):
        return (1.0, 1.0)
    return tuple(cosine_continuation(state["successful_optimizer_steps"],
                                    state[f"{prefix}_ramp_start_step"], state[f"{prefix}_ramp_steps"])
                 if state.get(f"{prefix}_ramp_armed", state[f"{prefix}_ramp_start_step"] is not None) else 0.0
                 for prefix in ("variance", "rollout"))


def curriculum_is_complete(state: Dict[str, Any]) -> bool:
    if state.get("phase_d_sliding_curriculum", False):
        return True
    steps = state["successful_optimizer_steps"]
    ramps_complete = all(
        state[f"{prefix}_ramp_armed"]
        and state[f"{prefix}_ramp_start_step"] is not None
        and steps >= state[f"{prefix}_ramp_start_step"] + state[f"{prefix}_ramp_steps"]
        for prefix in ("variance", "rollout")
    )
    return ramps_complete and curriculum_weights(state) == (1.0, 1.0)


def start_phase_d_if_ready(
    state: Dict[str, Any], completed_epoch: int, final_epoch: int,
) -> bool:
    """Store Phase D's immutable start epoch once both curriculum ramps finish."""
    if state.get("phase_d_start_epoch") is not None or not curriculum_is_complete(state):
        return False
    start_epoch = completed_epoch + 1
    cosine_end_epoch = (start_epoch + PHASE_D_REHEAT_EPOCHS - 1
                        + PHASE_D_COSINE_EPOCHS)
    if cosine_end_epoch > final_epoch:
        raise RuntimeError(
            "Curriculum completed too late for Phase D's five-epoch reheat and fixed "
            f"40-epoch cosine decay to finish by predetermined final epoch {final_epoch}"
        )
    state["phase_d_start_epoch"] = start_epoch
    return True


def apply_ready_phase_d_transition(
    state: Dict[str, Any], diagnostic: Optional[Dict[str, Any]],
    completed_epoch: int, final_epoch: int,
) -> bool:
    """Start Phase D only from the Phase-C readiness detector."""
    if diagnostic is None or not diagnostic.get("ready_for_phase_d", False):
        return False
    return start_phase_d_if_ready(state, completed_epoch, final_epoch)


def curriculum_phase(
    epoch: int, state: Dict[str, Any], lambda_v: float, lambda_r: float,
    config: str = "adaptive",
) -> Tuple[int, str]:
    if config == "three_phase_global_cosine":
        if lambda_v == 0:
            return 0, "mean_only"
        if lambda_r == 0:
            return 1, "variance_ramp_or_settle"
        if lambda_r < 1:
            return 2, "rollout_ramp"
        return 2, "rollout_settled"
    if config == "global_cosine":
        if lambda_v == 0:
            return 0, "mean_only"
        if lambda_r == 0:
            return 1, "variance_ramp_or_settle"
        if lambda_r < 1:
            return 2, "rollout_ramp"
        return 3, "sliding_rollout"
    phase_d_start = state.get("phase_d_start_epoch")
    if phase_d_start is not None and epoch >= phase_d_start:
        return 3, "joint_refinement"
    if lambda_v == 0:
        return 0, "mean" if config == "abcd_symmetric_lr" else "reconstruction"
    if lambda_r == 0:
        return 1, "variance"
    return 2, "rollout"


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
    global_cosine = getattr(args, "curriculum_config", "adaptive") in (
        "global_cosine", "three_phase_global_cosine",
    )

    def monitor_active(row):
        if global_cosine:
            return bool(row["epoch"] >= args.warmup_epochs)
        lr = row.get("learning_rate")
        return bool(row["epoch"] >= args.epochs and lr is not None and math.isfinite(lr)
                    and math.isclose(lr, args.min_lr, rel_tol=1e-9, abs_tol=0.0))

    segment_start = None
    for row in history:
        if not monitor_active(row):
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
    monitoring_reached = monitor_active(history[-1])
    lr = history[-1].get("learning_rate")
    floor_reached = (bool(lr is not None and math.isfinite(lr)
                          and math.isclose(lr, args.min_lr, rel_tol=1e-9, abs_tol=0.0))
                     if global_cosine else monitoring_reached)
    diagnostics["ema_val_mse"] = diagnostics.pop("ema_value")
    diagnostics.update(phase_a=phase_a, lr_floor_reached=floor_reached,
                       transition_monitor_active=monitoring_reached,
                       segment_start_epoch=segment_start,
                       threshold=getattr(args, "transition_rate_threshold", 0.005),
                       patience_target=getattr(args, "transition_patience", 4),
                       ready_for_variance_ramp=phase_a and monitoring_reached and diagnostics["converged"]
                       and diagnostics["burn_in_complete"])
    return diagnostics


def nll_metric_convergence(
    history, half_life=5.0, window=5, segment_start_epoch=None,
    metric="val_gaussian_nll",
):
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
        value = row.get(metric)
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
    objective = getattr(args, "uncertainty_loss", "gaussian_nll")
    is_gkl = objective in ("generalized_kl", "gkl")
    metric_name = "gkl" if is_gkl else "nll"
    if is_gkl:
        # GKL is nonnegative. Monitor relative changes in its positive loss,
        # as for MSE, instead of exponentiating a signed likelihood slope.
        uncertainty_rows = positive_metric_convergence(
            history, metric=f"val_{objective}", half_life=half_life, window=window,
            threshold=threshold, patience=patience, segment_start_epoch=cutoff,
        )
    else:
        nll_metric = f"val_{objective if objective in ('gaussian_nll', 'student_t_nll') else 'gaussian_nll'}"
        uncertainty_rows = nll_metric_convergence(
            history, half_life=half_life, window=window,
            segment_start_epoch=cutoff, metric=nll_metric,
        )
    stable_count, previous_epoch = 0, None
    for mse, uncertainty in zip(mse_rows, uncertainty_rows):
        if previous_epoch is not None and mse["epoch"] != previous_epoch + 1:
            stable_count = 0
        previous_epoch = mse["epoch"]
        burn_in_complete = mse["burn_in_complete"] and uncertainty["burn_in_complete"]
        rates = (mse["smoothed_rate_per_epoch"], uncertainty["smoothed_rate_per_epoch"])
        stable = burn_in_complete and all(rate is not None and math.isfinite(rate) and abs(rate) < threshold for rate in rates)
        stable_count = stable_count + 1 if stable else 0
    ages = (mse["segment_age"], uncertainty["segment_age"])
    phase_b = candidate and segment_start is not None
    return dict(epoch=history[-1]["epoch"], phase_b=phase_b, variance_ramp_complete=completed,
                segment_start_epoch=segment_start, ema_val_mse=mse["ema_value"],
                mse_rate_per_epoch=rates[0], stable_count=stable_count,
                segment_age=min(ages) if all(age is not None for age in ages) else None,
                minimum_segment_age=2.0 * half_life, burn_in_complete=burn_in_complete,
                threshold=threshold, patience_target=patience,
                ready_for_rollout_ramp=phase_b and burn_in_complete and stable_count >= patience,
                **{f"ema_val_{metric_name}": uncertainty["ema_value"],
                   f"{metric_name}_rate_per_epoch": rates[1]})


def phase_c_convergence(history, args, curriculum_state):
    """Detect rollout stability after its ramp using final K=2 reuse metrics."""
    state = curriculum_state or {}
    start = state.get("rollout_ramp_start_step")
    duration = state.get("rollout_ramp_steps")
    rollout_armed = state.get("rollout_ramp_armed", start is not None)
    completed = bool(
        rollout_armed and start is not None and duration is not None and duration > 0
        and state.get("successful_optimizer_steps", -1) >= start + duration
        and curriculum_weights(state) == (1.0, 1.0)
    )
    candidate = bool(
        getattr(args, "curriculum_training", False)
        and getattr(args, "curriculum_config", "adaptive") == "abcd_symmetric_lr"
        and completed and state.get("phase_d_start_epoch") is None
    )
    segment_start = None
    objective = getattr(args, "uncertainty_loss", "gaussian_nll")
    rollout_nll_metric = (
        f"val_rollout_{objective}"
        if objective in ("gaussian_nll", "student_t_nll")
        else "val_rollout_gaussian_nll"
    )
    if candidate:
        for row in history:
            count = row.get("curriculum", {}).get("successful_optimizer_steps", -1)
            valid = (row.get("val_rollout_mse") is not None
                     and row.get(rollout_nll_metric) is not None)
            if count < start + duration or not valid:
                segment_start = None
            elif segment_start is None:
                segment_start = row["epoch"]
    half_life = getattr(args, "transition_ema_half_life", 5.0)
    window = getattr(args, "transition_rate_window", 5)
    threshold = getattr(args, "transition_rate_threshold", 0.005)
    patience = getattr(args, "transition_patience", 4)
    cutoff = segment_start if segment_start is not None else history[-1]["epoch"] + 1
    mse_rows = positive_metric_convergence(
        history, metric="val_rollout_mse", half_life=half_life, window=window,
        threshold=threshold, patience=patience, segment_start_epoch=cutoff,
    )
    nll_rows = nll_metric_convergence(
        history, metric=rollout_nll_metric, half_life=half_life,
        window=window, segment_start_epoch=cutoff,
    )
    stable_count, previous_epoch = 0, None
    for mse, nll in zip(mse_rows, nll_rows):
        if previous_epoch is not None and mse["epoch"] != previous_epoch + 1:
            stable_count = 0
        previous_epoch = mse["epoch"]
        burn_in_complete = mse["burn_in_complete"] and nll["burn_in_complete"]
        rates = (mse["smoothed_rate_per_epoch"], nll["smoothed_rate_per_epoch"])
        stable = burn_in_complete and all(
            rate is not None and math.isfinite(rate) and abs(rate) < threshold
            for rate in rates
        )
        stable_count = stable_count + 1 if stable else 0
    ages = (mse["segment_age"], nll["segment_age"])
    phase_c = candidate and segment_start is not None
    return dict(
        epoch=history[-1]["epoch"], phase_c=phase_c,
        rollout_ramp_complete=completed, segment_start_epoch=segment_start,
        ema_val_rollout_mse=mse["ema_value"], ema_val_rollout_nll=nll["ema_value"],
        rollout_mse_rate_per_epoch=rates[0], rollout_nll_rate_per_epoch=rates[1],
        stable_count=stable_count,
        segment_age=min(ages) if all(age is not None for age in ages) else None,
        minimum_segment_age=2.0 * half_life, burn_in_complete=burn_in_complete,
        threshold=threshold, patience_target=patience,
        ready_for_phase_d=phase_c and burn_in_complete and stable_count >= patience,
    )


def sample_training_mask(y, pattern_weights, density_p_zero, density_p_full,
                         magnetic_p_zero, magnetic_p_full,
                         temporal_extrapolation=False,
                         temporal_extrapolation_only=False):
    """Shared existing mask sampling for the ordinary pass and rollout pass."""
    batch_size, _, time_steps, nx, nz = y.shape
    spatial_sites = nx * nz
    if temporal_extrapolation_only:
        # Populate these directly below using only the existing suffix helper.
        magnetic_patterns = [TEMPORAL_EXTRAPOLATION_PATTERN] * batch_size
        density_patterns = [TEMPORAL_EXTRAPOLATION_PATTERN] * batch_size
        magnetic_fractions, density_fractions = [0.0] * batch_size, [0.0] * batch_size
        magnetic_counts, density_counts = [None] * batch_size, [None] * batch_size
        mask = torch.empty_like(y)
        infos = [{} for _ in range(batch_size)]
    elif sample_training_patterns is not None:
        magnetic_patterns, density_patterns = sample_training_patterns(
            pattern_weights, batch_size
        )
        magnetic_fractions = torch.rand(batch_size).tolist()
        density_fractions = torch.rand(batch_size).tolist()
        magnetic_counts: List[Optional[int]] = [None] * batch_size
        density_counts: List[Optional[int]] = [None] * batch_size

        for i in range(batch_size):
            m_pat = magnetic_patterns[i]
            d_pat = density_patterns[i]
            if m_pat in TEMPORAL_MASK_PATTERNS:
                c = sample_training_severity_counts(
                    [m_pat], spatial_sites=spatial_sites, time_steps=time_steps,
                    p_zero=magnetic_p_zero, p_full=magnetic_p_full,
                )[0]
                magnetic_counts[i] = c
                density_counts[i] = c
                density_fractions[i] = magnetic_fractions[i]
            else:
                magnetic_counts[i] = sample_training_severity_counts(
                    [m_pat], spatial_sites=spatial_sites, time_steps=time_steps,
                    p_zero=magnetic_p_zero, p_full=magnetic_p_full,
                )[0]
                density_counts[i] = sample_training_severity_counts(
                    [d_pat], spatial_sites=spatial_sites, time_steps=time_steps,
                    p_zero=density_p_zero, p_full=density_p_full,
                )[0]

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
    else:
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
    if temporal_extrapolation or temporal_extrapolation_only:
        if sample_training_patterns is not None:
            for i in range(batch_size):
                if temporal_extrapolation_only or (magnetic_patterns[i] == "temporal_block" and temporal_extrapolation):
                    temporal_mask, temporal_info = temporal_extrapolation_frames(time_steps)
                    mask[i, :4] = temporal_mask[0].expand(4, time_steps, nx, nz).to(device=y.device, dtype=y.dtype)
                    magnetic_patterns[i] = TEMPORAL_EXTRAPOLATION_PATTERN
                    density_patterns[i] = TEMPORAL_EXTRAPOLATION_PATTERN
                    horizon = int(temporal_info["horizon"])
                    frac = horizon / time_steps
                    cnt = time_steps - horizon
                    magnetic_fractions[i] = frac
                    density_fractions[i] = frac
                    magnetic_counts[i] = cnt
                    density_counts[i] = cnt
                    for mod in ("magnetic", "density"):
                        infos[i][f"{mod}_pattern"] = TEMPORAL_EXTRAPOLATION_PATTERN
                        infos[i].update({f"{mod}_{k}": v for k, v in temporal_info.items()})
                        infos[i][f"{mod}_actual_mask_fraction"] = frac
                        infos[i][f"{mod}_visible_fraction"] = 1.0 - frac
                        if temporal_extrapolation_only:
                            infos[i][f"{mod}_target_mask_fraction"] = frac
                    infos[i]["actual_mask_fraction"] = frac
        else:
            for i in range(batch_size):
                for modality, channels, patterns, fractions, counts in (
                    ("magnetic", slice(0, 3), magnetic_patterns,
                     magnetic_fractions, magnetic_counts),
                    ("density", slice(3, 4), density_patterns,
                     density_fractions, density_counts),
                ):
                    if not temporal_extrapolation_only and patterns[i] != "temporal_block":
                        continue
                    temporal_mask, temporal_info = temporal_extrapolation_frames(
                        time_steps
                    )
                    mask[i, channels] = temporal_mask[0].expand(
                        channels.stop - channels.start, time_steps, nx, nz,
                    ).to(device=y.device, dtype=y.dtype)
                    patterns[i] = TEMPORAL_EXTRAPOLATION_PATTERN
                    horizon = int(temporal_info["horizon"])
                    fractions[i] = horizon / time_steps
                    counts[i] = time_steps - horizon
                    infos[i][f"{modality}_pattern"] = TEMPORAL_EXTRAPOLATION_PATTERN
                    infos[i].update({
                        f"{modality}_{key}": value
                        for key, value in temporal_info.items()
                    })
                    infos[i][f"{modality}_actual_mask_fraction"] = horizon / time_steps
                    infos[i][f"{modality}_visible_fraction"] = 1.0 - horizon / time_steps
                    if temporal_extrapolation_only:
                        infos[i][f"{modality}_target_mask_fraction"] = horizon / time_steps
                infos[i]["actual_mask_fraction"] = (
                    3.0 * infos[i]["magnetic_actual_mask_fraction"]
                    + infos[i]["density_actual_mask_fraction"]
                ) / 4.0
    return (mask, infos, magnetic_patterns, density_patterns, magnetic_fractions,
            density_fractions, magnetic_counts, density_counts)


class PhaseDTrajectory:
    """One local batch slot in the stateful Phase-D sliding stream."""

    def __init__(
        self, run_name: str, target: torch.Tensor, slide_step: int, start: int,
        rehearsal_length: Optional[int] = None,
    ):
        self.run_name = run_name
        self.target = target
        self.slide_step = int(slide_step)
        self.start = int(start)
        self.rehearsal_length = rehearsal_length
        self.reuse_depth = 0
        self.previous_mean: Optional[torch.Tensor] = None
        self.previous_variance: Optional[torch.Tensor] = None
        self.rehearsal_magnetic_pattern: Optional[str] = None
        self.rehearsal_density_pattern: Optional[str] = None


class PhaseDStreamingBatch:
    """Maintain independent recursive trajectories across optimizer steps.

    Trajectory tensors intentionally are not checkpointed. A process resume
    starts fresh local slots while retaining model/optimizer/curriculum state.
    """

    def __init__(
        self, dataset, train_runs: Sequence[str], batch_size: int,
        window_size: int, device: torch.device, mean: torch.Tensor,
        std: torch.Tensor, seed: int,
    ):
        if window_size != 24:
            raise ValueError("Stateful Phase D requires the production T=24 window")
        available = set(getattr(dataset, "run_file_indices", train_runs))
        self.train_runs = tuple(sorted(set(train_runs) & available))
        missing = set(train_runs) - available
        if missing:
            raise ValueError(f"Phase-D train runs are absent from the dataset: {sorted(missing)[:5]}")
        if not self.train_runs:
            raise ValueError("Phase D requires at least one training run")
        if hasattr(dataset, "samples"):
            window_runs = {str(sample[1]) for sample in dataset.samples}
            self.full_window_runs = tuple(
                name for name in self.train_runs if name in window_runs
            )
        elif hasattr(dataset, "runs"):
            self.full_window_runs = tuple(
                name for name in self.train_runs
                if int(dataset.runs[name].shape[1]) >= window_size
            )
        else:
            self.full_window_runs = self.train_runs
        if not self.full_window_runs:
            raise ValueError("Phase-D s=0 rehearsal requires a full physical window")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.window_size = int(window_size)
        self.device = device
        self.mean = mean[0]
        self.std = std[0]
        self.generator = torch.Generator().manual_seed(int(seed))
        self.slots: List[Optional[PhaseDTrajectory]] = [None] * self.batch_size
        self.begin_epoch()

    def begin_epoch(self) -> None:
        self._s_sum = 0.0
        self._abs_s_sum = 0.0
        self._reuse_sum = 0.0
        self._reuse_max = 0
        self._overlap_sum = 0.0
        self._resets = 0
        self._sign_counts = [0, 0, 0]  # negative, zero, positive
        self._finite_count = 0
        self._target_count = 0
        self._slot_steps = 0
        self._s_counts = [0] * (2 * self.window_size + 1)
        self._s_zero_rehearsal_steps = 0
        self._s_zero_rehearsal_trajectories = 0
        self._s_zero_rehearsal_length_sum = 0
        self._s_zero_mask_counts = {name: 0 for name in MASK_PATTERNS}

    @staticmethod
    def can_advance_start(
        start: int, slide_step: int, length: int, window_size: int = 24,
    ) -> bool:
        """Whether the existing boundary semantics permit one regular shift."""
        if slide_step == 0:
            return False
        reached_end = (
            start + window_size - 1 >= length - 1
            if slide_step > 0 else start <= 0
        )
        next_start = start + slide_step
        return (
            not reached_end
            and -(window_size - 1) <= next_start <= length - 1
        )

    def reset_slot(
        self, slot: int, run_name: Optional[str] = None,
        slide_step: Optional[int] = None, window_start: Optional[int] = None,
    ) -> PhaseDTrajectory:
        if slide_step is None:
            slide_step = int(torch.randint(
                -self.window_size, self.window_size + 1, (1,), generator=self.generator,
            ))
        if not -self.window_size <= slide_step <= self.window_size:
            raise ValueError(f"Invalid Phase-D slide step: {slide_step}")
        if run_name is None:
            candidates = self.full_window_runs if slide_step == 0 else self.train_runs
            index = int(torch.randint(len(candidates), (1,), generator=self.generator))
            run_name = candidates[index]
        elif run_name not in self.train_runs:
            raise ValueError(f"Phase-D slot reset requested a non-training run: {run_name}")

        physical = self.dataset.get_run(run_name).to(self.device, non_blocking=True)
        if physical.ndim != 4 or physical.shape[0] != 4:
            raise ValueError(f"Expected complete run shape (4,L,X,Z), got {tuple(physical.shape)}")
        target = (physical - self.mean) / (self.std + 1e-8)
        length = int(target.shape[1])
        rehearsal_length = None
        if slide_step == 0:
            if length < self.window_size:
                raise ValueError(
                    f"Phase-D s=0 rehearsal requires at least {self.window_size} "
                    f"physical frames, got {length} for {run_name}"
                )
            start = int(torch.randint(
                0, length - self.window_size + 1, (1,), generator=self.generator,
            ))
            rehearsal_length = int(torch.randint(16, 49, (1,), generator=self.generator))
        else:
            starts = [
                candidate for candidate in range(-(self.window_size - 1), length)
                if self.can_advance_start(
                    candidate, slide_step, length, self.window_size
                )
            ]
            if not starts:
                raise ValueError(
                    f"No Phase-D start permits a slide of {slide_step} for "
                    f"run {run_name} with length {length}"
                )
            start = starts[int(torch.randint(
                len(starts), (1,), generator=self.generator,
            ))]
        if window_start is not None:
            start = int(window_start)
        if slide_step == 0:
            if not 0 <= start <= length - self.window_size:
                raise ValueError(
                    f"Invalid Phase-D rehearsal window start {start} for length {length}"
                )
        elif not self.can_advance_start(start, slide_step, length, self.window_size):
            raise ValueError(
                f"Phase-D window start {start} cannot perform a slide of "
                f"{slide_step} for length {length}"
            )

        trajectory = PhaseDTrajectory(
            run_name, target, slide_step, start, rehearsal_length,
        )
        self.slots[slot] = trajectory
        self._resets += 1
        if slide_step == 0:
            self._s_zero_rehearsal_trajectories += 1
            self._s_zero_rehearsal_length_sum += rehearsal_length
        return trajectory

    def _ensure_slots(self) -> None:
        for slot, trajectory in enumerate(self.slots):
            if trajectory is None:
                self.reset_slot(slot)

    def _window(self, trajectory: PhaseDTrajectory) -> torch.Tensor:
        target = trajectory.target
        length, start = int(target.shape[1]), trajectory.start
        window = target.new_full(
            (4, self.window_size, target.shape[2], target.shape[3]), float("nan")
        )
        source_start = max(start, 0)
        source_end = min(start + self.window_size, length)
        if source_start < source_end:
            destination_start = source_start - start
            window[:, destination_start:destination_start + source_end - source_start] = (
                target[:, source_start:source_end]
            )
        return window

    def current_targets(self) -> Tuple[torch.Tensor, Dict[str, List[Any]]]:
        self._ensure_slots()
        trajectories = [slot for slot in self.slots if slot is not None]
        targets = torch.stack([self._window(slot) for slot in trajectories])
        metadata = {
            "run_name": [slot.run_name for slot in trajectories],
            "beta": [float(parse_run_name(slot.run_name)["beta"]) for slot in trajectories],
            "t0": [slot.start for slot in trajectories],
            "slide_step": [slot.slide_step for slot in trajectories],
            "reuse_depth": [slot.reuse_depth for slot in trajectories],
        }
        return targets, metadata

    @staticmethod
    def overlap_slices(slide_step: int, window_size: int = 24) -> Tuple[slice, slice]:
        """Return (current, previous) time slices for one regular shift."""
        if not -window_size <= slide_step <= window_size:
            raise ValueError(f"Invalid slide step: {slide_step}")
        if slide_step > 0:
            return slice(0, window_size - slide_step), slice(slide_step, window_size)
        if slide_step < 0:
            distance = -slide_step
            return slice(distance, window_size), slice(0, window_size - distance)
        return slice(0, window_size), slice(0, window_size)

    def make_model_input(
        self, targets: torch.Tensor, observation_mask: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        magnetic_patterns: Optional[Sequence[str]] = None,
        density_patterns: Optional[Sequence[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create finite input values/u and reuse detached temporal overlap."""
        if noise is None:
            noise = torch.randn_like(targets)
        values = noise.clone()
        input_variance = torch.ones_like(targets)
        for index, trajectory in enumerate(self.slots):
            assert trajectory is not None
            if trajectory.slide_step == 0:
                if magnetic_patterns is not None:
                    trajectory.rehearsal_magnetic_pattern = magnetic_patterns[index]
                if density_patterns is not None:
                    trajectory.rehearsal_density_pattern = density_patterns[index]
            if trajectory.previous_mean is not None:
                current_slice, previous_slice = self.overlap_slices(
                    trajectory.slide_step, self.window_size
                )
                values[index, :, current_slice] = trajectory.previous_mean[:, previous_slice]
                input_variance[index, :, current_slice] = (
                    trajectory.previous_variance[:, previous_slice]
                )

        finite = torch.isfinite(targets)
        exact = finite & (observation_mask > 0.5)
        values[exact] = targets[exact]
        input_variance[exact] = 0.0
        if not torch.isfinite(values).all() or not torch.isfinite(input_variance).all():
            raise FloatingPointError("Phase-D construction passed a non-finite value to the model")

        for trajectory in self.slots:
            assert trajectory is not None
            slide_step = trajectory.slide_step
            self._s_sum += slide_step
            self._abs_s_sum += abs(slide_step)
            self._reuse_sum += trajectory.reuse_depth
            self._reuse_max = max(self._reuse_max, trajectory.reuse_depth)
            overlap = (self.window_size - abs(slide_step)) / self.window_size
            self._overlap_sum += overlap if trajectory.reuse_depth > 0 else 0.0
            self._sign_counts[(slide_step > 0) + (slide_step >= 0)] += 1
            self._s_counts[slide_step + self.window_size] += 1
            self._slot_steps += 1
        self._finite_count += int(finite.sum().item())
        self._target_count += finite.numel()
        return torch.cat([values, input_variance], dim=1), exact.to(targets.dtype)

    def advance(self, prediction: torch.Tensor, variance: torch.Tensor) -> None:
        """Detach one successful update and independently advance/reset slots."""
        for index, trajectory in enumerate(self.slots):
            assert trajectory is not None
            mean = prediction[index].detach().float().clone()
            output_variance = variance[index].detach().float().clone()
            if mean.grad_fn is not None or output_variance.grad_fn is not None:
                raise RuntimeError("Phase-D state must be detached between optimizer steps")

            if trajectory.slide_step == 0:
                if trajectory.rehearsal_length is None:
                    raise RuntimeError("Phase-D s=0 trajectory has no rehearsal length")
                self._s_zero_rehearsal_steps += 1
                # Ordinary training draws B and Density patterns independently;
                # the per-pattern diagnostics count both standard-mask draws.
                for pattern in (
                    trajectory.rehearsal_magnetic_pattern,
                    trajectory.rehearsal_density_pattern,
                ):
                    if pattern is not None:
                        if pattern not in self._s_zero_mask_counts:
                            raise ValueError(f"Unknown Phase-D rehearsal mask: {pattern}")
                        self._s_zero_mask_counts[pattern] += 1
                trajectory.rehearsal_magnetic_pattern = None
                trajectory.rehearsal_density_pattern = None
                completed = trajectory.reuse_depth + 1
                if completed >= trajectory.rehearsal_length:
                    self.slots[index] = None
                else:
                    trajectory.previous_mean = mean
                    trajectory.previous_variance = output_variance
                    trajectory.reuse_depth = completed
                continue

            length = int(trajectory.target.shape[1])
            if not self.can_advance_start(
                trajectory.start, trajectory.slide_step, length, self.window_size
            ):
                self.slots[index] = None
                continue
            trajectory.previous_mean = mean
            trajectory.previous_variance = output_variance
            trajectory.start += trajectory.slide_step
            trajectory.reuse_depth += 1

    def local_epoch_metrics(self) -> Dict[str, Any]:
        count = max(self._slot_steps, 1)
        negative, zero, positive = self._sign_counts
        return {
            "s_mean": self._s_sum / count,
            "abs_s_mean": self._abs_s_sum / count,
            "reuse_depth_mean": self._reuse_sum / count,
            "reuse_depth_max": self._reuse_max,
            "overlap_fraction_mean": self._overlap_sum / count,
            "trajectory_resets": self._resets,
            "s_zero_fraction": zero / count,
            "s_positive_fraction": positive / count,
            "s_negative_fraction": negative / count,
            "finite_target_fraction": self._finite_count / max(self._target_count, 1),
            "s_zero_rehearsal_steps": self._s_zero_rehearsal_steps,
            "s_zero_rehearsal_trajectories": self._s_zero_rehearsal_trajectories,
            "s_zero_rehearsal_length_mean": (
                self._s_zero_rehearsal_length_sum
                / max(self._s_zero_rehearsal_trajectories, 1)
            ),
            **{
                f"s_zero_mask_count_{name}": self._s_zero_mask_counts[name]
                for name in MASK_PATTERNS
            },
            "slot_steps": self._slot_steps,
            "s_counts": {
                str(step): self._s_counts[step + self.window_size]
                for step in range(-self.window_size, self.window_size + 1)
            },
        }


def phase_c_geometric_weights(lambda_r: float) -> Tuple[float, float, float]:
    """Normalized K=2 reuse weights: 1, lambda_r, lambda_r squared."""
    lambda_r = float(lambda_r)
    if not math.isfinite(lambda_r) or not 0.0 <= lambda_r <= 1.0:
        raise ValueError(f"Phase-C lambda_r must be finite and in [0,1], got {lambda_r}")
    normalizer = 1.0 + lambda_r + lambda_r ** 2
    return 1.0 / normalizer, lambda_r / normalizer, lambda_r ** 2 / normalizer


def make_phase_c_reuse_input(
    target: torch.Tensor,
    prediction: torch.Tensor,
    output_variance: torch.Tensor,
    observation_mask: torch.Tensor,
    *,
    reuse_gradient_scale: float = 0.0,
    mean_only_phase_c_reuse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reuse same-window predictions with optional cross-forward gradients."""
    def scale_gradient(value):
        if reuse_gradient_scale == 1.0:
            return value
        detached = value.detach()
        if reuse_gradient_scale == 0.0:
            return detached
        return detached + reuse_gradient_scale * (value - detached)

    exact = observation_mask > 0.5
    values = torch.where(exact, target, scale_gradient(prediction))
    reused_variance = (
        output_variance.detach() if mean_only_phase_c_reuse else scale_gradient(output_variance)
    )
    input_variance = torch.where(
        exact, torch.zeros_like(target), reused_variance,
    )
    return torch.cat([values, input_variance], dim=1), observation_mask


def build_sliding_reuse_input(
    target_b: torch.Tensor,
    mask_b: torch.Tensor,
    pred_a: torch.Tensor,
    var_a: torch.Tensor,
    slide_step: int,
    *,
    noise_b: Optional[torch.Tensor] = None,
    reuse_gradient_scale: float = 1.0,
    mean_only_phase_c_reuse: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Construct differentiable second-pass input for Window B with temporal shift `slide_step`.

    When `slide_step == 0` and all targets are finite, delegates directly to
    `make_phase_c_reuse_input` to guarantee exact mathematical and numerical equivalence
    to settled Phase C.
    """
    window_size = int(target_b.shape[2])
    if not -window_size <= slide_step <= window_size:
        raise ValueError(f"Invalid slide_step {slide_step} for window_size={window_size}")

    finite = torch.isfinite(target_b)
    if slide_step == 0 and finite.all():
        return make_phase_c_reuse_input(
            target_b,
            pred_a,
            var_a,
            mask_b,
            reuse_gradient_scale=reuse_gradient_scale,
            mean_only_phase_c_reuse=mean_only_phase_c_reuse,
        )

    def scale_gradient(value: torch.Tensor) -> torch.Tensor:
        if reuse_gradient_scale == 1.0:
            return value
        detached = value.detach()
        if reuse_gradient_scale == 0.0:
            return detached
        return detached + reuse_gradient_scale * (value - detached)

    if noise_b is None:
        noise_b = torch.randn_like(target_b)
    values = noise_b.clone()
    input_variance = torch.ones_like(target_b)

    if abs(slide_step) < window_size:
        curr_slice, prev_slice = PhaseDStreamingBatch.overlap_slices(slide_step, window_size)
        reused_mean = scale_gradient(pred_a[:, :, prev_slice])
        reused_var = (
            var_a[:, :, prev_slice].detach()
            if mean_only_phase_c_reuse
            else scale_gradient(var_a[:, :, prev_slice])
        )
        values[:, :, curr_slice] = reused_mean
        input_variance[:, :, curr_slice] = reused_var

    exact = finite & (mask_b > 0.5)
    values = torch.where(exact, target_b, values)
    input_variance = torch.where(exact, torch.zeros_like(target_b), input_variance)
    exact_mask = exact.to(mask_b.dtype) if not finite.all() else mask_b
    return torch.cat([values, input_variance], dim=1), exact_mask


def sample_sliding_window_pair(
    dataset,
    run_name: str,
    t0_hint: Optional[int],
    slide_step: int,
    window_size: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """Extract normalized (window_A, window_B) of shape (4, T, X, Z) from one physical run."""
    physical = dataset.get_run(run_name).to(device, non_blocking=True)
    target = (physical - mean[0]) / (std[0] + 1e-8)
    length = int(target.shape[1])
    if length < window_size:
        raise ValueError(f"Run {run_name} has length {length} < window_size={window_size}")

    if slide_step == 0:
        max_start = length - window_size
        t_a = int(t0_hint) if (t0_hint is not None and 0 <= int(t0_hint) <= max_start) else int(
            torch.randint(0, max_start + 1, (1,), generator=generator).item()
        )
        win_a = target[:, t_a:t_a + window_size]
        return win_a, win_a, t_a, t_a

    # Prefer pairs where both A and B are 100% finite physical windows
    valid_full_starts = [
        s_a for s_a in range(0, length - window_size + 1)
        if 0 <= s_a + slide_step <= length - window_size
    ]
    if valid_full_starts:
        if t0_hint is not None and int(t0_hint) in valid_full_starts:
            t_a = int(t0_hint)
        else:
            idx = int(torch.randint(len(valid_full_starts), (1,), generator=generator).item())
            t_a = valid_full_starts[idx]
        t_b = t_a + slide_step
        return target[:, t_a:t_a + window_size], target[:, t_b:t_b + window_size], t_a, t_b

    # Fall back to established Phase-D boundary padding when length < window_size + |slide_step|
    valid_padded_starts = [
        s_a for s_a in range(0, length - window_size + 1)
        if PhaseDStreamingBatch.can_advance_start(s_a, slide_step, length, window_size)
    ]
    if not valid_padded_starts:
        raise ValueError(
            f"Run {run_name} (length={length}) has no valid window pair for slide_step={slide_step}"
        )
    if t0_hint is not None and int(t0_hint) in valid_padded_starts:
        t_a = int(t0_hint)
    else:
        idx = int(torch.randint(len(valid_padded_starts), (1,), generator=generator).item())
        t_a = valid_padded_starts[idx]
    t_b = t_a + slide_step

    win_a = target[:, t_a:t_a + window_size]
    win_b = target.new_full((4, window_size, target.shape[2], target.shape[3]), float("nan"))
    src_start = max(t_b, 0)
    src_end = min(t_b + window_size, length)
    if src_start < src_end:
        dst_start = src_start - t_b
        win_b[:, dst_start:dst_start + src_end - src_start] = target[:, src_start:src_end]
    return win_a, win_b, t_a, t_b


def curriculum_forward_loss(
    model, model_input, target, mask, lambda_v, lambda_r, reuse_masks=None,
    uncertainty_loss="gaussian_nll", student_t_df=3.0,
    curriculum_config="adaptive",
    detach_variance_features=True,
    detach_mean=True,
    gkl_alpha=None,
    reuse_gradient_scale=0.0,
    mean_only_phase_c_reuse=False,
    slide_step: int = 0,
    reuse_targets: Optional[Sequence[torch.Tensor]] = None,
    reuse_noises: Optional[Sequence[torch.Tensor]] = None,
):
    """Phase A/B pass or Phase-C reuse for the selected curriculum."""
    three_phase = curriculum_config == "three_phase_global_cosine"
    if three_phase:
        normalizer = 1.0 + float(lambda_r)
        weights = (1.0 / normalizer, float(lambda_r) / normalizer, 0.0)
        reuse_steps = 1
    else:
        weights = phase_c_geometric_weights(lambda_r)
        reuse_steps = 2

    def base_loss(mu, var):
        combined = mu + lambda_v * var
        return combined / (1.0 + lambda_v) if three_phase else combined

    pred, variance = model(model_input, detach_variance_features=detach_variance_features)
    mu_loss = full_mse_loss(pred, target, mask)
    if gkl_alpha is not None and lambda_v > 0:
        alpha = float(gkl_alpha)
        l_st = (
            predicted_error_mse_loss(
                variance, pred, target, "student_t_nll", student_t_df,
                detach_mean=False,
            )
            if alpha < 1.0 else mu_loss.new_zeros(())
        )
        l_gkl = (
            predicted_error_mse_loss(
                variance, pred, target, "generalized_kl", student_t_df,
                detach_mean=detach_mean,
            )
            if alpha > 0.0 else mu_loss.new_zeros(())
        )
        var_loss = (1.0 - alpha) * l_st + alpha * l_gkl
    else:
        var_loss = (predicted_error_mse_loss(
                        variance, pred, target, uncertainty_loss, student_t_df,
                        detach_mean=detach_mean,
                    )
                    if lambda_v > 0 else mu_loss.new_zeros(()))
    check_stage1(torch.isfinite(pred).all() & torch.isfinite(variance).all() & (variance > 0).all()
                 & torch.isfinite(mu_loss) & (mu_loss <= 100) & torch.isfinite(var_loss),
                 "Invalid curriculum first-pass prediction/variance/loss or catastrophic reconstruction MSE (>100)")
    pass_losses = [base_loss(mu_loss, var_loss)]
    zero = mu_loss.new_zeros(())
    components = {
        "L_mu_0": mu_loss, "L_var_0": var_loss,
        "L_mu_1": zero, "L_var_1": zero,
        "L_mu_2": zero, "L_var_2": zero,
        "reuse_weight_0": mu_loss.new_tensor(weights[0]),
        "reuse_weight_1": mu_loss.new_tensor(weights[1]),
        "reuse_weight_2": mu_loss.new_tensor(weights[2]),
        "forwards": 1,
    }
    if gkl_alpha is not None:
        components["gkl_alpha"] = mu_loss.new_tensor(alpha)
        components["L_student_t_0"] = l_st.detach()
        components["L_gkl_0"] = l_gkl.detach()
    if lambda_r > 0:
        if reuse_masks is None or len(reuse_masks) != reuse_steps:
            raise ValueError(
                "Positive Phase-C rollout weight requires exactly "
                f"{reuse_steps} new mask{'s' if reuse_steps != 1 else ''}"
            )
        previous_mean, previous_variance = pred, variance
        for pass_index, reuse_mask in enumerate(reuse_masks, 1):
            reuse_target = (
                reuse_targets[pass_index - 1]
                if reuse_targets is not None
                else target
            )
            reuse_noise = (
                reuse_noises[pass_index - 1]
                if reuse_noises is not None
                else None
            )
            reuse_input, exact_mask = build_sliding_reuse_input(
                reuse_target,
                reuse_mask,
                previous_mean,
                previous_variance,
                slide_step=slide_step,
                noise_b=reuse_noise,
                reuse_gradient_scale=reuse_gradient_scale,
                mean_only_phase_c_reuse=mean_only_phase_c_reuse,
            )
            next_mean, next_variance = model(
                reuse_input, detach_variance_features=detach_variance_features,
            )
            next_mu_loss = full_mse_loss(next_mean, reuse_target, exact_mask)
            if gkl_alpha is not None and lambda_v > 0:
                alpha = float(gkl_alpha)
                next_l_st = (
                    predicted_error_mse_loss(
                        next_variance, next_mean, reuse_target, "student_t_nll", student_t_df,
                        detach_mean=False,
                    )
                    if alpha < 1.0 else next_mu_loss.new_zeros(())
                )
                next_l_gkl = (
                    predicted_error_mse_loss(
                        next_variance, next_mean, reuse_target, "generalized_kl", student_t_df,
                        detach_mean=detach_mean,
                    )
                    if alpha > 0.0 else next_mu_loss.new_zeros(())
                )
                next_var_loss = (1.0 - alpha) * next_l_st + alpha * next_l_gkl
                components[f"L_student_t_{pass_index}"] = next_l_st.detach()
                components[f"L_gkl_{pass_index}"] = next_l_gkl.detach()
            else:
                next_var_loss = predicted_error_mse_loss(
                    next_variance, next_mean, reuse_target, uncertainty_loss, student_t_df,
                    detach_mean=detach_mean,
                )
            check_stage1(
                torch.isfinite(next_mean).all()
                & torch.isfinite(next_variance).all()
                & (next_variance > 0).all()
                & torch.isfinite(next_mu_loss)
                & (next_mu_loss <= 100)
                & torch.isfinite(next_var_loss),
                f"Invalid curriculum pass-{pass_index} prediction/variance/loss "
                "or catastrophic reconstruction MSE (>100)",
            )
            components[f"L_mu_{pass_index}"] = next_mu_loss
            components[f"L_var_{pass_index}"] = next_var_loss
            pass_losses.append(base_loss(next_mu_loss, next_var_loss))
            previous_mean, previous_variance = next_mean, next_variance
        components["forwards"] = 1 + reuse_steps
    loss = sum(weight * pass_loss for weight, pass_loss in zip(weights, pass_losses))
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
    student_t_df: float = 3.0,
    curriculum_state: Optional[Dict[str, Any]] = None,
    curriculum_log_every: int = 50,
    phase_d_stream: Optional[PhaseDStreamingBatch] = None,
    curriculum_config: str = "adaptive",
    temporal_extrapolation_continuation: bool = False,
    temporal_extrapolation_only_continuation: bool = False,
    uncertainty_feature_grad_continuation: bool = False,
    joint_nll_grad_continuation: bool = False,
    joint_nll_grad: bool = False,
    gkl_continuation_20ep: bool = False,
    gkl_joint_grad_continuation_20ep: bool = False,
    gkl_masking_continuation_20ep: bool = False,
    differentiable_phase_c_reuse: bool = False,
    mean_only_phase_c_reuse: bool = False,
    student_t_unnormalized_control_20ep: bool = False,
    phase_d_sliding_curriculum: bool = False,
    dataset: Optional[Any] = None,
    slide_rng: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    model.train()
    is_gkl = gkl_continuation_20ep or gkl_joint_grad_continuation_20ep
    stage1 = uncertainty_aware and uncertainty_loss in (
        "gaussian_nll", "student_t_nll", "generalized_kl", "gkl",
    )
    uncertainty_key = uncertainty_loss if stage1 else "uncertainty_mse"

    n_patterns = len(MASK_PATTERNS)

    # Accumulate on device and read back once per epoch. Training patterns are
    # independent marginals, so no ambiguous 5x5 combination loss is logged.
    sum_mse = torch.zeros((), device=device)
    sum_uncertainty_mse = torch.zeros((), device=device)
    gkl_totals = torch.zeros(5, dtype=torch.float64, device=device) if is_gkl else None
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
    curriculum_totals = (torch.zeros(15, device=device, dtype=torch.float64)
                         if curriculum_state is not None else None)
    lambda_start = lambda_end = None
    phase_d_active = phase_d_stream is not None
    if phase_d_active:
        if curriculum_state is None or not curriculum_is_complete(curriculum_state):
            raise ValueError("Stateful Phase D requires a complete curriculum")
        phase_d_stream.begin_epoch()
    sliding_active = bool(phase_d_sliding_curriculum and curriculum_state is not None)
    if sliding_active and dataset is None:
        raise ValueError("Unified sliding-window Phase D requires dataset access")
    if sliding_active and slide_rng is None:
        rank = dist.get_rank() if distributed_is_initialized() else 0
        slide_rng = torch.Generator().manual_seed(int(epoch_number) * 1009 + rank)
    slide_counts = torch.zeros(49, device=device, dtype=torch.float64) if sliding_active else None
    slide_lambda_sum = 0.0
    slide_lambda_start = None
    slide_lambda_end = None
    slide_batch_count = 0

    def count_optimizer_step(_optimizer, _args, _kwargs):
        nonlocal applied_steps
        applied_steps += 1

    step_counter = optimizer.register_step_post_hook(count_optimizer_step)

    batches_per_epoch = max(len(loader), 1)

    batches = range(len(loader)) if phase_d_active else loader
    for step, batch in enumerate(batches):
        data_wait_seconds += time.perf_counter() - last_batch_end
        if phase_d_active:
            y, metadata = phase_d_stream.current_targets()
            batch = {"metadata": metadata}
        else:
            y = batch["block"].to(device, non_blocking=True)
            y = normalize(y, mean, std)

        step_slide = 0
        reuse_targets = None
        if sliding_active:
            lambda_slide = slide_curriculum_weight(curriculum_state)
            slide_lambda_sum += lambda_slide
            if slide_lambda_start is None:
                slide_lambda_start = lambda_slide
            slide_lambda_end = lambda_slide
            slide_batch_count += 1
            step_slide = sample_slide_step(
                lambda_slide, window_size=int(y.shape[-3]), generator=slide_rng,
            )
            slide_counts[step_slide + int(y.shape[-3])] += 1
            if step_slide != 0:
                run_names = batch["metadata"]["run_name"]
                t0_hints = batch["metadata"]["t0"]
                pair_a, pair_b = [], []
                for r_name, t0_h in zip(run_names, t0_hints):
                    wa, wb, _, _ = sample_sliding_window_pair(
                        dataset, str(r_name), int(t0_h), step_slide,
                        int(y.shape[-3]), mean, std, device, generator=slide_rng,
                    )
                    pair_a.append(wa)
                    pair_b.append(wb)
                y = torch.stack(pair_a, dim=0)
                reuse_targets = (torch.stack(pair_b, dim=0),)
        for beta in batch.get("metadata", {}).get("beta", []):
            key = str(float(beta))
            beta_sample_counts[key] = beta_sample_counts.get(key, 0) + 1

        batch_size = y.shape[0]
        time_steps = int(y.shape[-3])
        spatial_sites = int(y.shape[-2] * y.shape[-1])

        (mask, mask_infos, magnetic_patterns, density_patterns, magnetic_fractions,
         density_fractions, magnetic_visible_counts, density_visible_counts) = sample_training_mask(
            y, pattern_weights, density_visible_p_zero, density_visible_p_full,
            magnetic_visible_p_zero, magnetic_visible_p_full,
            temporal_extrapolation_continuation,
            temporal_extrapolation_only_continuation)
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
        if phase_d_active:
            model_input, mask = phase_d_stream.make_model_input(
                y, mask, magnetic_patterns=magnetic_patterns,
                density_patterns=density_patterns,
            )
            input_variance = model_input[:, 4:]
        elif uncertainty_aware:
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
            finite_target_ok = (torch.isfinite(y).any() if phase_d_active
                                else torch.isfinite(y).all())
            check_stage1(torch.isfinite(model_input).all() & finite_target_ok,
                         f"Nonfinite input/target: epoch {epoch_number}, batch {step + 1}")

        if curriculum_state is not None:
            lambda_v, lambda_r = curriculum_weights(curriculum_state)
            lambda_start = lambda_start or (lambda_v, lambda_r)
            lambda_end = (lambda_v, lambda_r)
            # Phase D is already a recursive state, so it performs one forward.
            reuse_steps = (1 if curriculum_config == "three_phase_global_cosine" else 2)
            reuse_masks = (tuple(
                sample_training_mask(
                    y, pattern_weights, density_visible_p_zero, density_visible_p_full,
                    magnetic_visible_p_zero, magnetic_visible_p_full,
                    temporal_extrapolation_continuation,
                    temporal_extrapolation_only_continuation,
                )[0]
                for _ in range(reuse_steps)
            ) if lambda_r > 0 and not phase_d_active else None)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda"), dtype=amp_dtype):
            if phase_d_active:
                pred, output_variance = model(model_input, detach_variance_features=True)
                reconstruction_loss = full_mse_loss(pred, y, mask)
                variance_loss = predicted_error_mse_loss(
                    output_variance, pred, y, uncertainty_loss, student_t_df,
                )
                loss = reconstruction_loss + variance_loss
                components = {
                    "L_mu_1": reconstruction_loss,
                    "L_var_1": variance_loss,
                    "forwards": 1,
                    "total_loss": loss,
                }
            elif curriculum_state is not None:
                enable_variance_feature_grad = (
                    uncertainty_feature_grad_continuation
                    or joint_nll_grad_continuation
                    or joint_nll_grad
                    or gkl_continuation_20ep
                    or gkl_joint_grad_continuation_20ep
                    or gkl_masking_continuation_20ep
                    or student_t_unnormalized_control_20ep
                )
                enable_mean_nll_grad = (
                    joint_nll_grad_continuation
                    or joint_nll_grad
                    or gkl_joint_grad_continuation_20ep
                    or gkl_masking_continuation_20ep
                    or student_t_unnormalized_control_20ep
                )
                gkl_alpha = None
                if is_gkl:
                    gkl_start_step = 256 * curriculum_state.get("updates_per_epoch", len(loader))
                    gkl_duration_steps = 2 * curriculum_state.get("updates_per_epoch", len(loader))
                    current_step = curriculum_state["successful_optimizer_steps"]
                    step_progress = min(max((current_step - gkl_start_step) / gkl_duration_steps, 0.0), 1.0)
                    gkl_alpha = 0.5 * (1.0 - math.cos(math.pi * step_progress))
                reuse_gradient_scale = 0.0
                if differentiable_phase_c_reuse:
                    reuse_gradient_scale = lambda_r if gkl_alpha is None else gkl_alpha
                loss, pred, output_variance, components = curriculum_forward_loss(
                    model, model_input, y, mask, lambda_v, lambda_r, reuse_masks,
                    uncertainty_loss, student_t_df, curriculum_config,
                    detach_variance_features=not enable_variance_feature_grad,
                    detach_mean=not enable_mean_nll_grad,
                    gkl_alpha=gkl_alpha,
                    reuse_gradient_scale=reuse_gradient_scale,
                    mean_only_phase_c_reuse=mean_only_phase_c_reuse,
                    slide_step=step_slide,
                    reuse_targets=reuse_targets,
                )
                reconstruction_loss = components["L_mu_0"]
                variance_loss = components["L_var_0"]
            else:
                pred = model(model_input)
                if uncertainty_aware:
                    pred, output_variance = pred
                reconstruction_loss = full_mse_loss(pred, y, mask)
                loss = reconstruction_loss
                if uncertainty_aware:
                    variance_loss = predicted_error_mse_loss(
                        output_variance, pred, y, objective=uncertainty_loss,
                        student_t_df=student_t_df,
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
            if phase_d_active and applied:
                phase_d_stream.advance(pred, output_variance)
            coordinate = curriculum_state["successful_optimizer_steps"]
            curriculum_state["successful_optimizer_steps"] += applied
            with torch.no_grad():
                component_zero = loss.new_zeros(())
                mu_0 = components.get("L_mu_0", components.get("L_mu_1", component_zero))
                var_0 = components.get("L_var_0", components.get("L_var_1", component_zero))
                # Preserve Phase-D's existing single-forward L_mu_1/L_var_1
                # diagnostics while also exposing that loss as pass 0.
                mu_1 = components.get("L_mu_1", component_zero)
                var_1 = components.get("L_var_1", component_zero)
                curriculum_totals += torch.stack([
                    mu_0, var_0, mu_1, var_1,
                    components.get("L_mu_2", component_zero),
                    components.get("L_var_2", component_zero), loss,
                    loss.new_tensor(lambda_v), loss.new_tensor(lambda_r),
                    components.get("reuse_weight_0", loss.new_tensor(1.0)),
                    components.get("reuse_weight_1", component_zero),
                    components.get("reuse_weight_2", component_zero),
                    loss.new_tensor(components["forwards"]),
                    loss.new_tensor(components["forwards"] == 2),
                    loss.new_tensor(components["forwards"] == 3),
                ]).detach().double()
                if is_gkl and gkl_totals is not None:
                    gkl_totals += torch.stack([
                        components.get("gkl_alpha", component_zero),
                        components.get("L_student_t_0", component_zero),
                        components.get("L_gkl_0", component_zero),
                        components.get("L_student_t_1", component_zero),
                        components.get("L_gkl_1", component_zero),
                    ]).detach().double()
            if step % curriculum_log_every == 0 or step + 1 == len(loader):
                region = ("D" if phase_d_active else
                          ("A" if lambda_v == 0 else ("B" if lambda_r == 0 else "C")))
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
                [PATTERN_TO_ID[
                    "temporal_block"
                    if name == TEMPORAL_EXTRAPOLATION_PATTERN else name
                ] for name in magnetic_patterns],
                device=device,
            )
            density_pattern_ids = torch.tensor(
                [PATTERN_TO_ID[
                    "temporal_block"
                    if name == TEMPORAL_EXTRAPOLATION_PATTERN else name
                ] for name in density_patterns],
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
                if magnetic_patterns[i] in {
                    "temporal_block", TEMPORAL_EXTRAPOLATION_PATTERN,
                }
                else magnetic_fractions[i]
            )
            density_target = (
                info["density_actual_mask_fraction"]
                if density_patterns[i] in {
                    "temporal_block", TEMPORAL_EXTRAPOLATION_PATTERN,
                }
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

    logged_pattern_names = tuple(
        TEMPORAL_EXTRAPOLATION_PATTERN if (
            (temporal_extrapolation_continuation or temporal_extrapolation_only_continuation)
            and name == "temporal_block"
        ) else name
        for name in MASK_PATTERNS
    )
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
            for i, name in enumerate(logged_pattern_names)
        },
        "density_pattern_share": {
            name: float(density_pattern_counts[i].cpu()) / total_count
            for i, name in enumerate(logged_pattern_names)
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
        names = (
            "L_mu_0", "L_var_0", "L_mu_1", "L_var_1", "L_mu_2", "L_var_2",
            "total_loss", "lambda_v_mean", "lambda_r_mean",
            "reuse_weight_0", "reuse_weight_1", "reuse_weight_2",
            "phase_c_forwards", "two_forward_fraction", "three_forward_fraction",
        )
        metrics["curriculum"] = dict(zip(names, (curriculum_totals / max(global_batches, 1)).cpu().tolist()))
        metrics["curriculum"]["lambda_r"] = metrics["curriculum"]["lambda_r_mean"]
        metrics["curriculum"].update(lambda_v_start=lambda_start[0], lambda_r_start=lambda_start[1],
                                     lambda_v_end=lambda_end[0], lambda_r_end=lambda_end[1],
                                     successful_optimizer_steps=curriculum_state["successful_optimizer_steps"])
        if is_gkl and gkl_totals is not None:
            if distributed_is_initialized():
                dist.all_reduce(gkl_totals, op=dist.ReduceOp.SUM)
            gkl_names = (
                "gkl_alpha", "L_student_t_0", "L_gkl_0", "L_student_t_1", "L_gkl_1",
            )
            metrics["curriculum"].update(
                dict(zip(gkl_names, (gkl_totals / max(global_batches, 1)).cpu().tolist()))
            )
    if phase_d_active:
        local = phase_d_stream.local_epoch_metrics()
        count = local["slot_steps"]
        sums = torch.tensor([
            local["s_mean"] * count,
            local["abs_s_mean"] * count,
            local["reuse_depth_mean"] * count,
            local["overlap_fraction_mean"] * count,
            local["trajectory_resets"],
            local["s_zero_fraction"] * count,
            local["s_positive_fraction"] * count,
            local["s_negative_fraction"] * count,
            local["finite_target_fraction"] * count,
            count,
            *[local["s_counts"][str(step)] for step in range(-24, 25)],
            local["s_zero_rehearsal_steps"],
            local["s_zero_rehearsal_trajectories"],
            (local["s_zero_rehearsal_length_mean"]
             * local["s_zero_rehearsal_trajectories"]),
            *[local[f"s_zero_mask_count_{name}"] for name in MASK_PATTERNS],
        ], dtype=torch.float64, device=device)
        reuse_max = torch.tensor(local["reuse_depth_max"], device=device, dtype=torch.int64)
        if distributed_is_initialized():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(reuse_max, op=dist.ReduceOp.MAX)
        values = sums.cpu().tolist()
        global_slot_steps = max(values[9], 1.0)
        rehearsal_offset = 10 + 2 * 24 + 1
        rehearsal_trajectories = values[rehearsal_offset + 1]
        metrics["phase_d"] = {
            "s_mean": values[0] / global_slot_steps,
            "abs_s_mean": values[1] / global_slot_steps,
            "reuse_depth_mean": values[2] / global_slot_steps,
            "reuse_depth_max": int(reuse_max.cpu()),
            "overlap_fraction_mean": values[3] / global_slot_steps,
            "trajectory_resets": int(values[4]),
            "s_zero_fraction": values[5] / global_slot_steps,
            "s_positive_fraction": values[6] / global_slot_steps,
            "s_negative_fraction": values[7] / global_slot_steps,
            "finite_target_fraction": values[8] / global_slot_steps,
            "s_zero_rehearsal_steps": int(values[rehearsal_offset]),
            "s_zero_rehearsal_trajectories": int(rehearsal_trajectories),
            "s_zero_rehearsal_length_mean": (
                values[rehearsal_offset + 2] / max(rehearsal_trajectories, 1.0)
            ),
            **{
                f"s_zero_mask_count_{name}": int(values[rehearsal_offset + 3 + index])
                for index, name in enumerate(MASK_PATTERNS)
            },
            "s_counts": {
                str(step): int(values[10 + step + 24]) for step in range(-24, 25)
            },
        }
    if sliding_active and slide_counts is not None:
        if distributed_is_initialized():
            dist.all_reduce(slide_counts, op=dist.ReduceOp.SUM)
        counts_list = [int(c) for c in slide_counts.cpu().tolist()]
        total_s = max(sum(counts_list), 1)
        s_vals = list(range(-24, 25))
        neg_cnt = sum(c for s, c in zip(s_vals, counts_list) if s < 0)
        zero_cnt = counts_list[24]
        pos_cnt = sum(c for s, c in zip(s_vals, counts_list) if s > 0)
        abs_s_mean = sum(abs(s) * c for s, c in zip(s_vals, counts_list)) / total_s
        metrics["sliding_curriculum"] = {
            "lambda_slide_start": float(slide_lambda_start if slide_lambda_start is not None else 0.0),
            "lambda_slide_end": float(slide_lambda_end if slide_lambda_end is not None else 0.0),
            "lambda_slide_mean": float(slide_lambda_sum / max(slide_batch_count, 1)),
            "s_zero_fraction": zero_cnt / total_s,
            "s_positive_fraction": pos_cnt / total_s,
            "s_negative_fraction": neg_cnt / total_s,
            "abs_s_mean": abs_s_mean,
            "s_counts": {str(s): c for s, c in zip(s_vals, counts_list)},
        }
    if stage1:
        if distributed_is_initialized():
            dist.all_reduce(gradient_norm_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(gradient_norm_max, op=dist.ReduceOp.MAX)
        metrics.update(gradient_norm_mean=float(gradient_norm_sum.cpu()) / max(global_batches, 1),
                       gradient_norm_max=float(gradient_norm_max.cpu()),
                       binary_input_variance_verified=not phase_d_active,
                       finite_model_input_verified=True, nonfinite_events=0,
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


def rollout_validation_reuse_pattern(pattern: str) -> str:
    """Pair each fixed validation mask with the next existing fixed layout."""
    index = MASK_PATTERNS.index(pattern)
    return MASK_PATTERNS[(index + 1) % len(MASK_PATTERNS)]


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
    student_t_df: float = 3.0,
    rollout_validation: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate every mask pattern on the same validation blocks.

    All five patterns use deterministic ~50% paper-style layouts with an
    identical Bx/By/Bz/Density observation mask. Training samplers are
    unchanged. The loader is iterated once and each block is reused for all
    patterns.
    """
    model.eval()
    stage1 = uncertainty_aware and uncertainty_loss in (
        "gaussian_nll", "student_t_nll", "generalized_kl", "gkl",
    )
    if rollout_validation and not stage1:
        raise ValueError("Rollout validation requires an uncertainty-aware NLL mode")
    uncertainty_key = uncertainty_loss if stage1 else "uncertainty_mse"
    rollout_uncertainty_key = f"rollout_{uncertainty_key}"
    beta_details = {}
    _ = (mask_seed, density_probe_min, density_probe_max)

    sums = torch.zeros(len(patterns), device=device)
    uncertainty_sums = torch.zeros_like(sums)
    rollout_mse_sums = torch.zeros_like(sums)
    rollout_nll_sums = torch.zeros_like(sums)
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
                    variance_loss = (predicted_error_mse_loss(
                                         output_variance, pred, y,
                                         objective=uncertainty_loss,
                                         student_t_df=student_t_df,
                                     )
                                     if stage1 else predicted_error_mse_loss(output_variance, pred, y))
                if rollout_validation:
                    reuse_mask_1, _ = make_fixed_validation_mask(
                        rollout_validation_reuse_pattern(pattern), y.shape,
                        device=device, dtype=y.dtype,
                    )
                    second_input, exact_mask_1 = make_phase_c_reuse_input(
                        y, pred, output_variance, reuse_mask_1,
                    )
                    pred2, variance2 = model(second_input, detach_variance_features=True)
                    reuse_mask_2, _ = make_fixed_validation_mask(
                        rollout_validation_reuse_pattern(
                            rollout_validation_reuse_pattern(pattern)
                        ),
                        y.shape, device=device, dtype=y.dtype,
                    )
                    third_input, exact_mask_2 = make_phase_c_reuse_input(
                        y, pred2, variance2, reuse_mask_2,
                    )
                    pred3, variance3 = model(third_input, detach_variance_features=True)
                    rollout_mse = full_mse_loss(pred3, y, exact_mask_2)
                    rollout_nll = predicted_error_mse_loss(
                        variance3, pred3, y, objective=uncertainty_loss,
                        student_t_df=student_t_df,
                    )

            if stage1:
                # Validation shards have unequal lengths: no per-batch collectives.
                check_stage1(torch.isfinite(model_input).all() & torch.isfinite(y).all()
                             & torch.isfinite(pred).all() & torch.isfinite(output_variance).all()
                             & (output_variance > 0).all() & torch.isfinite(variance_loss)
                             & torch.isfinite(loss_mse) & (loss_mse <= 100),
                             f"Invalid validation values or catastrophic reconstruction MSE (>100): {batch.get('metadata')}",
                             synchronize=False)
                if rollout_validation:
                    check_stage1(
                        torch.isfinite(second_input).all() & torch.isfinite(pred2).all()
                        & torch.isfinite(variance2).all() & (variance2 > 0).all()
                        & torch.isfinite(third_input).all() & torch.isfinite(pred3).all()
                        & torch.isfinite(variance3).all() & (variance3 > 0).all()
                        & torch.isfinite(rollout_mse) & (rollout_mse <= 100)
                        & torch.isfinite(rollout_nll),
                        f"Invalid rollout validation values or catastrophic reconstruction MSE (>100): "
                        f"{batch.get('metadata')}", synchronize=False,
                    )
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
            if rollout_validation:
                rollout_mse_sums[i] += rollout_mse.detach().float() * batch_size
                rollout_nll_sums[i] += rollout_nll.detach().float() * batch_size

    if distributed_is_initialized():
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
        if uncertainty_aware:
            dist.all_reduce(uncertainty_sums, op=dist.ReduceOp.SUM)
        if rollout_validation:
            dist.all_reduce(rollout_mse_sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(rollout_nll_sums, op=dist.ReduceOp.SUM)

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
    if rollout_validation:
        rollout_mse_sums = rollout_mse_sums.cpu()
        rollout_nll_sums = rollout_nll_sums.cpu()
        for i, pattern in enumerate(patterns):
            metrics[pattern]["rollout_mse"] = float(rollout_mse_sums[i] / counts[i])
            metrics[pattern][rollout_uncertainty_key] = float(
                rollout_nll_sums[i] / counts[i]
            )
        metrics["mean"]["rollout_mse"] = sum(
            metrics[p]["rollout_mse"] for p in patterns
        ) / len(patterns)
        metrics["mean"][rollout_uncertainty_key] = sum(
            metrics[p][rollout_uncertainty_key] for p in patterns
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


@torch.no_grad()
def validate_sliding_windows(
    model: torch.nn.Module,
    dataset: Any,
    val_runs: Sequence[str],
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    slide_steps: Sequence[int] = (0, 1, -1, 12, -12),
    window_size: int = 24,
    pattern: str = "spatial_random",
    amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    uncertainty_loss: str = "generalized_kl",
    student_t_df: float = 3.0,
    seed: int = 4321,
    max_runs: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Evaluate two-forward sliding reconstruction on validation runs."""
    model.eval()
    available = [r for r in sorted(val_runs) if r in getattr(dataset, "run_file_indices", {})]
    selected_runs = available[:max_runs] if max_runs > 0 else available
    rng = torch.Generator().manual_seed(int(seed))
    results: Dict[str, Dict[str, float]] = {}

    for s in slide_steps:
        s_int = int(s)
        mse_vals, u_vals = [], []
        for run_idx, run_name in enumerate(selected_runs):
            try:
                win_a, win_b, _, _ = sample_sliding_window_pair(
                    dataset, run_name, None, s_int, window_size, mean, std, device, generator=rng,
                )
            except ValueError:
                continue
            win_a_b = win_a.unsqueeze(0)
            win_b_b = win_b.unsqueeze(0)
            mask_a, _ = make_fixed_validation_mask(pattern, win_a_b.shape, device=device, dtype=win_a_b.dtype)
            mask_b, _ = make_fixed_validation_mask(pattern, win_b_b.shape, device=device, dtype=win_b_b.dtype)
            noise_a = torch.randn(win_a_b.shape, generator=rng, device="cpu", dtype=win_a_b.dtype).to(device)
            noise_b = torch.randn(win_b_b.shape, generator=rng, device="cpu", dtype=win_b_b.dtype).to(device)
            iv_a = 1.0 - mask_a
            inp_a = torch.cat([torch.where(iv_a > 0.5, noise_a, win_a_b), iv_a], dim=1)
            with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda"), dtype=amp_dtype):
                pred_a, var_a = model(inp_a, detach_variance_features=True)
                inp_b, exact_b = build_sliding_reuse_input(
                    win_b_b, mask_b, pred_a, var_a, s_int,
                    noise_b=noise_b, reuse_gradient_scale=0.0,
                )
                pred_b, var_b = model(inp_b, detach_variance_features=True)
                mse_b = full_mse_loss(pred_b, win_b_b, exact_b)
                u_b = predicted_error_mse_loss(
                    var_b, pred_b, win_b_b, objective=uncertainty_loss, student_t_df=student_t_df,
                )
            mse_vals.append(float(mse_b.item()))
            u_vals.append(float(u_b.item()))
        results[str(s_int)] = {
            "mse": float(np.mean(mse_vals)) if mse_vals else float("nan"),
            uncertainty_loss: float(np.mean(u_vals)) if u_vals else float("nan"),
            "n_runs": len(mse_vals),
        }
    return results


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

    for modality in ("magnetic", "density"):
        for pattern, share in train_metrics[f"{modality}_pattern_share"].items():
            log_data[f"mask/{modality}_pattern_share/{pattern}"] = share

    for pattern in val_patterns:
        log_data[f"val/{pattern}/mse"] = val_metrics[pattern]["mse"]
        if "rollout_mse" in val_metrics[pattern]:
            log_data[f"val/{pattern}/rollout_mse"] = val_metrics[pattern]["rollout_mse"]
            rollout_key = next(
                key for key in (
                    "rollout_gaussian_nll", "rollout_student_t_nll",
                    "rollout_generalized_kl", "rollout_gkl",
                )
                if key in val_metrics[pattern]
            )
            log_data[f"val/{pattern}/{rollout_key}"] = val_metrics[pattern][rollout_key]
    if "val_rollout_mse" in row:
        log_data["val/rollout_mse"] = row["val_rollout_mse"]
        rollout_key = next(
            key for key in (
                "val_rollout_gaussian_nll", "val_rollout_student_t_nll",
                "val_rollout_generalized_kl", "val_rollout_gkl",
            )
            if key in row
        )
        log_data[f"val/{rollout_key.removeprefix('val_')}"] = row[rollout_key]
    if "train_generalized_kl" in row:
        log_data["train/generalized_kl"] = row["train_generalized_kl"]
        log_data["val/generalized_kl"] = row["val_generalized_kl"]
    elif "train_gkl" in row:
        log_data["train/gkl"] = row["train_gkl"]
        log_data["val/gkl"] = row["val_gkl"]
    elif "train_gaussian_nll" in row:
        log_data["train/gaussian_nll"] = row["train_gaussian_nll"]
        log_data["val/gaussian_nll"] = row["val_gaussian_nll"]
    elif "train_student_t_nll" in row:
        log_data["train/student_t_nll"] = row["train_student_t_nll"]
        log_data["val/student_t_nll"] = row["val_student_t_nll"]
    elif "train_uncertainty_mse" in row:
        log_data["train/uncertainty_smooth_l1"] = row["train_uncertainty_mse"]
        log_data["val/uncertainty_smooth_l1"] = row["val_uncertainty_mse"]
    if "curriculum" in row:
        log_data.update({f"curriculum/{key}": value for key, value in row["curriculum"].items()})
        log_data["curriculum/ready_for_variance_ramp"] = False
        log_data["curriculum/ready_for_rollout_ramp"] = False
        log_data["curriculum/ready_for_phase_d"] = False
    if "phase_d" in row:
        log_data.update({
            f"phase_d/{key}": value for key, value in row["phase_d"].items()
            if key != "s_counts"
        })
        for step, count in row["phase_d"]["s_counts"].items():
            label = f"neg_{step[1:]}" if step.startswith("-") else f"pos_{step}"
            if step == "0":
                label = "zero"
            log_data[f"phase_d/s_count/{label}"] = count
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
        for key in ("nll_rate_per_epoch", "gkl_rate_per_epoch"):
            if diagnostic.get(key) is not None:
                log_data[f"curriculum/{key}"] = diagnostic[key]
        log_data.update({
            "curriculum/convergence_stable_count": diagnostic["stable_count"],
            "curriculum/convergence_segment_age": diagnostic["segment_age"],
            "curriculum/ready_for_rollout_ramp": diagnostic["ready_for_rollout_ramp"],
        })
    if "phase_c_convergence" in row:
        log_data.update({f"phase_c_convergence/{key}": value
                         for key, value in row["phase_c_convergence"].items()
                         if value is not None})
        diagnostic = row["phase_c_convergence"]
        log_data.update({
            "curriculum/rollout_mse_rate_per_epoch": diagnostic[
                "rollout_mse_rate_per_epoch"
            ],
            "curriculum/rollout_nll_rate_per_epoch": diagnostic[
                "rollout_nll_rate_per_epoch"
            ],
            "curriculum/convergence_stable_count": diagnostic["stable_count"],
            "curriculum/convergence_segment_age": diagnostic["segment_age"],
            "curriculum/ready_for_phase_d": diagnostic["ready_for_phase_d"],
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
    path = Path(path)
    if path.name == "latest.pt":
        temporary_path = path.with_name(path.name + ".tmp")
        temporary_path.unlink(missing_ok=True)
        try:
            torch.save(ckpt, temporary_path)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
    else:
        torch.save(ckpt, path)


def save_history_atomic(path: Path, history: List[Dict[str, Any]]) -> None:
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        with open(temporary_path, "w") as f:
            json.dump(history, f, indent=2)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def history_through_checkpoint(
    history: List[Dict[str, Any]], checkpoint_epoch: int, first_epoch: int = 1,
) -> List[Dict[str, Any]]:
    """Return the contiguous history committed through the authoritative checkpoint."""
    epochs = [row.get("epoch") for row in history]
    if any(not isinstance(epoch, int) or isinstance(epoch, bool) for epoch in epochs):
        raise ValueError("Cannot resume: history contains an invalid epoch.")
    trimmed = [row for row, epoch in zip(history, epochs) if epoch <= checkpoint_epoch]
    if [row["epoch"] for row in trimmed] != list(range(first_epoch, checkpoint_epoch + 1)):
        raise ValueError("Cannot resume: history is behind or inconsistent with the completed checkpoint epoch.")
    return trimmed


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
    if args.post_final_continuation:
        continuation_start = (
            277 if (args.temporal_extrapolation_only_continuation
                    or args.uncertainty_feature_grad_continuation
                    or args.joint_nll_grad_continuation) else args.epochs + 1
        )
        rank_print(
            f"Post-final continuation: epochs {continuation_start}-{args.final_epoch}, full objective "
            f"(lambda_v=lambda_r=1), fixed LR={args.min_lr:g}; no ramps, "
            "convergence transitions, warmup, or reheat."
        )
        if args.gaussian_u_continuation:
            rank_print(
                "Gaussian-u diagnostic continuation: source objective="
                "student_t_nll (df=3), continuation objective=gaussian_nll; "
                "settled three-phase K=1 reuse remains unchanged; Phase D is disabled."
            )
        if args.temporal_extrapolation_continuation:
            rank_print(
                "Temporal-task diagnostic continuation: objective=student_t_nll "
                "(df=3); temporal_block draw replaced by "
                "temporal_extrapolation with probability=0.2; horizon is "
                "UniformInteger(1,23), visible prefix -> hidden future suffix; "
                "settled three-phase K=1 reuse remains unchanged; Phase D is disabled."
            )
        if args.uncertainty_feature_grad_continuation:
            rank_print(
                "Uncertainty feature-gradient diagnostic: variance features are not "
                "detached in training; NLL mean predictions remain detached; "
                "source epoch=276; 10 additional epochs; 20% extrapolation masks unchanged."
            )
        if args.joint_nll_grad_continuation:
            rank_print(
                "Joint NLL gradient diagnostic: Student-t NLL gradients enabled "
                "in both the mean head (Path A) and variance head/shared features (Path B); "
                "source epoch=276; 10 additional epochs; 20% extrapolation masks unchanged."
            )
        if args.temporal_extrapolation_only_continuation:
            rank_print(
                "Temporal-task 100% extrapolation diagnostic continuation: "
                "diagnostic=temporal_extrapolation_only; "
                f"source checkpoint={args.post_final_continuation}; source epoch=276; "
                "objective=student_t_nll; student_t_df=3; "
                "temporal_extrapolation training fraction=1.0; "
                "horizon range=1..23 uniform; lambda_v=1; lambda_r=1; "
                f"LR={args.min_lr:g}; settled three-phase K=1 reuse remains unchanged; "
                "Phase D is disabled."
            )
    rank_print("Mask pattern weights:", args.mask_pattern_weights)
    rank_print("Validation mask patterns:", args.val_patterns)
    if args.uncertainty_aware:
        rank_print("Uncertainty objective:", args.uncertainty_loss)
        if args.uncertainty_loss == "student_t_nll":
            rank_print("Student-t df:", args.student_t_df)
        if args.joint_nll_grad:
            rank_print(
                f"Joint gradient curriculum: {args.uncertainty_loss} gradients enabled "
                "in both the mean head (Path A) and variance head/shared features (Path B) "
                "during Phase B and Phase C."
            )
            if args.differentiable_phase_c_reuse:
                rank_print("Phase C: cross-forward mean and variance gradients scaled by lambda_r.")
        if args.gkl_continuation_20ep:
            rank_print(
                "GKL continuation: diagnostic=generalized_kl_continuation; "
                f"source checkpoint={args.post_final_continuation}; source epoch=256; "
                "target epoch=276; 2-epoch cosine transition from Student-t A+B to GKL; "
                f"LR={args.min_lr:g}; settled three-phase K=1 reuse remains unchanged; "
                "Phase D is disabled."
            )
        if args.gkl_joint_grad_continuation_20ep:
            reuse_description = "detached reuse"
            if args.mean_only_phase_c_reuse:
                reuse_description = "cross-forward mean gradients scaled by alpha, detached variance reuse"
            elif args.differentiable_phase_c_reuse:
                reuse_description = "cross-forward mean and variance gradients scaled by alpha"
            rank_print(
                "GKL joint-gradient continuation: diagnostic=generalized_kl_joint_grad_continuation; "
                f"source checkpoint={args.post_final_continuation}; source epoch=256; "
                "target epoch=276; 2-epoch cosine transition from Student-t A+B to GKL; "
                "Path A ON; Path B ON; "
                f"LR={args.min_lr:g}; settled three-phase K=1 with {reuse_description}; "
                "Phase D is disabled."
            )
        if args.student_t_unnormalized_control_20ep:
            rank_print(
                "Student-t unnormalized control: diagnostic=student_t_unnormalized_control; "
                f"source checkpoint={args.post_final_continuation}; source epoch=256; "
                "target epoch=276; objective=student_t_nll; student_t_df=3; "
                "Path A ON; Path B ON; "
                f"LR={args.min_lr:g}; settled three-phase K=1 reuse remains unchanged; "
                "Phase D is disabled."
            )
        if getattr(args, "gkl_masking_continuation_20ep", False):
            rank_print(
                "GKL masking continuation: diagnostic=gkl_masking_continuation; "
                f"source checkpoint={args.post_final_continuation}; source epoch=256; "
                f"target epoch={args.final_epoch}; objective=generalized_kl; "
                "Path A ON; Path B ON; "
                f"LR={args.min_lr:g}; settled three-phase K=1 with differentiable Phase C reuse; "
                f"masking_version={args.masking_version}; "
                "Phase D is disabled."
            )
        if args.uncertainty_loss in ("gaussian_nll", "student_t_nll", "generalized_kl", "gkl"):
            input_label = "Curriculum first pass" if args.curriculum_training else "Stage-1"
            rank_print(f"{input_label}: binary input variance checked every batch; FP32 NLL floor={GAUSSIAN_NLL_VARIANCE_FLOOR:g}; no upper cap")
    if args.curriculum_training:
        rank_print("Curriculum: binary first pass; detached variance features; direct, unclipped rollout reuse; "
                   "no calibration scale. Phase D uses per-slot stateful T=24 sliding with reset-on-resume; "
                   "validation retains the standard single-pass benchmark.")
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
        post_final_continuation=args.post_final_continuation,
    )
    initial_post_final_continuation = checkpoint_plan.mode == "continuation"
    if checkpoint_plan.mode == "resume":
        resume_checkpoint = torch.load(checkpoint_plan.path, map_location=device)
        validate_checkpoint_split(resume_checkpoint)
        validate_resume_parameters(args, resume_checkpoint.get("args", {}))
        rank_print(
            f"Auto-resuming from {checkpoint_plan.path} "
            f"(completed epoch {resume_checkpoint['epoch']})"
        )
    elif checkpoint_plan.mode == "continuation":
        if checkpoint_plan.path is None or not checkpoint_plan.path.is_file():
            raise FileNotFoundError(
                f"Post-final continuation checkpoint not found: {checkpoint_plan.path}"
            )
        resume_checkpoint = torch.load(checkpoint_plan.path, map_location=device)
        validate_checkpoint_split(resume_checkpoint)
        validate_post_final_continuation(
            args, resume_checkpoint, checkpoint_plan.path, out_dir,
        )
        rank_print(
            f"Starting isolated post-final continuation from {checkpoint_plan.path} "
            f"(completed epoch {resume_checkpoint['epoch']}); model and optimizer "
            f"state restored, full objective fixed, LR={args.min_lr:g}."
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
    elif getattr(args, "phase_d_sliding_curriculum", False) and init_checkpoint is not None:
        stats = init_checkpoint["stats"]
        rank_print("Reusing channel statistics from init checkpoint for Phase C -> D continuation")
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
    if args.post_final_continuation and not curriculum_is_complete(curriculum_state):
        raise ValueError("Post-final continuation restored an incomplete curriculum state")
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

    phase_d_stream = None
    if (curriculum_state is not None
            and args.curriculum_config != "three_phase_global_cosine"):
        phase_d_stream = PhaseDStreamingBatch(
            dataset=dataset,
            train_runs=sorted(selected_train_runs),
            batch_size=args.batch_size,
            window_size=args.delta_t,
            device=device,
            mean=mean,
            std=std,
            seed=args.seed + rank,
        )
        if (resume_checkpoint is not None
                and curriculum_phase(
                    start_epoch, curriculum_state, *curriculum_weights(curriculum_state),
                    args.curriculum_config,
                )[0] == 3):
            rank_print(
                "Phase-D resume: local streaming trajectory slots reset safely; "
                "model, optimizer, LR, and curriculum state remain restored."
            )

    if is_main_process():
        with open(out_dir / "config.json", "w") as f:
            json.dump({**vars(args), "data_split": split_metadata}, f, indent=2)

    final_epoch = args.stop_after_epochs or args.epochs
    if (curriculum_state is not None and resume_checkpoint is not None
            and not args.post_final_continuation
            and args.curriculum_config == "adaptive"):
        if start_phase_d_if_ready(curriculum_state, start_epoch - 1, args.final_epoch):
            rank_print(
                f"Started Phase D at epoch {curriculum_state['phase_d_start_epoch']} "
                "from a pre-Phase-D checkpoint whose rollout ramp was already complete."
            )
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
    if resume_checkpoint is not None and not initial_post_final_continuation:
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
                **({
                    "diagnostic": "temporal_extrapolation_sampling",
                    "temporal_extrapolation_probability": 0.2,
                    "temporal_extrapolation_horizon_range": "1..23 uniform",
                } if args.temporal_extrapolation_continuation else {}),
                **({
                    "diagnostic": "temporal_extrapolation_only",
                    "temporal_extrapolation_probability": 1.0,
                    "temporal_extrapolation_horizon_range": "1..23 uniform",
                } if args.temporal_extrapolation_only_continuation else {}),
                **({
                    "diagnostic": "joint_nll_gradient",
                    "nll_grad_path_a_mean": True,
                    "nll_grad_path_b_variance": True,
                    "temporal_extrapolation_probability": 0.2,
                    "temporal_extrapolation_horizon_range": "1..23 uniform",
                } if args.joint_nll_grad_continuation else {}),
                **({
                    "joint_nll_gradient": True,
                    "nll_grad_path_a_mean": True,
                    "nll_grad_path_b_variance": True,
                } if args.joint_nll_grad else {}),
                **({
                    "diagnostic": "gkl_continuation",
                    "uncertainty_objective": "generalized_kl",
                    "transition_epochs": 2,
                    "transition_schedule": "cosine",
                    "nll_grad_path_a_mean": False,
                    "nll_grad_path_b_variance": True,
                } if args.gkl_continuation_20ep else {}),
                **({
                    "diagnostic": "gkl_joint_grad_continuation",
                    "uncertainty_objective": "generalized_kl",
                    "transition_epochs": 2,
                    "transition_schedule": "cosine",
                    "nll_grad_path_a_mean": True,
                    "nll_grad_path_b_variance": True,
                } if args.gkl_joint_grad_continuation_20ep else {}),
                **({
                    "diagnostic": "student_t_unnormalized_control",
                    "uncertainty_objective": "student_t_nll",
                    "student_t_df": 3.0,
                    "nll_grad_path_a_mean": True,
                    "nll_grad_path_b_variance": True,
                } if args.student_t_unnormalized_control_20ep else {}),
                **({
                    "diagnostic": "gkl_masking_continuation",
                    "uncertainty_objective": "generalized_kl",
                    "nll_grad_path_a_mean": True,
                    "nll_grad_path_b_variance": True,
                    "differentiable_phase_c_reuse": True,
                    "masking_version": args.masking_version,
                } if getattr(args, "gkl_masking_continuation_20ep", False) else {}),
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
        wandb_run.define_metric("phase_c_convergence/*", step_metric="epoch")
        wandb_run.define_metric("phase_d/*", step_metric="epoch")

    history = []
    history_path = out_dir / "history.json"
    if is_main_process() and resume_checkpoint is not None and history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        first_history_epoch = args.epochs + 1 if args.post_final_continuation else 1
        if (args.temporal_extrapolation_only_continuation
                or args.uncertainty_feature_grad_continuation
                or args.joint_nll_grad_continuation):
            first_history_epoch = 277
        history = history_through_checkpoint(
            history, start_epoch - 1, first_epoch=first_history_epoch,
        )

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
        epoch_lr = (curriculum_learning_rate_for_epoch(epoch, args, curriculum_state)
                    if curriculum_state is not None else learning_rate_for_epoch(
                        epoch=epoch,
                        total_epochs=args.epochs,
                        base_lr=args.lr,
                        warmup_epochs=args.warmup_epochs,
                        min_lr=args.min_lr,
                    ))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = epoch_lr
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if epoch_progress is not None:
            epoch_progress.set_postfix(phase="train", lr=f"{epoch_lr:.8g}")

        active_phase_d_stream = None
        if curriculum_state is not None:
            phase_index_at_start = curriculum_phase(
                epoch, curriculum_state, *curriculum_weights(curriculum_state),
                args.curriculum_config,
            )[0]
            if phase_index_at_start == 3:
                active_phase_d_stream = phase_d_stream

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
            student_t_df=args.student_t_df,
            curriculum_state=curriculum_state,
            curriculum_log_every=args.curriculum_log_every,
            phase_d_stream=active_phase_d_stream,
            curriculum_config=args.curriculum_config,
            temporal_extrapolation_continuation=(
                args.temporal_extrapolation_continuation
            ),
            temporal_extrapolation_only_continuation=(
                args.temporal_extrapolation_only_continuation
            ),
            uncertainty_feature_grad_continuation=(
                args.uncertainty_feature_grad_continuation
            ),
            joint_nll_grad_continuation=(
                args.joint_nll_grad_continuation
            ),
            joint_nll_grad=args.joint_nll_grad,
            gkl_continuation_20ep=getattr(args, "gkl_continuation_20ep", False),
            gkl_joint_grad_continuation_20ep=getattr(
                args, "gkl_joint_grad_continuation_20ep", False,
            ),
            differentiable_phase_c_reuse=args.differentiable_phase_c_reuse,
            mean_only_phase_c_reuse=args.mean_only_phase_c_reuse,
            student_t_unnormalized_control_20ep=getattr(
                args, "student_t_unnormalized_control_20ep", False,
            ),
            gkl_masking_continuation_20ep=getattr(
                args, "gkl_masking_continuation_20ep", False,
            ),
            phase_d_sliding_curriculum=getattr(args, "phase_d_sliding_curriculum", False),
            dataset=train_set,
        )

        rollout_validation = bool(
            curriculum_state is not None
            and curriculum_is_complete(curriculum_state)
            and (
                args.curriculum_config == "global_cosine"
                or getattr(args, "phase_d_sliding_curriculum", False)
                or (args.curriculum_config == "abcd_symmetric_lr"
                    and curriculum_state.get("phase_d_start_epoch") is None)
            )
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
            student_t_df=args.student_t_df,
            rollout_validation=rollout_validation,
        )
        sliding_val_metrics = None
        if getattr(args, "phase_d_sliding_curriculum", False) and is_main_process():
            sliding_val_metrics = validate_sliding_windows(
                model=(model.module if isinstance(model, DistributedDataParallel) else model),
                dataset=val_set,
                val_runs=val_runs,
                device=device,
                mean=mean,
                std=std,
                slide_steps=args.val_slide_steps,
                window_size=args.delta_t,
                amp=args.amp,
                amp_dtype=getattr(torch, args.amp_dtype),
                uncertainty_loss=args.uncertainty_loss,
                student_t_df=args.student_t_df,
                seed=args.val_mask_seed,
            )
        if distributed_is_initialized():
            dist.barrier()
        validation_seconds = time.perf_counter() - validation_started

        row = {
            "epoch": epoch,
            "learning_rate": epoch_lr,
            "train_mse": train_metrics["mse"],
            "val_mse": val_metrics["mean"]["mse"],
            **({"sliding_curriculum": train_metrics["sliding_curriculum"]}
               if "sliding_curriculum" in train_metrics else {}),
            **({"sliding_validation": sliding_val_metrics}
               if sliding_val_metrics is not None else {}),
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
            phase_index, phase_name = curriculum_phase(
                epoch, curriculum_state, lambda_v_end, lambda_r_end,
                args.curriculum_config,
            )
            row["curriculum"].update(
                phase_index=phase_index,
                phase_name=phase_name,
                phase_a_start_epoch=1,
                phase_b_start_epoch=curriculum_state.get("variance_phase_start_epoch"),
                phase_c_start_epoch=curriculum_state.get("rollout_phase_start_epoch"),
                phase_d_start_epoch=curriculum_state.get("phase_d_start_epoch"),
                convergence_rate_threshold=args.transition_rate_threshold,
                convergence_patience_target=args.transition_patience,
            )
            epoch_log("  curriculum (second-pass losses averaged over all batches, inactive = 0): "
                      + json.dumps(row["curriculum"]))
        if "phase_d" in train_metrics:
            row["phase_d"] = train_metrics["phase_d"]
            epoch_log("  Phase-D streaming diagnostics: " + json.dumps(row["phase_d"]))
        if args.uncertainty_aware:
            row["uncertainty_objective"] = args.uncertainty_loss
            nll_active = args.uncertainty_loss in (
                "gaussian_nll", "student_t_nll", "generalized_kl", "gkl",
            )
            key = args.uncertainty_loss if nll_active else "uncertainty_mse"
            row[f"train_{key}"] = train_metrics[key]
            row[f"val_{key}"] = val_metrics["mean"][key]
            if nll_active:
                row.update({key: train_metrics[key] for key in (
                    "gradient_norm_mean", "gradient_norm_max", "binary_input_variance_verified",
                    "finite_model_input_verified", "nonfinite_events", "optimizer_updates")})
                row["val_variance_statistics"] = val_metrics["variance_statistics"]
                row["val_per_beta"] = val_metrics["per_beta"]
                if rollout_validation:
                    row["val_rollout_mse"] = val_metrics["mean"]["rollout_mse"]
                    row[f"val_rollout_{key}"] = val_metrics["mean"][
                        f"rollout_{key}"
                    ]
        transition_diagnostic = None
        if is_main_process():
            history.append(row)
            if curriculum_state is not None and not args.post_final_continuation:
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
                              f"transition_monitor_active={'yes' if diagnostic['transition_monitor_active'] else 'no'} "
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
                        uncertainty_name = "gkl" if "ema_val_gkl" in diagnostic else "nll"
                        for name in ("ema_val_mse", "mse_rate_per_epoch",
                                     f"ema_val_{uncertainty_name}", f"{uncertainty_name}_rate_per_epoch"):
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
                    else:
                        diagnostic = phase_c_convergence(history, args, curriculum_state)
                        if diagnostic["phase_c"]:
                            row["phase_c_convergence"] = diagnostic
                            transition_diagnostic = diagnostic
                            fields = []
                            for name in (
                                "ema_val_rollout_mse", "rollout_mse_rate_per_epoch",
                                "ema_val_rollout_nll", "rollout_nll_rate_per_epoch",
                            ):
                                value = diagnostic[name]
                                text = (f"{100 * value:+.3f}%" if "rate" in name
                                        else f"{value:.8g}") if value is not None else "n/a"
                                fields.append(f"{name}={text}")
                            epoch_log(
                                "Phase C convergence: " + " ".join(fields)
                                + f" threshold={100 * args.transition_rate_threshold:.3f}% "
                                f"stable_count={diagnostic['stable_count']}/{args.transition_patience} "
                                f"segment_age={diagnostic['segment_age'] if diagnostic['segment_age'] is not None else 'n/a'}"
                                f"/{diagnostic['minimum_segment_age']:g} "
                                f"ready_for_phase_d={'yes' if diagnostic['ready_for_phase_d'] else 'no'}"
                            )
                            if diagnostic["ready_for_phase_d"]:
                                epoch_log("READY_FOR_PHASE_D=true")

        if curriculum_state is not None and not args.post_final_continuation:
            synchronized = [transition_diagnostic]
            if distributed_is_initialized():
                dist.broadcast_object_list(synchronized, src=0)
            transition_diagnostic = synchronized[0]
            transition = apply_ready_curriculum_transition(curriculum_state, transition_diagnostic)
            if transition is not None:
                ramp, start_step = transition
                curriculum_state[f"{ramp}_phase_start_epoch"] = epoch + 1
                if is_main_process():
                    row["automatic_curriculum_transition"] = {
                        "ramp": ramp,
                        "start_step": start_step,
                    }
                    epoch_log(f"Automatically armed {ramp} ramp at successful optimizer step {start_step}")
            if args.curriculum_config == "abcd_symmetric_lr":
                phase_d_started = apply_ready_phase_d_transition(
                    curriculum_state, transition_diagnostic, epoch, args.final_epoch,
                )
            elif args.curriculum_config == "adaptive":
                phase_d_started = start_phase_d_if_ready(
                    curriculum_state, epoch, args.final_epoch,
                )
            else:
                phase_d_started = False
            if phase_d_started:
                if is_main_process():
                    row["phase_d_transition"] = {
                        "start_epoch": curriculum_state["phase_d_start_epoch"],
                    }
                    epoch_log(
                        f"Phase D joint refinement starts at epoch "
                        f"{curriculum_state['phase_d_start_epoch']}"
                    )
            row["curriculum"].update(
                phase_b_start_epoch=curriculum_state.get("variance_phase_start_epoch"),
                phase_c_start_epoch=curriculum_state.get("rollout_phase_start_epoch"),
                phase_d_start_epoch=curriculum_state.get("phase_d_start_epoch"),
            )

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
            label = {
                "gaussian_nll": "Gaussian NLL",
                "student_t_nll": "Student-t NLL",
            }.get(args.uncertainty_loss, "uncertainty Smooth L1")
            epoch_log(f"  {label}: train={row[f'train_{key}']:.6f} val={row[f'val_{key}']:.6f}")
            if args.uncertainty_loss in ("gaussian_nll", "student_t_nll"):
                input_check = ("binary v_in verified" if row["binary_input_variance_verified"]
                               else "finite stateful Phase-D input verified")
                epoch_log(f"  synchronized preclip gradient norm: mean={row['gradient_norm_mean']:.6g} "
                          f"max={row['gradient_norm_max']:.6g}; {input_check}; nonfinite_events=0")
                epoch_log("  validation variance statistics: " + json.dumps(row["val_variance_statistics"]))
                epoch_log("  validation per beta: " + json.dumps(row["val_per_beta"]))
        epoch_log(f"  epoch_seconds={row['epoch_seconds']:.2f} "
                  f"AMP skipped rank-steps={row['amp_skipped_steps']:.0f} "
                  f"scale={row['amp_scale_start']:g}->{row['amp_scale_end']:g}")
        if is_main_process():
            save_history_atomic(history_path, history)
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
