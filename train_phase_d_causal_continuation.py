#!/usr/bin/env python3
"""Phase D Stateful Causal Training and Continuation Pilot for ai4plasma v11.

Implements sequential causal sliding-window training (stride +1) initialized from
the converged v11 checkpoint (final.pt) to train the model to effectively utilize
historical predictions, their uncertainties, and genuine probes during online deployment.

Key Features:
  - Architecture & Loss: UNet3D loaded with strict=True from v11 checkpoint;
    trained with Generalized KL uncertainty loss + joint gradient flow (detach_mean=False).
  - Trajectory Protocol: Consecutive physical frames advancing with stride +1.
    Context window T=24; rollout depth K=6 consecutive causal steps per update.
  - Deployment-matched Observations: 1,000 fixed spatial Density probes with u_in=0.0.
    Trained under both B_full (100% visible B) and B_hidden (unobserved B with u_in=1.0).
  - Uncertainty-aware Historical Mixture:
      * 50% Original predicted uncertainty: u_in = u_hat
      * 30% Moderately inflated uncertainty: u_in = max(u_hat, 0.1)
      * 20% High uncertainty robustness:    u_in = 1.0
  - Detached Recursive Feedback: Predictions and uncertainties are detached between steps
    to avoid backpropagation through long horizons while providing realistic input drift.
  - Multi-GPU DDP Support: DistributedDataParallel across 16 GPUs (4 nodes x 4 GPUs).
  - Checkpoint & Auto-Resume: Saves latest.pt and history.json at each epoch;
    recovers seamlessly if an allocation expires.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from data.masking import (
    full_mse_loss,
    predicted_error_mse_loss,
)
from data.vpic_hdf5_dataset import (
    load_fixed_split,
    VPICWindowDataset,
)
from make_paper_figures import (
    sliding_density_probe_mask,
)
from models.unet3d import UNet3D


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def rank_print(*args, **kwargs) -> None:
    if is_main_process():
        print(*args, **kwargs)


def initialize_distributed() -> Tuple[torch.device, int, int, int]:
    """Initialize torchrun DDP state, or fall back to single process."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase D stateful causal training continuation pilot for ai4plasma v11."
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=Path("runs/multi_beta_gkl_v11_fromscratch_256ep_20261122/final.pt"),
        help="Path to completed v11 source checkpoint.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/multi_beta_gkl_v11_phase_d_pilot_6ep_20261123"),
        help="Output directory for Phase D continuation run.",
    )
    parser.add_argument(
        "--h5-dir",
        type=Path,
        default=Path("/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta_official2500_none_compat"),
        help="Directory of HDF5 VPIC data.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("split.json"),
        help="Path to split.json.",
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2],
        help="Beta values to train on.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=6,
        help="Number of Phase D pilot continuation epochs (default: 6).",
    )
    parser.add_argument(
        "--updates-per-epoch",
        type=int,
        default=500,
        help="Optimizer updates per epoch (default: 500).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-6,
        help="Base fine-tuning learning rate (default: 2e-6).",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=2e-7,
        help="Minimum learning rate for cosine schedule (default: 2e-7).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=float,
        default=0.5,
        help="Warmup epochs for learning rate schedule (default: 0.5).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4).",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Maximum gradient norm (default: 1.0).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (number of parallel trajectories) per GPU (default: 2).",
    )
    parser.add_argument(
        "--rollout-depth",
        type=int,
        default=6,
        help="Consecutive causal steps (K) per trajectory update (default: 6).",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=24,
        help="Context window size (default: 24).",
    )
    parser.add_argument(
        "--density-probe-count",
        type=int,
        default=1000,
        help="Fixed spatial density probes per trajectory (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed base.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Enable automatic mixed precision (BF16 on GPU).",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=True,
        help="Automatically resume from latest.pt if present.",
    )
    parser.add_argument(
        "--val-runs-per-epoch",
        type=int,
        default=10,
        help="Number of validation runs evaluated per epoch (default: 10).",
    )
    return parser.parse_args()


def load_source_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[UNet3D, torch.Tensor, torch.Tensor, Dict]:
    """Instantiate UNet3D using exact arguments from checkpoint and load weights."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {checkpoint_path}")

    rank_print(f"Loading source checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt.get("args", {})
    if isinstance(args, argparse.Namespace):
        args = vars(args)

    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=int(args.get("base_channels", 24)),
        channel_mults=tuple(args.get("channel_mults", (1, 2, 4, 8))),
        architecture=str(args.get("model_version", "direct_unet3d_no_activation_norm_v3")),
        use_attention=bool(args.get("use_attention", True)),
        spatial_only_pooling=bool(args.get("spatial_only_pooling", True)),
        uncertainty_aware=bool(args.get("uncertainty_aware", True)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    stats = ckpt["stats"]
    mean = torch.tensor(stats["mean"], dtype=torch.float32, device=device).reshape(4, 1, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32, device=device).reshape(4, 1, 1, 1)

    return model, mean, std, args


def compute_learning_rate(
    global_step: int,
    total_steps: int,
    warmup_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if global_step < warmup_steps:
        return min_lr + (base_lr - min_lr) * (global_step / max(1, warmup_steps))
    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


class CausalTrajectoryManager:
    """Manages sampling and stateful causal rollouts for training trajectories."""

    def __init__(
        self,
        dataset: VPICWindowDataset,
        train_runs: Sequence[str],
        window_size: int,
        rollout_depth: int,
        probe_count: int,
        mean: torch.Tensor,
        std: torch.Tensor,
        device: torch.device,
        seed: int,
    ):
        self.dataset = dataset
        self.train_runs = list(train_runs)
        self.window_size = window_size
        self.rollout_depth = rollout_depth
        self.probe_count = probe_count
        self.mean = mean
        self.std = std
        self.device = device
        self.rng = random.Random(seed)
        self.torch_rng = torch.Generator(device=device).manual_seed(seed)

    def sample_trajectory_batch(
        self, batch_size: int
    ) -> List[Dict]:
        """Samples a batch of independent trajectory descriptors."""
        trajectories = []
        for _ in range(batch_size):
            run_name = self.rng.choice(self.train_runs)
            # Fetch physical run data: [4, L, X, Z]
            run_data = self.dataset.get_run(run_name).to(self.device, non_blocking=True)
            run_norm = (run_data - self.mean) / (self.std + 1e-8)
            L = int(run_norm.shape[1])

            # Select start frame t0 such that t0 + rollout_depth <= L
            max_t0 = max(0, L - self.rollout_depth)
            t0 = self.rng.randint(0, max_t0) if max_t0 > 0 else 0

            # 1,000 fixed spatial probes for this trajectory
            probe_mask, _ = sliding_density_probe_mask(
                run_norm.unsqueeze(0), self.probe_count
            )
            probe_bool = (probe_mask > 0.5)

            # Randomly select magnetic visibility: 50% B_full, 50% B_hidden
            hide_magnetic = (self.rng.random() < 0.5)

            # Global noise tensor for this trajectory: [4, L, H, W]
            traj_noise = torch.randn(
                run_norm.shape, generator=self.torch_rng, device=self.device, dtype=run_norm.dtype
            )

            # Pre-allocate trajectory prediction and variance buffers: [L, H, W]
            H, W = run_norm.shape[2], run_norm.shape[3]
            pred_buffer = torch.zeros((L, H, W), device=self.device, dtype=run_norm.dtype)
            var_buffer = torch.ones((L, H, W), device=self.device, dtype=run_norm.dtype)

            trajectories.append({
                "run_name": run_name,
                "target_norm": run_norm,
                "traj_noise": traj_noise,
                "probe_bool": probe_bool,
                "hide_magnetic": hide_magnetic,
                "t0": t0,
                "L": L,
                "pred_buffer": pred_buffer,
                "var_buffer": var_buffer,
            })
        return trajectories


def build_trajectory_step_input(
    t: int,
    traj: Dict,
    window_size: int = 24,
    uncertainty_mode: str = "original",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Constructs the [1, 8, window_size, H, W] input for physical frame t.

    Returns:
      model_input: [1, 8, window_size, H, W]
      target_frame: [1, 4, 1, H, W] ground truth at frame t
      visible_mask: [1, 4, 1, H, W] visible mask at frame t
    """
    target_norm = traj["target_norm"]
    traj_noise = traj["traj_noise"]
    probe_bool = traj["probe_bool"]
    hide_magnetic = traj["hide_magnetic"]
    pred_buffer = traj["pred_buffer"]
    var_buffer = traj["var_buffer"]

    device = target_norm.device
    dtype = target_norm.dtype
    _, _, H, W = target_norm.shape

    values = torch.zeros((1, 4, window_size, H, W), device=device, dtype=dtype)
    input_variance = torch.ones_like(values)

    for l in range(window_size):
        tau = t - window_size + 1 + l
        if tau < 0:
            # Deterministic noise padding for nonexistent historical frame
            g_pad = torch.Generator(device=device)
            g_pad.manual_seed((10007 * tau + 54321) & 0x7FFFFFFF)
            values[0, :, l] = torch.randn((4, H, W), generator=g_pad, device=device, dtype=dtype)
            input_variance[0, :, l] = 1.0
        else:
            # Valid frame tau in [0, t]
            # 1. Magnetic components
            if not hide_magnetic:
                values[0, :3, l] = target_norm[:3, tau]
                input_variance[0, :3, l] = 0.0
            else:
                values[0, :3, l] = traj_noise[:3, tau]
                input_variance[0, :3, l] = 1.0

            # 2. Density component
            if tau == t:
                # Current frame: probes known, unobserved is unknown noise with u_in=1.0
                values[0, 3, l] = traj_noise[3, tau]
                input_variance[0, 3, l] = 1.0
                values[0, 3, l, probe_bool] = target_norm[3, tau, probe_bool]
                input_variance[0, 3, l, probe_bool] = 0.0
            else:
                # Historical frame tau < t: reuse previous prediction
                values[0, 3, l] = pred_buffer[tau]

                # Assign historical uncertainty based on sampled mixture condition
                if uncertainty_mode == "original":
                    input_variance[0, 3, l] = var_buffer[tau]
                elif uncertainty_mode == "inflated":
                    input_variance[0, 3, l] = torch.clamp_min(var_buffer[tau], 0.1)
                elif uncertainty_mode == "high":
                    input_variance[0, 3, l] = 1.0
                else:
                    input_variance[0, 3, l] = var_buffer[tau]

                # Genuine probes always override with zero variance
                values[0, 3, l, probe_bool] = target_norm[3, tau, probe_bool]
                input_variance[0, 3, l, probe_bool] = 0.0

    model_input = torch.cat([values, input_variance], dim=1)

    # Current frame target and visible mask for supervision
    target_frame = target_norm[:, t : t + 1].unsqueeze(0)  # [1, 4, 1, H, W]
    visible_mask = torch.zeros_like(target_frame)
    if not hide_magnetic:
        visible_mask[0, :3, 0] = 1.0
    visible_mask[0, 3, 0, probe_bool] = 1.0

    return model_input, target_frame, visible_mask


def evaluate_validation(
    model: torch.nn.Module,
    dataset: VPICWindowDataset,
    val_runs: Sequence[str],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    probe_count: int = 1000,
    window_size: int = 24,
    max_runs: int = 10,
) -> Dict[str, float]:
    """Evaluates online causal rollout on validation runs."""
    model.eval()
    val_subset = list(val_runs)[:max_runs]

    mse_full_list = []
    mse_hidden_list = []

    with torch.no_grad():
        for run_name in val_subset:
            try:
                run_data = dataset.get_run(run_name).to(device)
            except Exception:
                continue
            run_norm = (run_data - mean) / (std + 1e-8)
            L = int(run_norm.shape[1])
            if L <= window_size:
                continue
            n_eval = L

            probe_mask, _ = sliding_density_probe_mask(run_norm.unsqueeze(0), probe_count)
            probe_bool = (probe_mask > 0.5)
            hidden_bool = ~probe_bool

            for hide_b in [False, True]:
                pred_buf = torch.zeros((n_eval, run_norm.shape[2], run_norm.shape[3]), device=device)
                var_buf = torch.ones_like(pred_buf)
                g = torch.Generator(device=device).manual_seed(9876)
                noise = torch.randn((4, n_eval, run_norm.shape[2], run_norm.shape[3]), generator=g, device=device)

                run_mse_settled = []

                for t in range(n_eval):
                    traj_stub = {
                        "target_norm": run_norm[:, :n_eval],
                        "traj_noise": noise,
                        "probe_bool": probe_bool,
                        "hide_magnetic": hide_b,
                        "pred_buffer": pred_buf,
                        "var_buffer": var_buf,
                    }
                    inp, _, _ = build_trajectory_step_input(
                        t, traj_stub, window_size=window_size, uncertainty_mode="original"
                    )
                    pred, var = model(inp)
                    pred_t = pred[0, 3, window_size - 1].float()
                    var_t = var[0, 3, window_size - 1].float()

                    if t >= window_size - 1:
                        err = (pred_t[hidden_bool] - run_norm[3, t, hidden_bool]).square().mean().item()
                        run_mse_settled.append(err)

                    pred_t = pred_t.clone()
                    var_t = var_t.clone()
                    pred_t[probe_bool] = run_norm[3, t, probe_bool]
                    var_t[probe_bool] = 0.0
                    pred_buf[t] = pred_t
                    var_buf[t] = var_t

                if run_mse_settled:
                    avg_mse = float(np.mean(run_mse_settled))
                    if not hide_b:
                        mse_full_list.append(avg_mse)
                    else:
                        mse_hidden_list.append(avg_mse)

    model.train()
    return {
        "val_mse_bfull": float(np.mean(mse_full_list)) if mse_full_list else float("nan"),
        "val_mse_bhidden": float(np.mean(mse_hidden_list)) if mse_hidden_list else float("nan"),
    }


def main():
    args = parse_args()
    device, rank, world_size, local_rank = initialize_distributed()

    out_dir = args.out_dir
    if is_main_process():
        out_dir.mkdir(parents=True, exist_ok=True)

    rank_print(f"--- ai4plasma v11: Phase D Stateful Causal Training Pilot ---")
    rank_print(f"World size: {world_size}, Device: {device}, Local rank: {local_rank}")
    rank_print(f"Output directory: {out_dir}")

    # Set seeds
    seed = args.seed + rank * 1000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 1. Load source model and stats from converged v11 checkpoint
    model, mean, std, source_args = load_source_model(args.source_checkpoint, device)

    # 2. Setup DDP
    if dist.is_initialized():
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # 3. Load split and initialize datasets
    with open(args.split) as f:
        split_data = json.load(f)
    train_runs = split_data.get("train_runs", [])
    val_runs = split_data.get("val_runs", [])

    rank_print(f"Loaded split: {len(train_runs)} train runs, {len(val_runs)} val runs")

    train_dataset = VPICWindowDataset(
        h5_dir=args.h5_dir,
        betas=args.betas,
        delta_t=args.window_size,
        stride_t=2,
        include_runs=train_runs,
    )
    val_dataset = VPICWindowDataset(
        h5_dir=args.h5_dir,
        betas=args.betas,
        delta_t=args.window_size,
        stride_t=2,
        include_runs=val_runs,
    )

    traj_manager = CausalTrajectoryManager(
        dataset=train_dataset,
        train_runs=train_runs,
        window_size=args.window_size,
        rollout_depth=args.rollout_depth,
        probe_count=args.density_probe_count,
        mean=mean,
        std=std,
        device=device,
        seed=seed,
    )

    # 4. Setup Optimizer and LR schedule
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = args.amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)

    total_steps = args.epochs * args.updates_per_epoch
    warmup_steps = int(args.warmup_epochs * args.updates_per_epoch)

    # 5. Handle Auto-Resume
    start_epoch = 1
    global_step = 0
    history = []

    latest_ckpt_path = out_dir / "latest.pt"
    if args.auto_resume and latest_ckpt_path.exists():
        rank_print(f"Resuming training from {latest_ckpt_path}...")
        ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", (start_epoch - 1) * args.updates_per_epoch)
        history = ckpt.get("history", [])
        rank_print(f"Resumed at epoch {start_epoch}, global step {global_step}")

    if is_main_process():
        with open(out_dir / "config.json", "w") as f:
            json.dump({**vars(args), "source_args": source_args}, f, indent=2, default=str)

    # Uncertainty mixture distribution
    # 50% original (u=u_hat), 30% inflated (u=max(u_hat, 0.1)), 20% high (u=1.0)
    unc_modes = ["original", "inflated", "high"]
    unc_weights = [0.50, 0.30, 0.20]

    rank_print(f"\nStarting Phase D causal training pilot from epoch {start_epoch} to {args.epochs}...")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_start_time = time.time()

        sum_loss = 0.0
        sum_mu_loss = 0.0
        sum_var_loss = 0.0
        n_updates = 0

        pbar = range(args.updates_per_epoch)
        if is_main_process():
            pbar = tqdm(pbar, desc=f"Epoch {epoch}/{args.epochs}")

        for step in pbar:
            global_step += 1
            current_lr = compute_learning_rate(
                global_step, total_steps, warmup_steps, args.lr, args.min_lr
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)

            # Sample batch of trajectories
            trajectories = traj_manager.sample_trajectory_batch(args.batch_size)

            batch_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            batch_mu = 0.0
            batch_var = 0.0

            for traj in trajectories:
                t0 = traj["t0"]
                # Sample uncertainty condition for this trajectory
                unc_mode = random.choices(unc_modes, weights=unc_weights, k=1)[0]

                traj_steps_loss = []

                # Execute K consecutive causal sliding steps
                for step_idx in range(args.rollout_depth):
                    t = t0 + step_idx
                    if t >= traj["L"]:
                        break

                    model_input, target_frame, visible_mask = build_trajectory_step_input(
                        t=t,
                        traj=traj,
                        window_size=args.window_size,
                        uncertainty_mode=unc_mode,
                    )

                    with torch.amp.autocast(device_type, enabled=amp_enabled, dtype=torch.bfloat16):
                        pred, variance = model(model_input)

                        # Primary supervision at target frame t (local index window_size - 1)
                        pred_t = pred[:, :, args.window_size - 1 : args.window_size]
                        var_t = variance[:, :, args.window_size - 1 : args.window_size]

                        mu_loss = full_mse_loss(pred_t, target_frame, visible_mask)
                        var_loss = predicted_error_mse_loss(
                            var_t, pred_t, target_frame,
                            objective="generalized_kl", student_t_df=3.0, detach_mean=False
                        )
                        step_loss = 0.5 * (mu_loss + var_loss)

                    traj_steps_loss.append(step_loss)
                    batch_mu += mu_loss.item()
                    batch_var += var_loss.item()

                    # Detach predictions for stateful causal rollout to next physical step
                    with torch.no_grad():
                        pred_detach = pred[0, 3, args.window_size - 1].detach().float().clone()
                        var_detach = variance[0, 3, args.window_size - 1].detach().float().clone()

                        # Genuine probes override
                        pred_detach[traj["probe_bool"]] = traj["target_norm"][3, t, traj["probe_bool"]]
                        var_detach[traj["probe_bool"]] = 0.0

                        traj["pred_buffer"][t] = pred_detach
                        traj["var_buffer"][t] = var_detach

                if traj_steps_loss:
                    batch_loss = batch_loss + torch.stack(traj_steps_loss).mean() / len(trajectories)

            # Check finite loss
            if not torch.isfinite(batch_loss):
                raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}, step {step}: {batch_loss.item()}")

            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not math.isfinite(grad_norm):
                rank_print(f"Warning: Non-finite gradient norm {grad_norm} at step {global_step}, skipping update.")
                scaler.update()
                continue

            scaler.step(optimizer)
            scaler.update()

            sum_loss += batch_loss.item()
            n_updates += 1

            if is_main_process() and hasattr(pbar, "set_postfix"):
                pbar.set_postfix({
                    "loss": f"{batch_loss.item():.4f}",
                    "lr": f"{current_lr:.2e}",
                    "gnorm": f"{grad_norm:.2f}",
                })

        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = sum_loss / max(1, n_updates)
        avg_mu = batch_mu / max(1, n_updates * args.batch_size * args.rollout_depth)
        avg_var = batch_var / max(1, n_updates * args.batch_size * args.rollout_depth)

        rank_print(
            f"Epoch {epoch:02d}/{args.epochs:02d} completed in {epoch_duration:.1f}s | "
            f"Train Loss: {avg_train_loss:.5f} (mu={avg_mu:.5f}, var={avg_var:.5f}) | "
            f"LR: {current_lr:.2e}"
        )

        # Validation evaluation
        val_metrics = {}
        if is_main_process():
            unwrapped = model.module if hasattr(model, "module") else model
            val_metrics = evaluate_validation(
                model=unwrapped,
                dataset=val_dataset,
                val_runs=val_runs,
                mean=mean,
                std=std,
                device=device,
                probe_count=args.density_probe_count,
                window_size=args.window_size,
                max_runs=args.val_runs_per_epoch,
            )
            rank_print(
                f"  Validation (Online Rollout): "
                f"B_full MSE={val_metrics.get('val_mse_bfull', float('nan')):.5f} | "
                f"B_hidden MSE={val_metrics.get('val_mse_bhidden', float('nan')):.5f}"
            )

        epoch_record = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "train_mu_loss": avg_mu,
            "train_var_loss": avg_var,
            "learning_rate": current_lr,
            "epoch_duration_seconds": epoch_duration,
            "global_step": global_step,
            **val_metrics,
        }
        history.append(epoch_record)

        # Save checkpoint
        if is_main_process():
            unwrapped = model.module if hasattr(model, "module") else model
            ckpt_state = {
                "epoch": epoch,
                "global_step": global_step,
                "model": unwrapped.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "stats": {"mean": mean.squeeze().tolist(), "std": std.squeeze().tolist()},
                "args": vars(args),
                "history": history,
            }
            torch.save(ckpt_state, latest_ckpt_path)
            with open(out_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

            if epoch == args.epochs:
                final_ckpt_path = out_dir / "final.pt"
                torch.save(ckpt_state, final_ckpt_path)
                rank_print(f"Saved final Phase D checkpoint to {final_ckpt_path}")

    rank_print(f"\nPhase D continuation pilot complete! Checkpoint saved in {out_dir}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
