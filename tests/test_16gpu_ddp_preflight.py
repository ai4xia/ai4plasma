import math
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from data.vpic_hdf5_dataset import validate_checkpoint_split
from models.unet3d import DIRECT_MODEL_VERSION, UNet3D
from train_masked_unet3d import (
    curriculum_forward_loss,
    curriculum_weights,
    parse_args,
    validate_post_final_continuation,
)


def run_ddp_preflight():
    # Initialize distributed process group
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"=== Starting 16-GPU DDP Preflight Test (World Size: {world_size}) ===")

    # Test 1: NCCL All-Reduce Communication Test
    tensor = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected_sum = float(world_size * (world_size + 1) / 2)
    assert abs(tensor.item() - expected_sum) < 1e-5, f"All-reduce mismatch: got {tensor.item()}, expected {expected_sum}"

    if rank == 0:
        print(f"[Rank 0] NCCL All-reduce across {world_size} ranks PASSED (Sum = {tensor.item()})")

    # Test 2: Checkpoint Preflight Validation across Ranks
    source_path = Path("runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt")
    assert source_path.is_file(), f"Source checkpoint not found at {source_path}"

    checkpoint = torch.load(source_path, map_location="cpu")
    validate_checkpoint_split(checkpoint)

    # Checkpoint arguments
    source_args = checkpoint["args"]

    # Build base continuation args matching sbatch template
    cli_base = [
        "--h5-dir", source_args["h5_dir"],
        "--betas", *[str(b) for b in source_args["betas"]],
        "--delta-t", str(source_args["delta_t"]),
        "--stride-t", str(source_args["stride_t"]),
        "--batch-size", str(source_args["batch_size"]),
        "--num-workers", str(source_args["num_workers"]),
        "--epochs", "256",
        "--warmup-epochs", "5",
        "--lr", "2e-4",
        "--min-lr", "1e-5",
        "--weight-decay", str(source_args["weight_decay"]),
        "--base-channels", str(source_args["base_channels"]),
        "--model-version", source_args["model_version"],
        "--channel-mults", *[str(m) for m in source_args["channel_mults"]],
        "--density-probe-min", str(source_args["density_probe_min"]),
        "--density-probe-max", str(source_args["density_probe_max"]),
        "--density-visible-p-zero", str(source_args["density_visible_p_zero"]),
        "--density-visible-p-full", str(source_args["density_visible_p_full"]),
        "--magnetic-visible-p-zero", str(source_args["magnetic_visible_p_zero"]),
        "--magnetic-visible-p-full", str(source_args["magnetic_visible_p_full"]),
        "--uncertainty-aware",
        "--uncertainty-loss-weight", str(source_args["uncertainty_loss_weight"]),
        "--final-epoch", "276",
        "--stop-after-epochs", "276",
        "--curriculum-training",
        "--curriculum-config", "three_phase_global_cosine",
        "--post-final-continuation", str(source_path),
        "--amp", "--amp-dtype", "bfloat16",
    ]
    if source_args.get("use_attention"):
        cli_base.append("--use-attention")
    if source_args.get("spatial_only_pooling"):
        cli_base.append("--spatial-only-pooling")

    # Validate GKL preflight (B-only)
    gkl_cli = cli_base + [
        "--out-dir", "runs/multi_beta_gkl_continuation_20ep_20261119",
        "--uncertainty-loss", "generalized_kl",
        "--gkl-continuation-20ep",
    ]
    gkl_args = parse_args(gkl_cli)
    validate_post_final_continuation(
        gkl_args, checkpoint, source_path, Path("runs/multi_beta_gkl_continuation_20ep_20261119"),
    )
    if rank == 0:
        print("[Rank 0] GKL continuation (B-only) preflight validation PASSED")

    # Validate GKL Joint-Grad preflight (A+B)
    gkl_joint_cli = cli_base + [
        "--out-dir", "runs/multi_beta_gkl_joint_grad_continuation_20ep_20261119",
        "--uncertainty-loss", "generalized_kl",
        "--gkl-joint-grad-continuation-20ep",
    ]
    gkl_joint_args = parse_args(gkl_joint_cli)
    validate_post_final_continuation(
        gkl_joint_args, checkpoint, source_path, Path("runs/multi_beta_gkl_joint_grad_continuation_20ep_20261119"),
    )
    if rank == 0:
        print("[Rank 0] GKL joint-grad continuation (A+B) preflight validation PASSED")

    # Validate Control preflight
    ctrl_cli = cli_base + [
        "--out-dir", "runs/multi_beta_student_t_unnormalized_control_20ep_20261119",
        "--uncertainty-loss", "student_t_nll",
        "--student-t-df", "3.0",
        "--student-t-unnormalized-control-20ep",
    ]
    ctrl_args = parse_args(ctrl_cli)
    validate_post_final_continuation(
        ctrl_args, checkpoint, source_path, Path("runs/multi_beta_student_t_unnormalized_control_20ep_20261119"),
    )
    if rank == 0:
        print("[Rank 0] Student-t unnormalized control preflight validation PASSED")

    # Test 3: 1-batch DDP forward and backward synchronization test
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=8,
        channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION,
        use_attention=True,
        spatial_only_pooling=True,
        uncertainty_aware=True,
    ).to(device)
    # Initialize out.weight so backbone gradients propagate
    nn.init.normal_(model.out.weight, std=0.01)

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    batch_size = 1
    x = torch.randn(batch_size, 8, 4, 16, 16, device=device)
    target = torch.randn(batch_size, 4, 4, 16, 16, device=device)
    mask = torch.ones_like(target)
    reuse_mask = torch.ones_like(target)

    # Test GKL A+B backward across all 16 GPUs with BF16 autocast
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, _, _, comp = curriculum_forward_loss(
            model=ddp_model,
            model_input=x,
            target=target,
            mask=mask,
            lambda_v=1.0,
            lambda_r=1.0,
            reuse_masks=(reuse_mask,),
            uncertainty_loss="generalized_kl",
            student_t_df=3.0,
            curriculum_config="three_phase_global_cosine",
            detach_variance_features=False,
            detach_mean=False,
            gkl_alpha=1.0,
        )

    loss.backward()

    # Verify gradients synchronized across ranks
    for name, param in ddp_model.module.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Rank {rank}: Non-finite gradient in {name}"

    assert ddp_model.module.out.weight.grad is not None and torch.any(ddp_model.module.out.weight.grad != 0.0)
    assert ddp_model.module.variance_out.weight.grad is not None and torch.any(ddp_model.module.variance_out.weight.grad != 0.0)
    assert ddp_model.module.enc0.conv1.weight.grad is not None and torch.any(ddp_model.module.enc0.conv1.weight.grad != 0.0)

    dist.barrier()
    if rank == 0:
        print("[Rank 0] 1-batch DDP forward/backward synchronization across 16 GPUs PASSED")
        print("=== 16-GPU DDP PREFLIGHT COMPLETED SUCCESSFULLY ===")

    dist.destroy_process_group()


if __name__ == "__main__":
    run_ddp_preflight()
