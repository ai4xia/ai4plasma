import math
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from data.masking import make_uncertain_input
from data.vpic_hdf5_dataset import validate_checkpoint_split
from models.unet3d import DIRECT_MODEL_VERSION, UNet3D
from train_masked_unet3d import (
    curriculum_forward_loss,
    curriculum_weights,
    parse_args,
    validate_post_final_continuation,
)


def check_fresh_gkl_ddp(local_rank, rank, device, differentiable_reuse=False, mean_only=False):
    assert dist.get_world_size() == 16, "GKL preflight requires all 16 ranks"
    torch.manual_seed(1234 + rank)
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=24, channel_mults=(1, 2, 4, 8),
        architecture=DIRECT_MODEL_VERSION, use_attention=True, spatial_only_pooling=True,
        uncertainty_aware=True,
    ).to(device)
    target = torch.randn(1, 4, 8, 32, 32, device=device)
    target[..., 0] = 0  # Exact zero residuals with the default zero mean readout.
    mask = torch.ones_like(target)
    mask[:, :, 4:] = 0
    reuse_mask = 1 - mask
    x = torch.cat([make_uncertain_input(target, 1 - mask), 1 - mask], dim=1)

    def forward(net, lv, lr, alpha=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return curriculum_forward_loss(
                net, x, target, mask, lv, lr, (reuse_mask,) if lr else None,
                uncertainty_loss="generalized_kl", curriculum_config="three_phase_global_cosine",
                detach_variance_features=False, detach_mean=False,
                gkl_alpha=alpha,
                reuse_gradient_scale=lr if not differentiable_reuse else (alpha or 0.0),
                mean_only_phase_c_reuse=mean_only,
            )

    # Isolate GKL from MSE before installing the DDP reducer.
    _, mu, u, parts = forward(model, 1.0, 0.5)
    for grad in torch.autograd.grad(
        parts["L_var_0"],
        (model.out.weight, model.variance_out.weight, model.enc0.conv1.weight),
        retain_graph=True,
    ):
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad), f"Rank {rank}: GKL gradient"
    cross_grad = torch.autograd.grad(
        parts["L_mu_1"] + parts["L_var_1"], (mu, u), allow_unused=True,
    )
    if differentiable_reuse:
        assert cross_grad == (None, None), f"Rank {rank}: control reuse graph was not detached"
    else:
        for grad in cross_grad:
            assert grad is not None and torch.isfinite(grad).all() and grad.count_nonzero()
    del cross_grad
    del mu, u, parts
    if mean_only:
        _, mu, u, parts = forward(model, 1.0, 1.0, 1.0)
        grad_mean, grad_u = torch.autograd.grad(
            parts["L_mu_1"] + parts["L_var_1"], (mu, u), allow_unused=True,
        )
        assert grad_mean is not None and torch.isfinite(grad_mean).all()
        assert grad_mean.count_nonzero() and grad_u is None
        del mu, u, parts, grad_mean

    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    cases = (
        [(f"C alpha={alpha}", 1.0, 1.0, alpha) for alpha in (0.0, 0.5, 1.0)]
        if differentiable_reuse else
        [("A", 0.0, 0.0, None), ("B", 0.5, 0.0, None),
         ("C lambda_r=0.5", 1.0, 0.5, None), ("C lambda_r=1", 1.0, 1.0, None)]
    )
    for phase, lv, lr, alpha in cases:
        ddp.zero_grad(set_to_none=True)
        loss, mu, u, parts = forward(ddp, lv, lr, alpha)
        assert torch.isfinite(loss) and torch.isfinite(mu).all() and torch.isfinite(u).all()
        assert parts["forwards"] == (2 if lr else 1)
        assert ("gkl_alpha" in parts) == differentiable_reuse
        loss.backward()
        assert (model.variance_out.weight.grad is None) == (lv == 0)
        params = list(model.parameters())
        present = torch.tensor([p.grad is not None for p in params], device=device)
        reference = present.clone()
        dist.broadcast(reference, src=0)
        assert torch.equal(present, reference), f"Rank {rank}: gradient presence differs in phase {phase}"
        grads = torch.cat([p.grad.flatten() for p in params if p.grad is not None])
        reference = grads.clone()
        dist.broadcast(reference, src=0)
        assert torch.isfinite(grads).all() and torch.equal(grads, reference), (
            f"Rank {rank}: nonfinite or unsynchronized gradients in phase {phase}"
        )
        if rank == 0:
            print(f"GKL phase {phase}: finite BF16 loss and identical gradients on 16 ranks", flush=True)
    dist.barrier()
    if rank == 0:
        label = "Differentiable GKL reuse" if differentiable_reuse else "Fresh fully differentiable GKL A+B"
        if mean_only:
            label = "Mean-only GKL reuse"
        print(f"{label} DDP preflight: PASS on all 16 ranks", flush=True)


def run_ddp_preflight(fresh_gkl=False, differentiable_reuse=False, mean_only=False):
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

    if fresh_gkl or differentiable_reuse or mean_only:
        try:
            check_fresh_gkl_ddp(local_rank, rank, device, differentiable_reuse or mean_only, mean_only)
        finally:
            dist.destroy_process_group()
        return

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
    run_ddp_preflight(
        fresh_gkl=sys.argv[1:] == ["--fresh-gkl"],
        differentiable_reuse=sys.argv[1:] == ["--differentiable-reuse"],
        mean_only=sys.argv[1:] == ["--mean-only-reuse"],
    )
