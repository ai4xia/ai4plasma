import sys
import torch
import torch.nn as nn

from data.masking import full_mse_loss, predicted_error_mse_loss
from models.unet3d import DIRECT_MODEL_VERSION, UNet3D
from train_masked_unet3d import curriculum_forward_loss, make_phase_c_reuse_input


def run_cuda_smoke_test():
    if not torch.cuda.is_available():
        print("CUDA not available! Aborting.")
        sys.exit(1)

    device = torch.device("cuda:0")
    print(f"Running CUDA smoke test on device: {torch.cuda.get_device_name(device)}")

    # Instantiate model on CUDA
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=24,
        channel_mults=(1, 2, 4, 8),
        architecture=DIRECT_MODEL_VERSION,
        use_attention=True,
        spatial_only_pooling=True,
        uncertainty_aware=True,
    ).to(device)

    # Initialize out.weight with small non-zero values to verify gradient propagation
    nn.init.normal_(model.out.weight, std=0.01)

    batch_size = 2
    # Standard spatial dimensions for 3D UNet (depth=8, height=32, width=32)
    x = torch.randn(batch_size, 8, 8, 32, 32, device=device)
    target = torch.randn(batch_size, 4, 8, 32, 32, device=device)
    mask = torch.ones_like(target)
    reuse_mask = torch.ones_like(target)
    reuse_mask[:, :, 4:] = 0.0  # Hide second half

    print("\n--- Verifying Phase C 2-Pass with BF16 Autocast for alpha in {0.0, 0.5, 1.0} (GKL A+B) ---")

    for alpha in [0.0, 0.5, 1.0]:
        model.zero_grad()
        # For GKL A+B, Path A is ON throughout: detach_mean=False
        detach_mean = False

        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _, _, comp = curriculum_forward_loss(
                model=model,
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
                detach_mean=detach_mean,
                gkl_alpha=alpha,
            )

        print(f"GKL A+B alpha={alpha:.1f}:")
        print(f"  total loss: {loss.item():.6f}")
        print(f"  L_mu_0: {comp['L_mu_0'].item():.6f}, L_var_0: {comp['L_var_0'].item():.6f}")
        print(f"  L_mu_1: {comp['L_mu_1'].item():.6f}, L_var_1: {comp['L_var_1'].item():.6f}")
        print(f"  forwards: {comp['forwards']}")

        # Verify loss finiteness
        assert torch.isfinite(loss).all(), f"Non-finite loss at alpha={alpha}"
        assert torch.isfinite(comp["L_mu_0"]).all()
        assert torch.isfinite(comp["L_var_0"]).all()
        assert torch.isfinite(comp["L_mu_1"]).all()
        assert torch.isfinite(comp["L_var_1"]).all()

        # Backward
        loss.backward()

        # Check gradients
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Non-finite grad in {name} at alpha={alpha}"
                grad_norms[name] = param.grad.norm().item()

        print(f"  mean head out.weight grad norm: {grad_norms.get('out.weight', 0.0):.6f}")
        print(f"  variance head out.weight grad norm: {grad_norms.get('variance_out.weight', 0.0):.6f}")
        print(f"  backbone enc0.conv1 grad norm: {grad_norms.get('enc0.conv1.weight', 0.0):.6f}")
        print(f"  backbone up0 grad norm: {grad_norms.get('up0.conv.conv1.weight', 0.0):.6f}")

        assert grad_norms.get("out.weight", 0.0) > 0.0
        assert grad_norms.get("variance_out.weight", 0.0) > 0.0
        assert grad_norms.get("enc0.conv1.weight", 0.0) > 0.0

    print("\n--- Verifying Forward Equivalence: GKL A+B vs B-only at alpha=1.0 ---")
    model.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss_ab, _, _, _ = curriculum_forward_loss(
            model=model, model_input=x, target=target, mask=mask,
            lambda_v=1.0, lambda_r=1.0, reuse_masks=(reuse_mask,),
            uncertainty_loss="generalized_kl", student_t_df=3.0,
            curriculum_config="three_phase_global_cosine",
            detach_variance_features=False, detach_mean=False,
            gkl_alpha=1.0,
        )
        loss_b, _, _, _ = curriculum_forward_loss(
            model=model, model_input=x, target=target, mask=mask,
            lambda_v=1.0, lambda_r=1.0, reuse_masks=(reuse_mask,),
            uncertainty_loss="generalized_kl", student_t_df=3.0,
            curriculum_config="three_phase_global_cosine",
            detach_variance_features=False, detach_mean=True,
            gkl_alpha=1.0,
        )
    torch.testing.assert_close(loss_ab, loss_b)
    print(f"Forward equivalence verified: loss_ab={loss_ab.item():.6f}, loss_b={loss_b.item():.6f}")

    print("\n--- Verifying Exact Mathematical Equivalence at Boundary Conditions ---")
    # alpha=0 must match student_t_nll
    model.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss_gkl_a0, _, _, comp_gkl_a0 = curriculum_forward_loss(
            model=model, model_input=x, target=target, mask=mask,
            lambda_v=1.0, lambda_r=0.0,
            uncertainty_loss="generalized_kl", student_t_df=3.0,
            curriculum_config="three_phase_global_cosine",
            detach_variance_features=False, detach_mean=False,
            gkl_alpha=0.0,
        )
        loss_st, _, _, comp_st = curriculum_forward_loss(
            model=model, model_input=x, target=target, mask=mask,
            lambda_v=1.0, lambda_r=0.0,
            uncertainty_loss="student_t_nll", student_t_df=3.0,
            curriculum_config="three_phase_global_cosine",
            detach_variance_features=False, detach_mean=False,
            gkl_alpha=None,
        )
    torch.testing.assert_close(loss_gkl_a0, loss_st)
    torch.testing.assert_close(comp_gkl_a0["L_var_0"], comp_st["L_var_0"])
    print("Exact equivalence at alpha=0 (Student-t NLL) PASSED!")

    # Phase C detached reuse check
    target_c = torch.randn(batch_size, 4, 8, 32, 32, device=device)
    pred0 = torch.randn(batch_size, 4, 8, 32, 32, device=device, requires_grad=True)
    var0 = (torch.rand(batch_size, 4, 8, 32, 32, device=device) + 0.1).requires_grad_(True)
    reuse_m = torch.ones_like(target_c)
    reuse_m[:, :, 4:] = 0.0
    reuse_input, exact_mask = make_phase_c_reuse_input(target_c, pred0, var0, reuse_m)
    assert not reuse_input.requires_grad, "reuse_input must not require grad"
    print("Phase C detached reuse on CUDA PASSED!")

    print("\nALL CUDA SMOKE TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_cuda_smoke_test()
