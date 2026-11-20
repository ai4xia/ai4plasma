import sys
from statistics import median
from time import perf_counter
import torch
import torch.nn as nn

from data.masking import DEFAULT_PATTERN_WEIGHTS, full_mse_loss, make_uncertain_input, predicted_error_mse_loss
from models.unet3d import DIRECT_MODEL_VERSION, UNet3D
from train_masked_unet3d import curriculum_forward_loss, make_phase_c_reuse_input, sample_training_mask


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


def run_fresh_gkl_smoke_test():
    assert torch.cuda.is_available()
    torch.manual_seed(1234)
    device = torch.device("cuda")
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=24,
        channel_mults=(1, 2, 4, 8), architecture=DIRECT_MODEL_VERSION,
        use_attention=True, spatial_only_pooling=True, uncertainty_aware=True,
    ).to(device)
    target = torch.randn(2, 4, 8, 32, 32, device=device)
    target[..., 0] = 0  # Exact zero residuals at random initialization.
    mask = torch.ones_like(target)
    mask[:, :, 4:] = 0
    x = torch.cat([make_uncertain_input(target, 1 - mask), 1 - mask], dim=1)
    for phase, lv, lr in [("A", 0.0, 0.0), ("B", 0.5, 0.0), ("C", 1.0, 1.0)]:
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, mu, u, parts = curriculum_forward_loss(
                model, x, target, mask, lv, lr, (1 - mask,) if lr else None,
                uncertainty_loss="generalized_kl", curriculum_config="three_phase_global_cosine",
                detach_variance_features=False, detach_mean=False,
                reuse_gradient_scale=lr,
            )
        assert torch.isfinite(loss) and "gkl_alpha" not in parts
        assert parts["forwards"] == (2 if lr else 1)
        if lv:
            for grad in torch.autograd.grad(
                parts["L_var_0"],
                (model.out.weight, model.variance_out.weight, model.enc0.conv1.weight),
                retain_graph=True,
            ):
                assert torch.isfinite(grad).all() and torch.count_nonzero(grad)
        if not lv:
            torch.testing.assert_close(loss, parts["L_mu_0"], atol=0, rtol=0)
            assert not parts["L_var_0"].requires_grad
        elif not lr:
            torch.testing.assert_close(loss, (parts["L_mu_0"] + lv * parts["L_var_0"]) / (1 + lv))
        loss.backward()
        assert (model.variance_out.weight.grad is None) == (lv == 0)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        print(f"Fresh GKL phase {phase}: finite BF16 loss={loss.item():.6f}, finite gradients")

    exact = (1 - mask).bool()
    reference = None
    for scale in (1.0, 0.5, 0.0):
        second_pass = []
        hook = model.register_forward_pre_hook(lambda _, inputs: second_pass.append(inputs[0]))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Hold the second-pass loss weight fixed to isolate the reuse multiplier.
                _, mu, u, parts = curriculum_forward_loss(
                    model, x, target, mask, 1.0, 1.0, (1 - mask,),
                    uncertainty_loss="generalized_kl", curriculum_config="three_phase_global_cosine",
                    detach_variance_features=False, detach_mean=False, reuse_gradient_scale=scale,
                )
        finally:
            hook.remove()
        values, variance = second_pass[1].chunk(2, dim=1)
        torch.testing.assert_close(values, torch.where(exact, target, mu), atol=0, rtol=0)
        torch.testing.assert_close(variance, torch.where(exact, 0., u), atol=0, rtol=0)
        cross_grads = torch.autograd.grad(parts["L_mu_1"] + parts["L_var_1"], (mu, u),
                                          allow_unused=True)
        if scale == 1:
            reference = cross_grads
        if scale == 0:
            assert cross_grads == (None, None)
        else:
            for actual, full in zip(cross_grads, reference):
                assert torch.isfinite(actual).all() and actual[~exact].count_nonzero()
                assert actual[exact].count_nonzero() == 0
                torch.testing.assert_close(actual, full * scale)
    print("Fresh GKL forward reuse and both cross-forward gradient scales 0/0.5/1: PASS")


def run_differentiable_reuse_smoke(mean_only=False):
    assert torch.cuda.is_available()
    torch.manual_seed(1234)
    device = torch.device("cuda")
    source = "runs/multi_beta_three_phase_joint_nll_grad_globalcosine_256ep_20261117/final.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 256
    args = checkpoint["args"]
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=args["base_channels"],
        channel_mults=args["channel_mults"], architecture=args["model_version"],
        use_attention=args["use_attention"], spatial_only_pooling=args["spatial_only_pooling"],
        uncertainty_aware=True,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    def batch(shape):
        target = torch.randn(shape, device=device)
        mask, reuse = (sample_training_mask(target, DEFAULT_PATTERN_WEIGHTS, .05, .05, .05, .05)[0]
                       for _ in range(2))
        x = torch.cat([make_uncertain_input(target, 1 - mask), 1 - mask], dim=1)
        return x, target, mask, reuse

    def forward(inputs, alpha, scale, mean_only=False):
        x, target, mask, reuse = inputs
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return curriculum_forward_loss(
                model, x, target, mask, 1.0, 1.0, (reuse,),
                uncertainty_loss="generalized_kl", curriculum_config="three_phase_global_cosine",
                detach_variance_features=False, detach_mean=False,
                gkl_alpha=alpha, reuse_gradient_scale=scale,
                mean_only_phase_c_reuse=mean_only,
            )

    inputs = batch((2, 4, 8, 32, 32))
    _, target, _, reuse = inputs
    exact = reuse.bool()
    assert exact.any() and (~exact).any()
    record = []
    hook = model.register_forward_hook(lambda _m, x, outputs: record.append((x[0], *outputs)))
    try:
        for alpha in (0.0, 0.5, 1.0):
            # Hold the objective fixed while comparing routes at a given alpha.
            # Use pure GKL in the scaling check below to isolate the gradient multiplier.
            model.zero_grad(set_to_none=True)
            control, _, _, _ = forward(inputs, alpha, 0.0)
            baseline = tuple(t.detach() for t in record[1])
            record.clear()
            loss, _, _, _ = forward(inputs, alpha, alpha, mean_only)
            for actual, expected in zip(record[1], baseline):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            torch.testing.assert_close(loss, control, atol=0, rtol=0)
            reuse_values, reuse_u = record[1][0].chunk(2, dim=1)
            assert torch.equal(reuse_values[exact], target[exact])
            assert torch.count_nonzero(reuse_u[exact]) == 0
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            record.clear()
            if mean_only:
                full, _, _, _ = forward(inputs, alpha, alpha)
                for actual, expected in zip(record[1], baseline):
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                torch.testing.assert_close(full, loss, atol=0, rtol=0)
                record.clear()
        reference = {}
        for scale in (1.0, 0.5, 0.0):
            _, mu0, u0, parts = forward(inputs, 1.0, scale, mean_only)
            mu1, u1 = record[1][1:]
            if mean_only:
                for key, outputs in (("L_var_0", (mu0, u0)), ("L_var_1", (mu1, u1))):
                    for grad in torch.autograd.grad(parts[key], outputs, retain_graph=True):
                        assert torch.isfinite(grad).all() and grad.count_nonzero()
            for key in ("L_mu_1", "L_var_1"):
                grads = torch.autograd.grad(
                    parts[key], (mu0, u0, mu1, u1), retain_graph=True, allow_unused=True,
                )
                if scale == 1:
                    reference[key] = grads
                for route, (actual, full) in enumerate(zip(grads[:2], reference[key][:2])):
                    if scale == 0 or (mean_only and route == 1):
                        assert actual is None
                    else:
                        assert torch.isfinite(actual).all() and actual[~exact].count_nonzero()
                        assert actual[exact].count_nonzero() == 0
                        torch.testing.assert_close(actual, full * scale)
                for actual, full in zip(grads[2:], reference[key][2:]):
                    if full is None:
                        assert actual is None
                    else:
                        torch.testing.assert_close(actual, full, atol=0, rtol=0)
            record.clear()
    finally:
        hook.remove()
    routes = "mean reuse with blocked variance" if mean_only else "mean/variance reuse"
    print(f"Forward equivalence, exact observation preservation, 0/0.5/1 {routes} "
          "gradients, direct second-pass gradients, and finite BF16 backward: PASS", flush=True)
    if mean_only:
        return

    # Full batch dimensions, synthetic normalized values, same model and masks
    # for both routes. Time synchronized forward+backward without optimizer/I/O.
    del loss, control, baseline, parts, reference, grads, mu0, u0, mu1, u1, reuse_values, reuse_u, actual, full
    model.zero_grad(set_to_none=True)
    inputs = batch((4, 4, 24, 154, 62))
    samples = {0.0: [], 1.0: []}
    for repeat in range(14):
        for scale in ((0.0, 1.0) if repeat % 2 == 0 else (1.0, 0.0)):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            loss, _, _, _ = forward(inputs, 1.0, scale)
            loss.backward()
            torch.cuda.synchronize()
            elapsed_ms = 1000 * (perf_counter() - start)
            peak_mib = torch.cuda.max_memory_allocated() / 2**20
            if repeat >= 4:
                samples[scale].append((elapsed_ms, peak_mib))
    for scale, measurements in samples.items():
        print(f"Reuse scale={scale:g}: median {median(t for t, _ in measurements):.2f} ms, "
              f"peak allocated {max(m for _, m in measurements):.1f} MiB, n={len(measurements)}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--fresh-gkl"]:
        run_fresh_gkl_smoke_test()
    elif sys.argv[1:] == ["--differentiable-reuse"]:
        run_differentiable_reuse_smoke()
    elif sys.argv[1:] == ["--mean-only-reuse"]:
        run_differentiable_reuse_smoke(mean_only=True)
    else:
        run_cuda_smoke_test()
