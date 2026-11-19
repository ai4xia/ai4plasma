import math
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F

from data.masking import (
    DEFAULT_PATTERN_WEIGHTS,
    GAUSSIAN_NLL_VARIANCE_FLOOR,
    full_mse_loss,
    make_uncertain_input,
    predicted_error_mse_loss,
)
from models.unet3d import DIRECT_MODEL_VERSION, UNet3D
from train_masked_unet3d import (
    curriculum_forward_loss,
    curriculum_learning_rate_for_epoch,
    make_curriculum_state,
    make_phase_c_reuse_input,
    parse_args,
    validate_post_final_continuation,
    validate_resume_parameters,
)


# ---------------------------------------------------------------------------
# 1. Mathematical verification
# ---------------------------------------------------------------------------


def test_gkl_finite_at_zero_and_continuous_extension():
    u = torch.tensor([0.05, 0.5, 2.0], dtype=torch.float32, requires_grad=True)
    target = torch.zeros(3, dtype=torch.float32)
    pred = torch.zeros(3, dtype=torch.float32)

    loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl")
    assert torch.isfinite(loss).all()
    # At q=0: D(0||u) = u, so mean(u)
    expected = u.mean()
    torch.testing.assert_close(loss, expected)

    loss.backward()
    # grad w.r.t u at q=0: d/du (u - 0) = 1 (scaled by 1/N)
    torch.testing.assert_close(u.grad, torch.full_like(u, 1.0 / 3.0))


def test_gkl_zero_at_u_equals_q():
    q_vals = torch.tensor([0.01, 0.1, 0.5, 1.0, 5.0], dtype=torch.float32)
    u = q_vals.clone().requires_grad_(True)
    target = torch.zeros_like(q_vals)
    pred = torch.sqrt(q_vals)

    loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl")
    assert abs(loss.item()) < 1e-7


def test_gkl_gradient_vanishes_at_u_equals_q():
    q_vals = torch.tensor([0.05, 0.2, 1.5], dtype=torch.float32)
    u = q_vals.clone().requires_grad_(True)
    target = torch.zeros_like(q_vals)
    pred = torch.sqrt(q_vals)

    loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl")
    loss.backward()
    torch.testing.assert_close(u.grad, torch.zeros_like(u), atol=1e-7, rtol=1e-7)


def test_gkl_gradient_pushes_u_toward_q():
    q = torch.tensor([1.0, 1.0], dtype=torch.float32)
    # Case 1: u < q -> gradient w.r.t u is negative (reducing loss requires increasing u)
    # Case 2: u > q -> gradient w.r.t u is positive (reducing loss requires decreasing u)
    u = torch.tensor([0.5, 2.0], dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(q)
    pred = torch.sqrt(q)

    loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl")
    loss.backward()
    assert u.grad[0] < 0.0, "Expected negative gradient when u < q"
    assert u.grad[1] > 0.0, "Expected positive gradient when u > q"


def test_gkl_log_u_derivative():
    q = torch.tensor([0.2, 0.8, 1.5], dtype=torch.float32)
    log_u = torch.tensor([-1.0, 0.0, 0.5], dtype=torch.float32, requires_grad=True)
    u = torch.exp(log_u)
    target = torch.zeros_like(q)
    pred = torch.sqrt(q)

    loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl")
    loss.backward()

    # dL/d(log u) = (dL/du) * (du/d log u) = (1 - q/u) * u = u - q
    expected_grad = (u.detach() - q) / len(q)
    torch.testing.assert_close(log_u.grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_gkl_softplus_parameterization_gradient():
    z = torch.tensor([-2.0, 0.0, 2.0], dtype=torch.float32, requires_grad=True)
    u_min = 1e-6
    u = F.softplus(z) + u_min
    q = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    target = torch.zeros_like(q)
    pred = torch.sqrt(q)

    loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl")
    loss.backward()

    # Analytical: dL/dz = (1 - q/u) * sigmoid(z)
    sigmoid_z = torch.sigmoid(z)
    expected_dz = ((1.0 - q / u.detach()) * sigmoid_z) / len(z)
    torch.testing.assert_close(z.grad, expected_dz, rtol=1e-5, atol=1e-6)


def test_gkl_population_optimum_is_arithmetic_mean():
    # In GKL, for an empirical distribution of q_i, the optimal constant u* is the arithmetic mean
    torch.manual_seed(42)
    q_samples = torch.distributions.Exponential(rate=2.0).sample((10000,))
    sample_mean = q_samples.mean().item()

    # Minimize GKL w.r.t a scalar parameter log_u
    log_u = torch.tensor([0.0], requires_grad=True)
    optimizer = torch.optim.Adam([log_u], lr=0.1)

    target = torch.zeros_like(q_samples)
    pred = torch.sqrt(q_samples)

    for _ in range(300):
        optimizer.zero_grad()
        u = torch.exp(log_u).expand_as(q_samples)
        loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl")
        loss.backward()
        optimizer.step()

    learned_u = torch.exp(log_u).item()
    # Expected: learned_u matches arithmetic mean of q within statistical/numerical tolerance
    assert abs(learned_u - sample_mean) / sample_mean < 0.02


def test_gkl_matches_standalone_diagnostic():
    # Diagnostic definition: D(q || u) = q * log(q / u) - (q - u) = q*log(q) - q*log(u) + u - q
    torch.manual_seed(123)
    q = torch.rand(20, dtype=torch.float32) * 5.0
    # Include zero entries
    q[0] = 0.0
    q[1] = 0.0
    u = torch.rand(20, dtype=torch.float32) * 4.0 + 0.1

    def diagnostic_gkl(q_in, u_in):
        q_pos = q_in > 0
        term1 = torch.zeros_like(q_in)
        term1[q_pos] = q_in[q_pos] * torch.log(q_in[q_pos] / u_in[q_pos])
        term2 = -(q_in - u_in)
        return (term1 + term2).mean()

    diagnostic_loss = diagnostic_gkl(q, u)

    target = torch.zeros_like(q)
    pred = torch.sqrt(q)
    production_loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl")

    # Match within numerical precision
    torch.testing.assert_close(production_loss, diagnostic_loss, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. Gradient routing verification
# ---------------------------------------------------------------------------


def test_gkl_isolated_gradient_routing():
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION, use_attention=True,
        spatial_only_pooling=True, uncertainty_aware=True,
    )
    x = torch.randn(1, 8, 4, 16, 16)
    target = torch.randn(1, 4, 4, 16, 16)

    # 1. Forward with detach_variance_features=False (Path B enabled into backbone)
    pred, variance = model(x, detach_variance_features=False)

    # 2. Backward purely from GKL with detach_mean=True (Path A OFF)
    loss_gkl = predicted_error_mse_loss(
        variance, pred, target, objective="generalized_kl", detach_mean=True,
    )
    model.zero_grad()
    loss_gkl.backward()

    # Path A check: mean head parameters (out.weight, out.bias) MUST be zero
    assert model.out.weight.grad is None or torch.all(model.out.weight.grad == 0.0)
    assert model.out.bias.grad is None or torch.all(model.out.bias.grad == 0.0)

    # Path B check: variance head parameters (variance_out.weight, variance_out.bias) MUST be non-zero
    assert model.variance_out.weight.grad is not None
    assert torch.any(model.variance_out.weight.grad != 0.0)

    # Shared backbone check: encoder, attention, decoder parameters MUST be non-zero
    assert model.enc0.conv1.weight.grad is not None
    assert torch.any(model.enc0.conv1.weight.grad != 0.0)
    assert model.up0.conv.conv1.weight.grad is not None
    assert torch.any(model.up0.conv.conv1.weight.grad != 0.0)

    # 3. Reconstruction MSE backward check: updates mean head and backbone, NOT variance head
    model.zero_grad()
    torch.nn.init.normal_(model.out.weight)
    pred_mse, _ = model(x)
    loss_mse = full_mse_loss(pred_mse, target, torch.ones_like(target))
    loss_mse.backward()
    assert model.out.weight.grad is not None and torch.any(model.out.weight.grad != 0.0)
    assert model.variance_out.weight.grad is None or torch.all(model.variance_out.weight.grad == 0.0)
    assert model.enc0.conv1.weight.grad is not None and torch.any(model.enc0.conv1.weight.grad != 0.0)


def test_student_t_retains_both_pathways():
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION, use_attention=True,
        spatial_only_pooling=True, uncertainty_aware=True,
    )
    x = torch.randn(1, 8, 4, 16, 16)
    target = torch.randn(1, 4, 4, 16, 16)

    # With detach_variance_features=False and detach_mean=False: Path A and B both active
    pred, variance = model(x, detach_variance_features=False)
    loss_st = predicted_error_mse_loss(
        variance, pred, target, objective="student_t_nll", student_t_df=3.0, detach_mean=False,
    )
    model.zero_grad()
    loss_st.backward()

    # Mean head has non-zero gradient from Student-t NLL (Path A)
    assert model.out.weight.grad is not None and torch.any(model.out.weight.grad != 0.0)
    # Variance head has non-zero gradient from Student-t NLL (Path B)
    assert model.variance_out.weight.grad is not None and torch.any(model.variance_out.weight.grad != 0.0)
    # Backbone has non-zero gradient
    assert model.enc0.conv1.weight.grad is not None and torch.any(model.enc0.conv1.weight.grad != 0.0)


def test_phase_c_detached_reuse():
    target = torch.randn(1, 4, 4, 16, 16)
    pred0 = torch.randn(1, 4, 4, 16, 16, requires_grad=True)
    var0 = (torch.rand(1, 4, 4, 16, 16) + 0.1).requires_grad_(True)
    reuse_mask = torch.ones_like(target)
    reuse_mask[:, :, 2:] = 0.0  # Hide second half

    reuse_input, exact_mask = make_phase_c_reuse_input(target, pred0, var0, reuse_mask)
    # reuse_input must be strictly detached from pred0 and var0
    assert not reuse_input.requires_grad
    downstream_loss = (reuse_input * 2.0).sum()
    assert not downstream_loss.requires_grad
    assert pred0.grad is None
    assert var0.grad is None


# ---------------------------------------------------------------------------
# 3. Curriculum and continuation verification
# ---------------------------------------------------------------------------


def test_curriculum_alpha_transition_equivalence():
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION, use_attention=True,
        spatial_only_pooling=True, uncertainty_aware=True,
    )
    x = torch.randn(1, 8, 4, 16, 16)
    target = torch.randn(1, 4, 4, 16, 16)
    mask = torch.ones_like(target)

    # 1. Test alpha = 0: exactly matches Student-t A+B
    loss_a0, _, _, comp_a0 = curriculum_forward_loss(
        model, x, target, mask, lambda_v=1.0, lambda_r=0.0,
        uncertainty_loss="generalized_kl", student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False, detach_mean=False,
        gkl_alpha=0.0,
    )
    loss_st, _, _, comp_st = curriculum_forward_loss(
        model, x, target, mask, lambda_v=1.0, lambda_r=0.0,
        uncertainty_loss="student_t_nll", student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False, detach_mean=False,
        gkl_alpha=None,
    )
    torch.testing.assert_close(loss_a0, loss_st)
    torch.testing.assert_close(comp_a0["L_var_0"], comp_st["L_var_0"])

    # 2. Test alpha = 1.0: exactly matches pure GKL (Path A OFF, Path B ON)
    loss_a1, _, _, comp_a1 = curriculum_forward_loss(
        model, x, target, mask, lambda_v=1.0, lambda_r=0.0,
        uncertainty_loss="generalized_kl", student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False, detach_mean=True,
        gkl_alpha=1.0,
    )
    model.zero_grad()
    loss_a1.backward()
    # At alpha = 1.0, mean head receives gradients ONLY from L_mu_0
    mean_grad_a1 = model.out.weight.grad.clone()

    model.zero_grad()
    loss_mu_only = full_mse_loss(model(x)[0], target, mask) / 2.0
    loss_mu_only.backward()
    mean_grad_mu = model.out.weight.grad.clone()
    torch.testing.assert_close(mean_grad_a1, mean_grad_mu)

    # 3. Test alpha = 0.5: midpoint blend
    loss_a05, _, _, comp_a05 = curriculum_forward_loss(
        model, x, target, mask, lambda_v=1.0, lambda_r=0.0,
        uncertainty_loss="generalized_kl", student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False, detach_mean=False,
        gkl_alpha=0.5,
    )
    expected_var = 0.5 * comp_a0["L_var_0"] + 0.5 * comp_a1["L_var_0"]
    torch.testing.assert_close(comp_a05["L_var_0"], expected_var)


def test_phase_c_passes_use_independent_uncertainty_targets():
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION, use_attention=True,
        spatial_only_pooling=True, uncertainty_aware=True,
    )
    x = torch.randn(1, 8, 4, 16, 16)
    target = torch.randn(1, 4, 4, 16, 16)
    mask = torch.ones_like(target)
    reuse_mask = torch.ones_like(target)

    loss, _, _, comp = curriculum_forward_loss(
        model, x, target, mask, lambda_v=1.0, lambda_r=1.0,
        reuse_masks=(reuse_mask,),
        uncertainty_loss="generalized_kl", student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False, detach_mean=True,
        gkl_alpha=1.0,
    )
    assert "L_mu_0" in comp and "L_var_0" in comp
    assert "L_mu_1" in comp and "L_var_1" in comp
    assert comp["forwards"] == 2
    assert torch.isfinite(comp["L_var_0"])
    assert torch.isfinite(comp["L_var_1"])


def test_cli_and_continuation_validation():
    cli_base = [
        "--h5-dir", "dummy",
        "--betas", "0.2",
        "--delta-t", "24",
        "--stride-t", "2",
        "--batch-size", "4",
        "--num-workers", "4",
        "--epochs", "256",
        "--warmup-epochs", "5",
        "--lr", "2e-4",
        "--min-lr", "1e-5",
        "--weight-decay", "1e-4",
        "--base-channels", "24",
        "--model-version", "direct_unet3d_no_activation_norm_v3",
        "--channel-mults", "1", "2", "4", "8",
        "--density-probe-min", "0",
        "--density-probe-max", "0",
        "--density-visible-p-zero", "0.05",
        "--density-visible-p-full", "0.05",
        "--magnetic-visible-p-zero", "0.05",
        "--magnetic-visible-p-full", "0.05",
        "--use-attention",
        "--spatial-only-pooling",
        "--amp", "--amp-dtype", "bfloat16",
        "--uncertainty-aware",
        "--uncertainty-loss-weight", "1.0",
        "--curriculum-training",
        "--curriculum-config", "three_phase_global_cosine",
    ]

    # Source checkpoint simulated args from parser
    source_parsed = parse_args(
        cli_base + [
            "--out-dir", "runs/source",
            "--uncertainty-loss", "student_t_nll",
            "--student-t-df", "3.0",
            "--joint-nll-grad",
        ]
    )
    source_ckpt = {
        "epoch": 256,
        "args": vars(source_parsed).copy(),
        "curriculum_state": {
            "variance_ramp_armed": True,
            "rollout_ramp_armed": True,
            "variance_ramp_start_step": 79106,
            "variance_ramp_steps": 42760.0,
            "rollout_ramp_start_step": 135763,
            "rollout_ramp_steps": 42760.0,
            "lambda_v": 1.0,
            "lambda_r": 1.0,
            "updates_per_epoch": 1069,
            "successful_optimizer_steps": 273664,
            "variance_phase_start_epoch": 75,
            "rollout_phase_start_epoch": 128,
            "phase_d_start_epoch": None,
            "phase_d_trajectory_resume_policy": "reset_on_resume",
        },
        "optimizer": {
            "state": {0: {"step": 273664}},
            "param_groups": [{"lr": 1e-5}],
        },
        "model": {},
        "stats": {},
    }

    continuation_flags = [
        "--final-epoch", "276",
        "--stop-after-epochs", "276",
        "--post-final-continuation", "runs/source/final.pt",
    ]

    # GKL continuation args
    gkl_cli = cli_base + continuation_flags + [
        "--out-dir", "runs/gkl_continuation",
        "--uncertainty-loss", "generalized_kl",
        "--gkl-continuation-20ep",
    ]
    gkl_args = parse_args(gkl_cli)
    validate_post_final_continuation(
        gkl_args, source_ckpt, Path("runs/source/final.pt"), Path("runs/gkl_continuation/"),
    )

    # Student-t control args
    ctrl_cli = cli_base + continuation_flags + [
        "--out-dir", "runs/control",
        "--uncertainty-loss", "student_t_nll",
        "--student-t-df", "3.0",
        "--student-t-unnormalized-control-20ep",
    ]
    ctrl_args = parse_args(ctrl_cli)
    validate_post_final_continuation(
        ctrl_args, source_ckpt, Path("runs/source/final.pt"), Path("runs/control/"),
    )


def test_continuation_effective_learning_rate_and_resume():
    cli_base = [
        "--h5-dir", "dummy",
        "--betas", "0.2",
        "--delta-t", "24",
        "--stride-t", "2",
        "--batch-size", "4",
        "--num-workers", "4",
        "--epochs", "256",
        "--warmup-epochs", "5",
        "--lr", "2e-4",
        "--min-lr", "1e-5",
        "--weight-decay", "1e-4",
        "--base-channels", "24",
        "--model-version", "direct_unet3d_no_activation_norm_v3",
        "--channel-mults", "1", "2", "4", "8",
        "--density-probe-min", "0",
        "--density-probe-max", "0",
        "--density-visible-p-zero", "0.05",
        "--density-visible-p-full", "0.05",
        "--magnetic-visible-p-zero", "0.05",
        "--magnetic-visible-p-full", "0.05",
        "--use-attention",
        "--spatial-only-pooling",
        "--amp", "--amp-dtype", "bfloat16",
        "--uncertainty-aware",
        "--uncertainty-loss-weight", "1.0",
        "--curriculum-training",
        "--curriculum-config", "three_phase_global_cosine",
        "--final-epoch", "276",
        "--stop-after-epochs", "276",
        "--post-final-continuation", "runs/source/final.pt",
    ]

    # 1. GKL continuation
    gkl_args = parse_args(cli_base + [
        "--out-dir", "runs/gkl_continuation",
        "--uncertainty-loss", "generalized_kl",
        "--gkl-continuation-20ep",
    ])
    # Nominal curriculum retains standard three_phase_global_cosine endpoints
    assert math.isclose(gkl_args.lr, 2e-4)
    assert math.isclose(gkl_args.min_lr, 1e-5)
    # Effective continuation learning rate is constant at 1e-5 across all continuation epochs
    state = {}
    for epoch in range(256, 277):
        effective_lr = curriculum_learning_rate_for_epoch(epoch, gkl_args, state)
        assert math.isclose(effective_lr, 1e-5), f"Epoch {epoch} effective lr {effective_lr} != 1e-5"

    # Resume compatibility: args saved during continuation resume cleanly with --auto-resume
    sim_saved_gkl_args = vars(gkl_args).copy()
    gkl_resume_args = parse_args(cli_base + [
        "--out-dir", "runs/gkl_continuation",
        "--uncertainty-loss", "generalized_kl",
        "--gkl-continuation-20ep",
        "--auto-resume",
    ])
    validate_resume_parameters(gkl_resume_args, sim_saved_gkl_args)

    # 2. Student-t control
    ctrl_args = parse_args(cli_base + [
        "--out-dir", "runs/control",
        "--uncertainty-loss", "student_t_nll",
        "--student-t-df", "3.0",
        "--student-t-unnormalized-control-20ep",
    ])
    assert math.isclose(ctrl_args.lr, 2e-4)
    assert math.isclose(ctrl_args.min_lr, 1e-5)
    for epoch in range(256, 277):
        effective_lr = curriculum_learning_rate_for_epoch(epoch, ctrl_args, state)
        assert math.isclose(effective_lr, 1e-5), f"Epoch {epoch} effective lr {effective_lr} != 1e-5"

    sim_saved_ctrl_args = vars(ctrl_args).copy()
    ctrl_resume_args = parse_args(cli_base + [
        "--out-dir", "runs/control",
        "--uncertainty-loss", "student_t_nll",
        "--student-t-df", "3.0",
        "--student-t-unnormalized-control-20ep",
        "--auto-resume",
    ])
    validate_resume_parameters(ctrl_resume_args, sim_saved_ctrl_args)


# ---------------------------------------------------------------------------
# 4. Comprehensive Section 10 Verification (Items 1-16)
# ---------------------------------------------------------------------------


def test_item1_forward_loss_equivalence():
    torch.manual_seed(42)
    pred = torch.randn(2, 4, 4, 8, 8, requires_grad=True)
    target = torch.randn(2, 4, 4, 8, 8)
    u = (torch.rand(2, 4, 4, 8, 8) + 0.1).requires_grad_(True)

    loss_b_only = predicted_error_mse_loss(u, pred, target, objective="generalized_kl", detach_mean=True)
    loss_ab = predicted_error_mse_loss(u, pred, target, objective="generalized_kl", detach_mean=False)

    torch.testing.assert_close(loss_ab, loss_b_only)
    assert abs(loss_ab.item() - loss_b_only.item()) == 0.0


def test_item2_analytical_mean_gradient():
    torch.manual_seed(42)
    pred = torch.randn(10, dtype=torch.float32, requires_grad=True)
    target = torch.randn(10, dtype=torch.float32)
    u = torch.rand(10, dtype=torch.float32) + 0.2

    loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl", detach_mean=False)
    loss.backward()

    q = (pred.detach() - target).square()
    expected_grad = 2.0 * (pred.detach() - target) * torch.log(q / u) / len(pred)
    torch.testing.assert_close(pred.grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_item3_uncertainty_gradient_identity():
    torch.manual_seed(42)
    pred = torch.randn(10, dtype=torch.float32)
    target = torch.randn(10, dtype=torch.float32)

    u_b = (torch.rand(10, dtype=torch.float32) + 0.1).requires_grad_(True)
    loss_b = predicted_error_mse_loss(u_b, pred, target, objective="generalized_kl", detach_mean=True)
    loss_b.backward()

    u_ab = u_b.detach().clone().requires_grad_(True)
    loss_ab = predicted_error_mse_loss(u_ab, pred, target, objective="generalized_kl", detach_mean=False)
    loss_ab.backward()

    torch.testing.assert_close(u_ab.grad, u_b.grad)


def test_item4_gradient_decomposition_path_a_plus_b():
    torch.manual_seed(42)
    pred_val = torch.randn(8, dtype=torch.float32)
    target = torch.randn(8, dtype=torch.float32)
    u_val = torch.rand(8, dtype=torch.float32) + 0.2

    pred_a = pred_val.clone().requires_grad_(True)
    u_a = u_val.clone()
    loss_a = predicted_error_mse_loss(u_a, pred_a, target, objective="generalized_kl", detach_mean=False)
    grad_a = torch.autograd.grad(loss_a, pred_a)[0]

    pred_b = pred_val.clone()
    u_b = u_val.clone().requires_grad_(True)
    loss_b = predicted_error_mse_loss(u_b, pred_b, target, objective="generalized_kl", detach_mean=True)
    grad_b = torch.autograd.grad(loss_b, u_b)[0]

    pred_ab = pred_val.clone().requires_grad_(True)
    u_ab = u_val.clone().requires_grad_(True)
    loss_ab = predicted_error_mse_loss(u_ab, pred_ab, target, objective="generalized_kl", detach_mean=False)
    grad_ab_pred, grad_ab_u = torch.autograd.grad(loss_ab, (pred_ab, u_ab))

    torch.testing.assert_close(grad_ab_pred, grad_a)
    torch.testing.assert_close(grad_ab_u, grad_b)


def test_item5_a_plus_b_updates_both_heads_and_backbone():
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION, use_attention=True,
        spatial_only_pooling=True, uncertainty_aware=True,
    )
    x = torch.randn(1, 8, 4, 16, 16)
    target = torch.randn(1, 4, 4, 16, 16)

    pred, variance = model(x, detach_variance_features=False)
    loss = predicted_error_mse_loss(variance, pred, target, objective="generalized_kl", detach_mean=False)
    model.zero_grad()
    loss.backward()

    assert model.out.weight.grad is not None and torch.any(model.out.weight.grad != 0.0)
    assert model.variance_out.weight.grad is not None and torch.any(model.variance_out.weight.grad != 0.0)
    assert model.enc0.conv1.weight.grad is not None and torch.any(model.enc0.conv1.weight.grad != 0.0)


def test_item6_b_only_no_direct_mean_gradient():
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION, use_attention=True,
        spatial_only_pooling=True, uncertainty_aware=True,
    )
    x = torch.randn(1, 8, 4, 16, 16)
    target = torch.randn(1, 4, 4, 16, 16)

    pred, variance = model(x, detach_variance_features=False)
    loss = predicted_error_mse_loss(variance, pred, target, objective="generalized_kl", detach_mean=True)
    model.zero_grad()
    loss.backward()

    assert model.out.weight.grad is None or torch.all(model.out.weight.grad == 0.0)
    assert model.variance_out.weight.grad is not None and torch.any(model.variance_out.weight.grad != 0.0)
    assert model.enc0.conv1.weight.grad is not None and torch.any(model.enc0.conv1.weight.grad != 0.0)


def test_item7_zero_residual_stability():
    pred = torch.tensor([1.5, 2.5], dtype=torch.float32, requires_grad=True)
    target = torch.tensor([1.5, 2.5], dtype=torch.float32)
    u = torch.tensor([0.2, 0.8], dtype=torch.float32, requires_grad=True)

    loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl", detach_mean=False)
    assert torch.isfinite(loss).all()
    torch.testing.assert_close(loss, u.mean())

    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.isfinite(u.grad).all()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))
    torch.testing.assert_close(u.grad, torch.full_like(u, 1.0 / len(u)))


def test_item8_extremely_small_residuals_stability():
    epsilons = [1e-3, 1e-5, 1e-7, 1e-15, 1e-20, 0.0]
    for eps in epsilons:
        pred = torch.tensor([eps], dtype=torch.float32, requires_grad=True)
        target = torch.tensor([0.0], dtype=torch.float32)
        u = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)

        loss = predicted_error_mse_loss(u, pred, target, objective="generalized_kl", detach_mean=False)
        assert torch.isfinite(loss).all()
        loss.backward()
        assert torch.isfinite(pred.grad).all()
        assert torch.isfinite(u.grad).all()
        q = (pred.detach() - target).square()
        expected_pred_grad = torch.where(
            q > 0, 2.0 * (pred.detach() - target) * torch.log(q / u.detach()), torch.zeros_like(q)
        )
        torch.testing.assert_close(pred.grad, expected_pred_grad, rtol=1e-4, atol=1e-5)


def test_item9_alpha_0_reproduces_student_t():
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION, use_attention=True,
        spatial_only_pooling=True, uncertainty_aware=True,
    )
    x = torch.randn(1, 8, 4, 16, 16)
    target = torch.randn(1, 4, 4, 16, 16)
    mask = torch.ones_like(target)

    loss_a0, _, _, comp_a0 = curriculum_forward_loss(
        model, x, target, mask, lambda_v=1.0, lambda_r=0.0,
        uncertainty_loss="generalized_kl", student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False, detach_mean=False,
        gkl_alpha=0.0,
    )
    model.zero_grad()
    loss_a0.backward()
    grads_a0 = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    loss_st, _, _, comp_st = curriculum_forward_loss(
        model, x, target, mask, lambda_v=1.0, lambda_r=0.0,
        uncertainty_loss="student_t_nll", student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False, detach_mean=False,
        gkl_alpha=None,
    )
    model.zero_grad()
    loss_st.backward()
    grads_st = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    torch.testing.assert_close(loss_a0, loss_st)
    for g_a0, g_st in zip(grads_a0, grads_st):
        torch.testing.assert_close(g_a0, g_st)


def test_item10_alpha_05_produces_mixed_objective_and_gradients():
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION, use_attention=True,
        spatial_only_pooling=True, uncertainty_aware=True,
    )
    x = torch.randn(1, 8, 4, 16, 16)
    target = torch.randn(1, 4, 4, 16, 16)
    mask = torch.ones_like(target)

    loss_mid, _, _, comp_mid = curriculum_forward_loss(
        model, x, target, mask, lambda_v=1.0, lambda_r=0.0,
        uncertainty_loss="generalized_kl", student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False, detach_mean=False,
        gkl_alpha=0.5,
    )
    model.zero_grad()
    loss_mid.backward()

    assert model.out.weight.grad is not None and torch.any(model.out.weight.grad != 0.0)
    assert model.variance_out.weight.grad is not None and torch.any(model.variance_out.weight.grad != 0.0)
    assert model.enc0.conv1.weight.grad is not None and torch.any(model.enc0.conv1.weight.grad != 0.0)


def test_item11_alpha_1_produces_pure_gkl_ab():
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION, use_attention=True,
        spatial_only_pooling=True, uncertainty_aware=True,
    )
    x = torch.randn(1, 8, 4, 16, 16)
    target = torch.randn(1, 4, 4, 16, 16)
    mask = torch.ones_like(target)

    loss_a1, _, _, comp_a1 = curriculum_forward_loss(
        model, x, target, mask, lambda_v=1.0, lambda_r=0.0,
        uncertainty_loss="generalized_kl", student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False, detach_mean=False,
        gkl_alpha=1.0,
    )
    model.zero_grad()
    loss_a1.backward()
    grads_a1 = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    loss_pure, _, _, comp_pure = curriculum_forward_loss(
        model, x, target, mask, lambda_v=1.0, lambda_r=0.0,
        uncertainty_loss="generalized_kl", student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False, detach_mean=False,
        gkl_alpha=None,
    )
    model.zero_grad()
    loss_pure.backward()
    grads_pure = [p.grad.clone() for p in model.parameters() if p.grad is not None]

    torch.testing.assert_close(loss_a1, loss_pure)
    for g_a1, g_pure in zip(grads_a1, grads_pure):
        torch.testing.assert_close(g_a1, g_pure)


def test_item12_phase_c_detached_reuse():
    target = torch.randn(1, 4, 4, 16, 16)
    pred0 = torch.randn(1, 4, 4, 16, 16, requires_grad=True)
    var0 = (torch.rand(1, 4, 4, 16, 16) + 0.1).requires_grad_(True)
    reuse_mask = torch.ones_like(target)
    reuse_mask[:, :, 2:] = 0.0

    reuse_input, exact_mask = make_phase_c_reuse_input(target, pred0, var0, reuse_mask)
    assert not reuse_input.requires_grad
    downstream_loss = (reuse_input * 2.0).sum()
    assert not downstream_loss.requires_grad
    assert pred0.grad is None
    assert var0.grad is None


def test_item13_optimizer_state_restoration():
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture=DIRECT_MODEL_VERSION, use_attention=True,
        spatial_only_pooling=True, uncertainty_aware=True,
    )
    opt1 = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    x = torch.randn(1, 8, 4, 16, 16)
    pred, var = model(x)
    loss = pred.sum() + var.sum()
    loss.backward()
    opt1.step()

    state_dict = opt1.state_dict()

    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    opt2.load_state_dict(state_dict)

    assert opt2.state_dict()["state"] == opt1.state_dict()["state"]
    assert opt2.param_groups[0]["lr"] == 1e-5


def test_item14_effective_lr_fixed_1e5():
    cli_base = [
        "--h5-dir", "dummy", "--betas", "0.2", "--delta-t", "24", "--stride-t", "2",
        "--batch-size", "4", "--num-workers", "4", "--epochs", "256", "--warmup-epochs", "5",
        "--lr", "2e-4", "--min-lr", "1e-5", "--weight-decay", "1e-4", "--base-channels", "24",
        "--model-version", "direct_unet3d_no_activation_norm_v3", "--channel-mults", "1", "2", "4", "8",
        "--density-probe-min", "0", "--density-probe-max", "0",
        "--density-visible-p-zero", "0.05", "--density-visible-p-full", "0.05",
        "--magnetic-visible-p-zero", "0.05", "--magnetic-visible-p-full", "0.05",
        "--use-attention", "--spatial-only-pooling", "--amp", "--amp-dtype", "bfloat16",
        "--uncertainty-aware", "--uncertainty-loss-weight", "1.0",
        "--curriculum-training", "--curriculum-config", "three_phase_global_cosine",
        "--final-epoch", "276", "--stop-after-epochs", "276",
        "--post-final-continuation", "runs/source/final.pt",
        "--out-dir", "runs/gkl_joint_continuation",
        "--uncertainty-loss", "generalized_kl",
        "--gkl-joint-grad-continuation-20ep",
    ]
    args = parse_args(cli_base)
    state = {}
    for epoch in range(256, 277):
        effective_lr = curriculum_learning_rate_for_epoch(epoch, args, state)
        assert math.isclose(effective_lr, 1e-5)


def test_item15_resume_preserves_routing_and_transition():
    cli_base = [
        "--h5-dir", "dummy", "--betas", "0.2", "--delta-t", "24", "--stride-t", "2",
        "--batch-size", "4", "--num-workers", "4", "--epochs", "256", "--warmup-epochs", "5",
        "--lr", "2e-4", "--min-lr", "1e-5", "--weight-decay", "1e-4", "--base-channels", "24",
        "--model-version", "direct_unet3d_no_activation_norm_v3", "--channel-mults", "1", "2", "4", "8",
        "--density-probe-min", "0", "--density-probe-max", "0",
        "--density-visible-p-zero", "0.05", "--density-visible-p-full", "0.05",
        "--magnetic-visible-p-zero", "0.05", "--magnetic-visible-p-full", "0.05",
        "--use-attention", "--spatial-only-pooling", "--amp", "--amp-dtype", "bfloat16",
        "--uncertainty-aware", "--uncertainty-loss-weight", "1.0",
        "--curriculum-training", "--curriculum-config", "three_phase_global_cosine",
        "--final-epoch", "276", "--stop-after-epochs", "276",
        "--post-final-continuation", "runs/source/final.pt",
        "--out-dir", "runs/gkl_joint_continuation",
        "--uncertainty-loss", "generalized_kl",
        "--gkl-joint-grad-continuation-20ep",
    ]
    args = parse_args(cli_base)
    saved_args = vars(args).copy()

    resume_args = parse_args(cli_base + ["--auto-resume"])
    validate_resume_parameters(resume_args, saved_args)
    assert resume_args.gkl_joint_grad_continuation_20ep is True


def test_item16_existing_modes_remain_unchanged():
    cli_base = [
        "--h5-dir", "dummy", "--betas", "0.2", "--delta-t", "24", "--stride-t", "2",
        "--batch-size", "4", "--num-workers", "4", "--epochs", "256", "--warmup-epochs", "5",
        "--lr", "2e-4", "--min-lr", "1e-5", "--weight-decay", "1e-4", "--base-channels", "24",
        "--model-version", "direct_unet3d_no_activation_norm_v3", "--channel-mults", "1", "2", "4", "8",
        "--density-probe-min", "0", "--density-probe-max", "0",
        "--density-visible-p-zero", "0.05", "--density-visible-p-full", "0.05",
        "--magnetic-visible-p-zero", "0.05", "--magnetic-visible-p-full", "0.05",
        "--use-attention", "--spatial-only-pooling", "--amp", "--amp-dtype", "bfloat16",
        "--uncertainty-aware", "--uncertainty-loss-weight", "1.0",
        "--curriculum-training", "--curriculum-config", "three_phase_global_cosine",
        "--final-epoch", "276", "--stop-after-epochs", "276",
        "--post-final-continuation", "runs/source/final.pt",
    ]
    gkl_b_args = parse_args(cli_base + [
        "--out-dir", "runs/gkl_b_continuation",
        "--uncertainty-loss", "generalized_kl",
        "--gkl-continuation-20ep",
    ])
    assert gkl_b_args.gkl_continuation_20ep is True
    assert gkl_b_args.gkl_joint_grad_continuation_20ep is False

    st_args = parse_args(cli_base + [
        "--out-dir", "runs/student_t_control",
        "--uncertainty-loss", "student_t_nll",
        "--student-t-df", "3.0",
        "--student-t-unnormalized-control-20ep",
    ])
    assert st_args.student_t_unnormalized_control_20ep is True
    assert st_args.gkl_joint_grad_continuation_20ep is False
