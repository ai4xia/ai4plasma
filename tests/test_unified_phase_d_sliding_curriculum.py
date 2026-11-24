# tests/test_unified_phase_d_sliding_curriculum.py

from __future__ import annotations

import argparse
import copy
import math
from typing import Dict

import pytest
import torch

from data.masking import full_mse_loss, predicted_error_mse_loss
from models.unet3d import UNet3D
from train_masked_unet3d import (
    build_sliding_reuse_input,
    curriculum_forward_loss,
    curriculum_weights,
    make_curriculum_state,
    make_phase_c_reuse_input,
    sample_slide_step,
    sample_sliding_window_pair,
    slide_curriculum_weight,
)


def _make_small_unet3d(seed: int = 0) -> UNet3D:
    torch.manual_seed(seed)
    return UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=8,
        channel_mults=(1, 2, 4),
        architecture="direct_unet3d_no_activation_norm_v3",
        use_attention=False,
        spatial_only_pooling=True,
        uncertainty_aware=True,
    )


def test_strict_s0_phase_c_equivalence_and_gradients():
    """Verify L_D(s=0) == L_C across forwards, inputs, losses, and all parameter gradients."""
    torch.manual_seed(123)
    B, C, T, X, Z = 1, 4, 24, 16, 16
    target = torch.randn((B, C, T, X, Z), dtype=torch.float32)
    noise_a = torch.randn((B, C, T, X, Z), dtype=torch.float32)
    mask_a = (torch.rand((B, C, T, X, Z)) > 0.5).to(torch.float32)
    mask_b = (torch.rand((B, C, T, X, Z)) > 0.5).to(torch.float32)

    iv_a = 1.0 - mask_a
    input_a = torch.cat([torch.where(iv_a > 0.5, noise_a, target), iv_a], dim=1)

    model_ref = _make_small_unet3d(seed=7)
    model_new = copy.deepcopy(model_ref)

    # 1. Reference settled Phase C
    model_ref.zero_grad(set_to_none=True)
    loss_ref, pred_a_ref, var_a_ref, comp_ref = curriculum_forward_loss(
        model_ref,
        input_a.clone(),
        target,
        mask_a,
        lambda_v=1.0,
        lambda_r=1.0,
        reuse_masks=(mask_b,),
        uncertainty_loss="generalized_kl",
        student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False,
        detach_mean=False,
        gkl_alpha=None,
        reuse_gradient_scale=1.0,
        mean_only_phase_c_reuse=False,
    )
    loss_ref.backward()

    # 2. Generalized sliding Phase D with s=0
    model_new.zero_grad(set_to_none=True)
    loss_new, pred_a_new, var_a_new, comp_new = curriculum_forward_loss(
        model_new,
        input_a.clone(),
        target,
        mask_a,
        lambda_v=1.0,
        lambda_r=1.0,
        reuse_masks=(mask_b,),
        uncertainty_loss="generalized_kl",
        student_t_df=3.0,
        curriculum_config="three_phase_global_cosine",
        detach_variance_features=False,
        detach_mean=False,
        gkl_alpha=None,
        reuse_gradient_scale=1.0,
        mean_only_phase_c_reuse=False,
        slide_step=0,
        reuse_targets=(target,),
    )
    loss_new.backward()

    # Compare forward 1
    assert torch.allclose(pred_a_ref, pred_a_new, atol=1e-6, rtol=1e-6)
    assert torch.allclose(var_a_ref, var_a_new, atol=1e-6, rtol=1e-6)

    # Compare second-forward input tensors & second-forward outputs
    in_b_ref, m_b_ref = make_phase_c_reuse_input(
        target, pred_a_ref.detach(), var_a_ref.detach(), mask_b,
        reuse_gradient_scale=1.0, mean_only_phase_c_reuse=False,
    )
    in_b_new, m_b_new = build_sliding_reuse_input(
        target, mask_b, pred_a_new.detach(), var_a_new.detach(), slide_step=0,
        reuse_gradient_scale=1.0, mean_only_phase_c_reuse=False,
    )
    assert torch.allclose(in_b_ref, in_b_new, atol=1e-7, rtol=1e-7)
    assert torch.equal(m_b_ref, m_b_new)

    with torch.no_grad():
        pred_b_ref, var_b_ref = model_ref(in_b_ref, detach_variance_features=False)
        pred_b_new, var_b_new = model_new(in_b_new, detach_variance_features=False)
    assert torch.allclose(pred_b_ref, pred_b_new, atol=1e-6, rtol=1e-6)
    assert torch.allclose(var_b_ref, var_b_new, atol=1e-6, rtol=1e-6)

    # Compare all loss components
    for key in ("L_mu_0", "L_var_0", "L_mu_1", "L_var_1", "total_loss"):
        assert torch.allclose(comp_ref[key], comp_new[key], atol=1e-6, rtol=1e-6), f"Mismatch in {key}"

    # Compare ALL parameter gradients (including uncertainty head and shared layers)
    for (name, p_ref), (_, p_new) in zip(model_ref.named_parameters(), model_new.named_parameters()):
        assert p_ref.grad is not None and p_new.grad is not None, f"Missing grad for {name}"
        assert torch.allclose(p_ref.grad, p_new.grad, atol=1e-6, rtol=1e-6), f"Grad mismatch for {name}"

    # Explicitly verify second-forward loss backprops non-zero gradients through pred_A and var_A
    pred_a_probe = pred_a_ref.detach().clone().requires_grad_(True)
    var_a_probe = var_a_ref.detach().clone().requires_grad_(True)
    in_b_probe, exact_b_probe = build_sliding_reuse_input(
        target, mask_b, pred_a_probe, var_a_probe, slide_step=0,
        reuse_gradient_scale=1.0, mean_only_phase_c_reuse=False,
    )
    pred_b_probe, var_b_probe = model_ref(in_b_probe, detach_variance_features=False)
    l2 = full_mse_loss(pred_b_probe, target, exact_b_probe) + predicted_error_mse_loss(
        var_b_probe, pred_b_probe, target, "generalized_kl", 3.0, detach_mean=False,
    )
    g_pred_a, g_var_a = torch.autograd.grad(l2, (pred_a_probe, var_a_probe))
    assert torch.isfinite(g_pred_a).all() and g_pred_a.abs().max() > 0
    assert torch.isfinite(g_var_a).all() and g_var_a.abs().max() > 0


@pytest.mark.parametrize("slide_step", [0, 1, -1, 12, -12, 24, -24])
def test_bidirectional_sliding_overlap_and_gradients(slide_step: int):
    """Verify temporal alignment, masking, noise, and gradient flow for s in {0, +-1, +-12, +-24}."""
    torch.manual_seed(42 + slide_step)
    B, C, T, X, Z = 1, 4, 24, 8, 8
    # Construct target_b with distinct temporal values per frame
    target_b = torch.arange(T, dtype=torch.float32).view(1, 1, T, 1, 1).expand(B, C, T, X, Z).clone()
    noise_b = torch.full((B, C, T, X, Z), -99.0, dtype=torch.float32)

    # Known pred_a and var_a with gradients enabled
    pred_a_base = (torch.arange(T, dtype=torch.float32).view(1, 1, T, 1, 1).expand(B, C, T, X, Z) + 100.0)
    var_a_base = torch.full((B, C, T, X, Z), 0.25, dtype=torch.float32)
    pred_a = pred_a_base.clone().requires_grad_(True)
    var_a = var_a_base.clone().requires_grad_(True)

    mask_b = torch.zeros((B, C, T, X, Z), dtype=torch.float32)
    mask_b[:, :, :, 0, 0] = 1.0  # genuine observation at spatial (0, 0) for all frames

    inp_b, exact_b = build_sliding_reuse_input(
        target_b, mask_b, pred_a, var_a, slide_step=slide_step,
        noise_b=noise_b, reuse_gradient_scale=1.0, mean_only_phase_c_reuse=False,
    )
    vals = inp_b[:, :4]
    u_in = inp_b[:, 4:]
    assert torch.equal(exact_b, mask_b)

    # 1. Genuine observations override predictions everywhere
    assert torch.equal(vals[:, :, :, 0, 0], target_b[:, :, :, 0, 0])
    assert torch.all(u_in[:, :, :, 0, 0] == 0.0)

    # 2. Check overlap vs non-overlap at unobserved voxel (1, 1)
    overlap_len = T - abs(slide_step)
    if slide_step == 0:
        assert torch.equal(vals[:, :, :, 1, 1], pred_a_base[:, :, :, 1, 1])
        assert torch.all(u_in[:, :, :, 1, 1] == 0.25)
    elif slide_step > 0:
        if overlap_len > 0:
            assert torch.equal(vals[:, :, :overlap_len, 1, 1], pred_a_base[:, :, slide_step:, 1, 1])
            assert torch.all(u_in[:, :, :overlap_len, 1, 1] == 0.25)
        assert torch.all(vals[:, :, overlap_len:, 1, 1] == -99.0)
        assert torch.all(u_in[:, :, overlap_len:, 1, 1] == 1.0)
    else:  # slide_step < 0
        d = -slide_step
        assert torch.all(vals[:, :, :d, 1, 1] == -99.0)
        assert torch.all(u_in[:, :, :d, 1, 1] == 1.0)
        if overlap_len > 0:
            assert torch.equal(vals[:, :, d:, 1, 1], pred_a_base[:, :, :T - d, 1, 1])
            assert torch.all(u_in[:, :, d:, 1, 1] == 0.25)

    # 3. Verify gradient flow through overlapping region and zero transfer at |s|=24
    if abs(slide_step) == 24:
        assert not inp_b.requires_grad
    else:
        assert inp_b.requires_grad
        loss = inp_b.sum()
        g_pred, g_var = torch.autograd.grad(loss, (pred_a, var_a))
        assert g_pred.abs().sum() > 0
        assert g_var.abs().sum() > 0


class _MockRunDataset:
    def __init__(self, lengths: Dict[str, int]):
        self.runs = {}
        for name, length in lengths.items():
            # Encode physical frame index into values so alignment is directly verifiable
            t_idx = torch.arange(length, dtype=torch.float32).view(1, length, 1, 1).expand(4, length, 8, 8).clone()
            self.runs[name] = t_idx

    def get_run(self, run_name: str) -> torch.Tensor:
        return self.runs[run_name]


@pytest.mark.parametrize("slide_step", [0, 1, -1, 12, -12, 24, -24])
def test_sample_sliding_window_pair_alignment(slide_step: int):
    """Verify physical frame alignment and no cross-simulation leakage."""
    ds = _MockRunDataset({"long_run": 60, "short_run": 29})
    mean = torch.zeros((1, 4, 1, 1, 1))
    std = torch.ones((1, 4, 1, 1, 1))
    rng = torch.Generator().manual_seed(99)

    win_a, win_b, t_a, t_b = sample_sliding_window_pair(
        ds, "long_run", t0_hint=None, slide_step=slide_step,
        window_size=24, mean=mean, std=std, device=torch.device("cpu"), generator=rng,
    )
    assert t_b == t_a + slide_step
    assert 0 <= t_a <= 60 - 24
    assert 0 <= t_b <= 60 - 24
    assert torch.isfinite(win_a).all() and torch.isfinite(win_b).all()
    assert int(win_a[0, 0, 0, 0].item()) == t_a
    assert int(win_b[0, 0, 0, 0].item()) == t_b

    # Short run (L=29 < 24+24) uses boundary padding cleanly when |s| > 5 without inventing frames
    win_a_s, win_b_s, t_a_s, t_b_s = sample_sliding_window_pair(
        ds, "short_run", t0_hint=None, slide_step=slide_step,
        window_size=24, mean=mean, std=std, device=torch.device("cpu"), generator=rng,
    )
    assert t_b_s == t_a_s + slide_step
    assert torch.isfinite(win_a_s).all()
    assert torch.isfinite(win_b_s).any()


def test_cosine_slide_curriculum_schedule_and_resume():
    """Verify cosine lambda_slide at p=0, 0.5, 1.0, sampling distribution, and resume state."""
    args = argparse.Namespace(
        variance_ramp_start_epoch=None,
        rollout_ramp_start_epoch=None,
        variance_ramp_epochs=40.0,
        rollout_ramp_epochs=40.0,
        arm_variance_ramp=False,
        arm_rollout_ramp=False,
        phase_d_sliding_curriculum=True,
        slide_ramp_epochs=10.0,
    )
    state = make_curriculum_state(args, updates_per_epoch=100, checkpoint=None)
    assert curriculum_weights(state) == (1.0, 1.0)

    # p = 0.0 -> lambda_slide = 0.0
    state["successful_optimizer_steps"] = 0
    assert math.isclose(slide_curriculum_weight(state), 0.0, abs_tol=1e-7)
    assert all(sample_slide_step(0.0, 24) == 0 for _ in range(20))

    # p = 0.5 -> lambda_slide = 0.5
    state["successful_optimizer_steps"] = 500
    assert math.isclose(slide_curriculum_weight(state), 0.5, abs_tol=1e-7)

    # p = 1.0 -> lambda_slide = 1.0
    state["successful_optimizer_steps"] = 1000
    assert math.isclose(slide_curriculum_weight(state), 1.0, abs_tol=1e-7)
    rng = torch.Generator().manual_seed(123)
    samples = [sample_slide_step(1.0, 24, generator=rng) for _ in range(500)]
    assert any(s < 0 for s in samples) and any(s > 0 for s in samples)

    # Resume restores exact progress and rejects mismatched slide_ramp_epochs
    restored = make_curriculum_state(args, updates_per_epoch=100, checkpoint={"curriculum_state": state})
    assert restored["successful_optimizer_steps"] == 1000
    assert math.isclose(slide_curriculum_weight(restored), 1.0, abs_tol=1e-7)

    args_bad = copy.deepcopy(args)
    args_bad.slide_ramp_epochs = 20.0
    with pytest.raises(ValueError, match="slide_ramp_steps"):
        make_curriculum_state(args_bad, updates_per_epoch=100, checkpoint={"curriculum_state": state})
