# tests/test_phase_d_causal_continuation.py

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import torch

from data.masking import (
    full_mse_loss,
    predicted_error_mse_loss,
)
from run_phase_d_evaluation import (
    compute_paired_statistics,
    run_condition_rollout,
)
from train_phase_d_causal_continuation import (
    build_trajectory_step_input,
    compute_learning_rate,
)


class MockUNet3D(torch.nn.Module):
    """Deterministic mock UNet3D for testing gradient flow and evaluation rollouts."""
    uncertainty_aware = True

    def __init__(self):
        super().__init__()
        self.conv_mean = torch.nn.Conv3d(8, 4, kernel_size=1)
        self.conv_var = torch.nn.Conv3d(8, 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, 8, T, H, W]
        mean = self.conv_mean(x)
        raw_var = self.conv_var(x)
        # Softplus to ensure positive variance
        var = torch.nn.functional.softplus(raw_var) + 1e-4
        return mean, var


def test_causal_indexing_and_padding():
    """Verify temporal alignment, padding for tau < 0, and current frame boundary."""
    device = torch.device("cpu")
    H, W = 16, 16
    window_size = 24
    L = 48

    target_norm = torch.randn((4, L, H, W), device=device)
    traj_noise = torch.randn((4, L, H, W), device=device)
    probe_bool = torch.zeros((H, W), dtype=torch.bool, device=device)
    probe_bool[::2, ::2] = True  # sparse probes
    hidden_bool = ~probe_bool

    pred_buffer = torch.randn((L, H, W), device=device)
    var_buffer = torch.rand((L, H, W), device=device) * 0.5 + 0.05

    traj = {
        "target_norm": target_norm,
        "traj_noise": traj_noise,
        "probe_bool": probe_bool,
        "hide_magnetic": False,
        "pred_buffer": pred_buffer,
        "var_buffer": var_buffer,
        "L": L,
    }

    # Test early cold start (t=0), transition (t=10), settled boundary (t=23), and late (t=35)
    for t in [0, 10, 23, 35]:
        inp, target_frame, visible_mask = build_trajectory_step_input(
            t=t, traj=traj, window_size=window_size, uncertainty_mode="original"
        )

        assert inp.shape == (1, 8, window_size, H, W)
        assert target_frame.shape == (1, 4, 1, H, W)
        assert visible_mask.shape == (1, 4, 1, H, W)

        # Check current frame at local index window_size - 1 (index 23)
        curr_vals = inp[0, :4, window_size - 1]
        curr_vars = inp[0, 4:, window_size - 1]

        # B fields are fully visible in B_full
        assert torch.equal(curr_vals[:3], target_norm[:3, t])
        assert torch.all(curr_vars[:3] == 0.0)
        assert torch.all(visible_mask[0, :3, 0] == 1.0)

        # Density probes are visible with zero variance
        assert torch.equal(curr_vals[3, probe_bool], target_norm[3, t, probe_bool])
        assert torch.all(curr_vars[3, probe_bool] == 0.0)
        assert torch.all(visible_mask[0, 3, 0, probe_bool] == 1.0)

        # Hidden density voxels at current frame are noise with variance 1.0
        assert torch.equal(curr_vals[3, hidden_bool], traj_noise[3, t, hidden_bool])
        assert torch.all(curr_vars[3, hidden_bool] == 1.0)
        assert torch.all(visible_mask[0, 3, 0, hidden_bool] == 0.0)

        # Verify historical padding for tau < 0
        for l in range(window_size):
            tau = t - window_size + 1 + l
            if tau < 0:
                assert torch.all(inp[0, 4:, l] == 1.0)


def test_uncertainty_mixture_modes():
    """Verify uncertainty mixture handling (original, inflated, high)."""
    device = torch.device("cpu")
    H, W = 8, 8
    window_size = 24
    L = 30
    t = 24  # Historical frames tau in [1, 23] exist

    target_norm = torch.randn((4, L, H, W), device=device)
    traj_noise = torch.randn((4, L, H, W), device=device)
    probe_bool = torch.zeros((H, W), dtype=torch.bool, device=device)
    probe_bool[0, 0] = True
    hidden_bool = ~probe_bool

    pred_buffer = torch.ones((L, H, W), device=device) * 2.5
    # Small variances below 0.1
    var_buffer = torch.ones((L, H, W), device=device) * 0.03

    traj = {
        "target_norm": target_norm,
        "traj_noise": traj_noise,
        "probe_bool": probe_bool,
        "hide_magnetic": True,  # B_hidden
        "pred_buffer": pred_buffer,
        "var_buffer": var_buffer,
        "L": L,
    }

    # 1. Mode 'original': historical unobserved gets var_buffer (0.03)
    inp_orig, _, _ = build_trajectory_step_input(t=t, traj=traj, window_size=window_size, uncertainty_mode="original")
    for l in range(window_size - 1):
        tau = t - window_size + 1 + l
        if tau >= 0:
            assert torch.allclose(inp_orig[0, 7, l, hidden_bool], torch.tensor(0.03))
            assert torch.allclose(inp_orig[0, 3, l, hidden_bool], torch.tensor(2.5))
            # Probe preservation
            assert torch.all(inp_orig[0, 7, l, probe_bool] == 0.0)
            assert torch.all(inp_orig[0, 3, l, probe_bool] == target_norm[3, tau, probe_bool])

    # 2. Mode 'inflated': historical unobserved clamped to >= 0.1
    inp_inf, _, _ = build_trajectory_step_input(t=t, traj=traj, window_size=window_size, uncertainty_mode="inflated")
    for l in range(window_size - 1):
        tau = t - window_size + 1 + l
        if tau >= 0:
            assert torch.allclose(inp_inf[0, 7, l, hidden_bool], torch.tensor(0.1))
            assert torch.allclose(inp_inf[0, 3, l, hidden_bool], torch.tensor(2.5))
            # Probe preservation
            assert torch.all(inp_inf[0, 7, l, probe_bool] == 0.0)

    # 3. Mode 'high': historical unobserved gets 1.0
    inp_high, _, _ = build_trajectory_step_input(t=t, traj=traj, window_size=window_size, uncertainty_mode="high")
    for l in range(window_size - 1):
        tau = t - window_size + 1 + l
        if tau >= 0:
            assert torch.allclose(inp_high[0, 7, l, hidden_bool], torch.tensor(1.0))
            assert torch.allclose(inp_high[0, 3, l, hidden_bool], torch.tensor(2.5))
            # Probe preservation
            assert torch.all(inp_high[0, 7, l, probe_bool] == 0.0)


def test_loss_gradients_and_joint_flow():
    """Verify that Generalized KL loss with detach_mean=False backpropagates

    gradients into both mean and variance prediction parameters.
    """
    model = MockUNet3D()
    x = torch.randn((1, 8, 24, 8, 8), requires_grad=True)
    target = torch.randn((1, 4, 1, 8, 8))
    visible_mask = torch.zeros((1, 4, 1, 8, 8))
    visible_mask[:, :3] = 1.0  # B_full

    pred, variance = model(x)
    pred_t = pred[:, :, 23:24]
    var_t = variance[:, :, 23:24]

    mu_loss = full_mse_loss(pred_t, target, visible_mask)
    var_loss = predicted_error_mse_loss(
        var_t, pred_t, target,
        objective="generalized_kl", student_t_df=3.0, detach_mean=False
    )
    total_loss = 0.5 * (mu_loss + var_loss)

    assert torch.isfinite(mu_loss)
    assert torch.isfinite(var_loss)
    assert torch.isfinite(total_loss)

    total_loss.backward()

    # Both heads must receive non-zero gradients
    assert model.conv_mean.weight.grad is not None
    assert torch.isfinite(model.conv_mean.weight.grad).all()
    assert (model.conv_mean.weight.grad.abs() > 0).any()

    assert model.conv_var.weight.grad is not None
    assert torch.isfinite(model.conv_var.weight.grad).all()
    assert (model.conv_var.weight.grad.abs() > 0).any()


def test_stateful_recursive_feedback_detachment():
    """Verify that stateful feedback updating buffers does not cause in-place graph corruption."""
    model = MockUNet3D()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    H, W = 8, 8
    L = 10
    target_norm = torch.randn((4, L, H, W))
    traj_noise = torch.randn((4, L, H, W))
    probe_bool = torch.zeros((H, W), dtype=torch.bool)
    probe_bool[0, 0] = True

    pred_buffer = torch.zeros((L, H, W))
    var_buffer = torch.ones((L, H, W))

    traj = {
        "target_norm": target_norm,
        "traj_noise": traj_noise,
        "probe_bool": probe_bool,
        "hide_magnetic": False,
        "pred_buffer": pred_buffer,
        "var_buffer": var_buffer,
        "L": L,
    }

    # Simulate 3 consecutive steps
    losses = []
    for step_idx in range(3):
        t = step_idx
        inp, target_frame, visible_mask = build_trajectory_step_input(
            t=t, traj=traj, window_size=6, uncertainty_mode="original"
        )
        pred, var = model(inp)
        pred_t = pred[:, :, 5:6]
        var_t = var[:, :, 5:6]

        loss = 0.5 * (full_mse_loss(pred_t, target_frame, visible_mask) +
                      predicted_error_mse_loss(var_t, pred_t, target_frame, objective="generalized_kl", detach_mean=False))
        losses.append(loss)

        # Detach and update
        with torch.no_grad():
            pred_detach = pred[0, 3, 5].detach().clone()
            var_detach = var[0, 3, 5].detach().clone()
            pred_detach[probe_bool] = target_norm[3, t, probe_bool]
            var_detach[probe_bool] = 0.0
            traj["pred_buffer"][t] = pred_detach
            traj["var_buffer"][t] = var_detach

    total_batch_loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    total_batch_loss.backward()
    optimizer.step()

    # Step succeeded with no runtime error or in-place error
    assert torch.isfinite(total_batch_loss)


def test_learning_rate_schedule():
    """Verify warmup and cosine decay learning rate schedule."""
    base_lr = 2e-6
    min_lr = 2e-7
    total_steps = 3000
    warmup_steps = 250

    # Step 0: min_lr
    lr_0 = compute_learning_rate(0, total_steps, warmup_steps, base_lr, min_lr)
    assert math.isclose(lr_0, min_lr, rel_tol=1e-5)

    # End of warmup: base_lr
    lr_warmup = compute_learning_rate(warmup_steps, total_steps, warmup_steps, base_lr, min_lr)
    assert math.isclose(lr_warmup, base_lr, rel_tol=1e-5)

    # Halfway through decay
    mid_step = warmup_steps + (total_steps - warmup_steps) // 2
    lr_mid = compute_learning_rate(mid_step, total_steps, warmup_steps, base_lr, min_lr)
    assert min_lr < lr_mid < base_lr

    # Final step: min_lr
    lr_final = compute_learning_rate(total_steps, total_steps, warmup_steps, base_lr, min_lr)
    assert math.isclose(lr_final, min_lr, rel_tol=1e-5)


def test_paired_statistics_calculation():
    """Verify paired t-test and Wilcoxon signed rank test calculations."""
    import pandas as pd

    runs = [f"run_{i}" for i in range(10)]
    # phased outperforms no_feedback on 9 of 10 runs
    mse_nf = [0.00160 + 0.0001 * i for i in range(10)]
    mse_ph = [0.00140 + 0.00008 * i for i in range(10)]

    records = []
    for r, nf, ph in zip(runs, mse_nf, mse_ph):
        records.append({"run_name": r, "b_condition": "B_full", "condition": "no_feedback", "mean_hidden_mse_f24_f48": nf})
        records.append({"run_name": r, "b_condition": "B_full", "condition": "phased_learned", "mean_hidden_mse_f24_f48": ph})

    df = pd.DataFrame(records)
    stats_res = compute_paired_statistics(df, "B_full", "phased_learned", "no_feedback")

    assert stats_res["n_runs"] == 10
    assert stats_res["runs_outperformed"] == "10/10"
    assert stats_res["relative_improvement_pct"] > 0
    assert stats_res["mean_diff"] < 0
    assert stats_res["p_value_wilcoxon"] < 0.05
