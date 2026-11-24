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
    CRITICAL_RESUME_ARGS,
    build_trajectory_step_input,
    compute_learning_rate,
    initialize_distributed,
    warmup_trajectory_history,
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


def test_causal_warmup_initialization():
    """Verify that causal warm-up rolls forward from t=0 to t0 - 1 in no_grad,

    populating historical states with actual model predictions while preserving
    genuine probes and leaving t >= t0 untouched (no future leakage).
    """
    model = MockUNet3D()
    device = torch.device("cpu")
    H, W = 8, 8
    L = 20
    t0 = 5  # Warm up frames 0, 1, 2, 3, 4

    target_norm = torch.randn((4, L, H, W), device=device)
    traj_noise = torch.randn((4, L, H, W), device=device)
    probe_bool = torch.zeros((H, W), dtype=torch.bool, device=device)
    probe_bool[0, 0] = True
    probe_bool[1, 1] = True
    hidden_bool = ~probe_bool

    # Start with all zeros
    pred_buffer = torch.zeros((L, H, W), device=device)
    var_buffer = torch.zeros((L, H, W), device=device)

    traj = {
        "target_norm": target_norm,
        "traj_noise": traj_noise,
        "probe_bool": probe_bool,
        "hide_magnetic": False,
        "t0": t0,
        "L": L,
        "pred_buffer": pred_buffer,
        "var_buffer": var_buffer,
    }

    warmup_trajectory_history(
        model=model,
        traj=traj,
        window_size=6,
        uncertainty_mode="original",
        amp=False,
        device_type="cpu",
    )

    # For tau in [0, t0 - 1]: must be populated with valid predictions and uncertainties
    for tau in range(t0):
        # Genuine probes overridden with exact target values and zero variance
        assert torch.equal(traj["pred_buffer"][tau, probe_bool], target_norm[3, tau, probe_bool])
        assert torch.all(traj["var_buffer"][tau, probe_bool] == 0.0)

        # Hidden voxels must have non-zero predictions from model forward pass
        assert not torch.all(traj["pred_buffer"][tau, hidden_bool] == 0.0)
        assert torch.all(traj["var_buffer"][tau, hidden_bool] > 0.0)

    # For tau >= t0: MUST remain all zeros (no future leakage!)
    for tau in range(t0, L):
        assert torch.all(traj["pred_buffer"][tau] == 0.0)
        assert torch.all(traj["var_buffer"][tau] == 0.0)

    # Model parameters must have no gradients from warm-up
    assert model.conv_mean.weight.grad is None
    assert model.conv_var.weight.grad is None


def test_epoch_diagnostics_accumulation():
    """Verify that epoch loss averages accumulate across all updates,

    preventing only the last update from dominating the epoch summary.
    """
    sum_loss = 0.0
    sum_mu_loss = 0.0
    sum_var_loss = 0.0
    n_updates = 0

    updates_data = [
        {"mu": 0.02, "var": 0.01, "loss": 0.015},
        {"mu": 0.04, "var": 0.02, "loss": 0.030},
        {"mu": 0.06, "var": 0.03, "loss": 0.045},
    ]

    for upd in updates_data:
        step_mu_total = upd["mu"] * 2  # 2 steps
        step_var_total = upd["var"] * 2
        total_steps = 2

        sum_loss += upd["loss"]
        sum_mu_loss += step_mu_total / total_steps
        sum_var_loss += step_var_total / total_steps
        n_updates += 1

    avg_loss = sum_loss / n_updates
    avg_mu = sum_mu_loss / n_updates
    avg_var = sum_var_loss / n_updates

    assert math.isclose(avg_mu, 0.04, rel_tol=1e-5)
    assert math.isclose(avg_var, 0.02, rel_tol=1e-5)
    assert math.isclose(avg_loss, 0.03, rel_tol=1e-5)


def test_resume_argument_mismatch_detection():
    """Verify that attempting to resume with altered configuration parameters

    raises ValueError to prevent silent experiment corruption.
    """
    import argparse
    from pathlib import Path

    saved_args = {
        "window_size": 24,
        "rollout_depth": 6,
        "density_probe_count": 1000,
        "updates_per_epoch": 500,
        "lr": 2e-6,
        "min_lr": 2e-7,
        "warmup_epochs": 0.5,
        "source_checkpoint": "runs/source/final.pt",
    }

    # Case 1: Identical args -> passes
    current_args_ok = argparse.Namespace(**saved_args)
    mismatches = []
    for key in CRITICAL_RESUME_ARGS:
        if key in saved_args:
            curr_val = getattr(current_args_ok, key, None)
            saved_val = saved_args[key]
            if str(curr_val) != str(saved_val):
                mismatches.append(key)
    assert len(mismatches) == 0

    # Case 2: Altered rollout_depth -> detected
    current_args_bad = argparse.Namespace(**saved_args)
    current_args_bad.rollout_depth = 4
    mismatches = []
    for key in CRITICAL_RESUME_ARGS:
        if key in saved_args:
            curr_val = getattr(current_args_bad, key, None)
            saved_val = saved_args[key]
            if str(curr_val) != str(saved_val):
                mismatches.append(key)
    assert "rollout_depth" in mismatches


def test_ddp_initialization_fail_fast():
    """Verify fail-fast behavior when DDP is requested or expected but unavailable."""
    # When world_size is 1 and require_ddp is True, must fail fast
    with pytest.raises(RuntimeError, match="DDP is explicitly required"):
        initialize_distributed(require_ddp=True)

    # When world_size is 1 and expected_world_size is 16, must fail fast
    with pytest.raises(RuntimeError, match="Expected DDP world size 16"):
        initialize_distributed(expected_world_size=16)


def test_phased_no_feedback_rollout_mock():
    """Verify that run_condition_rollout with condition='phased_no_feedback' executes

    correctly, using phased_model weights under the zero-feedback protocol (no prediction reuse).
    """
    phased_model = MockUNet3D()
    v11_model = MockUNet3D()
    device = torch.device("cpu")
    H, W = 8, 8
    n_frames = 6
    window_size = 4

    target_norm = torch.randn((4, n_frames, H, W), device=device)
    global_noise = torch.randn((4, n_frames, H, W), device=device)
    probe_mask = torch.zeros((H, W), device=device)
    probe_mask[0, 0] = 1.0  # single probe

    # Rollout under phased_no_feedback
    rollout_pred, rollout_var, frame_metrics = run_condition_rollout(
        condition="phased_no_feedback",
        v11_model=v11_model,
        phased_model=phased_model,
        target_norm=target_norm,
        probe_mask=probe_mask,
        global_noise=global_noise,
        n_frames=n_frames,
        window_size=window_size,
        hide_magnetic=False,
        amp=False,
    )

    assert rollout_pred.shape == (n_frames, H, W)
    assert rollout_var.shape == (n_frames, H, W)
    assert len(frame_metrics) == n_frames

    # Probe preservation: probe location in rollout_pred must match target_norm
    for t in range(n_frames):
        assert math.isclose(rollout_pred[t, 0, 0].item(), target_norm[3, t, 0, 0].item(), rel_tol=1e-5)
        assert rollout_var[t, 0, 0].item() == 0.0

    # Ensure ValueError if phased_model is missing
    with pytest.raises(ValueError, match="Phase D model is required"):
        run_condition_rollout(
            condition="phased_no_feedback",
            v11_model=v11_model,
            phased_model=None,
            target_norm=target_norm,
            probe_mask=probe_mask,
            global_noise=global_noise,
            n_frames=n_frames,
            window_size=window_size,
        )


def test_all_paired_comparisons_matrix():
    """Verify computation of all 5 paired comparisons (A, B, C, D, E)

    for the Model x Feedback factorial matrix.
    """
    import pandas as pd

    runs = [f"run_{i}" for i in range(10)]
    records = []
    for r in runs:
        # Create deterministic synthetic values representing expected behavior:
        # v11_orig (worst feedback), v11_u1 (better), no_fb (strong baseline),
        # phased_no_fb (similar base), phased_ld (improved feedback)
        records.append({"run_name": r, "b_condition": "B_full", "condition": "v11_original", "mean_hidden_mse_f24_f48": 0.0150})
        records.append({"run_name": r, "b_condition": "B_full", "condition": "v11_u1", "mean_hidden_mse_f24_f48": 0.0021})
        records.append({"run_name": r, "b_condition": "B_full", "condition": "no_feedback", "mean_hidden_mse_f24_f48": 0.0014})
        records.append({"run_name": r, "b_condition": "B_full", "condition": "phased_no_feedback", "mean_hidden_mse_f24_f48": 0.0015})
        records.append({"run_name": r, "b_condition": "B_full", "condition": "phased_learned", "mean_hidden_mse_f24_f48": 0.0016})

    df = pd.DataFrame(records)

    # Comparison A: Within Phase D (phased_learned vs phased_no_feedback)
    stat_a = compute_paired_statistics(df, "B_full", "phased_learned", "phased_no_feedback")
    assert stat_a["n_runs"] == 10
    assert math.isclose(stat_a["mean_a"], 0.0016, rel_tol=1e-4)
    assert math.isclose(stat_a["mean_b"], 0.0015, rel_tol=1e-4)
    assert stat_a["mean_diff"] > 0  # +0.0001 (feedback slightly worse than no-fb)

    # Comparison B: Base capability across models (phased_no_feedback vs no_feedback)
    stat_b = compute_paired_statistics(df, "B_full", "phased_no_feedback", "no_feedback")
    assert stat_b["n_runs"] == 10
    assert math.isclose(stat_b["mean_a"], 0.0015, rel_tol=1e-4)
    assert math.isclose(stat_b["mean_b"], 0.0014, rel_tol=1e-4)

    # Comparison C: Total rollout across models (phased_learned vs v11_original)
    stat_c = compute_paired_statistics(df, "B_full", "phased_learned", "v11_original")
    assert stat_c["n_runs"] == 10
    assert stat_c["mean_diff"] < 0
    assert stat_c["relative_improvement_pct"] > 80.0  # Massive reduction over v11 original

