# tests/test_full_online_uncertainty_ablation.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple
import numpy as np
import pytest
import torch

from run_full_online_uncertainty_ablation import (
    build_rollout_step_input,
    run_online_causal_rollout,
)


class MockCausalRolloutModel(torch.nn.Module):
    """Deterministic mock UNet3D returning (mean, variance)."""
    uncertainty_aware = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is [B, 8, T, H, W]
        # output is mean [B, 4, T, H, W], variance [B, 4, T, H, W]
        vals = x[:, :4]
        in_var = x[:, 4:]
        mean = vals * 0.95 + 0.01
        var = in_var * 0.5 + 0.01
        return mean, var


def test_current_frame_bitwise_identical_across_strategies():
    """Verify that current target frame t (local index 23) receives

    bitwise identical values and input variances across Original, U1, and No-feedback.
    """
    torch.manual_seed(42)
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True  # 64 probes
    hidden_mask = ~probe_mask
    global_noise = torch.randn(4, n_frames, H, W)

    history_pred = torch.randn(n_frames, H, W)
    history_var = torch.rand(n_frames, H, W) * 0.5 + 0.05

    strategies = ["original", "u1", "no_feedback"]

    # Test across multiple representative frames: cold-start, transition, and fully populated
    for t in [0, 5, 23, 35, 47]:
        for hide_b in [False, True]:
            inps = {}
            for s in strategies:
                inps[s] = build_rollout_step_input(
                    t=t,
                    strategy=s,
                    target_normalized=target_norm,
                    probe_mask=probe_mask,
                    global_noise=global_noise,
                    history_pred=history_pred,
                    history_var=history_var,
                    hide_magnetic=hide_b,
                    window_size=24,
                )

            ref_vals = inps["original"][0, :4, 23]
            ref_var = inps["original"][0, 4:, 23]

            for s in ["u1", "no_feedback"]:
                s_vals = inps[s][0, :4, 23]
                s_var = inps[s][0, 4:, 23]

                # Bitwise identical values at current frame
                assert torch.equal(ref_vals, s_vals), f"Current frame values differ between original and {s} at t={t} (hide_b={hide_b})"
                assert torch.equal(ref_var, s_var), f"Current frame variances differ between original and {s} at t={t} (hide_b={hide_b})"

            # Genuine probes at current frame have exact target values and zero variance
            assert torch.equal(ref_vals[3, probe_mask], target_norm[3, t, probe_mask])
            assert torch.all(ref_var[3, probe_mask] == 0.0)

            # Hidden voxels at current frame have exact global noise and variance 1.0
            assert torch.equal(ref_vals[3, hidden_mask], global_noise[3, t, hidden_mask])
            assert torch.all(ref_var[3, hidden_mask] == 1.0)

            # Magnetic fields check
            if not hide_b:
                assert torch.equal(ref_vals[:3], target_norm[:3, t])
                assert torch.all(ref_var[:3] == 0.0)
            else:
                assert torch.equal(ref_vals[:3], global_noise[:3, t])
                assert torch.all(ref_var[:3] == 1.0)


def test_historical_conditioning_and_probe_preservation():
    """Verify that historical unobserved Density strictly follows strategy definitions

    while genuine historical probes remain exact across all strategies.
    """
    torch.manual_seed(101)
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    hidden_mask = ~probe_mask
    global_noise = torch.randn(4, n_frames, H, W)

    history_pred = torch.randn(n_frames, H, W)
    history_var = torch.rand(n_frames, H, W) * 0.5 + 0.05

    t = 30  # Settled regime (tau in 7..30)
    window_size = 24

    inp_orig = build_rollout_step_input(t, "original", target_norm, probe_mask, global_noise, history_pred, history_var, False, window_size)
    inp_u1 = build_rollout_step_input(t, "u1", target_norm, probe_mask, global_noise, history_pred, history_var, False, window_size)
    inp_nofb = build_rollout_step_input(t, "no_feedback", target_norm, probe_mask, global_noise, history_pred, history_var, False, window_size)

    for l in range(window_size - 1):
        tau = t - window_size + 1 + l  # historical frames 7..29
        assert 0 <= tau < t

        # Genuine probes: exact target and zero variance across all strategies
        for inp in [inp_orig, inp_u1, inp_nofb]:
            assert torch.equal(inp[0, 3, l, probe_mask], target_norm[3, tau, probe_mask])
            assert torch.all(inp[0, 7, l, probe_mask] == 0.0)

        # Original: history_pred with history_var
        assert torch.equal(inp_orig[0, 3, l, hidden_mask], history_pred[tau, hidden_mask])
        assert torch.equal(inp_orig[0, 7, l, hidden_mask], history_var[tau, hidden_mask])

        # U1: history_pred with u_in = 1.0
        assert torch.equal(inp_u1[0, 3, l, hidden_mask], history_pred[tau, hidden_mask])
        assert torch.all(inp_u1[0, 7, l, hidden_mask] == 1.0)

        # No-feedback: global_noise with u_in = 1.0
        assert torch.equal(inp_nofb[0, 3, l, hidden_mask], global_noise[3, tau, hidden_mask])
        assert torch.all(inp_nofb[0, 7, l, hidden_mask] == 1.0)


def test_no_feedback_independence_from_predictions():
    """Verify that changing history_pred has zero effect on the No-feedback strategy."""
    torch.manual_seed(202)
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    global_noise = torch.randn(4, n_frames, H, W)

    history_pred_1 = torch.randn(n_frames, H, W)
    history_pred_2 = torch.randn(n_frames, H, W) + 100.0  # radically different
    history_var = torch.rand(n_frames, H, W) * 0.5 + 0.05

    inp_1 = build_rollout_step_input(25, "no_feedback", target_norm, probe_mask, global_noise, history_pred_1, history_var, False, 24)
    inp_2 = build_rollout_step_input(25, "no_feedback", target_norm, probe_mask, global_noise, history_pred_2, history_var, False, 24)

    assert torch.equal(inp_1, inp_2), "No-feedback input depended on history_pred!"


def test_causal_padding_for_early_frames():
    """Verify that for t < 23, nonexistent historical frames (tau < 0) are padded with noise and u=1.0."""
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    global_noise = torch.randn(4, n_frames, H, W)
    history_pred = torch.randn(n_frames, H, W)
    history_var = torch.rand(n_frames, H, W) * 0.5

    t = 5
    window_size = 24
    inp = build_rollout_step_input(t, "original", target_norm, probe_mask, global_noise, history_pred, history_var, False, window_size)

    # For t=5 and window_size=24, local indices l in 0 .. 17 have tau in -18 .. -1
    for l in range(window_size - 1 - t):
        tau = t - window_size + 1 + l
        assert tau < 0
        # All 4 channels must have variance 1.0
        assert torch.all(inp[0, 4:, l] == 1.0)


def test_no_target_leakage():
    """Verify that target frame hidden ground truth is never exposed in the input."""
    torch.manual_seed(303)
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    hidden_mask = ~probe_mask
    global_noise = torch.randn(4, n_frames, H, W)

    history_pred = torch.randn(n_frames, H, W)
    history_var = torch.rand(n_frames, H, W) * 0.5

    for s in ["original", "u1", "no_feedback"]:
        inp = build_rollout_step_input(24, s, target_norm, probe_mask, global_noise, history_pred, history_var, False, 24)
        hidden_input = inp[0, 3, 23, hidden_mask]
        hidden_gt = target_norm[3, 24, hidden_mask]

        assert torch.equal(hidden_input, global_noise[3, 24, hidden_mask])
        assert not torch.equal(hidden_input, hidden_gt)


def test_mock_rollout_execution():
    """Verify that run_online_causal_rollout executes properly across all strategies."""
    model = MockCausalRolloutModel()
    H, W = 16, 16
    n_frames = 10
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    global_noise = torch.randn(4, n_frames, H, W)
    mean_stat = torch.zeros(4, 1, 1)
    std_stat = torch.ones(4, 1, 1)

    for s in ["original", "u1", "no_feedback"]:
        pred, var, metrics = run_online_causal_rollout(
            strategy=s,
            model=model,
            mean=mean_stat,
            std=std_stat,
            target_normalized=target_norm,
            probe_mask=probe_mask,
            global_noise=global_noise,
            n_frames=n_frames,
            window_size=6,
            hide_magnetic=False,
            amp=False,
        )

        assert pred.shape == (n_frames, H, W)
        assert var.shape == (n_frames, H, W)
        assert len(metrics) == n_frames

        # Check probe preservation at all frames
        for t in range(n_frames):
            assert torch.equal(pred[t, probe_mask], target_norm[3, t, probe_mask])
            assert torch.all(var[t, probe_mask] == 0.0)
            assert metrics[t]["density_mse_hidden"] >= 0.0
