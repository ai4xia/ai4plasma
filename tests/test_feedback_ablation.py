# tests/test_feedback_ablation.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple
import numpy as np
import pytest
import torch

from run_feedback_ablation import (
    build_feedback_ablation_input,
    evaluate_feedback_strategy,
    run_online_history_rollout,
)


class MockFeedbackModel(torch.nn.Module):
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


def test_target_frame_input_identical_across_strategies():
    """Verify that current target frame F24 (local index 23) receives

    bitwise identical input values and input variances across Strategies A, B, and C.
    """
    torch.manual_seed(42)
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True  # 64 probes
    global_noise = torch.randn(4, n_frames, H, W)

    history_pred = torch.randn(24, H, W)
    history_var = torch.rand(24, H, W) * 0.5

    for hide_b in [False, True]:
        inp_A = build_feedback_ablation_input(
            "A", target_norm, probe_mask, global_noise,
            history_pred=None, history_var=None,
            target_frame=24, hide_magnetic=hide_b
        )
        inp_B = build_feedback_ablation_input(
            "B", target_norm, probe_mask, global_noise,
            history_pred=history_pred, history_var=history_var,
            target_frame=24, hide_magnetic=hide_b
        )
        inp_C = build_feedback_ablation_input(
            "C", target_norm, probe_mask, global_noise,
            history_pred=None, history_var=None,
            target_frame=24, hide_magnetic=hide_b
        )

        # Local index 23 is global frame 24
        f24_A_vals = inp_A[0, :4, 23]
        f24_B_vals = inp_B[0, :4, 23]
        f24_C_vals = inp_C[0, :4, 23]

        f24_A_var = inp_A[0, 4:, 23]
        f24_B_var = inp_B[0, 4:, 23]
        f24_C_var = inp_C[0, 4:, 23]

        # Bitwise identical values
        assert torch.equal(f24_A_vals, f24_B_vals), f"Values differ between A and B (hide_b={hide_b})"
        assert torch.equal(f24_A_vals, f24_C_vals), f"Values differ between A and C (hide_b={hide_b})"

        # Bitwise identical input variances
        assert torch.equal(f24_A_var, f24_B_var), f"Variances differ between A and B (hide_b={hide_b})"
        assert torch.equal(f24_A_var, f24_C_var), f"Variances differ between A and C (hide_b={hide_b})"

        # Probes must have exact target value and 0.0 input variance
        assert torch.equal(f24_A_vals[3, probe_mask], target_norm[3, 24, probe_mask])
        assert torch.all(f24_A_var[3, probe_mask] == 0.0)

        # Hidden voxels must have exact global_noise value and 1.0 input variance
        hidden_mask = ~probe_mask
        assert torch.equal(f24_A_vals[3, hidden_mask], global_noise[3, 24, hidden_mask])
        assert torch.all(f24_A_var[3, hidden_mask] == 1.0)

        # Magnetic components check
        if not hide_b:
            assert torch.equal(f24_A_vals[:3], target_norm[:3, 24])
            assert torch.all(f24_A_var[:3] == 0.0)
        else:
            assert torch.equal(f24_A_vals[:3], global_noise[:3, 24])
            assert torch.all(f24_A_var[:3] == 1.0)


def test_historical_conditioning_differentiates_strategies():
    """Verify that historical frames F1..F23 strictly follow each strategy definition:

    - Strategy A: unobserved Density has global_noise, u_in = 1.0
    - Strategy B: unobserved Density has history_pred, u_in = history_var
    - Strategy C: unobserved Density has target_norm, u_in = 0.0
    - Probes in all strategies have target_norm, u_in = 0.0
    """
    torch.manual_seed(101)
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    hidden_mask = ~probe_mask
    global_noise = torch.randn(4, n_frames, H, W)

    history_pred = torch.randn(24, H, W)
    history_var = torch.rand(24, H, W) * 0.5 + 0.1

    inp_A = build_feedback_ablation_input(
        "A", target_norm, probe_mask, global_noise,
        history_pred=None, history_var=None,
        target_frame=24, hide_magnetic=False
    )
    inp_B = build_feedback_ablation_input(
        "B", target_norm, probe_mask, global_noise,
        history_pred=history_pred, history_var=history_var,
        target_frame=24, hide_magnetic=False
    )
    inp_C = build_feedback_ablation_input(
        "C", target_norm, probe_mask, global_noise,
        history_pred=None, history_var=None,
        target_frame=24, hide_magnetic=False
    )

    for l in range(23):
        tau = 1 + l  # global frames 1 through 23
        # Probes must be genuine target across all strategies
        assert torch.equal(inp_A[0, 3, l, probe_mask], target_norm[3, tau, probe_mask])
        assert torch.equal(inp_B[0, 3, l, probe_mask], target_norm[3, tau, probe_mask])
        assert torch.equal(inp_C[0, 3, l, probe_mask], target_norm[3, tau, probe_mask])
        assert torch.all(inp_A[0, 7, l, probe_mask] == 0.0)
        assert torch.all(inp_B[0, 7, l, probe_mask] == 0.0)
        assert torch.all(inp_C[0, 7, l, probe_mask] == 0.0)

        # Hidden voxels for Strategy A
        assert torch.equal(inp_A[0, 3, l, hidden_mask], global_noise[3, tau, hidden_mask])
        assert torch.all(inp_A[0, 7, l, hidden_mask] == 1.0)

        # Hidden voxels for Strategy B
        assert torch.equal(inp_B[0, 3, l, hidden_mask], history_pred[tau, hidden_mask])
        assert torch.equal(inp_B[0, 7, l, hidden_mask], history_var[tau, hidden_mask])

        # Hidden voxels for Strategy C
        assert torch.equal(inp_C[0, 3, l, hidden_mask], target_norm[3, tau, hidden_mask])
        assert torch.all(inp_C[0, 7, l, hidden_mask] == 0.0)


def test_online_history_rollout_mock():
    """Verify that run_online_history_rollout produces outputs of shape [24, H, W]

    and correctly preserves probe values.
    """
    model = MockFeedbackModel()
    H, W = 16, 16
    target_norm = torch.randn(4, 48, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    global_noise = torch.randn(4, 48, H, W)

    pred, var = run_online_history_rollout(
        model=model,
        target_normalized=target_norm,
        probe_mask=probe_mask,
        global_noise=global_noise,
        history_steps=24,
        window_size=24,
        hide_magnetic=False,
        amp=False,
    )

    assert pred.shape == (24, H, W)
    assert var.shape == (24, H, W)

    # Check probe preservation at each historical frame
    for t in range(24):
        assert torch.equal(pred[t, probe_mask], target_norm[3, t, probe_mask])
        assert torch.all(var[t, probe_mask] == 0.0)


def test_evaluate_feedback_strategy_metrics():
    """Verify evaluate_feedback_strategy produces valid metric dictionaries."""
    model = MockFeedbackModel()
    H, W = 16, 16
    target_norm = torch.randn(4, 48, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    global_noise = torch.randn(4, 48, H, W)

    mean_stat = torch.zeros(4, 1, 1)
    std_stat = torch.ones(4, 1, 1)

    res = evaluate_feedback_strategy(
        model=model,
        mean=mean_stat,
        std=std_stat,
        target_normalized=target_norm,
        probe_mask=probe_mask,
        global_noise=global_noise,
        history_pred=None,
        history_var=None,
        strategy="A",
        target_frame=24,
        hide_magnetic=False,
        amp=False,
    )

    expected_keys = [
        "strategy", "target_frame", "density_mse_hidden", "density_rmse_hidden",
        "full_field_rmse", "mean_var_hidden", "u_ratio_hidden",
        "pearson_r", "spearman_r"
    ]
    for k in expected_keys:
        assert k in res, f"Missing key {k} in evaluate_feedback_strategy result"

    assert res["density_mse_hidden"] >= 0.0
    assert res["mean_var_hidden"] >= 0.0
    assert res["u_ratio_hidden"] >= 0.0
    if math.isfinite(res["pearson_r"]):
        assert -1.0 <= res["pearson_r"] <= 1.0
