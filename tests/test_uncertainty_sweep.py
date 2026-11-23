# tests/test_uncertainty_sweep.py

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple
import numpy as np
import pytest
import torch

from run_uncertainty_sweep import (
    build_uncertainty_sweep_input,
    evaluate_uncertainty_sweep_setting,
)


class MockSweepModel(torch.nn.Module):
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


def test_target_frame_input_identical_across_sweep_settings():
    """Verify that current target frame F24 (local index 23) receives

    bitwise identical input values and input variances across all sweep settings.
    """
    torch.manual_seed(42)
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True  # 64 probes
    hidden_mask = ~probe_mask
    global_noise = torch.randn(4, n_frames, H, W)

    history_pred = torch.randn(24, H, W)
    history_var = torch.rand(24, H, W) * 0.5 + 0.05

    settings = ["baseline", "U001", "U01", "U1", "no_feedback_ref"]

    for hide_b in [False, True]:
        inps = {}
        for s in settings:
            inps[s] = build_uncertainty_sweep_input(
                setting=s,
                target_normalized=target_norm,
                probe_mask=probe_mask,
                global_noise=global_noise,
                history_pred=history_pred,
                history_var=history_var,
                target_frame=24,
                hide_magnetic=hide_b,
            )

        ref_vals = inps["baseline"][0, :4, 23]
        ref_var = inps["baseline"][0, 4:, 23]

        for s in settings[1:]:
            s_vals = inps[s][0, :4, 23]
            s_var = inps[s][0, 4:, 23]

            assert torch.equal(ref_vals, s_vals), f"F24 values differ between baseline and {s} (hide_b={hide_b})"
            assert torch.equal(ref_var, s_var), f"F24 variances differ between baseline and {s} (hide_b={hide_b})"

        # Probes must have exact target value and 0.0 input variance
        assert torch.equal(ref_vals[3, probe_mask], target_norm[3, 24, probe_mask])
        assert torch.all(ref_var[3, probe_mask] == 0.0)

        # Hidden voxels must have exact global_noise value and 1.0 input variance
        assert torch.equal(ref_vals[3, hidden_mask], global_noise[3, 24, hidden_mask])
        assert torch.all(ref_var[3, hidden_mask] == 1.0)

        # Magnetic components check
        if not hide_b:
            assert torch.equal(ref_vals[:3], target_norm[:3, 24])
            assert torch.all(ref_var[:3] == 0.0)
        else:
            assert torch.equal(ref_vals[:3], global_noise[:3, 24])
            assert torch.all(ref_var[:3] == 1.0)


def test_historical_prediction_values_identical_across_uncertainty_settings():
    """Verify that historical unobserved Density values across F1..F23 are

    EXACTLY IDENTICAL to history_pred across baseline, U001, U01, and U1,
    while their input uncertainty strictly reflects the configured value.
    """
    torch.manual_seed(101)
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    hidden_mask = ~probe_mask
    global_noise = torch.randn(4, n_frames, H, W)

    history_pred = torch.randn(24, H, W)
    history_var = torch.rand(24, H, W) * 0.5 + 0.05

    inps = {
        s: build_uncertainty_sweep_input(
            s, target_norm, probe_mask, global_noise, history_pred, history_var, 24, False
        )
        for s in ["baseline", "U001", "U01", "U1"]
    }

    expected_u = {
        "baseline": None,  # matches history_var
        "U001": 0.01,
        "U01": 0.1,
        "U1": 1.0,
    }

    for l in range(23):
        tau = 1 + l
        # 1. Probes must be genuine target with zero variance across all settings
        for s in inps:
            assert torch.equal(inps[s][0, 3, l, probe_mask], target_norm[3, tau, probe_mask])
            assert torch.all(inps[s][0, 7, l, probe_mask] == 0.0)

        # 2. Historical prediction values must be BITWISE IDENTICAL across baseline, U001, U01, U1
        base_vals = inps["baseline"][0, 3, l, hidden_mask]
        assert torch.equal(base_vals, history_pred[tau, hidden_mask])

        for s in ["U001", "U01", "U1"]:
            s_vals = inps[s][0, 3, l, hidden_mask]
            assert torch.equal(base_vals, s_vals), f"Historical prediction values differ between baseline and {s} at l={l}"

        # 3. Input variance check
        assert torch.equal(inps["baseline"][0, 7, l, hidden_mask], history_var[tau, hidden_mask])
        for s, u_val in [("U001", 0.01), ("U01", 0.1), ("U1", 1.0)]:
            assert torch.allclose(inps[s][0, 7, l, hidden_mask], torch.tensor(u_val, dtype=torch.float32))


def test_no_target_leakage():
    """Verify that target frame F24 hidden ground-truth values are NEVER exposed in the model input."""
    torch.manual_seed(999)
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    hidden_mask = ~probe_mask
    global_noise = torch.randn(4, n_frames, H, W)

    history_pred = torch.randn(24, H, W)
    history_var = torch.rand(24, H, W) * 0.5 + 0.05

    for s in ["baseline", "U001", "U01", "U1"]:
        inp = build_uncertainty_sweep_input(
            s, target_norm, probe_mask, global_noise, history_pred, history_var, 24, False
        )
        f24_hidden_input = inp[0, 3, 23, hidden_mask]
        f24_hidden_gt = target_norm[3, 24, hidden_mask]

        # The input must match global_noise, NOT target_norm
        assert torch.equal(f24_hidden_input, global_noise[3, 24, hidden_mask])
        assert not torch.equal(f24_hidden_input, f24_hidden_gt)


def test_evaluate_uncertainty_sweep_setting_metrics():
    """Verify evaluate_uncertainty_sweep_setting produces valid metric dictionaries across settings."""
    model = MockSweepModel()
    H, W = 16, 16
    target_norm = torch.randn(4, 48, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    global_noise = torch.randn(4, 48, H, W)

    history_pred = torch.randn(24, H, W)
    history_var = torch.rand(24, H, W) * 0.5 + 0.1

    mean_stat = torch.zeros(4, 1, 1)
    std_stat = torch.ones(4, 1, 1)

    expected_keys = [
        "setting", "target_frame", "density_mse_hidden", "density_rmse_hidden",
        "full_field_rmse", "mean_var_hidden", "u_ratio_hidden",
        "pearson_r", "spearman_r"
    ]

    for setting in ["baseline", "U001", "U01", "U1", "no_feedback_ref"]:
        res = evaluate_uncertainty_sweep_setting(
            model=model,
            mean=mean_stat,
            std=std_stat,
            target_normalized=target_norm,
            probe_mask=probe_mask,
            global_noise=global_noise,
            history_pred=history_pred,
            history_var=history_var,
            setting=setting,
            target_frame=24,
            hide_magnetic=False,
            amp=False,
        )

        for k in expected_keys:
            assert k in res, f"Missing key {k} in sweep result for setting {setting}"

        assert res["setting"] == setting
        assert res["density_mse_hidden"] >= 0.0
        assert res["mean_var_hidden"] >= 0.0
        assert res["u_ratio_hidden"] >= 0.0
        if math.isfinite(res["pearson_r"]):
            assert -1.0 <= res["pearson_r"] <= 1.0
