# tests/test_online_causal_reconstruction.py

from __future__ import annotations

import numpy as np
import pytest
import torch

from run_online_causal_reconstruction import (
    build_causal_window_input,
    reconstruct_online_causal,
)


class MockCausalModel(torch.nn.Module):
    """Deterministic mock UNet3D returning (mean, variance)."""
    uncertainty_aware = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is [B, 8, T, H, W]
        # output is mean [B, 4, T, H, W], variance [B, 4, T, H, W]
        vals = x[:, :4]
        in_var = x[:, 4:]
        # Output prediction is a scaled linear function of values
        mean = vals * 0.8
        # Output variance is half the input variance plus baseline
        var = in_var * 0.5 + 0.05
        return mean, var


def test_causal_window_input_padding_and_positions():
    """Verify window positioning and padding semantics:
    1. At t=0, slots 0..22 are negative global times: input value is Gaussian noise, u_in=1.0.
    2. Current frame t always occupies local index 23.
    3. At t=1, slots 0..21 are negative global times (u_in=1.0), slot 22 is frame 0, slot 23 is frame 1.
    4. At t=23, slots 0..23 correspond to frames 0..23 (no negative padding).
    """
    torch.manual_seed(42)
    T_total, H, W = 48, 8, 8
    target = torch.randn(4, T_total, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True  # 16 probes

    history_pred = torch.zeros(T_total, H, W)
    history_var = torch.ones(T_total, H, W)

    # --- Test t = 0 ---
    g0 = torch.Generator().manual_seed(123)
    model_input_0, _ = build_causal_window_input(
        t=0,
        target_normalized=target,
        probe_mask=probe_mask,
        history_pred=history_pred,
        history_var=history_var,
        hide_magnetic=False,
        window_size=24,
        generator=g0,
    )
    # Check shape: [1, 8, 24, 8, 8]
    assert model_input_0.shape == (1, 8, 24, H, W)
    vals_0 = model_input_0[0, :4]
    var_0 = model_input_0[0, 4:]

    # Local indices 0..22 correspond to global tau = -23 .. -1
    # Must have variance exactly 1.0 across all 4 channels
    assert torch.all(var_0[:, :23] == 1.0), "Historical padded slots must have variance 1.0"
    # Local index 23 corresponds to global tau = 0 (current frame)
    # Magnetic channels 0..2 are observed (variance 0.0)
    assert torch.all(var_0[:3, 23] == 0.0), "B components at t=0 must have variance 0.0 under B_full"
    assert torch.allclose(vals_0[:3, 23], target[:3, 0]), "B values at t=0 must match target"
    # Density (channel 3): probe locations have variance 0.0, non-probe locations have variance 1.0
    assert torch.all(var_0[3, 23, probe_mask] == 0.0), "Density probes must have variance 0.0"
    assert torch.all(var_0[3, 23, ~probe_mask] == 1.0), "Unobserved Density must have variance 1.0"
    assert torch.allclose(vals_0[3, 23, probe_mask], target[3, 0, probe_mask]), "Probe values must match"

    # --- Test t = 1 ---
    # Set history_pred[0] and history_var[0] to specific test values
    history_pred[0] = 3.14
    history_var[0] = 0.42
    g1 = torch.Generator().manual_seed(456)
    model_input_1, _ = build_causal_window_input(
        t=1,
        target_normalized=target,
        probe_mask=probe_mask,
        history_pred=history_pred,
        history_var=history_var,
        hide_magnetic=False,
        window_size=24,
        generator=g1,
    )
    var_1 = model_input_1[0, 4:]
    vals_1 = model_input_1[0, :4]

    # Local indices 0..21: global tau < 0 -> variance 1.0
    assert torch.all(var_1[:, :22] == 1.0)
    # Local index 22: global tau = 0
    # Density at non-probe should reuse history_pred[0] and history_var[0]
    assert torch.allclose(vals_1[3, 22, ~probe_mask], torch.tensor(3.14))
    assert torch.allclose(var_1[3, 22, ~probe_mask], torch.tensor(0.42))
    # Density at probe should be exact target[3, 0] with variance 0.0
    assert torch.allclose(vals_1[3, 22, probe_mask], target[3, 0, probe_mask])
    assert torch.all(var_1[3, 22, probe_mask] == 0.0)
    # Local index 23: global tau = 1 (current frame)
    assert torch.all(var_1[3, 23, ~probe_mask] == 1.0)
    assert torch.all(var_1[3, 23, probe_mask] == 0.0)

    # --- Test t = 23 ---
    model_input_23, _ = build_causal_window_input(
        t=23,
        target_normalized=target,
        probe_mask=probe_mask,
        history_pred=history_pred,
        history_var=history_var,
        hide_magnetic=False,
        window_size=24,
    )
    # In window t=23, local 0 is tau=0 (no negative global indices)
    var_23 = model_input_23[0, 4:]
    assert torch.all(var_23[:3, :] == 0.0), "All B components in valid frames must have variance 0.0"


def test_b_hidden_semantics():
    """Verify B_hidden semantics:
    All B components (channels 0..2) have input variance = 1.0 across all frames,
    and their values are Gaussian noise.
    """
    T_total, H, W = 48, 8, 8
    target = torch.randn(4, T_total, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    history_pred = torch.zeros(T_total, H, W)
    history_var = torch.ones(T_total, H, W)

    model_input, _ = build_causal_window_input(
        t=10,
        target_normalized=target,
        probe_mask=probe_mask,
        history_pred=history_pred,
        history_var=history_var,
        hide_magnetic=True,
        window_size=24,
    )
    var = model_input[0, 4:]
    # Channels 0..2 must be variance 1.0 everywhere in the window
    assert torch.all(var[:3] == 1.0), "Under B_hidden, B input variance must be 1.0 everywhere"


def test_strict_causality_invariance_to_future():
    """STRICT CAUSALITY TEST:
    Modifying future ground-truth target frames (tau > t) while keeping the
    available history and random noise fixed must NOT change the prediction
    at global time t.
    """
    torch.manual_seed(101)
    T_total, H, W = 48, 8, 8
    target_A = torch.randn(4, T_total, H, W)

    # Target B is identical to Target A for t <= 5, but completely different for t > 5
    target_B = target_A.clone()
    target_B[:, 6:] = torch.randn_like(target_B[:, 6:]) * 100.0 + 50.0

    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True

    model = MockCausalModel()

    # Run online causal rollout up to t=5 for both target_A and target_B
    pred_A, var_A = reconstruct_online_causal(
        model=model,
        target_normalized=target_A,
        probe_mask=probe_mask,
        n_frames=6,  # evaluate frames 0 .. 5
        window_size=24,
        hide_magnetic=False,
        seed=42,
        amp=False,
    )

    pred_B, var_B = reconstruct_online_causal(
        model=model,
        target_normalized=target_B,
        probe_mask=probe_mask,
        n_frames=6,  # evaluate frames 0 .. 5
        window_size=24,
        hide_magnetic=False,
        seed=42,
        amp=False,
    )

    # The predictions for frames 0..5 must be mathematically identical
    max_diff_pred = np.max(np.abs(pred_A - pred_B))
    max_diff_var = np.max(np.abs(var_A - var_B))

    assert max_diff_pred == 0.0, f"Future leakage detected! pred diff = {max_diff_pred}"
    assert max_diff_var == 0.0, f"Future leakage detected! var diff = {max_diff_var}"
