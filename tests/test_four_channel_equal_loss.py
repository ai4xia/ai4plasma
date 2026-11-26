# tests/test_four_channel_equal_loss.py

from __future__ import annotations

import torch

from data.masking import (
    four_channel_equal_mse_loss,
    full_mse_loss,
    reconstruction_mse_loss,
)


def test_four_channel_equal_mse_loss_exact_weights():
    """Verify 0.25 per channel and 0.5 visible/hidden split per channel."""
    pred = torch.zeros((1, 4, 2, 4, 4), dtype=torch.float32)
    target = torch.zeros((1, 4, 2, 4, 4), dtype=torch.float32)
    mask = torch.zeros((1, 4, 2, 4, 4), dtype=torch.float32)
    mask[:, :, :, :2, :] = 1.0  # 50% visible, 50% hidden

    # Put distinct errors in each channel (visible=1, hidden=2 for ch c -> 0.5*(1+4)=2.5 * scale)
    errs = [1.0, 2.0, 3.0, 4.0]
    for c, e in enumerate(errs):
        target[:, c] = e

    # Equal weights: 0.25 * mean(1^2, 4^2 ... wait: both vis and hid have error e -> L_c = e^2)
    expected_equal = 0.25 * sum(e**2 for e in errs)
    got_equal = four_channel_equal_mse_loss(pred, target, mask)
    assert torch.isclose(got_equal, torch.tensor(expected_equal), atol=1e-6)

    # Group-balanced: 0.5 * mean(1, 4, 9) + 0.5 * 16
    expected_group = 0.5 * ((1.0 + 4.0 + 9.0) / 3.0) + 0.5 * 16.0
    got_group = full_mse_loss(pred, target, mask)
    assert torch.isclose(got_group, torch.tensor(expected_group), atol=1e-6)
    assert torch.isclose(reconstruction_mse_loss(pred, target, mask, "group_balanced"), got_group)
    assert torch.isclose(reconstruction_mse_loss(pred, target, mask, "channel_equal"), got_equal)


def test_four_channel_equal_mse_loss_extreme_masks_and_nans():
    """Verify fully visible, fully hidden, nonfinite temporal padding, and finite gradients."""
    pred = torch.randn((2, 4, 6, 4, 4), requires_grad=True)
    target = torch.randn((2, 4, 6, 4, 4))
    target[:, :, -2:, :, :] = float("nan")  # padded frames

    for vis_val in (0.0, 1.0):
        mask = torch.full((2, 4, 6, 4, 4), vis_val)
        loss = four_channel_equal_mse_loss(pred, target, mask)
        assert torch.isfinite(loss)
        loss.backward()
        assert pred.grad is not None and torch.isfinite(pred.grad).all()
        pred.grad.zero_()
