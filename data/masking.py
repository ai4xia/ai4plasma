# data/masking.py

from __future__ import annotations

from typing import Tuple

import torch


def random_voxel_mask(
    block: torch.Tensor,
    keep_prob: float = 0.2,
) -> torch.Tensor:
    """
    block: (B, C, T, X, Z)

    Return:
        mask with same shape.
        1 = visible
        0 = hidden / target to recover
    """
    if not (0.0 < keep_prob <= 1.0):
        raise ValueError(f"keep_prob must be in (0, 1], got {keep_prob}")

    return (torch.rand_like(block) < keep_prob).to(block.dtype)


def make_visible_input(
    block: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return block * mask


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MSE only on hidden region.

    visible_mask:
        1 = visible input
        0 = hidden target region
    """
    hidden_mask = 1.0 - visible_mask
    loss = ((pred - target) ** 2) * hidden_mask
    return loss.sum() / (hidden_mask.sum() + eps)


def masked_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    visible_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MAE only on hidden region.
    """
    hidden_mask = 1.0 - visible_mask
    loss = torch.abs(pred - target) * hidden_mask
    return loss.sum() / (hidden_mask.sum() + eps)


def full_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    MSE over all voxels, including visible and hidden regions.
    """
    return torch.mean((pred - target) ** 2)


def full_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    MAE over all voxels, including visible and hidden regions.
    """
    return torch.mean(torch.abs(pred - target))


def temporal_interpolation_mask(
    block: torch.Tensor,
    keep_every: int = 2,
) -> torch.Tensor:
    if keep_every < 1:
        raise ValueError(f"keep_every must be >= 1, got {keep_every}")

    mask = torch.zeros_like(block)
    mask[:, :, ::keep_every, :, :] = 1.0
    return mask


def future_extrapolation_mask(
    block: torch.Tensor,
    num_context_frames: int,
) -> torch.Tensor:
    B, C, T, X, Z = block.shape

    if not (1 <= num_context_frames < T):
        raise ValueError(
            f"num_context_frames must be in [1, T-1], "
            f"got {num_context_frames}, T={T}"
        )

    mask = torch.zeros_like(block)
    mask[:, :, :num_context_frames, :, :] = 1.0
    return mask


def channel_completion_mask(
    block: torch.Tensor,
    visible_channels: Tuple[int, ...],
) -> torch.Tensor:
    B, C, T, X, Z = block.shape

    mask = torch.zeros_like(block)
    for c in visible_channels:
        if not (0 <= c < C):
            raise ValueError(f"Invalid channel index {c}, C={C}")
        mask[:, c, :, :, :] = 1.0

    return mask