from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.unet3d import UNet3D  # noqa: E402


def test_four_level_unet_preserves_shape():
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=24,
        channel_mults=(1, 2, 4, 8),
    )
    x = torch.randn(1, 8, 24, 31, 17)

    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (1, 4, 24, 31, 17)


def test_legacy_three_level_unet_preserves_shape():
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=16,
        channel_mults=(1, 2, 4),
    )
    x = torch.randn(1, 8, 8, 31, 17)

    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (1, 4, 8, 31, 17)


def test_unet_rejects_unsupported_depth():
    try:
        UNet3D(channel_mults=(1, 2))
    except ValueError:
        return
    raise AssertionError("Expected unsupported U-Net depth to raise ValueError")
