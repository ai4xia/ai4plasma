from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.unet3d import (  # noqa: E402
    LEGACY_MODEL_VERSION,
    MODEL_VERSION,
    UNet3D,
)


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


def test_residual_unet_starts_as_visible_input_identity():
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=4,
        channel_mults=(1, 2, 4),
        architecture=MODEL_VERSION,
    )
    model.eval()
    x = torch.randn(1, 8, 8, 15, 9)

    with torch.no_grad():
        y = model(x)

    assert torch.equal(y, x[:, :4])

    # The residual is learned everywhere rather than mask-gated: once the head
    # is nonzero, even visible values can be adjusted by the model.
    with torch.no_grad():
        model.out.bias.fill_(0.25)
        adjusted = model(x)
    assert torch.allclose(adjusted, x[:, :4] + 0.25)


def test_residual_unet_has_no_activation_normalization():
    model = UNet3D(
        base_channels=4,
        channel_mults=(1, 2, 4),
        architecture=MODEL_VERSION,
    )
    assert not any(isinstance(module, nn.GroupNorm) for module in model.modules())


def test_legacy_architecture_keeps_groupnorm_for_old_checkpoints():
    model = UNet3D(
        base_channels=4,
        channel_mults=(1, 2, 4),
        architecture=LEGACY_MODEL_VERSION,
    )
    assert any(isinstance(module, nn.GroupNorm) for module in model.modules())
