# models/unet3d.py

from __future__ import annotations

from typing import Sequence, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


LEGACY_MODEL_VERSION = "unet3d_groupnorm_direct_v1"
MODEL_VERSION = "residual_unet3d_no_activation_norm_v2"


def _num_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    Pick a valid GroupNorm group number.
    """
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1


class ConvBlock3D(nn.Module):
    """Legacy Conv3d + GroupNorm block kept for old checkpoint inference."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_num_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualConvBlock3D(nn.Module):
    """Residual convolution block that preserves activation mean and scale."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, kernel_size=1)
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        # Keep the skip path strictly linear so absolute feature levels can
        # propagate unchanged; only the learned correction is nonlinear.
        return x + residual


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_cls: Type[nn.Module] = ConvBlock3D,
    ):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv = block_cls(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        block_cls: Type[nn.Module] = ConvBlock3D,
    ):
        super().__init__()
        self.conv = block_cls(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Handles odd input sizes such as 154 x 62.
        x = F.interpolate(
            x,
            size=skip.shape[2:],
            mode="trilinear",
            align_corners=False,
        )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Lightweight 3D U-Net.

    Input:
        (B, in_channels, T, X, Z)

    Output:
        (B, out_channels, T, X, Z)
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 4,
        base_channels: int = 24,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        architecture: str = MODEL_VERSION,
    ):
        super().__init__()

        channel_mults = tuple(int(mult) for mult in channel_mults)
        if len(channel_mults) not in {3, 4}:
            raise ValueError(
                "UNet3D supports three or four resolution levels, got "
                f"channel_mults={channel_mults}."
            )

        channels = [base_channels * mult for mult in channel_mults]
        c0, c1, c2 = channels[:3]
        self.num_levels = len(channels)
        self.out_channels = int(out_channels)
        self.architecture = architecture

        if architecture == MODEL_VERSION:
            block_cls = ResidualConvBlock3D
            self.residual_output = True
        elif architecture == LEGACY_MODEL_VERSION:
            block_cls = ConvBlock3D
            self.residual_output = False
        else:
            raise ValueError(f"Unsupported UNet3D architecture: {architecture!r}")
        if self.residual_output and in_channels < out_channels:
            raise ValueError(
                "Residual UNet3D expects its first out_channels inputs to be "
                f"the visible fields, got in_channels={in_channels}, "
                f"out_channels={out_channels}."
            )

        self.enc0 = block_cls(in_channels, c0)
        self.enc1 = DownBlock3D(c0, c1, block_cls)
        self.enc2 = DownBlock3D(c1, c2, block_cls)

        if self.num_levels == 4:
            c3 = channels[3]
            self.enc3 = DownBlock3D(c2, c3, block_cls)
            self.mid = block_cls(c3, c3)
            self.up2 = UpBlock3D(c3, c2, c2, block_cls)
        else:
            self.mid = block_cls(c2, c2)

        self.up1 = UpBlock3D(c2, c1, c1, block_cls)
        self.up0 = UpBlock3D(c1, c0, c0, block_cls)

        self.out = nn.Conv3d(c0, out_channels, kernel_size=1)
        if self.residual_output:
            # Start from identity on visible normalized fields and zero (the
            # physical channel mean) in missing regions. The network learns a
            # correction everywhere; no mask-gated value replacement is used.
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        visible_fields = x[:, : self.out_channels] if self.residual_output else None
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        x = self.enc2(s1)

        if self.num_levels == 4:
            s2 = x
            x = self.enc3(s2)

        x = self.mid(x)

        if self.num_levels == 4:
            x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.up0(x, s0)

        correction = self.out(x)
        if self.residual_output:
            return visible_fields + correction
        return correction


if __name__ == "__main__":
    model = UNet3D(in_channels=8, out_channels=4, base_channels=24)
    x = torch.randn(1, 8, 24, 154, 62)
    y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)
