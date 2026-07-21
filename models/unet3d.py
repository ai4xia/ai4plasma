# models/unet3d.py

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(num_channels: int, max_groups: int = 8) -> int:
    """
    Pick a valid GroupNorm group number.
    """
    for g in reversed(range(1, max_groups + 1)):
        if num_channels % g == 0:
            return g
    return 1


class ConvBlock3D(nn.Module):
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


class DownBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv = ConvBlock3D(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = ConvBlock3D(in_channels + skip_channels, out_channels)

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
        base_channels: int = 16,
        channel_mults: Sequence[int] = (1, 2, 4),
    ):
        super().__init__()

        c0 = base_channels * channel_mults[0]
        c1 = base_channels * channel_mults[1]
        c2 = base_channels * channel_mults[2]

        self.enc0 = ConvBlock3D(in_channels, c0)
        self.enc1 = DownBlock3D(c0, c1)
        self.enc2 = DownBlock3D(c1, c2)

        self.mid = ConvBlock3D(c2, c2)

        self.up1 = UpBlock3D(c2, c1, c1)
        self.up0 = UpBlock3D(c1, c0, c0)

        self.out = nn.Conv3d(c0, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.enc0(x)
        s1 = self.enc1(s0)
        x = self.enc2(s1)

        x = self.mid(x)

        x = self.up1(x, s1)
        x = self.up0(x, s0)

        return self.out(x)


if __name__ == "__main__":
    model = UNet3D(in_channels=8, out_channels=4, base_channels=16)
    x = torch.randn(2, 8, 8, 154, 62)
    y = model(x)
    print("input:", x.shape)
    print("output:", y.shape)
