# models/unet3d.py

from __future__ import annotations

from typing import Sequence, Tuple, Type

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
        pooling_kernel: Tuple[int, int, int] = (2, 2, 2),
    ):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=pooling_kernel)
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


class SpatiotemporalAttention3D(nn.Module):
    """Global self-attention over unified (T, X, Z) tokens with axial 3D RoPE."""

    _HEAD_CANDIDATES: Tuple[int, ...] = (8, 6, 4, 3, 2, 1)

    def __init__(
        self,
        channels: int,
        max_heads: int = 8,
        rope_base: float = 10_000.0,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_heads = next(
            heads
            for heads in self._HEAD_CANDIDATES
            if heads <= max_heads and self.channels % heads == 0
        )
        self.head_dim = self.channels // self.num_heads
        self.rotary_dim = (self.head_dim // 6) * 6
        self.axis_rotary_dim = self.rotary_dim // 3
        self.rope_base = float(rope_base)

        self.norm = nn.LayerNorm(self.channels)
        self.qkv = nn.Linear(self.channels, 3 * self.channels)
        self.out_proj = nn.Linear(self.channels, self.channels)

        # The attention branch starts as an exact zero correction, leaving the
        # convolutional U-Net identity path unchanged at initialization.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _apply_axis_rope(
        self,
        tensor: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply RoPE to adjacent pairs in one axis-specific head slice."""
        axis_dim = tensor.shape[-1]
        inv_freq = self.rope_base ** (
            -torch.arange(
                0,
                axis_dim,
                2,
                device=tensor.device,
                dtype=torch.float32,
            )
            / axis_dim
        )
        angles = positions.to(dtype=torch.float32).unsqueeze(-1) * inv_freq
        cos = angles.cos().to(dtype=tensor.dtype)[None, None]
        sin = angles.sin().to(dtype=tensor.dtype)[None, None]

        even = tensor[..., 0::2]
        odd = tensor[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)

    def _apply_3d_rope(
        self,
        tensor: torch.Tensor,
        time: int,
        size_x: int,
        size_z: int,
    ) -> torch.Tensor:
        """Apply separate temporal, X and Z rotations to (B, H, N, D)."""
        if self.rotary_dim == 0:
            return tensor

        device = tensor.device
        t_positions = (
            torch.arange(time, device=device)
            .view(time, 1, 1)
            .expand(time, size_x, size_z)
            .reshape(-1)
        )
        x_positions = (
            torch.arange(size_x, device=device)
            .view(1, size_x, 1)
            .expand(time, size_x, size_z)
            .reshape(-1)
        )
        z_positions = (
            torch.arange(size_z, device=device)
            .view(1, 1, size_z)
            .expand(time, size_x, size_z)
            .reshape(-1)
        )

        axis_dim = self.axis_rotary_dim
        temporal = self._apply_axis_rope(tensor[..., :axis_dim], t_positions)
        spatial_x = self._apply_axis_rope(
            tensor[..., axis_dim : 2 * axis_dim],
            x_positions,
        )
        spatial_z = self._apply_axis_rope(
            tensor[..., 2 * axis_dim : 3 * axis_dim],
            z_positions,
        )
        return torch.cat(
            (temporal, spatial_x, spatial_z, tensor[..., self.rotary_dim :]),
            dim=-1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time, size_x, size_z = x.shape
        if channels != self.channels:
            raise ValueError(
                f"Expected {self.channels} attention channels, got {channels}."
            )

        tokens = x.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)
        normalized = self.norm(tokens)
        qkv = self.qkv(normalized).reshape(
            batch,
            tokens.shape[1],
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q = self._apply_3d_rope(q, time, size_x, size_z)
        k = self._apply_3d_rope(k, time, size_x, size_z)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, -1, channels)
        correction = self.out_proj(attended)
        output = tokens + correction
        return output.reshape(batch, time, size_x, size_z, channels).permute(
            0, 4, 1, 2, 3
        )


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
        use_attention: bool = False,
        spatial_only_pooling: bool = False,
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
        self.use_attention = bool(use_attention)
        self.spatial_only_pooling = bool(spatial_only_pooling)

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

        pooling_kernel = (1, 2, 2) if self.spatial_only_pooling else (2, 2, 2)
        self.enc0 = block_cls(in_channels, c0)
        self.enc1 = DownBlock3D(c0, c1, block_cls, pooling_kernel)
        self.enc2 = DownBlock3D(c1, c2, block_cls, pooling_kernel)

        if self.num_levels == 4:
            c3 = channels[3]
            self.enc3 = DownBlock3D(c2, c3, block_cls, pooling_kernel)
            self.mid = block_cls(c3, c3)
            self.up2 = UpBlock3D(c3, c2, c2, block_cls)
        else:
            self.mid = block_cls(c2, c2)

        self.up1 = UpBlock3D(c2, c1, c1, block_cls)
        self.up0 = UpBlock3D(c1, c0, c0, block_cls)

        # Do not instantiate these modules when attention is disabled, so old
        # checkpoint state_dict keys remain exactly compatible.
        self.attention_enc2 = (
            SpatiotemporalAttention3D(c2)
            if self.use_attention and self.num_levels == 4
            else None
        )
        mid_channels = channels[-1]
        self.attention_mid = (
            SpatiotemporalAttention3D(mid_channels)
            if self.use_attention
            else None
        )

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
            if self.attention_enc2 is not None:
                x = self.attention_enc2(x)
            s2 = x
            x = self.enc3(s2)

        x = self.mid(x)
        if self.attention_mid is not None:
            x = self.attention_mid(x)

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
