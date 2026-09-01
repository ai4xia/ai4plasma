from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.unet3d import (  # noqa: E402
    LEGACY_MODEL_VERSION,
    MODEL_VERSION,
    SpatiotemporalAttention3D,
    UNet3D,
)


def _full_resolution_shape(
    *,
    use_attention: bool,
    spatial_only_pooling: bool,
):
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=24,
        channel_mults=(1, 2, 4, 8),
        use_attention=use_attention,
        spatial_only_pooling=spatial_only_pooling,
    ).to("meta")
    x = torch.empty(1, 8, 24, 154, 62, device="meta")
    return model, model(x)


def test_full_resolution_attention_off_old_pooling_preserves_shape():
    _, y = _full_resolution_shape(
        use_attention=False,
        spatial_only_pooling=False,
    )
    assert tuple(y.shape) == (1, 4, 24, 154, 62)


def test_full_resolution_attention_on_old_pooling_preserves_shape():
    _, y = _full_resolution_shape(
        use_attention=True,
        spatial_only_pooling=False,
    )
    assert tuple(y.shape) == (1, 4, 24, 154, 62)


def test_full_resolution_attention_on_spatial_pooling_preserves_time_and_shape():
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=24,
        channel_mults=(1, 2, 4, 8),
        use_attention=True,
        spatial_only_pooling=True,
    ).to("meta")
    encoder_shapes = {}
    handles = []
    for name in ("enc0", "enc1", "enc2", "enc3"):
        handles.append(
            getattr(model, name).register_forward_hook(
                lambda _module, _inputs, output, name=name: encoder_shapes.__setitem__(
                    name, tuple(output.shape[2:])
                )
            )
        )

    x = torch.empty(1, 8, 24, 154, 62, device="meta")
    y = model(x)
    for handle in handles:
        handle.remove()

    assert tuple(y.shape) == (1, 4, 24, 154, 62)
    assert encoder_shapes == {
        "enc0": (24, 154, 62),
        "enc1": (24, 77, 31),
        "enc2": (24, 38, 15),
        "enc3": (24, 19, 7),
    }


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


def test_attention_selects_heads_and_partitions_3d_rope():
    attention = SpatiotemporalAttention3D(channels=96)
    assert attention.num_heads == 8
    assert attention.head_dim == 12
    assert attention.rotary_dim == 12
    assert attention.axis_rotary_dim == 4

    assert SpatiotemporalAttention3D(channels=60).num_heads == 6


def test_3d_rope_uses_distinct_temporal_x_and_z_coordinates():
    attention = SpatiotemporalAttention3D(channels=48)
    tensor = torch.zeros(1, attention.num_heads, 8, attention.head_dim)
    tensor[..., 0::2] = 1.0

    rotated = attention._apply_3d_rope(tensor, time=2, size_x=2, size_z=2)
    origin = rotated[0, 0, 0]
    z_shift = rotated[0, 0, 1]
    x_shift = rotated[0, 0, 2]
    t_shift = rotated[0, 0, 4]

    assert not torch.equal(t_shift[0:2], origin[0:2])
    assert torch.equal(t_shift[2:], origin[2:])
    assert not torch.equal(x_shift[2:4], origin[2:4])
    assert torch.equal(x_shift[0:2], origin[0:2])
    assert torch.equal(x_shift[4:], origin[4:])
    assert not torch.equal(z_shift[4:6], origin[4:6])
    assert torch.equal(z_shift[:4], origin[:4])


def test_attention_zero_output_projection_starts_as_identity():
    attention = SpatiotemporalAttention3D(channels=24)
    x = torch.randn(2, 24, 3, 5, 4)

    with torch.no_grad():
        y = attention(x)

    assert torch.equal(y, x)
    assert torch.count_nonzero(attention.out_proj.weight) == 0
    assert torch.count_nonzero(attention.out_proj.bias) == 0


def test_four_level_attention_placement_and_shape():
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=6,
        channel_mults=(1, 2, 4, 8),
        use_attention=True,
    )
    assert isinstance(model.attention_enc2, SpatiotemporalAttention3D)
    assert isinstance(model.attention_mid, SpatiotemporalAttention3D)

    x = torch.randn(1, 8, 8, 16, 16)
    with torch.no_grad():
        y = model(x)
    assert tuple(y.shape) == (1, 4, 8, 16, 16)


def test_three_level_attention_only_uses_mid_block():
    model = UNet3D(
        in_channels=8,
        out_channels=4,
        base_channels=6,
        channel_mults=(1, 2, 4),
        use_attention=True,
    )
    assert model.attention_enc2 is None
    assert isinstance(model.attention_mid, SpatiotemporalAttention3D)


def test_attention_off_keeps_original_state_dict_structure():
    default_model = UNet3D(base_channels=4, channel_mults=(1, 2, 4))
    attention_off_model = UNet3D(
        base_channels=4,
        channel_mults=(1, 2, 4),
        use_attention=False,
    )

    assert default_model.state_dict().keys() == attention_off_model.state_dict().keys()
    assert not any(
        key.startswith(("attention_enc2.", "attention_mid."))
        for key in attention_off_model.state_dict()
    )


def test_spatial_pooling_keeps_parameter_and_state_dict_structure():
    old_pooling_model = UNet3D(
        base_channels=6,
        channel_mults=(1, 2, 4, 8),
        use_attention=True,
        spatial_only_pooling=False,
    )
    spatial_pooling_model = UNet3D(
        base_channels=6,
        channel_mults=(1, 2, 4, 8),
        use_attention=True,
        spatial_only_pooling=True,
    )

    assert old_pooling_model.enc1.pool.kernel_size == (2, 2, 2)
    assert spatial_pooling_model.enc1.pool.kernel_size == (1, 2, 2)
    assert (
        old_pooling_model.state_dict().keys()
        == spatial_pooling_model.state_dict().keys()
    )
    assert sum(p.numel() for p in old_pooling_model.parameters()) == sum(
        p.numel() for p in spatial_pooling_model.parameters()
    )
    spatial_pooling_model.load_state_dict(old_pooling_model.state_dict())


def test_zero_initialized_attention_preserves_pretrained_unet_output():
    attention_off_model = UNet3D(
        base_channels=6,
        channel_mults=(1, 2, 4, 8),
        use_attention=False,
    ).eval()
    with torch.no_grad():
        attention_off_model.out.weight.normal_()
        attention_off_model.out.bias.normal_()

    attention_on_model = UNet3D(
        base_channels=6,
        channel_mults=(1, 2, 4, 8),
        use_attention=True,
    ).eval()
    incompatible = attention_on_model.load_state_dict(
        attention_off_model.state_dict(),
        strict=False,
    )
    assert not incompatible.unexpected_keys
    assert all(
        key.startswith(("attention_enc2.", "attention_mid."))
        for key in incompatible.missing_keys
    )

    x = torch.randn(1, 8, 8, 16, 16)
    with torch.no_grad():
        attention_off_output = attention_off_model(x)
        attention_on_output = attention_on_model(x)
    assert torch.equal(attention_on_output, attention_off_output)
