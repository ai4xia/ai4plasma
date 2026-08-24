from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_mask_patterns_unet3d import (  # noqa: E402
    build_density_only_multifunction_rows,
    build_density_superres_rows,
    build_magnetic_ablation_rows,
)


SHAPE = (1, 4, 3, 154, 62)


def make_block() -> torch.Tensor:
    return torch.zeros(SHAPE)


def make_generator() -> torch.Generator:
    return torch.Generator().manual_seed(1234)


def test_multifunction_masks_only_density():
    rows = build_density_only_multifunction_rows(
        block=make_block(),
        patterns=[
            "spatial_random",
            "spatial_grid",
            "spatial_block",
            "temporal_random",
        ],
        mask_fraction=0.8,
        block_fraction=0.5,
        grid_stride=4,
        magnetic_grid_stride=2,
        generator=make_generator(),
    )

    assert len(rows) == 4
    assert all(torch.all(mask[:, :3] == 1) for _, _, mask in rows)


def test_density_superres_tracks_requested_probe_ratios():
    targets = [0.08, 0.04, 0.02, 0.01]
    rows = build_density_superres_rows(make_block(), targets, make_generator())

    for (_, _, mask), target in zip(rows, targets):
        assert torch.all(mask[:, :3] == 1)
        actual = float(mask[:, 3:4].mean())
        assert abs(actual - target) < 0.003


def test_magnetic_ablation_is_nested_and_reuses_density_grid():
    targets = [1.0, 0.8, 0.6, 0.4]
    rows = build_magnetic_ablation_rows(
        block=make_block(),
        magnetic_visible_fractions=targets,
        density_visible_fraction=0.08,
        generator=make_generator(),
    )

    density_reference = rows[0][2][:, 3:4]
    for (_, _, mask), target in zip(rows, targets):
        assert abs(float(mask[:, :3].mean()) - target) < 1e-4
        assert torch.equal(mask[:, 3:4], density_reference)

    for (_, _, higher), (_, _, lower) in zip(rows, rows[1:]):
        assert torch.all(lower[:, :3] <= higher[:, :3])
