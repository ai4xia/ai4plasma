from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_mask_patterns_unet3d import (  # noqa: E402
    build_density_forecast_rows,
    build_density_only_multifunction_rows,
    build_density_superres_rows,
    build_magnetic_ablation_rows,
    default_density_forecast_visible_frames,
    compute_normalized_metrics,
    normalized_residual,
    relative_error_epsilon,
    select_run_t0_index,
    write_animation,
)


SHAPE = (1, 4, 3, 154, 62)


def make_block() -> torch.Tensor:
    return torch.zeros(SHAPE)


def make_generator() -> torch.Generator:
    return torch.Generator().manual_seed(1234)


def test_stabilized_pointwise_normalized_residual_is_finite_near_zero():
    target = np.asarray([0.0, 1.0, -2.0])
    prediction = np.asarray([1.0, 2.0, -1.0])
    epsilon_abs = relative_error_epsilon([target], epsilon_fraction=0.05)
    residual = normalized_residual(prediction, target, epsilon_abs)
    nrmse, nmae = compute_normalized_metrics(residual)

    assert epsilon_abs > 0
    assert np.isfinite(residual).all()
    assert np.isfinite(nrmse)
    assert np.isfinite(nmae)


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
    block_mask = next(mask for name, _, mask in rows if name == "spatial_block")
    midpoint = SHAPE[-2] // 2
    assert torch.all(block_mask[:, 3:4, :, :midpoint] == 1)
    assert torch.all(block_mask[:, 3:4, :, midpoint:] == 0)


def test_density_superres_uses_exact_requested_probe_counts():
    targets = [30, 20, 10, 0]
    rows = build_density_superres_rows(make_block(), targets)

    for (_, _, mask), target in zip(rows, targets):
        assert torch.all(mask[:, :3] == 1)
        assert int(mask[0, 3, 0].sum()) == target


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


def test_density_forecast_uses_complete_prefixes_and_full_magnetic_history():
    block = torch.zeros(1, 4, 24, 20, 12)
    history_lengths = [23, 18, 12, 6]
    rows = build_density_forecast_rows(block, history_lengths)

    assert len(rows) == 4
    for (_, label, mask), history_length in zip(rows, history_lengths):
        assert torch.all(mask[:, :3] == 1)
        assert torch.all(mask[:, 3:4, :history_length] == 1)
        assert torch.all(mask[:, 3:4, history_length:] == 0)
        assert f"forecast horizon={24 - history_length} step" in label

    for (_, _, longer), (_, _, shorter) in zip(rows, rows[1:]):
        assert torch.all(shorter[:, 3:4] <= longer[:, 3:4])


def test_density_forecast_defaults_scale_with_context_length():
    assert default_density_forecast_visible_frames(24) == [23, 18, 12, 6]
    assert default_density_forecast_visible_frames(8) == [7, 6, 4, 2]


class FakeDataset:
    samples = [
        (0, "train_run", 0),
        (0, "canonical_run", 70),
        (0, "canonical_run", 72),
        (0, "canonical_run", 74),
    ]


def test_named_sample_selection_uses_exact_run_and_t0():
    index = select_run_t0_index(
        dataset=FakeDataset(),
        val_runs={"canonical_run"},
        run_name="canonical_run",
        t0=72,
    )
    assert index == 2


def test_named_sample_selection_rejects_training_run():
    try:
        select_run_t0_index(
            dataset=FakeDataset(),
            val_runs={"canonical_run"},
            run_name="train_run",
            t0=0,
        )
    except ValueError as error:
        assert "not in split.json validation runs" in str(error)
    else:
        raise AssertionError("Expected a non-validation run to be rejected.")


def test_gif_animation_has_no_infinite_loop_extension(tmp_path):
    frame_paths = []
    for index, color in enumerate(("red", "blue")):
        frame_path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (8, 8), color=color).save(frame_path)
        frame_paths.append(frame_path)

    gif_path = tmp_path / "single-play.gif"
    write_animation(frame_paths, gif_path, fps=2.0)

    with Image.open(gif_path) as animation:
        assert animation.n_frames == 2
        assert "loop" not in animation.info
