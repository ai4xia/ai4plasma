from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize_mask_patterns_unet3d import (  # noqa: E402
    DEFAULT_MAGNETIC_ABLATION_VISIBLE_FRACTIONS,
    apply_log_yscale_if_strictly_positive,
    build_density_forecast_rows,
    build_density_only_multifunction_rows,
    build_density_superres_rows,
    build_magnetic_ablation_rows,
    aggregate_run_window_profiles,
    compute_ay_jy,
    compute_jy,
    collect_validation_statistics,
    default_density_forecast_visible_frames,
    format_magnetic_visible_percent,
    magnetic_ablation_visible_count,
    compute_normalized_metrics,
    normalized_residual,
    save_information_suite_error_plot,
    save_validation_statistics_plot,
    select_validation_statistics_indices,
    select_run_t0_index,
    validation_statistics_legend_label,
    write_animation,
)


SHAPE = (1, 4, 3, 154, 62)


def make_block() -> torch.Tensor:
    return torch.zeros(SHAPE)


def make_generator() -> torch.Generator:
    return torch.Generator().manual_seed(1234)


def test_normalized_residual_is_direct_difference_in_standardized_space():
    target_normalized = np.asarray([0.0, 1.0, -2.0, np.nan])
    prediction_normalized = np.asarray([1.0, 2.0, -1.0, 5.0])
    residual = normalized_residual(prediction_normalized, target_normalized)
    nrmse, nmae = compute_normalized_metrics(residual)

    np.testing.assert_allclose(residual[:3], [1.0, 1.0, 1.0])
    assert np.isnan(residual[3])
    assert np.isclose(nrmse, 1.0)
    assert np.isclose(nmae, 1.0)


def test_compute_jy_matches_full_derived_field_helper():
    field = np.arange(4 * 3 * 4 * 5, dtype=np.float64).reshape(4, 3, 4, 5)
    _ay, jy_with_ay = compute_ay_jy(field, [-2.0, 2.0, -3.0, 3.0])
    jy_only = compute_jy(field, [-2.0, 2.0, -3.0, 3.0])
    np.testing.assert_allclose(jy_only, jy_with_ay)


def test_validation_statistics_equal_weight_runs_after_combining_windows():
    profiles = {
        "run_a": [np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])],
        "run_b": [np.asarray([10.0, 20.0])],
    }
    stats = aggregate_run_window_profiles(profiles)

    np.testing.assert_allclose(stats["run_profiles"], [[2.0, 3.0], [10.0, 20.0]])
    np.testing.assert_allclose(stats["median"], [6.0, 11.5])
    assert stats["window_counts"] == {"run_a": 2, "run_b": 1}


class ZeroFieldModel(torch.nn.Module):
    def forward(self, model_input):
        return torch.zeros_like(model_input[:, :4])


class TinyWindowDataset:
    samples = [(0, "run_a", 0), (0, "run_a", 2)]

    def __getitem__(self, index):
        block = torch.zeros((4, 2, 4, 3), dtype=torch.float32)
        block[3] = float(index + 1)
        return {
            "block": block,
            "metadata": {"run_name": "run_a", "t0": 2 * index},
        }


def test_collect_validation_statistics_crops_and_combines_windows():
    template = torch.zeros((1, 4, 2, 4, 3), dtype=torch.float32)
    mask_rows = build_density_superres_rows(template, [1, 0])
    canonical_rows = {
        "density_superres": [
            {"name": name, "label": label, "mask": mask[0].numpy()}
            for name, label, mask in mask_rows
        ]
    }
    args = SimpleNamespace(
        seed=1234,
        experiment="density_superres",
        density_probe_counts=[1, 0],
        magnetic_visible_fractions=[1.0, 0.8, 0.6, 0.4],
        extent=[-2.0, 2.0, -3.0, 3.0],
    )
    stats = collect_validation_statistics(
        model=ZeroFieldModel(),
        dataset=TinyWindowDataset(),
        sample_indices=[0, 1],
        args=args,
        mean=torch.zeros((1, 4, 1, 1, 1)),
        std=torch.ones((1, 4, 1, 1, 1)),
        device=torch.device("cpu"),
        canonical_rows=canonical_rows,
    )

    first_row = stats["density_superres"][0]
    np.testing.assert_allclose(first_row["density"]["median"], [1.5, 1.5])
    assert first_row["density"]["window_counts"] == {"run_a": 2}


def test_multifunction_masks_only_density():
    rows = build_density_only_multifunction_rows(
        block=make_block(),
        patterns=[
            "spatial_random",
            "spatial_grid",
            "spatial_block",
            "temporal_random",
            "temporal_block",
        ],
        mask_fraction=0.8,
        block_fraction=0.5,
        grid_stride=4,
        magnetic_grid_stride=2,
        generator=make_generator(),
    )

    assert len(rows) == 5
    assert all(torch.all(mask[:, :3] == 1) for _, _, mask in rows)
    block_mask = next(mask for name, _, mask in rows if name == "spatial_block")
    midpoint = SHAPE[-2] // 2
    assert torch.all(block_mask[:, 3:4, :, :midpoint] == 1)
    assert torch.all(block_mask[:, 3:4, :, midpoint:] == 0)

    temporal_random = next(
        mask for name, _, mask in rows if name == "temporal_random"
    )
    assert torch.all(temporal_random[:, 3:4, 0::2] == 1)
    assert torch.all(temporal_random[:, 3:4, 1::2] == 0)

    temporal_block = next(
        mask for name, _, mask in rows if name == "temporal_block"
    )
    half = SHAPE[2] // 2
    assert torch.all(temporal_block[:, 3:4, :half] == 1)
    assert torch.all(temporal_block[:, 3:4, half:] == 0)


def test_hide_magnetic_keeps_density_geometry():
    patterns = [
        "spatial_random",
        "spatial_grid",
        "spatial_block",
        "temporal_random",
        "temporal_block",
    ]
    kwargs = dict(
        block=make_block(),
        patterns=patterns,
        mask_fraction=0.8,
        block_fraction=0.5,
        grid_stride=4,
        magnetic_grid_stride=2,
    )
    visible_rows = build_density_only_multifunction_rows(
        **kwargs, generator=make_generator(), magnetic_visible=True
    )
    hidden_rows = build_density_only_multifunction_rows(
        **kwargs, generator=make_generator(), magnetic_visible=False
    )

    assert all(torch.all(mask[:, :3] == 0) for _, _, mask in hidden_rows)
    for (_, visible_label, visible_mask), (_, hidden_label, hidden_mask) in zip(
        visible_rows, hidden_rows
    ):
        assert torch.equal(visible_mask[:, 3:4], hidden_mask[:, 3:4])
        assert "B visible=100%" in visible_label
        assert "B visible=0%" in hidden_label

    superres_hidden = build_density_superres_rows(
        make_block(), [0, 10], magnetic_visible=False
    )
    assert all(torch.all(mask[:, :3] == 0) for _, _, mask in superres_hidden)
    assert int(superres_hidden[1][2][0, 3, 0].sum()) == 10

    forecast_hidden = build_density_forecast_rows(
        torch.zeros(1, 4, 24, 20, 12), [12], magnetic_visible=False
    )
    _, forecast_label, forecast_mask = forecast_hidden[0]
    assert torch.all(forecast_mask[:, :3] == 0)
    assert torch.all(forecast_mask[:, 3:4, :12] == 1)
    assert torch.all(forecast_mask[:, 3:4, 12:] == 0)
    assert "B visible=0%" in forecast_label


def test_density_superres_uses_exact_requested_probe_counts():
    targets = [0, 10, 100, 1000]
    rows = build_density_superres_rows(make_block(), targets)

    for (_, _, mask), target in zip(rows, targets):
        assert torch.all(mask[:, :3] == 1)
        assert int(mask[0, 3, 0].sum()) == target


def test_magnetic_ablation_is_nested_and_hides_all_density():
    targets = list(DEFAULT_MAGNETIC_ABLATION_VISIBLE_FRACTIONS)
    rows = build_magnetic_ablation_rows(
        block=make_block(),
        magnetic_visible_fractions=targets,
        generator=make_generator(),
    )
    num_sites = SHAPE[-2] * SHAPE[-1]

    assert len(rows) == 8
    expected_labels = [
        "B visible=100%",
        "B visible=30%",
        "B visible=10%",
        "B visible=3%",
        "B visible=1%",
        "B visible=0.3%",
        "B visible=0.1%",
        "B visible=0%",
    ]
    for (name, label, mask), target, expected_label in zip(
        rows, targets, expected_labels
    ):
        visible_count = int(mask[0, 0, 0].sum())
        assert visible_count == magnetic_ablation_visible_count(num_sites, target)
        assert torch.all(mask[:, 0] == mask[:, 1])
        assert torch.all(mask[:, 1] == mask[:, 2])
        assert torch.all(mask[:, 3:4] == 0)
        assert expected_label in label
        assert validation_statistics_legend_label(
            "magnetic_ablation", {"name": name}, context_length=24
        ) == expected_label

    assert torch.all(rows[0][2][:, :3] == 1)
    assert torch.all(rows[-1][2][:, :3] == 0)
    assert magnetic_ablation_visible_count(num_sites, 0.0) == 0
    assert magnetic_ablation_visible_count(num_sites, 1.0) == num_sites

    for (_, _, higher), (_, _, lower) in zip(rows, rows[1:]):
        assert torch.all(lower[:, :3] <= higher[:, :3])


def test_magnetic_ablation_nrmse_panels_use_log_scale_when_positive(tmp_path):
    frames = np.arange(3)
    rows = []
    for fraction, offset in zip((1.0, 0.001, 0.0), (0.2, 0.5, 0.9)):
        pred = np.full((4, 3, 2, 2), offset, dtype=np.float64)
        pred_jy = np.full((3, 2, 2), offset + 0.1, dtype=np.float64)
        rows.append(
            {
                "name": f"magnetic_ablation_{fraction:g}",
                "label": f"Magnetic information ablation\n{format_magnetic_visible_percent(fraction)} (nested random)",
                "pred_normalized": pred,
                "pred_jy_normalized": pred_jy,
            }
        )
    target = np.zeros((4, 3, 2, 2), dtype=np.float64)
    target_jy = np.zeros((3, 2, 2), dtype=np.float64)
    payload = save_information_suite_error_plot(
        target_field_normalized=target,
        target_jy_normalized=target_jy,
        rows=rows,
        frame_ids=frames,
        out_path=tmp_path / "named_magnetic_ablation.png",
        title="magnetic_ablation: error by frame",
        experiment_name="magnetic_ablation",
    )
    assert payload["density_nrmse_yscale"] == "log"
    assert payload["jy_nrmse_yscale"] == "log"
    assert [row["legend_label"] for row in payload["rows"]] == [
        "B visible=100%",
        "B visible=0.1%",
        "B visible=0%",
    ]

    statistics_rows = []
    for fraction, value in zip((1.0, 0.0), (0.4, 0.8)):
        curve = np.full(3, value, dtype=np.float64)
        statistics_rows.append(
            {
                "name": f"magnetic_ablation_{fraction:g}",
                "label": format_magnetic_visible_percent(fraction),
                "density": {
                    "run_names": ["run_a"],
                    "window_counts": {"run_a": 1},
                    "run_profiles": curve[None, :],
                    "median": curve,
                    "p16": curve * 0.9,
                    "p84": curve * 1.1,
                },
                "jy": {
                    "run_names": ["run_a"],
                    "window_counts": {"run_a": 1},
                    "run_profiles": (curve + 0.05)[None, :],
                    "median": curve + 0.05,
                    "p16": (curve + 0.05) * 0.9,
                    "p84": (curve + 0.05) * 1.1,
                },
            }
        )
    stats_payload = save_validation_statistics_plot(
        experiment_name="magnetic_ablation",
        row_statistics=statistics_rows,
        context_length=3,
        total_windows=1,
        out_path=tmp_path / "validation_magnetic_ablation.png",
    )
    assert stats_payload["density_nrmse_yscale"] == "log"
    assert stats_payload["jy_nrmse_yscale"] == "log"
    assert [row["legend_label"] for row in stats_payload["rows"]] == [
        "B visible=100%",
        "B visible=0%",
    ]


def test_nrmse_log_scale_is_skipped_when_a_panel_contains_zero():
    class FakeAxis:
        def __init__(self):
            self.yscale = "linear"

        def set_yscale(self, value):
            self.yscale = value

    axis = FakeAxis()
    scale = apply_log_yscale_if_strictly_positive(
        axis,
        np.asarray([0.2, 0.0, 0.4]),
        "unit-test Density NRMSE",
    )
    assert scale == "linear"
    assert axis.yscale == "linear"

    axis = FakeAxis()
    scale = apply_log_yscale_if_strictly_positive(
        axis,
        np.asarray([0.2, 0.1, 0.4]),
        "unit-test Jy NRMSE",
    )
    assert scale == "log"
    assert axis.yscale == "log"


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


class FakeStatisticsDataset:
    samples = [
        (0, "run_a", 0),
        (0, "run_a", 24),
        (0, "run_a", 48),
        (0, "run_b", 0),
        (0, "train_run", 0),
    ]


def test_validation_statistics_selects_even_windows_from_each_validation_run():
    indices = select_validation_statistics_indices(
        dataset=FakeStatisticsDataset(),
        val_runs={"run_a", "run_b"},
        window_stride=24,
        max_windows_per_run=2,
    )
    assert indices == [0, 2, 3]


def test_validation_statistics_includes_run_end_aligned_window():
    dataset = SimpleNamespace(
        samples=[(0, "run_a", t0) for t0 in range(6)]
    )
    indices = select_validation_statistics_indices(
        dataset=dataset,
        val_runs={"run_a"},
        window_stride=4,
        max_windows_per_run=None,
    )
    assert indices == [0, 4, 5]


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


def test_gif_animation_loops_infinitely(tmp_path):
    frame_paths = []
    for index, color in enumerate(("red", "blue")):
        frame_path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (8, 8), color=color).save(frame_path)
        frame_paths.append(frame_path)

    gif_path = tmp_path / "looping.gif"
    write_animation(frame_paths, gif_path, fps=2.0)

    with Image.open(gif_path) as animation:
        assert animation.n_frames == 2
        assert animation.info.get("loop") == 0
