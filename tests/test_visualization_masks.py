from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.masking import _spatial_block_plane, sample_mask  # noqa: E402

from visualize_mask_patterns_unet3d import (  # noqa: E402
    DEFAULT_MAGNETIC_ABLATION_VISIBLE_FRACTIONS,
    JY_NRMSE_YLABEL,
    make_centered_spatial_block_mask,
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
    information_suite_plot_cache_signature,
    jy_nrmse_from_residual,
    jy_stats_cache_matches,
    normalize_field_np,
    normalized_residual,
    save_information_suite_error_plot,
    save_plot_cache,
    save_validation_statistics_plot,
    summarize_jy_values,
    select_validation_statistics_indices,
    select_run_t0_index,
    try_reuse_plot_cache,
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
    np.testing.assert_allclose(first_row["jy"]["median"], [0.0, 0.0])
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

    assert len(rows) == 6
    assert all(torch.all(mask[:, :3] == 1) for _, _, mask in rows)
    inpaint = next(
        mask for name, _, mask in rows if name == "spatial_block_inpainting"
    )
    outpaint = next(
        mask for name, _, mask in rows if name == "spatial_block_outpainting"
    )
    assert torch.equal(outpaint[:, 3:4], 1.0 - inpaint[:, 3:4])
    inpaint_label = next(
        label for name, label, _ in rows if name == "spatial_block_inpainting"
    )
    outpaint_label = next(
        label for name, label, _ in rows if name == "spatial_block_outpainting"
    )
    assert "Spatial block — inpainting" in inpaint_label
    assert "Spatial block — outpainting" in outpaint_label

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
    assert payload["jy_nrmse_ylabel"] == JY_NRMSE_YLABEL
    np.testing.assert_allclose(payload["rows"][0]["jy_nrmse"], [0.3, 0.3, 0.3])
    scaled_payload = save_information_suite_error_plot(
        target_field_normalized=target,
        target_jy_normalized=target_jy,
        rows=rows,
        frame_ids=frames,
        out_path=tmp_path / "named_magnetic_ablation_scaled.png",
        title="magnetic_ablation: error by frame",
        experiment_name="magnetic_ablation",
        jy_std_train=0.1,
    )
    np.testing.assert_allclose(scaled_payload["rows"][0]["jy_nrmse"], [3.0, 3.0, 3.0])
    np.testing.assert_allclose(scaled_payload["rows"][0]["density_nrmse"], [0.2, 0.2, 0.2])
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
    assert stats_payload["jy_nrmse_ylabel"] == JY_NRMSE_YLABEL
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


def test_jy_nrmse_divides_residual_rms_by_training_std():
    residual = np.full((3, 4, 5), 0.2, dtype=np.float64)
    np.testing.assert_allclose(jy_nrmse_from_residual(residual, 0.1), [2.0, 2.0, 2.0])
    nrmse, nmae = compute_normalized_metrics(residual, scale=0.1)
    assert np.isclose(nrmse, 2.0)
    assert np.isclose(nmae, 0.2)


def test_jy_training_summary_and_cache_keys_match_visualization_definition():
    field = np.zeros((4, 2, 6, 5), dtype=np.float64)
    z = np.linspace(-21.0, 21.0, 5)
    field[0] = z[None, None, :]
    mean = np.array([0.0, 0.0, 0.0, 0.0])
    std = np.array([2.0, 1.0, 1.0, 1.0])
    extent = [-21.0, 21.0, -50.0, 50.0]
    jy = compute_jy(normalize_field_np(field, mean, std), extent)
    summary = summarize_jy_values(jy)
    # Linear Bx / std=2 has constant dBx/dz = 0.5 after channel standardization.
    np.testing.assert_allclose(jy, 0.5, atol=1e-12)
    assert np.isclose(summary["jy_mean_train"], 0.5)
    assert np.isclose(summary["jy_std_train"], 0.0)
    assert np.isclose(summary["jy_rms_train"], 0.5)
    cached = {
        "train_runs": ["run_a"],
        "jy_definition": "dBx/dz - dBz/dx",
        "preprocessing": "checkpoint channel-standardized Bx,Bz",
        "extent": extent,
        "channel_mean": mean.tolist(),
        "channel_std": std.tolist(),
    }
    assert jy_stats_cache_matches(cached, {"run_a"}, mean, std, extent)
    assert not jy_stats_cache_matches(cached, {"run_b"}, mean, std, extent)


def test_collect_validation_statistics_scales_jy_nrmse_by_training_std():
    class LinearBxDataset:
        samples = [(0, "run_a", 0)]

        def __getitem__(self, index):
            block = torch.zeros((4, 2, 4, 5), dtype=torch.float32)
            z = torch.linspace(-21.0, 21.0, 5)
            block[0] = z.view(1, 1, 5)
            return {
                "block": block,
                "metadata": {"run_name": "run_a", "t0": 0},
            }

    template = torch.zeros((1, 4, 2, 4, 5), dtype=torch.float32)
    mask_rows = build_density_superres_rows(template, [0])
    canonical_rows = {
        "density_superres": [
            {"name": name, "label": label, "mask": mask[0].numpy()}
            for name, label, mask in mask_rows
        ]
    }
    args = SimpleNamespace(
        seed=1234,
        experiment="density_superres",
        density_probe_counts=[0],
        magnetic_visible_fractions=[0.0],
        extent=[-21.0, 21.0, -50.0, 50.0],
    )
    raw = collect_validation_statistics(
        model=ZeroFieldModel(),
        dataset=LinearBxDataset(),
        sample_indices=[0],
        args=args,
        mean=torch.zeros((1, 4, 1, 1, 1)),
        std=torch.ones((1, 4, 1, 1, 1)),
        device=torch.device("cpu"),
        canonical_rows=canonical_rows,
        jy_std_train=1.0,
    )
    scaled = collect_validation_statistics(
        model=ZeroFieldModel(),
        dataset=LinearBxDataset(),
        sample_indices=[0],
        args=args,
        mean=torch.zeros((1, 4, 1, 1, 1)),
        std=torch.ones((1, 4, 1, 1, 1)),
        device=torch.device("cpu"),
        canonical_rows=canonical_rows,
        jy_std_train=0.2,
    )
    raw_jy = raw["density_superres"][0]["jy"]["median"]
    scaled_jy = scaled["density_superres"][0]["jy"]["median"]
    assert np.all(raw_jy > 0.0)
    np.testing.assert_allclose(scaled_jy, raw_jy / 0.2)
    # Density is unchanged by the Jy scale.
    np.testing.assert_allclose(
        scaled["density_superres"][0]["density"]["median"],
        raw["density_superres"][0]["density"]["median"],
    )


def test_centered_spatial_block_is_complementary_and_half_area():
    size_x, size_z = 154, 62
    inpaint, in_info = make_centered_spatial_block_mask(
        size_x, size_z, orientation="inside_masked"
    )
    outpaint, out_info = make_centered_spatial_block_mask(
        size_x, size_z, orientation="inside_visible"
    )

    assert in_info["rect_height"] == 109
    assert in_info["rect_width"] == 44
    assert in_info["rect_x0"] == (size_x - 109) // 2
    assert in_info["rect_z0"] == (size_z - 44) // 2
    assert in_info["rect_x0"] == out_info["rect_x0"]
    assert in_info["rect_z0"] == out_info["rect_z0"]
    assert abs(in_info["rect_x0"] - (size_x - in_info["rect_x0"] - in_info["rect_height"])) <= 1
    assert abs(in_info["rect_z0"] - (size_z - in_info["rect_z0"] - in_info["rect_width"])) <= 1

    aspect = in_info["rect_height"] / in_info["rect_width"]
    domain_aspect = size_x / size_z
    assert abs(math.log(aspect / domain_aspect)) < 0.02
    assert abs(in_info["actual_area_fraction"] - 0.5) < 0.01
    assert in_info["target_area_fraction"] == 0.5

    x0, z0 = in_info["rect_x0"], in_info["rect_z0"]
    hx, hz = in_info["rect_height"], in_info["rect_width"]
    assert torch.all(inpaint[x0 : x0 + hx, z0 : z0 + hz] == 0)
    outside = inpaint.clone()
    outside[x0 : x0 + hx, z0 : z0 + hz] = float("nan")
    assert torch.all(outside[torch.isfinite(outside)] == 1)
    assert torch.all(outpaint[x0 : x0 + hx, z0 : z0 + hz] == 1)
    outside_out = outpaint.clone()
    outside_out[x0 : x0 + hx, z0 : z0 + hz] = float("nan")
    assert torch.all(outside_out[torch.isfinite(outside_out)] == 0)
    assert torch.equal(outpaint, 1.0 - inpaint)


def test_training_and_validation_spatial_block_are_not_the_viz_rectangle():
    size_x, size_z = 154, 62
    _, centered = make_centered_spatial_block_mask(size_x, size_z)
    origins = set()
    for seed in range(20):
        mask, info = sample_mask(
            (1, 4, 4, size_x, size_z),
            "spatial_block",
            0.5,
            generator=torch.Generator().manual_seed(seed),
        )
        assert info["orientation"] == "inside_masked"
        origins.add((info["rect_x0"], info["rect_z0"], info["rect_height"], info["rect_width"]))
        assert torch.all(mask[:, 0] == mask[:, 3])
    assert len(origins) > 1
    assert any(
        origin
        != (
            centered["rect_x0"],
            centered["rect_z0"],
            centered["rect_height"],
            centered["rect_width"],
        )
        for origin in origins
    )

    orientations = {
        _spatial_block_plane(
            1, 1, 32, 24, 0.35, torch.Generator().manual_seed(seed), orientation="random"
        )[1]["orientation"]
        for seed in range(40)
    }
    assert orientations == {"inside_masked", "inside_visible"}


def _information_suite_cache_args(run_dir: Path, **overrides) -> SimpleNamespace:
    checkpoint = run_dir / "latest.pt"
    if not checkpoint.exists():
        checkpoint.write_bytes(b"ckpt")
    fields = {
        "checkpoint": "latest.pt",
        "run_name": "beta0.2_nu2_Bz0_dt2_tau70",
        "t0": 28,
        "sample_index": None,
        "seed": 1234,
        "mask_fraction": 0.8,
        "block_fraction": 0.5,
        "grid_stride": 4,
        "magnetic_grid_stride": 2,
        "mask_patterns": ["spatial_random"],
        "experiment": "all",
        "hide_magnetic": False,
        "density_probe_counts": [0, 10, 100, 1000],
        "magnetic_visible_fractions": [1.0, 0.0],
        "density_forecast_visible_frames": None,
        "skip_validation_statistics": False,
        "statistics_window_stride": None,
        "statistics_max_windows_per_run": None,
        "plot_units": "physical",
        "extent": [-21.0, 21.0, -50.0, 50.0],
        "h5_dir": None,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_information_suite_plot_cache_signature_tracks_inference_settings(tmp_path):
    args = _information_suite_cache_args(tmp_path)
    signature = information_suite_plot_cache_signature(args, tmp_path)
    again = information_suite_plot_cache_signature(
        _information_suite_cache_args(tmp_path), tmp_path
    )
    assert signature == again
    hidden = information_suite_plot_cache_signature(
        _information_suite_cache_args(tmp_path, hide_magnetic=True),
        tmp_path,
    )
    assert hidden != signature


def test_plot_cache_roundtrip_and_signature_mismatch(tmp_path):
    args = _information_suite_cache_args(tmp_path)
    signature = information_suite_plot_cache_signature(args, tmp_path)
    cache_path = tmp_path / "plot_cache.pkl"
    save_plot_cache(cache_path, {"signature": signature, "delta_t": 24})

    reused = try_reuse_plot_cache(cache_path, signature, enabled=True)
    assert reused is not None
    assert reused["delta_t"] == 24
    assert try_reuse_plot_cache(cache_path, signature, enabled=False) is None
    mismatched = dict(signature)
    mismatched["t0"] = 0
    assert try_reuse_plot_cache(cache_path, mismatched, enabled=True) is None

