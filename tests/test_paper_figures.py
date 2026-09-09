from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_paper_figures import (  # noqa: E402
    COMPATIBLE_PAPER_CACHE_VERSIONS,
    DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS,
    DEFAULT_SLIDING_DENSITY_PROBE_COUNT,
    DEFAULT_SLIDE_STEPS,
    EXPECTED_FORECAST_ROW_NAMES,
    EXPECTED_MAGNETIC_ROW_NAMES,
    EXPECTED_SUPERRES_ROW_NAMES,
    FIGURE_STEMS,
    MAGNETIC_DENSITY_CONDITIONS,
    JY_INCLUDES_MU0,
    JY_PAPER_PANEL_TITLE,
    JY_STATS_CACHE_VERSION,
    JY_STATS_PREPROCESSING,
    PAPER_CACHE_VERSION,
    SUPERRES_B_CONDITION_STYLES,
    SLIDING_AGGREGATE_N_FRAMES,
    SPATIAL_ROW_ORDER,
    _val_json_path,
    _magnetic_ablation_masks_with_density_full,
    _magnetic_ablation_rows_from_payload,
    base_signature,
    build_magnetic_ablation_cache,
    build_paired_b_condition_cache,
    clip_positive_for_log,
    density_nrmse_from_residual,
    density_visible_ratio,
    format_density_nrmse_label,
    compute_sliding_qualitative_density_nrmse,
    ensure_sliding_qualitative_nrmse_metadata,
    load_figure_cache,
    load_or_build_figure_cache,
    local_index_for_global_frame,
    manifest_figure_entry,
    paper_signatures_match,
    plot_density_forecast_summary,
    plot_density_superres_summary,
    plot_magnetic_ablation_summary,
    plot_sliding_window_appendix,
    plot_spatial_qualitative,
    select_sliding_aggregate_runs,
    sliding_density_probe_mask,
    sliding_provenance_error,
    stack_rmse_by_global_frame,
    try_load_validation_json,
    validation_run_order,
)


def _touch_checkpoint(path: Path, payload: bytes = b"ckpt") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _write_validation_json(
    path: Path,
    checkpoint_path: Path,
    names: list[str],
    n_runs: int = 3,
    n_frames: int = 4,
    checkpoint_epoch: int = 4500,
    experiment: str | None = None,
    hide_magnetic: bool | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        per_run = {}
        for run_index in range(n_runs):
            per_run[f"run{run_index}"] = {
                "density_nrmse": [0.1 * (run_index + 1)] * n_frames,
                "jy_nrmse": [0.2 * (run_index + 1)] * n_frames,
            }
        rows.append({"name": name, "per_run": per_run})
    payload = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint_epoch,
        "rows": rows,
        "local_frames": list(range(n_frames)),
        "context_length": n_frames,
        "run_count": n_runs,
        "jy_preprocessing": JY_STATS_PREPROCESSING,
        "jy_includes_mu0": JY_INCLUDES_MU0,
    }
    if experiment is not None:
        payload["experiment"] = experiment
    if hide_magnetic is not None:
        payload["hide_magnetic"] = hide_magnetic
    path.write_text(json.dumps(payload))


def test_paper_cache_write_read_and_redraw_skips_builder(tmp_path):
    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return {"v": np.arange(3, dtype=np.float32)}, {"note": "built"}

    signature = {"cache_version": 1, "checkpoint": {"size": 4, "mtime_ns": 1}}
    arrays, metadata = load_or_build_figure_cache(
        tmp_path, "demo", signature, builder, force=False
    )
    np.testing.assert_array_equal(arrays["v"], np.arange(3))
    assert metadata["note"] == "built"
    assert metadata["signature"] == signature
    assert calls["n"] == 1

    loaded = load_figure_cache(tmp_path, "demo")
    assert loaded is not None
    arrays2, metadata2 = load_or_build_figure_cache(
        tmp_path, "demo", signature, builder, force=False
    )
    assert calls["n"] == 1
    np.testing.assert_array_equal(arrays2["v"], arrays["v"])
    assert metadata2["signature"] == signature


def test_cache_signature_invalidates_when_checkpoint_changes(tmp_path):
    checkpoint = _touch_checkpoint(tmp_path / "latest.pt", b"abc")
    args = SimpleNamespace(
        run_name="beta0.2_nu2_Bz0_dt2_tau70",
        t0=28,
        global_frame=45,
        seed=1234,
        extent=[-21.0, 21.0, -50.0, 50.0],
    )
    signature = base_signature(args, tmp_path, checkpoint)
    assert "dpi" not in signature
    assert "residual_vmax" not in signature
    assert signature["checkpoint"]["size"] == 3

    calls = {"n": 0}

    def builder():
        calls["n"] += 1
        return {"v": np.asarray([calls["n"]], dtype=np.float32)}, {"built": calls["n"]}

    load_or_build_figure_cache(tmp_path, "sig", signature, builder)
    load_or_build_figure_cache(tmp_path, "sig", signature, builder)
    assert calls["n"] == 1

    checkpoint.write_bytes(b"abcd")
    changed = base_signature(args, tmp_path, checkpoint)
    assert changed != signature
    load_or_build_figure_cache(tmp_path, "sig", changed, builder)
    assert calls["n"] == 2


def test_spatial_qualitative_figure_is_4x4(tmp_path):
    rng = np.random.default_rng(0)
    target = rng.random((8, 6)).astype(np.float32)
    arrays = {"target_density": target}
    for name in SPATIAL_ROW_ORDER:
        arrays[f"{name}_visible"] = target.copy()
        arrays[f"{name}_prediction"] = target + 0.01
        arrays[f"{name}_residual"] = rng.normal(0.0, 0.1, target.shape).astype(np.float32)
        arrays[f"{name}_mask"] = np.ones_like(target)
    metadata = {
        "row_names": list(SPATIAL_ROW_ORDER),
        "global_frame": 45,
    }
    n_rows, n_cols = plot_spatial_qualitative(
        arrays,
        metadata,
        tmp_path,
        extent=[-1.0, 1.0, -1.0, 1.0],
        residual_vmax=1.0,
        field_q=99.0,
        dpi=80,
    )
    assert (n_rows, n_cols) == (4, 4)
    assert len(SPATIAL_ROW_ORDER) == 4
    assert (tmp_path / f"{FIGURE_STEMS['spatial']}.png").exists()
    assert (tmp_path / f"{FIGURE_STEMS['spatial']}.pdf").exists()


def test_superres_visible_ratios_use_actual_xz():
    assert density_visible_ratio(0, 154, 62) == 0.0
    assert density_visible_ratio(10, 10, 10) == 0.1
    ratio = density_visible_ratio(10, 154, 62)
    assert abs(ratio - 10 / (154 * 62)) < 1e-15
    other = density_visible_ratio(10, 200, 80)
    assert other != ratio
    assert abs(other - 10 / (200 * 80)) < 1e-15


def test_forecast_cache_contains_both_b_conditions(tmp_path):
    checkpoint = _touch_checkpoint(tmp_path / "latest.pt")
    names = [
        "density_forecast_history_23",
        "density_forecast_history_18",
        "density_forecast_history_12",
        "density_forecast_history_6",
    ]
    _write_validation_json(
        tmp_path
        / "figures_information_suite_plasmoid_merger"
        / "validation-runs_stride-24_experiment-density_forecast_error_vs_local_frame.json",
        checkpoint,
        names,
    )
    _write_validation_json(
        tmp_path
        / "figures_information_suite_plasmoid_merger_no_magnetic"
        / "validation-runs_stride-24_experiment-density_forecast_error_vs_local_frame.json",
        checkpoint,
        names,
    )
    args = Namespace(force_recompute=False)
    arrays, meta = build_paired_b_condition_cache(
        args, tmp_path, checkpoint, None, "forecast", checkpoint_epoch=4500
    )
    assert meta["b_conditions"] == ["B_full", "B_hidden"]
    assert set(meta["rows"]) == {"B_full", "B_hidden"}
    assert [row["history"] for row in meta["rows"]["B_full"]] == [23, 18, 12, 6]
    assert [row["history"] for row in meta["rows"]["B_hidden"]] == [23, 18, 12, 6]
    result = plot_density_forecast_summary(arrays, meta, tmp_path, dpi=80)
    assert result["layout"] == (1, 2)
    assert result["titles"] == ["Density NRMSE", JY_PAPER_PANEL_TITLE]
    assert result["sharey"] is True
    assert result["ylims"][0] == result["ylims"][1]
    assert (tmp_path / f"{FIGURE_STEMS['forecast']}.png").exists()


def test_magnetic_ablation_contains_all_eight_levels(tmp_path):
    checkpoint = _touch_checkpoint(tmp_path / "latest.pt")
    names = [
        f"magnetic_ablation_{percent / 100.0:g}"
        for percent in DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS
    ]
    _write_validation_json(
        tmp_path
        / "figures_information_suite_plasmoid_merger"
        / "validation-runs_stride-24_experiment-magnetic_ablation_error_vs_local_frame.json",
        checkpoint,
        names,
    )
    args = Namespace(force_recompute=False)
    payload = try_load_validation_json(
        tmp_path, checkpoint, "magnetic_ablation", checkpoint_epoch=4500
    )
    assert payload is not None
    rows = _magnetic_ablation_rows_from_payload(payload)
    percents = [100.0 * float(row["visible_fraction"]) for row in rows]
    assert len(rows) == 8
    assert percents == sorted(percents)
    np.testing.assert_allclose(sorted(percents), sorted(DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS))
    try:
        build_magnetic_ablation_cache(
            args, tmp_path, checkpoint, None, checkpoint_epoch=4500
        )
        assert False, "density-full magnetic ablation should need inference context"
    except RuntimeError as exc:
        assert "paired density conditions" in str(exc)


def test_magnetic_ablation_density_conditions_share_nested_b_layout():
    import torch
    from visualize_mask_patterns_unet3d import build_magnetic_ablation_rows

    block = torch.zeros(1, 4, 2, 8, 6)
    generator = torch.Generator().manual_seed(0)
    hidden_rows = build_magnetic_ablation_rows(
        block, [0.0, 0.1, 0.5, 1.0], generator
    )
    full_rows = _magnetic_ablation_masks_with_density_full(hidden_rows)
    assert len(hidden_rows) == len(full_rows) == 4
    previous_b = None
    for (_h_name, _h_label, hidden), (_f_name, _f_label, full) in zip(
        hidden_rows, full_rows
    ):
        assert torch.equal(hidden[:, :3], full[:, :3])
        assert torch.all(hidden[:, 3] == 0)
        assert torch.all(full[:, 3] == 1)
        b_plane = hidden[0, 0, 0]
        if previous_b is not None:
            assert torch.all(b_plane[previous_b > 0.5] > 0.5)
        previous_b = b_plane


def test_magnetic_ablation_plot_is_1x2_with_both_density_conditions(tmp_path):
    percents = np.asarray(DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS, dtype=np.float64)
    n = len(percents)
    arrays = {"b_visible_percent": percents}
    for prefix, scale in (("density_hidden", 0.2), ("density_full", 0.05)):
        median = np.linspace(scale, scale / 2.0, n)
        arrays[f"{prefix}_density_median"] = median
        arrays[f"{prefix}_density_p16"] = median * 0.8
        arrays[f"{prefix}_density_p84"] = median * 1.2
        arrays[f"{prefix}_jy_median"] = median * 1.4
        arrays[f"{prefix}_jy_p16"] = median * 1.1
        arrays[f"{prefix}_jy_p84"] = median * 1.7
    result = plot_magnetic_ablation_summary(arrays, {}, tmp_path, dpi=80)
    assert result["layout"] == (1, 2)
    assert result["titles"] == ["Density NRMSE", JY_PAPER_PANEL_TITLE]
    assert result["sharey"] is True
    assert result["ylims"][0] == result["ylims"][1]
    assert result["density_conditions"] == [
        "Density fully hidden",
        "Density visible 100%",
    ]
    assert [label for _prefix, label in MAGNETIC_DENSITY_CONDITIONS] == result[
        "density_conditions"
    ]
    assert (tmp_path / f"{FIGURE_STEMS['magnetic_ablation']}.png").exists()


def test_superres_plot_uses_probe_over_xz(tmp_path):
    checkpoint = _touch_checkpoint(tmp_path / "latest.pt")
    names = [f"density_superres_{count}" for count in (0, 10, 100, 1000)]
    _write_validation_json(
        tmp_path
        / "figures_information_suite_plasmoid_merger"
        / "validation-runs_stride-24_experiment-density_superres_error_vs_local_frame.json",
        checkpoint,
        names,
    )
    _write_validation_json(
        tmp_path
        / "figures_information_suite_plasmoid_merger_no_magnetic"
        / "validation-runs_stride-24_experiment-density_superres_error_vs_local_frame.json",
        checkpoint,
        names,
    )
    args = Namespace(force_recompute=False, density_probe_counts=[0, 10, 100, 1000])
    sliding_dir = tmp_path / "figures_sliding_density_reconstruction_plasmoid_merger_no_magnetic"
    sliding_dir.mkdir()
    np.savez(
        sliding_dir / "dummy_steps-8-4-2-1_density-visible-0.08_B-hidden_physical_final_reconstructions.npz",
        target_density=np.zeros((2, 12, 7), dtype=np.float32),
    )
    (sliding_dir / "dummy_steps-8-4-2-1_density-visible-0.08_B-hidden_physical_metrics.json").write_text(
        json.dumps({"slide_steps": [8, 4, 2, 1]})
    )
    arrays, meta = build_paired_b_condition_cache(
        args, tmp_path, checkpoint, None, "superres", checkpoint_epoch=4500
    )
    assert meta["b_conditions"] == ["B_full", "B_hidden"]
    size_x, size_z = 154, 50
    result = plot_density_superres_summary(
        arrays, meta, tmp_path, dpi=80, size_x=size_x, size_z=size_z
    )
    expected = [100.0 * density_visible_ratio(count, size_x, size_z) for count in (0, 10, 100, 1000)]
    np.testing.assert_allclose(meta["visible_ratio_percent"], expected)
    assert size_x * size_z != 154 * 62
    assert result["layout"] == (1, 2)
    assert result["titles"] == ["Density NRMSE", JY_PAPER_PANEL_TITLE]
    assert result["sharey"] is True
    assert result["ylims"][0] == result["ylims"][1]
    assert result["b_condition_styles"] == {
        "B_full": {
            "color": "C0",
            "linestyle": "-",
            "marker": "o",
            "label": "B visible 100%",
        },
        "B_hidden": {
            "color": "C1",
            "linestyle": "--",
            "marker": "^",
            "label": "B visible 0%",
        },
    }
    assert SUPERRES_B_CONDITION_STYLES["B_full"]["color"] == "C0"
    assert SUPERRES_B_CONDITION_STYLES["B_hidden"]["color"] == "C1"
    assert (tmp_path / f"{FIGURE_STEMS['superres']}.png").exists()


def test_global_frame_maps_through_t0():
    assert local_index_for_global_frame(28, 45, 24) == 17
    try:
        local_index_for_global_frame(28, 10, 24)
        assert False, "expected out-of-window global frame to fail"
    except ValueError:
        pass


def test_validation_json_uses_density_experiment_names(tmp_path):
    checkpoint = _touch_checkpoint(tmp_path / "latest.pt")
    forecast_path = _val_json_path(tmp_path, hide_magnetic=False, experiment="forecast")
    assert forecast_path.name.endswith("experiment-density_forecast_error_vs_local_frame.json")
    superres_path = _val_json_path(tmp_path, hide_magnetic=True, experiment="superres")
    assert "no_magnetic" in str(superres_path)
    assert superres_path.name.endswith("experiment-density_superres_error_vs_local_frame.json")
    _write_validation_json(
        forecast_path,
        checkpoint,
        list(EXPECTED_FORECAST_ROW_NAMES),
        experiment="density_forecast",
    )
    loaded = try_load_validation_json(
        tmp_path,
        checkpoint,
        "forecast",
        hide_magnetic=False,
        checkpoint_epoch=4500,
    )
    assert loaded is not None
    assert loaded["rows"][0]["name"] == "density_forecast_history_23"


def test_validation_json_rejects_wrong_epoch(tmp_path):
    checkpoint = _touch_checkpoint(tmp_path / "latest.pt")
    path = _val_json_path(tmp_path, hide_magnetic=False, experiment="forecast")
    _write_validation_json(
        path,
        checkpoint,
        list(EXPECTED_FORECAST_ROW_NAMES),
        checkpoint_epoch=3972,
        experiment="density_forecast",
    )
    assert (
        try_load_validation_json(
            tmp_path, checkpoint, "forecast", checkpoint_epoch=4500
        )
        is None
    )
    missing_epoch = json.loads(path.read_text())
    missing_epoch.pop("checkpoint_epoch")
    path.write_text(json.dumps(missing_epoch))
    assert (
        try_load_validation_json(
            tmp_path, checkpoint, "forecast", checkpoint_epoch=4500
        )
        is None
    )


def test_validation_json_rejects_wrong_forecast_histories(tmp_path):
    checkpoint = _touch_checkpoint(tmp_path / "latest.pt")
    path = _val_json_path(tmp_path, hide_magnetic=False, experiment="forecast")
    _write_validation_json(
        path,
        checkpoint,
        [
            "density_forecast_history_22",
            "density_forecast_history_18",
            "density_forecast_history_12",
            "density_forecast_history_6",
        ],
        experiment="density_forecast",
    )
    assert (
        try_load_validation_json(
            tmp_path, checkpoint, "forecast", checkpoint_epoch=4500
        )
        is None
    )


def test_validation_json_rejects_wrong_magnetic_levels(tmp_path):
    checkpoint = _touch_checkpoint(tmp_path / "latest.pt")
    path = _val_json_path(tmp_path, hide_magnetic=False, experiment="magnetic_ablation")
    names = list(EXPECTED_MAGNETIC_ROW_NAMES[:-1])
    _write_validation_json(path, checkpoint, names, experiment="magnetic_ablation")
    assert (
        try_load_validation_json(
            tmp_path, checkpoint, "magnetic_ablation", checkpoint_epoch=4500
        )
        is None
    )


def test_validation_json_rejects_wrong_superres_probes(tmp_path):
    checkpoint = _touch_checkpoint(tmp_path / "latest.pt")
    path = _val_json_path(tmp_path, hide_magnetic=False, experiment="superres")
    _write_validation_json(
        path,
        checkpoint,
        ["density_superres_0", "density_superres_10", "density_superres_100", "density_superres_500"],
        experiment="density_superres",
    )
    assert (
        try_load_validation_json(
            tmp_path, checkpoint, "superres", checkpoint_epoch=4500
        )
        is None
    )


def test_sliding_rejects_wrong_checkpoint(tmp_path):
    current = _touch_checkpoint(tmp_path / "latest.pt")
    other = tmp_path / "best.pt"
    other.write_bytes(b"best")
    assert (
        sliding_provenance_error(
            {"checkpoint": str(other), "checkpoint_epoch": 4500},
            current,
            4500,
        )
        is not None
    )
    assert (
        sliding_provenance_error(
            {"checkpoint": str(current), "checkpoint_epoch": 3972},
            current,
            4500,
        )
        is not None
    )
    assert (
        sliding_provenance_error(
            {"checkpoint": str(current), "checkpoint_epoch": 4500},
            current,
            4500,
        )
        is None
    )


def test_manifest_records_per_figure_checkpoint_path_and_epoch(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    metadata = {
        "source_checkpoint": str(tmp_path / "latest.pt"),
        "source_checkpoint_epoch": 4500,
        "source_files": [str(tmp_path / "validation.json")],
    }
    entry = manifest_figure_entry("forecast", FIGURE_STEMS["forecast"], data_dir, metadata)
    assert entry["figure"] == "forecast"
    assert entry["source_checkpoint"].endswith("latest.pt")
    assert entry["source_checkpoint_epoch"] == 4500
    assert entry["source_data"].endswith("density_forecast_summary.json")
    assert entry["source_files"] == [str(tmp_path / "validation.json")]
    manifest = {"figure_provenance": [entry]}
    assert all(
        item["source_checkpoint"] and item["source_checkpoint_epoch"] == 4500
        for item in manifest["figure_provenance"]
    )


def test_spatial_prediction_and_residual_are_full_field(tmp_path):
    target = np.ones((6, 5), dtype=np.float32) * 2.0
    prediction = target + 0.4
    residual = np.full_like(target, 0.25)
    visible = target.copy()
    visible[:, 3:] = np.nan
    arrays = {"target_density": target}
    for name in SPATIAL_ROW_ORDER:
        arrays[f"{name}_visible"] = visible
        arrays[f"{name}_prediction"] = prediction
        arrays[f"{name}_residual"] = residual
        arrays[f"{name}_mask"] = np.isfinite(visible).astype(np.float32)
    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(residual))
    assert abs(density_nrmse_from_residual(residual) - 0.25) < 1e-6
    assert format_density_nrmse_label(0.25) == "NRMSE = 0.25"
    n_rows, n_cols = plot_spatial_qualitative(
        arrays,
        {
            "row_names": list(SPATIAL_ROW_ORDER),
            "global_frame": 45,
            "rows": [
                {"name": name, "density_nrmse": 0.25} for name in SPATIAL_ROW_ORDER
            ],
        },
        tmp_path,
        extent=[-1.0, 1.0, -1.0, 1.0],
        residual_vmax=1.0,
        field_q=99.0,
        dpi=80,
    )
    assert (n_rows, n_cols) == (4, 4)


def test_clip_positive_for_log_avoids_nonpositive():
    clipped = clip_positive_for_log(np.array([0.0, -1.0, 0.5]))
    assert np.all(clipped > 0)
    assert clipped[-1] == 0.5


def test_select_sliding_aggregate_runs_uses_split_order():
    assert select_sliding_aggregate_runs(
        ["short", "long_a", "canonical", "long_b"], max_runs=2
    ) == ["short", "long_a"]
    assert select_sliding_aggregate_runs(["b", "a", "c"], max_runs=2) == ["b", "a"]


def test_sliding_first_48_frames_skips_short_and_drops_later_frames():
    def first_48_or_skip(n_frames: int) -> tuple[int, int] | None:
        if n_frames < SLIDING_AGGREGATE_N_FRAMES:
            return None
        return (0, SLIDING_AGGREGATE_N_FRAMES)

    assert SLIDING_AGGREGATE_N_FRAMES == 48
    assert first_48_or_skip(47) is None
    assert first_48_or_skip(48) == (0, 48)
    assert first_48_or_skip(52) == (0, 48)
    assert first_48_or_skip(104) == (0, 48)
    assert first_48_or_skip(130) == (0, 48)


def test_sliding_default_steps_are_1_12_24():
    assert tuple(DEFAULT_SLIDE_STEPS) == (1, 12, 24)


def test_validation_run_order_preserves_split_json(tmp_path):
    (tmp_path / "split.json").write_text(
        json.dumps({"val_runs": ["run_b", "run_a"], "train_runs": []})
    )
    assert validation_run_order(tmp_path) == ["run_b", "run_a"]


def test_stack_rmse_aligns_global_frames():
    frames, stacked = stack_rmse_by_global_frame(
        [np.array([0, 1]), np.array([1, 2])],
        [np.array([10.0, 20.0]), np.array([30.0, 40.0])],
    )
    np.testing.assert_array_equal(frames, [0, 1, 2])
    assert np.isnan(stacked[0, 2])
    assert stacked[1, 1] == 30.0


def test_sliding_density_probes_match_superres_1000_count_grid():
    import torch
    from visualize_mask_patterns_unet3d import (  # noqa: E402
        _density_probe_count_grid,
        _density_probe_grid,
    )

    size_x, size_z = 154, 62
    block = torch.zeros(1, 4, 3, size_x, size_z)
    assert DEFAULT_SLIDING_DENSITY_PROBE_COUNT == 1000
    mask, info = sliding_density_probe_mask(block, DEFAULT_SLIDING_DENSITY_PROBE_COUNT)
    superres_mask, superres_info = _density_probe_count_grid(block, 1000)
    assert int(mask.sum().item()) == 1000
    assert torch.equal(mask, superres_mask[0, 0, 0])
    assert info == superres_info == {"count_x": 50, "count_z": 20}
    assert mask.shape == (size_x, size_z)
    assert torch.equal(superres_mask[0, 0, 0], superres_mask[0, 0, 1])
    ratio = density_visible_ratio(1000, size_x, size_z)
    assert ratio == 1000 / (154 * 62)
    assert abs(100.0 * ratio - 10.47) < 0.005
    fraction_mask, _info = _density_probe_grid(
        block, 0.08, torch.Generator().manual_seed(1234)
    )
    assert int(fraction_mask[0, 0, 0].sum().item()) != 1000


def test_sliding_signature_change_invalidates_old_single_run_cache():
    old = {
        "cache_version": 2,
        "figure": "sliding",
        "slide_steps": [8, 4, 2, 1],
    }
    new = {
        **old,
        "cache_version": PAPER_CACHE_VERSION,
        "left_panel": "multi_run_framewise_rmse",
        "right_panel": "single_run_qualitative",
        "statistical_unit": "run",
        "sliding_max_runs": 16,
        "b_conditions": ["B_full", "B_hidden"],
        "frame_range": [0, 48],
        "slide_steps": [1, 12, 24],
    }
    assert not paper_signatures_match(old, new)


def test_sliding_probe_count_signature_invalidates_old_fraction_cache():
    old = {
        "cache_version": 6,
        "figure": "sliding",
        "slide_steps": [8, 4, 2, 1],
        "density_visible_fraction": 0.08,
        "left_panel": "multi_run_framewise_rmse",
        "right_panel": "single_run_qualitative",
        "statistical_unit": "run",
        "sliding_max_runs": 16,
        "b_conditions": ["B_full", "B_hidden"],
        "frame_range": [0, 52],
    }
    new = {
        **{key: value for key, value in old.items() if key != "density_visible_fraction"},
        "cache_version": PAPER_CACHE_VERSION,
        "density_probe_count": 1000,
        "probe_layout_semantics": "same as density_superres",
        "slide_steps": [1, 12, 24],
        "frame_range": [0, 48],
        "frame_selection": "first 48 frames",
    }
    assert not paper_signatures_match(old, new)


def test_validation_json_rejects_stale_standardized_b_jy(tmp_path):
    checkpoint = _touch_checkpoint(tmp_path / "latest.pt")
    path = _val_json_path(tmp_path, hide_magnetic=False, experiment="forecast")
    _write_validation_json(
        path,
        checkpoint,
        list(EXPECTED_FORECAST_ROW_NAMES),
        experiment="density_forecast",
    )
    payload = json.loads(path.read_text())
    payload["jy_preprocessing"] = "checkpoint channel-standardized Bx,Bz"
    path.write_text(json.dumps(payload))
    assert (
        try_load_validation_json(
            tmp_path,
            checkpoint,
            "forecast",
            hide_magnetic=False,
            checkpoint_epoch=4500,
        )
        is None
    )


def test_compatible_cache_versions_reuse_unchanged_signatures():
    assert PAPER_CACHE_VERSION == 13
    assert 12 in COMPATIBLE_PAPER_CACHE_VERSIONS
    for figure in ("spatial", "sliding"):
        stored = {"cache_version": 12, "figure": figure, "payload": 1}
        current = {"cache_version": 13, "figure": figure, "payload": 1}
        assert paper_signatures_match(stored, current)
    old_forecast = {
        "cache_version": 12,
        "figure": "forecast",
        "histories": [23],
        "b_conditions": ["B_full", "B_hidden"],
        "jy_includes_mu0": False,
    }
    new_forecast = {
        **old_forecast,
        "cache_version": 13,
        "jy_preprocessing": JY_STATS_PREPROCESSING,
        "jy_stats_cache_version": JY_STATS_CACHE_VERSION,
        "jy_includes_mu0": JY_INCLUDES_MU0,
        "B_unit": "T",
        "jy_unit": "A/m^2",
    }
    assert not paper_signatures_match(old_forecast, new_forecast)
    old_mag = {
        "cache_version": 12,
        "figure": "magnetic_ablation",
        "density_conditions": ["fully_hidden", "visible_100"],
        "shared_b_probe_layout": True,
        "layout": "1x2_density_jy",
        "b_visible_levels": list(DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS),
        "jy_includes_mu0": False,
    }
    new_mag = {
        **old_mag,
        "cache_version": 13,
        "jy_preprocessing": JY_STATS_PREPROCESSING,
        "jy_stats_cache_version": JY_STATS_CACHE_VERSION,
        "jy_includes_mu0": JY_INCLUDES_MU0,
        "B_unit": "T",
        "jy_unit": "A/m^2",
    }
    assert not paper_signatures_match(old_mag, new_mag)
    old_superres = {
        "cache_version": 12,
        "figure": "superres",
        "probe_counts": [0, 10, 100, 1000],
        "b_conditions": ["B_full", "B_hidden"],
        "line_style": "two_color_b_conditions",
        "jy_includes_mu0": False,
    }
    new_superres = {
        **old_superres,
        "cache_version": 13,
        "jy_preprocessing": JY_STATS_PREPROCESSING,
        "jy_stats_cache_version": JY_STATS_CACHE_VERSION,
        "jy_includes_mu0": JY_INCLUDES_MU0,
        "B_unit": "T",
        "jy_unit": "A/m^2",
    }
    assert not paper_signatures_match(old_superres, new_superres)


def test_sliding_plot_uses_framewise_rmse_and_both_b_conditions(tmp_path):
    frames = np.arange(48)
    arrays = {
        "frame_ids": frames,
        "aggregate_frame_ids": frames,
        "target_tz": np.zeros((48, 6), dtype=np.float32),
    }
    for step in (1, 12, 24):
        arrays[f"step_{step}_tz"] = np.zeros((48, 6), dtype=np.float32)
        arrays[f"step_{step}_tz_residual"] = np.zeros((48, 6), dtype=np.float32)
        median_full = np.linspace(0.1, 0.2, 48)
        median_hidden = np.linspace(0.12, 0.22, 48)
        arrays[f"B_full_step_{step}_rmse_median"] = median_full
        arrays[f"B_full_step_{step}_rmse_p16"] = median_full * 0.8
        arrays[f"B_full_step_{step}_rmse_p84"] = median_full * 1.2
        arrays[f"B_hidden_step_{step}_rmse_median"] = median_hidden
        arrays[f"B_hidden_step_{step}_rmse_p16"] = median_hidden * 0.8
        arrays[f"B_hidden_step_{step}_rmse_p84"] = median_hidden * 1.2
    result = plot_sliding_window_appendix(
        arrays,
        {
            "slide_steps": [1, 12, 24],
            "representative_step": 1,
            "aggregate_n_runs": 16,
            "n_frames": 48,
            "density_probe_count": 1000,
            "density_visible_ratio_percent": 100.0 * density_visible_ratio(1000, 154, 62),
            "frame_range": [0, 48],
            "b_conditions": ["B_full", "B_hidden"],
            "statistical_unit": "run",
        },
        tmp_path,
        extent=[-21.0, 21.0, -50.0, 50.0],
        dpi=80,
    )
    titles = result["panel_titles"]
    assert "Target" in titles
    assert "RMSE vs global frame" in titles
    for step in (1, 12, 24):
        assert f"Prediction (step={step})" in titles
        assert f"Residual (step={step})" in titles
    assert result["slide_steps"] == [1, 12, 24]
    expected_b100 = float(np.mean(np.linspace(0.1, 0.2, 48)))
    expected_b0 = float(np.mean(np.linspace(0.12, 0.22, 48)))
    for step in (1, 12, 24):
        label = result["legend_labels"][result["slide_steps"].index(step)]
        assert f"step={step}" in label
        assert f"{expected_b100:.3f} / {expected_b0:.3f}" in label
        assert abs(result["mean_rmse"][step]["B100"] - expected_b100) < 1e-12
        assert abs(result["mean_rmse"][step]["B0"] - expected_b0) < 1e-12
    assert result["legend_title"] == "step   mean RMSE (B100 / B0)"
    assert result["target_has_colorbar"] is False
    assert result["n_column_colorbars"] == 2
    assert result["pred_res_sharey"] is True
    assert "B visible 100%" in result["legend_labels"]
    assert "B visible 0%" in result["legend_labels"]
    for step in (1, 12, 24):
        assert result["qualitative_density_nrmse"][step] == 0.0
    assert (tmp_path / f"{FIGURE_STEMS['sliding']}.png").exists()
    assert (tmp_path / f"{FIGURE_STEMS['sliding']}.pdf").exists()


def test_sliding_qualitative_nrmse_uses_paper_density_definition():
    residual = np.full((48, 6), 0.25, dtype=np.float64)
    arrays = {
        "target_tz": np.zeros((48, 6), dtype=np.float32),
        "step_1_tz": np.zeros((48, 6), dtype=np.float32),
        "step_1_tz_residual": residual,
    }
    nrmse = compute_sliding_qualitative_density_nrmse(arrays, 1)
    assert abs(nrmse - density_nrmse_from_residual(residual)) < 1e-12
    assert abs(nrmse - 0.25) < 1e-12


def test_sliding_qualitative_nrmse_prefers_full_field_prediction_target():
    target = np.ones((48, 3, 4), dtype=np.float64)
    pred = target + 0.4
    tz_target = np.ones((48, 4), dtype=np.float64)
    tz_pred = tz_target + 0.4
    tz_residual = np.full((48, 4), 0.2, dtype=np.float64)
    arrays = {
        "target_density": target,
        "step_12_prediction": pred,
        "target_tz": tz_target,
        "step_12_tz": tz_pred,
        "step_12_tz_residual": tz_residual,
    }
    # tz physical RMSE=0.4, tz NRMSE=0.2 => std=2; full-field NRMSE=0.4/2=0.2
    nrmse = compute_sliding_qualitative_density_nrmse(arrays, 12)
    assert abs(nrmse - 0.2) < 1e-12


def test_sliding_nrmse_metadata_does_not_change_signature():
    arrays = {"step_1_tz_residual": np.full((48, 4), 0.1)}
    metadata = {
        "slide_steps": [1],
        "signature": {"cache_version": 9, "figure": "sliding", "layout": "x"},
    }
    signature = dict(metadata["signature"])
    assert ensure_sliding_qualitative_nrmse_metadata(arrays, metadata)
    assert abs(metadata["qualitative_density_nrmse_step1"] - 0.1) < 1e-12
    assert metadata["signature"] == signature
    assert paper_signatures_match(signature, metadata["signature"])
    assert not ensure_sliding_qualitative_nrmse_metadata(arrays, metadata)
