from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_paper_figures import (  # noqa: E402
    DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS,
    EXPECTED_FORECAST_ROW_NAMES,
    EXPECTED_MAGNETIC_ROW_NAMES,
    EXPECTED_SUPERRES_ROW_NAMES,
    FIGURE_STEMS,
    SPATIAL_ROW_ORDER,
    _val_json_path,
    base_signature,
    build_magnetic_ablation_cache,
    build_paired_b_condition_cache,
    density_visible_ratio,
    load_figure_cache,
    load_or_build_figure_cache,
    local_index_for_global_frame,
    manifest_figure_entry,
    plot_density_forecast_summary,
    plot_density_superres_summary,
    plot_magnetic_ablation_summary,
    plot_spatial_qualitative,
    sliding_provenance_error,
    try_load_validation_json,
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
    plot_density_forecast_summary(arrays, meta, tmp_path, dpi=80)
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
    arrays, meta = build_magnetic_ablation_cache(
        args, tmp_path, checkpoint, None, checkpoint_epoch=4500
    )
    percents = [float(value) for value in arrays["b_visible_percent"]]
    assert meta["n_levels"] == 8
    assert percents == sorted(percents)
    np.testing.assert_allclose(sorted(percents), sorted(DEFAULT_MAGNETIC_ABLATION_VISIBLE_PERCENTS))
    plot_magnetic_ablation_summary(arrays, meta, tmp_path, dpi=80)
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
    plot_density_superres_summary(
        arrays, meta, tmp_path, dpi=80, size_x=size_x, size_z=size_z
    )
    expected = [100.0 * density_visible_ratio(count, size_x, size_z) for count in (0, 10, 100, 1000)]
    np.testing.assert_allclose(meta["visible_ratio_percent"], expected)
    assert size_x * size_z != 154 * 62
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
