from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.masking import (  # noqa: E402
    FIXED_VALIDATION_PATTERNS,
    make_fixed_validation_mask,
)
from evaluate_standardized_validation import (  # noqa: E402
    CACHE_STEM,
    CACHE_VERSION,
    build_standardized_masks,
    cache_signature,
    cross_run_stats,
    evaluate_windows,
    load_cached_payload,
    per_run_means,
    summarize_from_records,
    summarize_density_calibration,
    select_calibration_indices,
    save_density_calibration,
    split_calibration_runs,
    fit_global_calibration_scale,
    posthoc_calibration_summary,
    window_density_nrmse,
    window_jy_nrmse,
    window_overall_mse,
    write_tables_from_payload,
)
from visualize_mask_patterns_unet3d import (  # noqa: E402
    CM_TO_M,
    JY_INCLUDES_MU0,
    JY_STATS_CACHE_VERSION,
    JY_STATS_PREPROCESSING,
    MU0,
    compute_jy,
    denormalize_field_np,
    normalize_field_np,
)


@pytest.mark.parametrize("flags", [[], ["--density-calibration"], ["--fit-global-scale"]])
def test_uncertainty_checkpoint_keeps_standardized_default_and_explicit_calibration(tmp_path, monkeypatch, flags):
    import evaluate_standardized_validation as evaluator

    torch.save(dict(args={"uncertainty_aware": True}, epoch=240,
                    stats=dict(mean=[0.] * 4, std=[1.] * 4)), tmp_path / "latest.pt")
    (tmp_path / "split.json").write_text(json.dumps(dict(val_runs=["held_out"], train_runs=["train"])))
    monkeypatch.setattr(sys, "argv", ["evaluate", "--run-dir", str(tmp_path), *flags])

    def stop_before_inference(*args, **kwargs):
        raise RuntimeError("explicit calibration selected")

    monkeypatch.setattr(evaluator, "load_checkpoint", stop_before_inference)
    if flags:
        with pytest.raises(RuntimeError, match="explicit calibration selected"):
            evaluator.main()
    else:
        # The paper table requires Jy stats; stop here without evaluating data.
        with pytest.raises(FileNotFoundError, match="jy_stats.json"):
            evaluator.main()
        assert not (tmp_path / "density_calibration").exists()


def test_standardized_pattern_order_matches_fixed_validation():
    assert FIXED_VALIDATION_PATTERNS == (
        "spatial_random",
        "spatial_grid",
        "spatial_block",
        "temporal_random",
        "temporal_block",
    )


def test_build_standardized_masks_uses_make_fixed_validation_mask():
    shape = (1, 4, 4, 8, 6)
    masks, infos = build_standardized_masks(shape)
    assert [info["pattern"] for info in infos] == list(FIXED_VALIDATION_PATTERNS)
    reference = [
        make_fixed_validation_mask(pattern, shape)[0] for pattern in FIXED_VALIDATION_PATTERNS
    ]
    assert len(masks) == 5
    for mask, expected in zip(masks, reference):
        torch.testing.assert_close(mask, expected)
        for channel in range(1, 4):
            torch.testing.assert_close(mask[:, 0], mask[:, channel])
        assert abs(float(1.0 - mask.mean().item()) - 0.5) < 0.08


def test_per_run_aggregation_is_mean_over_windows():
    records = [
        {
            "run_name": "run_a",
            "t0": 0,
            "pattern": "spatial_random",
            "overall_mse": 0.2,
            "density_nrmse": 0.4,
            "jy_nrmse": 0.6,
        },
        {
            "run_name": "run_a",
            "t0": 24,
            "pattern": "spatial_random",
            "overall_mse": 0.4,
            "density_nrmse": 0.8,
            "jy_nrmse": 1.0,
        },
        {
            "run_name": "run_b",
            "t0": 0,
            "pattern": "spatial_random",
            "overall_mse": 1.0,
            "density_nrmse": 2.0,
            "jy_nrmse": 3.0,
        },
    ]
    per_run = per_run_means(records)
    assert np.isclose(per_run["spatial_random"]["run_a"]["overall_mse"], 0.3)
    assert np.isclose(per_run["spatial_random"]["run_a"]["density_nrmse"], 0.6)
    assert np.isclose(per_run["spatial_random"]["run_a"]["jy_nrmse"], 0.8)
    assert np.isclose(per_run["spatial_random"]["run_b"]["overall_mse"], 1.0)


def test_cross_run_median_p16_p84():
    stats = cross_run_stats([1.0, 2.0, 3.0, 4.0, 5.0])
    assert stats["median"] == 3.0
    assert np.isclose(stats["p16"], np.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 16.0))
    assert np.isclose(stats["p84"], np.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 84.0))
    assert stats["n_runs"] == 5


def test_window_metrics_match_definitions():
    target = np.zeros((4, 2, 2, 2), dtype=np.float64)
    prediction = np.zeros_like(target)
    prediction[3] = 2.0
    prediction[:3] = 1.0
    assert np.isclose(window_overall_mse(prediction, target), (3 * 1.0 + 4.0) / 4.0)
    assert np.isclose(window_density_nrmse(prediction, target), 2.0)


def test_cache_reuse_and_checkpoint_change_invalidates(tmp_path):
    records = []
    for pattern in FIXED_VALIDATION_PATTERNS:
        for run_name, value in (("run_a", 0.2), ("run_b", 0.4)):
            records.append(
                {
                    "run_name": run_name,
                    "t0": 0,
                    "pattern": pattern,
                    "overall_mse": value,
                    "density_nrmse": value,
                    "jy_nrmse": value,
                    "validate_style_mse": value,
                }
            )
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"abc")
    signature = cache_signature(
        checkpoint_path=checkpoint,
        checkpoint_epoch=4500,
        split={"val_runs": ["run_a", "run_b"], "train_runs": ["train"]},
        mean=[0.0, 0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0, 1.0],
        jy_stats={
            "jy_std_train": 1.0,
            "jy_mean_train": 0.0,
            "jy_definition": "dBx/dz - dBz/dx",
            "n_runs": 1,
            "count": 1,
        },
        extent=[-1.0, 1.0, -1.0, 1.0],
    )
    payload = {
        "signature": signature,
        "records": records,
        "summary": summarize_from_records(records),
    }
    (tmp_path / f"{CACHE_STEM}.json").write_text(json.dumps(payload))
    np.savez(tmp_path / f"{CACHE_STEM}.npz", overall_mse=np.array([0.2, 0.4]))

    reused = load_cached_payload(tmp_path, signature)
    assert reused is not None
    write_tables_from_payload(reused, tmp_path)
    assert (tmp_path / f"{CACHE_STEM}.csv").exists()
    assert (tmp_path / f"{CACHE_STEM}_table.md").exists()
    assert (tmp_path / f"{CACHE_STEM}_table.tex").exists()

    checkpoint.write_bytes(b"abcd")
    changed = cache_signature(
        checkpoint_path=checkpoint,
        checkpoint_epoch=4500,
        split={"val_runs": ["run_a", "run_b"], "train_runs": ["train"]},
        mean=[0.0, 0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0, 1.0],
        jy_stats={
            "jy_std_train": 1.0,
            "jy_mean_train": 0.0,
            "jy_definition": "dBx/dz - dBz/dx",
            "n_runs": 1,
            "count": 1,
        },
        extent=[-1.0, 1.0, -1.0, 1.0],
    )
    assert changed != signature
    assert load_cached_payload(tmp_path, changed) is None


def test_standardized_validation_cache_version_and_jy_preprocessing(tmp_path):
    assert CACHE_VERSION == 3
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"abc")
    jy_old = {
        "jy_std_train": 1.0,
        "jy_mean_train": 0.0,
        "jy_definition": "dBx/dz - dBz/dx",
        "jy_preprocessing": "checkpoint channel-standardized Bx,Bz",
        "jy_includes_mu0": False,
        "n_runs": 1,
        "count": 1,
    }
    jy_new = {
        **jy_old,
        "jy_preprocessing": JY_STATS_PREPROCESSING,
        "jy_stats_cache_version": JY_STATS_CACHE_VERSION,
        "jy_includes_mu0": JY_INCLUDES_MU0,
        "B_unit": "T",
        "jy_unit": "A/m^2",
    }
    split = {"val_runs": ["run_a"], "train_runs": ["train"]}
    old_sig = cache_signature(
        checkpoint_path=checkpoint,
        checkpoint_epoch=4500,
        split=split,
        mean=[0.0, 0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0, 1.0],
        jy_stats=jy_old,
        extent=[-21.0, 21.0, -50.0, 50.0],
    )
    new_sig = cache_signature(
        checkpoint_path=checkpoint,
        checkpoint_epoch=4500,
        split=split,
        mean=[0.0, 0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0, 1.0],
        jy_stats=jy_new,
        extent=[-21.0, 21.0, -50.0, 50.0],
    )
    assert old_sig != new_sig
    assert new_sig["cache_version"] == 3
    assert new_sig["jy_stats"]["jy_preprocessing"] == JY_STATS_PREPROCESSING
    assert new_sig["jy_stats"]["jy_includes_mu0"] is True


def test_window_jy_nrmse_denormalizes_before_derivatives():
    field = np.zeros((4, 2, 6, 5), dtype=np.float64)
    z = np.linspace(-21.0, 21.0, 5)
    field[0] = z[None, None, :]
    pred = np.zeros_like(field)
    extent = [-21.0, 21.0, -50.0, 50.0]
    jy_scale = float(np.abs(np.mean(compute_jy(field, extent))))
    np.testing.assert_allclose(window_jy_nrmse(pred, field, extent, jy_scale), 1.0)
    np.testing.assert_allclose(jy_scale, (1.0 / CM_TO_M) / MU0, rtol=1e-12)
    mean = np.zeros(4)
    std = np.array([2.0, 1.0, 4.0, 1.0])
    pred_norm = normalize_field_np(pred, mean, std)
    target_norm = normalize_field_np(field, mean, std)
    old_wrong = window_jy_nrmse(pred_norm, target_norm, extent, jy_scale)
    assert not np.isclose(old_wrong, 1.0)
    new = window_jy_nrmse(
        denormalize_field_np(pred_norm, mean, std),
        denormalize_field_np(target_norm, mean, std),
        extent,
        jy_scale,
    )
    np.testing.assert_allclose(new, 1.0)


def test_calibration_selection_is_reproducible_stratified_and_metadata_only():
    class IndexOnlyDataset:
        file_infos = [{"beta_file": b} for b in [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]]
        samples = [(f, f"run_{f}_{r}", t) for f in range(10) for r in range(4) for t in (0, 2, 4)]

        def __getitem__(self, index):
            raise AssertionError("Selection must not read any field data")

    ds = IndexOnlyDataset()
    val = {f"run_{f}_{r}" for f in range(10) for r in range(3)}
    betas = [info["beta_file"] for info in ds.file_infos]
    first = select_calibration_indices(ds, val, betas, 2, 1, 1234)
    assert first == select_calibration_indices(ds, val, betas, 2, 1, 1234)
    assert len(first) == len({ds.samples[i][1] for i in first}) == 20
    assert all(ds.samples[i][1] in val for i in first)
    assert all(sum(ds.samples[i][0] == f for i in first) == 2 for f in range(10))
    with pytest.raises(ValueError, match="eligible validation runs"):
        select_calibration_indices(ds, val, betas, 4, 1, 1234)


def test_calibration_bins_preserve_extremes_means_medians_and_counts():
    summary = summarize_density_calibration([1, 2, 3, 4, 5, 6], [0, 0, 0.9, 0, 0, 3], 2)
    assert [b["count"] for b in summary["bins"]] == [3, 3]
    np.testing.assert_allclose([b["mean_predicted"] for b in summary["bins"]], [2, 5])
    np.testing.assert_allclose([b["mean_actual"] for b in summary["bins"]], [0.3, 1])
    assert [b["median_actual"] for b in summary["bins"]] == [0, 0]
    assert summary["bins"][-1]["predicted_max"] == 6  # Never clip to one.
    assert np.isclose(summary["mean_bias_predicted_minus_actual"], 2.85)
    assert summary["increasing_mean_transitions"] == 1
    assert summarize_density_calibration(np.ones(12), np.zeros(12))["bins"][0]["count"] == 12
    with pytest.raises(ValueError, match="finite, nonnegative"):
        summarize_density_calibration([1, np.nan], [1, 2])


@pytest.mark.parametrize("uncertainty", [False, True])
@pytest.mark.parametrize("prediction_value", [0.0, 0.25])
def test_evaluation_reuses_uncertain_inputs_and_samples_only_hidden_density(uncertainty, prediction_value):
    class Dataset:
        def __getitem__(self, index):
            return {"block": torch.full((4, 2, 4, 4), 5.0),
                    "metadata": dict(run_name="held_out", beta=0.08, t0=2)}

    class Model(torch.nn.Module):
        uncertainty_aware = uncertainty

        def forward(self, inputs):
            self.inputs = inputs
            pred = torch.full_like(inputs[:, :4], prediction_value)
            return (pred, torch.full_like(pred, 2.5)) if uncertainty else pred

    model = Model()
    samples = [] if uncertainty else None
    records, _ = evaluate_windows(
        model, Dataset(), [0], torch.full((1, 4, 1, 1, 1), 3.0),
        torch.full((1, 4, 1, 1, 1), 2.0), torch.device("cpu"), [-2, 2, -3, 3],
        None if uncertainty else 1.0, calibration_samples=samples,
        calibration_voxels_per_pattern=3,
    )
    masks, _ = build_standardized_masks((1, 4, 2, 4, 4))
    stacked = torch.cat(masks)
    torch.testing.assert_close(model.inputs[:, 4:], 1 - stacked if uncertainty else stacked)
    assert len(records) == 5
    expected_error = (1.0 - prediction_value) ** 2
    assert all(np.isclose(record["overall_mse"], expected_error) for record in records)
    if uncertainty:
        assert len(samples) == 5
        for mask, part in zip(masks, samples):
            assert len(part["observed"]) == 3 and not part["observed"].any()
            assert (mask[0, 3].numpy().ravel()[part["voxel_index"]] == 0).all()
            np.testing.assert_allclose(part["predicted_squared_error"], 2.5)
            np.testing.assert_allclose(part["actual_squared_error"], expected_error)
            assert set(part["run_name"]) == {"held_out"}
            assert set(part["beta"]) == {0.08}
    else:
        assert all(np.isclose(record["jy_nrmse"], 0, atol=1e-6) for record in records)


def test_calibration_outputs_include_raw_voxels_bins_and_reliability_plot(tmp_path):
    samples = [dict(predicted_squared_error=np.arange(1, 7, dtype=np.float32),
                    actual_squared_error=np.array([0, 0, 0.9, 0, 0, 3], dtype=np.float32),
                    observed=np.zeros(6, dtype=bool), beta=np.full(6, 0.08),
                    pattern=np.full(6, "spatial_random"), run_name=np.full(6, "held_out"))]
    save_density_calibration(samples, [], {"checkpoint_epoch": 19}, tmp_path, 2)
    raw = np.load(tmp_path / "density_calibration_voxels.npz", allow_pickle=False)
    assert raw["predicted_squared_error"].max() == 6
    report = json.loads((tmp_path / "density_calibration.json").read_text())
    assert sum(b["count"] for b in report["bins"]) == 6
    assert "median is descriptive" in report["interpretation"]
    assert (tmp_path / "density_calibration.png").stat().st_size > 0
    assert "median_actual" in (tmp_path / "density_calibration_bins.csv").read_text()


def test_posthoc_split_is_beta_balanced_and_keeps_all_windows_of_a_run_together():
    class Dataset:
        file_infos = [{"beta_file": float(i + 1)} for i in range(10)]
        samples = [(f, f"run_{f}_{r}", t) for f in range(10) for r in range(10) for t in (0, 2)]

    ds = Dataset()
    split = split_calibration_runs(ds, range(200), 1234)
    assert split == split_calibration_runs(ds, range(200), 1234)
    fit, evaluation = set(split["calibration_runs"]), set(split["evaluation_runs"])
    assert len(fit) == len(evaluation) == 50 and not fit & evaluation
    assert fit | evaluation == {name for _, name, _ in ds.samples}
    assert all(len(g["calibration"]) == len(g["evaluation"]) == 5 for g in split["by_beta"].values())
    with pytest.raises(ValueError, match="even number"):
        split_calibration_runs(ds, range(198), 1234)


def _posthoc_samples():
    names = np.repeat(["fit_a", "fit_b", "eval_a", "eval_b"], 10)
    patterns = np.tile(np.repeat(FIXED_VALIDATION_PATTERNS, 2), 4)
    predicted = np.linspace(0.1, 4, 40)
    fit = np.char.startswith(names, "fit")
    return dict(run_name=names, pattern=patterns, beta=np.repeat([0.01, 0.2, 0.01, 0.2], 10),
                predicted_squared_error=predicted, actual_squared_error=predicted * np.where(fit, 3, 8),
                observed=np.zeros(40, dtype=bool))


def test_global_scale_uses_calibration_runs_only_and_preserves_raw_bins_and_ranking():
    arrays = _posthoc_samples()
    original = arrays["predicted_squared_error"].copy()
    split = dict(calibration_runs=["fit_a", "fit_b"], evaluation_runs=["eval_a", "eval_b"])
    summary = posthoc_calibration_summary(arrays, split, 4)
    assert np.isclose(summary["global_scale"], 3)
    assert summary["calibration"]["n_runs"] == summary["evaluation"]["n_runs"] == 2
    assert summary["ranking"]["unchanged"]
    assert np.isclose(summary["calibration"]["calibrated_mean_bias"], 0)
    assert np.isclose(summary["evaluation"]["calibrated_actual_over_predicted"], 8 / 3)
    raw = summarize_density_calibration(original[20:], arrays["actual_squared_error"][20:], 4)
    for before, after in zip(raw["bins"], summary["bins"]):
        for key in ("count", "mean_actual", "median_actual", "mean_predicted"):
            assert before[key] == after[key]
        assert np.isclose(after["mean_calibrated_predicted"], 3 * after["mean_predicted"])
    arrays["actual_squared_error"][20:] *= 100  # Evaluation labels must never affect fitting.
    assert posthoc_calibration_summary(arrays, split, 4)["global_scale"] == summary["global_scale"]
    np.testing.assert_array_equal(arrays["predicted_squared_error"], original)
    with pytest.raises(ValueError, match="disjoint"):
        posthoc_calibration_summary(arrays, dict(calibration_runs=["fit_a"], evaluation_runs=["fit_a"]), 4)


def test_global_scale_handles_zero_predictions_and_zero_error_explicitly():
    assert fit_global_calibration_scale([0, 0], [0, 0]) == 1
    assert fit_global_calibration_scale([1, 2], [0, 0]) == 0
    with pytest.raises(ValueError, match="zero predicted mean"):
        fit_global_calibration_scale([0, 0], [0, 1])
    arrays = _posthoc_samples()
    arrays["actual_squared_error"][:20] = 0
    summary = posthoc_calibration_summary(
        arrays, dict(calibration_runs=["fit_a", "fit_b"], evaluation_runs=["eval_a", "eval_b"]), 4)
    assert summary["global_scale"] == 0 and not summary["ranking"]["unchanged"]
    assert summary["evaluation"]["calibrated_actual_over_predicted"] is None


@pytest.mark.parametrize("objective", ["smooth_l1", "gaussian_nll"])
def test_posthoc_outputs_preserve_raw_predictions_and_save_split_groups_and_heatmap(tmp_path, objective):
    arrays = _posthoc_samples()
    split = dict(calibration_runs=["fit_a", "fit_b"], evaluation_runs=["eval_a", "eval_b"])
    save_density_calibration([arrays], [], {"checkpoint_epoch": 19, "uncertainty_objective": objective},
                             tmp_path, 4, run_split=split)
    raw = np.load(tmp_path / "density_calibration_voxels.npz", allow_pickle=False)
    np.testing.assert_array_equal(raw["predicted_squared_error"], arrays["predicted_squared_error"])
    assert set(raw["subset"][:20]) == {"calibration"} and set(raw["subset"][20:]) == {"evaluation"}
    report = json.loads((tmp_path / "density_calibration.json").read_text())
    assert "not an independent test set" in report["interpretation"]
    assert report["uncertainty_objective"] == objective
    assert ("Gaussian NLL" if objective == "gaussian_nll" else "Smooth L1") in report["interpretation"]
    assert json.loads((tmp_path / "calibration_run_split.json").read_text()) == split
    assert sum(b["count"] for b in report["bins"]) == 20
    assert (tmp_path / "density_calibration_heatmap.png").stat().st_size > 0
    assert "calibrated_actual_over_predicted" in (tmp_path / "density_calibration_groups.csv").read_text()
