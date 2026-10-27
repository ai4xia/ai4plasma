from __future__ import annotations

import numpy as np

from journal_statistics import (
    calibration_bias,
    group_run_records,
    pearson_spearman,
    predicted_voxel_mask,
    quantile_calibration,
)


def test_uncertainty_correlations_and_equal_count_bins():
    predicted = np.arange(1, 21, dtype=float)
    actual = predicted**2
    correlation = pearson_spearman(predicted, actual)
    assert correlation["pearson"] > 0.95
    assert correlation["spearman"] == 1.0
    bins = quantile_calibration(predicted, actual, 10)
    assert len(bins) == 10
    assert [row["count"] for row in bins] == [2] * 10
    assert sum(row["count"] for row in bins) == predicted.size


def test_parameter_summary_groups_run_records_before_cross_run_statistics():
    records = [
        {"beta": .1, "metric": 1.0},
        {"beta": .1, "metric": 3.0},
        {"beta": .2, "metric": 8.0},
    ]
    groups = group_run_records(records, "beta", ["metric"])
    assert groups[0]["run_count"] == 2
    assert groups[0]["metrics"]["metric"]["median"] == 2.0
    assert groups[1]["run_count"] == 1


def test_calibration_values_remain_in_same_standardized_squared_units():
    residual = np.array([-2.0, -1.0, 1.0, 2.0])
    predicted_variance = np.array([4.0, 1.0, 1.0, 4.0])
    bins = quantile_calibration(predicted_variance, residual**2, 2)
    for row in bins:
        assert row["mean_predicted_variance"] == row["mean_actual_squared_error"]


def test_primary_uq_mask_selects_only_unknown_not_exact_observations():
    observation_mask = np.array([
        [[[1, 0], [1, 0]]],
        [[[1, 1], [1, 1]]],
        [[[0, 0], [1, 1]]],
        [[[1, 0], [0, 1]]],
    ], dtype=np.float32)
    density = predicted_voxel_mask(observation_mask, [3])
    assert density.tolist() == [[[[False, True], [True, False]]]]
    assert not predicted_voxel_mask(observation_mask, [1]).any()
    magnetic = predicted_voxel_mask(observation_mask, [0, 1, 2])
    assert magnetic.shape[0] == 3
    assert magnetic.sum() == 4


def test_signed_calibration_bias_preserves_direction():
    assert calibration_bias(3.0, 2.0) == {"signed_bias": 1.0, "calibration_gap": 1.0}
    assert calibration_bias(1.0, 2.5) == {"signed_bias": -1.5, "calibration_gap": 1.5}
