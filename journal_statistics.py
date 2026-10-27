"""Small shared statistical helpers for journal evaluation figures."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np


def _finite_pair(predicted_variance, squared_error):
    predicted = np.asarray(predicted_variance, dtype=np.float64).ravel()
    actual = np.asarray(squared_error, dtype=np.float64).ravel()
    keep = np.isfinite(predicted) & np.isfinite(actual)
    predicted, actual = predicted[keep], actual[keep]
    if predicted.size == 0:
        raise ValueError("No finite uncertainty/error pairs.")
    return predicted, actual


def rankdata(values: Sequence[float]) -> np.ndarray:
    """Average ranks for ties, equivalent to scipy.stats.rankdata(method='average')."""
    values = np.asarray(values, dtype=np.float64).ravel()
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def pearson_spearman(predicted_variance, squared_error) -> Dict[str, float]:
    predicted, actual = _finite_pair(predicted_variance, squared_error)
    if predicted.size < 2 or np.ptp(predicted) == 0 or np.ptp(actual) == 0:
        return {"pearson": float("nan"), "spearman": float("nan")}
    return {
        "pearson": float(np.corrcoef(predicted, actual)[0, 1]),
        "spearman": float(np.corrcoef(rankdata(predicted), rankdata(actual))[0, 1]),
    }


def predicted_voxel_mask(observation_mask, channels: Sequence[int]) -> np.ndarray:
    """Select only values absent from the exact-observation input mask."""
    mask = np.asarray(observation_mask)
    selected = mask[list(channels)] < 0.5
    return np.asarray(selected, dtype=bool)


def calibration_bias(mean_squared_error: float, mean_predicted_variance: float) -> Dict[str, float]:
    """Signed bias is positive for underestimated variance (overconfidence)."""
    signed = float(mean_squared_error) - float(mean_predicted_variance)
    return {"signed_bias": signed, "calibration_gap": abs(signed)}


def quantile_calibration(predicted_variance, squared_error, n_bins: int = 10) -> list[Dict]:
    """Equal-count bins sorted by raw predicted variance."""
    if int(n_bins) < 1:
        raise ValueError("n_bins must be positive")
    predicted, actual = _finite_pair(predicted_variance, squared_error)
    order = np.argsort(predicted, kind="mergesort")
    bins = []
    for index, selection in enumerate(np.array_split(order, min(int(n_bins), order.size))):
        if selection.size:
            bins.append({
                "bin": index + 1,
                "mean_predicted_variance": float(np.mean(predicted[selection])),
                "mean_actual_squared_error": float(np.mean(actual[selection])),
                "count": int(selection.size),
            })
    return bins


def summarize(values: Iterable[float]) -> Dict[str, float | int]:
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in ("mean", "std", "median", "p16", "p84")} | {"n": 0}
    return {
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "median": float(np.median(values)), "p16": float(np.percentile(values, 16)),
        "p84": float(np.percentile(values, 84)), "n": int(values.size),
    }


def summarize_run_records(records: Sequence[Mapping], keys: Sequence[str]) -> Dict[str, Dict]:
    """Cross-run summary; caller must provide one record per independent run."""
    return {key: summarize(record[key] for record in records) for key in keys}


def group_run_records(records: Sequence[Mapping], parameter: str, metrics: Sequence[str]) -> list[Dict]:
    """Group already run-aggregated records, never windows or voxels."""
    groups = {}
    for record in records:
        groups.setdefault(record[parameter], []).append(record)
    return [
        {"parameter": parameter, "value": value, "run_count": len(group),
         "metrics": summarize_run_records(group, metrics)}
        for value, group in sorted(groups.items(), key=lambda item: float(item[0]))
    ]
