from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.masking import (  # noqa: E402
    FIXED_VALIDATION_PATTERNS,
    make_fixed_validation_mask,
)
from evaluate_standardized_validation import (  # noqa: E402
    CACHE_STEM,
    build_standardized_masks,
    cache_signature,
    cross_run_stats,
    load_cached_payload,
    per_run_means,
    summarize_from_records,
    window_density_nrmse,
    window_overall_mse,
    write_tables_from_payload,
)


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
