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
    CACHE_VERSION,
    build_standardized_masks,
    cache_signature,
    cross_run_stats,
    load_cached_payload,
    per_run_means,
    summarize_from_records,
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
