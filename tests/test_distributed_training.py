from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_masked_unet3d import (  # noqa: E402
    DistributedEvalSampler,
    RESUME_PARAMETER_KEYS,
    learning_rate_for_epoch,
    parse_args,
    sample_mixed_magnetic_visible_counts,
    sample_mixed_density_probe_counts,
    validate_resume_parameters,
)


def test_distributed_eval_sampler_covers_each_sample_once():
    dataset = list(range(23))
    world_size = 4
    shards = [
        list(DistributedEvalSampler(dataset, rank=rank, world_size=world_size))
        for rank in range(world_size)
    ]

    combined = [index for shard in shards for index in shard]
    assert sorted(combined) == list(range(len(dataset)))
    assert len(combined) == len(set(combined))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def test_warmup_cosine_schedule_hits_endpoints_and_is_monotonic():
    learning_rates = [
        learning_rate_for_epoch(
            epoch=epoch,
            total_epochs=200,
            base_lr=2e-4,
            warmup_epochs=10,
            min_lr=2e-6,
        )
        for epoch in range(1, 201)
    ]

    assert abs(learning_rates[0] - 2e-5) < 1e-12
    assert abs(learning_rates[9] - 2e-4) < 1e-12
    assert abs(learning_rates[-1] - 2e-6) < 1e-12
    assert all(a < b for a, b in zip(learning_rates[:9], learning_rates[1:10]))
    assert all(a > b for a, b in zip(learning_rates[9:], learning_rates[10:]))


def test_probe_patterns_use_balanced_sparse_dense_count_mixture():
    patterns = [
        "spatial_random",
        "spatial_grid",
        "spatial_block",
        "temporal_random",
    ] * 2000
    spatial_sites = 4096
    counts = sample_mixed_density_probe_counts(
        patterns,
        0,
        30,
        spatial_sites=spatial_sites,
    )

    for pattern, count in zip(patterns, counts):
        if pattern in {"spatial_grid", "spatial_random"}:
            assert 0 <= count <= spatial_sites
        else:
            assert count is None

    for pattern in ("spatial_grid", "spatial_random"):
        pattern_counts = [
            count
            for sampled_pattern, count in zip(patterns, counts)
            if sampled_pattern == pattern
        ]
        sparse_share = sum(count <= 30 for count in pattern_counts) / len(
            pattern_counts
        )
        assert 0.46 < sparse_share < 0.54
        assert any(count > 30 for count in pattern_counts)


def test_probe_patterns_use_half_full_half_random_magnetic_counts():
    patterns = [
        "spatial_random",
        "spatial_grid",
        "spatial_block",
        "temporal_random",
    ] * 2000
    spatial_sites = 4096
    counts = sample_mixed_magnetic_visible_counts(
        patterns,
        spatial_sites=spatial_sites,
    )

    for pattern, count in zip(patterns, counts):
        if pattern in {"spatial_grid", "spatial_random"}:
            assert 0 <= count <= spatial_sites
        else:
            assert count is None

    for pattern in ("spatial_grid", "spatial_random"):
        pattern_counts = [
            count
            for sampled_pattern, count in zip(patterns, counts)
            if sampled_pattern == pattern
        ]
        full_share = sum(count == spatial_sites for count in pattern_counts) / len(
            pattern_counts
        )
        mean_visible_fraction = sum(pattern_counts) / (
            len(pattern_counts) * spatial_sites
        )
        assert 0.46 < full_share < 0.54
        assert 0.72 < mean_visible_fraction < 0.78
        assert any(count < spatial_sites for count in pattern_counts)


def test_spatial_only_pooling_cli_defaults_off_and_flag_enables(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_masked_unet3d.py"])
    assert parse_args().spatial_only_pooling is False

    monkeypatch.setattr(
        sys,
        "argv",
        ["train_masked_unet3d.py", "--spatial-only-pooling"],
    )
    assert parse_args().spatial_only_pooling is True


def test_old_resume_args_default_to_optional_architecture_features_off():
    values = {key: None for key in RESUME_PARAMETER_KEYS}
    values["mask_pattern_weights"] = {"temporal_random": 1.0}
    values["use_attention"] = False
    values["spatial_only_pooling"] = False
    args = SimpleNamespace(**values)
    old_checkpoint_args = {
        key: value
        for key, value in values.items()
        if key not in {"use_attention", "spatial_only_pooling"}
    }

    validate_resume_parameters(args, old_checkpoint_args)

    args.use_attention = True
    try:
        validate_resume_parameters(args, old_checkpoint_args)
    except ValueError as exc:
        assert "use_attention" in str(exc)
    else:
        raise AssertionError("Expected attention mismatch to block auto-resume")

    args.use_attention = False
    args.spatial_only_pooling = True
    try:
        validate_resume_parameters(args, old_checkpoint_args)
    except ValueError as exc:
        assert "spatial_only_pooling" in str(exc)
    else:
        raise AssertionError("Expected pooling mismatch to block auto-resume")
