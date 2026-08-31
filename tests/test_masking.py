# tests/test_masking.py

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.masking import (  # noqa: E402
    DEFAULT_PATTERN_WEIGHTS,
    MASK_PATTERNS,
    SPATIAL_MASK_PATTERNS,
    TEMPORAL_MASK_PATTERNS,
    parse_pattern_weights,
    sample_batch_masks,
    sample_mask,
    sample_patterns,
)


SHAPE = (2, 4, 8, 154, 62)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def test_shape_and_binary():
    for pattern in MASK_PATTERNS:
        for p in [0.0, 0.15, 0.5, 0.85, 1.0]:
            mask, info = sample_mask(SHAPE, pattern, p, generator=make_generator(0))

            assert tuple(mask.shape) == SHAPE, (pattern, p, mask.shape)
            assert torch.all((mask == 0) | (mask == 1)), (pattern, p)
            assert info["pattern"] == pattern
            assert info["target_mask_fraction"] == p


def test_spatial_masks_are_constant_in_time():
    T = SHAPE[2]

    for pattern in SPATIAL_MASK_PATTERNS:
        for p in [0.1, 0.5, 0.9]:
            mask, _ = sample_mask(SHAPE, pattern, p, generator=make_generator(1))

            for t in range(T):
                assert torch.equal(mask[:, :, t], mask[:, :, 0]), (pattern, p, t)


def test_temporal_masks_are_constant_in_space():
    for pattern in TEMPORAL_MASK_PATTERNS:
        mask, _ = sample_mask(SHAPE, pattern, 0.5, generator=make_generator(2))

        per_frame = mask.flatten(3)
        assert torch.all(per_frame == per_frame[..., :1]), pattern


def test_actual_fraction_tracks_target():
    # Continuous patterns should land close to the target. Discrete patterns are
    # only expected to follow the target on average over a range of p.
    tolerances = {
        "spatial_random": 0.02,
        "spatial_grid": 0.30,
        "spatial_block": 0.05,
        "temporal_random": 0.10,
    }

    for pattern in MASK_PATTERNS:
        errors = []

        for i in range(24):
            generator = make_generator(100 + i)
            p = float(torch.rand((), generator=generator))

            mask, info = sample_mask(SHAPE, pattern, p, generator=generator)

            measured = float(1.0 - mask.float().mean())
            assert abs(measured - info["actual_mask_fraction"]) < 1e-5, pattern

            errors.append(measured - p)

        mean_error = sum(errors) / len(errors)
        max_error = max(abs(e) for e in errors)

        assert abs(mean_error) < tolerances[pattern], (pattern, mean_error)
        assert max_error < 3 * tolerances[pattern] + 0.05, (pattern, max_error)


def test_temporal_random_masks_whole_frames():
    mask, info = sample_mask(SHAPE, "temporal_random", 0.5, generator=make_generator(3))

    visible_frames = mask[0, 0, :, 0, 0]
    expected = torch.zeros(SHAPE[2])
    expected[info["visible_frames"]] = 1.0

    assert torch.equal(visible_frames, expected)
    assert len(info["visible_frames"]) == info["num_visible_frames"]

    density_visible_frames = mask[0, 3, :, 0, 0]
    density_expected = torch.zeros(SHAPE[2])
    density_expected[info["density_visible_frames"]] = 1.0
    assert torch.equal(density_visible_frames, density_expected)
    assert (
        len(info["density_visible_frames"])
        == info["density_num_visible_frames"]
    )
    assert torch.equal(mask[:, 0], mask[:, 3])


def test_temporal_random_mixes_scattered_and_bidirectional_boundary_masks():
    shape = (1, 4, 16, 3, 2)
    mode_counts = {"random_frames": 0, "contiguous_boundary": 0}
    direction_counts = {"visible_prefix": 0, "visible_suffix": 0}

    for seed in range(400):
        mask, info = sample_mask(
            shape,
            "temporal_random",
            0.375,
            generator=make_generator(seed),
        )
        mode_counts[info["temporal_mode"]] += 1

        if info["temporal_mode"] != "contiguous_boundary":
            continue

        direction = info["temporal_direction"]
        direction_counts[direction] += 1
        frames = mask[0, 0, :, 0, 0]
        transition_index = info["transition_index"]

        if direction == "visible_prefix":
            assert torch.all(frames[:transition_index] == 1)
            assert torch.all(frames[transition_index:] == 0)
        elif direction == "visible_suffix":
            assert torch.all(frames[:transition_index] == 0)
            assert torch.all(frames[transition_index:] == 1)
        else:
            raise AssertionError(f"Unexpected temporal direction: {direction}")

    assert 0.4 < mode_counts["random_frames"] / 400 < 0.6
    assert 0.4 < mode_counts["contiguous_boundary"] / 400 < 0.6
    boundary_count = mode_counts["contiguous_boundary"]
    assert 0.4 < direction_counts["visible_prefix"] / boundary_count < 0.6
    assert 0.4 < direction_counts["visible_suffix"] / boundary_count < 0.6


def test_spatial_grid_is_a_regular_lattice():
    X, Z = SHAPE[3], SHAPE[4]

    for i in range(12):
        generator = make_generator(300 + i)
        p = float(torch.rand((), generator=generator))

        mask, info = sample_mask(SHAPE, "spatial_grid", p, generator=generator)

        stride = info["stride"]
        expected = torch.zeros(X, Z)
        expected[info["offset_x"] :: stride, info["offset_z"] :: stride] = 1.0

        assert torch.equal(mask[0, 0, 0], expected), (p, stride)
        # Isotropic by construction, and the offset stays inside one cell.
        assert 0 <= info["offset_x"] < stride
        assert 0 <= info["offset_z"] < stride

        # All magnetic channels observe the same lattice.
        for c in (1, 2):
            assert torch.equal(mask[0, c, 0], mask[0, 0, 0])

        density_expected = torch.zeros(X, Z)
        density_expected[
            info["density_offset_x"] :: info["density_stride"],
            info["density_offset_z"] :: info["density_stride"],
        ] = 1.0
        assert torch.equal(mask[0, 3, 0], density_expected)


def test_spatial_grid_stride_can_be_pinned():
    mask, info = sample_mask(
        SHAPE,
        "spatial_grid",
        0.5,
        generator=make_generator(4),
        grid_stride=4,
    )

    assert info["stride"] == 4
    assert abs(info["actual_mask_fraction"] - (1.0 - 1.0 / 16.0)) < 0.01


def test_probe_count_grid_keeps_b_visible_and_uses_exact_count():
    spatial_sites = SHAPE[-2] * SHAPE[-1]
    for probe_count in (0, 1, 10, 30, 31, spatial_sites // 2, spatial_sites):
        target_fraction = 0.25 * (1.0 - probe_count / spatial_sites)
        mask, info = sample_mask(
            SHAPE,
            "spatial_grid",
            target_fraction,
            generator=make_generator(400 + probe_count),
            density_probe_count=probe_count,
        )

        assert torch.all(mask[:, :3] == 1)
        assert int(mask[0, 3, 0].sum()) == probe_count
        assert torch.equal(mask[:, 3, 0], mask[:, 3, -1])
        assert info["density_probe_count"] == probe_count
        assert abs(info["actual_mask_fraction"] - target_fraction) < 1e-6


def test_probe_count_spatial_random_keeps_b_visible_and_uses_exact_count():
    spatial_sites = SHAPE[-2] * SHAPE[-1]
    for probe_count in (0, 1, 10, 30, 31, spatial_sites // 2, spatial_sites):
        target_fraction = 0.25 * (1.0 - probe_count / spatial_sites)
        mask, info = sample_mask(
            SHAPE,
            "spatial_random",
            target_fraction,
            generator=make_generator(700 + probe_count),
            density_probe_count=probe_count,
        )

        assert torch.all(mask[:, :3] == 1)
        assert int(mask[0, 3, 0].sum()) == probe_count
        assert torch.equal(mask[:, 3, 0], mask[:, 3, -1])
        assert info["density_probe_count"] == probe_count
        assert abs(info["actual_mask_fraction"] - target_fraction) < 1e-6


def test_exact_magnetic_count_is_shared_by_all_b_channels():
    spatial_sites = SHAPE[-2] * SHAPE[-1]
    density_count = 17
    for pattern in ("spatial_grid", "spatial_random"):
        for magnetic_count in (0, 1, 30, spatial_sites // 2, spatial_sites):
            target_fraction = 1.0 - (
                (3 * magnetic_count + density_count) / (4.0 * spatial_sites)
            )
            mask, info = sample_mask(
                SHAPE,
                pattern,
                target_fraction,
                generator=make_generator(900 + magnetic_count),
                density_probe_count=density_count,
                magnetic_visible_count=magnetic_count,
            )

            assert int(mask[0, 0, 0].sum()) == magnetic_count
            assert torch.equal(mask[:, 0], mask[:, 1])
            assert torch.equal(mask[:, 1], mask[:, 2])
            assert int(mask[0, 3, 0].sum()) == density_count
            assert torch.equal(mask[:, :, 0], mask[:, :, -1])
            assert info["magnetic_visible_count"] == magnetic_count
            assert abs(info["actual_mask_fraction"] - target_fraction) < 1e-6


def test_spatial_block_is_one_contiguous_rectangle():
    mask, info = sample_mask(SHAPE, "spatial_block", 0.5, generator=make_generator(5))

    plane = mask[0, 0, 0]
    hidden_rows = (plane == 0).any(dim=1).nonzero().flatten()
    hidden_cols = (plane == 0).any(dim=0).nonzero().flatten()

    assert len(hidden_rows) == info["rect_height"]
    assert len(hidden_cols) == info["rect_width"]
    # Contiguous in both directions.
    assert torch.equal(hidden_rows, torch.arange(hidden_rows[0], hidden_rows[-1] + 1))
    assert torch.equal(hidden_cols, torch.arange(hidden_cols[0], hidden_cols[-1] + 1))

    density_plane = mask[0, 3, 0]
    density_hidden_rows = (density_plane == 0).any(dim=1).nonzero().flatten()
    density_hidden_cols = (density_plane == 0).any(dim=0).nonzero().flatten()
    assert len(density_hidden_rows) == info["density_rect_height"]
    assert len(density_hidden_cols) == info["density_rect_width"]
    assert torch.equal(mask[:, 0], mask[:, 3])


def test_channel_sharing_rules_by_pattern():
    for pattern in MASK_PATTERNS:
        density_differs = False

        for seed in range(20, 32):
            mask, _ = sample_mask(
                SHAPE,
                pattern,
                0.5,
                generator=make_generator(seed),
            )

            assert torch.equal(mask[:, 0], mask[:, 1]), pattern
            assert torch.equal(mask[:, 1], mask[:, 2]), pattern
            density_differs |= not torch.equal(mask[:, 0], mask[:, 3])

        if pattern in {"spatial_random", "spatial_grid"}:
            assert density_differs, pattern
        else:
            assert not density_differs, pattern


def test_layout_and_ratio_vary_with_seed():
    for pattern in MASK_PATTERNS:
        fractions = set()
        distinct_masks = []

        for i in range(12):
            generator = make_generator(500 + i)
            p = float(torch.rand((), generator=generator))

            mask, info = sample_mask(SHAPE, pattern, p, generator=generator)

            fractions.add(round(info["actual_mask_fraction"], 6))

            if not any(torch.equal(mask, other) for other in distinct_masks):
                distinct_masks.append(mask)

        assert len(fractions) > 1, pattern
        assert len(distinct_masks) > 1, pattern


def test_layout_varies_at_fixed_fraction():
    # Same p, different seed: the location of the mask must still move.
    for pattern in MASK_PATTERNS:
        masks = []
        for i in range(8):
            mask, _ = sample_mask(SHAPE, pattern, 0.5, generator=make_generator(600 + i))
            masks.append(mask)

        assert any(not torch.equal(masks[0], m) for m in masks[1:]), pattern


def test_same_seed_is_reproducible():
    for pattern in MASK_PATTERNS:
        a, info_a = sample_mask(SHAPE, pattern, 0.42, generator=make_generator(7))
        b, info_b = sample_mask(SHAPE, pattern, 0.42, generator=make_generator(7))

        assert torch.equal(a, b), pattern
        assert info_a["actual_mask_fraction"] == info_b["actual_mask_fraction"]


def test_batch_masks_mix_patterns_per_sample():
    patterns = ["spatial_random", "temporal_random", "spatial_grid", "spatial_block"]
    fractions = [0.1, 0.4, 0.7, 0.9]
    shape = (len(patterns), 4, 8, 40, 30)

    mask, infos = sample_batch_masks(
        shape,
        patterns=patterns,
        mask_fractions=fractions,
        generator=make_generator(8),
    )

    assert tuple(mask.shape) == shape
    assert [info["pattern"] for info in infos] == patterns
    assert [info["target_mask_fraction"] for info in infos] == fractions

    for i in range(len(patterns)):
        measured = float(1.0 - mask[i].float().mean())
        assert abs(measured - infos[i]["actual_mask_fraction"]) < 1e-5


def test_parse_pattern_weights():
    weights = parse_pattern_weights(["spatial_random", "2", "spatial_grid", "2.5"])
    assert weights == {"spatial_random": 2.0, "spatial_grid": 2.5}

    for bad in (
        ["spatial_random"],
        ["nonsense", "1"],
        ["spatial_random", "abc"],
        ["spatial_random", "-1"],
        ["spatial_random", "0"],
        ["spatial_random", "1", "spatial_random", "2"],
    ):
        try:
            parse_pattern_weights(bad)
        except ValueError:
            continue
        raise AssertionError(f"Expected ValueError for {bad}")


def test_sample_patterns_follows_weights():
    weights = {"spatial_random": 1.0, "spatial_grid": 3.0}
    drawn = sample_patterns(weights, 4000, generator=make_generator(9))

    assert set(drawn) == {"spatial_random", "spatial_grid"}

    grid_share = drawn.count("spatial_grid") / len(drawn)
    assert 0.70 < grid_share < 0.80

    only_one = sample_patterns({"spatial_random": 1.0}, 50, generator=make_generator(10))
    assert set(only_one) == {"spatial_random"}


def test_default_weights_cover_every_pattern():
    assert set(DEFAULT_PATTERN_WEIGHTS) == set(MASK_PATTERNS)
    assert all(w > 0 for w in DEFAULT_PATTERN_WEIGHTS.values())


if __name__ == "__main__":
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]

    failures = 0

    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")

    print(f"\n{len(tests) - failures} passed, {failures} failed")
    sys.exit(1 if failures else 0)
