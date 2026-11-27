#!/usr/bin/env python3
"""CPU-only validation, empirical comparison, and visualization for general_v12 masking."""
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data.masking import (
    DEFAULT_PATTERN_WEIGHTS,
    GENERAL_MASKING_VERSION_V12,
    LEGACY_MASKING_VERSION_V11,
    _diverse_probe_grid_plane,
    _sparse_probe_grid_plane,
    sample_endpoint_log_uniform_count,
    sample_endpoint_mixed_count,
)
from train_masked_unet3d import (
    parse_args,
    sample_training_mask,
    validate_resume_parameters,
)

OUT = Path("evaluations/masking_sampler_generalization_20261127")
OUT.mkdir(parents=True, exist_ok=True)


def run_targeted_tests() -> dict:
    results = {}
    T, X, Z = 24, 154, 62
    S = X * Z  # 9548

    # 1. Every valid probe count supported + 3. Interior mixture has sparse & high counts
    gen = torch.Generator("cpu").manual_seed(101)
    draws_mix = np.array([sample_endpoint_mixed_count(S, 0.05, 0.05, 0.5, gen) for _ in range(4000)])
    assert draws_mix.min() >= 0 and draws_mix.max() <= S
    assert np.any((draws_mix >= 1) & (draws_mix < 100))
    assert np.any((draws_mix > 4000) & (draws_mix < S))
    # Small n_max=5: verify every integer 0..5 is hit
    draws_small = {sample_endpoint_mixed_count(5, 0.05, 0.05, 0.5, gen) for _ in range(500)}
    assert draws_small == {0, 1, 2, 3, 4, 5}, f"Missing counts in small test: {draws_small}"
    results["test_1_3_all_counts_and_mixture"] = "PASS"

    # 2. Zero/full point masses retain configured probabilities (0.05 each)
    p0 = float(np.mean(draws_mix == 0))
    p1 = float(np.mean(draws_mix == S))
    assert abs(p0 - 0.05) < 0.015 and abs(p1 - 0.05) < 0.015, f"p0={p0}, p1={p1}"
    results["test_2_point_masses"] = f"PASS (p0={p0:.4f}, p1={p1:.4f})"

    # 4 & 7. Independent B/D spatial positions, no forced overlap or disjointness
    y = torch.zeros((4, 4, T, X, Z), dtype=torch.float32)
    torch.manual_seed(202)
    overlaps = []
    for _ in range(50):
        m, infos, mp, dp, *_ = sample_training_mask(
            y, DEFAULT_PATTERN_WEIGHTS, 0.05, 0.05, 0.05, 0.05,
            grid_match_prob=0.0, interior_uniform_weight=0.5,
            diverse_spatial_grid=True, reject_joint_empty=True,
        )
        for i in range(4):
            if mp[i] in ("spatial_random", "spatial_grid") and dp[i] in ("spatial_random", "spatial_grid"):
                mb = m[i, 0, 0] > 0.5
                md = m[i, 3, 0] > 0.5
                if mb.any() and md.any():
                    inter = float((mb & md).sum().item())
                    union = float((mb | md).sum().item())
                    overlaps.append(inter / union)
    assert any(o > 0.0 for o in overlaps) and any(o < 0.99 for o in overlaps)
    results["test_4_7_bd_independence"] = f"PASS (n={len(overlaps)}, jaccard mean={np.mean(overlaps):.3f})"

    # 5 & 6. Grid generator exact unique probe counts + diversity
    families = set()
    aspect_pairs = set()
    for cnt in [1, 7, 64, 500, 2387, 4774, 7161, 9547, 9548]:
        for _ in range(15):
            pl, info = _diverse_probe_grid_plane(1, 1, X, Z, cnt, gen)
            assert int(pl.sum().item()) == cnt, f"Expected {cnt}, got {int(pl.sum().item())}"
            assert torch.all((pl == 0.0) | (pl == 1.0))
            families.add(info["grid_family"])
            aspect_pairs.add((info["grid_count_x"], info["grid_count_z"]))
    assert {"aligned", "staggered", "jittered"}.issubset(families)
    assert len(aspect_pairs) > 20
    results["test_5_6_grid_exact_and_diverse"] = f"PASS (families={sorted(families)}, aspect_pairs={len(aspect_pairs)})"

    # 8 & 9. Single-modality valid, joint-empty absent when reject_joint_empty=True, bounded retries
    torch.manual_seed(303)
    saw_mag_only = saw_den_only = False
    for _ in range(250):
        m, *_ = sample_training_mask(
            y, DEFAULT_PATTERN_WEIGHTS, 0.25, 0.05, 0.25, 0.05,
            grid_match_prob=0.0, interior_uniform_weight=0.5,
            diverse_spatial_grid=True, reject_joint_empty=True,
        )
        for i in range(4):
            has_b = bool((m[i, :3] > 0.5).any().item())
            has_d = bool((m[i, 3:] > 0.5).any().item())
            assert has_b or has_d, "Joint-empty mask slipped through!"
            saw_mag_only = saw_mag_only or (has_b and not has_d)
            saw_den_only = saw_den_only or (has_d and not has_b)
    assert saw_mag_only and saw_den_only
    # Bounded retry raises RuntimeError when p_zero=1.0 and reject_joint_empty=True
    try:
        sample_training_mask(
            y, {"spatial_random": 1.0}, 1.0, 0.0, 1.0, 0.0,
            reject_joint_empty=True, _max_empty_retries=4,
        )
        raise AssertionError("Expected RuntimeError when all retries yield joint-empty masks!")
    except RuntimeError as exc:
        assert "nonempty" in str(exc).lower() and "_max_empty_retries=4" in str(exc)
    # Legacy reject_joint_empty=False remains unchanged
    m_leg_empty, *_ = sample_training_mask(
        y, {"spatial_random": 1.0}, 1.0, 0.0, 1.0, 0.0,
        reject_joint_empty=False,
    )
    assert m_leg_empty.shape == y.shape and float(m_leg_empty.sum().item()) == 0.0
    results["test_8_9_single_modality_and_nonempty"] = (
        "PASS (mag_only=True, den_only=True, joint_empty=0, exhausted_retry_raises=PASS, legacy_allow_empty=PASS)"
    )

    # 10 & 11. Legacy reproducibility + masking_version & resume compatibility
    a_leg = parse_args(["--masking-sampler", "legacy_v11", "--grid-match-prob", "0.15"])
    a_new = parse_args(["--masking-sampler", "general_v12", "--grid-match-prob", "0.0"])
    assert a_leg.masking_version == LEGACY_MASKING_VERSION_V11
    assert a_new.masking_version == GENERAL_MASKING_VERSION_V12
    assert a_leg.interior_uniform_weight == 0.0 and not a_leg.diverse_spatial_grid and not a_leg.reject_joint_empty
    assert a_new.interior_uniform_weight == 0.5 and a_new.diverse_spatial_grid and a_new.reject_joint_empty

    # Verify resume compatibility rejects switching v11 <-> v12
    try:
        validate_resume_parameters(a_new, vars(a_leg))
        raise AssertionError("Should have rejected resuming legacy_v11 with general_v12!")
    except ValueError:
        pass
    # Verify same-version resume succeeds
    validate_resume_parameters(a_new, vars(a_new))
    validate_resume_parameters(a_leg, vars(a_leg))
    results["test_10_11_legacy_and_resume_compat"] = "PASS"
    return results


def benchmark_sampler(name: str, n_batches: int, batch_size: int, **kwargs) -> tuple:
    T, X, Z = 24, 154, 62
    S = X * Z
    y = torch.zeros((batch_size, 4, T, X, Z), dtype=torch.float32)
    torch.manual_seed(42)
    rows = []
    t0 = time.perf_counter()
    for _ in range(n_batches):
        m, infos, mp, dp, mf, df, mc, dc = sample_training_mask(
            y, DEFAULT_PATTERN_WEIGHTS, 0.05, 0.05, 0.05, 0.05, **kwargs
        )
        for i in range(batch_size):
            b_vis = float((m[i, 0] > 0.5).float().mean().item())
            d_vis = float((m[i, 3] > 0.5).float().mean().item())
            has_b = bool((m[i, :3] > 0.5).any().item())
            has_d = bool((m[i, 3:] > 0.5).any().item())
            mb = m[i, 0, 0] > 0.5
            md = m[i, 3, 0] > 0.5
            is_spatial_pair = mp[i] in ("spatial_random", "spatial_grid", "spatial_block")
            jacc = float((mb & md).sum().item() / max(1, int((mb | md).sum().item()))) if is_spatial_pair else np.nan
            rows.append({
                "sampler": name,
                "m_pat": mp[i],
                "d_pat": dp[i],
                "b_vis": b_vis,
                "d_vis": d_vis,
                "mag_only": int(has_b and not has_d),
                "den_only": int(has_d and not has_b),
                "joint_empty": int(not has_b and not has_d),
                "spatial_jaccard": jacc,
                "m_grid_family": infos[i].get("magnetic_grid_family", "orig"),
                "m_gx": infos[i].get("magnetic_grid_count_x", np.nan),
                "m_gz": infos[i].get("magnetic_grid_count_z", np.nan),
            })
    dt = time.perf_counter() - t0
    return pd.DataFrame(rows), dt / n_batches * 1000.0


def make_visualization():
    T, X, Z = 24, 154, 62
    y = torch.zeros((1, 4, T, X, Z), dtype=torch.float32)
    torch.manual_seed(777)
    examples = {}
    targets = {
        "sparse": lambda m, mp, dp, bv, dv: mp[0] in ("spatial_random", "spatial_grid") and 0 < bv < 0.05 and 0 < dv < 0.05,
        "intermediate": lambda m, mp, dp, bv, dv: mp[0] == "spatial_grid" and 0.25 <= bv <= 0.65 and 0.25 <= dv <= 0.65,
        "dense": lambda m, mp, dp, bv, dv: mp[0] in ("spatial_random", "spatial_grid") and bv > 0.75 and dv > 0.75,
        "staggered_grid": lambda m, mp, dp, bv, dv: mp[0] == "spatial_grid" and 0.15 < bv < 0.7 and dp[0] == "spatial_grid",
        "spatial_random": lambda m, mp, dp, bv, dv: mp[0] == "spatial_random" and dp[0] == "spatial_random" and 0.1 < bv < 0.8,
        "single_modality": lambda m, mp, dp, bv, dv: (bv == 0.0 and dv > 0.05) or (dv == 0.0 and bv > 0.05),
    }
    for _ in range(800):
        m, infos, mp, dp, *_ = sample_training_mask(
            y, DEFAULT_PATTERN_WEIGHTS, 0.05, 0.05, 0.05, 0.05,
            grid_match_prob=0.0, interior_uniform_weight=0.5,
            diverse_spatial_grid=True, reject_joint_empty=True,
        )
        bv = float((m[0, 0] > 0.5).float().mean().item())
        dv = float((m[0, 3] > 0.5).float().mean().item())
        for k, fn in targets.items():
            if k not in examples and fn(m, mp, dp, bv, dv):
                if k == "staggered_grid" and infos[0].get("magnetic_grid_family") != "staggered":
                    continue
                examples[k] = (
                    m[0, 0, 0].numpy(),
                    m[0, 3, 0].numpy(),
                    f"{k}\nB:{mp[0]} ({bv*100:.1f}%) | D:{dp[0]} ({dv*100:.1f}%)",
                )
        if len(examples) == len(targets):
            break

    fig, axes = plt.subplots(len(examples), 2, figsize=(9, 2.3 * len(examples)), dpi=150)
    for row_i, (k, (bm, dm, title)) in enumerate(examples.items()):
        axes[row_i, 0].imshow(bm.T, origin="lower", aspect="auto", cmap="Greys_r", vmin=0, vmax=1)
        axes[row_i, 0].set_title(f"Magnetic — {title}", fontsize=8)
        axes[row_i, 1].imshow(dm.T, origin="lower", aspect="auto", cmap="Greys_r", vmin=0, vmax=1)
        axes[row_i, 1].set_title(f"Density — {title}", fontsize=8)
        for c in range(2):
            axes[row_i, c].set_xticks([])
            axes[row_i, c].set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT / "sampling_comparison.png")
    plt.close(fig)
    print(f"Saved sampling_comparison.png with {len(examples)}/{len(targets)} categories")


def main():
    test_res = run_targeted_tests()
    print("=== TARGETED TESTS ===")
    print(json.dumps(test_res, indent=2))

    df_a, ms_a = benchmark_sampler(
        "A_existing_new256", 1250, 4,
        grid_match_prob=0.15, grid_match_mode="checkerboard_shared",
        interior_uniform_weight=0.0, diverse_spatial_grid=False, reject_joint_empty=False,
    )
    df_b, ms_b = benchmark_sampler(
        "B_improved_v12", 1250, 4,
        grid_match_prob=0.0,
        interior_uniform_weight=0.5, diverse_spatial_grid=True, reject_joint_empty=True,
    )
    df = pd.concat([df_a, df_b], ignore_index=True)

    summary_rows = []
    for name, ms in [("A_existing_new256", ms_a), ("B_improved_v12", ms_b)]:
        sub = df[df["sampler"] == name]
        probe_b = sub[sub["m_pat"].isin(["spatial_random", "spatial_grid"])]["b_vis"]
        probe_d = sub[sub["d_pat"].isin(["spatial_random", "spatial_grid"])]["d_vis"]
        grid_m = sub[sub["m_pat"] == "spatial_grid"]
        unique_aspects = int(grid_m[["m_gx", "m_gz"]].dropna().drop_duplicates().shape[0])
        row = {
            "sampler": name,
            "n_samples": len(sub),
            "ms_per_batch4": round(ms, 3),
            "b_probe_0_10pct": round(100.0 * float(((probe_b >= 0.0) & (probe_b < 0.10)).mean()), 2),
            "b_probe_10_25pct": round(100.0 * float(((probe_b >= 0.10) & (probe_b < 0.25)).mean()), 2),
            "b_probe_25_50pct": round(100.0 * float(((probe_b >= 0.25) & (probe_b < 0.50)).mean()), 2),
            "b_probe_50_75pct": round(100.0 * float(((probe_b >= 0.50) & (probe_b < 0.75)).mean()), 2),
            "b_probe_75_100pct": round(100.0 * float((probe_b >= 0.75).mean()), 2),
            "d_probe_0_10pct": round(100.0 * float(((probe_d >= 0.0) & (probe_d < 0.10)).mean()), 2),
            "d_probe_10_25pct": round(100.0 * float(((probe_d >= 0.10) & (probe_d < 0.25)).mean()), 2),
            "d_probe_25_50pct": round(100.0 * float(((probe_d >= 0.25) & (probe_d < 0.50)).mean()), 2),
            "d_probe_50_75pct": round(100.0 * float(((probe_d >= 0.50) & (probe_d < 0.75)).mean()), 2),
            "d_probe_75_100pct": round(100.0 * float((probe_d >= 0.75).mean()), 2),
            "b_zero_pct": round(100.0 * float((probe_b == 0.0).mean()), 2),
            "b_full_pct": round(100.0 * float((probe_b == 1.0).mean()), 2),
            "d_zero_pct": round(100.0 * float((probe_d == 0.0).mean()), 2),
            "d_full_pct": round(100.0 * float((probe_d == 1.0).mean()), 2),
            "mag_only_pct": round(100.0 * float(sub["mag_only"].mean()), 2),
            "den_only_pct": round(100.0 * float(sub["den_only"].mean()), 2),
            "joint_empty_pct": round(100.0 * float(sub["joint_empty"].mean()), 2),
            "spatial_jaccard_mean": round(float(sub["spatial_jaccard"].dropna().mean()), 4),
            "spatial_jaccard_exact1_pct": round(100.0 * float((sub["spatial_jaccard"].dropna() > 0.999).mean()), 2),
            "unique_grid_aspects": unique_aspects,
        }
        summary_rows.append(row)
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(OUT / "sampling_statistics.csv", index=False)
    print("\n=== SAMPLING STATISTICS (5000 SAMPLES EACH) ===")
    print(df_sum.T.to_string())
    print("\n=== GRID FAMILIES (B_improved_v12) ===")
    print(df_b[df_b["m_pat"] == "spatial_grid"]["m_grid_family"].value_counts().to_dict())

    make_visualization()


if __name__ == "__main__":
    main()
