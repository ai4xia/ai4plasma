# tests/test_temporal_position_ablation.py

from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pytest
import torch

from run_temporal_position_ablation import (
    build_controlled_window_input,
    evaluate_position_configuration,
    get_window_configurations,
    load_model_and_stats,
)


class MockPositionModel(torch.nn.Module):
    """Deterministic mock UNet3D returning (mean, variance)."""
    uncertainty_aware = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is [B, 8, T, H, W]
        # output is mean [B, 4, T, H, W], variance [B, 4, T, H, W]
        vals = x[:, :4]
        in_var = x[:, 4:]
        mean = vals * 0.9
        var = in_var * 0.4 + 0.02
        return mean, var


def test_window_configurations_target_mapping():
    """Verify that start + local_index exactly equals the target frame."""
    # Test target_frame = 24
    cfgs_24 = get_window_configurations(target_frame=24)
    for k, cfg in cfgs_24.items():
        assert cfg["start"] + cfg["local_index"] == 24
        assert cfg["end"] - cfg["start"] == 24
        assert 0 <= cfg["start"] and cfg["end"] <= 48

    # Check specific indices for target 24
    assert cfgs_24["left_edge"]["local_index"] == 0
    assert cfgs_24["left_edge"]["start"] == 24 and cfgs_24["left_edge"]["end"] == 48

    assert cfgs_24["interior"]["local_index"] == 11
    assert cfgs_24["interior"]["start"] == 13 and cfgs_24["interior"]["end"] == 37

    assert cfgs_24["right_edge"]["local_index"] == 23
    assert cfgs_24["right_edge"]["start"] == 1 and cfgs_24["right_edge"]["end"] == 25

    # Test target_frame = 23
    cfgs_23 = get_window_configurations(target_frame=23)
    for k, cfg in cfgs_23.items():
        assert cfg["start"] + cfg["local_index"] == 23
        assert cfg["end"] - cfg["start"] == 24
        assert 0 <= cfg["start"] and cfg["end"] <= 48

    assert cfgs_23["left_edge"]["local_index"] == 0
    assert cfgs_23["left_edge"]["start"] == 23 and cfgs_23["left_edge"]["end"] == 47

    assert cfgs_23["interior"]["local_index"] == 11
    assert cfgs_23["interior"]["start"] == 12 and cfgs_23["interior"]["end"] == 36

    assert cfgs_23["right_edge"]["local_index"] == 23
    assert cfgs_23["right_edge"]["start"] == 0 and cfgs_23["right_edge"]["end"] == 24


def test_global_noise_and_target_consistency_across_windows():
    """Verify that matching global frames receive the exact same noise

    and genuine observations regardless of their window position.
    """
    torch.manual_seed(42)
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True  # 64 probes

    # Global noise tensor
    global_noise = torch.randn(4, n_frames, H, W)

    cfgs = get_window_configurations(target_frame=24)

    # Build inputs for all three configurations under B_full
    inp_left = build_controlled_window_input(
        target_norm, probe_mask, global_noise,
        start=cfgs["left_edge"]["start"], end=cfgs["left_edge"]["end"], hide_magnetic=False
    )
    inp_interior = build_controlled_window_input(
        target_norm, probe_mask, global_noise,
        start=cfgs["interior"]["start"], end=cfgs["interior"]["end"], hide_magnetic=False
    )
    inp_right = build_controlled_window_input(
        target_norm, probe_mask, global_noise,
        start=cfgs["right_edge"]["start"], end=cfgs["right_edge"]["end"], hide_magnetic=False
    )

    # 1. Target frame F24:
    # In Left edge: local index 0
    # In Interior: local index 11
    # In Right edge: local index 23
    f24_left_vals = inp_left[0, :4, 0]
    f24_interior_vals = inp_interior[0, :4, 11]
    f24_right_vals = inp_right[0, :4, 23]

    f24_left_var = inp_left[0, 4:, 0]
    f24_interior_var = inp_interior[0, 4:, 11]
    f24_right_var = inp_right[0, 4:, 23]

    # Values and variances for F24 must be IDENTICAL across all three inputs
    diff_vals_li = torch.max(torch.abs(f24_left_vals - f24_interior_vals)).item()
    diff_vals_lr = torch.max(torch.abs(f24_left_vals - f24_right_vals)).item()
    assert diff_vals_li == 0.0, f"Target F24 values differ between left and interior: {diff_vals_li}"
    assert diff_vals_lr == 0.0, f"Target F24 values differ between left and right: {diff_vals_lr}"

    diff_var_li = torch.max(torch.abs(f24_left_var - f24_interior_var)).item()
    diff_var_lr = torch.max(torch.abs(f24_left_var - f24_right_var)).item()
    assert diff_var_li == 0.0, f"Target F24 variance differs between left and interior: {diff_var_li}"
    assert diff_var_lr == 0.0, f"Target F24 variance differs between left and right: {diff_var_lr}"

    # 2. Check an overlapping non-target frame: e.g. global frame F20
    # In Left edge (start=24): F20 is NOT in window
    # In Interior (start=13): F20 is at local index 20 - 13 = 7
    # In Right edge (start=1): F20 is at local index 20 - 1 = 19
    f20_interior_vals = inp_interior[0, :4, 7]
    f20_right_vals = inp_right[0, :4, 19]
    diff_f20 = torch.max(torch.abs(f20_interior_vals - f20_right_vals)).item()
    assert diff_f20 == 0.0, f"Overlapping frame F20 values differ across windows: {diff_f20}"


def test_b_hidden_controlled_noise():
    """Verify that under B_hidden, magnetic values use the exact global_noise."""
    n_frames, H, W = 48, 8, 8
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    global_noise = torch.randn(4, n_frames, H, W)

    inp = build_controlled_window_input(
        target_norm, probe_mask, global_noise,
        start=13, end=37, hide_magnetic=True
    )
    # Channels 0..2 should equal global_noise[:3, 13:37]
    diff_b = torch.max(torch.abs(inp[0, :3] - global_noise[:3, 13:37])).item()
    assert diff_b == 0.0, f"B_hidden values must equal sliced global noise: {diff_b}"
    # Magnetic input variance should be 1.0 everywhere
    assert torch.all(inp[0, 4:7] == 1.0)


def test_mock_evaluation_run():
    """Verify metric computation and shape consistency with MockPositionModel."""
    n_frames, H, W = 48, 16, 16
    target_norm = torch.randn(4, n_frames, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    global_noise = torch.randn(4, n_frames, H, W)

    mean = torch.zeros(4, 1, 1, 1)
    std = torch.ones(4, 1, 1, 1)

    model = MockPositionModel()
    cfgs = get_window_configurations(target_frame=24)

    for cfg_key, cfg in cfgs.items():
        res = evaluate_position_configuration(
            model=model,
            mean=mean,
            std=std,
            target_normalized=target_norm,
            probe_mask=probe_mask,
            global_noise=global_noise,
            target_frame=24,
            config=cfg,
            hide_magnetic=False,
            amp=False,
        )
        assert res["local_index"] == cfg["local_index"]
        assert res["density_mse_hidden"] > 0.0
        assert math.isclose(res["density_rmse_hidden"], math.sqrt(res["density_mse_hidden"]), rel_tol=1e-5)
        assert res["mean_var_hidden"] > 0.0
        # Pearson/Spearman are valid floats or NaN when variance is completely constant
        assert isinstance(res["pearson_r"], float)
        assert isinstance(res["spearman_r"], float)
        if math.isfinite(res["pearson_r"]):
            assert -1.0 <= res["pearson_r"] <= 1.0
        if math.isfinite(res["spearman_r"]):
            assert -1.0 <= res["spearman_r"] <= 1.0


def test_strict_v11_checkpoint_loading_cpu():
    """Verify strict loading of v11 final.pt on CPU."""
    run_dir = Path("runs/multi_beta_gkl_v11_fromscratch_256ep_20261122")
    if not (run_dir / "final.pt").exists():
        pytest.skip("Checkpoint final.pt not found")

    model, mean, std, args = load_model_and_stats(run_dir, "final.pt", torch.device("cpu"))
    assert getattr(model, "uncertainty_aware", False) is True
    assert getattr(model, "spatial_only_pooling", False) is True
    assert getattr(model, "architecture", None) == "direct_unet3d_no_activation_norm_v3"
