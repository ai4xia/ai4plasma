# tests/test_online_causal_reconstruction.py

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from journal_statistics import pearson_spearman
from make_paper_figures import find_and_load_run, sliding_density_probe_mask
from run_online_causal_reconstruction import (
    build_causal_window_input,
    evaluate_run_online,
    load_model_and_stats,
    reconstruct_online_causal,
)


class MockCausalModel(torch.nn.Module):
    """Deterministic mock UNet3D returning (mean, variance)."""
    uncertainty_aware = True

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x is [B, 8, T, H, W]
        # output is mean [B, 4, T, H, W], variance [B, 4, T, H, W]
        vals = x[:, :4]
        in_var = x[:, 4:]
        mean = vals * 0.8
        var = in_var * 0.5 + 0.05
        return mean, var


def test_causal_window_input_padding_and_positions():
    """Verify window positioning and padding semantics:
    1. At t=0, slots 0..22 are negative global times: input value is Gaussian noise, u_in=1.0.
    2. Current frame t always occupies local index 23.
    3. At t=1, slots 0..21 are negative global times (u_in=1.0), slot 22 is frame 0, slot 23 is frame 1.
    4. At t=23, slots 0..23 correspond to frames 0..23 (no negative padding).
    """
    torch.manual_seed(42)
    T_total, H, W = 48, 8, 8
    target = torch.randn(4, T_total, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True  # 16 probes

    history_pred = torch.zeros(T_total, H, W)
    history_var = torch.ones(T_total, H, W)

    # --- Test t = 0 ---
    g0 = torch.Generator().manual_seed(123)
    model_input_0, _ = build_causal_window_input(
        t=0,
        target_normalized=target,
        probe_mask=probe_mask,
        history_pred=history_pred,
        history_var=history_var,
        hide_magnetic=False,
        window_size=24,
        generator=g0,
    )
    assert model_input_0.shape == (1, 8, 24, H, W)
    vals_0 = model_input_0[0, :4]
    var_0 = model_input_0[0, 4:]

    # Local indices 0..22 correspond to global tau = -23 .. -1: variance 1.0 across all 4 channels
    assert torch.all(var_0[:, :23] == 1.0), "Historical padded slots must have variance 1.0"
    # Local index 23 corresponds to global tau = 0 (current frame)
    # B components: variance 0.0
    assert torch.all(var_0[:3, 23] == 0.0), "B components at t=0 must have variance 0.0 under B_full"
    assert torch.allclose(vals_0[:3, 23], target[:3, 0]), "B values at t=0 must match target"
    # Density: probe locations variance 0.0, non-probe locations variance 1.0
    assert torch.all(var_0[3, 23, probe_mask] == 0.0), "Density probes must have variance 0.0"
    assert torch.all(var_0[3, 23, ~probe_mask] == 1.0), "Unobserved Density must have variance 1.0"
    assert torch.allclose(vals_0[3, 23, probe_mask], target[3, 0, probe_mask]), "Probe values must match"

    # --- Test t = 1 ---
    history_pred[0] = 3.14
    history_var[0] = 0.42
    g1 = torch.Generator().manual_seed(456)
    model_input_1, _ = build_causal_window_input(
        t=1,
        target_normalized=target,
        probe_mask=probe_mask,
        history_pred=history_pred,
        history_var=history_var,
        hide_magnetic=False,
        window_size=24,
        generator=g1,
    )
    var_1 = model_input_1[0, 4:]
    vals_1 = model_input_1[0, :4]

    # Local indices 0..21: global tau < 0 -> variance 1.0
    assert torch.all(var_1[:, :22] == 1.0)
    # Local index 22: global tau = 0
    # Density at non-probe should reuse history_pred[0] and history_var[0]
    assert torch.allclose(vals_1[3, 22, ~probe_mask], torch.tensor(3.14))
    assert torch.allclose(var_1[3, 22, ~probe_mask], torch.tensor(0.42))
    # Density at probe should be exact target[3, 0] with variance 0.0
    assert torch.allclose(vals_1[3, 22, probe_mask], target[3, 0, probe_mask])
    assert torch.all(var_1[3, 22, probe_mask] == 0.0)
    # Local index 23: global tau = 1 (current frame)
    assert torch.all(var_1[3, 23, ~probe_mask] == 1.0)
    assert torch.all(var_1[3, 23, probe_mask] == 0.0)

    # --- Test t = 23 ---
    model_input_23, _ = build_causal_window_input(
        t=23,
        target_normalized=target,
        probe_mask=probe_mask,
        history_pred=history_pred,
        history_var=history_var,
        hide_magnetic=False,
        window_size=24,
    )
    var_23 = model_input_23[0, 4:]
    assert torch.all(var_23[:3, :] == 0.0), "All B components in valid frames must have variance 0.0"


def test_b_hidden_semantics():
    """Verify B_hidden semantics:
    All B components (channels 0..2) have input variance = 1.0 across all frames,
    and their values are Gaussian noise.
    """
    T_total, H, W = 48, 8, 8
    target = torch.randn(4, T_total, H, W)
    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    history_pred = torch.zeros(T_total, H, W)
    history_var = torch.ones(T_total, H, W)

    model_input, _ = build_causal_window_input(
        t=10,
        target_normalized=target,
        probe_mask=probe_mask,
        history_pred=history_pred,
        history_var=history_var,
        hide_magnetic=True,
        window_size=24,
    )
    var = model_input[0, 4:]
    assert torch.all(var[:3] == 1.0), "Under B_hidden, B input variance must be 1.0 everywhere"


def test_strict_causality_invariance_to_future():
    """STRICT CAUSALITY TEST:
    Modifying future ground-truth target frames (tau > t) while keeping the
    available history and random noise fixed must NOT change the prediction
    at global time t.
    """
    torch.manual_seed(101)
    T_total, H, W = 48, 8, 8
    target_A = torch.randn(4, T_total, H, W)
    target_B = target_A.clone()
    target_B[:, 6:] = torch.randn_like(target_B[:, 6:]) * 100.0 + 50.0

    probe_mask = torch.zeros(H, W, dtype=torch.bool)
    probe_mask[::2, ::2] = True
    model = MockCausalModel()

    pred_A, var_A = reconstruct_online_causal(
        model=model,
        target_normalized=target_A,
        probe_mask=probe_mask,
        n_frames=6,
        window_size=24,
        hide_magnetic=False,
        seed=42,
        amp=False,
    )
    pred_B, var_B = reconstruct_online_causal(
        model=model,
        target_normalized=target_B,
        probe_mask=probe_mask,
        n_frames=6,
        window_size=24,
        hide_magnetic=False,
        seed=42,
        amp=False,
    )

    max_diff_pred = np.max(np.abs(pred_A - pred_B))
    max_diff_var = np.max(np.abs(var_A - var_B))
    assert max_diff_pred == 0.0, f"Future leakage detected! pred diff = {max_diff_pred}"
    assert max_diff_var == 0.0, f"Future leakage detected! var diff = {max_diff_var}"


def test_hdf5_loading_and_probe_mask_helpers():
    """Verify real HDF5 loading return shape and probe mask helper."""
    h5_dir = Path("/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta_official2500_none_compat")
    run_name = "beta0.2_nu10_Bz0.15_dt2_tau200"
    if not h5_dir.exists():
        pytest.skip(f"HDF5 directory {h5_dir} not found on this machine")

    fields, frame_ids, h5_path = find_and_load_run(h5_dir, [0.2], run_name)
    assert isinstance(fields, np.ndarray)
    assert fields.ndim == 4
    # Fields is (T, 4, X, Z) where (X, Z) = (154, 62)
    assert fields.shape[1] == 4
    assert fields.shape[2:] == (154, 62)

    # Transpose to (4, T, X, Z)
    fields_48 = fields[:48]
    target = torch.from_numpy(np.transpose(fields_48, (1, 0, 2, 3))).to(dtype=torch.float32)
    assert target.shape == (4, 48, 154, 62)

    # Sliding probe mask helper
    target_norm = target.unsqueeze(0)  # [1, 4, 48, 154, 62]
    probe_mask, info = sliding_density_probe_mask(target_norm, probe_count=1000)
    assert probe_mask.shape == (154, 62)
    # Count of visible probes in mask should match requested count
    assert int(probe_mask.sum().item()) == 1000


def test_correlation_and_metrics_calculation():
    """Verify pearson_spearman dict return extraction and NRMSE formulas."""
    u = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    e2 = np.array([0.12, 0.19, 0.31, 0.38, 0.52])
    corr_dict = pearson_spearman(u, e2)
    assert isinstance(corr_dict, dict)
    assert "pearson" in corr_dict and "spearman" in corr_dict
    assert isinstance(corr_dict["pearson"], float)
    assert isinstance(corr_dict["spearman"], float)
    assert corr_dict["pearson"] > 0.95

    # Test full-field vs hidden voxel NRMSE separation
    pred = np.zeros((48, 10, 10))
    tgt = np.ones((48, 10, 10))
    probe_mask = np.zeros((10, 10), dtype=bool)
    probe_mask[:2, :2] = True  # 4 probes
    # If probes are clamped to target:
    pred[:, probe_mask] = 1.0

    diff = pred - tgt
    full_nrmse = float(np.sqrt(np.mean(diff**2)))
    hidden_mask = ~probe_mask
    hidden_nrmse = float(np.sqrt(np.mean(diff[:, hidden_mask]**2)))

    # Hidden error should be 1.0, full error should be sqrt(96/100) < 1.0
    assert np.isclose(hidden_nrmse, 1.0)
    assert full_nrmse < hidden_nrmse


def test_offline_csv_filtering_and_pairing():
    """Verify offline ablation summary filtering (step=24, variant='A') and inner join."""
    offline_summary_path = Path("evaluations/physworld_v11_sliding_overwrite_ablation_20261123/data/ablation_summary.csv")
    if not offline_summary_path.exists():
        pytest.skip("Offline summary CSV not found")

    df_offline = pd.read_csv(offline_summary_path)
    # Filter step==24 and variant=='A'
    filtered = df_offline[(df_offline["step"] == 24) & (df_offline["variant"] == "A")]
    assert len(filtered) == 20  # 10 runs x 2 conditions (B_full and B_hidden)

    # Mock online summary
    mock_online = pd.DataFrame({
        "run_name": filtered["run_name"].unique(),
        "b_condition": "B_full",
        "mean_e2_hidden": 0.005,
    })
    # Inner join by run_name
    merged = pd.merge(mock_online, filtered[filtered["b_condition"] == "B_full"], on="run_name")
    assert len(merged) == 10
    assert "mean_e2_hidden_x" in merged.columns and "mean_e2_hidden_y" in merged.columns


def test_strict_v11_checkpoint_loading():
    """Verify strict loading of v11 final.pt on CPU."""
    run_dir = Path("runs/multi_beta_gkl_v11_fromscratch_256ep_20261122")
    if not (run_dir / "final.pt").exists():
        pytest.skip("Checkpoint final.pt not found")

    model, mean, std, args = load_model_and_stats(run_dir, "final.pt", torch.device("cpu"))
    assert getattr(model, "uncertainty_aware", False) is True
    assert getattr(model, "spatial_only_pooling", False) is True
    assert getattr(model, "architecture", None) == "direct_unet3d_no_activation_norm_v3"
