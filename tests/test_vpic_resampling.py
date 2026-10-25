"""Dataset-level checks for spatial alignment and legacy data identity."""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vpic_hdf5_dataset import VPICWindowDataset


def _write_dataset(directory, beta, fields):
    name = f"beta{beta}_nu0_Bz0_dt2_tau100"
    path = directory / f"VPIC_PPPL_CSV_Data_official2500_beta{beta}_none.h5"
    with h5py.File(path, "w") as handle:
        index = handle.create_group("index")
        index["run_names"] = np.asarray([name], dtype=h5py.string_dtype())
        for key, value in dict(beta=beta, nu=0, Bz0=0, dt=2, tau=100,
                               T=fields.shape[0], Nx=fields.shape[2],
                               Nz=fields.shape[3]).items():
            index[key] = [value]
        group = handle.create_group(f"runs/{name}")
        group["fields"] = fields
        group["frame_ids"] = np.arange(fields.shape[0])
    return path


@pytest.mark.parametrize("layout", ["C T X Z", "C X Z T"])
def test_cell_center_alignment_and_no_time_or_channel_mixing(tmp_path, layout):
    x = (np.arange(308, dtype=np.float32) + 0.5) / 308
    z = (np.arange(124, dtype=np.float32) + 0.5) / 124
    plane = 2 * x[:, None] + 3 * z[None, :]
    fields = np.stack([np.stack([plane + 10*c + 100*t for c in range(4)])
                       for t in range(3)])
    _write_dataset(tmp_path, 0.1, fields)
    dataset = VPICWindowDataset(tmp_path, delta_t=3, layout=layout)
    actual = dataset[0]["block"]
    if layout == "C X Z T":
        actual = actual.permute(0, 3, 1, 2)
    assert actual.shape == (4, 3, 154, 62)
    x_target = (torch.arange(154) + 0.5) / 154
    z_target = (torch.arange(62) + 0.5) / 62
    expected = 2*x_target[:, None] + 3*z_target[None, :]
    # Antialias filtering changes boundary samples; interior affine coordinates
    # must retain their cell-centered physical locations in every frame/channel.
    for channel in range(4):
        for frame in range(3):
            torch.testing.assert_close(
                actual[channel, frame, 3:-3, 3:-3],
                (expected + 10*channel + 100*frame)[3:-3, 3:-3],
                rtol=1e-6, atol=5e-5,
            )
    dataset.close()


@pytest.mark.parametrize("layout", ["C T X Z", "C X Z T"])
def test_reference_grid_is_exact_and_skips_interpolation(tmp_path, monkeypatch, layout):
    fields = np.random.default_rng(0).normal(size=(3, 4, 154, 62)).astype(np.float32)
    path = _write_dataset(tmp_path, 0.2, fields)
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Reference-grid data must not be interpolated")

    monkeypatch.setattr("data.vpic_hdf5_dataset.F.interpolate", forbidden)
    dataset = VPICWindowDataset(tmp_path, delta_t=3, layout=layout)
    expected = np.transpose(fields, (1, 0, 2, 3) if layout == "C T X Z" else (1, 2, 3, 0))
    assert torch.equal(dataset[0]["block"], torch.from_numpy(expected))
    dataset.close()
    assert path.read_bytes() == before


def test_bicubic_overshoot_is_not_clamped(tmp_path):
    fields = np.zeros((2, 4, 308, 124), dtype=np.float32)
    fields[:, :, 154:, :] = 1.0
    _write_dataset(tmp_path, 0.1, fields)
    dataset = VPICWindowDataset(tmp_path, delta_t=2, return_metadata=False)
    block = dataset[0]
    assert block.min() < 0.0 and block.max() > 1.0
    assert torch.isfinite(block).all()
    dataset.close()


@pytest.mark.parametrize("beta,shape", [(0.01, (308, 124)), (0.2, (154, 62))])
def test_full_run_sliding_loader_matches_window_preprocessing(tmp_path, monkeypatch, beta, shape):
    from visualize_sliding_density_reconstruction import find_and_load_run

    fields = np.random.default_rng(12).normal(size=(3, 4, *shape)).astype(np.float32)
    path = _write_dataset(tmp_path, beta, fields)
    run_name = f"beta{beta}_nu0_Bz0_dt2_tau100"
    frame_ids = np.array([10, 25, 40], dtype=np.int64)
    with h5py.File(path, "r+") as handle:
        handle[f"runs/{run_name}/frame_ids"][:] = frame_ids
    if shape == (154, 62):
        def forbidden(*args, **kwargs):
            raise AssertionError("Reference-grid full runs must not be interpolated")
        monkeypatch.setattr("torch.nn.functional.interpolate", forbidden)
    dataset = VPICWindowDataset(tmp_path, betas=[beta], delta_t=3)
    expected = dataset[0]["block"].permute(1, 0, 2, 3).numpy()
    actual, actual_ids, actual_path = find_and_load_run(tmp_path, [beta], run_name)
    assert actual.shape == (3, 4, 154, 62) and actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_ids, frame_ids)
    assert actual_path == path
    if shape == (154, 62):
        np.testing.assert_array_equal(actual, fields)
    dataset.close()
