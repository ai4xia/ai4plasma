from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vpic_hdf5_dataset import (
    CANONICAL_TEST_RUN,
    FIXED_SPLIT_COUNTS,
    FIXED_SPLIT_PATH,
    KNOWN_SHORT_TRAIN_RUNS,
    fixed_split_sha256,
    h5_run_lengths,
    load_fixed_split,
    validate_final_checkpoint,
)
from train_masked_unet3d import save_checkpoint, split_by_run


def test_authoritative_split_counts_roles_and_canonical_case():
    split = load_fixed_split()
    roles = {key: set(split[key]) for key in FIXED_SPLIT_COUNTS}
    assert {key: len(value) for key, value in roles.items()} == FIXED_SPLIT_COUNTS
    assert not (roles["train_runs"] & roles["val_runs"])
    assert not (roles["train_runs"] & roles["test_runs"])
    assert not (roles["val_runs"] & roles["test_runs"])
    assert CANONICAL_TEST_RUN in roles["test_runs"]
    assert KNOWN_SHORT_TRAIN_RUNS <= roles["train_runs"]
    assert len(fixed_split_sha256()) == 64


def test_hdf5_coverage_and_usable_window_counts():
    h5_dir = Path("/pscratch/sd/b/binxia/VPIC_PPPL_HDF5_by_beta_official2500_none_compat")
    if not h5_dir.is_dir():
        pytest.skip("VPIC dataset is unavailable")
    split = load_fixed_split()
    lengths = h5_run_lengths(h5_dir)
    assert set(lengths) == set().union(*(set(split[key]) for key in FIXED_SPLIT_COUNTS))
    expected = {
        "train_runs": (2248, 68463),
        "val_runs": (125, 3821),
        "test_runs": (125, 3819),
    }
    for role, (usable_count, window_count) in expected.items():
        usable = [name for name in split[role] if lengths[name] >= 24]
        assert len(usable) == usable_count
        assert sum((lengths[name] - 24) // 2 + 1 for name in usable) == window_count


def test_training_index_excludes_test_windows():
    split = load_fixed_split()
    train_name = split["train_runs"][0]
    val_name = split["val_runs"][0]
    test_name = split["test_runs"][0]
    dataset = SimpleNamespace(samples=[
        (0, train_name, 0), (0, val_name, 0), (0, test_name, 0),
    ])
    train_idx, val_idx, _, _ = split_by_run(dataset, split)
    assert train_idx == [0]
    assert val_idx == [1]
    assert 2 not in train_idx + val_idx


def test_checkpoint_hash_and_predetermined_final_epoch(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    args = SimpleNamespace(split_sha256=fixed_split_sha256(), final_epoch=7)
    latest = tmp_path / "latest.pt"
    save_checkpoint(latest, model, optimizer, 6, args, {"mean": [], "std": []})
    checkpoint = torch.load(latest, weights_only=False)
    assert checkpoint["split_path"] == str(FIXED_SPLIT_PATH)
    assert checkpoint["split_sha256"] == fixed_split_sha256()
    with pytest.raises(ValueError, match="predetermined final checkpoint"):
        validate_final_checkpoint(checkpoint)
    checkpoint["epoch"] = 7
    validate_final_checkpoint(checkpoint)
    checkpoint["split_sha256"] = "changed"
    with pytest.raises(ValueError, match="split hash"):
        validate_final_checkpoint(checkpoint)


def test_split_json_is_not_generated_in_run_directories():
    source = Path("train_masked_unet3d.py").read_text()
    assert 'out_dir / "split.json"' not in source
    assert 'out_dir / "best.pt"' not in source
    assert 'out_dir / "final.pt"' in source
