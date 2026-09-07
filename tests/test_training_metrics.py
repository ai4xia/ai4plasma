from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.masking import DEFAULT_PATTERN_WEIGHTS, MASK_PATTERNS  # noqa: E402
from train_masked_unet3d import (  # noqa: E402
    build_epoch_wandb_log,
    train_one_epoch,
    validate,
)


class TinyBlockDataset(Dataset):
    def __init__(self, n: int = 2):
        self.blocks = [torch.randn(4, 2, 4, 4) for _ in range(n)]

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        return {"block": self.blocks[index]}


class ScaleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, model_input):
        return model_input[:, :4] * self.scale


def _channel_stats(device: torch.device):
    mean = torch.zeros((1, 4, 1, 1, 1), device=device)
    std = torch.ones((1, 4, 1, 1, 1), device=device)
    return mean, std


def test_train_one_epoch_backprop_and_mse_only_metrics():
    device = torch.device("cpu")
    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = DataLoader(TinyBlockDataset(), batch_size=1)
    mean, std = _channel_stats(device)
    scale_before = float(model.scale.detach())

    metrics = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        mean=mean,
        std=std,
        pattern_weights=DEFAULT_PATTERN_WEIGHTS,
        density_visible_p_zero=0.05,
        density_visible_p_full=0.05,
        magnetic_visible_p_zero=0.05,
        magnetic_visible_p_full=0.05,
        amp=False,
        grad_clip=1.0,
    )

    assert "mse" in metrics
    assert "mae" not in metrics
    assert metrics["mse"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["mse"]))
    assert float(model.scale.detach()) != scale_before


def test_validate_returns_mse_only_and_mean_is_average():
    device = torch.device("cpu")
    model = ScaleModel()
    loader = DataLoader(TinyBlockDataset(), batch_size=1)
    mean, std = _channel_stats(device)
    patterns = list(MASK_PATTERNS)

    metrics = validate(
        model=model,
        loader=loader,
        device=device,
        mean=mean,
        std=std,
        patterns=patterns,
        mask_seed=4321,
        density_probe_min=0,
        density_probe_max=2,
        amp=False,
    )

    assert set(metrics) == set(patterns) | {"mean"}
    for name, values in metrics.items():
        assert list(values) == ["mse"]
        assert values["mse"] >= 0.0
    expected_mean = sum(metrics[pattern]["mse"] for pattern in patterns) / len(patterns)
    assert abs(metrics["mean"]["mse"] - expected_mean) < 1e-12


def test_wandb_epoch_log_has_no_mae_and_keeps_mse_and_sampling_stats():
    train_metrics = {
        "magnetic_pattern_share": {name: 0.2 for name in MASK_PATTERNS},
        "density_pattern_share": {name: 0.2 for name in MASK_PATTERNS},
    }
    val_metrics = {
        pattern: {"mse": 0.01 * (index + 1)}
        for index, pattern in enumerate(MASK_PATTERNS)
    }
    val_metrics["mean"] = {"mse": 0.03}
    row = {
        "epoch": 3,
        "learning_rate": 2e-4,
        "train_mse": 0.04,
        "val_mse": 0.03,
        "train_mask_target_fraction": 0.5,
        "train_mask_actual_fraction": 0.5,
        "train_density_probe_count": 10.0,
        "train_density_visible_fraction": 0.1,
        "train_density_zero_sample_share": 0.05,
        "train_density_full_sample_share": 0.05,
        "train_magnetic_visible_count": 20.0,
        "train_magnetic_visible_fraction": 0.2,
        "train_magnetic_zero_sample_share": 0.05,
        "train_magnetic_full_sample_share": 0.05,
    }

    log_data = build_epoch_wandb_log(
        row=row,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        val_patterns=list(MASK_PATTERNS),
    )

    mae_keys = [key for key in log_data if "mae" in key.lower()]
    assert mae_keys == []
    for key in (
        "train/mse",
        "val/mse",
        "val/mean/mse",
        "val/spatial_random/mse",
        "val/spatial_grid/mse",
        "val/spatial_block/mse",
        "val/temporal_random/mse",
        "val/temporal_block/mse",
        "optimization/learning_rate",
        "mask/density_probe_patterns/zero_sample_share",
        "mask/density_probe_patterns/full_sample_share",
        "mask/magnetic_probe_patterns/zero_sample_share",
        "mask/magnetic_probe_patterns/full_sample_share",
    ):
        assert key in log_data


def test_resume_history_filter_tolerates_old_mae_fields():
    history = [
        {"epoch": 1, "train_mse": 0.2, "train_mae": 0.1, "val_mse": 0.3, "val_mae": 0.2},
        {"epoch": 2, "train_mse": 0.1, "train_mae": 0.05, "val_mse": 0.2, "val_mae": 0.1},
    ]
    start_epoch = 2
    filtered = [row for row in history if row.get("epoch", 0) < start_epoch]
    assert [row["epoch"] for row in filtered] == [1]
    assert filtered[0]["val_mse"] == 0.3


def test_best_checkpoint_selection_uses_val_mse_only():
    best_val_mse = float("inf")
    chosen = None
    for epoch, val_mse, val_mae in ((1, 0.20, 0.01), (2, 0.25, 0.001), (3, 0.15, 0.9)):
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            chosen = epoch
    assert chosen == 3
    assert best_val_mse == 0.15
