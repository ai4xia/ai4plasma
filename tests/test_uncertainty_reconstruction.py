"""Small CPU checks for variance regression and recursive conditioning."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.masking import full_mse_loss, make_uncertain_input, predicted_error_mse_loss
from models.unet3d import UNet3D
from train_masked_unet3d import (
    RESUME_PARAMETER_KEYS, parse_args, train_one_epoch, validate,
    validate_resume_parameters,
)
from visualize_sliding_density_reconstruction import (
    build_bidirectional_rows, reconstruct_with_slide_step, refine_reconstruction_pass,
)


def test_continuous_corruption_and_endpoints():
    truth = torch.tensor([2.0, 4.0, 8.0])
    variance = torch.tensor([0.0, 0.25, 1.0])
    noise = torch.tensor([-3.0, -2.0, -1.0])
    actual = make_uncertain_input(truth, variance, noise)
    expected = torch.tensor([2.0, 4.0 * 0.75**0.5 - 1.0, -1.0])
    torch.testing.assert_close(actual, expected)
    # Unknown truth cannot leak into the noise-filled input.
    torch.testing.assert_close(
        make_uncertain_input(truth, torch.ones(3), noise), noise
    )


def test_error_regression_detaches_prediction_and_does_not_clip_target():
    prediction = torch.tensor([3.0, 2.0, 1.0, 1.0], requires_grad=True)
    variance = torch.tensor([1.0, 0.5, 0.75, 1.5], requires_grad=True)
    target = torch.zeros(4)
    loss = predicted_error_mse_loss(variance, prediction, target)
    expected = F.smooth_l1_loss(variance, (prediction.detach() - target).square(), beta=1.0, reduction="mean")
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(loss, torch.tensor((7.5 + 3.0 + 0.03125 + 0.125) / 4))
    loss.backward()
    assert prediction.grad is None
    torch.testing.assert_close(variance.grad, torch.tensor([-0.25, -0.25, -0.0625, 0.125]))


def test_reconstruction_keeps_balanced_mse():
    prediction = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 3 + [[2.0, 4.0, 6.0, 8.0]]).view(1, 4, 1, 1, 4)
    mask = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3 + [[1.0, 1.0, 1.0, 0.0]]).view_as(prediction)
    # Equal B/Density weighting, each balanced across visible/hidden regions.
    torch.testing.assert_close(full_mse_loss(prediction, torch.zeros_like(prediction), mask), torch.tensor(70.0 / 3))


def test_unet_head_and_checkpoint_roundtrip(tmp_path):
    options = dict(base_channels=2, channel_mults=(1, 2, 4))
    deterministic = UNet3D(**options).eval()
    extended = UNet3D(**options, uncertainty_aware=True).eval()
    missing = extended.load_state_dict(deterministic.state_dict(), strict=False)
    assert set(missing.missing_keys) == {"variance_out.weight", "variance_out.bias"}
    assert not missing.unexpected_keys
    inputs = torch.randn(1, 8, 4, 4, 4)
    with torch.no_grad():
        prediction, variance = extended(inputs)
        torch.testing.assert_close(prediction, deterministic(inputs), rtol=0, atol=0)
    assert prediction.shape == variance.shape == (1, 4, 4, 4, 4)
    assert (variance >= 0).all()
    checkpoint_path = tmp_path / "smoke.pt"
    torch.save(extended.state_dict(), checkpoint_path)
    restored = UNet3D(**options, uncertainty_aware=True).eval()
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    with torch.no_grad():
        restored_prediction, restored_variance = restored(inputs)
    torch.testing.assert_close(restored_prediction, prediction)
    torch.testing.assert_close(restored_variance, variance)


def test_one_synthetic_batch_and_validation():
    model = UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True)
    loader = [{"block": torch.randn(1, 4, 4, 4, 4)}]
    mean = torch.zeros(1, 4, 1, 1, 1)
    std = torch.ones_like(mean)
    before = model.variance_out.weight.detach().clone()
    metrics = train_one_epoch(
        model, loader, torch.optim.SGD(model.parameters(), lr=0.01),
        torch.device("cpu"), mean, std, {"spatial_random": 1.0},
        0.05, 0.05, 0.05, 0.05, False, 1.0,
        uncertainty_aware=True, uncertainty_loss_weight=1.0,
    )
    assert torch.isfinite(torch.tensor(metrics["mse"]))
    assert torch.isfinite(torch.tensor(metrics["uncertainty_mse"]))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not torch.equal(before, model.variance_out.weight)
    metrics = validate(
        model, loader, torch.device("cpu"), mean, std, ["spatial_random"],
        4321, 0, 30, False, uncertainty_aware=True,
    )
    assert torch.isfinite(torch.tensor(metrics["mean"]["mse"]))
    assert torch.isfinite(torch.tensor(metrics["mean"]["uncertainty_mse"]))


def test_cli_and_old_checkpoint_resume(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--uncertainty-aware", "--uncertainty-loss-weight", "0.2"])
    args = parse_args()
    assert args.uncertainty_aware and args.uncertainty_loss_weight == 0.2
    old_args = {key: getattr(args, key) for key in RESUME_PARAMETER_KEYS}
    with pytest.raises(ValueError, match="uncertainty_aware"):
        validate_resume_parameters(args, old_args)
    args.uncertainty_aware = False
    validate_resume_parameters(args, old_args)
    args.uncertainty_aware = True
    changed_weight = dict(old_args, uncertainty_aware=True, uncertainty_loss_weight=0.1)
    with pytest.raises(ValueError, match="uncertainty_loss_weight"):
        validate_resume_parameters(args, changed_weight)


class RecordingUncertainModel(torch.nn.Module):
    uncertainty_aware = True

    def __init__(self, error):
        super().__init__()
        self.error = error
        self.inputs = []

    def forward(self, inputs):
        self.inputs.append(inputs.detach().clone())
        prediction = torch.full_like(inputs[:, :4], 5.0)
        return prediction, torch.full_like(prediction, self.error)


@pytest.mark.parametrize("error", [-0.5, 0.25, 2.0])
def test_sliding_reuses_values_without_noise_and_carries_clipped_error(error):
    model = RecordingUncertainModel(error)
    target = torch.full((4, 6, 2, 2), 2.0)
    probes = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    result = reconstruct_with_slide_step(
        model, target, target[3].numpy(), torch.tensor(0.0), torch.tensor(1.0),
        probes, 4, 2, "normalized", None, False, retain_state=True,
    )
    first, second = model.inputs[:2]
    assert torch.equal(first[0, :3], target[:3, :4])
    assert torch.all(first[0, 4:7] == 0)
    assert torch.all(first[0, 7, :, 0, 0] == 0)
    assert torch.all(first[0, 7, :, 1, 1] == 1)
    assert torch.all(second[0, 3, :2, 1, 1] == 5.0)
    assert torch.all(second[0, 7, :2, 1, 1] == min(max(error, 0.0), 1.0))
    assert torch.all(second[0, 3, :, 0, 0] == 2.0)
    assert torch.all(second[0, 7, :, 0, 0] == 0)
    assert torch.all(second[0, 7, 2:, 1, 1] == 1)
    # Refinement must retain the error state too, including padded windows.
    refine_reconstruction_pass(
        model, target, result["raw_state_normalized"],
        result["conditioning_state_normalized"], probes, 4, 2, "reverse", 0,
        False, variance_state_normalized=result["variance_state_normalized"],
    )
    reused = model.inputs[-1]
    assert torch.all(reused[0, 3, :, 1, 1] == 5.0)
    assert torch.all(reused[0, 7, :, 1, 1] == min(max(error, 0.0), 1.0))
    assert torch.all(reused[0, 3, :, 0, 0] == 2.0)
    assert torch.all(reused[0, 7, :, 0, 0] == 0)


def test_bidirectional_passes_carry_variance_and_keep_hidden_magnetic_unknown():
    model = RecordingUncertainModel(0.25)
    target = torch.full((4, 6, 2, 2), 2.0)
    rows = build_bidirectional_rows(
        model, target, target[3].numpy(), torch.tensor(0.0), torch.tensor(1.0),
        torch.zeros(2, 2), 4, 2, 2, 2, "normalized", None, False,
        hide_magnetic=True,
    )
    assert len(rows) == 4
    assert all(torch.all(inputs[0, 4:7] == 1) for inputs in model.inputs)
    assert any(torch.any(inputs[0, 7] == 0.25) for inputs in model.inputs)
