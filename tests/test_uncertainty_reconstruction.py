"""Small CPU checks for variance regression and recursive conditioning."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.masking import (
    MASK_PATTERNS, GAUSSIAN_NLL_VARIANCE_FLOOR,
    full_mse_loss, make_uncertain_input, predicted_error_mse_loss,
)
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
    assert (variance > 0).all()
    checkpoint_path = tmp_path / "smoke.pt"
    torch.save(extended.state_dict(), checkpoint_path)
    restored = UNet3D(**options, uncertainty_aware=True).eval()
    restored.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    with torch.no_grad():
        restored_prediction, restored_variance = restored(inputs)
    torch.testing.assert_close(restored_prediction, prediction)
    torch.testing.assert_close(restored_variance, variance)


@pytest.mark.parametrize("objective,pattern", [("smooth_l1", "spatial_random")] +
                         [("gaussian_nll", p) for p in MASK_PATTERNS])
def test_one_synthetic_batch_and_validation(objective, pattern):
    model = UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True)
    loader = [{"block": torch.randn(1, 4, 4, 4, 4), "metadata": {"beta": [0.01]}}]
    def check_binary_input(_model, inputs):
        values, variance = inputs[0][:, :4], inputs[0][:, 4:]
        assert ((variance == 0) | (variance == 1)).all()
        torch.testing.assert_close(values[variance == 0], loader[0]["block"][variance == 0])
    hook = model.register_forward_pre_hook(check_binary_input)
    seen_variances = []
    output_hook = model.register_forward_hook(lambda _m, _i, out: seen_variances.append(out[1].detach().float()))
    mean = torch.zeros(1, 4, 1, 1, 1)
    std = torch.ones_like(mean)
    before = model.variance_out.weight.detach().clone()
    metrics = train_one_epoch(
        model, loader, torch.optim.SGD(model.parameters(), lr=0.01),
        torch.device("cpu"), mean, std, {pattern: 1.0},
        0.05, 0.05, 0.05, 0.05, False, 1.0,
        uncertainty_aware=True, uncertainty_loss_weight=1.0,
        uncertainty_loss=objective,
    )
    assert torch.isfinite(torch.tensor(metrics["mse"]))
    key = "gaussian_nll" if objective == "gaussian_nll" else "uncertainty_mse"
    assert torch.isfinite(torch.tensor(metrics[key]))
    if objective == "gaussian_nll":
        assert metrics["binary_input_variance_verified"] and metrics["nonfinite_events"] == 0
        assert metrics["gradient_norm_max"] >= metrics["gradient_norm_mean"] > 0
        assert metrics["optimizer_updates"] == 1 and metrics["amp_skipped_steps"] == 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert not torch.equal(before, model.variance_out.weight)
    metrics = validate(
        model, loader, torch.device("cpu"), mean, std, [pattern],
        4321, 0, 30, False, uncertainty_aware=True,
        uncertainty_loss=objective,
    )
    assert torch.isfinite(torch.tensor(metrics["mean"]["mse"]))
    assert torch.isfinite(torch.tensor(metrics["mean"][key]))
    hook.remove()
    output_hook.remove()
    if objective == "gaussian_nll":
        assert metrics["per_beta"]["0.01"]["mse"] == pytest.approx(metrics["mean"]["mse"])
        assert set(metrics["variance_statistics"]) == {"Bx", "By", "Bz", "Density"}
        for i, row in enumerate(metrics["variance_statistics"].values()):
            assert 0 < row["median"] <= row["p90"] <= row["p99"]
            assert row["count"] == 64 and row["quantile_sample_count"] == 64
            assert row["mean"] == pytest.approx(float(seen_variances[-1][:, i].mean()))
            assert row["fraction_gt_one"] == pytest.approx(float((seen_variances[-1][:, i] > 1).float().mean()))


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
    args.uncertainty_loss = "gaussian_nll"
    compatible = dict(old_args, uncertainty_aware=True, uncertainty_loss_weight=args.uncertainty_loss_weight)
    with pytest.raises(ValueError, match="uncertainty_loss:"):
        validate_resume_parameters(args, compatible)
    validate_resume_parameters(args, dict(compatible, uncertainty_loss="gaussian_nll"))


def test_gaussian_nll_matches_fp32_formula_detaches_mu_and_keeps_backbone_gradients():
    prediction = torch.tensor([3., 0., 2.], dtype=torch.bfloat16, requires_grad=True)
    variance = torch.tensor([2., .5, 1e-8], dtype=torch.bfloat16, requires_grad=True)
    target = torch.zeros_like(prediction)
    loss = predicted_error_mse_loss(variance, prediction, target, objective="gaussian_nll")
    safe = variance.float().clamp_min(GAUSSIAN_NLL_VARIANCE_FLOOR)
    expected = .5 * (safe.log() + prediction.detach().float().square() / safe).mean()
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prediction.grad is None and torch.isfinite(variance.grad).all()
    assert variance.grad[-1] == 0
    model = UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True)
    prediction, variance = model(torch.randn(1, 8, 4, 4, 4))
    predicted_error_mse_loss(variance, prediction, torch.zeros_like(prediction), "gaussian_nll").backward()
    assert model.out.weight.grad is None
    assert model.variance_out.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.enc0.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("fault", ["mask", "variance", "gradient", "reconstruction"])
def test_stage1_stops_before_optimizer_step_on_invalid_values(monkeypatch, fault):
    import train_masked_unet3d as train
    model = UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True)
    optimizer = torch.optim.AdamW(model.parameters())
    if fault == "mask":
        original = train.sample_independent_batch_masks
        def nonbinary(*args, **kwargs):
            mask, infos = original(*args, **kwargs)
            return mask.fill_(.5), infos
        monkeypatch.setattr(train, "sample_independent_batch_masks", nonbinary)
    elif fault == "variance":
        model.register_forward_hook(lambda _m, _i, out: (out[0], out[1] * float("nan")))
    elif fault == "reconstruction":
        model.register_forward_hook(lambda _m, _i, out: (out[0] * 0 + 1e6, out[1]))
    else:
        model.variance_out.weight.register_hook(lambda grad: grad * float("nan"))
    mean = torch.zeros(1, 4, 1, 1, 1)
    with pytest.raises(FloatingPointError):
        train_one_epoch(model, [{"block": torch.randn(1, 4, 4, 4, 4)}], optimizer,
                        torch.device("cpu"), mean, torch.ones_like(mean), {"spatial_random": 1.},
                        .05, .05, .05, .05, False, 1., uncertainty_aware=True,
                        uncertainty_loss="gaussian_nll")
    assert not optimizer.state


def test_six_epoch_main_preserves_twenty_epoch_schedule_and_reference_stats(monkeypatch, tmp_path, capsys):
    import json
    import train_masked_unet3d as train

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2
        def __getitem__(self, i):
            return {"block": torch.ones(4, 2, 4, 4), "metadata": {"beta": .01}}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.1))
            self.logit = torch.nn.Parameter(torch.tensor(0.))
        def forward(self, x):
            return x[:, :4] * self.scale, F.softplus(self.logit).expand_as(x[:, :4])

    reference = tmp_path / "reference"
    reference.mkdir()
    stats = {"mean": [0.] * 4, "std": [1.] * 4}
    (reference / "stats.json").write_text(json.dumps(stats))
    (reference / "split.json").write_text(json.dumps({"train_runs": ["train"], "val_runs": ["val"]}))
    out = tmp_path / "experiment"
    monkeypatch.setattr(sys, "argv", ["train", "--epochs", "20", "--stop-after-epochs", "6",
                        "--warmup-epochs", "1", "--lr", "2e-5", "--min-lr", "2e-6",
                        "--batch-size", "1", "--num-workers", "0", "--uncertainty-aware",
                        "--uncertainty-loss", "gaussian_nll", "--no-auto-resume", "--no-wandb",
                        "--mask-patterns", "spatial_random", "1", "--val-patterns", "spatial_random",
                        "--reference-run-dir", str(reference), "--out-dir", str(out)])
    monkeypatch.setattr(train, "initialize_distributed", lambda: (torch.device("cpu"), 0, 1, 0))
    monkeypatch.setattr(train, "VPICWindowDataset", lambda **kw: Dataset())
    monkeypatch.setattr(train, "DataLoader", lambda *a, **kw: torch.utils.data.DataLoader(
        *a, **dict(kw, pin_memory=False)))
    monkeypatch.setattr(train, "split_by_run", lambda *a, **kw: ([0], [1], ["train"], ["val"]))
    monkeypatch.setattr(train, "UNet3D", lambda **kw: Model())
    def no_recompute(*a, **kw):
        pytest.fail("Reference normalization must not be recomputed")
    monkeypatch.setattr(train, "estimate_channel_stats", no_recompute)
    train.main()
    history = json.loads((out / "history.json").read_text())
    assert [r["epoch"] for r in history] == list(range(1, 7))
    assert [r["learning_rate"] for r in history] == [
        train.learning_rate_for_epoch(e, 20, 2e-5, 1, 2e-6) for e in range(1, 7)]
    assert history[-1]["learning_rate"] > 1.7e-5  # Not the compressed six-epoch cosine minimum.
    assert all("train_gaussian_nll" in r and "train_uncertainty_mse" not in r for r in history)
    checkpoint = torch.load(out / "latest.pt", weights_only=False)
    assert checkpoint["epoch"] == 6 and checkpoint["args"]["epochs"] == 20
    assert checkpoint["stats"] == stats
    Model().load_state_dict(checkpoint["model"], strict=True)
    assert all(float(state["step"]) == 6 for state in checkpoint["optimizer"]["state"].values())
    printed = capsys.readouterr().out
    assert "Gaussian NLL" in printed and "uncertainty MSE" not in printed


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


def test_curriculum_cosine_schedule_and_nonoverlap(monkeypatch):
    from train_masked_unet3d import cosine_continuation, curriculum_weights, make_curriculum_state, default_run_name
    values = [cosine_continuation(s, 10, 100) for s in range(121)]
    assert values[0] == values[10] == 0 and values[110] == values[-1] == 1
    assert values[60] == pytest.approx(.5)
    assert all(a < b for a, b in zip(values[10:110], values[11:111]))
    assert max(b - a for a, b in zip(values, values[1:])) < .016
    h = 1e-3
    assert cosine_continuation(10 + h, 10, 100) / h < 1e-6
    assert (1 - cosine_continuation(110 - h, 10, 100)) / h < 1e-6
    argv = ["train", "--curriculum-training", "--uncertainty-aware", "--uncertainty-loss", "gaussian_nll",
            "--variance-ramp-start-epoch", "1", "--variance-ramp-epochs", "2",
            "--rollout-ramp-start-epoch", "4", "--rollout-ramp-epochs", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    args = parse_args()
    curriculum_name = default_run_name(args)
    args.curriculum_training = False
    assert default_run_name(args) != curriculum_name
    args.curriculum_training = True
    state = make_curriculum_state(args, 100)
    for step, expected in ((0, (0, 0)), (200, (.5, 0)), (300, (1, 0)),
                           (500, (1, .5)), (600, (1, 1))):
        state["successful_optimizer_steps"] = step
        assert curriculum_weights(state) == pytest.approx(expected)
    state["successful_optimizer_steps"] = 201
    restored = make_curriculum_state(args, 100, {"curriculum_state": state})
    assert curriculum_weights(restored) == curriculum_weights(state)
    with pytest.raises(ValueError, match="updates_per_epoch"):
        make_curriculum_state(args, 101, {"curriculum_state": state})
    with pytest.raises(ValueError, match="requires saved"):
        make_curriculum_state(args, 100, {})
    validate_resume_parameters(args, vars(args).copy())
    with pytest.raises(ValueError, match="variance_ramp_epochs"):
        validate_resume_parameters(args, dict(vars(args), variance_ramp_epochs=3))
    monkeypatch.setattr(sys, "argv", argv + ["--rollout-ramp-start-epoch", "2"])
    with pytest.raises(SystemExit):
        parse_args()
    monkeypatch.setattr(sys, "argv", argv[:6])
    with pytest.raises(SystemExit):
        parse_args()  # Malformed explicit start flags must still be rejected.


def test_manual_curriculum_arming_and_checkpoint_continuity(monkeypatch, tmp_path):
    from train_masked_unet3d import make_curriculum_state, curriculum_weights, save_checkpoint
    base = ["train", "--curriculum-training", "--uncertainty-aware", "--uncertainty-loss", "gaussian_nll"]
    monkeypatch.setattr(sys, "argv", base)
    args = parse_args()
    state = make_curriculum_state(args, 10)
    assert curriculum_weights(state) == (0., 0.)
    assert state["variance_ramp_start_step"] is state["rollout_ramp_start_step"] is None
    assert not state["variance_ramp_armed"] and not state["rollout_ramp_armed"]
    assert state["variance_ramp_steps"] == state["rollout_ramp_steps"] == 400
    state["successful_optimizer_steps"] = 1000
    assert curriculum_weights(state) == (0., 0.)  # No automatic transition.
    args.arm_rollout_ramp = True
    with pytest.raises(ValueError, match="fully completed"):
        make_curriculum_state(args, 10, {"curriculum_state": state})
    args.arm_rollout_ramp = False
    original = dict(state)
    args.arm_variance_ramp = True
    with pytest.raises(ValueError, match="resuming"):
        make_curriculum_state(args, 10)
    state = make_curriculum_state(args, 10, {"curriculum_state": state})
    assert curriculum_weights(state) == (0., 0.)
    assert state["variance_ramp_start_step"] == 1000
    assert {k for k in state if state[k] != original[k]} == {"variance_ramp_armed", "variance_ramp_start_step"}
    with pytest.raises(ValueError, match="already armed"):
        make_curriculum_state(args, 10, {"curriculum_state": state})
    args.arm_variance_ramp = False
    state["successful_optimizer_steps"] += 1
    assert 0 < curriculum_weights(state)[0] < .0001
    args.arm_rollout_ramp = True
    with pytest.raises(ValueError, match="fully completed"):
        make_curriculum_state(args, 10, {"curriculum_state": state})
    args.arm_rollout_ramp = False
    model = UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-6)
    checkpoint_path = tmp_path / "latest.pt"
    for prefix, mid_step in (("variance", 1200), ("rollout", 1600)):
        state["successful_optimizer_steps"] = mid_step
        expected = curriculum_weights(state)
        save_checkpoint(checkpoint_path, model, optimizer, 20, .1, args, {}, curriculum_state=state)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        assert (checkpoint["curriculum_state"]["lambda_v"], checkpoint["curriculum_state"]["lambda_r"]) == expected
        assert checkpoint["lr_schedule"]["horizon_epochs"] == 20
        restored = make_curriculum_state(args, 10, checkpoint)
        for step in range(mid_step, mid_step + 20):
            restored["successful_optimizer_steps"] = state["successful_optimizer_steps"] = step
            assert curriculum_weights(restored) == curriculum_weights(state)
        if prefix == "variance":
            state["successful_optimizer_steps"] = 1400
            assert curriculum_weights(state) == (1., 0.)
            args.arm_rollout_ramp = True
            before = dict(state)
            state = make_curriculum_state(args, 10, {"curriculum_state": state})
            assert curriculum_weights(state) == (1., 0.)
            assert state["rollout_ramp_start_step"] == 1400
            assert {k for k in state if state[k] != before[k]} == {"rollout_ramp_armed", "rollout_ramp_start_step"}
            with pytest.raises(ValueError, match="already armed"):
                make_curriculum_state(args, 10, {"curriculum_state": state})
            args.arm_rollout_ramp = False
    invalid = dict(state, rollout_ramp_start_step=1399)
    with pytest.raises(ValueError, match="overlap"):
        make_curriculum_state(args, 10, {"curriculum_state": invalid})
    # Previous curriculum checkpoints did not store explicit armed flags.
    legacy_args = type(args)(**dict(vars(args), variance_ramp_start_epoch=1., rollout_ramp_start_epoch=41.))
    legacy = make_curriculum_state(legacy_args, 10)
    legacy.pop("variance_ramp_armed")
    legacy.pop("rollout_ramp_armed")
    legacy["successful_optimizer_steps"] = 250
    restored = make_curriculum_state(legacy_args, 10, {"curriculum_state": legacy})
    assert curriculum_weights(restored) == curriculum_weights(legacy)


def test_manual_curriculum_cli_guards(monkeypatch):
    base = ["train", "--curriculum-training", "--uncertainty-aware", "--uncertainty-loss", "gaussian_nll"]
    for extra in (["--arm-variance-ramp", "--no-auto-resume"],
                  ["--arm-variance-ramp", "--arm-rollout-ramp"]):
        monkeypatch.setattr(sys, "argv", base + extra)
        with pytest.raises(SystemExit): parse_args()
    monkeypatch.setattr(sys, "argv", ["train", "--arm-variance-ramp"])
    with pytest.raises(SystemExit): parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--stop-after-epochs", "60"])
    assert parse_args().epochs == 20 and parse_args().stop_after_epochs == 60
    monkeypatch.setattr(sys, "argv", ["train", "--stop-after-epochs", "60"])
    with pytest.raises(SystemExit): parse_args()


def test_curriculum_rollout_input_exact_reuse_noise_and_detach():
    from data.masking import make_rollout_input
    truth = torch.full((1, 4, 1, 1, 4), 7.)
    pred = torch.full_like(truth, 3., requires_grad=True)
    variance = torch.tensor([.25, 2., 4., 5.]).expand_as(truth).clone().requires_grad_()
    observed = torch.tensor([1., 0., 0., 1.]).expand_as(truth)
    available = torch.tensor([1., 1., 0., 0.]).expand_as(truth)
    noise = torch.full_like(truth, -2.)
    inputs, exact = make_rollout_input(truth, pred, variance, observed, available, noise)
    assert not inputs.requires_grad
    assert torch.equal(inputs[:, :4], torch.tensor([7., 3., -2., -2.]).expand_as(truth))
    assert torch.equal(inputs[:, 4:], torch.tensor([0., 2., 1., 1.]).expand_as(truth))
    assert torch.equal(exact, observed * available)


@pytest.mark.parametrize("lambda_v,lambda_r,forwards", [(0., 0., 1), (.3, 0., 1), (1., .4, 2)])
def test_curriculum_phases_formula_and_gradient_routing(lambda_v, lambda_r, forwards, monkeypatch):
    from train_masked_unet3d import curriculum_forward_loss
    if lambda_v == 0:
        monkeypatch.setattr("train_masked_unet3d.predicted_error_mse_loss",
                            lambda *a, **kw: pytest.fail("Phase A must not compute a variance loss"))
    model = UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True)
    truth = torch.randn(1, 4, 4, 4, 4)
    mask = torch.randint(2, truth.shape).float()
    inputs = torch.cat([make_uncertain_input(truth, 1 - mask), 1 - mask], 1)
    outputs = []
    def record(_model, _inputs, out):
        for tensor in out:
            tensor.retain_grad()
        outputs.append(out)
    hook = model.register_forward_hook(record)
    reuse = torch.randint(2, truth.shape).float() if lambda_r else None
    loss, pred, variance, parts = curriculum_forward_loss(model, inputs, truth, mask, lambda_v, lambda_r, reuse)
    assert len(outputs) == parts["forwards"] == forwards
    assert variance.min() > 0 and torch.isfinite(variance).all()
    torch.testing.assert_close(parts["L_mu_1"], full_mse_loss(pred, truth, mask))
    expected = parts["L_mu_1"] + lambda_v * parts["L_var_1"]
    if lambda_r:
        expected = expected + lambda_r * (parts["L_mu_2"] + lambda_v * parts["L_var_2"])
        # Backward of pass two alone cannot reach either first-pass output.
        (parts["L_mu_2"] + parts["L_var_2"]).backward(retain_graph=True)
        assert pred.grad is None and variance.grad is None
        model.zero_grad(set_to_none=True)
    torch.testing.assert_close(loss, expected)  # No weight-sum normalization.
    if lambda_v:
        parts["L_var_1"].backward(retain_graph=True)
        assert model.variance_out.weight.grad.abs().sum() > 0
        assert model.out.weight.grad is None
        assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("variance_out."))
        model.zero_grad(set_to_none=True)
    else:
        assert parts["L_var_1"].item() == 0
    loss.backward()
    assert model.out.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert (model.variance_out.weight.grad is None) == (lambda_v == 0)
    hook.remove()
    # Strict loading uses identical keys in both gradient-routing modes.
    restored = UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True)
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        model.variance_out.weight.zero_()
        model.variance_out.bias.fill_(3.)
        assert (model(inputs, detach_variance_features=True)[1] > 1).all()
    assert GAUSSIAN_NLL_VARIANCE_FLOOR == 1e-6


def test_curriculum_skipped_updates_do_not_advance_ramps():
    from train_masked_unet3d import make_curriculum_state
    from types import SimpleNamespace
    state = make_curriculum_state(SimpleNamespace(variance_ramp_start_epoch=0., variance_ramp_epochs=2.,
                                  rollout_ramp_start_epoch=3., rollout_ramp_epochs=2.), 3)
    model = UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True)
    class SkipFirstScaler:
        def __init__(self): self.skipped = False
        def get_scale(self): return 1.
        def scale(self, loss): return loss
        def unscale_(self, optimizer): pass
        def update(self): pass
        def step(self, optimizer):
            if self.skipped: optimizer.step()
            self.skipped = True
    loader = [{"block": torch.randn(1, 4, 4, 4, 4)} for _ in range(3)]
    mean = torch.zeros(1, 4, 1, 1, 1)
    result = train_one_epoch(model, loader, torch.optim.AdamW(model.parameters(), lr=2e-5),
                             torch.device("cpu"), mean, torch.ones_like(mean), {"spatial_random": 1.},
                             .05, .05, .05, .05, False, 1., uncertainty_aware=True,
                             uncertainty_loss="gaussian_nll", curriculum_state=state, scaler=SkipFirstScaler())
    assert state["successful_optimizer_steps"] == 2
    assert result["amp_skipped_steps"] == 1 and result["optimizer_updates"] == 2
    from train_masked_unet3d import cosine_continuation
    assert result["curriculum"]["lambda_v_end"] == cosine_continuation(1, 0, 6)


def _curriculum_ddp_worker(rank, rendezvous):
    """Exercise changing unused parameters and two forwards on two CPU ranks."""
    import datetime
    from types import SimpleNamespace
    import torch.distributed as dist
    from train_masked_unet3d import make_curriculum_state
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=45))
    try:
        torch.manual_seed(1234)
        model = torch.nn.parallel.DistributedDataParallel(
            UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True),
            find_unused_parameters=True)
        state = make_curriculum_state(SimpleNamespace(variance_ramp_start_epoch=0., variance_ramp_epochs=2.,
                                      rollout_ramp_start_epoch=2., rollout_ramp_epochs=2.), 1)
        mean = torch.zeros(1, 4, 1, 1, 1)
        loader = [{"block": torch.randn(1, 4, 4, 4, 4)} for _ in range(5)]
        metrics = train_one_epoch(model, loader, torch.optim.AdamW(model.parameters(), lr=2e-5),
                                   torch.device("cpu"), mean, torch.ones_like(mean), {"spatial_random": 1.},
                                   .05, .05, .05, .05, False, 1., uncertainty_aware=True,
                                   uncertainty_loss="gaussian_nll", curriculum_state=state)
        assert state["successful_optimizer_steps"] == 5
        assert metrics["optimizer_updates"] == 5 and metrics["amp_skipped_steps"] == 0
        assert metrics["curriculum"]["two_forward_fraction"] == pytest.approx(2 / 5)
    finally:
        dist.destroy_process_group()


def test_curriculum_two_rank_ddp_phase_transition(tmp_path):
    torch.multiprocessing.start_processes(_curriculum_ddp_worker, args=(str(tmp_path / "rendezvous"),),
                                          nprocs=2, join=True, start_method="spawn")


@pytest.mark.parametrize("manual", [False, True], ids=["scheduled", "manual"])
def test_curriculum_compressed_main_save_resume_smoke(monkeypatch, tmp_path, capsys, manual):
    """Synthetic control-flow test only; opt into one allocated CUDA GPU via env."""
    import json
    import os
    import re
    import train_masked_unet3d as train
    device = torch.device(os.environ.get("AI4PLASMA_CURRICULUM_SMOKE_DEVICE", "cpu"))
    if device.type == "cuda":
        assert torch.cuda.is_available(), "GPU smoke explicitly requested but CUDA is unavailable"
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return 5
        def __getitem__(self, i):
            block = torch.randn(4, 2, 8, 8, generator=torch.Generator().manual_seed(i))
            return {"block": block, "metadata": {"beta": .01}}
    monkeypatch.setattr(train, "initialize_distributed", lambda: (device, 0, 1, 0))
    monkeypatch.setattr(train, "VPICWindowDataset", lambda **kw: Dataset())
    monkeypatch.setattr(train, "split_by_run", lambda *a, **kw: ([0, 1, 2, 3], [4], ["train"], ["val"]))
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "stats.json").write_text(json.dumps({"mean": [0.] * 4, "std": [1.] * 4}))
    (reference / "split.json").write_text(json.dumps({"train_runs": ["train"], "val_runs": ["val"]}))
    base = ["train", "--epochs", "20", "--warmup-epochs", "1", "--lr", "2e-5", "--min-lr", "2e-6",
            "--batch-size", "2", "--num-workers", "0", "--base-channels", "2", "--channel-mults", "1", "2", "4", "8",
            "--spatial-only-pooling", "--use-attention", "--uncertainty-aware", "--uncertainty-loss", "gaussian_nll",
            "--curriculum-training", "--variance-ramp-epochs", "2", "--rollout-ramp-epochs", "2", "--curriculum-log-every", "1",
            "--no-wandb", "--val-patterns", "spatial_random", "--reference-run-dir", str(reference)]
    if not manual:
        base += ["--variance-ramp-start-epoch", "1", "--rollout-ramp-start-epoch", "4"]
    if device.type == "cuda": base += ["--amp", "--amp-dtype", "bfloat16"]
    if manual and device.type == "cpu":
        # Exercise scaler restoration too, without requiring a CUDA allocation.
        monkeypatch.setattr(torch.cuda.amp, "GradScaler", lambda **kw: torch.amp.GradScaler("cpu", init_scale=8., growth_interval=1))
    def run(out, stop, fresh, arm=None):
        monkeypatch.setattr(sys, "argv", base + ["--out-dir", str(out), "--stop-after-epochs", str(stop),
                                               "--no-auto-resume" if fresh else "--auto-resume"] + ([arm] if arm else []))
        train.main()
        return capsys.readouterr().out
    if manual:
        out = tmp_path / "manual"
        def checkpoint(): return torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        log = run(out, 20, True)
        state = checkpoint()["curriculum_state"]
        assert state["successful_optimizer_steps"] == 40
        assert not state["variance_ramp_armed"] and not state["rollout_ramp_armed"]
        log += run(out, 21, False)  # No arm flag: remain in Phase A after the LR horizon.
        assert checkpoint()["curriculum_state"]["lambda_v"] == 0
        log += run(out, 22, False, "--arm-variance-ramp")
        state = checkpoint()["curriculum_state"]
        assert state["variance_ramp_start_step"] == 42 and state["lambda_v"] == pytest.approx(.5)
        log += run(out, 24, False)  # Resume in the middle of the variance ramp.
        assert checkpoint()["curriculum_state"]["lambda_v"] == 1
        log += run(out, 25, False, "--arm-rollout-ramp")
        state = checkpoint()["curriculum_state"]
        assert state["rollout_ramp_start_step"] == 48 and state["lambda_r"] == pytest.approx(.5)
        log += run(out, 27, False)  # Resume in the middle of the rollout ramp.
        saved = checkpoint()
        assert saved["curriculum_state"]["successful_optimizer_steps"] == 54
        assert saved["curriculum_state"]["lambda_v"] == saved["curriculum_state"]["lambda_r"] == 1
        assert {int(s["step"]) for s in saved["optimizer"]["state"].values()} == {11, 54}
        assert saved["lr_schedule"] == dict(horizon_epochs=20, warmup_epochs=1, base_lr=2e-5,
                                            min_lr=2e-6, current_lr=[2e-6])
        history = json.loads((out / "history.json").read_text())
        assert [r["epoch"] for r in history] == list(range(1, 28))
        assert all(r["learning_rate"] == 2e-6 for r in history if r["epoch"] >= 20)
        assert all(r["curriculum"]["total_loss"] == pytest.approx(r["train_mse"], rel=1e-7) and
                   r["curriculum"]["two_forward_fraction"] == 0 for r in history[:21])
        assert saved["best_val_mse"] == min(r["val_mse"] for r in history)
        assert all(r["amp_skipped_steps"] == r["nonfinite_events"] == 0 for r in history)
        assert all(torch.isfinite(torch.tensor(r["gradient_norm_max"])) for r in history)
        if device.type == "cpu":
            assert saved["scaler"]["scale"] == 8. * 2 ** 54
            assert all(r["amp_scale_start"] == 8. * 2 ** (2 * (r["epoch"] - 1)) for r in history)
        (tmp_path / "manual_curriculum_smoke.log").write_text(log)
        print(log)
        print("Manual arming/resume preserved optimizer, scaler, history, best metric and min LR; checkpoints:", out)
        return
    full, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    complete_log = run(full, 8, True)
    resumed_log = run(resumed, 2, True)  # saved step 4, halfway through variance ramp
    resumed_log += run(resumed, 5, False)  # saved step 10, halfway through rollout ramp
    resumed_log += run(resumed, 8, False)
    (tmp_path / "curriculum_smoke.log").write_text(complete_log + "\nSAVE/RESUME REPLAY\n" + resumed_log)
    regex = r"Curriculum ([ABC]): successful_step=(\d+) next_successful_step=(\d+) lambda_v=([\d.e+-]+) lambda_r=([\d.e+-]+) forwards=(\d+)"
    full_rows, resumed_rows = re.findall(regex, complete_log), re.findall(regex, resumed_log)
    assert len(full_rows) == 16 and full_rows == resumed_rows
    for region, step, next_step, lv, lr, forwards in full_rows:
        assert int(next_step) == int(step) + 1
        assert int(forwards) == (2 if float(lr) > 0 else 1)
        if float(lr) > 0: assert float(lv) == 1
    assert full_rows[0][3:] == ("0", "0", "1")
    assert full_rows[-1][3:] == ("1", "1", "2")
    checkpoint = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 8 and checkpoint["args"]["epochs"] == 20
    assert checkpoint["curriculum_state"]["successful_optimizer_steps"] == 16
    assert {int(s["step"]) for s in checkpoint["optimizer"]["state"].values()} == {13, 16}
    history = json.loads((resumed / "history.json").read_text())
    uninterrupted_history = json.loads((full / "history.json").read_text())
    for full_epoch, resumed_epoch in zip(uninterrupted_history, history):
        for key in ("lambda_v_start", "lambda_v_end", "lambda_v_mean",
                    "lambda_r_start", "lambda_r_end", "lambda_r_mean", "successful_optimizer_steps"):
            assert full_epoch["curriculum"][key] == resumed_epoch["curriculum"][key]
    assert [r["epoch"] for r in history] == list(range(1, 9))
    assert checkpoint["best_val_mse"] == min(r["val_mse"] for r in history)
    assert all(r["amp_skipped_steps"] == r["nonfinite_events"] == 0 for r in history)
    assert all(torch.isfinite(torch.tensor(r["gradient_norm_max"])) for r in history)
    assert history[-1]["curriculum"]["two_forward_fraction"] == 1
    assert all(r["learning_rate"] == train.learning_rate_for_epoch(r["epoch"], 20, 2e-5, 1, 2e-6) for r in history)
    print(complete_log)
    print("Save/resume at steps 4 and 10 exactly reproduced all 16 lambda pairs; checkpoints:", resumed)
