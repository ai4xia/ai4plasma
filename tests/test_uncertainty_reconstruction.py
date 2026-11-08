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
from models.unet3d import DIRECT_MODEL_VERSION, MODEL_VERSION, UNet3D
from train_masked_unet3d import (
    PhaseDStreamingBatch, RESUME_PARAMETER_KEYS, make_curriculum_state,
    parse_args, train_one_epoch, validate,
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


class _PhaseDRunDataset:
    def __init__(self, lengths=(30, 31), spatial_shape=(2, 2)):
        self.names = [
            "beta0.01_nu0_Bz0_dt2_tau100",
            "beta0.02_nu1_Bz0.1_dt2_tau150",
        ]
        self.run_file_indices = {name: index for index, name in enumerate(self.names)}
        self.runs = {}
        for run_index, (name, length) in enumerate(zip(self.names, lengths)):
            time = torch.arange(length, dtype=torch.float32).view(1, length, 1, 1)
            channel = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
            self.runs[name] = (
                time + 10 * channel + 100 * run_index
            ).expand(4, length, *spatial_shape).clone() / 100.0

    def get_run(self, run_name):
        return self.runs[run_name].clone()


def _phase_d_stream(dataset=None, batch_size=1, seed=1234):
    dataset = dataset or _PhaseDRunDataset()
    mean = torch.zeros(1, 4, 1, 1, 1)
    std = torch.ones_like(mean)
    return PhaseDStreamingBatch(
        dataset, dataset.names, batch_size, 24, torch.device("cpu"), mean, std, seed,
    )


def test_phase_d_finite_target_losses_exclude_nan_before_arithmetic():
    target = torch.full((1, 4, 24, 2, 2), float("nan"))
    target[:, :, 20:] = torch.randn(1, 4, 4, 2, 2)
    prediction = torch.randn_like(target, requires_grad=True)
    variance = torch.full_like(target, 2.0, requires_grad=True)
    mask = torch.randint(2, target.shape).float()
    finite = torch.isfinite(target)
    mse = full_mse_loss(prediction, target, mask)
    nll = predicted_error_mse_loss(variance, prediction, target, "gaussian_nll")
    assert torch.isfinite(mse) and torch.isfinite(nll)
    safe_error = (prediction.detach()[finite] - target[finite]).square()
    expected_nll = 0.5 * (torch.log(torch.tensor(2.0)) + safe_error / 2.0).mean()
    torch.testing.assert_close(nll, expected_nll)
    (mse + nll).backward()
    assert torch.isfinite(prediction.grad).all() and torch.isfinite(variance.grad).all()
    assert (variance.grad[~finite] == 0).all()


def test_phase_d_positive_and_zero_overlap_input_construction():
    stream = _phase_d_stream()
    name = stream.train_runs[0]
    trajectory = stream.reset_slot(0, name, slide_step=1, window_start=-23)
    assert trajectory.start == -23
    target, _ = stream.current_targets()
    finite_times = torch.isfinite(target[0, 0, :, 0, 0]).nonzero().flatten().tolist()
    assert finite_times == [23]  # only physical frame zero
    hidden = torch.zeros_like(target)
    inputs, exact = stream.make_model_input(target, hidden, torch.full_like(target, 7.0))
    assert torch.isfinite(inputs).all() and not exact.any()
    assert (inputs[:, :4] == 7).all() and (inputs[:, 4:] == 1).all()

    prediction = torch.full_like(target, 3.0, requires_grad=True)
    variance = torch.full_like(target, 2.0, requires_grad=True)
    stream.advance(prediction, variance)
    trajectory = stream.slots[0]
    assert trajectory.start == -22 and trajectory.reuse_depth == 1
    assert trajectory.previous_mean.grad_fn is None
    target, _ = stream.current_targets()
    observation = torch.zeros_like(target)
    observation[:, :, 22] = 1  # physical frame zero overrides overlap reuse
    inputs, exact = stream.make_model_input(target, observation, torch.full_like(target, -5.0))
    assert (inputs[:, :4, :22] == 3).all()  # padded predictions persist
    assert (inputs[:, 4:, :22] == 2).all()  # raw u > 1 is retained
    assert (inputs[:, :4, 23] == -5).all() and (inputs[:, 4:, 23] == 1).all()
    assert exact[:, :, 22].all() and (inputs[:, 4:, 22] == 0).all()

    zero_overlap = _phase_d_stream()
    zero_overlap.reset_slot(0, name, slide_step=24, window_start=-23)
    first, _ = zero_overlap.current_targets()
    zero_overlap.advance(torch.full_like(first, 9.0), torch.full_like(first, 4.0))
    assert zero_overlap.slots[0].start == 1
    second, _ = zero_overlap.current_targets()
    inputs, _ = zero_overlap.make_model_input(
        second, torch.zeros_like(second), torch.full_like(second, -2.0)
    )
    assert (inputs[:, :4] == -2).all() and (inputs[:, 4:] == 1).all()
    current, previous = PhaseDStreamingBatch.overlap_slices(24)
    assert (current.start, current.stop) == (0, 0)
    assert (previous.start, previous.stop) == (24, 24)


def test_phase_d_negative_sliding_is_exact_mirror():
    stream = _phase_d_stream()
    name = stream.train_runs[0]
    length = stream.dataset.runs[name].shape[1]
    trajectory = stream.reset_slot(
        0, name, slide_step=-1, window_start=length - 1,
    )
    assert trajectory.start == length - 1
    target, _ = stream.current_targets()
    assert torch.isfinite(target[0, 0, :, 0, 0]).nonzero().flatten().tolist() == [0]
    stream.advance(torch.full_like(target, 6.0), torch.full_like(target, 3.0))
    assert stream.slots[0].start == length - 2
    target, _ = stream.current_targets()
    inputs, _ = stream.make_model_input(
        target, torch.zeros_like(target), torch.full_like(target, -4.0)
    )
    assert (inputs[:, :4, 0] == -4).all() and (inputs[:, 4:, 0] == 1).all()
    assert (inputs[:, :4, 1:] == 6).all() and (inputs[:, 4:, 1:] == 3).all()

    zero_overlap = _phase_d_stream()
    zero_overlap.reset_slot(
        0, name, slide_step=-24, window_start=length - 1,
    )
    first, _ = zero_overlap.current_targets()
    zero_overlap.advance(torch.full_like(first, 8.0), torch.full_like(first, 5.0))
    assert zero_overlap.slots[0].start == length - 25
    second, _ = zero_overlap.current_targets()
    inputs, _ = zero_overlap.make_model_input(
        second, torch.zeros_like(second), torch.full_like(second, -3.0)
    )
    assert (inputs[:, :4] == -3).all() and (inputs[:, 4:] == 1).all()


def test_phase_d_zero_step_uses_full_windows_and_exact_sampled_length():
    stream = _phase_d_stream()
    name = stream.train_runs[0]
    length = stream.dataset.runs[name].shape[1]
    starts, lengths = set(), set()
    for _ in range(200):
        trajectory = stream.reset_slot(0, name, slide_step=0)
        starts.add(trajectory.start)
        lengths.add(trajectory.rehearsal_length)
        target, _ = stream.current_targets()
        assert torch.isfinite(target).all()
    assert min(starts) >= 0 and max(starts) <= length - 24
    assert min(lengths) >= 16 and max(lengths) <= 48
    assert len(lengths) > 1
    with pytest.raises(ValueError, match="Invalid Phase-D rehearsal window start"):
        stream.reset_slot(0, name, slide_step=0, window_start=-1)
    with pytest.raises(ValueError, match="Invalid Phase-D rehearsal window start"):
        stream.reset_slot(0, name, slide_step=0, window_start=length - 23)

    trajectory = stream.reset_slot(0, name, slide_step=0, window_start=2)
    rehearsal_length = trajectory.rehearsal_length
    target, _ = stream.current_targets()
    for update in range(rehearsal_length):
        prediction = torch.full_like(target, 4.0, requires_grad=True)
        variance = torch.full_like(target, 2.5, requires_grad=True)
        stream.advance(prediction, variance)
        if update + 1 < rehearsal_length:
            assert stream.slots[0] is trajectory
            assert trajectory.reuse_depth == update + 1
            assert trajectory.previous_mean.grad_fn is None
            assert trajectory.previous_variance.grad_fn is None
        else:
            assert stream.slots[0] is None


def test_phase_d_zero_step_selects_only_runs_with_full_physical_windows():
    dataset = _PhaseDRunDataset(lengths=(8, 30))
    stream = _phase_d_stream(dataset)
    assert stream.full_window_runs == (dataset.names[1],)
    for _ in range(20):
        trajectory = stream.reset_slot(0, slide_step=0)
        assert trajectory.run_name == dataset.names[1]
        target, _ = stream.current_targets()
        assert torch.isfinite(target).all()
    with pytest.raises(ValueError, match="requires at least 24 physical frames"):
        stream.reset_slot(0, dataset.names[0], slide_step=0)


def test_phase_d_zero_step_reuses_detached_state_and_new_exact_observations():
    stream = _phase_d_stream()
    name = stream.train_runs[0]
    trajectory = stream.reset_slot(0, name, slide_step=0, window_start=0)
    trajectory.rehearsal_length = 16
    target, _ = stream.current_targets()
    first_mask = torch.zeros_like(target)
    first_mask[:, :, :, 0, 0] = 1
    first_input, first_exact = stream.make_model_input(
        target, first_mask, torch.full_like(target, -7.0),
        magnetic_patterns=["spatial_random"],
        density_patterns=["temporal_block"],
    )
    first_exact = first_exact.bool()
    assert first_exact.any()
    assert (first_input[:, :4][first_exact] == target[first_exact]).all()
    assert (first_input[:, 4:][first_exact] == 0).all()
    assert (first_input[:, :4][~first_exact] == -7).all()
    stream.advance(
        torch.full_like(target, 4.0, requires_grad=True),
        torch.full_like(target, 2.5, requires_grad=True),
    )
    assert trajectory.previous_mean.grad_fn is None
    assert trajectory.previous_variance.grad_fn is None

    second_mask = torch.zeros_like(target)
    second_mask[:, :, :, 1, 1] = 1
    second_input, second_exact = stream.make_model_input(
        target, second_mask, torch.full_like(target, 99.0),
        magnetic_patterns=["spatial_grid"],
        density_patterns=["temporal_random"],
    )
    second_exact = second_exact.bool()
    unknown = ~second_exact
    assert (second_input[:, :4][unknown] == 4).all()
    assert (second_input[:, 4:][unknown] == 2.5).all()
    assert (second_input[:, :4][second_exact] == target[second_exact]).all()
    assert (second_input[:, 4:][second_exact] == 0).all()
    assert not (second_input[:, :4][unknown] == 99).any()
    stream.advance(torch.zeros_like(target), torch.ones_like(target))
    local = stream.local_epoch_metrics()
    assert local["s_zero_rehearsal_steps"] == 2
    assert local["s_zero_mask_count_spatial_random"] == 1
    assert local["s_zero_mask_count_spatial_grid"] == 1
    assert local["s_zero_mask_count_temporal_random"] == 1
    assert local["s_zero_mask_count_temporal_block"] == 1


def test_phase_d_nonzero_start_sampling_permits_at_least_one_slide():
    stream = _phase_d_stream()
    name = stream.train_runs[0]
    length = stream.dataset.runs[name].shape[1]
    positive, negative = set(), set()
    for _ in range(200):
        for step, seen in (
            (1, positive), (-1, negative), (24, positive), (-24, negative),
        ):
            trajectory = stream.reset_slot(0, name, slide_step=step)
            seen.add(trajectory.start)
            assert -23 <= trajectory.start <= length - 1
            assert stream.can_advance_start(trajectory.start, step, length)
            old_start = trajectory.start
            target, _ = stream.current_targets()
            stream.advance(torch.zeros_like(target), torch.ones_like(target))
            assert stream.slots[0].start == old_start + step
    assert min(positive) < 0
    assert max(negative) > length - 24
    with pytest.raises(ValueError, match="cannot perform a slide"):
        stream.reset_slot(0, name, slide_step=1, window_start=length - 24)
    with pytest.raises(ValueError, match="cannot perform a slide"):
        stream.reset_slot(0, name, slide_step=-1, window_start=0)


class _TinyPhaseDModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.nn.Conv3d(8, 4, 1)
        self.variance = torch.nn.Conv3d(8, 4, 1)
        torch.nn.init.zeros_(self.variance.weight)
        torch.nn.init.constant_(self.variance.bias, 2.0)
        self.seen_inputs = []

    def forward(self, inputs, detach_variance_features=False):
        self.seen_inputs.append(inputs.detach().clone())
        prediction = self.mean(inputs)
        variance = F.softplus(self.variance(inputs).float()).clamp_min(1e-6)
        return prediction, variance


@pytest.mark.parametrize("objective", ["gaussian_nll", "student_t_nll"])
def test_phase_d_streaming_optimizer_smoke_all_slide_cases_and_independent_slots(objective):
    dataset = _PhaseDRunDataset(lengths=(30, 31))
    stream = _phase_d_stream(dataset, batch_size=5)
    for slot, slide_step in enumerate((-24, -1, 0, 1, 24)):
        stream.reset_slot(slot, dataset.names[slot % 2], slide_step=slide_step)
    stream.slots[2].rehearsal_length = 20
    assert len({slot.run_name for slot in stream.slots}) == 2
    assert {slot.slide_step for slot in stream.slots} == {-24, -1, 0, 1, 24}

    model = _TinyPhaseDModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.mean.weight.detach().clone()
    args = type("Args", (), dict(
        variance_ramp_start_epoch=0.0, variance_ramp_epochs=1.0,
        rollout_ramp_start_epoch=1.0, rollout_ramp_epochs=1.0,
        arm_variance_ramp=False, arm_rollout_ramp=False,
    ))()
    state = make_curriculum_state(args, updates_per_epoch=1)
    state["successful_optimizer_steps"] = 2
    torch.manual_seed(1234)
    metrics = train_one_epoch(
        model, [None] * 20, optimizer, torch.device("cpu"),
        torch.zeros(1, 4, 1, 1, 1), torch.ones(1, 4, 1, 1, 1),
        {name: 1.0 for name in MASK_PATTERNS}, .05, .05, .05, .05, False, 1.0,
        uncertainty_aware=True, uncertainty_loss=objective,
        curriculum_state=state, phase_d_stream=stream,
    )
    assert metrics["optimizer_updates"] == 20 and metrics["amp_skipped_steps"] == 0
    assert stream.slots[2] is None  # the forced K=20 rehearsal reset exactly here
    assert not torch.equal(model.mean.weight, before)
    assert torch.isfinite(torch.tensor(metrics["mse"]))
    assert metrics["curriculum"]["two_forward_fraction"] == 0
    assert metrics["curriculum"]["L_mu_1"] == pytest.approx(metrics["mse"])
    assert metrics["curriculum"]["L_mu_0"] == pytest.approx(metrics["mse"])
    assert metrics["phase_d"]["finite_target_fraction"] < 1
    assert metrics["phase_d"]["reuse_depth_max"] >= 1
    assert all(metrics["phase_d"]["s_counts"][str(step)] > 0
               for step in (-24, -1, 0, 1, 24))
    assert metrics["phase_d"]["s_zero_rehearsal_steps"] >= 20
    mask_counts = [
        metrics["phase_d"][f"s_zero_mask_count_{name}"]
        for name in MASK_PATTERNS
    ]
    assert sum(mask_counts) == 2 * metrics["phase_d"]["s_zero_rehearsal_steps"]
    assert sum(count > 0 for count in mask_counts) > 1
    assert all(torch.isfinite(inputs).all() for inputs in model.seen_inputs)
    assert any((inputs[:, 4:] > 1).any() for inputs in model.seen_inputs[1:])
    assert all(slot is None or slot.previous_mean is None
               or slot.previous_mean.grad_fn is None for slot in stream.slots)


def _phase_d_ddp_worker(rank, rendezvous):
    import datetime
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        dataset = _PhaseDRunDataset(lengths=(30, 31))
        stream = _phase_d_stream(dataset, batch_size=2, seed=1234 + rank)
        stream.reset_slot(0, dataset.names[0], slide_step=-1)
        stream.reset_slot(1, dataset.names[1], slide_step=1)
        model = torch.nn.parallel.DistributedDataParallel(
            _TinyPhaseDModel(), find_unused_parameters=True,
        )
        args = type("Args", (), dict(
            variance_ramp_start_epoch=0.0, variance_ramp_epochs=1.0,
            rollout_ramp_start_epoch=1.0, rollout_ramp_epochs=1.0,
            arm_variance_ramp=False, arm_rollout_ramp=False,
        ))()
        state = make_curriculum_state(args, updates_per_epoch=1)
        state["successful_optimizer_steps"] = 2
        metrics = train_one_epoch(
            model, [None, None], torch.optim.AdamW(model.parameters(), lr=1e-3),
            torch.device("cpu"), torch.zeros(1, 4, 1, 1, 1),
            torch.ones(1, 4, 1, 1, 1), {"spatial_random": 1.0},
            .05, .05, .05, .05, False, 1.0, uncertainty_aware=True,
            uncertainty_loss="gaussian_nll", curriculum_state=state,
            phase_d_stream=stream,
        )
        assert state["successful_optimizer_steps"] == 4
        assert metrics["optimizer_updates"] == 2
        assert metrics["phase_d"]["s_negative_fraction"] == pytest.approx(.5)
        assert metrics["phase_d"]["s_positive_fraction"] == pytest.approx(.5)
        assert metrics["phase_d"]["finite_target_fraction"] < 1
    finally:
        dist.destroy_process_group()


def test_phase_d_two_rank_ddp_keeps_optimizer_step_control_flow_synchronized(tmp_path):
    torch.multiprocessing.start_processes(
        _phase_d_ddp_worker, args=(str(tmp_path / "phase_d_rendezvous"),),
        nprocs=2, join=True, start_method="spawn",
    )


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


@pytest.mark.parametrize("architecture", [MODEL_VERSION, DIRECT_MODEL_VERSION])
@pytest.mark.parametrize("bf16", [False, True])
def test_returned_variance_fp32_floor_and_gradients(architecture, bf16):
    model = UNet3D(base_channels=2, channel_mults=(1, 2, 4),
                   uncertainty_aware=True, architecture=architecture)
    logits = torch.tensor([-114.5, -1., 0., 3.])
    with torch.no_grad():
        model.variance_out.weight.zero_()
        model.variance_out.bias.copy_(logits)
    inputs = torch.randn(1, 8, 4, 4, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        prediction, variance = model(inputs)
    expected = F.softplus(logits.float()).clamp_min(1e-6)
    assert GAUSSIAN_NLL_VARIANCE_FLOOR == 1e-6
    assert variance.dtype == torch.float32
    torch.testing.assert_close(variance, expected.view(1, 4, 1, 1, 1).expand_as(variance), rtol=0, atol=0)
    assert (variance[:, 0] == torch.tensor(1e-6)).all()
    assert (variance[:, 3] > 1).all()
    expected_prediction = inputs[:, :4] if architecture == MODEL_VERSION else torch.zeros_like(prediction)
    torch.testing.assert_close(prediction, expected_prediction, rtol=0, atol=0)
    variance.sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.variance_out.bias.grad[0] == 0
    assert (model.variance_out.bias.grad[1:] > 0).all()
    assert model.variance_out.weight.grad is not None


@pytest.mark.parametrize("objective,pattern", [("smooth_l1", "spatial_random"),
                         ("student_t_nll", "spatial_random")] +
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
    key = objective if objective in ("gaussian_nll", "student_t_nll") else "uncertainty_mse"
    assert torch.isfinite(torch.tensor(metrics[key]))
    if objective in ("gaussian_nll", "student_t_nll"):
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
    if objective in ("gaussian_nll", "student_t_nll"):
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

    monkeypatch.setattr(sys, "argv", [
        "train", "--uncertainty-aware", "--uncertainty-loss", "student_t_nll",
        "--student-t-df", "5",
    ])
    student_args = parse_args()
    assert student_args.student_t_df == 5.0
    with pytest.raises(ValueError, match="student_t_df"):
        validate_resume_parameters(student_args, {
            **vars(student_args), "student_t_df": 3.0,
        })


def test_direct_model_cli_records_version_and_rejects_cross_version_resume(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train"])
    old_args = parse_args()
    assert old_args.model_version == MODEL_VERSION
    monkeypatch.setattr(sys, "argv", ["train", "--model-version", DIRECT_MODEL_VERSION])
    args = parse_args()
    assert args.model_version == DIRECT_MODEL_VERSION
    validate_resume_parameters(args, vars(args))
    with pytest.raises(ValueError, match="model_version"):
        validate_resume_parameters(args, vars(old_args))
    with pytest.raises(ValueError, match="model_version"):
        validate_resume_parameters(old_args, vars(args))


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


@pytest.mark.parametrize("df", [3.0, 5.0])
def test_student_t_nll_matches_torch_distribution_and_preserves_variance_semantics(df):
    torch.manual_seed(1234)
    prediction = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    target = torch.randn_like(prediction)
    variance = torch.logspace(-8, 2, prediction.numel()).reshape_as(prediction).requires_grad_()
    loss = predicted_error_mse_loss(
        variance, prediction, target, "student_t_nll", student_t_df=df,
    )
    safe_u = variance.float().clamp_min(GAUSSIAN_NLL_VARIANCE_FLOOR)
    scale_squared = safe_u * ((df - 2.0) / df)
    expected = -torch.distributions.StudentT(
        df=df, loc=prediction.detach().float(), scale=scale_squared.sqrt(),
    ).log_prob(target.float()).mean()
    torch.testing.assert_close(loss, expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(scale_squared * df / (df - 2.0), safe_u)
    assert loss.dtype == torch.float32
    loss.backward()
    assert prediction.grad is None
    assert torch.isfinite(variance.grad).all()


def test_student_t_nll_robust_residuals_floor_and_invalid_df(monkeypatch):
    target = torch.tensor([0.0, 1e-4, 100.0, float("nan")])
    prediction = torch.zeros_like(target)
    variance = torch.tensor([1e-12, 0.2, 1.0, 2.0], requires_grad=True)
    student = predicted_error_mse_loss(
        variance, prediction, target, "student_t_nll", student_t_df=3.0,
    )
    gaussian = predicted_error_mse_loss(variance, prediction, target, "gaussian_nll")
    assert torch.isfinite(student) and student < gaussian
    student.backward()
    assert variance.grad[-1] == 0  # NaN-padded target was filtered before arithmetic.
    assert torch.isfinite(variance.grad).all()
    for invalid in (2.0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="degrees of freedom"):
            predicted_error_mse_loss(
                variance, prediction, target, "student_t_nll", student_t_df=invalid,
            )
        monkeypatch.setattr(sys, "argv", [
            "train", "--uncertainty-aware", "--uncertainty-loss", "student_t_nll",
            "--student-t-df", str(invalid),
        ])
        with pytest.raises(SystemExit):
            parse_args()


def test_phase_c_k2_student_t_uses_three_detached_reuses():
    from train_masked_unet3d import curriculum_forward_loss

    model = UNet3D(
        base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True,
        architecture=DIRECT_MODEL_VERSION,
    )
    target = torch.randn(1, 4, 4, 4, 4)
    masks = [torch.randint(2, target.shape).float() for _ in range(3)]
    model_input = torch.cat([
        make_uncertain_input(target, 1 - masks[0]), 1 - masks[0],
    ], dim=1)
    phase_b_loss, _, _, phase_b_parts = curriculum_forward_loss(
        model, model_input, target, masks[0], 1.0, 0.0, None,
        "student_t_nll", 3.0,
    )
    assert phase_b_parts["forwards"] == 1 and torch.isfinite(phase_b_loss)
    model.zero_grad(set_to_none=True)
    loss, _, _, parts = curriculum_forward_loss(
        model, model_input, target, masks[0], 1.0, 1.0, tuple(masks[1:]),
        "student_t_nll", 3.0,
    )
    assert parts["forwards"] == 3
    assert all(torch.isfinite(parts[f"L_var_{index}"]) for index in range(3))
    loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


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


def test_six_epoch_main_preserves_twenty_epoch_schedule_and_fixed_train_stats(monkeypatch, tmp_path, capsys):
    import json
    import train_masked_unet3d as train

    from data.vpic_hdf5_dataset import load_fixed_split
    split = load_fixed_split()
    train_name = next(name for name in split["train_runs"] if name.startswith("beta0.2_"))
    val_name = next(name for name in split["val_runs"] if name.startswith("beta0.2_"))

    class Dataset(torch.utils.data.Dataset):
        samples = [(0, train_name, 0), (0, val_name, 0)]
        def __len__(self):
            return 2
        def __getitem__(self, i):
            return {"block": torch.ones(4, 2, 4, 4), "metadata": {"beta": .2}}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(.1))
            self.logit = torch.nn.Parameter(torch.tensor(0.))
        def forward(self, x):
            return x[:, :4] * self.scale, F.softplus(self.logit).expand_as(x[:, :4])

    stats = {"mean": [0.] * 4, "std": [1.] * 4}
    out = tmp_path / "experiment"
    monkeypatch.setattr(sys, "argv", ["train", "--epochs", "20", "--stop-after-epochs", "6",
                        "--warmup-epochs", "1", "--lr", "2e-5", "--min-lr", "2e-6",
                        "--batch-size", "1", "--num-workers", "0", "--uncertainty-aware",
                        "--uncertainty-loss", "gaussian_nll", "--no-auto-resume", "--no-wandb",
                        "--mask-patterns", "spatial_random", "1", "--val-patterns", "spatial_random",
                        "--final-epoch", "20", "--out-dir", str(out)])
    monkeypatch.setattr(train, "initialize_distributed", lambda: (torch.device("cpu"), 0, 1, 0))
    monkeypatch.setattr(train, "VPICWindowDataset", lambda **kw: Dataset())
    monkeypatch.setattr(train, "DataLoader", lambda *a, **kw: torch.utils.data.DataLoader(
        *a, **dict(kw, pin_memory=False)))
    monkeypatch.setattr(train, "split_by_run", lambda *a, **kw: ([0], [1], [train_name], [val_name]))
    monkeypatch.setattr(train, "UNet3D", lambda **kw: Model())
    stats_calls = []
    monkeypatch.setattr(train, "estimate_channel_stats", lambda *a, **kw: stats_calls.append(True) or stats)
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
    assert stats_calls == [True]
    assert not (out / "final.pt").exists()
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


@pytest.mark.parametrize("error", [0.25, 1.5, 2.0])
def test_sliding_reuses_values_without_noise_and_carries_raw_variance(error, monkeypatch):
    draw_noise, noises = torch.randn_like, []

    def record_noise(values):
        noise = draw_noise(values)
        noises.append(noise)
        return noise

    monkeypatch.setattr("data.masking.torch.randn_like", record_noise)
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
    torch.testing.assert_close(first[0, 3, :, 1, 1], noises[0][0, 3, :, 1, 1])
    assert torch.all(second[0, 3, :2, 1, 1] == 5.0)
    assert torch.all(second[0, 7, :2, 1, 1] == error)
    assert torch.all(second[0, 3, :, 0, 0] == 2.0)
    assert torch.all(second[0, 7, :, 0, 0] == 0)
    assert torch.all(second[0, 7, 2:, 1, 1] == 1)
    torch.testing.assert_close(second[0, 3, 2:, 1, 1], noises[1][0, 3, 2:, 1, 1])
    # Refinement must retain the error state too, including padded windows.
    refine_reconstruction_pass(
        model, target, result["raw_state_normalized"],
        result["conditioning_state_normalized"], probes, 4, 2, "reverse", 1,
        False, include_boundary_padding=True,
        variance_state_normalized=result["variance_state_normalized"],
    )
    reused = model.inputs[-1]
    assert torch.all(reused[0, 3, 1:, 1, 1] == 5.0)
    assert torch.all(reused[0, 7, 1:, 1, 1] == error)
    assert torch.all(reused[0, 3, 1:, 0, 0] == 2.0)
    assert torch.all(reused[0, 7, 1:, 0, 0] == 0)
    assert torch.all(reused[0, 4:, 0] == 1)
    torch.testing.assert_close(reused[0, :4, 0], noises[-1][0, :, 0])


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
    # Successful steps alone do not replace the validation-readiness actuator.
    assert curriculum_weights(state) == (0., 0.)
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
        save_checkpoint(checkpoint_path, model, optimizer, 20, args, {}, curriculum_state=state)
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


@pytest.mark.parametrize("reused_variance", [1.5, 2.0])
def test_curriculum_rollout_input_exact_reuse_noise_and_detach(reused_variance):
    from data.masking import make_rollout_input
    truth = torch.full((1, 4, 1, 1, 4), 7.)
    pred = torch.full_like(truth, 3., requires_grad=True)
    variance = torch.tensor([.25, reused_variance, 4., 5.]).expand_as(truth).clone().requires_grad_()
    observed = torch.tensor([1., 0., 0., 1.]).expand_as(truth)
    available = torch.tensor([1., 1., 0., 0.]).expand_as(truth)
    noise = torch.full_like(truth, -2.)
    inputs, exact = make_rollout_input(truth, pred, variance, observed, available, noise)
    assert not inputs.requires_grad
    assert torch.equal(inputs[:, :4], torch.tensor([7., 3., -2., -2.]).expand_as(truth))
    assert torch.equal(inputs[:, 4:], torch.tensor([0., reused_variance, 1., 1.]).expand_as(truth))
    assert torch.equal(exact, observed * available)


@pytest.mark.parametrize(
    "lambda_r,expected",
    [
        (0.0, (1.0, 0.0, 0.0)),
        (0.5, (1.0 / 1.75, 0.5 / 1.75, 0.25 / 1.75)),
        (1.0, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
    ],
)
def test_phase_c_k2_geometric_weights(lambda_r, expected):
    from train_masked_unet3d import phase_c_geometric_weights
    assert phase_c_geometric_weights(lambda_r) == pytest.approx(expected)


@pytest.mark.parametrize("lambda_v,lambda_r,forwards", [(0., 0., 1), (.3, 0., 1), (1., .4, 3)])
@pytest.mark.parametrize("architecture", [MODEL_VERSION, DIRECT_MODEL_VERSION])
def test_curriculum_phases_formula_and_gradient_routing(lambda_v, lambda_r, forwards, monkeypatch, architecture):
    from train_masked_unet3d import curriculum_forward_loss, phase_c_geometric_weights
    if lambda_v == 0:
        monkeypatch.setattr("train_masked_unet3d.predicted_error_mse_loss",
                            lambda *a, **kw: pytest.fail("Phase A must not compute a variance loss"))
    model = UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True, architecture=architecture)
    truth = torch.randn(1, 4, 4, 4, 4)
    mask = torch.randint(2, truth.shape).float()
    inputs = torch.cat([make_uncertain_input(truth, 1 - mask), 1 - mask], 1)
    outputs = []
    def record(_model, _inputs, out):
        for tensor in out:
            tensor.retain_grad()
        outputs.append(out)
    hook = model.register_forward_hook(record)
    reuse = tuple(torch.randint(2, truth.shape).float() for _ in range(2)) if lambda_r else None
    loss, pred, variance, parts = curriculum_forward_loss(model, inputs, truth, mask, lambda_v, lambda_r, reuse)
    assert len(outputs) == parts["forwards"] == forwards
    assert variance.min() > 0 and torch.isfinite(variance).all()
    torch.testing.assert_close(parts["L_mu_0"], full_mse_loss(pred, truth, mask))
    pass_losses = [parts[f"L_mu_{index}"] + lambda_v * parts[f"L_var_{index}"]
                   for index in range(3)]
    weights = phase_c_geometric_weights(lambda_r)
    expected = sum(weight * value for weight, value in zip(weights, pass_losses))
    for index, weight in enumerate(weights):
        assert parts[f"reuse_weight_{index}"].item() == pytest.approx(weight)
    if lambda_r:
        # Backward of either reuse pass cannot reach an earlier pass output.
        (parts["L_mu_1"] + parts["L_var_1"]).backward(retain_graph=True)
        assert pred.grad is None and variance.grad is None
        model.zero_grad(set_to_none=True)
        for output in outputs:
            for tensor in output:
                tensor.grad = None
        (parts["L_mu_2"] + parts["L_var_2"]).backward(retain_graph=True)
        assert all(tensor.grad is None for tensor in outputs[0] + outputs[1])
        model.zero_grad(set_to_none=True)
    torch.testing.assert_close(loss, expected)
    if lambda_v:
        parts["L_var_0"].backward(retain_graph=True)
        assert model.variance_out.weight.grad.abs().sum() > 0
        assert model.out.weight.grad is None
        assert all(p.grad is None for n, p in model.named_parameters() if not n.startswith("variance_out."))
        model.zero_grad(set_to_none=True)
    else:
        assert parts["L_var_0"].item() == 0
    loss.backward()
    assert model.out.weight.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert (model.variance_out.weight.grad is None) == (lambda_v == 0)
    hook.remove()
    # Strict loading uses identical keys in both gradient-routing modes.
    restored = UNet3D(base_channels=2, channel_mults=(1, 2, 4), uncertainty_aware=True, architecture=architecture)
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad():
        model.variance_out.weight.zero_()
        model.variance_out.bias.fill_(3.)
        assert (model(inputs, detach_variance_features=True)[1] > 1).all()
    assert GAUSSIAN_NLL_VARIANCE_FLOOR == 1e-6


@pytest.mark.parametrize("bf16", [False, True])
def test_direct_three_pass_optimization_preserves_reuse_and_raw_variance(monkeypatch, bf16):
    from train_masked_unet3d import curriculum_forward_loss

    model = UNet3D(base_channels=2, channel_mults=(1, 2, 4), use_attention=True,
                   spatial_only_pooling=True, uncertainty_aware=True, architecture=DIRECT_MODEL_VERSION)
    with torch.no_grad():
        model.variance_out.weight.zero_()
        model.variance_out.bias.fill_(torch.log(torch.expm1(torch.tensor(1.5))).item())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    target = torch.randn(1, 4, 4, 8, 8)
    observed = torch.zeros_like(target)
    observed[..., ::2] = 1
    reuse_1, reuse_2 = torch.zeros_like(target), torch.zeros_like(target)
    reuse_1[..., :4] = 1
    reuse_2[..., 4:] = 1
    draw_noise, noises, inputs, outputs = torch.randn_like, [], [], []

    def record_noise(values):
        noise = draw_noise(values)
        noises.append(noise)
        return noise

    monkeypatch.setattr("data.masking.torch.randn_like", record_noise)
    pre = model.register_forward_pre_hook(lambda m, a: inputs.append((a[0].detach().clone(), a[0].requires_grad)))
    post = model.register_forward_hook(lambda m, a, out: outputs.append(out))
    model_input = torch.cat([make_uncertain_input(target, 1 - observed), 1 - observed], dim=1)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        loss, _, _, parts = curriculum_forward_loss(
            model, model_input, target, observed, 1., 1., (reuse_1, reuse_2),
        )
    pre.remove()
    post.remove()
    assert len(inputs) == len(outputs) == 3
    assert len(noises) == 1  # Gaussian initialization occurs only in pass 0.
    first, second, third = (entry[0] for entry in inputs)
    assert not inputs[1][1] and not inputs[2][1]
    assert torch.equal(first[:, 4:], 1 - observed)
    assert torch.equal(first[:, :4][observed.bool()], target[observed.bool()])
    assert torch.equal(first[:, :4][~observed.bool()], noises[0][~observed.bool()])
    assert torch.count_nonzero(first[:, :4][~observed.bool()]) > 0
    assert torch.equal(second[:, :4][reuse_1.bool()], target[reuse_1.bool()])
    assert (second[:, 4:][reuse_1.bool()] == 0).all()
    torch.testing.assert_close(second[:, :4][~reuse_1.bool()], outputs[0][0].float()[~reuse_1.bool()], rtol=0, atol=0)
    torch.testing.assert_close(second[:, 4:][~reuse_1.bool()], outputs[0][1].float()[~reuse_1.bool()], rtol=0, atol=0)
    assert (second[:, 4:][~reuse_1.bool()] > 1).all()
    assert torch.equal(third[:, :4][reuse_2.bool()], target[reuse_2.bool()])
    assert (third[:, 4:][reuse_2.bool()] == 0).all()
    torch.testing.assert_close(third[:, :4][~reuse_2.bool()], outputs[1][0].float()[~reuse_2.bool()], rtol=0, atol=0)
    torch.testing.assert_close(third[:, 4:][~reuse_2.bool()], outputs[1][1].float()[~reuse_2.bool()], rtol=0, atol=0)
    assert (third[:, 4:][~reuse_2.bool()] > 1).all()
    for mean, variance in outputs:
        assert mean.shape == variance.shape == target.shape
        assert torch.isfinite(mean).all() and torch.isfinite(variance).all() and (variance > 0).all()
        assert mean.dtype == (torch.bfloat16 if bf16 else torch.float32)
    assert torch.isfinite(loss) and all(
        torch.isfinite(parts[f"L_{kind}_{index}"])
        for kind in ("mu", "var") for index in range(3)
    )
    assert all(parts[f"L_var_{index}"].dtype == torch.float32 for index in range(3))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    before = model.out.weight.detach().clone()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    assert not torch.equal(model.out.weight, before)


def test_phase_c_resamples_two_masks_and_uses_one_optimizer_update(monkeypatch):
    from types import SimpleNamespace
    import train_masked_unet3d as train

    original_sampler = train.sample_training_mask
    sampled_masks = []

    def record_sampler(*args, **kwargs):
        result = original_sampler(*args, **kwargs)
        sampled_masks.append(result[0].detach().clone())
        return result

    monkeypatch.setattr(train, "sample_training_mask", record_sampler)
    model = UNet3D(
        base_channels=2, channel_mults=(1, 2, 4),
        uncertainty_aware=True, architecture=DIRECT_MODEL_VERSION,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    args = SimpleNamespace(
        variance_ramp_start_epoch=0.0, variance_ramp_epochs=1.0,
        rollout_ramp_start_epoch=1.0, rollout_ramp_epochs=1.0,
        arm_variance_ramp=False, arm_rollout_ramp=False,
    )
    state = make_curriculum_state(args, updates_per_epoch=1)
    state["successful_optimizer_steps"] = 2
    metrics = train_one_epoch(
        model, [{"block": torch.randn(1, 4, 4, 4, 4)}], optimizer,
        torch.device("cpu"), torch.zeros(1, 4, 1, 1, 1),
        torch.ones(1, 4, 1, 1, 1), {name: 1.0 for name in MASK_PATTERNS},
        .05, .05, .05, .05, False, 1.0,
        uncertainty_aware=True, uncertainty_loss="gaussian_nll",
        curriculum_state=state,
    )
    assert len(sampled_masks) == 3  # pass 0 plus two independently sampled reuse masks
    assert len({mask.data_ptr() for mask in sampled_masks}) == 3
    assert metrics["optimizer_updates"] == 1
    assert metrics["curriculum"]["phase_c_forwards"] == 3
    assert metrics["curriculum"]["three_forward_fraction"] == 1
    assert metrics["curriculum"]["two_forward_fraction"] == 0
    expected_keys = {
        "L_mu_0", "L_var_0", "L_mu_1", "L_var_1", "L_mu_2", "L_var_2",
        "reuse_weight_0", "reuse_weight_1", "reuse_weight_2", "lambda_r",
        "phase_c_forwards",
    }
    assert expected_keys <= metrics["curriculum"].keys()


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
        assert metrics["curriculum"]["two_forward_fraction"] == 0
        assert metrics["curriculum"]["three_forward_fraction"] == pytest.approx(2 / 5)
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
    from types import SimpleNamespace
    import train_masked_unet3d as train
    device = torch.device(os.environ.get("AI4PLASMA_CURRICULUM_SMOKE_DEVICE", "cpu"))
    if device.type == "cuda":
        assert torch.cuda.is_available(), "GPU smoke explicitly requested but CUDA is unavailable"
    class Dataset(torch.utils.data.Dataset):
        samples = []
        run_file_indices = {}
        def __len__(self): return 5
        def __getitem__(self, i):
            block = torch.randn(4, 2, 8, 8, generator=torch.Generator().manual_seed(i))
            return {"block": block, "metadata": {"beta": .01}}
        def get_run(self, run_name):
            generator = torch.Generator().manual_seed(sum(map(ord, run_name)))
            return torch.randn(4, 30, 8, 8, generator=generator)
    from data.vpic_hdf5_dataset import load_fixed_split
    split = load_fixed_split()
    train_name = next(name for name in split["train_runs"] if name.startswith("beta0.2_"))
    val_name = next(name for name in split["val_runs"] if name.startswith("beta0.2_"))
    Dataset.samples = [(0, train_name, i) for i in range(4)] + [(0, val_name, 0)]
    Dataset.run_file_indices = {train_name: 0, val_name: 0}
    monkeypatch.setattr(train, "initialize_distributed", lambda: (device, 0, 1, 0))
    monkeypatch.setattr(train, "VPICWindowDataset", lambda **kw: Dataset())
    monkeypatch.setattr(train, "split_by_run", lambda *a, **kw: ([0, 1, 2, 3], [4], [train_name], [val_name]))
    monkeypatch.setattr(train, "estimate_channel_stats", lambda *a, **kw: {"mean": [0.] * 4, "std": [1.] * 4})
    base = ["train", "--epochs", "20", "--warmup-epochs", "1", "--lr", "2e-5", "--min-lr", "2e-6",
            "--batch-size", "2", "--num-workers", "0", "--base-channels", "2", "--channel-mults", "1", "2", "4", "8",
            "--spatial-only-pooling", "--use-attention", "--uncertainty-aware", "--uncertainty-loss", "gaussian_nll",
            "--curriculum-training", "--variance-ramp-epochs", "2", "--rollout-ramp-epochs", "2", "--curriculum-log-every", "1",
            "--no-wandb", "--val-patterns", "spatial_random", "--final-epoch", "200"]
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
        log += run(out, 27, False)  # Finish rollout; Phase D begins at epoch 27.
        log += run(out, 33, False)  # Resume during the fixed Phase-D cosine decay.
        saved = checkpoint()
        assert saved["curriculum_state"]["successful_optimizer_steps"] == 66
        assert saved["curriculum_state"]["lambda_v"] == saved["curriculum_state"]["lambda_r"] == 1
        assert saved["curriculum_state"]["phase_d_start_epoch"] == 27
        assert {int(s["step"]) for s in saved["optimizer"]["state"].values()} == {23, 66}
        phase_d_epoch_33_lr = train.curriculum_learning_rate_for_epoch(
            33, SimpleNamespace(epochs=20, warmup_epochs=1, lr=2e-5,
                                min_lr=2e-6, final_epoch=200), saved["curriculum_state"])
        assert saved["lr_schedule"] == dict(horizon_epochs=20, warmup_epochs=1, base_lr=2e-5,
                                            min_lr=2e-6, current_lr=[phase_d_epoch_33_lr])
        history = json.loads((out / "history.json").read_text())
        assert [r["epoch"] for r in history] == list(range(1, 34))
        assert all(r["learning_rate"] == 2e-6 for r in history if 20 <= r["epoch"] < 27)
        assert [r["learning_rate"] for r in history[26:31]] == pytest.approx(
            [3.6e-6, 5.2e-6, 6.8e-6, 8.4e-6, 1e-5]
        )
        assert history[-1]["learning_rate"] == phase_d_epoch_33_lr
        assert all(r["curriculum"]["total_loss"] == pytest.approx(r["train_mse"], rel=1e-7) and
                   r["curriculum"]["two_forward_fraction"] == 0 for r in history[:21])
        assert "best_val_mse" not in saved and not (out / "best.pt").exists()
        assert not (out / "final.pt").exists()
        assert all(r["amp_skipped_steps"] == r["nonfinite_events"] == 0 for r in history)
        assert all(torch.isfinite(torch.tensor(r["gradient_norm_max"])) for r in history)
        if device.type == "cpu":
            assert saved["scaler"]["scale"] == 8. * 2 ** 66
            assert all(r["amp_scale_start"] == 8. * 2 ** (2 * (r["epoch"] - 1)) for r in history)
        (tmp_path / "manual_curriculum_smoke.log").write_text(log)
        print(log)
        print("Manual arming/resume preserved optimizer, scaler, history and min LR; checkpoints:", out)
        return
    full, resumed = tmp_path / "uninterrupted", tmp_path / "resumed"
    complete_log = run(full, 8, True)
    resumed_log = run(resumed, 2, True)  # saved step 4, halfway through variance ramp
    resumed_log += run(resumed, 5, False)  # saved step 10, halfway through rollout ramp
    resumed_log += run(resumed, 8, False)
    (tmp_path / "curriculum_smoke.log").write_text(complete_log + "\nSAVE/RESUME REPLAY\n" + resumed_log)
    regex = r"Curriculum ([ABCD]): successful_step=(\d+) next_successful_step=(\d+) lambda_v=([\d.e+-]+) lambda_r=([\d.e+-]+) forwards=(\d+)"
    full_rows, resumed_rows = re.findall(regex, complete_log), re.findall(regex, resumed_log)
    assert len(full_rows) == 16 and full_rows == resumed_rows
    for region, step, next_step, lv, lr, forwards in full_rows:
        assert int(next_step) == int(step) + 1
        assert int(forwards) == (3 if float(lr) > 0 and region != "D" else 1)
        if float(lr) > 0: assert float(lv) == 1
    assert full_rows[0][3:] == ("0", "0", "1")
    assert full_rows[-1][3:] == ("1", "1", "1")
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
    assert "best_val_mse" not in checkpoint and not (resumed / "best.pt").exists()
    assert all(r["amp_skipped_steps"] == r["nonfinite_events"] == 0 for r in history)
    assert all(torch.isfinite(torch.tensor(r["gradient_norm_max"])) for r in history)
    assert history[-1]["curriculum"]["two_forward_fraction"] == 0
    assert history[-1]["phase_d"]["finite_target_fraction"] < 1
    assert all(r["learning_rate"] == train.learning_rate_for_epoch(r["epoch"], 20, 2e-5, 1, 2e-6)
               for r in history[:6])
    assert [r["learning_rate"] for r in history[6:]] == pytest.approx([3.6e-6, 5.2e-6])
    print(complete_log)
    print("Save/resume at steps 4 and 10 exactly reproduced all 16 lambda pairs; checkpoints:", resumed)
