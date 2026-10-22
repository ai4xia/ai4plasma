from __future__ import annotations

import sys
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.masking import DEFAULT_PATTERN_WEIGHTS, MASK_PATTERNS  # noqa: E402
from train_masked_unet3d import (  # noqa: E402
    build_epoch_wandb_log,
    save_checkpoint,
    train_one_epoch,
    validate,
    positive_metric_convergence,
    phase_a_convergence,
    nll_metric_convergence,
    phase_b_convergence,
)


def test_transition_half_life_and_ema_initialization():
    history = [{"epoch": 1, "val_mse": 1.}, {"epoch": 2, "val_mse": 2.}]
    for half_life in (1., 5.):
        alpha = 1 - 2 ** (-1 / half_life)
        rows = positive_metric_convergence(history, half_life=half_life)
        assert rows[0]["ema_value"] == 1
        assert rows[1]["ema_value"] == pytest.approx(math.exp(alpha * math.log(2)))
    assert 1 - 2 ** (-1 / 5) == pytest.approx(.12944943670387588)


@pytest.mark.parametrize("factor,sign", [(1., 0), (.98, -1), (1.02, 1)])
def test_transition_constant_and_signed_geometric_trends(factor, sign):
    history = [{"epoch": e, "val_mse": factor ** e} for e in range(1, 61)]
    diagnostics = positive_metric_convergence(history)
    assert all(r["smoothed_rate_per_epoch"] is None for r in diagnostics[:5])
    rate = diagnostics[-1]["smoothed_rate_per_epoch"]
    if sign == 0:
        assert rate == 0 and diagnostics[-1]["converged"]
    else:
        assert rate * sign > 0
        assert rate == pytest.approx(factor - 1, abs=1e-5)
        assert not diagnostics[-1]["converged"]


def test_transition_threshold_patience_and_invalid_history():
    history = [{"epoch": e, "val_mse": 1.} for e in range(1, 10)]
    rows = positive_metric_convergence(history)
    assert rows[5]["stable_count"] == 1
    assert rows[7]["stable_count"] == 3 and not rows[7]["converged"]
    assert rows[8]["stable_count"] == 4 and rows[8]["converged"]
    assert not rows[8]["burn_in_complete"]  # Burn-in does not alter the patience criterion.
    assert not positive_metric_convergence(history, threshold=0)[-1]["converged"]  # Strict <.
    assert positive_metric_convergence(history + [{"epoch": 10, "val_mse": 10.}])[-1]["stable_count"] == 0
    for bad in (0., -1., float("nan"), float("inf"), None):
        row = positive_metric_convergence(history + [{"epoch": 10, "val_mse": bad}])[-1]
        assert row["ema_value"] is row["smoothed_rate_per_epoch"] is None
        assert row["stable_count"] == 0 and not row["converged"]
    assert positive_metric_convergence(history + [{"epoch": 11, "val_mse": 1.}])[-1]["smoothed_rate_per_epoch"] is None


def test_transition_default_threshold_is_point_five_percent():
    history = [{"epoch": e, "val_mse": .996 ** e} for e in range(1, 61)]
    default = positive_metric_convergence(history)
    assert default == positive_metric_convergence(history, threshold=.005)
    assert default[-1]["converged"]
    old_threshold = positive_metric_convergence(history, threshold=.003)
    assert not old_threshold[-1]["converged"]
    assert default[-1]["smoothed_rate_per_epoch"] == old_threshold[-1]["smoothed_rate_per_epoch"]


def test_transition_readiness_phase_and_lr_guards():
    from train_masked_unet3d import make_curriculum_state
    args = SimpleNamespace(curriculum_training=True, epochs=20, min_lr=2e-6,
                           variance_ramp_epochs=40., rollout_ramp_epochs=40.)
    state = make_curriculum_state(args, 10)
    history = [{"epoch": e, "val_mse": 1., "learning_rate": 2e-6} for e in range(1, 31)]
    assert phase_a_convergence(history, args, state)["ready_for_variance_ramp"]
    assert not phase_a_convergence(history[:-1], args, state)["ready_for_variance_ramp"]
    assert phase_a_convergence(history[:-1], args, state)["converged"]
    assert all(not phase_a_convergence(history[:e], args, state)["ready_for_variance_ramp"] for e in range(20, 30))
    assert phase_a_convergence(history, args, state)["segment_age"] == 10
    assert not phase_a_convergence(history[:19], args, state)["ready_for_variance_ramp"]
    assert phase_a_convergence(history[:19], args, state)["ema_val_mse"] is None
    start = phase_a_convergence(history[:20], args, state)
    assert start["ema_val_mse"] == 1 and start["smoothed_rate_per_epoch"] is None
    assert start["segment_start_epoch"] == 20 and start["stable_count"] == 0
    assert not phase_a_convergence([dict(r, learning_rate=3e-6) for r in history], args, state)["ready_for_variance_ramp"]
    for prefix in ("variance", "rollout"):
        # Even an armed ramp scheduled in the future at lambda=0 blocks readiness.
        armed = dict(state, **{f"{prefix}_ramp_armed": True, f"{prefix}_ramp_start_step": 100})
        assert not phase_a_convergence(history, args, armed)["ready_for_variance_ramp"]
        armed["successful_optimizer_steps"] = 200
        assert not phase_a_convergence(history, args, armed)["ready_for_variance_ramp"]
    assert not phase_a_convergence(history, args, None)["ready_for_variance_ramp"]
    args.curriculum_training = False
    assert not phase_a_convergence(history, args, state)["ready_for_variance_ramp"]


def test_transition_history_resume_is_exact_and_passive(tmp_path):
    import json
    from train_masked_unet3d import make_curriculum_state
    args = SimpleNamespace(curriculum_training=True, epochs=20, min_lr=2e-6,
                           variance_ramp_epochs=40., rollout_ramp_epochs=40.)
    state = make_curriculum_state(args, 10)
    initial_state = dict(state)
    rng = torch.get_rng_state().clone()
    history = [{"epoch": e, "val_mse": 1 + .001 * math.sin(e), "val_gaussian_nll": -2.,
                "learning_rate": 2e-6} for e in range(1, 31)]
    expected = [phase_a_convergence(history[:e], args, state) for e in range(1, 31)]
    path = tmp_path / "history.json"
    # Old histories have no diagnostic fields; later diagnostic fields are ignored on replay.
    path.write_text(json.dumps(history[:23]))
    resumed = json.loads(path.read_text())
    actual = expected[:23]
    for row in history[23:]:
        resumed.append(row.copy())
        actual.append(phase_a_convergence(resumed, args, state))
        resumed[-1]["phase_convergence"] = actual[-1]
    assert actual == expected
    assert expected[-1]["ready_for_variance_ramp"]
    assert state == initial_state  # Advisory readiness never arms a ramp or advances a counter.
    assert torch.equal(torch.get_rng_state(), rng)


def test_transition_segments_discard_previous_regime_memory():
    history = [{"epoch": e, "val_mse": 100. if e < 20 else .1} for e in range(1, 31)]
    segmented = positive_metric_convergence(history, segment_start_epoch=20)
    assert all(row["ema_value"] is None and row["stable_count"] == 0 for row in segmented[:19])
    assert segmented[19]["ema_value"] == pytest.approx(.1)
    assert segmented[19:] == positive_metric_convergence(history[19:])
    assert segmented[23]["smoothed_rate_per_epoch"] is None
    assert segmented[24]["smoothed_rate_per_epoch"] == pytest.approx(0., abs=1e-14)
    # A later phase can explicitly start a fresh segment after its ramp completes.
    assert positive_metric_convergence(history, segment_start_epoch=27)[26:] == positive_metric_convergence(history[26:])


@pytest.mark.parametrize("half_life,start", [(5., 20), (2., 50), (2.5, 100)])
def test_transition_burn_in_is_generic_and_readiness_only(half_life, start):
    age_required = 2 * half_life
    history = [{"epoch": e, "val_mse": 1.} for e in range(start, start + int(age_required) + 1)]
    rows = positive_metric_convergence(history, half_life=half_life, window=1, patience=1,
                                      segment_start_epoch=start)
    assert all(not r["burn_in_complete"] for r in rows[:-1])
    assert rows[-1]["burn_in_complete"] and rows[-1]["segment_age"] == age_required
    assert all(r["minimum_segment_age"] == age_required for r in rows)
    assert all(r["ema_value"] == 1 for r in rows)
    assert all(r["smoothed_rate_per_epoch"] == 0 and r["converged"] for r in rows[1:])
    assert [r["stable_count"] for r in rows] == list(range(len(rows)))


def test_transition_phase_a_resets_when_lr_regime_changes():
    from train_masked_unet3d import make_curriculum_state
    args = SimpleNamespace(curriculum_training=True, epochs=20, min_lr=2e-6,
                           variance_ramp_epochs=40., rollout_ramp_epochs=40.)
    state = make_curriculum_state(args, 10)
    history = [{"epoch": e, "val_mse": 10. if e < 25 else .1,
                "learning_rate": 3e-6 if e in (23, 24) else 2e-6} for e in range(1, 31)]
    reset = phase_a_convergence(history[:25], args, state)
    assert reset["segment_start_epoch"] == 25 and reset["ema_val_mse"] == pytest.approx(.1)
    assert reset["segment_age"] == 0 and not reset["burn_in_complete"]
    assert reset["smoothed_rate_per_epoch"] is None and reset["stable_count"] == 0
    final = phase_a_convergence(history, args, state)
    assert final["smoothed_rate_per_epoch"] == pytest.approx(0., abs=1e-14)
    assert final["stable_count"] == 1 and not final["ready_for_variance_ramp"]


def test_transition_cli_defaults_and_old_checkpoint_compatibility(monkeypatch):
    from train_masked_unet3d import parse_args, validate_resume_parameters
    options = {"transition_ema_half_life": 5., "transition_rate_window": 5,
               "transition_rate_threshold": .005, "transition_patience": 4}
    for mode in ([], ["--uncertainty-aware"], ["--uncertainty-aware", "--uncertainty-loss", "gaussian_nll"],
                 ["--curriculum-training", "--uncertainty-aware", "--uncertainty-loss", "gaussian_nll"]):
        monkeypatch.setattr(sys, "argv", ["train"] + mode)
        args = parse_args()
        assert all(getattr(args, name) == value for name, value in options.items())
        old = {key: value for key, value in vars(args).items() if key not in options}
        validate_resume_parameters(args, old)
        args.transition_rate_threshold = .001
        validate_resume_parameters(args, old)  # Advisory settings are not immutable training parameters.
    for flag, value in (("--transition-ema-half-life", "0"), ("--transition-ema-half-life", "nan"),
                        ("--transition-rate-window", "0"), ("--transition-rate-threshold", "-1"),
                        ("--transition-patience", "0")):
        monkeypatch.setattr(sys, "argv", ["train", flag, value])
        with pytest.raises(SystemExit): parse_args()


@pytest.mark.parametrize("slope,sign", [(-.02, -1), (0., 0), (.02, 1)])
def test_nll_convergence_signed_negative_trends(slope, sign):
    history = [{"epoch": e, "val_gaussian_nll": -2. + slope * e} for e in range(1, 61)]
    rows = nll_metric_convergence(history)
    assert rows[0]["ema_value"] == history[0]["val_gaussian_nll"]
    assert all(row["smoothed_rate_per_epoch"] is None for row in rows[:5])
    rate = rows[-1]["smoothed_rate_per_epoch"]
    if sign:
        assert rate * sign > 0
        assert rate == pytest.approx(math.expm1(slope), abs=1e-5)
    else:
        assert rate == pytest.approx(0., abs=1e-14)
    assert all(math.isfinite(row["ema_value"]) for row in rows)


def test_nll_convergence_crosses_zero_without_log(monkeypatch):
    history = [{"epoch": e, "val_gaussian_nll": value} for e, value in enumerate((-.02, 0., .02), 1)]
    monkeypatch.setattr(math, "log", lambda *a: pytest.fail("NLL must not be logged"))
    rows = nll_metric_convergence(history, half_life=1., window=1)
    assert [row["ema_value"] for row in rows] == pytest.approx([-.02, -.01, .005])
    assert rows[1]["smoothed_rate_per_epoch"] == pytest.approx(math.expm1(.01))
    assert rows[2]["smoothed_rate_per_epoch"] == pytest.approx(math.expm1(.015))


def _phase_b_example():
    from train_masked_unet3d import make_curriculum_state
    args = SimpleNamespace(curriculum_training=True, epochs=20, min_lr=2e-6,
                           variance_ramp_epochs=40., rollout_ramp_epochs=40.)
    state = make_curriculum_state(args, 10)
    state.update(variance_ramp_armed=True, variance_ramp_start_step=800, successful_optimizer_steps=1350)
    history = [{"epoch": e, "val_mse": 10. if e < 120 else .1,
                "val_gaussian_nll": 50. if e < 120 else -.4, "learning_rate": 2e-6,
                "curriculum": {"successful_optimizer_steps": e * 10, "lambda_v_end": .9999999986}}
               for e in range(80, 136)]
    return args, state, history


@pytest.mark.parametrize("unstable", [None, "val_mse", "val_gaussian_nll"])
def test_phase_b_requires_both_metrics_stable(unstable):
    args, state, history = _phase_b_example()
    if unstable:
        for row in history:
            if row["epoch"] >= 120:
                age = row["epoch"] - 120
                row[unstable] = .1 * math.exp(-.02 * age) if unstable == "val_mse" else -.4 - .02 * age
    result = phase_b_convergence(history, args, state)
    assert result["segment_start_epoch"] == 120 and result["phase_b"]
    assert result["ready_for_rollout_ramp"] == (unstable is None)
    assert result["stable_count"] == (6 if unstable is None else 0)


def test_phase_b_completion_burn_in_patience_and_guards():
    args, state, history = _phase_b_example()
    through = lambda epoch: [r for r in history if r["epoch"] <= epoch]
    assert not phase_b_convergence(through(119), args, state)["phase_b"]
    start = phase_b_convergence(through(120), args, state)
    assert start["phase_b"] and start["segment_start_epoch"] == 120
    assert start["ema_val_mse"] == pytest.approx(.1) and start["ema_val_nll"] == -.4
    assert start["segment_age"] == start["stable_count"] == 0
    assert start["mse_rate_per_epoch"] is start["nll_rate_per_epoch"] is None
    for epoch in range(120, 130):
        result = phase_b_convergence(through(epoch), args, state)
        assert result["stable_count"] == 0 and not result["ready_for_rollout_ramp"]
    assert phase_b_convergence(through(130), args, state)["stable_count"] == 1
    assert not phase_b_convergence(through(132), args, state)["ready_for_rollout_ramp"]
    assert phase_b_convergence(through(133), args, state)["ready_for_rollout_ramp"]
    for invalid in (dict(state, successful_optimizer_steps=1199),
                    dict(state, variance_ramp_armed=False), dict(state, rollout_ramp_armed=True)):
        assert not phase_b_convergence(history, args, invalid)["ready_for_rollout_ramp"]
    args.curriculum_training = False
    assert not phase_b_convergence(history, args, state)["ready_for_rollout_ramp"]


def test_phase_b_joint_patience_resets_if_either_metric_moves():
    args, state, history = _phase_b_example()
    # A change in either metric must break the shared stability streak.
    for metric in ("val_mse", "val_gaussian_nll"):
        changed = [dict(row) for row in history]
        changed[-2][metric] = 10.
        result = phase_b_convergence(changed, args, state)
        assert result["stable_count"] == 0 and not result["ready_for_rollout_ramp"]


def test_phase_b_history_resume_is_exact_and_passive(tmp_path):
    import json
    args, state, history = _phase_b_example()
    before = dict(state)
    rng = torch.get_rng_state().clone()
    expected = [phase_b_convergence(history[:i + 1], args, state) for i in range(len(history))]
    # Include epoch 120 and part of settling in the saved pre-monitor history.
    split = 48
    path = tmp_path / "history.json"
    path.write_text(json.dumps(history[:split]))
    resumed = json.loads(path.read_text())
    actual = expected[:split]
    for row in history[split:]:
        resumed.append(dict(row))
        actual.append(phase_b_convergence(resumed, args, state))
        resumed[-1]["phase_b_convergence"] = actual[-1]
    assert actual == expected and state == before
    assert torch.equal(torch.get_rng_state(), rng)
    missing = [dict(row) for row in history]
    missing[-2].pop("val_gaussian_nll")
    assert not phase_b_convergence(missing, args, state)["ready_for_rollout_ramp"]


def test_supplied_scaler_persists_across_epochs_and_checkpoints(tmp_path):
    from types import SimpleNamespace

    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=1)
    mean, std = _channel_stats(torch.device("cpu"))
    loader = DataLoader(TinyBlockDataset(n=1), batch_size=1)
    scales = []
    for epoch in (1, 2):
        metrics = train_one_epoch(
            model, loader, optimizer, torch.device("cpu"), mean, std,
            DEFAULT_PATTERN_WEIGHTS, .05, .05, .05, .05, False, 1.0,
            scaler=scaler,
        )
        scales.append((metrics["amp_scale_start"], metrics["amp_scale_end"]))
        assert metrics["optimizer_steps"] == 1
        assert metrics["amp_skipped_steps"] == 0
    assert scales == [(8.0, 16.0), (16.0, 32.0)]
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, epoch=2, best_val_mse=1.0,
                    args=SimpleNamespace(), stats={}, scaler=scaler)
    checkpoint = torch.load(path, weights_only=False)
    restored = torch.amp.GradScaler("cpu")
    restored.load_state_dict(checkpoint["scaler"])
    assert restored.state_dict() == scaler.state_dict()


def test_amp_precision_defaults_and_resume_compatibility(monkeypatch):
    from train_masked_unet3d import parse_args, validate_resume_parameters
    import pytest

    monkeypatch.setattr(sys, "argv", ["train", "--amp"])
    args = parse_args()
    assert args.amp_dtype == "float16"
    old_args = vars(args).copy()
    old_args.pop("amp_dtype")
    validate_resume_parameters(args, old_args)
    args.amp_dtype = "bfloat16"
    with pytest.raises(ValueError, match="amp_dtype"):
        validate_resume_parameters(args, old_args)
    validate_resume_parameters(args, dict(old_args, amp_dtype="bfloat16"))


@pytest.mark.parametrize("amp,device_type,dtype,scaling", [
    (True, "cuda", "bfloat16", False), (True, "cuda", "float16", True),
    (False, "cuda", "float16", False), (True, "cpu", "float16", False),
])
def test_amp_constructor_policy_and_autocast(amp, device_type, dtype, scaling):
    import ast
    import train_masked_unet3d as train

    # Evaluate the actual constructor arguments without initializing CUDA.
    tree = ast.parse(Path(train.__file__).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr in {"GradScaler", "autocast"}]
    assert sum(call.func.attr == "GradScaler" for call in calls) == 2
    assert sum(call.func.attr == "autocast" for call in calls) == 2
    fake_torch = SimpleNamespace(float16=torch.float16, cuda=SimpleNamespace(
        amp=SimpleNamespace(GradScaler=lambda **kw: kw, autocast=lambda **kw: kw)))
    for call in calls:
        options = eval(compile(ast.Expression(call), train.__file__, "eval"), {
            "torch": fake_torch, "amp": amp, "amp_dtype": getattr(torch, dtype),
            "device": torch.device(device_type),
            "args": SimpleNamespace(amp=amp, amp_dtype=dtype),
        })
        if call.func.attr == "GradScaler":
            assert options["enabled"] is scaling
        else:
            assert options == {"enabled": amp and device_type == "cuda",
                               "dtype": getattr(torch, dtype)}


@pytest.mark.parametrize("enabled", [False, True])
def test_resume_scaler_state_disabled_bf16_and_enabled_fp16(tmp_path, enabled):
    # CPU scaler exercises the same state API without requiring a GPU.
    scaler = torch.amp.GradScaler("cpu", enabled=enabled)
    historical = torch.amp.GradScaler("cpu", init_scale=1.0141204801825835e31).state_dict()
    scaler.load_state_dict(historical)
    assert scaler.get_scale() == (historical["scale"] if enabled else 1.0)
    assert bool(scaler.state_dict()) is enabled
    model = ScaleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scaler.scale(model.scale.square()).backward()
    scaler.unscale_(optimizer)
    assert model.scale.grad.item() == pytest.approx(1.0)
    scaler.step(optimizer)
    scaler.update()
    assert model.scale.item() == pytest.approx(.4)
    path = tmp_path / "latest.pt"
    save_checkpoint(path, model, optimizer, 1, 1., SimpleNamespace(), {}, scaler=scaler)
    assert ("scaler" in torch.load(path, weights_only=False)) is enabled


@pytest.mark.parametrize("dtype,scaler_state,missing_model,accepted", [
    ("bfloat16", "old", False, True), ("bfloat16", "missing", False, True),
    ("bfloat16", "empty", False, True), ("float16", "old", False, True),
    ("float16", "missing", False, False), ("float16", "empty", False, False),
    ("bfloat16", "missing", True, False),
])
def test_launcher_resume_scaler_preflight(tmp_path, monkeypatch, dtype, scaler_state,
                                        missing_model, accepted):
    import json
    import train_masked_unet3d as train

    monkeypatch.setattr(sys, "argv", ["train", "--out-dir", str(tmp_path), "--amp",
        "--amp-dtype", dtype, "--uncertainty-aware", "--uncertainty-loss", "gaussian_nll",
        "--curriculum-training", "--stop-after-epochs", "7"])
    args = train.parse_args()
    state = train.make_curriculum_state(args, 1)
    state["successful_optimizer_steps"] = 6
    checkpoint = dict(model={"weight": torch.ones(1)}, optimizer={"state": {}},
        stats={"mean": [0.] * 4, "std": [1.] * 4}, epoch=6, args=vars(args),
        curriculum_state=state, best_val_mse=.1)
    if scaler_state != "missing":
        checkpoint["scaler"] = (torch.amp.GradScaler("cpu", init_scale=1e31).state_dict()
                                if scaler_state == "old" else {})
    if missing_model:
        checkpoint.pop("model")
    torch.save(checkpoint, tmp_path / "latest.pt")
    (tmp_path / "history.json").write_text(json.dumps([{"epoch": e} for e in range(1, 7)]))
    # Execute only the launcher's existing Python preflight, never its srun command.
    launcher = Path(train.__file__).with_name("train_masked_unet3d_4n16g.sbatch").read_text()
    preflight = launcher.split('COMPLETED_EPOCH=$(', 1)[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    if accepted:
        exec(compile(preflight, "launcher_preflight", "exec"), {})
    else:
        with pytest.raises(ValueError, match="missing or empty checkpoint '(model|scaler)'"):
            exec(compile(preflight, "launcher_preflight", "exec"), {})


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
