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
    apply_ready_curriculum_transition,
    apply_ready_phase_d_transition,
    build_epoch_wandb_log,
    curriculum_is_complete,
    curriculum_learning_rate_for_epoch,
    curriculum_phase,
    curriculum_weights,
    global_cosine_learning_rate,
    save_checkpoint,
    start_phase_d_if_ready,
    train_one_epoch,
    validate,
    positive_metric_convergence,
    require_complete_curriculum,
    phase_a_convergence,
    nll_metric_convergence,
    phase_b_convergence,
    phase_c_convergence,
    symmetric_lr_cycle,
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


def test_transition_history_resume_is_exact_and_detector_is_pure(tmp_path):
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
    assert state == initial_state  # The epoch-boundary actuator, not the detector, mutates ramp state.
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


def test_transition_cli_defaults_and_curriculum_resume_compatibility(monkeypatch):
    from train_masked_unet3d import parse_args, validate_resume_parameters
    options = {"transition_ema_half_life": 5., "transition_rate_window": 5,
               "transition_rate_threshold": .005, "transition_patience": 4}
    for mode in ([], ["--uncertainty-aware"], ["--uncertainty-aware", "--uncertainty-loss", "gaussian_nll"],
                 ["--curriculum-training", "--uncertainty-aware", "--uncertainty-loss", "gaussian_nll"]):
        monkeypatch.setattr(sys, "argv", ["train"] + mode)
        args = parse_args()
        assert all(getattr(args, name) == value for name, value in options.items())
        saved = vars(args).copy()
        validate_resume_parameters(args, saved)
        args.transition_rate_threshold = .001
        if args.curriculum_training:
            with pytest.raises(ValueError, match="transition_rate_threshold"):
                validate_resume_parameters(args, saved)
        else:
            validate_resume_parameters(args, saved)
    monkeypatch.setattr(sys, "argv", ["train", "--curriculum-training", "--uncertainty-aware",
                                      "--uncertainty-loss", "gaussian_nll", "--final-epoch", "200"])
    args = parse_args()
    validate_resume_parameters(args, dict(vars(args), final_epoch=240))
    with pytest.raises(ValueError, match="final_epoch"):
        validate_resume_parameters(args, dict(vars(args), final_epoch=220))
    for flag, value in (("--transition-ema-half-life", "0"), ("--transition-ema-half-life", "nan"),
                        ("--transition-rate-window", "0"), ("--transition-rate-threshold", "-1"),
                        ("--transition-patience", "0")):
        monkeypatch.setattr(sys, "argv", ["train", flag, value])
        with pytest.raises(SystemExit): parse_args()


def test_abcd_symmetric_cli_configuration(monkeypatch):
    from train_masked_unet3d import parse_args

    monkeypatch.setattr(sys, "argv", [
        "train", "--curriculum-training", "--curriculum-config", "abcd_symmetric_lr",
        "--uncertainty-aware", "--uncertainty-loss", "gaussian_nll",
        "--epochs", "45", "--warmup-epochs", "5", "--lr", "2e-5",
        "--min-lr", "2e-6", "--variance-ramp-epochs", "40",
        "--rollout-ramp-epochs", "40", "--final-epoch", "240",
        "--stop-after-epochs", "240", "--no-auto-resume",
    ])
    args = parse_args()
    assert args.curriculum_config == "abcd_symmetric_lr"
    assert (args.epochs, args.warmup_epochs, args.final_epoch,
            args.stop_after_epochs) == (45, 5, 240, 240)


def test_global_cosine_cli_configuration(monkeypatch):
    from train_masked_unet3d import parse_args

    monkeypatch.setattr(sys, "argv", [
        "train", "--curriculum-training", "--curriculum-config", "global_cosine",
        "--uncertainty-aware", "--uncertainty-loss", "gaussian_nll",
        "--epochs", "240", "--warmup-epochs", "5", "--lr", "2e-4",
        "--min-lr", "1e-6", "--variance-ramp-epochs", "40",
        "--rollout-ramp-epochs", "40", "--final-epoch", "240",
        "--stop-after-epochs", "240", "--no-auto-resume",
    ])
    args = parse_args()
    assert args.curriculum_config == "global_cosine"
    assert (args.epochs, args.warmup_epochs, args.lr, args.min_lr,
            args.final_epoch, args.stop_after_epochs) == (240, 5, 2e-4, 1e-6, 240, 240)


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


def test_phase_b_history_resume_is_exact_and_detector_is_pure(tmp_path):
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


def test_ready_detectors_automatically_arm_existing_cosine_ramps():
    from train_masked_unet3d import curriculum_weights, make_curriculum_state

    args = SimpleNamespace(curriculum_training=True, epochs=20, min_lr=2e-6,
                           variance_ramp_epochs=40., rollout_ramp_epochs=40.,
                           transition_ema_half_life=5., transition_rate_window=5,
                           transition_rate_threshold=.005, transition_patience=4)
    state = make_curriculum_state(args, updates_per_epoch=10)
    state["successful_optimizer_steps"] = 300
    history = [{"epoch": epoch, "val_mse": 1., "val_gaussian_nll": -1.,
                "learning_rate": 2e-6,
                "curriculum": {"successful_optimizer_steps": epoch * 10}}
               for epoch in range(1, 31)]

    phase_a = phase_a_convergence(history, args, state)
    assert phase_a["ready_for_variance_ramp"]
    assert apply_ready_curriculum_transition(state, phase_a) == ("variance", 300)
    assert curriculum_weights(state) == (0., 0.)
    state["successful_optimizer_steps"] += 1
    assert 0 < curriculum_weights(state)[0] < 1 and curriculum_weights(state)[1] == 0

    variance_end = state["variance_ramp_start_step"] + state["variance_ramp_steps"]
    state["successful_optimizer_steps"] = variance_end
    for epoch in range(31, 45):
        history.append({"epoch": epoch, "val_mse": 1., "val_gaussian_nll": -1.,
                        "learning_rate": 2e-6,
                        "curriculum": {"successful_optimizer_steps": variance_end}})
    phase_b = phase_b_convergence(history, args, state)
    assert phase_b["ready_for_rollout_ramp"]
    assert apply_ready_curriculum_transition(state, phase_b) == ("rollout", variance_end)
    assert curriculum_weights(state) == (1., 0.)
    state["successful_optimizer_steps"] += state["rollout_ramp_steps"]
    assert curriculum_weights(state) == (1., 1.)
    assert curriculum_is_complete(state)
    require_complete_curriculum(state, epoch=200)


def test_phase_d_restores_old_checkpoint_and_has_exact_lr_trajectory():
    from train_masked_unet3d import make_curriculum_state

    args = SimpleNamespace(
        epochs=20, warmup_epochs=1, lr=2e-5, min_lr=2e-6, final_epoch=200,
        variance_ramp_epochs=40., rollout_ramp_epochs=40.,
        variance_ramp_start_epoch=None, rollout_ramp_start_epoch=None,
        arm_variance_ramp=False, arm_rollout_ramp=False,
    )
    # This is the current epoch-121 checkpoint state, minus the new Phase-D key.
    saved = {
        "updates_per_epoch": 1069,
        "variance_ramp_start_step": 32070,
        "variance_ramp_armed": True,
        "variance_ramp_steps": 42760.,
        "rollout_ramp_start_step": 88727,
        "rollout_ramp_armed": True,
        "rollout_ramp_steps": 42760.,
        "successful_optimizer_steps": 129349,
    }
    state = make_curriculum_state(args, 1069, {"curriculum_state": saved})
    assert state["phase_d_start_epoch"] is None
    assert state["successful_optimizer_steps"] == 129349
    assert curriculum_learning_rate_for_epoch(122, args, state) == 2e-6
    assert not start_phase_d_if_ready(state, completed_epoch=121, final_epoch=200)

    state["successful_optimizer_steps"] = 131487
    assert curriculum_is_complete(state)
    assert start_phase_d_if_ready(state, completed_epoch=123, final_epoch=200)
    assert state["phase_d_start_epoch"] == 124
    assert not start_phase_d_if_ready(state, completed_epoch=124, final_epoch=200)

    expected_reheat = {124: 3.6e-6, 125: 5.2e-6, 126: 6.8e-6,
                       127: 8.4e-6, 128: 1e-5}
    for epoch, expected in expected_reheat.items():
        assert curriculum_learning_rate_for_epoch(epoch, args, state) == pytest.approx(expected)
    assert curriculum_learning_rate_for_epoch(129, args, state) == 1e-5
    cosine_epochs = list(range(129, 169))
    assert len(cosine_epochs) == 40
    cosine_lrs = [curriculum_learning_rate_for_epoch(epoch, args, state)
                  for epoch in cosine_epochs]
    assert all(left > right for left, right in zip(cosine_lrs, cosine_lrs[1:]))
    assert curriculum_learning_rate_for_epoch(167, args, state) > 2e-6
    assert curriculum_learning_rate_for_epoch(168, args, state) == 2e-6
    assert all(curriculum_learning_rate_for_epoch(epoch, args, state) == 2e-6
               for epoch in range(169, 201))
    assert curriculum_weights(state) == (1., 1.)
    assert curriculum_phase(123, state, 1., 1.) == (2, "rollout")
    assert curriculum_phase(124, state, 1., 1.) == (3, "joint_refinement")

    resumed = make_curriculum_state(
        args, 1069, {"curriculum_state": dict(state)},
    )
    assert resumed["phase_d_start_epoch"] == 124
    assert [curriculum_learning_rate_for_epoch(epoch, args, resumed)
            for epoch in range(124, 201)] == [
                curriculum_learning_rate_for_epoch(epoch, args, state)
                for epoch in range(124, 201)
            ]


def _abcd_args():
    return SimpleNamespace(
        curriculum_training=True, curriculum_config="abcd_symmetric_lr",
        epochs=45, warmup_epochs=5, lr=2e-5, min_lr=2e-6,
        final_epoch=240, variance_ramp_epochs=40., rollout_ramp_epochs=40.,
        variance_ramp_start_epoch=None, rollout_ramp_start_epoch=None,
        arm_variance_ramp=False, arm_rollout_ramp=False,
        transition_ema_half_life=5., transition_rate_window=5,
        transition_rate_threshold=.005, transition_patience=4,
    )


def _global_cosine_args():
    return SimpleNamespace(
        curriculum_training=True, curriculum_config="global_cosine",
        epochs=240, warmup_epochs=5, lr=2e-4, min_lr=1e-6,
        final_epoch=240, variance_ramp_epochs=40., rollout_ramp_epochs=40.,
        variance_ramp_start_epoch=None, rollout_ramp_start_epoch=None,
        arm_variance_ramp=False, arm_rollout_ramp=False,
        transition_ema_half_life=5., transition_rate_window=5,
        transition_rate_threshold=.005, transition_patience=4,
    )


def test_global_cosine_lr_has_exact_endpoints_and_never_restarts():
    from train_masked_unet3d import make_curriculum_state

    args = _global_cosine_args()
    state = make_curriculum_state(args, updates_per_epoch=10)
    expected_warmup = [1e-6, 50.75e-6, 100.5e-6, 150.25e-6, 2e-4]
    lrs = [curriculum_learning_rate_for_epoch(epoch, args, state)
           for epoch in range(1, 241)]
    assert lrs[:5] == pytest.approx(expected_warmup)
    assert lrs[5] == pytest.approx(2e-4)
    assert lrs[-1] == pytest.approx(1e-6)
    assert all(left >= right for left, right in zip(lrs[4:], lrs[5:]))
    assert all(lr >= 1e-6 for lr in lrs)
    assert global_cosine_learning_rate(240, 240, 5, 1e-6, 2e-4) == 1e-6

    # Curriculum state, including a legacy Phase-D coordinate, cannot affect LR.
    for steps, phase_d_start in ((0, None), (400, None), (800, None), (1200, 100)):
        changed = dict(state, successful_optimizer_steps=steps,
                       phase_d_start_epoch=phase_d_start)
        assert [curriculum_learning_rate_for_epoch(epoch, args, changed)
                for epoch in range(1, 241)] == lrs


def test_global_cosine_phase_a_monitors_after_warmup_while_lr_decays():
    from train_masked_unet3d import make_curriculum_state

    args = _global_cosine_args()
    state = make_curriculum_state(args, updates_per_epoch=10)
    history = [
        {"epoch": epoch, "val_mse": 1.,
         "learning_rate": curriculum_learning_rate_for_epoch(epoch, args, state)}
        for epoch in range(1, 16)
    ]
    assert all(not phase_a_convergence(history[:end], args, state)[
        "ready_for_variance_ramp"] for end in range(1, 15))
    ready = phase_a_convergence(history, args, state)
    assert ready["segment_start_epoch"] == 5 and ready["segment_age"] == 10
    assert ready["transition_monitor_active"] and not ready["lr_floor_reached"]
    assert history[-1]["learning_rate"] > args.min_lr
    assert ready["ready_for_variance_ramp"]
    assert apply_ready_curriculum_transition(state, ready) == ("variance", 0)
    assert curriculum_weights(state) == (0., 0.)


def test_global_cosine_phase_b_controls_existing_rollout_ramp_and_resume():
    from train_masked_unet3d import make_curriculum_state

    args = _global_cosine_args()
    state = make_curriculum_state(args, updates_per_epoch=10)
    state.update(variance_ramp_armed=True, variance_ramp_start_step=100,
                 successful_optimizer_steps=500)
    history = [
        {"epoch": epoch, "val_mse": .2, "val_gaussian_nll": -.5,
         "learning_rate": curriculum_learning_rate_for_epoch(epoch, args, state),
         "curriculum": {"successful_optimizer_steps": 500}}
        for epoch in range(60, 74)
    ]
    ready = phase_b_convergence(history, args, state)
    assert ready["ready_for_rollout_ramp"]
    assert apply_ready_curriculum_transition(state, ready) == ("rollout", 500)
    assert state["variance_ramp_steps"] == state["rollout_ramp_steps"] == 400
    assert curriculum_weights(dict(state, successful_optimizer_steps=700)) == (1., pytest.approx(.5))
    assert curriculum_weights(dict(state, successful_optimizer_steps=900)) == (1., 1.)

    baseline_lrs = [curriculum_learning_rate_for_epoch(epoch, args, state)
                    for epoch in range(1, 241)]
    resumed = make_curriculum_state(
        args, updates_per_epoch=10, checkpoint={"curriculum_state": dict(state)},
    )
    assert resumed == state
    assert [curriculum_learning_rate_for_epoch(epoch, args, resumed)
            for epoch in range(1, 241)] == baseline_lrs
    assert curriculum_weights(resumed) == curriculum_weights(state)
    assert curriculum_phase(1, state, 0., 0., "global_cosine") == (0, "mean_only")
    assert curriculum_phase(60, state, .5, 0., "global_cosine") == (
        1, "variance_ramp_or_settle")
    assert curriculum_phase(100, state, 1., .5, "global_cosine") == (2, "rollout_ramp")
    assert curriculum_phase(140, state, 1., 1., "global_cosine") == (3, "full_objective")


def test_abcd_phase_a_and_phase_d_use_identical_fixed_lr_cycles():
    from train_masked_unet3d import make_curriculum_state

    args = _abcd_args()
    state = make_curriculum_state(args, updates_per_epoch=10)
    phase_a = [curriculum_learning_rate_for_epoch(epoch, args, state)
               for epoch in range(1, 46)]
    assert phase_a[:5] == pytest.approx([2e-6, 6.5e-6, 11e-6, 15.5e-6, 2e-5])
    assert phase_a[5] == pytest.approx(2e-5)
    assert phase_a[-1] == pytest.approx(2e-6)
    assert len(phase_a[5:]) == 40

    state["phase_d_start_epoch"] = 150
    phase_d = [curriculum_learning_rate_for_epoch(epoch, args, state)
               for epoch in range(150, 195)]
    assert phase_d == pytest.approx(phase_a)
    assert all(curriculum_learning_rate_for_epoch(epoch, args, state) == 2e-6
               for epoch in range(195, 241))
    assert symmetric_lr_cycle(194, 150, 2e-6, 2e-5) == pytest.approx(2e-6)


def test_abcd_phase_a_waits_for_floor_and_existing_readiness_detector():
    from train_masked_unet3d import make_curriculum_state

    args = _abcd_args()
    state = make_curriculum_state(args, updates_per_epoch=10)
    history = [
        {"epoch": epoch, "val_mse": 1.,
         "learning_rate": curriculum_learning_rate_for_epoch(epoch, args, state)}
        for epoch in range(1, 56)
    ]
    assert all(not phase_a_convergence(history[:epoch], args, state)[
        "ready_for_variance_ramp"] for epoch in range(1, 55))
    ready = phase_a_convergence(history, args, state)
    assert ready["segment_start_epoch"] == 45
    assert ready["segment_age"] == 10
    assert ready["ready_for_variance_ramp"]


def _phase_c_example():
    from train_masked_unet3d import make_curriculum_state

    args = _abcd_args()
    state = make_curriculum_state(args, updates_per_epoch=10)
    state.update(
        variance_ramp_armed=True, variance_ramp_start_step=200,
        rollout_ramp_armed=True, rollout_ramp_start_step=800,
        successful_optimizer_steps=1200,
    )
    history = [
        {
            "epoch": epoch,
            "val_mse": math.exp(.1 * epoch),
            "val_gaussian_nll": -10. - epoch,
            "val_rollout_mse": .2,
            "val_rollout_gaussian_nll": -.5,
            "learning_rate": 2e-6,
            "curriculum": {"successful_optimizer_steps": 1200},
        }
        for epoch in range(100, 114)
    ]
    return args, state, history


def test_abcd_existing_successful_step_ramps_and_phase_c_rollout_gate():
    args, state, history = _phase_c_example()
    assert state["variance_ramp_steps"] == state["rollout_ramp_steps"] == 400
    assert curriculum_weights(dict(state, successful_optimizer_steps=600)) == (1., 0.)
    assert curriculum_weights(dict(state, successful_optimizer_steps=1000))[1] == pytest.approx(.5)
    before = dict(state, successful_optimizer_steps=1199)
    assert not phase_c_convergence(history, args, before)["phase_c"]

    for end in range(1, 14):
        assert not phase_c_convergence(history[:end], args, state)["ready_for_phase_d"]
    ready = phase_c_convergence(history, args, state)
    assert ready["phase_c"] and ready["rollout_ramp_complete"]
    assert ready["segment_start_epoch"] == 100 and ready["segment_age"] == 13
    assert ready["stable_count"] == 4 and ready["ready_for_phase_d"]
    # Deliberately divergent first-pass metrics above do not enter this gate.
    assert ready["rollout_mse_rate_per_epoch"] == pytest.approx(0.)
    assert ready["rollout_nll_rate_per_epoch"] == pytest.approx(0.)

    changed = [dict(row) for row in history]
    changed[-1]["val_rollout_mse"] = 10.
    assert not phase_c_convergence(changed, args, state)["ready_for_phase_d"]
    changed = [dict(row) for row in history]
    changed[-1]["val_rollout_gaussian_nll"] = 10.
    assert not phase_c_convergence(changed, args, state)["ready_for_phase_d"]


def test_abcd_phase_d_requires_phase_c_readiness_and_resume_is_exact(tmp_path):
    import json
    from train_masked_unet3d import make_curriculum_state

    args, state, history = _phase_c_example()
    diagnostic = phase_c_convergence(history, args, state)
    path = tmp_path / "history.json"
    path.write_text(json.dumps(history))
    resumed_history = json.loads(path.read_text())
    resumed = make_curriculum_state(
        args, updates_per_epoch=10, checkpoint={"curriculum_state": dict(state)},
    )
    assert resumed == state
    assert phase_c_convergence(resumed_history, args, resumed) == diagnostic

    assert not apply_ready_phase_d_transition(
        resumed, {"ready_for_phase_d": False}, completed_epoch=113, final_epoch=240,
    )
    assert resumed["phase_d_start_epoch"] is None
    assert apply_ready_phase_d_transition(
        resumed, diagnostic, completed_epoch=113, final_epoch=240,
    )
    assert resumed["phase_d_start_epoch"] == 114
    assert not apply_ready_phase_d_transition(
        resumed, diagnostic, completed_epoch=114, final_epoch=240,
    )
    checkpoint_resumed = make_curriculum_state(
        args, updates_per_epoch=10, checkpoint={"curriculum_state": dict(resumed)},
    )
    assert checkpoint_resumed == resumed
    assert [curriculum_learning_rate_for_epoch(epoch, args, checkpoint_resumed)
            for epoch in range(114, 241)] == [
                curriculum_learning_rate_for_epoch(epoch, args, resumed)
                for epoch in range(114, 241)
            ]


def test_incomplete_curriculum_is_rejected_as_official_final():
    from train_masked_unet3d import make_curriculum_state

    args = SimpleNamespace(variance_ramp_epochs=40., rollout_ramp_epochs=40.)
    state = make_curriculum_state(args, updates_per_epoch=10)
    with pytest.raises(RuntimeError, match="final epoch 200 reached before the curriculum completed"):
        require_complete_curriculum(state, epoch=200)
    assert not curriculum_is_complete(state)
    assert state["successful_optimizer_steps"] == 0


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
    save_checkpoint(path, model, optimizer, epoch=2,
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
    save_checkpoint(path, model, optimizer, 1, SimpleNamespace(), {}, scaler=scaler)
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
        curriculum_state=state, split_sha256=args.split_sha256)
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


class TinyUncertaintyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, model_input, detach_variance_features=False):
        del detach_variance_features
        prediction = model_input[:, :4] * self.scale
        variance = torch.ones_like(prediction)
        return prediction, variance


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


def test_validate_emits_second_pass_rollout_objectives_for_phase_c():
    device = torch.device("cpu")
    model = TinyUncertaintyModel()
    loader = DataLoader(TinyBlockDataset(n=1), batch_size=1)
    mean, std = _channel_stats(device)
    metrics = validate(
        model=model, loader=loader, device=device, mean=mean, std=std,
        patterns=["spatial_random"], mask_seed=4321,
        density_probe_min=0, density_probe_max=2, amp=False,
        uncertainty_aware=True, uncertainty_loss="gaussian_nll",
        rollout_validation=True,
    )
    for scope in ("spatial_random", "mean"):
        assert set(metrics[scope]) == {
            "mse", "gaussian_nll", "rollout_mse", "rollout_gaussian_nll",
        }
        assert metrics[scope]["rollout_mse"] >= 0
        assert all(math.isfinite(value) for value in metrics[scope].values())


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


def test_wandb_epoch_log_includes_curriculum_phase_and_convergence_state():
    train_metrics = {
        "magnetic_pattern_share": {name: 0.2 for name in MASK_PATTERNS},
        "density_pattern_share": {name: 0.2 for name in MASK_PATTERNS},
    }
    val_metrics = {pattern: {"mse": 0.1} for pattern in MASK_PATTERNS}
    val_metrics["mean"] = {"mse": 0.1}
    row = {
        "epoch": 30, "learning_rate": 2e-6, "train_mse": 0.1, "val_mse": 0.1,
        "train_mask_target_fraction": 0.5, "train_mask_actual_fraction": 0.5,
        "train_density_probe_count": 10., "train_density_visible_fraction": 0.1,
        "train_density_zero_sample_share": 0.05, "train_density_full_sample_share": 0.05,
        "train_magnetic_visible_count": 20., "train_magnetic_visible_fraction": 0.2,
        "train_magnetic_zero_sample_share": 0.05, "train_magnetic_full_sample_share": 0.05,
        "curriculum": {
            "lambda_v_start": 0., "lambda_v_mean": 0., "lambda_v_end": 0.,
            "lambda_r_start": 0., "lambda_r_mean": 0., "lambda_r_end": 0.,
            "phase_index": 0, "phase_name": "reconstruction",
            "convergence_rate_threshold": .005, "convergence_patience_target": 4,
        },
        "phase_convergence": {
            "smoothed_rate_per_epoch": -.001, "stable_count": 4, "segment_age": 10,
            "ready_for_variance_ramp": True,
        },
    }
    log_data = build_epoch_wandb_log(row, train_metrics, val_metrics, list(MASK_PATTERNS))
    expected = {
        "optimization/learning_rate": 2e-6,
        "curriculum/lambda_v_start": 0., "curriculum/lambda_v_mean": 0.,
        "curriculum/lambda_v_end": 0., "curriculum/lambda_r_start": 0.,
        "curriculum/lambda_r_mean": 0., "curriculum/lambda_r_end": 0.,
        "curriculum/phase_index": 0, "curriculum/phase_name": "reconstruction",
        "curriculum/convergence_rate_threshold": .005,
        "curriculum/convergence_patience_target": 4,
        "curriculum/mse_rate_per_epoch": -.001,
        "curriculum/convergence_stable_count": 4,
        "curriculum/convergence_segment_age": 10,
        "curriculum/ready_for_variance_ramp": True,
        "curriculum/ready_for_rollout_ramp": False,
    }
    assert {key: log_data[key] for key in expected} == expected

    row.pop("phase_convergence")
    row["phase_b_convergence"] = {
        "mse_rate_per_epoch": -.002, "nll_rate_per_epoch": .003,
        "stable_count": 3, "segment_age": 12, "ready_for_rollout_ramp": False,
    }
    log_data = build_epoch_wandb_log(row, train_metrics, val_metrics, list(MASK_PATTERNS))
    assert log_data["curriculum/mse_rate_per_epoch"] == -.002
    assert log_data["curriculum/nll_rate_per_epoch"] == .003
    assert log_data["curriculum/convergence_stable_count"] == 3
    assert not log_data["curriculum/ready_for_rollout_ramp"]


def test_resume_history_filter_tolerates_old_mae_fields():
    history = [
        {"epoch": 1, "train_mse": 0.2, "train_mae": 0.1, "val_mse": 0.3, "val_mae": 0.2},
        {"epoch": 2, "train_mse": 0.1, "train_mae": 0.05, "val_mse": 0.2, "val_mae": 0.1},
    ]
    start_epoch = 2
    filtered = [row for row in history if row.get("epoch", 0) < start_epoch]
    assert [row["epoch"] for row in filtered] == [1]
    assert filtered[0]["val_mse"] == 0.3


def test_validation_metrics_do_not_select_a_checkpoint():
    source = Path(__file__).resolve().parents[1].joinpath("train_masked_unet3d.py").read_text()
    assert 'out_dir / "best.pt"' not in source
    assert 'out_dir / "final.pt"' in source
