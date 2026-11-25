# tests/test_adaptive_abcd_240ep.py

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
from pathlib import Path

import pytest
import torch

from models.unet3d import UNet3D
from train_masked_unet3d import (
    apply_ready_curriculum_transition,
    curriculum_is_complete,
    curriculum_learning_rate_for_epoch,
    curriculum_phase,
    curriculum_weights,
    global_cosine_learning_rate,
    is_slide_cosine_complete,
    make_curriculum_state,
    parse_args,
    phase_a_convergence,
    phase_b_convergence,
    phase_c_convergence,
    phase_d_convergence,
    require_complete_curriculum,
    sample_slide_step,
    save_checkpoint,
    slide_curriculum_weight,
)


def _make_240ep_args(epochs: int = 240) -> argparse.Namespace:
    return parse_args([
        "--out-dir", "/tmp/test_abcd_240",
        "--epochs", str(epochs),
        "--warmup-epochs", "5",
        "--lr", "2e-4",
        "--min-lr", "1e-5",
        "--curriculum-training",
        "--curriculum-config", "three_phase_global_cosine",
        "--uncertainty-aware",
        "--uncertainty-loss", "generalized_kl",
        "--student-t-df", "3.0",
        "--joint-nll-grad",
        "--differentiable-phase-c-reuse",
        "--adaptive-phase-d-sliding",
        "--slide-ramp-epochs", "40",
    ])


def test_adaptive_4phase_transitions_and_gkl_phase_c():
    """Verify A->B->C->D convergence-triggered transitions, 40-ep ramps, and incomplete guard."""
    args = _make_240ep_args(240)
    updates_per_ep = 100
    state = make_curriculum_state(args, updates_per_epoch=updates_per_ep)

    # Fresh start: all lambdas 0, all ramps unarmed
    assert curriculum_weights(state) == (0.0, 0.0)
    assert slide_curriculum_weight(state) == 0.0
    assert curriculum_phase(1, state, *curriculum_weights(state), "three_phase_global_cosine") == (0, "mean_only")
    assert all(sample_slide_step(slide_curriculum_weight(state), 24) == 0 for _ in range(10))

    # Simulate Phase A convergence (20 epochs stable val_mse)
    history = []
    for ep in range(1, 21):
        state["successful_optimizer_steps"] = ep * updates_per_ep
        history.append({
            "epoch": ep,
            "val_mse": 0.05,
            "val_generalized_kl": 0.05,
            "curriculum": {"successful_optimizer_steps": state["successful_optimizer_steps"]},
        })
    diag_a = phase_a_convergence(history, args, state)
    assert diag_a["phase_a"] and diag_a["ready_for_variance_ramp"]
    ramp, start_step = apply_ready_curriculum_transition(state, diag_a)
    assert ramp == "variance" and start_step == 20 * updates_per_ep

    # Advance 40 epochs for variance ramp (epochs 21..60) -> lambda_v == 1.0, lambda_r == 0.0
    for ep in range(21, 61):
        state["successful_optimizer_steps"] = ep * updates_per_ep
        history.append({
            "epoch": ep,
            "val_mse": 0.05,
            "val_generalized_kl": 0.04,
            "curriculum": {"successful_optimizer_steps": state["successful_optimizer_steps"]},
        })
    assert curriculum_weights(state) == (1.0, 0.0)
    assert slide_curriculum_weight(state) == 0.0

    # Settle Phase B (epochs 61..90) -> Phase B->C transition
    for ep in range(61, 91):
        state["successful_optimizer_steps"] = ep * updates_per_ep
        history.append({
            "epoch": ep,
            "val_mse": 0.04,
            "val_generalized_kl": 0.03,
            "curriculum": {"successful_optimizer_steps": state["successful_optimizer_steps"]},
        })
    diag_b = phase_b_convergence(history, args, state)
    assert diag_b["phase_b"] and diag_b["ready_for_rollout_ramp"]
    ramp, start_step = apply_ready_curriculum_transition(state, diag_b)
    assert ramp == "rollout" and start_step == 90 * updates_per_ep

    # Advance 40 epochs for rollout ramp (epochs 91..130)
    for ep in range(91, 131):
        state["successful_optimizer_steps"] = ep * updates_per_ep
        history.append({
            "epoch": ep,
            "val_mse": 0.03,
            "val_generalized_kl": 0.025,
            "val_rollout_mse": 0.02,
            "val_rollout_generalized_kl": 0.02,
            "curriculum": {"successful_optimizer_steps": state["successful_optimizer_steps"]},
        })
    assert curriculum_weights(state) == (1.0, 1.0)
    assert slide_curriculum_weight(state) == 0.0

    # Incomplete before Phase D completes
    assert not curriculum_is_complete(state)
    with pytest.raises(RuntimeError, match="lambda_slide=0"):
        require_complete_curriculum(state, 130)

    # Settle Phase C (rollout MSE + GKL stable for epochs 131..160) -> Phase C->D transition
    for ep in range(131, 161):
        state["successful_optimizer_steps"] = ep * updates_per_ep
        history.append({
            "epoch": ep,
            "val_mse": 0.03,
            "val_generalized_kl": 0.025,
            "val_rollout_mse": 0.015,
            "val_rollout_generalized_kl": 0.015,
            "curriculum": {"successful_optimizer_steps": state["successful_optimizer_steps"]},
        })
    diag_c = phase_c_convergence(history, args, state)
    assert diag_c["phase_c"] and diag_c["ready_for_phase_d"]
    ramp, start_step = apply_ready_curriculum_transition(state, diag_c)
    assert ramp == "slide" and start_step == 160 * updates_per_ep
    state["phase_d_start_epoch"] = 161

    # Mid-Phase D ramp (20 epochs into 40-epoch slide ramp) -> lambda_slide == 0.5
    state["successful_optimizer_steps"] = 180 * updates_per_ep
    assert math.isclose(slide_curriculum_weight(state), 0.5, abs_tol=1e-7)
    assert curriculum_phase(180, state, *curriculum_weights(state), "three_phase_global_cosine") == (3, "sliding_ramp")
    assert not curriculum_is_complete(state)

    # End of 40-epoch Phase D ramp -> lambda_slide == 1.0, curriculum complete!
    state["successful_optimizer_steps"] = 200 * updates_per_ep
    assert math.isclose(slide_curriculum_weight(state), 1.0, abs_tol=1e-7)
    assert curriculum_phase(200, state, *curriculum_weights(state), "three_phase_global_cosine") == (3, "sliding_settled")
    assert curriculum_is_complete(state)
    require_complete_curriculum(state, 240)

    # Verify global cosine LR does not reheat across transitions (monotonically decays after warmup ep 5)
    lrs = [curriculum_learning_rate_for_epoch(ep, args, state) for ep in range(1, 241)]
    assert math.isclose(lrs[0], 1e-5, rel_tol=1e-6)
    assert math.isclose(lrs[4], 2e-4, rel_tol=1e-6)
    assert math.isclose(lrs[-1], 1e-5, rel_tol=1e-6)
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(4, 239))


def test_resume_mid_phase_d_ramp_parity(tmp_path: Path):
    """Verify checkpoint save & resume mid-Phase D ramp restores exact LR, weights, and lambdas."""
    args = _make_240ep_args(240)
    updates_per_ep = 50
    state = make_curriculum_state(args, updates_per_epoch=updates_per_ep)
    state["variance_ramp_start_step"] = 10 * updates_per_ep
    state["variance_ramp_armed"] = True
    state["rollout_ramp_start_step"] = 60 * updates_per_ep
    state["rollout_ramp_armed"] = True
    state["slide_ramp_start_step"] = 110 * updates_per_ep
    state["slide_ramp_armed"] = True
    state["phase_d_start_epoch"] = 111
    state["successful_optimizer_steps"] = 130 * updates_per_ep  # 20 epochs into 40-ep slide ramp

    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture="direct_unet3d_no_activation_norm_v3", uncertainty_aware=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    ckpt_path = tmp_path / "latest.pt"
    save_checkpoint(
        ckpt_path, model=model, optimizer=opt, epoch=130, args=args,
        stats={"mean": [0.0]*4, "std": [1.0]*4}, curriculum_state=state,
    )

    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    restored = make_curriculum_state(args, updates_per_epoch=updates_per_ep, checkpoint=loaded)
    assert restored["successful_optimizer_steps"] == 130 * updates_per_ep
    assert curriculum_weights(restored) == (1.0, 1.0)
    assert math.isclose(slide_curriculum_weight(restored), 0.5, abs_tol=1e-7)
    assert restored["phase_d_start_epoch"] == 111
    assert math.isclose(
        curriculum_learning_rate_for_epoch(131, args, restored),
        curriculum_learning_rate_for_epoch(131, args, state),
        rel_tol=1e-9,
    )


def test_auto_train_monitor_mock_lifecycle(tmp_path: Path):
    """Test auto_train_phase_d_240ep.sh: fresh launch, transient retry, resume, completion, no-progress halt."""
    script = Path("/pscratch/sd/b/binxia/ai4plasma/auto_train_phase_d_240ep.sh")
    proj = tmp_path / "proj"
    proj.mkdir()
    run_name = "test_run"
    run_dir = proj / "runs" / run_name

    # Create mock train_masked_unet3d_4n16g.sbatch and mock salloc/squeue
    mock_sbatch = proj / "train_masked_unet3d_4n16g.sbatch"
    mock_sbatch.write_text("""#!/bin/bash
set -euo pipefail
if [[ "$1" == "--adaptive-abcd-sliding-240ep" ]] && [[ -e "${PWD}/runs/test_run" ]]; then
    echo "ERROR: fresh output directory already exists: ${PWD}/runs/test_run; refusing overwrite/resume." >&2
    exit 1
fi
if [[ "${MOCK_MODE:-normal}" == "no_progress" ]]; then
    exit 0
fi
mkdir -p "${PWD}/runs/test_run"
LOG_FLAGS="${PWD}/runs/test_run/invocations.log"
echo "$1" >> "${LOG_FLAGS}"
COUNT=$(wc -l < "${LOG_FLAGS}")
if [[ "${MOCK_MODE:-normal}" == "continue_past_target" ]]; then
    if [[ "${COUNT}" -eq 1 ]]; then
        python3 -c "
import json, torch
cs = {'adaptive_phase_d_sliding': True, 'variance_ramp_armed': True, 'rollout_ramp_armed': True, 'slide_ramp_armed': True, 'slide_ramp_start_step': 40, 'slide_ramp_steps': 20, 'successful_optimizer_steps': 60}
torch.save({'epoch': 10, 'curriculum_state': cs}, '${PWD}/runs/test_run/latest.pt')
json.dump([{'epoch': 10}], open('${PWD}/runs/test_run/history.json', 'w'))
"
    else
        python3 -c "
import json, torch
cs = {'adaptive_phase_d_sliding': True, 'variance_ramp_armed': True, 'rollout_ramp_armed': True, 'slide_ramp_armed': True, 'slide_ramp_start_step': 40, 'slide_ramp_steps': 20, 'successful_optimizer_steps': 80, 'completion_conditions': {'lr_cosine_complete': True, 'slide_cosine_complete': True, 'phase_d_converged': True, 'actual_final_epoch': 12}}
torch.save({'epoch': 12, 'curriculum_state': cs}, '${PWD}/runs/test_run/latest.pt')
torch.save({'epoch': 12, 'curriculum_state': cs}, '${PWD}/runs/test_run/final.pt')
json.dump([{'epoch': 10}, {'epoch': 12}], open('${PWD}/runs/test_run/history.json', 'w'))
"
    fi
    exit 0
fi
if [[ "${COUNT}" -eq 1 ]]; then
    # Simulate fresh run reaching epoch 2, saving latest.pt
    python3 -c "
import json, torch
torch.save({'epoch': 2}, '${PWD}/runs/test_run/latest.pt')
json.dump([{'epoch': 2}], open('${PWD}/runs/test_run/history.json', 'w'))
"
elif [[ "${COUNT}" -ge 2 ]]; then
    # Simulate resume reaching epoch 10 and completing full A->B->C->D
    python3 -c "
import json, torch
cs = {'adaptive_phase_d_sliding': True, 'variance_ramp_armed': True, 'rollout_ramp_armed': True, 'slide_ramp_armed': True, 'slide_ramp_start_step': 40, 'slide_ramp_steps': 20, 'successful_optimizer_steps': 60}
torch.save({'epoch': 10, 'curriculum_state': cs}, '${PWD}/runs/test_run/final.pt')
json.dump([{'epoch': 2}, {'epoch': 10}], open('${PWD}/runs/test_run/history.json', 'w'))
"
fi
""")
    mock_sbatch.chmod(0o755)

    mock_salloc = tmp_path / "mock_salloc.sh"
    mock_salloc.write_text("""#!/bin/bash
# Mock salloc: skip flags until 'bash -c ...'
while [[ "$1" != "bash" ]]; do shift; done
shift # bash
shift # -c
if [[ "${MOCK_FAIL_FIRST:-0}" == "1" ]] && [[ ! -f "${PWD}/failed_once" ]]; then
    touch "${PWD}/failed_once"
    exit 1
fi
bash -c "$1"
""")
    mock_salloc.chmod(0o755)

    mock_squeue = tmp_path / "mock_squeue.sh"
    mock_squeue.write_text("#!/bin/bash\nexit 0\n")
    mock_squeue.chmod(0o755)

    env = os.environ.copy()
    env.update({
        "PROJECT_DIR": str(proj),
        "CURRICULUM_RUN_NAME": run_name,
        "TARGET_EPOCH": "10",
        "POLL_INTERVAL": "0",
        "BASE_RETRY_DELAY": "0",
        "MAX_ALLOC_RETRIES": "5",
        "MAX_NO_PROGRESS": "2",
        "MIN_CKPT_BYTES": "1",
        "SALLOC_CMD": str(mock_salloc),
        "SQUEUE_CMD": str(mock_squeue),
        "MOCK_FAIL_FIRST": "1",
    })

    # 1. Test full lifecycle: transient salloc fail -> fresh launch -> resume -> final.pt completion
    res = subprocess.run(["bash", str(script)], cwd=str(proj), env=env, capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr + "\n" + res.stdout
    invocations = (run_dir / "invocations.log").read_text().splitlines()
    assert invocations == ["--adaptive-abcd-sliding-240ep", "--resume-adaptive-abcd-sliding-240ep"]

    # 2. Test no-progress failure halt (and monitor log location outside RUN_DIR)
    proj2 = tmp_path / "proj2"
    proj2.mkdir()
    (proj2 / "train_masked_unet3d_4n16g.sbatch").write_text(mock_sbatch.read_text())
    (proj2 / "train_masked_unet3d_4n16g.sbatch").chmod(0o755)
    env2 = env.copy()
    env2.update({"PROJECT_DIR": str(proj2), "MOCK_MODE": "no_progress", "MOCK_FAIL_FIRST": "0"})
    res2 = subprocess.run(["bash", str(script)], cwd=str(proj2), env=env2, capture_output=True, text=True, timeout=30)
    assert res2.returncode != 0
    mon_log2 = proj2 / "runs" / f".monitor_{run_name}" / "auto_resume_monitor.log"
    assert "consecutive sessions made zero progress" in mon_log2.read_text()

    # 3. Test continuing past TARGET_EPOCH without final.pt, then completing at epoch 12
    proj3 = tmp_path / "proj3"
    proj3.mkdir()
    (proj3 / "train_masked_unet3d_4n16g.sbatch").write_text(mock_sbatch.read_text())
    (proj3 / "train_masked_unet3d_4n16g.sbatch").chmod(0o755)
    env3 = env.copy()
    env3.update({"PROJECT_DIR": str(proj3), "MOCK_MODE": "continue_past_target", "MOCK_FAIL_FIRST": "0"})
    res3 = subprocess.run(["bash", str(script)], cwd=str(proj3), env=env3, capture_output=True, text=True, timeout=30)
    assert res3.returncode == 0, res3.stderr + "\n" + res3.stdout
    inv3 = (proj3 / "runs" / run_name / "invocations.log").read_text().splitlines()
    assert inv3 == ["--adaptive-abcd-sliding-240ep", "--resume-adaptive-abcd-sliding-240ep"]
    assert (proj3 / "runs" / run_name / "final.pt").is_file()


def test_post_240_lr_tail_and_three_condition_termination(tmp_path: Path):
    """Verify post-240 constant min LR, three-condition gating, Phase D burn-in/patience, and resume."""
    args = _make_240ep_args(240)
    updates_per_ep = 100
    state = make_curriculum_state(args, updates_per_epoch=updates_per_ep)

    # 1. LR 1..240 identical to original global_cosine_learning_rate, and 241..300 == 1e-5
    orig_lrs = [global_cosine_learning_rate(ep, 240, 5, 1e-5, 2e-4) for ep in range(1, 241)]
    new_lrs = [curriculum_learning_rate_for_epoch(ep, args, state) for ep in range(1, 301)]
    assert new_lrs[:240] == orig_lrs
    assert math.isclose(new_lrs[0], 1e-5, rel_tol=1e-9)
    assert math.isclose(new_lrs[4], 2e-4, rel_tol=1e-9)
    assert math.isclose(new_lrs[239], 1e-5, rel_tol=1e-9)
    assert math.isclose(new_lrs[240], 1e-5, rel_tol=1e-9)
    assert math.isclose(new_lrs[259], 1e-5, rel_tol=1e-9)
    assert math.isclose(new_lrs[299], 1e-5, rel_tol=1e-9)
    with pytest.raises(ValueError):
        global_cosine_learning_rate(241, 240, 5, 1e-5, 2e-4)

    # Helper to build sliding_validation dict for an epoch
    def _sv(mse: float, gkl: float, zero_mse: float = 9.9, n_runs: int = 20):
        return {
            "0": {"mse": zero_mse, "generalized_kl": zero_mse, "n_runs": n_runs},
            "1": {"mse": mse, "generalized_kl": gkl, "n_runs": n_runs},
            "-1": {"mse": mse, "generalized_kl": gkl, "n_runs": n_runs},
            "12": {"mse": mse, "generalized_kl": gkl, "n_runs": n_runs},
            "-12": {"mse": mse, "generalized_kl": gkl, "n_runs": n_runs},
        }

    # 2. Simulate Phase D starting at epoch 180 (slide ramp 181..220, settled 221+)
    state["variance_ramp_armed"] = True
    state["variance_ramp_start_step"] = 20 * updates_per_ep
    state["rollout_ramp_armed"] = True
    state["rollout_ramp_start_step"] = 80 * updates_per_ep
    state["slide_ramp_armed"] = True
    state["slide_ramp_start_step"] = 180 * updates_per_ep
    state["phase_d_start_epoch"] = 181

    history = []
    # Epochs 181..219: still in slide ramp -> must NOT count toward Phase D burn-in
    for ep in range(181, 220):
        state["successful_optimizer_steps"] = ep * updates_per_ep
        history.append({
            "epoch": ep,
            "curriculum": {"successful_optimizer_steps": state["successful_optimizer_steps"]},
            "sliding_validation": _sv(0.01, 0.01),
        })
    diag_ramp = phase_d_convergence(history, args, state)
    assert not diag_ramp["slide_cosine_complete"]
    assert not diag_ramp["phase_d_converged"]
    assert not diag_ramp["training_complete"]

    # Epochs 220..235: settled slide (lambda_slide=1.0), converges by ep 235, but LR < 240!
    for ep in range(220, 236):
        state["successful_optimizer_steps"] = ep * updates_per_ep
        history.append({
            "epoch": ep,
            "curriculum": {"successful_optimizer_steps": state["successful_optimizer_steps"]},
            "sliding_validation": _sv(0.01, 0.01),
        })
    diag_pre240 = phase_d_convergence(history, args, state)
    assert not diag_pre240["lr_cosine_complete"]
    assert diag_pre240["slide_cosine_complete"]
    assert diag_pre240["phase_d_converged"]
    assert not diag_pre240["training_complete"]

    # Nonfinite / missing direction at ep 236 resets burn-in & patience!
    state["successful_optimizer_steps"] = 236 * updates_per_ep
    bad_sv = _sv(0.01, 0.01)
    bad_sv["-12"]["generalized_kl"] = float("nan")
    history.append({
        "epoch": 236,
        "curriculum": {"successful_optimizer_steps": state["successful_optimizer_steps"]},
        "sliding_validation": bad_sv,
    })
    diag_bad = phase_d_convergence(history, args, state)
    assert not diag_bad["phase_d_converged"]
    assert diag_bad["stable_count"] == 0

    # Recover for epochs 237..246 (10 burn-in epochs -> segment_age=10 at ep 247, patience=4 -> ep 250)
    for ep in range(237, 250):
        state["successful_optimizer_steps"] = ep * updates_per_ep
        history.append({
            "epoch": ep,
            "curriculum": {"successful_optimizer_steps": state["successful_optimizer_steps"]},
            "sliding_validation": _sv(0.01, 0.01),
        })
        d = phase_d_convergence(history, args, state)
        assert not d["training_complete"], f"Unexpectedly completed at epoch {ep}"

    # At epoch 250 (14th valid settled epoch: 0..10 burn-in, 11..13 stable 1..3, 14th stable=4):
    state["successful_optimizer_steps"] = 250 * updates_per_ep
    history.append({
        "epoch": 250,
        "curriculum": {"successful_optimizer_steps": state["successful_optimizer_steps"]},
        "sliding_validation": _sv(0.01, 0.01),
    })
    diag_final = phase_d_convergence(history, args, state)
    assert diag_final["lr_cosine_complete"]
    assert diag_final["slide_cosine_complete"]
    assert diag_final["phase_d_converged"]
    assert diag_final["training_complete"]
    assert diag_final["stable_count"] == 4

    # Verify s=0 is excluded from aggregate (changing s=0 does not change agg)
    assert math.isclose(diag_final["val_sliding_mse_agg"], 0.01, rel_tol=1e-9)
    assert math.isclose(diag_final["val_sliding_gkl_agg"], 0.01, rel_tol=1e-9)

    # Verify checkpoint save & resume at epoch 241 preserves exact convergence decision
    model = UNet3D(
        in_channels=8, out_channels=4, base_channels=8, channel_mults=(1, 2, 4),
        architecture="direct_unet3d_no_activation_norm_v3", uncertainty_aware=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    ckpt_path = tmp_path / "latest_241.pt"
    save_checkpoint(
        ckpt_path, model=model, optimizer=opt, epoch=241, args=args,
        stats={"mean": [0.0]*4, "std": [1.0]*4}, curriculum_state=state,
    )
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    restored = make_curriculum_state(args, updates_per_epoch=updates_per_ep, checkpoint=loaded)
    assert is_slide_cosine_complete(restored)
    assert phase_d_convergence(history[:61], args, restored)["stable_count"] == phase_d_convergence(history[:61], args, state)["stable_count"]
