from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import run_final_campaign

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> list[dict]:
    result = subprocess.run([sys.executable, str(ROOT / "run_final_campaign.py"), *args],
                            cwd=ROOT, text=True, capture_output=True, check=True)
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def test_calibration_is_cifar10_dense_three_seed_and_preregistered() -> None:
    plans = _run("calibrate", "--dry-run", "--out", "/tmp/ddp-calibration-plan")
    assert len(plans) == 3
    for seed, plan in zip((17, 29, 41), plans):
        command = plan["command"]
        assert command[command.index("--dataset") + 1] == "cifar10"
        assert command[command.index("--arm") + 1] == "dense"
        assert command[command.index("--seed") + 1] == str(seed)
        assert command[command.index("--target-accuracy") + 1] == "0.7"
        assert command[command.index("--steps") + 1] == "1000"


def test_final_matrix_matches_accumulation_reference_and_profiles_separately(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "schema": "ddp-crossover.calibration/v1", "dataset": "cifar10",
        "target_accuracy": 0.70, "steps_cap": 1000,
        "global_batch_size": 128, "precision": "fp16", "world_size": 2,
        "seeds": [17, 29, 41], "status": "target_frozen",
    }))
    plans = _run("final", "--calibration", str(calibration), "--dry-run",
                 "--out", str(tmp_path / "final"))
    assert len(plans) == 40
    assert sum("profile-" in Path(plan["log"]).name and "-control-" not in Path(plan["log"]).name
               for plan in plans) == 8
    training_commands = [plan["command"] for plan in plans if "/profile/" not in " ".join(plan["command"])]
    assert len(training_commands) == 24
    assert {command[command.index("--seed") + 1] for command in training_commands} == {"17", "29", "41"}
    assert {command[command.index("--accumulate-steps") + 1] for command in training_commands} == {"1", "4"}
    fixed = [command for command in training_commands
             if command[command.index("--arm") + 1] == "powersgd"]
    assert {command[command.index("--rank") + 1] for command in fixed} == {"1", "2", "4", "8"}
    assert all(command[command.index("--accumulate-steps") + 1] == "4" for command in fixed)


def test_campaign_orders_gpu_restart_gate_before_calibration_and_quality_matrix(tmp_path: Path) -> None:
    plans = _run("campaign", "--dry-run", "--out", str(tmp_path / "campaign"))
    assert len(plans) == 48
    commands = [" ".join(plan["command"]) for plan in plans]
    assert sum("nccl_fault_check.py" in command for command in commands) == 1
    assert sum(Path(plan["log"]).name == "adaptive-partial.log" for plan in plans) == 1
    assert sum(Path(plan["log"]).name == "adaptive-resumed.log" for plan in plans) == 1
    assert sum(Path(plan["log"]).name == "adaptive-reference.log" for plan in plans) == 1
    assert sum("verify_restart_equivalence.py" in command for command in commands) == 1
    assert sum("calibration-dense-seed" in plan["log"] for plan in plans) == 3
    assert sum("profile-" in Path(plan["log"]).name for plan in plans) == 16
    fault_position = next(i for i, command in enumerate(commands) if "nccl_fault_check.py" in command)
    restart_position = next(i for i, command in enumerate(commands)
                            if "--output" in command and "adaptive-partial" in command)
    assert fault_position < restart_position


def test_restart_correctness_dispatches_verifier_without_torchrun(tmp_path: Path, monkeypatch) -> None:
    launched: list[list[str]] = []

    def fake_launch(command, log, *, dry_run):
        launched.append(command)
        if any("nccl_fault_check.py" in part for part in command):
            Path(command[command.index("--out") + 1]).write_text('{"status":"pass"}')
        elif any("verify_restart_equivalence.py" in part for part in command):
            Path(command[command.index("--out") + 1]).write_text(
                '{"status":"pass","fields_compared":["model"]}')
        return 0

    monkeypatch.setattr(run_final_campaign, "launch", fake_launch)
    monkeypatch.setattr(run_final_campaign, "run", lambda *args, **kwargs: 0)

    result = run_final_campaign.run_restart_correctness(tmp_path, dry_run=False)

    verifier = launched[-1]
    assert verifier[0] == sys.executable
    assert verifier[1] == str(ROOT / "verify_restart_equivalence.py")
    assert "-m" not in verifier
    assert result["status"] == "pass"


def test_campaign_uses_external_watchdog_without_elapsed_time_flags(tmp_path: Path) -> None:
    plans = _run("campaign", "--dry-run", "--out", str(tmp_path / "campaign"))
    commands = [part for plan in plans for part in plan["command"]]
    assert not any("wall-clock" in part or "provider-start" in part or "max-rental" in part
                   or "no-elapsed" in part for part in commands)


def test_wall_clock_timeout_logs_and_kills_entire_process_group(tmp_path: Path) -> None:
    descendant_pid_file = tmp_path / "descendant.pid"
    child = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(descendant_pid_file)!r}).write_text(str(p.pid)); "
        "time.sleep(60)"
    )
    log = tmp_path / "timeout.log"
    started = time.monotonic()
    result = run_final_campaign.run_process([sys.executable, "-c", child], log, 0.25)
    assert result == 124
    assert time.monotonic() - started < 5
    assert "Campaign wall-clock cap reached" in log.read_text()
    descendant_pid = int(descendant_pid_file.read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            break
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(descendant_pid)],
                               text=True, capture_output=True).stdout.strip()
        if not state or state.startswith("Z"):
            break
        time.sleep(0.05)
    else:
        raise AssertionError(f"descendant process {descendant_pid} survived process-group timeout")
