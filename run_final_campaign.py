"""Run the preregistered CIFAR-10 DDP calibration and final arm matrix.

The final campaign is deliberately serial on one two-GPU host. It creates no
cloud resources and refuses to run without two visible CUDA devices.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = Path.home() / ".codex/work/project-extensions/05-ddp/runs"
SEEDS = (17, 29, 41)
TARGET = 0.70
STEPS = 1000
GLOBAL_BATCH = 128
ACCUMULATION = 4
PRECISION = "fp16"
@dataclass(frozen=True)
class Arm:
    name: str
    arm: str
    accumulation: int
    rank: int = 2


ARMS = (
    Arm("dense-accum1", "dense", 1),
    Arm("dense-accum4", "dense", ACCUMULATION),
    Arm("fp16-accum4", "fp16", ACCUMULATION),
    Arm("powersgd-rank1-accum4", "powersgd", ACCUMULATION, 1),
    Arm("powersgd-rank2-accum4", "powersgd", ACCUMULATION, 2),
    Arm("powersgd-rank4-accum4", "powersgd", ACCUMULATION, 4),
    Arm("powersgd-rank8-accum4", "powersgd", ACCUMULATION, 8),
    Arm("adaptive-rank2-accum4", "adaptive", ACCUMULATION, 2),
)


def run(argv: list[str], log: Path, *, dry_run: bool) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
               "--nproc_per_node=2", str(ROOT / "05_learnable_train.py"), *argv]
    return launch(command, log, dry_run=dry_run)


def run_process(command: list[str], log: Path, timeout: float | None) -> int:
    """Run one command in its own process group, terminating descendants on cap."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                 env=os.environ.copy(), start_new_session=True)
        try:
            return child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stream.write("\nCampaign wall-clock cap reached. Terminating the training process group.\n")
            stream.flush()
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
            return 124


def launch(command: list[str], log: Path, *, dry_run: bool) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(json.dumps({"command": command, "log": str(log)}, sort_keys=True))
        return 0
    return run_process(command, log, None)


def latest_row(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"training emitted no validation rows: {path}")
    return rows[-1]


def ensure_gpu(dry_run: bool) -> None:
    if dry_run:
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise SystemExit("final campaign requires exactly two visible CUDA devices; no run was started")


def common(seed: int, output: Path, *, stop_after: int, checkpoint: Path | None,
           profile: Path | None = None) -> list[str]:
    args = ["--dataset", "cifar10", "--seed", str(seed), "--steps", str(STEPS),
            "--stop-after", str(stop_after), "--batch-size", str(GLOBAL_BATCH),
            "--eval-every", "100", "--target-accuracy", str(TARGET),
            "--precision", PRECISION, "--output", str(output)]
    if checkpoint:
        args.extend(("--checkpoint", str(checkpoint)))
    if profile:
        args.extend(("--profiler-trace", str(profile), "--profiler-steps", "32"))
    return args


def run_restart_correctness(run_dir: Path, *, dry_run: bool) -> dict:
    """On the rented two-GPU NCCL host, compare full adaptive resume to reference."""
    fault_report = run_dir / "nccl-gradient-fault.json"
    fault_command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                     "--nproc_per_node=2", str(ROOT / "nccl_fault_check.py"),
                     "--out", str(fault_report)]
    result = launch(fault_command, run_dir / "logs" / "nccl-gradient-fault.log", dry_run=dry_run)
    if result:
        return {"status": "failed", "exit_code": result, "fault_report": str(fault_report)}
    arm = ("--arm", "adaptive", "--accumulate-steps", str(ACCUMULATION), "--rank", "2",
           "--update-every", "1")
    partial_ckpt = run_dir / "adaptive-partial.pt"
    partial_trace = run_dir / "adaptive-partial.jsonl"
    partial = common(SEEDS[0], partial_trace, stop_after=3, checkpoint=partial_ckpt)
    partial.extend(arm)
    partial_log = run_dir / "logs" / "adaptive-partial.log"
    result = run(partial, partial_log, dry_run=dry_run)
    if result:
        return {"status": "failed", "exit_code": result}

    resumed_ckpt = run_dir / "adaptive-resumed.pt"
    resumed_trace = run_dir / "adaptive-resumed.jsonl"
    resumed = common(SEEDS[0], resumed_trace, stop_after=6, checkpoint=resumed_ckpt)
    resumed.extend(("--resume", str(partial_ckpt), *arm))
    result = run(resumed, run_dir / "logs" / "adaptive-resumed.log", dry_run=dry_run)
    if result:
        return {"status": "failed", "exit_code": result}

    reference_ckpt = run_dir / "adaptive-reference.pt"
    reference_trace = run_dir / "adaptive-reference.jsonl"
    reference = common(SEEDS[0], reference_trace, stop_after=6, checkpoint=reference_ckpt)
    reference.extend(arm)
    result = run(reference, run_dir / "logs" / "adaptive-reference.log", dry_run=dry_run)
    if result:
        return {"status": "failed", "exit_code": result}
    if dry_run:
        command = [sys.executable, str(ROOT / "verify_restart_equivalence.py"),
                   str(partial_ckpt), str(resumed_ckpt), str(reference_ckpt),
                   "--out", str(run_dir / "restart-equivalence.json")]
        print(json.dumps({"command": command, "log": str(run_dir / "logs" / "restart-equivalence.log")}, sort_keys=True))
        return {"status": "planned", "fault_report": str(fault_report), "partial_checkpoint": str(partial_ckpt),
                "resumed_checkpoint": str(resumed_ckpt), "reference_checkpoint": str(reference_ckpt)}
    report = run_dir / "restart-equivalence.json"
    command = [sys.executable, str(ROOT / "verify_restart_equivalence.py"),
               str(partial_ckpt), str(resumed_ckpt), str(reference_ckpt), "--out", str(report)]
    result = launch(command, run_dir / "logs" / "restart-equivalence.log", dry_run=False)
    if result and not report.exists():
        return {"status": "failed", "exit_code": result, "report": str(report)}
    outcome = json.loads(report.read_text())
    fault_result = json.loads(fault_report.read_text())
    status = "pass" if outcome.get("status") == "pass" and fault_result.get("status") == "pass" else "fail"
    return {"status": status, "fault_report": str(fault_report), "report": str(report),
            "partial_checkpoint": str(partial_ckpt), "resumed_checkpoint": str(resumed_ckpt),
            "reference_checkpoint": str(reference_ckpt), "fields_compared": outcome["fields_compared"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("campaign", "calibrate", "final"))
    parser.add_argument("--out", type=Path, default=None,
                        help="private run directory; default is a timestamp under ~/.codex/work")
    parser.add_argument("--calibration", type=Path,
                        help="calibration.json required by final phase")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact command matrix without training")
    args = parser.parse_args()
    run_dir = args.out or DEFAULT_OUT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if not args.dry_run and run_dir.exists() and any(run_dir.iterdir()):
        parser.error("output directory must be new or empty so traces cannot be mixed across runs")
    campaign_root = run_dir if args.phase == "campaign" else None
    if args.phase == "campaign":
        run_dir = run_dir / "calibration"
    if args.phase == "final" and not args.dry_run:
        if not args.calibration:
            parser.error("final phase requires --calibration from a completed target-freeze phase")
    ensure_gpu(args.dry_run)

    protocol = {"schema": "ddp-crossover.campaign-protocol/v1", "phase": args.phase,
                "dataset": "CIFAR-10 official train and test batches",
                "target_accuracy": TARGET, "steps_cap": STEPS,
                "global_batch_size": GLOBAL_BATCH, "precision": PRECISION,
                "world_size": 2, "seeds": list(SEEDS), "arms": [arm.__dict__ for arm in ARMS],
                "profile_seed": SEEDS[0], "profile_steps": 32,
                "wall_clock_cap_hours": None,
                "runner_deadline_epoch": None,
                "provider_start_epoch": None,
                "provider_deadline_epoch": None,
                "deadline_scope": "none; the external credit watchdog remains authoritative",
                "restart_correctness_stage": {
                    "gradient_fault_check": "NCCL DDP injects a finite +1000 gradient perturbation on rank 1, bucket 0 at compressed update 2, then verifies finite weights, identical adaptive controller decisions and replica updates",
                    "adaptive_arm": "PowerSGD adaptive rank 2, update cadence 1, global batch 128, fp16, seed 17",
                    "procedure": "checkpoint after update 3, resume to update 6, compare with uninterrupted update-6 checkpoint",
                    "fields": ["model", "optimizer", "scheduler", "scaler", "compressor_by_rank", "rng_by_rank"],
                    "tolerance_atol_rtol": 1e-5,
                    "required_before_calibration_and_final": True,
                },
                "paid_resource_cap_usd": 8.0,
                "provider_request": {"provider": "JarvisLabs", "gpu": "2 x A100 40GB",
                                     "paid_resource_cap_usd": 8.0,
                                     "disk_gb": 100},
                "training_claim": "No speedup claim without two-device CUDA/NCCL evidence and full report."}
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    correctness = None
    if campaign_root is not None:
        correctness = run_restart_correctness(campaign_root / "correctness", dry_run=args.dry_run)
        if correctness["status"] in {"failed", "fail"}:
            (campaign_root / "campaign-status.json").write_text(json.dumps({
                "status": "stopped_before_calibration",
                "reason": "restart correctness did not pass",
                "correctness": correctness,
            }, indent=2, sort_keys=True) + "\n")
            return correctness.get("exit_code", 1)

    if args.phase in ("calibrate", "campaign"):
        summary = {"schema": "ddp-crossover.calibration/v1", "dataset": "cifar10",
                   "target_accuracy": TARGET, "steps_cap": STEPS,
                   "global_batch_size": GLOBAL_BATCH, "precision": PRECISION,
                   "world_size": 2, "seeds": list(SEEDS), "runs": [],
                   "runner_deadline_epoch": None,
                   "provider_start_epoch": None,
                   "provider_deadline_epoch": None}
        for seed in SEEDS:
            output = run_dir / "calibration" / f"dense-seed{seed}.jsonl"
            log = run_dir / "logs" / f"calibration-dense-seed{seed}.log"
            argv = common(seed, output, stop_after=STEPS,
                          checkpoint=run_dir / "calibration" / f"dense-seed{seed}.pt")
            argv.extend(("--arm", "dense", "--accumulate-steps", "1", "--rank", "2"))
            exit_code = run(argv, log, dry_run=args.dry_run)
            if args.dry_run:
                continue
            if exit_code:
                return exit_code
            row = latest_row(output)
            summary["runs"].append({"seed": seed, "trace": str(output),
                                    "final_accuracy": row["validation_accuracy"],
                                    "target_reached": row["validation_accuracy"] >= TARGET,
                                    "elapsed_s": row["elapsed_s"]})
        if not args.dry_run:
            summary["status"] = "complete"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "calibration.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if args.phase == "calibrate":
            return 0
        if args.dry_run:
            calibration = {"schema": "ddp-crossover.calibration/v1", "dataset": "cifar10",
                           "target_accuracy": TARGET, "steps_cap": STEPS,
                           "global_batch_size": GLOBAL_BATCH, "precision": PRECISION,
                           "world_size": 2, "seeds": list(SEEDS), "status": "complete",
                           "runner_deadline_epoch": None}
        else:
            args.calibration = run_dir / "calibration.json"
            calibration = json.loads(args.calibration.read_text())
        args.phase = "final"
        run_dir = campaign_root / "final"
        if not args.dry_run:
            run_dir.mkdir(parents=True, exist_ok=True)
        protocol["phase"] = "final"
        if not args.dry_run:
            (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    else:
        calibration = json.loads(args.calibration.read_text())
    ensure_gpu(args.dry_run)
    manifest = {"schema": "ddp-crossover.final-campaign/v1", "calibration": str(args.calibration) if args.calibration else None,
                "target_accuracy": TARGET, "steps_cap": STEPS, "global_batch_size": GLOBAL_BATCH,
                "precision": PRECISION, "world_size": 2, "seeds": list(SEEDS),
                "arms": [arm.__dict__ for arm in ARMS], "runs": [],
                "restart_correctness": correctness}
    for seed in SEEDS:
        for arm in ARMS:
            output = run_dir / "final" / f"{arm.name}-seed{seed}.jsonl"
            log = run_dir / "logs" / f"final-{arm.name}-seed{seed}.log"
            checkpoint = run_dir / "checkpoints" / f"{arm.name}-seed{seed}.pt"
            argv = common(seed, output, stop_after=STEPS, checkpoint=checkpoint)
            argv.extend(("--arm", arm.arm, "--accumulate-steps", str(arm.accumulation),
                         "--rank", str(arm.rank)))
            if run(argv, log, dry_run=args.dry_run):
                return 1
            if args.dry_run:
                continue
            row = latest_row(output)
            manifest["runs"].append({"arm": arm.name, "seed": seed, "trace": str(output),
                                     "log": str(log), "final_accuracy": row["validation_accuracy"],
                                     "target_reached": row["validation_accuracy"] >= TARGET,
                                     "elapsed_s": row["elapsed_s"],
                                     "communication_payload_bytes_per_rank": row.get("communication_payload_bytes_per_rank"),
                                     "controller_payload_bytes_per_rank": row.get("controller_payload_bytes_per_rank", 0),
                                     "estimated_ring_bus_bytes_per_rank": row.get("estimated_ring_bus_bytes_per_rank")})
    profile_runs = []
    profile_seed = SEEDS[0]
    for arm in ARMS:
        rows = {}
        for mode in ("control", "profile"):
            output = run_dir / "profile" / f"{arm.name}-{mode}-seed{profile_seed}.jsonl"
            log = run_dir / "logs" / f"profile-{arm.name}-{mode}-seed{profile_seed}.log"
            trace = (run_dir / "profile" / f"{arm.name}-rank{{rank}}.trace.json" if mode == "profile" else None)
            argv = common(profile_seed, output, stop_after=32, checkpoint=None, profile=trace)
            argv.extend(("--arm", arm.arm, "--accumulate-steps", str(arm.accumulation),
                         "--rank", str(arm.rank)))
            if run(argv, log, dry_run=args.dry_run):
                return 1
            if not args.dry_run:
                rows[mode] = latest_row(output)
        if not args.dry_run:
            control_s = rows["control"]["elapsed_s"]
            profile_s = rows["profile"]["elapsed_s"]
            profile_runs.append({"arm": arm.name, "seed": profile_seed, "steps": 32,
                                 "control_elapsed_s": control_s,
                                 "profile_elapsed_s": profile_s,
                                 "profiler_overhead_s": profile_s - control_s,
                                 "profiler_overhead_fraction": profile_s / control_s - 1.0,
                                 "trace_pattern": str(run_dir / "profile" / f"{arm.name}-rank{{rank}}.trace.json"),
                                 "logical_communication_payload_bytes_per_rank":
                                     rows["profile"].get("communication_payload_bytes_per_rank"),
                                 "controller_payload_bytes_per_rank":
                                     rows["profile"].get("controller_payload_bytes_per_rank", 0)})
    manifest["profiling"] = {"profile_seed": profile_seed, "profile_steps": 32,
                             "runs": profile_runs,
                             "note": "Short profiler runs are separate from time-to-quality runs; overhead is paired at the same seed, arm and 32-step cap."}
    if not args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        final_paths = [Path(row["trace"]) for row in manifest["runs"]]
        report = run_dir / "time-to-quality.json"
        report_command = [sys.executable, str(ROOT / "06_compare.py"),
                          *[str(path) for path in final_paths], "--out", str(report)]
        report_log = run_dir / "logs" / "time-to-quality.log"
        report_result = run_process(report_command, report_log, None)
        if report_result:
            return report_result
        manifest["report"] = str(report)
        (run_dir / "campaign-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
