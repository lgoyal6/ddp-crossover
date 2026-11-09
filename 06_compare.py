"""Summarize matched training traces, including arms censored before target."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def summarize(paths: list[Path]) -> dict:
    runs = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["arm"], row["dataset"], row["seed"], row["world_size"],
                   row.get("global_batch_size"), row.get("accumulate_steps", 1),
                   row.get("power_rank", 1))
            runs.setdefault(key, []).append(row)
    arms = []
    for (arm, dataset, seed, world, global_batch_size, accumulate_steps, power_rank), rows in sorted(runs.items()):
        rows.sort(key=lambda row: row["step"])
        target = rows[-1]["target_accuracy"]
        reached = next((r for r in rows if r["validation_accuracy"] >= target), None)
        arms.append({
            "arm": arm, "dataset": dataset, "seed": seed, "world_size": world,
            "global_batch_size": global_batch_size,
            "accumulate_steps": accumulate_steps, "power_rank": power_rank,
            "target_accuracy": target,
            "target_reached": reached is not None,
            "time_to_target_s": reached["elapsed_s"] if reached else None,
            "censored_at_step": None if reached else rows[-1]["step"],
            "final_validation_accuracy": rows[-1]["validation_accuracy"],
            "final_validation_loss": rows[-1]["validation_loss"],
            "final_elapsed_s": rows[-1]["elapsed_s"],
            "communication_payload_bytes_per_rank": rows[-1].get("communication_payload_bytes_per_rank"),
            "controller_payload_bytes_per_rank": rows[-1].get("controller_payload_bytes_per_rank", 0),
            "estimated_ring_bus_bytes_per_rank": rows[-1].get("estimated_ring_bus_bytes_per_rank"),
            "profiled_steps": rows[-1].get("profiled_steps", 0),
        })
    groups = {}
    for row in arms:
        key = (row["arm"], row["dataset"], row["world_size"], row["global_batch_size"],
               row["accumulate_steps"], row["power_rank"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, rows in sorted(groups.items()):
        times = [row["time_to_target_s"] for row in rows if row["time_to_target_s"] is not None]
        summaries.append({
            "arm": key[0], "dataset": key[1], "world_size": key[2],
            "global_batch_size": key[3], "accumulate_steps": key[4], "power_rank": key[5],
            "seed_count": len(rows), "target_reached_seeds": len(times),
            "median_time_to_target_s": statistics.median(times) if times else None,
            "median_final_validation_accuracy": statistics.median(
                row["final_validation_accuracy"] for row in rows),
            "median_final_validation_loss": statistics.median(
                row["final_validation_loss"] for row in rows),
            "median_censored_at_step": statistics.median(
                row["censored_at_step"] for row in rows if row["censored_at_step"] is not None)
                if any(row["censored_at_step"] is not None for row in rows) else None,
            "median_communication_payload_bytes_per_rank": statistics.median(
                row["communication_payload_bytes_per_rank"] for row in rows
                if row["communication_payload_bytes_per_rank"] is not None)
                if any(row["communication_payload_bytes_per_rank"] is not None for row in rows) else None,
        })
    return {"schema": "ddp-crossover.time-to-quality/v2", "runs": arms, "arm_summaries": summaries,
            "limits": ["Each JSONL file must contain a single matched run.",
                       "Wall time is process elapsed time; GPU-hours require actual device and process accounting.",
                       "Collective payload bytes are per-rank logical tensor bytes; ring bus bytes are a model, not a profiler observation.",
                       "The first target hit is interval-censored by the validation cadence.",
                       "Synthetic data is a smoke fixture, not a convergence claim."]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.traces)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({len(report['runs'])} runs)")


if __name__ == "__main__":
    main()
