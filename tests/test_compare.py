from __future__ import annotations

import json
from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location

SPEC = spec_from_file_location("ddp_compare", Path(__file__).resolve().parents[1] / "06_compare.py")
assert SPEC and SPEC.loader
compare = module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def test_report_keeps_accumulation_and_fixed_rank_arms_separate(tmp_path: Path) -> None:
    rows = [
        {"arm": "dense", "dataset": "cifar10", "seed": 1, "world_size": 2,
         "global_batch_size": 128, "accumulate_steps": 1, "power_rank": 2,
         "step": 100, "target_accuracy": 0.7, "validation_accuracy": 0.7,
         "validation_loss": 1.0, "elapsed_s": 10.0},
        {"arm": "dense", "dataset": "cifar10", "seed": 1, "world_size": 2,
         "global_batch_size": 128, "accumulate_steps": 8, "power_rank": 2,
         "step": 100, "target_accuracy": 0.7, "validation_accuracy": 0.6,
         "validation_loss": 1.2, "elapsed_s": 11.0},
        {"arm": "powersgd", "dataset": "cifar10", "seed": 1, "world_size": 2,
         "global_batch_size": 128, "accumulate_steps": 1, "power_rank": 4,
         "step": 100, "target_accuracy": 0.7, "validation_accuracy": 0.6,
         "validation_loss": 1.2, "elapsed_s": 12.0},
    ]
    traces = []
    for index, row in enumerate(rows):
        path = tmp_path / f"{index}.jsonl"
        path.write_text(json.dumps(row) + "\n")
        traces.append(path)
    report = compare.summarize(traces)
    assert len(report["runs"]) == 3
    assert len(report["arm_summaries"]) == 3
    assert {item["accumulate_steps"] for item in report["runs"]} == {1, 8}
    assert {item["power_rank"] for item in report["runs"]} == {2, 4}
    dense_unreached = next(item for item in report["runs"]
                           if item["arm"] == "dense" and item["accumulate_steps"] == 8)
    assert dense_unreached["censored_at_step"] == 100
