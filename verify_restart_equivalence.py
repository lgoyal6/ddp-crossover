"""Check same-topology adaptive restart against uninterrupted training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from training_checkpoint import load_checkpoint


def compare(left, right, name: str, *, atol: float = 1e-5, failures: list[str]) -> None:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape or left.dtype != right.dtype or not torch.allclose(left, right, atol=atol, rtol=atol):
            delta = float("inf") if left.shape != right.shape else float((left - right).abs().max().item())
            failures.append(f"{name}: max_abs_error={delta}")
    elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if not np.array_equal(left, right):
            failures.append(f"{name}: array mismatch")
    elif isinstance(left, np.random.RandomState) and isinstance(right, np.random.RandomState):
        compare(left.get_state(), right.get_state(), name, atol=atol, failures=failures)
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            failures.append(f"{name}: key mismatch")
            return
        for key in left:
            compare(left[key], right[key], f"{name}.{key}", atol=atol, failures=failures)
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            failures.append(f"{name}: sequence shape mismatch")
            return
        for index, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f"{name}[{index}]", atol=atol, failures=failures)
    elif left != right:
        failures.append(f"{name}: value mismatch ({left!r} != {right!r})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("partial", type=Path)
    parser.add_argument("resumed", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise SystemExit("restart correctness stage requires exactly two visible CUDA devices")
    partial = load_checkpoint(args.partial, world_size=2)
    resumed = load_checkpoint(args.resumed, world_size=2)
    reference = load_checkpoint(args.reference, world_size=2)
    fields = ("model", "optimizer", "scheduler", "scaler", "compressor_by_rank", "rng_by_rank")
    failures: list[str] = []
    if partial.get("step") != 3:
        failures.append(f"partial.step: expected 3, got {partial.get('step')}")
    if resumed.get("step") != 6 or reference.get("step") != 6:
        failures.append(f"final steps: resumed={resumed.get('step')} reference={reference.get('step')}")
    partial_compressors = partial.get("compressor_by_rank", [])
    if len(partial_compressors) != 2 or any(not state.get("power_by_bucket") for state in partial_compressors):
        failures.append("partial compressor checkpoint is missing per-rank bucket state")
    for rank, state in enumerate(partial_compressors):
        if not any(bucket.get("error_dict") for bucket in state.get("power_by_bucket", {}).values()):
            failures.append(f"partial compressor checkpoint has no rank-{rank} error feedback")
    for field in fields:
        compare(resumed.get(field), reference.get(field), field, failures=failures)
    result = {"schema": "ddp-crossover.restart-equivalence/v1", "status": "pass" if not failures else "fail",
              "topology": {"world_size": 2, "backend": "nccl"}, "atol_rtol": 1e-5,
              "fields_compared": list(fields), "partial_checkpoint": str(args.partial),
              "resumed_checkpoint": str(args.resumed),
              "reference_checkpoint": str(args.reference), "failures": failures}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
