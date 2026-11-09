"""Two-GPU NCCL control check with one rank-local gradient fault injection."""
from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from adaptive_powersgd import AdaptivePowerSGDState, adaptive_power_sgd_hook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dist-timeout-seconds", type=int, default=20)
    args = parser.parse_args()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world != 2 or not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise SystemExit("NCCL fault control requires exactly two visible CUDA devices")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.dist_timeout_seconds))
    try:
        torch.manual_seed(701)
        model = DDP(torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Tanh(),
                                        torch.nn.Linear(256, 4)).to(device), device_ids=[local_rank],
                    bucket_cap_mb=0.1)
        state = AdaptivePowerSGDState(
            dist.group.WORLD, ranks=(1, 2, 4), initial_rank=1, update_every=1,
            residual_budget=1e-6, start_compression_iter=2, min_compression_rate=1.0)
        injected = False

        def faulted_hook(hook_state: AdaptivePowerSGDState, bucket):
            nonlocal injected
            # Distinct rank-local gradients are normal in DDP. Add one finite,
            # deterministic anomaly to one rank/bucket to pressure the shared
            # controller and verify that the compressed collective still
            # produces identical replica updates.
            if rank == 1 and hook_state.step == 2 and bucket.index() == 0 and not injected:
                bucket.buffer()[0].add_(1000.0)
                injected = True
            return adaptive_power_sgd_hook(hook_state, bucket)

        model.register_comm_hook(state, faulted_hook)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for step in range(6):
            torch.manual_seed(900 + step + rank * 10000)
            x = torch.randn(16, 256, device=device)
            y = torch.randint(0, 4, (16,), device=device)
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(model(x), y).backward()
            optimizer.step()
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise AssertionError(f"rank {rank} produced non-finite model state at step {step}")

        hit = torch.tensor([int(injected)], device=device)
        dist.all_reduce(hit, op=dist.ReduceOp.SUM)
        if int(hit.item()) != 1:
            raise AssertionError("rank-local gradient fault was not injected exactly once")
        schedules = [None] * world
        dist.all_gather_object(schedules, (state.step, state.decisions))
        if schedules[0] != schedules[1]:
            raise AssertionError("adaptive rank/controller schedule diverged across NCCL ranks")
        replica_errors = []
        for parameter in model.module.parameters():
            minimum, maximum = parameter.detach().clone(), parameter.detach().clone()
            dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            replica_errors.append(float((maximum - minimum).abs().max().item()))
        worst_replica_error = max(replica_errors, default=0.0)
        if worst_replica_error > 1e-6:
            raise AssertionError(f"replica update mismatch {worst_replica_error}")
        if rank == 0:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({
                "schema": "ddp-crossover.nccl-gradient-fault/v1", "status": "pass",
                "backend": "nccl", "world_size": world, "steps": state.step,
                "fault": "rank 1 bucket 0 finite +1000 gradient perturbation at compressed step 2",
                "fault_injection_count": int(hit.item()), "adaptive_schedule": state.decisions,
                "worst_replica_parameter_error": worst_replica_error,
                "communication_payload_bytes_per_rank": state.communication_payload_bytes,
                "controller_payload_bytes_per_rank": state.controller_payload_bytes,
                "rank_states_identical": True,
            }, indent=2, sort_keys=True) + "\n")
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
