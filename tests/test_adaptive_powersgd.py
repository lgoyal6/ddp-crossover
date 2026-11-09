from __future__ import annotations

import os
import tempfile
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from adaptive_powersgd import AdaptivePowerSGDState, adaptive_power_sgd_hook


def _distributed_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}",
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    torch.manual_seed(7)
    model = DDP(torch.nn.Sequential(torch.nn.Linear(256, 256), torch.nn.Tanh(),
                                    torch.nn.Linear(256, 256), torch.nn.Tanh(),
                                    torch.nn.Linear(256, 4)), bucket_cap_mb=0.1)
    state = AdaptivePowerSGDState(
        dist.group.WORLD, ranks=(1, 2, 4), initial_rank=1,
        update_every=1, residual_budget=1e-6, start_compression_iter=2,
        min_compression_rate=1.0,
    )
    model.register_comm_hook(state, adaptive_power_sgd_hook)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    for step in range(5):
        torch.manual_seed(100 + step + rank * 1000)
        x = torch.randn(8, 256)
        y = torch.randint(0, 4, (8,))
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x), y).backward()
        opt.step()

    # Rank decisions must be bit-identical on all replicas.
    ranks = {index: power.matrix_approximation_rank
             for index, power in state.power_by_bucket.items()}
    agreed = [None, None]
    dist.all_gather_object(agreed, (state.step, ranks, state.decisions))
    assert agreed[0] == agreed[1]
    assert len(ranks) > 1, f"expected multiple DDP buckets, got {sorted(ranks)}"
    assert ranks and all(rank in state.ranks for rank in ranks.values())
    assert state.communication_payload_bytes > 0
    assert state.communication_collectives > 0
    assert state.controller_payload_bytes > 0
    assert state.controller_collectives > 0
    assert state.samples_by_bucket
    assert any(power.error_dict for power in state.power_by_bucket.values())

    # Compressor internals, rank, controller history and feedback survive a
    # same-world-size process restart through the state_dict contract.
    fresh = AdaptivePowerSGDState(
        dist.group.WORLD, ranks=(1, 2, 4), initial_rank=1,
        update_every=1, residual_budget=1e-6, start_compression_iter=2,
        min_compression_rate=1.0,
    )
    fresh.load_state_dict(state.state_dict())
    assert fresh.step == state.step
    assert fresh.communication_payload_bytes == state.communication_payload_bytes
    assert fresh.controller_payload_bytes == state.controller_payload_bytes
    assert fresh.power_by_bucket.keys() == state.power_by_bucket.keys()
    for bucket_id in state.power_by_bucket:
        assert fresh.power_by_bucket[bucket_id].matrix_approximation_rank == state.power_by_bucket[bucket_id].matrix_approximation_rank
        assert fresh.power_by_bucket[bucket_id].error_dict.keys() == state.power_by_bucket[bucket_id].error_dict.keys()
        for key in state.power_by_bucket[bucket_id].error_dict:
            assert torch.equal(fresh.power_by_bucket[bucket_id].error_dict[key], state.power_by_bucket[bucket_id].error_dict[key])
    dist.destroy_process_group()


def test_gloo_adaptive_hook_and_state_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        rendezvous = os.path.join(directory, "gloo-init")
        context = mp.spawn(_distributed_worker, args=(rendezvous,), nprocs=2, join=False)
        deadline = time.monotonic() + 35
        while time.monotonic() < deadline:
            if context.join(timeout=min(1.0, deadline - time.monotonic())):
                break
        else:
            live = [process.pid for process in context.processes if process.is_alive()]
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
            for process in context.processes:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            raise TimeoutError(f"adaptive Gloo hook did not exit; live PIDs: {live}")


def test_configuration_and_checkpoint_mismatch_are_rejected() -> None:
    state = AdaptivePowerSGDState(None)
    snapshot = state.state_dict()
    snapshot["config"]["ranks"] = (1, 3)
    try:
        state.load_state_dict(snapshot)
    except ValueError as exc:
        assert "configuration mismatch" in str(exc)
    else:
        raise AssertionError("mismatched compressor config was accepted")
