"""Step 5.1 -- raw NCCL collective bandwidth on 2x T4 over PCIe.

This is the roofline the training study needs: every gradient synchronization in
DDP is an all-reduce, so the crossover where a second GPU stops paying is set by
this curve and by how much of a step is spent on it.

Two collectives, swept from a few KB to a few hundred MB:

  * **all-reduce**, which is what DDP gradient sync uses.
  * **all-gather**, which is what FSDP additionally needs for parameters, and is
    the reason FSDP should be *worse* than DDP at small scale.

Bus bandwidth is reported alongside raw size/time, using NCCL's own convention:
for ring all-reduce each rank sends and receives 2*(N-1)/N of the buffer, so
bus_bw = algbw * 2*(N-1)/N. Quoting algorithm bandwidth instead would flatter
all-reduce by ~2x against all-gather and make the comparison meaningless.

Turing (sm_75) has **no bf16**. Everything here is fp16 or fp32.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
import torch.distributed as dist


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    start, end = (torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True))
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1e3 / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--out", default="results/nccl.json")
    a = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dt = torch.float16 if a.dtype == "fp16" else torch.float32
    esize = 2 if dt == torch.float16 else 4

    if rank == 0:
        print(f"world={world} dtype={a.dtype} "
              f"device={torch.cuda.get_device_name(rank)} "
              f"cap={torch.cuda.get_device_capability(rank)}")
        print(f"\n{'bytes':>12}{'MiB':>9}{'allreduce ms':>14}{'algbw GB/s':>12}"
              f"{'busbw GB/s':>12}{'allgather ms':>14}{'ag busbw':>10}")

    sizes = [2**i for i in range(12, 29)]      # 4 KiB .. 256 MiB
    rows = []
    for nbytes in sizes:
        n = nbytes // esize
        try:
            x = torch.ones(n, dtype=dt, device=rank)
            g = torch.empty(n * world, dtype=dt, device=rank)
        except torch.cuda.OutOfMemoryError:
            break
        iters = 20 if nbytes <= 2**24 else 5
        t_ar = bench(lambda: dist.all_reduce(x), iters=iters)
        t_ag = bench(lambda: dist.all_gather_into_tensor(g, x), iters=iters)

        # NCCL convention. Ring all-reduce moves 2*(N-1)/N of the buffer per
        # rank; all-gather moves (N-1)/N.
        alg_ar = nbytes / t_ar / 1e9
        bus_ar = alg_ar * 2 * (world - 1) / world
        bus_ag = (nbytes / t_ag / 1e9) * (world - 1) / world
        rows.append({"bytes": nbytes, "allreduce_s": t_ar,
                     "allreduce_algbw_gbs": alg_ar,
                     "allreduce_busbw_gbs": bus_ar,
                     "allgather_s": t_ag, "allgather_busbw_gbs": bus_ag})
        if rank == 0:
            print(f"{nbytes:>12}{nbytes/2**20:>9.2f}{t_ar*1e3:>14.3f}"
                  f"{alg_ar:>12.2f}{bus_ar:>12.2f}{t_ag*1e3:>14.3f}"
                  f"{bus_ag:>10.2f}")
        del x, g
        torch.cuda.empty_cache()

    if rank == 0:
        peak = max(r["allreduce_busbw_gbs"] for r in rows)
        small = rows[0]
        print(f"\npeak all-reduce bus bandwidth: {peak:.2f} GB/s")
        print(f"small-message latency floor ({small['bytes']} B): "
              f"{small['allreduce_s']*1e6:.1f} us")
        half = peak / 2
        cross = next((r["bytes"] for r in rows
                      if r["allreduce_busbw_gbs"] >= half), None)
        print(f"reaches half of peak at {cross} bytes "
              f"({cross/2**20:.2f} MiB)" if cross else "")
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"world": world, "dtype": a.dtype,
                       "device": torch.cuda.get_device_name(0),
                       "peak_allreduce_busbw_gbs": peak, "rows": rows}, f,
                      indent=2)
        print(f"wrote {a.out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
