"""Step 5.2/5.3/5.4 -- where does the second GPU stop paying, and what moves it.

Trains the same model 1-GPU and 2-GPU DDP across a parameter-count sweep and
finds the size below which 2-GPU throughput falls *below* 1-GPU. The small
hardware is the point: 2x T4 over PCIe with no NVLink is the regime where
interconnect cost is visible, and the finding is about the interconnect.

All-reduce fraction of step time is instrumented directly rather than inferred,
by timing the backward pass with and without gradient synchronization
(`model.no_sync()` suppresses the all-reduce while doing identical compute), so
the difference is the synchronization cost and nothing else.

Mitigations swept, each of which moves the crossover for a different reason:

  * `bucket_cap_mb` -- how much gradient is coalesced per all-reduce. Small
    buckets mean more, smaller collectives, and the NCCL curve from
    01_nccl_bench.py shows small messages are latency-bound.
  * gradient accumulation -- syncs once per K micro-batches instead of every
    one, which divides communication by K and is the bluntest fix available.
  * fp16 gradient compression via a DDP comm hook -- halves the bytes on the
    wire. **fp16, not bf16**: T4 is Turing (sm_75) and has no bf16 support at
    all, so a bf16 hook would either fail or silently emulate.
  * FSDP vs DDP at the same size -- FSDP adds an all-gather of parameters on top
    of the gradient reduce-scatter, so at small scale it should be *worse*, and
    that is the counterintuitive part worth measuring rather than asserting.

Throughput is samples/second across both ranks, so a perfectly scaling 2-GPU run
is 2.0x and anything below 1.0x means the second GPU is actively harmful.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


class MLP(nn.Module):
    """Plain stack of square linears, so parameter count is a clean dial.

    Deliberately not a transformer: the question is how communication volume
    trades against compute, and an MLP lets parameter count be swept smoothly
    without also changing attention cost, sequence length or memory layout.
    """

    def __init__(self, width: int, depth: int, n_classes: int = 64):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(width, width), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        return self.head(self.body(x))


def fp16_compress_hook(state, bucket):
    """Cast the bucket to fp16 for the wire, all-reduce, cast back.

    torch ships `default_hooks.fp16_compress_hook`; this is written out so the
    bytes-on-the-wire claim is visible rather than taken on trust.
    """
    group = state if state is not None else dist.group.WORLD
    world = dist.get_world_size(group)
    buf = bucket.buffer().to(torch.float16).div_(world)
    fut = dist.all_reduce(buf, group=group, async_op=True).get_future()

    def decompress(f):
        return f.value()[0].to(bucket.buffer().dtype)

    return fut.then(decompress)


def build(width, depth, device):
    torch.manual_seed(0)
    return MLP(width, depth).to(device)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train_steps(model, opt, batch, width, device, steps, accum=1,
                no_sync_probe=False):
    """Return (seconds per optimizer step, seconds of gradient sync per step).

    `no_sync_probe` reruns the same work with DDP's all-reduce suppressed; the
    difference isolates communication from compute.
    """
    x = torch.randn(batch, width, device=device)
    y = torch.randint(0, 64, (batch,), device=device)
    lossf = nn.CrossEntropyLoss()

    def one_step(sync=True):
        opt.zero_grad(set_to_none=True)
        for k in range(accum):
            last = (k == accum - 1)
            # DDP all-reduces on the last backward of an accumulation group.
            # Suppress it for every earlier micro-batch, and for all of them
            # when probing the no-communication baseline.
            skip = isinstance(model, DDP) and (not last or not sync)
            ctx = model.no_sync() if skip else contextlib.nullcontext()
            with ctx:
                lossf(model(x), y).backward()
        opt.step()

    for _ in range(5):
        one_step()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step(sync=True)
    torch.cuda.synchronize()
    t_sync = (time.perf_counter() - t0) / steps

    t_nosync = t_sync
    if no_sync_probe and isinstance(model, DDP):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            one_step(sync=False)
        torch.cuda.synchronize()
        t_nosync = (time.perf_counter() - t0) / steps
    return t_sync, max(0.0, t_sync - t_nosync)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="*",
                    default=[256, 512, 1024, 2048, 4096, 8192])
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bucket-mbs", type=int, nargs="*", default=[25])
    ap.add_argument("--accums", type=int, nargs="*", default=[1])
    ap.add_argument("--fp16-hook", action="store_true")
    ap.add_argument("--fsdp", action="store_true")
    ap.add_argument("--tag", default="ddp")
    ap.add_argument("--out", default="results/ddp_sweep.json")
    a = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        rank, world, device = 0, 1, torch.device("cuda", 0)

    if rank == 0:
        print(f"{a.tag}: world={world} depth={a.depth} batch={a.batch} "
              f"device={torch.cuda.get_device_name(0)} "
              f"cap={torch.cuda.get_device_capability(0)}")
        hdr = (f"\n{'width':>7}{'params_M':>10}{'grad_MiB':>10}{'bucket':>8}"
               f"{'accum':>7}{'s/step':>10}{'samples/s':>12}{'comm_ms':>10}"
               f"{'comm%':>8}")
        print(hdr)

    rows = []
    for width in a.widths:
        for bucket in a.bucket_mbs:
            for accum in a.accums:
                model = build(width, a.depth, device)
                p = n_params(model)
                if ddp:
                    if a.fsdp:
                        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                        model = FSDP(model, device_id=rank)
                    else:
                        model = DDP(model, device_ids=[rank],
                                    bucket_cap_mb=bucket)
                        if a.fp16_hook:
                            model.register_comm_hook(None, fp16_compress_hook)
                opt = torch.optim.SGD(model.parameters(), lr=0.01)
                t, comm = train_steps(model, opt, a.batch, width, device,
                                      a.steps, accum=accum,
                                      no_sync_probe=ddp and not a.fsdp)
                # samples/s counts every rank's batch and every accumulation
                # micro-batch, so 1-GPU and 2-GPU numbers are comparable.
                sps = a.batch * accum * world / t
                rows.append({
                    "tag": a.tag, "world": world, "width": width,
                    "params": p, "grad_mib": p * 4 / 2**20,
                    "bucket_mb": bucket, "accum": accum,
                    "fp16_hook": a.fp16_hook, "fsdp": a.fsdp,
                    "s_per_step": t, "samples_per_s": sps,
                    "comm_s": comm, "comm_frac": comm / t if t else 0.0,
                })
                if rank == 0:
                    print(f"{width:>7}{p/1e6:>10.2f}{p*4/2**20:>10.1f}"
                          f"{bucket:>8}{accum:>7}{t*1e3:>10.2f}{sps:>12.0f}"
                          f"{comm*1e3:>10.2f}{100*comm/t:>7.1f}%")
                del model, opt
                torch.cuda.empty_cache()

    if rank == 0:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        prev = []
        if os.path.exists(a.out):
            with open(a.out) as f:
                prev = json.load(f).get("rows", [])
        with open(a.out, "w") as f:
            json.dump({"rows": prev + rows}, f, indent=2)
        print(f"\nwrote {a.out} ({len(prev)+len(rows)} rows total)")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
