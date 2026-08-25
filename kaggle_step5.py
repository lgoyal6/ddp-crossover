"""Kaggle entry point for the 2x T4 study.

A Kaggle kernel is a single process, so `torchrun` is not available. This spawns
the ranks itself with `torch.multiprocessing`, setting the same environment
variables torchrun would, and then runs the unmodified study scripts through
`runpy`. Keeping the scripts unchanged means the Kaggle run and a local
`torchrun` run execute identical code.

Expects `nvidiaTeslaT4x2`. Fails loudly rather than silently measuring a
single-GPU machine, because a 1-GPU run of this study would produce a table full
of 1.00x scaling that looks like a result.
"""
from __future__ import annotations

import os
import runpy
import sys

import torch
import torch.multiprocessing as mp


def _worker(rank: int, world: int, script: str, argv: list[str], port: str):
    os.environ.update({
        "RANK": str(rank), "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world),
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": port,
    })
    sys.argv = [script] + argv
    runpy.run_path(script, run_name="__main__")


def launch(script: str, argv: list[str], world: int = 2, port: str = "29517"):
    print(f"\n$ [{world} ranks] {script} {' '.join(argv)}", flush=True)
    mp.spawn(_worker, args=(world, script, argv, port), nprocs=world,
             join=True)


def single(script: str, argv: list[str]):
    """Run a script as a plain 1-GPU process (no distributed env set)."""
    print(f"\n$ [1 rank] {script} {' '.join(argv)}", flush=True)
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        os.environ.pop(k, None)
    sys.argv = [script] + argv
    runpy.run_path(script, run_name="__main__")


def main():
    n = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n)]
    caps = [torch.cuda.get_device_capability(i) for i in range(n)]
    print(f"visible GPUs: {n} -> {names} {caps}")
    if n < 2:
        raise SystemExit(
            f"This study needs 2 GPUs and sees {n}. In Kaggle, set "
            "Settings > Accelerator > GPU T4 x2. Running it on one GPU would "
            "produce a table of 1.00x scaling that looks like a finding.")
    if caps[0][0] >= 8:
        print("NOTE: this GPU supports bf16; the study is written for Turing "
              "(sm_75) where fp16 is the only option. Results will not be "
              "comparable to a T4 run.")

    os.makedirs("results", exist_ok=True)

    launch("01_nccl_bench.py", ["--dtype", "fp16",
                                "--out", "results/nccl_fp16.json"])
    launch("01_nccl_bench.py", ["--dtype", "fp32",
                                "--out", "results/nccl_fp32.json"])

    single("02_ddp_sweep.py", ["--tag", "1gpu"])
    launch("02_ddp_sweep.py", ["--tag", "2gpu_ddp"])

    launch("02_ddp_sweep.py", ["--tag", "bucket",
                               "--bucket-mbs", "1", "5", "25", "100"])
    launch("02_ddp_sweep.py", ["--tag", "accum",
                               "--accums", "1", "2", "4", "8"])
    launch("02_ddp_sweep.py", ["--tag", "fp16hook", "--fp16-hook"])
    launch("02_ddp_sweep.py", ["--tag", "fsdp", "--fsdp"])

    single("03_report.py", [])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
