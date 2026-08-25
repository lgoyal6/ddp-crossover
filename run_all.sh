#!/usr/bin/env bash
# Full step 5 study on 2x T4. Run from the repo root on a 2-GPU machine.
#
# Every arm writes into the same results/ddp_sweep.json (appending), so
# 03_report.py can compare them without re-running anything.
set -euo pipefail
mkdir -p results

echo "=== 1. NCCL collective roofline ==="
torchrun --nproc_per_node=2 01_nccl_bench.py --dtype fp16 --out results/nccl_fp16.json
torchrun --nproc_per_node=2 01_nccl_bench.py --dtype fp32 --out results/nccl_fp32.json

echo "=== 2. baseline: 1 GPU vs 2 GPU DDP across model size ==="
python 02_ddp_sweep.py --tag 1gpu
torchrun --nproc_per_node=2 02_ddp_sweep.py --tag 2gpu_ddp

echo "=== 3. mitigations ==="
# bucket size: small buckets mean more, smaller collectives
torchrun --nproc_per_node=2 02_ddp_sweep.py --tag bucket --bucket-mbs 1 5 25 100
# gradient accumulation: sync once per K micro-batches
torchrun --nproc_per_node=2 02_ddp_sweep.py --tag accum --accums 1 2 4 8
# fp16 gradient compression (T4 is Turing: fp16, never bf16)
torchrun --nproc_per_node=2 02_ddp_sweep.py --tag fp16hook --fp16-hook
# FSDP: extra all-gather of parameters should make it worse at small scale
torchrun --nproc_per_node=2 02_ddp_sweep.py --tag fsdp --fsdp

echo "=== 4. report ==="
python 03_report.py
