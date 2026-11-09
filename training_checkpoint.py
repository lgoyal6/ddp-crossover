"""Atomic full-state checkpoint helpers for fixed-world-size DDP resumes."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    try:
        torch.save(state, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
            digest = hashlib.sha256(stream.read()).hexdigest()
        os.replace(temporary, target)
        manifest = {"schema": "ddp-crossover.checkpoint/v1",
                    "sha256": digest, "complete": True}
        manifest_path = target.with_suffix(target.suffix + ".manifest.json")
        manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
        manifest_tmp.write_text(json.dumps(manifest, sort_keys=True) + "\n")
        with manifest_tmp.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(manifest_tmp, manifest_path)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path, *, world_size: int) -> dict[str, Any]:
    target = Path(path)
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "ddp-crossover.checkpoint/v1" or not manifest.get("complete"):
        raise ValueError("checkpoint has no complete supported manifest")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != manifest.get("sha256"):
        raise ValueError("checkpoint integrity hash mismatch")
    state = torch.load(target, map_location="cpu", weights_only=False)
    if state.get("world_size") != world_size:
        raise ValueError("checkpoint world size differs from requested world size")
    required = {"model", "optimizer", "scheduler", "scaler", "step", "epoch",
                "sampler_epoch", "batch_in_epoch", "rng", "rng_by_rank",
                "compressor_by_rank", "world_size", "config"}
    missing = required - state.keys()
    if missing:
        raise ValueError(f"incomplete training checkpoint: {sorted(missing)}")
    rng_by_rank = state["rng_by_rank"]
    if not isinstance(rng_by_rank, list) or len(rng_by_rank) != world_size:
        raise ValueError("checkpoint RNG state count differs from world size")
    for rank, rng in enumerate(rng_by_rank):
        if (not isinstance(rng, dict) or not isinstance(rng.get("python"), tuple) or
                not torch.is_tensor(rng.get("torch")) or rng["torch"].dtype != torch.uint8):
            raise ValueError(f"checkpoint RNG state for rank {rank} is incomplete")
        if "cuda" in rng and (not isinstance(rng["cuda"], list) or
                              any(not torch.is_tensor(value) or value.dtype != torch.uint8
                                  for value in rng["cuda"])):
            raise ValueError(f"checkpoint CUDA RNG state for rank {rank} is malformed")
    compressor_by_rank = state["compressor_by_rank"]
    if not isinstance(compressor_by_rank, list) or len(compressor_by_rank) != world_size:
        raise ValueError("checkpoint compressor state count differs from world size")
    if any(not isinstance(compressor, dict) for compressor in compressor_by_rank):
        raise ValueError("checkpoint compressor state must be a dictionary for every rank")
    return state
