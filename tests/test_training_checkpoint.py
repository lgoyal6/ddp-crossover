from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest
import torch

from training_checkpoint import load_checkpoint, save_checkpoint


def _state(world_size: int = 2) -> dict:
    rng_by_rank = [{"python": random.getstate(), "torch": torch.get_rng_state()}
                   for _ in range(world_size)]
    return {
        "model": {}, "optimizer": {}, "scheduler": {}, "scaler": {},
        "step": 1, "epoch": 0, "sampler_epoch": 0, "batch_in_epoch": 1,
        "rng": rng_by_rank[0], "rng_by_rank": rng_by_rank,
        "compressor_by_rank": [{} for _ in range(world_size)],
        "world_size": world_size, "config": {},
    }


def test_atomic_checkpoint_requires_integrity_and_all_rank_states() -> None:
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "train.pt"
        save_checkpoint(checkpoint, _state())
        assert len(load_checkpoint(checkpoint, world_size=2)["compressor_by_rank"]) == 2

        checkpoint.write_bytes(checkpoint.read_bytes() + b"corruption")
        with pytest.raises(ValueError, match="integrity hash mismatch"):
            load_checkpoint(checkpoint, world_size=2)

        checkpoint = Path(directory) / "wrong-ranks.pt"
        invalid = _state()
        invalid["compressor_by_rank"] = [{}]
        save_checkpoint(checkpoint, invalid)
        with pytest.raises(ValueError, match="compressor state count"):
            load_checkpoint(checkpoint, world_size=2)

        invalid["compressor_by_rank"] = [{}, {}]
        invalid["rng_by_rank"] = [invalid["rng_by_rank"][0]]
        save_checkpoint(checkpoint, invalid)
        with pytest.raises(ValueError, match="RNG state count"):
            load_checkpoint(checkpoint, world_size=2)
