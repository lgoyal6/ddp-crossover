from __future__ import annotations

import json
import os
import runpy
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from training_checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def _run(rank: int, init_file: str, argv: list[str]) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    sys.argv = [str(ROOT / "05_learnable_train.py"), *argv,
                "--init-method", f"file://{init_file}"]
    runpy.run_path(str(ROOT / "05_learnable_train.py"), run_name="__main__")


def _spawn_bounded(argv: list[str], *, rendezvous: str, timeout: float = 40) -> None:
    context = mp.spawn(_run, args=(rendezvous, argv), nprocs=2, join=False)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if context.join(timeout=min(1.0, deadline - time.monotonic())):
            return
    live = [process.pid for process in context.processes if process.is_alive()]
    for process in context.processes:
        if process.is_alive():
            process.terminate()
    for process in context.processes:
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
    raise TimeoutError(f"Gloo training processes failed to exit within {timeout}s; live PIDs: {live}")


def _assert_tree_equal(left, right, path="state") -> None:
    if type(left).__name__ == "RandomState" and type(right).__name__ == "RandomState":
        _assert_tree_equal(left.get_state(), right.get_state(), path)
    elif torch.is_tensor(left):
        assert torch.equal(left, right), path
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right), path
    elif isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_tree_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right)):
            _assert_tree_equal(a, b, f"{path}[{index}]")
    else:
        assert left == right, path


def test_adaptive_learnable_smoke_and_full_checkpoint_resume() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first_file = str(root / "gloo-first")
        first_ckpt = root / "first.pt"
        common = ["--dataset", "synthetic", "--arm", "adaptive", "--batch-size", "8",
                  "--accumulate-steps", "2", "--update-every", "1",
                  "--target-accuracy", "0.99", "--steps", "5",
                  "--dist-timeout-seconds", "20"]
        _spawn_bounded(common + ["--stop-after", "4", "--output",
                       str(root / "first.jsonl"), "--checkpoint", str(first_ckpt)],
                       rendezvous=first_file)
        state = load_checkpoint(first_ckpt, world_size=2)
        assert state["step"] == 4
        assert state["batch_in_epoch"] == 8
        assert len(state["compressor_by_rank"]) == 2
        assert state["compressor_by_rank"][0]["power_by_bucket"]
        assert state["compressor_by_rank"][0]["decisions"]

        prefix_file = str(root / "gloo-prefix")
        prefix_ckpt = root / "prefix.pt"
        _spawn_bounded(common + ["--stop-after", "4", "--output",
                       str(root / "prefix.jsonl"), "--checkpoint", str(prefix_ckpt)],
                       rendezvous=prefix_file)
        prefix = load_checkpoint(prefix_ckpt, world_size=2)
        for key in ("model", "optimizer", "scheduler", "scaler", "compressor_by_rank", "rng_by_rank"):
            _assert_tree_equal(state[key], prefix[key], key)

        second_file = str(root / "gloo-resume")
        _spawn_bounded(common + ["--resume", str(first_ckpt), "--output",
                       str(root / "resumed.jsonl"), "--checkpoint",
                       str(root / "resumed.pt")], rendezvous=second_file)
        resumed = load_checkpoint(root / "resumed.pt", world_size=2)
        assert resumed["step"] == 5
        assert all(compressor["step"] >= state["compressor_by_rank"][rank]["step"]
                   for rank, compressor in enumerate(resumed["compressor_by_rank"]))

        reference_file = str(root / "gloo-reference")
        reference_ckpt = root / "reference.pt"
        _spawn_bounded(common + ["--output", str(root / "reference.jsonl"),
                       "--checkpoint", str(reference_ckpt)], rendezvous=reference_file)
        reference = load_checkpoint(reference_ckpt, world_size=2)
        repeat_file = str(root / "gloo-repeat")
        repeat_ckpt = root / "repeat.pt"
        _spawn_bounded(common + ["--output", str(root / "repeat.jsonl"),
                       "--checkpoint", str(repeat_ckpt)], rendezvous=repeat_file)
        repeat = load_checkpoint(repeat_ckpt, world_size=2)
        repeat_delta = max((value - repeat["model"][key]).abs().max().item()
                           for key, value in reference["model"].items())
        assert repeat_delta <= 1e-4, f"independent uninterrupted repeats differ by {repeat_delta}"
        resumed_loss = json.loads((root / "resumed.jsonl").read_text().splitlines()[-1])["train_loss"]
        reference_loss = json.loads((root / "reference.jsonl").read_text().splitlines()[-1])["train_loss"]
        assert resumed_loss == reference_loss, (
            f"step-5 rank-0 loss differs: resumed {resumed_loss}, reference {reference_loss}")
        _assert_tree_equal(reference["compressor_by_rank"], resumed["compressor_by_rank"],
                           "compressor_by_rank")
        for key, value in reference["model"].items():
            max_error = (value - resumed["model"][key]).abs().max().item()
            assert max_error <= 1e-4, (
                f"{key} resume max absolute error {max_error} exceeds 1e-4 "
                f"(uninterrupted-repeat delta {repeat_delta})")


def test_dense_checkpoint_resume_matches_uninterrupted() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        common = ["--dataset", "synthetic", "--arm", "dense", "--batch-size", "8",
                  "--accumulate-steps", "2", "--steps", "5",
                  "--dist-timeout-seconds", "20"]
        first = root / "dense-first.pt"
        _spawn_bounded(common + ["--stop-after", "4", "--output", str(root / "first.jsonl"),
                       "--checkpoint", str(first), "--profiler-trace",
                       str(root / "dense-rank{rank}.trace.json"), "--profiler-steps", "2"],
                       rendezvous=str(root / "first-init"))
        assert (root / "dense-rank0.trace.json").is_file()
        assert (root / "dense-rank1.trace.json").is_file()
        first_row = json.loads((root / "first.jsonl").read_text().splitlines()[-1])
        assert first_row["profiled_steps"] == 2
        assert first_row["communication_payload_bytes_per_rank"] > 0
        assert first_row["estimated_ring_bus_bytes_per_rank"] == first_row["communication_payload_bytes_per_rank"]
        _spawn_bounded(common + ["--resume", str(first), "--output", str(root / "resume.jsonl"),
                       "--checkpoint", str(root / "resumed.pt")], rendezvous=str(root / "resume-init"))
        _spawn_bounded(common + ["--output", str(root / "reference.jsonl"),
                       "--checkpoint", str(root / "reference.pt")], rendezvous=str(root / "reference-init"))
        resumed = load_checkpoint(root / "resumed.pt", world_size=2)
        reference = load_checkpoint(root / "reference.pt", world_size=2)
        max_error = max((value - resumed["model"][key]).abs().max().item()
                        for key, value in reference["model"].items())
        assert max_error <= 1e-4, f"dense resume max absolute error {max_error} exceeds 1e-4"
