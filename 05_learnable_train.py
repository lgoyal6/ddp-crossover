"""Matched CIFAR-10 training arms for dense, FP16, fixed and adaptive DDP.

`--dataset synthetic` is only a fast executable smoke fixture. Study claims
must use the pinned CIFAR-10 archive and a predeclared validation target.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
import pickle
import random
import tarfile
import time
import urllib.request
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, TensorDataset

from adaptive_powersgd import AdaptivePowerSGDState, adaptive_power_sgd_hook
from training_checkpoint import load_checkpoint, save_checkpoint

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"


@dataclass
class CommunicationMeter:
    """Count logical all-reduce payload bytes per rank, before ring factors."""
    payload_bytes: int = 0
    collectives: int = 0


def _counted_hook(delegate, meter: CommunicationMeter, *, bytes_for_bucket):
    def hook(process_group, bucket):
        payload_bytes, collectives = bytes_for_bucket(bucket)
        future = delegate(process_group, bucket)

        def counted(done):
            meter.payload_bytes += payload_bytes
            meter.collectives += collectives
            return done.value()[0]

        return future.then(counted)

    return hook


def _power_sgd_hook(meter: CommunicationMeter):
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook

    def hook(state, bucket):
        before = state.total_numel_after_compression
        warmup = state.iter < state.start_powerSGD_iter
        future = powerSGD_hook.powerSGD_hook(state, bucket)
        compressed_elements = state.total_numel_after_compression - before
        if warmup:
            payload_bytes = bucket.buffer().numel() * bucket.buffer().element_size()
            collectives = 1
        else:
            payload_bytes = compressed_elements * bucket.buffer().element_size()
            collectives = 3

        def counted(done):
            meter.payload_bytes += payload_bytes
            meter.collectives += collectives
            return done.value()[0]

        return future.then(counted)

    return hook


class TinyConv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2), torch.nn.Conv2d(16, 32, 3, padding=1),
            torch.nn.ReLU(), torch.nn.MaxPool2d(2), torch.nn.Flatten(),
            torch.nn.Linear(32 * 8 * 8, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(),
                                       torch.nn.Linear(32, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CifarData(Dataset):
    def __init__(self, root: Path, train: bool, seed: int) -> None:
        archive = root / "cifar-10-python.tar.gz"
        root.mkdir(parents=True, exist_ok=True)
        extracted = root / "cifar-10-batches-py"
        if not extracted.exists():
            if not archive.exists():
                urllib.request.urlretrieve(CIFAR_URL, archive)
            md5 = hashlib.md5(archive.read_bytes()).hexdigest()
            if md5 != CIFAR_MD5:
                raise ValueError(f"CIFAR-10 archive MD5 mismatch: {md5}")
            with tarfile.open(archive, "r:gz") as bundle:
                bundle.extractall(root, filter="data")
        names = ([f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"])
        images, labels = [], []
        for name in names:
            with (extracted / name).open("rb") as stream:
                record = pickle.load(stream, encoding="bytes")
            images.append(torch.from_numpy(np.asarray(record[b"data"]).copy()).view(-1, 3, 32, 32))
            labels.extend(record[b"labels"])
        self.images = torch.cat(images).float().div_(255)
        mean = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
        std = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
        self.images = (self.images - mean) / std
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.augment = train
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.images[index]
        if self.augment:
            generator = torch.Generator().manual_seed(self.seed + self.epoch * len(self) + index)
            image = torch.nn.functional.pad(image, (4, 4, 4, 4))
            top = int(torch.randint(0, 9, (), generator=generator))
            left = int(torch.randint(0, 9, (), generator=generator))
            image = image[:, top:top + 32, left:left + 32]
            if bool(torch.randint(0, 2, (), generator=generator)):
                image = image.flip(-1)
        return image, self.labels[index]


def synthetic_dataset(seed: int, count: int = 1024) -> Dataset:
    generator = torch.Generator().manual_seed(seed)
    prototypes = torch.randn(10, 16, generator=generator)
    labels = torch.arange(count) % 10
    images = prototypes[labels] + 0.35 * torch.randn(count, 16, generator=generator)
    return TensorDataset(images, labels)


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("dense", "fp16", "powersgd", "adaptive"), default="dense")
    parser.add_argument("--dataset", choices=("cifar10", "synthetic"), default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("run.jsonl"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--stop-after", type=int,
                        help="stop this process segment at this global step while keeping --steps as the total schedule horizon")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="effective global batch size per optimizer update")
    parser.add_argument("--accumulate-steps", type=int, default=1,
                        help="micro-batches per optimizer update; effective global batch stays fixed")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-accuracy", type=float, default=0.70)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--update-every", type=int, default=32)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--init-method", default="env://",
                        help="torch.distributed init method; file:// is useful for local Gloo checks")
    parser.add_argument("--dist-timeout-seconds", type=int, default=180,
                        help="distributed collective timeout for detecting stalled collectives")
    parser.add_argument("--profiler-trace", type=Path,
                        help="optional Chrome trace path; `{rank}` is replaced for each process")
    parser.add_argument("--profiler-steps", type=int, default=16,
                        help="number of leading optimizer steps to capture in each rank trace")
    args = parser.parse_args()
    stop_after = args.steps if args.stop_after is None else args.stop_after
    if (args.steps < 1 or args.batch_size < 1 or args.accumulate_steps < 1 or args.eval_every < 1 or
            args.dist_timeout_seconds < 1 or args.profiler_steps < 1 or
            not 1 <= stop_after <= args.steps):
        parser.error("steps, stop-after, batch-size, accumulate-steps, eval-every, profiler-steps, and dist-timeout-seconds are invalid")

    distributed = "RANK" in os.environ
    if distributed:
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                init_method=args.init_method, rank=rank, world_size=world,
                                timeout=timedelta(seconds=args.dist_timeout_seconds))
    else:
        rank, world = 0, 1
    divisor = world * args.accumulate_steps
    if args.batch_size % divisor:
        parser.error("global batch size must be divisible by world size times accumulate-steps")
    local_batch_size = args.batch_size // divisor
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    amp_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    if args.precision == "fp16" and device.type != "cuda":
        parser.error("fp16 training requires CUDA")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        parser.error("bf16 is not supported by this CUDA device")
    seed_all(args.seed)

    train_data = (CifarData(args.data_dir, True, args.seed) if args.dataset == "cifar10"
                  else synthetic_dataset(args.seed))
    valid_data = (CifarData(args.data_dir, False, args.seed + 1) if args.dataset == "cifar10"
                  else synthetic_dataset(args.seed + 1, 256))
    sampler = DistributedSampler(train_data, num_replicas=world, rank=rank,
                                 shuffle=True, seed=args.seed) if distributed else None
    train_loader = DataLoader(train_data, batch_size=local_batch_size, sampler=sampler,
                              shuffle=sampler is None, num_workers=0, drop_last=True)
    valid_generator = torch.Generator().manual_seed(args.seed + 1 + rank)
    valid_loader = DataLoader(valid_data, batch_size=local_batch_size * 2, shuffle=False,
                              generator=valid_generator)

    raw_model = (TinyConv() if args.dataset == "cifar10" else TinyMLP()).to(device)
    model = DDP(raw_model, device_ids=[rank] if device.type == "cuda" else None) if distributed else raw_model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    comm_meter = CommunicationMeter()
    compressor = None
    if distributed and args.arm in ("fp16", "powersgd", "adaptive"):
        if args.arm == "fp16":
            pass
        elif args.arm == "powersgd":
            from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
            compressor = powerSGD_hook.PowerSGDState(
                dist.group.WORLD, matrix_approximation_rank=args.rank,
                start_powerSGD_iter=2, use_error_feedback=True, warm_start=True)
        else:
            compressor = AdaptivePowerSGDState(
                dist.group.WORLD, initial_rank=args.rank, update_every=args.update_every,
                start_compression_iter=2)
            compressor.parameter_order = {id(parameter): index
                                          for index, parameter in enumerate(model.parameters())}

    step, epoch, sampler_epoch = 0, 0, 0
    batch_in_epoch = 0
    resume_rng = None
    if args.resume:
        restored = load_checkpoint(args.resume, world_size=world)
        expected = {"dataset": args.dataset, "data_dir": str(args.data_dir),
                    "arm": args.arm, "steps": args.steps,
                    "batch_size": args.batch_size, "accumulate_steps": args.accumulate_steps,
                    "eval_every": args.eval_every,
                    "seed": args.seed, "target_accuracy": args.target_accuracy,
                    "rank": args.rank, "update_every": args.update_every,
                    "precision": args.precision}
        saved = restored["config"]
        mismatched = [key for key, value in expected.items() if saved.get(key) != value]
        if mismatched:
            parser.error("resume training configuration differs for: " + ", ".join(mismatched))
        (model.module if hasattr(model, "module") else model).load_state_dict(restored["model"])
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        scaler.load_state_dict(restored["scaler"])
        step, epoch, sampler_epoch = restored["step"], restored["epoch"], restored["sampler_epoch"]
        batch_in_epoch = restored["batch_in_epoch"]
        if compressor is not None:
            compressor_state = restored["compressor_by_rank"][rank]
        resume_rng = restored["rng_by_rank"][rank]
        saved_meter = restored.get("communication_metrics", {})
        comm_meter.payload_bytes = int(saved_meter.get("payload_bytes", 0))
        comm_meter.collectives = int(saved_meter.get("collectives", 0))

    def evaluate() -> tuple[float, float]:
        model.eval()
        correct = total = loss_sum = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    logits = model(x)
                loss_sum += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
                correct += (logits.argmax(1) == y).sum().item()
                total += len(y)
        counts = torch.tensor((correct, total, loss_sum), device=device, dtype=torch.float64)
        if distributed:
            dist.all_reduce(counts)
        model.train()
        return counts[0].item() / max(counts[1].item(), 1), counts[2].item() / max(counts[1].item(), 1)

    started = time.perf_counter()
    model.train()
    if sampler is not None:
        sampler.set_epoch(sampler_epoch)
    if hasattr(train_data, "set_epoch"):
        train_data.set_epoch(sampler_epoch)
    iterator = iter(train_loader)
    for _ in range(batch_in_epoch):
        next(iterator)
    resume_first_batch = None
    resume_bucket_rebuild_s = 0.0
    if args.resume and distributed:
        rebuild_started = time.perf_counter()
        try:
            resume_first_batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler_epoch = epoch
            batch_in_epoch = 0
            sampler.set_epoch(epoch)
            if hasattr(train_data, "set_epoch"):
                train_data.set_epoch(epoch)
            iterator = iter(train_loader)
            resume_first_batch = next(iterator)
        # DDP rebuilds its bucket layout after the first backward. Build that
        # layout before restoring per-bucket PowerSGD state, otherwise a saved
        # error buffer can be applied to a fresh reducer's different parameter
        # order. Reuse the next real batch and restore mutable model buffers.
        buffers = [(buffer, buffer.detach().clone()) for buffer in raw_model.buffers()]
        x_warm, y_warm = (tensor.to(device) for tensor in resume_first_batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            warm_loss = torch.nn.functional.cross_entropy(model(x_warm), y_warm)
        warm_loss.backward()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for buffer, saved in buffers:
                buffer.copy_(saved)
        if compressor is not None:
            if hasattr(compressor, "load_state_dict"):
                compressor.load_state_dict(compressor_state)
            else:
                compressor.__setstate__(compressor_state)
                compressor.process_group = dist.group.WORLD
        resume_bucket_rebuild_s = time.perf_counter() - rebuild_started
    if distributed and args.arm == "dense":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        model.register_comm_hook(
            dist.group.WORLD,
            _counted_hook(default_hooks.allreduce_hook, comm_meter,
                          bytes_for_bucket=lambda bucket: (
                              bucket.buffer().numel() * bucket.buffer().element_size(), 1)),
        )
    elif distributed and args.arm == "fp16":
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        model.register_comm_hook(
            dist.group.WORLD,
            _counted_hook(default_hooks.fp16_compress_hook, comm_meter,
                          bytes_for_bucket=lambda bucket: (
                              bucket.buffer().numel() * 2, 1)),
        )
    elif compressor is not None and args.arm == "powersgd":
        model.register_comm_hook(compressor, _power_sgd_hook(comm_meter))
    elif compressor is not None and args.arm == "adaptive":
        model.register_comm_hook(compressor, adaptive_power_sgd_hook)
    # DataLoader iterator construction draws a base seed from the global CPU
    # generator even with num_workers=0. Restore the saved RNG after rebuilding
    # and fast-forwarding the iterator so resumed stochastic ops match the
    # uninterrupted run's next optimizer step.
    if resume_rng is not None:
        random.setstate(resume_rng["python"])
        torch.set_rng_state(resume_rng["torch"])
        if device.type == "cuda" and "cuda" in resume_rng:
            torch.cuda.set_rng_state_all(resume_rng["cuda"])
    profiler = None
    profiler_active = False
    profile_path = None
    profiled_steps = 0
    if args.profiler_trace:
        from torch.profiler import ProfilerActivity
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        profile_path = Path(str(args.profiler_trace).format(rank=rank))
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profiler = torch.profiler.profile(activities=activities, record_shapes=True,
                                         profile_memory=True, with_stack=False)
        profiler.start()
        profiler_active = True
    while step < stop_after:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for micro_step in range(args.accumulate_steps):
            if resume_first_batch is not None:
                x, y = resume_first_batch
                resume_first_batch = None
            else:
                try:
                    x, y = next(iterator)
                except StopIteration:
                    epoch += 1
                    sampler_epoch = epoch
                    batch_in_epoch = 0
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    if hasattr(train_data, "set_epoch"):
                        train_data.set_epoch(epoch)
                    iterator = iter(train_loader)
                    x, y = next(iterator)
            batch_in_epoch += 1
            x, y = x.to(device), y.to(device)
            sync_context = (model.no_sync() if distributed and micro_step < args.accumulate_steps - 1
                            else nullcontext())
            with sync_context:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    loss = torch.nn.functional.cross_entropy(model(x), y)
                scaler.scale(loss / args.accumulate_steps).backward()
            accumulated_loss += loss.item() / args.accumulate_steps
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step += 1
        if profiler_active:
            profiler.step()
            profiled_steps += 1
            if profiled_steps >= args.profiler_steps:
                profiler.stop()
                profiler_active = False
        if step % args.eval_every == 0 or step == stop_after:
            accuracy, val_loss = evaluate()
            if rank == 0:
                row = {"schema": "ddp-crossover.train-step/v1", "arm": args.arm,
                       "dataset": args.dataset, "seed": args.seed, "step": step,
                       "world_size": world, "global_batch_size": args.batch_size,
                       "accumulate_steps": args.accumulate_steps,
                       "micro_batch_size_per_rank": local_batch_size,
                       "power_rank": args.rank,
                       "precision": args.precision, "elapsed_s": time.perf_counter() - started,
                       "resume_bucket_rebuild_s": resume_bucket_rebuild_s,
                       "communication_payload_bytes_per_rank": comm_meter.payload_bytes,
                       "communication_collectives_per_rank": comm_meter.collectives,
                       "estimated_ring_bus_bytes_per_rank": comm_meter.payload_bytes *
                           (2 * (world - 1) / world if world > 1 else 0),
                       "profiled_steps": profiled_steps,
                       "train_loss": accumulated_loss, "validation_accuracy": accuracy,
                       "validation_loss": val_loss, "target_accuracy": args.target_accuracy,
                       "target_reached": accuracy >= args.target_accuracy}
                if compressor is not None and hasattr(compressor, "decisions"):
                    row["rank_decisions"] = compressor.decisions
                    row["communication_payload_bytes_per_rank"] = compressor.communication_payload_bytes
                    row["communication_collectives_per_rank"] = compressor.communication_collectives
                    row["controller_payload_bytes_per_rank"] = compressor.controller_payload_bytes
                    row["controller_collectives_per_rank"] = compressor.controller_collectives
                    row["estimated_ring_bus_bytes_per_rank"] = (
                        compressor.communication_payload_bytes + compressor.controller_payload_bytes
                    ) * (2 * (world - 1) / world if world > 1 else 0)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("a") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
        if args.checkpoint and (step % max(1, args.steps // 5) == 0 or step == stop_after):
            rng_by_rank = [None] * world
            local_compressor_state = (
                compressor.state_dict() if hasattr(compressor, "state_dict") else
                compressor.__getstate__() if compressor is not None else {}
            )
            compressor_by_rank = [None] * world if rank == 0 else None
            if distributed:
                dist.all_gather_object(rng_by_rank, rng_state())
                dist.gather_object(local_compressor_state, compressor_by_rank, dst=0)
            else:
                rng_by_rank[0] = rng_state()
                compressor_by_rank[0] = local_compressor_state
            if rank == 0:
                packed = {"model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                          "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                          "scaler": scaler.state_dict(),
                          "step": step, "epoch": epoch, "sampler_epoch": sampler_epoch,
                          "batch_in_epoch": batch_in_epoch,
                          "rng": rng_by_rank[0], "rng_by_rank": rng_by_rank,
                          "compressor_by_rank": compressor_by_rank,
                          "communication_metrics": {"payload_bytes": comm_meter.payload_bytes,
                                                     "collectives": comm_meter.collectives},
                          "world_size": world,
                          "config": vars(args) | {"data_dir": str(args.data_dir),
                                                   "output": str(args.output),
                                                   "checkpoint": str(args.checkpoint) if args.checkpoint else None}}
                save_checkpoint(args.checkpoint, packed)
            if distributed:
                dist.barrier()
    if profiler is not None:
        if profiler_active:
            profiler.stop()
        profiler.export_chrome_trace(str(profile_path))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
