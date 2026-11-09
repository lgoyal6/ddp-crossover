"""Adaptive, checkpointable per-bucket PowerSGD hook for torch DDP.

PyTorch owns the compressor implementation. This module gives each DDP bucket
its own adaptive rank and PowerSGD feedback/factor state. All ranks reconcile a
fixed-shape bucket signature before reducing controller statistics.
"""
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook


@dataclass
class AdaptivePowerSGDState:
    process_group: Any
    ranks: tuple[int, ...] = (1, 2, 4, 8)
    initial_rank: int = 1
    update_every: int = 32
    residual_budget: float = 0.10
    hysteresis: float = 0.20
    start_compression_iter: int = 2
    parameter_order: dict[int, int] = field(default_factory=dict)
    min_compression_rate: float = 2.0
    residual_ema_decay: float = 0.8
    step: int = 0
    residual_ema_by_bucket: dict[int, float] = field(default_factory=dict)
    samples_by_bucket: dict[int, int] = field(default_factory=dict)
    power_by_bucket: dict[int, powerSGD_hook.PowerSGDState] = field(default_factory=dict)
    pending_stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    communication_step: int = -1
    communication_tail: torch.futures.Future[torch.Tensor] | None = None
    communication_payload_bytes: int = 0
    communication_collectives: int = 0
    controller_payload_bytes: int = 0
    controller_collectives: int = 0
    bucket_signatures: dict[int, int] = field(default_factory=dict)
    decisions: list[dict[str, float | int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.ranks or tuple(sorted(set(self.ranks))) != self.ranks:
            raise ValueError("ranks must be a strictly increasing tuple")
        if any(rank < 1 for rank in self.ranks):
            raise ValueError("ranks must be positive")
        if self.initial_rank not in self.ranks:
            raise ValueError("initial_rank must be present in ranks")
        if self.update_every < 1 or not 0 < self.residual_budget < 1:
            raise ValueError("update_every and residual_budget are invalid")
        if not 0 <= self.residual_ema_decay < 1:
            raise ValueError("residual_ema_decay must be in [0, 1)")

    @property
    def current_rank(self) -> int:
        return self.initial_rank

    def _power_state(self, bucket_index: int) -> powerSGD_hook.PowerSGDState:
        if bucket_index not in self.power_by_bucket:
            self.power_by_bucket[bucket_index] = powerSGD_hook.PowerSGDState(
                process_group=self.process_group,
                matrix_approximation_rank=self.initial_rank,
                start_powerSGD_iter=self.start_compression_iter,
                min_compression_rate=self.min_compression_rate,
                use_error_feedback=True,
                warm_start=True,
            )
        return self.power_by_bucket[bucket_index]

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "config": {"ranks": self.ranks, "initial_rank": self.initial_rank,
                       "update_every": self.update_every, "residual_budget": self.residual_budget,
                       "hysteresis": self.hysteresis,
                       "start_compression_iter": self.start_compression_iter,
                       "min_compression_rate": self.min_compression_rate,
                       "residual_ema_decay": self.residual_ema_decay},
            "step": self.step,
            "communication_payload_bytes": self.communication_payload_bytes,
            "communication_collectives": self.communication_collectives,
            "controller_payload_bytes": self.controller_payload_bytes,
            "controller_collectives": self.controller_collectives,
            "residual_ema_by_bucket": dict(self.residual_ema_by_bucket),
            "samples_by_bucket": dict(self.samples_by_bucket),
            "power_by_bucket": {key: value.__getstate__() for key, value in self.power_by_bucket.items()},
            "bucket_signatures": dict(self.bucket_signatures),
            "decisions": list(self.decisions),
        }

    def load_state_dict(self, saved: dict[str, Any]) -> None:
        if saved.get("version") != 2:
            raise ValueError("unsupported adaptive PowerSGD checkpoint version")
        expected = self.state_dict()["config"]
        if saved.get("config") != expected:
            raise ValueError("checkpoint adaptive PowerSGD configuration mismatch")
        self.step = int(saved["step"])
        self.communication_payload_bytes = int(saved.get("communication_payload_bytes", 0))
        self.communication_collectives = int(saved.get("communication_collectives", 0))
        self.controller_payload_bytes = int(saved.get("controller_payload_bytes", 0))
        self.controller_collectives = int(saved.get("controller_collectives", 0))
        self.residual_ema_by_bucket = {int(k): float(v) for k, v in saved["residual_ema_by_bucket"].items()}
        self.samples_by_bucket = {int(k): int(v) for k, v in saved["samples_by_bucket"].items()}
        self.bucket_signatures = {int(k): int(v) for k, v in saved["bucket_signatures"].items()}
        self.decisions = list(saved["decisions"])
        self.power_by_bucket = {}
        for key, value in saved["power_by_bucket"].items():
            restored = self._power_state(int(key))
            restored.__setstate__(value)
            restored.process_group = self.process_group


def _signature(bucket: dist.GradBucket, parameter_order: dict[int, int]) -> int:
    value = 1469598103934665603
    parameters = bucket.parameters()
    for index, tensor in enumerate(bucket.gradients()):
        parameter_id = id(parameters[index]) if index < len(parameters) else -1
        stable_order = parameter_order.get(parameter_id, index)
        value = ((value ^ stable_order) * 1099511628211) & ((1 << 63) - 1)
        for dim in tensor.shape:
            value = ((value ^ int(dim)) * 1099511628211) & ((1 << 63) - 1)
    value ^= bucket.buffer().numel()
    return value


def _move_power_state_to_device(power: powerSGD_hook.PowerSGDState,
                                device: torch.device) -> None:
    """Lazily relocate checkpoint-loaded CPU state to the active bucket device."""
    for memory in (power.error_dict, power.p_memory_dict, power.q_memory_dict):
        for index, tensor in tuple(memory.items()):
            if isinstance(tensor, torch.Tensor) and tensor.device != device:
                memory[index] = tensor.to(device=device)


def adaptive_power_sgd_hook(
    state: AdaptivePowerSGDState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    """Run per-bucket PowerSGD; adapt ranks at a synchronized fixed cadence."""
    index = bucket.index()
    debug = __import__("os").environ.get("ADAPT_DEBUG")
    if debug:
        print(f"rank={dist.get_rank()} enter bucket={index} last={bucket.is_last()} step={state.step}", flush=True)
    power = state._power_state(index)
    _move_power_state_to_device(power, bucket.buffer().device)
    # Built-in PowerSGD advances its iteration only for bucket zero. Give each
    # bucket the same optimizer step so its warm-up and compression schedule
    # are synchronized even though factor state is bucket-local.
    power.iter = state.step
    signature_value = _signature(bucket, state.parameter_order)
    old_signature = state.bucket_signatures.get(index)
    if old_signature is not None and old_signature != signature_value:
        power.p_memory_dict.pop(index, None)
        power.q_memory_dict.pop(index, None)
        # Equal total size does not prove the flat residual maps to the same
        # parameters. Keep feedback only when the full bucket mapping matches.
        power.error_dict.pop(index, None)
    state.bucket_signatures[index] = signature_value
    prior_error = power.error_dict.get(index)
    compensated = bucket.buffer().detach().clone()
    if prior_error is not None:
        compensated.add_(prior_error)

    output_future: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    prior_future = (state.communication_tail if state.communication_step == state.step
                    and index != 0 else None)
    if index == 0 or state.communication_step != state.step:
        state.communication_step = state.step
    state.communication_tail = output_future

    def set_error(error: BaseException) -> None:
        if not output_future.done():
            output_future.set_exception(error)

    def finish(done: torch.futures.Future[torch.Tensor]) -> None:
        try:
            result = done.value()
            error = power.error_dict.get(index)
            if error is None:
                err_sq = torch.zeros((), device=result.device, dtype=torch.float64)
            else:
                err_sq = error.double().square().sum()
            norm_sq = compensated.double().square().sum()
            state.pending_stats[index] = (err_sq, norm_sq)
            if not bucket.is_last():
                if debug: print(f"rank={dist.get_rank()} bucket={index} complete", flush=True)
                output_future.set_result(result)
                return

            next_step = state.step + 1
            if next_step % state.update_every != 0 or state.step < state.start_compression_iter:
                state.pending_stats.clear()
                state.step = next_step
                if debug: print(f"rank={dist.get_rank()} step={state.step} warm complete", flush=True)
                output_future.set_result(result)
                return

            indices = sorted(state.pending_stats)
            if debug: print(f"rank={dist.get_rank()} control step={next_step} buckets={indices}", flush=True)
            combined_signature = 1469598103934665603
            for bucket_id in indices:
                combined_signature = ((combined_signature * 1099511628211) ^
                                      state.bucket_signatures[bucket_id]) & ((1 << 63) - 1)
            signature = torch.tensor([len(indices), combined_signature],
                                     device=result.device, dtype=torch.int64)
            state.controller_payload_bytes += 32 + len(indices) * 16
            state.controller_collectives += 3
            low, high = signature.clone(), signature.clone()
            group = state.process_group or dist.group.WORLD

            def after_signature(done_signature: torch.futures.Future[torch.Tensor]) -> None:
                try:
                    if not torch.equal(low, high):
                        raise RuntimeError("DDP bucket order or shape diverged across ranks")
                    values = torch.stack([component for bucket_id in indices
                                          for component in state.pending_stats[bucket_id]])
                    reduced = dist.all_reduce(values, group=group, async_op=True).get_future()
                    if debug: print(f"rank={dist.get_rank()} control stats reduce", flush=True)

                    def apply_decisions(done_values: torch.futures.Future[torch.Tensor]) -> None:
                        try:
                            global_values = done_values.value()[0]
                            for position, bucket_id in enumerate(indices):
                                error_total, norm_total = global_values[position * 2:position * 2 + 2].tolist()
                                residual = (error_total / max(norm_total, 1e-30)) ** 0.5
                                samples = state.samples_by_bucket.get(bucket_id, 0)
                                old_ema = state.residual_ema_by_bucket.get(bucket_id, 0.0)
                                ema = residual if samples == 0 else (
                                    state.residual_ema_decay * old_ema + (1 - state.residual_ema_decay) * residual
                                )
                                state.samples_by_bucket[bucket_id] = samples + 1
                                state.residual_ema_by_bucket[bucket_id] = ema
                                bucket_power = state.power_by_bucket[bucket_id]
                                old_rank = bucket_power.matrix_approximation_rank
                                pos = state.ranks.index(old_rank)
                                if ema > state.residual_budget and pos + 1 < len(state.ranks):
                                    new_rank = state.ranks[pos + 1]
                                elif ema < state.residual_budget * (1 - state.hysteresis) and pos > 0:
                                    new_rank = state.ranks[pos - 1]
                                else:
                                    new_rank = old_rank
                                if new_rank != old_rank:
                                    bucket_power.p_memory_dict.pop(bucket_id, None)
                                    bucket_power.q_memory_dict.pop(bucket_id, None)
                                    bucket_power.matrix_approximation_rank = new_rank
                                state.decisions.append({"step": next_step, "bucket": bucket_id,
                                                        "rank": new_rank, "residual": ema})
                            state.pending_stats.clear()
                            state.step = next_step
                            if debug: print(f"rank={dist.get_rank()} decisions step={state.step}", flush=True)
                            output_future.set_result(result)
                        except BaseException as error:
                            set_error(error)

                    reduced.then(apply_decisions)
                except BaseException as error:
                    set_error(error)

            def after_low(done_low: torch.futures.Future[torch.Tensor]) -> None:
                try:
                    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group,
                                    async_op=True).get_future().then(after_signature)
                except BaseException as error:
                    set_error(error)

            dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group,
                            async_op=True).get_future().then(after_low)
        except BaseException as error:
            set_error(error)

    def launch_compressor(done_prior: torch.futures.Future[torch.Tensor] | None = None) -> None:
        try:
            if done_prior is not None:
                done_prior.value()
            before = power.total_numel_after_compression
            warmup = power.iter < power.start_powerSGD_iter
            future = powerSGD_hook.powerSGD_hook(power, bucket)
            compressed_elements = power.total_numel_after_compression - before
            if warmup:
                payload_bytes = bucket.buffer().numel() * bucket.buffer().element_size()
                collectives = 1
            else:
                payload_bytes = compressed_elements * bucket.buffer().element_size()
                collectives = 3
            state.communication_payload_bytes += payload_bytes
            state.communication_collectives += collectives
            future.then(finish)
        except BaseException as error:
            set_error(error)

    if prior_future is None:
        launch_compressor()
    else:
        prior_future.then(launch_compressor)
    return output_future
