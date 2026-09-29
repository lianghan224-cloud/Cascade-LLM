"""OLMoE qualification bridge using Cascade Expert streaming.

This module keeps Hugging Face attention/KV as a reference harness while the
MoE FFN itself uses Cascade's WeightStore, cache, scheduler and CUDA streams.
It is intentionally an adapter boundary, not the generic MoE executor API.
"""

from dataclasses import dataclass, field
import time

import torch
from torch import nn
import torch.nn.functional as F

from .backends import _linear
from .dispatch import ExpertDispatcher
from .routing import Router


class SelectedExpertHostBridge:
    """One targeted Event fence for dynamic cache metadata decisions."""

    def __init__(self, max_assignments, compute_stream):
        self.buffer = torch.empty(
            int(max_assignments), dtype=torch.long, device="cpu", pin_memory=True
        )
        self.compute_stream = compute_stream
        self.ready = torch.cuda.Event(enable_timing=False)
        self.fence_ms = 0.0
        self.copies = 0

    def copy(self, expert_ids):
        flat = expert_ids.reshape(-1)
        if flat.numel() > self.buffer.numel():
            raise RuntimeError("routing batch exceeds the preallocated host bridge")
        started = time.perf_counter()
        with torch.cuda.stream(self.compute_stream):
            self.buffer[: flat.numel()].copy_(flat, non_blocking=True)
            self.ready.record(self.compute_stream)
        # This is an explicit narrow Event/Fence, not a device-wide barrier.
        self.ready.synchronize()
        self.fence_ms += (time.perf_counter() - started) * 1000.0
        self.copies += 1
        return tuple(int(item) for item in self.buffer[: flat.numel()].tolist())


@dataclass
class MoEStageMetrics:
    tokens: int = 0
    steps: int = 0
    selected_experts_per_step: list = field(default_factory=list)
    event_pairs: dict = field(
        default_factory=lambda: {
            name: []
            for name in ("router", "dispatch", "expert_compute", "combine")
        }
    )

    def record_pair(self, name, start, end):
        self.event_pairs[name].append((start, end))

    def finalize(self):
        pairs = [pair for values in self.event_pairs.values() for pair in values]
        if pairs:
            pairs[-1][1].synchronize()
        event_ms = {
            name + "_time_ms": sum(start.elapsed_time(end) for start, end in values)
            for name, values in self.event_pairs.items()
        }
        return {
            "tokens": self.tokens,
            "steps": self.steps,
            "selected_experts_per_step": list(self.selected_experts_per_step),
            **event_ms,
        }


class StreamingOlmoeSparseMoeBlock(nn.Module):
    """Drop-in HF OLMoE block backed by exact Cascade Expert streaming."""

    def __init__(
        self,
        gate,
        moe_config,
        layer_id,
        scheduler,
        max_iteration_tokens=32,
        metrics=None,
    ):
        super().__init__()
        self.gate = gate
        self.moe_config = moe_config
        self.layer_id = int(layer_id)
        self.scheduler = scheduler
        self.transfer_engine = scheduler.transfer_engine
        self.compute_stream = self.transfer_engine.compute_stream
        self.router = Router(moe_config, self.gate.weight)
        self.dispatcher = ExpertDispatcher(moe_config.num_experts)
        self.host_bridge = SelectedExpertHostBridge(
            int(max_iteration_tokens) * moe_config.experts_per_token,
            self.compute_stream,
        )
        self.metrics = metrics or MoEStageMetrics()

    @staticmethod
    def _events():
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def _recorded(self, stage, callback):
        start, end = self._events()
        start.record(self.compute_stream)
        result = callback()
        end.record(self.compute_stream)
        self.metrics.record_pair(stage, start, end)
        return result

    def _expert_weight_names(self, unit):
        names = {}
        for tensor in unit.tensors:
            role = self.transfer_engine.plan.weights[tensor.weight_name].role
            if role in {"mlp_gate", "mlp_up", "mlp_down"}:
                names[role] = tensor.weight_name
        if set(names) != {"mlp_gate", "mlp_up", "mlp_down"}:
            raise RuntimeError("Expert unit does not contain gate/up/down")
        return names

    def forward(self, hidden_states):
        shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, shape[-1])
        caller_stream = torch.cuda.current_stream(hidden.device)
        if caller_stream != self.compute_stream:
            input_ready = torch.cuda.Event(enable_timing=False)
            input_ready.record(caller_stream)
            self.compute_stream.wait_event(input_ready)
        with torch.cuda.stream(self.compute_stream):
            routing = self._recorded("router", lambda: self.router(hidden))
            plan = self._recorded("dispatch", lambda: self.dispatcher(routing))
        selected_ids = self.host_bridge.copy(routing.expert_ids)
        schedule = self.scheduler.build_schedule(self.layer_id, selected_ids)
        dispatched = self.dispatcher.dispatch_hidden(hidden, plan)
        assignment_outputs = torch.empty_like(dispatched)

        def compute(key, views):
            unit = self.scheduler.unit_lookup(key)
            names = self._expert_weight_names(unit)
            assignment_ids = torch.nonzero(
                plan.assignment_expert_ids == key.expert_id, as_tuple=False
            ).flatten()
            current = dispatched.index_select(0, assignment_ids)
            gate = F.silu(_linear(current, views[names["mlp_gate"]]))
            up = _linear(current, views[names["mlp_up"]])
            output = _linear(gate * up, views[names["mlp_down"]])
            assignment_outputs.index_copy_(0, assignment_ids, output)
            return output

        before = len(self.transfer_engine._compute_timing_pairs)
        self.scheduler.execute(schedule, compute)
        for _key, pair in self.transfer_engine._compute_timing_pairs[before:]:
            self.metrics.record_pair("expert_compute", pair[0], pair[1])
        with torch.cuda.stream(self.compute_stream):
            output = self._recorded(
                "combine",
                lambda: self.dispatcher.combine(
                    assignment_outputs, plan, output_dtype=hidden.dtype
                ).reshape(shape),
            )
        self.metrics.tokens += int(hidden.shape[0])
        self.metrics.steps += 1
        self.metrics.selected_experts_per_step.append(len(set(selected_ids)))
        if caller_stream != self.compute_stream:
            output_ready = torch.cuda.Event(enable_timing=False)
            output_ready.record(self.compute_stream)
            caller_stream.wait_event(output_ready)
        return output, routing.router_logits

    def profile_stats(self):
        result = self.metrics.finalize()
        result.update(
            {
                "routing_host_fence_ms": self.host_bridge.fence_ms,
                "routing_host_fence_count": self.host_bridge.copies,
            }
        )
        return result
