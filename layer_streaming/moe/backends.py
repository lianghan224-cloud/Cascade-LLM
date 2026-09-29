"""Reference and tensorized Expert compute backends.

The reference backend accepts arbitrary executable linear weights, including
the existing runtime's ephemeral LinearBackend views.  The grouped backend is
an additive dense-weight optimization and never replaces that reference path.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

from .dispatch import ExpertDispatcher, ExpertExecutionPlan


def _linear(hidden_states, weight):
    execute = getattr(weight, "execute", None)
    if callable(execute):
        return execute(hidden_states)
    return F.linear(hidden_states.to(weight.dtype), weight)


@dataclass(frozen=True)
class ExpertWeights:
    gate: object
    up: object
    down: object


@dataclass(frozen=True)
class StackedExpertWeights:
    gate: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor

    def __post_init__(self):
        if self.gate.ndim != 3 or self.up.ndim != 3 or self.down.ndim != 3:
            raise ValueError("stacked Expert weights must be rank-3")
        if self.gate.shape != self.up.shape:
            raise ValueError("stacked gate/up shapes differ")
        experts, intermediate, hidden = self.gate.shape
        if self.down.shape != (experts, hidden, intermediate):
            raise ValueError("stacked down shape is inconsistent")

    @classmethod
    def from_experts(cls, experts: Sequence[ExpertWeights]):
        if not experts:
            raise ValueError("at least one Expert is required")
        tensors = []
        for name in ("gate", "up", "down"):
            values = [getattr(item, name) for item in experts]
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise TypeError("grouped backend requires dense Tensor weights")
            tensors.append(torch.stack(values, dim=0))
        return cls(*tensors)

    def expert(self, expert_id):
        return ExpertWeights(
            self.gate[expert_id], self.up[expert_id], self.down[expert_id]
        )


class NaiveExpertBackend:
    """Correctness backend that executes one selected Expert at a time."""

    name = "naive"

    def __init__(self, activation: Callable = F.silu):
        self.activation = activation

    def execute(
        self,
        dispatched_hidden: torch.Tensor,
        plan: ExpertExecutionPlan,
        expert_weights: Mapping[int, ExpertWeights],
    ):
        output = torch.empty_like(dispatched_hidden)
        # This reference bridge may copy selected IDs to the host.  Streaming
        # production scheduling uses ExpertScheduler and never calls it.
        for bucket in plan.materialize_buckets():
            assignment_ids = torch.nonzero(
                plan.assignment_expert_ids == bucket.expert_id, as_tuple=False
            ).flatten()
            hidden = dispatched_hidden.index_select(0, assignment_ids)
            weights = expert_weights[bucket.expert_id]
            intermediate = self.activation(_linear(hidden, weights.gate))
            intermediate = intermediate * _linear(hidden, weights.up)
            result = _linear(intermediate, weights.down)
            output.index_copy_(0, assignment_ids, result.to(output.dtype))
        return output


class GroupedExpertBackend:
    """Tensorized batched Expert GEMM for dense, stacked weights."""

    name = "grouped"

    def __init__(self, activation: Callable = F.silu):
        self.activation = activation

    @staticmethod
    def _bmm(weight, expert_ids, hidden):
        selected = weight.index_select(0, expert_ids)
        return torch.bmm(selected, hidden.unsqueeze(-1)).squeeze(-1)

    def execute(self, dispatched_hidden, plan, expert_weights):
        if not isinstance(expert_weights, StackedExpertWeights):
            raise TypeError("GroupedExpertBackend requires StackedExpertWeights")
        expert_ids = plan.assignment_expert_ids
        gate = self._bmm(expert_weights.gate, expert_ids, dispatched_hidden)
        up = self._bmm(expert_weights.up, expert_ids, dispatched_hidden)
        intermediate = self.activation(gate) * up
        return self._bmm(expert_weights.down, expert_ids, intermediate)


class AdaptiveExpertBackend:
    """Use the reference fast path for tiny assignment counts."""

    name = "adaptive"

    def __init__(self, activation=F.silu, grouped_min_assignments=16):
        self.naive = NaiveExpertBackend(activation)
        self.grouped = GroupedExpertBackend(activation)
        self.grouped_min_assignments = int(grouped_min_assignments)
        if self.grouped_min_assignments < 1:
            raise ValueError("grouped_min_assignments must be positive")

    def choose(self, plan, expert_weights):
        if (
            isinstance(expert_weights, StackedExpertWeights)
            and plan.assignment_count >= self.grouped_min_assignments
        ):
            return self.grouped
        return self.naive

    def execute(self, dispatched_hidden, plan, expert_weights):
        backend = self.choose(plan, expert_weights)
        if backend is self.naive and isinstance(expert_weights, StackedExpertWeights):
            expert_weights = {
                expert_id: expert_weights.expert(expert_id)
                for expert_id in range(expert_weights.gate.shape[0])
            }
        return backend.execute(dispatched_hidden, plan, expert_weights)


class MoEFFN:
    """Model-neutral resident MoE FFN composition."""

    def __init__(self, router, dispatcher, backend, expert_weights):
        self.router = router
        self.dispatcher = dispatcher
        self.backend = backend
        self.expert_weights = expert_weights

    def __call__(self, hidden_states, return_router_logits=True, force_general=False):
        original_shape = hidden_states.shape
        hidden = hidden_states.reshape(-1, original_shape[-1])
        routing = self.router(hidden, return_logits=return_router_logits)
        plan = self.dispatcher(routing, force_general=force_general)
        dispatched = self.dispatcher.dispatch_hidden(hidden, plan)
        assignment_outputs = self.backend.execute(
            dispatched, plan, self.expert_weights
        )
        output = self.dispatcher.combine(
            assignment_outputs, plan, output_dtype=hidden.dtype
        ).reshape(original_shape)
        return output, routing, plan
