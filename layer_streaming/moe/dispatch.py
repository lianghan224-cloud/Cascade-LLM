"""Tensorized token-to-Expert dispatch and weighted combine."""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .routing import RoutingResult


@dataclass(frozen=True)
class ExpertBucket:
    expert_id: int
    token_ids: torch.Tensor
    routing_slots: torch.Tensor
    routing_weights: torch.Tensor


@dataclass(frozen=True)
class ExpertExecutionPlan:
    token_count: int
    top_k: int
    num_experts: int
    selected_experts: torch.Tensor
    expert_offsets: torch.Tensor
    assignment_expert_ids: torch.Tensor
    assignment_token_ids: torch.Tensor
    assignment_routing_slots: torch.Tensor
    assignment_weights: torch.Tensor
    inverse_permutation: torch.Tensor
    small_batch_fast_path: bool

    @property
    def assignment_count(self):
        return int(self.assignment_expert_ids.numel())

    def materialize_buckets(self) -> Tuple[ExpertBucket, ...]:
        """Debug/reference bridge; production tensorized paths need not call it."""
        selected = self.selected_experts.detach().to("cpu").tolist()
        result = []
        offsets = None
        if not self.small_batch_fast_path:
            offsets = self.expert_offsets.detach().to("cpu").tolist()
        for expert_id in selected:
            if self.small_batch_fast_path:
                assignment_ids = torch.nonzero(
                    self.assignment_expert_ids == expert_id, as_tuple=False
                ).flatten()
            else:
                start = offsets[expert_id]
                end = offsets[expert_id + 1]
                assignment_ids = slice(start, end)
            result.append(
                ExpertBucket(
                    expert_id=int(expert_id),
                    token_ids=self.assignment_token_ids[assignment_ids],
                    routing_slots=self.assignment_routing_slots[assignment_ids],
                    routing_weights=self.assignment_weights[assignment_ids],
                )
            )
        return tuple(result)


class ExpertDispatcher:
    def __init__(self, num_experts, small_batch_threshold=4):
        self.num_experts = int(num_experts)
        self.small_batch_threshold = int(small_batch_threshold)
        if self.num_experts < 1 or self.small_batch_threshold < 1:
            raise ValueError("dispatcher sizes must be positive")

    def __call__(self, routing: RoutingResult, force_general=False):
        token_count, top_k = routing.expert_ids.shape
        use_small = token_count <= self.small_batch_threshold and not force_general
        flat_experts = routing.expert_ids.reshape(-1)
        flat_weights = routing.expert_weights.reshape(-1)
        token_ids = torch.arange(
            token_count, dtype=torch.long, device=flat_experts.device
        ).repeat_interleave(top_k)
        routing_slots = torch.arange(
            top_k, dtype=torch.long, device=flat_experts.device
        ).repeat(token_count)
        if use_small:
            # Preserve Router order and avoid the general argsort workspace.
            permutation = torch.arange(
                flat_experts.numel(), dtype=torch.long, device=flat_experts.device
            )
        else:
            permutation = torch.argsort(flat_experts, stable=True)
        assignment_experts = flat_experts[permutation]
        assignment_tokens = token_ids[permutation]
        assignment_slots = routing_slots[permutation]
        assignment_weights = flat_weights[permutation]
        counts = torch.bincount(assignment_experts, minlength=self.num_experts)
        offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=counts.device),
                torch.cumsum(counts, dim=0),
            )
        )
        inverse = torch.empty_like(permutation)
        inverse.scatter_(0, permutation, torch.arange(
            permutation.numel(), dtype=torch.long, device=permutation.device
        ))
        selected = torch.unique(flat_experts, sorted=True)
        return ExpertExecutionPlan(
            token_count=int(token_count),
            top_k=int(top_k),
            num_experts=self.num_experts,
            selected_experts=selected,
            expert_offsets=offsets,
            assignment_expert_ids=assignment_experts,
            assignment_token_ids=assignment_tokens,
            assignment_routing_slots=assignment_slots,
            assignment_weights=assignment_weights,
            inverse_permutation=inverse,
            small_batch_fast_path=use_small,
        )

    @staticmethod
    def dispatch_hidden(hidden_states, plan: ExpertExecutionPlan):
        hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        if hidden.shape[0] != plan.token_count:
            raise ValueError("hidden token count does not match dispatch plan")
        return hidden.index_select(0, plan.assignment_token_ids)

    @staticmethod
    def combine(assignment_outputs, plan: ExpertExecutionPlan, output_dtype=None):
        if assignment_outputs.ndim != 2:
            raise ValueError("assignment_outputs must be [assignments, hidden]")
        if assignment_outputs.shape[0] != plan.assignment_count:
            raise ValueError("assignment output count mismatch")
        weighted = assignment_outputs * plan.assignment_weights[:, None].to(
            assignment_outputs.dtype
        )
        output = torch.zeros(
            (plan.token_count, assignment_outputs.shape[-1]),
            dtype=output_dtype or assignment_outputs.dtype,
            device=assignment_outputs.device,
        )
        output.index_add_(0, plan.assignment_token_ids, weighted.to(output.dtype))
        return output
