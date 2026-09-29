"""Resident Router execution with model-neutral Top-K semantics."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .config import MoEConfig


@dataclass(frozen=True)
class RoutingResult:
    expert_ids: torch.Tensor
    expert_weights: torch.Tensor
    router_logits: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.expert_ids.ndim != 2 or self.expert_weights.ndim != 2:
            raise ValueError("routing ids/weights must be rank-2 [tokens, top_k]")
        if self.expert_ids.shape != self.expert_weights.shape:
            raise ValueError("routing ids/weights shapes differ")
        if self.expert_ids.dtype != torch.long:
            raise ValueError("expert_ids must use torch.long")
        if self.router_logits is not None:
            if self.router_logits.ndim != 2:
                raise ValueError("router_logits must be rank-2")
            if self.router_logits.shape[0] != self.expert_ids.shape[0]:
                raise ValueError("router logits/token count mismatch")

    @property
    def token_count(self):
        return int(self.expert_ids.shape[0])

    @property
    def top_k(self):
        return int(self.expert_ids.shape[1])


class Router:
    """Linear Top-K Router whose weight is expected to be GPU resident."""

    def __init__(self, config: MoEConfig, weight=None):
        self.config = config
        self.weight = weight

    @staticmethod
    def _linear(hidden_states, weight):
        execute = getattr(weight, "execute", None)
        if callable(execute):
            return execute(hidden_states)
        return F.linear(hidden_states.to(weight.dtype), weight)

    def __call__(self, hidden_states, weight=None, return_logits=True):
        if hidden_states.ndim < 2:
            raise ValueError("hidden_states must end in [tokens, hidden]")
        weight = self.weight if weight is None else weight
        if weight is None:
            raise ValueError("Router requires a resident weight")
        hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = self._linear(hidden, weight)
        if logits.shape[-1] != self.config.num_experts:
            raise ValueError("Router output does not match num_experts")
        probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
        routing_weights, expert_ids = torch.topk(
            probabilities, self.config.experts_per_token, dim=-1
        )
        if self.config.normalize_topk:
            routing_weights = routing_weights / routing_weights.sum(
                dim=-1, keepdim=True
            )
        if self.config.routed_scaling_factor != 1.0:
            routing_weights = (
                routing_weights * self.config.routed_scaling_factor
            )
        routing_weights = routing_weights.to(hidden.dtype)
        return RoutingResult(
            expert_ids=expert_ids,
            expert_weights=routing_weights,
            router_logits=logits if return_logits else None,
        )
