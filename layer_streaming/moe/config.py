"""Model-family-neutral MoE configuration sidecar."""

from dataclasses import dataclass
from typing import Optional

from ..specs import normalize_dtype


@dataclass(frozen=True)
class MoEConfig:
    num_experts: int
    experts_per_token: int
    expert_intermediate_size: int
    router_type: str = "linear_topk"
    router_dtype: str = "bfloat16"
    normalize_topk: bool = True
    routed_scaling_factor: float = 1.0
    has_shared_expert: bool = False
    num_shared_experts: int = 0
    expert_weight_layout: str = "gate_up_down"
    expert_groups: int = 1
    experts_per_group: Optional[int] = None
    group_topk: Optional[int] = None

    def __post_init__(self):
        for name in ("num_experts", "experts_per_token", "expert_intermediate_size"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError("{} must be positive".format(name))
            object.__setattr__(self, name, value)
        if self.experts_per_token > self.num_experts:
            raise ValueError("experts_per_token exceeds num_experts")
        router_type = str(self.router_type).strip()
        if router_type not in {"linear_topk", "grouped_topk"}:
            raise ValueError("unsupported router_type {}".format(router_type))
        layout = str(self.expert_weight_layout).strip()
        if layout not in {"gate_up_down", "w1_w3_w2"}:
            raise ValueError("unsupported expert_weight_layout {}".format(layout))
        if float(self.routed_scaling_factor) <= 0:
            raise ValueError("routed_scaling_factor must be positive")
        shared = int(self.num_shared_experts)
        if shared < 0 or bool(self.has_shared_expert) != (shared > 0):
            raise ValueError("shared-expert flag/count mismatch")
        groups = int(self.expert_groups)
        if groups < 1 or self.num_experts % groups:
            raise ValueError("expert_groups must divide num_experts")
        per_group = self.experts_per_group
        if per_group is None:
            per_group = self.num_experts // groups
        if int(per_group) * groups != self.num_experts:
            raise ValueError("experts_per_group does not match num_experts")
        group_topk = self.group_topk
        if router_type == "grouped_topk" and (
            group_topk is None or int(group_topk) < 1 or int(group_topk) > groups
        ):
            raise ValueError("grouped_topk requires a valid group_topk")
        object.__setattr__(self, "router_type", router_type)
        object.__setattr__(self, "router_dtype", normalize_dtype(self.router_dtype))
        object.__setattr__(self, "expert_weight_layout", layout)
        object.__setattr__(self, "routed_scaling_factor", float(self.routed_scaling_factor))
        object.__setattr__(self, "num_shared_experts", shared)
        object.__setattr__(self, "expert_groups", groups)
        object.__setattr__(self, "experts_per_group", int(per_group))
        object.__setattr__(self, "group_topk", None if group_topk is None else int(group_topk))

    def as_dict(self):
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }
