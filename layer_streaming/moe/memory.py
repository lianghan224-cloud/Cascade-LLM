"""Additive MemoryPlanner result for MoE cache and dispatch resources."""

from dataclasses import dataclass

from .cache import UnifiedResidentWeightBudget


@dataclass(frozen=True)
class MoEMemoryEstimate:
    dense_estimate: object
    unified_resident_weight_budget_bytes: int
    fixed_resident_weight_bytes: int
    expert_cache_bytes: int
    dispatch_workspace_bytes: int
    estimated_gpu_peak_bytes: int
    gpu_weight_budget_bytes: int

    def as_dict(self):
        return {
            "dense_estimate": self.dense_estimate.as_dict(),
            "unified_resident_weight_budget_bytes": self.unified_resident_weight_budget_bytes,
            "fixed_resident_weight_bytes": self.fixed_resident_weight_bytes,
            "expert_cache_bytes": self.expert_cache_bytes,
            "dispatch_workspace_bytes": self.dispatch_workspace_bytes,
            "estimated_gpu_peak_bytes": self.estimated_gpu_peak_bytes,
            "gpu_weight_budget_bytes": self.gpu_weight_budget_bytes,
        }


class MoEMemoryPlanner:
    """Compose the existing planner without changing its Dense estimate API."""

    def __init__(
        self,
        dense_planner,
        unified_resident_weight_budget_bytes,
        expert_cache_bytes,
        dispatch_workspace_bytes=0,
    ):
        self.dense_planner = dense_planner
        self.total_budget = int(unified_resident_weight_budget_bytes)
        self.expert_cache_bytes = int(expert_cache_bytes)
        self.dispatch_workspace_bytes = int(dispatch_workspace_bytes)
        if min(self.total_budget, self.expert_cache_bytes, self.dispatch_workspace_bytes) < 0:
            raise ValueError("MoE memory sizes must be nonnegative")

    def estimate(self):
        dense = self.dense_planner.estimate()
        fixed = int(dense.resident_parameters_bytes)
        budget = UnifiedResidentWeightBudget(self.total_budget, fixed)
        if self.expert_cache_bytes > budget.expert_capacity_bytes:
            raise MemoryError(
                "Expert cache plus fixed resident weights exceeds "
                "gpu-resident-weight-budget"
            )
        return MoEMemoryEstimate(
            dense_estimate=dense,
            unified_resident_weight_budget_bytes=self.total_budget,
            fixed_resident_weight_bytes=fixed,
            expert_cache_bytes=self.expert_cache_bytes,
            dispatch_workspace_bytes=self.dispatch_workspace_bytes,
            estimated_gpu_peak_bytes=(
                int(dense.estimated_gpu_peak_bytes)
                + self.expert_cache_bytes
                + self.dispatch_workspace_bytes
            ),
            gpu_weight_budget_bytes=(
                int(dense.gpu_weight_budget_bytes) + self.expert_cache_bytes
            ),
        )
