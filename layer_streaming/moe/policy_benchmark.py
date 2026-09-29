"""Deterministic cache-policy traces used before expensive model runs."""

from dataclasses import dataclass

from .cache import (
    ExpertCache,
    FrequencyAwareExpertPolicy,
    LRUExpertPolicy,
    UnifiedResidentWeightBudget,
)
from .residency import ExpertKey, ExpertResidencyManager


@dataclass(frozen=True)
class PolicyTraceResult:
    policy: str
    workload: str
    accesses: int
    hits: int
    misses: int
    hit_rate: float
    evictions: int
    h2d_bytes: int
    h2d_bytes_per_token: float

    def as_dict(self):
        return dict(self.__dict__)


def standard_policy_workloads():
    return {
        "uniform": tuple(index % 8 for index in range(128)),
        "skewed": tuple(0 if index % 4 else (index // 4) % 8 for index in range(128)),
        "hot_cold": tuple(
            [0, 1] * 8
            + [cold for cold in range(2, 8)]
            + [0, 1] * 8
            + [cold for cold in range(7, 1, -1)]
            + [0, 1] * 8
        ),
        "alternating": tuple(index % 2 for index in range(128)),
    }


def simulate_policy_trace(
    accesses,
    policy,
    workload="custom",
    capacity_experts=2,
    expert_bytes=1,
):
    accesses = tuple(int(item) for item in accesses)
    manager = ExpertResidencyManager()
    keys = {ExpertKey(0, item) for item in accesses}
    for key in keys:
        manager.register(key)
    cache = ExpertCache(
        manager,
        UnifiedResidentWeightBudget(
            int(capacity_experts) * int(expert_bytes), 0
        ),
        policy=policy,
    )
    for expert_id in accesses:
        key = ExpertKey(0, expert_id)
        if cache.lookup(key) is None:
            cache.reserve(key, expert_bytes)
            manager.mark_inflight(key)
            manager.mark_ready(key, location="simulated")
            cache.mark_ready(key, location="simulated")
        cache.acquire(key)
        cache.release(key)
    stats = cache.stats()
    return PolicyTraceResult(
        policy=stats["policy"],
        workload=str(workload),
        accesses=len(accesses),
        hits=stats["hits"],
        misses=stats["misses"],
        hit_rate=stats["hit_rate"],
        evictions=stats["evictions"],
        h2d_bytes=stats["loaded_bytes"],
        h2d_bytes_per_token=(
            0.0 if not accesses else stats["loaded_bytes"] / len(accesses)
        ),
    )


def compare_standard_policies(capacity_experts=2, expert_bytes=1):
    result = []
    for workload, accesses in standard_policy_workloads().items():
        for policy in (LRUExpertPolicy(), FrequencyAwareExpertPolicy()):
            result.append(
                simulate_policy_trace(
                    accesses,
                    policy,
                    workload=workload,
                    capacity_experts=capacity_experts,
                    expert_bytes=expert_bytes,
                )
            )
    return tuple(result)
