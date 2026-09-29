"""Additive Mixture-of-Experts inference sidecars.

The Dense ExecutionPlan v1 and WeightSpec contracts remain unchanged.  MoE
semantic identity, routing, dispatch, residency and cache state live here.
"""

from .objects import (
    WeightObjectCatalog,
    WeightObjectKey,
    WeightObjectKind,
    WeightObjectRecord,
    WeightObjectSource,
)
from .config import MoEConfig
from .adapters import MoEExecutionSidecar, OlmoeModelAdapter
from .dispatch import ExpertBucket, ExpertDispatcher, ExpertExecutionPlan
from .routing import Router, RoutingResult
from .backends import (
    AdaptiveExpertBackend,
    ExpertWeights,
    GroupedExpertBackend,
    MoEFFN,
    NaiveExpertBackend,
    StackedExpertWeights,
)
from .cache import (
    ExpertCache,
    ExpertCacheEntry,
    FrequencyAwareExpertPolicy,
    LRUExpertPolicy,
    UnifiedResidentWeightBudget,
)
from .memory import MoEMemoryEstimate, MoEMemoryPlanner
from .residency import (
    ExpertKey,
    ExpertResidencyManager,
    ExpertResidencyRecord,
    ExpertResidencyState,
)
from .runtime import (
    ExpertDeviceArena,
    ExpertExecutionSchedule,
    ExpertScheduler,
    ExpertTransferEngine,
    ExpertTransferTicket,
)
from .hf_runtime import (
    MoEStageMetrics,
    SelectedExpertHostBridge,
    StreamingOlmoeSparseMoeBlock,
)
from .policy_benchmark import (
    PolicyTraceResult,
    compare_standard_policies,
    simulate_policy_trace,
    standard_policy_workloads,
)

__all__ = [
    "MoEConfig",
    "MoEExecutionSidecar",
    "OlmoeModelAdapter",
    "ExpertBucket",
    "ExpertDispatcher",
    "ExpertExecutionPlan",
    "Router",
    "RoutingResult",
    "AdaptiveExpertBackend",
    "ExpertWeights",
    "GroupedExpertBackend",
    "MoEFFN",
    "NaiveExpertBackend",
    "StackedExpertWeights",
    "ExpertCache",
    "ExpertCacheEntry",
    "FrequencyAwareExpertPolicy",
    "LRUExpertPolicy",
    "UnifiedResidentWeightBudget",
    "MoEMemoryEstimate",
    "MoEMemoryPlanner",
    "ExpertKey",
    "ExpertResidencyManager",
    "ExpertResidencyRecord",
    "ExpertResidencyState",
    "ExpertDeviceArena",
    "ExpertExecutionSchedule",
    "ExpertScheduler",
    "ExpertTransferEngine",
    "ExpertTransferTicket",
    "MoEStageMetrics",
    "SelectedExpertHostBridge",
    "StreamingOlmoeSparseMoeBlock",
    "PolicyTraceResult",
    "compare_standard_policies",
    "simulate_policy_trace",
    "standard_policy_workloads",
    "WeightObjectCatalog",
    "WeightObjectKey",
    "WeightObjectKind",
    "WeightObjectRecord",
    "WeightObjectSource",
]
