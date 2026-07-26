"""CPU-resident, CUDA-streamed LLM inference primitives."""

from .llama31 import (
    LLAMA31_8B_MODEL_ID,
    Llama31DecodeExecutor,
    SimpleKVCache,
)
from .plan import (
    Granularity,
    HostPlacement,
    ModelPlan,
    TensorSpec,
    TransferUnit,
    VocabPlan,
    build_llama31_8b_plan,
)
from .runtime import DoubleBufferRuntime, ResidentDeviceArena
from .vocab import VocabStreamingRuntime, merge_topk
from .weight_store import (
    FullPinnedWeightStore,
    PinnedStagingWeightStore,
    WeightStoreMode,
    create_weight_store,
)

__all__ = [
    "DoubleBufferRuntime",
    "FullPinnedWeightStore",
    "Granularity",
    "HostPlacement",
    "LLAMA31_8B_MODEL_ID",
    "Llama31DecodeExecutor",
    "ModelPlan",
    "PinnedStagingWeightStore",
    "ResidentDeviceArena",
    "SimpleKVCache",
    "TensorSpec",
    "TransferUnit",
    "VocabPlan",
    "VocabStreamingRuntime",
    "WeightStoreMode",
    "build_llama31_8b_plan",
    "create_weight_store",
    "merge_topk",
]
