"""CPU-resident, CUDA-streamed LLM inference primitives."""

from .llama31 import (
    LLAMA31_8B_MODEL_ID,
    Llama31DecodeExecutor,
    SimpleKVCache,
)
from .plan import (
    Granularity,
    ModelPlan,
    TensorSpec,
    TransferUnit,
    build_llama31_8b_plan,
)
from .runtime import DoubleBufferRuntime, ResidentDeviceArena
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
    "LLAMA31_8B_MODEL_ID",
    "Llama31DecodeExecutor",
    "ModelPlan",
    "PinnedStagingWeightStore",
    "ResidentDeviceArena",
    "SimpleKVCache",
    "TensorSpec",
    "TransferUnit",
    "WeightStoreMode",
    "build_llama31_8b_plan",
    "create_weight_store",
]
