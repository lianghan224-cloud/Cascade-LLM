"""CPU-resident, CUDA-streamed LLM inference primitives."""

from .chat import (
    RenderedChat,
    SamplingConfig,
    collect_stop_token_ids,
    first_stop_string,
    render_chat_prompt,
    select_next_token,
)
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
from .int8 import (
    Int8DoubleBufferRuntime,
    Int8FullPinnedWeightStore,
    Int8PinnedStagingWeightStore,
    Int8ResidentDeviceArena,
    Int8WeightStoreMode,
    build_llama31_70b_int8_plan,
    create_int8_weight_store,
)
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
    "Int8DoubleBufferRuntime",
    "Int8FullPinnedWeightStore",
    "Int8PinnedStagingWeightStore",
    "Int8ResidentDeviceArena",
    "Int8WeightStoreMode",
    "LLAMA31_8B_MODEL_ID",
    "Llama31DecodeExecutor",
    "ModelPlan",
    "PinnedStagingWeightStore",
    "RenderedChat",
    "SamplingConfig",
    "ResidentDeviceArena",
    "SimpleKVCache",
    "TensorSpec",
    "TransferUnit",
    "VocabPlan",
    "VocabStreamingRuntime",
    "WeightStoreMode",
    "build_llama31_8b_plan",
    "build_llama31_70b_int8_plan",
    "collect_stop_token_ids",
    "create_int8_weight_store",
    "create_weight_store",
    "first_stop_string",
    "merge_topk",
    "render_chat_prompt",
    "select_next_token",
]
