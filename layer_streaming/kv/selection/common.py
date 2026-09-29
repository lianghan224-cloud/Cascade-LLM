"""Public RGKV contracts shared by selection providers.

The concrete incremental implementation lives in :mod:`rgkv_index`.  This
module is the stable import surface for callers that do not need to know how
the index is stored.  Keeping one set of value types avoids subtly different
budget or epoch semantics between the CPU oracle and the GPU hot path.
"""

from .rgkv_index import (
    RGKVBudget,
    RGKVPageSummary,
    RGKVSelectionResult,
    RGKVStaleIndexError,
    STALE_INDEX,
)


RGKV_NAME = "rgkv"
RGKV_DISPLAY_NAME = "Relevance-Guided KV"

# Old symbols remain importable while callers migrate.  This is metadata, not
# a runtime warning: library imports must not make normal applications noisy.
DEPRECATED_QUEST_APIS = {
    "QuestCPUIndex": "RGKVIndex",
    "QuestFlatSelection": "RGKVSelectionPolicy",
    "TorchTensorizedQuestScorer": "RGKVGPUScorer",
    "quest_scorer_provider": "rgkv_scorer_provider",
}


__all__ = [
    "DEPRECATED_QUEST_APIS",
    "RGKVBudget",
    "RGKV_DISPLAY_NAME",
    "RGKV_NAME",
    "RGKVPageSummary",
    "RGKVSelectionResult",
    "RGKVStaleIndexError",
    "STALE_INDEX",
]
