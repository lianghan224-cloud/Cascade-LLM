"""Explicit opt-in experimental and legacy implementations.

Importing :mod:`layer_streaming` never imports this package.
"""

from .legacy_kv_cache import (
    BlockKVCache,
    DensePagedAttention,
    DensePagedOnlineAttention,
    KVAttentionBackend,
    KVCacheCapacityError,
    KVCacheError,
    KVCacheHandle,
    KVCacheManager,
    KVPageMetadata,
    KVPagePool,
    KVPageState,
    KVTier,
    RequestBlockTable,
    SimpleKVCache,
)

__all__ = [
    "BlockKVCache",
    "DensePagedAttention",
    "DensePagedOnlineAttention",
    "KVAttentionBackend",
    "KVCacheCapacityError",
    "KVCacheError",
    "KVCacheHandle",
    "KVCacheManager",
    "KVPageMetadata",
    "KVPagePool",
    "KVPageState",
    "KVTier",
    "RequestBlockTable",
    "SimpleKVCache",
]
