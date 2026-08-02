"""Pinned CPU store contract; active offload is intentionally unsupported."""

from .base import KVStore, KVStoreCapability


class PinnedCPUKVStore(KVStore):
    store_id = "pinned_cpu"

    def capability(self):
        return KVStoreCapability(
            store_id=self.store_id,
            tiers=("cpu",),
            dtypes=("bf16", "fp16", "int8", "fp8", "int4"),
            layouts=("hnd",),
            supports_active_attention=False,
            supports_async_copy=False,
            implemented=False,
        )
