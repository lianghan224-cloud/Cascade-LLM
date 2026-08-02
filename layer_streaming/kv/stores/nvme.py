"""NVMe store contract; no fake real-time decode implementation."""

from .base import KVStore, KVStoreCapability


class NVMeKVStore(KVStore):
    store_id = "nvme"

    def capability(self):
        return KVStoreCapability(
            store_id=self.store_id,
            tiers=("nvme",),
            dtypes=("bf16", "fp16", "int8", "fp8", "int4"),
            layouts=("hnd",),
            supports_active_attention=False,
            supports_async_copy=False,
            implemented=False,
        )
