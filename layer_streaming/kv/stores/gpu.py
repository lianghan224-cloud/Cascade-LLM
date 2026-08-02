"""Preallocated HND GPU page store."""

import torch

from .base import KVStore, KVStoreCapability


class GPUKVStore(KVStore):
    store_id = "gpu"

    def __init__(
        self,
        layer_count,
        page_count,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda:0",
        tensor_factory=None,
    ):
        self.layer_count = int(layer_count)
        self.page_count = int(page_count)
        self.num_kv_heads = int(num_kv_heads)
        self.page_size = int(page_size)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type != "cuda" and tensor_factory is None:
            # CPU allocation is retained for deterministic ABI tests only.
            pass
        factory = tensor_factory or torch.empty
        shape = (
            self.layer_count,
            self.page_count,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
        )
        self.keys = factory(shape, dtype=dtype, device=self.device)
        self.values = factory(shape, dtype=dtype, device=self.device)
        self._closed = False

    def capability(self):
        return KVStoreCapability(
            store_id=self.store_id,
            tiers=("gpu",),
            dtypes=("bf16", "fp16"),
            layouts=("hnd",),
            supports_active_attention=True,
            supports_async_copy=True,
            implemented=True,
        )

    def layer_view(self, layer):
        if self._closed:
            raise RuntimeError("GPU KV store is closed")
        layer = int(layer)
        if layer < 0 or layer >= self.layer_count:
            raise IndexError("layer is outside the KV store")
        return self.keys[layer], self.values[layer]

    def copy_page(self, source_page_id, target_page_id, valid_tokens, stream=None):
        source_page_id = int(source_page_id)
        target_page_id = int(target_page_id)
        valid_tokens = int(valid_tokens)
        if stream is None:
            self.keys[:, target_page_id, :, :valid_tokens, :].copy_(
                self.keys[:, source_page_id, :, :valid_tokens, :]
            )
            self.values[:, target_page_id, :, :valid_tokens, :].copy_(
                self.values[:, source_page_id, :, :valid_tokens, :]
            )
        else:
            with torch.cuda.stream(stream):
                self.keys[:, target_page_id, :, :valid_tokens, :].copy_(
                    self.keys[:, source_page_id, :, :valid_tokens, :],
                    non_blocking=True,
                )
                self.values[:, target_page_id, :, :valid_tokens, :].copy_(
                    self.values[:, source_page_id, :, :valid_tokens, :],
                    non_blocking=True,
                )

    def _page_indices(self, page_ids):
        indices = torch.as_tensor(
            tuple(int(item) for item in page_ids),
            dtype=torch.long,
            device=self.device,
        )
        if indices.ndim != 1:
            raise ValueError("page IDs must be one-dimensional")
        if indices.numel() and (
            int(indices.min().item()) < 0
            or int(indices.max().item()) >= self.page_count
        ):
            raise IndexError("page ID is outside the KV store")
        return indices

    def read_pages(self, layer, page_ids, stream=None):
        key_pool, value_pool = self.layer_view(layer)
        indices = self._page_indices(page_ids)
        if stream is None:
            return (
                torch.index_select(key_pool, 0, indices),
                torch.index_select(value_pool, 0, indices),
            )
        with torch.cuda.stream(stream):
            return (
                torch.index_select(key_pool, 0, indices),
                torch.index_select(value_pool, 0, indices),
            )

    def write_pages(self, layer, page_ids, key, value, stream=None):
        key_pool, value_pool = self.layer_view(layer)
        indices = self._page_indices(page_ids)
        expected = (indices.numel(),) + tuple(key_pool.shape[1:])
        if key.shape != expected or value.shape != expected:
            raise ValueError("page payload shape does not match the KV store")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("page payload dtype does not match the KV store")
        if key.device != self.device or value.device != self.device:
            raise ValueError("page payload device does not match the KV store")
        if stream is None:
            key_pool.index_copy_(0, indices, key)
            value_pool.index_copy_(0, indices, value)
        else:
            with torch.cuda.stream(stream):
                key_pool.index_copy_(0, indices, key)
                value_pool.index_copy_(0, indices, value)

    @property
    def nbytes(self):
        if self._closed:
            return 0
        return (
            self.keys.numel() * self.keys.element_size()
            + self.values.numel() * self.values.element_size()
        )

    def close(self):
        if self._closed:
            return
        self.keys = None
        self.values = None
        self._closed = True
