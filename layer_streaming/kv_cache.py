"""Fixed-block, preallocated KV cache for single-request inference."""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple

import torch


class KVCacheError(RuntimeError):
    pass


class KVCacheCapacityError(KVCacheError):
    pass


@dataclass
class KVCacheHandle:
    block_ids: Tuple[int, ...]
    length: int
    max_length: int
    batch_size: int
    allocation_id: int
    _layer_lengths: List[int] = field(repr=False)
    _next_layer: int = field(default=0, repr=False)
    _active_batch_size: Optional[int] = field(default=None, repr=False)
    _released: bool = field(default=False, repr=False)


class KVCacheManager:
    """Own K/V arenas and allocate contiguous runs of fixed-size blocks.

    The physical layout is ``[layer, batch, kv_head, block, token, head_dim]``.
    Contiguous block runs therefore expose an ordinary strided
    ``[batch, kv_head, sequence, head_dim]`` view without gathering or copying
    prior tokens during decode.
    """

    def __init__(
        self,
        layer_count,
        num_key_value_heads,
        head_dim,
        total_blocks,
        block_size=16,
        max_batch_size=1,
        dtype=torch.bfloat16,
        device="cuda:0",
        tensor_factory=None,
    ):
        self.layer_count = int(layer_count)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.total_blocks = int(total_blocks)
        self.block_size = int(block_size)
        self.max_batch_size = int(max_batch_size)
        for name in (
            "layer_count",
            "num_key_value_heads",
            "head_dim",
            "total_blocks",
            "block_size",
            "max_batch_size",
        ):
            if getattr(self, name) <= 0:
                raise ValueError("{} must be positive".format(name))
        self.dtype = dtype
        self.device = torch.device(device)
        factory = tensor_factory or torch.empty
        shape = (
            self.layer_count,
            self.max_batch_size,
            self.num_key_value_heads,
            self.total_blocks,
            self.block_size,
            self.head_dim,
        )
        self.keys = factory(shape, dtype=dtype, device=self.device)
        self.values = factory(shape, dtype=dtype, device=self.device)
        self._free_ranges = [(0, self.total_blocks)]
        self._handles = {}
        self._next_allocation_id = 1
        self._closed = False

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.numel() * self.keys.element_size() * 2

    @property
    def free_blocks(self):
        return sum(count for _, count in self._free_ranges)

    @property
    def allocated_blocks(self):
        return self.total_blocks - self.free_blocks

    def resource_stats(self):
        return {
            "closed": self._closed,
            "total_blocks": self.total_blocks,
            "free_blocks": self.free_blocks,
            "allocated_blocks": self.allocated_blocks,
            "active_handles": len(self._handles),
            "kv_cache_bytes": self.nbytes,
        }

    def _check_open(self):
        if self._closed:
            raise KVCacheError("KV cache manager is closed")

    def _check_handle(self, handle):
        self._check_open()
        if not isinstance(handle, KVCacheHandle):
            raise TypeError("handle must be a KVCacheHandle")
        if handle._released or self._handles.get(handle.allocation_id) is not handle:
            raise KVCacheError("KV cache handle has been released")

    def allocate(self, max_length, batch_size=1):
        self._check_open()
        max_length = int(max_length)
        batch_size = int(batch_size)
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if batch_size <= 0 or batch_size > self.max_batch_size:
            raise ValueError(
                "batch_size must be between 1 and {}".format(
                    self.max_batch_size
                )
            )
        needed = int(math.ceil(max_length / float(self.block_size)))
        for range_index, (start, count) in enumerate(self._free_ranges):
            if count < needed:
                continue
            if count == needed:
                self._free_ranges.pop(range_index)
            else:
                self._free_ranges[range_index] = (start + needed, count - needed)
            allocation_id = self._next_allocation_id
            self._next_allocation_id += 1
            handle = KVCacheHandle(
                block_ids=tuple(range(start, start + needed)),
                length=0,
                max_length=max_length,
                batch_size=batch_size,
                allocation_id=allocation_id,
                _layer_lengths=[0] * self.layer_count,
            )
            self._handles[allocation_id] = handle
            return handle
        raise KVCacheCapacityError(
            "KV cache needs {} contiguous blocks for {} tokens, but only {} "
            "blocks are free".format(needed, max_length, self.free_blocks)
        )

    def bind(self, handle):
        self._check_handle(handle)
        return BlockKVCache(self, handle)

    def _capacity_view(self, arena, handle, layer_index, batch_size):
        start = handle.block_ids[0]
        base = arena[layer_index, :batch_size, :, start, 0, :]
        return torch.as_strided(
            base,
            size=(
                batch_size,
                self.num_key_value_heads,
                len(handle.block_ids) * self.block_size,
                self.head_dim,
            ),
            stride=(
                arena.stride(1),
                arena.stride(2),
                self.head_dim,
                1,
            ),
        )

    def append(self, handle, layer, key, value):
        self._check_handle(handle)
        layer = int(layer)
        if layer < 0 or layer >= self.layer_count:
            raise IndexError("layer index is outside the KV cache")
        if layer != handle._next_layer:
            raise KVCacheError(
                "KV layers must append in order; expected {}, got {}".format(
                    handle._next_layer, layer
                )
            )
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("key and value must share [batch, kv_head, token, dim]")
        batch, kv_heads, token_count, head_dim = key.shape
        if batch > handle.batch_size:
            raise ValueError("KV batch exceeds the handle batch capacity")
        if kv_heads != self.num_key_value_heads or head_dim != self.head_dim:
            raise ValueError("KV tensor geometry does not match the cache")
        if key.device != self.device or value.device != self.device:
            raise ValueError("KV tensors and cache must use the same device")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("KV tensors and cache must use the same dtype")
        if layer == 0:
            handle._active_batch_size = batch
        elif batch != handle._active_batch_size:
            raise ValueError("KV batch size changed between layers")
        start = handle.length
        end = start + int(token_count)
        if end > handle.max_length:
            raise KVCacheCapacityError(
                "KV append would reach {} tokens, exceeding max_length {}".format(
                    end, handle.max_length
                )
            )
        key_target = self._capacity_view(self.keys, handle, layer, batch)
        value_target = self._capacity_view(self.values, handle, layer, batch)
        key_target[:, :, start:end, :].copy_(key)
        value_target[:, :, start:end, :].copy_(value)
        handle._layer_lengths[layer] = end
        handle._next_layer += 1
        if handle._next_layer == self.layer_count:
            if any(length != end for length in handle._layer_lengths):
                raise KVCacheError("KV layer lengths diverged")
            handle.length = end
            handle._next_layer = 0
        return (
            key_target[:, :, :end, :],
            value_target[:, :, :end, :],
        )

    def get_view(self, handle, layer):
        self._check_handle(handle)
        layer = int(layer)
        if layer < 0 or layer >= self.layer_count:
            raise IndexError("layer index is outside the KV cache")
        batch = handle._active_batch_size or handle.batch_size
        length = handle._layer_lengths[layer]
        key = self._capacity_view(self.keys, handle, layer, batch)
        value = self._capacity_view(self.values, handle, layer, batch)
        return key[:, :, :length, :], value[:, :, :length, :]

    def reset(self, handle):
        self._check_handle(handle)
        handle.length = 0
        handle._layer_lengths[:] = [0] * self.layer_count
        handle._next_layer = 0
        handle._active_batch_size = None

    def release(self, handle):
        if not isinstance(handle, KVCacheHandle) or handle._released:
            return
        owned = self._handles.pop(handle.allocation_id, None)
        if owned is not handle:
            raise KVCacheError("KV cache handle belongs to another manager")
        start = handle.block_ids[0]
        count = len(handle.block_ids)
        handle._released = True
        self._free_ranges.append((start, count))
        self._free_ranges.sort()
        merged = []
        for range_start, range_count in self._free_ranges:
            if merged and merged[-1][0] + merged[-1][1] == range_start:
                previous_start, previous_count = merged[-1]
                merged[-1] = (previous_start, previous_count + range_count)
            else:
                merged.append((range_start, range_count))
        self._free_ranges = merged

    def close(self):
        if self._closed:
            return
        for handle in tuple(self._handles.values()):
            handle._released = True
        self._handles.clear()
        self._free_ranges = []
        self.keys = None
        self.values = None
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class BlockKVCache:
    """Request-bound compatibility surface used by the model executor."""

    def __init__(self, manager, handle):
        self.manager = manager
        self.handle = handle

    @property
    def max_length(self):
        return self.handle.max_length

    @property
    def nbytes(self):
        blocks = len(self.handle.block_ids)
        elements = (
            2
            * self.manager.layer_count
            * self.handle.batch_size
            * self.manager.num_key_value_heads
            * blocks
            * self.manager.block_size
            * self.manager.head_dim
        )
        return elements * torch.empty((), dtype=self.manager.dtype).element_size()

    def sequence_length(self):
        return self.handle.length

    def append(self, layer_index, key, value):
        return self.manager.append(self.handle, layer_index, key, value)

    def get_view(self, layer_index):
        return self.manager.get_view(self.handle, layer_index)

    def clear(self):
        self.manager.reset(self.handle)

    def close(self):
        self.manager.release(self.handle)


class SimpleKVCache:
    """Deprecated compatibility wrapper backed by a fixed block cache."""

    def __init__(self, layer_count=32, max_length=2048, block_size=16):
        self.layer_count = int(layer_count)
        self.max_length = int(max_length)
        self.block_size = int(block_size)
        self._cache = None

    def _initialize(self, key):
        blocks = int(math.ceil(self.max_length / float(self.block_size)))
        manager = KVCacheManager(
            layer_count=self.layer_count,
            num_key_value_heads=key.shape[1],
            head_dim=key.shape[-1],
            total_blocks=blocks,
            block_size=self.block_size,
            max_batch_size=key.shape[0],
            dtype=key.dtype,
            device=key.device,
        )
        self._cache = manager.bind(
            manager.allocate(self.max_length, batch_size=key.shape[0])
        )

    def sequence_length(self):
        return 0 if self._cache is None else self._cache.sequence_length()

    def append(self, layer_index, key, value):
        if self._cache is None:
            self._initialize(key)
        return self._cache.append(layer_index, key, value)

    def get_view(self, layer_index):
        if self._cache is None:
            raise KVCacheError("KV cache is empty")
        return self._cache.get_view(layer_index)

    def clear(self):
        if self._cache is not None:
            self._cache.clear()

    def close(self):
        if self._cache is not None:
            manager = self._cache.manager
            self._cache.close()
            manager.close()
            self._cache = None
