"""Experimental legacy KV cache retained outside the production import graph.

D1 intentionally implements only exact BF16/FP16 GPU-resident KV.  The page
pool and policy contracts are shared by later prefix, tiering, quantization,
and sparse-index milestones, but unsupported modes fail before allocation.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import heapq
import math
import threading
import time
from typing import Dict, List, Optional

import torch

from ..kv_policy import (
    KVDataType,
    KVLayout,
    KVPolicy,
    KVReusePolicy,
)


class KVCacheError(RuntimeError):
    pass


class KVCacheCapacityError(KVCacheError):
    pass


class KVPageState(str, Enum):
    FREE = "free"
    ACTIVE_MUTABLE = "active_mutable"
    SEALED_PRIVATE = "sealed_private"
    SEALED_SHARED = "sealed_shared"
    OFFLOADING = "offloading"
    RESTORING = "restoring"
    CPU_RESIDENT = "cpu_resident"
    NVME_RESIDENT = "nvme_resident"
    EVICTING = "evicting"


class KVTier(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    NVME = "nvme"


@dataclass
class KVPageMetadata:
    page_id: int
    logical_block_id: Optional[int] = None
    token_start: int = 0
    valid_tokens: int = 0
    tier: KVTier = KVTier.GPU
    dtype: str = "bf16"
    state: KVPageState = KVPageState.FREE
    ref_count: int = 0
    pin_count: int = 0
    last_access: float = 0.0
    block_hash: Optional[str] = None
    checksum: Optional[str] = None
    index_version: int = 0
    cuda_event: object = field(default=None, repr=False)

    def as_dict(self):
        return {
            "page_id": self.page_id,
            "logical_block_id": self.logical_block_id,
            "token_start": self.token_start,
            "valid_tokens": self.valid_tokens,
            "tier": self.tier.value,
            "dtype": self.dtype,
            "state": self.state.value,
            "ref_count": self.ref_count,
            "pin_count": self.pin_count,
            "last_access": self.last_access,
            "block_hash": self.block_hash,
            "checksum": self.checksum,
            "index_version": self.index_version,
            "has_cuda_event": self.cuda_event is not None,
        }


class KVPagePool:
    """Own page metadata, deterministic free allocation, and references."""

    def __init__(self, page_count, dtype):
        self.page_count = int(page_count)
        if self.page_count <= 0:
            raise ValueError("page_count must be positive")
        self.dtype = str(dtype)
        self.pages = [
            KVPageMetadata(page_id=index, dtype=self.dtype)
            for index in range(self.page_count)
        ]
        self._free = list(range(self.page_count))
        heapq.heapify(self._free)
        self._lock = threading.RLock()

    @property
    def free_pages(self):
        with self._lock:
            return len(self._free)

    @property
    def allocated_pages(self):
        return self.page_count - self.free_pages

    def metadata(self, page_id):
        page_id = int(page_id)
        if page_id < 0 or page_id >= self.page_count:
            raise IndexError("KV page id is outside the pool")
        return self.pages[page_id]

    def allocate(self, logical_block_id, token_start):
        with self._lock:
            if not self._free:
                raise KVCacheCapacityError("KV page pool is exhausted")
            page_id = heapq.heappop(self._free)
            page = self.pages[page_id]
            if page.state != KVPageState.FREE or page.ref_count:
                raise KVCacheError("free-page metadata is inconsistent")
            page.logical_block_id = int(logical_block_id)
            page.token_start = int(token_start)
            page.valid_tokens = 0
            page.tier = KVTier.GPU
            page.state = KVPageState.ACTIVE_MUTABLE
            page.ref_count = 1
            page.pin_count = 0
            page.last_access = time.monotonic()
            page.block_hash = None
            page.checksum = None
            page.index_version = 0
            page.cuda_event = None
            return page_id

    def retain(self, page_id):
        with self._lock:
            page = self.metadata(page_id)
            if page.state not in {
                KVPageState.SEALED_PRIVATE,
                KVPageState.SEALED_SHARED,
            }:
                raise KVCacheError("only sealed pages can be shared")
            page.ref_count += 1
            page.state = KVPageState.SEALED_SHARED
            page.last_access = time.monotonic()

    def seal(self, page_id, valid_tokens):
        with self._lock:
            page = self.metadata(page_id)
            if page.state not in {
                KVPageState.ACTIVE_MUTABLE,
                KVPageState.SEALED_PRIVATE,
                KVPageState.SEALED_SHARED,
            }:
                raise KVCacheError("cannot seal page in state {}".format(page.state))
            page.valid_tokens = int(valid_tokens)
            page.state = (
                KVPageState.SEALED_SHARED
                if page.ref_count > 1
                else KVPageState.SEALED_PRIVATE
            )
            page.last_access = time.monotonic()

    def mark_active(self, page_id, valid_tokens):
        with self._lock:
            page = self.metadata(page_id)
            if page.ref_count != 1:
                raise KVCacheError("a shared KV page cannot become mutable")
            page.valid_tokens = int(valid_tokens)
            page.state = KVPageState.ACTIVE_MUTABLE
            page.last_access = time.monotonic()

    def pin(self, page_id):
        with self._lock:
            page = self.metadata(page_id)
            if page.state == KVPageState.FREE:
                raise KVCacheError("cannot pin a free KV page")
            page.pin_count += 1
            page.last_access = time.monotonic()

    def unpin(self, page_id):
        with self._lock:
            page = self.metadata(page_id)
            if page.pin_count <= 0:
                raise KVCacheError("KV page pin_count underflow")
            page.pin_count -= 1
            page.last_access = time.monotonic()

    def release(self, page_id):
        with self._lock:
            page = self.metadata(page_id)
            if page.ref_count <= 0:
                raise KVCacheError("KV page ref_count underflow")
            if page.pin_count:
                raise KVCacheError("cannot release a pinned KV page")
            if page.state in {
                KVPageState.OFFLOADING,
                KVPageState.RESTORING,
                KVPageState.EVICTING,
            }:
                raise KVCacheError("cannot release a page during transfer")
            page.ref_count -= 1
            if page.ref_count:
                page.state = (
                    KVPageState.SEALED_SHARED
                    if page.ref_count > 1
                    else KVPageState.SEALED_PRIVATE
                )
                return False
            page.logical_block_id = None
            page.token_start = 0
            page.valid_tokens = 0
            page.tier = KVTier.GPU
            page.state = KVPageState.FREE
            page.last_access = time.monotonic()
            page.block_hash = None
            page.checksum = None
            page.index_version = 0
            page.cuda_event = None
            heapq.heappush(self._free, page.page_id)
            return True

    def state_counts(self):
        result = {state.value: 0 for state in KVPageState}
        with self._lock:
            for page in self.pages:
                result[page.state.value] += 1
        return result


@dataclass
class RequestBlockTable:
    request_id: int
    max_length: int
    batch_size: int
    page_size: int
    physical_page_ids: List[int] = field(default_factory=list)
    length: int = 0

    @property
    def block_ids(self):
        return tuple(self.physical_page_ids)

    @property
    def logical_block_count(self):
        return len(self.physical_page_ids)

    def physical_page(self, logical_block_id):
        logical_block_id = int(logical_block_id)
        if logical_block_id < 0 or logical_block_id >= len(
            self.physical_page_ids
        ):
            raise IndexError("logical KV block is not allocated")
        return self.physical_page_ids[logical_block_id]

    def as_dict(self):
        return {
            "request_id": self.request_id,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "page_size": self.page_size,
            "length": self.length,
            "logical_to_physical": list(self.physical_page_ids),
        }


@dataclass
class KVCacheHandle:
    max_length: int
    batch_size: int
    allocation_id: int
    request_id: int
    block_table: RequestBlockTable = field(repr=False)
    length: int = 0
    _layer_lengths: List[int] = field(default_factory=list, repr=False)
    _next_layer: int = field(default=0, repr=False)
    _active_batch_size: Optional[int] = field(default=None, repr=False)
    _released: bool = field(default=False, repr=False)

    @property
    def block_ids(self):
        return self.block_table.block_ids


class KVAttentionBackend:
    name = "abstract"
    accuracy = "exact"
    materializes_full_kv = False

    def execute(
        self,
        manager,
        handle,
        layer,
        query,
        kv_groups,
        position_ids,
    ):
        raise NotImplementedError


def _validate_attention_request(
    manager,
    handle,
    layer,
    query,
    kv_groups,
    position_ids,
):
    """Validate and normalize inputs shared by exact reference backends."""

    manager._check_handle(handle)
    layer = int(layer)
    if layer < 0 or layer >= manager.layer_count:
        raise IndexError("layer index is outside the KV cache")
    if query.ndim != 4:
        raise ValueError("query must have [batch, attention_head, token, dim]")
    batch, query_heads, query_tokens, head_dim = query.shape
    kv_groups = int(kv_groups)
    active_batch = handle._active_batch_size
    if active_batch is None:
        raise KVCacheError("cannot attend before KV has been appended")
    if batch != active_batch or head_dim != manager.head_dim:
        raise ValueError("query geometry does not match the KV cache")
    if kv_groups <= 0 or query_heads != manager.num_key_value_heads * kv_groups:
        raise ValueError("query heads do not match KV groups")
    if query_tokens <= 0:
        raise ValueError("query token count must be positive")
    if query.device != manager.device:
        raise ValueError("query and KV cache must use the same device")
    if query.dtype != manager.dtype:
        raise ValueError("query and KV cache must use the same dtype")
    attention_length = handle._layer_lengths[layer]
    if attention_length <= 0 or query_tokens > attention_length:
        raise KVCacheError("query token range is outside the populated KV cache")
    if position_ids is None:
        start = attention_length - query_tokens
        position_ids = torch.arange(
            start,
            attention_length,
            dtype=torch.long,
            device=query.device,
        ).unsqueeze(0).expand(batch, -1)
    if tuple(position_ids.shape) not in {
        (1, query_tokens),
        (batch, query_tokens),
    }:
        raise ValueError("position_ids do not match the query")
    if position_ids.device != query.device:
        raise ValueError("position_ids and query must use the same device")
    if position_ids.shape[0] == 1 and batch > 1:
        position_ids = position_ids.expand(batch, -1)
    return (
        layer,
        batch,
        query_heads,
        query_tokens,
        head_dim,
        kv_groups,
        attention_length,
        position_ids,
    )


class DensePagedOnlineAttention(KVAttentionBackend):
    """Numerically stable exact page-wise attention reference.

    This backend reads the request block table directly and never creates a
    full contiguous BF16/FP16 KV copy.  It performs online-softmax reduction
    page by page.  A fused CUDA provider is still required to meet the D1
    short-context performance target.
    """

    name = "dense_paged_online_reference"

    def execute(
        self,
        manager,
        handle,
        layer,
        query,
        kv_groups,
        position_ids,
    ):
        (
            layer,
            batch,
            query_heads,
            query_tokens,
            head_dim,
            kv_groups,
            attention_length,
            position_ids,
        ) = _validate_attention_request(
            manager,
            handle,
            layer,
            query,
            kv_groups,
            position_ids,
        )

        page_ids = tuple(handle.block_ids)
        if len(page_ids) == 1:
            valid = attention_length
            page_id = page_ids[0]
            with manager.pin_pages(page_ids):
                key = manager.keys[
                    layer, page_id, :batch, :, :valid, :
                ].repeat_interleave(kv_groups, dim=1)
                value = manager.values[
                    layer, page_id, :batch, :, :valid, :
                ].repeat_interleave(kv_groups, dim=1)
                # The executor permits multi-token input only when the cache
                # is empty, so equal query/KV lengths identify full prefill
                # without a synchronizing inspection of CUDA position values.
                full_prefill = query_tokens == valid
                if full_prefill or query_tokens == 1:
                    return torch.nn.functional.scaled_dot_product_attention(
                        query,
                        key,
                        value,
                        dropout_p=0.0,
                        is_causal=(full_prefill and query_tokens > 1),
                    )
                key_positions = torch.arange(
                    valid, dtype=torch.long, device=query.device
                )
                mask = key_positions.view(1, 1, 1, valid) <= (
                    position_ids.view(batch, 1, query_tokens, 1)
                )
                return torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                )

        grouped_query = query.reshape(
            batch,
            manager.num_key_value_heads,
            kv_groups,
            query_tokens,
            head_dim,
        ).float()
        running_max = torch.full(
            (
                batch,
                manager.num_key_value_heads,
                kv_groups,
                query_tokens,
            ),
            -float("inf"),
            dtype=torch.float32,
            device=query.device,
        )
        running_sum = torch.zeros_like(running_max)
        running_output = torch.zeros(
            running_max.shape + (head_dim,),
            dtype=torch.float32,
            device=query.device,
        )
        scale = 1.0 / math.sqrt(float(head_dim))

        with manager.pin_pages(page_ids):
            for logical_block, page_id in enumerate(page_ids):
                page_start = logical_block * manager.block_size
                valid = min(
                    manager.block_size,
                    max(0, attention_length - page_start),
                )
                if valid <= 0:
                    break
                key = manager.keys[
                    layer, page_id, :batch, :, :valid, :
                ].float()
                value = manager.values[
                    layer, page_id, :batch, :, :valid, :
                ].float()
                scores = torch.matmul(
                    grouped_query,
                    key.unsqueeze(2).transpose(-1, -2),
                ) * scale
                key_positions = torch.arange(
                    page_start,
                    page_start + valid,
                    dtype=torch.long,
                    device=query.device,
                )
                allowed = key_positions.view(1, 1, 1, 1, valid) <= (
                    position_ids.view(batch, 1, 1, query_tokens, 1)
                )
                scores = scores.masked_fill(~allowed, -float("inf"))
                page_max = scores.amax(dim=-1)
                new_max = torch.maximum(running_max, page_max)
                previous_scale = torch.exp(running_max - new_max)
                page_exp = torch.exp(scores - new_max.unsqueeze(-1))
                running_output = (
                    running_output * previous_scale.unsqueeze(-1)
                    + torch.matmul(page_exp, value.unsqueeze(2))
                )
                running_sum = (
                    running_sum * previous_scale + page_exp.sum(dim=-1)
                )
                running_max = new_max

        if torch.any(running_sum == 0):
            raise KVCacheError("attention query has no visible KV page")
        output = running_output / running_sum.unsqueeze(-1)
        return output.reshape(
            batch, query_heads, query_tokens, head_dim
        ).to(query.dtype)


class DensePagedAttention(KVAttentionBackend):
    """Correctness-first paged-storage SDPA reference.

    A multi-page layer is materialized into a contiguous temporary tensor so
    PyTorch follows its established dense SDPA numerical path.  This keeps the
    page pool and block table authoritative while making the growing copy and
    workspace explicit.  It is not the fused production D1 backend.
    """

    name = "dense_paged_sdpa_reference"
    materializes_full_kv = True

    def __init__(self):
        self._online_single_page = DensePagedOnlineAttention()

    def execute(
        self,
        manager,
        handle,
        layer,
        query,
        kv_groups,
        position_ids,
    ):
        (
            layer,
            _batch,
            _query_heads,
            query_tokens,
            _head_dim,
            kv_groups,
            attention_length,
            position_ids,
        ) = _validate_attention_request(
            manager,
            handle,
            layer,
            query,
            kv_groups,
            position_ids,
        )
        if len(handle.block_ids) <= 1:
            return self._online_single_page.execute(
                manager,
                handle,
                layer,
                query,
                kv_groups,
                position_ids,
            )
        with manager.pin_pages(tuple(handle.block_ids)):
            key, value = manager._materialize_layer(handle, layer)
        key = key.repeat_interleave(kv_groups, dim=1)
        value = value.repeat_interleave(kv_groups, dim=1)
        full_prefill = query_tokens == attention_length
        if full_prefill or query_tokens == 1:
            return torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=0.0,
                is_causal=(full_prefill and query_tokens > 1),
            )
        key_positions = torch.arange(
            attention_length, dtype=torch.long, device=query.device
        )
        mask = key_positions.view(1, 1, 1, attention_length) <= (
            position_ids.view(query.shape[0], 1, query_tokens, 1)
        )
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
        )


class KVCacheManager:
    """Own an HND page pool and per-request logical block tables.

    Physical K/V layout is
    ``[layer, page, batch, kv_head, page_token, head_dim]``.  For each layer
    and batch item this is the block-major HND layout
    ``[page, kv_head, page_token, head_dim]``.
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
        policy=None,
        attention_backend=None,
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
        inferred_dtype = {
            torch.bfloat16: KVDataType.BF16,
            torch.float16: KVDataType.FP16,
        }.get(dtype)
        if inferred_dtype is None:
            # float32 is retained for deterministic CPU unit tests.  It is not
            # a user-visible production KV format.
            if not (self.device.type == "cpu" and dtype == torch.float32):
                raise ValueError("D1 KV supports BF16/FP16 only")
            inferred_dtype = KVDataType.BF16
        self.policy = policy or KVPolicy(
            dtype=inferred_dtype,
            page_size=self.block_size,
            reuse=KVReusePolicy.NONE,
        )
        if self.policy.page_size != self.block_size:
            raise ValueError("KV policy page_size does not match block_size")
        self.policy.require_d1_supported()
        self.layout = KVLayout.HND
        factory = tensor_factory or torch.empty
        shape = (
            self.layer_count,
            self.total_blocks,
            self.max_batch_size,
            self.num_key_value_heads,
            self.block_size,
            self.head_dim,
        )
        self.keys = factory(shape, dtype=dtype, device=self.device)
        self.values = factory(shape, dtype=dtype, device=self.device)
        self.page_pool = KVPagePool(
            self.total_blocks, self.policy.dtype.value
        )
        self.attention_backend = attention_backend or DensePagedAttention()
        self._handles: Dict[int, KVCacheHandle] = {}
        self._next_allocation_id = 1
        self._next_request_id = 1
        self._closed = False
        self._profile = {}
        self.reset_profile()

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.numel() * self.keys.element_size() * 2

    @property
    def free_blocks(self):
        return self.page_pool.free_pages

    @property
    def allocated_blocks(self):
        return self.page_pool.allocated_pages

    def reset_profile(self):
        self._profile = {
            "append_calls": 0,
            "appended_tokens": 0,
            "attention_calls": 0,
            "attention_wall_ms": 0.0,
            "materialize_calls": 0,
            "materialized_bytes": 0,
            "fork_calls": 0,
            "cow_page_copies": 0,
            "page_allocations": 0,
            "page_releases": 0,
        }

    def profile_stats(self):
        result = dict(self._profile)
        result.update(
            {
                "attention_backend": self.attention_backend.name,
                "attention_accuracy": self.attention_backend.accuracy,
                "materializes_full_kv": bool(
                    self.attention_backend.materializes_full_kv
                ),
                "layout": self.layout.value,
                "policy": self.policy.as_dict(),
            }
        )
        return result

    def resource_stats(self):
        state_counts = self.page_pool.state_counts()
        return {
            "closed": self._closed,
            "total_blocks": self.total_blocks,
            "free_blocks": self.free_blocks,
            "allocated_blocks": self.allocated_blocks,
            "active_handles": len(self._handles),
            "kv_cache_bytes": self.nbytes,
            "page_size": self.block_size,
            "layout": self.layout.value,
            "dtype": self.policy.dtype.value,
            "accuracy": self.policy.accuracy.value,
            "storage": self.policy.storage.value,
            "selection": self.policy.selection.value,
            "reuse": self.policy.reuse.value,
            "shared_pages": state_counts[KVPageState.SEALED_SHARED.value],
            "pinned_pages": sum(page.pin_count > 0 for page in self.page_pool.pages),
            "page_state_counts": state_counts,
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

    def allocate(self, max_length, batch_size=1, request_id=None):
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
        if needed > self.total_blocks:
            raise KVCacheCapacityError(
                "request capacity needs {} pages, pool contains {}".format(
                    needed, self.total_blocks
                )
            )
        allocation_id = self._next_allocation_id
        self._next_allocation_id += 1
        if request_id is None:
            request_id = self._next_request_id
            self._next_request_id += 1
        table = RequestBlockTable(
            request_id=int(request_id),
            max_length=max_length,
            batch_size=batch_size,
            page_size=self.block_size,
        )
        handle = KVCacheHandle(
            max_length=max_length,
            batch_size=batch_size,
            allocation_id=allocation_id,
            request_id=int(request_id),
            block_table=table,
            _layer_lengths=[0] * self.layer_count,
        )
        self._handles[allocation_id] = handle
        return handle

    def bind(self, handle):
        self._check_handle(handle)
        return BlockKVCache(self, handle)

    def block_table(self, handle):
        self._check_handle(handle)
        return handle.block_table

    def page_metadata(self, page_id):
        self._check_open()
        return self.page_pool.metadata(page_id)

    def _allocate_page(self, handle, logical_block):
        page_id = self.page_pool.allocate(
            logical_block_id=logical_block,
            token_start=logical_block * self.block_size,
        )
        handle.block_table.physical_page_ids.append(page_id)
        self._profile["page_allocations"] += 1
        return page_id

    def _ensure_pages(self, handle, end):
        required = int(math.ceil(end / float(self.block_size))) if end else 0
        missing = required - len(handle.block_ids)
        if missing <= 0:
            return
        if missing > self.free_blocks:
            raise KVCacheCapacityError(
                "KV append needs {} new pages, only {} are free".format(
                    missing, self.free_blocks
                )
            )
        allocated = []
        try:
            while len(handle.block_ids) < required:
                allocated.append(
                    self._allocate_page(handle, len(handle.block_ids))
                )
        except BaseException:
            for page_id in reversed(allocated):
                handle.block_table.physical_page_ids.pop()
                self.page_pool.release(page_id)
            raise

    def _clone_page(self, source_page_id, logical_block, handle):
        target_page_id = self.page_pool.allocate(
            logical_block_id=logical_block,
            token_start=logical_block * self.block_size,
        )
        try:
            source = self.page_pool.metadata(source_page_id)
            batch = handle._active_batch_size or handle.batch_size
            valid = source.valid_tokens
            self.keys[
                :, target_page_id, :batch, :, :valid, :
            ].copy_(
                self.keys[:, source_page_id, :batch, :, :valid, :]
            )
            self.values[
                :, target_page_id, :batch, :, :valid, :
            ].copy_(
                self.values[:, source_page_id, :batch, :, :valid, :]
            )
            target = self.page_pool.metadata(target_page_id)
            target.valid_tokens = valid
            target.state = KVPageState.ACTIVE_MUTABLE
        except BaseException:
            self.page_pool.release(target_page_id)
            raise
        handle.block_table.physical_page_ids[logical_block] = target_page_id
        self.page_pool.release(source_page_id)
        self._profile["cow_page_copies"] += 1
        self._profile["page_allocations"] += 1
        self._profile["page_releases"] += 1
        return target_page_id

    def _ensure_mutable_tail(self, handle):
        if not handle.block_ids or handle.length % self.block_size == 0:
            return
        logical_block = handle.length // self.block_size
        page_id = handle.block_table.physical_page(logical_block)
        page = self.page_pool.metadata(page_id)
        if page.ref_count > 1 or page.state == KVPageState.SEALED_SHARED:
            self._clone_page(page_id, logical_block, handle)
        else:
            self.page_pool.mark_active(
                page_id, handle.length % self.block_size
            )

    def append_only(self, handle, layer, key, value):
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
        if layer == 0:
            self._ensure_mutable_tail(handle)
            self._ensure_pages(handle, end)

        source_offset = 0
        cursor = start
        while cursor < end:
            logical_block = cursor // self.block_size
            page_offset = cursor % self.block_size
            count = min(end - cursor, self.block_size - page_offset)
            page_id = handle.block_table.physical_page(logical_block)
            key_target = self.keys[
                layer,
                page_id,
                :batch,
                :,
                page_offset : page_offset + count,
                :,
            ]
            value_target = self.values[
                layer,
                page_id,
                :batch,
                :,
                page_offset : page_offset + count,
                :,
            ]
            key_target.copy_(key[:, :, source_offset : source_offset + count, :])
            value_target.copy_(
                value[:, :, source_offset : source_offset + count, :]
            )
            source_offset += count
            cursor += count

        handle._layer_lengths[layer] = end
        handle._next_layer += 1
        if handle._next_layer == self.layer_count:
            if any(length != end for length in handle._layer_lengths):
                raise KVCacheError("KV layer lengths diverged")
            handle.length = end
            handle.block_table.length = end
            handle._next_layer = 0
            for logical_block, page_id in enumerate(handle.block_ids):
                valid = min(
                    self.block_size,
                    max(0, end - logical_block * self.block_size),
                )
                if valid == self.block_size:
                    self.page_pool.seal(page_id, valid)
                elif valid:
                    self.page_pool.mark_active(page_id, valid)
            self._profile["append_calls"] += 1
            self._profile["appended_tokens"] += int(token_count)

    def append(self, handle, layer, key, value):
        self.append_only(handle, layer, key, value)
        return self.get_view(handle, layer)

    def _materialize_layer(self, handle, layer):
        batch = handle._active_batch_size or handle.batch_size
        length = handle._layer_lengths[int(layer)]
        shape = (
            batch,
            self.num_key_value_heads,
            length,
            self.head_dim,
        )
        key = torch.empty(shape, dtype=self.dtype, device=self.device)
        value = torch.empty_like(key)
        for logical_block, page_id in enumerate(handle.block_ids):
            start = logical_block * self.block_size
            valid = min(self.block_size, max(0, length - start))
            if valid <= 0:
                break
            key[:, :, start : start + valid, :].copy_(
                self.keys[int(layer), page_id, :batch, :, :valid, :]
            )
            value[:, :, start : start + valid, :].copy_(
                self.values[int(layer), page_id, :batch, :, :valid, :]
            )
        self._profile["materialize_calls"] += 1
        self._profile["materialized_bytes"] += (
            key.numel() * key.element_size() * 2
        )
        return key, value

    def get_view(self, handle, layer):
        self._check_handle(handle)
        layer = int(layer)
        if layer < 0 or layer >= self.layer_count:
            raise IndexError("layer index is outside the KV cache")
        return self._materialize_layer(handle, layer)

    @contextmanager
    def pin_pages(self, page_ids):
        pinned = []
        try:
            for page_id in page_ids:
                self.page_pool.pin(page_id)
                pinned.append(page_id)
            yield
        finally:
            for page_id in reversed(pinned):
                self.page_pool.unpin(page_id)

    def attend(
        self,
        handle,
        layer,
        query,
        kv_groups=1,
        position_ids=None,
    ):
        self._check_handle(handle)
        started = time.perf_counter()
        result = self.attention_backend.execute(
            self,
            handle,
            int(layer),
            query,
            int(kv_groups),
            position_ids,
        )
        self._profile["attention_calls"] += 1
        self._profile["attention_wall_ms"] += (
            time.perf_counter() - started
        ) * 1000.0
        return result

    def fork(self, handle, max_length=None, request_id=None):
        self._check_handle(handle)
        if handle._next_layer:
            raise KVCacheError("cannot fork during a partial layer append")
        max_length = handle.max_length if max_length is None else int(max_length)
        if max_length < handle.length:
            raise ValueError("fork max_length is shorter than the prefix")
        child = self.allocate(
            max_length=max_length,
            batch_size=handle.batch_size,
            request_id=request_id,
        )
        try:
            full_pages = handle.length // self.block_size
            for logical_block in range(full_pages):
                page_id = handle.block_table.physical_page(logical_block)
                self.page_pool.seal(page_id, self.block_size)
                self.page_pool.retain(page_id)
                child.block_table.physical_page_ids.append(page_id)
            if handle.length % self.block_size:
                logical_block = full_pages
                source_page = handle.block_table.physical_page(logical_block)
                target_page = self.page_pool.allocate(
                    logical_block_id=logical_block,
                    token_start=logical_block * self.block_size,
                )
                child.block_table.physical_page_ids.append(target_page)
                valid = handle.length % self.block_size
                batch = handle._active_batch_size or handle.batch_size
                self.keys[
                    :, target_page, :batch, :, :valid, :
                ].copy_(
                    self.keys[:, source_page, :batch, :, :valid, :]
                )
                self.values[
                    :, target_page, :batch, :, :valid, :
                ].copy_(
                    self.values[:, source_page, :batch, :, :valid, :]
                )
                self.page_pool.mark_active(target_page, valid)
                self._profile["cow_page_copies"] += 1
                self._profile["page_allocations"] += 1
            child.length = handle.length
            child.block_table.length = handle.length
            child._layer_lengths[:] = list(handle._layer_lengths)
            child._active_batch_size = handle._active_batch_size
            self._profile["fork_calls"] += 1
            return child
        except BaseException:
            self.release(child)
            raise

    def _assert_pages_releasable(self, page_ids, operation):
        blocked = []
        for page_id in page_ids:
            page = self.page_pool.metadata(page_id)
            reason = None
            if page.pin_count:
                reason = "pinned"
            elif page.state in {
                KVPageState.OFFLOADING,
                KVPageState.RESTORING,
                KVPageState.EVICTING,
            }:
                reason = page.state.value
            elif page.ref_count <= 0:
                reason = "invalid_ref_count"
            if reason is not None:
                blocked.append({"page_id": int(page_id), "reason": reason})
        if blocked:
            raise KVCacheError(
                "cannot {} request; KV pages are busy: {}".format(
                    operation,
                    blocked,
                )
            )

    def reset(self, handle):
        self._check_handle(handle)
        if handle._next_layer:
            raise KVCacheError("cannot reset during a partial layer append")
        self._assert_pages_releasable(handle.block_ids, "reset")
        for page_id in reversed(handle.block_ids):
            self.page_pool.release(page_id)
            self._profile["page_releases"] += 1
        handle.block_table.physical_page_ids[:] = []
        handle.block_table.length = 0
        handle.length = 0
        handle._layer_lengths[:] = [0] * self.layer_count
        handle._next_layer = 0
        handle._active_batch_size = None

    def release(self, handle):
        if not isinstance(handle, KVCacheHandle) or handle._released:
            return
        self._assert_pages_releasable(handle.block_ids, "release")
        owned = self._handles.pop(handle.allocation_id, None)
        if owned is not handle:
            raise KVCacheError("KV cache handle belongs to another manager")
        for page_id in reversed(handle.block_ids):
            self.page_pool.release(page_id)
            self._profile["page_releases"] += 1
        handle.block_table.physical_page_ids[:] = []
        handle.block_table.length = 0
        handle._released = True

    def close(self):
        if self._closed:
            return
        all_page_ids = tuple(
            page_id
            for handle in self._handles.values()
            for page_id in handle.block_ids
        )
        self._assert_pages_releasable(all_page_ids, "close")
        for handle in tuple(self._handles.values()):
            self.release(handle)
        self._handles.clear()
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
        return self.manager.nbytes

    @property
    def policy(self):
        return self.manager.policy

    def sequence_length(self):
        return self.handle.length

    def append(self, layer_index, key, value):
        return self.manager.append(self.handle, layer_index, key, value)

    def append_only(self, layer_index, key, value):
        return self.manager.append_only(self.handle, layer_index, key, value)

    def attend(
        self,
        layer_index,
        query,
        kv_groups=1,
        position_ids=None,
    ):
        return self.manager.attend(
            self.handle,
            layer_index,
            query,
            kv_groups=kv_groups,
            position_ids=position_ids,
        )

    def get_view(self, layer_index):
        return self.manager.get_view(self.handle, layer_index)

    def block_table(self):
        return self.manager.block_table(self.handle)

    def fork(self, max_length=None, request_id=None):
        return self.manager.bind(
            self.manager.fork(
                self.handle,
                max_length=max_length,
                request_id=request_id,
            )
        )

    def profile_stats(self):
        return self.manager.profile_stats()

    def resource_stats(self):
        return self.manager.resource_stats()

    def clear(self):
        self.manager.reset(self.handle)

    def close(self):
        self.manager.release(self.handle)


class SimpleKVCache:
    """Deprecated compatibility wrapper backed by the paged cache."""

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

    def append_only(self, layer_index, key, value):
        if self._cache is None:
            self._initialize(key)
        return self._cache.append_only(layer_index, key, value)

    def attend(self, layer_index, query, kv_groups=1, position_ids=None):
        if self._cache is None:
            raise KVCacheError("KV cache is empty")
        return self._cache.attend(
            layer_index,
            query,
            kv_groups=kv_groups,
            position_ids=position_ids,
        )

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
