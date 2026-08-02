"""KV Framework V1 transactional paged runtime."""

import math
import threading
import time

import torch

from ..attention.paged import (
    PagedAttentionDispatcher,
    PagedAttentionInput,
    PagedKVAppendInput,
    default_paged_registry,
    detected_architecture,
)
from ..kv_policy import (
    KVAccuracy,
    KVDataType,
    KVPolicy,
    KVReusePolicy,
    KVSelectionPolicy,
    KVStoragePolicy,
)
from .batch_state import build_paged_batch_view
from .block_table import LogicalBlockTable
from .errors import KVCapacityError, KVLifecycleError, KVUnsupportedError
from .metrics import KVMetrics
from .page_pool import KVPagePoolV1
from .request_state import PendingAppend, RequestKVState
from .reuse import (
    InMemoryPrefixIndex,
    PrefixMemoryReuse,
    RequestOnlyReuse,
    SessionReuse,
)
from .selection import DenseSelection
from .slot_mapping import SlotMapping
from .stores import GPUKVStore
from .types import PageState, RequestLifecycleState


_TORCH_DTYPES = {
    KVDataType.BF16: torch.bfloat16,
    KVDataType.FP16: torch.float16,
}


class PagedKVRuntime:
    """Final page/request/batch owner used by model executors.

    The logical block table stores generation-checked PageHandle objects. Only
    `prepare_batch` compacts them into GPU physical page IDs for a provider.
    """

    abi_version = 1

    def __init__(
        self,
        layer_count,
        num_query_heads,
        num_kv_heads,
        head_dim,
        page_count,
        page_size=16,
        dtype=torch.bfloat16,
        device="cuda:0",
        policy=None,
        softmax_scale=None,
        reserved_free_pages=0,
        provider_registry=None,
        allow_reference=False,
        store=None,
    ):
        self.layer_count = int(layer_count)
        self.num_query_heads = int(num_query_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_count = int(page_count)
        self.page_size = int(page_size)
        self.device = torch.device(device)
        self.dtype = dtype
        if self.layer_count <= 0 or self.page_count <= 0:
            raise ValueError("layer_count and page_count must be positive")
        if self.num_query_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.head_dim <= 0 or self.page_size <= 0:
            raise ValueError("head_dim and page_size must be positive")
        inferred_dtype = {
            torch.bfloat16: KVDataType.BF16,
            torch.float16: KVDataType.FP16,
        }.get(dtype)
        if inferred_dtype is None:
            # FP32 is permitted only for CPU reference unit tests.
            if self.device.type != "cpu" or dtype != torch.float32:
                raise ValueError("V1 executable KV supports BF16/FP16")
            inferred_dtype = KVDataType.BF16
        self.policy = policy or KVPolicy(
            dtype=inferred_dtype,
            page_size=self.page_size,
            attention_backend=(
                "reference_paged_exact"
                if self.device.type == "cpu"
                else "generic_cuda"
            ),
        )
        self._validate_policy(inferred_dtype)
        self.softmax_scale = float(
            softmax_scale
            if softmax_scale is not None
            else 1.0 / math.sqrt(float(self.head_dim))
        )
        self.registry = provider_registry or default_paged_registry(load_cuda=True)
        self.dispatcher = PagedAttentionDispatcher(
            self.registry,
            self.policy.attention_backend,
            allow_reference=allow_reference,
        )
        # Resolve static capability before allocating the GPU page arena.
        attention_backend = self.dispatcher.attention_backend
        kv_kernel_backend = self.dispatcher.kv_kernel_backend
        if attention_backend.is_reference and not allow_reference:
            raise ValueError("reference provider requires allow_reference=True")
        capability = attention_backend.capability()
        kernel_capability = kv_kernel_backend.capability()
        architecture = detected_architecture(self.device)
        static_errors = []
        if architecture not in capability.architectures:
            static_errors.append("architecture {}".format(architecture))
        if self.policy.dtype.value not in capability.dtypes:
            static_errors.append("dtype {}".format(self.policy.dtype.value))
        if self.page_size not in capability.page_sizes:
            static_errors.append("page size {}".format(self.page_size))
        if capability.head_dims and self.head_dim not in capability.head_dims:
            static_errors.append("head dim {}".format(self.head_dim))
        if self.num_kv_heads == 1 and self.num_query_heads > 1:
            if not capability.supports_mqa:
                static_errors.append("MQA")
        elif self.num_query_heads == self.num_kv_heads:
            if not capability.supports_mha:
                static_errors.append("MHA")
        elif not capability.supports_gqa:
            static_errors.append("GQA")
        if architecture not in kernel_capability.architectures:
            static_errors.append("KV kernel architecture {}".format(architecture))
        if self.policy.dtype.value not in kernel_capability.dtypes:
            static_errors.append(
                "KV kernel dtype {}".format(self.policy.dtype.value)
            )
        if self.page_size not in kernel_capability.page_sizes:
            static_errors.append("KV kernel page size {}".format(self.page_size))
        if (
            kernel_capability.head_dims
            and self.head_dim not in kernel_capability.head_dims
        ):
            static_errors.append("KV kernel head dim {}".format(self.head_dim))
        if not kernel_capability.supports_append:
            static_errors.append("KV append")
        if not kernel_capability.supports_copy:
            static_errors.append("KV page copy")
        if static_errors:
            raise KVUnsupportedError(
                "provider {} rejected runtime before allocation: {}".format(
                    self.dispatcher.bundle.name,
                    ", ".join(static_errors),
                )
            )
        self.store = store or GPUKVStore(
            layer_count=self.layer_count,
            page_count=self.page_count,
            num_kv_heads=self.num_kv_heads,
            page_size=self.page_size,
            head_dim=self.head_dim,
            dtype=dtype,
            device=self.device,
        )
        self.page_pool = KVPagePoolV1(
            page_count=self.page_count,
            store_id=self.store.store_id,
            dtype=self.policy.dtype.value,
            layout="hnd",
            format_version=1,
            reserved_free_pages=reserved_free_pages,
        )
        self.selection = DenseSelection()
        self.reuse = {
            KVReusePolicy.REQUEST_ONLY: RequestOnlyReuse,
            KVReusePolicy.SESSION: SessionReuse,
            KVReusePolicy.PREFIX_MEMORY: PrefixMemoryReuse,
        }[self.policy.reuse]()
        self.prefix_index = InMemoryPrefixIndex(self.page_size)
        self._prefix_owned = {}
        self._requests = {}
        self._next_request_id = 1
        self._metrics = KVMetrics()
        self._closed = False
        self._lock = threading.RLock()
        # One event per layer bounds async page pins and avoids per-token event
        # allocation. Before reusing an event its previous pages are drained.
        self._attention_events = (
            [torch.cuda.Event(enable_timing=False) for _ in range(self.layer_count)]
            if self.device.type == "cuda"
            else []
        )
        self._attention_pins = [[] for _ in range(self.layer_count)]
        self._append_events = (
            [torch.cuda.Event(enable_timing=False) for _ in range(self.layer_count)]
            if self.device.type == "cuda"
            else []
        )
        self._append_pins = [[] for _ in range(self.layer_count)]

    def _validate_policy(self, inferred_dtype):
        if self.policy.page_size != self.page_size:
            raise ValueError("KV policy page size does not match runtime")
        if self.policy.accuracy != KVAccuracy.EXACT:
            raise KVUnsupportedError("V1 executable path currently supports exact KV only")
        if self.policy.storage != KVStoragePolicy.GPU:
            raise KVUnsupportedError("active CPU/NVMe KV stores are not implemented")
        if self.policy.dtype not in {KVDataType.BF16, KVDataType.FP16}:
            raise KVUnsupportedError("quantized KV formats are not implemented")
        if inferred_dtype != self.policy.dtype and self.dtype != torch.float32:
            raise ValueError("runtime tensor dtype does not match KV policy")
        if self.policy.selection != KVSelectionPolicy.DENSE:
            raise KVUnsupportedError("sparse page selection is not implemented")
        if self.policy.reuse == KVReusePolicy.PREFIX_PERSISTENT:
            raise KVUnsupportedError("persistent prefix reuse is not implemented")

    def _check_open(self):
        if self._closed:
            raise KVLifecycleError("KV runtime is closed")

    @property
    def provider(self):
        """Compatibility alias for the attention-only backend."""
        return self.attention_backend

    @property
    def provider_bundle(self):
        return self.dispatcher.bundle

    @property
    def attention_backend(self):
        return self.dispatcher.attention_backend

    @property
    def kv_kernel_backend(self):
        return self.dispatcher.kv_kernel_backend

    def capacity(self, max_length=None):
        requested_pages = (
            None
            if max_length is None
            else int(math.ceil(int(max_length) / float(self.page_size)))
        )
        return {
            "total_pages": self.page_pool.page_count,
            "free_pages": self.page_pool.free_pages,
            "reserved_free_pages": self.page_pool.reserved_free_pages,
            "requested_pages": requested_pages,
            "admissible": (
                True
                if requested_pages is None
                else self.page_pool.can_allocate(requested_pages)
            ),
            "max_new_tokens": max(
                0,
                (
                    self.page_pool.free_pages
                    - self.page_pool.reserved_free_pages
                )
                * self.page_size,
            ),
        }

    def create_request(
        self,
        max_length,
        request_id=None,
        reuse_namespace="default",
        lifecycle_state=RequestLifecycleState.ACTIVE,
    ):
        with self._lock:
            self._check_open()
            max_length = int(max_length)
            if max_length <= 0:
                raise ValueError("max_length must be positive")
            if int(math.ceil(max_length / float(self.page_size))) > self.page_count:
                raise KVCapacityError("request max_length exceeds total KV capacity")
            if request_id is None:
                request_id = self._next_request_id
                self._next_request_id += 1
            request_id = int(request_id)
            if request_id in self._requests:
                raise ValueError("duplicate request_id {}".format(request_id))
            state = RequestKVState(
                request_id=request_id,
                block_table=LogicalBlockTable(self.page_size, max_length),
                reuse_namespace=str(reuse_namespace),
                lifecycle_state=lifecycle_state,
                layer_lengths=[0] * self.layer_count,
            )
            self._requests[request_id] = state
            return state

    def request(self, request_id):
        self._check_open()
        try:
            return self._requests[int(request_id)]
        except KeyError:
            raise KeyError("unknown KV request {}".format(request_id))

    def _drain_layer_use(self, layer):
        if self.device.type != "cuda":
            return
        layer = int(layer)
        handles = self._attention_pins[layer]
        if not handles:
            return
        self._attention_events[layer].synchronize()
        for handle in reversed(handles):
            self.page_pool.unpin(handle)
        self._attention_pins[layer] = []

    def _drain_layer_append(self, layer):
        if self.device.type != "cuda":
            return
        layer = int(layer)
        handles = self._append_pins[layer]
        if not handles:
            return
        self._append_events[layer].synchronize()
        for handle in reversed(handles):
            self.page_pool.unpin(handle)
        self._append_pins[layer] = []

    def _drain_handles(self, handles):
        identities = {item.identity() for item in handles}
        for layer, pinned in enumerate(self._append_pins):
            if any(item.identity() in identities for item in pinned):
                self._drain_layer_append(layer)
        for layer, pinned in enumerate(self._attention_pins):
            if any(item.identity() in identities for item in pinned):
                self._drain_layer_use(layer)

    @staticmethod
    def _append_target_handles(requests, pending, page_size):
        result = []
        seen = set()
        for state, transaction in zip(requests, pending):
            first_block = transaction.start // int(page_size)
            last_block = (transaction.end - 1) // int(page_size)
            for handle in state.block_table.handles[first_block : last_block + 1]:
                identity = handle.identity()
                if identity not in seen:
                    result.append(handle)
                    seen.add(identity)
        return result

    def _ensure_mutable_tail(self, state):
        if not state.block_table.handles or state.sequence_length % self.page_size == 0:
            return None, None
        logical_tail = state.sequence_length // self.page_size
        source = state.block_table.handles[logical_tail]
        descriptor = self.page_pool.descriptor(source)
        if descriptor.ref_count == 1 and descriptor.state != PageState.SHARED:
            self.page_pool.activate(source, state.sequence_length % self.page_size)
            return None, None
        self._drain_handles((source,))
        target = self.page_pool.allocate(owner_hint=state.request_id)
        valid = state.sequence_length % self.page_size
        self.page_pool.begin_copy(source, target)
        try:
            self.kv_kernel_backend.copy_pages(
                self.store,
                (source.page_id,),
                (target.page_id,),
                (valid,),
            )
        except BaseException:
            # Restore target to a releasable state before rolling back.
            self.page_pool.end_copy(source, target, valid)
            self.page_pool.release(target)
            raise
        self.page_pool.end_copy(source, target, valid)
        state.block_table.replace(logical_tail, target)
        self.page_pool.release(source)
        self._metrics.cow_count += 1
        return source, target

    def _begin_append(self, state, token_count):
        state.ensure_active()
        if state.pending_append is not None:
            if state.pending_append.token_count != int(token_count):
                raise KVLifecycleError("append transaction token count changed")
            return state.pending_append
        token_count = int(token_count)
        if token_count <= 0:
            raise ValueError("append token count must be positive")
        end = state.sequence_length + token_count
        if end > state.block_table.max_length:
            raise KVCapacityError("append exceeds request max_length")
        original_count = len(state.block_table.handles)
        original_lengths = tuple(state.layer_lengths)
        cow_original, cow_replacement = self._ensure_mutable_tail(state)
        required = int(math.ceil(end / float(self.page_size)))
        missing = required - len(state.block_table.handles)
        if not self.page_pool.can_allocate(missing):
            # Restore a COW replacement before rejecting admission.
            if cow_replacement is not None:
                self.page_pool.retain(cow_original)
                state.block_table.replace(original_count - 1, cow_original)
                self.page_pool.release(cow_replacement)
            raise KVCapacityError(
                "append needs {} new pages, only {} are admissible".format(
                    missing,
                    self.page_pool.free_pages - self.page_pool.reserved_free_pages,
                )
            )
        allocated = []
        try:
            while len(state.block_table.handles) < required:
                handle = self.page_pool.allocate(owner_hint=state.request_id)
                state.block_table.append(handle)
                allocated.append(handle)
        except BaseException:
            for handle in reversed(allocated):
                state.block_table.truncate(len(state.block_table.handles) - 1)
                self.page_pool.release(handle)
            if cow_replacement is not None:
                self.page_pool.retain(cow_original)
                state.block_table.replace(original_count - 1, cow_original)
                self.page_pool.release(cow_replacement)
            raise
        pages = []
        offsets = []
        for position in range(state.sequence_length, end):
            logical = position // self.page_size
            pages.append(state.block_table.handles[logical].page_id)
            offsets.append(position % self.page_size)
        pending = PendingAppend(
            start=state.sequence_length,
            token_count=token_count,
            slot_page_ids=torch.tensor(pages, dtype=torch.int32, device=self.device),
            slot_offsets=torch.tensor(offsets, dtype=torch.int32, device=self.device),
            original_block_count=original_count,
            original_layer_lengths=original_lengths,
            allocated_handles=allocated,
            cow_original=cow_original,
            cow_replacement=cow_replacement,
        )
        state.pending_append = pending
        return pending

    def abort_append(self, state):
        with self._lock:
            pending = state.pending_append
            if pending is None:
                return
            for handle in reversed(pending.allocated_handles):
                if state.block_table.handles and state.block_table.handles[-1] == handle:
                    state.block_table.truncate(len(state.block_table.handles) - 1)
                self.page_pool.release(handle)
            if pending.cow_replacement is not None:
                self.page_pool.retain(pending.cow_original)
                state.block_table.replace(
                    pending.original_block_count - 1,
                    pending.cow_original,
                )
                self.page_pool.release(pending.cow_replacement)
            elif pending.original_block_count and state.sequence_length:
                # `_begin_append` makes a private partial tail mutable.  An
                # aborted transaction must restore the sealed committed state.
                tail = state.block_table.handles[pending.original_block_count - 1]
                valid = state.sequence_length % self.page_size or self.page_size
                self.page_pool.seal(tail, valid)
            state.layer_lengths[:] = list(pending.original_layer_lengths)
            state.pending_append = None

    @staticmethod
    def _normalize_kv_tensor(tensor, total_tokens, num_kv_heads, head_dim):
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0).transpose(0, 1)
        if tensor.ndim != 3 or tuple(tensor.shape) != (
            int(total_tokens),
            int(num_kv_heads),
            int(head_dim),
        ):
            raise ValueError(
                "KV tensor must be [total_token, kv_head, head_dim], got {}".format(
                    tuple(tensor.shape)
                )
            )
        return tensor.contiguous()

    def append(self, requests, layer, key, value, query_lengths):
        with self._lock:
            self._check_open()
            requests = tuple(requests)
            query_lengths = tuple(int(item) for item in query_lengths)
            if len(requests) != len(query_lengths) or not requests:
                raise ValueError("append batch metadata is invalid")
            layer = int(layer)
            if layer < 0 or layer >= self.layer_count:
                raise IndexError("layer is outside KV runtime")
            total_tokens = sum(query_lengths)
            key = self._normalize_kv_tensor(
                key, total_tokens, self.num_kv_heads, self.head_dim
            )
            value = self._normalize_kv_tensor(
                value, total_tokens, self.num_kv_heads, self.head_dim
            )
            if key.device != self.device or value.device != self.device:
                raise ValueError("KV append tensors are on the wrong device")
            if key.dtype != self.dtype or value.dtype != self.dtype:
                raise ValueError("KV append tensors have the wrong dtype")
            pending = []
            try:
                for state, token_count in zip(requests, query_lengths):
                    transaction = self._begin_append(state, token_count)
                    if layer in transaction.completed_layers:
                        raise KVLifecycleError("layer was appended twice in one transaction")
                    pending.append(transaction)
                pages = torch.cat([item.slot_page_ids for item in pending])
                offsets = torch.cat([item.slot_offsets for item in pending])
                key_pool, value_pool = self.store.layer_view(layer)
                append_handles = []
                if self.device.type == "cuda":
                    # Reuse a bounded per-layer Event only after its previous
                    # launch has completed. Pin every target until that Event.
                    self._drain_layer_append(layer)
                    append_handles = self._append_target_handles(
                        requests,
                        pending,
                        self.page_size,
                    )
                    for handle in append_handles:
                        self.page_pool.pin(handle)
                try:
                    self.kv_kernel_backend.append_kv(
                        PagedKVAppendInput(
                            key=key,
                            value=value,
                            key_pool_view=key_pool,
                            value_pool_view=value_pool,
                            slot_mapping=SlotMapping(pages, offsets),
                            page_size=self.page_size,
                            num_kv_heads=self.num_kv_heads,
                            head_dim=self.head_dim,
                        )
                    )
                except BaseException:
                    for handle in reversed(append_handles):
                        self.page_pool.unpin(handle)
                    raise
                if self.device.type == "cuda":
                    self._append_events[layer].record(
                        torch.cuda.current_stream(self.device)
                    )
                    self._append_pins[layer] = list(append_handles)
                for state, transaction in zip(requests, pending):
                    transaction.completed_layers.add(layer)
                    state.layer_lengths[layer] = transaction.end
                self._metrics.append_calls += 1
                self._metrics.append_tokens += total_tokens
            except BaseException:
                for state in requests:
                    self.abort_append(state)
                raise
            for state in requests:
                if len(state.pending_append.completed_layers) == self.layer_count:
                    self.commit(state)
            return tuple(
                SlotMapping(item.slot_page_ids, item.slot_offsets)
                for item in pending
            )

    def commit(self, state):
        pending = state.pending_append
        if pending is None:
            return state
        if len(pending.completed_layers) != self.layer_count:
            raise KVLifecycleError("cannot commit an incomplete KV append")
        if any(length != pending.end for length in state.layer_lengths):
            raise KVLifecycleError("layer KV lengths diverged")
        state.sequence_length = pending.end
        self._metrics.committed_tokens += pending.token_count
        state.tail_valid_tokens = state.sequence_length % self.page_size or self.page_size
        for logical, handle in enumerate(state.block_table.handles):
            valid = min(
                self.page_size,
                max(0, state.sequence_length - logical * self.page_size),
            )
            if valid:
                self.page_pool.seal(handle, valid)
        state.pending_append = None
        state.version += 1
        return state

    def prepare_batch(
        self,
        requests,
        query_lengths,
        layer,
        query_positions=None,
        slot_mappings=None,
    ):
        with self._lock:
            self._check_open()
            return build_paged_batch_view(
                requests=requests,
                query_lengths=query_lengths,
                layer=layer,
                device=self.device,
                page_size=self.page_size,
                query_positions=query_positions,
                slot_mappings=slot_mappings,
            )

    def attend(
        self,
        requests,
        layer,
        query,
        query_lengths,
        query_positions=None,
        causal=True,
        return_logsumexp=False,
        phase=None,
    ):
        # Page-table compaction, page pins, launch, and Event recording form a
        # single lifecycle transaction. A concurrent release can enter only
        # after the Event exists, so `_drain_handles` has a safe wait target.
        with self._lock:
            return self._attend_locked(
                requests=requests,
                layer=layer,
                query=query,
                query_lengths=query_lengths,
                query_positions=query_positions,
                causal=causal,
                return_logsumexp=return_logsumexp,
                phase=phase,
            )

    def _attend_locked(
        self,
        requests,
        layer,
        query,
        query_lengths,
        query_positions=None,
        causal=True,
        return_logsumexp=False,
        phase=None,
    ):
        self._check_open()
        requests = tuple(requests)
        query_lengths = tuple(int(item) for item in query_lengths)
        total_tokens = sum(query_lengths)
        if query.ndim == 4 and query.shape[0] == 1 and len(requests) == 1:
            query = query.squeeze(0).transpose(0, 1)
        if query.ndim != 3 or tuple(query.shape) != (
            total_tokens,
            self.num_query_heads,
            self.head_dim,
        ):
            raise ValueError("query must be flattened [token, query_head, head_dim]")
        query = query.contiguous()
        batch = self.prepare_batch(
            requests,
            query_lengths,
            layer,
            query_positions=query_positions,
        )
        selected = self.selection.select(requests, layer, query, batch)
        key_pool, value_pool = self.store.layer_view(layer)
        request = PagedAttentionInput(
            query=query,
            key_pool_view=key_pool,
            value_pool_view=value_pool,
            batch_view=batch,
            page_size=self.page_size,
            num_query_heads=self.num_query_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            softmax_scale=self.softmax_scale,
            causal=bool(causal),
            kv_dtype=self.policy.dtype.value,
            output_dtype=(
                "bf16" if self.dtype == torch.bfloat16
                else "fp16" if self.dtype == torch.float16
                else "fp32"
            ),
            selected_pages=selected,
            workspace=None,
            return_logsumexp=bool(return_logsumexp),
        )
        handles = []
        for state in requests:
            required = int(
                math.ceil(state.layer_lengths[int(layer)] / float(self.page_size))
            )
            handles.extend(state.block_table.handles[:required])
        if self.device.type == "cuda":
            # Attention may run on a caller-selected stream. Explicitly order
            # it after the latest append without synchronizing the host.
            if self._append_pins[int(layer)]:
                torch.cuda.current_stream(self.device).wait_event(
                    self._append_events[int(layer)]
                )
            self._drain_layer_use(layer)
            for handle in handles:
                self.page_pool.pin(handle)
            self._attention_pins[int(layer)] = list(handles)
        started = time.perf_counter()
        try:
            result = self.dispatcher.execute(request, phase=phase)
        except BaseException:
            if self.device.type == "cuda":
                for handle in reversed(handles):
                    self.page_pool.unpin(handle)
                self._attention_pins[int(layer)] = []
            raise
        elapsed = (time.perf_counter() - started) * 1000.0
        self._metrics.attention_calls += 1
        resolved_phase = phase or (
            "decode" if max(query_lengths) == 1 else "prefill"
        )
        if resolved_phase == "decode":
            self._metrics.decode_attention_ms += elapsed
        else:
            self._metrics.prefill_attention_ms += elapsed
        workspace_bytes = int(result.provider_metrics.get("workspace_bytes", 0))
        self._metrics.workspace_peak_bytes = max(
            self._metrics.workspace_peak_bytes, workspace_bytes
        )
        if self.device.type == "cuda":
            self._attention_events[int(layer)].record(torch.cuda.current_stream(self.device))
        return result

    def fork(self, state, request_id=None, max_length=None, branch=True):
        with self._lock:
            state.ensure_active()
            if state.pending_append is not None:
                raise KVLifecycleError("cannot fork during append transaction")
            child = self.create_request(
                max_length=(state.block_table.max_length if max_length is None else max_length),
                request_id=request_id,
                reuse_namespace=state.reuse_namespace,
                lifecycle_state=(
                    RequestLifecycleState.BRANCH
                    if branch
                    else RequestLifecycleState.ACTIVE
                ),
            )
            if child.block_table.max_length < state.sequence_length:
                self._requests.pop(child.request_id, None)
                raise ValueError("fork max_length is shorter than prefix")
            try:
                for handle in state.block_table.handles:
                    descriptor = self.page_pool.descriptor(handle)
                    self.page_pool.seal(handle, descriptor.valid_tokens)
                    self.page_pool.retain(handle)
                    child.block_table.append(handle)
                child.sequence_length = state.sequence_length
                child.tail_valid_tokens = state.tail_valid_tokens
                child.layer_lengths[:] = list(state.layer_lengths)
                child.parent_request_id = state.request_id
                child.fork_position = state.sequence_length
                child.version = state.version
                self._metrics.fork_count += 1
                return child
            except BaseException:
                self.release(child)
                raise

    def session_fork(self, state, request_id=None, max_length=None):
        return self.reuse.fork(
            self,
            state,
            request_id=request_id,
            max_length=max_length,
        )

    def commit_branch(self, parent, branch):
        with self._lock:
            if branch.parent_request_id != parent.request_id:
                raise KVLifecycleError("branch does not belong to parent")
            if branch.pending_append is not None or parent.pending_append is not None:
                raise KVLifecycleError("cannot commit branch during append")
            self._drain_handles(parent.block_table.handles)
            self.page_pool.assert_releasable(parent.block_table.handles, "commit branch")
            for handle in reversed(parent.block_table.handles):
                self.page_pool.release(handle)
            parent.block_table.handles[:] = branch.block_table.handles
            parent.block_table.version += 1
            parent.sequence_length = branch.sequence_length
            parent.tail_valid_tokens = branch.tail_valid_tokens
            parent.layer_lengths[:] = list(branch.layer_lengths)
            parent.version += 1
            branch.block_table.handles[:] = []
            branch.lifecycle_state = RequestLifecycleState.RELEASED
            self._requests.pop(branch.request_id, None)
            return parent

    def discard_branch(self, branch):
        self.release(branch)

    def register_prefix(self, state, token_ids):
        return self.reuse.register_prefix(self, state, token_ids)

    def _register_prefix_impl(self, state, token_ids):
        with self._lock:
            if state.pending_append is not None:
                raise KVLifecycleError("cannot register an uncommitted prefix")
            full_pages = state.sequence_length // self.page_size
            handles = tuple(state.block_table.handles[:full_pages])
            hashes = self.prefix_index.register(
                state.reuse_namespace,
                token_ids,
                handles,
            )
            for handle in handles:
                identity = handle.identity()
                if identity not in self._prefix_owned:
                    self.page_pool.retain(handle)
                    self._prefix_owned[identity] = handle
            state.token_block_hashes[:] = list(hashes)
            return hashes

    def reuse_prefix(
        self,
        token_ids,
        max_length,
        reuse_namespace="default",
        request_id=None,
    ):
        return self.reuse.lookup_prefix(
            self,
            token_ids,
            max_length,
            reuse_namespace=reuse_namespace,
            request_id=request_id,
        )

    def _reuse_prefix_impl(
        self,
        token_ids,
        max_length,
        reuse_namespace="default",
        request_id=None,
    ):
        with self._lock:
            match = self.prefix_index.lookup(reuse_namespace, token_ids)
            if not match.page_handles:
                self._metrics.prefix_misses += 1
                return self.create_request(
                    max_length,
                    request_id=request_id,
                    reuse_namespace=reuse_namespace,
                ), match
            state = self.create_request(
                max_length,
                request_id=request_id,
                reuse_namespace=reuse_namespace,
            )
            try:
                for handle in match.page_handles:
                    self.page_pool.retain(handle)
                    state.block_table.append(handle)
                state.sequence_length = match.matched_tokens
                state.tail_valid_tokens = self.page_size
                state.layer_lengths[:] = [match.matched_tokens] * self.layer_count
                state.token_block_hashes[:] = list(match.block_hashes)
                state.version += 1
            except BaseException:
                self.release(state)
                raise
            self._metrics.prefix_hits += 1
            return state, match

    def reset(self, state):
        with self._lock:
            state.ensure_active()
            if state.pending_append is not None:
                self.abort_append(state)
            self._drain_handles(state.block_table.handles)
            self.page_pool.assert_releasable(state.block_table.handles, "reset request")
            for handle in reversed(state.block_table.handles):
                self.page_pool.release(handle)
            state.block_table.handles[:] = []
            state.block_table.version += 1
            state.sequence_length = 0
            state.tail_valid_tokens = 0
            state.layer_lengths[:] = [0] * self.layer_count
            state.version += 1

    def release(self, state):
        with self._lock:
            if state.lifecycle_state == RequestLifecycleState.RELEASED:
                return
            owned = self._requests.get(state.request_id)
            if owned is not state:
                raise KVLifecycleError("request belongs to another KV runtime")
            if state.pending_append is not None:
                self.abort_append(state)
            self._drain_handles(state.block_table.handles)
            self.page_pool.assert_releasable(state.block_table.handles, "release request")
            state.lifecycle_state = RequestLifecycleState.RELEASING
            for handle in reversed(state.block_table.handles):
                self.page_pool.release(handle)
            state.block_table.handles[:] = []
            state.lifecycle_state = RequestLifecycleState.RELEASED
            self._requests.pop(state.request_id, None)
            self._metrics.release_count += 1

    def profile_stats(self):
        pool = self.page_pool.profile()
        self._metrics.pool_peak_pages = pool["peak_allocated_pages"]
        self._metrics.shared_pages = pool["shared_pages"]
        result = self._metrics.as_dict()
        result.update(
            {
                "abi_version": self.abi_version,
                "kv_policy_resolved": self.policy.as_dict(),
                "attention_backend": self.attention_backend.name,
                "attention_accuracy": self.policy.accuracy.value,
                "layout": "hnd",
                "kv_store": self.store.store_id,
                "kv_dtype": self.policy.dtype.value,
                "kv_selection": self.selection.name,
                "kv_reuse": self.reuse.name,
                "kv_page_size": self.page_size,
                "kv_pool_total_pages": self.page_count,
                "kv_pool_allocated_pages": pool["allocated_pages"],
                "active_requests": len(self._requests),
                "prefix_index_entries": len(self.prefix_index),
                "paged_attention_provider": self.attention_backend.name,
                "paged_kv_kernel_backend": self.kv_kernel_backend.name,
                "paged_provider_bundle": self.provider_bundle.name,
                "provider_fallback_reason": (
                    None
                    if self.dispatcher.last_decision is None
                    else self.dispatcher.last_decision.get("fallback_reason")
                ),
                "provider_decision": self.dispatcher.last_decision,
                "store_bytes": self.store.nbytes,
                "page_state_counts": pool["state_counts"],
                "page_allocations": pool["allocation_count"],
                "page_releases": pool["release_count"],
                "total_ref_count": pool["total_ref_count"],
                "max_ref_count": pool["max_ref_count"],
                "total_pin_count": pool["total_pin_count"],
                "max_pin_count": pool["max_pin_count"],
                "cuda_event_count": (
                    len(self._attention_events) + len(self._append_events)
                ),
            }
        )
        return result

    def quiesce(self):
        """Wait for bounded page kernels and release all transient page pins."""
        with self._lock:
            self._check_open()
            for layer in range(self.layer_count):
                self._drain_layer_append(layer)
                self._drain_layer_use(layer)
            return self.profile_stats()

    def close(self):
        with self._lock:
            if self._closed:
                return
            for layer in range(self.layer_count):
                self._drain_layer_append(layer)
                self._drain_layer_use(layer)
            for state in tuple(self._requests.values()):
                self.release(state)
            prefix_handles = tuple(self._prefix_owned.values())
            self.page_pool.assert_releasable(prefix_handles, "close prefix cache")
            for handle in reversed(prefix_handles):
                self.page_pool.release(handle)
            self._prefix_owned.clear()
            self._attention_events = []
            self._attention_pins = []
            self._append_events = []
            self._append_pins = []
            self.store.close()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class RequestKVCacheV1:
    """Single-request executor adapter over the batch-first runtime ABI."""

    def __init__(self, runtime, state):
        self.runtime = runtime
        self.state = state
        self.manager = runtime

    @property
    def max_length(self):
        return self.state.block_table.max_length

    @property
    def nbytes(self):
        return self.runtime.store.nbytes

    @property
    def policy(self):
        return self.runtime.policy

    def sequence_length(self):
        return self.state.sequence_length

    def append_only(self, layer_index, key, value):
        token_count = int(key.shape[2])
        return self.runtime.append(
            (self.state,),
            layer_index,
            key,
            value,
            (token_count,),
        )[0]

    def attend(self, layer_index, query, kv_groups=1, position_ids=None):
        if int(kv_groups) != self.runtime.num_query_heads // self.runtime.num_kv_heads:
            raise ValueError("executor KV group count does not match runtime")
        token_count = int(query.shape[2])
        positions = None
        if position_ids is not None:
            positions = (position_ids.detach().reshape(-1),)
        result = self.runtime.attend(
            (self.state,),
            layer_index,
            query,
            (token_count,),
            query_positions=positions,
            causal=True,
        )
        return result.output.transpose(0, 1).unsqueeze(0)

    def clear(self):
        self.runtime.reset(self.state)

    def close(self):
        self.runtime.release(self.state)

    def profile_stats(self):
        return self.runtime.profile_stats()
