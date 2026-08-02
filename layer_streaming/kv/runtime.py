"""KV Framework V1 transactional paged runtime."""

import math
import threading

import torch

from ..attention.paged import (
    PagedAttentionDispatcher,
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
from .errors import KVLifecycleError, KVUnsupportedError
from .execution import KVExecutionCoordinator
from .metrics import KVMetrics
from .ownership import OwnershipManager
from .page_pool import KVPagePoolV1
from .prefix_cache import PrefixCache
from .request_table import RequestTable
from .reuse import (
    PrefixMemoryReuse,
    RequestOnlyReuse,
    SessionReuse,
)
from .selection import DenseSelection
from .stores import GPUKVStore
from .types import RequestLifecycleState

class PagedKVRuntime:
    """Final page/request/batch owner used by model executors.

    The logical block table stores generation-checked PageHandle objects. Only
    `prepare_batch` compacts them into GPU physical page IDs for a provider.
    """

    abi_version = 1
    schema_version = 1

    @property
    def provider_abi(self):
        return self.attention_backend.provider_abi

    @property
    def qualification_status(self):
        return self.attention_backend.qualification_status

    def capability(self):
        return {
            "schema_version": self.schema_version,
            "provider_abi": self.provider_abi,
            "qualification_status": self.qualification_status,
            "policy": self.policy.capability(),
            "attention": self.attention_backend.capability().as_dict(),
            "page_kernel": self.kv_kernel_backend.capability().as_dict(),
        }

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
        self._metrics = KVMetrics()
        self._closed = False
        self._lock = threading.RLock()
        self.request_table = RequestTable(
            self.page_size,
            self.page_count,
            self.layer_count,
        )
        self._requests = self.request_table
        self.prefix_cache = PrefixCache(
            self.page_size,
            self.page_pool,
            self.layer_count,
            self._metrics,
        )
        # Compatibility inspection aliases; ownership remains in PrefixCache.
        self.prefix_index = self.prefix_cache.index
        self._prefix_owned = self.prefix_cache.owned_handles
        self.ownership = OwnershipManager(self)
        self.execution = KVExecutionCoordinator(self)

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
            return self.request_table.create(
                max_length=max_length,
                request_id=request_id,
                reuse_namespace=reuse_namespace,
                lifecycle_state=lifecycle_state,
            )

    def request(self, request_id):
        self._check_open()
        return self.request_table.request(request_id)

    def abort_append(self, state):
        with self._lock:
            return self.ownership.abort_append(state)

    def append(self, requests, layer, key, value, query_lengths):
        with self._lock:
            self._check_open()
            return self.execution.append(
                requests, layer, key, value, query_lengths
            )

    def commit(self, state):
        return self.ownership.commit(state)

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
            self._check_open()
            return self.execution.attend(
                requests=requests,
                layer=layer,
                query=query,
                query_lengths=query_lengths,
                query_positions=query_positions,
                causal=causal,
                return_logsumexp=return_logsumexp,
                phase=phase,
            )

    def fork(self, state, request_id=None, max_length=None, branch=True):
        with self._lock:
            return self.ownership.fork(
                state,
                request_id=request_id,
                max_length=max_length,
                branch=branch,
            )

    def session_fork(self, state, request_id=None, max_length=None):
        return self.reuse.fork(
            self,
            state,
            request_id=request_id,
            max_length=max_length,
        )

    def commit_branch(self, parent, branch):
        with self._lock:
            return self.ownership.commit_branch(parent, branch)

    def discard_branch(self, branch):
        self.release(branch)

    def register_prefix(self, state, token_ids):
        return self.reuse.register_prefix(self, state, token_ids)

    def _register_prefix_impl(self, state, token_ids):
        with self._lock:
            return self.prefix_cache.register(state, token_ids)

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
            return self.prefix_cache.reuse(
                self,
                token_ids,
                max_length,
                reuse_namespace=reuse_namespace,
                request_id=request_id,
            )

    def reset(self, state):
        with self._lock:
            return self.ownership.reset(state)

    def release(self, state):
        with self._lock:
            return self.ownership.release(state)

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
                "cuda_event_count": self.ownership.event_count,
            }
        )
        return result

    def quiesce(self):
        """Wait for bounded page kernels and release all transient page pins."""
        with self._lock:
            self._check_open()
            self.ownership.quiesce()
            return self.profile_stats()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.ownership.quiesce()
            for state in tuple(self._requests.values()):
                self.release(state)
            self.prefix_cache.close()
            self.ownership.close()
            self.store.close()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
