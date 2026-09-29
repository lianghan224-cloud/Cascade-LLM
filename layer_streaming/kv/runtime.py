"""KV Framework V1 transactional paged runtime."""
import math
import threading
import torch
from ..attention.paged import (
    PagedAttentionDispatcher,
    default_paged_registry,
)
from ..kv_policy import (
    KVDataType,
    KVPolicy,
    KVReusePolicy,
    KVSelectionPolicy,
    KVStoragePolicy,
)
from .batch_state import build_paged_batch_view
from .active_runtime import ActiveTierRuntimeMixin
from .errors import KVLifecycleError, KVUnsupportedError
from .execution import KVExecutionCoordinator
from .metrics import KVMetrics
from .ownership import OwnershipManager
from .page_pool import KVPagePoolV1
from .prefix_cache import PrefixCache
from .request_table import RequestTable
from .runtime_profile import build_runtime_profile
from .runtime_validation import validate_runtime_provider
from .reuse import (
    PrefixMemoryReuse,
    RequestOnlyReuse,
    SessionReuse,
)
from .selection import (
    DenseSelection,
    RGKVBudget,
    RGKVSelectionPolicy,
    rgkv_scorer_provider,
)
from .stores import GPUKVStore
from .types import RequestLifecycleState
class PagedKVRuntime(ActiveTierRuntimeMixin):
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
        prefill_backend="reference_paged_exact",
        max_prefix_pages=None,
        max_prefix_bytes=None,
        prefetch_timeout_seconds=10.0,
        tier_tensor_factory=None,
    ):
        self.layer_count = int(layer_count)
        self.num_query_heads = int(num_query_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_count = int(page_count)
        self.page_size = int(page_size)
        self.device = torch.device(device)
        self.dtype = dtype
        self.prefetch_timeout_seconds = float(prefetch_timeout_seconds)
        if self.prefetch_timeout_seconds <= 0:
            raise ValueError("prefetch_timeout_seconds must be positive")
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
            prefill_provider_name=prefill_backend,
        )
        validate_runtime_provider(self, allow_reference=allow_reference)
        self._initialize_active_tier_state()
        self._logical_page_bytes = self._kv_logical_page_bytes()
        if self.policy.storage == KVStoragePolicy.GPU_CPU:
            if store is not None:
                raise ValueError(
                    "GPU_CPU runtime constructs its bounded hot store from policy"
                )
            self._initialize_active_tier(
                reserved_free_pages, tier_tensor_factory
            )
        else:
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
        self.selection = (
            RGKVSelectionPolicy(
                budget=RGKVBudget(
                    total_page_budget=(
                        self.policy.page_budget or self.page_count
                    ),
                    recent_pages=min(
                        self.policy.page_budget or self.page_count,
                        int(
                            math.ceil(
                                self.policy.recent_window
                                / float(self.page_size)
                            )
                        ),
                    ),
                ),
                mode=("full" if self.policy.page_budget == 0 else "budget"),
                scorer=rgkv_scorer_provider(self.policy.rgkv_scorer),
            )
            if self.policy.selection == KVSelectionPolicy.RGKV
            else DenseSelection()
        )
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
            max_prefix_pages=max_prefix_pages,
            max_prefix_bytes=max_prefix_bytes,
            page_bytes=self._logical_page_bytes,
        )
        # Compatibility inspection aliases; ownership remains in PrefixCache.
        self.prefix_index = self.prefix_cache.index
        self._prefix_owned = self.prefix_cache.owned_handles
        self.ownership = OwnershipManager(self)
        self.execution = KVExecutionCoordinator(self)

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
        return self._runtime_capacity(max_length)
    def create_request(
        self,
        max_length,
        request_id=None,
        reuse_namespace="default",
        lifecycle_state=RequestLifecycleState.ACTIVE,
    ):
        with self._lock:
            self._check_open()
            self._active_tier_admit(max_length)
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
            if self.active_tier_enabled:
                self._tier_abort_append(state)
            return self.ownership.abort_append(state)
    def append(self, requests, layer, key, value, query_lengths):
        with self._lock:
            self._check_open()
            return self.execution.append(
                requests, layer, key, value, query_lengths
            )

    def commit(self, state):
        return self.ownership.commit(state)

    def wait_attention_fence(self, fence_or_id, timeout_seconds=5.0):
        with self._lock:
            return self.ownership.wait_attention_fence(fence_or_id, timeout_seconds)

    def drain_attention_fence(self, fence_or_id, timeout_seconds=5.0):
        with self._lock:
            return self.ownership.drain_attention_fence(fence_or_id, timeout_seconds)

    def rollback(self, state, target_length, target_version=None):
        with self._lock:
            if self.active_tier_enabled:
                raise KVUnsupportedError(
                    "Active Tier rollback is not implemented in the minimal "
                    "request-only runtime"
                )
            return self.ownership.rollback(
                state,
                target_length=target_length,
                target_version=target_version,
            )

    def commit_speculative(self, state, draft_start, accepted_tokens):
        draft_start = int(draft_start)
        accepted_tokens = int(accepted_tokens)
        if accepted_tokens < 0:
            raise ValueError("accepted speculative tokens must not be negative")
        target = draft_start + accepted_tokens
        if target > state.sequence_length:
            raise ValueError("speculative commit exceeds the appended draft")
        return self.rollback(state, target)

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
            if self.active_tier_enabled:
                raise KVUnsupportedError(
                    "Active Tier fork/COW is outside request_only support"
                )
            return self.ownership.fork(
                state,
                request_id=request_id,
                max_length=max_length,
                branch=branch,
            )

    def session_fork(self, state, request_id=None, max_length=None):
        if self.active_tier_enabled:
            raise KVUnsupportedError(
                "Active Tier session fork is outside request_only support"
            )
        return self.reuse.fork(
            self,
            state,
            request_id=request_id,
            max_length=max_length,
        )

    def commit_branch(self, parent, branch):
        with self._lock:
            if self.active_tier_enabled:
                raise KVUnsupportedError(
                    "Active Tier branch commit is outside request_only support"
                )
            return self.ownership.commit_branch(parent, branch)

    def discard_branch(self, branch):
        self.release(branch)

    def register_prefix(self, state, token_ids):
        if self.active_tier_enabled:
            raise KVUnsupportedError(
                "Active Tier prefix reuse is outside request_only support"
            )
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
        if self.active_tier_enabled:
            raise KVUnsupportedError(
                "Active Tier prefix reuse is outside request_only support"
            )
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
            result = self.prefix_cache.reuse(
                self,
                token_ids,
                max_length,
                reuse_namespace=reuse_namespace,
                request_id=request_id,
            )
            state, _ = result
            try:
                if hasattr(self.selection, "reuse_request_from_page_metadata"):
                    self.selection.reuse_request_from_page_metadata(
                        self, state
                    )
                elif hasattr(self.selection, "build_request_layer"):
                    for layer in range(self.layer_count):
                        self.selection.build_request_layer(self, state, layer)
            except BaseException:
                self.release(state)
                raise
            return result
    def reset(self, state):
        with self._lock:
            if self.active_tier_enabled:
                self._tier_unregister_state(state)
            return self.ownership.reset(state)
    def release(self, state):
        with self._lock:
            if self.active_tier_enabled:
                self._tier_unregister_state(state)
            return self.ownership.release(state)

    def profile_stats(self):
        return build_runtime_profile(self)

    def quiesce(self, timeout_seconds=5.0):
        """Wait for bounded page kernels and release all transient page pins."""
        with self._lock:
            self._check_open()
            self.ownership.quiesce(timeout_seconds=timeout_seconds)
            if self.active_tier_enabled:
                self.active_tier.quiesce(timeout=timeout_seconds)
                tier_stats = self.active_tier.stats()
                if tier_stats["pending_tier_operations"]:
                    raise KVLifecycleError(
                        "Active Tier still has pending migration/attention work"
                    )
            self.page_pool.wait_quiescent(timeout_seconds=timeout_seconds)
            self.page_pool.validate_invariants()
            return self.profile_stats()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self.ownership.quiesce()
            for state in tuple(self._requests.values()):
                self.release(state)
            self.prefix_cache.close()
            if self.active_tier_enabled:
                self.active_tier.close()
            self.ownership.close()
            self.store.close()
            if self.cpu_store is not None:
                self.cpu_store.close()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
