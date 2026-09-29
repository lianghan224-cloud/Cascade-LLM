"""Active-Tier composition kept outside the thin PagedKVRuntime shell."""

import math

import torch

from ..attention.paged.tiered_streaming import TieredStreamingExactAttention
from ..kv_policy import (
    KVAccuracy,
    KVDataType,
    KVReusePolicy,
    KVSelectionPolicy,
    KVStoragePolicy,
)
from .errors import KVCapacityError, KVLifecycleError, KVUnsupportedError
from .stores import (
    ActiveTierCoordinator,
    GPUHotKVCache,
    PinnedCPUKVStore,
    ResidencyState,
)


class ActiveTierRuntimeMixin:
    """Location/Execution helpers; this mixin never manages Request refs."""

    def _initialize_active_tier_state(self):
        self.active_tier = None
        self.cpu_store = None
        self._tier_keys = {}
        self._tier_pending_appends = {}
        self._tier_streaming_attention = None

    def _validate_policy(self, inferred_dtype):
        if self.policy.page_size != self.page_size:
            raise ValueError("KV policy page size does not match runtime")
        if self.policy.accuracy not in {KVAccuracy.EXACT, KVAccuracy.SPARSE}:
            raise KVUnsupportedError(
                "executable KV supports exact or RGKV sparse accuracy"
            )
        if self.policy.storage not in {
            KVStoragePolicy.GPU,
            KVStoragePolicy.GPU_CPU,
        }:
            raise KVUnsupportedError("active NVMe KV stores are not implemented")
        if self.policy.dtype not in {KVDataType.BF16, KVDataType.FP16}:
            raise KVUnsupportedError("quantized KV formats are not implemented")
        if inferred_dtype != self.policy.dtype and self.dtype != torch.float32:
            raise ValueError("runtime tensor dtype does not match KV policy")
        if self.policy.selection not in {
            KVSelectionPolicy.DENSE,
            KVSelectionPolicy.RGKV,
        }:
            raise KVUnsupportedError(
                "only dense and RGKV page selection are implemented"
            )
        if self.policy.reuse == KVReusePolicy.PREFIX_PERSISTENT:
            raise KVUnsupportedError("persistent prefix reuse is not implemented")
        if self.policy.storage == KVStoragePolicy.GPU_CPU:
            if self.policy.accuracy not in {
                KVAccuracy.EXACT,
                KVAccuracy.SPARSE,
            }:
                raise KVUnsupportedError(
                    "gpu_cpu Active Tier requires exact or RGKV sparse KV"
                )
            if self.policy.selection not in {
                KVSelectionPolicy.DENSE,
                KVSelectionPolicy.RGKV,
            }:
                raise KVUnsupportedError(
                    "gpu_cpu Active Tier requires dense or RGKV selection"
                )
            if self.policy.reuse != KVReusePolicy.REQUEST_ONLY:
                raise KVUnsupportedError(
                    "gpu_cpu Active Tier currently requires request_only reuse"
                )
            if self.policy.gpu_hot_budget_bytes <= 0:
                raise KVUnsupportedError(
                    "gpu_cpu Active Tier requires a positive GPU hot budget"
                )
            if self.policy.cpu_budget_bytes <= 0:
                raise KVUnsupportedError(
                    "gpu_cpu Active Tier requires a positive pinned CPU budget"
                )

    def _kv_logical_page_bytes(self):
        return (
            2
            * self.layer_count
            * self.num_kv_heads
            * self.page_size
            * self.head_dim
            * torch.empty((), dtype=self.dtype).element_size()
        )

    @staticmethod
    def _tier_watermark_pages(value, page_bytes, capacity_pages, round_up):
        value = int(value)
        if value == 0:
            return None
        pages = (
            int(math.ceil(value / float(page_bytes)))
            if round_up
            else value // int(page_bytes)
        )
        return min(int(capacity_pages), max(0, pages))

    def _initialize_active_tier(self, reserved_free_pages, tensor_factory):
        from .page_pool import KVPagePoolV1

        self.page_pool = KVPagePoolV1(
            page_count=self.page_count,
            # PageHandle is logical ownership identity, never a hot slot.  The
            # frozen block-table ABI still labels executable pages as "gpu".
            store_id="gpu",
            dtype=self.policy.dtype.value,
            layout="hnd",
            format_version=1,
            reserved_free_pages=reserved_free_pages,
        )
        hot_pages = self.policy.gpu_hot_budget_bytes // self._logical_page_bytes
        if hot_pages <= 0:
            raise KVUnsupportedError(
                "gpu_cpu KV requires a complete cross-layer GPU hot page"
            )
        hot_pages = min(hot_pages, self.page_count)
        high_pages = self._tier_watermark_pages(
            self.policy.gpu_high_watermark_bytes,
            self._logical_page_bytes,
            hot_pages,
            True,
        )
        low_pages = self._tier_watermark_pages(
            self.policy.gpu_low_watermark_bytes,
            self._logical_page_bytes,
            hot_pages,
            False,
        )
        self.store = GPUHotKVCache(
            page_pool=self.page_pool,
            layer_count=self.layer_count,
            gpu_capacity_pages=hot_pages,
            num_kv_heads=self.num_kv_heads,
            page_size=self.page_size,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
            high_watermark_pages=high_pages,
            low_watermark_pages=low_pages,
            tensor_factory=tensor_factory,
        )
        self.cpu_store = PinnedCPUKVStore(
            capacity_bytes=self.policy.cpu_budget_bytes,
            num_kv_heads=self.num_kv_heads,
            page_size=self.page_size,
            head_dim=self.head_dim,
            dtype=self.dtype,
            high_watermark_bytes=self.policy.cpu_high_watermark_bytes or None,
            low_watermark_bytes=self.policy.cpu_low_watermark_bytes or None,
            tensor_factory=tensor_factory,
        )
        self.active_tier = ActiveTierCoordinator(
            self.store, self.cpu_store, self.page_pool
        )
        self._tier_streaming_attention = TieredStreamingExactAttention()

    @property
    def active_tier_enabled(self):
        return self.active_tier is not None

    def _tier_capacity_pages(self):
        if not self.active_tier_enabled:
            return None
        return (
            self.store.gpu_capacity_pages
            + self.cpu_store.slot_count // self.layer_count
        )

    def _runtime_capacity(self, max_length=None):
        requested = (
            None
            if max_length is None
            else int(math.ceil(int(max_length) / float(self.page_size)))
        )
        tier_capacity = self._tier_capacity_pages()
        return {
            "total_pages": self.page_pool.page_count,
            "free_pages": self.page_pool.free_pages,
            "reserved_free_pages": self.page_pool.reserved_free_pages,
            "requested_pages": requested,
            "admissible": (
                True
                if requested is None
                else self.page_pool.can_allocate(requested)
                and (tier_capacity is None or requested <= tier_capacity)
            ),
            "tier_capacity_pages": tier_capacity,
            "max_new_tokens": max(
                0,
                (self.page_pool.free_pages - self.page_pool.reserved_free_pages)
                * self.page_size,
            ),
        }

    def _active_tier_admit(self, max_length):
        if not self.active_tier_enabled:
            return
        requested = int(math.ceil(int(max_length) / float(self.page_size)))
        capacity = self._tier_capacity_pages()
        if requested > capacity:
            raise KVCapacityError(
                "request needs {} logical KV pages, above GPU+CPU Active Tier "
                "capacity {}".format(requested, capacity)
            )

    def _reject_active_tier(self, operation):
        if self.active_tier_enabled:
            raise KVUnsupportedError(
                "Active Tier {} is outside minimal request_only support".format(
                    operation
                )
            )

    @staticmethod
    def _tier_handle_identity(handle):
        return handle.identity()

    @staticmethod
    def _tier_logical_id(state, logical, handle):
        return "request:{}:logical:{}:pool:{}:page:{}:generation:{}".format(
            state.request_id,
            int(logical),
            handle.pool_uuid,
            handle.page_id,
            handle.generation,
        )

    def _tier_registered_key(self, handle):
        return self._tier_keys.get(self._tier_handle_identity(handle))

    def _tier_gpu_slot(self, key):
        handles = {item.gpu_slot for item in self.store.location_sets(key)}
        if len(handles) != 1 or None in handles:
            raise KVLifecycleError(
                "Active Tier page has no single cross-layer GPU slot"
            )
        return next(iter(handles))

    def _tier_begin_append(self, state, transaction):
        existing = self._tier_pending_appends.get(state.request_id)
        if existing is not None:
            return existing
        first = transaction.start // self.page_size
        last = (transaction.end - 1) // self.page_size
        logicals = tuple(range(first, last + 1))
        reserve = 1 if transaction.start > 0 else 0
        capacity = self.store.gpu_capacity_pages - reserve
        if len(logicals) > capacity:
            raise KVCapacityError(
                "Active Tier append touches {} pages, above {} writable hot "
                "pages after reserving {} streaming page; chunk Prefill".format(
                    len(logicals), capacity, reserve
                )
            )
        entries = []
        newly_registered = []
        try:
            for logical in logicals:
                handle = state.block_table.handles[logical]
                key = self._tier_registered_key(handle)
                if key is None:
                    descriptor = self.page_pool.descriptor(handle)
                    valid = min(
                        self.page_size,
                        max(0, transaction.end - logical * self.page_size),
                    )
                    key = self.active_tier.register_page(
                        self._tier_logical_id(state, logical, handle),
                        handle,
                        valid_tokens=valid,
                        data_epoch=descriptor.data_version,
                    )
                    self._tier_keys[self._tier_handle_identity(handle)] = key
                    newly_registered.append((handle, key))
                entries.append(
                    {
                        "logical": logical,
                        "handle": handle,
                        "key": key,
                        "operation": "append:{}:{}:{}".format(
                            state.request_id, transaction.start, logical
                        ),
                    }
                )
            all_keys = tuple(item["key"] for item in entries)
            for item in entries:
                slot, operation = self.active_tier.prepare_append(
                    item["key"],
                    request_id=state.request_id,
                    timeout=self.prefetch_timeout_seconds,
                    exclude_keys=all_keys,
                    operation=item["operation"],
                )
                item["slot"] = slot
                item["operation"] = operation
            self._tier_pending_appends[state.request_id] = entries
            return entries
        except BaseException:
            for item in reversed(entries):
                if "slot" not in item:
                    continue
                try:
                    self.active_tier.abort_append(
                        item["key"], item["slot"], item["operation"]
                    )
                except BaseException:
                    pass
            for handle, key in reversed(newly_registered):
                try:
                    self.active_tier.unregister_page(key)
                except BaseException:
                    pass
                self._tier_keys.pop(self._tier_handle_identity(handle), None)
            raise

    def _tier_append_slot_mapping(self, state, transaction):
        entries = self._tier_begin_append(state, transaction)
        by_logical = {item["logical"]: item for item in entries}
        slot_ids = [
            self._tier_gpu_slot(
                by_logical[position // self.page_size]["key"]
            ).slot_id
            for position in range(transaction.start, transaction.end)
        ]
        return torch.tensor(slot_ids, dtype=torch.int32, device=self.device)

    def _tier_publish_append_layer(self, state, layer, append_fence):
        append_fence.wait(timeout_seconds=self.prefetch_timeout_seconds)
        for item in self._tier_pending_appends[state.request_id]:
            location = self.store.location_set(item["key"], int(layer))
            if location.gpu_state == ResidencyState.LOADING:
                self.store.commit_gpu_layer(
                    item["key"],
                    int(layer),
                    location.gpu_slot,
                    expected_data_epoch=item["key"].data_epoch,
                )

    def _tier_commit_append(self, state, transaction):
        entries = self._tier_pending_appends.get(state.request_id, ())
        for item in entries:
            descriptor = self.page_pool.descriptor(item["handle"])
            valid = min(
                self.page_size,
                max(0, state.sequence_length - item["logical"] * self.page_size),
            )
            new_key = self.active_tier.advance_data_epoch(
                item["key"],
                descriptor.data_version,
                valid_tokens=valid,
                end_operation=item["operation"],
            )
            item["key"] = new_key
            self._tier_keys[self._tier_handle_identity(item["handle"])] = new_key
        self._tier_pending_appends.pop(state.request_id, None)
        self._tier_evict_pressure()

    def _tier_evict_pressure(self):
        while (
            self.store.above_high_watermark
            and self.store.used_pages > self.store.low_watermark_pages
        ):
            victims = self.store.lru_victims(limit=1)
            if not victims:
                raise KVLifecycleError(
                    "GPU hot cache crossed high watermark without a victim"
                )
            self.active_tier.evict(victims[0]).wait(
                timeout_seconds=self.prefetch_timeout_seconds
            )

    def _tier_abort_append(self, state, *, validate_epoch=True):
        entries = self._tier_pending_appends.pop(state.request_id, ())
        pending = state.pending_append
        allocated = set(() if pending is None else pending.allocated_handles)
        for item in reversed(tuple(entries)):
            try:
                if validate_epoch:
                    self.active_tier.abort_append(
                        item["key"], item["slot"], item["operation"]
                    )
                else:
                    # Authoritative Page publication may have advanced one
                    # or more epochs before the Request is invalidated. Every
                    # append layer is already quiescent here, so remove the
                    # stale-key pending marker using identity-safe cleanup;
                    # unregister_page below then releases the resident slot.
                    self.store.cleanup_operation(
                        item["key"], operation=item["operation"]
                    )
            except BaseException:
                pass
            if item["handle"] in allocated:
                try:
                    self.active_tier.unregister_page(
                        item["key"], validate_epoch=bool(validate_epoch)
                    )
                finally:
                    self._tier_keys.pop(
                        self._tier_handle_identity(item["handle"]), None
                    )

    def _tier_unregister_state(self, state, *, validate_epoch=True):
        if not self.active_tier_enabled:
            return
        # Device-selected attention owns no Page reference, but its arena
        # execution guard must reach the recorded CUDA event before Location
        # slots can be removed for reset/release/close.
        self.active_tier.quiesce(
            timeout=self.prefetch_timeout_seconds,
            request_id=state.request_id,
        )
        self.active_tier.cancel_request(
            state.request_id, timeout=self.prefetch_timeout_seconds
        )
        self._tier_abort_append(state, validate_epoch=validate_epoch)
        for handle in tuple(state.block_table.handles):
            identity = self._tier_handle_identity(handle)
            key = self._tier_keys.pop(identity, None)
            if key is not None:
                self.active_tier.unregister_page(
                    key, validate_epoch=bool(validate_epoch)
                )
