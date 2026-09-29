"""Page-kernel and attention execution coordination without page ownership."""

from contextlib import contextmanager
import time

import torch

from ..attention.paged import (
    DevicePagedAttentionInput,
    PagedAttentionInput,
    PagedKVAppendInput,
)
from .errors import KVLifecycleError
from .page_view import DeviceSelectedPageView
from .slot_mapping import SlotMapping


class KVExecutionCoordinator:
    """Build kernel inputs while OwnershipManager controls every reference."""

    def __init__(self, runtime):
        self.runtime = runtime

    @staticmethod
    def normalize_kv_tensor(tensor, total_tokens, num_kv_heads, head_dim):
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
        runtime = self.runtime
        requests = tuple(requests)
        query_lengths = tuple(int(item) for item in query_lengths)
        if len(requests) != len(query_lengths) or not requests:
            raise ValueError("append batch metadata is invalid")
        if runtime.active_tier_enabled and len(requests) != 1:
            raise NotImplementedError(
                "Active Tier append currently supports batch 1"
            )
        if runtime.selection.name == "rgkv" and len(requests) != 1:
            raise NotImplementedError("RGKV append currently supports batch 1")
        layer = int(layer)
        if layer < 0 or layer >= runtime.layer_count:
            raise IndexError("layer is outside KV runtime")
        total_tokens = sum(query_lengths)
        key = self.normalize_kv_tensor(
            key, total_tokens, runtime.num_kv_heads, runtime.head_dim
        )
        value = self.normalize_kv_tensor(
            value, total_tokens, runtime.num_kv_heads, runtime.head_dim
        )
        if key.device != runtime.device or value.device != runtime.device:
            raise ValueError("KV append tensors are on the wrong device")
        if key.dtype != runtime.dtype or value.dtype != runtime.dtype:
            raise ValueError("KV append tensors have the wrong dtype")
        pending = []
        try:
            for state, token_count in zip(requests, query_lengths):
                transaction = runtime.ownership.begin_append(state, token_count)
                if layer in transaction.completed_layers:
                    raise KVLifecycleError(
                        "layer was appended twice in one transaction"
                    )
                pending.append(transaction)
            pages = (
                runtime._tier_append_slot_mapping(requests[0], pending[0])
                if runtime.active_tier_enabled
                else torch.cat([item.slot_page_ids for item in pending])
            )
            offsets = torch.cat([item.slot_offsets for item in pending])
            key_pool, value_pool = runtime.store.layer_view(layer)
            append_handles = runtime.ownership.begin_append_kernel(
                layer, requests, pending
            )
            try:
                runtime.kv_kernel_backend.append_kv(
                    PagedKVAppendInput(
                        key=key,
                        value=value,
                        key_pool_view=key_pool,
                        value_pool_view=value_pool,
                        slot_mapping=SlotMapping(pages, offsets),
                        page_size=runtime.page_size,
                        num_kv_heads=runtime.num_kv_heads,
                        head_dim=runtime.head_dim,
                    )
                )
            except BaseException as original_error:
                try:
                    runtime.ownership.abort_append_kernel(append_handles)
                except BaseException as cleanup_error:
                    try:
                        original_error.kv_cleanup_errors = (cleanup_error,)
                    except BaseException:
                        pass
                raise
            append_fence = runtime.ownership.record_append_kernel(
                layer, append_handles
            )
            if hasattr(runtime.selection, "stage_append_layer"):
                offset = 0
                update_started = time.perf_counter()
                for state, transaction, token_count in zip(
                    requests, pending, query_lengths
                ):
                    runtime.selection.stage_append_layer(
                        runtime,
                        state,
                        layer,
                        key[offset : offset + token_count],
                        transaction,
                        (token_count,),
                    )
                    offset += token_count
                runtime._metrics.rgkv_update_ms += (
                    time.perf_counter() - update_started
                ) * 1000.0
            if runtime.active_tier_enabled:
                runtime._tier_publish_append_layer(
                    requests[0], layer, append_fence
                )
            for state, transaction in zip(requests, pending):
                transaction.completed_layers.add(layer)
                transaction.layer_fences[layer] = append_fence
                state.layer_lengths[layer] = transaction.end
            runtime._metrics.append_calls += 1
            runtime._metrics.append_tokens += total_tokens
        except BaseException as original_error:
            cleanup_errors = []
            for state, transaction in zip(requests, pending):
                try:
                    if runtime.active_tier_enabled:
                        runtime._tier_abort_append(state)
                    runtime.ownership.abort_append(state)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            if cleanup_errors:
                # Keep the actual append/provider exception as the raised
                # error; cleanup diagnostics remain inspectable without
                # masking the cause on Python 3.10.
                try:
                    original_error.kv_cleanup_errors = tuple(cleanup_errors)
                except BaseException:
                    pass
            raise
        for state in requests:
            if len(state.pending_append.completed_layers) == runtime.layer_count:
                transaction = state.pending_append
                committed = False
                try:
                    # Ownership first drains every layer Append Fence and
                    # publishes the one authoritative global Data Epoch.
                    runtime.ownership.commit(state)
                    committed = True
                    if runtime.active_tier_enabled:
                        runtime._tier_commit_append(state, transaction)
                    if hasattr(runtime.selection, "commit_append"):
                        update_started = time.perf_counter()
                        runtime.selection.commit_append(
                            runtime, state, transaction
                        )
                        runtime._metrics.rgkv_update_ms += (
                            time.perf_counter() - update_started
                        ) * 1000.0
                    elif hasattr(runtime.selection, "build_request_layer"):
                        changed = range(
                            transaction.start // runtime.page_size,
                            (transaction.end - 1) // runtime.page_size + 1,
                        )
                        for current_layer in range(runtime.layer_count):
                            runtime.selection.build_request_layer(
                                runtime,
                                state,
                                current_layer,
                                changed_blocks=changed,
                            )
                except BaseException as original_error:
                    cleanup_errors = []
                    publication_started = bool(
                        transaction.ownership_publication_started
                    )
                    if not committed and not publication_started:
                        # Ownership commit can fail while draining an Append
                        # Fence, before the authoritative Data Epoch exists.
                        # Discard transaction-local summaries and restore the
                        # original logical mappings/lengths exactly as any
                        # earlier append-layer failure would.
                        try:
                            if runtime.active_tier_enabled:
                                runtime._tier_abort_append(state)
                            runtime.ownership.abort_append(state)
                        except BaseException as cleanup_error:
                            cleanup_errors.append(cleanup_error)
                    # Once payload and Page epoch have committed, the prior
                    # payload cannot be reconstructed from compact summaries.
                    # Never expose a half-published Request: invalidate and
                    # release it completely while preserving the root error.
                    if committed or publication_started:
                        if runtime.active_tier_enabled:
                            try:
                                runtime._tier_unregister_state(
                                    state, validate_epoch=False
                                )
                            except BaseException as cleanup_error:
                                cleanup_errors.append(cleanup_error)
                        try:
                            if hasattr(runtime.selection, "abort_append"):
                                runtime.selection.abort_append(
                                    state, transaction
                                )
                        except BaseException as cleanup_error:
                            cleanup_errors.append(cleanup_error)
                        # release() must not re-enter ordinary append rollback:
                        # this request is being invalidated, not restored.
                        state.pending_append = None
                        try:
                            runtime.ownership.release(state)
                        except BaseException as cleanup_error:
                            cleanup_errors.append(cleanup_error)
                    if cleanup_errors:
                        try:
                            original_error.kv_cleanup_errors = tuple(
                                cleanup_errors
                            )
                        except BaseException:
                            pass
                    raise
        if runtime.active_tier_enabled:
            return (SlotMapping(pages, offsets),)
        return tuple(
            SlotMapping(item.slot_page_ids, item.slot_offsets)
            for item in pending
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
        runtime = self.runtime
        requests = tuple(requests)
        if runtime.active_tier_enabled and len(requests) != 1:
            raise NotImplementedError(
                "Active Tier attention currently supports batch 1"
            )
        query_lengths = tuple(int(item) for item in query_lengths)
        total_tokens = sum(query_lengths)
        if query.ndim == 4 and query.shape[0] == 1 and len(requests) == 1:
            query = query.squeeze(0).transpose(0, 1)
        if query.ndim != 3 or tuple(query.shape) != (
            total_tokens,
            runtime.num_query_heads,
            runtime.head_dim,
        ):
            raise ValueError(
                "query must be flattened [token, query_head, head_dim]"
            )
        query = query.contiguous()
        batch = runtime.prepare_batch(
            requests,
            query_lengths,
            layer,
            query_positions=query_positions,
        )
        device_guard = None
        device_decode_candidate = (
            runtime.active_tier_enabled
            and runtime.selection.name == "rgkv"
            and query.device.type == "cuda"
            and all(length == 1 for length in query_lengths)
            and phase in {None, "decode"}
        )
        if device_decode_candidate:
            # Freeze hot-slot identity before selection gathers Location
            # metadata. This closes the selection-to-launch reuse window while
            # leaving Request refs and Page ownership untouched.
            device_guard = runtime.active_tier.begin_hot_cache_execution(
                request_id=requests[0].request_id
            )
        selection_started = time.perf_counter()
        try:
            selected = runtime.selection.select(requests, layer, query, batch)
        except BaseException as error:
            if device_guard is not None:
                # Selection can enqueue scorer/assert kernels. Preserve arena
                # identity until the current stream reaches this event.
                try:
                    fence = device_guard.submit_close()
                    error.attention_fence_id = fence.operation_id
                except BaseException as cleanup_error:
                    try:
                        error.kv_cleanup_errors = (cleanup_error,)
                    except BaseException:
                        pass
            if "stale" in str(error).lower() and runtime.selection.name != "dense":
                runtime._metrics.rgkv_stale_index_count += 1
            raise
        selection_enqueue_ms = (time.perf_counter() - selection_started) * 1000.0
        device_decode = (
            device_decode_candidate
            and isinstance(selected, DeviceSelectedPageView)
        )
        if device_decode_candidate and not device_decode:
            fence = device_guard.submit_close()
            error = KVLifecycleError(
                "RGKV CUDA Decode did not produce DeviceSelectedPageView"
            )
            error.attention_fence_id = fence.operation_id
            raise error
        if runtime.selection.name != "dense":
            scorer = getattr(runtime.selection, "scorer", None)
            selection_stats = (
                runtime.selection.stats()
                if hasattr(runtime.selection, "stats")
                else {}
            )
            runtime._metrics.observe_rgkv_selection(
                pages_total=int(batch.flat_block_table.shape[0]),
                pages_selected=int(selected.flat_page_ids.shape[0]),
                index_bytes=int(selection_stats.get("index_bytes", 0)),
                selection_enqueue_ms=float(selection_enqueue_ms),
                # CUDA score/top-k duration needs device events resolved after
                # a later synchronization.  Do not mislabel host enqueue time
                # as kernel execution time.
                timing_sampled=False,
                # Scoring itself stays tensorized. The current CPU
                # Ownership/Location bridge packs boundaries, logical IDs,
                # physical IDs and valid-token metadata into one bulk Host
                # transfer. It remains an explicit synchronization, but is no
                # longer one synchronization per selected scalar/page.
                cpu_sync_count=(
                    0
                    if device_decode
                    else int(selected.logical_block_ids.device.type == "cuda")
                ),
                # Device epoch validation rejects stale metadata without a
                # candidate-wide host scan, but PageHandle generation and
                # lifecycle authority are still checked once per selected
                # entry by resolve_entries(). Keep that bridge visible.
                host_authority_page_checks=(
                    0
                    if device_decode
                    else int(selected.flat_page_ids.shape[0])
                ),
            )
        key_pool, value_pool = runtime.store.layer_view(layer)
        request_type = DevicePagedAttentionInput if device_decode else PagedAttentionInput
        request = request_type(
            query=query,
            key_pool_view=key_pool,
            value_pool_view=value_pool,
            batch_view=batch,
            page_size=runtime.page_size,
            num_query_heads=runtime.num_query_heads,
            num_kv_heads=runtime.num_kv_heads,
            head_dim=runtime.head_dim,
            softmax_scale=runtime.softmax_scale,
            causal=bool(causal),
            kv_dtype=runtime.policy.dtype.value,
            output_dtype=(
                "bf16" if runtime.dtype == torch.bfloat16
                else "fp16" if runtime.dtype == torch.float16
                else "fp32"
            ),
            selected_pages=selected,
            workspace=None,
            return_logsumexp=bool(return_logsumexp),
        )
        if device_decode:
            return self._attend_active_tier_device_hit(
                requests[0], layer, request, device_guard
            )
        if runtime.active_tier_enabled:
            selected_count = int(selected.flat_page_ids.shape[0])
            is_cuda_bridge = selected.logical_block_ids.device.type == "cuda"
            is_decode = all(length == 1 for length in query_lengths) and phase in {
                None,
                "decode",
            }
            if is_decode:
                runtime._metrics.observe_decode_execution(
                    selected_metadata_d2h=int(is_cuda_bridge),
                    selected_metadata_d2h_bytes=(
                        8 * (2 + 3 * selected_count) if is_cuda_bridge else 0
                    ),
                    host_scalar_readbacks=int(is_cuda_bridge),
                    # Each page is visited twice by the bounded two-pass
                    # implementation. CUDA waves synchronously close after
                    # each visit; CPU reference waves are not CUDA syncs.
                    explicit_sync_count=(
                        2 * selected_count if is_cuda_bridge else 0
                    ),
                    python_attention_wave_count=2 * selected_count,
                )
            return self._attend_active_tier(
                requests[0], layer, request, selected, query_lengths, phase
            )
        if selected.logical_block_ids.device.type == "cuda":
            selected_count = int(selected.flat_page_ids.shape[0])
            runtime._metrics.observe_decode_execution(
                selected_metadata_d2h=1,
                selected_metadata_d2h_bytes=8 * (len(requests) + 1 + 3 * selected_count),
            )
        handles = selected.resolve_handles(requests, runtime.page_pool)
        runtime.ownership.begin_attention_kernel(layer, handles)
        started = time.perf_counter()
        try:
            result = runtime.dispatcher.execute(request, phase=phase)
        except BaseException as original_error:
            try:
                failed_fence = runtime.ownership.abort_attention_kernel(
                    layer, handles, error=original_error
                )
                original_error.attention_fence_id = failed_fence.operation_id
            except BaseException as cleanup_error:
                try:
                    original_error.kv_cleanup_errors = (cleanup_error,)
                except BaseException:
                    pass
            raise
        elapsed = (time.perf_counter() - started) * 1000.0
        runtime._metrics.attention_calls += 1
        resolved_phase = phase or (
            "decode" if max(query_lengths) == 1 else "prefill"
        )
        if resolved_phase == "decode":
            runtime._metrics.decode_attention_ms += elapsed
        else:
            runtime._metrics.prefill_attention_ms += elapsed
        workspace_bytes = int(
            result.provider_metrics.get("workspace_bytes", 0)
        )
        runtime._metrics.workspace_peak_bytes = max(
            runtime._metrics.workspace_peak_bytes,
            workspace_bytes,
        )
        fence = runtime.ownership.record_attention_kernel(
            layer,
            handles,
            request_ids=(state.request_id for state in requests),
        )
        result.provider_metrics["attention_fence_id"] = fence.operation_id
        result.provider_metrics["selected_pin_count"] = len(handles)
        return result

    def _attend_active_tier_device_hit(self, state, layer, request, guard):
        """Launch the strict all-GPU-resident device-selected Decode path.

        Selection IDs, Page generation/epoch validation and logical-to-hot-slot
        resolution remain device resident.  The arena-wide execution guard is
        ownership-neutral: it only prevents Location slot reuse until the
        compute-stream event completes.
        """

        runtime = self.runtime
        started = time.perf_counter()
        try:
            result = runtime.dispatcher.execute_device(
                request, phase="decode"
            )
        except BaseException as original_error:
            # The provider may have enqueued validation or partial kernel work
            # before raising. Record the current stream and keep slot identity
            # frozen until that work is quiescent; never abort optimistically.
            cleanup_error = None
            try:
                fence = guard.submit_close()
                original_error.attention_fence_id = fence.operation_id
            except BaseException as current_error:
                cleanup_error = current_error
            if cleanup_error is not None:
                try:
                    original_error.kv_cleanup_errors = (cleanup_error,)
                except BaseException:
                    pass
            raise
        elapsed = (time.perf_counter() - started) * 1000.0
        fence = guard.submit_close()
        runtime._metrics.attention_calls += 1
        runtime._metrics.decode_attention_ms += elapsed
        runtime._metrics.workspace_peak_bytes = max(
            runtime._metrics.workspace_peak_bytes,
            int(result.provider_metrics.get("workspace_bytes", 0)),
        )
        runtime._metrics.observe_decode_execution(
            gpu_hit_device_attention_calls=1,
        )
        result.provider_metrics["attention_fence_id"] = fence.operation_id
        result.provider_metrics["selected_view"] = "device"
        result.provider_metrics["selected_metadata_d2h"] = 0
        result.provider_metrics["host_scalar_readbacks"] = 0
        result.provider_metrics["explicit_sync_count"] = 0
        result.provider_metrics["python_attention_wave_count"] = 0
        result.provider_metrics["device_view_fallback_count"] = 0
        result.provider_metrics["hot_cache_execution_guard"] = "arena"
        return result

    def _attend_active_tier(
        self, state, layer, request, selected, query_lengths, phase
    ):
        """Run exact attention through one-page residency waves."""

        runtime = self.runtime
        selected_entries = selected.resolve_entries(
            (state,), runtime.page_pool
        )
        handles = tuple(entry.handle for entry in selected_entries)
        selected_keys_ordered = tuple(
            runtime._tier_registered_key(entry.handle)
            for entry in selected_entries
        )
        selected_keys = set(selected_keys_ordered)
        if None in selected_keys:
            raise KVLifecycleError(
                "selected logical page has no Active Tier location"
            )
        visited_keys = set()
        prefetched_keys = set()
        prefetch_submissions = 0
        visit = 0

        @contextmanager
        def page_loader(index):
            nonlocal visit, prefetch_submissions
            key = selected_keys_ordered[index]
            if key is None:
                raise KVLifecycleError(
                    "selected logical page has no Active Tier location"
                )
            visit += 1
            visited_keys.add(key)
            wave = runtime.active_tier.acquire_wave(
                (key,),
                request_id="tier-attention:{}:{}:{}".format(
                    state.request_id, int(layer), visit
                ),
                timeout=runtime.prefetch_timeout_seconds,
                layer=int(layer),
            )
            try:
                prefetched_keys.update(wave.prefetched_page_keys)
                prefetch_submissions += len(wave.prefetched_page_keys)
                page = wave.layer_pages(layer)[0]
                yield page.key, page.value
            finally:
                wave.close(timeout=runtime.prefetch_timeout_seconds)

        started = time.perf_counter()
        result = runtime._tier_streaming_attention.execute(
            request,
            page_loader,
            page_metadata=tuple(
                (entry.logical_block_id, entry.valid_tokens)
                for entry in selected_entries
            ),
        )
        if visited_keys != selected_keys:
            raise KVLifecycleError(
                "temporary attention pin/read set diverged from Selection"
            )
        if not prefetched_keys.issubset(selected_keys):
            raise KVLifecycleError(
                "Active Tier prefetched a page outside Selection"
            )
        elapsed = (time.perf_counter() - started) * 1000.0
        runtime._metrics.attention_calls += 1
        resolved_phase = phase or (
            "decode" if max(query_lengths) == 1 else "prefill"
        )
        if resolved_phase == "decode":
            runtime._metrics.decode_attention_ms += elapsed
        else:
            runtime._metrics.prefill_attention_ms += elapsed
        workspace_bytes = int(
            result.provider_metrics.get("workspace_bytes", 0)
        )
        runtime._metrics.workspace_peak_bytes = max(
            runtime._metrics.workspace_peak_bytes, workspace_bytes
        )
        fence = runtime.ownership._fence(
            "attention_tiered_streaming",
            request_id=(state.request_id,),
            source_handles=handles,
        ).mark_completed(runtime.ownership._next_operation_epoch())
        runtime.ownership._remember_fence(fence)
        result.provider_metrics["attention_fence_id"] = fence.operation_id
        result.provider_metrics["selected_pin_count"] = 1
        result.provider_metrics["tiered_selected_pages"] = len(handles)
        result.provider_metrics["kernel_actual_read_pages"] = len(visited_keys)
        result.provider_metrics["temporary_attention_pin_peak"] = 1
        result.provider_metrics["prefetch_selected_only"] = True
        result.provider_metrics["prefetch_unique_pages"] = len(prefetched_keys)
        result.provider_metrics["prefetch_page_submissions"] = prefetch_submissions
        return result
