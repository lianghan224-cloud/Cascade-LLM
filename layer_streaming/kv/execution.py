"""Page-kernel and attention execution coordination without page ownership."""

import math
import time

import torch

from ..attention.paged import PagedAttentionInput, PagedKVAppendInput
from .errors import KVLifecycleError
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
            pages = torch.cat([item.slot_page_ids for item in pending])
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
            except BaseException:
                runtime.ownership.abort_append_kernel(append_handles)
                raise
            runtime.ownership.record_append_kernel(layer, append_handles)
            for state, transaction in zip(requests, pending):
                transaction.completed_layers.add(layer)
                state.layer_lengths[layer] = transaction.end
                if hasattr(runtime.selection, "build_request_layer"):
                    runtime.selection.build_request_layer(
                        runtime,
                        state,
                        layer,
                        version=state.version + 1,
                        changed_blocks=range(
                            transaction.start // runtime.page_size,
                            (transaction.end - 1) // runtime.page_size + 1,
                        ),
                    )
            runtime._metrics.append_calls += 1
            runtime._metrics.append_tokens += total_tokens
        except BaseException:
            for state in requests:
                runtime.ownership.abort_append(state)
            raise
        for state in requests:
            if len(state.pending_append.completed_layers) == runtime.layer_count:
                runtime.ownership.commit(state)
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
        selected = runtime.selection.select(requests, layer, query, batch)
        key_pool, value_pool = runtime.store.layer_view(layer)
        request = PagedAttentionInput(
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
        handles = []
        for state in requests:
            required = int(
                math.ceil(
                    state.layer_lengths[int(layer)]
                    / float(runtime.page_size)
                )
            )
            handles.extend(state.block_table.handles[:required])
        runtime.ownership.begin_attention_kernel(layer, handles)
        started = time.perf_counter()
        try:
            result = runtime.dispatcher.execute(request, phase=phase)
        except BaseException:
            runtime.ownership.abort_attention_kernel(layer, handles)
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
        runtime.ownership.record_attention_kernel(layer)
        return result
