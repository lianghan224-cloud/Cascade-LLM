"""Mixed-dtype streamed Embedding and LM Head."""

import math
import time

import torch

from .multi_dtype_store import MultiDtypeStoreMode
from .providers.generic_cuda import deterministic_lm_head
from .specs import DTYPE_BYTES
from .vocab import merge_topk


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class MixedVocabStreamingRuntime:
    def __init__(
        self,
        plan,
        store,
        transformer_runtime,
        embedding_staging_rows=256,
        profile=False,
    ):
        self.plan = plan
        self.vocab = plan.vocab
        self.store = store
        self.runtime = transformer_runtime
        self.device = transformer_runtime.device
        self.device_slots = transformer_runtime.transfer_slots
        self.copy_stream = transformer_runtime.copy_stream
        self.compute_stream = transformer_runtime.compute_stream
        self.coordinator = transformer_runtime.coordinator
        self.profile = bool(profile)
        self.embedding_staging_rows = int(embedding_staging_rows)
        if self.embedding_staging_rows < 1:
            raise ValueError("embedding_staging_rows must be positive")
        self.stream_embedding = self.vocab.embedding_mode == "streamed"
        self.stream_lm_head = self.vocab.lm_head_mode == "streamed"
        embedding_spec = plan.weights[self.vocab.embedding_name]
        lm_spec = plan.weights[self.vocab.lm_head_name]
        if lm_spec.alias_of is not None:
            lm_spec = plan.weights[lm_spec.alias_of]
        if embedding_spec.quantization is not None or lm_spec.quantization is not None:
            raise ValueError("quantized vocabulary weights are not supported")
        self.embedding_spec = embedding_spec
        self.lm_spec = lm_spec
        self.embedding_staging = None
        self.embedding_ready = None
        if self.stream_embedding:
            self.embedding_staging = torch.empty(
                (self.embedding_staging_rows, plan.geometry.hidden_size),
                dtype=_TORCH_DTYPES[embedding_spec.storage_dtype],
                device="cpu",
                pin_memory=True,
            )
            self.embedding_ready = torch.cuda.Event(enable_timing=False)
        self.chunk_count = int(
            math.ceil(plan.geometry.vocab_size / float(self.vocab.chunk_rows))
        )
        self.ready_events = (
            [torch.cuda.Event(enable_timing=False) for _ in range(self.chunk_count)]
            if self.stream_lm_head
            else []
        )
        self.free_events = (
            [torch.cuda.Event(enable_timing=False) for _ in range(self.chunk_count)]
            if self.stream_lm_head
            else []
        )
        self.head_staging_slots = None
        if (
            self.stream_lm_head
            and store.mode == MultiDtypeStoreMode.PINNED_STAGING
        ):
            self.head_staging_slots = [
                torch.empty(
                    (
                        self.vocab.chunk_rows,
                        plan.geometry.hidden_size,
                    ),
                    dtype=_TORCH_DTYPES[lm_spec.storage_dtype],
                    device="cpu",
                    pin_memory=True,
                )
                for _ in self.device_slots
            ]
        self.last_profile = None
        self.last_embedding_wall_ms = 0.0
        self._closed = False

    @property
    def extra_pinned_cpu_bytes(self):
        result = 0
        if self.embedding_staging is not None:
            result += self.embedding_staging.numel() * self.embedding_staging.element_size()
        if self.head_staging_slots is not None:
            result += sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.head_staging_slots
            )
        return result

    def embedding(self, input_ids):
        if not self.stream_embedding:
            raise RuntimeError("embedding streaming is disabled")
        started = time.perf_counter()
        shape = tuple(input_ids.shape)
        ids = input_ids.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        if ids.numel() > self.embedding_staging_rows:
            raise ValueError("embedding staging capacity exceeded")
        source = self.store.view(self.vocab.embedding_name)
        staging = self.embedding_staging[: ids.numel()]
        torch.index_select(source, 0, ids, out=staging)
        output = torch.empty(
            (ids.numel(), self.plan.geometry.hidden_size),
            dtype=staging.dtype,
            device=self.device,
        )
        with torch.cuda.stream(self.copy_stream):
            output.copy_(staging, non_blocking=True)
            self.embedding_ready.record(self.copy_stream)
        self.compute_stream.wait_event(self.embedding_ready)
        self.last_embedding_wall_ms = (time.perf_counter() - started) * 1000.0
        return output.view(shape + (self.plan.geometry.hidden_size,))

    def _head_source(self, start, end, slot_index, previous_ready):
        source = self.store.view(self.vocab.lm_head_name)[start:end]
        if self.head_staging_slots is None:
            if not source.is_pinned():
                raise RuntimeError("full_pinned LM Head source is not pinned")
            return source
        if previous_ready is not None:
            previous_ready.synchronize()
        target = self.head_staging_slots[slot_index][: end - start]
        target.copy_(source)
        return target

    def lm_head_topk(self, hidden_states, top_k=10, return_full_logits=False):
        if not self.stream_lm_head:
            raise RuntimeError("LM Head streaming is disabled")
        if top_k < 1 or top_k > self.plan.geometry.vocab_size:
            raise ValueError("top_k is outside vocabulary range")
        started = time.perf_counter()
        global_values = None
        global_indices = None
        logits = None
        if return_full_logits:
            logits = torch.empty(
                hidden_states.shape[:-1] + (self.plan.geometry.vocab_size,),
                dtype=torch.float32,
                device=self.device,
            )
        transferred = 0
        slot_count = len(self.device_slots)
        for index in range(self.chunk_count):
            slot_index = index % slot_count
            start = index * self.vocab.chunk_rows
            end = min(start + self.vocab.chunk_rows, self.plan.geometry.vocab_size)
            rows = end - start
            source = self._head_source(
                start,
                end,
                slot_index,
                self.ready_events[index - slot_count] if index >= slot_count else None,
            )
            if index >= slot_count:
                self.copy_stream.wait_event(self.free_events[index - slot_count])
            byte_count = source.numel() * source.element_size()
            with torch.cuda.stream(self.copy_stream):
                self.device_slots[slot_index][:byte_count].copy_(
                    source.reshape(-1).view(torch.uint8),
                    non_blocking=True,
                )
                self.ready_events[index].record(self.copy_stream)
            transferred += byte_count
            self.compute_stream.wait_event(self.ready_events[index])
            with torch.cuda.stream(self.compute_stream):
                weight = self.device_slots[slot_index][:byte_count].view(
                    _TORCH_DTYPES[self.lm_spec.storage_dtype]
                ).view(rows, self.plan.geometry.hidden_size)
                partial = deterministic_lm_head(hidden_states, weight)
                local_k = min(top_k, rows)
                values, indices = torch.topk(partial, local_k, dim=-1)
                global_values, global_indices = merge_topk(
                    global_values,
                    global_indices,
                    values,
                    indices + start,
                    top_k,
                )
                if logits is not None:
                    logits[..., start:end] = partial
                self.free_events[index].record(self.compute_stream)
        self.coordinator.wait_event(self.free_events[-1])
        self.coordinator.synchronize()
        self.last_profile = {
            "wall_ms": (time.perf_counter() - started) * 1000.0,
            "h2d_bytes": transferred,
            "chunk_count": self.chunk_count,
            "chunk_rows": self.vocab.chunk_rows,
            "embedding_wall_ms": self.last_embedding_wall_ms,
            "lm_head_backend": "deterministic_cuda_fp32_accum_native_output",
        }
        return {
            "values": global_values,
            "indices": global_indices,
            "logits": logits,
        }

    def profile_stats(self):
        result = dict(self.last_profile or {})
        result["embedding_wall_ms"] = self.last_embedding_wall_ms
        return result

    def resource_stats(self):
        return {
            "closed": self._closed,
            "pinned_bytes": (
                0 if self._closed else self.extra_pinned_cpu_bytes
            ),
            "event_count": (
                len(self.ready_events)
                + len(self.free_events)
                + int(self.embedding_ready is not None)
            ),
            "stream_embedding": self.stream_embedding,
            "stream_lm_head": self.stream_lm_head,
        }

    def close(self):
        if self._closed:
            return
        self.embedding_staging = None
        self.embedding_ready = None
        self.head_staging_slots = None
        self.ready_events = []
        self.free_events = []
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
