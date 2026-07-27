"""Row lookup and vocabulary-sharded LM-head execution."""

import time

import torch
import torch.nn.functional as F

from .weight_store import WeightStoreMode


def merge_topk(
    current_values,
    current_indices,
    candidate_values,
    candidate_indices,
    top_k,
):
    """Merge two independently sorted or unsorted candidate sets exactly."""

    if current_values is None:
        merged_values = candidate_values
        merged_indices = candidate_indices
    else:
        merged_values = torch.cat((current_values, candidate_values), dim=-1)
        merged_indices = torch.cat(
            (current_indices, candidate_indices),
            dim=-1,
        )
    keep = min(int(top_k), merged_values.shape[-1])
    values, positions = torch.topk(merged_values, k=keep, dim=-1)
    indices = torch.gather(merged_indices, dim=-1, index=positions)
    return values, indices


class VocabStreamingRuntime:
    """Share Transformer slots for embedding lookup and LM-head shards."""

    def __init__(
        self,
        plan,
        store,
        transformer_runtime,
        embedding_staging_rows=256,
        profile=False,
    ):
        if plan.vocab is None:
            raise ValueError("plan does not enable vocabulary streaming")
        self.plan = plan
        self.vocab = plan.vocab
        self.store = store
        self.transformer_runtime = transformer_runtime
        self.device = transformer_runtime.device
        self.device_slots = transformer_runtime.device_slots
        self.copy_stream = transformer_runtime.copy_stream
        self.compute_stream = transformer_runtime.compute_stream
        self.coordinator = transformer_runtime.coordinator
        self.profile = bool(profile)
        self.last_profile = None
        self.last_embedding_bytes = 0
        self.slot_count = len(self.device_slots)
        self.embedding_staging_rows = int(embedding_staging_rows)
        if self.embedding_staging_rows < 1:
            raise ValueError("embedding_staging_rows must be positive")
        if self.slot_count not in (1, 2):
            raise ValueError(
                "vocabulary streaming requires one or two GPU slots"
            )
        if self.device_slots[0].numel() < self.vocab.chunk_elements:
            raise ValueError("shared GPU slot is smaller than vocab chunk")

        self.embedding_staging = torch.empty(
            (self.embedding_staging_rows, self.vocab.hidden_size),
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=True,
        )
        self.head_staging_slots = None
        if self.store.mode == WeightStoreMode.PINNED_STAGING:
            self.head_staging_slots = [
                torch.empty(
                    self.vocab.chunk_elements,
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(self.slot_count)
            ]

    @property
    def gpu_slot_bytes(self):
        return sum(slot.numel() * slot.element_size() for slot in self.device_slots)

    @property
    def extra_pinned_cpu_bytes(self):
        total = self.embedding_staging.numel() * self.embedding_staging.element_size()
        if self.head_staging_slots is not None:
            total += sum(
                slot.numel() * slot.element_size()
                for slot in self.head_staging_slots
            )
        return total

    def embedding(self, input_ids):
        """Gather only requested CPU rows and transfer them to the GPU."""

        shape = tuple(input_ids.shape)
        ids_cpu = input_ids.detach().reshape(-1).to(
            device="cpu",
            dtype=torch.long,
        )
        source = self.store.tensor(self.vocab.embedding_key)
        output = torch.empty(
            (ids_cpu.numel(), self.vocab.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.last_embedding_bytes = (
            ids_cpu.numel() * self.vocab.hidden_size * 2
        )
        for start in range(0, ids_cpu.numel(), self.embedding_staging_rows):
            end = min(start + self.embedding_staging_rows, ids_cpu.numel())
            rows = end - start
            staging = self.embedding_staging[:rows]
            torch.index_select(
                source,
                dim=0,
                index=ids_cpu[start:end],
                out=staging,
            )
            with torch.cuda.stream(self.copy_stream):
                output[start:end].copy_(staging, non_blocking=True)
            # The same small staging buffer is reused for the next group.
            self.copy_stream.synchronize()
        return output.view(shape + (self.vocab.hidden_size,))

    def _head_source(self, start_row, end_row, slot_index, ready_event):
        source = self.store.tensor(self.vocab.lm_head_key)[
            start_row:end_row
        ].reshape(-1)
        if self.head_staging_slots is None:
            if not source.is_pinned():
                raise RuntimeError(
                    "full_pinned LM-head source unexpectedly is not pinned"
                )
            return source
        if ready_event is not None:
            ready_event.synchronize()
        target = self.head_staging_slots[slot_index][: source.numel()]
        target.copy_(source)
        return target

    def lm_head_topk(self, hidden_states, top_k=10, return_full_logits=False):
        """Stream row shards and maintain an exact global top-k."""

        if hidden_states.shape[-1] != self.vocab.hidden_size:
            raise ValueError("hidden state width does not match vocabulary")
        if top_k < 1 or top_k > self.vocab.vocab_size:
            raise ValueError("top_k is outside vocabulary range")

        wall_started = time.perf_counter()
        chunk_count = self.vocab.chunk_count
        ready_events = [
            torch.cuda.Event(enable_timing=False)
            for _ in range(chunk_count)
        ]
        free_events = [
            torch.cuda.Event(enable_timing=False)
            for _ in range(chunk_count)
        ]
        copy_starts = []
        copy_ends = []
        compute_starts = []
        compute_ends = []
        pipeline_start = None
        pipeline_end = None
        if self.profile:
            pipeline_start = torch.cuda.Event(enable_timing=True)
            pipeline_end = torch.cuda.Event(enable_timing=True)
            copy_starts = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(chunk_count)
            ]
            copy_ends = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(chunk_count)
            ]
            compute_starts = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(chunk_count)
            ]
            compute_ends = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(chunk_count)
            ]
            pipeline_start.record(self.coordinator)
            self.copy_stream.wait_event(pipeline_start)
            self.compute_stream.wait_event(pipeline_start)

        global_values = None
        global_indices = None
        logits = None
        if return_full_logits:
            logits = torch.empty(
                hidden_states.shape[:-1] + (self.vocab.vocab_size,),
                dtype=torch.float32,
                device=self.device,
            )

        transferred_bytes = 0
        for chunk_index in range(chunk_count):
            slot_index = chunk_index % self.slot_count
            start_row = chunk_index * self.vocab.chunk_rows
            end_row = min(
                start_row + self.vocab.chunk_rows,
                self.vocab.vocab_size,
            )
            rows = end_row - start_row
            elements = rows * self.vocab.hidden_size
            previous_ready = (
                ready_events[chunk_index - self.slot_count]
                if chunk_index >= self.slot_count
                else None
            )
            source = self._head_source(
                start_row,
                end_row,
                slot_index,
                previous_ready,
            )
            slot = self.device_slots[slot_index]
            if chunk_index >= self.slot_count:
                self.copy_stream.wait_event(
                    free_events[chunk_index - self.slot_count]
                )
            with torch.cuda.stream(self.copy_stream):
                if self.profile:
                    copy_starts[chunk_index].record(self.copy_stream)
                slot[:elements].copy_(source, non_blocking=True)
                if self.profile:
                    copy_ends[chunk_index].record(self.copy_stream)
                ready_events[chunk_index].record(self.copy_stream)
            transferred_bytes += elements * 2

            self.compute_stream.wait_event(ready_events[chunk_index])
            with torch.cuda.stream(self.compute_stream):
                if self.profile:
                    compute_starts[chunk_index].record(self.compute_stream)
                weight = slot[:elements].view(
                    rows,
                    self.vocab.hidden_size,
                )
                partial_logits = F.linear(hidden_states, weight).float()
                local_k = min(top_k, rows)
                local_values, local_indices = torch.topk(
                    partial_logits,
                    k=local_k,
                    dim=-1,
                )
                local_indices = local_indices + start_row
                global_values, global_indices = merge_topk(
                    global_values,
                    global_indices,
                    local_values,
                    local_indices,
                    top_k,
                )
                if logits is not None:
                    logits[..., start_row:end_row] = partial_logits
                if self.profile:
                    compute_ends[chunk_index].record(self.compute_stream)
                free_events[chunk_index].record(self.compute_stream)

        self.coordinator.wait_event(free_events[-1])
        if self.profile:
            pipeline_end.record(self.coordinator)
        self.coordinator.synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000.0

        if self.profile:
            h2d_ms = sum(
                start.elapsed_time(end)
                for start, end in zip(copy_starts, copy_ends)
            )
            compute_ms = sum(
                start.elapsed_time(end)
                for start, end in zip(compute_starts, compute_ends)
            )
            self.last_profile = {
                "wall_ms": wall_ms,
                "pipeline_ms": pipeline_start.elapsed_time(pipeline_end),
                "h2d_event_sum_ms": h2d_ms,
                "compute_event_sum_ms": compute_ms,
                "h2d_bytes": transferred_bytes,
                "h2d_effective_gbps": (
                    (transferred_bytes / 1e9) / (h2d_ms / 1000.0)
                ),
                "chunk_count": chunk_count,
                "chunk_rows": self.vocab.chunk_rows,
                "requested_top_k": int(top_k),
            }
        return {
            "values": global_values,
            "indices": global_indices,
            "logits": logits,
        }
