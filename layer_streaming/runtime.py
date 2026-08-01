"""CUDA resident arena and two-slot asynchronous transfer runtime."""

from dataclasses import dataclass
import time

import torch

from .plan import ModelPlan
from .pipeline import PipelineRuntimeCore


@dataclass
class RuntimeStats:
    transfer_units: int
    slot_bytes: int
    two_slot_bytes: int
    device_slot_count: int
    device_slots_bytes: int
    resident_bytes: int
    weight_gpu_bytes: int

    def as_dict(self):
        return {
            "transfer_units": self.transfer_units,
            "slot_bytes": self.slot_bytes,
            "two_slot_bytes": self.two_slot_bytes,
            "device_slot_count": self.device_slot_count,
            "device_slots_bytes": self.device_slots_bytes,
            "resident_bytes": self.resident_bytes,
            "weight_gpu_bytes": self.weight_gpu_bytes,
        }


class ResidentDeviceArena:
    """GPU allocation for the plan's selected resident tensors and aliases."""

    def __init__(self, plan, store, device):
        if not isinstance(plan, ModelPlan):
            raise TypeError("plan must be a ModelPlan")
        self.plan = plan
        self.store = store
        self.device = torch.device(device)
        self.arena = torch.empty(
            plan.resident_arena_elements,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.views = {}
        for placement in plan.resident:
            spec = placement.tensor
            start = placement.device_offset_elements
            view = self.arena[start : start + spec.numel].view(spec.shape)
            source = store.tensor(spec.key)
            view.copy_(source, non_blocking=source.is_pinned())
            self.views[spec.key] = view
        for alias, target in plan.aliases.items():
            if target in self.views:
                self.views[alias] = self.views[target]
        torch.cuda.current_stream(self.device).synchronize()

    @property
    def nbytes(self):
        return self.plan.resident_bytes

    def __getitem__(self, key):
        return self.views[key]

    def close(self):
        self.views = {}
        self.arena = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class DoubleBufferRuntime:
    """Two device slots with a copy stream and a compute stream.

    ``compute_unit`` is called as ``compute_unit(unit, weight_views, state)``.
    It must enqueue all work that consumes the supplied views on the current
    compute stream and return the next state. Weight views must not escape the
    call because their slot is reused two transfer units later.
    """

    def __init__(
        self,
        plan,
        store,
        resident,
        device="cuda:0",
        slot_count=None,
        profile=False,
        source_timeout_seconds=120.0,
        prefetch_depth=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.plan = plan
        self.store = store
        self.resident = resident
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DoubleBufferRuntime requires a CUDA device")
        self.slot_count = (
            plan.slot_count if slot_count is None else int(slot_count)
        )
        if self.slot_count < 1:
            raise ValueError("slot_count must be positive")
        self.profile = bool(profile)
        self.source_timeout_seconds = float(source_timeout_seconds)
        if self.source_timeout_seconds <= 0:
            raise ValueError("source_timeout_seconds must be positive")
        self.last_profile = None
        self._closed = False
        self.prefetch_depth = (
            self.slot_count
            if prefetch_depth is None
            else int(prefetch_depth)
        )
        if self.prefetch_depth < 1:
            raise ValueError("prefetch_depth must be positive")
        self.device_slots = [
            torch.empty(
                plan.slot_elements,
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(self.slot_count)
        ]
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.coordinator = torch.cuda.current_stream(self.device)
        self.ready_events = [
            torch.cuda.Event(enable_timing=False) for _ in self.plan.units
        ]
        self.free_events = [
            torch.cuda.Event(enable_timing=False) for _ in self.plan.units
        ]
        self.run_start_event = torch.cuda.Event(enable_timing=False)
        self._profile_events = None
        if self.profile:
            self._profile_events = {
                name: [
                    torch.cuda.Event(enable_timing=True)
                    for _ in self.plan.units
                ]
                for name in (
                    "copy_starts",
                    "copy_ends",
                    "compute_starts",
                    "compute_ends",
                )
            }
            self._profile_events["pipeline_start"] = torch.cuda.Event(
                enable_timing=True
            )
            self._profile_events["pipeline_end"] = torch.cuda.Event(
                enable_timing=True
            )
        self.pipeline = PipelineRuntimeCore(
            plan=self.plan,
            store=self.store,
            slot_count=self.slot_count,
            prefetch_depth=self.prefetch_depth,
            source_timeout_seconds=self.source_timeout_seconds,
            submit_copy=self._submit_copy,
        )

    @property
    def stats(self):
        return RuntimeStats(
            transfer_units=len(self.plan.units),
            slot_bytes=self.plan.slot_bytes,
            two_slot_bytes=self.plan.two_slot_bytes,
            device_slot_count=self.slot_count,
            device_slots_bytes=self.slot_count * self.plan.slot_bytes,
            resident_bytes=self.plan.resident_bytes,
            weight_gpu_bytes=(
                self.slot_count * self.plan.slot_bytes
                + self.plan.resident_bytes
            ),
        )

    def _unit_views(self, unit, slot):
        views = {}
        for piece in unit.pieces:
            begin = piece.unit_offset_elements
            spec = piece.tensor
            views[spec.key] = slot[
                begin : begin + spec.numel
            ].view(spec.shape)
        return views

    def _submit_copy(self, prepared, lease):
        index = prepared.index
        unit = prepared.unit
        slot = self.device_slots[lease.slot_index]
        with torch.cuda.device(self.device):
            if lease.reuse_event is not None:
                self.copy_stream.wait_event(lease.reuse_event)
            with torch.cuda.stream(self.copy_stream):
                if self.profile:
                    self._profile_events["copy_starts"][index].record(
                        self.copy_stream
                    )
                slot[: unit.elements].copy_(
                    prepared.source, non_blocking=True
                )
                if self.profile:
                    self._profile_events["copy_ends"][index].record(
                        self.copy_stream
                    )
                ready_event = self.ready_events[index]
                ready_event.record(self.copy_stream)
        return ready_event

    def _consume_ready(self, ready, compute_unit, state):
        index = ready.index
        unit = ready.unit
        slot = self.device_slots[ready.device_slot_index]
        self.compute_stream.wait_event(ready.ready_event)
        with torch.cuda.stream(self.compute_stream):
            if self.profile:
                self._profile_events["compute_starts"][index].record(
                    self.compute_stream
                )
            try:
                views = self._unit_views(unit, slot)
                state = compute_unit(unit, views, state)
            finally:
                if self.profile:
                    self._profile_events["compute_ends"][index].record(
                        self.compute_stream
                    )
                free_event = self.free_events[index]
                free_event.record(self.compute_stream)
        return state, free_event

    def run(self, compute_unit, state):
        if self._closed:
            raise RuntimeError("runtime is closed")
        wall_started = time.perf_counter()
        units = self.plan.units
        if not units:
            return state

        pipeline_start = None
        pipeline_end = None
        copy_starts = []
        copy_ends = []
        compute_starts = []
        compute_ends = []
        if self.profile:
            pipeline_start = self._profile_events["pipeline_start"]
            pipeline_end = self._profile_events["pipeline_end"]
            pipeline_start.record(self.coordinator)
            self.copy_stream.wait_event(pipeline_start)
            self.compute_stream.wait_event(pipeline_start)
        else:
            self.run_start_event.record(self.coordinator)
            self.copy_stream.wait_event(self.run_start_event)
            self.compute_stream.wait_event(self.run_start_event)
        if hasattr(self.store, "reset_profile"):
            self.store.reset_profile()
        state, last_free_event = self.pipeline.run(
            lambda ready, current: self._consume_ready(
                ready, compute_unit, current
            ),
            state,
            timeout_seconds=self.source_timeout_seconds,
        )
        self.coordinator.wait_event(last_free_event)
        if self.profile:
            pipeline_end.record(self.coordinator)
        self.coordinator.synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        if self.profile:
            h2d_durations = [
                start.elapsed_time(end)
                for start, end in zip(
                    self._profile_events["copy_starts"],
                    self._profile_events["copy_ends"],
                )
            ]
            compute_durations = [
                start.elapsed_time(end)
                for start, end in zip(
                    self._profile_events["compute_starts"],
                    self._profile_events["compute_ends"],
                )
            ]
            h2d_ms = sum(h2d_durations)
            compute_ms = sum(compute_durations)
            attention_operations = {
                "q_proj", "k_proj", "v_proj", "o_proj", "qkv"
            }
            mlp_operations = {
                "gate_proj", "up_proj", "down_proj", "gate_up"
            }
            attention_ms = sum(
                duration
                for unit, duration in zip(units, compute_durations)
                if unit.operation in attention_operations
            )
            mlp_ms = sum(
                duration
                for unit, duration in zip(units, compute_durations)
                if unit.operation in mlp_operations
            )
            store_profile = (
                self.store.profile_stats()
                if hasattr(self.store, "profile_stats")
                else {}
            )
            self.last_profile = {
                "wall_ms": wall_ms,
                "host_submit_ms": self.pipeline.last_stats[
                    "pipeline_host_wall_ms"
                ],
                "source_wait_ms": self.pipeline.last_stats[
                    "source_prepare_wait_ms"
                ],
                "gpu_pipeline_ms": pipeline_start.elapsed_time(
                    pipeline_end
                ),
                "h2d_event_sum_ms": h2d_ms,
                "compute_event_sum_ms": compute_ms,
                "attention_event_sum_ms": (
                    None
                    if self.plan.granularity.value == "layer"
                    else attention_ms
                ),
                "mlp_event_sum_ms": (
                    None
                    if self.plan.granularity.value == "layer"
                    else mlp_ms
                ),
                "h2d_bytes": self.plan.stream_bytes_per_token,
                "h2d_effective_gbps": (
                    self.plan.stream_bytes_per_token / 1e9
                )
                / (h2d_ms / 1000.0),
            }
            self.last_profile.update(self.pipeline.last_stats)
            self.last_profile.update(store_profile)
        elif self.pipeline.last_stats:
            self.last_profile = dict(self.pipeline.last_stats)
        return state

    def close(self):
        if self._closed:
            return
        self.pipeline.close()
        self.device_slots = []
        self.ready_events = []
        self.free_events = []
        self.run_start_event = None
        self._profile_events = None
        self.copy_stream = None
        self.compute_stream = None
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("runtime is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
