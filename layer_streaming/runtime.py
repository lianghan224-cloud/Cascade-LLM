"""CUDA resident arena and two-slot asynchronous transfer runtime."""

from dataclasses import dataclass
import time

import torch

from .plan import ModelPlan


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
    """GPU allocation for embedding, LM head, norms, and aliases."""

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
            self.views[alias] = self.views[target]
        torch.cuda.current_stream(self.device).synchronize()

    @property
    def nbytes(self):
        return self.plan.resident_bytes

    def __getitem__(self, key):
        return self.views[key]


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
        if self.slot_count not in (1, 2):
            raise ValueError("slot_count must be 1 or 2")
        self.profile = bool(profile)
        self.last_profile = None
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

    def run(self, compute_unit, state):
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
            pipeline_start = torch.cuda.Event(enable_timing=True)
            pipeline_end = torch.cuda.Event(enable_timing=True)
            pipeline_start.record(self.coordinator)
            self.copy_stream.wait_event(pipeline_start)
            self.compute_stream.wait_event(pipeline_start)
        ready_events = [
            torch.cuda.Event(enable_timing=False) for _ in units
        ]
        free_events = [
            torch.cuda.Event(enable_timing=False) for _ in units
        ]
        if self.profile:
            copy_starts = [
                torch.cuda.Event(enable_timing=True) for _ in units
            ]
            copy_ends = [
                torch.cuda.Event(enable_timing=True) for _ in units
            ]
            compute_starts = [
                torch.cuda.Event(enable_timing=True) for _ in units
            ]
            compute_ends = [
                torch.cuda.Event(enable_timing=True) for _ in units
            ]
        if hasattr(self.store, "reset_profile"):
            self.store.reset_profile()
        staged = {}
        for index in range(min(self.slot_count, len(units))):
            staged[index] = self.store.prepare_unit(
                units[index],
                index,
            )

        source_wait_ms = 0.0
        submit_started = time.perf_counter()
        for index, unit in enumerate(units):
            slot_index = index % self.slot_count
            slot = self.device_slots[slot_index]
            source_wait_started = time.perf_counter()
            source = staged.pop(index).result()
            source_wait_ms += (
                time.perf_counter() - source_wait_started
            ) * 1000.0

            if index >= self.slot_count:
                self.copy_stream.wait_event(
                    free_events[index - self.slot_count]
                )
            with torch.cuda.stream(self.copy_stream):
                if self.profile:
                    copy_starts[index].record(self.copy_stream)
                slot[: unit.elements].copy_(source, non_blocking=True)
                if self.profile:
                    copy_ends[index].record(self.copy_stream)
                ready_events[index].record(self.copy_stream)

            stage_ahead = index + self.slot_count
            if stage_ahead < len(units):
                # The CPU staging worker may overwrite this host slot as soon
                # as the corresponding H2D DMA has completed. It need not wait
                # for GPU compute, which uses the separate device slot.
                reuse_event = (
                    ready_events[index]
                    if self.store.mode.value == "pinned_staging"
                    else None
                )
                staged[stage_ahead] = self.store.prepare_unit(
                    units[stage_ahead],
                    slot_index,
                    reuse_event=reuse_event,
                )

            self.compute_stream.wait_event(ready_events[index])
            with torch.cuda.stream(self.compute_stream):
                if self.profile:
                    compute_starts[index].record(self.compute_stream)
                views = self._unit_views(unit, slot)
                state = compute_unit(unit, views, state)
                if self.profile:
                    compute_ends[index].record(self.compute_stream)
                free_events[index].record(self.compute_stream)

        submit_ms = (time.perf_counter() - submit_started) * 1000.0
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
            store_profile = (
                self.store.profile_stats()
                if hasattr(self.store, "profile_stats")
                else {}
            )
            self.last_profile = {
                "wall_ms": wall_ms,
                "host_submit_ms": submit_ms,
                "source_wait_ms": source_wait_ms,
                "gpu_pipeline_ms": pipeline_start.elapsed_time(
                    pipeline_end
                ),
                "h2d_event_sum_ms": h2d_ms,
                "compute_event_sum_ms": compute_ms,
                "h2d_bytes": self.plan.stream_bytes_per_token,
                "h2d_effective_gbps": (
                    self.plan.stream_bytes_per_token / 1e9
                )
                / (h2d_ms / 1000.0),
            }
            self.last_profile.update(store_profile)
        return state
