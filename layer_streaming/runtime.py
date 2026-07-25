"""CUDA resident arena and two-slot asynchronous transfer runtime."""

from dataclasses import dataclass

import torch

from .plan import ModelPlan


@dataclass
class RuntimeStats:
    transfer_units: int
    slot_bytes: int
    two_slot_bytes: int
    resident_bytes: int
    weight_gpu_bytes: int

    def as_dict(self):
        return {
            "transfer_units": self.transfer_units,
            "slot_bytes": self.slot_bytes,
            "two_slot_bytes": self.two_slot_bytes,
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
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.plan = plan
        self.store = store
        self.resident = resident
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("DoubleBufferRuntime requires a CUDA device")
        self.device_slots = [
            torch.empty(
                plan.slot_elements,
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(plan.slot_count)
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
            resident_bytes=self.plan.resident_bytes,
            weight_gpu_bytes=(
                self.plan.two_slot_bytes + self.plan.resident_bytes
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
        units = self.plan.units
        if not units:
            return state

        ready_events = [
            torch.cuda.Event(enable_timing=False) for _ in units
        ]
        free_events = [
            torch.cuda.Event(enable_timing=False) for _ in units
        ]
        staged = {}
        for index in range(min(self.plan.slot_count, len(units))):
            staged[index] = self.store.prepare_unit(
                units[index],
                index,
            )

        for index, unit in enumerate(units):
            slot_index = index % self.plan.slot_count
            slot = self.device_slots[slot_index]
            source = staged.pop(index).result()

            if index >= self.plan.slot_count:
                self.copy_stream.wait_event(
                    free_events[index - self.plan.slot_count]
                )
            with torch.cuda.stream(self.copy_stream):
                slot[: unit.elements].copy_(source, non_blocking=True)
                ready_events[index].record(self.copy_stream)

            stage_ahead = index + self.plan.slot_count
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
                views = self._unit_views(unit, slot)
                state = compute_unit(unit, views, state)
                free_events[index].record(self.compute_stream)

        self.coordinator.wait_event(free_events[-1])
        self.coordinator.synchronize()
        return state
