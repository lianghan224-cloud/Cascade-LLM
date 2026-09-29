"""Exact Expert streaming through the existing WeightStore staging path."""

from dataclasses import dataclass
import time
from typing import Dict, Tuple

import torch

from ..mixed_runtime import _BackendWeightViews
from ..pipeline import StageLease
from ..timeline import analyze_copy_compute_timeline
from .cache import ExpertCache
from .residency import ExpertKey, ExpertResidencyState


class ExpertDeviceArena:
    """Bounded cache allocation covered exactly by the unified budget."""

    def __init__(self, capacity_bytes, device):
        self.capacity_bytes = int(capacity_bytes)
        self.device = torch.device(device)
        if self.capacity_bytes < 0:
            raise ValueError("Expert arena capacity must be nonnegative")
        if self.device.type != "cuda":
            raise ValueError("Expert cache arena requires CUDA")
        self.arena = torch.empty(
            self.capacity_bytes, dtype=torch.uint8, device=self.device
        )
        self._allocations: Dict[ExpertKey, Tuple[int, int]] = {}
        self._free = [(0, self.capacity_bytes)] if self.capacity_bytes else []

    def allocate(self, key, nbytes):
        key = key if isinstance(key, ExpertKey) else ExpertKey(*key)
        nbytes = int(nbytes)
        if key in self._allocations:
            raise ValueError("Expert already has an arena allocation")
        for index, (offset, size) in enumerate(self._free):
            if size < nbytes:
                continue
            self._allocations[key] = (offset, nbytes)
            replacement = [] if size == nbytes else [(offset + nbytes, size - nbytes)]
            self._free[index : index + 1] = replacement
            return offset
        raise MemoryError("Expert cache arena is fragmented or exhausted")

    def free(self, key):
        key = key if isinstance(key, ExpertKey) else ExpertKey(*key)
        offset, nbytes = self._allocations.pop(key)
        self._free.append((offset, nbytes))
        self._free.sort()
        merged = []
        for start, size in self._free:
            if merged and merged[-1][0] + merged[-1][1] == start:
                previous = merged[-1]
                merged[-1] = (previous[0], previous[1] + size)
            else:
                merged.append((start, size))
        self._free = merged

    def bytes_view(self, key):
        key = key if isinstance(key, ExpertKey) else ExpertKey(*key)
        offset, nbytes = self._allocations[key]
        return self.arena[offset : offset + nbytes]

    def raw_views(self, key, unit, plan):
        slot = self.bytes_view(key)
        result = {}
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "int8": torch.int8,
            "uint8": torch.uint8,
        }
        for item in unit.tensors:
            spec = plan.weights[item.weight_name]
            raw = slot[item.device_offset : item.device_offset + item.storage_bytes]
            result[item.weight_name] = raw.view(dtype_map[spec.storage_dtype]).view(
                spec.storage_shape
            )
        return result

    @property
    def used_bytes(self):
        return sum(size for _offset, size in self._allocations.values())

    def close(self):
        self._allocations.clear()
        self._free.clear()
        self.arena = None


@dataclass(frozen=True)
class ExpertTransferTicket:
    key: ExpertKey
    unit: object
    cache_entry: object
    ready_event: object
    staging_slot_index: int
    submitted_ns: int


@dataclass(frozen=True)
class ExpertExecutionSchedule:
    resident_keys: Tuple[ExpertKey, ...]
    missing_keys: Tuple[ExpertKey, ...]
    tickets: Tuple[ExpertTransferTicket, ...]


class ExpertTransferEngine:
    """Reuse MultiDtypeWeightStore.prepare_unit and existing CUDA streams."""

    def __init__(self, runtime, cache, arena=None, profile=False):
        if not isinstance(cache, ExpertCache):
            raise TypeError("cache must be ExpertCache")
        self.runtime = runtime
        self.plan = runtime.plan
        self.store = runtime.store
        self.cache = cache
        self.residency = cache.residency
        self.device = runtime.device
        self.copy_stream = runtime.copy_stream
        self.compute_stream = runtime.compute_stream
        self.coordinator = runtime.coordinator
        self.profile = bool(profile)
        self.arena = arena or ExpertDeviceArena(cache.capacity_bytes, self.device)
        if self.arena.capacity_bytes != cache.capacity_bytes:
            raise ValueError("cache metadata and Expert arena budgets differ")
        self.workspace = torch.empty(
            self.plan.workspace_bytes, dtype=torch.uint8, device=self.device
        )
        self._tickets = {}
        self._closed = False
        self.h2d_bytes = 0
        self.h2d_submissions = 0
        self.host_stage_wait_ms = 0.0
        self._timing_pairs = []
        self._compute_timing_pairs = []
        self._epoch = torch.cuda.Event(enable_timing=True) if self.profile else None
        if self._epoch is not None:
            self._epoch.record(self.coordinator)
            self.copy_stream.wait_event(self._epoch)
            self.compute_stream.wait_event(self._epoch)

    def _remove_for_space(self, key, nbytes):
        before = set(self.cache.entries)
        entry = self.cache.reserve(key, nbytes)
        removed = before.difference(self.cache.entries)
        for victim in removed:
            self.arena.free(victim)
        return entry

    def submit(self, key, unit):
        if self._closed:
            raise RuntimeError("ExpertTransferEngine is closed")
        key = key if isinstance(key, ExpertKey) else ExpertKey(*key)
        existing = self.cache.lookup(key, count=False)
        if existing is not None:
            return None
        entry = self._remove_for_space(key, unit.transfer_bytes)
        allocated = False
        inflight = False
        stage_lease = None
        try:
            self.arena.allocate(key, unit.transfer_bytes)
            allocated = True
            self.residency.mark_inflight(key)
            inflight = True
            stage_lease = self.runtime.pipeline.staging_queue.get(
                timeout=self.runtime.source_timeout_seconds
            )
            stage_slot = stage_lease.slot_index
            started = time.perf_counter()
            source = self.store.prepare_unit(
                unit,
                stage_slot,
                reuse_event=stage_lease.reuse_event,
            ).result(timeout=self.runtime.source_timeout_seconds)
            self.host_stage_wait_ms += (time.perf_counter() - started) * 1000.0
            ready = torch.cuda.Event(enable_timing=False)
            timing = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            ) if self.profile else None
            destination = self.arena.bytes_view(key)
            with torch.cuda.device(self.device), torch.cuda.stream(self.copy_stream):
                if timing is not None:
                    timing[0].record(self.copy_stream)
                if torch.is_tensor(source):
                    destination[: unit.transfer_bytes].copy_(source, non_blocking=True)
                else:
                    for offset, value in source:
                        destination[offset : offset + value.numel()].copy_(
                            value, non_blocking=True
                        )
                if timing is not None:
                    timing[1].record(self.copy_stream)
                ready.record(self.copy_stream)
            self.runtime.pipeline.staging_queue.put(StageLease(stage_slot, ready))
            stage_lease = None
            self.residency.attach_inflight(key, destination, ready)
            ticket = ExpertTransferTicket(
                key, unit, entry, ready, stage_slot, time.monotonic_ns()
            )
            self._tickets[key] = ticket
            if timing is not None:
                self._timing_pairs.append((unit.transfer_bytes, timing))
            self.h2d_bytes += int(unit.transfer_bytes)
            self.h2d_submissions += 1
            return ticket
        except BaseException:
            if stage_lease is not None:
                self.runtime.pipeline.staging_queue.put(stage_lease)
            if inflight:
                self.residency.rollback_inflight(key)
            if allocated:
                self.arena.free(key)
            if key in self.cache.entries:
                self.cache.rollback_reservation(key)
            raise

    def activate(self, ticket):
        """Insert a stream dependency; never globally synchronize CUDA."""
        record = self.residency.require(ticket.key)
        if record.state == ExpertResidencyState.H2D_INFLIGHT:
            self.compute_stream.wait_event(ticket.ready_event)
            self.residency.mark_ready(
                ticket.key,
                location=self.arena.bytes_view(ticket.key),
                ready_event=ticket.ready_event,
            )
            self.cache.mark_ready(
                ticket.key,
                location=self.arena.bytes_view(ticket.key),
                ready_event=ticket.ready_event,
            )
        self._tickets.pop(ticket.key, None)
        return self.weight_views(ticket.key, ticket.unit)

    def weight_views(self, key, unit):
        backend_names = {
            tensor.weight_name: tensor.backend
            for tensor in unit.tensors
            if tensor.backend
        }
        return _BackendWeightViews(
            self.plan,
            self.arena.raw_views(key, unit, self.plan),
            self.workspace,
            backend_names,
        )

    def evict(self, key):
        entry = self.cache.remove(key)
        self.arena.free(entry.key)
        return entry

    def profile_stats(self, finalize=False):
        h2d_time_ms = None
        timeline = {}
        all_pairs = self._timing_pairs + self._compute_timing_pairs
        if finalize and all_pairs:
            # Explicit reporting boundary only; normal decode never calls a
            # device-wide synchronization function.
            for _label, pair in all_pairs:
                pair[1].synchronize()
            h2d_time_ms = sum(
                start.elapsed_time(end) for _bytes, (start, end) in self._timing_pairs
            )
            copy_intervals = [
                (self._epoch.elapsed_time(start), self._epoch.elapsed_time(end))
                for _bytes, (start, end) in self._timing_pairs
            ]
            compute_intervals = [
                (self._epoch.elapsed_time(start), self._epoch.elapsed_time(end))
                for _key, (start, end) in self._compute_timing_pairs
            ]
            wall_ms = max(
                [0.0]
                + [end for _start, end in copy_intervals]
                + [end for _start, end in compute_intervals]
            )
            timeline = analyze_copy_compute_timeline(
                copy_intervals, compute_intervals, wall_ms
            )
            timeline["transfer_compute_overlap_ratio"] = timeline[
                "copy_overlap_ratio"
            ]
            timeline["expert_h2d_stall_ms"] = timeline["copy_only_ms"]
        return {
            "expert_h2d_bytes": self.h2d_bytes,
            "expert_h2d_submissions": self.h2d_submissions,
            "expert_h2d_time_ms": h2d_time_ms,
            "host_stage_wait_ms": self.host_stage_wait_ms,
            "inflight": len(self._tickets),
            "arena_used_bytes": self.arena.used_bytes,
            **timeline,
        }

    def close(self):
        if self._closed:
            return
        if self._tickets:
            raise RuntimeError("cannot close with inflight Expert transfers")
        if not self.residency.quiescent():
            raise RuntimeError("cannot close with acquired or pinned Experts")
        self.workspace = None
        self.arena.close()
        self._closed = True


class ExpertScheduler:
    """Batch residency lookup and exact-prefetch scheduling."""

    def __init__(self, transfer_engine, unit_lookup):
        self.transfer_engine = transfer_engine
        self.cache = transfer_engine.cache
        self.unit_lookup = unit_lookup

    def build_schedule(self, layer_id, selected_expert_ids):
        ids = tuple(dict.fromkeys(int(item) for item in selected_expert_ids))
        resident = []
        missing = []
        tickets = []
        for expert_id in ids:
            key = ExpertKey(int(layer_id), expert_id)
            entry = self.cache.lookup(key)
            if entry is not None:
                resident.append(key)
                continue
            missing.append(key)
            tickets.append(
                self.transfer_engine.submit(key, self.unit_lookup(key))
            )
        return ExpertExecutionSchedule(
            tuple(resident), tuple(missing), tuple(tickets)
        )

    def execute(self, schedule, compute_expert):
        """Compute hits first, then each miss as its own event becomes ready."""
        results = {}
        for key in schedule.resident_keys:
            unit = self.unit_lookup(key)
            self.cache.acquire(key)
            try:
                with torch.cuda.stream(self.transfer_engine.compute_stream):
                    timing = self._timing_pair()
                    if timing is not None:
                        timing[0].record(self.transfer_engine.compute_stream)
                    results[key] = compute_expert(
                        key, self.transfer_engine.weight_views(key, unit)
                    )
                    if timing is not None:
                        timing[1].record(self.transfer_engine.compute_stream)
                        self.transfer_engine._compute_timing_pairs.append((key, timing))
            finally:
                self.cache.release(key)
        for ticket in schedule.tickets:
            views = self.transfer_engine.activate(ticket)
            self.cache.acquire(ticket.key)
            try:
                with torch.cuda.stream(self.transfer_engine.compute_stream):
                    timing = self._timing_pair()
                    if timing is not None:
                        timing[0].record(self.transfer_engine.compute_stream)
                    results[ticket.key] = compute_expert(ticket.key, views)
                    if timing is not None:
                        timing[1].record(self.transfer_engine.compute_stream)
                        self.transfer_engine._compute_timing_pairs.append(
                            (ticket.key, timing)
                        )
            finally:
                self.cache.release(ticket.key)
        return results

    def _timing_pair(self):
        if not self.transfer_engine.profile:
            return None
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
