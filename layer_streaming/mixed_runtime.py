"""Generic byte-slot runtime for dense, INT8, and packed INT4 weights."""

import threading
import time

import torch

from .backends import backend_for_weight
from .pipeline import PipelineRuntimeCore
from .placement import build_static_transformer_placement
from .specs import DTYPE_BYTES


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
    "uint8": torch.uint8,
}

_STREAM_POOL_LOCK = threading.Lock()
_STREAM_PAIRS = {}


def _shared_stream_pair(device):
    """Reuse native streams so cuBLAS workspaces do not grow per model load."""

    device = torch.device(device)
    key = device.index
    if key is None:
        key = torch.cuda.current_device()
    with _STREAM_POOL_LOCK:
        pair = _STREAM_PAIRS.get(key)
        if pair is None:
            with torch.cuda.device(device):
                pair = (
                    torch.cuda.Stream(device=device),
                    torch.cuda.Stream(device=device),
                )
            _STREAM_PAIRS[key] = pair
        return pair


class MixedResidentDeviceArena:
    def __init__(
        self,
        plan,
        store,
        device="cuda:0",
        transformer_placement=None,
    ):
        self.plan = plan
        self.store = store
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("resident device arena requires CUDA")
        self.transformer_placement = (
            transformer_placement
            or build_static_transformer_placement(plan, 0)
        )
        placements = tuple(plan.resident) + tuple(
            self.transformer_placement.resident_tensors
        )
        names = [item.weight_name for item in placements]
        if len(names) != len(set(names)):
            raise ValueError("resident placement contains duplicate weights")
        arena_bytes = max(
            (
                item.device_offset + item.storage_bytes
                for item in placements
            ),
            default=0,
        )
        self.arena = torch.empty(
            arena_bytes, dtype=torch.uint8, device=self.device
        )
        self.views = {}
        for item in placements:
            spec = plan.weights[item.weight_name]
            raw = self.arena[
                item.device_offset : item.device_offset + item.storage_bytes
            ]
            view = raw.view(_TORCH_DTYPES[spec.storage_dtype]).view(
                spec.storage_shape
            )
            source = store.view(item.weight_name)
            view.copy_(source, non_blocking=source.is_pinned())
            self.views[item.weight_name] = view
        for alias, target in plan.aliases.items():
            if target in self.views:
                self.views[alias] = self.views[target]
        torch.cuda.current_stream(self.device).synchronize()

    @property
    def nbytes(self):
        return 0 if self.arena is None else self.arena.numel()

    def __getitem__(self, name):
        return self.views[name]

    def raw_views(self, unit):
        return {
            tensor.weight_name: self.views[tensor.weight_name]
            for tensor in unit.tensors
        }

    def close(self):
        self.views = {}
        self.arena = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class _BackendWeightViews:
    """Lazily dequantize into a slot-owned reusable workspace."""

    def __init__(
        self,
        plan,
        raw_views,
        workspace,
        backend_names,
        dequant_events=None,
    ):
        self.plan = plan
        self.raw_views = raw_views
        self.workspace = workspace
        self.backend_names = backend_names
        self.dequant_events = dequant_events or {}
        self.used_backends = []
        self.used_providers = []
        self.used_dequant_events = []

    def __getitem__(self, name):
        spec = self.plan.weights[name]
        raw = self.raw_views[name]
        backend = backend_for_weight(
            spec, backend_name=self.backend_names.get(name)
        )
        self.used_backends.append(backend.name)
        self.used_providers.append(
            getattr(backend, "provider_name", backend.name)
        )
        if spec.quantization is None:
            return raw
        scale = self.raw_views[name + "_scale"]
        compute_dtype = _TORCH_DTYPES[spec.compute_dtype]
        workspace = self.workspace[
            : backend.workspace_bytes(spec, batch_tokens=1)
        ].view(compute_dtype)
        quant_views = {"scale": scale, "weight_spec": spec}
        if backend.is_fallback:
            event_pair = self.dequant_events.get(name)
            if event_pair is not None:
                event_pair[0].record()
            result = backend.dequantize(
                spec,
                raw,
                quant_views,
                workspace=workspace,
            )
            if event_pair is not None:
                event_pair[1].record()
                self.used_dequant_events.append(event_pair)
            return result
        return _ExecutableBackendWeight(
            backend, raw, quant_views, workspace
        )


class _ExecutableBackendWeight:
    """Ephemeral slot view routed through a registered fused backend."""

    def __init__(self, backend, weight_view, quant_views, workspace):
        self.backend = backend
        self.weight_view = weight_view
        self.quant_views = quant_views
        self.workspace = workspace

    def execute(self, activations):
        return self.backend.execute(
            activations,
            self.weight_view,
            self.quant_views,
            self.workspace,
        )


class MixedDtypeRuntime:
    """Bounded producer/copy/consumer runtime with typed views over byte slots."""

    def __init__(
        self,
        plan,
        store,
        resident,
        device="cuda:0",
        slot_count=None,
        prefetch_depth=None,
        source_timeout_seconds=120.0,
        profile=False,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.plan = plan
        self.store = store
        self.resident = resident
        self.transformer_placement = (
            getattr(resident, "transformer_placement", None)
            or build_static_transformer_placement(plan, 0)
        )
        resident_ids = set(self.transformer_placement.resident_unit_ids)
        streamed_ids = set(self.transformer_placement.streamed_unit_ids)
        all_ids = {unit.unit_id for unit in plan.units}
        if resident_ids.intersection(streamed_ids) or (
            resident_ids.union(streamed_ids) != all_ids
        ):
            raise ValueError("Transformer placement does not partition plan units")
        self.resident_units = tuple(
            unit for unit in plan.units if unit.unit_id in resident_ids
        )
        self.streamed_units = tuple(
            unit for unit in plan.units if unit.unit_id in streamed_ids
        )
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("MixedDtypeRuntime requires CUDA")
        self.slot_count = int(
            plan.slot_count if slot_count is None else slot_count
        )
        self.prefetch_depth = int(
            self.slot_count if prefetch_depth is None else prefetch_depth
        )
        self.source_timeout_seconds = float(source_timeout_seconds)
        self.profile = bool(profile)
        if self.slot_count < 1 or self.prefetch_depth < 1:
            raise ValueError("slot_count and prefetch_depth must be positive")
        self.transfer_slots = [
            torch.empty(plan.slot_bytes, dtype=torch.uint8, device=self.device)
            for _ in range(self.slot_count)
        ]
        self.workspace_slots = [
            torch.empty(plan.workspace_bytes, dtype=torch.uint8, device=self.device)
            for _ in range(self.slot_count)
        ]
        self.copy_stream, self.compute_stream = _shared_stream_pair(
            self.device
        )
        self.coordinator = torch.cuda.current_stream(self.device)
        self.ready_events = [
            torch.cuda.Event(enable_timing=False) for _ in plan.units
        ]
        self.free_events = [
            torch.cuda.Event(enable_timing=False) for _ in plan.units
        ]
        self.start_event = torch.cuda.Event(enable_timing=False)
        self.resident_done_event = (
            torch.cuda.Event(enable_timing=False)
            if self.resident_units
            else None
        )
        self._timing_events = None
        self._dequant_events = {}
        self._linear_events = {}
        self._stage_events = {}
        self._resident_compute_events = {}
        if self.profile:
            self._timing_events = {
                name: [
                    torch.cuda.Event(enable_timing=True) for _ in plan.units
                ]
                for name in (
                    "copy_start",
                    "copy_end",
                    "compute_start",
                    "compute_end",
                )
            }
            self._dequant_events = {
                spec.name: (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for spec in plan.weights.values()
                if spec.quantization is not None
            }
            self._linear_events = {
                spec.name: (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for spec in plan.weights.values()
                if spec.role
                in {
                    "attention_q",
                    "attention_k",
                    "attention_v",
                    "attention_o",
                    "mlp_gate",
                    "mlp_up",
                    "mlp_down",
                }
            }
            self._stage_events = {
                ("attention", layer_index): (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for layer_index in range(plan.geometry.num_hidden_layers)
            }
            self._resident_compute_events = {
                unit.unit_id: (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for unit in self.resident_units
            }
        self.used_backends = set()
        self.used_providers = set()
        self.backend_names = {
            tensor.weight_name: tensor.backend
            for unit in plan.units
            for tensor in unit.tensors
            if tensor.backend
        }
        self.backend_phase_plan = None
        self._active_phase = None
        self._used_dequant_events = []
        self.last_profile = None
        self._closed = False
        self.pipeline = PipelineRuntimeCore(
            plan=plan,
            store=store,
            slot_count=self.slot_count,
            prefetch_depth=self.prefetch_depth,
            source_timeout_seconds=self.source_timeout_seconds,
            submit_copy=self._submit_copy,
        )

    @property
    def stats(self):
        transfer = self.slot_count * self.plan.slot_bytes
        workspace = self.slot_count * self.plan.workspace_bytes
        return {
            "transfer_units": len(self.plan.units),
            "slot_count": self.slot_count,
            "transfer_slot_bytes": self.plan.slot_bytes,
            "workspace_bytes_per_slot": self.plan.workspace_bytes,
            "device_transfer_slots_bytes": transfer,
            "device_dequant_workspaces_bytes": workspace,
            "resident_bytes": self.resident.nbytes,
            "resident_transformer_weight_bytes": (
                self.transformer_placement.resident_weight_bytes
            ),
            "streamed_transformer_weight_bytes": (
                self.transformer_placement.streamed_weight_bytes
            ),
            "weight_gpu_bytes": transfer + workspace + self.resident.nbytes,
        }

    def configure_backend_phase_plan(self, phase_plan):
        """Install an explicitly validated prefill/decode dispatch sidecar.

        This additive method deliberately leaves the frozen constructor and
        ExecutionPlan schema unchanged.
        """

        if int(phase_plan.workspace_bytes) > int(self.plan.workspace_bytes):
            raise ValueError(
                "backend phase plan requires {} workspace bytes, plan has "
                "{}".format(
                    phase_plan.workspace_bytes, self.plan.workspace_bytes
                )
            )
        expected = {
            tensor.weight_name
            for unit in self.plan.units
            for tensor in unit.tensors
            if tensor.backend
        }
        for phase in ("prefill", "decode"):
            actual = set(phase_plan.backends_for_phase(phase))
            if actual != expected:
                raise ValueError(
                    "{} backend mapping keys do not match streamed weights: "
                    "missing={}, extra={}".format(
                        phase,
                        sorted(expected - actual),
                        sorted(actual - expected),
                    )
                )
        self.backend_phase_plan = phase_plan
        return self

    @staticmethod
    def _phase_for_state(state):
        hidden = getattr(state, "hidden_states", None)
        if hidden is None or hidden.ndim < 2:
            raise ValueError(
                "explicit phase dispatch requires state.hidden_states"
            )
        tokens = 1
        for dimension in hidden.shape[:-1]:
            tokens *= int(dimension)
        return "decode" if tokens == 1 else "prefill"

    def resource_stats(self):
        timing_event_count = 0
        if self._timing_events is not None:
            timing_event_count = sum(
                len(events) for events in self._timing_events.values()
            )
        pipeline = self.pipeline.resource_stats()
        result = {
            "closed": self._closed,
            "event_count": (
                len(self.ready_events)
                + len(self.free_events)
                + (0 if self._closed else 1)
                + int(self.resident_done_event is not None)
                + timing_event_count
                + 2 * len(self._dequant_events)
                + 2 * len(self._linear_events)
                + 2 * len(self._stage_events)
                + 2 * len(self._resident_compute_events)
            ),
            "stream_count": 0 if self._closed else 2,
            "transfer_slot_count": len(self.transfer_slots),
            "workspace_slot_count": len(self.workspace_slots),
            "used_backends": sorted(self.used_backends),
            "pipeline": pipeline,
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            result.update(
                {
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(
                        self.device
                    ),
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(
                        self.device
                    ),
                    "cuda_max_allocated_bytes": torch.cuda.max_memory_allocated(
                        self.device
                    ),
                    "cuda_max_reserved_bytes": torch.cuda.max_memory_reserved(
                        self.device
                    ),
                }
            )
        return result

    def _submit_copy(self, prepared, lease):
        index = prepared.index
        unit = prepared.unit
        slot = self.transfer_slots[lease.slot_index]
        with torch.cuda.device(self.device):
            if lease.reuse_event is not None:
                self.copy_stream.wait_event(lease.reuse_event)
            with torch.cuda.stream(self.copy_stream):
                if self.profile:
                    self._timing_events["copy_start"][index].record(
                        self.copy_stream
                    )
                if torch.is_tensor(prepared.source):
                    slot[: unit.transfer_bytes].copy_(
                        prepared.source, non_blocking=True
                    )
                else:
                    for offset, source in prepared.source:
                        slot[offset : offset + source.numel()].copy_(
                            source, non_blocking=True
                        )
                if self.profile:
                    self._timing_events["copy_end"][index].record(
                        self.copy_stream
                    )
                ready = self.ready_events[index]
                ready.record(self.copy_stream)
        return ready

    def _raw_views(self, unit, slot):
        result = {}
        for item in unit.tensors:
            spec = self.plan.weights[item.weight_name]
            raw = slot[
                item.device_offset : item.device_offset + item.storage_bytes
            ]
            result[item.weight_name] = raw.view(
                _TORCH_DTYPES[spec.storage_dtype]
            ).view(spec.storage_shape)
        return result

    def _consume(self, ready, compute_unit, state):
        index = ready.index
        unit = ready.unit
        slot_index = ready.device_slot_index
        self.compute_stream.wait_event(ready.ready_event)
        with torch.cuda.stream(self.compute_stream):
            if self.profile:
                self._timing_events["compute_start"][index].record(
                    self.compute_stream
                )
            try:
                views = _BackendWeightViews(
                    self.plan,
                    self._raw_views(unit, self.transfer_slots[slot_index]),
                    self.workspace_slots[slot_index],
                    self.backend_names,
                    dequant_events=self._dequant_events,
                )
                state = compute_unit(unit, views, state)
                self.used_backends.update(views.used_backends)
                self.used_providers.update(views.used_providers)
                self._used_dequant_events.extend(views.used_dequant_events)
            finally:
                if self.profile:
                    self._timing_events["compute_end"][index].record(
                        self.compute_stream
                    )
                free = self.free_events[index]
                free.record(self.compute_stream)
        return state, free

    def _consume_resident(self, unit, compute_unit, state):
        event_pair = self._resident_compute_events.get(unit.unit_id)
        if event_pair is not None:
            event_pair[0].record(self.compute_stream)
        views = _BackendWeightViews(
            self.plan,
            self.resident.raw_views(unit),
            self.workspace_slots[0],
            self.backend_names,
            dequant_events=self._dequant_events,
        )
        state = compute_unit(unit, views, state)
        self.used_backends.update(views.used_backends)
        self.used_providers.update(views.used_providers)
        self._used_dequant_events.extend(views.used_dequant_events)
        if event_pair is not None:
            event_pair[1].record(self.compute_stream)
        return state

    def run(self, compute_unit, state):
        if self._closed:
            raise RuntimeError("runtime is closed")
        if not self.plan.units:
            return state
        self.used_backends = set()
        self.used_providers = set()
        self._active_phase = (
            self._phase_for_state(state)
            if self.backend_phase_plan is not None
            else None
        )
        if self.backend_phase_plan is not None:
            self.backend_names = self.backend_phase_plan.backends_for_phase(
                self._active_phase
            )
        started = time.perf_counter()
        if self.profile and hasattr(compute_unit, "set_linear_profiler"):
            def record_linear(phase, name):
                pair = self._linear_events[name]
                pair[0 if phase == "start" else 1].record(
                    self.compute_stream
                )

            compute_unit.set_linear_profiler(record_linear)
        if self.profile and hasattr(compute_unit, "set_stage_profiler"):
            def record_stage(phase, stage, layer_index):
                pair = self._stage_events.get((stage, layer_index))
                if pair is not None:
                    pair[0 if phase == "start" else 1].record(
                        self.compute_stream
                    )

            compute_unit.set_stage_profiler(record_stage)
        self.start_event.record(self.coordinator)
        self.copy_stream.wait_event(self.start_event)
        self.compute_stream.wait_event(self.start_event)
        self.store.reset_profile()
        self._used_dequant_events = []
        resident_done = None
        try:
            if self.resident_units:
                with torch.cuda.stream(self.compute_stream):
                    for unit in self.resident_units:
                        state = self._consume_resident(
                            unit, compute_unit, state
                        )
                    resident_done = self.resident_done_event
                    resident_done.record(self.compute_stream)
            if self.streamed_units:
                state, last_free = self.pipeline.run(
                    lambda ready, current: self._consume(
                        ready, compute_unit, current
                    ),
                    state,
                    timeout_seconds=self.source_timeout_seconds,
                    units=self.streamed_units,
                )
            else:
                last_free = resident_done
                self.pipeline.last_stats = {
                    "pipeline_host_wall_ms": 0.0,
                    "source_prepare_wait_ms": 0.0,
                    "ready_wait_ms": 0.0,
                    "free_slot_wait_ms": 0.0,
                    "source_queue_max_depth": 0,
                    "ready_queue_max_depth": 0,
                    "source_queue_capacity": self.pipeline.source_queue.maxsize,
                    "ready_queue_capacity": self.pipeline.ready_queue.maxsize,
                }
        finally:
            if self.profile and hasattr(compute_unit, "set_linear_profiler"):
                compute_unit.set_linear_profiler(None)
            if self.profile and hasattr(compute_unit, "set_stage_profiler"):
                compute_unit.set_stage_profiler(None)
        if last_free is not None:
            self.coordinator.wait_event(last_free)
        self.coordinator.synchronize()
        result = dict(self.pipeline.last_stats)
        result.update(self.store.profile_stats())
        result["wall_ms"] = (time.perf_counter() - started) * 1000.0
        result["h2d_bytes"] = sum(
            unit.transfer_bytes for unit in self.streamed_units
        )
        result["quant_weight_h2d_bytes"] = sum(
            tensor.storage_bytes
            for unit in self.streamed_units
            for tensor in unit.tensors
            if self.plan.weights[tensor.weight_name].quantization is not None
        )
        result["scale_h2d_bytes"] = sum(
            tensor.storage_bytes
            for unit in self.streamed_units
            for tensor in unit.tensors
            if self.plan.weights[tensor.weight_name].role == "scale"
        )
        result["weight_h2d_bytes"] = (
            result["h2d_bytes"] - result["scale_h2d_bytes"]
        )
        result.update(self.transformer_placement.as_dict())
        result["backends"] = sorted(self.used_backends)
        result["fallback_backends"] = sorted(
            name for name in self.used_backends if "fallback" in name
        )
        result["backend_providers"] = sorted(self.used_providers)
        if self.backend_phase_plan is not None:
            result.update(self.backend_phase_plan.selection.as_dict())
            result["backend_phase"] = self._active_phase
            result["phase_backend"] = (
                self.backend_phase_plan.selection.prefill
                if self._active_phase == "prefill"
                else self.backend_phase_plan.selection.decode
            )
        if self.profile:
            copy_ms = sum(
                self._timing_events["copy_start"][index].elapsed_time(
                    self._timing_events["copy_end"][index]
                )
                for index in range(len(self.streamed_units))
            )
            compute_ms = sum(
                self._timing_events["compute_start"][index].elapsed_time(
                    self._timing_events["compute_end"][index]
                )
                for index in range(len(self.streamed_units))
            )
            compute_ms += sum(
                start.elapsed_time(end)
                for start, end in self._resident_compute_events.values()
            )
            result["h2d_event_sum_ms"] = copy_ms
            result["compute_event_sum_ms"] = compute_ms
            result["dequant_event_sum_ms"] = sum(
                start.elapsed_time(end)
                for start, end in self._used_dequant_events
            )
            result["gemm_event_sum_ms"] = sum(
                start.elapsed_time(end)
                for start, end in self._linear_events.values()
            )
            result["attention_event_sum_ms"] = sum(
                start.elapsed_time(end)
                for start, end in self._stage_events.values()
            )
        self.last_profile = result
        return state

    def close(self):
        if self._closed:
            return
        self.pipeline.close()
        self.transfer_slots = []
        self.workspace_slots = []
        self.ready_events = []
        self.free_events = []
        self.start_event = None
        self.resident_done_event = None
        self._timing_events = None
        self._dequant_events = {}
        self._linear_events = {}
        self._stage_events = {}
        self._resident_compute_events = {}
        self.copy_stream = None
        self.compute_stream = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


# Frozen M5 public name.  The longer original name remains an exact alias.
MixedRuntime = MixedDtypeRuntime
