"""Mixed-dtype CPU weight regions and pinned staging resources."""

from dataclasses import dataclass
from enum import Enum
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import time
from typing import Dict

import torch
from safetensors import safe_open

from .checkpoint import CheckpointManifest
from .specs import DTYPE_BYTES


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}


class MultiDtypeStoreMode(str, Enum):
    FULL_PINNED = "full_pinned"
    PINNED_STAGING = "pinned_staging"


@dataclass
class WeightRegion:
    name: str
    storage_dtype: str
    base_tensor: object
    bytes: int
    pinned: bool


class MultiDtypeWeightStore:
    """Own independent typed arenas and optionally reusable pinned byte slots."""

    def __init__(
        self,
        plan,
        mode=MultiDtypeStoreMode.PINNED_STAGING,
        staging_slot_count=None,
    ):
        self.plan = plan
        self.mode = MultiDtypeStoreMode(mode)
        self.staging_slot_count = int(
            staging_slot_count
            if staging_slot_count is not None
            else plan.slot_count
        )
        if self.staging_slot_count < 1:
            raise ValueError("staging_slot_count must be positive")
        self.slot_count = self.staging_slot_count
        self.regions = {}
        self.staging_slots = []
        self._executor = None
        self._profile_lock = threading.Lock()
        self._stage_durations_ms = []
        self._closed = False
        try:
            self._allocate()
        except BaseException:
            self.close()
            raise

    def _allocate(self):
        pinned = self.mode == MultiDtypeStoreMode.FULL_PINNED
        for name, region_plan in self.plan.regions.items():
            item_bytes = DTYPE_BYTES[region_plan.storage_dtype]
            if region_plan.bytes % item_bytes:
                raise ValueError(
                    "region {} byte count is not dtype aligned".format(name)
                )
            tensor = torch.empty(
                region_plan.bytes // item_bytes,
                dtype=_TORCH_DTYPES[region_plan.storage_dtype],
                device="cpu",
                pin_memory=pinned,
            )
            self.regions[name] = WeightRegion(
                name=name,
                storage_dtype=region_plan.storage_dtype,
                base_tensor=tensor,
                bytes=region_plan.bytes,
                pinned=bool(tensor.is_pinned()),
            )
        if self.mode == MultiDtypeStoreMode.PINNED_STAGING:
            self.staging_slots = [
                torch.empty(
                    self.plan.slot_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(self.staging_slot_count)
            ]
            self._executor = ThreadPoolExecutor(
                max_workers=self.staging_slot_count,
                thread_name_prefix="mixed-weight-stage",
            )

    @property
    def pinned_bytes(self):
        if self.mode == MultiDtypeStoreMode.FULL_PINNED:
            return sum(region.bytes for region in self.regions.values())
        return sum(slot.numel() for slot in self.staging_slots)

    @property
    def pageable_bytes(self):
        if self.mode == MultiDtypeStoreMode.FULL_PINNED:
            return 0
        return sum(region.bytes for region in self.regions.values())

    def _canonical_name(self, name):
        seen = set()
        while name in self.plan.aliases:
            if name in seen:
                raise ValueError("cyclic weight alias involving {}".format(name))
            seen.add(name)
            name = self.plan.aliases[name]
        return name

    def view(self, name):
        if self._closed:
            raise RuntimeError("weight store is closed")
        name = self._canonical_name(name)
        spec = self.plan.weights[name]
        region_name = next(
            item.storage_region
            for unit in self.plan.units
            for item in unit.tensors
            if item.weight_name == name
        ) if any(
            item.weight_name == name
            for unit in self.plan.units
            for item in unit.tensors
        ) else None
        if region_name is None:
            from .execution_plan import storage_region_name

            region_name = storage_region_name(spec)
        region = self.regions[region_name]
        item_bytes = DTYPE_BYTES[spec.storage_dtype]
        offset = self.plan.host_offsets[name]
        if offset % item_bytes:
            raise RuntimeError("{} host offset is dtype-unaligned".format(name))
        start = offset // item_bytes
        end = start + spec.storage_numel
        return region.base_tensor[start:end].view(spec.storage_shape)

    def load_checkpoint(self, checkpoint):
        aliases = dict(self.plan.aliases)
        manifest = CheckpointManifest.from_path(checkpoint, aliases=aliases)
        manifest.validate(tuple(self.plan.weights.values())).raise_for_error()
        by_file = {}
        for name, tensor_manifest in manifest.tensors.items():
            if name in self.plan.weights and self.plan.weights[name].alias_of is None:
                by_file.setdefault(tensor_manifest.file, []).append(name)
        try:
            for filename, names in by_file.items():
                with safe_open(filename, framework="pt", device="cpu") as source:
                    for name in names:
                        destination = self.view(name)
                        value = source.get_tensor(name)
                        if value.dtype != destination.dtype:
                            raise ValueError(
                                "{} changed dtype while loading".format(name)
                            )
                        destination.copy_(value)
        except BaseException:
            # Loaded bytes are not externally visible as a valid store.
            self.close()
            raise
        return manifest

    def stage_unit(self, unit, staging_slot_index):
        if self.mode != MultiDtypeStoreMode.PINNED_STAGING:
            raise RuntimeError("stage_unit is only used by pinned_staging")
        slot = self.staging_slots[int(staging_slot_index)]
        for tensor in unit.tensors:
            source = self.view(tensor.weight_name).reshape(-1).view(torch.uint8)
            start = int(tensor.device_offset)
            end = start + int(tensor.storage_bytes)
            slot[start:end].copy_(source)
        return slot[: unit.transfer_bytes]

    def source_bytes(self, unit):
        """Return direct pinned pieces for full_pinned mode."""

        if self.mode != MultiDtypeStoreMode.FULL_PINNED:
            raise RuntimeError("source_bytes is only used by full_pinned")
        return tuple(
            (
                tensor.device_offset,
                self.view(tensor.weight_name).reshape(-1).view(torch.uint8),
            )
            for tensor in unit.tensors
        )

    def _stage(self, unit, staging_slot_index, reuse_event):
        started = time.perf_counter()
        if reuse_event is not None:
            reuse_event.synchronize()
        result = self.stage_unit(unit, staging_slot_index)
        elapsed = (time.perf_counter() - started) * 1000.0
        with self._profile_lock:
            self._stage_durations_ms.append(elapsed)
        return result

    def prepare_unit(self, unit, slot_index, reuse_event=None):
        if self._closed:
            raise RuntimeError("weight store is closed")
        if self.mode == MultiDtypeStoreMode.FULL_PINNED:
            future = Future()
            future.set_result(self.source_bytes(unit))
            return future
        if self._executor is None:
            raise RuntimeError("pinned staging executor is closed")
        return self._executor.submit(
            self._stage, unit, slot_index, reuse_event
        )

    def reset_profile(self):
        with self._profile_lock:
            self._stage_durations_ms = []

    def profile_stats(self):
        with self._profile_lock:
            values = list(self._stage_durations_ms)
        return {
            "staging_copy_count": len(values),
            "staging_event_sum_ms": sum(values),
            "staging_event_max_ms": max(values) if values else 0.0,
        }

    def resource_stats(self):
        threads = (
            tuple(self._executor._threads)
            if self._executor is not None
            else ()
        )
        return {
            "closed": self._closed,
            "mode": self.mode.value,
            "pinned_bytes": 0 if self._closed else self.pinned_bytes,
            "pageable_bytes": 0 if self._closed else self.pageable_bytes,
            "region_count": len(self.regions),
            "staging_slot_count": len(self.staging_slots),
            "worker_thread_count": sum(thread.is_alive() for thread in threads),
        }

    def close(self):
        if self._closed:
            return
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.staging_slots.clear()
        self.regions.clear()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("weight store is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
