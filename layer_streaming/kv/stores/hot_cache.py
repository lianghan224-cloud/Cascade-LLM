"""Bounded GPU KV hot cache and generation-safe location plane.

The cache slot is a complete token page across every model layer.  A
``PageHandle`` identifies the logical/ownership page but is never used as a
GPU slot index.  This module owns no Request reference: ``KVPagePoolV1``
remains the sole ownership and lifecycle authority.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import uuid

import torch

from ..device_metadata import (
    DEVICE_PAGE_ABSENT,
    DEVICE_PAGE_CPU,
    DEVICE_PAGE_GPU,
    DEVICE_PAGE_GPU_CPU,
    DeviceKVPageTable,
)
from ..errors import KVCapacityError, KVLifecycleError
from ..types import PageState
from .base import KVStoreCapability
from .tiered import ResidencyState


_BUSY_PAGE_STATES = {
    PageState.COPYING,
    PageState.MIGRATING,
    PageState.EVICT_PENDING,
    PageState.RELEASING,
}


@dataclass(frozen=True)
class GPUHotPageKey:
    """Stable identity for one logical token page at one data epoch."""

    logical_key: str
    pool_uuid: str
    page_id: int
    page_generation: int
    data_epoch: int


@dataclass(frozen=True)
class GPUHotSlotHandle:
    """Generation-bearing handle into the fixed GPU hot arena."""

    cache_uuid: str
    slot_id: int
    generation: int

    def as_dict(self):
        return {
            "cache_uuid": self.cache_uuid,
            "slot_id": self.slot_id,
            "generation": self.generation,
        }


@dataclass
class KVLocationSet:
    """GPU/CPU locations for one layer of a logical token page."""

    page_key: GPUHotPageKey
    layer: int
    gpu_slot: object = None
    gpu_state: ResidencyState = ResidencyState.ABSENT
    cpu_slot: object = None
    cpu_state: ResidencyState = ResidencyState.ABSENT
    authoritative_tier: object = None

    def as_dict(self):
        return {
            "logical_key": self.page_key.logical_key,
            "layer": self.layer,
            "data_epoch": self.page_key.data_epoch,
            "gpu_slot": (
                None if self.gpu_slot is None else self.gpu_slot.as_dict()
            ),
            "gpu_state": self.gpu_state.value,
            "cpu_slot": None if self.cpu_slot is None else repr(self.cpu_slot),
            "cpu_state": self.cpu_state.value,
            "authoritative_tier": self.authoritative_tier,
        }


@dataclass(frozen=True)
class GPUHotLayerPage:
    """Validated zero-copy view for one resident layer page."""

    page_key: GPUHotPageKey
    layer: int
    slot: GPUHotSlotHandle
    key: torch.Tensor
    value: torch.Tensor
    authoritative_tier: object


@dataclass
class _HotSlotRecord:
    generation: int = 0
    state: ResidencyState = ResidencyState.ABSENT
    page_key: object = None
    pin_count: int = 0
    inflight_compute: int = 0
    inflight_io: int = 0
    selected_count: int = 0
    pending_operations: set = field(default_factory=set)
    last_access_epoch: int = 0


@dataclass
class _PageLocationRecord:
    key: GPUHotPageKey
    page_handle: object
    locations: dict


class GPUHotKVCache:
    """Fixed-capacity GPU page arena with an independent Location Plane.

    Capacity is measured in complete token pages.  Each slot contains K and V
    for every layer, so all ``KVLocationSet`` objects belonging to a logical
    page share exactly one slot handle.
    """

    store_id = "gpu_hot"

    def __init__(
        self,
        *,
        page_pool,
        layer_count,
        gpu_capacity_pages,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda:0",
        high_watermark_pages=None,
        low_watermark_pages=None,
        tensor_factory=None,
    ):
        self.page_pool = page_pool
        self.layer_count = self._positive_int(layer_count, "layer_count")
        self.gpu_capacity_pages = self._positive_int(
            gpu_capacity_pages, "gpu_capacity_pages"
        )
        self.num_kv_heads = self._positive_int(
            num_kv_heads, "num_kv_heads"
        )
        self.page_size = self._positive_int(page_size, "page_size")
        self.head_dim = self._positive_int(head_dim, "head_dim")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        self.dtype = dtype
        self.device = torch.device(device)
        element_size = torch.empty((), dtype=dtype).element_size()
        self.layer_page_bytes = (
            2
            * self.num_kv_heads
            * self.page_size
            * self.head_dim
            * element_size
        )
        self.gpu_page_bytes = self.layer_count * self.layer_page_bytes
        self.gpu_capacity_bytes = (
            self.gpu_capacity_pages * self.gpu_page_bytes
        )
        # Compatibility alias for low-level GPUKVStore consumers.  Values are
        # hot slot IDs, never PageHandle.page_id values.
        self.page_count = self.gpu_capacity_pages
        default_high = max(1, int(self.gpu_capacity_pages * 0.9))
        default_low = int(self.gpu_capacity_pages * 0.7)
        self.high_watermark_pages = self._watermark(
            high_watermark_pages, default_high, "high_watermark_pages"
        )
        self.low_watermark_pages = self._watermark(
            low_watermark_pages, default_low, "low_watermark_pages"
        )
        if self.low_watermark_pages > self.high_watermark_pages:
            raise ValueError(
                "low_watermark_pages must not exceed high_watermark_pages"
            )
        self.high_watermark_bytes = (
            self.high_watermark_pages * self.gpu_page_bytes
        )
        self.low_watermark_bytes = (
            self.low_watermark_pages * self.gpu_page_bytes
        )

        self.cache_uuid = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._clock = 0
        self._used_pages = 0
        self._peak_used_pages = 0
        self._closed = False
        self._slots = [
            _HotSlotRecord() for _ in range(self.gpu_capacity_pages)
        ]
        # Pop from the end for deterministic ascending allocation.
        self._free_slot_ids = list(
            reversed(range(self.gpu_capacity_pages))
        )
        self._pages = {}
        # Device-driven attention must not read selected slots back to the Host
        # merely to protect them from reuse.  Until slot-granular device
        # inflight accounting exists, these opaque execution guards freeze the
        # whole arena's slot identity.  They own no PagePool ref or owner.
        self._execution_guards = set()
        self.device_page_table = DeviceKVPageTable(
            self.page_pool.page_count, device=self.device
        )

        shape = (
            self.layer_count,
            self.gpu_capacity_pages,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
        )
        factory = tensor_factory or torch.empty
        self.keys = factory(shape, dtype=dtype, device=self.device)
        try:
            self.values = factory(shape, dtype=dtype, device=self.device)
        except BaseException:
            self.keys = None
            raise

    @staticmethod
    def _positive_int(value, name):
        value = int(value)
        if value <= 0:
            raise ValueError("{} must be positive".format(name))
        return value

    def _watermark(self, value, default, name):
        value = default if value is None else int(value)
        if value < 0 or value > self.gpu_capacity_pages:
            raise ValueError(
                "{} must be within [0, gpu_capacity_pages]".format(name)
            )
        return value

    @staticmethod
    def _logical_key(value):
        return str(value.key()) if hasattr(value, "key") else str(value)

    def _ensure_open(self):
        if self._closed:
            raise KVLifecycleError("GPU hot cache is closed")

    def begin_execution_guard(self, token=None):
        """Freeze arena slot identity until the caller's Fence completes."""

        token = uuid.uuid4().hex if token is None else str(token)
        if not token:
            raise ValueError("execution guard token must not be empty")
        with self._lock:
            self._ensure_open()
            if token in self._execution_guards:
                raise KVLifecycleError(
                    "GPU hot-cache execution guard already exists"
                )
            self._execution_guards.add(token)
        return token

    def end_execution_guard(self, token):
        token = str(token)
        with self._lock:
            if token not in self._execution_guards:
                raise KVLifecycleError(
                    "GPU hot-cache execution guard is not active"
                )
            self._execution_guards.remove(token)
        return token

    def _assert_arena_mutable(self, operation):
        if self._execution_guards:
            raise KVLifecycleError(
                "cannot {} while GPU hot-cache execution is inflight".format(
                    operation
                )
            )

    @property
    def active_execution_guards(self):
        with self._lock:
            return len(self._execution_guards)

    def _touch(self, slot):
        self._clock += 1
        slot.last_access_epoch = self._clock

    def register_page(
        self,
        logical_block_id,
        page_handle,
        data_epoch,
        *,
        cpu_slots=None,
    ):
        """Register a logical page without consuming a GPU slot.

        ``cpu_slots`` may map layer number to an already-valid backing slot.
        Registration of many CPU-backed logical pages therefore does not
        depend on the much smaller GPU hot capacity.
        """

        data_epoch = int(data_epoch)
        descriptor = self.page_pool.descriptor(page_handle)
        if descriptor.data_version != data_epoch:
            raise KVLifecycleError(
                "logical page/data epoch mismatch: expected {}, found {}".format(
                    data_epoch, descriptor.data_version
                )
            )
        key = GPUHotPageKey(
            logical_key=self._logical_key(logical_block_id),
            pool_uuid=page_handle.pool_uuid,
            page_id=page_handle.page_id,
            page_generation=page_handle.generation,
            data_epoch=data_epoch,
        )
        supplied_cpu = {} if cpu_slots is None else dict(cpu_slots)
        invalid_layers = set(supplied_cpu) - set(range(self.layer_count))
        if invalid_layers:
            raise ValueError(
                "cpu_slots contains invalid layers: {}".format(
                    sorted(invalid_layers)
                )
            )
        with self._lock:
            self._ensure_open()
            if key in self._pages:
                raise KVLifecycleError("logical page is already registered")
            for existing in self._pages.values():
                if existing.key.logical_key == key.logical_key:
                    raise KVLifecycleError(
                        "logical page key is already registered at another identity"
                    )
            locations = {}
            for layer in range(self.layer_count):
                cpu_slot = supplied_cpu.get(layer)
                locations[layer] = KVLocationSet(
                    page_key=key,
                    layer=layer,
                    cpu_slot=cpu_slot,
                    cpu_state=(
                        ResidencyState.RESIDENT
                        if cpu_slot is not None
                        else ResidencyState.ABSENT
                    ),
                    authoritative_tier=(
                        "cpu" if cpu_slot is not None else None
                    ),
                )
            self._pages[key] = _PageLocationRecord(
                key=key,
                page_handle=page_handle,
                locations=locations,
            )
            cpu_resident = len(supplied_cpu) == self.layer_count
            self.device_page_table.publish_page(
                page_handle.page_id,
                generation=page_handle.generation,
                data_epoch=data_epoch,
                valid_tokens=descriptor.valid_tokens,
                physical_gpu_slot=-1,
                location_state=(
                    DEVICE_PAGE_CPU if cpu_resident else DEVICE_PAGE_ABSENT
                ),
            )
            return key

    def _record_identity(self, key):
        self._ensure_open()
        try:
            record = self._pages[key]
        except KeyError:
            raise KeyError("unknown GPU hot logical page")
        descriptor = self.page_pool.descriptor(record.page_handle)
        if (
            record.page_handle.pool_uuid != key.pool_uuid
            or record.page_handle.page_id != key.page_id
            or record.page_handle.generation != key.page_generation
        ):
            raise KVLifecycleError("logical page generation identity mismatch")
        return record, descriptor

    def _record(self, key, expected_data_epoch=None):
        record, descriptor = self._record_identity(key)
        expected = (
            key.data_epoch
            if expected_data_epoch is None
            else int(expected_data_epoch)
        )
        if expected != key.data_epoch or descriptor.data_version != expected:
            raise KVLifecycleError(
                "GPU hot page data epoch mismatch: key={}, expected={}, current={}".format(
                    key.data_epoch, expected, descriptor.data_version
                )
            )
        return record

    def advance_data_epoch(self, key, new_data_epoch, *, end_operation=None):
        """Publish a newer page epoch while retaining its GPU slot.

        Append writes all layers under the previous committed epoch.  After
        PagePool publishes the new global data epoch, this method atomically
        rekeys the Location Plane, invalidates stale CPU replicas, and may end
        the caller's pending ``append`` marker.  Returned CPU handles remain
        owned by the caller and must be released through the CPU store.
        """

        new_data_epoch = int(new_data_epoch)
        with self._lock:
            self._assert_arena_mutable("remap a GPU hot page epoch")
            record, descriptor = self._record_identity(key)
            if new_data_epoch <= key.data_epoch:
                raise KVLifecycleError("data epoch must advance strictly")
            if descriptor.data_version != new_data_epoch:
                raise KVLifecycleError(
                    "PagePool has not published data epoch {}".format(
                        new_data_epoch
                    )
                )
            handles = {location.gpu_slot for location in record.locations.values()}
            slot = None
            if handles != {None}:
                if len(handles) != 1 or None in handles:
                    raise KVLifecycleError("cross-layer GPU slot is inconsistent")
                handle = next(iter(handles))
                slot = self._slot_record(handle, page_key=key)
                if any(
                    location.gpu_state != ResidencyState.RESIDENT
                    for location in record.locations.values()
                ):
                    raise KVLifecycleError(
                        "cannot advance epoch before every GPU layer is resident"
                    )
            new_key = GPUHotPageKey(
                logical_key=key.logical_key,
                pool_uuid=key.pool_uuid,
                page_id=key.page_id,
                page_generation=key.page_generation,
                data_epoch=new_data_epoch,
            )
            if new_key in self._pages:
                raise KVLifecycleError("new data epoch is already registered")
            operation_to_end = None
            if slot is not None and end_operation is not None:
                operation_to_end = str(end_operation)
                if operation_to_end not in slot.pending_operations:
                    raise KVLifecycleError(
                        "GPU hot operation is not pending"
                    )
            stale_cpu_slots = []
            for location in record.locations.values():
                if location.cpu_slot is not None:
                    stale_cpu_slots.append(location.cpu_slot)
                location.page_key = new_key
                location.cpu_slot = None
                location.cpu_state = ResidencyState.ABSENT
                if location.gpu_state == ResidencyState.RESIDENT:
                    location.authoritative_tier = "gpu"
                else:
                    location.authoritative_tier = None
            if slot is not None:
                if operation_to_end is not None:
                    slot.pending_operations.remove(operation_to_end)
                slot.page_key = new_key
                slot.state = ResidencyState.RESIDENT
                self._touch(slot)
            del self._pages[key]
            record.key = new_key
            self._pages[new_key] = record
            self.device_page_table.publish_page(
                new_key.page_id,
                generation=new_key.page_generation,
                data_epoch=new_key.data_epoch,
                valid_tokens=descriptor.valid_tokens,
                physical_gpu_slot=(-1 if slot is None else handle.slot_id),
                location_state=(
                    DEVICE_PAGE_ABSENT if slot is None else DEVICE_PAGE_GPU
                ),
            )
            return new_key, tuple(stale_cpu_slots)

    def location_set(self, key, layer, *, expected_data_epoch=None):
        layer = int(layer)
        with self._lock:
            record = self._record(key, expected_data_epoch)
            try:
                return record.locations[layer]
            except KeyError:
                raise IndexError("layer is outside the GPU hot cache")

    def location_sets(self, key, *, expected_data_epoch=None):
        with self._lock:
            record = self._record(key, expected_data_epoch)
            return tuple(record.locations[layer] for layer in range(self.layer_count))

    def _slot_record(self, handle, *, page_key=None):
        if not isinstance(handle, GPUHotSlotHandle):
            raise TypeError("expected GPUHotSlotHandle")
        if handle.cache_uuid != self.cache_uuid:
            raise KVLifecycleError("GPU slot belongs to another cache")
        if handle.slot_id < 0 or handle.slot_id >= self.gpu_capacity_pages:
            raise KVLifecycleError("GPU slot is outside this cache")
        slot = self._slots[handle.slot_id]
        if (
            slot.generation != handle.generation
            or slot.state == ResidencyState.ABSENT
        ):
            raise KVLifecycleError("GPU slot handle is stale or inactive")
        if page_key is not None and slot.page_key != page_key:
            raise KVLifecycleError("GPU slot is assigned to another logical page")
        return slot

    def reserve_gpu(self, key, *, expected_data_epoch=None):
        """Reserve one complete cross-layer slot as a non-readable target."""

        with self._lock:
            self._assert_arena_mutable("assign a GPU hot slot")
            record = self._record(key, expected_data_epoch)
            existing = {location.gpu_slot for location in record.locations.values()}
            if existing != {None}:
                raise KVLifecycleError(
                    "logical page already has a current GPU slot"
                )
            if not self._free_slot_ids:
                raise KVCapacityError(
                    "GPU hot cache exhausted: capacity_pages={}, used_pages={}".format(
                        self.gpu_capacity_pages, self._used_pages
                    )
                )
            slot_id = self._free_slot_ids.pop()
            slot = self._slots[slot_id]
            handle = GPUHotSlotHandle(
                cache_uuid=self.cache_uuid,
                slot_id=slot_id,
                generation=slot.generation,
            )
            slot.state = ResidencyState.LOADING
            slot.page_key = key
            for location in record.locations.values():
                location.gpu_slot = handle
                location.gpu_state = ResidencyState.LOADING
            self._used_pages += 1
            self._peak_used_pages = max(
                self._peak_used_pages, self._used_pages
            )
            self._touch(slot)
            return handle

    def layer_write_target(
        self, key, layer, slot_handle, *, expected_data_epoch=None
    ):
        """Return a preallocated layer target while it remains non-resident."""

        layer = int(layer)
        with self._lock:
            record = self._record(key, expected_data_epoch)
            slot = self._slot_record(slot_handle, page_key=key)
            location = record.locations.get(layer)
            if location is None:
                raise IndexError("layer is outside the GPU hot cache")
            if (
                location.gpu_slot != slot_handle
                or location.gpu_state != ResidencyState.LOADING
                or slot.state != ResidencyState.LOADING
            ):
                raise KVLifecycleError("GPU layer target is not loading")
            return (
                self.keys[layer, slot_handle.slot_id],
                self.values[layer, slot_handle.slot_id],
            )

    def commit_gpu(
        self,
        key,
        slot_handle,
        *,
        expected_data_epoch=None,
        make_authoritative=True,
    ):
        """Atomically publish every layer in a completed GPU page write."""

        with self._lock:
            record = self._record(key, expected_data_epoch)
            slot = self._slot_record(slot_handle, page_key=key)
            if slot.state not in {
                ResidencyState.LOADING,
                ResidencyState.RESIDENT,
            }:
                raise KVLifecycleError("GPU slot is not a loading target")
            for location in record.locations.values():
                if (
                    location.gpu_slot != slot_handle
                    or location.gpu_state
                    not in {ResidencyState.LOADING, ResidencyState.RESIDENT}
                ):
                    raise KVLifecycleError(
                        "cross-layer GPU slot publication is inconsistent"
                    )
            for location in record.locations.values():
                location.gpu_state = ResidencyState.RESIDENT
                if make_authoritative or location.authoritative_tier is None:
                    location.authoritative_tier = "gpu"
            slot.state = ResidencyState.RESIDENT
            self._touch(slot)
            self.device_page_table.publish_gpu_slot(
                key.page_id,
                slot_handle.slot_id,
                has_cpu_replica=all(
                    item.cpu_state == ResidencyState.RESIDENT
                    for item in record.locations.values()
                ),
            )
            return tuple(record.locations.values())

    def commit_gpu_layer(
        self,
        key,
        layer,
        slot_handle,
        *,
        expected_data_epoch=None,
        make_authoritative=True,
    ):
        """Publish one layer while keeping an incomplete cross-layer slot non-LRU."""

        layer = int(layer)
        with self._lock:
            record = self._record(key, expected_data_epoch)
            slot = self._slot_record(slot_handle, page_key=key)
            location = record.locations.get(layer)
            if location is None:
                raise IndexError("layer is outside the GPU hot cache")
            if (
                location.gpu_slot != slot_handle
                or location.gpu_state != ResidencyState.LOADING
                or slot.state != ResidencyState.LOADING
            ):
                raise KVLifecycleError("GPU layer target is not loading")
            location.gpu_state = ResidencyState.RESIDENT
            if make_authoritative or location.authoritative_tier is None:
                location.authoritative_tier = "gpu"
            if all(
                item.gpu_state == ResidencyState.RESIDENT
                for item in record.locations.values()
            ):
                slot.state = ResidencyState.RESIDENT
            # The slot identity is page-wide even while layer publication is
            # incremental.  A layer consumer must still obey that layer's
            # Append Fence before reading it.
            self.device_page_table.publish_gpu_slot(
                key.page_id,
                slot_handle.slot_id,
                has_cpu_replica=all(
                    item.cpu_state == ResidencyState.RESIDENT
                    for item in record.locations.values()
                ),
            )
            self._touch(slot)
            return location

    def resolve_gpu_layer(
        self, key, layer, *, expected_data_epoch=None, touch=True
    ):
        """Resolve one resident layer without deriving slot from page_id."""

        layer = int(layer)
        with self._lock:
            record = self._record(key, expected_data_epoch)
            location = record.locations.get(layer)
            if location is None:
                raise IndexError("layer is outside the GPU hot cache")
            if (
                location.gpu_slot is None
                or location.gpu_state != ResidencyState.RESIDENT
            ):
                raise KVLifecycleError("logical layer page is not GPU resident")
            slot = self._slot_record(location.gpu_slot, page_key=key)
            if slot.state not in {
                ResidencyState.LOADING,
                ResidencyState.RESIDENT,
            }:
                raise KVLifecycleError("GPU hot slot is not readable")
            if touch:
                self._touch(slot)
            return GPUHotLayerPage(
                page_key=key,
                layer=layer,
                slot=location.gpu_slot,
                key=self.keys[layer, location.gpu_slot.slot_id],
                value=self.values[layer, location.gpu_slot.slot_id],
                authoritative_tier=location.authoritative_tier,
            )

    def attach_cpu(
        self,
        key,
        layer,
        cpu_slot,
        *,
        expected_data_epoch=None,
        make_authoritative=False,
    ):
        if cpu_slot is None:
            raise ValueError("cpu_slot must not be None")
        layer = int(layer)
        with self._lock:
            location = self.location_set(
                key, layer, expected_data_epoch=expected_data_epoch
            )
            if location.cpu_slot is not None and location.cpu_slot != cpu_slot:
                raise KVLifecycleError("logical layer page already has a CPU slot")
            location.cpu_slot = cpu_slot
            location.cpu_state = ResidencyState.RESIDENT
            if make_authoritative or location.authoritative_tier is None:
                location.authoritative_tier = "cpu"
            return location

    def attach_cpu_page(
        self,
        key,
        cpu_slots,
        *,
        expected_data_epoch=None,
        make_authoritative=False,
    ):
        """Atomically publish a complete cross-layer CPU page replica."""

        supplied = dict(cpu_slots)
        expected_layers = set(range(self.layer_count))
        if set(supplied) != expected_layers:
            raise ValueError(
                "cpu_slots must contain every layer exactly once"
            )
        if any(slot is None for slot in supplied.values()):
            raise ValueError("CPU page slots must not be None")
        with self._lock:
            record = self._record(key, expected_data_epoch)
            for layer, cpu_slot in supplied.items():
                location = record.locations[layer]
                if (
                    location.cpu_slot is not None
                    and location.cpu_slot != cpu_slot
                ):
                    raise KVLifecycleError(
                        "logical layer page already has a CPU slot"
                    )
            # Publication happens only after validation of the entire page.
            for layer, cpu_slot in supplied.items():
                location = record.locations[layer]
                location.cpu_slot = cpu_slot
                location.cpu_state = ResidencyState.RESIDENT
                if make_authoritative or location.authoritative_tier is None:
                    location.authoritative_tier = "cpu"
            gpu_resident = all(
                item.gpu_state == ResidencyState.RESIDENT
                for item in record.locations.values()
            )
            self.device_page_table.publish_cpu_replica(
                key.page_id, gpu_resident=gpu_resident
            )
            return tuple(
                record.locations[layer] for layer in range(self.layer_count)
            )

    def detach_cpu(self, key, layer, *, expected_data_epoch=None):
        layer = int(layer)
        with self._lock:
            location = self.location_set(
                key, layer, expected_data_epoch=expected_data_epoch
            )
            if location.authoritative_tier == "cpu":
                raise KVLifecycleError("cannot detach authoritative CPU location")
            cpu_slot = location.cpu_slot
            location.cpu_slot = None
            location.cpu_state = ResidencyState.ABSENT
            return cpu_slot

    def _page_busy_reason(self, record, slot):
        descriptor = self.page_pool.descriptor(record.page_handle)
        if slot.pin_count:
            return "slot_pinned"
        if slot.inflight_compute or slot.inflight_io:
            return "slot_inflight"
        if slot.selected_count:
            return "selected"
        if slot.pending_operations:
            return "pending_operation"
        if descriptor.pin_count:
            return "page_pinned"
        if descriptor.inflight_compute or descriptor.inflight_io:
            return "page_inflight"
        if descriptor.state in _BUSY_PAGE_STATES:
            return descriptor.state.value
        return None

    def release_gpu(
        self,
        key,
        *,
        expected_data_epoch=None,
        require_cpu_replica=True,
    ):
        """Release and invalidate a slot for reuse by another logical page."""

        with self._lock:
            record = self._record(key, expected_data_epoch)
            handles = {location.gpu_slot for location in record.locations.values()}
            if len(handles) != 1 or None in handles:
                raise KVLifecycleError("logical page has no single current GPU slot")
            handle = next(iter(handles))
            slot = self._slot_record(handle, page_key=key)
            busy = self._page_busy_reason(record, slot)
            if busy is not None:
                raise KVLifecycleError(
                    "cannot release GPU hot slot: {}".format(busy)
                )
            for location in record.locations.values():
                if location.authoritative_tier == "gpu":
                    if (
                        location.cpu_slot is None
                        or location.cpu_state != ResidencyState.RESIDENT
                    ):
                        if require_cpu_replica:
                            raise KVLifecycleError(
                                "cannot release authoritative GPU page without "
                                "a resident CPU replica for every layer"
                            )
                        location.authoritative_tier = None
                    else:
                        location.authoritative_tier = "cpu"
            self._release_slot(record, handle)
            return handle

    def _release_slot(self, record, handle):
        self._assert_arena_mutable("release a GPU hot slot")
        slot = self._slot_record(handle, page_key=record.key)
        for location in record.locations.values():
            location.gpu_slot = None
            location.gpu_state = ResidencyState.ABSENT
        slot.generation += 1
        slot.state = ResidencyState.ABSENT
        slot.page_key = None
        slot.pin_count = 0
        slot.inflight_compute = 0
        slot.inflight_io = 0
        slot.selected_count = 0
        slot.pending_operations.clear()
        slot.last_access_epoch = 0
        self._free_slot_ids.append(handle.slot_id)
        self._used_pages -= 1
        self.device_page_table.drop_gpu_slot(
            record.key.page_id,
            has_cpu_replica=all(
                item.cpu_state == ResidencyState.RESIDENT
                for item in record.locations.values()
            ),
        )

    def pin_slot(
        self,
        key,
        *,
        kind="compute",
        layer=None,
        expected_data_epoch=None,
    ):
        if kind not in {"compute", "io"}:
            raise ValueError("pin kind must be compute or io")
        with self._lock:
            record = self._record(key, expected_data_epoch)
            handle = next(
                iter({location.gpu_slot for location in record.locations.values()})
            )
            slot = self._slot_record(handle, page_key=key)
            if slot.state != ResidencyState.RESIDENT and kind == "compute":
                layer = None if layer is None else int(layer)
                location = record.locations.get(layer)
                if (
                    location is None
                    or location.gpu_state != ResidencyState.RESIDENT
                ):
                    raise KVLifecycleError(
                        "compute cannot pin a non-resident GPU layer"
                    )
            slot.pin_count += 1
            if kind == "compute":
                slot.inflight_compute += 1
            else:
                slot.inflight_io += 1
            self._touch(slot)
        return handle

    def unpin_slot(
        self,
        key,
        *,
        kind="compute",
        expected_data_epoch=None,
        validate_epoch=True,
    ):
        if kind not in {"compute", "io"}:
            raise ValueError("pin kind must be compute or io")
        with self._lock:
            record = (
                self._record(key, expected_data_epoch)
                if validate_epoch
                else self._record_identity(key)[0]
            )
            handle = next(
                iter({location.gpu_slot for location in record.locations.values()})
            )
            slot = self._slot_record(handle, page_key=key)
            counter = (
                slot.inflight_compute if kind == "compute" else slot.inflight_io
            )
            if slot.pin_count <= 0 or counter <= 0:
                raise KVLifecycleError("GPU hot slot pin accounting underflow")
            slot.pin_count -= 1
            if kind == "compute":
                slot.inflight_compute -= 1
            else:
                slot.inflight_io -= 1

    @contextmanager
    def pinned_slot(self, key, *, kind="compute", expected_data_epoch=None):
        self.pin_slot(key, kind=kind, expected_data_epoch=expected_data_epoch)
        try:
            yield
        finally:
            self.unpin_slot(
                key,
                kind=kind,
                expected_data_epoch=expected_data_epoch,
                validate_epoch=False,
            )

    def mark_selected(self, key, *, expected_data_epoch=None):
        with self._lock:
            record = self._record(key, expected_data_epoch)
            handle = next(
                iter({location.gpu_slot for location in record.locations.values()})
            )
            slot = self._slot_record(handle, page_key=key)
            slot.selected_count += 1

    def unmark_selected(self, key, *, expected_data_epoch=None):
        with self._lock:
            record = self._record(key, expected_data_epoch)
            handle = next(
                iter({location.gpu_slot for location in record.locations.values()})
            )
            slot = self._slot_record(handle, page_key=key)
            if slot.selected_count <= 0:
                raise KVLifecycleError("GPU hot slot selected count underflow")
            slot.selected_count -= 1

    def begin_pending(self, key, operation, *, expected_data_epoch=None):
        operation = str(operation)
        if not operation:
            raise ValueError("operation must not be empty")
        with self._lock:
            record = self._record(key, expected_data_epoch)
            handle = next(
                iter({location.gpu_slot for location in record.locations.values()})
            )
            slot = self._slot_record(handle, page_key=key)
            if operation in slot.pending_operations:
                raise KVLifecycleError("GPU hot operation is already pending")
            slot.pending_operations.add(operation)

    def end_pending(self, key, operation, *, expected_data_epoch=None):
        operation = str(operation)
        with self._lock:
            record = self._record(key, expected_data_epoch)
            handle = next(
                iter({location.gpu_slot for location in record.locations.values()})
            )
            slot = self._slot_record(handle, page_key=key)
            if operation not in slot.pending_operations:
                raise KVLifecycleError("GPU hot operation is not pending")
            slot.pending_operations.remove(operation)

    def cleanup_operation(self, key, operation=None, *, kind=None):
        """Unwind local operation accounting after an epoch mismatch.

        Generation and cache/slot identity are still validated.  Only the
        PagePool data-epoch comparison is skipped so failed asynchronous work
        cannot strand a pin or pending marker after the epoch changed.
        """

        if kind not in {None, "compute", "io"}:
            raise ValueError("pin kind must be compute, io, or None")
        with self._lock:
            record = self._record_identity(key)[0]
            handles = {location.gpu_slot for location in record.locations.values()}
            if len(handles) != 1 or None in handles:
                raise KVLifecycleError("logical page has no single current GPU slot")
            handle = next(iter(handles))
            slot = self._slot_record(handle, page_key=key)
            if operation is not None:
                operation = str(operation)
                if operation not in slot.pending_operations:
                    raise KVLifecycleError("GPU hot operation is not pending")
                slot.pending_operations.remove(operation)
            if kind is not None:
                counter = (
                    slot.inflight_compute
                    if kind == "compute"
                    else slot.inflight_io
                )
                if slot.pin_count <= 0 or counter <= 0:
                    raise KVLifecycleError(
                        "GPU hot slot pin accounting underflow"
                    )
                slot.pin_count -= 1
                if kind == "compute":
                    slot.inflight_compute -= 1
                else:
                    slot.inflight_io -= 1
            return handle

    def abort_gpu_load(self, key, slot_handle, *, operation=None, kind=None):
        """Discard a non-resident target with cleanup-safe identity checks."""

        with self._lock:
            record = self._record_identity(key)[0]
            slot = self._slot_record(slot_handle, page_key=key)
            if slot.state != ResidencyState.LOADING:
                raise KVLifecycleError("only a loading GPU slot can be aborted")
            if operation is not None:
                operation = str(operation)
                if operation not in slot.pending_operations:
                    raise KVLifecycleError("GPU hot operation is not pending")
                slot.pending_operations.remove(operation)
            if kind is not None:
                if kind not in {"compute", "io"}:
                    raise ValueError("pin kind must be compute or io")
                counter = (
                    slot.inflight_compute
                    if kind == "compute"
                    else slot.inflight_io
                )
                if slot.pin_count <= 0 or counter <= 0:
                    raise KVLifecycleError(
                        "GPU hot slot pin accounting underflow"
                    )
                slot.pin_count -= 1
                if kind == "compute":
                    slot.inflight_compute -= 1
                else:
                    slot.inflight_io -= 1
            if (
                slot.pin_count
                or slot.inflight_compute
                or slot.inflight_io
                or slot.pending_operations
                or slot.selected_count
            ):
                raise KVLifecycleError(
                    "cannot abort GPU load before local work quiesces"
                )
            for location in record.locations.values():
                if location.authoritative_tier == "gpu":
                    location.authoritative_tier = (
                        "cpu"
                        if location.cpu_slot is not None
                        and location.cpu_state == ResidencyState.RESIDENT
                        else None
                    )
            self._release_slot(record, slot_handle)
            return slot_handle

    def lru_victims(self, *, exclude_keys=(), limit=None):
        """Enumerate legal GPU-resident victims in oldest-first order."""

        excluded = set(exclude_keys)
        if limit is not None and int(limit) < 0:
            raise ValueError("limit must be non-negative")
        with self._lock:
            self._ensure_open()
            if self._execution_guards:
                return ()
            candidates = []
            for slot in self._slots:
                if (
                    slot.state != ResidencyState.RESIDENT
                    or slot.page_key in excluded
                ):
                    continue
                record = self._record(slot.page_key)
                if self._page_busy_reason(record, slot) is None:
                    candidates.append((slot.last_access_epoch, slot.page_key))
            candidates.sort(key=lambda item: (item[0], repr(item[1])))
            keys = tuple(item[1] for item in candidates)
            return keys if limit is None else keys[: int(limit)]

    def unregister_page(
        self,
        key,
        *,
        expected_data_epoch=None,
        validate_epoch=True,
    ):
        """Forget all locations and return opaque CPU slots to their owner.

        CPU slots are not released here because this cache does not own the CPU
        store.  The caller receives them for explicit store cleanup.
        """

        with self._lock:
            record = (
                self._record(key, expected_data_epoch)
                if validate_epoch
                else self._record_identity(key)[0]
            )
            handles = {location.gpu_slot for location in record.locations.values()}
            if handles != {None}:
                if len(handles) != 1 or None in handles:
                    raise KVLifecycleError("cross-layer GPU slot is inconsistent")
                handle = next(iter(handles))
                slot = self._slot_record(handle, page_key=key)
                busy = self._page_busy_reason(record, slot)
                if busy is not None:
                    raise KVLifecycleError(
                        "cannot unregister GPU hot page: {}".format(busy)
                    )
                self._release_slot(record, handle)
            cpu_slots = tuple(
                record.locations[layer].cpu_slot
                for layer in range(self.layer_count)
                if record.locations[layer].cpu_slot is not None
            )
            del self._pages[key]
            self.device_page_table.clear_page(key.page_id)
            return cpu_slots

    def capability(self):
        return KVStoreCapability(
            store_id=self.store_id,
            tiers=("gpu", "cpu"),
            dtypes=("bf16", "fp16", "fp32"),
            layouts=("hnd",),
            supports_active_attention=True,
            supports_async_copy=True,
            implemented=True,
        )

    def layer_view(self, layer):
        """Return the raw hot-slot arena for a layer.

        Indices into this view are GPU hot slot IDs.  Callers must resolve a
        ``GPUHotSlotHandle`` first; PageHandle.page_id is not a valid index.
        """

        with self._lock:
            self._ensure_open()
            layer = int(layer)
            if layer < 0 or layer >= self.layer_count:
                raise IndexError("layer is outside the GPU hot cache")
            return self.keys[layer], self.values[layer]

    def _slot_indices(self, slot_ids):
        indices = torch.as_tensor(
            tuple(int(item) for item in slot_ids),
            dtype=torch.long,
            device=self.device,
        )
        if indices.ndim != 1:
            raise ValueError("slot IDs must be one-dimensional")
        if indices.numel() and (
            int(indices.min().item()) < 0
            or int(indices.max().item()) >= self.gpu_capacity_pages
        ):
            raise IndexError("slot ID is outside the GPU hot cache")
        return indices

    def read_pages(self, layer, slot_ids, stream=None):
        """GPUKVStore-compatible read using resolved hot slot IDs."""

        key_pool, value_pool = self.layer_view(layer)
        indices = self._slot_indices(slot_ids)
        if stream is None:
            return (
                torch.index_select(key_pool, 0, indices),
                torch.index_select(value_pool, 0, indices),
            )
        with torch.cuda.stream(stream):
            return (
                torch.index_select(key_pool, 0, indices),
                torch.index_select(value_pool, 0, indices),
            )

    def write_pages(self, layer, slot_ids, key, value, stream=None):
        """GPUKVStore-compatible write using reserved hot slot IDs."""

        key_pool, value_pool = self.layer_view(layer)
        indices = self._slot_indices(slot_ids)
        expected = (indices.numel(),) + tuple(key_pool.shape[1:])
        if tuple(key.shape) != expected or tuple(value.shape) != expected:
            raise ValueError("page payload shape does not match the GPU hot cache")
        if key.dtype != self.dtype or value.dtype != self.dtype:
            raise ValueError("page payload dtype does not match the GPU hot cache")
        if key.device != self.device or value.device != self.device:
            raise ValueError("page payload device does not match the GPU hot cache")
        if stream is None:
            key_pool.index_copy_(0, indices, key)
            value_pool.index_copy_(0, indices, value)
        else:
            with torch.cuda.stream(stream):
                key_pool.index_copy_(0, indices, key)
                value_pool.index_copy_(0, indices, value)

    def copy_page(self, source_slot_id, target_slot_id, valid_tokens, stream=None):
        """Copy raw hot slots; Location publication remains caller-owned."""

        source_slot_id = int(source_slot_id)
        target_slot_id = int(target_slot_id)
        valid_tokens = int(valid_tokens)
        if valid_tokens < 0 or valid_tokens > self.page_size:
            raise ValueError("valid_tokens is outside the page")
        for slot_id in (source_slot_id, target_slot_id):
            if slot_id < 0 or slot_id >= self.gpu_capacity_pages:
                raise IndexError("slot ID is outside the GPU hot cache")
        if stream is None:
            self.keys[:, target_slot_id, :, :valid_tokens, :].copy_(
                self.keys[:, source_slot_id, :, :valid_tokens, :]
            )
            self.values[:, target_slot_id, :, :valid_tokens, :].copy_(
                self.values[:, source_slot_id, :, :valid_tokens, :]
            )
        else:
            with torch.cuda.stream(stream):
                self.keys[:, target_slot_id, :, :valid_tokens, :].copy_(
                    self.keys[:, source_slot_id, :, :valid_tokens, :],
                    non_blocking=True,
                )
                self.values[:, target_slot_id, :, :valid_tokens, :].copy_(
                    self.values[:, source_slot_id, :, :valid_tokens, :],
                    non_blocking=True,
                )

    @property
    def used_pages(self):
        with self._lock:
            return self._used_pages

    @property
    def free_pages(self):
        with self._lock:
            return self.gpu_capacity_pages - self._used_pages

    @property
    def used_bytes(self):
        return self.used_pages * self.gpu_page_bytes

    @property
    def free_bytes(self):
        return self.free_pages * self.gpu_page_bytes

    @property
    def nbytes(self):
        return 0 if self._closed else self.gpu_capacity_bytes

    @property
    def above_high_watermark(self):
        return self.used_pages >= self.high_watermark_pages

    @property
    def below_low_watermark(self):
        return self.used_pages <= self.low_watermark_pages

    def stats(self):
        with self._lock:
            used_bytes = self._used_pages * self.gpu_page_bytes
            return {
                "gpu_kv_capacity_pages": self.gpu_capacity_pages,
                "gpu_kv_used_pages": self._used_pages,
                "gpu_kv_free_pages": self.gpu_capacity_pages
                - self._used_pages,
                "gpu_kv_capacity_bytes": self.gpu_capacity_bytes,
                "gpu_kv_used_bytes": used_bytes,
                "gpu_kv_free_bytes": self.gpu_capacity_bytes - used_bytes,
                "gpu_kv_high_watermark_pages": self.high_watermark_pages,
                "gpu_kv_low_watermark_pages": self.low_watermark_pages,
                "gpu_kv_high_watermark_bytes": self.high_watermark_bytes,
                "gpu_kv_low_watermark_bytes": self.low_watermark_bytes,
                "gpu_kv_above_high_watermark": self._used_pages
                >= self.high_watermark_pages,
                "gpu_kv_below_low_watermark": self._used_pages
                <= self.low_watermark_pages,
                "gpu_kv_peak_used_pages": self._peak_used_pages,
                "gpu_kv_peak_used_bytes": self._peak_used_pages
                * self.gpu_page_bytes,
                "registered_logical_pages": len(self._pages),
                "active_hot_cache_execution_guards": len(
                    self._execution_guards
                ),
            }

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._execution_guards:
                raise KVLifecycleError(
                    "cannot close GPU hot cache with active execution guards"
                )
            busy = []
            for slot_id, slot in enumerate(self._slots):
                if (
                    slot.pin_count
                    or slot.inflight_compute
                    or slot.inflight_io
                    or slot.selected_count
                    or slot.pending_operations
                ):
                    busy.append(slot_id)
            if busy:
                raise KVLifecycleError(
                    "cannot close GPU hot cache with busy slots: {}".format(busy)
                )
            for slot in self._slots:
                slot.generation += 1
                slot.state = ResidencyState.ABSENT
                slot.page_key = None
                slot.last_access_epoch = 0
            self._pages.clear()
            self._free_slot_ids.clear()
            self._used_pages = 0
            self.keys = None
            self.values = None
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
