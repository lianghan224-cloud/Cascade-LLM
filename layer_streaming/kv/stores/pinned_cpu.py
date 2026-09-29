"""Byte-bounded, preallocated pinned-CPU storage for KV layer pages.

This module implements only the storage half of the KV location plane.  It
does not own logical pages, publish replicas, or choose an authoritative
location.  A slot is deliberately not a ``PageHandle``: it is a private,
generation-checked handle into this store's fixed pinned-memory pool.
"""

from dataclasses import dataclass
import hashlib
import threading
import uuid

import torch

from ..errors import KVCapacityError, KVLifecycleError
from .base import KVStore, KVStoreCapability


@dataclass(frozen=True)
class PinnedCPUReservation:
    """One byte-accounted reservation for one layer-page slot."""

    store_uuid: str
    reservation_id: int
    slot_id: int
    generation: int
    nbytes: int

    def as_dict(self):
        return {
            "store_uuid": self.store_uuid,
            "reservation_id": self.reservation_id,
            "slot_id": self.slot_id,
            "generation": self.generation,
            "nbytes": self.nbytes,
        }


@dataclass(frozen=True)
class PinnedCPUSlotHandle:
    """Generation-bearing handle for an allocated pinned layer page."""

    store_uuid: str
    slot_id: int
    generation: int

    def as_dict(self):
        return {
            "store_uuid": self.store_uuid,
            "slot_id": self.slot_id,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class PinnedCPUPage:
    """Validated read view and metadata for one resident layer page."""

    handle: PinnedCPUSlotHandle
    logical_block_id: object
    layer: int
    key: torch.Tensor
    value: torch.Tensor
    dtype: torch.dtype
    layout: str
    valid_tokens: int
    data_epoch: int
    checksum: object = None


@dataclass
class _SlotRecord:
    generation: int = 0
    state: str = "free"
    reservation_id: object = None
    logical_block_id: object = None
    layer: object = None
    data_epoch: object = None
    valid_tokens: int = 0
    checksum: object = None
    initialized: bool = False


class PinnedCPUKVStore(KVStore):
    """A fixed-size pinned-memory pool whose allocation unit is a layer page.

    ``tensor_factory`` is an explicit test seam.  Production construction
    always asks PyTorch for pinned CPU memory and propagates allocation errors;
    it never falls back to pageable memory.  Tests may inject a factory that
    returns pageable CPU tensors to exercise lifecycle logic without CUDA.
    """

    store_id = "pinned_cpu"
    layout_name = "hnd"

    def __init__(
        self,
        capacity_bytes,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=torch.bfloat16,
        layout="hnd",
        high_watermark_bytes=None,
        low_watermark_bytes=None,
        checksum_enabled=False,
        tensor_factory=None,
    ):
        requested_capacity = self._positive_int(
            capacity_bytes, "capacity_bytes"
        )
        self.num_kv_heads = self._positive_int(
            num_kv_heads, "num_kv_heads"
        )
        self.page_size = self._positive_int(page_size, "page_size")
        self.head_dim = self._positive_int(head_dim, "head_dim")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        self.dtype = dtype
        self.layout = str(layout).lower()
        if self.layout != self.layout_name:
            raise ValueError(
                "Pinned CPU KV store only supports the HND layout"
            )
        self.checksum_enabled = bool(checksum_enabled)

        element_size = torch.empty((), dtype=dtype).element_size()
        self.layer_page_bytes = (
            2
            * self.num_kv_heads
            * self.page_size
            * self.head_dim
            * element_size
        )
        self.slot_count = requested_capacity // self.layer_page_bytes
        if self.slot_count == 0:
            raise KVCapacityError(
                "capacity_bytes={} cannot hold one {}-byte KV layer page".format(
                    requested_capacity, self.layer_page_bytes
                )
            )
        # Capacity is the physically backed, allocatable pool size.  Any tail
        # smaller than one layer page is intentionally not advertised as free.
        self.requested_capacity_bytes = requested_capacity
        self.capacity_bytes = self.slot_count * self.layer_page_bytes
        self.high_watermark_bytes = self._watermark(
            high_watermark_bytes,
            default=int(self.capacity_bytes * 0.9),
            name="high_watermark_bytes",
        )
        self.low_watermark_bytes = self._watermark(
            low_watermark_bytes,
            default=int(self.capacity_bytes * 0.7),
            name="low_watermark_bytes",
        )
        if self.low_watermark_bytes > self.high_watermark_bytes:
            raise ValueError(
                "low_watermark_bytes must not exceed high_watermark_bytes"
            )

        self.store_uuid = str(uuid.uuid4())
        self._lock = threading.RLock()
        self._next_reservation_id = 1
        self._reservations = {}
        self._slots = [_SlotRecord() for _ in range(self.slot_count)]
        # Pop from the end to allocate deterministic ascending slot IDs.
        self._free_slot_ids = list(reversed(range(self.slot_count)))
        self._used_bytes = 0
        self._reserved_bytes = 0
        self._closed = False

        shape = (
            self.slot_count,
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
        )
        factory = tensor_factory or torch.empty
        self.keys = self._allocate_pool_tensor(factory, shape)
        try:
            self.values = self._allocate_pool_tensor(factory, shape)
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
        if value < 0 or value > self.capacity_bytes:
            raise ValueError(
                "{} must be within [0, capacity_bytes]".format(name)
            )
        return value

    def _allocate_pool_tensor(self, factory, shape):
        tensor = factory(
            shape,
            dtype=self.dtype,
            device=torch.device("cpu"),
            pin_memory=True,
        )
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor_factory must return a torch.Tensor")
        if tuple(tensor.shape) != shape:
            raise ValueError("tensor_factory returned an invalid pool shape")
        if tensor.dtype != self.dtype or tensor.device.type != "cpu":
            raise ValueError("tensor_factory returned an invalid dtype or device")
        if not tensor.is_contiguous():
            raise ValueError("pinned CPU KV pool must be contiguous")
        # The injection is the only permitted pageable-memory test path.
        # Production allocation must prove that pin_memory=True took effect.
        if factory is torch.empty and not tensor.is_pinned():
            raise RuntimeError("PyTorch did not allocate pinned CPU memory")
        return tensor

    def capability(self):
        return KVStoreCapability(
            store_id=self.store_id,
            tiers=("cpu",),
            dtypes=("bf16", "fp16", "fp32", "int8", "uint8"),
            layouts=(self.layout_name,),
            supports_active_attention=False,
            supports_async_copy=False,
            implemented=True,
        )

    def _ensure_open(self):
        if self._closed:
            raise KVLifecycleError("Pinned CPU KV store is closed")

    def _reservation_record(self, reservation):
        self._ensure_open()
        if not isinstance(reservation, PinnedCPUReservation):
            raise TypeError("expected PinnedCPUReservation")
        if reservation.store_uuid != self.store_uuid:
            raise KVLifecycleError("reservation belongs to another store")
        current = self._reservations.get(reservation.reservation_id)
        if current != reservation:
            raise KVLifecycleError("reservation is stale or inactive")
        record = self._slots[reservation.slot_id]
        if (
            record.state != "reserved"
            or record.reservation_id != reservation.reservation_id
            or record.generation != reservation.generation
        ):
            raise KVLifecycleError("reservation does not match its slot")
        return record

    def _slot_record(self, handle, require_initialized=False):
        self._ensure_open()
        if not isinstance(handle, PinnedCPUSlotHandle):
            raise TypeError("expected PinnedCPUSlotHandle")
        if handle.store_uuid != self.store_uuid:
            raise KVLifecycleError("slot handle belongs to another store")
        if handle.slot_id < 0 or handle.slot_id >= self.slot_count:
            raise KVLifecycleError("slot handle is outside this store")
        record = self._slots[handle.slot_id]
        if record.generation != handle.generation or record.state != "allocated":
            raise KVLifecycleError("slot handle is stale or inactive")
        if require_initialized and not record.initialized:
            raise KVLifecycleError("pinned CPU KV page has not been committed")
        return record

    @staticmethod
    def _validate_epoch(record, expected_data_epoch):
        if (
            expected_data_epoch is not None
            and record.data_epoch != int(expected_data_epoch)
        ):
            raise KVLifecycleError(
                "pinned CPU KV data epoch mismatch: expected {}, found {}".format(
                    int(expected_data_epoch), record.data_epoch
                )
            )

    def reserve(self, required_bytes=None):
        """Reserve one layer-page slot without publishing it as allocated."""

        required_bytes = (
            self.layer_page_bytes
            if required_bytes is None
            else int(required_bytes)
        )
        if required_bytes != self.layer_page_bytes:
            raise ValueError(
                "a reservation must equal one layer page ({} bytes)".format(
                    self.layer_page_bytes
                )
            )
        with self._lock:
            self._ensure_open()
            if not self._free_slot_ids:
                raise KVCapacityError(
                    "Pinned CPU KV capacity exhausted: capacity={}, used={}, "
                    "reserved={}, required={}".format(
                        self.capacity_bytes,
                        self._used_bytes,
                        self._reserved_bytes,
                        required_bytes,
                    )
                )
            slot_id = self._free_slot_ids.pop()
            record = self._slots[slot_id]
            reservation_id = self._next_reservation_id
            self._next_reservation_id += 1
            reservation = PinnedCPUReservation(
                store_uuid=self.store_uuid,
                reservation_id=reservation_id,
                slot_id=slot_id,
                generation=record.generation,
                nbytes=self.layer_page_bytes,
            )
            record.state = "reserved"
            record.reservation_id = reservation_id
            self._reservations[reservation_id] = reservation
            self._reserved_bytes += self.layer_page_bytes
            return reservation

    def cancel_reservation(self, reservation):
        """Cancel a live reservation and return its slot to the pool."""

        with self._lock:
            record = self._reservation_record(reservation)
            del self._reservations[reservation.reservation_id]
            self._reserved_bytes -= self.layer_page_bytes
            self._reset_slot(reservation.slot_id, record)

    def allocate(
        self, reservation, *, logical_block_id, layer, data_epoch
    ):
        """Consume a reservation and bind it to one logical layer page."""

        if logical_block_id is None:
            raise ValueError("logical_block_id must not be None")
        layer = int(layer)
        data_epoch = int(data_epoch)
        if layer < 0:
            raise ValueError("layer must be non-negative")
        if data_epoch < 0:
            raise ValueError("data_epoch must be non-negative")
        with self._lock:
            record = self._reservation_record(reservation)
            del self._reservations[reservation.reservation_id]
            self._reserved_bytes -= self.layer_page_bytes
            self._used_bytes += self.layer_page_bytes
            record.state = "allocated"
            record.reservation_id = None
            record.logical_block_id = logical_block_id
            record.layer = layer
            record.data_epoch = data_epoch
            record.valid_tokens = 0
            record.checksum = None
            record.initialized = False
            return PinnedCPUSlotHandle(
                store_uuid=self.store_uuid,
                slot_id=reservation.slot_id,
                generation=record.generation,
            )

    def _validate_payload(self, key, value):
        expected_shape = (
            self.num_kv_heads,
            self.page_size,
            self.head_dim,
        )
        for name, tensor in (("key", key), ("value", value)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("{} must be a torch.Tensor".format(name))
            if tuple(tensor.shape) != expected_shape:
                raise ValueError("{} has an invalid layer-page shape".format(name))
            if tensor.dtype != self.dtype:
                raise ValueError("{} has an invalid dtype".format(name))
            if tensor.device.type != "cpu":
                raise ValueError(
                    "synchronous store.write requires CPU tensors; use "
                    "write_target/commit_write for asynchronous migration"
                )

    def write_target(self, handle, expected_data_epoch=None):
        """Return preallocated target tensors without publishing readable data.

        Migration code may submit an asynchronous D2H copy into these views.
        It must call :meth:`commit_write` only after its common operation Fence
        completes successfully.
        """

        with self._lock:
            record = self._slot_record(handle)
            self._validate_epoch(record, expected_data_epoch)
            if record.initialized:
                raise KVLifecycleError("pinned CPU KV page is already committed")
            return self.keys[handle.slot_id], self.values[handle.slot_id]

    def commit_write(
        self, handle, *, valid_tokens, data_epoch, checksum=None
    ):
        """Publish a completed write after epoch and optional checksum checks."""

        valid_tokens = int(valid_tokens)
        data_epoch = int(data_epoch)
        if valid_tokens < 0 or valid_tokens > self.page_size:
            raise ValueError("valid_tokens is outside the layer page")
        with self._lock:
            record = self._slot_record(handle)
            self._validate_epoch(record, data_epoch)
            if record.initialized:
                raise KVLifecycleError("pinned CPU KV page is already committed")
            stored_checksum = None
            if self.checksum_enabled or checksum is not None:
                stored_checksum = self._page_checksum(
                    handle.slot_id, valid_tokens
                )
                if checksum is not None and checksum != stored_checksum:
                    raise IOError("pinned CPU KV checksum mismatch")
            record.valid_tokens = valid_tokens
            record.checksum = stored_checksum
            record.initialized = True
            return self._page_view(handle, record)

    def write(
        self,
        handle,
        key,
        value,
        *,
        valid_tokens,
        data_epoch,
        checksum=None,
    ):
        """Synchronously copy and commit one CPU layer-page payload."""

        self._validate_payload(key, value)
        target_key, target_value = self.write_target(
            handle, expected_data_epoch=data_epoch
        )
        target_key.copy_(key)
        target_value.copy_(value)
        return self.commit_write(
            handle,
            valid_tokens=valid_tokens,
            data_epoch=data_epoch,
            checksum=checksum,
        )

    def read(self, handle, *, expected_data_epoch=None, verify_checksum=False):
        """Return a zero-copy validated view of a committed layer page."""

        with self._lock:
            record = self._slot_record(handle, require_initialized=True)
            self._validate_epoch(record, expected_data_epoch)
            if verify_checksum:
                if record.checksum is None:
                    raise KVLifecycleError(
                        "checksum verification requested for an unchecked page"
                    )
                actual = self._page_checksum(
                    handle.slot_id, record.valid_tokens
                )
                if actual != record.checksum:
                    raise IOError("pinned CPU KV checksum mismatch")
            return self._page_view(handle, record)

    def _page_view(self, handle, record):
        return PinnedCPUPage(
            handle=handle,
            logical_block_id=record.logical_block_id,
            layer=record.layer,
            key=self.keys[handle.slot_id],
            value=self.values[handle.slot_id],
            dtype=self.dtype,
            layout=self.layout,
            valid_tokens=record.valid_tokens,
            data_epoch=record.data_epoch,
            checksum=record.checksum,
        )

    def _page_checksum(self, slot_id, valid_tokens):
        digest = hashlib.sha256()
        digest.update(str(int(valid_tokens)).encode("ascii"))
        for tensor in (self.keys[slot_id], self.values[slot_id]):
            # A HND slice for one head is contiguous.  Viewing it as bytes
            # supports BF16 without asking NumPy to understand BF16 itself.
            for head in range(self.num_kv_heads):
                byte_view = tensor[head, :valid_tokens, :].view(torch.uint8)
                digest.update(memoryview(byte_view.numpy()))
        return "sha256:" + digest.hexdigest()

    def release(self, handle, *, expected_data_epoch=None):
        """Invalidate an allocated handle and return its slot to the pool."""

        with self._lock:
            record = self._slot_record(handle)
            self._validate_epoch(record, expected_data_epoch)
            self._used_bytes -= self.layer_page_bytes
            self._reset_slot(handle.slot_id, record)

    def _reset_slot(self, slot_id, record):
        record.generation += 1
        record.state = "free"
        record.reservation_id = None
        record.logical_block_id = None
        record.layer = None
        record.data_epoch = None
        record.valid_tokens = 0
        record.checksum = None
        record.initialized = False
        self._free_slot_ids.append(slot_id)

    @property
    def used_bytes(self):
        with self._lock:
            return self._used_bytes

    @property
    def reserved_bytes(self):
        with self._lock:
            return self._reserved_bytes

    @property
    def free_bytes(self):
        with self._lock:
            return self.capacity_bytes - self._used_bytes - self._reserved_bytes

    @property
    def above_high_watermark(self):
        with self._lock:
            return (
                self._used_bytes + self._reserved_bytes
                >= self.high_watermark_bytes
            )

    @property
    def below_low_watermark(self):
        with self._lock:
            return (
                self._used_bytes + self._reserved_bytes
                <= self.low_watermark_bytes
            )

    def stats(self):
        with self._lock:
            occupied = self._used_bytes + self._reserved_bytes
            return {
                "capacity_bytes": self.capacity_bytes,
                "requested_capacity_bytes": self.requested_capacity_bytes,
                "used_bytes": self._used_bytes,
                "reserved_bytes": self._reserved_bytes,
                "free_bytes": self.capacity_bytes - occupied,
                "high_watermark_bytes": self.high_watermark_bytes,
                "low_watermark_bytes": self.low_watermark_bytes,
                "above_high_watermark": occupied
                >= self.high_watermark_bytes,
                "below_low_watermark": occupied
                <= self.low_watermark_bytes,
                "slot_count": self.slot_count,
                "free_slots": len(self._free_slot_ids),
                "layer_page_bytes": self.layer_page_bytes,
            }

    @property
    def nbytes(self):
        return 0 if self._closed else self.capacity_bytes

    def close(self):
        with self._lock:
            if self._closed:
                return
            for record in self._slots:
                record.generation += 1
                record.state = "free"
                record.reservation_id = None
                record.logical_block_id = None
                record.layer = None
                record.data_epoch = None
                record.valid_tokens = 0
                record.checksum = None
                record.initialized = False
            self._reservations.clear()
            self._free_slot_ids.clear()
            self._used_bytes = 0
            self._reserved_bytes = 0
            self.keys = None
            self.values = None
            self._closed = True

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
