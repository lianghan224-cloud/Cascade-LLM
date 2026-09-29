"""Device execution metadata for KV selection and location resolution.

This module deliberately owns no Page reference and never changes lifecycle
state.  PagePool/Ownership and the Location Plane publish snapshots at their
existing commit points; device consumers may then validate and gather those
snapshots without reading scalar IDs back on the host.
"""

from dataclasses import dataclass

import torch


DEVICE_PAGE_ABSENT = 0
DEVICE_PAGE_GPU = 1
DEVICE_PAGE_CPU = 2
DEVICE_PAGE_GPU_CPU = DEVICE_PAGE_GPU | DEVICE_PAGE_CPU

DEVICE_PAGE_OK = 0
DEVICE_PAGE_STALE_EPOCH = 1
DEVICE_PAGE_STALE_GENERATION = 2
DEVICE_PAGE_INVALID_ID = 4
DEVICE_PAGE_STALE_VALID_TOKENS = 8


@dataclass(frozen=True)
class DeviceKVPageMetadata:
    """A tensor-only gather result for one selected logical-page set."""

    logical_page_ids: torch.Tensor
    page_ids: torch.Tensor
    physical_gpu_slots: torch.Tensor
    valid_tokens: torch.Tensor
    data_epochs: torch.Tensor
    generations: torch.Tensor
    location_states: torch.Tensor
    error_states: torch.Tensor

    def __post_init__(self):
        count = int(self.logical_page_ids.numel())
        fields = (
            self.page_ids,
            self.physical_gpu_slots,
            self.valid_tokens,
            self.data_epochs,
            self.generations,
            self.location_states,
            self.error_states,
        )
        if any(item.shape != (count,) for item in fields):
            raise ValueError("device page metadata tensors must align")
        if len({item.device for item in (self.logical_page_ids,) + fields}) != 1:
            raise ValueError("device page metadata tensors must share a device")


class DeviceKVPageTable:
    """Physical-page-indexed execution mirror on one torch device.

    Updates happen only at existing PagePool/Location publication boundaries.
    The table is not an authority: generation and epoch still originate in
    PagePool, while physical GPU slot and residency originate in Location.
    """

    def __init__(self, page_capacity, *, device):
        self.page_capacity = int(page_capacity)
        if self.page_capacity <= 0:
            raise ValueError("device page-table capacity must be positive")
        self.device = torch.device(device)
        shape = (self.page_capacity,)
        self.physical_gpu_slots = torch.full(
            shape, -1, dtype=torch.int32, device=self.device
        )
        self.valid_tokens = torch.zeros(
            shape, dtype=torch.int32, device=self.device
        )
        self.data_epochs = torch.zeros(
            shape, dtype=torch.int64, device=self.device
        )
        self.generations = torch.zeros(
            shape, dtype=torch.int64, device=self.device
        )
        self.location_states = torch.zeros(
            shape, dtype=torch.int32, device=self.device
        )

    @staticmethod
    def _page_id(value):
        value = int(value)
        if value < 0:
            raise IndexError("page ID must not be negative")
        return value

    def _check_page_id(self, page_id):
        page_id = self._page_id(page_id)
        if page_id >= self.page_capacity:
            raise IndexError("page ID is outside the device page table")
        return page_id

    def publish_page(
        self,
        page_id,
        *,
        generation,
        data_epoch,
        valid_tokens,
        physical_gpu_slot=-1,
        location_state=DEVICE_PAGE_ABSENT,
    ):
        """Enqueue one authoritative metadata-row publication."""

        page_id = self._check_page_id(page_id)
        generation = int(generation)
        data_epoch = int(data_epoch)
        valid_tokens = int(valid_tokens)
        physical_gpu_slot = int(physical_gpu_slot)
        location_state = int(location_state)
        if generation <= 0:
            raise ValueError("published page generation must be positive")
        # Epoch/token zero represents a transaction-local newly allocated
        # page.  It is never a committed readable epoch and must be paired
        # with a staged-tail Append Fence by the caller.
        if data_epoch < 0:
            raise ValueError("published data epoch must not be negative")
        if valid_tokens < 0:
            raise ValueError("published valid token count must not be negative")
        self.generations[page_id] = generation
        self.data_epochs[page_id] = data_epoch
        self.valid_tokens[page_id] = valid_tokens
        self.physical_gpu_slots[page_id] = physical_gpu_slot
        self.location_states[page_id] = location_state

    def publish_gpu_slot(self, page_id, slot_id, *, has_cpu_replica):
        page_id = self._check_page_id(page_id)
        slot_id = int(slot_id)
        if slot_id < 0:
            raise ValueError("resident GPU slot must not be negative")
        self.physical_gpu_slots[page_id] = slot_id
        self.location_states[page_id] = (
            DEVICE_PAGE_GPU_CPU if bool(has_cpu_replica) else DEVICE_PAGE_GPU
        )

    def drop_gpu_slot(self, page_id, *, has_cpu_replica):
        page_id = self._check_page_id(page_id)
        self.physical_gpu_slots[page_id] = -1
        self.location_states[page_id] = (
            DEVICE_PAGE_CPU if bool(has_cpu_replica) else DEVICE_PAGE_ABSENT
        )

    def publish_cpu_replica(self, page_id, *, gpu_resident):
        page_id = self._check_page_id(page_id)
        self.location_states[page_id] = (
            DEVICE_PAGE_GPU_CPU if bool(gpu_resident) else DEVICE_PAGE_CPU
        )

    def clear_page(self, page_id):
        page_id = self._check_page_id(page_id)
        self.physical_gpu_slots[page_id] = -1
        self.valid_tokens[page_id] = 0
        self.data_epochs[page_id] = 0
        self.generations[page_id] = 0
        self.location_states[page_id] = DEVICE_PAGE_ABSENT

    def gather(
        self,
        logical_page_ids,
        page_ids,
        *,
        expected_epochs=None,
        expected_generations=None,
        expected_valid_tokens=None,
    ):
        """Gather and validate without host scalar extraction or sync."""

        logical_page_ids = logical_page_ids.to(
            device=self.device, dtype=torch.int64
        ).reshape(-1)
        page_ids = page_ids.to(device=self.device, dtype=torch.int64).reshape(-1)
        if logical_page_ids.shape != page_ids.shape:
            raise ValueError("logical and physical page IDs must align")
        invalid = (page_ids < 0) | (page_ids >= self.page_capacity)
        safe_page_ids = page_ids.clamp(0, self.page_capacity - 1)
        slots = self.physical_gpu_slots[safe_page_ids].contiguous()
        valid_tokens = self.valid_tokens[safe_page_ids].contiguous()
        epochs = self.data_epochs[safe_page_ids].contiguous()
        generations = self.generations[safe_page_ids].contiguous()
        locations = self.location_states[safe_page_ids].contiguous()
        errors = invalid.to(torch.int32) * DEVICE_PAGE_INVALID_ID
        if expected_epochs is not None:
            expected_epochs = expected_epochs.to(
                device=self.device, dtype=torch.int64
            ).reshape(-1)
            if expected_epochs.shape != page_ids.shape:
                raise ValueError("expected epochs must align with selected pages")
            errors = errors | (
                (epochs != expected_epochs).to(torch.int32)
                * DEVICE_PAGE_STALE_EPOCH
            )
        if expected_generations is not None:
            expected_generations = expected_generations.to(
                device=self.device, dtype=torch.int64
            ).reshape(-1)
            if expected_generations.shape != page_ids.shape:
                raise ValueError(
                    "expected generations must align with selected pages"
                )
            errors = errors | (
                (generations != expected_generations).to(torch.int32)
                * DEVICE_PAGE_STALE_GENERATION
            )
        if expected_valid_tokens is not None:
            expected_valid_tokens = expected_valid_tokens.to(
                device=self.device, dtype=torch.int32
            ).reshape(-1)
            if expected_valid_tokens.shape != page_ids.shape:
                raise ValueError(
                    "expected valid tokens must align with selected pages"
                )
            errors = errors | (
                (valid_tokens != expected_valid_tokens).to(torch.int32)
                * DEVICE_PAGE_STALE_VALID_TOKENS
            )
        return DeviceKVPageMetadata(
            logical_page_ids=logical_page_ids,
            page_ids=page_ids,
            physical_gpu_slots=slots,
            valid_tokens=valid_tokens,
            data_epochs=epochs,
            generations=generations,
            location_states=locations,
            error_states=errors,
        )


__all__ = [
    "DEVICE_PAGE_ABSENT",
    "DEVICE_PAGE_CPU",
    "DEVICE_PAGE_GPU",
    "DEVICE_PAGE_GPU_CPU",
    "DEVICE_PAGE_INVALID_ID",
    "DEVICE_PAGE_OK",
    "DEVICE_PAGE_STALE_EPOCH",
    "DEVICE_PAGE_STALE_GENERATION",
    "DEVICE_PAGE_STALE_VALID_TOKENS",
    "DeviceKVPageMetadata",
    "DeviceKVPageTable",
]
