"""Storage-independent selected page views passed to dispatchers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResolvedSelectedPage:
    """One host bridge result; it carries no reference ownership."""

    handle: object
    logical_block_id: int
    valid_tokens: int


@dataclass(frozen=True)
class SelectedPageView:
    flat_page_ids: object
    block_table_indptr: object
    logical_block_ids: object
    page_valid_tokens: object
    selection_name: str
    exact: bool
    metadata: dict

    def __post_init__(self):
        count = int(self.flat_page_ids.numel())
        if self.logical_block_ids.shape != (count,):
            raise ValueError("logical block IDs must align with selected pages")
        if self.page_valid_tokens.shape != (count,):
            raise ValueError("valid-token metadata must align with selected pages")
        devices = {
            self.flat_page_ids.device,
            self.block_table_indptr.device,
            self.logical_block_ids.device,
            self.page_valid_tokens.device,
        }
        if len(devices) != 1:
            raise ValueError("selected page metadata must share a device")

    def resolve_entries(self, requests, page_pool):
        """Resolve selected logical IDs to generation-checked PageHandles.

        Handles are deliberately resolved after Selection: providers choose
        logical pages, while Ownership remains the sole authority that turns
        that choice into physical pins.
        """

        requests = tuple(requests)
        if self.block_table_indptr.shape != (len(requests) + 1,):
            raise ValueError("selected request boundaries do not match requests")
        count = int(self.flat_page_ids.numel())
        boundary_count = len(requests) + 1
        # This is the one explicit device -> host bridge that remains after
        # tensorized top-k. Pack all selected execution metadata into one
        # transfer; never synchronize once per scalar/page. Ownership still
        # validates each resulting generation handle below.
        packed = torch.cat(
            (
                self.block_table_indptr.to(dtype=torch.int64).reshape(-1),
                self.flat_page_ids.to(dtype=torch.int64).reshape(-1),
                self.logical_block_ids.to(dtype=torch.int64).reshape(-1),
                self.page_valid_tokens.to(dtype=torch.int64).reshape(-1),
            ),
            dim=0,
        ).to(device="cpu").tolist()
        boundaries = packed[:boundary_count]
        offset = boundary_count
        physical_ids = packed[offset : offset + count]
        offset += count
        logical_ids = packed[offset : offset + count]
        offset += count
        valid_tokens = packed[offset : offset + count]
        resolved = []
        seen_logical = set()
        for request_index, state in enumerate(requests):
            start = int(boundaries[request_index])
            end = int(boundaries[request_index + 1])
            for selected_index in range(start, end):
                logical = int(logical_ids[selected_index])
                if logical < 0 or logical >= len(state.block_table.handles):
                    raise IndexError("selected logical page is outside request")
                logical_identity = (int(state.request_id), logical)
                if logical_identity in seen_logical:
                    raise ValueError("selected logical page is duplicated")
                seen_logical.add(logical_identity)
                handle = state.block_table.handles[logical]
                descriptor = page_pool.descriptor(handle)
                physical = int(physical_ids[selected_index])
                if descriptor.page_id != physical:
                    raise ValueError(
                        "selected physical page does not match generation handle"
                    )
                valid = int(valid_tokens[selected_index])
                if valid <= 0:
                    raise ValueError("selected page has no valid tokens")
                resolved.append(ResolvedSelectedPage(handle, logical, valid))
        return tuple(resolved)

    def resolve_handles(self, requests, page_pool):
        """Compatibility projection of the single selected-page host bridge."""

        entries = self.resolve_entries(requests, page_pool)
        handles = []
        seen = set()
        for entry in entries:
            identity = entry.handle.identity()
            if identity not in seen:
                handles.append(entry.handle)
                seen.add(identity)
        return tuple(handles)


@dataclass(frozen=True)
class DeviceSelectedPageView(SelectedPageView):
    """Selected pages with device-resident execution/location metadata.

    This is additive to the host/debug ``SelectedPageView`` contract.  It does
    not resolve or retain PageHandles and therefore cannot own lifecycle.
    ``error_state_tensor`` remains device-visible until an existing safe
    synchronization boundary chooses to surface an error on the host.
    """

    gpu_physical_slots: object
    expected_epochs: object
    current_epochs: object
    expected_generations: object
    current_generations: object
    location_flags: object
    valid_mask: object
    error_mask: object
    selection_count: int
    staged_tail_mask: object = None

    def __post_init__(self):
        super().__post_init__()
        count = int(self.flat_page_ids.numel())
        fields = (
            self.gpu_physical_slots,
            self.expected_epochs,
            self.current_epochs,
            self.expected_generations,
            self.current_generations,
            self.location_flags,
            self.valid_mask,
            self.error_mask,
        )
        if any(item.shape != (count,) for item in fields):
            raise ValueError("device selected-page metadata must align")
        if int(self.selection_count) != count:
            raise ValueError("selection_count must match selected tensors")
        if len({item.device for item in fields + (self.flat_page_ids,)}) != 1:
            raise ValueError("device selected-page metadata must share a device")
        if self.gpu_physical_slots.dtype != torch.int32:
            raise TypeError("GPU physical slots must use int32")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("device selected-page valid mask must be bool")
        if self.error_mask.dtype != torch.int32:
            raise TypeError("device selected-page error mask must be int32")
        if any(not item.is_contiguous() for item in fields):
            raise ValueError("device selected-page metadata must be contiguous")
        if self.staged_tail_mask is not None:
            if self.staged_tail_mask.shape != (count,):
                raise ValueError("staged-tail mask must align with selected pages")
            if self.staged_tail_mask.device != self.flat_page_ids.device:
                raise ValueError("staged-tail mask must share the selected device")

    @property
    def logical_ids_tensor(self):
        return self.logical_block_ids

    @property
    def valid_tokens_tensor(self):
        return self.page_valid_tokens

    def device_epoch_error_state(self):
        """Return the device error vector without reducing it on the host."""

        return self.error_mask

    # Compatibility projections for the early additive ABI.  Production
    # consumers should use the explicit canonical names above.
    @property
    def physical_slot_tensor(self):
        return self.gpu_physical_slots

    @property
    def epoch_tensor(self):
        return self.expected_epochs

    @property
    def current_epoch_tensor(self):
        return self.current_epochs

    @property
    def generation_tensor(self):
        return self.current_generations

    @property
    def location_state_tensor(self):
        return self.location_flags

    @property
    def error_state_tensor(self):
        return self.error_mask


__all__ = [
    "DeviceSelectedPageView",
    "ResolvedSelectedPage",
    "SelectedPageView",
]
