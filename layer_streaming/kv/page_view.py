"""Storage-independent selected page views passed to dispatchers."""

from dataclasses import dataclass


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
