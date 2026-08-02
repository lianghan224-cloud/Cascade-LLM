"""Token-to-physical-page append mappings."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SlotMapping:
    page_ids: torch.Tensor
    offsets: torch.Tensor

    def __post_init__(self):
        if self.page_ids.ndim != 1 or self.offsets.ndim != 1:
            raise ValueError("slot mapping tensors must be one-dimensional")
        if self.page_ids.shape != self.offsets.shape:
            raise ValueError("slot mapping tensors must have equal length")
        if self.page_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("page_ids must be an integer tensor")
        if self.offsets.dtype not in {torch.int32, torch.int64}:
            raise ValueError("offsets must be an integer tensor")
        if self.page_ids.device != self.offsets.device:
            raise ValueError("slot mapping tensors must share a device")

    @property
    def token_count(self):
        return int(self.page_ids.numel())

    def as_tensor(self):
        return torch.stack((self.page_ids, self.offsets), dim=-1)
