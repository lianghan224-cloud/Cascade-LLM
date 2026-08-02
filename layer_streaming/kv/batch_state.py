"""Ragged batch metadata; batch size one uses the identical ABI."""

from dataclasses import dataclass

import torch

from .errors import KVLifecycleError
from .slot_mapping import SlotMapping


PAGED_BATCH_ABI_VERSION = 1


@dataclass(frozen=True)
class PagedBatchView:
    request_ids: tuple
    query_indptr: torch.Tensor
    block_table_indptr: torch.Tensor
    flat_block_table: torch.Tensor
    flat_logical_block_ids: torch.Tensor
    flat_page_valid_tokens: torch.Tensor
    sequence_lengths: torch.Tensor
    query_lengths: torch.Tensor
    tail_valid_tokens: torch.Tensor
    slot_mapping: SlotMapping
    query_positions: torch.Tensor
    page_size: int
    layer: int
    version: int = PAGED_BATCH_ABI_VERSION

    def __post_init__(self):
        batch_size = len(self.request_ids)
        if self.version != PAGED_BATCH_ABI_VERSION:
            raise ValueError("unsupported PagedBatchView ABI")
        if self.query_indptr.shape != (batch_size + 1,):
            raise ValueError("query_indptr shape does not match batch")
        if self.block_table_indptr.shape != (batch_size + 1,):
            raise ValueError("block_table_indptr shape does not match batch")
        for name in ("sequence_lengths", "query_lengths", "tail_valid_tokens"):
            if getattr(self, name).shape != (batch_size,):
                raise ValueError("{} shape does not match batch".format(name))
        total_query = int(self.query_positions.numel())
        if total_query != int(self.query_indptr[-1].item()):
            raise ValueError("query positions do not match query_indptr")
        if self.slot_mapping.token_count not in {0, total_query}:
            raise ValueError("slot_mapping must be empty or cover every query token")
        flat_count = int(self.flat_block_table.numel())
        if self.flat_logical_block_ids.shape != (flat_count,):
            raise ValueError("flat logical block IDs do not match block table")
        if self.flat_page_valid_tokens.shape != (flat_count,):
            raise ValueError("flat valid-token metadata does not match block table")
        devices = {
            tensor.device
            for tensor in (
                self.query_indptr,
                self.block_table_indptr,
                self.flat_block_table,
                self.flat_logical_block_ids,
                self.flat_page_valid_tokens,
                self.sequence_lengths,
                self.query_lengths,
                self.tail_valid_tokens,
                self.query_positions,
                self.slot_mapping.page_ids,
                self.slot_mapping.offsets,
            )
        }
        if len(devices) != 1:
            raise ValueError("all PagedBatchView tensors must share a device")

    @property
    def batch_size(self):
        return len(self.request_ids)

    @property
    def total_query_tokens(self):
        return int(self.query_positions.numel())

    @property
    def max_query_length(self):
        return int(self.query_lengths.max().item()) if self.batch_size else 0

    @property
    def device(self):
        return self.sequence_lengths.device

    def as_dict(self):
        return {
            "version": self.version,
            "request_ids": list(self.request_ids),
            "page_size": int(self.page_size),
            "layer": int(self.layer),
            "batch_size": self.batch_size,
            "total_query_tokens": self.total_query_tokens,
            "max_query_length": self.max_query_length,
            "flat_block_count": int(self.flat_block_table.numel()),
            "sequence_lengths": self.sequence_lengths.tolist(),
            "query_lengths": self.query_lengths.tolist(),
            "tail_valid_tokens": self.tail_valid_tokens.tolist(),
        }


def _prefix_sum(values):
    result = [0]
    for value in values:
        result.append(result[-1] + int(value))
    return result


def build_paged_batch_view(
    requests,
    query_lengths,
    layer,
    device,
    page_size,
    query_positions=None,
    slot_mappings=None,
):
    requests = tuple(requests)
    query_lengths = tuple(int(item) for item in query_lengths)
    if len(requests) != len(query_lengths):
        raise ValueError("query_lengths must have one item per request")
    if not requests:
        raise ValueError("PagedBatchView requires at least one request")
    if any(item <= 0 for item in query_lengths):
        raise ValueError("query lengths must be positive")
    request_ids = tuple(int(item.request_id) for item in requests)
    if len(set(request_ids)) != len(request_ids):
        raise ValueError("a request may appear only once in a batch")
    layer = int(layer)
    sequence_lengths = []
    tails = []
    flat_blocks = []
    flat_logical_blocks = []
    flat_valid_tokens = []
    block_counts = []
    positions = []
    slot_page_ids = []
    slot_offsets = []
    if query_positions is not None:
        query_positions = tuple(query_positions)
        if len(query_positions) != len(requests):
            raise ValueError("query_positions must have one tensor per request")
    if slot_mappings is not None and len(slot_mappings) != len(requests):
        raise ValueError("slot_mappings must have one item per request")
    for index, (request, query_length) in enumerate(zip(requests, query_lengths)):
        request.ensure_active()
        if layer < 0 or layer >= len(request.layer_lengths):
            raise IndexError("layer is outside request KV state")
        sequence_length = int(request.layer_lengths[layer])
        if query_length > sequence_length:
            raise KVLifecycleError("query exceeds populated layer KV")
        sequence_lengths.append(sequence_length)
        tails.append(sequence_length % int(page_size) or int(page_size))
        page_ids = request.block_table.physical_page_ids("gpu")
        required_blocks = (sequence_length + int(page_size) - 1) // int(page_size)
        page_ids = page_ids[:required_blocks]
        block_counts.append(len(page_ids))
        flat_blocks.extend(page_ids)
        flat_logical_blocks.extend(range(len(page_ids)))
        flat_valid_tokens.extend(
            min(int(page_size), sequence_length - logical * int(page_size))
            for logical in range(len(page_ids))
        )
        if query_positions is None:
            positions.extend(range(sequence_length - query_length, sequence_length))
        else:
            item = query_positions[index]
            if isinstance(item, torch.Tensor):
                item = item.detach().reshape(-1).tolist()
            item = [int(value) for value in item]
            if len(item) != query_length:
                raise ValueError("query position count does not match query length")
            positions.extend(item)
        if slot_mappings is not None:
            mapping = slot_mappings[index]
            if mapping.token_count != query_length:
                raise ValueError("slot mapping does not match query length")
            slot_page_ids.extend(mapping.page_ids.detach().cpu().tolist())
            slot_offsets.extend(mapping.offsets.detach().cpu().tolist())
    tensor_device = torch.device(device)
    int_dtype = torch.int32
    query_indptr = torch.tensor(
        _prefix_sum(query_lengths), dtype=int_dtype, device=tensor_device
    )
    block_indptr = torch.tensor(
        _prefix_sum(block_counts), dtype=int_dtype, device=tensor_device
    )
    empty_or_pages = slot_page_ids if slot_mappings is not None else []
    empty_or_offsets = slot_offsets if slot_mappings is not None else []
    return PagedBatchView(
        request_ids=request_ids,
        query_indptr=query_indptr,
        block_table_indptr=block_indptr,
        flat_block_table=torch.tensor(
            flat_blocks, dtype=int_dtype, device=tensor_device
        ),
        flat_logical_block_ids=torch.tensor(
            flat_logical_blocks, dtype=int_dtype, device=tensor_device
        ),
        flat_page_valid_tokens=torch.tensor(
            flat_valid_tokens, dtype=int_dtype, device=tensor_device
        ),
        sequence_lengths=torch.tensor(
            sequence_lengths, dtype=int_dtype, device=tensor_device
        ),
        query_lengths=torch.tensor(
            query_lengths, dtype=int_dtype, device=tensor_device
        ),
        tail_valid_tokens=torch.tensor(
            tails, dtype=int_dtype, device=tensor_device
        ),
        slot_mapping=SlotMapping(
            torch.tensor(empty_or_pages, dtype=int_dtype, device=tensor_device),
            torch.tensor(empty_or_offsets, dtype=int_dtype, device=tensor_device),
        ),
        query_positions=torch.tensor(
            positions, dtype=int_dtype, device=tensor_device
        ),
        page_size=int(page_size),
        layer=layer,
    )
