"""Frozen Paged Attention Provider ABI V1."""

from dataclasses import dataclass
import math

import torch

from ...kv.batch_state import PagedBatchView
from ...kv.page_view import SelectedPageView
from ...kv.slot_mapping import SlotMapping


PAGED_ATTENTION_ABI_VERSION = 1


@dataclass(frozen=True)
class PagedKVAppendInput:
    key: torch.Tensor
    value: torch.Tensor
    key_pool_view: torch.Tensor
    value_pool_view: torch.Tensor
    slot_mapping: SlotMapping
    page_size: int
    num_kv_heads: int
    head_dim: int

    def validate(self):
        if self.key.shape != self.value.shape or self.key.ndim != 3:
            raise ValueError("key/value must be [token, kv_head, head_dim]")
        if self.key.shape != (
            self.slot_mapping.token_count,
            int(self.num_kv_heads),
            int(self.head_dim),
        ):
            raise ValueError("append key/value geometry is invalid")
        expected_tail = (
            int(self.num_kv_heads),
            int(self.page_size),
            int(self.head_dim),
        )
        if self.key_pool_view.ndim != 4 or tuple(self.key_pool_view.shape[1:]) != expected_tail:
            raise ValueError("key page pool geometry is invalid")
        if self.value_pool_view.shape != self.key_pool_view.shape:
            raise ValueError("key/value page pools must share a shape")
        tensors = (
            self.key,
            self.value,
            self.key_pool_view,
            self.value_pool_view,
            self.slot_mapping.page_ids,
            self.slot_mapping.offsets,
        )
        if len({item.device for item in tensors}) != 1:
            raise ValueError("append tensors must share a device")
        if self.key.dtype != self.key_pool_view.dtype or self.value.dtype != self.value_pool_view.dtype:
            raise ValueError("append dtype does not match page pool")
        return self


@dataclass(frozen=True)
class PagedAttentionInput:
    query: torch.Tensor
    key_pool_view: torch.Tensor
    value_pool_view: torch.Tensor
    batch_view: PagedBatchView
    page_size: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    softmax_scale: float
    causal: bool
    kv_dtype: str
    output_dtype: str
    selected_pages: SelectedPageView
    workspace: object = None
    return_logsumexp: bool = False

    @property
    def flat_block_table(self):
        return self.selected_pages.flat_page_ids

    @property
    def block_table_indptr(self):
        return self.selected_pages.block_table_indptr

    @property
    def logical_block_ids(self):
        return self.selected_pages.logical_block_ids

    @property
    def page_valid_tokens(self):
        return self.selected_pages.page_valid_tokens

    @property
    def sequence_lengths(self):
        return self.batch_view.sequence_lengths

    @property
    def query_indptr(self):
        return self.batch_view.query_indptr

    @property
    def query_positions(self):
        return self.batch_view.query_positions

    @property
    def tail_valid_tokens(self):
        return self.batch_view.tail_valid_tokens

    @property
    def slot_mapping(self):
        return self.batch_view.slot_mapping

    def validate(self):
        if self.query.ndim != 3:
            raise ValueError("query must be [total_query_token, query_head, head_dim]")
        if tuple(self.query.shape[1:]) != (
            int(self.num_query_heads),
            int(self.head_dim),
        ):
            raise ValueError("query geometry is invalid")
        if int(self.query.shape[0]) != self.batch_view.total_query_tokens:
            raise ValueError("query token count does not match batch view")
        if self.block_table_indptr.shape != (self.batch_view.batch_size + 1,):
            raise ValueError("selected block indptr does not match batch")
        if int(self.block_table_indptr[-1].item()) != int(
            self.flat_block_table.numel()
        ):
            raise ValueError("selected block indptr does not cover selected pages")
        if int(self.num_query_heads) % int(self.num_kv_heads):
            raise ValueError("query heads must be divisible by KV heads")
        expected_tail = (
            int(self.num_kv_heads),
            int(self.page_size),
            int(self.head_dim),
        )
        if self.key_pool_view.ndim != 4 or tuple(self.key_pool_view.shape[1:]) != expected_tail:
            raise ValueError("key pool must be HND [page, head, token, dim]")
        if self.value_pool_view.shape != self.key_pool_view.shape:
            raise ValueError("K/V pools must share a shape")
        tensors = (
            self.query,
            self.key_pool_view,
            self.value_pool_view,
            self.flat_block_table,
            self.block_table_indptr,
            self.logical_block_ids,
            self.page_valid_tokens,
            self.sequence_lengths,
            self.query_indptr,
            self.query_positions,
        )
        if len({item.device for item in tensors}) != 1:
            raise ValueError("paged attention tensors must share a device")
        expected_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }.get(str(self.kv_dtype))
        if expected_dtype is None:
            raise ValueError("unsupported executable KV dtype {}".format(self.kv_dtype))
        if self.key_pool_view.dtype != expected_dtype:
            raise ValueError("KV dtype does not match page pool")
        if self.query.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError("unsupported query dtype")
        if not math.isfinite(float(self.softmax_scale)) or float(self.softmax_scale) <= 0:
            raise ValueError("softmax_scale must be finite and positive")
        return self


@dataclass(frozen=True)
class PagedAttentionOutput:
    output: torch.Tensor
    logsumexp: object
    provider_metrics: dict
