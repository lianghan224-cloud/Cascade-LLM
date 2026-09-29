from contextlib import contextmanager
import unittest

import torch
import torch.nn.functional as F

from layer_streaming.attention.paged.abi import PagedAttentionInput
from layer_streaming.attention.paged.tiered_streaming import (
    TieredStreamingExactAttention,
)
from layer_streaming.kv.batch_state import build_paged_batch_view
from layer_streaming.kv.page_pool import KVPagePoolV1
from layer_streaming.kv.page_view import SelectedPageView
from layer_streaming.kv.request_state import RequestKVState
from layer_streaming.kv.block_table import LogicalBlockTable


class TieredStreamingAttentionTest(unittest.TestCase):
    def make_request(self, page_count=6, page_size=2, query_length=12):
        pool = KVPagePoolV1(page_count, "gpu", "bf16")
        state = RequestKVState(
            request_id=1,
            block_table=LogicalBlockTable(page_size, page_count * page_size),
            layer_lengths=[query_length],
        )
        for logical in range(page_count):
            handle = pool.allocate(owner_hint=1)
            pool.activate(handle, page_size)
            pool.seal(handle, page_size)
            state.block_table.append(handle)
        state.sequence_length = query_length
        state.tail_valid_tokens = page_size
        batch = build_paged_batch_view(
            (state,), (query_length,), 0, "cpu", page_size
        )
        selected = SelectedPageView(
            flat_page_ids=batch.flat_block_table,
            block_table_indptr=batch.block_table_indptr,
            logical_block_ids=batch.flat_logical_block_ids,
            page_valid_tokens=batch.flat_page_valid_tokens,
            selection_name="dense",
            exact=True,
            metadata={},
        )
        return batch, selected

    def test_six_page_request_uses_one_page_working_set_and_matches_reference(self):
        torch.manual_seed(7)
        page_count, page_size, kv_heads, query_heads, head_dim = 6, 2, 2, 4, 3
        batch, selected = self.make_request(page_count, page_size, page_count * page_size)
        keys = torch.randn(page_count, kv_heads, page_size, head_dim)
        values = torch.randn_like(keys)
        query = torch.randn(page_count * page_size, query_heads, head_dim)
        request = PagedAttentionInput(
            query=query,
            key_pool_view=keys,
            value_pool_view=values,
            batch_view=batch,
            page_size=page_size,
            num_query_heads=query_heads,
            num_kv_heads=kv_heads,
            head_dim=head_dim,
            softmax_scale=head_dim ** -0.5,
            causal=True,
            kv_dtype="bf16",
            output_dtype="fp32",
            selected_pages=selected,
            return_logsumexp=True,
        )
        active = 0
        peak = 0
        visits = []

        @contextmanager
        def loader(index):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            visits.append(index)
            try:
                yield keys[index], values[index]
            finally:
                active -= 1

        actual = TieredStreamingExactAttention().execute(request, loader)
        dense_key = keys.transpose(0, 1).reshape(kv_heads, -1, head_dim)
        dense_value = values.transpose(0, 1).reshape(kv_heads, -1, head_dim)
        dense_key = dense_key.repeat_interleave(query_heads // kv_heads, dim=0)
        dense_value = dense_value.repeat_interleave(query_heads // kv_heads, dim=0)
        q = query.transpose(0, 1).unsqueeze(0)
        expected_output = F.scaled_dot_product_attention(
            q,
            dense_key.unsqueeze(0),
            dense_value.unsqueeze(0),
            dropout_p=0.0,
            is_causal=True,
            scale=head_dim ** -0.5,
        ).squeeze(0).transpose(0, 1)
        scores = torch.einsum("qhd,htd->qht", query, dense_key)
        scores.mul_(head_dim ** -0.5)
        causal_mask = torch.arange(page_count * page_size).view(1, 1, -1) <= torch.arange(
            page_count * page_size
        ).view(-1, 1, 1)
        expected_lse = torch.logsumexp(
            scores.masked_fill(~causal_mask, -float("inf")), dim=-1
        )
        self.assertTrue(torch.allclose(actual.output, expected_output, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(actual.logsumexp, expected_lse, atol=1e-5, rtol=1e-5))
        self.assertEqual(peak, 1)
        self.assertEqual(visits, list(range(page_count)) * 2)
        self.assertFalse(actual.provider_metrics["full_kv_workspace"])

    def test_batch_greater_than_one_fails_explicitly(self):
        batch, selected = self.make_request(page_count=1, page_size=2, query_length=2)
        object.__setattr__(batch, "request_ids", (1, 2))
        request = type("Request", (), {"batch_view": batch, "selected_pages": selected})()
        with self.assertRaisesRegex(NotImplementedError, "batch 1"):
            TieredStreamingExactAttention().execute(request, None)


if __name__ == "__main__":
    unittest.main()
