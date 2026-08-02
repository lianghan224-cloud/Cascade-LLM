import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from layer_streaming.experimental import (
    DensePagedOnlineAttention,
    KVCacheCapacityError,
    KVCacheError,
    KVCacheManager,
    KVPageState,
)


class KVCacheTest(unittest.TestCase):
    def make_manager(self, **overrides):
        values = {
            "layer_count": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "total_blocks": 4,
            "block_size": 2,
            "max_batch_size": 1,
            "dtype": torch.float32,
            "device": "cpu",
        }
        values.update(overrides)
        return KVCacheManager(**values)

    def append_step(self, cache, values):
        for layer in range(cache.manager.layer_count):
            key = torch.stack(
                [
                    torch.full(
                        (1, 2, 1, 4),
                        value + layer,
                        dtype=torch.float32,
                    )
                    for value in values
                ],
                dim=2,
            ).squeeze(3)
            value = key + 100
            cache.append_only(layer, key, value)

    def test_append_uses_stable_preallocated_page_arena(self):
        manager = self.make_manager()
        cache = manager.bind(manager.allocate(4))
        arena_pointer = manager.keys.data_ptr()
        self.append_step(cache, [1, 2])
        self.append_step(cache, [3])
        self.assertEqual(manager.keys.data_ptr(), arena_pointer)
        self.assertEqual(cache.sequence_length(), 3)
        key, _ = cache.get_view(0)
        self.assertEqual(key.shape, (1, 2, 3, 4))
        self.assertTrue(torch.equal(key[0, 0, :, 0], torch.tensor([1., 2., 3.])))
        self.assertEqual(cache.block_table().length, 3)

    def test_reset_release_and_page_reuse(self):
        manager = self.make_manager()
        handle = manager.allocate(4)
        cache = manager.bind(handle)
        self.append_step(cache, [1, 2, 3])
        page_ids = handle.block_ids
        self.assertEqual(manager.allocated_blocks, 2)
        cache.clear()
        self.assertEqual(cache.sequence_length(), 0)
        self.assertEqual(manager.free_blocks, 4)
        self.append_step(cache, [4, 5, 6])
        self.assertEqual(handle.block_ids, page_ids)
        cache.close()
        self.assertEqual(manager.resource_stats()["active_handles"], 0)

    def test_capacity_and_order_errors_are_controlled(self):
        manager = self.make_manager(total_blocks=1)
        cache = manager.bind(manager.allocate(2))
        tensor = torch.zeros((1, 2, 1, 4), dtype=torch.float32)
        with self.assertRaises(KVCacheError):
            cache.append_only(1, tensor, tensor)
        for layer in range(2):
            cache.append_only(layer, tensor.repeat(1, 1, 2, 1), tensor.repeat(1, 1, 2, 1))
        with self.assertRaises(KVCacheCapacityError):
            cache.append_only(0, tensor, tensor)
        with self.assertRaises(KVCacheCapacityError):
            manager.allocate(3)

    def test_noncontiguous_physical_pages_follow_request_table(self):
        manager = self.make_manager(layer_count=1, total_blocks=5)
        first = manager.bind(manager.allocate(2))
        hole = manager.bind(manager.allocate(2))
        third = manager.bind(manager.allocate(2))
        self.append_step(first, [10])
        self.append_step(hole, [20])
        self.append_step(third, [30])
        hole.close()
        fragmented = manager.bind(manager.allocate(4))
        self.append_step(fragmented, [1, 2, 3])
        self.assertEqual(fragmented.handle.block_ids, (1, 3))
        key, _ = fragmented.get_view(0)
        self.assertTrue(torch.equal(key[0, 0, :, 0], torch.tensor([1., 2., 3.])))

    def test_fork_shares_sealed_pages_and_copies_mutable_tail(self):
        manager = self.make_manager(layer_count=1, total_blocks=8)
        parent = manager.bind(manager.allocate(8, request_id=11))
        self.append_step(parent, [1, 2, 3])
        child = parent.fork(request_id=12)
        self.assertEqual(parent.handle.block_ids[0], child.handle.block_ids[0])
        self.assertNotEqual(parent.handle.block_ids[1], child.handle.block_ids[1])
        shared = manager.page_metadata(parent.handle.block_ids[0])
        self.assertEqual(shared.state, KVPageState.SEALED_SHARED)
        self.assertEqual(shared.ref_count, 2)
        self.append_step(parent, [4])
        self.append_step(child, [9])
        parent_key, _ = parent.get_view(0)
        child_key, _ = child.get_view(0)
        self.assertEqual(parent_key[0, 0, -1, 0].item(), 4)
        self.assertEqual(child_key[0, 0, -1, 0].item(), 9)
        child.close()
        self.assertEqual(shared.ref_count, 1)
        self.assertEqual(shared.state, KVPageState.SEALED_PRIVATE)
        parent.close()
        self.assertEqual(manager.allocated_blocks, 0)

    def test_dense_paged_attention_matches_sdpa_for_gqa_prefill_and_decode(self):
        torch.manual_seed(7)
        manager = self.make_manager(layer_count=1, total_blocks=4)
        cache = manager.bind(manager.allocate(8))
        query = torch.randn((1, 4, 3, 4), dtype=torch.float32)
        key = torch.randn((1, 2, 3, 4), dtype=torch.float32)
        value = torch.randn((1, 2, 3, 4), dtype=torch.float32)
        cache.append_only(0, key, value)
        actual = cache.attend(
            0,
            query,
            kv_groups=2,
            position_ids=torch.arange(3).unsqueeze(0),
        )
        expected = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(2, dim=1),
            value.repeat_interleave(2, dim=1),
            dropout_p=0.0,
            is_causal=True,
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

        decode_query = torch.randn((1, 4, 1, 4), dtype=torch.float32)
        decode_key = torch.randn((1, 2, 1, 4), dtype=torch.float32)
        decode_value = torch.randn((1, 2, 1, 4), dtype=torch.float32)
        cache.append_only(0, decode_key, decode_value)
        actual_decode = cache.attend(
            0,
            decode_query,
            kv_groups=2,
            position_ids=torch.tensor([[3]]),
        )
        full_key = torch.cat((key, decode_key), dim=2).repeat_interleave(2, dim=1)
        full_value = torch.cat((value, decode_value), dim=2).repeat_interleave(2, dim=1)
        expected_decode = F.scaled_dot_product_attention(
            decode_query,
            full_key,
            full_value,
            dropout_p=0.0,
            is_causal=False,
        )
        torch.testing.assert_close(
            actual_decode, expected_decode, atol=1e-5, rtol=1e-5
        )
        profile = cache.profile_stats()
        self.assertEqual(profile["attention_backend"], "dense_paged_sdpa_reference")
        self.assertEqual(profile["materialize_calls"], 2)

    def test_online_paged_attention_avoids_full_kv_materialization(self):
        torch.manual_seed(9)
        manager = self.make_manager(
            layer_count=1,
            total_blocks=3,
            attention_backend=DensePagedOnlineAttention(),
        )
        cache = manager.bind(manager.allocate(6))
        query = torch.randn((1, 4, 3, 4), dtype=torch.float32)
        key = torch.randn((1, 2, 3, 4), dtype=torch.float32)
        value = torch.randn((1, 2, 3, 4), dtype=torch.float32)
        cache.append_only(0, key, value)
        actual = cache.attend(
            0,
            query,
            kv_groups=2,
            position_ids=torch.arange(3).unsqueeze(0),
        )
        expected = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(2, dim=1),
            value.repeat_interleave(2, dim=1),
            dropout_p=0.0,
            is_causal=True,
        )
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        profile = cache.profile_stats()
        self.assertEqual(
            profile["attention_backend"], "dense_paged_online_reference"
        )
        self.assertEqual(profile["materialize_calls"], 0)
        self.assertEqual(profile["materialized_bytes"], 0)

    def test_single_page_decode_does_not_pass_an_all_true_mask_to_sdpa(self):
        manager = self.make_manager(
            layer_count=1, total_blocks=1, block_size=4
        )
        cache = manager.bind(manager.allocate(4))
        key = torch.randn((1, 2, 4, 4), dtype=torch.float32)
        value = torch.randn((1, 2, 4, 4), dtype=torch.float32)
        query = torch.randn((1, 4, 1, 4), dtype=torch.float32)
        cache.append_only(0, key, value)
        with mock.patch(
            "torch.nn.functional.scaled_dot_product_attention",
            wraps=F.scaled_dot_product_attention,
        ) as sdpa:
            cache.attend(
                0,
                query,
                kv_groups=2,
                position_ids=torch.tensor([[3]]),
            )
        self.assertEqual(sdpa.call_count, 1)
        self.assertNotIn("attn_mask", sdpa.call_args.kwargs)
        self.assertFalse(sdpa.call_args.kwargs["is_causal"])

    def test_attention_inputs_fail_before_materialization(self):
        manager = self.make_manager(layer_count=1, total_blocks=2)
        cache = manager.bind(manager.allocate(4))
        query = torch.zeros((1, 4, 1, 4), dtype=torch.float32)
        with self.assertRaisesRegex(KVCacheError, "before KV"):
            cache.attend(0, query, kv_groups=2)

        key = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
        cache.append_only(0, key, key)
        with self.assertRaises(IndexError):
            cache.attend(1, query, kv_groups=2)
        with self.assertRaisesRegex(ValueError, "query heads"):
            cache.attend(0, query[:, :3], kv_groups=2)
        self.assertEqual(cache.profile_stats()["materialize_calls"], 0)

    def test_reset_is_atomic_when_a_page_is_pinned(self):
        manager = self.make_manager(layer_count=1, total_blocks=3)
        cache = manager.bind(manager.allocate(6))
        self.append_step(cache, [1, 2, 3])
        original_page_ids = cache.handle.block_ids
        manager.page_pool.pin(original_page_ids[0])
        try:
            with self.assertRaisesRegex(KVCacheError, "cannot reset"):
                cache.clear()
            self.assertEqual(cache.handle.block_ids, original_page_ids)
            self.assertEqual(cache.sequence_length(), 3)
            self.assertEqual(manager.allocated_blocks, 2)
        finally:
            manager.page_pool.unpin(original_page_ids[0])
        cache.clear()
        self.assertEqual(manager.allocated_blocks, 0)

    def test_one_hundred_allocate_release_cycles_restore_all_pages(self):
        manager = self.make_manager()
        for index in range(100):
            handle = manager.allocate(4)
            cache = manager.bind(handle)
            self.append_step(cache, [index])
            cache.close()
            stats = manager.resource_stats()
            self.assertEqual(stats["allocated_blocks"], 0)
            self.assertEqual(stats["active_handles"], 0)
            self.assertEqual(stats["free_blocks"], stats["total_blocks"])
        manager.close()
        manager.close()


if __name__ == "__main__":
    unittest.main()
