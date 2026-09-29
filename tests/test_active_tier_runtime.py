import unittest

import torch

from layer_streaming.kv.runtime import PagedKVRuntime
from layer_streaming.kv_policy import KVPolicy


def _cpu_factory(shape, **kwargs):
    return torch.empty(shape, dtype=kwargs["dtype"], device="cpu")


class ActiveTierRuntimeTest(unittest.TestCase):
    layers = 2
    query_heads = 2
    kv_heads = 1
    head_dim = 4
    page_size = 16
    logical_pages = 16
    hot_pages = 4

    @property
    def logical_page_bytes(self):
        return (
            2
            * self.layers
            * self.kv_heads
            * self.page_size
            * self.head_dim
            * torch.empty((), dtype=torch.float16).element_size()
        )

    def make_runtime(self, tiered):
        policy = KVPolicy(
            storage="gpu_cpu" if tiered else "gpu",
            dtype="fp16",
            selection="none",
            reuse="request_only",
            attention_backend="reference_paged_exact",
            page_size=self.page_size,
            gpu_hot_budget_bytes=(
                self.hot_pages * self.logical_page_bytes if tiered else 0
            ),
            cpu_budget_bytes=(
                self.logical_pages * self.logical_page_bytes if tiered else 0
            ),
            gpu_high_watermark_bytes=(
                self.hot_pages * self.logical_page_bytes if tiered else 0
            ),
            gpu_low_watermark_bytes=(
                (self.hot_pages - 1) * self.logical_page_bytes
                if tiered
                else 0
            ),
        )
        return PagedKVRuntime(
            layer_count=self.layers,
            num_query_heads=self.query_heads,
            num_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            page_count=self.logical_pages,
            page_size=self.page_size,
            dtype=torch.float16,
            device="cpu",
            policy=policy,
            allow_reference=True,
            prefetch_timeout_seconds=2.0,
            tier_tensor_factory=(_cpu_factory if tiered else None),
        )

    def payload(self, layer, start, count):
        values = torch.arange(
            start * self.head_dim,
            (start + count) * self.head_dim,
            dtype=torch.float16,
        ).reshape(count, self.kv_heads, self.head_dim)
        key = values.mul(0.001).add_(float(layer + 1))
        value = values.mul(0.002).add_(float(10 + layer))
        return key, value

    def query(self, layer, position):
        base = torch.arange(
            self.query_heads * self.head_dim, dtype=torch.float16
        ).reshape(1, self.query_heads, self.head_dim)
        return base.mul(0.01).add_(float(layer + 1 + position * 0.0001))

    def test_four_hot_pages_run_sixteen_logical_pages_end_to_end(self):
        dense = self.make_runtime(tiered=False)
        tiered = self.make_runtime(tiered=True)
        dense_state = dense.create_request(self.logical_pages * self.page_size)
        tiered_state = tiered.create_request(self.logical_pages * self.page_size)

        total_tokens = self.logical_pages * self.page_size
        # Three pages leave one bounded streaming slot for historical pages.
        chunk_tokens = (self.hot_pages - 1) * self.page_size
        start = 0
        outputs_checked = 0
        while start < total_tokens:
            count = min(chunk_tokens, total_tokens - start)
            for layer in range(self.layers):
                key, value = self.payload(layer, start, count)
                dense.append((dense_state,), layer, key, value, (count,))
                tiered.append((tiered_state,), layer, key, value, (count,))
                query = self.query(layer, start + count - 1)
                dense_result = dense.attend(
                    (dense_state,), layer, query, (1,), phase="decode"
                )
                tiered_result = tiered.attend(
                    (tiered_state,), layer, query, (1,), phase="decode"
                )
                self.assertTrue(
                    torch.allclose(
                        tiered_result.output,
                        dense_result.output,
                        atol=3e-3,
                        rtol=3e-3,
                    )
                )
                self.assertFalse(
                    tiered_result.provider_metrics["full_kv_workspace"]
                )
                self.assertEqual(
                    tiered_result.provider_metrics[
                        "resident_working_set_pages"
                    ],
                    1,
                )
                outputs_checked += 1
            start += count
            for handle in tiered_state.block_table.handles:
                key = tiered._tier_registered_key(handle)
                self.assertIsNotNone(key)
                self.assertEqual(
                    key.data_epoch,
                    tiered.page_pool.descriptor(handle).data_version,
                )

        self.assertEqual(tiered.page_pool.page_count, self.logical_pages)
        self.assertEqual(tiered.store.gpu_capacity_pages, self.hot_pages)
        self.assertEqual(tiered_state.sequence_length, total_tokens)
        self.assertGreater(outputs_checked, 0)
        stats = tiered.profile_stats()
        self.assertTrue(stats["tier_metrics_sampled"])
        self.assertGreater(stats["eviction_count"], 0)
        self.assertGreater(stats["prefetch_count"], 0)
        self.assertGreater(stats["d2h_kv_bytes"], 0)
        self.assertGreater(stats["h2d_kv_bytes"], 0)
        self.assertEqual(stats["pending_tier_operations"], 0)
        self.assertEqual(stats["total_ref_count"], stats["logical_owner_count"])

        tiered.reset(tiered_state)
        reset_stats = tiered.profile_stats()
        self.assertEqual(reset_stats["kv_pool_allocated_pages"], 0)
        self.assertEqual(reset_stats["gpu_kv_used_pages"], 0)
        self.assertEqual(reset_stats["cpu_kv_used_bytes"], 0)
        tiered.close()
        dense.close()

    def test_active_tier_batch_gate_and_close_with_cpu_pages(self):
        runtime = self.make_runtime(tiered=True)
        first = runtime.create_request(6 * self.page_size)
        second = runtime.create_request(6 * self.page_size)
        key, value = self.payload(0, 0, self.page_size)
        with self.assertRaisesRegex(NotImplementedError, "batch 1"):
            runtime.append(
                (first, second),
                0,
                torch.cat((key, key), dim=0),
                torch.cat((value, value), dim=0),
                (self.page_size, self.page_size),
            )
        for start, count in ((0, 3 * self.page_size), (3 * self.page_size, 2 * self.page_size)):
            for layer in range(self.layers):
                chunk_key, chunk_value = self.payload(layer, start, count)
                runtime.append(
                    (first,), layer, chunk_key, chunk_value, (count,)
                )
        before_close = runtime.profile_stats()
        self.assertGreater(before_close["cpu_kv_used_bytes"], 0)
        self.assertEqual(before_close["pending_tier_operations"], 0)
        self.assertEqual(runtime.prefetch_timeout_seconds, 2.0)
        runtime.close()

    def test_continuation_prefetches_an_evicted_partial_tail(self):
        dense = self.make_runtime(tiered=False)
        tiered = self.make_runtime(tiered=True)
        dense_state = dense.create_request(4 * self.page_size)
        tiered_state = tiered.create_request(4 * self.page_size)
        initial = self.page_size + self.page_size // 2
        for layer in range(self.layers):
            key, value = self.payload(layer, 0, initial)
            dense.append((dense_state,), layer, key, value, (initial,))
            tiered.append((tiered_state,), layer, key, value, (initial,))

        tail_handle = tiered_state.block_table.handles[-1]
        old_key = tiered._tier_registered_key(tail_handle)
        tiered.active_tier.evict(old_key).wait(timeout_seconds=2.0)
        self.assertTrue(
            all(
                location.cpu_slot is not None
                for location in tiered.store.location_sets(old_key)
            )
        )
        for layer in range(self.layers):
            key, value = self.payload(layer, initial, 1)
            dense.append((dense_state,), layer, key, value, (1,))
            tiered.append((tiered_state,), layer, key, value, (1,))
            query = self.query(layer, initial)
            dense_result = dense.attend(
                (dense_state,), layer, query, (1,), phase="decode"
            )
            tiered_result = tiered.attend(
                (tiered_state,), layer, query, (1,), phase="decode"
            )
            self.assertTrue(
                torch.allclose(
                    tiered_result.output,
                    dense_result.output,
                    atol=3e-3,
                    rtol=3e-3,
                )
            )
        new_key = tiered._tier_registered_key(tail_handle)
        self.assertGreater(new_key.data_epoch, old_key.data_epoch)
        stats = tiered.profile_stats()
        self.assertGreater(stats["d2h_kv_bytes"], 0)
        self.assertGreater(stats["h2d_kv_bytes"], 0)
        tiered.close()
        dense.close()


if __name__ == "__main__":
    unittest.main()
