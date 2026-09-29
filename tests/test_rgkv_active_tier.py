import unittest

import torch

from layer_streaming.kv.runtime import PagedKVRuntime
from layer_streaming.kv.types import RequestLifecycleState
from layer_streaming.kv_policy import KVPolicy


def _cpu_factory(shape, **kwargs):
    return torch.empty(shape, dtype=kwargs["dtype"], device="cpu")


class RGKVActiveTierTest(unittest.TestCase):
    layers = 2
    page_size = 16
    logical_pages = 8
    hot_pages = 3
    kv_heads = 1
    query_heads = 2
    head_dim = 4

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

    def make_runtime(self):
        return PagedKVRuntime(
            layer_count=self.layers,
            num_query_heads=self.query_heads,
            num_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            page_count=self.logical_pages,
            page_size=self.page_size,
            dtype=torch.float16,
            device="cpu",
            policy=KVPolicy(
                accuracy="sparse",
                storage="gpu_cpu",
                dtype="fp16",
                selection="rgkv",
                rgkv_scorer="torch_tensorized",
                page_budget=2,
                recent_window=self.page_size,
                attention_backend="reference_paged_exact",
                page_size=self.page_size,
                gpu_hot_budget_bytes=self.hot_pages * self.logical_page_bytes,
                cpu_budget_bytes=self.logical_pages * self.logical_page_bytes,
                gpu_high_watermark_bytes=self.hot_pages * self.logical_page_bytes,
                gpu_low_watermark_bytes=(self.hot_pages - 1)
                * self.logical_page_bytes,
            ),
            allow_reference=True,
            tier_tensor_factory=_cpu_factory,
        )

    def payload(self, layer, start):
        values = torch.arange(
            start * self.head_dim,
            (start + self.page_size) * self.head_dim,
            dtype=torch.float16,
        ).reshape(self.page_size, self.kv_heads, self.head_dim)
        return values.mul(0.01).add(layer + 1), values.add(20 + layer)

    def query(self):
        return torch.ones(
            (1, self.query_heads, self.head_dim), dtype=torch.float16
        )

    def test_staged_decode_and_selected_only_prefetch(self):
        runtime = self.make_runtime()
        state = runtime.create_request(self.logical_pages * self.page_size)
        try:
            for logical in range(6):
                start = logical * self.page_size
                for layer in range(self.layers):
                    key, value = self.payload(layer, start)
                    runtime.append(
                        (state,), layer, key, value, (self.page_size,)
                    )
                    result = runtime.attend(
                        (state,), layer, self.query(), (1,), phase="decode"
                    )
                    self.assertLessEqual(
                        result.provider_metrics["tiered_selected_pages"], 2
                    )

            batch = runtime.prepare_batch((state,), (1,), 0)
            selected = runtime.selection.select(
                (state,), 0, self.query(), batch
            )
            self.assertEqual(type(selected).__name__, "DeviceSelectedPageView")
            expected_generations = batch.flat_page_generations[
                selected.logical_block_ids.to(torch.long)
            ]
            self.assertTrue(
                torch.equal(selected.expected_generations, expected_generations)
            )
            self.assertTrue(
                torch.equal(
                    selected.expected_generations,
                    selected.current_generations,
                )
            )
            self.assertTrue(bool(selected.valid_mask.all()))
            self.assertEqual(selected.gpu_physical_slots.dtype, torch.int32)
            selected_ids = set(selected.logical_block_ids.tolist())
            cold_logical = next(
                logical
                for logical in range(len(state.block_table.handles))
                if logical not in selected_ids
            )
            cold_handle = state.block_table.handles[cold_logical]
            cold_key = runtime._tier_registered_key(cold_handle)
            locations = runtime.store.location_sets(cold_key)
            if any(item.gpu_slot is not None for item in locations):
                runtime.active_tier.evict(cold_key).wait(timeout_seconds=2.0)
            before = runtime.profile_stats()["prefetch_count"]

            result = runtime.attend(
                (state,), 0, self.query(), (1,), phase="decode"
            )

            after = runtime.profile_stats()["prefetch_count"]
            self.assertEqual(result.provider_metrics["tiered_selected_pages"], 2)
            self.assertEqual(result.provider_metrics["kernel_actual_read_pages"], 2)
            self.assertEqual(result.provider_metrics["temporary_attention_pin_peak"], 1)
            self.assertTrue(result.provider_metrics["prefetch_selected_only"])
            self.assertLessEqual(result.provider_metrics["prefetch_unique_pages"], 2)
            self.assertLessEqual(after - before, 2)
            for location in runtime.store.location_sets(cold_key):
                self.assertIsNone(location.gpu_slot)
                self.assertIsNotNone(location.cpu_slot)
            stats = runtime.profile_stats()
            self.assertEqual(stats["rgkv_pages_selected"] % 2, 0)
            self.assertGreater(stats["rgkv_update_ms"], 0.0)
            self.assertEqual(
                stats["total_ref_count"], stats["logical_owner_count"]
            )
            self.assertGreater(stats["rgkv_host_authority_page_checks"], 0)
            self.assertEqual(stats["rgkv_cpu_sync_count"], 0)
        finally:
            runtime.close()

    def test_device_epoch_mismatch_fails_before_attention(self):
        runtime = self.make_runtime()
        state = runtime.create_request(self.logical_pages * self.page_size)
        try:
            for layer in range(self.layers):
                key, value = self.payload(layer, 0)
                runtime.append(
                    (state,), layer, key, value, (self.page_size,)
                )
            handle = state.block_table.handles[0]
            descriptor = runtime.page_pool.descriptor(handle)
            runtime.store.device_page_table.data_epochs[handle.page_id] = (
                descriptor.data_version - 1
            )
            with self.assertRaisesRegex(RuntimeError, "STALE_INDEX"):
                runtime.attend(
                    (state,), 0, self.query(), (1,), phase="decode"
                )
            self.assertEqual(
                runtime.profile_stats()["rgkv_stale_index_count"], 1
            )
        finally:
            runtime.close()

    def test_device_generation_mismatch_fails_before_attention(self):
        runtime = self.make_runtime()
        state = runtime.create_request(self.logical_pages * self.page_size)
        try:
            for layer in range(self.layers):
                key, value = self.payload(layer, 0)
                runtime.append(
                    (state,), layer, key, value, (self.page_size,)
                )
            handle = state.block_table.handles[0]
            runtime.store.device_page_table.generations[handle.page_id] = (
                handle.generation + 1
            )
            with self.assertRaisesRegex(RuntimeError, "STALE_INDEX"):
                runtime.attend(
                    (state,), 0, self.query(), (1,), phase="decode"
                )
            self.assertEqual(
                runtime.profile_stats()["rgkv_stale_index_count"], 1
            )
        finally:
            runtime.close()

    def test_gpu_topk_failure_precedes_tier_pin_and_prefetch(self):
        runtime = self.make_runtime()
        state = runtime.create_request(self.logical_pages * self.page_size)
        try:
            for layer in range(self.layers):
                key, value = self.payload(layer, 0)
                runtime.append((state,), layer, key, value, (self.page_size,))
            before = runtime.profile_stats()
            original_select = runtime.selection.scorer.select

            def fail_topk(*args, **kwargs):
                raise RuntimeError("injected RGKV GPU top-k failure")

            runtime.selection.scorer.select = fail_topk
            try:
                with self.assertRaisesRegex(RuntimeError, "top-k failure"):
                    runtime.attend(
                        (state,), 0, self.query(), (1,), phase="decode"
                    )
            finally:
                runtime.selection.scorer.select = original_select
            after = runtime.profile_stats()
            self.assertEqual(after["total_pin_count"], 0)
            self.assertEqual(after["prefetch_count"], before["prefetch_count"])
            self.assertEqual(after["pending_tier_operations"], 0)
            self.assertEqual(after["active_prefetch_groups"], 0)
        finally:
            runtime.close()

    def test_page_epoch_failure_removes_stale_active_tier_location(self):
        runtime = self.make_runtime()
        state = runtime.create_request(self.logical_pages * self.page_size)
        key, value = self.payload(0, 0)
        # Complete all but the final layer so failure occurs exactly at the
        # Ownership publication boundary, after Active Tier append slots exist.
        runtime.append((state,), 0, key, value, (self.page_size,))
        original_publish = runtime.page_pool.mark_data_updated

        def fail_epoch(*args, **kwargs):
            raise RuntimeError("injected Active Tier epoch publication failure")

        runtime.page_pool.mark_data_updated = fail_epoch
        try:
            key, value = self.payload(1, 0)
            with self.assertRaisesRegex(RuntimeError, "epoch publication"):
                runtime.append((state,), 1, key, value, (self.page_size,))
        finally:
            runtime.page_pool.mark_data_updated = original_publish
        self.assertEqual(state.lifecycle_state, RequestLifecycleState.RELEASED)
        self.assertEqual(runtime._tier_keys, {})
        self.assertEqual(runtime._tier_pending_appends, {})
        stats = runtime.profile_stats()
        self.assertEqual(stats["registered_logical_pages"], 0)
        self.assertEqual(stats["gpu_kv_used_pages"], 0)
        self.assertEqual(stats["cpu_kv_used_bytes"], 0)
        self.assertEqual(stats["total_pin_count"], 0)
        self.assertEqual(stats["logical_owner_count"], 0)
        self.assertEqual(stats["committed_tokens"], 0)
        runtime.close()

    def test_partial_multi_page_epoch_failure_removes_every_location(self):
        runtime = self.make_runtime()
        state = runtime.create_request(self.logical_pages * self.page_size)
        keys = []
        values = []
        for layer in range(self.layers):
            first_key, first_value = self.payload(layer, 0)
            second_key, second_value = self.payload(layer, self.page_size)
            keys.append(torch.cat((first_key, second_key), dim=0))
            values.append(torch.cat((first_value, second_value), dim=0))
        runtime.append((state,), 0, keys[0], values[0], (2 * self.page_size,))
        original_publish = runtime.page_pool.mark_data_updated
        calls = {"count": 0}

        def fail_second_epoch(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("injected second Page epoch failure")
            return original_publish(*args, **kwargs)

        runtime.page_pool.mark_data_updated = fail_second_epoch
        try:
            with self.assertRaisesRegex(RuntimeError, "second Page epoch"):
                runtime.append(
                    (state,), 1, keys[1], values[1], (2 * self.page_size,)
                )
        finally:
            runtime.page_pool.mark_data_updated = original_publish
        self.assertEqual(calls["count"], 2)
        self.assertEqual(state.lifecycle_state, RequestLifecycleState.RELEASED)
        self.assertEqual(runtime._tier_keys, {})
        self.assertEqual(runtime._tier_pending_appends, {})
        stats = runtime.profile_stats()
        self.assertEqual(stats["registered_logical_pages"], 0)
        self.assertEqual(stats["gpu_kv_used_pages"], 0)
        self.assertEqual(stats["cpu_kv_used_bytes"], 0)
        self.assertEqual(stats["total_ref_count"], 0)
        self.assertEqual(stats["logical_owner_count"], 0)
        self.assertEqual(stats["total_pin_count"], 0)
        self.assertEqual(stats["committed_tokens"], 0)
        runtime.close()


if __name__ == "__main__":
    unittest.main()
