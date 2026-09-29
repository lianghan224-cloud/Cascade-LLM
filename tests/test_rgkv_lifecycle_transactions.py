import unittest

import torch

from layer_streaming.kv.runtime import PagedKVRuntime
from layer_streaming.kv.types import RequestLifecycleState
from layer_streaming.kv_policy import KVPolicy


class RgkvLifecycleTransactionTest(unittest.TestCase):
    def make_runtime(self, layers=1, pages=8):
        return PagedKVRuntime(
            layer_count=layers,
            num_query_heads=2,
            num_kv_heads=1,
            head_dim=4,
            page_count=pages,
            page_size=16,
            dtype=torch.float32,
            device="cpu",
            policy=KVPolicy(
                accuracy="sparse",
                storage="gpu",
                dtype="bf16",
                selection="rgkv",
                reuse="request_only",
                attention_backend="reference_paged_exact",
                page_size=16,
                page_budget=2,
                recent_window=16,
                rgkv_scorer="cpu_reference",
            ),
            allow_reference=True,
        )

    @staticmethod
    def payload(count, offset=0):
        return torch.arange(
            offset * 4,
            (offset + count) * 4,
            dtype=torch.float32,
        ).reshape(count, 1, 4)

    def assert_summary_epochs(self, runtime, state):
        for logical, handle in enumerate(state.block_table.handles):
            descriptor = runtime.page_pool.descriptor(handle)
            self.assertEqual(descriptor.index_version, descriptor.data_version)
            self.assertEqual(
                len(descriptor.index_metadata_handle), runtime.layer_count
            )
            for layer, summary in enumerate(descriptor.index_metadata_handle):
                self.assertEqual(summary.logical_page_id, logical)
                self.assertEqual(summary.data_epoch, descriptor.data_version)
                self.assertIs(
                    runtime.selection.records[
                        (state.request_id, layer, logical)
                    ],
                    summary,
                )

    def test_repeated_append_and_rollback_keep_page_and_summary_epochs_equal(self):
        runtime = self.make_runtime(layers=2)
        state = runtime.create_request(32)
        try:
            position = 0
            for count in (1, 2, 1, 3, 1):
                payload = self.payload(count, position)
                for layer in range(runtime.layer_count):
                    runtime.append((state,), layer, payload + layer, payload, (count,))
                position += count
                self.assert_summary_epochs(runtime, state)
            runtime.rollback(state, 5)
            self.assert_summary_epochs(runtime, state)
            self.assertEqual(
                runtime.profile_stats()["total_ref_count"],
                runtime.profile_stats()["logical_owner_count"],
            )
        finally:
            runtime.close()

    def test_layer_failure_discards_staged_summaries_and_all_owners(self):
        runtime = self.make_runtime(layers=2)
        state = runtime.create_request(32)
        payload = self.payload(2)
        original = runtime.kv_kernel_backend.append_kv
        calls = {"count": 0}

        def fail_second(request):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("injected RGKV layer failure")
            return original(request)

        runtime.kv_kernel_backend.append_kv = fail_second
        runtime.append((state,), 0, payload, payload, (2,))
        with self.assertRaisesRegex(RuntimeError, "injected RGKV"):
            runtime.append((state,), 1, payload, payload, (2,))
        self.assertIsNone(state.pending_append)
        self.assertEqual(state.sequence_length, 0)
        self.assertEqual(runtime.selection.records, {})
        self.assertEqual(runtime.selection.indexes, {})
        self.assertEqual(runtime.page_pool.allocated_pages, 0)
        self.assertTrue(runtime.page_pool.validate_invariants())
        runtime.close()

    def test_summary_publication_failure_invalidates_committed_request(self):
        runtime = self.make_runtime()
        state = runtime.create_request(32)
        payload = self.payload(2)
        publish = runtime.selection.commit_append

        def publish_then_fail(*args, **kwargs):
            publish(*args, **kwargs)
            raise RuntimeError("injected RGKV publication failure")

        runtime.selection.commit_append = publish_then_fail
        with self.assertRaisesRegex(RuntimeError, "publication failure"):
            runtime.append((state,), 0, payload, payload, (2,))
        self.assertEqual(state.lifecycle_state, RequestLifecycleState.RELEASED)
        self.assertEqual(runtime.selection.records, {})
        self.assertEqual(runtime.selection.indexes, {})
        self.assertEqual(runtime.page_pool.allocated_pages, 0)
        self.assertEqual(runtime.profile_stats()["logical_owner_count"], 0)
        self.assertTrue(runtime.page_pool.validate_invariants())
        runtime.close()

    def test_summary_build_failure_aborts_before_epoch_publication(self):
        runtime = self.make_runtime()
        state = runtime.create_request(32)
        payload = self.payload(2)
        original_stage = runtime.selection.stage_append_layer

        def fail_stage(*args, **kwargs):
            raise RuntimeError("injected RGKV summary build failure")

        runtime.selection.stage_append_layer = fail_stage
        try:
            with self.assertRaisesRegex(RuntimeError, "summary build failure"):
                runtime.append((state,), 0, payload, payload, (2,))
        finally:
            runtime.selection.stage_append_layer = original_stage
        self.assertIsNone(state.pending_append)
        self.assertEqual(state.sequence_length, 0)
        self.assertEqual(runtime.selection.records, {})
        self.assertEqual(runtime.selection.indexes, {})
        stats = runtime.profile_stats()
        self.assertEqual(stats["total_pin_count"], 0)
        self.assertEqual(stats["total_ref_count"], stats["logical_owner_count"])
        self.assertEqual(runtime.page_pool.allocated_pages, 0)
        runtime.close()

    def test_append_fence_drain_failure_aborts_staging_and_transaction(self):
        runtime = self.make_runtime()
        state = runtime.create_request(32)
        payload = self.payload(2)
        original_drain = runtime.ownership.drain_handles
        calls = {"count": 0}

        def fail_first_drain(handles):
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("injected append Fence drain failure")
            return original_drain(handles)

        runtime.ownership.drain_handles = fail_first_drain
        try:
            with self.assertRaisesRegex(RuntimeError, "Fence drain failure"):
                runtime.append((state,), 0, payload, payload, (2,))
        finally:
            runtime.ownership.drain_handles = original_drain
        self.assertGreaterEqual(calls["count"], 2)
        self.assertIsNone(state.pending_append)
        self.assertEqual(state.sequence_length, 0)
        self.assertEqual(state.layer_lengths, [0])
        self.assertEqual(runtime.selection.records, {})
        self.assertEqual(runtime.selection.indexes, {})
        stats = runtime.profile_stats()
        self.assertEqual(stats["total_pin_count"], 0)
        self.assertEqual(stats["total_ref_count"], stats["logical_owner_count"])
        self.assertEqual(runtime.page_pool.allocated_pages, 0)
        runtime.close()

    def test_page_epoch_publication_failure_invalidates_request(self):
        runtime = self.make_runtime()
        state = runtime.create_request(32)
        payload = self.payload(2)
        original_publish = runtime.page_pool.mark_data_updated

        def fail_epoch(*args, **kwargs):
            raise RuntimeError("injected Page Data Epoch publication failure")

        runtime.page_pool.mark_data_updated = fail_epoch
        try:
            with self.assertRaisesRegex(RuntimeError, "Epoch publication"):
                runtime.append((state,), 0, payload, payload, (2,))
        finally:
            runtime.page_pool.mark_data_updated = original_publish
        self.assertEqual(state.lifecycle_state, RequestLifecycleState.RELEASED)
        self.assertIsNone(state.pending_append)
        self.assertEqual(runtime.selection.records, {})
        self.assertEqual(runtime.selection.indexes, {})
        self.assertEqual(runtime.page_pool.allocated_pages, 0)
        stats = runtime.profile_stats()
        self.assertEqual(stats["total_pin_count"], 0)
        self.assertEqual(stats["total_ref_count"], stats["logical_owner_count"])
        self.assertEqual(stats["committed_tokens"], 0)
        runtime.close()

    def test_fork_cow_rollback_release_1000_cycles_has_no_index_drift(self):
        runtime = self.make_runtime(pages=6)
        parent = runtime.create_request(32, request_id=100)
        initial = self.payload(2)
        runtime.append((parent,), 0, initial, initial, (2,))
        baseline_pages = runtime.page_pool.allocated_pages
        baseline_index_bytes = runtime.selection.stats()["index_bytes"]
        for cycle in range(1000):
            branch = runtime.fork(parent, request_id=1000 + cycle)
            token = self.payload(1, cycle + 2)
            runtime.append((branch,), 0, token, token, (1,))
            runtime.rollback(branch, parent.sequence_length)
            runtime.release(branch)
            self.assertEqual(runtime.page_pool.allocated_pages, baseline_pages)
        self.assert_summary_epochs(runtime, parent)
        self.assertEqual(runtime.selection.stats()["index_bytes"], baseline_index_bytes)
        stats = runtime.profile_stats()
        self.assertEqual(stats["total_ref_count"], stats["logical_owner_count"])
        self.assertEqual(stats["total_pin_count"], 0)
        for descriptor in runtime.page_pool.descriptors:
            self.assertEqual(descriptor.inflight_compute, 0)
            self.assertEqual(descriptor.inflight_io, 0)
        runtime.close()


if __name__ == "__main__":
    unittest.main()
