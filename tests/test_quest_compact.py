import unittest

import torch

from layer_streaming.kv import PagedKVRuntime
from layer_streaming.kv_policy import (
    KVAccuracy,
    KVDataType,
    KVPolicy,
    KVSelectionPolicy,
)


class RGKVCompactRuntimeTest(unittest.TestCase):
    def make_runtime(self):
        return PagedKVRuntime(
            layer_count=1,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=8,
            page_count=8,
            page_size=16,
            dtype=torch.bfloat16,
            device="cpu",
            policy=KVPolicy(
                accuracy=KVAccuracy.SPARSE,
                dtype=KVDataType.BF16,
                selection=KVSelectionPolicy.QUEST_FLAT,
                attention_backend="reference_paged_exact",
                page_size=16,
                page_budget=2,
                recent_window=16,
            ),
            allow_reference=True,
        )

    def test_runtime_records_are_compact_and_tail_update_reads_no_history(self):
        runtime = self.make_runtime()
        state = runtime.create_request(64)
        key = torch.randn(33, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (33,))
        records = tuple(runtime.selection.records.values())
        self.assertEqual(len(records), 3)
        self.assertTrue(all(tuple(item.values.shape) == (3, 2, 8) for item in records))
        self.assertGreater(runtime.selection.stats()["index_bytes"], 0)

        calls = []
        original = runtime.store.read_pages

        def capture(layer, page_ids, stream=None):
            calls.append(tuple(page_ids))
            return original(layer, page_ids, stream=stream)

        runtime.store.read_pages = capture
        extension = torch.randn(1, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, extension, extension, (1,))
        self.assertEqual(calls, [])
        runtime.close()

    def test_recent_is_inside_total_budget(self):
        runtime = self.make_runtime()
        state = runtime.create_request(64)
        key = torch.randn(33, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (33,))
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        batch = runtime.prepare_batch((state,), (1,), 0)
        selected = runtime.selection.select((state,), 0, query, batch)
        stats = selected.metadata["requests"][0]
        self.assertEqual(stats["relevance_selected_pages"], 1)
        self.assertEqual(stats["mandatory_recent_pages"], 1)
        self.assertEqual(stats["total_page_budget"], 2)
        self.assertEqual(int(selected.flat_page_ids.numel()), 2)
        runtime.close()

    def test_page_metadata_holds_complete_rgkv_layer_bundle(self):
        runtime = self.make_runtime()
        state = runtime.create_request(32)
        key = torch.randn(5, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (5,))
        handle = state.block_table.handles[0]
        descriptor = runtime.page_pool.descriptor(handle)
        self.assertEqual(len(descriptor.index_metadata_handle), 1)
        record = descriptor.index_metadata_handle[0]
        self.assertEqual(tuple(record.values.shape), (3, 2, 8))
        self.assertEqual(record.valid_tokens, 5)
        runtime.close()

    def test_rollback_index_uses_global_page_data_epoch(self):
        runtime = self.make_runtime()
        state = runtime.create_request(32)
        key = torch.randn(9, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (9,))
        previous_epoch = runtime.ownership._data_epoch

        runtime.rollback(state, 5)

        handle = state.block_table.handles[0]
        descriptor = runtime.page_pool.descriptor(handle)
        record = runtime.selection.records[(state.request_id, 0, 0)]
        self.assertGreater(descriptor.data_version, previous_epoch)
        self.assertEqual(record.data_epoch, descriptor.data_version)
        self.assertEqual(descriptor.index_version, descriptor.data_version)
        runtime.close()


if __name__ == "__main__":
    unittest.main()
