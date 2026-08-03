import unittest

import torch

from layer_streaming.kv import PagedKVRuntime
from layer_streaming.kv_policy import (
    KVAccuracy,
    KVDataType,
    KVPolicy,
    KVSelectionPolicy,
)
from tools.kv_validation_cases import (
    run_index_suite,
    run_lifecycle_suite,
    run_routing_suite,
    run_scheduler_suite,
    run_sharing_suite,
    run_tier_suite,
)
from layer_streaming.kv.stores import KVTier, TieredKVStore


class KVRemediationLogicTest(unittest.TestCase):
    def test_lifecycle_faults_and_random_state_machine(self):
        result = run_lifecycle_suite(operations=5000)
        self.assertGreaterEqual(result["V14"]["random_operations"], 5000)
        self.assertEqual(result["V15"]["minimized_repro"], ["unpin"])

    def test_sharing_index_tiering_scheduler_and_routing(self):
        for suite, expected in (
            (run_sharing_suite, 11),
            (run_index_suite, 12),
            (run_tier_suite, 13),
            (run_scheduler_suite, 4),
            (run_routing_suite, 4),
        ):
            with self.subTest(suite=suite.__name__):
                result = suite()
                self.assertEqual(len(result), expected)

    def test_quest_runtime_builds_real_records_and_obeys_budget(self):
        policy = KVPolicy(
            accuracy=KVAccuracy.SPARSE,
            dtype=KVDataType.BF16,
            selection=KVSelectionPolicy.QUEST_FLAT,
            attention_backend="reference_paged_exact",
            page_size=16,
            page_budget=2,
            recent_window=16,
        )
        runtime = PagedKVRuntime(
            layer_count=1,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=8,
            page_count=8,
            page_size=16,
            dtype=torch.bfloat16,
            device="cpu",
            policy=policy,
            allow_reference=True,
        )
        state = runtime.create_request(128)
        torch.manual_seed(101)
        key = torch.randn(33, 2, 8, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        runtime.append((state,), 0, key, value, (33,))
        self.assertEqual(len(runtime.selection.records), 3)
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        output = runtime.attend(
            (state,), 0, query, (1,), phase="decode"
        ).output
        self.assertTrue(torch.isfinite(output).all())
        stats = runtime.selection.index.stats()
        self.assertEqual(stats["candidates"], 3)
        self.assertEqual(stats["selected"], 2)
        branch = runtime.fork(state)
        extension = torch.randn(2, 2, 8, dtype=torch.bfloat16)
        runtime.append((branch,), 0, extension, extension, (2,))
        runtime.rollback(branch, 33)
        self.assertEqual(state.sequence_length, 33)
        self.assertEqual(branch.sequence_length, 33)
        runtime.close()

    def test_tiered_replace_and_move_do_not_delete_new_authority(self):
        with TieredKVStore() as store:
            store.put("replace", b"old", KVTier.GPU, version=1)
            store.put("replace", b"new", KVTier.GPU, version=2)
            self.assertEqual(store.get("replace"), b"new")
            store.migrate(
                "replace", KVTier.CPU, keep_source=False
            )
            self.assertEqual(store.get("replace"), b"new")
            self.assertFalse(store.is_resident("replace", KVTier.GPU))
            self.assertEqual(
                store.record("replace").authoritative_tier, KVTier.CPU
            )


if __name__ == "__main__":
    unittest.main()
