import unittest

from layer_streaming import (
    KVAccuracy,
    KVDataType,
    KVPolicy,
    KVReusePolicy,
    KVSelectionPolicy,
    KVStoragePolicy,
    expand_kv_preset,
    kv_page_pool_bytes,
)


class KVPolicyTest(unittest.TestCase):
    def test_exact_quantized_and_sparse_contracts_are_distinct(self):
        exact = KVPolicy()
        self.assertFalse(exact.d1_support_errors())
        quantized = KVPolicy(
            accuracy=KVAccuracy.QUANTIZED,
            dtype=KVDataType.INT8,
        )
        self.assertIn("D1 implements exact KV only", quantized.d1_support_errors())
        sparse = KVPolicy(
            accuracy=KVAccuracy.SPARSE,
            selection=KVSelectionPolicy.QUEST_FLAT,
        )
        self.assertIn("dense page selection only", sparse.d1_support_errors()[1])

    def test_illegal_accuracy_switches_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "exact KV"):
            KVPolicy(accuracy="exact", dtype="int8")
        with self.assertRaisesRegex(ValueError, "cannot discard"):
            KVPolicy(accuracy="exact", selection="quest_flat")
        with self.assertRaisesRegex(ValueError, "explicit sparse"):
            KVPolicy(accuracy="sparse", selection="none")

    def test_persistent_prefix_requires_nvme_tier(self):
        with self.assertRaisesRegex(ValueError, "persistent prefix"):
            KVPolicy(reuse=KVReusePolicy.PREFIX_PERSISTENT)
        policy = KVPolicy(
            storage=KVStoragePolicy.GPU_CPU_NVME,
            reuse=KVReusePolicy.PREFIX_PERSISTENT,
            cpu_budget_bytes=1024,
            nvme_budget_bytes=4096,
        )
        self.assertEqual(policy.storage, KVStoragePolicy.GPU_CPU_NVME)

    def test_presets_expand_to_full_explicit_policy(self):
        policy = expand_kv_preset("long-context")
        self.assertEqual(policy.accuracy, KVAccuracy.SPARSE)
        self.assertEqual(policy.selection, KVSelectionPolicy.QUEST_FLAT)
        self.assertEqual(set(policy.as_dict()), {
            "accuracy", "storage", "dtype", "selection", "reuse",
            "attention_backend",
            "page_size", "cpu_budget_bytes", "nvme_budget_bytes",
            "page_budget", "recent_window",
        })

    def test_page_pool_calculator_covers_mha_gqa_and_low_precision(self):
        bf16 = kv_page_pool_bytes(
            layer_count=2,
            page_count=3,
            num_key_value_heads=2,
            page_size=16,
            head_dim=8,
            dtype="bf16",
        )
        int8 = kv_page_pool_bytes(
            layer_count=2,
            page_count=3,
            num_key_value_heads=2,
            page_size=16,
            head_dim=8,
            dtype="int8",
        )
        self.assertEqual(bf16, 2 * int8)


if __name__ == "__main__":
    unittest.main()
