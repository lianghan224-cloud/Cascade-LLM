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
        self.assertEqual(policy.selection, KVSelectionPolicy.RGKV)
        self.assertEqual(set(policy.as_dict()), {
            "accuracy", "storage", "dtype", "selection", "reuse",
            "attention_backend",
            "page_size", "gpu_hot_budget_bytes", "cpu_budget_bytes",
            "nvme_budget_bytes", "gpu_migration_slots_bytes",
            "cpu_migration_slots_bytes", "gpu_high_watermark_bytes",
            "gpu_low_watermark_bytes", "cpu_high_watermark_bytes",
            "cpu_low_watermark_bytes",
            "page_budget", "recent_window", "rgkv_scorer",
        })

    def test_tensorized_rgkv_scorer_is_explicit_and_selection_scoped(self):
        policy = KVPolicy(
            accuracy="sparse",
            selection="rgkv",
            rgkv_scorer="torch-tensorized",
        )
        self.assertEqual(policy.rgkv_scorer, "torch_tensorized")
        self.assertFalse(policy.executable_support_errors())
        self.assertEqual(
            policy.require_executable_supported(), policy
        )
        with self.assertRaisesRegex(ValueError, "requires rgkv"):
            KVPolicy(rgkv_scorer="torch_tensorized")
        with self.assertRaisesRegex(ValueError, "unknown RGKV scorer"):
            KVPolicy(
                accuracy="sparse",
                selection="rgkv",
                rgkv_scorer="hidden_fallback",
            )
        replaced = expand_kv_preset(
            "long-context", {"rgkv_scorer": "torch_tensorized"}
        )
        self.assertEqual(replaced.rgkv_scorer, "torch_tensorized")

    def test_tier_policy_fields_are_additive_and_gpu_only_stays_strict(self):
        policy = KVPolicy(
            storage="gpu_cpu",
            gpu_hot_budget_bytes=4096,
            cpu_budget_bytes=8192,
            gpu_migration_slots_bytes=1024,
            cpu_migration_slots_bytes=2048,
            gpu_high_watermark_bytes=4096,
            gpu_low_watermark_bytes=2048,
            cpu_high_watermark_bytes=8192,
            cpu_low_watermark_bytes=4096,
        )
        self.assertEqual(policy.as_dict()["gpu_hot_budget_bytes"], 4096)
        self.assertFalse(policy.executable_support_errors())
        with self.assertRaisesRegex(ValueError, "GPU-only KV"):
            KVPolicy(gpu_hot_budget_bytes=4096)
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            KVPolicy(storage="gpu_cpu", cpu_migration_slots_bytes=-1)

    def test_active_gpu_cpu_executable_gate_is_narrow_and_explicit(self):
        supported = KVPolicy(
            storage="gpu_cpu",
            gpu_hot_budget_bytes=4096,
            cpu_budget_bytes=8192,
        )
        self.assertEqual(supported.require_executable_supported(), supported)
        unsupported = KVPolicy(
            storage="gpu_cpu",
            accuracy="sparse",
            selection="quest_flat",
            gpu_hot_budget_bytes=4096,
            cpu_budget_bytes=8192,
        )
        self.assertEqual(
            unsupported.require_executable_supported(), unsupported
        )
        self.assertEqual(unsupported.selection, KVSelectionPolicy.RGKV)

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
