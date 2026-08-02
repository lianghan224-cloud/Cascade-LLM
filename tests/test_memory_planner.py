import unittest

from layer_streaming import (
    ExecutionPolicy,
    LlamaModelAdapter,
    MemoryPlanner,
    KVPolicy,
    PlacementMode,
    SystemCapacity,
)

from test_plan import tiny_config


class MemoryPlannerTest(unittest.TestCase):
    def make_planner(self, **kwargs):
        adapter = LlamaModelAdapter()
        config = tiny_config(num_hidden_layers=2, vocab_size=100)
        geometry = adapter.build_geometry(config)
        policy = ExecutionPolicy(
            cpu_weight_mode="pinned_staging",
            embedding_mode=PlacementMode.STREAMED,
            lm_head_mode=PlacementMode.STREAMED,
            vocab_chunk_bytes=4096,
        )
        plan = adapter.build_plan(config, policy)
        return MemoryPlanner(
            plan,
            geometry,
            policy=policy,
            max_context=33,
            max_prefill_tokens=8,
            kv_block_size=16,
            cuda_safety_margin_bytes=0,
            **kwargs
        )

    def test_kv_budget_rounds_to_blocks(self):
        planner = self.make_planner()
        estimate = planner.estimate()
        expected = 2 * 2 * 1 * 2 * 48 * 16 * 2
        self.assertEqual(estimate.kv_cache_bytes, expected)
        self.assertGreater(estimate.pinned_staging_bytes, 0)
        self.assertGreater(estimate.estimated_gpu_peak_bytes, expected)

    def test_batch_kv_page_count_and_descriptor_budget_are_explicit(self):
        single = self.make_planner().estimate()
        batched = self.make_planner(
            batch_size=3,
            kv_reserved_free_pages=2,
        ).estimate()
        self.assertEqual(batched.kv_page_count, 3 * single.kv_page_count)
        self.assertEqual(batched.kv_page_bytes, single.kv_page_bytes)
        self.assertEqual(batched.kv_cache_bytes, 3 * single.kv_cache_bytes)
        self.assertEqual(
            batched.kv_reserved_page_bytes,
            2 * single.kv_page_bytes,
        )
        pages_per_request = 3
        expected_metadata = (
            3 * pages_per_request * 4
            + (3 + 1) * 4 * 2
            + 3 * 4 * 3
            + 3 * 8 * 2 * 4
        )
        self.assertEqual(batched.kv_block_table_bytes, expected_metadata)

    def test_preflight_reports_all_capacity_failures(self):
        planner = self.make_planner()
        result = planner.preflight(
            capacity=SystemCapacity(
                cpu_available_bytes=1,
                memlock_limit_bytes=1,
                gpu_free_bytes=1,
                gpu_total_bytes=1,
            ),
            raise_on_error=False,
        )
        self.assertFalse(result.ok)
        self.assertEqual(len(result.errors), 3)
        self.assertTrue(result.suggestions)

    def test_full_logits_are_explicitly_budgeted(self):
        planner = self.make_planner(return_full_logits=True, logits_tokens=8)
        estimate = planner.estimate()
        self.assertEqual(estimate.full_logits_bytes, 1 * 8 * 100 * 4)
        self.assertEqual(estimate.lm_head_buffer_bytes, 0)

    def test_kv_dtype_tiers_and_index_are_budgeted_independently(self):
        bf16 = self.make_planner().estimate()
        int8 = self.make_planner(
            kv_policy=KVPolicy(
                accuracy="quantized",
                dtype="int8",
                page_size=16,
            )
        ).estimate()
        self.assertEqual(bf16.kv_gpu_pool_bytes, 2 * int8.kv_gpu_pool_bytes)

        tiered_sparse = self.make_planner(
            kv_policy=KVPolicy(
                accuracy="sparse",
                storage="gpu_cpu_nvme",
                dtype="bf16",
                selection="quest_flat",
                reuse="prefix_persistent",
                page_size=16,
                cpu_budget_bytes=8192,
                nvme_budget_bytes=16384,
            )
        ).estimate()
        self.assertEqual(tiered_sparse.kv_cpu_pool_bytes, 8192)
        self.assertEqual(tiered_sparse.kv_nvme_budget_bytes, 16384)
        self.assertGreater(tiered_sparse.kv_index_bytes, 0)
        self.assertGreaterEqual(
            tiered_sparse.pinned_staging_bytes,
            bf16.pinned_staging_bytes + 8192,
        )


if __name__ == "__main__":
    unittest.main()
