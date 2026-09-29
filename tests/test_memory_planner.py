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
from layer_streaming.attention.paged import (
    PagedWorkspaceEstimate,
    ReferencePagedExactBackend,
)
from layer_streaming.kv.selection import RGKVGPUScorer


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
        self.assertEqual(estimate.kv_total_context_bytes, expected)
        self.assertEqual(estimate.kv_gpu_cache_capacity_bytes, expected)
        self.assertEqual(estimate.kv_cpu_pinned_backing_bytes, 0)
        self.assertEqual(estimate.kv_gpu_migration_slots_bytes, 0)
        self.assertEqual(estimate.kv_cpu_migration_slots_bytes, 0)
        self.assertEqual(
            estimate.gpu_weight_budget_bytes,
            estimate.gpu_transfer_slots_bytes
            + estimate.resident_parameters_bytes,
        )
        self.assertEqual(
            estimate.kv_admission_required_bytes,
            estimate.kv_total_context_bytes,
        )
        self.assertEqual(
            estimate.kv_admission_capacity_bytes,
            estimate.kv_total_context_bytes,
        )
        self.assertEqual(estimate.kv_admission_headroom_bytes, 0)
        self.assertEqual(estimate.kv_decode_attention_workspace_bytes, 0)

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

    def test_deterministic_topk_sort_workspace_is_explicit(self):
        planner = self.make_planner(top_k=10)
        estimate = planner.estimate()
        expected = 10 * (4 + 8) + planner.plan.vocab.chunk_rows * 48
        self.assertEqual(estimate.topk_bytes, expected)

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
                gpu_migration_slots_bytes=2048,
                cpu_migration_slots_bytes=2048,
            )
        ).estimate()
        self.assertEqual(tiered_sparse.kv_cpu_pool_bytes, 8192)
        self.assertEqual(tiered_sparse.kv_nvme_budget_bytes, 16384)
        self.assertGreater(tiered_sparse.kv_index_bytes, 0)
        self.assertGreaterEqual(
            tiered_sparse.pinned_staging_bytes,
            bf16.pinned_staging_bytes + 8192,
        )

    def test_gpu_cpu_tier_admission_uses_explicit_byte_capacities(self):
        dense = self.make_planner().estimate()
        page_bytes = dense.kv_page_bytes
        tiered = self.make_planner(
            kv_policy=KVPolicy(
                storage="gpu_cpu",
                page_size=16,
                gpu_hot_budget_bytes=page_bytes,
                cpu_budget_bytes=dense.kv_total_context_bytes - page_bytes,
                gpu_migration_slots_bytes=2048,
                cpu_migration_slots_bytes=4096,
                gpu_high_watermark_bytes=page_bytes,
                gpu_low_watermark_bytes=0,
                cpu_high_watermark_bytes=(
                    dense.kv_total_context_bytes - page_bytes
                ),
                cpu_low_watermark_bytes=page_bytes,
            )
        ).estimate()
        self.assertEqual(
            tiered.kv_total_context_bytes, dense.kv_total_context_bytes
        )
        self.assertEqual(tiered.kv_gpu_cache_capacity_bytes, page_bytes)
        self.assertEqual(
            tiered.kv_cpu_pinned_backing_bytes,
            dense.kv_total_context_bytes - page_bytes,
        )
        self.assertEqual(tiered.kv_gpu_migration_slots_bytes, 2048)
        self.assertEqual(tiered.kv_cpu_migration_slots_bytes, 4096)
        self.assertEqual(tiered.kv_layer_page_bytes, 2048)
        self.assertEqual(tiered.kv_gpu_migration_slot_count, 1)
        self.assertEqual(tiered.kv_cpu_migration_slot_count, 2)
        self.assertEqual(
            tiered.kv_admission_capacity_bytes,
            dense.kv_total_context_bytes,
        )
        self.assertEqual(tiered.kv_admission_headroom_bytes, 0)
        self.assertEqual(
            tiered.estimated_gpu_peak_bytes,
            dense.estimated_gpu_peak_bytes
            - dense.kv_total_context_bytes
            + page_bytes
            + 2048
            + tiered.kv_device_page_table_bytes,
        )
        self.assertEqual(
            tiered.pinned_staging_bytes,
            dense.pinned_staging_bytes
            + dense.kv_total_context_bytes
            - page_bytes
            + 4096,
        )

    def test_gpu_cpu_tier_rejects_insufficient_total_capacity(self):
        dense = self.make_planner().estimate()
        with self.assertRaisesRegex(ValueError, "below total context"):
            self.make_planner(
                kv_policy=KVPolicy(
                    storage="gpu_cpu",
                    page_size=16,
                    gpu_hot_budget_bytes=dense.kv_page_bytes,
                    cpu_budget_bytes=(
                        dense.kv_total_context_bytes
                        - dense.kv_page_bytes
                        - 1
                    ),
                    gpu_migration_slots_bytes=2048,
                    cpu_migration_slots_bytes=2048,
                )
            ).estimate()

    def test_tier_migration_slots_are_complete_layer_pages(self):
        dense = self.make_planner().estimate()
        common = {
            "storage": "gpu_cpu",
            "page_size": 16,
            "gpu_hot_budget_bytes": dense.kv_page_bytes,
            "cpu_budget_bytes": dense.kv_total_context_bytes,
        }
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.make_planner(kv_policy=KVPolicy(**common)).estimate()
        with self.assertRaisesRegex(ValueError, "multiple"):
            self.make_planner(
                kv_policy=KVPolicy(
                    **common,
                    gpu_migration_slots_bytes=2049,
                    cpu_migration_slots_bytes=2048,
                )
            ).estimate()

    def test_tensorized_rgkv_index_and_workspace_are_provider_estimated(self):
        planner = self.make_planner(
            kv_policy=KVPolicy(
                accuracy="sparse",
                selection="quest_flat",
                page_size=16,
                page_budget=2,
                recent_window=16,
                quest_scorer="torch_tensorized",
            )
        )
        estimate = planner.estimate()
        pages = estimate.kv_page_count
        dimensions = (
            planner.geometry.num_key_value_heads * planner.geometry.head_dim
        )
        per_layer = RGKVGPUScorer.estimate_workspace(
            pages, dimensions
        )
        self.assertEqual(estimate.rgkv_cpu_reference_index_bytes, 0)
        self.assertEqual(
            estimate.rgkv_gpu_index_bytes,
            planner.geometry.num_hidden_layers * per_layer.index_bytes,
        )
        self.assertEqual(
            estimate.rgkv_scoring_workspace_bytes
            + estimate.rgkv_topk_workspace_bytes,
            per_layer.temporary_bytes,
        )
        self.assertEqual(
            estimate.kv_index_bytes,
            estimate.rgkv_cpu_reference_index_bytes
            + estimate.rgkv_gpu_index_bytes,
        )
        self.assertEqual(estimate.kv_selected_metadata_bytes, 2 * 64)

    def test_tier_watermarks_are_byte_based_and_bounded(self):
        dense = self.make_planner().estimate()
        common = {
            "storage": "gpu_cpu",
            "page_size": 16,
            "gpu_hot_budget_bytes": dense.kv_page_bytes,
            "cpu_budget_bytes": dense.kv_total_context_bytes,
            "gpu_migration_slots_bytes": 2048,
            "cpu_migration_slots_bytes": 2048,
        }
        with self.assertRaisesRegex(ValueError, "below its high"):
            self.make_planner(
                kv_policy=KVPolicy(
                    **common,
                    gpu_high_watermark_bytes=dense.kv_page_bytes,
                    gpu_low_watermark_bytes=dense.kv_page_bytes,
                )
            ).estimate()
        with self.assertRaisesRegex(ValueError, "exceeds tier capacity"):
            self.make_planner(
                kv_policy=KVPolicy(
                    **common,
                    cpu_high_watermark_bytes=(
                        dense.kv_total_context_bytes + 1
                    ),
                )
            ).estimate()

    def test_gather_sdpa_prefill_workspace_is_admitted(self):
        planner = self.make_planner(
            kv_prefill_backend="gather_sdpa_prefill"
        )
        estimate = planner.estimate()
        groups = (
            planner.geometry.num_attention_heads
            // planner.geometry.num_key_value_heads
        )
        gather_only = (
            estimate.kv_cache_bytes
            // planner.geometry.num_hidden_layers
        ) * (1 + groups)
        # The Provider contract also covers output, worst-case chunked mask
        # and positions, plus its local SDPA/allocator guard.  The Planner
        # must not budget only the gathered and GQA-expanded K/V tensors.
        self.assertGreater(
            estimate.kv_attention_workspace_bytes,
            gather_only + 256 * 1024,
        )
        reference = self.make_planner().estimate()
        self.assertEqual(reference.kv_attention_workspace_bytes, 0)
        self.assertEqual(
            estimate.estimated_gpu_peak_bytes
            - reference.estimated_gpu_peak_bytes,
            estimate.kv_attention_workspace_bytes,
        )

    def test_reserved_pages_equal_to_pool_are_rejected_by_planner(self):
        with self.assertRaisesRegex(ValueError, "smaller than the page pool"):
            self.make_planner(kv_reserved_free_pages=3).estimate()

    def test_workspace_budget_is_driven_by_provider_shape_interface(self):
        class FixedWorkspaceProvider(ReferencePagedExactBackend):
            name = "fixed_workspace_test"

            def __init__(self):
                self.seen_shape = None

            def estimate_workspace_shape(self, shape):
                self.validate_shape(shape)
                self.seen_shape = shape
                return PagedWorkspaceEstimate(
                    12345, "test", False, False, False
                )

        provider = FixedWorkspaceProvider()
        estimate = self.make_planner(
            kv_prefill_backend=provider.name,
            kv_prefill_provider=provider,
        ).estimate()
        self.assertEqual(estimate.kv_attention_workspace_bytes, 12345)
        self.assertEqual(provider.seen_shape.max_sequence_length, 48)
        self.assertEqual(provider.seen_shape.page_size, 16)


if __name__ == "__main__":
    unittest.main()
