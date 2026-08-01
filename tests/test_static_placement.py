import unittest

from layer_streaming import (
    ExecutionPolicy,
    LlamaModelAdapter,
    MemoryPlanner,
    build_static_transformer_placement,
)

from test_plan import tiny_config


class StaticTransformerPlacementTest(unittest.TestCase):
    def make_plan(self, granularity="matrix_group", layers=3):
        config = tiny_config(num_hidden_layers=layers)
        return LlamaModelAdapter().build_execution_plan(
            config,
            ExecutionPolicy(
                granularity=granularity,
                vocab_chunk_bytes=4096,
            ),
        )

    def test_zero_budget_preserves_frozen_execution_plan(self):
        plan = self.make_plan()
        before = plan.as_dict()
        placement = build_static_transformer_placement(plan, 0)
        self.assertEqual(placement.resident_layer_ids, ())
        self.assertEqual(len(placement.streamed_unit_ids), len(plan.units))
        self.assertEqual(placement.resident_weight_bytes, 0)
        self.assertEqual(plan.as_dict(), before)

    def test_budget_selects_only_complete_prefix_layers(self):
        plan = self.make_plan(layers=3)
        all_resident = build_static_transformer_placement(plan, 1 << 60)
        first_layer_bytes = max(
            item.device_offset + item.storage_bytes
            for item in all_resident.resident_tensors
            if item.weight_name.startswith("model.layers.0.")
        ) - plan.resident_bytes
        none = build_static_transformer_placement(
            plan, first_layer_bytes - 1
        )
        one = build_static_transformer_placement(plan, first_layer_bytes)
        self.assertEqual(none.resident_layer_ids, ())
        self.assertEqual(one.resident_layer_ids, (0,))
        self.assertTrue(
            all(name.startswith("layer_0000.") for name in one.resident_unit_ids)
        )
        self.assertAlmostEqual(
            one.resident_hit_ratio,
            one.resident_weight_bytes
            / (one.resident_weight_bytes + one.streamed_weight_bytes),
        )

    def test_layer_choice_is_independent_of_transfer_granularity(self):
        results = []
        for granularity in ("matrix", "matrix_group", "layer"):
            plan = self.make_plan(granularity=granularity, layers=2)
            placement = build_static_transformer_placement(plan, 1 << 60)
            results.append(
                (
                    placement.resident_layer_ids,
                    placement.resident_weight_bytes,
                    placement.streamed_weight_bytes,
                )
            )
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_negative_budget_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            build_static_transformer_placement(self.make_plan(), -1)

    def test_memory_planner_budgets_selected_transformer_arena(self):
        plan = self.make_plan(layers=2)
        placement = build_static_transformer_placement(plan, 1 << 60)
        estimate = MemoryPlanner(
            plan,
            plan.geometry,
            max_context=16,
            max_prefill_tokens=8,
            transformer_placement=placement,
        ).estimate()
        self.assertEqual(
            estimate.gpu_resident_transformer_bytes,
            placement.resident_weight_bytes,
        )
        self.assertEqual(
            estimate.resident_parameters_bytes,
            placement.resident_arena_bytes,
        )


if __name__ == "__main__":
    unittest.main()
