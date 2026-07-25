import unittest

from layer_streaming.plan import (
    Granularity,
    MIB,
    build_llama31_8b_plan,
)


class Llama31PlanTest(unittest.TestCase):
    def test_matrix_is_default_and_uses_largest_projection(self):
        plan = build_llama31_8b_plan()
        self.assertEqual(plan.granularity, Granularity.MATRIX)
        self.assertEqual(len(plan.units), 32 * 7)
        self.assertEqual(plan.slot_bytes, 112 * MIB)
        self.assertEqual(plan.two_slot_bytes, 224 * MIB)
        self.assertEqual(
            plan.stream_bytes_per_token,
            32 * 416 * MIB,
        )

    def test_layer_plan_allocates_two_whole_layer_slots(self):
        plan = build_llama31_8b_plan("layer")
        self.assertEqual(len(plan.units), 32)
        self.assertEqual(plan.slot_bytes, 416 * MIB)
        self.assertEqual(plan.two_slot_bytes, 832 * MIB)

    def test_separate_embedding_and_head_are_resident(self):
        plan = build_llama31_8b_plan("matrix")
        resident_keys = {item.tensor.key for item in plan.resident}
        self.assertIn("model.embed_tokens.weight", resident_keys)
        self.assertIn("lm_head.weight", resident_keys)
        self.assertEqual(plan.resident_bytes, int(2004.5078125 * MIB))
        self.assertEqual(plan.aliases, {})

    def test_tied_head_alias_does_not_duplicate_gpu_memory(self):
        untied = build_llama31_8b_plan(
            "matrix",
            tie_word_embeddings=False,
        )
        tied = build_llama31_8b_plan(
            "matrix",
            tie_word_embeddings=True,
        )
        self.assertEqual(
            tied.aliases["lm_head.weight"],
            "model.embed_tokens.weight",
        )
        self.assertEqual(
            untied.resident_bytes - tied.resident_bytes,
            1002 * MIB,
        )

    def test_matrix_plan_saves_608_mib_of_device_slots(self):
        layer = build_llama31_8b_plan("layer")
        matrix = build_llama31_8b_plan("matrix")
        saved = layer.two_slot_bytes - matrix.two_slot_bytes
        self.assertEqual(saved, 608 * MIB)
        self.assertAlmostEqual(
            1.0 - matrix.two_slot_bytes / layer.two_slot_bytes,
            0.7307692307692308,
        )


if __name__ == "__main__":
    unittest.main()
