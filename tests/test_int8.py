import unittest

from layer_streaming import build_llama31_70b_int8_plan


class Llama31Int8PlanTest(unittest.TestCase):
    def test_checkpoint_payload_and_tensor_count(self):
        plan = build_llama31_70b_int8_plan("matrix")
        self.assertEqual(len(plan.tensors), 1283)
        self.assertEqual(plan.host_arena_bytes, 72669806592)
        self.assertEqual(plan.stream_bytes_per_token, 68464476160)

    def test_matrix_plan_uses_weight_plus_scale_units(self):
        plan = build_llama31_70b_int8_plan("matrix")
        self.assertEqual(len(plan.units), 80 * 7)
        self.assertEqual(plan.transfer_slot_bytes, 234938368)
        self.assertEqual(plan.dequant_workspace_bytes, 469762048)
        first = plan.units[0]
        self.assertEqual(first.pieces[0].tensor.dtype, "int8")
        self.assertEqual(first.pieces[1].tensor.dtype, "bfloat16")
        self.assertTrue(first.pieces[1].tensor.key.endswith("weight_scale"))

    def test_granularity_tradeoff(self):
        matrix = build_llama31_70b_int8_plan("matrix")
        group = build_llama31_70b_int8_plan("matrix_group")
        layer = build_llama31_70b_int8_plan("layer")
        self.assertEqual(len(group.units), 80 * 4)
        self.assertEqual(len(layer.units), 80)
        self.assertLess(matrix.slot_bytes, group.slot_bytes)
        self.assertLess(group.slot_bytes, layer.slot_bytes)
        self.assertEqual(
            matrix.stream_bytes_per_token,
            layer.stream_bytes_per_token,
        )

    def test_vocab_is_streamed_and_norms_are_resident(self):
        plan = build_llama31_70b_int8_plan("matrix")
        self.assertEqual(plan.vocab.chunk_bytes, 128 * 1024**2)
        self.assertEqual(plan.vocab.chunk_count, 16)
        self.assertEqual(len(plan.host_only), 2)
        self.assertEqual(len(plan.resident), 80 * 2 + 1)
        self.assertEqual(plan.resident_bytes, 2637824)


if __name__ == "__main__":
    unittest.main()
