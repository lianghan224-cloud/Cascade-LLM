import unittest

from layer_streaming import QuantizationSpec, WeightSpec


class QuantizationSpecTest(unittest.TestCase):
    def test_per_channel_and_per_group_shapes(self):
        channel = QuantizationSpec(bits=8, granularity="per_channel")
        self.assertEqual(channel.scale_shape((17, 64)), (17, 1))
        self.assertEqual(channel.storage_shape((17, 64)), (17, 64))
        group = QuantizationSpec(
            bits=4,
            granularity="per_group",
            group_size=32,
            scale_dtype="fp16",
        )
        self.assertEqual(group.scale_shape((17, 64)), (17, 2))
        self.assertEqual(group.storage_shape((17, 64)), (544,))
        spec = WeightSpec.quantized(
            "weight", (17, 64), group, "bf16", "attention_q"
        )
        self.assertEqual(spec.storage_dtype, "uint8")
        self.assertEqual(spec.storage_nbytes, 544)

    def test_invalid_group_and_zero_point_fail_early(self):
        group = QuantizationSpec(
            bits=8, granularity="per_group", group_size=32
        )
        with self.assertRaisesRegex(ValueError, "does not divide"):
            group.scale_shape((8, 65))
        with self.assertRaisesRegex(ValueError, "zero-point"):
            QuantizationSpec(bits=8, scheme="symmetric", zero_point=True)
        with self.assertRaisesRegex(ValueError, "requires a zero-point"):
            QuantizationSpec(bits=8, scheme="asymmetric")

    def test_weight_storage_shape_and_dtype_are_not_implicit(self):
        quant = QuantizationSpec(
            bits=4, granularity="per_group", group_size=32
        )
        with self.assertRaisesRegex(ValueError, "storage shape"):
            WeightSpec(
                name="weight",
                logical_shape=(2, 32),
                storage_shape=(2, 32),
                storage_dtype="uint8",
                compute_dtype="bf16",
                role="mlp_up",
                quantization=quant,
            )


if __name__ == "__main__":
    unittest.main()
