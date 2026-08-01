import unittest

from layer_streaming import (
    ExecutionPolicy,
    MemoryPlanner,
    QuantizationSpec,
    WeightFormat,
)
from test_plan import tiny_config


class MetadataOnlyMemoryTest(unittest.TestCase):
    def test_70b_layout_uses_offsets_above_32_bit_without_allocation(self):
        config = tiny_config(
            _name_or_path="synthetic-70b",
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            vocab_size=128256,
            max_position_embeddings=131072,
        )
        result = MemoryPlanner.plan_metadata_only(
            config,
            ExecutionPolicy(
                weight_format=WeightFormat.INT8_DEQUANT_BF16_FALLBACK,
                quantization=QuantizationSpec(
                    bits=8,
                    granularity="per_group",
                    group_size=128,
                    scale_dtype="fp16",
                ),
            ),
            max_context=4096,
            cuda_safety_margin_bytes=0,
        )
        self.assertTrue(result.offsets_are_int64_safe)
        self.assertGreater(result.max_host_offset, 2 ** 32)
        self.assertGreater(result.estimated_shard_count, 1)
        self.assertGreater(result.estimate.cpu_int8_bytes, 60 * 1024 ** 3)
        self.assertGreater(result.estimate.cpu_scale_bytes, 0)
        self.assertGreater(result.estimate.gpu_dequant_workspace_bytes, 0)

    def test_metadata_plan_rejects_context_above_model_limit(self):
        with self.assertRaisesRegex(ValueError, "exceeds model limit"):
            MemoryPlanner.plan_metadata_only(
                tiny_config(max_position_embeddings=128),
                max_context=129,
            )

    def test_over_100b_int4_metadata_plan_remains_allocation_free(self):
        config = tiny_config(
            _name_or_path="synthetic-over-100b",
            hidden_size=12288,
            intermediate_size=32768,
            num_hidden_layers=96,
            num_attention_heads=96,
            num_key_value_heads=8,
            vocab_size=128256,
            max_position_embeddings=131072,
        )
        result = MemoryPlanner.plan_metadata_only(
            config,
            ExecutionPolicy(
                weight_format=WeightFormat.INT4_DEQUANT_BF16_FALLBACK,
                quantization=QuantizationSpec(
                    bits=4,
                    granularity="per_group",
                    group_size=128,
                ),
            ),
            max_context=8192,
            cuda_safety_margin_bytes=0,
        )
        self.assertTrue(result.offsets_are_int64_safe)
        self.assertGreater(result.estimate.cpu_int4_packed_bytes, 64 * 1024 ** 3)
        self.assertGreater(result.estimated_shard_count, 10)


if __name__ == "__main__":
    unittest.main()
