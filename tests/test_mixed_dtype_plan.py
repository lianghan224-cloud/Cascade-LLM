import unittest

from layer_streaming import (
    ExecutionPolicy,
    Granularity,
    LlamaModelAdapter,
    PlacementMode,
    QuantizationSpec,
    WeightFormat,
)
from test_plan import tiny_config


class MixedDtypePlanTest(unittest.TestCase):
    def test_int4_transformer_with_fp16_vocabulary_and_bf16_norm(self):
        adapter = LlamaModelAdapter()
        policy = ExecutionPolicy(
            weight_format=WeightFormat.INT4_DEQUANT_FP16_FALLBACK,
            quantization=QuantizationSpec(
                bits=4,
                granularity="per_group",
                group_size=32,
                scale_dtype="bf16",
            ),
            embedding_dtype="fp16",
            lm_head_dtype="fp16",
            norm_dtype="bf16",
            granularity=Granularity.MATRIX_GROUP,
            embedding_mode=PlacementMode.STREAMED,
            lm_head_mode=PlacementMode.RESIDENT,
            vocab_chunk_bytes=4096,
        )
        plan = adapter.build_execution_plan(tiny_config(), policy)
        self.assertIn("int4_packed", plan.regions)
        self.assertIn("float16", plan.regions)
        self.assertIn("bfloat16", plan.regions)
        self.assertIn("scale_bfloat16", plan.regions)
        self.assertEqual(len(plan.units), 3 * 4)
        self.assertTrue(
            all(
                unit.workspace_bytes > 0 and unit.transfer_bytes <= plan.slot_bytes
                for unit in plan.units
            )
        )
        backends = {
            tensor.backend
            for unit in plan.units
            for tensor in unit.tensors
            if tensor.backend
        }
        self.assertEqual(backends, {"int4_dequant_fp16_fallback"})

    def test_tied_weight_has_one_storage_and_one_resident_allocation(self):
        adapter = LlamaModelAdapter()
        config = tiny_config(tie_word_embeddings=True)
        policy = ExecutionPolicy(
            embedding_mode=PlacementMode.STREAMED,
            lm_head_mode=PlacementMode.RESIDENT,
            vocab_chunk_bytes=4096,
        )
        plan = adapter.build_execution_plan(config, policy)
        self.assertEqual(
            plan.aliases["lm_head.weight"], "model.embed_tokens.weight"
        )
        self.assertNotIn("lm_head.weight", plan.host_offsets)
        resident = [item.weight_name for item in plan.resident]
        self.assertEqual(resident.count("model.embed_tokens.weight"), 1)
        self.assertEqual(
            plan.shared_weight_savings_bytes,
            config["vocab_size"] * config["hidden_size"] * 2,
        )

    def test_tied_weight_rejects_conflicting_storage_dtypes(self):
        with self.assertRaisesRegex(ValueError, "same dtype"):
            LlamaModelAdapter().build_execution_plan(
                tiny_config(tie_word_embeddings=True),
                ExecutionPolicy(
                    embedding_dtype="bf16",
                    lm_head_dtype="fp16",
                    vocab_chunk_bytes=4096,
                ),
            )


if __name__ == "__main__":
    unittest.main()
