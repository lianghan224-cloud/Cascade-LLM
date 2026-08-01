import unittest

from layer_streaming import (
    ExecutionPolicy,
    Granularity,
    LlamaModelAdapter,
    PlacementMode,
    WeightFormat,
)


def tiny_config(**overrides):
    config = {
        "model_type": "llama",
        "_name_or_path": "tiny-test",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 257,
        "max_position_embeddings": 512,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-5,
    }
    config.update(overrides)
    return config


class PlanTest(unittest.TestCase):
    def test_geometry_and_plan_are_config_driven(self):
        adapter = LlamaModelAdapter()
        config = tiny_config(num_hidden_layers=5, vocab_size=301)
        geometry = adapter.build_geometry(config)
        policy = ExecutionPolicy(
            granularity=Granularity.MATRIX_GROUP,
            embedding_mode=PlacementMode.STREAMED,
            lm_head_mode=PlacementMode.RESIDENT,
        )
        plan = adapter.build_plan(config, policy)
        self.assertEqual(geometry.num_hidden_layers, 5)
        self.assertEqual(geometry.vocab_size, 301)
        self.assertEqual(len(plan.units), 5 * 4)
        self.assertEqual(
            plan.tensors["model.layers.4.self_attn.k_proj.weight"].shape,
            (32, 64),
        )
        resident_keys = {item.tensor.key for item in plan.resident}
        host_only_keys = {item.tensor.key for item in plan.host_only}
        self.assertIn("lm_head.weight", resident_keys)
        self.assertIn("model.embed_tokens.weight", host_only_keys)
        self.assertTrue(plan.vocab.stream_embedding)
        self.assertFalse(plan.vocab.stream_lm_head)

    def test_all_granularities_have_stable_unit_counts(self):
        adapter = LlamaModelAdapter()
        expected = {
            Granularity.MATRIX: 3 * 7,
            Granularity.MATRIX_GROUP: 3 * 4,
            Granularity.LAYER: 3,
        }
        for granularity, unit_count in expected.items():
            with self.subTest(granularity=granularity):
                plan = adapter.build_plan(
                    tiny_config(),
                    ExecutionPolicy(granularity=granularity),
                )
                self.assertEqual(len(plan.units), unit_count)

    def test_int8_fallback_is_explicit_and_geometry_driven(self):
        adapter = LlamaModelAdapter()
        policy = ExecutionPolicy(
            weight_format=WeightFormat.INT8_DEQUANT_BF16_FALLBACK,
            embedding_mode=PlacementMode.STREAMED,
            lm_head_mode=PlacementMode.STREAMED,
        )
        plan = adapter.build_plan(tiny_config(), policy)
        key = "model.layers.2.mlp.down_proj.weight"
        self.assertEqual(plan.tensors[key].dtype, "int8")
        self.assertEqual(plan.tensors[key].shape, (64, 128))
        self.assertEqual(plan.tensors[key + "_scale"].shape, (64, 1))
        self.assertEqual(plan.geometry.num_hidden_layers, 3)

    def test_invalid_head_geometry_fails_early(self):
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            LlamaModelAdapter().build_geometry(
                tiny_config(num_attention_heads=6)
            )

    def test_large_geometries_are_config_driven(self):
        adapter = LlamaModelAdapter()
        config_8b = tiny_config(
            _name_or_path="synthetic-8b",
            hidden_size=4096,
            intermediate_size=14336,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,
            vocab_size=128256,
            max_position_embeddings=131072,
        )
        plan_8b = adapter.build_execution_plan(
            config_8b,
            ExecutionPolicy(
                granularity=Granularity.MATRIX_GROUP,
                embedding_mode=PlacementMode.STREAMED,
                lm_head_mode=PlacementMode.STREAMED,
            ),
        )
        self.assertEqual(plan_8b.geometry.num_hidden_layers, 32)
        self.assertEqual(plan_8b.geometry.num_key_value_heads, 8)
        self.assertEqual(len(plan_8b.units), 32 * 4)
        config_70b = tiny_config(
            _name_or_path="synthetic-70b",
            hidden_size=8192,
            intermediate_size=28672,
            num_hidden_layers=80,
            num_attention_heads=64,
            num_key_value_heads=8,
            vocab_size=128256,
            max_position_embeddings=131072,
        )
        plan_70b = adapter.build_execution_plan(
            config_70b,
            ExecutionPolicy(
                granularity=Granularity.MATRIX_GROUP,
                weight_format=WeightFormat.INT8_DEQUANT_BF16_FALLBACK,
            ),
        )
        self.assertEqual(plan_70b.geometry.num_hidden_layers, 80)
        self.assertEqual(plan_70b.geometry.hidden_size, 8192)
        self.assertEqual(len(plan_70b.units), 80 * 4)


if __name__ == "__main__":
    unittest.main()
