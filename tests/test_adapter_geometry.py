import unittest

from layer_streaming import LlamaModelAdapter
from test_plan import tiny_config


class AdapterGeometryTest(unittest.TestCase):
    def test_rope_activation_and_mha_are_config_driven(self):
        config = tiny_config(
            num_key_value_heads=4,
            rope_theta=500000.0,
            rope_scaling={"rope_type": "llama3", "factor": 8.0},
            hidden_act="gelu",
        )
        geometry = LlamaModelAdapter().build_geometry(config)
        self.assertEqual(geometry.num_key_value_heads, 4)
        self.assertEqual(geometry.rope_theta, 500000.0)
        self.assertEqual(geometry.rope_scaling["factor"], 8.0)
        self.assertEqual(geometry.hidden_act, "gelu")

    def test_invalid_rope_and_gqa_fail_before_planning(self):
        with self.assertRaisesRegex(ValueError, "rope_theta"):
            LlamaModelAdapter().build_geometry(tiny_config(rope_theta=0))
        with self.assertRaisesRegex(ValueError, "key_value_heads"):
            LlamaModelAdapter().build_geometry(
                tiny_config(num_attention_heads=4, num_key_value_heads=3)
            )


if __name__ == "__main__":
    unittest.main()
