import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from layer_streaming import ExecutionPolicy, LlamaModelAdapter

from test_plan import tiny_config


def tensors_for(specs):
    result = {}
    dtypes = {"bfloat16": torch.bfloat16, "int8": torch.int8}
    for spec in specs:
        if spec.alias_of is None:
            result[spec.key] = torch.zeros(spec.shape, dtype=dtypes[spec.dtype])
    return result


class CheckpointValidationTest(unittest.TestCase):
    def setUp(self):
        self.adapter = LlamaModelAdapter()
        self.config = tiny_config(num_hidden_layers=1, vocab_size=31)
        self.policy = ExecutionPolicy(vocab_chunk_bytes=4096)
        self.specs = self.adapter.enumerate_weights(
            self.config, policy=self.policy
        )

    def test_valid_single_file_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                tensors_for(self.specs),
                str(Path(directory) / "model.safetensors"),
            )
            result = self.adapter.validate_checkpoint(
                directory, self.config, self.policy
            )
            self.assertTrue(result.ok, result.format_errors())
            self.assertGreater(result.tensor_count, 0)

    def test_reports_complete_key_shape_and_dtype_conflicts(self):
        with tempfile.TemporaryDirectory() as directory:
            tensors = tensors_for(self.specs)
            missing_key = "model.norm.weight"
            tensors.pop(missing_key)
            shape_key = "model.layers.0.self_attn.q_proj.weight"
            tensors[shape_key] = torch.zeros((63, 64), dtype=torch.bfloat16)
            dtype_key = "model.layers.0.input_layernorm.weight"
            tensors[dtype_key] = tensors[dtype_key].float()
            tensors["unexpected.weight"] = torch.zeros(1, dtype=torch.bfloat16)
            save_file(tensors, str(Path(directory) / "model.safetensors"))
            result = self.adapter.validate_checkpoint(
                directory, self.config, self.policy
            )
            self.assertFalse(result.ok)
            self.assertIn(missing_key, result.missing)
            self.assertIn("unexpected.weight", result.unexpected)
            self.assertEqual(result.shape_mismatches[0].key, shape_key)
            self.assertEqual(result.dtype_mismatches[0].key, dtype_key)
            rendered = result.format_errors()
            self.assertIn("missing weights", rendered)
            self.assertIn("shape conflicts", rendered)
            self.assertIn("dtype conflicts", rendered)

    def test_index_mapping_is_checked_against_shard_header(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = tensors_for(self.specs)
            save_file(tensors, str(root / "part-1.safetensors"))
            weight_map = {key: "part-1.safetensors" for key in tensors}
            weight_map["ghost.weight"] = "part-1.safetensors"
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}), encoding="utf-8"
            )
            result = self.adapter.validate_checkpoint(
                root, self.config, self.policy
            )
            self.assertFalse(result.ok)
            self.assertTrue(
                any("ghost.weight" in item for item in result.index_errors)
            )


if __name__ == "__main__":
    unittest.main()
