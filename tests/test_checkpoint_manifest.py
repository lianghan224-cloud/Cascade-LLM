import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from layer_streaming import (
    CheckpointManifest,
    ExecutionPolicy,
    LlamaModelAdapter,
)
from test_plan import tiny_config


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "int8": torch.int8,
    "uint8": torch.uint8,
}


def expected_tensors(specs):
    return {
        spec.name: torch.zeros(
            spec.storage_shape, dtype=DTYPES[spec.storage_dtype]
        )
        for spec in specs
        if spec.alias_of is None
    }


class CheckpointManifestTest(unittest.TestCase):
    def setUp(self):
        self.adapter = LlamaModelAdapter()
        self.config = tiny_config(num_hidden_layers=1, vocab_size=33)
        self.policy = ExecutionPolicy(vocab_chunk_bytes=4096)
        self.specs = self.adapter.enumerate_weights(
            self.config, policy=self.policy
        )

    def test_multi_shard_index_and_unreferenced_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensors = expected_tensors(self.specs)
            names = sorted(tensors)
            split = len(names) // 2
            first = {name: tensors[name] for name in names[:split]}
            second = {name: tensors[name] for name in names[split:]}
            first["unindexed.weight"] = torch.zeros(1, dtype=torch.bfloat16)
            save_file(first, str(root / "part-1.safetensors"))
            save_file(second, str(root / "part-2.safetensors"))
            weight_map = {
                name: (
                    "part-1.safetensors"
                    if name in first and name != "unindexed.weight"
                    else "part-2.safetensors"
                )
                for name in tensors
            }
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}), encoding="utf-8"
            )
            manifest = CheckpointManifest.from_path(root)
            result = manifest.validate(self.specs)
            self.assertFalse(result.ok)
            self.assertTrue(
                any("unindexed.weight" in item for item in result.errors)
            )

    def test_missing_shard_is_reported_without_tensor_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"model.norm.weight": "missing.safetensors"}}
                ),
                encoding="utf-8",
            )
            manifest = CheckpointManifest.from_path(root)
            self.assertTrue(
                any("missing shard" in item for item in manifest.errors)
            )

    def test_tied_alias_is_not_required_as_physical_tensor(self):
        config = tiny_config(
            num_hidden_layers=1,
            vocab_size=33,
            tie_word_embeddings=True,
        )
        specs = self.adapter.enumerate_weights(config, policy=self.policy)
        aliases = {
            spec.name: spec.alias_of for spec in specs if spec.alias_of is not None
        }
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                expected_tensors(specs),
                str(Path(directory) / "model.safetensors"),
            )
            manifest = CheckpointManifest.from_path(directory, aliases=aliases)
            result = manifest.validate(specs)
            self.assertTrue(result.ok, result.format_errors())
            self.assertGreater(manifest.shared_bytes, 0)


if __name__ == "__main__":
    unittest.main()
