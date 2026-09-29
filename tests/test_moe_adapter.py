import tempfile
from pathlib import Path
import unittest

import torch
from safetensors.torch import save_file
from transformers import OlmoeConfig

from layer_streaming import ExecutionPolicy, adapter_for_config
from layer_streaming.moe.adapters import OlmoeModelAdapter
from layer_streaming.moe.objects import WeightObjectKind


def tiny_olmoe_config(**overrides):
    values = dict(
        vocab_size=97,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        attention_bias=False,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    values.update(overrides)
    return OlmoeConfig(**values)


class OlmoeAdapterTest(unittest.TestCase):
    def setUp(self):
        self.config = tiny_olmoe_config()
        self.policy = ExecutionPolicy(vocab_chunk_bytes=4096)
        self.adapter = OlmoeModelAdapter()

    def test_registry_and_config_parse(self):
        self.assertIsInstance(adapter_for_config(self.config), OlmoeModelAdapter)
        geometry = self.adapter.build_geometry(self.config)
        moe = self.adapter.build_moe_config(self.config, self.policy)
        self.assertEqual(geometry.model_type, "olmoe")
        self.assertEqual(moe.num_experts, 4)
        self.assertEqual(moe.experts_per_token, 2)
        self.assertEqual(moe.expert_intermediate_size, 16)
        self.assertFalse(moe.normalize_topk)
        self.assertFalse(moe.has_shared_expert)

    def test_tensor_discovery_count_shapes_and_catalog(self):
        sidecar = self.adapter.build_moe_plan(self.config, self.policy)
        # 2 vocab + per layer: 4 attention + 4 norms + router + 4*3 experts,
        # plus one final norm.
        self.assertEqual(len(sidecar.execution_plan.weights), 2 + 2 * 21 + 1)
        experts = sidecar.weight_objects.iter_experts()
        self.assertEqual(len(experts), 8)
        first = experts[0]
        self.assertEqual(first.key.kind, WeightObjectKind.EXPERT)
        self.assertEqual(len(first.tensor_names), 3)
        self.assertEqual(
            first.shapes,
            ((16, 32), (16, 32), (32, 16)),
        )
        routers = [
            item for item in sidecar.weight_objects
            if item.key.kind == WeightObjectKind.ROUTER
        ]
        self.assertEqual(len(routers), 2)
        self.assertEqual(routers[0].shape, (4, 32))
        self.assertEqual(
            tuple(item.key.sort_key for item in sidecar.weight_objects),
            tuple(sorted(item.key.sort_key for item in sidecar.weight_objects)),
        )
        self.assertEqual(
            sidecar.execution_plan.slot_bytes,
            max(unit.transfer_bytes for unit in sidecar.execution_plan.units),
        )
        self.assertLessEqual(
            sidecar.execution_plan.slot_bytes,
            sidecar.execution_plan.vocab.chunk_bytes,
        )

    def test_missing_tensor_is_reported_by_checkpoint_validation(self):
        specs = self.adapter.enumerate_weights(self.config, policy=self.policy)
        tensors = {
            item.name: torch.zeros(item.storage_shape, dtype=torch.bfloat16)
            for item in specs
            if item.alias_of is None
        }
        missing = next(name for name in tensors if ".experts.2." in name)
        tensors.pop(missing)
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, str(Path(directory) / "model.safetensors"))
            result = self.adapter.validate_checkpoint(
                directory, self.config, self.policy
            )
        self.assertFalse(result.ok)
        self.assertIn(missing, result.missing)

    def test_corrupt_expert_shape_is_reported(self):
        specs = self.adapter.enumerate_weights(self.config, policy=self.policy)
        tensors = {
            item.name: torch.zeros(item.storage_shape, dtype=torch.bfloat16)
            for item in specs
            if item.alias_of is None
        }
        target = next(name for name in tensors if ".experts.1.down_proj" in name)
        tensors[target] = torch.zeros((31, 16), dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, str(Path(directory) / "model.safetensors"))
            result = self.adapter.validate_checkpoint(
                directory, self.config, self.policy
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.shape_mismatches[0].key, target)

    def test_tied_vocab_keeps_two_semantic_weight_objects(self):
        sidecar = self.adapter.build_moe_plan(
            tiny_olmoe_config(tie_word_embeddings=True), self.policy
        )
        embedding = next(
            item for item in sidecar.weight_objects
            if item.key.kind == WeightObjectKind.EMBEDDING
        )
        lm_head = next(
            item for item in sidecar.weight_objects
            if item.key.kind == WeightObjectKind.LM_HEAD
        )
        self.assertEqual(embedding.tensor_names, lm_head.tensor_names)


if __name__ == "__main__":
    unittest.main()
