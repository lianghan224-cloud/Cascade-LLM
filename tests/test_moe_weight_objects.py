import unittest

from layer_streaming import ExecutionPolicy, LlamaModelAdapter
from layer_streaming.api_contract import core_api_contract_sha256
from layer_streaming.moe.objects import (
    WeightObjectCatalog,
    WeightObjectKey,
    WeightObjectKind,
    WeightObjectRecord,
    WeightObjectSource,
)

from test_plan import tiny_config


def record(key, name="tensor", offset=0, nbytes=32):
    return WeightObjectRecord(
        key=key,
        tensor_names=(name,),
        storage_dtypes=("bfloat16",),
        shapes=((4, 4),),
        nbytes=nbytes,
        sources=(WeightObjectSource(name, "bfloat16", offset, nbytes),),
    )


class WeightObjectTest(unittest.TestCase):
    def test_key_equality_hash_and_expert_isolation(self):
        left = WeightObjectKey(2, "expert", "L2/E7", 7)
        same = WeightObjectKey(2, WeightObjectKind.EXPERT, "L2/E7", 7)
        other = WeightObjectKey(2, "expert", "L2/E8", 8)
        self.assertEqual(left, same)
        self.assertEqual(hash(left), hash(same))
        self.assertNotEqual(left, other)
        self.assertEqual(len({left, same, other}), 2)

    def test_invalid_expert_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "require"):
            WeightObjectKey(1, "expert", "missing")
        with self.assertRaisesRegex(ValueError, "only"):
            WeightObjectKey(1, "dense", "bad", 2)

    def test_catalog_is_deterministic_and_counts_bytes(self):
        expert = record(
            WeightObjectKey(1, "expert", "L1/E0", 0), "expert", 32, 64
        )
        dense = record(WeightObjectKey(0, "dense", "q"), "q", 0, 32)
        first = WeightObjectCatalog((expert, dense))
        second = WeightObjectCatalog((dense, expert))
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(first.total_bytes(), 96)
        self.assertEqual(first.total_bytes("expert"), 64)
        self.assertEqual(first.key_for_tensor("expert"), expert.key)
        self.assertEqual(first.iter_experts(1), (expert,))

    def test_missing_key_and_duplicate_tensor_fail_explicitly(self):
        key = WeightObjectKey(0, "dense", "q")
        catalog = WeightObjectCatalog((record(key),))
        with self.assertRaises(KeyError):
            catalog.get(WeightObjectKey(0, "dense", "k"))
        with self.assertRaisesRegex(ValueError, "multiple"):
            WeightObjectCatalog(
                (record(key), record(WeightObjectKey(0, "dense", "k")))
            )

    def test_dense_plan_maps_all_seven_matrices_without_hot_path_change(self):
        plan = LlamaModelAdapter().build_execution_plan(
            tiny_config(num_hidden_layers=2),
            ExecutionPolicy(vocab_chunk_bytes=4096),
        )
        before = plan.as_dict()
        contract_before = core_api_contract_sha256()
        catalog = WeightObjectCatalog.from_dense_plan(plan)
        for layer in range(2):
            names = {item.key.name for item in catalog.iter_layer(layer)}
            for operation in (
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ):
                self.assertTrue(any(operation in name for name in names), operation)
        self.assertEqual(plan.as_dict(), before)
        self.assertEqual(core_api_contract_sha256(), contract_before)


if __name__ == "__main__":
    unittest.main()
