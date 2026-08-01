import unittest

import torch

from layer_streaming import (
    ExecutionPolicy,
    LlamaModelAdapter,
    create_weight_store,
)

from test_plan import tiny_config


def pageable_factory(*shape, **kwargs):
    return torch.empty(*shape, dtype=kwargs["dtype"], device="cpu")


class WeightStoreLifecycleTest(unittest.TestCase):
    def test_context_close_is_idempotent_and_reopenable(self):
        plan = LlamaModelAdapter().build_plan(
            tiny_config(num_hidden_layers=1),
            ExecutionPolicy(vocab_chunk_bytes=4096),
        )
        store = create_weight_store(
            plan,
            mode="pinned_staging",
            allocate=False,
            tensor_factory=pageable_factory,
            slot_count=3,
        )
        with store as entered:
            self.assertIs(entered, store)
            self.assertTrue(store.allocated)
            self.assertEqual(len(store.staging_slots), 3)
        self.assertFalse(store.allocated)
        store.close()
        with store:
            self.assertTrue(store.allocated)
        self.assertFalse(store.allocated)


if __name__ == "__main__":
    unittest.main()
