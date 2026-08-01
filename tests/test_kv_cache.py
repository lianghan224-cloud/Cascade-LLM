import unittest

import torch

from layer_streaming import (
    KVCacheCapacityError,
    KVCacheError,
    KVCacheManager,
)


class KVCacheTest(unittest.TestCase):
    def make_manager(self):
        return KVCacheManager(
            layer_count=2,
            num_key_value_heads=2,
            head_dim=4,
            total_blocks=4,
            block_size=2,
            max_batch_size=1,
            dtype=torch.float32,
            device="cpu",
        )

    def append_step(self, cache, values):
        for layer in range(2):
            key = torch.full(
                (1, 2, len(values), 4),
                values[0] + layer,
                dtype=torch.float32,
            )
            value = torch.full(
                (1, 2, len(values), 4),
                values[-1] + layer,
                dtype=torch.float32,
            )
            cache.append(layer, key, value)

    def test_append_reuses_stable_preallocated_address(self):
        manager = self.make_manager()
        cache = manager.bind(manager.allocate(4))
        self.append_step(cache, [1, 2])
        first_pointer = cache.get_view(0)[0].data_ptr()
        self.append_step(cache, [3])
        second_pointer = cache.get_view(0)[0].data_ptr()
        self.assertEqual(first_pointer, second_pointer)
        self.assertEqual(cache.sequence_length(), 3)
        self.assertEqual(cache.get_view(0)[0].shape, (1, 2, 3, 4))

    def test_reset_release_and_block_reuse(self):
        manager = self.make_manager()
        handle = manager.allocate(4)
        block_ids = handle.block_ids
        cache = manager.bind(handle)
        self.append_step(cache, [1])
        cache.clear()
        self.assertEqual(cache.sequence_length(), 0)
        cache.close()
        self.assertEqual(manager.free_blocks, 4)
        replacement = manager.allocate(4)
        self.assertEqual(replacement.block_ids, block_ids)

    def test_overflow_and_order_errors_are_controlled(self):
        manager = self.make_manager()
        cache = manager.bind(manager.allocate(2))
        tensor = torch.zeros((1, 2, 1, 4), dtype=torch.float32)
        with self.assertRaises(KVCacheError):
            cache.append(1, tensor, tensor)
        self.append_step(cache, [1, 2])
        with self.assertRaises(KVCacheCapacityError):
            cache.append(0, tensor, tensor)

    def test_capacity_error_does_not_consume_blocks(self):
        manager = self.make_manager()
        manager.allocate(6)
        with self.assertRaises(KVCacheCapacityError):
            manager.allocate(4)
        self.assertEqual(manager.free_blocks, 1)

    def test_one_hundred_allocate_release_cycles_restore_all_blocks(self):
        manager = self.make_manager()
        for index in range(100):
            handle = manager.allocate(4)
            cache = manager.bind(handle)
            self.append_step(cache, [index])
            cache.close()
            stats = manager.resource_stats()
            self.assertEqual(stats["allocated_blocks"], 0)
            self.assertEqual(stats["active_handles"], 0)
            self.assertEqual(stats["free_blocks"], stats["total_blocks"])
        manager.close()
        manager.close()


if __name__ == "__main__":
    unittest.main()
