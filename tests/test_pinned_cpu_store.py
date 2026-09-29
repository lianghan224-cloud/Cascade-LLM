import unittest
from unittest import mock

import torch

from layer_streaming.kv.errors import KVCapacityError, KVLifecycleError
from layer_streaming.kv.stores import PinnedCPUKVStore


class RecordingTensorFactory:
    """Explicitly simulate pinned tensors on CPU-only test machines."""

    def __init__(self):
        self.calls = []

    def __call__(self, shape, **kwargs):
        self.calls.append((tuple(shape), dict(kwargs)))
        return torch.empty(shape, dtype=kwargs["dtype"], device="cpu")


class PinnedCPUKVStoreTest(unittest.TestCase):
    def make_store(self, slots=2, checksum_enabled=False, **kwargs):
        factory = RecordingTensorFactory()
        page_bytes = 2 * 2 * 4 * 3 * torch.tensor([], dtype=torch.float32).element_size()
        store = PinnedCPUKVStore(
            capacity_bytes=slots * page_bytes,
            num_kv_heads=2,
            page_size=4,
            head_dim=3,
            dtype=torch.float32,
            checksum_enabled=checksum_enabled,
            tensor_factory=factory,
            **kwargs,
        )
        return store, factory, page_bytes

    def payload(self, offset=0):
        key = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        value = key + 100
        return key + offset, value + offset

    def test_preallocates_two_pinned_pool_tensors_and_reports_bytes(self):
        store, factory, page_bytes = self.make_store(slots=2)
        self.assertEqual(len(factory.calls), 2)
        for shape, kwargs in factory.calls:
            self.assertEqual(shape, (2, 2, 4, 3))
            self.assertEqual(kwargs["device"], torch.device("cpu"))
            self.assertTrue(kwargs["pin_memory"])
        self.assertEqual(store.capacity_bytes, 2 * page_bytes)
        self.assertEqual(store.nbytes, 2 * page_bytes)
        self.assertTrue(store.capability().implemented)
        self.assertFalse(store.capability().supports_async_copy)

    def test_reserve_allocate_write_read_release_lifecycle(self):
        store, _, page_bytes = self.make_store(slots=2, checksum_enabled=True)
        reservation = store.reserve()
        self.assertEqual(store.reserved_bytes, page_bytes)
        self.assertEqual(store.used_bytes, 0)
        handle = store.allocate(
            reservation, logical_block_id="request:page-3", layer=7, data_epoch=11
        )
        self.assertEqual(store.reserved_bytes, 0)
        self.assertEqual(store.used_bytes, page_bytes)
        key, value = self.payload()
        written = store.write(
            handle,
            key,
            value,
            valid_tokens=3,
            data_epoch=11,
        )
        self.assertEqual(written.logical_block_id, "request:page-3")
        self.assertEqual(written.layer, 7)
        self.assertEqual(written.dtype, torch.float32)
        self.assertEqual(written.layout, "hnd")
        self.assertEqual(written.valid_tokens, 3)
        self.assertEqual(written.data_epoch, 11)
        self.assertTrue(written.checksum.startswith("sha256:"))
        page = store.read(
            handle, expected_data_epoch=11, verify_checksum=True
        )
        self.assertTrue(torch.equal(page.key, key))
        self.assertTrue(torch.equal(page.value, value))
        store.release(handle, expected_data_epoch=11)
        self.assertEqual(store.used_bytes, 0)
        self.assertEqual(store.free_bytes, 2 * page_bytes)
        with self.assertRaisesRegex(KVLifecycleError, "stale"):
            store.read(handle)

    def test_write_target_is_invisible_until_explicit_commit(self):
        store, _, _ = self.make_store(slots=1)
        reservation = store.reserve()
        handle = store.allocate(
            reservation, logical_block_id="block", layer=2, data_epoch=5
        )
        target_key, target_value = store.write_target(
            handle, expected_data_epoch=5
        )
        key, value = self.payload(4)
        target_key.copy_(key)
        target_value.copy_(value)
        with self.assertRaisesRegex(KVLifecycleError, "not been committed"):
            store.read(handle)
        page = store.commit_write(
            handle, valid_tokens=4, data_epoch=5
        )
        self.assertTrue(torch.equal(page.key, key))
        self.assertTrue(torch.equal(page.value, value))
        with self.assertRaisesRegex(KVLifecycleError, "already committed"):
            store.commit_write(handle, valid_tokens=4, data_epoch=5)

    def test_capacity_failure_and_cancel_restore_accounting(self):
        store, _, page_bytes = self.make_store(slots=1)
        reservation = store.reserve(required_bytes=page_bytes)
        with self.assertRaises(KVCapacityError):
            store.reserve()
        self.assertEqual(store.stats()["reserved_bytes"], page_bytes)
        store.cancel_reservation(reservation)
        self.assertEqual(store.stats()["reserved_bytes"], 0)
        self.assertEqual(store.stats()["free_bytes"], page_bytes)
        with self.assertRaisesRegex(KVLifecycleError, "stale or inactive"):
            store.cancel_reservation(reservation)
        next_reservation = store.reserve()
        self.assertNotEqual(next_reservation.generation, reservation.generation)

    def test_epoch_shape_dtype_checksum_and_foreign_handle_validation(self):
        first, _, _ = self.make_store(slots=1, checksum_enabled=True)
        second, _, _ = self.make_store(slots=1)
        handle = first.allocate(
            first.reserve(), logical_block_id="block", layer=0, data_epoch=9
        )
        key, value = self.payload()
        with self.assertRaisesRegex(KVLifecycleError, "epoch mismatch"):
            first.write(handle, key, value, valid_tokens=4, data_epoch=8)
        with self.assertRaisesRegex(ValueError, "shape"):
            first.write(
                handle,
                key[:, :3],
                value[:, :3],
                valid_tokens=3,
                data_epoch=9,
            )
        first.write(handle, key, value, valid_tokens=4, data_epoch=9)
        first.read(handle).key[0, 0, 0] += 1
        with self.assertRaisesRegex(IOError, "checksum mismatch"):
            first.read(handle, verify_checksum=True)
        with self.assertRaisesRegex(KVLifecycleError, "another store"):
            second.read(handle)

    def test_byte_watermarks_and_capacity_round_down(self):
        store, _, page_bytes = self.make_store(
            slots=2,
            high_watermark_bytes=2 * 192,
            low_watermark_bytes=192,
        )
        # This test geometry has a 192-byte K/V layer page.
        self.assertEqual(page_bytes, 192)
        first = store.reserve()
        self.assertTrue(store.below_low_watermark)
        second = store.reserve()
        self.assertTrue(store.above_high_watermark)
        store.cancel_reservation(first)
        store.cancel_reservation(second)
        self.assertEqual(store.stats()["free_slots"], 2)

        factory = RecordingTensorFactory()
        rounded = PinnedCPUKVStore(
            capacity_bytes=2 * page_bytes + page_bytes // 2,
            num_kv_heads=2,
            page_size=4,
            head_dim=3,
            dtype=torch.float32,
            tensor_factory=factory,
        )
        self.assertEqual(rounded.requested_capacity_bytes, 480)
        self.assertEqual(rounded.capacity_bytes, 384)
        self.assertEqual(rounded.free_bytes, 384)

    def test_lifecycle_cycles_reuse_fixed_pool_without_new_allocations(self):
        store, factory, _ = self.make_store(slots=2)
        key, value = self.payload()
        for epoch in range(1, 101):
            reservation = store.reserve()
            handle = store.allocate(
                reservation,
                logical_block_id="block-{}".format(epoch),
                layer=epoch % 8,
                data_epoch=epoch,
            )
            store.write(
                handle,
                key,
                value,
                valid_tokens=4,
                data_epoch=epoch,
            )
            store.release(handle, expected_data_epoch=epoch)
        self.assertEqual(len(factory.calls), 2)
        self.assertEqual(store.used_bytes, 0)
        self.assertEqual(store.reserved_bytes, 0)
        self.assertEqual(store.stats()["free_slots"], 2)

    def test_production_pin_failure_is_not_silently_retried_pageable(self):
        with mock.patch(
            "layer_streaming.kv.stores.pinned_cpu.torch.empty",
            side_effect=RuntimeError("pin allocator unavailable"),
        ) as empty:
            with self.assertRaisesRegex(RuntimeError, "pin allocator unavailable"):
                PinnedCPUKVStore(
                    capacity_bytes=192,
                    num_kv_heads=2,
                    page_size=4,
                    head_dim=3,
                    dtype=torch.float32,
                )
        self.assertEqual(empty.call_count, 1)

    def test_close_is_idempotent_and_rejects_future_use(self):
        store, _, _ = self.make_store(slots=1)
        store.reserve()
        store.close()
        store.close()
        self.assertEqual(store.nbytes, 0)
        self.assertEqual(store.used_bytes, 0)
        self.assertEqual(store.reserved_bytes, 0)
        with self.assertRaisesRegex(KVLifecycleError, "closed"):
            store.reserve()


if __name__ == "__main__":
    unittest.main()
