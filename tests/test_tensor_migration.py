import threading
import unittest

import torch

from layer_streaming.kv.errors import KVLifecycleError
from layer_streaming.kv.page_pool import KVPagePoolV1
from layer_streaming.kv.stores import (
    GPUKVStore,
    PinnedCPUKVStore,
    PinnedCPUTensorMigration,
)


def _pageable_factory(shape, **kwargs):
    return torch.empty(shape, dtype=kwargs["dtype"], device="cpu")


class _ControlledEvent:
    def __init__(self):
        self.ready = threading.Event()

    def synchronize(self):
        if not self.ready.wait(2.0):
            raise TimeoutError("controlled migration event timed out")


class TensorMigrationTest(unittest.TestCase):
    def make_fixture(self, event_factory=None):
        gpu = GPUKVStore(
            layer_count=1,
            page_count=2,
            num_kv_heads=2,
            page_size=4,
            head_dim=3,
            dtype=torch.float32,
            device="cpu",
        )
        page_bytes = 2 * 2 * 4 * 3 * 4
        cpu = PinnedCPUKVStore(
            capacity_bytes=2 * page_bytes,
            num_kv_heads=2,
            page_size=4,
            head_dim=3,
            dtype=torch.float32,
            tensor_factory=_pageable_factory,
        )
        pool = KVPagePoolV1(2, store_id="gpu", dtype="fp32")
        handle = pool.allocate(owner_hint="request")
        pool.activate(handle, 4)
        pool.mark_data_updated(handle, version=1)
        pool.seal(handle, 4)
        gpu.keys[0, handle.page_id].copy_(
            torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
        )
        gpu.values[0, handle.page_id].copy_(gpu.keys[0, handle.page_id] + 100)
        manager = PinnedCPUTensorMigration(
            gpu, cpu, pool, event_factory=event_factory
        )
        key = manager.register_gpu_authority("block", 0, handle, 4, 1)
        return manager, gpu, cpu, pool, handle, key

    def test_d2h_h2d_roundtrip_commits_one_authority_after_fence(self):
        manager, gpu, cpu, pool, handle, key = self.make_fixture()
        expected_key = gpu.keys[0, handle.page_id].clone()
        expected_value = gpu.values[0, handle.page_id].clone()

        d2h = manager.migrate_d2h(key)
        d2h.wait()
        self.assertEqual(manager.record(key).authority, "cpu")
        self.assertEqual(pool.descriptor(handle).inflight_io, 0)

        gpu.keys[0, handle.page_id].zero_()
        gpu.values[0, handle.page_id].zero_()
        h2d = manager.migrate_h2d(key)
        h2d.wait()
        self.assertEqual(manager.record(key).authority, "gpu")
        self.assertTrue(torch.equal(gpu.keys[0, handle.page_id], expected_key))
        self.assertTrue(torch.equal(gpu.values[0, handle.page_id], expected_value))
        self.assertEqual(manager.stats()["authority_changes"], 2)
        self.assertEqual(manager.stats()["pending_fences"], 0)
        self.assertEqual(pool.descriptor(handle).pin_count, 0)
        manager.close()
        cpu.close()
        gpu.close()

    def test_epoch_change_aborts_target_and_keeps_gpu_authority(self):
        event = _ControlledEvent()
        manager, gpu, cpu, pool, handle, key = self.make_fixture(
            event_factory=lambda: event
        )
        fence = manager.migrate_d2h(key)
        pool.mark_data_updated(handle, version=2)
        event.ready.set()
        with self.assertRaises(KVLifecycleError):
            fence.wait()
        self.assertEqual(manager.record(key).authority, "gpu")
        self.assertIsNone(manager.record(key).cpu_handle)
        self.assertEqual(cpu.used_bytes, 0)
        self.assertEqual(pool.descriptor(handle).pin_count, 0)
        self.assertEqual(manager.stats()["migration_failures"], 1)
        manager.close()
        cpu.close()
        gpu.close()

    def test_cancelled_d2h_releases_reservation_after_quiescence(self):
        event = _ControlledEvent()
        manager, gpu, cpu, pool, handle, key = self.make_fixture(
            event_factory=lambda: event
        )
        fence = manager.migrate_d2h(key)
        fence.cancel()
        self.assertEqual(pool.descriptor(handle).inflight_io, 1)
        event.ready.set()
        with self.assertRaises(KVLifecycleError):
            fence.wait()
        self.assertEqual(manager.record(key).authority, "gpu")
        self.assertEqual(cpu.used_bytes, 0)
        self.assertEqual(pool.descriptor(handle).pin_count, 0)
        self.assertEqual(manager.stats()["migration_cancellations"], 1)
        manager.close()
        cpu.close()
        gpu.close()


if __name__ == "__main__":
    unittest.main()
