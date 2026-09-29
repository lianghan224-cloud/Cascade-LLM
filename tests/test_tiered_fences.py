import threading
import time
import unittest

from layer_streaming.kv import KVOperationFence
from layer_streaming.kv.scheduler import TieredAttentionCoordinator
from layer_streaming.kv.stores import (
    KVTier,
    PrefetchCancelled,
    ResidencyState,
    TieredKVStore,
)


def _assert_quiescent(test_case, store, logical_block_id="block"):
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        stats = store.stats()
        if (
            stats["pins"] == 0
            and stats["inflight_io"] == 0
            and stats["reservations"] == 0
            and stats["pending_prefetches"] == 0
            and stats["pending_tier_operations"] == 0
        ):
            break
        time.sleep(0.001)
    else:
        test_case.fail("mock tier did not quiesce: {!r}".format(stats))
    test_case.assertTrue(store.validate_record(logical_block_id))
    return stats


class TieredKVFenceTest(unittest.TestCase):
    def test_prefetch_exposes_common_fence_and_cleans_accounting(self):
        with TieredKVStore(io_delay=0.005) as store:
            store.put("block", b"payload", KVTier.CPU, version=7)
            fence = store.prefetch(
                "block", KVTier.GPU, request_id="request-success"
            )
            self.assertIsInstance(fence, KVOperationFence)
            self.assertEqual(fence.kind, "prefetch")
            self.assertEqual(fence.request_id, "request-success")
            location = fence.result(timeout=2.0)
            self.assertEqual(location.tier, KVTier.GPU)
            self.assertTrue(fence.query())
            self.assertEqual(fence.status, "completed")
            self.assertEqual(store.get("block", KVTier.GPU), b"payload")
            _assert_quiescent(self, store)

    def test_timeout_cancels_and_preserves_source_authority(self):
        with TieredKVStore(io_delay=0.03) as store:
            store.put("block", b"payload", KVTier.CPU)
            group = store.prefetch_group(
                "request-timeout", cleanup_timeout=1.0
            )
            fence = group.add(
                store.prefetch(
                    "block", KVTier.GPU, request_id="request-timeout"
                )
            )
            with self.assertRaises(TimeoutError):
                group.wait(timeout=0.001)
            self.assertTrue(fence.cancelled)
            record = store.record("block")
            self.assertEqual(record.authoritative_tier, KVTier.CPU)
            self.assertEqual(store.get("block"), b"payload")
            self.assertNotEqual(
                record.locations[KVTier.GPU].state,
                ResidencyState.RESIDENT,
            )
            _assert_quiescent(self, store)

    def test_deduplicated_prefetch_is_not_cancelled_with_live_consumer(self):
        with TieredKVStore(io_delay=0.02) as store:
            store.put("block", b"payload", KVTier.CPU)
            first = store.prefetch_group("request-a")
            second = store.prefetch_group("request-b")
            first_fence = first.add(
                store.prefetch(
                    "block", KVTier.GPU, request_id="request-a"
                )
            )
            second_fence = second.add(
                store.prefetch(
                    "block", KVTier.GPU, request_id="request-b"
                )
            )
            self.assertIs(first_fence, second_fence)
            first.cancel()
            first.quiesce(timeout=2.0).release()
            second.wait(timeout=2.0)
            self.assertFalse(second_fence.cancelled)
            self.assertEqual(second_fence.status, "completed")
            self.assertEqual(store.get("block", KVTier.GPU), b"payload")
            self.assertEqual(store.stats()["prefetch_deduplicated"], 1)
            _assert_quiescent(self, store)

    def test_async_migration_failures_keep_one_authority(self):
        for stage in ("submit", "copy", "completion"):
            with self.subTest(stage=stage):
                with TieredKVStore(io_delay=0.001) as store:
                    store.put("block", b"payload", KVTier.CPU)
                    fence = store.migrate_async(
                        "block",
                        KVTier.GPU,
                        failure_stage=stage,
                        request_id="failure-{}".format(stage),
                    )
                    with self.assertRaises(IOError):
                        fence.result(timeout=2.0)
                    record = store.record("block")
                    self.assertEqual(record.authoritative_tier, KVTier.CPU)
                    self.assertEqual(store.get("block"), b"payload")
                    self.assertEqual(fence.status, "failed")
                    _assert_quiescent(self, store)

    def test_checksum_failure_keeps_source_authority(self):
        with TieredKVStore() as store:
            store.put("block", b"expected", KVTier.CPU)
            source_backend = store.backends[KVTier.CPU]
            original_read = source_backend.read

            def corrupt_in_transit(slot):
                original_read(slot)
                return b"corrupt!"

            source_backend.read = corrupt_in_transit
            try:
                fence = store.migrate_async(
                    "block", KVTier.GPU, request_id="checksum"
                )
                with self.assertRaisesRegex(IOError, "checksum"):
                    fence.result(timeout=2.0)
            finally:
                source_backend.read = original_read
            record = store.record("block")
            self.assertEqual(record.authoritative_tier, KVTier.CPU)
            self.assertEqual(store.get("block"), b"expected")
            self.assertEqual(fence.status, "failed")
            _assert_quiescent(self, store)

    def test_target_write_failure_releases_reservation(self):
        with TieredKVStore(io_delay=0.001) as store:
            store.put("block", b"payload", KVTier.CPU)
            store.backends[KVTier.GPU].fail_next_write = True
            group = store.prefetch_group("target-write")
            group.add(
                store.prefetch(
                    "block", KVTier.GPU, request_id="target-write"
                )
            )
            with self.assertRaisesRegex(IOError, "write failure"):
                group.wait(timeout=2.0)
            record = store.record("block")
            self.assertEqual(record.authoritative_tier, KVTier.CPU)
            self.assertEqual(store.get("block"), b"payload")
            _assert_quiescent(self, store)

    def test_bounded_cleanup_never_releases_running_io_early(self):
        with TieredKVStore(io_delay=0.05) as store:
            store.put("block", b"payload", KVTier.CPU)
            group = store.prefetch_group(
                "bounded-cleanup", cleanup_timeout=0.001
            )
            group.add(
                store.prefetch(
                    "block", KVTier.GPU, request_id="bounded-cleanup"
                )
            )
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if store.stats()["inflight_io"] == 1:
                    break
                time.sleep(0.001)
            self.assertEqual(store.stats()["inflight_io"], 1)
            with self.assertRaises(TimeoutError):
                group.cleanup(timeout=0.001)
            # The worker is still inside the mock read.  Its reservation and
            # IO pin must remain visible until cooperative cancellation is
            # observed and the migration finally block unwinds them.
            in_flight = store.stats()
            self.assertEqual(in_flight["inflight_io"], 1)
            self.assertEqual(in_flight["reservations"], 1)
            self.assertEqual(
                store.record("block").authoritative_tier, KVTier.CPU
            )
            _assert_quiescent(self, store)

    def test_request_cancel_quiesces_active_coordinator_group(self):
        with TieredKVStore(io_delay=0.03) as store:
            store.put("block", b"payload", KVTier.CPU)
            coordinator = TieredAttentionCoordinator(store)
            error = []

            def prepare():
                try:
                    coordinator.prepare(
                        ("block",), timeout=2.0, request_id="request-close"
                    )
                except BaseException as caught:
                    error.append(caught)

            worker = threading.Thread(target=prepare)
            worker.start()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if coordinator.stats()["active_prefetch_groups"] == 1:
                    break
                time.sleep(0.001)
            self.assertTrue(
                coordinator.cancel_request("request-close", timeout=1.0)
            )
            worker.join(timeout=2.0)
            self.assertFalse(worker.is_alive())
            self.assertEqual(len(error), 1)
            self.assertIsInstance(error[0], PrefetchCancelled)
            self.assertEqual(
                store.record("block").authoritative_tier, KVTier.CPU
            )
            self.assertEqual(
                coordinator.stats()["active_prefetch_groups"], 0
            )
            _assert_quiescent(self, store)


if __name__ == "__main__":
    unittest.main()
