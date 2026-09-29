import threading
import time
import unittest

import torch

from layer_streaming.kv.errors import KVCapacityError, KVLifecycleError
from layer_streaming.kv.page_pool import KVPagePoolV1
from layer_streaming.kv.stores import (
    ActiveTierCoordinator,
    GPUHotKVCache,
    PinnedCPUKVStore,
)


def _cpu_factory(shape, **kwargs):
    return torch.empty(shape, dtype=kwargs["dtype"], device="cpu")


class _ControlledEvent:
    def __init__(self):
        self.ready = threading.Event()
        self.started = threading.Event()

    def synchronize(self):
        self.started.set()
        if not self.ready.wait(2.0):
            raise TimeoutError("controlled Active Tier event timed out")


class _FailingEvent:
    def synchronize(self):
        raise RuntimeError("injected hot-cache execution event failure")


class ActiveTierCoordinatorTest(unittest.TestCase):
    def make_fixture(
        self,
        page_count=16,
        gpu_pages=4,
        event_factory=None,
        compute_event_factory=None,
    ):
        layers = 2
        heads = 2
        page_size = 4
        head_dim = 3
        pool = KVPagePoolV1(
            page_count, store_id="active-tier", dtype="fp32"
        )
        handles = []
        for index in range(page_count):
            handle = pool.allocate(owner_hint="request")
            pool.activate(handle, page_size)
            pool.mark_data_updated(handle, version=index + 1)
            pool.seal(handle, page_size)
            handles.append(handle)
        hot = GPUHotKVCache(
            page_pool=pool,
            layer_count=layers,
            gpu_capacity_pages=gpu_pages,
            num_kv_heads=heads,
            page_size=page_size,
            head_dim=head_dim,
            dtype=torch.float32,
            device="cpu",
            high_watermark_pages=gpu_pages,
            low_watermark_pages=max(0, gpu_pages - 1),
            tensor_factory=_cpu_factory,
        )
        cpu = PinnedCPUKVStore(
            capacity_bytes=page_count * layers * hot.layer_page_bytes,
            num_kv_heads=heads,
            page_size=page_size,
            head_dim=head_dim,
            dtype=torch.float32,
            tensor_factory=_cpu_factory,
        )
        coordinator = ActiveTierCoordinator(
            hot,
            cpu,
            pool,
            event_factory=event_factory,
            compute_event_factory=compute_event_factory,
        )
        return coordinator, hot, cpu, pool, handles

    @staticmethod
    def populate_gpu(coordinator, hot, logical, handle, epoch, value):
        key = coordinator.register_page(
            logical,
            handle,
            valid_tokens=hot.page_size,
            data_epoch=epoch,
        )
        slot = hot.reserve_gpu(key)
        expected = {}
        for layer in range(hot.layer_count):
            key_tensor, value_tensor = hot.layer_write_target(
                key, layer, slot
            )
            key_tensor.fill_(float(value + layer))
            value_tensor.fill_(float(value + layer + 100))
            expected[layer] = (key_tensor.clone(), value_tensor.clone())
        hot.commit_gpu(key, slot)
        return key, expected

    def test_four_hot_pages_stream_sixteen_request_pages_bitwise(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture()
        keys = []
        expected = {}
        initial_refs = tuple(pool.descriptor(item).ref_count for item in handles)
        for index, handle in enumerate(handles):
            if hot.free_pages == 0:
                coordinator.evict(hot.lru_victims(limit=1)[0]).wait()
            key, payload = self.populate_gpu(
                coordinator,
                hot,
                "logical-{}".format(index),
                handle,
                index + 1,
                index * 10,
            )
            keys.append(key)
            expected[key] = payload

        self.assertEqual(hot.gpu_capacity_pages, 4)
        self.assertEqual(len(keys), 16)
        for wave_index in range(4):
            wave_keys = keys[wave_index * 4 : (wave_index + 1) * 4]
            with coordinator.acquire_wave(
                wave_keys,
                request_id="wave-{}".format(wave_index),
                timeout=2.0,
            ) as wave:
                for layer in range(hot.layer_count):
                    pages = wave.layer_pages(layer)
                    self.assertEqual(len(pages), 4)
                    for key, page in zip(wave_keys, pages):
                        expected_key, expected_value = expected[key][layer]
                        self.assertTrue(torch.equal(page.key, expected_key))
                        self.assertTrue(torch.equal(page.value, expected_value))
            pool.validate_invariants()

        stats = coordinator.stats()
        self.assertTrue(stats["tier_metrics_sampled"])
        self.assertGreater(stats["eviction_count"], 0)
        self.assertGreater(stats["prefetch_count"], 0)
        self.assertGreater(stats["h2d_kv_bytes"], 0)
        self.assertGreater(stats["d2h_kv_bytes"], 0)
        self.assertEqual(stats["thrash_window_operations"], 8)
        self.assertGreater(stats["eviction_bytes"], stats["d2h_kv_bytes"])
        self.assertEqual(stats["pending_tier_operations"], 0)
        self.assertEqual(
            tuple(pool.descriptor(item).ref_count for item in handles),
            initial_refs,
        )
        coordinator.close()
        hot.close()
        cpu.close()

    def test_deduplicated_prefetch_survives_one_consumer_cancel(self):
        event = _ControlledEvent()
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 7
        )
        coordinator.evict(key).wait()
        coordinator._event_factory = lambda: event

        first = coordinator.prefetch_group("request-a")
        second = coordinator.prefetch_group("request-b")
        first_fence = first.add(
            coordinator.prefetch(key, request_id="request-a")
        )
        second_fence = second.add(
            coordinator.prefetch(key, request_id="request-b")
        )
        self.assertIs(first_fence, second_fence)
        first.cancel()
        event.ready.set()
        first.quiesce(timeout=2.0).release()
        second.wait(timeout=2.0)
        self.assertFalse(second_fence.cancelled)
        self.assertTrue(coordinator._gpu_resident(key))
        self.assertEqual(coordinator.stats()["prefetch_deduplicated"], 1)
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_evict_then_immediate_prefetch_records_thrashing(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 29
        )
        coordinator.evict(key).wait(timeout_seconds=2.0)
        coordinator.prefetch(key, request_id="thrash").wait(
            timeout_seconds=2.0
        )
        stats = coordinator.stats()
        self.assertEqual(stats["thrashing_count"], 1)
        self.assertEqual(stats["thrash_window_operations"], 8)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_epoch_mismatch_aborts_loading_slot_and_unpins(self):
        event = _ControlledEvent()
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 3
        )
        coordinator.evict(key).wait()
        coordinator._event_factory = lambda: event
        fence = coordinator.prefetch(key, request_id="epoch")
        pool.mark_data_updated(handles[0], version=99)
        event.ready.set()
        with self.assertRaises(KVLifecycleError):
            fence.wait(timeout_seconds=2.0)
        self.assertEqual(hot.free_pages, 1)
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        self.assertEqual(coordinator.stats()["migration_failures"], 1)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_prefetch_complete_then_epoch_change_rejects_old_attention_replica(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 23
        )
        coordinator.evict(key).wait(timeout_seconds=2.0)
        coordinator.prefetch(key, request_id="restore-old").wait(
            timeout_seconds=2.0
        )
        self.assertTrue(coordinator._gpu_resident(key))

        # Simulate authoritative payload publication racing after Prefetch but
        # before the stale selected view reaches Attention.  Runtime normally
        # follows with advance_data_epoch(); until then the old Location key
        # must be unreadable rather than silently serving its resident bytes.
        pool.mark_data_updated(handles[0], version=2)
        with self.assertRaisesRegex(KVLifecycleError, "data epoch mismatch"):
            coordinator.acquire_wave(
                (key,), request_id="stale-attention", timeout=2.0
            )
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        # close() is also the public hot-slot pin/inflight closure gate.
        self.assertEqual(coordinator.stats()["pending_tier_operations"], 0)
        self.assertEqual(coordinator.stats()["active_prefetch_groups"], 0)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_prefetch_timeout_cancels_and_quiesces_before_cleanup(self):
        event = _ControlledEvent()
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 5
        )
        coordinator.evict(key).wait()
        coordinator._event_factory = lambda: event
        timer = threading.Timer(0.02, event.ready.set)
        timer.start()
        try:
            with self.assertRaises(TimeoutError):
                coordinator.acquire_wave(
                    (key,), request_id="timeout", timeout=0.001
                )
        finally:
            timer.join(timeout=1.0)
        self.assertEqual(hot.free_pages, 1)
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        stats = coordinator.stats()
        self.assertEqual(stats["prefetch_timeouts"], 1)
        self.assertEqual(stats["migration_cancellations"], 1)
        self.assertEqual(stats["pending_tier_operations"], 0)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_cancel_during_d2h_preserves_gpu_and_releases_cpu_targets(self):
        event = _ControlledEvent()
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1, event_factory=lambda: event
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 9
        )
        fence = coordinator.evict(key)
        fence.cancel()
        event.ready.set()
        with self.assertRaises(KVLifecycleError):
            fence.wait(timeout_seconds=2.0)
        self.assertTrue(coordinator._gpu_resident(key))
        self.assertEqual(cpu.used_bytes, 0)
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        self.assertEqual(coordinator.stats()["migration_cancellations"], 1)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_d2h_target_allocation_failure_cancels_reservation(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 10
        )
        original_allocate = cpu.allocate

        def fail_allocate(*args, **kwargs):
            raise RuntimeError("injected CPU target allocation failure")

        cpu.allocate = fail_allocate
        try:
            fence = coordinator.evict(key)
            with self.assertRaisesRegex(RuntimeError, "target allocation"):
                fence.wait(timeout_seconds=2.0)
        finally:
            cpu.allocate = original_allocate
        self.assertTrue(coordinator._gpu_resident(key))
        self.assertEqual(cpu.used_bytes, 0)
        self.assertEqual(cpu.reserved_bytes, 0)
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_h2d_submit_failure_releases_loading_slot_and_io_pin(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 31
        )
        coordinator.evict(key).wait(timeout_seconds=2.0)
        original_submit = coordinator._executor.submit

        def fail_submit(*args, **kwargs):
            raise RuntimeError("injected H2D submit failure")

        coordinator._executor.submit = fail_submit
        try:
            with self.assertRaisesRegex(RuntimeError, "H2D submit"):
                coordinator.prefetch(key, request_id="h2d-submit")
        finally:
            coordinator._executor.submit = original_submit
        self.assertFalse(coordinator._gpu_resident(key))
        self.assertTrue(coordinator._cpu_resident(key))
        self.assertEqual(hot.free_pages, 1)
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        stats = coordinator.stats()
        self.assertEqual(stats["migration_failures"], 1)
        self.assertEqual(stats["pending_tier_operations"], 0)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_d2h_submit_failure_keeps_gpu_page_and_releases_io_pin(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 37
        )
        original_submit = coordinator._executor.submit

        def fail_submit(*args, **kwargs):
            raise RuntimeError("injected D2H submit failure")

        coordinator._executor.submit = fail_submit
        try:
            with self.assertRaisesRegex(RuntimeError, "D2H submit"):
                coordinator.evict(key)
        finally:
            coordinator._executor.submit = original_submit
        self.assertTrue(coordinator._gpu_resident(key))
        self.assertFalse(coordinator._cpu_resident(key))
        self.assertEqual(cpu.used_bytes, 0)
        descriptor = pool.descriptor(handles[0])
        self.assertEqual(descriptor.pin_count, 0)
        self.assertEqual(descriptor.inflight_io, 0)
        stats = coordinator.stats()
        self.assertEqual(stats["migration_failures"], 1)
        self.assertEqual(stats["pending_tier_operations"], 0)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_selected_wave_is_not_an_eviction_victim(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=2, gpu_pages=1
        )
        first, _ = self.populate_gpu(
            coordinator, hot, "first", handles[0], 1, 1
        )
        with coordinator.acquire_wave((first,), request_id="active"):
            second = coordinator.register_page(
                "second", handles[1], valid_tokens=4, data_epoch=2
            )
            reservations = []
            for layer in range(hot.layer_count):
                reservation = cpu.reserve()
                handle = cpu.allocate(
                    reservation,
                    logical_block_id="second",
                    layer=layer,
                    data_epoch=2,
                )
                cpu.write(
                    handle,
                    torch.ones(2, 4, 3),
                    torch.ones(2, 4, 3),
                    valid_tokens=4,
                    data_epoch=2,
                )
                reservations.append(handle)
            hot.attach_cpu_page(second, dict(enumerate(reservations)))
            with self.assertRaisesRegex(KVCapacityError, "no legal"):
                coordinator.prefetch(second, request_id="blocked")
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_compute_event_keeps_wave_pinned_until_fence_completion(self):
        event = _ControlledEvent()
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        coordinator._compute_event_factory = lambda: event
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 11
        )
        wave = coordinator.acquire_wave((key,), request_id="compute-fence")
        fence = wave.submit_close()
        self.assertEqual(pool.descriptor(handles[0]).inflight_compute, 1)
        self.assertEqual(hot.lru_victims(), ())
        event.ready.set()
        fence.wait(timeout_seconds=2.0)
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        self.assertEqual(pool.descriptor(handles[0]).inflight_compute, 0)
        self.assertEqual(hot.lru_victims(), (key,))
        coordinator.close()
        hot.close()
        cpu.close()

    def test_advance_epoch_rekeys_and_releases_stale_cpu_replicas(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 13
        )
        coordinator.evict(key).wait()
        coordinator.prefetch(key, request_id="restore").wait()
        self.assertGreater(cpu.used_bytes, 0)
        original_ref = pool.descriptor(handles[0]).ref_count
        hot.begin_pending(key, "append")
        pool.mark_data_updated(handles[0], version=2)
        new_key = coordinator.advance_data_epoch(
            key, 2, valid_tokens=4
        )
        self.assertEqual(new_key.data_epoch, 2)
        self.assertEqual(cpu.used_bytes, 0)
        self.assertTrue(coordinator._gpu_resident(new_key))
        self.assertEqual(pool.descriptor(handles[0]).ref_count, original_ref)
        with self.assertRaises(KeyError):
            coordinator._registration(key)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_prepare_append_evicts_for_new_page_and_epoch_commit_rekeys(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=2, gpu_pages=1
        )
        first, _ = self.populate_gpu(
            coordinator, hot, "first", handles[0], 1, 1
        )
        second = coordinator.register_page(
            "second", handles[1], valid_tokens=4, data_epoch=2
        )
        slot, operation = coordinator.prepare_append(
            second, request_id="append", timeout=2.0
        )
        self.assertFalse(coordinator._gpu_resident(first))
        self.assertTrue(coordinator._cpu_resident(first))
        for layer in range(hot.layer_count):
            key_target, value_target = hot.layer_write_target(
                second, layer, slot
            )
            key_target.fill_(float(layer + 20))
            value_target.fill_(float(layer + 120))
            hot.commit_gpu_layer(second, layer, slot)
        pool.mark_data_updated(handles[1], version=3)
        committed = coordinator.advance_data_epoch(
            second, 3, valid_tokens=4, end_operation=operation
        )
        self.assertTrue(coordinator._gpu_resident(committed))
        self.assertEqual(hot.lru_victims(), (committed,))
        coordinator.close()
        hot.close()
        cpu.close()

    def test_current_append_layer_can_attend_while_slot_is_loading(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key = coordinator.register_page(
            "partial", handles[0], valid_tokens=4, data_epoch=1
        )
        slot, operation = coordinator.prepare_append(
            key, request_id="partial-append"
        )
        key_target, value_target = hot.layer_write_target(key, 0, slot)
        key_target.fill_(23.0)
        value_target.fill_(123.0)
        hot.commit_gpu_layer(key, 0, slot)

        with coordinator.acquire_wave(
            (key,), request_id="partial-layer-0", layer=0
        ) as wave:
            page = wave.layer_pages(0)[0]
            self.assertTrue(torch.equal(page.key, key_target))
            self.assertTrue(torch.equal(page.value, value_target))
            self.assertEqual(pool.descriptor(handles[0]).inflight_compute, 1)
        with self.assertRaisesRegex(
            KVLifecycleError, "every CPU layer replica"
        ):
            coordinator.acquire_wave(
                (key,), request_id="partial-layer-1", layer=1
            )
        coordinator.abort_append(key, slot, operation)
        coordinator.close()
        hot.close()
        cpu.close()

    def test_close_waits_for_pending_migration_then_releases_locations(self):
        event = _ControlledEvent()
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 17
        )
        coordinator.evict(key).wait()
        coordinator._event_factory = lambda: event
        coordinator.prefetch(key, request_id="close")
        errors = []

        def close_coordinator():
            try:
                coordinator.close()
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=close_coordinator)
        worker.start()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if pool.descriptor(handles[0]).inflight_io:
                break
            time.sleep(0.001)
        self.assertEqual(pool.descriptor(handles[0]).inflight_io, 1)
        self.assertTrue(worker.is_alive())
        event.ready.set()
        worker.join(timeout=2.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        self.assertEqual(cpu.used_bytes, 0)
        self.assertEqual(hot.stats()["registered_logical_pages"], 0)
        hot.close()
        cpu.close()

    def test_close_does_not_cancel_queued_cleanup_worker_twenty_rounds(self):
        for iteration in range(20):
            with self.subTest(iteration=iteration):
                coordinator, hot, cpu, pool, handles = self.make_fixture(
                    page_count=1, gpu_pages=1
                )
                key, _ = self.populate_gpu(
                    coordinator, hot, "logical", handles[0], 1, iteration
                )
                coordinator.evict(key).wait()
                blocker = threading.Event()
                occupying = tuple(
                    coordinator._executor.submit(blocker.wait)
                    for _ in range(2)
                )
                fence = coordinator.prefetch(key, request_id="queued-close")
                errors = []

                def close_coordinator():
                    try:
                        coordinator.close()
                    except BaseException as error:
                        errors.append(error)

                worker = threading.Thread(target=close_coordinator)
                worker.start()
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline and not fence.cancelled:
                    time.sleep(0.001)
                self.assertTrue(fence.cancelled)
                self.assertFalse(fence.io_future.cancelled())
                self.assertEqual(
                    pool.descriptor(handles[0]).inflight_io, 1
                )
                blocker.set()
                for item in occupying:
                    item.result(timeout=1.0)
                worker.join(timeout=2.0)
                self.assertFalse(worker.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
                self.assertEqual(hot.stats()["registered_logical_pages"], 0)
                hot.close()
                cpu.close()

    def test_direct_cancel_before_worker_start_runs_submitter_cleanup(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "logical", handles[0], 1, 19
        )
        coordinator.evict(key).wait()
        blocker = threading.Event()
        occupying = tuple(
            coordinator._executor.submit(blocker.wait) for _ in range(2)
        )
        fence = coordinator.prefetch(key, request_id="direct-cancel")
        fence.cancel()
        self.assertTrue(fence.io_future.cancelled())
        self.assertEqual(pool.descriptor(handles[0]).pin_count, 0)
        self.assertEqual(hot.free_pages, 1)
        self.assertTrue(coordinator._cpu_resident(key))
        blocker.set()
        for item in occupying:
            item.result(timeout=1.0)
        coordinator.close()
        hot.close()
        cpu.close()


    def test_hot_cache_execution_fence_blocks_slot_reuse_until_event(self):
        event = _ControlledEvent()
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=2,
            gpu_pages=2,
            compute_event_factory=lambda: event,
        )
        key, _ = self.populate_gpu(
            coordinator, hot, "guarded", handles[0], 1, 7
        )
        # Establish a same-epoch CPU replica so release/eviction would otherwise
        # be legal and the guard is the only blocking condition.
        coordinator.evict(key).wait()
        coordinator.prefetch(key).wait()
        ref_before = pool.descriptor(handles[0]).ref_count

        guard = coordinator.begin_hot_cache_execution(request_id="device-attn")
        fence = guard.submit_close()
        self.assertTrue(event.started.wait(timeout=1.0))
        self.assertEqual(hot.active_execution_guards, 1)
        self.assertEqual(hot.lru_victims(), ())
        with self.assertRaisesRegex(KVLifecycleError, "execution is inflight"):
            hot.release_gpu(key)
        with self.assertRaisesRegex(KVLifecycleError, "execution is inflight"):
            hot.reserve_gpu(key)
        with self.assertRaisesRegex(KVLifecycleError, "execution is inflight"):
            coordinator.evict(key)
        descriptor = pool.descriptor(handles[0])
        self.assertEqual(descriptor.ref_count, ref_before)
        self.assertEqual(descriptor.pin_count, 0)
        self.assertEqual(descriptor.inflight_compute, 0)

        event.ready.set()
        fence.wait(timeout_seconds=2.0)
        self.assertTrue(guard.closed)
        self.assertEqual(hot.active_execution_guards, 0)
        self.assertEqual(
            coordinator.stats()["active_hot_cache_execution_guards"], 0
        )
        hot.release_gpu(key)
        coordinator.close()

    def test_hot_cache_execution_guard_closes_on_abort_and_event_failure(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        guard = coordinator.begin_hot_cache_execution(request_id="abort")
        guard.abort()
        self.assertTrue(guard.closed)
        self.assertEqual(hot.active_execution_guards, 0)
        guarded_error = coordinator.begin_hot_cache_execution(
            request_id="context-error"
        )
        with self.assertRaisesRegex(RuntimeError, "provider launch failed"):
            with guarded_error:
                raise RuntimeError("provider launch failed")
        self.assertTrue(guarded_error.closed)
        self.assertEqual(hot.active_execution_guards, 0)
        coordinator.close()

        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1,
            gpu_pages=1,
            compute_event_factory=_FailingEvent,
        )
        guard = coordinator.begin_hot_cache_execution(request_id="failure")
        fence = guard.submit_close()
        with self.assertRaisesRegex(RuntimeError, "execution event failure"):
            fence.wait(timeout_seconds=2.0)
        self.assertTrue(guard.closed)
        self.assertEqual(hot.active_execution_guards, 0)
        self.assertEqual(coordinator.stats()["pending_tier_operations"], 0)
        coordinator.close()

    def test_coordinator_close_reclaims_unsubmitted_execution_guard(self):
        coordinator, hot, cpu, pool, handles = self.make_fixture(
            page_count=1, gpu_pages=1
        )
        guard = coordinator.begin_hot_cache_execution(request_id="close")
        self.assertEqual(hot.active_execution_guards, 1)
        coordinator.close()
        self.assertTrue(guard.closed)
        self.assertEqual(hot.active_execution_guards, 0)


if __name__ == "__main__":
    unittest.main()
