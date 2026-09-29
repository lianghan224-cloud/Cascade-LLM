import unittest

import torch

from layer_streaming.kv.device_metadata import (
    DEVICE_PAGE_ABSENT,
    DEVICE_PAGE_CPU,
    DEVICE_PAGE_GPU,
    DEVICE_PAGE_GPU_CPU,
)
from layer_streaming.kv.errors import KVLifecycleError
from layer_streaming.kv.page_pool import KVPagePoolV1
from layer_streaming.kv.stores import GPUHotKVCache


def _cpu_factory(shape, **kwargs):
    return torch.empty(shape, dtype=kwargs["dtype"], device="cpu")


class GPUHotKVCacheTest(unittest.TestCase):
    def make_pages(self, count=16, layers=2):
        pool = KVPagePoolV1(
            count, store_id="gpu", dtype="fp32", layout="hnd"
        )
        handles = []
        for index in range(count):
            handle = pool.allocate(owner_hint="request-{}".format(index))
            pool.activate(handle, 4)
            pool.mark_data_updated(handle, version=index + 1)
            pool.seal(handle, 4)
            handles.append(handle)
        cache = GPUHotKVCache(
            page_pool=pool,
            layer_count=layers,
            gpu_capacity_pages=4,
            num_kv_heads=2,
            page_size=4,
            head_dim=3,
            dtype=torch.float32,
            device="cpu",
            high_watermark_pages=4,
            low_watermark_pages=1,
            tensor_factory=_cpu_factory,
        )
        return pool, handles, cache

    @staticmethod
    def fill_and_commit(cache, key, slot, value):
        for layer in range(cache.layer_count):
            target_key, target_value = cache.layer_write_target(
                key, layer, slot
            )
            target_key.fill_(float(value + layer))
            target_value.fill_(float(value + layer + 100))
        cache.commit_gpu(key, slot, make_authoritative=False)

    def test_four_gpu_slots_cycle_through_sixteen_logical_pages(self):
        pool, handles, cache = self.make_pages()
        original_refs = [pool.descriptor(handle).ref_count for handle in handles]
        keys = [
            cache.register_page(
                "logical-{}".format(index),
                handle,
                index + 1,
                cpu_slots={
                    layer: "cpu-{}-{}".format(index, layer)
                    for layer in range(cache.layer_count)
                },
            )
            for index, handle in enumerate(handles)
        ]
        self.assertEqual(cache.stats()["registered_logical_pages"], 16)
        self.assertEqual(cache.free_pages, 4)

        previous_handles = None
        for wave in range(4):
            wave_keys = keys[wave * 4 : (wave + 1) * 4]
            slot_handles = []
            for offset, key in enumerate(wave_keys):
                slot = cache.reserve_gpu(key)
                self.fill_and_commit(cache, key, slot, wave * 10 + offset)
                metadata = cache.device_page_table.gather(
                    torch.tensor((offset,)), torch.tensor((key.page_id,))
                )
                self.assertEqual(
                    metadata.physical_gpu_slots.tolist(), [slot.slot_id]
                )
                self.assertEqual(
                    metadata.location_states.tolist(), [DEVICE_PAGE_GPU_CPU]
                )
                slot_handles.append(slot)
                locations = cache.location_sets(key)
                self.assertEqual(
                    {location.gpu_slot for location in locations}, {slot}
                )
            self.assertEqual(cache.used_pages, 4)
            self.assertEqual(cache.free_pages, 0)
            self.assertTrue(cache.above_high_watermark)
            if previous_handles is not None:
                self.assertEqual(
                    {handle.slot_id for handle in slot_handles},
                    {handle.slot_id for handle in previous_handles},
                )
                previous_generations = {
                    handle.slot_id: handle.generation
                    for handle in previous_handles
                }
                self.assertTrue(
                    all(
                        handle.generation
                        == previous_generations[handle.slot_id] + 1
                        for handle in slot_handles
                    )
                )
                with self.assertRaises(KVLifecycleError):
                    cache.layer_write_target(
                        wave_keys[0], 0, previous_handles[0]
                    )
            for key in wave_keys:
                cache.release_gpu(key)
                metadata = cache.device_page_table.gather(
                    torch.tensor((0,)), torch.tensor((key.page_id,))
                )
                self.assertEqual(metadata.physical_gpu_slots.tolist(), [-1])
                self.assertEqual(
                    metadata.location_states.tolist(), [DEVICE_PAGE_CPU]
                )
            self.assertEqual(cache.used_pages, 0)
            self.assertEqual(cache.free_pages, 4)
            self.assertTrue(cache.below_low_watermark)
            previous_handles = slot_handles

        self.assertEqual(
            [pool.descriptor(handle).ref_count for handle in handles],
            original_refs,
        )
        stats = cache.stats()
        self.assertEqual(stats["gpu_kv_capacity_pages"], 4)
        self.assertEqual(stats["gpu_kv_peak_used_pages"], 4)
        self.assertEqual(
            stats["gpu_kv_capacity_bytes"], 4 * cache.gpu_page_bytes
        )
        cache.close()

    def test_layer_publish_epoch_advance_and_cleanup_safe_abort(self):
        pool, handles, cache = self.make_pages(count=2, layers=2)
        key = cache.register_page(
            "append",
            handles[0],
            1,
            cpu_slots={0: "stale-cpu-0", 1: "stale-cpu-1"},
        )
        slot = cache.reserve_gpu(key)
        cache.begin_pending(key, "append")
        for layer in range(2):
            target_key, target_value = cache.layer_write_target(
                key, layer, slot
            )
            target_key.fill_(layer + 1)
            target_value.fill_(layer + 11)
            cache.commit_gpu_layer(key, layer, slot)
            self.assertEqual(
                cache.resolve_gpu_layer(key, layer).slot, slot
            )
            # The page remains excluded until the cross-layer append commits.
            self.assertEqual(cache.lru_victims(), ())

        pool.mark_data_updated(handles[0], version=9)
        new_key, stale_cpu = cache.advance_data_epoch(
            key, 9, end_operation="append"
        )
        self.assertEqual(stale_cpu, ("stale-cpu-0", "stale-cpu-1"))
        self.assertEqual(cache.resolve_gpu_layer(new_key, 0).slot, slot)
        self.assertEqual(cache.lru_victims(), (new_key,))
        self.assertEqual(
            {location.authoritative_tier for location in cache.location_sets(new_key)},
            {"gpu"},
        )
        metadata = cache.device_page_table.gather(
            torch.tensor((0,)),
            torch.tensor((new_key.page_id,)),
            expected_epochs=torch.tensor((9,)),
        )
        self.assertEqual(metadata.data_epochs.tolist(), [9])
        self.assertEqual(metadata.location_states.tolist(), [DEVICE_PAGE_GPU])
        cache.unregister_page(new_key)
        metadata = cache.device_page_table.gather(
            torch.tensor((0,)), torch.tensor((new_key.page_id,))
        )
        self.assertEqual(metadata.location_states.tolist(), [DEVICE_PAGE_ABSENT])

        abort_key = cache.register_page(
            "abort",
            handles[1],
            2,
            cpu_slots={0: "cpu-0", 1: "cpu-1"},
        )
        abort_slot = cache.reserve_gpu(abort_key)
        cache.begin_pending(abort_key, "prefetch")
        cache.pin_slot(abort_key, kind="io")
        pool.mark_data_updated(handles[1], version=10)
        cache.abort_gpu_load(
            abort_key,
            abort_slot,
            operation="prefetch",
            kind="io",
        )
        self.assertEqual(cache.used_pages, 0)
        self.assertEqual(
            cache.unregister_page(abort_key, validate_epoch=False),
            ("cpu-0", "cpu-1"),
        )
        cache.close()

    def test_one_cross_layer_slot_and_cpu_replica_before_release(self):
        pool, handles, cache = self.make_pages(count=2, layers=3)
        key = cache.register_page("logical", handles[0], 1)
        slot = cache.reserve_gpu(key)
        with self.assertRaises(KVLifecycleError):
            cache.reserve_gpu(key)
        self.fill_and_commit(cache, key, slot, 7)

        resolved = [cache.resolve_gpu_layer(key, layer) for layer in range(3)]
        self.assertEqual({page.slot for page in resolved}, {slot})
        self.assertEqual(slot.slot_id, 0)
        self.assertNotEqual(slot.slot_id, handles[0].page_id + 1)
        with self.assertRaises(KVLifecycleError):
            cache.release_gpu(key)
        for layer in range(3):
            cache.attach_cpu(key, layer, "cpu-{}".format(layer))
        cache.release_gpu(key)
        self.assertEqual(cache.free_pages, 4)
        cache.close()

    def test_cpu_page_location_publication_is_atomic(self):
        pool, handles, cache = self.make_pages(count=1, layers=2)
        key = cache.register_page("logical", handles[0], 1)
        with self.assertRaises(ValueError):
            cache.attach_cpu_page(key, {0: "cpu0"})
        self.assertTrue(
            all(location.cpu_slot is None for location in cache.location_sets(key))
        )
        locations = cache.attach_cpu_page(
            key, {0: "cpu0", 1: "cpu1"}, make_authoritative=True
        )
        self.assertEqual(
            tuple(location.cpu_slot for location in locations),
            ("cpu0", "cpu1"),
        )
        self.assertEqual(
            {location.authoritative_tier for location in locations}, {"cpu"}
        )
        cache.close()

    def test_gpu_store_compatibility_surface_uses_hot_slot_ids(self):
        pool, handles, cache = self.make_pages(count=2, layers=2)
        keys = [
            cache.register_page(
                "logical-{}".format(index),
                handle,
                index + 1,
                cpu_slots={0: "cpu-{}-0".format(index), 1: "cpu-{}-1".format(index)},
            )
            for index, handle in enumerate(handles)
        ]
        slots = [cache.reserve_gpu(key) for key in keys]
        payload_shape = (2, cache.num_kv_heads, cache.page_size, cache.head_dim)
        key_payload = torch.full(payload_shape, 3.0, dtype=cache.dtype)
        value_payload = torch.full(payload_shape, 5.0, dtype=cache.dtype)
        cache.write_pages(
            0, [slot.slot_id for slot in slots], key_payload, value_payload
        )
        read_key, read_value = cache.read_pages(
            0, [slot.slot_id for slot in slots]
        )
        self.assertTrue(torch.equal(read_key, key_payload))
        self.assertTrue(torch.equal(read_value, value_payload))
        self.assertEqual(cache.page_count, cache.gpu_capacity_pages)
        self.assertEqual(cache.nbytes, cache.gpu_capacity_bytes)
        self.assertTrue(cache.capability().supports_active_attention)
        for key, slot in zip(keys, slots):
            # Finish untouched layer 1 before publishing the complete slot.
            target_key, target_value = cache.layer_write_target(key, 1, slot)
            target_key.zero_()
            target_value.zero_()
            cache.commit_gpu(key, slot, make_authoritative=False)
        cache.close()

    def test_generation_and_data_epoch_are_revalidated(self):
        pool, handles, cache = self.make_pages(count=2)
        with self.assertRaises(KVLifecycleError):
            cache.register_page("wrong-epoch", handles[0], 99)

        key = cache.register_page(
            "epoch", handles[0], 1, cpu_slots={0: "cpu0", 1: "cpu1"}
        )
        pool.mark_data_updated(handles[0], version=17)
        with self.assertRaises(KVLifecycleError):
            cache.location_set(key, 0)
        cache.close()

        pool2, handles2, cache2 = self.make_pages(count=2)
        stale_key = cache2.register_page(
            "generation",
            handles2[0],
            1,
            cpu_slots={0: "cpu0", 1: "cpu1"},
        )
        pool2.release(handles2[0])
        reused = pool2.allocate(owner_hint="new-request")
        self.assertEqual(reused.page_id, handles2[0].page_id)
        self.assertNotEqual(reused.generation, handles2[0].generation)
        with self.assertRaises(KVLifecycleError):
            cache2.location_set(stale_key, 0)
        cache2.close()

    def test_lru_excludes_pin_inflight_pending_selected_and_page_busy(self):
        pool, handles, cache = self.make_pages(count=4)
        keys = []
        for index, handle in enumerate(handles):
            key = cache.register_page(
                "logical-{}".format(index),
                handle,
                index + 1,
                cpu_slots={0: "cpu-{}-0".format(index), 1: "cpu-{}-1".format(index)},
            )
            slot = cache.reserve_gpu(key)
            self.fill_and_commit(cache, key, slot, index)
            keys.append(key)

        cache.pin_slot(keys[0], kind="compute")
        cache.mark_selected(keys[1])
        cache.begin_pending(keys[2], "append")
        pool.pin(handles[3], kind="io")
        self.assertEqual(cache.lru_victims(), ())

        cache.unpin_slot(keys[0], kind="compute")
        cache.unmark_selected(keys[1])
        cache.end_pending(keys[2], "append")
        pool.unpin(handles[3], kind="io")
        victims = cache.lru_victims()
        self.assertEqual(set(victims), set(keys))
        self.assertEqual(len(victims), 4)
        self.assertEqual(len(cache.lru_victims(exclude_keys={keys[1]}, limit=2)), 2)
        cache.close()

    def test_unregister_releases_gpu_slot_and_returns_cpu_locations(self):
        pool, handles, cache = self.make_pages(count=1)
        key = cache.register_page(
            "logical",
            handles[0],
            1,
            cpu_slots={0: "cpu0", 1: "cpu1"},
        )
        slot = cache.reserve_gpu(key)
        self.fill_and_commit(cache, key, slot, 1)
        cpu_slots = cache.unregister_page(key)
        self.assertEqual(cpu_slots, ("cpu0", "cpu1"))
        self.assertEqual(cache.used_pages, 0)
        self.assertEqual(cache.stats()["registered_logical_pages"], 0)
        with self.assertRaises(KeyError):
            cache.location_set(key, 0)
        cache.close()


if __name__ == "__main__":
    unittest.main()
