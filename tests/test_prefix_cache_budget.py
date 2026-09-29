import unittest
from types import SimpleNamespace

from layer_streaming.kv.errors import KVLifecycleError
from layer_streaming.kv.metrics import KVMetrics
from layer_streaming.kv.page_pool import KVPagePoolV1
from layer_streaming.kv.prefix_cache import PrefixCache


PAGE_SIZE = 4
PAGE_BYTES = 64


class PrefixCacheBudgetTest(unittest.TestCase):
    def setUp(self):
        self.pool = KVPagePoolV1(8, "prefix-test", "bf16")
        self.metrics = KVMetrics()

    def make_cache(self, max_pages=None, max_bytes=None, page_bytes=PAGE_BYTES):
        return PrefixCache(
            PAGE_SIZE,
            self.pool,
            layer_count=1,
            metrics=self.metrics,
            max_prefix_pages=max_pages,
            max_prefix_bytes=max_bytes,
            page_bytes=page_bytes,
        )

    def make_state(self, request_id, page_count=1, namespace="default"):
        handles = []
        for logical in range(page_count):
            mapping = ("request", int(request_id), logical)
            handle = self.pool.allocate(
                owner_hint=request_id,
                logical_mapping=mapping,
            )
            self.pool.activate(handle, PAGE_SIZE)
            self.pool.seal(handle, PAGE_SIZE)
            handles.append(handle)
        return SimpleNamespace(
            request_id=int(request_id),
            reuse_namespace=namespace,
            pending_append=None,
            block_table=SimpleNamespace(handles=handles),
            token_block_hashes=[],
        )

    def release_state(self, state):
        for logical, handle in reversed(tuple(enumerate(state.block_table.handles))):
            self.pool.release(
                handle,
                logical_mapping=("request", state.request_id, logical),
            )

    def test_page_budget_uses_lru_touch(self):
        cache = self.make_cache(max_pages=2)
        first = self.make_state(1)
        second = self.make_state(2)
        third = self.make_state(3)
        first_tokens = [1, 2, 3, 4]
        second_tokens = [5, 6, 7, 8]
        third_tokens = [9, 10, 11, 12]

        cache.register(first, first_tokens)
        cache.register(second, second_tokens)
        first_hash = first.token_block_hashes[0]
        second_hash = second.token_block_hashes[0]
        cache.touch("default", first_hash)
        cache.register(third, third_tokens)

        self.assertIsNotNone(cache.index.get_entry("default", first_hash))
        self.assertIsNone(cache.index.get_entry("default", second_hash))
        self.assertEqual(cache.stats()["prefix_pages"], 2)
        self.assertEqual(cache.stats()["prefix_evictions"], 1)
        self.assertEqual(self.pool.descriptor(second.block_table.handles[0]).ref_count, 1)

        cache.evict_all()
        for state in (first, second, third):
            self.release_state(state)
        self.assertEqual(self.pool.allocated_pages, 0)
        self.assertTrue(self.pool.validate_invariants())

    def test_byte_budget_retains_shortest_admissible_prefix(self):
        cache = self.make_cache(max_bytes=PAGE_BYTES)
        state = self.make_state(10, page_count=2)
        tokens = list(range(PAGE_SIZE * 2))

        cache.register(state, tokens)
        match = cache.index.lookup("default", tokens)

        self.assertEqual(len(cache), 1)
        self.assertEqual(match.matched_tokens, PAGE_SIZE)
        self.assertEqual(cache.stats()["prefix_pages"], 1)
        self.assertEqual(cache.stats()["prefix_bytes"], PAGE_BYTES)
        entry = cache.index.entries()[0]
        self.assertEqual(entry.page_count, 1)
        self.assertEqual(entry.retained_bytes, PAGE_BYTES)

        cache.evict_all()
        self.release_state(state)
        self.assertEqual(self.pool.allocated_pages, 0)

    def test_same_hash_replace_releases_old_owner_once(self):
        cache = self.make_cache()
        old_state = self.make_state(20)
        new_state = self.make_state(21)
        tokens = [11, 12, 13, 14]

        cache.register(old_state, tokens)
        old_handle = old_state.block_table.handles[0]
        self.assertEqual(self.pool.descriptor(old_handle).ref_count, 2)
        cache.register(new_state, tokens)

        new_handle = new_state.block_table.handles[0]
        self.assertEqual(self.pool.descriptor(old_handle).ref_count, 1)
        self.assertEqual(self.pool.descriptor(new_handle).ref_count, 2)
        self.assertEqual(cache.stats()["prefix_replacements"], 1)
        self.assertEqual(cache.stats()["prefix_pages"], 1)

        cache.evict_all()
        self.assertEqual(self.pool.descriptor(new_handle).ref_count, 1)
        self.release_state(old_state)
        self.release_state(new_state)
        self.assertEqual(self.pool.allocated_pages, 0)

    def test_stale_entry_cannot_evict_same_hash_replacement(self):
        cache = self.make_cache()
        old_state = self.make_state(22)
        new_state = self.make_state(23)
        tokens = [4, 3, 2, 1]
        cache.register(old_state, tokens)
        stale_entry = cache.index.entries()[0]
        cache.register(new_state, tokens)

        self.assertFalse(cache.evict(stale_entry))
        current = cache.index.entries()[0]
        self.assertNotEqual(current.created_epoch, stale_entry.created_epoch)
        self.assertEqual(current.handles, tuple(new_state.block_table.handles))

        cache.evict_all()
        self.release_state(old_state)
        self.release_state(new_state)

    def test_entry_eviction_releases_only_orphaned_prefix_owners(self):
        cache = self.make_cache()
        state = self.make_state(30, page_count=2)
        cache.register(state, list(range(PAGE_SIZE * 2)))
        first_entry, second_entry = sorted(
            cache.index.entries(), key=lambda item: item.page_count
        )

        self.assertTrue(cache.evict(first_entry))
        self.assertEqual(cache.stats()["prefix_pages"], 2)
        for handle in state.block_table.handles:
            self.assertEqual(self.pool.descriptor(handle).ref_count, 2)

        self.assertTrue(cache.evict(second_entry))
        self.assertEqual(cache.stats()["prefix_pages"], 0)
        for handle in state.block_table.handles:
            self.assertEqual(self.pool.descriptor(handle).ref_count, 1)

        self.release_state(state)
        self.assertEqual(self.pool.allocated_pages, 0)

    def test_origin_and_cache_owners_are_independent(self):
        cache = self.make_cache()
        state = self.make_state(40)
        cache.register(state, [1, 3, 5, 7])
        handle = state.block_table.handles[0]

        self.release_state(state)
        self.assertEqual(self.pool.descriptor(handle).ref_count, 1)
        self.assertEqual(cache.index.lookup("default", [1, 3, 5, 7]).page_handles, (handle,))
        cache.evict_all()
        self.assertEqual(self.pool.allocated_pages, 0)

    def test_configure_page_bytes_updates_entries_and_evicts(self):
        cache = self.make_cache(page_bytes=32)
        first = self.make_state(50)
        second = self.make_state(51)
        cache.register(first, [1, 1, 1, 1])
        cache.register(second, [2, 2, 2, 2])

        evicted = cache.configure_limits(max_prefix_bytes=64, page_bytes=64)

        self.assertEqual(evicted, 1)
        self.assertEqual(cache.stats()["prefix_pages"], 1)
        self.assertEqual(cache.stats()["prefix_bytes"], 64)
        self.assertEqual(cache.index.entries()[0].retained_bytes, 64)
        cache.evict_all()
        self.release_state(first)
        self.release_state(second)

    def test_thousand_replacements_do_not_grow_owners(self):
        cache = self.make_cache(max_pages=1)
        tokens = [7, 7, 7, 7]
        for request_id in range(1000, 2000):
            state = self.make_state(request_id)
            cache.register(state, tokens)
            self.release_state(state)
            self.assertEqual(cache.stats()["prefix_pages"], 1)
            self.assertEqual(self.pool.allocated_pages, 1)
            self.assertTrue(self.pool.validate_invariants())

        self.assertEqual(cache.stats()["prefix_replacements"], 999)
        cache.evict_all()
        self.assertEqual(self.pool.allocated_pages, 0)
        self.assertTrue(self.pool.validate_invariants())

    def test_rejects_unsealed_or_partial_pages(self):
        cache = self.make_cache()
        handle = self.pool.allocate(
            owner_hint=60,
            logical_mapping=("request", 60, 0),
        )
        self.pool.activate(handle, PAGE_SIZE - 1)
        partial = SimpleNamespace(
            request_id=60,
            reuse_namespace="default",
            pending_append=None,
            block_table=SimpleNamespace(handles=[handle]),
            token_block_hashes=[],
        )
        with self.assertRaisesRegex(KVLifecycleError, "not sealed"):
            cache.register(partial, [1, 2, 3, 4])
        self.pool.seal(handle, PAGE_SIZE - 1)
        with self.assertRaisesRegex(KVLifecycleError, "not full"):
            cache.register(partial, [1, 2, 3, 4])
        self.release_state(partial)

    def test_zero_and_exact_boundary_budgets(self):
        cases = (
            {"max_pages": 0, "max_bytes": None, "expected_pages": 0},
            {"max_pages": None, "max_bytes": 0, "expected_pages": 0},
            {"max_pages": 1, "max_bytes": None, "expected_pages": 1},
            {
                "max_pages": None,
                "max_bytes": PAGE_BYTES,
                "expected_pages": 1,
            },
        )
        for index, case in enumerate(cases):
            with self.subTest(**case):
                cache = self.make_cache(
                    max_pages=case["max_pages"],
                    max_bytes=case["max_bytes"],
                )
                state = self.make_state(70 + index)
                cache.register(state, [1, 2, 3, 4])
                self.assertEqual(
                    cache.stats()["prefix_pages"], case["expected_pages"]
                )
                self.assertEqual(
                    len(cache), 1 if case["expected_pages"] else 0
                )
                cache.evict_all()
                self.release_state(state)
                self.assertEqual(self.pool.allocated_pages, 0)

    def test_budget_rejects_negative_values(self):
        for keyword in ("max_prefix_pages", "max_prefix_bytes"):
            with self.subTest(keyword=keyword):
                values = {keyword: -1}
                if keyword == "max_prefix_bytes":
                    values["page_bytes"] = PAGE_BYTES
                with self.assertRaisesRegex(ValueError, "cannot be negative"):
                    PrefixCache(
                        PAGE_SIZE,
                        self.pool,
                        layer_count=1,
                        metrics=self.metrics,
                        **values,
                    )


if __name__ == "__main__":
    unittest.main()
