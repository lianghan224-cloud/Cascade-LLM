from dataclasses import dataclass
import unittest

from layer_streaming.moe import (
    ExpertCache,
    ExpertKey,
    ExpertResidencyManager,
    ExpertResidencyState,
    FrequencyAwareExpertPolicy,
    LRUExpertPolicy,
    MoEMemoryPlanner,
    UnifiedResidentWeightBudget,
)


class ResidencyTest(unittest.TestCase):
    def setUp(self):
        self.manager = ExpertResidencyManager()
        self.key = ExpertKey(2, 7)
        self.manager.register(self.key)

    def _ready(self):
        self.manager.mark_inflight(self.key)
        return self.manager.mark_ready(self.key, location="slot-1", ready_event="event")

    def test_acquire_release_pin_and_eviction_protection(self):
        self._ready()
        self.manager.acquire(self.key)
        self.assertEqual(self.manager.require(self.key).state, ExpertResidencyState.IN_USE)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            self.manager.evict(self.key)
        self.manager.release(self.key)
        self.manager.pin(self.key)
        with self.assertRaisesRegex(RuntimeError, "protected"):
            self.manager.evict(self.key)
        self.manager.unpin(self.key)
        self.manager.evict(self.key)
        self.assertEqual(
            self.manager.require(self.key).state, ExpertResidencyState.CPU_RESIDENT
        )

    def test_inflight_failure_rolls_back(self):
        def fail(_record):
            raise OSError("copy submission failed")

        with self.assertRaisesRegex(OSError, "submission"):
            self.manager.ensure_resident(self.key, fail)
        record = self.manager.require(self.key)
        self.assertEqual(record.state, ExpertResidencyState.CPU_RESIDENT)
        self.assertIsNone(record.ready_event)

    def test_exception_lease_releases_and_close_requires_quiescence(self):
        self._ready()
        with self.assertRaisesRegex(RuntimeError, "body"):
            with self.manager.lease(self.key):
                raise RuntimeError("body")
        self.assertEqual(self.manager.stats()["use_count"], 0)
        self.manager.pin(self.key)
        with self.assertRaisesRegex(RuntimeError, "non-quiescent"):
            self.manager.close()
        self.manager.unpin(self.key)
        self.manager.close()
        self.assertTrue(self.manager.stats()["closed"])

    def test_reset_rejects_inflight_then_reaches_quiescence(self):
        self.manager.mark_inflight(self.key)
        self.assertFalse(self.manager.quiescent())
        with self.assertRaisesRegex(RuntimeError, "non-quiescent"):
            self.manager.reset()
        self.manager.rollback_inflight(self.key)
        self.manager.reset()
        self.assertTrue(self.manager.quiescent())


class ExpertCacheTest(unittest.TestCase):
    def setUp(self):
        self.manager = ExpertResidencyManager()
        self.keys = [ExpertKey(0, value) for value in range(6)]
        for key in self.keys:
            self.manager.register(key)

    def _cache(self, count, policy=None):
        return ExpertCache(
            self.manager,
            UnifiedResidentWeightBudget(
                total_bytes=100 + count * 10,
                fixed_resident_bytes=100,
            ),
            policy=policy,
        )

    def _load(self, cache, key):
        cache.reserve(key, 10)
        self.manager.mark_inflight(key)
        self.manager.mark_ready(key, location="gpu:" + str(key.expert_id), ready_event=object())
        cache.mark_ready(key, location="gpu:" + str(key.expert_id), ready_event=object())
        # A completed but unused cache entry is immediately evictable.
        self.manager.acquire(key)
        self.manager.release(key)

    def test_budget_one_forces_whole_expert_eviction(self):
        cache = self._cache(1)
        self._load(cache, self.keys[0])
        self._load(cache, self.keys[1])
        self.assertFalse(cache.contains(self.keys[0]))
        self.assertTrue(cache.contains(self.keys[1]))
        self.assertEqual(cache.stats()["evictions"], 1)
        self.assertEqual(cache.stats()["used_bytes"], 10)

    def test_budget_two_lru_alternating_and_repeated_hits(self):
        cache = self._cache(2)
        self._load(cache, self.keys[0])
        self._load(cache, self.keys[1])
        self.assertIsNotNone(cache.lookup(self.keys[0]))
        self._load(cache, self.keys[2])
        self.assertTrue(cache.contains(self.keys[0]))
        self.assertFalse(cache.contains(self.keys[1]))
        self.assertIsNone(cache.lookup(self.keys[3]))
        for _ in range(4):
            self.assertIsNotNone(cache.lookup(self.keys[0]))
        stats = cache.stats()
        self.assertEqual(stats["hits"], 5)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 5 / 6)

    def test_pinned_inflight_and_in_use_entries_cannot_be_evicted(self):
        cache = self._cache(2)
        self._load(cache, self.keys[0])
        cache.reserve(self.keys[1], 10)
        self.manager.mark_inflight(self.keys[1])
        self.manager.pin(self.keys[0])
        with self.assertRaisesRegex(MemoryError, "no evictable"):
            cache.reserve(self.keys[2], 10)
        self.manager.unpin(self.keys[0])
        self.manager.rollback_inflight(self.keys[1])
        cache.rollback_reservation(self.keys[1])

    def test_hot_frequency_policy_retains_frequent_entry(self):
        cache = self._cache(2, FrequencyAwareExpertPolicy(frequency_weight=10.0))
        self._load(cache, self.keys[0])
        self._load(cache, self.keys[1])
        for _ in range(10):
            cache.lookup(self.keys[0])
        self._load(cache, self.keys[2])
        self.assertTrue(cache.contains(self.keys[0]))
        self.assertFalse(cache.contains(self.keys[1]))

    def test_budget_n_holds_all_and_oversize_fails(self):
        cache = self._cache(6, LRUExpertPolicy())
        for key in self.keys:
            self._load(cache, key)
        self.assertEqual(cache.stats()["entries"], 6)
        empty = self._cache(1)
        with self.assertRaisesRegex(MemoryError, "one Expert"):
            empty.reserve(self.keys[0], 11)


@dataclass
class _Estimate:
    resident_parameters_bytes: int = 100
    estimated_gpu_peak_bytes: int = 1000
    gpu_weight_budget_bytes: int = 180

    def as_dict(self):
        return dict(self.__dict__)


class _DensePlanner:
    def estimate(self):
        return _Estimate()


class MoEMemoryPlannerTest(unittest.TestCase):
    def test_cache_is_added_to_existing_peak_and_weight_budget(self):
        result = MoEMemoryPlanner(_DensePlanner(), 300, 150, 20).estimate()
        self.assertEqual(result.estimated_gpu_peak_bytes, 1170)
        self.assertEqual(result.gpu_weight_budget_bytes, 330)
        self.assertEqual(result.fixed_resident_weight_bytes, 100)

    def test_expert_cache_cannot_escape_unified_budget(self):
        with self.assertRaisesRegex(MemoryError, "gpu-resident-weight-budget"):
            MoEMemoryPlanner(_DensePlanner(), 200, 101).estimate()


if __name__ == "__main__":
    unittest.main()
