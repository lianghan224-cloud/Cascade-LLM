"""Whole-Expert cache metadata governed by a shared resident-weight budget."""

from dataclasses import dataclass
import math
from typing import Dict

from .residency import ExpertKey, ExpertResidencyManager


@dataclass(frozen=True)
class UnifiedResidentWeightBudget:
    total_bytes: int
    fixed_resident_bytes: int

    def __post_init__(self):
        if int(self.total_bytes) < 0 or int(self.fixed_resident_bytes) < 0:
            raise ValueError("resident weight budget fields must be nonnegative")
        if self.fixed_resident_bytes > self.total_bytes:
            raise ValueError("fixed resident weights exceed the unified budget")

    @property
    def expert_capacity_bytes(self):
        return int(self.total_bytes) - int(self.fixed_resident_bytes)


@dataclass
class ExpertCacheEntry:
    key: ExpertKey
    bytes: int
    location: object = None
    ready_event: object = None
    last_used: int = 0
    access_count: int = 0
    admitted_at: int = 0
    loaded_bytes: int = 0
    residency_ticks: int = 0


class LRUExpertPolicy:
    name = "lru"

    def score(self, entry, now):
        del now
        return (entry.last_used, entry.admitted_at, entry.key)


class FrequencyAwareExpertPolicy:
    name = "lru_frequency"

    def __init__(self, frequency_weight=4.0):
        self.frequency_weight = float(frequency_weight)

    def score(self, entry, now):
        age = max(0, int(now) - int(entry.last_used))
        retention = self.frequency_weight * math.log2(entry.access_count + 1.0)
        # Lower scores are evicted first: old and cold entries lose.
        return (retention - age, entry.last_used, entry.key)


class ExpertCache:
    def __init__(self, residency, budget, policy=None):
        if not isinstance(residency, ExpertResidencyManager):
            raise TypeError("residency must be ExpertResidencyManager")
        if not isinstance(budget, UnifiedResidentWeightBudget):
            raise TypeError("cache requires the unified resident-weight budget")
        self.residency = residency
        self.budget = budget
        self.policy = policy or LRUExpertPolicy()
        self.entries: Dict[ExpertKey, ExpertCacheEntry] = {}
        self.used_bytes = 0
        self.clock = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.loaded_bytes = 0

    @staticmethod
    def _key(key):
        return key if isinstance(key, ExpertKey) else ExpertKey(*key)

    @property
    def capacity_bytes(self):
        return self.budget.expert_capacity_bytes

    def _tick(self):
        self.clock += 1
        return self.clock

    def contains(self, key):
        return self._key(key) in self.entries

    def lookup(self, key, count=True):
        key = self._key(key)
        entry = self.entries.get(key)
        now = self._tick()
        if entry is None:
            if count:
                self.misses += 1
            return None
        if count:
            self.hits += 1
            entry.access_count += 1
        entry.last_used = now
        entry.residency_ticks = now - entry.admitted_at
        return entry

    def _victims(self, exclude=frozenset()):
        candidates = [
            entry
            for key, entry in self.entries.items()
            if key not in exclude and self.residency.require(key).evictable
        ]
        return sorted(candidates, key=lambda item: self.policy.score(item, self.clock))

    def reserve(self, key, nbytes):
        """Reserve one whole-Expert entry before the runtime starts H2D."""
        key = self._key(key)
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise ValueError("Expert cache entry bytes must be positive")
        if key in self.entries:
            raise ValueError("Expert is already cached")
        if nbytes > self.capacity_bytes:
            raise MemoryError("one Expert exceeds the unified cache capacity")
        required = self.used_bytes + nbytes - self.capacity_bytes
        for victim in self._victims(exclude={key}):
            if required <= 0:
                break
            self.remove(victim.key)
            required = self.used_bytes + nbytes - self.capacity_bytes
        if required > 0:
            raise MemoryError("no evictable Expert can satisfy the cache budget")
        now = self._tick()
        entry = ExpertCacheEntry(
            key=key,
            bytes=nbytes,
            last_used=now,
            access_count=1,
            admitted_at=now,
        )
        self.entries[key] = entry
        self.used_bytes += nbytes
        return entry

    def mark_ready(self, key, location=None, ready_event=None):
        entry = self.entries[self._key(key)]
        entry.location = location
        entry.ready_event = ready_event
        entry.loaded_bytes += entry.bytes
        self.loaded_bytes += entry.bytes
        return entry

    def acquire(self, key):
        key = self._key(key)
        entry = self.entries.get(key)
        if entry is None:
            raise KeyError(key)
        entry.last_used = self._tick()
        entry.access_count += 1
        self.residency.acquire(entry.key)
        return entry

    def release(self, key):
        self.residency.release(self._key(key))

    def remove(self, key):
        key = self._key(key)
        entry = self.entries[key]
        self.residency.evict(key)
        self.used_bytes -= entry.bytes
        del self.entries[key]
        self.evictions += 1
        return entry

    def rollback_reservation(self, key):
        key = self._key(key)
        entry = self.entries.pop(key)
        self.used_bytes -= entry.bytes
        return entry

    def stats(self):
        accesses = self.hits + self.misses
        return {
            "policy": self.policy.name,
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "entries": len(self.entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": 0.0 if accesses == 0 else self.hits / accesses,
            "evictions": self.evictions,
            "loaded_bytes": self.loaded_bytes,
        }
