"""Unified resource snapshots and drift classification for KV qualification.

This module deliberately separates resources that must return to their
baseline from CUDA allocator reservations.  A larger ``cuda_reserved`` value
with stable ``cuda_allocated`` is allocator cache, not by itself a leak.
Callers may capture once more after :func:`empty_cuda_cache_and_capture` when
they need the stricter post-cache qualification gate.
"""

from dataclasses import asdict, dataclass, field
import os
import threading
import time

import torch


PASS = "PASS"
FAIL = "FAIL"
SKIPPED_WITH_REASON = "SKIPPED_WITH_REASON"
BLOCKED = "BLOCKED"
RESOURCE_AUDIT_SCHEMA_VERSION = 2


def _process_rss_bytes():
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except BaseException:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as source:
                resident_pages = int(source.read().split()[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
        except BaseException:
            return None


def _prefix_stats(runtime):
    cache = getattr(runtime, "prefix_cache", None)
    stats = cache.stats() if cache is not None and hasattr(cache, "stats") else {}
    return (
        int(stats.get("prefix_entries", 0)),
        int(stats.get("prefix_pages", 0)),
        int(stats.get("prefix_bytes", 0)),
    )


def _rgkv_stats(runtime):
    selection = getattr(runtime, "selection", None)
    index = getattr(selection, "index", None)
    stats = index.stats() if index is not None and hasattr(index, "stats") else {}
    return int(stats.get("records", 0)), int(stats.get("compact_bytes", 0))


@dataclass(frozen=True)
class KVResourceSnapshot:
    """One point-in-time view of ownership, async, cache, and system state."""

    sample_index: int
    token_index: int
    monotonic_seconds: float
    page_total: int
    page_free: int
    page_allocated: int
    page_peak_allocated: int
    ref_total: int
    owner_total: int
    pin_total: int
    inflight_compute: int
    inflight_io: int
    pending_fences: int
    pending_append: int
    prefix_entries: int
    prefix_pages: int
    prefix_bytes: int
    rgkv_records: int
    rgkv_index_bytes: int
    cuda_allocated: int
    cuda_reserved: int
    cpu_rss: object
    pinned_bytes: int
    thread_count: int
    future_count: int
    metadata: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)

    @property
    def quest_records(self):
        """Deprecated compatibility spelling; new evidence uses RGKV."""

        return self.rgkv_records

    @property
    def quest_bytes(self):
        """Deprecated compatibility spelling; new evidence uses RGKV."""

        return self.rgkv_index_bytes

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        # Schema-v1 evidence used the experimental Quest name.  Accept it at
        # the deserialization boundary without emitting it in new reports.
        if "rgkv_records" not in value and "quest_records" in value:
            value["rgkv_records"] = value.pop("quest_records")
        if "rgkv_index_bytes" not in value and "quest_bytes" in value:
            value["rgkv_index_bytes"] = value.pop("quest_bytes")
        return cls(**value)

    @classmethod
    def capture(
        cls,
        runtime,
        sample_index=0,
        token_index=0,
        *,
        tier_store=None,
        pinned_bytes=None,
        metadata=None,
    ):
        pinned_bytes_explicit = pinned_bytes is not None
        pool = runtime.page_pool.profile()
        profile = runtime.profile_stats()
        requests = tuple(runtime.request_table.values())
        tier = tier_store.stats() if tier_store is not None else {}
        prefix_entries, prefix_pages, prefix_bytes = _prefix_stats(runtime)
        rgkv_records, rgkv_index_bytes = _rgkv_stats(runtime)
        device = torch.device(runtime.device)
        if device.type == "cuda":
            cuda_allocated = int(torch.cuda.memory_allocated(device))
            cuda_reserved = int(torch.cuda.memory_reserved(device))
        else:
            cuda_allocated = cuda_reserved = 0
        pending_append_fences = int(profile.get("pending_append_fences", 0))
        pending_attention = int(profile.get("pending_attention_fences", 0))
        pending_tier = int(tier.get("pending_tier_operations", 0))
        pending_prefetch = int(
            tier.get("pending_prefetches", pending_tier)
        )
        if pinned_bytes_explicit:
            pinned_bytes_source = "explicit"
        elif (
            "pinned_bytes" in tier
            or "cpu_kv_used_bytes" in tier
            or "pinned_bytes" in profile
        ):
            pinned_bytes_source = "runtime_or_tier_stats"
        else:
            pinned_bytes_source = "unavailable_reported_zero"
        if pinned_bytes is None:
            pinned_bytes = int(
                tier.get(
                    "pinned_bytes",
                    tier.get(
                        "cpu_kv_used_bytes", profile.get("pinned_bytes", 0)
                    ),
                )
            )
        snapshot_metadata = {
            "pinned_bytes_source": pinned_bytes_source,
            "future_count_source": "tier_pending_prefetches",
        }
        snapshot_metadata.update(metadata or {})
        return cls(
            sample_index=int(sample_index),
            token_index=int(token_index),
            monotonic_seconds=time.monotonic(),
            page_total=int(pool["total_pages"]),
            page_free=int(pool["free_pages"]),
            page_allocated=int(pool["allocated_pages"]),
            page_peak_allocated=int(pool["peak_allocated_pages"]),
            ref_total=int(pool["total_ref_count"]),
            owner_total=int(pool["logical_owner_count"]),
            pin_total=int(pool["total_pin_count"]),
            inflight_compute=int(pool["inflight_compute"]),
            inflight_io=int(pool["inflight_io"]),
            pending_fences=pending_append_fences + pending_attention + pending_tier,
            pending_append=sum(state.pending_append is not None for state in requests),
            prefix_entries=prefix_entries,
            prefix_pages=prefix_pages,
            prefix_bytes=prefix_bytes,
            rgkv_records=rgkv_records,
            rgkv_index_bytes=rgkv_index_bytes,
            cuda_allocated=cuda_allocated,
            cuda_reserved=cuda_reserved,
            cpu_rss=_process_rss_bytes(),
            pinned_bytes=int(pinned_bytes),
            thread_count=int(threading.active_count()),
            future_count=pending_prefetch,
            metadata=snapshot_metadata,
        )


@dataclass(frozen=True)
class ResourceDrift:
    field: str
    before: object
    after: object
    delta: object
    classification: str
    allowed_growth: int
    reason: str

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class KVResourceComparison:
    status: str
    reason: object
    before: object
    after: object
    after_empty_cache: object
    drift: tuple
    schema_version: int = RESOURCE_AUDIT_SCHEMA_VERSION

    @property
    def leaks(self):
        return tuple(item for item in self.drift if item.classification == "true_leak")

    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "reason": self.reason,
            "before": None if self.before is None else self.before.as_dict(),
            "after": None if self.after is None else self.after.as_dict(),
            "after_empty_cache": (
                None
                if self.after_empty_cache is None
                else self.after_empty_cache.as_dict()
            ),
            "drift": [item.as_dict() for item in self.drift],
            "leak_fields": [item.field for item in self.leaks],
        }

    @classmethod
    def not_run(cls, status, reason):
        if status not in {SKIPPED_WITH_REASON, BLOCKED}:
            raise ValueError("not-run status must be SKIPPED_WITH_REASON or BLOCKED")
        if not str(reason).strip():
            raise ValueError("not-run result requires a reason")
        return cls(status, str(reason), None, None, None, ())


_STRICT_FIELDS = (
    "page_free",
    "page_allocated",
    "ref_total",
    "owner_total",
    "pin_total",
    "inflight_compute",
    "inflight_io",
    "pending_fences",
    "pending_append",
    "prefix_entries",
    "prefix_pages",
    "prefix_bytes",
    "rgkv_records",
    "rgkv_index_bytes",
    "cuda_allocated",
    "pinned_bytes",
    "thread_count",
    "future_count",
)


def compare_resource_snapshots(
    before,
    after,
    *,
    expected_growth=None,
    after_empty_cache=None,
    cuda_allocated_tolerance=8 * 1024 * 1024,
    cpu_rss_tolerance=16 * 1024 * 1024,
):
    """Classify resource growth without treating CUDA cache as a leak.

    ``expected_growth`` maps a field to the maximum allowed positive delta.
    Negative deltas are releases.  ``cuda_reserved`` is classified as
    ``allocator_cache`` while allocated bytes are within tolerance.  When a
    post-empty-cache snapshot is supplied it becomes the strict final value.
    """

    if not isinstance(before, KVResourceSnapshot) or not isinstance(
        after, KVResourceSnapshot
    ):
        raise TypeError("before and after must be KVResourceSnapshot")
    if after_empty_cache is not None and not isinstance(
        after_empty_cache, KVResourceSnapshot
    ):
        raise TypeError("after_empty_cache must be KVResourceSnapshot")
    expected = {str(key): int(value) for key, value in (expected_growth or {}).items()}
    if any(value < 0 for value in expected.values()):
        raise ValueError("expected growth allowances cannot be negative")
    final = after_empty_cache or after
    drift = []
    for name in _STRICT_FIELDS:
        initial = getattr(before, name)
        value = getattr(final, name)
        delta = value - initial
        allowance = expected.get(name, 0)
        if name == "cuda_allocated":
            allowance = max(allowance, int(cuda_allocated_tolerance))
        if delta <= 0:
            classification, reason = "released_or_stable", "returned to or below baseline"
        elif delta <= allowance:
            classification, reason = "expected_growth", "within configured growth allowance"
        else:
            classification, reason = "true_leak", "positive drift exceeds allowance"
        drift.append(ResourceDrift(name, initial, value, delta, classification, allowance, reason))
    invariants = (
        (
            "ref_owner_invariant",
            final.ref_total,
            final.owner_total,
            final.ref_total - final.owner_total,
            "ref_total must equal owner_total",
        ),
        (
            "page_capacity_invariant",
            final.page_total,
            final.page_free + final.page_allocated,
            final.page_total - final.page_free - final.page_allocated,
            "page_total must equal page_free + page_allocated",
        ),
        (
            "pin_inflight_invariant",
            final.pin_total,
            final.inflight_compute + final.inflight_io,
            final.pin_total - final.inflight_compute - final.inflight_io,
            "pin_total must equal inflight_compute + inflight_io",
        ),
    )
    for name, left, right, delta, reason in invariants:
        drift.append(
            ResourceDrift(
                name,
                left,
                right,
                delta,
                "released_or_stable" if delta == 0 else "true_leak",
                0,
                reason,
            )
        )
    if before.cpu_rss is not None and final.cpu_rss is not None:
        delta = int(final.cpu_rss) - int(before.cpu_rss)
        allowance = max(expected.get("cpu_rss", 0), int(cpu_rss_tolerance))
        classification = (
            "released_or_stable"
            if delta <= 0
            else "expected_growth"
            if delta <= allowance
            else "true_leak"
        )
        drift.append(ResourceDrift("cpu_rss", before.cpu_rss, final.cpu_rss, delta, classification, allowance, "RSS tolerance gate"))
    reserved_delta = int(final.cuda_reserved) - int(before.cuda_reserved)
    allocated_ok = int(final.cuda_allocated) - int(before.cuda_allocated) <= max(
        expected.get("cuda_allocated", 0), int(cuda_allocated_tolerance)
    )
    reserved_allowance = expected.get("cuda_reserved", 0)
    if reserved_delta <= 0:
        reserved_class, reason = "released_or_stable", "returned to or below baseline"
    elif reserved_delta <= reserved_allowance:
        reserved_class, reason = "expected_growth", "within configured growth allowance"
    elif after_empty_cache is None and allocated_ok:
        reserved_class, reason = "allocator_cache", "reserved cache grew while allocated bytes closed"
    else:
        reserved_class, reason = "true_leak", "reserved bytes remain above allowance after strict check"
    drift.append(ResourceDrift("cuda_reserved", before.cuda_reserved, final.cuda_reserved, reserved_delta, reserved_class, reserved_allowance, reason))
    leaks = tuple(item for item in drift if item.classification == "true_leak")
    return KVResourceComparison(
        PASS if not leaks else FAIL,
        None if not leaks else "; ".join("{} +{}".format(item.field, item.delta) for item in leaks),
        before,
        after,
        after_empty_cache,
        tuple(drift),
    )


def empty_cuda_cache_and_capture(runtime, *args, **kwargs):
    """Synchronize, clear allocator cache, then take a strict final snapshot."""

    device = torch.device(runtime.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return KVResourceSnapshot.capture(runtime, *args, **kwargs)
