"""Resource snapshots and trend checks for long-running inference tests."""

from dataclasses import asdict, dataclass, field
import json
import statistics
import threading
import time
from typing import Optional, Tuple

import torch


STABILITY_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StabilityThresholds:
    cuda_allocated_drift_bytes: int = 8 * 1024 * 1024
    cuda_reserved_drift_bytes: int = 64 * 1024 * 1024
    pinned_drift_bytes: int = 0
    thread_count_drift: int = 0
    event_count_drift: int = 0
    kv_block_drift: int = 0
    latency_tail_ratio: float = 1.25

    def __post_init__(self):
        for name in (
            "cuda_allocated_drift_bytes",
            "cuda_reserved_drift_bytes",
            "pinned_drift_bytes",
            "thread_count_drift",
            "event_count_drift",
            "kv_block_drift",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError("{} must be non-negative".format(name))
        if float(self.latency_tail_ratio) < 1.0:
            raise ValueError("latency_tail_ratio must be at least 1.0")


@dataclass(frozen=True)
class ResourceSnapshot:
    sample_index: int
    token_index: int
    monotonic_seconds: float
    token_latency_ms: Optional[float]
    cuda_allocated_bytes: int
    cuda_reserved_bytes: int
    pinned_bytes: int
    pageable_bytes: int
    thread_count: int
    cascade_thread_count: int
    event_count: int
    kv_total_blocks: int
    kv_allocated_blocks: int
    kv_active_handles: int
    queue_depths: dict = field(default_factory=dict)
    queue_capacities: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class StabilityReport:
    schema_version: int
    passed: bool
    sample_count: int
    thresholds: StabilityThresholds
    drift: dict
    latency: dict
    failures: Tuple[str, ...]
    snapshots: Tuple[ResourceSnapshot, ...]

    def as_dict(self):
        result = asdict(self)
        result["failures"] = list(self.failures)
        result["snapshots"] = [item.as_dict() for item in self.snapshots]
        return result

    def to_json(self, indent=2):
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version = int(value.pop("schema_version"))
        if version != STABILITY_REPORT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported stability report schema version {}".format(
                    version
                )
            )
        return cls(
            schema_version=version,
            passed=bool(value["passed"]),
            sample_count=int(value["sample_count"]),
            thresholds=StabilityThresholds(**value["thresholds"]),
            drift=dict(value["drift"]),
            latency=dict(value["latency"]),
            failures=tuple(value["failures"]),
            snapshots=tuple(
                ResourceSnapshot.from_dict(item)
                for item in value["snapshots"]
            ),
        )


def _cascade_thread_count():
    return sum(
        thread.is_alive()
        and (
            thread.name.startswith("cascade-")
            or thread.name.startswith("mixed-weight-stage")
        )
        for thread in threading.enumerate()
    )


def capture_resource_snapshot(
    sample_index,
    token_index,
    *,
    runtime=None,
    store=None,
    kv_manager=None,
    vocab_runtime=None,
    token_latency_ms=None,
):
    runtime_stats = runtime.resource_stats() if runtime is not None else {}
    store_stats = store.resource_stats() if store is not None else {}
    kv_stats = (
        kv_manager.resource_stats() if kv_manager is not None else {}
    )
    vocab_stats = (
        vocab_runtime.resource_stats()
        if vocab_runtime is not None
        else {}
    )
    pipeline = runtime_stats.get("pipeline", {})
    queue_depths = {
        name: int(pipeline.get(name + "_depth", 0))
        for name in (
            "source_queue",
            "ready_queue",
            "free_slot_queue",
            "staging_queue",
            "producer_request_queue",
        )
    }
    queue_capacities = {
        name: int(pipeline.get(name + "_capacity", 0))
        for name in (
            "source_queue",
            "ready_queue",
            "free_slot_queue",
            "staging_queue",
        )
    }
    if runtime is None and torch.cuda.is_available():
        cuda_allocated = torch.cuda.memory_allocated()
        cuda_reserved = torch.cuda.memory_reserved()
    else:
        cuda_allocated = runtime_stats.get("cuda_allocated_bytes", 0)
        cuda_reserved = runtime_stats.get("cuda_reserved_bytes", 0)
    return ResourceSnapshot(
        sample_index=int(sample_index),
        token_index=int(token_index),
        monotonic_seconds=time.monotonic(),
        token_latency_ms=(
            None if token_latency_ms is None else float(token_latency_ms)
        ),
        cuda_allocated_bytes=int(cuda_allocated),
        cuda_reserved_bytes=int(cuda_reserved),
        pinned_bytes=int(
            store_stats.get("pinned_bytes", 0)
            + vocab_stats.get("pinned_bytes", 0)
        ),
        pageable_bytes=int(store_stats.get("pageable_bytes", 0)),
        thread_count=threading.active_count(),
        cascade_thread_count=_cascade_thread_count(),
        event_count=int(
            runtime_stats.get("event_count", 0)
            + vocab_stats.get("event_count", 0)
        ),
        kv_total_blocks=int(kv_stats.get("total_blocks", 0)),
        kv_allocated_blocks=int(kv_stats.get("allocated_blocks", 0)),
        kv_active_handles=int(kv_stats.get("active_handles", 0)),
        queue_depths=queue_depths,
        queue_capacities=queue_capacities,
    )


def _median_window(values, from_tail, window):
    selected = values[-window:] if from_tail else values[:window]
    return float(statistics.median(selected))


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    position = (len(ordered) - 1) * float(percentile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def analyze_stability(snapshots, thresholds=None, window=None):
    snapshots = tuple(snapshots)
    if len(snapshots) < 2:
        raise ValueError("at least two resource snapshots are required")
    thresholds = thresholds or StabilityThresholds()
    window = int(window or max(1, min(20, len(snapshots) // 5)))
    window = min(window, len(snapshots) // 2)
    fields = {
        "cuda_allocated_bytes": thresholds.cuda_allocated_drift_bytes,
        "cuda_reserved_bytes": thresholds.cuda_reserved_drift_bytes,
        "pinned_bytes": thresholds.pinned_drift_bytes,
        "cascade_thread_count": thresholds.thread_count_drift,
        "event_count": thresholds.event_count_drift,
        "kv_allocated_blocks": thresholds.kv_block_drift,
    }
    drift = {}
    failures = []
    for name, limit in fields.items():
        values = [getattr(item, name) for item in snapshots]
        head = _median_window(values, False, window)
        tail = _median_window(values, True, window)
        increase = max(0.0, tail - head)
        drift[name] = {
            "head_median": head,
            "tail_median": tail,
            "increase": increase,
            "limit": int(limit),
        }
        if increase > limit:
            failures.append(
                "{} increased by {:.0f}, limit {}".format(
                    name, increase, limit
                )
            )
    latencies = [
        item.token_latency_ms
        for item in snapshots
        if item.token_latency_ms is not None
    ]
    latency = {
        "count": len(latencies),
        "p50_ms": _percentile(latencies, 0.50),
        "p95_ms": _percentile(latencies, 0.95),
        "head_median_ms": 0.0,
        "tail_median_ms": 0.0,
        "tail_ratio": 1.0,
    }
    if len(latencies) >= 2:
        latency_window = min(window, len(latencies) // 2)
        head = _median_window(latencies, False, latency_window)
        tail = _median_window(latencies, True, latency_window)
        ratio = tail / head if head > 0 else 1.0
        latency.update(
            {
                "head_median_ms": head,
                "tail_median_ms": tail,
                "tail_ratio": ratio,
            }
        )
        if ratio > thresholds.latency_tail_ratio:
            failures.append(
                "token latency tail ratio {:.3f} exceeds {:.3f}".format(
                    ratio, thresholds.latency_tail_ratio
                )
            )
    return StabilityReport(
        schema_version=STABILITY_REPORT_SCHEMA_VERSION,
        passed=not failures,
        sample_count=len(snapshots),
        thresholds=thresholds,
        drift=drift,
        latency=latency,
        failures=tuple(failures),
        snapshots=snapshots,
    )
