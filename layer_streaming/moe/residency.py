"""Allocation-neutral Expert residency lifecycle."""

from dataclasses import dataclass
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Optional, Tuple


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer_id: int
    expert_id: int

    def __post_init__(self):
        if int(self.layer_id) < 0 or int(self.expert_id) < 0:
            raise ValueError("ExpertKey fields must be nonnegative")


class ExpertResidencyState(str, Enum):
    NOT_RESIDENT = "not_resident"
    CPU_RESIDENT = "cpu_resident"
    H2D_INFLIGHT = "h2d_inflight"
    GPU_RESIDENT = "gpu_resident"
    IN_USE = "in_use"
    EVICTABLE = "evictable"


@dataclass
class ExpertResidencyRecord:
    key: ExpertKey
    state: ExpertResidencyState = ExpertResidencyState.NOT_RESIDENT
    use_count: int = 0
    pin_count: int = 0
    ready_event: object = None
    location: object = None
    rollback_state: Optional[ExpertResidencyState] = None

    @property
    def evictable(self):
        return (
            self.state in {
                ExpertResidencyState.GPU_RESIDENT,
                ExpertResidencyState.EVICTABLE,
            }
            and self.use_count == 0
            and self.pin_count == 0
        )


class ExpertResidencyManager:
    """Tracks state only; payload memory remains owned by the weight runtime."""

    def __init__(self):
        self._records: Dict[ExpertKey, ExpertResidencyRecord] = {}
        self._closed = False

    @staticmethod
    def _key(key):
        return key if isinstance(key, ExpertKey) else ExpertKey(*key)

    def register(self, key, cpu_resident=True):
        self._require_open()
        key = self._key(key)
        if key in self._records:
            raise ValueError("Expert is already registered")
        record = ExpertResidencyRecord(
            key=key,
            state=(
                ExpertResidencyState.CPU_RESIDENT
                if cpu_resident
                else ExpertResidencyState.NOT_RESIDENT
            ),
        )
        self._records[key] = record
        return record

    def lookup(self, key):
        self._require_open()
        return self._records.get(self._key(key))

    def require(self, key):
        record = self.lookup(key)
        if record is None:
            raise KeyError(self._key(key))
        return record

    def mark_inflight(self, key):
        record = self.require(key)
        if record.use_count or record.pin_count:
            raise RuntimeError("cannot migrate an acquired or pinned Expert")
        if record.state not in {
            ExpertResidencyState.NOT_RESIDENT,
            ExpertResidencyState.CPU_RESIDENT,
        }:
            raise RuntimeError("Expert cannot enter H2D from {}".format(record.state.value))
        record.rollback_state = record.state
        record.state = ExpertResidencyState.H2D_INFLIGHT
        record.ready_event = None
        return record

    def mark_ready(self, key, location=None, ready_event=None):
        record = self.require(key)
        if record.state != ExpertResidencyState.H2D_INFLIGHT:
            raise RuntimeError("only an inflight Expert can become ready")
        record.state = ExpertResidencyState.GPU_RESIDENT
        record.location = location
        record.ready_event = ready_event
        record.rollback_state = None
        return record

    def attach_inflight(self, key, location, ready_event):
        record = self.require(key)
        if record.state != ExpertResidencyState.H2D_INFLIGHT:
            raise RuntimeError("Expert is not inflight")
        record.location = location
        record.ready_event = ready_event
        return record

    def ensure_resident(self, key, submit_load):
        """Transactionally submit an async load without taking ownership."""
        record = self.require(key)
        if record.state in {
            ExpertResidencyState.GPU_RESIDENT,
            ExpertResidencyState.EVICTABLE,
            ExpertResidencyState.IN_USE,
            ExpertResidencyState.H2D_INFLIGHT,
        }:
            return record
        self.mark_inflight(key)
        try:
            result = submit_load(record)
            if isinstance(result, tuple) and len(result) == 2:
                location, ready_event = result
            else:
                location, ready_event = result, None
            return self.mark_ready(key, location, ready_event)
        except BaseException:
            self.rollback_inflight(key)
            raise

    def rollback_inflight(self, key):
        record = self.require(key)
        if record.state != ExpertResidencyState.H2D_INFLIGHT:
            raise RuntimeError("Expert is not inflight")
        record.state = record.rollback_state or ExpertResidencyState.CPU_RESIDENT
        record.rollback_state = None
        record.ready_event = None
        record.location = None
        return record

    def acquire(self, key):
        record = self.require(key)
        if record.state not in {
            ExpertResidencyState.GPU_RESIDENT,
            ExpertResidencyState.EVICTABLE,
            ExpertResidencyState.IN_USE,
        }:
            raise RuntimeError("Expert is not GPU ready")
        record.use_count += 1
        record.state = ExpertResidencyState.IN_USE
        return record

    def release(self, key):
        record = self.require(key)
        if record.use_count < 1:
            raise RuntimeError("unbalanced Expert release")
        record.use_count -= 1
        if record.use_count == 0:
            record.state = (
                ExpertResidencyState.GPU_RESIDENT
                if record.pin_count
                else ExpertResidencyState.EVICTABLE
            )
        return record

    @contextmanager
    def lease(self, key):
        record = self.acquire(key)
        try:
            yield record
        finally:
            self.release(key)

    def pin(self, key):
        record = self.require(key)
        if record.state not in {
            ExpertResidencyState.GPU_RESIDENT,
            ExpertResidencyState.EVICTABLE,
            ExpertResidencyState.IN_USE,
        }:
            raise RuntimeError("only GPU-ready Experts can be pinned")
        record.pin_count += 1
        if record.state == ExpertResidencyState.EVICTABLE:
            record.state = ExpertResidencyState.GPU_RESIDENT
        return record

    def unpin(self, key):
        record = self.require(key)
        if record.pin_count < 1:
            raise RuntimeError("unbalanced Expert unpin")
        record.pin_count -= 1
        if record.pin_count == 0 and record.use_count == 0:
            record.state = ExpertResidencyState.EVICTABLE
        return record

    def evict(self, key):
        record = self.require(key)
        if not record.evictable:
            raise RuntimeError("Expert is protected from eviction")
        record.state = ExpertResidencyState.CPU_RESIDENT
        record.location = None
        record.ready_event = None
        return record

    def quiescent(self):
        return all(
            record.use_count == 0
            and record.pin_count == 0
            and record.state != ExpertResidencyState.H2D_INFLIGHT
            for record in self._records.values()
        )

    def stats(self):
        by_state = {state.value: 0 for state in ExpertResidencyState}
        for record in self._records.values():
            by_state[record.state.value] += 1
        return {
            "closed": self._closed,
            "experts": len(self._records),
            "inflight": by_state[ExpertResidencyState.H2D_INFLIGHT.value],
            "pin_count": sum(item.pin_count for item in self._records.values()),
            "use_count": sum(item.use_count for item in self._records.values()),
            "states": by_state,
        }

    def reset(self):
        self._require_open()
        if not self.quiescent():
            raise RuntimeError("cannot reset non-quiescent Expert residency")
        for record in self._records.values():
            record.state = ExpertResidencyState.CPU_RESIDENT
            record.ready_event = None
            record.location = None

    def close(self):
        if self._closed:
            return
        if not self.quiescent():
            raise RuntimeError("cannot close non-quiescent Expert residency")
        self._records.clear()
        self._closed = True

    def _require_open(self):
        if self._closed:
            raise RuntimeError("ExpertResidencyManager is closed")
