"""KVDrive-style logical location table and deterministic mock tiers."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
from pathlib import Path
import tempfile
import threading
import time

from ..errors import KVCapacityError, KVLifecycleError


class KVTier(str, Enum):
    GPU = "gpu"
    CPU = "cpu"
    SSD = "ssd"


class ResidencyState(str, Enum):
    ABSENT = "absent"
    LOADING = "loading"
    RESIDENT = "resident"
    EVICTING = "evicting"
    FAILED = "failed"


@dataclass(frozen=True)
class KVLocation:
    tier: KVTier
    device: str
    slot: str
    length: int
    layout: str
    dtype: str
    quant_format: str
    version: int
    state: ResidencyState
    checksum: str

    def as_dict(self):
        return {
            "tier": self.tier.value,
            "device": self.device,
            "slot": self.slot,
            "length": self.length,
            "layout": self.layout,
            "dtype": self.dtype,
            "quant_format": self.quant_format,
            "version": self.version,
            "state": self.state.value,
            "checksum": self.checksum,
        }


@dataclass
class KVLocationRecord:
    logical_block_id: str
    data_version: int
    locations: dict = field(default_factory=dict)
    authoritative_tier: object = None
    pin_count: int = 0
    inflight_io: int = 0
    reservations: set = field(default_factory=set)
    last_access_epoch: int = 0
    error: object = None

    def authoritative_location(self):
        if self.authoritative_tier is None:
            raise KVLifecycleError(
                "logical block {} has no authoritative location".format(
                    self.logical_block_id
                )
            )
        location = self.locations.get(self.authoritative_tier)
        if location is None or location.state != ResidencyState.RESIDENT:
            raise KVLifecycleError(
                "logical block {} authoritative location is not resident".format(
                    self.logical_block_id
                )
            )
        return location

    def as_dict(self):
        return {
            "logical_block_id": self.logical_block_id,
            "data_version": self.data_version,
            "locations": {
                tier.value: location.as_dict()
                for tier, location in sorted(
                    self.locations.items(), key=lambda item: item[0].value
                )
            },
            "authoritative_tier": (
                None
                if self.authoritative_tier is None
                else self.authoritative_tier.value
            ),
            "pin_count": self.pin_count,
            "inflight_io": self.inflight_io,
            "reservations": sorted(item.value for item in self.reservations),
            "last_access_epoch": self.last_access_epoch,
            "error": self.error,
        }


class PrefetchCancelled(KVLifecycleError):
    pass


class MockTierBackend:
    """Copies bytes into real buffers or per-slot files for the SSD tier."""

    def __init__(self, tier, root=None, io_delay=0.0):
        self.tier = KVTier(tier)
        self.io_delay = float(io_delay)
        self._buffers = {}
        self._temporary = None
        if self.tier == KVTier.SSD:
            if root is None:
                self._temporary = tempfile.TemporaryDirectory(
                    prefix="cascade-kv-mock-ssd-"
                )
                root = self._temporary.name
            self.root = Path(root)
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            self.root = None
        self.read_count = 0
        self.write_count = 0
        self.delete_count = 0
        self.fail_next_read = False
        self.fail_next_write = False
        self._lock = threading.RLock()

    def _path(self, slot):
        digest = hashlib.sha256(str(slot).encode("utf-8")).hexdigest()
        return self.root / (digest + ".kv")

    def write(self, slot, payload):
        payload = bytes(payload)
        if self.io_delay:
            time.sleep(self.io_delay)
        with self._lock:
            if self.fail_next_write:
                self.fail_next_write = False
                raise IOError("injected {} write failure".format(self.tier.value))
            if self.tier == KVTier.SSD:
                self._path(slot).write_bytes(payload)
            else:
                self._buffers[str(slot)] = bytes(bytearray(payload))
            self.write_count += 1

    def read(self, slot):
        if self.io_delay:
            time.sleep(self.io_delay)
        with self._lock:
            if self.fail_next_read:
                self.fail_next_read = False
                raise IOError("injected {} read failure".format(self.tier.value))
            if self.tier == KVTier.SSD:
                result = self._path(slot).read_bytes()
            else:
                result = self._buffers[str(slot)]
            self.read_count += 1
            return bytes(bytearray(result))

    def delete(self, slot):
        with self._lock:
            if self.tier == KVTier.SSD:
                path = self._path(slot)
                if path.exists():
                    path.unlink()
            else:
                self._buffers.pop(str(slot), None)
            self.delete_count += 1

    def contains(self, slot):
        with self._lock:
            if self.tier == KVTier.SSD:
                return self._path(slot).exists()
            return str(slot) in self._buffers

    def close(self):
        with self._lock:
            self._buffers.clear()
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None


def _digest(payload):
    return hashlib.sha256(bytes(payload)).hexdigest()


def _logical_key(logical_block_id):
    if hasattr(logical_block_id, "key"):
        return str(logical_block_id.key())
    return str(logical_block_id)


class TieredKVStore:
    """One metadata authority over mock GPU/CPU/SSD data copies."""

    def __init__(self, capacities=None, io_delay=0.0, ssd_root=None):
        self.backends = {
            KVTier.GPU: MockTierBackend(KVTier.GPU, io_delay=io_delay),
            KVTier.CPU: MockTierBackend(KVTier.CPU, io_delay=io_delay),
            KVTier.SSD: MockTierBackend(
                KVTier.SSD, root=ssd_root, io_delay=io_delay
            ),
        }
        capacities = capacities or {}
        self.capacities = {
            KVTier(tier): int(value) for tier, value in capacities.items()
        }
        self.location_table = {}
        self._epoch = 0
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="cascade-kv-prefetch"
        )
        self._prefetches = {}
        self._cancelled = set()
        self._closed = False
        self.migration_count = 0
        self.prefetch_requests = 0
        self.prefetch_deduplicated = 0
        self.prefetch_cancelled = 0
        self.eviction_count = 0
        self.failure_count = 0

    def _check_open(self):
        if self._closed:
            raise KVLifecycleError("tiered KV store is closed")

    def _touch(self, record):
        self._epoch += 1
        record.last_access_epoch = self._epoch

    def _tier_count(self, tier):
        return sum(
            1
            for record in self.location_table.values()
            if tier in record.locations
            and record.locations[tier].state == ResidencyState.RESIDENT
        )

    def _reserve_capacity(self, tier, exclude=None):
        limit = self.capacities.get(tier)
        if limit is None:
            return
        if self._tier_count(tier) < limit:
            return
        try:
            self.evict_lru(tier, exclude=exclude)
        except KVLifecycleError:
            raise KVCapacityError(
                "{} tier capacity {} has no evictable location".format(
                    tier.value, limit
                )
            )

    def put(
        self,
        logical_block_id,
        payload,
        tier=KVTier.GPU,
        version=1,
        layout="hnd",
        dtype="bf16",
        quant_format="none",
    ):
        self._check_open()
        key = _logical_key(logical_block_id)
        tier = KVTier(tier)
        payload = bytes(payload)
        with self._lock:
            existing = self.location_table.get(key)
            if existing is not None and (
                existing.pin_count or existing.inflight_io
            ):
                raise KVLifecycleError(
                    "cannot replace pinned/inflight logical block {}".format(key)
                )
            self._reserve_capacity(tier, exclude=key)
            slot = "{}-v{}-put{}".format(
                key, int(version), self._epoch + 1
            )
        self.backends[tier].write(slot, payload)
        location = KVLocation(
            tier=tier,
            device=("mock:0" if tier != KVTier.SSD else "mock-file"),
            slot=slot,
            length=len(payload),
            layout=str(layout),
            dtype=str(dtype),
            quant_format=str(quant_format),
            version=int(version),
            state=ResidencyState.RESIDENT,
            checksum=_digest(payload),
        )
        with self._lock:
            if existing is not None:
                for old_tier, old_location in existing.locations.items():
                    self.backends[old_tier].delete(old_location.slot)
            record = KVLocationRecord(
                logical_block_id=key,
                data_version=int(version),
                locations={tier: location},
                authoritative_tier=tier,
            )
            self._touch(record)
            self.location_table[key] = record
            self.validate_record(key)
            return location

    def record(self, logical_block_id):
        key = _logical_key(logical_block_id)
        try:
            return self.location_table[key]
        except KeyError:
            raise KeyError("unknown logical KV block {!r}".format(key))

    def is_resident(self, logical_block_id, tier):
        with self._lock:
            location = self.record(logical_block_id).locations.get(KVTier(tier))
            return location is not None and location.state == ResidencyState.RESIDENT

    def get(self, logical_block_id, tier=None):
        self._check_open()
        with self._lock:
            record = self.record(logical_block_id)
            location = (
                record.authoritative_location()
                if tier is None
                else record.locations.get(KVTier(tier))
            )
            if location is None or location.state != ResidencyState.RESIDENT:
                raise KVLifecycleError("requested KV location is not resident")
            if location.version != record.data_version:
                raise KVLifecycleError("requested KV location has a stale version")
            self._touch(record)
        payload = self.backends[location.tier].read(location.slot)
        if len(payload) != location.length or _digest(payload) != location.checksum:
            raise IOError("KV checksum validation failed")
        return payload

    def migrate(
        self,
        logical_block_id,
        target_tier,
        keep_source=True,
        failure_stage=None,
        cancellation_key=None,
    ):
        self._check_open()
        key = _logical_key(logical_block_id)
        target_tier = KVTier(target_tier)
        reservation = (key, target_tier)
        source_location = None
        with self._lock:
            record = self.record(key)
            existing = record.locations.get(target_tier)
            if existing is not None and existing.state == ResidencyState.RESIDENT:
                self._touch(record)
                return existing
            self._reserve_capacity(target_tier, exclude=key)
            source_location = record.authoritative_location()
            record.pin_count += 1
            record.inflight_io += 1
            record.reservations.add(target_tier)
            target_slot = "{}-v{}-{}".format(
                key, record.data_version, target_tier.value
            )
            record.locations[target_tier] = KVLocation(
                tier=target_tier,
                device=(
                    "mock:0" if target_tier != KVTier.SSD else "mock-file"
                ),
                slot=target_slot,
                length=source_location.length,
                layout=source_location.layout,
                dtype=source_location.dtype,
                quant_format=source_location.quant_format,
                version=record.data_version,
                state=ResidencyState.LOADING,
                checksum=source_location.checksum,
            )
            source_version = record.data_version
        try:
            if failure_stage == "submit":
                raise IOError("injected migration submit failure")
            payload = self.backends[source_location.tier].read(
                source_location.slot
            )
            if cancellation_key is not None:
                with self._lock:
                    if cancellation_key in self._cancelled:
                        raise PrefetchCancelled(
                            "prefetch cancelled before target write"
                        )
            if failure_stage == "copy":
                raise IOError("injected migration copy failure")
            self.backends[target_tier].write(target_slot, payload)
            if failure_stage == "completion":
                raise IOError("injected migration completion failure")
            if _digest(payload) != source_location.checksum:
                raise IOError("migration checksum changed during copy")
            with self._lock:
                record = self.record(key)
                if (
                    record.data_version != source_version
                    or record.authoritative_tier != source_location.tier
                ):
                    raise KVLifecycleError(
                        "authoritative KV version changed during migration"
                    )
                resident = replace(
                    record.locations[target_tier],
                    state=ResidencyState.RESIDENT,
                )
                record.locations[target_tier] = resident
                record.authoritative_tier = target_tier
                record.error = None
                self.migration_count += 1
                self._touch(record)
            if not keep_source and source_location.tier != target_tier:
                # This is the migration transaction's own protected source,
                # so remove it before returning rather than calling the public
                # eviction path (which correctly rejects inflight records).
                self.backends[source_location.tier].delete(
                    source_location.slot
                )
                with self._lock:
                    record = self.record(key)
                    record.locations.pop(source_location.tier, None)
                    self.eviction_count += 1
            with self._lock:
                self.validate_record(key)
                return self.record(key).locations[target_tier]
        except BaseException as exc:
            self.failure_count += 1
            cleanup_error = None
            try:
                self.backends[target_tier].delete(target_slot)
            except BaseException as cleanup_exc:
                cleanup_error = "{}: {}".format(
                    type(cleanup_exc).__name__, cleanup_exc
                )
            with self._lock:
                record = self.record(key)
                failed = record.locations.get(target_tier)
                if failed is not None and failed.state == ResidencyState.LOADING:
                    record.locations[target_tier] = replace(
                        failed, state=ResidencyState.FAILED
                    )
                record.authoritative_tier = source_location.tier
                record.error = "{}: {}{}".format(
                    type(exc).__name__,
                    exc,
                    (
                        " (target cleanup also failed: {})".format(cleanup_error)
                        if cleanup_error is not None
                        else ""
                    ),
                )
                self.validate_record(key)
            raise
        finally:
            with self._lock:
                record = self.record(key)
                record.pin_count -= 1
                record.inflight_io -= 1
                record.reservations.discard(target_tier)
                if record.pin_count < 0 or record.inflight_io < 0:
                    raise KVLifecycleError(
                        "migration accounting underflow for {}".format(key)
                    )

    def _prefetch_worker(self, key, target_tier, prefetch_key):
        return self.migrate(
            key,
            target_tier,
            keep_source=True,
            cancellation_key=prefetch_key,
        )

    def prefetch(self, logical_block_id, target_tier=KVTier.GPU):
        self._check_open()
        key = _logical_key(logical_block_id)
        target_tier = KVTier(target_tier)
        prefetch_key = (key, target_tier)
        with self._lock:
            self.prefetch_requests += 1
            if self.is_resident(key, target_tier):
                completed = self._executor.submit(
                    lambda: self.record(key).locations[target_tier]
                )
                return completed
            existing = self._prefetches.get(prefetch_key)
            if existing is not None and not existing.done():
                self.prefetch_deduplicated += 1
                return existing
            self._cancelled.discard(prefetch_key)
            future = self._executor.submit(
                self._prefetch_worker, key, target_tier, prefetch_key
            )
            self._prefetches[prefetch_key] = future

            def cleanup(completed):
                del completed
                with self._lock:
                    self._prefetches.pop(prefetch_key, None)
                    self._cancelled.discard(prefetch_key)

            future.add_done_callback(cleanup)
            return future

    def cancel_prefetch(self, logical_block_id, target_tier=KVTier.GPU):
        key = _logical_key(logical_block_id)
        target_tier = KVTier(target_tier)
        prefetch_key = (key, target_tier)
        with self._lock:
            future = self._prefetches.get(prefetch_key)
            if future is None:
                return False
            self._cancelled.add(prefetch_key)
            cancelled = future.cancel()
            self.prefetch_cancelled += 1
            if cancelled:
                self._prefetches.pop(prefetch_key, None)
                self._cancelled.discard(prefetch_key)
            return True

    def evict(self, logical_block_id, tier):
        self._check_open()
        key = _logical_key(logical_block_id)
        tier = KVTier(tier)
        with self._lock:
            record = self.record(key)
            if record.pin_count or record.inflight_io or tier in record.reservations:
                raise KVLifecycleError(
                    "cannot evict pinned/inflight block {}".format(key)
                )
            location = record.locations.get(tier)
            if location is None or location.state != ResidencyState.RESIDENT:
                raise KVLifecycleError("KV tier is not resident")
            other = [
                item
                for item in record.locations.values()
                if item.tier != tier
                and item.state == ResidencyState.RESIDENT
                and item.version == record.data_version
            ]
            if record.authoritative_tier == tier and not other:
                raise KVLifecycleError(
                    "cannot evict the only authoritative KV copy"
                )
            record.locations[tier] = replace(
                location, state=ResidencyState.EVICTING
            )
            if record.authoritative_tier == tier:
                record.authoritative_tier = other[0].tier
        self.backends[tier].delete(location.slot)
        with self._lock:
            record.locations.pop(tier, None)
            self.eviction_count += 1
            self.validate_record(key)

    def evict_lru(self, tier, exclude=None):
        tier = KVTier(tier)
        exclude = None if exclude is None else _logical_key(exclude)
        with self._lock:
            candidates = []
            for key, record in self.location_table.items():
                if key == exclude or record.pin_count or record.inflight_io:
                    continue
                location = record.locations.get(tier)
                if location is None or location.state != ResidencyState.RESIDENT:
                    continue
                other = [
                    item
                    for item in record.locations.values()
                    if item.tier != tier
                    and item.state == ResidencyState.RESIDENT
                    and item.version == record.data_version
                ]
                if record.authoritative_tier == tier and not other:
                    continue
                candidates.append((record.last_access_epoch, key))
            if not candidates:
                raise KVLifecycleError(
                    "no evictable {} KV location".format(tier.value)
                )
            _, key = min(candidates)
        self.evict(key, tier)
        return key

    def pin(self, logical_block_id):
        with self._lock:
            record = self.record(logical_block_id)
            record.authoritative_location()
            record.pin_count += 1
            self._touch(record)

    def unpin(self, logical_block_id):
        with self._lock:
            record = self.record(logical_block_id)
            if record.pin_count <= 0:
                raise KVLifecycleError("tiered KV pin underflow")
            record.pin_count -= 1
            self._touch(record)

    @contextmanager
    def pinned(self, logical_block_ids):
        pinned = []
        try:
            for logical_block_id in logical_block_ids:
                self.pin(logical_block_id)
                pinned.append(logical_block_id)
            yield
        finally:
            for logical_block_id in reversed(pinned):
                self.unpin(logical_block_id)

    def validate_record(self, logical_block_id):
        record = self.record(logical_block_id)
        if record.pin_count < 0 or record.inflight_io < 0:
            raise KVLifecycleError("tiered KV accounting is negative")
        authority = record.authoritative_location()
        if authority.version != record.data_version:
            raise KVLifecycleError("authoritative KV version is stale")
        authoritative_count = sum(
            1
            for location in record.locations.values()
            if location.tier == record.authoritative_tier
            and location.state == ResidencyState.RESIDENT
            and location.version == record.data_version
        )
        if authoritative_count != 1:
            raise KVLifecycleError(
                "logical block {} does not have one authority".format(
                    record.logical_block_id
                )
            )
        for location in record.locations.values():
            if (
                location.state == ResidencyState.RESIDENT
                and location.version != record.data_version
            ):
                raise KVLifecycleError("resident replica version is stale")
        return True

    def stats(self):
        with self._lock:
            return {
                "blocks": len(self.location_table),
                "resident": {
                    tier.value: self._tier_count(tier) for tier in KVTier
                },
                "pins": sum(item.pin_count for item in self.location_table.values()),
                "inflight_io": sum(
                    item.inflight_io for item in self.location_table.values()
                ),
                "reservations": sum(
                    len(item.reservations)
                    for item in self.location_table.values()
                ),
                "pending_prefetches": sum(
                    not item.done() for item in self._prefetches.values()
                ),
                "migrations": self.migration_count,
                "prefetch_requests": self.prefetch_requests,
                "prefetch_deduplicated": self.prefetch_deduplicated,
                "prefetch_cancelled": self.prefetch_cancelled,
                "evictions": self.eviction_count,
                "failures": self.failure_count,
            }

    def close(self):
        if self._closed:
            return
        with self._lock:
            pending = tuple(self._prefetches.values())
        for future in pending:
            future.cancel()
        self._executor.shutdown(wait=True)
        for backend in self.backends.values():
            backend.close()
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
