"""KVDrive-style logical location table and deterministic mock tiers."""

from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
from pathlib import Path
import tempfile
import threading
import time

from ..errors import KVCapacityError, KVLifecycleError
from ..fence import KVOperationFence


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


class TieredKVOperationFence(KVOperationFence):
    """KVOperationFence with the minimal legacy ``Future`` adapter.

    The mock tier used to expose ``Future`` directly.  ``done`` and ``result``
    remain temporarily available so existing diagnostics keep working, while
    all new callers can use the common fence contract.
    """

    def done(self):
        return self.query()

    def result(self, timeout=None):
        future = self.io_future
        if future is None:
            self.wait(timeout_seconds=0.0 if timeout is None else timeout)
            return getattr(self, "result_value", None)
        value = future.result(timeout=timeout)
        if self.cancelled:
            raise PrefetchCancelled(
                "KV operation {} ({}) was cancelled".format(
                    self.operation_id, self.kind
                )
            )
        self.result_value = value
        self.mark_completed(self.completion_epoch)
        return value


class RequestScopedPrefetchGroup:
    """Own and quiesce the prefetch fences submitted for one request.

    Cancellation is cooperative: outstanding workers are asked to stop, then
    observed for a bounded interval.  The store keeps every reservation and
    IO pin until its worker has actually stopped, so a timeout cannot publish
    an incomplete replica or release protected metadata early.
    """

    def __init__(self, store, request_id, cleanup_timeout=5.0):
        self.store = store
        self.request_id = request_id
        self.cleanup_timeout = float(cleanup_timeout)
        self.fences = []
        self._operation_ids = set()
        self._cancelled = False
        self._released = False
        self._lock = threading.RLock()

    def add(self, fence):
        if not isinstance(fence, KVOperationFence):
            raise TypeError("prefetch group only accepts KVOperationFence")
        with self._lock:
            if self._cancelled or self._released:
                # A request can be closed after the store submission but
                # before the coordinator attaches the returned Fence.  Adopt
                # and cancel that orphan here instead of letting it escape the
                # request-scoped cleanup boundary.
                self.store.cancel_fence(
                    fence, request_id=self.request_id
                )
                raise PrefetchCancelled(
                    "prefetch group {!r} is cancelled".format(
                        self.request_id
                    )
                )
            if fence.operation_id not in self._operation_ids:
                self.fences.append(fence)
                self._operation_ids.add(fence.operation_id)
        return fence

    def cancel(self):
        with self._lock:
            if self._cancelled:
                return self
            self._cancelled = True
            fences = tuple(self.fences)
        for fence in fences:
            self.store.cancel_fence(fence, request_id=self.request_id)
        return self

    def quiesce(self, timeout=None):
        timeout = self.cleanup_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + timeout
        with self._lock:
            pending = list(self.fences)
        while pending:
            pending = [fence for fence in pending if not fence.query()]
            if not pending:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "prefetch group {!r} did not quiesce within {:.3f}s".format(
                        self.request_id, timeout
                    )
                )
            time.sleep(0.001)
        return self

    def _abort_and_quiesce(self, original_error):
        self.cancel()
        try:
            self.quiesce()
        except BaseException as cleanup_error:
            try:
                original_error.kv_cleanup_error = cleanup_error
            except BaseException:
                pass
        self.release()

    def wait(self, timeout=10.0, cancel_event=None):
        deadline = time.monotonic() + float(timeout)
        try:
            with self._lock:
                fences = tuple(self.fences)
                cancelled = self._cancelled
            if cancelled:
                raise PrefetchCancelled(
                    "request {!r} prefetch was cancelled".format(
                        self.request_id
                    )
                )
            for fence in fences:
                while not fence.query():
                    if cancel_event is not None and cancel_event.is_set():
                        raise PrefetchCancelled(
                            "request {!r} cancelled while prefetch was pending".format(
                                self.request_id
                            )
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "KV prefetch timeout for request {!r}".format(
                                self.request_id
                            )
                        )
                    time.sleep(0.001)
                if fence.cancelled:
                    raise PrefetchCancelled(
                        "request {!r} prefetch was cancelled".format(
                            self.request_id
                        )
                    )
                fence.wait(timeout_seconds=0.0)
            self.release()
            return self
        except BaseException as error:
            self._abort_and_quiesce(error)
            raise

    def release(self):
        with self._lock:
            if self._released:
                return self
            self._released = True
            fences = tuple(self.fences)
        for fence in fences:
            self.store.release_fence_consumer(
                fence, request_id=self.request_id
            )
        return self

    def cleanup(self, timeout=None):
        self.cancel()
        try:
            self.quiesce(timeout=timeout)
        finally:
            self.release()
        return self


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
        self._active_fences = {}
        self._fence_consumers = {}
        self._cancelled = set()
        self._operation_sequence = 0
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

    def _next_operation(self, kind, request_id, prefetch_key):
        self._operation_sequence += 1
        fence = TieredKVOperationFence(
            operation_id="mock-tier-{}-{}".format(
                kind, self._operation_sequence
            ),
            request_id=request_id,
            kind=kind,
            submit_epoch=self._epoch,
        )
        fence.prefetch_key = prefetch_key
        fence.cancellation_key = (prefetch_key, fence.operation_id)
        fence.result_value = None
        # Cancellation and metadata publication are serialized by the store
        # lock.  This bit closes the small window between ``migrate``
        # publishing the target authority and the worker marking its Future
        # complete: a cancellation arriving in that window is too late and
        # must not relabel the committed transaction as cancelled.
        fence.metadata_committed = False
        return fence

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
        operation_fence=None,
    ):
        self._check_open()
        key = _logical_key(logical_block_id)
        target_tier = KVTier(target_tier)
        source_location = None

        def raise_if_cancelled(stage):
            if cancellation_key is None:
                return
            with self._lock:
                if cancellation_key in self._cancelled:
                    raise PrefetchCancelled(
                        "prefetch cancelled {}".format(stage)
                    )

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
            raise_if_cancelled("before source read")
            payload = self.backends[source_location.tier].read(
                source_location.slot
            )
            raise_if_cancelled("before target write")
            if failure_stage == "copy":
                raise IOError("injected migration copy failure")
            self.backends[target_tier].write(target_slot, payload)
            raise_if_cancelled("before metadata commit")
            if failure_stage == "completion":
                raise IOError("injected migration completion failure")
            if len(payload) != source_location.length:
                raise IOError("migration length changed during copy")
            if _digest(payload) != source_location.checksum:
                raise IOError("migration checksum changed during copy")
            with self._lock:
                # This is the transaction commit point.  Re-check cooperative
                # cancellation under the same lock used by ``cancel_fence``;
                # otherwise cancel could win immediately after the earlier
                # check and the target would still become authoritative.
                if (
                    cancellation_key is not None
                    and cancellation_key in self._cancelled
                ):
                    raise PrefetchCancelled(
                        "prefetch cancelled before metadata commit"
                    )
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
                if operation_fence is not None:
                    operation_fence.metadata_committed = True
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
                if operation_fence is not None:
                    operation_fence.metadata_committed = False
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

    def _submit_migration(
        self,
        key,
        target_tier,
        *,
        request_id=None,
        kind="prefetch",
        keep_source=True,
        failure_stage=None,
    ):
        prefetch_key = (key, target_tier)
        fence = self._next_operation(kind, request_id, prefetch_key)

        def worker():
            result = self.migrate(
                key,
                target_tier,
                keep_source=keep_source,
                failure_stage=failure_stage,
                cancellation_key=fence.cancellation_key,
                operation_fence=fence,
            )
            # Publish fence completion at the same serialization point used by
            # cancellation.  A cancel racing after this point observes a
            # completed operation instead of mislabelling a committed copy.
            with self._lock:
                fence.result_value = result
                fence.mark_completed(self._epoch)
            return result

        future = self._executor.submit(worker)
        fence.io_future = future
        self._active_fences[fence.operation_id] = fence

        def complete(completed):
            try:
                fence.result_value = completed.result()
            except CancelledError as error:
                fence.cancelled = True
                fence.status = "cancelled"
                fence.error = error
            except BaseException as error:
                if fence.cancelled:
                    fence.status = "cancelled"
                    fence.error = error
                else:
                    with self._lock:
                        completion_epoch = self._epoch
                    fence.mark_failed(error, completion_epoch)
            else:
                if fence.cancelled:
                    fence.status = "cancelled"
                else:
                    with self._lock:
                        completion_epoch = self._epoch
                    fence.mark_completed(completion_epoch)
            finally:
                with self._lock:
                    if self._prefetches.get(prefetch_key) is fence:
                        self._prefetches.pop(prefetch_key, None)
                    self._active_fences.pop(fence.operation_id, None)
                    self._fence_consumers.pop(fence.operation_id, None)
                    self._cancelled.discard(fence.cancellation_key)

        future.add_done_callback(complete)
        return fence

    def migrate_async(
        self,
        logical_block_id,
        target_tier,
        keep_source=True,
        failure_stage=None,
        request_id=None,
    ):
        """Submit a mock migration and expose only the common fence contract."""
        self._check_open()
        key = _logical_key(logical_block_id)
        target_tier = KVTier(target_tier)
        with self._lock:
            return self._submit_migration(
                key,
                target_tier,
                request_id=request_id,
                kind="migration",
                keep_source=keep_source,
                failure_stage=failure_stage,
            )

    def prefetch(
        self,
        logical_block_id,
        target_tier=KVTier.GPU,
        request_id=None,
    ):
        self._check_open()
        key = _logical_key(logical_block_id)
        target_tier = KVTier(target_tier)
        prefetch_key = (key, target_tier)
        with self._lock:
            self.prefetch_requests += 1
            if self.is_resident(key, target_tier):
                fence = self._next_operation(
                    "prefetch", request_id, prefetch_key
                )
                fence.result_value = self.record(key).locations[target_tier]
                fence.mark_completed(self._epoch)
                return fence
            existing = self._prefetches.get(prefetch_key)
            if existing is not None and not existing.query():
                self.prefetch_deduplicated += 1
                if request_id is not None:
                    self._fence_consumers.setdefault(
                        existing.operation_id, set()
                    ).add(request_id)
                return existing
            fence = self._submit_migration(
                key,
                target_tier,
                request_id=request_id,
                kind="prefetch",
            )
            self._prefetches[prefetch_key] = fence
            if request_id is not None:
                self._fence_consumers[fence.operation_id] = {request_id}
            return fence

    def prefetch_group(self, request_id, cleanup_timeout=5.0):
        return RequestScopedPrefetchGroup(
            self, request_id=request_id, cleanup_timeout=cleanup_timeout
        )

    def release_fence_consumer(self, fence, request_id):
        with self._lock:
            consumers = self._fence_consumers.get(fence.operation_id)
            if consumers is None:
                return False
            consumers.discard(request_id)
            if not consumers:
                self._fence_consumers.pop(fence.operation_id, None)
            return True

    def cancel_fence(self, fence, request_id=None):
        cancellation_key = getattr(fence, "cancellation_key", None)
        with self._lock:
            if request_id is not None:
                consumers = self._fence_consumers.get(fence.operation_id)
                if consumers is not None:
                    consumers.discard(request_id)
                    if consumers:
                        return False
                    self._fence_consumers.pop(fence.operation_id, None)
            if fence.query():
                return False
            if getattr(fence, "metadata_committed", False):
                return False
            if cancellation_key is not None:
                self._cancelled.add(cancellation_key)
            fence.cancel()
            self.prefetch_cancelled += 1
            return True

    def cancel_prefetch(self, logical_block_id, target_tier=KVTier.GPU):
        key = _logical_key(logical_block_id)
        target_tier = KVTier(target_tier)
        prefetch_key = (key, target_tier)
        with self._lock:
            fence = self._prefetches.get(prefetch_key)
            if fence is None:
                return False
        return self.cancel_fence(fence)

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
                    not item.query() for item in self._prefetches.values()
                ),
                "pending_tier_operations": sum(
                    not item.query() for item in self._active_fences.values()
                ),
                "prefetch_consumers": sum(
                    len(item) for item in self._fence_consumers.values()
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
            pending = tuple(self._active_fences.values())
        for fence in pending:
            self.cancel_fence(fence)
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
