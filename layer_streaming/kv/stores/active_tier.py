"""Active GPU-hot/Pinned-CPU KV residency coordinator.

This module composes the Location Plane only.  It never retains or releases a
Request reference: ``KVPagePoolV1`` remains the lifecycle authority.  A caller
acquires a bounded attention wave, reads its GPU layer views, and closes the
wave after the corresponding compute fence has quiesced.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import time
import uuid

import torch

from ..errors import KVCapacityError, KVLifecycleError
from ..fence import KVOperationFence
from .tiered import PrefetchCancelled, RequestScopedPrefetchGroup, ResidencyState


class _ImmediateEvent:
    def query(self):
        return True

    def synchronize(self):
        return None


@dataclass(frozen=True)
class _RegisteredPage:
    page_handle: object
    valid_tokens: int


@dataclass
class ActiveTierAttentionWave:
    """Compute-pinned GPU views for one bounded selected-page wave."""

    coordinator: object
    page_keys: tuple
    request_id: object
    prefetched_page_keys: tuple = ()
    closed: bool = False
    completion_fence: object = None

    def layer_pages(self, layer):
        if self.closed:
            raise KVLifecycleError("attention wave is already closed")
        return tuple(
            self.coordinator.hot_cache.resolve_gpu_layer(key, layer)
            for key in self.page_keys
        )

    def submit_close(self):
        if self.closed:
            return self.completion_fence
        self.completion_fence = self.coordinator._submit_attention_release(
            self.page_keys, self.request_id
        )
        self.closed = True
        return self.completion_fence

    def close(self, timeout=10.0):
        fence = self.submit_close()
        if fence is not None:
            fence.wait(timeout_seconds=timeout)
        return fence

    def __enter__(self):
        if self.closed:
            raise KVLifecycleError("attention wave is already closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


@dataclass
class HotCacheExecutionGuard:
    """Arena-wide slot-reuse guard for device-selected attention.

    This transitional guard deliberately protects Location identity rather than
    managing Page ownership.  ``submit_close`` records the current compute
    stream and releases the arena asynchronously only after that event.
    """

    coordinator: object
    token: str
    request_id: object = None
    submitted: bool = False
    closed: bool = False
    completion_fence: object = None

    def submit_close(self):
        if self.closed:
            return self.completion_fence
        if self.submitted:
            return self.completion_fence
        self.completion_fence = (
            self.coordinator._submit_hot_cache_execution_release(self)
        )
        self.submitted = True
        return self.completion_fence

    def abort(self):
        """Release a guard only when no device work was submitted under it."""

        if self.closed:
            return self.completion_fence
        if self.submitted:
            raise KVLifecycleError(
                "submitted hot-cache execution must complete through its Fence"
            )
        self.coordinator._abort_hot_cache_execution(self)
        self.closed = True
        return None

    def close(self, timeout=10.0):
        fence = self.submit_close()
        if fence is not None:
            fence.wait(timeout_seconds=timeout)
        return fence

    def __enter__(self):
        if self.closed:
            raise KVLifecycleError("hot-cache execution guard is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # A provider may raise after enqueueing partial stream work.  The
        # context-manager path therefore records and waits for the event even
        # on error.  ``abort`` remains an explicit API only for callers that
        # can prove no device work was submitted.
        self.close()
        return False


class ActiveTierCoordinator:
    """Prefetch, eviction, and migration over real tensor-backed tiers."""

    def __init__(
        self,
        hot_cache,
        cpu_store,
        page_pool,
        *,
        event_factory=None,
        compute_event_factory=None,
        max_workers=2,
    ):
        if hot_cache.page_pool is not page_pool:
            raise ValueError("hot cache and coordinator must share PagePool")
        if hot_cache.layer_page_bytes != cpu_store.layer_page_bytes:
            raise ValueError("GPU and CPU Layer Page byte sizes differ")
        if hot_cache.layer_count <= 0:
            raise ValueError("hot cache must contain at least one layer")
        self.hot_cache = hot_cache
        self.cpu_store = cpu_store
        self.page_pool = page_pool
        self._event_factory = event_factory
        self._compute_event_factory = compute_event_factory
        self._stream = (
            torch.cuda.Stream(device=hot_cache.device)
            if hot_cache.device.type == "cuda"
            else None
        )
        self._executor = ThreadPoolExecutor(
            max_workers=int(max_workers),
            thread_name_prefix="cascade-kv-active-tier",
        )
        self._lock = threading.RLock()
        self._capacity_lock = threading.RLock()
        self._registered = {}
        self._prefetches = {}
        self._operations = {}
        self._consumers = {}
        self._groups = {}
        self._execution_guards = {}
        self._closed = False
        self.gpu_hits = 0
        self.cpu_hits = 0
        self.prefetch_count = 0
        self.prefetch_deduplicated = 0
        self.prefetch_bytes = 0
        self.prefetch_wait_ms = 0.0
        self.prefetch_timeouts = 0
        self.eviction_count = 0
        self.eviction_bytes = 0
        self.d2h_bytes = 0
        self.authority_changes = 0
        self.migration_failures = 0
        self.migration_cancellations = 0
        # Count re-fetches occurring within this many completed migration
        # operations after eviction. This is diagnostic only; it never changes
        # LRU eligibility or authority.
        self.thrash_window_operations = 8
        self.thrashing_count = 0
        self._migration_epoch = 0
        self._last_eviction_epoch = {}

    def _check_open(self):
        if self._closed:
            raise KVLifecycleError("Active Tier coordinator is closed")

    def _event(self):
        if self._event_factory is not None:
            return self._event_factory()
        if self.hot_cache.device.type == "cuda":
            return torch.cuda.Event()
        return _ImmediateEvent()

    def _record_compute_event(self):
        if self._compute_event_factory is not None:
            event = self._compute_event_factory()
        elif self.hot_cache.device.type == "cuda":
            event = torch.cuda.Event()
        else:
            return _ImmediateEvent()
        if hasattr(event, "record"):
            event.record(torch.cuda.current_stream(self.hot_cache.device))
        return event

    def begin_hot_cache_execution(self, request_id=None):
        """Begin an ownership-neutral arena guard for device attention."""

        token = "hot-cache-execution-{}".format(uuid.uuid4().hex)
        with self._lock:
            self._check_open()
            self.hot_cache.begin_execution_guard(token)
            guard = HotCacheExecutionGuard(
                coordinator=self,
                token=token,
                request_id=request_id,
            )
            self._execution_guards[token] = guard
            return guard

    def _abort_hot_cache_execution(self, guard):
        with self._lock:
            current = self._execution_guards.get(guard.token)
            if current is not guard:
                raise KVLifecycleError("hot-cache execution guard is not active")
            if guard.submitted:
                raise KVLifecycleError(
                    "submitted hot-cache execution cannot be aborted early"
                )
            self.hot_cache.end_execution_guard(guard.token)
            self._execution_guards.pop(guard.token, None)
            guard.closed = True

    def _submit_hot_cache_execution_release(self, guard):
        with self._lock:
            current = self._execution_guards.get(guard.token)
            if current is not guard or guard.closed:
                raise KVLifecycleError("hot-cache execution guard is not active")
            if guard.submitted:
                return guard.completion_fence
            event = self._record_compute_event()
            fence = KVOperationFence(
                operation_id="hot-cache-execution-fence-{}".format(
                    uuid.uuid4().hex
                ),
                request_id=guard.request_id,
                kind="hot_cache_execution",
            )
            guard.submitted = True
            guard.completion_fence = fence
            self._operations[fence.operation_id] = fence
            try:
                future = self._executor.submit(
                    self._finish_hot_cache_execution, guard, event, fence
                )
            except BaseException as error:
                self._operations.pop(fence.operation_id, None)
                self._execution_guards.pop(guard.token, None)
                self.hot_cache.end_execution_guard(guard.token)
                guard.closed = True
                fence.mark_failed(error)
                raise
            fence.io_future = future
            future.add_done_callback(
                lambda completed: self._cleanup_cancelled_hot_cache_execution(
                    completed, guard, event, fence
                )
            )
            return fence

    def _finish_hot_cache_execution(self, guard, event, fence):
        error = None
        try:
            event.synchronize()
            fence.completion_epoch = fence.submit_epoch
            return True
        except BaseException as current_error:
            error = current_error
            fence.error = current_error
            raise
        finally:
            cleanup_error = None
            try:
                self.hot_cache.end_execution_guard(guard.token)
            except BaseException as current_error:
                cleanup_error = current_error
            with self._lock:
                self._execution_guards.pop(guard.token, None)
                self._operations.pop(fence.operation_id, None)
                guard.closed = True
            if error is None and cleanup_error is not None:
                fence.error = cleanup_error
                raise cleanup_error

    def _cleanup_cancelled_hot_cache_execution(
        self, future, guard, event, fence
    ):
        if not future.cancelled():
            return
        cleanup_error = None
        try:
            # A cancelled worker never observed the event. Slot identity must
            # remain frozen until this explicit cleanup observes completion.
            event.synchronize()
            self.hot_cache.end_execution_guard(guard.token)
        except BaseException as error:
            cleanup_error = error
        with self._lock:
            self._execution_guards.pop(guard.token, None)
            self._operations.pop(fence.operation_id, None)
            guard.closed = True
        fence.error = cleanup_error
        fence.status = "cancelled"

    def register_page(
        self,
        logical_block_id,
        page_handle,
        *,
        valid_tokens,
        data_epoch,
        cpu_slots=None,
    ):
        """Register one ownership page without adding a Request reference."""

        valid_tokens = int(valid_tokens)
        if valid_tokens < 0 or valid_tokens > self.hot_cache.page_size:
            raise ValueError("valid_tokens is outside the token page")
        key = self.hot_cache.register_page(
            logical_block_id,
            page_handle,
            data_epoch,
            cpu_slots=cpu_slots,
        )
        with self._lock:
            self._check_open()
            self._registered[key] = _RegisteredPage(
                page_handle=page_handle, valid_tokens=valid_tokens
            )
        return key

    def adopt_registered_page(self, key, page_handle, valid_tokens):
        """Attach coordinator bookkeeping to a page registered by Runtime."""

        self.hot_cache.location_sets(key)
        if self.page_pool.descriptor(page_handle).data_version != key.data_epoch:
            raise KVLifecycleError("adopted page epoch mismatch")
        with self._lock:
            self._check_open()
            if key in self._registered:
                raise KVLifecycleError("page is already adopted")
            self._registered[key] = _RegisteredPage(
                page_handle=page_handle, valid_tokens=int(valid_tokens)
            )
        return key

    def advance_data_epoch(
        self,
        key,
        new_epoch,
        *,
        valid_tokens,
        end_operation="append",
    ):
        """Rekey Location bookkeeping after PagePool publishes Append data."""

        valid_tokens = int(valid_tokens)
        if valid_tokens < 0 or valid_tokens > self.hot_cache.page_size:
            raise ValueError("valid_tokens is outside the token page")
        with self._lock:
            registration = self._registration(key)
            new_key, stale_cpu_handles = self.hot_cache.advance_data_epoch(
                key, int(new_epoch), end_operation=end_operation
            )
            del self._registered[key]
            self._registered[new_key] = _RegisteredPage(
                registration.page_handle, valid_tokens
            )
        first_error = None
        for handle in stale_cpu_handles:
            try:
                self.cpu_store.release(handle)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error
        return new_key

    def _registration(self, key):
        with self._lock:
            try:
                return self._registered[key]
            except KeyError:
                raise KeyError("unregistered Active Tier page")

    def _gpu_resident(self, key, layer=None):
        if layer is not None:
            location = self.hot_cache.location_set(key, int(layer))
            return (
                location.gpu_state == ResidencyState.RESIDENT
                and location.gpu_slot is not None
            )
        locations = self.hot_cache.location_sets(key)
        return all(
            item.gpu_state == ResidencyState.RESIDENT
            and item.gpu_slot is not None
            for item in locations
        )

    def _cpu_resident(self, key):
        locations = self.hot_cache.location_sets(key)
        return all(
            item.cpu_state == ResidencyState.RESIDENT
            and item.cpu_slot is not None
            for item in locations
        )

    def prefetch_group(self, request_id, cleanup_timeout=5.0):
        return RequestScopedPrefetchGroup(
            self, request_id=request_id, cleanup_timeout=cleanup_timeout
        )

    def _completed_fence(self, kind, request_id, key):
        fence = KVOperationFence(
            operation_id="active-{}-{}".format(kind, uuid.uuid4().hex),
            request_id=request_id,
            kind=kind,
            submit_epoch=key.data_epoch,
        )
        fence.mark_completed(key.data_epoch)
        return fence

    def _begin_gpu_io(self, key, operation):
        registration = self._registration(key)
        self.hot_cache.begin_pending(key, operation)
        try:
            self.hot_cache.pin_slot(key, kind="io")
            try:
                self.page_pool.pin(registration.page_handle, kind="io")
            except BaseException:
                self.hot_cache.cleanup_operation(
                    key, operation=operation, kind="io"
                )
                raise
        except BaseException:
            try:
                self.hot_cache.cleanup_operation(key, operation=operation)
            except BaseException:
                pass
            raise
        return registration

    def _finish_gpu_io(self, key, operation, registration, *, loading_slot=None):
        cleanup_error = None
        try:
            if loading_slot is None:
                self.hot_cache.cleanup_operation(
                    key, operation=operation, kind="io"
                )
            else:
                self.hot_cache.abort_gpu_load(
                    key,
                    loading_slot,
                    operation=operation,
                    kind="io",
                )
        except BaseException as error:
            cleanup_error = error
        try:
            self.page_pool.unpin(registration.page_handle, kind="io")
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if cleanup_error is not None:
            raise cleanup_error

    def _copy_pairs(self, pairs, event):
        if self._stream is None:
            for target, source in pairs:
                target.copy_(source)
            return
        with torch.cuda.stream(self._stream):
            for target, source in pairs:
                target.copy_(source, non_blocking=True)
            event.record(self._stream)

    def _ensure_free_slot(self, exclude_keys):
        with self._capacity_lock:
            if self.hot_cache.free_pages:
                return
            victims = self.hot_cache.lru_victims(
                exclude_keys=exclude_keys, limit=1
            )
            if not victims:
                raise KVCapacityError(
                    "GPU hot cache has no legal eviction victim"
                )
            self.evict(victims[0]).wait(timeout_seconds=10.0)
            if not self.hot_cache.free_pages:
                raise KVCapacityError("GPU eviction did not release a slot")

    def prepare_append(
        self,
        key,
        *,
        request_id,
        timeout=10.0,
        exclude_keys=(),
        operation=None,
    ):
        """Make one page GPU-writable and protect it until epoch publication."""

        operation = (
            "append-{}".format(uuid.uuid4().hex)
            if operation is None
            else str(operation)
        )
        if not operation:
            raise ValueError("append operation must not be empty")
        with self._capacity_lock:
            if not self._gpu_resident(key):
                if self._cpu_resident(key):
                    self.prefetch(
                        key,
                        request_id=request_id,
                        exclude_keys=set(exclude_keys) | {key},
                    ).wait(timeout_seconds=timeout)
                else:
                    self._ensure_free_slot(set(exclude_keys) | {key})
                    self.hot_cache.reserve_gpu(key)
            locations = self.hot_cache.location_sets(key)
            slots = {item.gpu_slot for item in locations}
            if len(slots) != 1 or None in slots:
                raise KVLifecycleError("append page has no single GPU slot")
            slot = next(iter(slots))
            try:
                self.hot_cache.begin_pending(key, operation)
            except BaseException:
                if any(
                    item.gpu_state == ResidencyState.LOADING
                    for item in locations
                ):
                    self.hot_cache.abort_gpu_load(key, slot)
                raise
            return slot, operation

    def abort_append(
        self, key, slot, operation, *, drop_loading=True
    ):
        """Unwind Location state after the Ownership append aborts."""

        locations = self.hot_cache.location_sets(key)
        loading = any(
            item.gpu_state == ResidencyState.LOADING for item in locations
        )
        if loading and drop_loading:
            return self.hot_cache.abort_gpu_load(
                key, slot, operation=operation
            )
        return self.hot_cache.cleanup_operation(key, operation=operation)

    def prefetch(self, key, request_id=None, *, exclude_keys=()):
        """Deduplicate and submit one complete cross-layer CPU->GPU page."""

        with self._lock:
            self._check_open()
            if self._gpu_resident(key):
                self.gpu_hits += 1
                return self._completed_fence("prefetch_hit", request_id, key)
            if not self._cpu_resident(key):
                raise KVLifecycleError("prefetch requires every CPU layer replica")
            existing = self._prefetches.get(key)
            if existing is not None and not existing.query():
                self.prefetch_deduplicated += 1
                if request_id is not None:
                    self._consumers.setdefault(existing.operation_id, set()).add(
                        request_id
                    )
                return existing

        self._ensure_free_slot(set(exclude_keys) | {key})
        with self._capacity_lock, self._lock:
            if self._gpu_resident(key):
                self.gpu_hits += 1
                return self._completed_fence("prefetch_hit", request_id, key)
            existing = self._prefetches.get(key)
            if existing is not None and not existing.query():
                self.prefetch_deduplicated += 1
                if request_id is not None:
                    self._consumers.setdefault(existing.operation_id, set()).add(
                        request_id
                    )
                return existing
            slot = self.hot_cache.reserve_gpu(key)
            operation = "prefetch-{}".format(uuid.uuid4().hex)
            try:
                registration = self._begin_gpu_io(key, operation)
            except BaseException:
                self.hot_cache.abort_gpu_load(key, slot)
                raise
            fence = KVOperationFence(
                operation_id=operation,
                request_id=request_id,
                kind="prefetch",
                source_handles=(registration.page_handle,),
                submit_epoch=key.data_epoch,
            )
            self.prefetch_count += 1
            self.cpu_hits += 1
            self._prefetches[key] = fence
            self._operations[operation] = fence
            if request_id is not None:
                self._consumers[operation] = {request_id}
            try:
                future = self._executor.submit(
                    self._run_h2d, key, slot, operation, registration, fence
                )
            except BaseException as error:
                # The worker never started, so its finally block cannot close
                # the PagePool/hot-cache IO pins or the LOADING slot.
                self.migration_failures += 1
                fence.error = error
                cleanup_error = None
                try:
                    self._finish_gpu_io(
                        key, operation, registration, loading_slot=slot
                    )
                except BaseException as current_error:
                    cleanup_error = current_error
                self._prefetches.pop(key, None)
                self._operations.pop(operation, None)
                self._consumers.pop(operation, None)
                if cleanup_error is not None:
                    try:
                        error.kv_cleanup_error = cleanup_error
                    except BaseException:
                        pass
                raise
            fence.io_future = future
            future.add_done_callback(
                lambda completed: self._cleanup_cancelled_h2d_submission(
                    completed, key, slot, operation, registration, fence
                )
            )
            return fence

    def _cleanup_cancelled_h2d_submission(
        self, future, key, slot, operation, registration, fence
    ):
        """Unwind pins when Future.cancel() prevented worker startup."""

        if not future.cancelled():
            return
        cleanup_error = None
        try:
            self._finish_gpu_io(
                key, operation, registration, loading_slot=slot
            )
        except BaseException as error:
            cleanup_error = error
        with self._lock:
            self.migration_cancellations += 1
            self._prefetches.pop(key, None)
            self._operations.pop(operation, None)
        fence.error = cleanup_error
        fence.status = "cancelled"

    def _run_h2d(self, key, slot, operation, registration, fence):
        try:
            pairs = []
            for layer, location in enumerate(self.hot_cache.location_sets(key)):
                page = self.cpu_store.read(
                    location.cpu_slot, expected_data_epoch=key.data_epoch
                )
                target_key, target_value = self.hot_cache.layer_write_target(
                    key, layer, slot
                )
                pairs.extend(((target_key, page.key), (target_value, page.value)))
            event = self._event()
            self._copy_pairs(pairs, event)
            event.synchronize()
            if fence.cancelled:
                raise PrefetchCancelled("Active Tier H2D was cancelled")
            self.hot_cache.commit_gpu(
                key, slot, expected_data_epoch=key.data_epoch,
                make_authoritative=False,
            )
            self.hot_cache.cleanup_operation(
                key, operation=operation, kind="io"
            )
            self.page_pool.unpin(registration.page_handle, kind="io")
            self.prefetch_bytes += self.hot_cache.gpu_page_bytes
            with self._lock:
                self._migration_epoch += 1
                evicted_at = self._last_eviction_epoch.get(key)
                if (
                    evicted_at is not None
                    and self._migration_epoch - evicted_at
                    <= self.thrash_window_operations
                ):
                    self.thrashing_count += 1
            fence.completion_epoch = key.data_epoch
            return slot
        except BaseException as error:
            self.migration_cancellations += int(fence.cancelled)
            self.migration_failures += int(not fence.cancelled)
            fence.error = error
            fence.completion_epoch = key.data_epoch
            try:
                self._finish_gpu_io(
                    key, operation, registration, loading_slot=slot
                )
            except BaseException as cleanup_error:
                try:
                    error.kv_cleanup_error = cleanup_error
                except BaseException:
                    pass
            raise
        finally:
            with self._lock:
                self._prefetches.pop(key, None)
                self._operations.pop(operation, None)

    def evict(self, key):
        """Ensure a CPU replica, then release the reusable GPU hot slot."""

        with self._lock:
            self._check_open()
            if not self._gpu_resident(key):
                raise KVLifecycleError("eviction requires a GPU-resident page")
            if self._cpu_resident(key):
                changed_authority = any(
                    item.authoritative_tier == "gpu"
                    for item in self.hot_cache.location_sets(key)
                )
                self.hot_cache.release_gpu(key)
                self.eviction_count += 1
                self.eviction_bytes += self.hot_cache.gpu_page_bytes
                self.authority_changes += int(changed_authority)
                self._migration_epoch += 1
                self._last_eviction_epoch[key] = self._migration_epoch
                return self._completed_fence("eviction", None, key)
            operation = "eviction-{}".format(uuid.uuid4().hex)
            registration = self._begin_gpu_io(key, operation)
            fence = KVOperationFence(
                operation_id=operation,
                request_id=None,
                kind="eviction",
                source_handles=(registration.page_handle,),
                submit_epoch=key.data_epoch,
            )
            self._operations[operation] = fence
            try:
                future = self._executor.submit(
                    self._run_d2h, key, operation, registration, fence
                )
            except BaseException as error:
                # No worker means no asynchronous cleanup path. Restore the
                # pre-eviction GPU-resident state before surfacing failure.
                self.migration_failures += 1
                fence.error = error
                cleanup_error = None
                try:
                    self._finish_gpu_io(key, operation, registration)
                except BaseException as current_error:
                    cleanup_error = current_error
                self._operations.pop(operation, None)
                if cleanup_error is not None:
                    try:
                        error.kv_cleanup_error = cleanup_error
                    except BaseException:
                        pass
                raise
            fence.io_future = future
            future.add_done_callback(
                lambda completed: self._cleanup_cancelled_d2h_submission(
                    completed, key, operation, registration, fence
                )
            )
            return fence

    def _cleanup_cancelled_d2h_submission(
        self, future, key, operation, registration, fence
    ):
        if not future.cancelled():
            return
        cleanup_error = None
        try:
            self._finish_gpu_io(key, operation, registration)
        except BaseException as error:
            cleanup_error = error
        with self._lock:
            self.migration_cancellations += 1
            self._operations.pop(operation, None)
        fence.error = cleanup_error
        fence.status = "cancelled"

    def _run_d2h(self, key, operation, registration, fence):
        cpu_handles = {}
        attached = False
        io_cleaned = False
        try:
            pairs = []
            for layer in range(self.hot_cache.layer_count):
                reservation = self.cpu_store.reserve()
                try:
                    handle = self.cpu_store.allocate(
                        reservation,
                        logical_block_id=key.logical_key,
                        layer=layer,
                        data_epoch=key.data_epoch,
                    )
                except BaseException:
                    self.cpu_store.cancel_reservation(reservation)
                    raise
                cpu_handles[layer] = handle
                target_key, target_value = self.cpu_store.write_target(
                    handle, expected_data_epoch=key.data_epoch
                )
                source = self.hot_cache.resolve_gpu_layer(
                    key, layer, touch=False
                )
                pairs.extend(((target_key, source.key), (target_value, source.value)))
            event = self._event()
            self._copy_pairs(pairs, event)
            event.synchronize()
            if fence.cancelled:
                raise PrefetchCancelled("Active Tier D2H was cancelled")
            for handle in cpu_handles.values():
                self.cpu_store.commit_write(
                    handle,
                    valid_tokens=registration.valid_tokens,
                    data_epoch=key.data_epoch,
                )
            self.hot_cache.attach_cpu_page(
                key,
                cpu_handles,
                expected_data_epoch=key.data_epoch,
                make_authoritative=False,
            )
            attached = True
            self.hot_cache.cleanup_operation(
                key, operation=operation, kind="io"
            )
            self.page_pool.unpin(registration.page_handle, kind="io")
            io_cleaned = True
            changed_authority = any(
                item.authoritative_tier == "gpu"
                for item in self.hot_cache.location_sets(key)
            )
            self.hot_cache.release_gpu(key)
            self.eviction_count += 1
            self.eviction_bytes += self.hot_cache.gpu_page_bytes
            self.d2h_bytes += self.hot_cache.gpu_page_bytes
            self.authority_changes += int(changed_authority)
            with self._lock:
                self._migration_epoch += 1
                self._last_eviction_epoch[key] = self._migration_epoch
            fence.completion_epoch = key.data_epoch
            return tuple(cpu_handles.values())
        except BaseException as error:
            self.migration_cancellations += int(fence.cancelled)
            self.migration_failures += int(not fence.cancelled)
            fence.error = error
            fence.completion_epoch = key.data_epoch
            if not io_cleaned:
                try:
                    self._finish_gpu_io(key, operation, registration)
                except BaseException as cleanup_error:
                    try:
                        error.kv_cleanup_error = cleanup_error
                    except BaseException:
                        pass
            if not attached:
                for handle in cpu_handles.values():
                    try:
                        self.cpu_store.release(handle)
                    except BaseException as cleanup_error:
                        try:
                            error.kv_cleanup_error = cleanup_error
                        except BaseException:
                            pass
            raise
        finally:
            with self._lock:
                self._operations.pop(operation, None)

    def release_fence_consumer(self, fence, request_id):
        with self._lock:
            consumers = self._consumers.get(fence.operation_id)
            if consumers is None:
                return False
            consumers.discard(request_id)
            if not consumers:
                self._consumers.pop(fence.operation_id, None)
            return True

    def cancel_fence(self, fence, request_id=None):
        with self._lock:
            if request_id is not None:
                consumers = self._consumers.get(fence.operation_id)
                if consumers is not None:
                    consumers.discard(request_id)
                    if consumers:
                        return False
                    self._consumers.pop(fence.operation_id, None)
            if fence.query():
                return False
            # Cooperative cancellation deliberately leaves the Future queued.
            # Future.cancel() may prevent the worker's cleanup finally block
            # from ever running, stranding the PagePool/hot-cache IO pins.
            fence.cancelled = True
            return True

    def cancel_request(self, request_id, timeout=5.0):
        with self._lock:
            group = self._groups.get(request_id)
        if group is None:
            return False
        group.cleanup(timeout=timeout)
        return True

    def quiesce(self, timeout=10.0, request_id=None):
        """Drain submitted tier/device fences at an explicit safe boundary."""

        deadline = time.monotonic() + float(timeout)
        while True:
            with self._lock:
                fences = tuple(
                    fence
                    for fence in self._operations.values()
                    if request_id is None or fence.request_id == request_id
                )
                unsubmitted = tuple(
                    guard
                    for guard in self._execution_guards.values()
                    if (request_id is None or guard.request_id == request_id)
                    and not guard.submitted
                )
            for guard in unsubmitted:
                guard.abort()
            if not fences:
                with self._lock:
                    remaining = any(
                        request_id is None or fence.request_id == request_id
                        for fence in self._operations.values()
                    )
                if not remaining:
                    return True
            for fence in fences:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise KVLifecycleError(
                        "Active Tier quiesce timed out with pending operations"
                    )
                fence.wait(timeout_seconds=remaining_seconds)

    def acquire_wave(
        self,
        page_keys,
        *,
        request_id,
        timeout=10.0,
        cancel_event=None,
        layer=None,
    ):
        """Prefetch and compute-pin one bounded attention wave."""

        keys = tuple(page_keys)
        if layer is not None:
            layer = int(layer)
            if layer < 0 or layer >= self.hot_cache.layer_count:
                raise IndexError("attention wave layer is outside the cache")
        if len(set(keys)) != len(keys):
            raise ValueError("attention wave page keys must be unique")
        if len(keys) > self.hot_cache.gpu_capacity_pages:
            raise KVCapacityError(
                "attention wave has {} pages, above GPU hot capacity {}".format(
                    len(keys), self.hot_cache.gpu_capacity_pages
                )
            )
        group = self.prefetch_group(
            request_id, cleanup_timeout=max(1.0, float(timeout))
        )
        with self._lock:
            if request_id in self._groups:
                raise KVLifecycleError("request already has an active wave")
            self._groups[request_id] = group
        started = time.perf_counter()
        try:
            prefetched_keys = []
            for key in keys:
                if cancel_event is not None and cancel_event.is_set():
                    raise PrefetchCancelled("attention wave was cancelled")
                if not self._gpu_resident(key, layer=layer):
                    prefetched_keys.append(key)
                    group.add(
                        self.prefetch(
                            key,
                            request_id=request_id,
                            exclude_keys=keys,
                        )
                    )
                else:
                    self.gpu_hits += 1
            group.wait(timeout=timeout, cancel_event=cancel_event)
            pinned = []
            try:
                for key in keys:
                    self._pin_attention_key(key, layer=layer)
                    pinned.append(key)
            except BaseException:
                self._release_attention_wave_now(tuple(pinned))
                raise
            return ActiveTierAttentionWave(
                self,
                keys,
                request_id,
                prefetched_page_keys=tuple(prefetched_keys),
            )
        except TimeoutError:
            self.prefetch_timeouts += 1
            group.cleanup(timeout=max(1.0, float(timeout)))
            raise
        except BaseException:
            try:
                group.cleanup(timeout=max(1.0, float(timeout)))
            except BaseException:
                pass
            raise
        finally:
            self.prefetch_wait_ms += (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._groups.pop(request_id, None)

    def _pin_attention_key(self, key, layer=None):
        registration = self._registration(key)
        selected = False
        hot_pinned = False
        try:
            self.hot_cache.mark_selected(key)
            selected = True
            self.hot_cache.pin_slot(key, kind="compute", layer=layer)
            hot_pinned = True
            self.page_pool.pin(registration.page_handle, kind="compute")
        except BaseException:
            if hot_pinned:
                self.hot_cache.unpin_slot(key, kind="compute")
            if selected:
                self.hot_cache.unmark_selected(key)
            raise

    def _submit_attention_release(self, keys, request_id):
        keys = tuple(keys)
        event = self._record_compute_event()
        fence = KVOperationFence(
            operation_id="attention-wave-{}".format(uuid.uuid4().hex),
            request_id=request_id,
            kind="attention_wave",
            source_handles=tuple(
                self._registration(key).page_handle for key in keys
            ),
            submit_epoch=max((key.data_epoch for key in keys), default=0),
        )
        with self._lock:
            self._operations[fence.operation_id] = fence
        future = self._executor.submit(
            self._finish_attention_release, keys, event, fence
        )
        fence.io_future = future
        future.add_done_callback(
            lambda completed: self._cleanup_cancelled_attention_submission(
                completed, keys, event, fence
            )
        )
        return fence

    def _cleanup_cancelled_attention_submission(
        self, future, keys, event, fence
    ):
        if not future.cancelled():
            return
        cleanup_error = None
        try:
            # Direct external Fence cancellation is allowed to block here:
            # compute pins cannot be released before the recorded stream Event.
            event.synchronize()
            self._release_attention_wave_now(keys)
        except BaseException as error:
            cleanup_error = error
        with self._lock:
            self._operations.pop(fence.operation_id, None)
        fence.error = cleanup_error
        fence.status = "cancelled"

    def _finish_attention_release(self, keys, event, fence):
        try:
            event.synchronize()
            self._release_attention_wave_now(keys)
            fence.completion_epoch = fence.submit_epoch
            return True
        except BaseException as error:
            fence.error = error
            raise
        finally:
            with self._lock:
                self._operations.pop(fence.operation_id, None)

    def _release_attention_wave_now(self, keys):
        first_error = None
        for key in reversed(tuple(keys)):
            registration = self._registration(key)
            for action in (
                lambda: self.page_pool.unpin(
                    registration.page_handle, kind="compute"
                ),
                lambda: self.hot_cache.unpin_slot(key, kind="compute"),
                lambda: self.hot_cache.unmark_selected(key),
            ):
                try:
                    action()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        if first_error is not None:
            raise first_error

    def unregister_page(self, key, *, validate_epoch=True):
        registration = self._registration(key)
        cpu_slots = self.hot_cache.unregister_page(
            key,
            expected_data_epoch=(key.data_epoch if validate_epoch else None),
            validate_epoch=validate_epoch,
        )
        for slot in cpu_slots:
            self.cpu_store.release(slot, expected_data_epoch=key.data_epoch)
        with self._lock:
            self._registered.pop(key, None)
        # Prove that this Location cleanup did not alter Request ownership.
        self.page_pool.descriptor(registration.page_handle)

    def stats(self):
        hot = self.hot_cache.stats()
        cpu = self.cpu_store.stats()
        with self._lock:
            return {
                "tier_metrics_sampled": bool(
                    self.prefetch_count or self.eviction_count
                ),
                **hot,
                "cpu_kv_capacity_bytes": cpu["capacity_bytes"],
                "cpu_kv_used_bytes": cpu["used_bytes"],
                "cpu_kv_capacity_pages": cpu["slot_count"]
                // self.hot_cache.layer_count,
                "cpu_kv_used_pages": (
                    cpu["used_bytes"]
                    + cpu["layer_page_bytes"] * self.hot_cache.layer_count
                    - 1
                ) // (cpu["layer_page_bytes"] * self.hot_cache.layer_count),
                "cpu_kv_free_pages": cpu["free_slots"]
                // self.hot_cache.layer_count,
                "cpu_kv_high_watermark_bytes": cpu[
                    "high_watermark_bytes"
                ],
                "cpu_kv_low_watermark_bytes": cpu[
                    "low_watermark_bytes"
                ],
                "gpu_hits": self.gpu_hits,
                "cpu_hits": self.cpu_hits,
                "prefetch_count": self.prefetch_count,
                "prefetch_pages": self.prefetch_count,
                "prefetch_deduplicated": self.prefetch_deduplicated,
                "prefetch_bytes": self.prefetch_bytes,
                "prefetch_wait_ms": self.prefetch_wait_ms,
                "prefetch_timeouts": self.prefetch_timeouts,
                "eviction_count": self.eviction_count,
                "eviction_pages": self.eviction_count,
                "eviction_bytes": self.eviction_bytes,
                "h2d_kv_bytes": self.prefetch_bytes,
                "d2h_kv_bytes": self.d2h_bytes,
                "authority_changes": self.authority_changes,
                "migration_failures": self.migration_failures,
                "migration_cancellations": self.migration_cancellations,
                "thrashing_count": self.thrashing_count,
                "thrash_window_operations": self.thrash_window_operations,
                "pending_tier_operations": len(self._operations),
                "active_prefetch_groups": len(self._groups),
                "active_hot_cache_execution_guards": len(
                    self._execution_guards
                ),
            }

    def close(self):
        with self._lock:
            if self._closed:
                return
            groups = tuple(self._groups.values())
            guards = tuple(self._execution_guards.values())
        for group in groups:
            try:
                group.cleanup()
            except BaseException:
                pass
        for guard in guards:
            if guard.submitted:
                continue
            try:
                guard.abort()
            except BaseException:
                pass
        with self._lock:
            fences = tuple(self._operations.values())
        for fence in fences:
            # See cancel_fence(): let queued workers run their cleanup paths.
            fence.cancelled = True
            try:
                fence.wait()
            except BaseException:
                pass
        self._executor.shutdown(wait=True)
        cleanup_error = None
        with self._lock:
            registered = tuple(self._registered)
        for key in registered:
            try:
                self.unregister_page(key, validate_epoch=False)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        with self._lock:
            self._operations.clear()
            self._prefetches.clear()
            self._consumers.clear()
            self._groups.clear()
            self._execution_guards.clear()
            self._last_eviction_epoch.clear()
            self._closed = True
        if cleanup_error is not None:
            raise cleanup_error

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
