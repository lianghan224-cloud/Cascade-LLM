"""Generation-safe V1 page allocator and lifecycle owner."""

from contextlib import contextmanager
import heapq
import threading
import time

from .errors import KVCapacityError, KVLifecycleError
from .types import PageDescriptor, PageHandle, PageState


_BUSY_STATES = {
    PageState.COPYING,
    PageState.MIGRATING,
    PageState.EVICT_PENDING,
    PageState.RELEASING,
}


class KVPagePoolV1:
    """Own descriptors and references; payload lives in a KVStore."""

    def __init__(
        self,
        page_count,
        store_id,
        dtype,
        layout="hnd",
        format_version=1,
        reserved_free_pages=0,
    ):
        self.page_count = int(page_count)
        self.store_id = str(store_id)
        self.dtype = str(dtype)
        self.layout = str(layout)
        self.format_version = int(format_version)
        self.reserved_free_pages = int(reserved_free_pages)
        if self.page_count <= 0:
            raise ValueError("page_count must be positive")
        if self.reserved_free_pages < 0 or self.reserved_free_pages >= self.page_count:
            raise ValueError("reserved_free_pages must be smaller than page_count")
        self.descriptors = [
            PageDescriptor(
                page_id=index,
                generation=0,
                store_id=self.store_id,
                dtype=self.dtype,
                layout=self.layout,
                format_version=self.format_version,
            )
            for index in range(self.page_count)
        ]
        self._free = list(range(self.page_count))
        heapq.heapify(self._free)
        self._epoch = 0
        self._peak_allocated = 0
        self._allocation_count = 0
        self._release_count = 0
        self._lock = threading.RLock()

    @property
    def free_pages(self):
        with self._lock:
            return len(self._free)

    @property
    def allocated_pages(self):
        return self.page_count - self.free_pages

    @property
    def peak_allocated_pages(self):
        with self._lock:
            return self._peak_allocated

    def can_allocate(self, count, include_reserve=False):
        count = int(count)
        available = self.free_pages
        if not include_reserve:
            available -= self.reserved_free_pages
        return count >= 0 and count <= available

    def _touch(self, descriptor):
        self._epoch += 1
        descriptor.last_access_epoch = self._epoch

    def allocate(self, owner_hint=None, include_reserve=False):
        with self._lock:
            if not self.can_allocate(1, include_reserve=include_reserve):
                raise KVCapacityError(
                    "KV page pool is exhausted (free={}, reserved={})".format(
                        len(self._free),
                        self.reserved_free_pages,
                    )
                )
            page_id = heapq.heappop(self._free)
            descriptor = self.descriptors[page_id]
            if descriptor.state != PageState.FREE or descriptor.ref_count:
                raise KVLifecycleError("free page descriptor is inconsistent")
            descriptor.generation += 1
            descriptor.state = PageState.ALLOCATED
            descriptor.valid_tokens = 0
            descriptor.ref_count = 1
            descriptor.pin_count = 0
            descriptor.owner_hint = owner_hint
            descriptor.index_metadata_handle = None
            descriptor.transfer_event = None
            descriptor.data_version = 0
            descriptor.index_version = 0
            descriptor.inflight_compute = 0
            descriptor.inflight_io = 0
            descriptor.dirty = False
            descriptor.error = None
            descriptor.logical_mappings.clear()
            self._touch(descriptor)
            self._peak_allocated = max(self._peak_allocated, self.allocated_pages)
            self._allocation_count += 1
            return PageHandle(descriptor)

    def descriptor(self, handle):
        if not isinstance(handle, PageHandle):
            raise TypeError("expected PageHandle")
        if handle.page_id < 0 or handle.page_id >= self.page_count:
            raise KVLifecycleError("page handle is outside the pool")
        descriptor = self.descriptors[handle.page_id]
        if (
            descriptor.generation != handle.generation
            or descriptor.store_id != handle.store_id
            or descriptor.format_version != self.format_version
        ):
            raise KVLifecycleError("stale or foreign PageHandle")
        return descriptor

    def activate(self, handle, valid_tokens):
        with self._lock:
            descriptor = self.descriptor(handle)
            if descriptor.ref_count != 1:
                raise KVLifecycleError("shared page cannot become mutable")
            if descriptor.state not in {
                PageState.ALLOCATED,
                PageState.ACTIVE,
                PageState.SEALED,
            }:
                raise KVLifecycleError(
                    "cannot activate page in state {}".format(descriptor.state.value)
                )
            descriptor.state = PageState.ACTIVE
            descriptor.valid_tokens = int(valid_tokens)
            self._touch(descriptor)

    def seal(self, handle, valid_tokens):
        with self._lock:
            descriptor = self.descriptor(handle)
            if descriptor.state in _BUSY_STATES or descriptor.state == PageState.FREE:
                raise KVLifecycleError(
                    "cannot seal page in state {}".format(descriptor.state.value)
                )
            descriptor.valid_tokens = int(valid_tokens)
            descriptor.state = (
                PageState.SHARED if descriptor.ref_count > 1 else PageState.SEALED
            )
            self._touch(descriptor)

    def retain(self, handle):
        with self._lock:
            descriptor = self.descriptor(handle)
            if descriptor.state not in {PageState.SEALED, PageState.SHARED}:
                raise KVLifecycleError("only sealed pages can be shared")
            descriptor.ref_count += 1
            descriptor.state = PageState.SHARED
            self._touch(descriptor)

    def begin_copy(self, source, target):
        with self._lock:
            source_descriptor = self.descriptor(source)
            target_descriptor = self.descriptor(target)
            if source_descriptor.state not in {PageState.SEALED, PageState.SHARED}:
                raise KVLifecycleError("copy source must be sealed")
            if target_descriptor.state != PageState.ALLOCATED:
                raise KVLifecycleError("copy target must be allocated")
            source_descriptor.pin_count += 1
            source_descriptor.inflight_io += 1
            target_descriptor.state = PageState.COPYING
            self._touch(source_descriptor)
            self._touch(target_descriptor)

    def end_copy(self, source, target, valid_tokens):
        with self._lock:
            source_descriptor = self.descriptor(source)
            target_descriptor = self.descriptor(target)
            if target_descriptor.state != PageState.COPYING:
                raise KVLifecycleError("copy target is not in COPYING state")
            if source_descriptor.pin_count <= 0:
                raise KVLifecycleError("copy source pin underflow")
            if source_descriptor.inflight_io <= 0:
                raise KVLifecycleError("copy source IO accounting underflow")
            source_descriptor.pin_count -= 1
            source_descriptor.inflight_io -= 1
            target_descriptor.valid_tokens = int(valid_tokens)
            target_descriptor.state = PageState.ACTIVE
            self._touch(source_descriptor)
            self._touch(target_descriptor)

    def pin(self, handle, kind="compute"):
        with self._lock:
            descriptor = self.descriptor(handle)
            if descriptor.state == PageState.FREE:
                raise KVLifecycleError("cannot pin a free page")
            if kind not in {"compute", "io"}:
                raise ValueError("pin kind must be compute or io")
            descriptor.pin_count += 1
            if kind == "compute":
                descriptor.inflight_compute += 1
            else:
                descriptor.inflight_io += 1
            self._touch(descriptor)

    def unpin(self, handle, kind="compute"):
        with self._lock:
            descriptor = self.descriptor(handle)
            if descriptor.pin_count <= 0:
                raise KVLifecycleError("page pin underflow")
            if kind not in {"compute", "io"}:
                raise ValueError("pin kind must be compute or io")
            counter = (
                descriptor.inflight_compute
                if kind == "compute"
                else descriptor.inflight_io
            )
            if counter <= 0:
                raise KVLifecycleError(
                    "page {} pin kind {} underflow".format(
                        descriptor.page_id, kind
                    )
                )
            descriptor.pin_count -= 1
            if kind == "compute":
                descriptor.inflight_compute -= 1
            else:
                descriptor.inflight_io -= 1
            self._touch(descriptor)

    @contextmanager
    def pinned(self, handles, kind="compute"):
        pinned = []
        try:
            for handle in handles:
                self.pin(handle, kind=kind)
                pinned.append(handle)
            yield
        finally:
            for handle in reversed(pinned):
                self.unpin(handle, kind=kind)

    def mark_data_updated(self, handle, version=None):
        """Commit a page data version and invalidate any older index."""

        with self._lock:
            descriptor = self.descriptor(handle)
            if descriptor.state == PageState.FREE:
                raise KVLifecycleError("cannot version a free page")
            next_version = (
                descriptor.data_version + 1
                if version is None
                else int(version)
            )
            if next_version < descriptor.data_version:
                raise KVLifecycleError("data version cannot move backwards")
            descriptor.data_version = next_version
            descriptor.dirty = descriptor.index_version != next_version
            self._touch(descriptor)
            return next_version

    def mark_index_built(self, handle, version):
        with self._lock:
            descriptor = self.descriptor(handle)
            version = int(version)
            if version != descriptor.data_version:
                raise KVLifecycleError(
                    "index version {} does not match page {} data version {}".format(
                        version,
                        descriptor.page_id,
                        descriptor.data_version,
                    )
                )
            descriptor.index_version = version
            descriptor.dirty = False
            self._touch(descriptor)

    def attach_index(self, handle, index_metadata_handle, version):
        """Atomically publish a queryable index record for a page version."""

        with self._lock:
            descriptor = self.descriptor(handle)
            self.mark_index_built(handle, version)
            descriptor.index_metadata_handle = index_metadata_handle
            descriptor.error = None
            self._touch(descriptor)

    def add_logical_mapping(self, handle, mapping):
        with self._lock:
            descriptor = self.descriptor(handle)
            descriptor.logical_mappings.add(tuple(mapping))

    def remove_logical_mapping(self, handle, mapping):
        with self._lock:
            descriptor = self.descriptor(handle)
            descriptor.logical_mappings.discard(tuple(mapping))

    def assert_releasable(self, handles, operation="release"):
        blocked = []
        for handle in handles:
            descriptor = self.descriptor(handle)
            reason = None
            if descriptor.pin_count:
                reason = "pinned"
            elif descriptor.inflight_compute or descriptor.inflight_io:
                reason = "inflight"
            elif descriptor.state in _BUSY_STATES:
                reason = descriptor.state.value
            elif descriptor.ref_count <= 0:
                reason = "invalid_ref_count"
            if reason:
                blocked.append((handle.page_id, reason))
        if blocked:
            raise KVLifecycleError(
                "cannot {}; busy KV pages: {}".format(operation, blocked)
            )

    def release(self, handle):
        with self._lock:
            self.assert_releasable((handle,))
            descriptor = self.descriptor(handle)
            descriptor.ref_count -= 1
            if descriptor.ref_count:
                descriptor.state = (
                    PageState.SHARED
                    if descriptor.ref_count > 1
                    else PageState.SEALED
                )
                self._touch(descriptor)
                return False
            descriptor.state = PageState.RELEASING
            self._touch(descriptor)
            descriptor.valid_tokens = 0
            descriptor.owner_hint = None
            descriptor.index_metadata_handle = None
            descriptor.transfer_event = None
            descriptor.data_version = 0
            descriptor.index_version = 0
            descriptor.inflight_compute = 0
            descriptor.inflight_io = 0
            descriptor.dirty = False
            descriptor.error = None
            descriptor.logical_mappings.clear()
            descriptor.state = PageState.FREE
            heapq.heappush(self._free, descriptor.page_id)
            self._release_count += 1
            return True

    def validate_invariants(self):
        """Raise with page-local diagnostics if ownership state is corrupt."""

        with self._lock:
            free_ids = set(self._free)
            if len(free_ids) != len(self._free):
                raise KVLifecycleError("free-list contains duplicate page IDs")
            for descriptor in self.descriptors:
                context = "page={} generation={}".format(
                    descriptor.page_id, descriptor.generation
                )
                if descriptor.ref_count < 0 or descriptor.pin_count < 0:
                    raise KVLifecycleError(context + " has a negative count")
                if (
                    descriptor.inflight_compute < 0
                    or descriptor.inflight_io < 0
                ):
                    raise KVLifecycleError(context + " has negative inflight work")
                if descriptor.pin_count != (
                    descriptor.inflight_compute + descriptor.inflight_io
                ):
                    raise KVLifecycleError(context + " pin/inflight counts diverged")
                is_free = descriptor.state == PageState.FREE
                if is_free != (descriptor.page_id in free_ids):
                    raise KVLifecycleError(context + " free-list state diverged")
                if is_free and any(
                    (
                        descriptor.ref_count,
                        descriptor.pin_count,
                        descriptor.inflight_compute,
                        descriptor.inflight_io,
                        len(descriptor.logical_mappings),
                    )
                ):
                    raise KVLifecycleError(context + " FREE invariant failed")
                if not is_free and descriptor.ref_count <= 0:
                    raise KVLifecycleError(context + " allocated page has no owner")
                if (
                    descriptor.index_version > descriptor.data_version
                    and descriptor.index_metadata_handle is not None
                ):
                    raise KVLifecycleError(context + " index is newer than data")
            return True

    def wait_quiescent(self, timeout_seconds=5.0):
        deadline = time.monotonic() + float(timeout_seconds)
        while True:
            with self._lock:
                busy = [
                    (
                        item.page_id,
                        item.generation,
                        item.pin_count,
                        item.inflight_compute,
                        item.inflight_io,
                        item.state.value,
                    )
                    for item in self.descriptors
                    if item.pin_count
                    or item.inflight_compute
                    or item.inflight_io
                    or item.state in _BUSY_STATES
                ]
            if not busy:
                return True
            if time.monotonic() >= deadline:
                raise KVLifecycleError(
                    "KV quiesce timed out after {:.3f}s; busy={}".format(
                        float(timeout_seconds), busy
                    )
                )
            time.sleep(0.001)

    def state_counts(self):
        result = {state.value: 0 for state in PageState}
        with self._lock:
            for descriptor in self.descriptors:
                result[descriptor.state.value] += 1
        return result

    def profile(self):
        allocated = [
            item for item in self.descriptors if item.state != PageState.FREE
        ]
        return {
            "total_pages": self.page_count,
            "free_pages": self.free_pages,
            "allocated_pages": self.allocated_pages,
            "peak_allocated_pages": self.peak_allocated_pages,
            "reserved_free_pages": self.reserved_free_pages,
            "shared_pages": sum(
                item.state == PageState.SHARED for item in self.descriptors
            ),
            "allocation_count": self._allocation_count,
            "release_count": self._release_count,
            "total_ref_count": sum(item.ref_count for item in allocated),
            "max_ref_count": max(
                (item.ref_count for item in allocated), default=0
            ),
            "total_pin_count": sum(item.pin_count for item in allocated),
            "max_pin_count": max(
                (item.pin_count for item in allocated), default=0
            ),
            "inflight_compute": sum(
                item.inflight_compute for item in allocated
            ),
            "inflight_io": sum(item.inflight_io for item in allocated),
            "dirty_pages": sum(item.dirty for item in allocated),
            "state_counts": self.state_counts(),
        }
