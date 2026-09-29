"""Selection-to-residency coordinator with paired compute pins."""

from dataclasses import dataclass
import threading
import time

from .errors import KVLifecycleError
from .stores.tiered import KVTier, PrefetchCancelled


@dataclass
class AttentionKVView:
    store: object
    logical_block_ids: tuple
    payloads: tuple
    closed: bool = False

    def close(self):
        if self.closed:
            return
        for logical_block_id in reversed(self.logical_block_ids):
            self.store.unpin(logical_block_id)
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise KVLifecycleError("attention KV view is already closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class TieredAttentionCoordinator:
    """FIFO prefetch admission followed by strict compute-pin pairing."""

    def __init__(self, store, target_tier=KVTier.GPU):
        self.store = store
        self.target_tier = KVTier(target_tier)
        self._ticket = 0
        self._served = 0
        self._abandoned = set()
        self._condition = threading.Condition()
        self.requests = 0
        self.cancelled = 0
        self.compute_failures = 0
        self.wait_ms = 0.0
        self._groups = {}
        self._groups_lock = threading.RLock()

    def _enter_fifo(self, cancel_event):
        with self._condition:
            ticket = self._ticket
            self._ticket += 1
            while ticket != self._served:
                if cancel_event is not None and cancel_event.is_set():
                    self._abandoned.add(ticket)
                    self._condition.notify_all()
                    raise PrefetchCancelled(
                        "request cancelled while waiting for fair admission"
                    )
                self._condition.wait(timeout=0.01)
            return ticket

    def _leave_fifo(self):
        with self._condition:
            self._served += 1
            while self._served in self._abandoned:
                self._abandoned.remove(self._served)
                self._served += 1
            self._condition.notify_all()

    def cancel_request(self, request_id, timeout=5.0):
        """Cancel and synchronously quiesce one request's prefetch group."""
        with self._groups_lock:
            group = self._groups.get(request_id)
        if group is None:
            return False
        group.cleanup(timeout=timeout)
        return True

    def prepare(
        self,
        logical_block_ids,
        cancel_event=None,
        timeout=10.0,
        request_id=None,
    ):
        logical_block_ids = tuple(logical_block_ids)
        if len(set(str(item) for item in logical_block_ids)) != len(
            logical_block_ids
        ):
            raise ValueError("selected KV blocks must be unique")
        self.requests += 1
        started = time.perf_counter()
        ticket = self._enter_fifo(cancel_event)
        if request_id is None:
            request_id = "tier-attention-{}".format(ticket)
        group = self.store.prefetch_group(
            request_id=request_id,
            cleanup_timeout=max(1.0, float(timeout)),
        )
        with self._groups_lock:
            if request_id in self._groups:
                self._leave_fifo()
                raise KVLifecycleError(
                    "request {!r} already has an active prefetch group".format(
                        request_id
                    )
                )
            self._groups[request_id] = group
        try:
            for logical_block_id in logical_block_ids:
                if cancel_event is not None and cancel_event.is_set():
                    raise PrefetchCancelled(
                        "request cancelled before prefetch submission"
                    )
                if not self.store.is_resident(
                    logical_block_id, self.target_tier
                ):
                    group.add(
                        self.store.prefetch(
                            logical_block_id,
                            self.target_tier,
                            request_id=request_id,
                        )
                    )
            group.wait(timeout=timeout, cancel_event=cancel_event)
            pinned = []
            try:
                for logical_block_id in logical_block_ids:
                    self.store.pin(logical_block_id)
                    pinned.append(logical_block_id)
                payloads = tuple(
                    self.store.get(logical_block_id, self.target_tier)
                    for logical_block_id in logical_block_ids
                )
            except BaseException:
                for logical_block_id in reversed(pinned):
                    self.store.unpin(logical_block_id)
                raise
            return AttentionKVView(
                store=self.store,
                logical_block_ids=logical_block_ids,
                payloads=payloads,
            )
        except BaseException as error:
            if isinstance(error, PrefetchCancelled):
                self.cancelled += 1
            # ``wait`` already performs cancellation and bounded quiescence;
            # this also covers failures before wait was entered.
            try:
                group.cleanup(timeout=max(1.0, float(timeout)))
            except BaseException as cleanup_error:
                # Cleanup diagnostics are useful, but must not replace the
                # request's original cancellation/copy/checksum error.
                try:
                    error.kv_cleanup_error = cleanup_error
                except BaseException:
                    pass
            raise
        finally:
            with self._groups_lock:
                self._groups.pop(request_id, None)
            self.wait_ms += (time.perf_counter() - started) * 1000.0
            self._leave_fifo()

    def execute(
        self,
        selection_result,
        compute,
        cancel_event=None,
        timeout=10.0,
        request_id=None,
    ):
        block_ids = getattr(
            selection_result, "logical_block_ids", selection_result
        )
        with self.prepare(
            block_ids,
            cancel_event=cancel_event,
            timeout=timeout,
            request_id=request_id,
        ) as view:
            try:
                return compute(view)
            except BaseException:
                self.compute_failures += 1
                raise

    def stats(self):
        with self._groups_lock:
            active_groups = len(self._groups)
        return {
            "requests": self.requests,
            "served": self._served,
            "cancelled": self.cancelled,
            "compute_failures": self.compute_failures,
            "wait_ms": self.wait_ms,
            "active_prefetch_groups": active_groups,
        }
