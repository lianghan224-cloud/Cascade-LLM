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

    def prepare(self, logical_block_ids, cancel_event=None, timeout=10.0):
        logical_block_ids = tuple(logical_block_ids)
        if len(set(str(item) for item in logical_block_ids)) != len(
            logical_block_ids
        ):
            raise ValueError("selected KV blocks must be unique")
        self.requests += 1
        started = time.perf_counter()
        self._enter_fifo(cancel_event)
        futures = []
        try:
            for logical_block_id in logical_block_ids:
                if cancel_event is not None and cancel_event.is_set():
                    raise PrefetchCancelled(
                        "request cancelled before prefetch submission"
                    )
                if not self.store.is_resident(
                    logical_block_id, self.target_tier
                ):
                    futures.append(
                        (
                            logical_block_id,
                            self.store.prefetch(
                                logical_block_id, self.target_tier
                            ),
                        )
                    )
            deadline = time.monotonic() + float(timeout)
            for logical_block_id, future in futures:
                if cancel_event is not None and cancel_event.is_set():
                    for pending_id, _ in futures:
                        self.store.cancel_prefetch(
                            pending_id, self.target_tier
                        )
                    raise PrefetchCancelled(
                        "request cancelled while prefetch was pending"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "KV prefetch timeout for {}".format(logical_block_id)
                    )
                future.result(timeout=remaining)
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
        except PrefetchCancelled:
            self.cancelled += 1
            raise
        finally:
            self.wait_ms += (time.perf_counter() - started) * 1000.0
            self._leave_fifo()

    def execute(
        self,
        selection_result,
        compute,
        cancel_event=None,
        timeout=10.0,
    ):
        block_ids = getattr(
            selection_result, "logical_block_ids", selection_result
        )
        with self.prepare(
            block_ids, cancel_event=cancel_event, timeout=timeout
        ) as view:
            try:
                return compute(view)
            except BaseException:
                self.compute_failures += 1
                raise

    def stats(self):
        return {
            "requests": self.requests,
            "served": self._served,
            "cancelled": self.cancelled,
            "compute_failures": self.compute_failures,
            "wait_ms": self.wait_ms,
        }
