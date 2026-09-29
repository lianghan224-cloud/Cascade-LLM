"""Unified completion fence for asynchronous KV lifecycle operations."""

from dataclasses import dataclass, field
import time

from .errors import KVLifecycleError


@dataclass
class KVOperationFence:
    """Describe and bound one asynchronous KV operation.

    The fence owns no Page references itself.  OwnershipManager keeps the
    source/target handles pinned until ``wait`` succeeds or cleanup aborts the
    operation.
    """

    operation_id: str
    request_id: object
    kind: str
    source_handles: tuple = ()
    target_handles: tuple = ()
    cuda_event: object = field(default=None, repr=False)
    io_future: object = field(default=None, repr=False)
    status: str = "submitted"
    error: object = field(default=None, repr=False)
    cancelled: bool = False
    submit_epoch: int = 0
    completion_epoch: object = None

    def query(self):
        if self.status in {"completed", "failed", "cancelled"}:
            return True
        if self.cuda_event is not None:
            return bool(self.cuda_event.query())
        if self.io_future is not None:
            return bool(self.io_future.done())
        return True

    def wait(self, timeout_seconds=5.0):
        deadline = time.monotonic() + float(timeout_seconds)
        try:
            while not self.query():
                if time.monotonic() >= deadline:
                    raise KVLifecycleError(
                        "KV operation {} ({}) timed out after {:.3f}s".format(
                            self.operation_id, self.kind, float(timeout_seconds)
                        )
                    )
                time.sleep(0.001)
            # Query provides bounded waiting; synchronize is the final CUDA
            # completion/error observation point and is intentionally
            # injectable by the qualification harness.
            if self.cuda_event is not None and hasattr(
                self.cuda_event, "synchronize"
            ):
                self.cuda_event.synchronize()
        except KVLifecycleError:
            raise
        except BaseException as error:
            self.mark_failed(error)
            raise KVLifecycleError(
                "KV operation {} ({}) event wait failed".format(
                    self.operation_id, self.kind
                )
            ) from error
        if self.io_future is not None and not self.cancelled:
            try:
                self.io_future.result(timeout=0)
            except BaseException as error:
                self.mark_failed(error)
                raise
        if self.status == "failed":
            raise KVLifecycleError(
                "KV operation {} ({}) failed".format(
                    self.operation_id, self.kind
                )
            ) from self.error
        if self.cancelled:
            self.status = "cancelled"
            raise KVLifecycleError(
                "KV operation {} ({}) was cancelled".format(
                    self.operation_id, self.kind
                )
            )
        self.status = "completed"
        return self

    def mark_completed(self, completion_epoch=None):
        self.status = "completed"
        self.completion_epoch = completion_epoch
        return self

    def mark_failed(self, error, completion_epoch=None):
        self.status = "failed"
        self.error = error
        self.completion_epoch = completion_epoch
        return self

    def cancel(self):
        self.cancelled = True
        future = self.io_future
        if future is not None:
            future.cancel()
        if self.query():
            self.status = "cancelled"
        return self
