"""Bounded producer/H2D/compute orchestration shared by runtimes."""

from collections import deque
from dataclasses import dataclass
from concurrent.futures import TimeoutError as FutureTimeoutError
import queue
import threading
import time


_STOP = object()


class PipelineError(RuntimeError):
    pass


class PipelineWorkerError(PipelineError):
    def __init__(self, phase, error):
        self.phase = phase
        self.original_error = error
        super().__init__("{} failed: {}".format(phase, error))


@dataclass
class PipelineJob:
    job_id: int
    units: tuple
    cancel: threading.Event


@dataclass
class StageLease:
    slot_index: int
    reuse_event: object = None


@dataclass
class DeviceLease:
    slot_index: int
    reuse_event: object = None


@dataclass
class PreparedSource:
    job: PipelineJob
    index: int
    unit: object
    source: object
    stage_slot_index: int


@dataclass
class ReadyTransfer:
    job: PipelineJob
    index: int
    unit: object
    device_slot_index: int
    ready_event: object


@dataclass
class PipelineEnd:
    job: PipelineJob


@dataclass
class PipelineFailure:
    job: PipelineJob
    phase: str
    error: BaseException


def _queue_get(target, stop, cancel=None, timeout=0.1):
    while not stop.is_set() and (cancel is None or not cancel.is_set()):
        try:
            return target.get(timeout=timeout)
        except queue.Empty:
            continue
    raise PipelineError("pipeline operation was cancelled")


def _queue_put(target, item, stop, cancel=None, timeout=0.1):
    while not stop.is_set() and (cancel is None or not cancel.is_set()):
        try:
            target.put(item, timeout=timeout)
            return
        except queue.Full:
            continue
    raise PipelineError("pipeline operation was cancelled")


class SourceProducer:
    """Resolve CPU sources into a bounded queue using reusable stage slots."""

    def __init__(
        self,
        store,
        source_queue,
        staging_queue,
        stop_event,
        prefetch_depth,
        source_timeout_seconds,
    ):
        self.store = store
        self.source_queue = source_queue
        self.staging_queue = staging_queue
        self.stop_event = stop_event
        self.prefetch_depth = int(prefetch_depth)
        self.source_timeout_seconds = float(source_timeout_seconds)
        self.request_queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(
            target=self._worker,
            name="cascade-source-producer",
            daemon=True,
        )
        self.prepare_wait_ms = 0.0
        self.max_queue_depth = 0
        self.thread.start()

    def submit(self, job):
        _queue_put(self.request_queue, job, self.stop_event, job.cancel)

    def _publish(self, item, job, allow_cancel=True):
        _queue_put(
            self.source_queue,
            item,
            self.stop_event,
            job.cancel if allow_cancel else None,
        )
        self.max_queue_depth = max(
            self.max_queue_depth, self.source_queue.qsize()
        )

    def _produce(self, job):
        pending = deque()
        next_index = 0
        while next_index < len(job.units) or pending:
            while (
                next_index < len(job.units)
                and len(pending) < self.prefetch_depth
                and not job.cancel.is_set()
            ):
                lease = _queue_get(
                    self.staging_queue, self.stop_event, job.cancel
                )
                unit = job.units[next_index]
                future = self.store.prepare_unit(
                    unit,
                    lease.slot_index,
                    reuse_event=lease.reuse_event,
                )
                pending.append((next_index, unit, lease, future))
                next_index += 1
            if not pending:
                break
            index, unit, lease, future = pending.popleft()
            started = time.perf_counter()
            try:
                source = future.result(timeout=self.source_timeout_seconds)
            except FutureTimeoutError as error:
                raise TimeoutError(
                    "source unit {} exceeded {:.1f}s".format(
                        unit.unit_id, self.source_timeout_seconds
                    )
                ) from error
            self.prepare_wait_ms += (
                time.perf_counter() - started
            ) * 1000.0
            self._publish(
                PreparedSource(
                    job=job,
                    index=index,
                    unit=unit,
                    source=source,
                    stage_slot_index=lease.slot_index,
                ),
                job,
            )
        if not job.cancel.is_set():
            self._publish(PipelineEnd(job), job)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                request = self.request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is _STOP:
                return
            job = request
            try:
                self._produce(job)
            except BaseException as error:
                job.cancel.set()
                try:
                    self._publish(
                        PipelineFailure(job, "source_producer", error),
                        job,
                        allow_cancel=False,
                    )
                except PipelineError:
                    return

    def reset_stats(self):
        self.prepare_wait_ms = 0.0
        self.max_queue_depth = 0

    def close(self):
        try:
            self.request_queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self.thread.join(timeout=5.0)


class H2DScheduler:
    """Consume prepared sources and submit H2D into available device slots."""

    def __init__(
        self,
        source_queue,
        ready_queue,
        staging_queue,
        free_slot_queue,
        stop_event,
        submit_copy,
        staging_requires_event,
    ):
        self.source_queue = source_queue
        self.ready_queue = ready_queue
        self.staging_queue = staging_queue
        self.free_slot_queue = free_slot_queue
        self.stop_event = stop_event
        self.submit_copy = submit_copy
        self.staging_requires_event = bool(staging_requires_event)
        self.free_slot_wait_ms = 0.0
        self.max_ready_queue_depth = 0
        self.thread = threading.Thread(
            target=self._worker,
            name="cascade-h2d-scheduler",
            daemon=True,
        )
        self.thread.start()

    def _publish(self, item, job, allow_cancel=True):
        _queue_put(
            self.ready_queue,
            item,
            self.stop_event,
            job.cancel if allow_cancel else None,
        )
        self.max_ready_queue_depth = max(
            self.max_ready_queue_depth, self.ready_queue.qsize()
        )

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.source_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is _STOP:
                return
            if isinstance(item, PipelineFailure):
                try:
                    self._publish(item, item.job, allow_cancel=False)
                except PipelineError:
                    return
                continue
            if isinstance(item, PipelineEnd):
                try:
                    self._publish(item, item.job)
                except PipelineError:
                    pass
                continue
            prepared = item
            job = prepared.job
            lease = None
            try:
                started = time.perf_counter()
                lease = _queue_get(
                    self.free_slot_queue, self.stop_event, job.cancel
                )
                self.free_slot_wait_ms += (
                    time.perf_counter() - started
                ) * 1000.0
                ready_event = self.submit_copy(prepared, lease)
                stage_event = (
                    ready_event if self.staging_requires_event else None
                )
                _queue_put(
                    self.staging_queue,
                    StageLease(prepared.stage_slot_index, stage_event),
                    self.stop_event,
                    job.cancel,
                )
                self._publish(
                    ReadyTransfer(
                        job=job,
                        index=prepared.index,
                        unit=prepared.unit,
                        device_slot_index=lease.slot_index,
                        ready_event=ready_event,
                    ),
                    job,
                )
            except BaseException as error:
                job.cancel.set()
                if lease is not None:
                    try:
                        self.free_slot_queue.put_nowait(lease)
                    except queue.Full:
                        pass
                try:
                    self._publish(
                        PipelineFailure(job, "h2d_scheduler", error),
                        job,
                        allow_cancel=False,
                    )
                except PipelineError:
                    return

    def reset_stats(self):
        self.free_slot_wait_ms = 0.0
        self.max_ready_queue_depth = 0

    def close(self):
        try:
            self.source_queue.put_nowait(_STOP)
        except queue.Full:
            pass
        self.thread.join(timeout=5.0)


class ComputeConsumer:
    """Ordered main-thread consumer for model callbacks and slot release."""

    def __init__(self, ready_queue, free_slot_queue, stop_event):
        self.ready_queue = ready_queue
        self.free_slot_queue = free_slot_queue
        self.stop_event = stop_event
        self.ready_wait_ms = 0.0

    def next(self, job, timeout_seconds):
        deadline = time.monotonic() + timeout_seconds
        while not self.stop_event.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("ready queue timed out")
            try:
                item = self.ready_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if item.job.job_id != job.job_id:
                raise PipelineError("received an item from another pipeline job")
            if isinstance(item, PipelineFailure):
                raise PipelineWorkerError(item.phase, item.error) from item.error
            return item
        raise PipelineError("pipeline is closed")

    def release(self, ready, free_event):
        self.free_slot_queue.put(
            DeviceLease(ready.device_slot_index, free_event)
        )

    def reset_stats(self):
        self.ready_wait_ms = 0.0


class PipelineRuntimeCore:
    """Persistent bounded queues and workers for one runtime instance."""

    def __init__(
        self,
        plan,
        store,
        slot_count,
        prefetch_depth,
        source_timeout_seconds,
        submit_copy,
    ):
        self.plan = plan
        self.store = store
        self.slot_count = int(slot_count)
        self.prefetch_depth = int(prefetch_depth)
        if self.prefetch_depth < 1:
            raise ValueError("prefetch_depth must be positive")
        self.stop_event = threading.Event()
        self.source_queue = queue.Queue(maxsize=self.prefetch_depth)
        self.ready_queue = queue.Queue(maxsize=self.prefetch_depth)
        self.free_slot_queue = queue.Queue(maxsize=self.slot_count)
        for slot_index in range(self.slot_count):
            self.free_slot_queue.put(DeviceLease(slot_index))
        staging_count = int(
            getattr(store, "slot_count", self.prefetch_depth)
        )
        self.staging_queue = queue.Queue(maxsize=staging_count)
        for slot_index in range(staging_count):
            self.staging_queue.put(StageLease(slot_index))
        self.producer = SourceProducer(
            store=store,
            source_queue=self.source_queue,
            staging_queue=self.staging_queue,
            stop_event=self.stop_event,
            prefetch_depth=min(self.prefetch_depth, staging_count),
            source_timeout_seconds=source_timeout_seconds,
        )
        mode = getattr(getattr(store, "mode", None), "value", None)
        self.scheduler = H2DScheduler(
            source_queue=self.source_queue,
            ready_queue=self.ready_queue,
            staging_queue=self.staging_queue,
            free_slot_queue=self.free_slot_queue,
            stop_event=self.stop_event,
            submit_copy=submit_copy,
            staging_requires_event=(mode == "pinned_staging"),
        )
        self.consumer = ComputeConsumer(
            self.ready_queue, self.free_slot_queue, self.stop_event
        )
        self._job_counter = 0
        self._run_lock = threading.Lock()
        self._failed = False
        self._closed = False
        self.last_stats = {}

    def resource_stats(self):
        """Return a non-blocking snapshot of bounded queues and workers."""

        return {
            "closed": self._closed,
            "failed": self._failed,
            "source_queue_depth": self.source_queue.qsize(),
            "source_queue_capacity": self.source_queue.maxsize,
            "ready_queue_depth": self.ready_queue.qsize(),
            "ready_queue_capacity": self.ready_queue.maxsize,
            "free_slot_queue_depth": self.free_slot_queue.qsize(),
            "free_slot_queue_capacity": self.free_slot_queue.maxsize,
            "staging_queue_depth": self.staging_queue.qsize(),
            "staging_queue_capacity": self.staging_queue.maxsize,
            "producer_request_queue_depth": self.producer.request_queue.qsize(),
            "producer_alive": self.producer.thread.is_alive(),
            "scheduler_alive": self.scheduler.thread.is_alive(),
            "worker_thread_count": sum(
                (
                    self.producer.thread.is_alive(),
                    self.scheduler.thread.is_alive(),
                )
            ),
        }

    def run(self, consume, state, timeout_seconds, units=None):
        if self._closed:
            raise PipelineError("pipeline runtime is closed")
        if self._failed:
            raise PipelineError("pipeline runtime cannot be reused after failure")
        if not self._run_lock.acquire(blocking=False):
            raise PipelineError("concurrent run calls are unsupported")
        self._job_counter += 1
        job = PipelineJob(
            self._job_counter,
            tuple(self.plan.units if units is None else units),
            threading.Event(),
        )
        self.producer.reset_stats()
        self.scheduler.reset_stats()
        self.consumer.reset_stats()
        started = time.perf_counter()
        last_free_event = None
        try:
            self.producer.submit(job)
            while True:
                wait_started = time.perf_counter()
                item = self.consumer.next(job, timeout_seconds)
                self.consumer.ready_wait_ms += (
                    time.perf_counter() - wait_started
                ) * 1000.0
                if isinstance(item, PipelineEnd):
                    break
                try:
                    state, free_event = consume(item, state)
                except BaseException:
                    job.cancel.set()
                    raise
                self.consumer.release(item, free_event)
                last_free_event = free_event
            self.last_stats = {
                "pipeline_host_wall_ms": (
                    time.perf_counter() - started
                ) * 1000.0,
                "source_prepare_wait_ms": self.producer.prepare_wait_ms,
                "ready_wait_ms": self.consumer.ready_wait_ms,
                "free_slot_wait_ms": self.scheduler.free_slot_wait_ms,
                "source_queue_max_depth": self.producer.max_queue_depth,
                "ready_queue_max_depth": self.scheduler.max_ready_queue_depth,
                "source_queue_capacity": self.source_queue.maxsize,
                "ready_queue_capacity": self.ready_queue.maxsize,
            }
            return state, last_free_event
        except BaseException:
            job.cancel.set()
            self._failed = True
            raise
        finally:
            self._run_lock.release()

    def close(self):
        if self._closed:
            return
        self.stop_event.set()
        self.producer.close()
        self.scheduler.close()
        self._closed = True
