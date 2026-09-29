"""Fenced Layer-Page copies between active GPU and pinned CPU stores."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import threading
import uuid

import torch

from ..errors import KVLifecycleError
from ..fence import KVOperationFence


@dataclass(frozen=True)
class TensorLayerPageKey:
    logical_key: str
    layer: int
    data_epoch: int


@dataclass
class TensorLayerLocationSet:
    key: TensorLayerPageKey
    page_handle: object
    valid_tokens: int
    authority: str = "gpu"
    gpu_resident: bool = True
    cpu_handle: object = None
    inflight_operation: object = None


class _ImmediateEvent:
    def query(self):
        return True

    def synchronize(self):
        return None


class PinnedCPUTensorMigration:
    """Location/Execution adapter; PagePool remains ownership authority."""

    def __init__(self, gpu_store, cpu_store, page_pool, event_factory=None):
        self.gpu_store = gpu_store
        self.cpu_store = cpu_store
        self.page_pool = page_pool
        self._event_factory = event_factory
        self._stream = (
            torch.cuda.Stream(device=gpu_store.device)
            if gpu_store.device.type == "cuda"
            else None
        )
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="cascade-kv-tensor-migration"
        )
        self._lock = threading.RLock()
        self._records = {}
        self._fences = {}
        self._closed = False
        self.h2d_bytes = 0
        self.d2h_bytes = 0
        self.authority_changes = 0
        self.failures = 0
        self.cancellations = 0

    @staticmethod
    def _logical_key(value):
        return str(value.key()) if hasattr(value, "key") else str(value)

    def _check_open(self):
        if self._closed:
            raise KVLifecycleError("tensor migration adapter is closed")

    def _event(self):
        if self._event_factory is not None:
            return self._event_factory()
        if self.gpu_store.device.type == "cuda":
            return torch.cuda.Event()
        return _ImmediateEvent()

    def register_gpu_authority(
        self, logical_block_id, layer, page_handle, valid_tokens, data_epoch
    ):
        key = TensorLayerPageKey(
            self._logical_key(logical_block_id), int(layer), int(data_epoch)
        )
        descriptor = self.page_pool.descriptor(page_handle)
        if descriptor.data_version != key.data_epoch:
            raise KVLifecycleError("GPU page/data epoch mismatch")
        if int(valid_tokens) < 0 or int(valid_tokens) > self.gpu_store.page_size:
            raise ValueError("valid_tokens is outside the layer page")
        with self._lock:
            self._check_open()
            if key in self._records:
                raise KVLifecycleError("Layer Page location is already registered")
            self._records[key] = TensorLayerLocationSet(
                key=key,
                page_handle=page_handle,
                valid_tokens=int(valid_tokens),
            )
        return key

    def record(self, key):
        with self._lock:
            try:
                return self._records[key]
            except KeyError:
                raise KeyError("unknown tensor Layer Page")

    def _begin(self, key, expected_authority, kind):
        with self._lock:
            self._check_open()
            record = self.record(key)
            if record.authority != expected_authority:
                raise KVLifecycleError(
                    "{} requires {} authority".format(kind, expected_authority)
                )
            if record.inflight_operation is not None:
                raise KVLifecycleError("Layer Page already has inflight migration")
            fence = KVOperationFence(
                operation_id="tensor-{}-{}".format(kind, uuid.uuid4().hex),
                request_id=None,
                kind=kind,
                source_handles=(record.page_handle,),
                submit_epoch=key.data_epoch,
            )
            record.inflight_operation = fence.operation_id
            self._fences[fence.operation_id] = fence
            self.page_pool.pin(record.page_handle, kind="io")
            return record, fence

    def _copy(self, target, source, event):
        if self._stream is None:
            target.copy_(source)
        else:
            with torch.cuda.stream(self._stream):
                target.copy_(source, non_blocking=True)
                event.record(self._stream)

    def migrate_d2h(self, key):
        record, fence = self._begin(key, "gpu", "d2h")
        reservation = None
        cpu_handle = None
        try:
            reservation = self.cpu_store.reserve()
            cpu_handle = self.cpu_store.allocate(
                reservation,
                logical_block_id=key.logical_key,
                layer=key.layer,
                data_epoch=key.data_epoch,
            )
            target_key, target_value = self.cpu_store.write_target(
                cpu_handle, expected_data_epoch=key.data_epoch
            )
            source_key = self.gpu_store.keys[key.layer, record.page_handle.page_id]
            source_value = self.gpu_store.values[key.layer, record.page_handle.page_id]
            event = self._event()
            self._copy(target_key, source_key, event)
            self._copy(target_value, source_value, event)
            future = self._executor.submit(
                self._finish_d2h, key, fence, event, cpu_handle
            )
            fence.io_future = future
            return fence
        except BaseException as error:
            if reservation is not None and cpu_handle is None:
                self.cpu_store.cancel_reservation(reservation)
            if cpu_handle is not None:
                self.cpu_store.release(cpu_handle)
            self._abort_submit(record, fence, error)
            raise

    def _finish_d2h(self, key, fence, event, cpu_handle):
        try:
            event.synchronize()
            if fence.cancelled:
                raise KVLifecycleError("D2H migration cancelled")
            page = self.cpu_store.commit_write(
                cpu_handle,
                valid_tokens=self.record(key).valid_tokens,
                data_epoch=key.data_epoch,
            )
            with self._lock:
                record = self.record(key)
                descriptor = self.page_pool.descriptor(record.page_handle)
                if descriptor.data_version != key.data_epoch:
                    raise KVLifecycleError("GPU source epoch changed during D2H")
                if fence.cancelled:
                    raise KVLifecycleError("D2H migration cancelled")
                record.cpu_handle = cpu_handle
                record.authority = "cpu"
                record.inflight_operation = None
                self.d2h_bytes += self.cpu_store.layer_page_bytes
                self.authority_changes += 1
                fence.mark_completed(key.data_epoch)
            return page
        except BaseException as error:
            self.failures += int(not fence.cancelled)
            self.cancellations += int(fence.cancelled)
            try:
                self.cpu_store.release(cpu_handle)
            except BaseException:
                pass
            if fence.cancelled:
                fence.status = "cancelled"
                fence.error = error
            else:
                fence.mark_failed(error, key.data_epoch)
            raise
        finally:
            self._finish_common(key, fence)

    def migrate_h2d(self, key):
        record, fence = self._begin(key, "cpu", "h2d")
        try:
            page = self.cpu_store.read(
                record.cpu_handle, expected_data_epoch=key.data_epoch
            )
            target_key = self.gpu_store.keys[key.layer, record.page_handle.page_id]
            target_value = self.gpu_store.values[key.layer, record.page_handle.page_id]
            event = self._event()
            self._copy(target_key, page.key, event)
            self._copy(target_value, page.value, event)
            future = self._executor.submit(self._finish_h2d, key, fence, event)
            fence.io_future = future
            return fence
        except BaseException as error:
            self._abort_submit(record, fence, error)
            raise

    def _finish_h2d(self, key, fence, event):
        try:
            event.synchronize()
            with self._lock:
                record = self.record(key)
                self.cpu_store.read(
                    record.cpu_handle, expected_data_epoch=key.data_epoch
                )
                if fence.cancelled:
                    raise KVLifecycleError("H2D migration cancelled")
                record.authority = "gpu"
                record.gpu_resident = True
                record.inflight_operation = None
                self.h2d_bytes += self.cpu_store.layer_page_bytes
                self.authority_changes += 1
                fence.mark_completed(key.data_epoch)
            return record
        except BaseException as error:
            self.failures += int(not fence.cancelled)
            self.cancellations += int(fence.cancelled)
            if fence.cancelled:
                fence.status = "cancelled"
                fence.error = error
            else:
                fence.mark_failed(error, key.data_epoch)
            raise
        finally:
            self._finish_common(key, fence)

    def _abort_submit(self, record, fence, error):
        with self._lock:
            record.inflight_operation = None
            self._fences.pop(fence.operation_id, None)
            fence.mark_failed(error)
        self.page_pool.unpin(record.page_handle, kind="io")

    def _finish_common(self, key, fence):
        with self._lock:
            record = self.record(key)
            if record.inflight_operation == fence.operation_id:
                record.inflight_operation = None
            self._fences.pop(fence.operation_id, None)
        self.page_pool.unpin(record.page_handle, kind="io")

    def stats(self):
        with self._lock:
            return {
                "records": len(self._records),
                "pending_fences": len(self._fences),
                "d2h_kv_bytes": self.d2h_bytes,
                "h2d_kv_bytes": self.h2d_bytes,
                "authority_changes": self.authority_changes,
                "migration_failures": self.failures,
                "migration_cancellations": self.cancellations,
            }

    def close(self):
        with self._lock:
            if self._closed:
                return
            fences = tuple(self._fences.values())
        for fence in fences:
            fence.cancel()
            try:
                fence.wait()
            except BaseException:
                pass
        self._executor.shutdown(wait=True)
        with self._lock:
            for record in self._records.values():
                if record.cpu_handle is not None:
                    self.cpu_store.release(record.cpu_handle)
            self._records.clear()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
