"""Exclusive owner of page references, pins, Fork and Copy-on-Write."""

import math
import time

import torch

from .errors import KVCapacityError, KVLifecycleError
from .request_state import PendingAppend
from .types import PageState, RequestLifecycleState


class OwnershipManager:
    """Manage page lifetime without delegating policy to a GPU provider.

    Providers may copy page bytes, but this object alone decides when a page
    is allocated, retained, pinned, shared, released or replaced by COW.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.page_pool = runtime.page_pool
        self.device = runtime.device
        self.layer_count = runtime.layer_count
        self.page_size = runtime.page_size
        self.attention_events = (
            [torch.cuda.Event(enable_timing=False) for _ in range(self.layer_count)]
            if self.device.type == "cuda"
            else []
        )
        self.attention_pins = [[] for _ in range(self.layer_count)]
        self.append_events = (
            [torch.cuda.Event(enable_timing=False) for _ in range(self.layer_count)]
            if self.device.type == "cuda"
            else []
        )
        self.append_pins = [[] for _ in range(self.layer_count)]

    @property
    def event_count(self):
        return len(self.attention_events) + len(self.append_events)

    @staticmethod
    def _wait_event(event, timeout_seconds, label):
        deadline = time.monotonic() + float(timeout_seconds)
        while not event.query():
            if time.monotonic() >= deadline:
                raise KVLifecycleError(
                    "KV quiesce timed out waiting for {} after {:.3f}s".format(
                        label, float(timeout_seconds)
                    )
                )
            time.sleep(0.001)

    def drain_layer_attention(self, layer, timeout_seconds=5.0):
        if self.device.type != "cuda":
            return
        layer = int(layer)
        handles = self.attention_pins[layer]
        if not handles:
            return
        self._wait_event(
            self.attention_events[layer],
            timeout_seconds,
            "attention layer {}".format(layer),
        )
        for handle in reversed(handles):
            self.page_pool.unpin(handle)
        self.attention_pins[layer] = []

    def drain_layer_append(self, layer, timeout_seconds=5.0):
        if self.device.type != "cuda":
            return
        layer = int(layer)
        handles = self.append_pins[layer]
        if not handles:
            return
        self._wait_event(
            self.append_events[layer],
            timeout_seconds,
            "append layer {}".format(layer),
        )
        for handle in reversed(handles):
            self.page_pool.unpin(handle)
        self.append_pins[layer] = []

    def drain_handles(self, handles):
        identities = {item.identity() for item in handles}
        for layer, pinned in enumerate(self.append_pins):
            if any(item.identity() in identities for item in pinned):
                self.drain_layer_append(layer)
        for layer, pinned in enumerate(self.attention_pins):
            if any(item.identity() in identities for item in pinned):
                self.drain_layer_attention(layer)

    @staticmethod
    def append_target_handles(requests, pending, page_size):
        result = []
        seen = set()
        for state, transaction in zip(requests, pending):
            first_block = transaction.start // int(page_size)
            last_block = (transaction.end - 1) // int(page_size)
            for handle in state.block_table.handles[first_block : last_block + 1]:
                identity = handle.identity()
                if identity not in seen:
                    result.append(handle)
                    seen.add(identity)
        return result

    def begin_append_kernel(self, layer, requests, pending):
        if self.device.type != "cuda":
            return ()
        self.drain_layer_append(layer)
        handles = self.append_target_handles(
            requests,
            pending,
            self.page_size,
        )
        for handle in handles:
            self.page_pool.pin(handle)
        return tuple(handles)

    def abort_append_kernel(self, handles):
        for handle in reversed(tuple(handles)):
            self.page_pool.unpin(handle)

    def record_append_kernel(self, layer, handles):
        if self.device.type != "cuda":
            return
        self.append_events[int(layer)].record(torch.cuda.current_stream(self.device))
        self.append_pins[int(layer)] = list(handles)

    def begin_attention_kernel(self, layer, handles):
        if self.device.type != "cuda":
            return
        layer = int(layer)
        if self.append_pins[layer]:
            torch.cuda.current_stream(self.device).wait_event(
                self.append_events[layer]
            )
        self.drain_layer_attention(layer)
        for handle in handles:
            self.page_pool.pin(handle)
        self.attention_pins[layer] = list(handles)

    def abort_attention_kernel(self, layer, handles):
        if self.device.type != "cuda":
            return
        for handle in reversed(tuple(handles)):
            self.page_pool.unpin(handle)
        self.attention_pins[int(layer)] = []

    def record_attention_kernel(self, layer):
        if self.device.type == "cuda":
            self.attention_events[int(layer)].record(
                torch.cuda.current_stream(self.device)
            )

    def ensure_mutable_tail(self, state):
        if (
            not state.block_table.handles
            or state.sequence_length % self.page_size == 0
        ):
            return None, None
        logical_tail = state.sequence_length // self.page_size
        source = state.block_table.handles[logical_tail]
        descriptor = self.page_pool.descriptor(source)
        if descriptor.ref_count == 1 and descriptor.state != PageState.SHARED:
            self.page_pool.activate(
                source, state.sequence_length % self.page_size
            )
            return None, None
        self.drain_handles((source,))
        target = self.page_pool.allocate(owner_hint=state.request_id)
        valid = state.sequence_length % self.page_size
        self.page_pool.begin_copy(source, target)
        try:
            self.runtime.kv_kernel_backend.copy_pages(
                self.runtime.store,
                (source.page_id,),
                (target.page_id,),
                (valid,),
            )
        except BaseException:
            self.page_pool.end_copy(source, target, valid)
            self.page_pool.release(target)
            raise
        self.page_pool.end_copy(source, target, valid)
        state.block_table.replace(logical_tail, target)
        self.page_pool.release(source)
        self.runtime._metrics.cow_count += 1
        return source, target

    def begin_append(self, state, token_count):
        state.ensure_active()
        if state.pending_append is not None:
            if state.pending_append.token_count != int(token_count):
                raise KVLifecycleError("append transaction token count changed")
            return state.pending_append
        token_count = int(token_count)
        if token_count <= 0:
            raise ValueError("append token count must be positive")
        end = state.sequence_length + token_count
        if end > state.block_table.max_length:
            raise KVCapacityError("append exceeds request max_length")
        original_count = len(state.block_table.handles)
        original_lengths = tuple(state.layer_lengths)
        cow_original, cow_replacement = self.ensure_mutable_tail(state)
        required = int(math.ceil(end / float(self.page_size)))
        missing = required - len(state.block_table.handles)
        if not self.page_pool.can_allocate(missing):
            if cow_replacement is not None:
                self.page_pool.retain(cow_original)
                state.block_table.replace(original_count - 1, cow_original)
                self.page_pool.release(cow_replacement)
            raise KVCapacityError(
                "append needs {} new pages, only {} are admissible".format(
                    missing,
                    self.page_pool.free_pages
                    - self.page_pool.reserved_free_pages,
                )
            )
        allocated = []
        try:
            while len(state.block_table.handles) < required:
                handle = self.page_pool.allocate(owner_hint=state.request_id)
                state.block_table.append(handle)
                allocated.append(handle)
        except BaseException:
            for handle in reversed(allocated):
                state.block_table.truncate(len(state.block_table.handles) - 1)
                self.page_pool.release(handle)
            if cow_replacement is not None:
                self.page_pool.retain(cow_original)
                state.block_table.replace(original_count - 1, cow_original)
                self.page_pool.release(cow_replacement)
            raise
        pages = []
        offsets = []
        for position in range(state.sequence_length, end):
            logical = position // self.page_size
            pages.append(state.block_table.handles[logical].page_id)
            offsets.append(position % self.page_size)
        pending = PendingAppend(
            start=state.sequence_length,
            token_count=token_count,
            slot_page_ids=torch.tensor(
                pages, dtype=torch.int32, device=self.device
            ),
            slot_offsets=torch.tensor(
                offsets, dtype=torch.int32, device=self.device
            ),
            original_block_count=original_count,
            original_layer_lengths=original_lengths,
            allocated_handles=allocated,
            cow_original=cow_original,
            cow_replacement=cow_replacement,
        )
        state.pending_append = pending
        return pending

    def abort_append(self, state):
        pending = state.pending_append
        if pending is None:
            return
        for handle in reversed(pending.allocated_handles):
            if state.block_table.handles and state.block_table.handles[-1] == handle:
                state.block_table.truncate(len(state.block_table.handles) - 1)
            self.page_pool.release(handle)
        if pending.cow_replacement is not None:
            self.page_pool.retain(pending.cow_original)
            state.block_table.replace(
                pending.original_block_count - 1,
                pending.cow_original,
            )
            self.page_pool.release(pending.cow_replacement)
        elif pending.original_block_count and state.sequence_length:
            tail = state.block_table.handles[pending.original_block_count - 1]
            valid = state.sequence_length % self.page_size or self.page_size
            self.page_pool.seal(tail, valid)
        state.layer_lengths[:] = list(pending.original_layer_lengths)
        state.pending_append = None

    def commit(self, state):
        pending = state.pending_append
        if pending is None:
            return state
        if len(pending.completed_layers) != self.layer_count:
            raise KVLifecycleError("cannot commit an incomplete KV append")
        if any(length != pending.end for length in state.layer_lengths):
            raise KVLifecycleError("layer KV lengths diverged")
        state.sequence_length = pending.end
        self.runtime._metrics.committed_tokens += pending.token_count
        state.tail_valid_tokens = (
            state.sequence_length % self.page_size or self.page_size
        )
        for logical, handle in enumerate(state.block_table.handles):
            valid = min(
                self.page_size,
                max(0, state.sequence_length - logical * self.page_size),
            )
            if valid:
                self.page_pool.seal(handle, valid)
        first_changed = pending.start // self.page_size
        last_changed = (pending.end - 1) // self.page_size
        next_version = state.version + 1
        for handle in state.block_table.handles[first_changed : last_changed + 1]:
            self.page_pool.mark_data_updated(handle, version=next_version)
        state.pending_append = None
        state.version = next_version
        return state

    def fork(self, state, request_id=None, max_length=None, branch=True):
        state.ensure_active()
        if state.pending_append is not None:
            raise KVLifecycleError("cannot fork during append transaction")
        child = self.runtime.create_request(
            max_length=(
                state.block_table.max_length
                if max_length is None
                else max_length
            ),
            request_id=request_id,
            reuse_namespace=state.reuse_namespace,
            lifecycle_state=(
                RequestLifecycleState.BRANCH
                if branch
                else RequestLifecycleState.ACTIVE
            ),
        )
        if child.block_table.max_length < state.sequence_length:
            self.runtime.request_table.pop(child.request_id, None)
            raise ValueError("fork max_length is shorter than prefix")
        try:
            for handle in state.block_table.handles:
                descriptor = self.page_pool.descriptor(handle)
                self.page_pool.seal(handle, descriptor.valid_tokens)
                self.page_pool.retain(handle)
                child.block_table.append(handle)
            child.sequence_length = state.sequence_length
            child.tail_valid_tokens = state.tail_valid_tokens
            child.layer_lengths[:] = list(state.layer_lengths)
            child.parent_request_id = state.request_id
            child.fork_position = state.sequence_length
            child.version = state.version
            if hasattr(self.runtime.selection, "fork_request"):
                self.runtime.selection.fork_request(state, child)
            self.runtime._metrics.fork_count += 1
            return child
        except BaseException:
            self.release(child)
            raise

    def commit_branch(self, parent, branch):
        if branch.parent_request_id != parent.request_id:
            raise KVLifecycleError("branch does not belong to parent")
        if branch.pending_append is not None or parent.pending_append is not None:
            raise KVLifecycleError("cannot commit branch during append")
        self.drain_handles(parent.block_table.handles)
        self.page_pool.assert_releasable(
            parent.block_table.handles, "commit branch"
        )
        for handle in reversed(parent.block_table.handles):
            self.page_pool.release(handle)
        parent.block_table.handles[:] = branch.block_table.handles
        parent.block_table.version += 1
        parent.sequence_length = branch.sequence_length
        parent.tail_valid_tokens = branch.tail_valid_tokens
        parent.layer_lengths[:] = list(branch.layer_lengths)
        parent.version += 1
        if hasattr(self.runtime.selection, "commit_branch"):
            self.runtime.selection.commit_branch(parent, branch)
        branch.block_table.handles[:] = []
        branch.lifecycle_state = RequestLifecycleState.RELEASED
        self.runtime.request_table.pop(branch.request_id, None)
        return parent

    def reset(self, state):
        state.ensure_active()
        if state.pending_append is not None:
            self.abort_append(state)
        self.drain_handles(state.block_table.handles)
        self.page_pool.assert_releasable(
            state.block_table.handles, "reset request"
        )
        for handle in reversed(state.block_table.handles):
            self.page_pool.release(handle)
        state.block_table.handles[:] = []
        state.block_table.version += 1
        state.sequence_length = 0
        state.tail_valid_tokens = 0
        state.layer_lengths[:] = [0] * self.layer_count
        state.version += 1
        if hasattr(self.runtime.selection, "release_request"):
            self.runtime.selection.release_request(state)

    def rollback(self, state, target_length, target_version=None):
        """Rollback a committed/speculative suffix, including page boundaries."""

        state.ensure_active()
        target_length = int(target_length)
        if target_length < 0 or target_length > state.sequence_length:
            raise ValueError("rollback target is outside the committed sequence")
        if state.pending_append is not None:
            self.abort_append(state)
        next_version = (
            state.version + 1 if target_version is None else int(target_version)
        )
        if next_version <= state.version:
            raise ValueError("rollback target_version must advance monotonically")
        self.drain_handles(state.block_table.handles)
        required = int(math.ceil(target_length / float(self.page_size)))
        removed = tuple(state.block_table.handles[required:])
        self.page_pool.assert_releasable(removed, "rollback suffix")
        changed_block = None
        if required and target_length % self.page_size:
            changed_block = required - 1
            source = state.block_table.handles[changed_block]
            descriptor = self.page_pool.descriptor(source)
            if descriptor.ref_count > 1:
                target = self.page_pool.allocate(owner_hint=state.request_id)
                valid = target_length % self.page_size
                self.page_pool.begin_copy(source, target)
                try:
                    self.runtime.kv_kernel_backend.copy_pages(
                        self.runtime.store,
                        (source.page_id,),
                        (target.page_id,),
                        (valid,),
                    )
                except BaseException:
                    self.page_pool.end_copy(source, target, valid)
                    self.page_pool.release(target)
                    raise
                self.page_pool.end_copy(source, target, valid)
                state.block_table.replace(changed_block, target)
                self.page_pool.release(source)
        state.block_table.truncate(required)
        for handle in reversed(removed):
            self.page_pool.release(handle)
        state.sequence_length = target_length
        state.tail_valid_tokens = (
            target_length % self.page_size or (self.page_size if target_length else 0)
        )
        state.layer_lengths[:] = [target_length] * self.layer_count
        if required:
            tail = state.block_table.handles[-1]
            descriptor = self.page_pool.descriptor(tail)
            if descriptor.ref_count == 1:
                self.page_pool.seal(tail, state.tail_valid_tokens)
            if changed_block is not None:
                self.page_pool.mark_data_updated(tail, version=next_version)
        state.version = next_version
        if hasattr(self.runtime.selection, "rollback_request"):
            self.runtime.selection.rollback_request(
                self.runtime, state, changed_block=changed_block
            )
        self.page_pool.validate_invariants()
        return state

    def release(self, state):
        if state.lifecycle_state == RequestLifecycleState.RELEASED:
            return
        if not self.runtime.request_table.owns(state):
            raise KVLifecycleError("request belongs to another KV runtime")
        if state.pending_append is not None:
            self.abort_append(state)
        self.drain_handles(state.block_table.handles)
        self.page_pool.assert_releasable(
            state.block_table.handles, "release request"
        )
        state.lifecycle_state = RequestLifecycleState.RELEASING
        for handle in reversed(state.block_table.handles):
            self.page_pool.release(handle)
        state.block_table.handles[:] = []
        state.lifecycle_state = RequestLifecycleState.RELEASED
        self.runtime.request_table.pop(state.request_id, None)
        if hasattr(self.runtime.selection, "release_request"):
            self.runtime.selection.release_request(state)
        self.runtime._metrics.release_count += 1

    def quiesce(self, timeout_seconds=5.0):
        for layer in range(self.layer_count):
            self.drain_layer_append(layer, timeout_seconds=timeout_seconds)
            self.drain_layer_attention(layer, timeout_seconds=timeout_seconds)
        self.page_pool.validate_invariants()

    def close(self):
        self.quiesce()
        self.attention_events = []
        self.attention_pins = []
        self.append_events = []
        self.append_pins = []
