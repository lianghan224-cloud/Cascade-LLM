from types import SimpleNamespace
import threading
import unittest

import torch

from layer_streaming import (
    GenerationSession,
    GenerationSessionState,
    SamplingConfig,
)
from layer_streaming.kv.api import RequestKVCacheV1
from layer_streaming.kv.runtime import PagedKVRuntime
from layer_streaming.kv.stores import ResidencyState
from layer_streaming.kv_policy import KVPolicy


def _cpu_factory(shape, **kwargs):
    return torch.empty(shape, dtype=kwargs["dtype"], device="cpu")


class _ImmediateEvent:
    def query(self):
        return True

    def synchronize(self):
        return None


class _BlockingEvent:
    def __init__(self):
        self.started = threading.Event()
        self.released = threading.Event()

    def query(self):
        return self.released.is_set()

    def synchronize(self):
        self.started.set()
        if not self.released.wait(timeout=5.0):
            raise TimeoutError("test did not release blocked tier copy")


class _OneShotBlockingEventFactory:
    def __init__(self):
        self.blocking = _BlockingEvent()
        self._used = False

    def __call__(self):
        if not self._used:
            self._used = True
            return self.blocking
        return _ImmediateEvent()


class _TierExecutor:
    """Small deterministic executor that exercises the real Request KV API."""

    def __init__(self, kv_cache, layer_count, query_heads, kv_heads, head_dim):
        self.kv_cache = kv_cache
        self.layer_count = int(layer_count)
        self.query_heads = int(query_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.finish_calls = 0

    def begin(self, input_ids):
        return SimpleNamespace(input_ids=input_ids)

    def run_step(self, state):
        count = int(state.input_ids.shape[1])
        start = self.kv_cache.sequence_length()
        positions = torch.arange(start, start + count).reshape(1, -1)
        for layer in range(self.layer_count):
            values = torch.arange(
                start * self.head_dim,
                (start + count) * self.head_dim,
                dtype=torch.float16,
            ).reshape(1, count, self.kv_heads, self.head_dim)
            values = values.transpose(1, 2).contiguous()
            key = values.mul(0.001).add_(float(layer + 1))
            value = values.mul(0.002).add_(float(layer + 10))
            self.kv_cache.append_only(layer, key, value)
            query = torch.ones(
                (1, self.query_heads, count, self.head_dim),
                dtype=torch.float16,
            ).mul_(float(layer + 1))
            self.kv_cache.attend(
                layer,
                query,
                kv_groups=self.query_heads // self.kv_heads,
                position_ids=positions,
            )
        return state

    def finish(self, state):
        self.finish_calls += 1
        token = 100 + self.finish_calls
        state.topk_values = torch.tensor([[[3.0, 2.0]]])
        state.topk_indices = torch.tensor([[[token, token + 1]]])
        return state


class _TierModelRuntime:
    def run(self, executor, state):
        return executor.run_step(state)


class GenerationSessionActiveTierTest(unittest.TestCase):
    layers = 2
    query_heads = 2
    kv_heads = 1
    head_dim = 4
    page_size = 16
    logical_pages = 16
    hot_pages = 4

    @property
    def logical_page_bytes(self):
        return (
            2
            * self.layers
            * self.kv_heads
            * self.page_size
            * self.head_dim
            * torch.empty((), dtype=torch.float16).element_size()
        )

    def make_session(self, selection="none"):
        sparse = selection == "rgkv"
        policy = KVPolicy(
            accuracy="sparse" if sparse else "exact",
            storage="gpu_cpu",
            dtype="fp16",
            selection=selection,
            reuse="request_only",
            attention_backend="reference_paged_exact",
            page_size=self.page_size,
            page_budget=2 if sparse else 0,
            recent_window=self.page_size if sparse else 0,
            rgkv_scorer="torch_tensorized" if sparse else "cpu_reference",
            gpu_hot_budget_bytes=self.hot_pages * self.logical_page_bytes,
            cpu_budget_bytes=self.logical_pages * self.logical_page_bytes,
            gpu_high_watermark_bytes=(
                self.hot_pages * self.logical_page_bytes
            ),
            gpu_low_watermark_bytes=(
                (self.hot_pages - 1) * self.logical_page_bytes
            ),
        )
        runtime = PagedKVRuntime(
            layer_count=self.layers,
            num_query_heads=self.query_heads,
            num_kv_heads=self.kv_heads,
            head_dim=self.head_dim,
            page_count=self.logical_pages,
            page_size=self.page_size,
            dtype=torch.float16,
            device="cpu",
            policy=policy,
            allow_reference=True,
            prefetch_timeout_seconds=2.0,
            tier_tensor_factory=_cpu_factory,
        )
        state = runtime.create_request(self.logical_pages * self.page_size)
        cache = RequestKVCacheV1(runtime, state)
        executor = _TierExecutor(
            cache,
            self.layers,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
        )
        session = GenerationSession(
            executor,
            _TierModelRuntime(),
            SamplingConfig(top_k=1, max_new_tokens=4),
            eos_token_ids=(),
        )
        return session, runtime, state

    def _prefill_with_cpu_residence(self, session, runtime, token_count=63):
        session.prefill(torch.arange(token_count))
        stats = runtime.profile_stats()
        self.assertGreater(stats["cpu_kv_used_bytes"], 0)
        self.assertGreater(stats["eviction_count"], 0)
        return stats

    def _make_cpu_only_page_and_free_slot(self, runtime, state):
        for handle in state.block_table.handles:
            key = runtime._tier_registered_key(handle)
            locations = runtime.store.location_sets(key)
            if all(
                item.gpu_state == ResidencyState.RESIDENT
                and item.cpu_state == ResidencyState.RESIDENT
                for item in locations
            ):
                runtime.active_tier.evict(key).wait(timeout_seconds=2.0)
                return key
        self.fail("fixture has no GPU+CPU page that can be made CPU-only")

    def _make_rgkv_selected_cpu_only_page(self, runtime, state):
        query = torch.ones(
            (1, self.query_heads, self.head_dim), dtype=torch.float16
        )
        batch = runtime.prepare_batch((state,), (1,), 0)
        selected = runtime.selection.select((state,), 0, query, batch)
        selected_ids = tuple(int(item) for item in selected.logical_block_ids.tolist())
        for logical in selected_ids:
            handle = state.block_table.handles[logical]
            key = runtime._tier_registered_key(handle)
            locations = runtime.store.location_sets(key)
            if any(item.gpu_slot is not None for item in locations):
                runtime.active_tier.evict(key).wait(timeout_seconds=2.0)
                return key
        self.fail("fixture has no selected GPU page that can be made CPU-only")

    @staticmethod
    def _start_call(callable_):
        result = {"value": None, "error": None}

        def target():
            try:
                result["value"] = callable_()
            except BaseException as error:
                result["error"] = error

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread, result

    def _start_blocked_decode(self, session, runtime, state, *, rgkv=False):
        # The first sampled token does not require another model step.  The
        # second Decode appends it and attends over the evicted historical page.
        session.decode_one()
        if rgkv:
            self._make_rgkv_selected_cpu_only_page(runtime, state)
        else:
            self._make_cpu_only_page_and_free_slot(runtime, state)
        events = _OneShotBlockingEventFactory()
        runtime.active_tier._event_factory = events
        thread, result = self._start_call(session.decode_one)
        self.assertTrue(events.blocking.started.wait(timeout=2.0))
        self.assertTrue(thread.is_alive())
        return thread, result, events.blocking

    def test_prefill_evicts_and_decode_prefetches_transparently(self):
        session, runtime, _ = self.make_session()
        before = self._prefill_with_cpu_residence(session, runtime)
        session.decode_one()
        session.decode_one()
        after = runtime.profile_stats()
        self.assertEqual(session.state, GenerationSessionState.DECODING)
        self.assertGreater(after["prefetch_count"], before["prefetch_count"])
        self.assertGreater(after["h2d_kv_bytes"], before["h2d_kv_bytes"])
        self.assertGreater(after["d2h_kv_bytes"], 0)
        self.assertEqual(after["pending_tier_operations"], 0)
        self.assertEqual(after["total_ref_count"], after["logical_owner_count"])
        session.close()
        runtime.close()

    def test_continuation_after_eviction_is_transparent(self):
        session, runtime, _ = self.make_session()
        before = self._prefill_with_cpu_residence(session, runtime)
        session.continue_prefill([70, 71])
        after = runtime.profile_stats()
        self.assertEqual(session.token_history[-2:], [70, 71])
        self.assertEqual(session.kv_cache.sequence_length(), 65)
        self.assertGreater(after["prefetch_count"], before["prefetch_count"])
        self.assertEqual(after["pending_tier_operations"], 0)
        session.close()
        runtime.close()

    def test_reset_reclaims_gpu_and_cpu_resident_pages(self):
        session, runtime, _ = self.make_session()
        self._prefill_with_cpu_residence(session, runtime)
        session.reset()
        stats = runtime.profile_stats()
        self.assertEqual(session.state, GenerationSessionState.NEW)
        self.assertEqual(stats["gpu_kv_used_pages"], 0)
        self.assertEqual(stats["cpu_kv_used_bytes"], 0)
        self.assertEqual(stats["kv_pool_allocated_pages"], 0)
        self.assertEqual(stats["pending_tier_operations"], 0)
        runtime.close()

    def test_cancel_waits_for_pending_prefetch_then_reclaims_request(self):
        session, runtime, state = self.make_session()
        self._prefill_with_cpu_residence(session, runtime)
        decode_thread, decode_result, blocking = self._start_blocked_decode(
            session, runtime, state
        )
        cancel_thread, cancel_result = self._start_call(session.cancel)
        self.assertTrue(cancel_thread.is_alive())
        blocking.released.set()
        decode_thread.join(timeout=3.0)
        cancel_thread.join(timeout=3.0)
        self.assertFalse(decode_thread.is_alive())
        self.assertFalse(cancel_thread.is_alive())
        self.assertIsNone(decode_result["error"])
        self.assertIsNone(cancel_result["error"])
        self.assertEqual(session.state, GenerationSessionState.CANCELLED)
        stats = runtime.profile_stats()
        self.assertEqual(stats["pending_tier_operations"], 0)
        self.assertEqual(stats["kv_pool_allocated_pages"], 0)
        self.assertEqual(stats["gpu_kv_used_pages"], 0)
        self.assertEqual(stats["cpu_kv_used_bytes"], 0)
        runtime.close()

    def test_rgkv_selected_cpu_page_cancel_during_h2d_closes_resources(self):
        session, runtime, state = self.make_session(selection="rgkv")
        self._prefill_with_cpu_residence(session, runtime)
        decode_thread, decode_result, blocking = self._start_blocked_decode(
            session, runtime, state, rgkv=True
        )
        cancel_thread, cancel_result = self._start_call(session.cancel)
        self.assertTrue(cancel_thread.is_alive())
        blocking.released.set()
        decode_thread.join(timeout=3.0)
        cancel_thread.join(timeout=3.0)
        self.assertFalse(decode_thread.is_alive())
        self.assertFalse(cancel_thread.is_alive())
        self.assertIsNone(decode_result["error"])
        self.assertIsNone(cancel_result["error"])
        self.assertEqual(session.state, GenerationSessionState.CANCELLED)
        stats = runtime.profile_stats()
        self.assertEqual(stats["pending_tier_operations"], 0)
        self.assertEqual(stats["active_prefetch_groups"], 0)
        self.assertEqual(stats["kv_pool_allocated_pages"], 0)
        self.assertEqual(stats["gpu_kv_used_pages"], 0)
        self.assertEqual(stats["cpu_kv_used_bytes"], 0)
        self.assertEqual(stats["total_ref_count"], 0)
        self.assertEqual(stats["logical_owner_count"], 0)
        self.assertEqual(stats["total_pin_count"], 0)
        self.assertGreater(stats["prefetch_count"], 0)
        self.assertEqual(stats["rgkv_stale_index_count"], 0)
        runtime.close()

    def test_close_waits_for_pending_prefetch_and_is_idempotent(self):
        session, runtime, state = self.make_session()
        self._prefill_with_cpu_residence(session, runtime)
        decode_thread, decode_result, blocking = self._start_blocked_decode(
            session, runtime, state
        )
        close_thread, close_result = self._start_call(session.close)
        self.assertTrue(close_thread.is_alive())
        blocking.released.set()
        decode_thread.join(timeout=3.0)
        close_thread.join(timeout=3.0)
        self.assertFalse(decode_thread.is_alive())
        self.assertFalse(close_thread.is_alive())
        self.assertIsNone(decode_result["error"])
        self.assertIsNone(close_result["error"])
        self.assertEqual(session.state, GenerationSessionState.CLOSED)
        stats = runtime.profile_stats()
        self.assertEqual(stats["pending_tier_operations"], 0)
        self.assertEqual(stats["kv_pool_allocated_pages"], 0)
        session.close()
        runtime.close()

    def test_session_exposes_no_manual_tier_control(self):
        session, runtime, _ = self.make_session()
        for name in (
            "move_kv_to_cpu",
            "prefetch_kv",
            "evict_kv",
            "migrate_kv",
        ):
            self.assertFalse(hasattr(session, name), name)
        for name in (
            "prefill",
            "decode_one",
            "stream",
            "cancel",
            "reset",
            "close",
        ):
            self.assertTrue(callable(getattr(session, name)))
        session.close()
        runtime.close()


if __name__ == "__main__":
    unittest.main()
