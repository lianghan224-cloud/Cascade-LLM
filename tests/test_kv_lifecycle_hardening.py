import unittest

import torch

from layer_streaming import (
    KVLifecycleError,
    KVPagePoolV1,
    KVPolicy,
    PagedKVRuntime,
    PagedProviderBundle,
    ReferencePagedExactBackend,
    SelectedPageView,
    TorchPagedKVKernelBackend,
    default_paged_registry,
)
from layer_streaming.kv.selection import LogicalKVBlockId, QuestCPUIndex
from layer_streaming.kv.selection.dense import DenseSelection


def make_runtime(layers=1, pages=8, reuse="request_only", backend=None):
    registry = default_paged_registry(load_cuda=False)
    provider_name = "reference_paged_exact"
    if backend is not None:
        provider_name = "lifecycle_hardening"
        registry.register(
            PagedProviderBundle(
                name=provider_name,
                attention_backend=ReferencePagedExactBackend(),
                kv_kernel_backend=backend,
            )
        )
    return PagedKVRuntime(
        layer_count=layers,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        page_count=pages,
        page_size=16,
        dtype=torch.bfloat16,
        device="cpu",
        policy=KVPolicy(
            dtype="bf16",
            page_size=16,
            reuse=reuse,
            attention_backend=provider_name,
        ),
        provider_registry=registry,
        allow_reference=True,
    )


class KVLifecycleHardeningTest(unittest.TestCase):
    def test_foreign_handle_is_rejected_even_when_v1_fields_match(self):
        first = KVPagePoolV1(2, "gpu", "bf16")
        second = KVPagePoolV1(2, "gpu", "bf16")
        handle = first.allocate(owner_hint=1)
        other = second.allocate(owner_hint=2)
        self.assertEqual(handle.page_id, other.page_id)
        self.assertEqual(handle.generation, other.generation)
        self.assertNotEqual(handle.pool_uuid, other.pool_uuid)
        with self.assertRaisesRegex(KVLifecycleError, "foreign"):
            second.descriptor(handle)
        first.release(handle)
        second.release(other)

    def test_copy_pins_source_and_target_until_abort(self):
        pool = KVPagePoolV1(2, "gpu", "bf16")
        source = pool.allocate(owner_hint=1)
        target = pool.allocate(owner_hint=1)
        pool.activate(source, 3)
        pool.seal(source, 3)
        pool.begin_copy(source, target)
        self.assertEqual(pool.descriptor(source).inflight_io, 1)
        self.assertEqual(pool.descriptor(target).inflight_io, 1)
        self.assertEqual(pool.descriptor(target).state.value, "copying")
        pool.abort_copy(source, target, error=RuntimeError("copy failed"))
        self.assertEqual(pool.descriptor(source).pin_count, 0)
        self.assertEqual(pool.descriptor(target).pin_count, 0)
        self.assertEqual(pool.descriptor(target).state.value, "allocated")
        pool.release(target)
        pool.release(source)
        self.assertTrue(pool.validate_invariants())

    def test_append_failure_quiesces_and_restores_all_owners(self):
        class FailSecondAppend(TorchPagedKVKernelBackend):
            name = "fail_second_append"

            def __init__(self):
                self.calls = 0

            def append_kv(self, append_input):
                self.calls += 1
                if self.calls == 2:
                    raise RuntimeError("injected layer append failure")
                return super().append_kv(append_input)

        runtime = make_runtime(layers=2, backend=FailSecondAppend())
        state = runtime.create_request(32, request_id=17)
        key = torch.randn(3, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (3,))
        with self.assertRaisesRegex(RuntimeError, "injected layer"):
            runtime.append((state,), 1, key, key, (3,))
        self.assertIsNone(state.pending_append)
        self.assertEqual(state.sequence_length, 0)
        self.assertEqual(state.block_table.handles, [])
        self.assertEqual(runtime.page_pool.allocated_pages, 0)
        self.assertTrue(runtime.page_pool.validate_invariants())
        runtime.close()

    def test_runtime_refs_equal_logical_owner_count_through_cow(self):
        runtime = make_runtime(layers=1)
        parent = runtime.create_request(64, request_id=21)
        key = torch.randn(3, 2, 8, dtype=torch.bfloat16)
        runtime.append((parent,), 0, key, key, (3,))
        shared = parent.block_table.handles[0]
        branch = runtime.fork(parent, request_id=22)
        descriptor = runtime.page_pool.descriptor(shared)
        self.assertEqual(descriptor.ref_count, len(descriptor.logical_mappings))
        runtime.append((branch,), 0, key[:1], key[:1], (1,))
        for state in (parent, branch):
            for handle in state.block_table.handles:
                descriptor = runtime.page_pool.descriptor(handle)
                self.assertEqual(
                    descriptor.ref_count, len(descriptor.logical_mappings)
                )
        runtime.release(branch)
        runtime.release(parent)
        self.assertTrue(runtime.page_pool.validate_invariants())
        runtime.close()

    def test_page_data_epoch_is_global_and_request_version_is_separate(self):
        runtime = make_runtime(layers=1)
        first = runtime.create_request(64, request_id=23)
        second = runtime.create_request(64, request_id=24)
        key = torch.randn(1, 2, 8, dtype=torch.bfloat16)
        runtime.append((first,), 0, key, key, (1,))
        runtime.append((second,), 0, key, key, (1,))
        first_descriptor = runtime.page_pool.descriptor(
            first.block_table.handles[0]
        )
        second_descriptor = runtime.page_pool.descriptor(
            second.block_table.handles[0]
        )
        self.assertEqual((first.version, second.version), (1, 1))
        self.assertEqual(
            (first_descriptor.data_version, second_descriptor.data_version),
            (1, 2),
        )
        self.assertEqual(runtime.profile_stats()["global_data_epoch"], 2)
        with self.assertRaisesRegex(KVLifecycleError, "strictly"):
            runtime.page_pool.mark_data_updated(
                second.block_table.handles[0], version=2
            )
        runtime.close()

    def test_quest_records_are_deleted_at_zero_refs(self):
        index = QuestCPUIndex()
        record = index.build(
            [[1.0, 2.0]],
            {
                "logical_block_id": LogicalKVBlockId(
                    "model", "session", "branch", 0, 0
                ),
                "data_version": 1,
            },
        )
        index.fork_ref(record)
        self.assertEqual(index.stats()["references"], 2)
        self.assertFalse(index.release_ref(record))
        self.assertTrue(index.release_ref(record))
        self.assertEqual(index.stats()["records"], 0)
        self.assertEqual(index.stats()["references"], 0)

    def test_prefix_replacement_releases_stale_cache_owner(self):
        runtime = make_runtime(reuse="prefix_memory")
        first = runtime.create_request(32, request_id=31)
        second = runtime.create_request(32, request_id=32)
        key = torch.randn(16, 2, 8, dtype=torch.bfloat16)
        runtime.append((first,), 0, key, key, (16,))
        runtime.append((second,), 0, key + 1, key + 1, (16,))
        runtime.register_prefix(first, list(range(16)))
        old = first.block_table.handles[0]
        self.assertEqual(runtime.page_pool.descriptor(old).ref_count, 2)
        runtime.register_prefix(second, list(range(16)))
        self.assertEqual(runtime.page_pool.descriptor(old).ref_count, 1)
        self.assertNotIn(old.identity(), runtime.prefix_cache.owned_handles)
        runtime.release(first)
        runtime.release(second)
        runtime.close()

    def test_attention_pins_only_selected_generation_handles(self):
        class FirstAndLastSelection:
            name = "first_and_last"

            def select(self, requests, layer, query, batch_view):
                del requests, layer, query
                indices = torch.tensor([0, 2], dtype=torch.long)
                return SelectedPageView(
                    flat_page_ids=batch_view.flat_block_table[indices],
                    block_table_indptr=torch.tensor([0, 2], dtype=torch.int32),
                    logical_block_ids=batch_view.flat_logical_block_ids[indices],
                    page_valid_tokens=batch_view.flat_page_valid_tokens[indices],
                    selection_name=self.name,
                    exact=False,
                    metadata={},
                )

        runtime = make_runtime(pages=8)
        state = runtime.create_request(64, request_id=41)
        key = torch.randn(33, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (33,))
        runtime.selection = FirstAndLastSelection()
        captured = []
        original = runtime.ownership.begin_attention_kernel

        def capture(layer, handles):
            captured.extend(handles)
            return original(layer, handles)

        runtime.ownership.begin_attention_kernel = capture
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        result = runtime.attend((state,), 0, query, (1,), phase="decode")
        self.assertEqual(
            [item.page_id for item in captured],
            [state.block_table.handles[0].page_id,
             state.block_table.handles[2].page_id],
        )
        self.assertEqual(result.provider_metrics["selected_pin_count"], 2)
        fence = runtime.wait_attention_fence(
            result.provider_metrics["attention_fence_id"]
        )
        self.assertEqual(fence.status, "completed")
        runtime.close()

    def test_selected_handle_resolution_deduplicates_shared_page(self):
        runtime = make_runtime(pages=4)
        parent = runtime.create_request(32, request_id=51)
        key = torch.randn(16, 2, 8, dtype=torch.bfloat16)
        runtime.append((parent,), 0, key, key, (16,))
        child = runtime.fork(parent, request_id=52)
        batch = runtime.prepare_batch((parent, child), (1, 1), 0)
        selected = DenseSelection().select(
            (parent, child), 0, torch.empty(0), batch
        )
        handles = selected.resolve_handles(
            (parent, child), runtime.page_pool
        )
        self.assertEqual(len(handles), 1)
        self.assertEqual(handles[0], parent.block_table.handles[0])
        runtime.close()

    def test_duplicate_selected_logical_id_is_rejected_before_pin(self):
        class DuplicateSelection:
            name = "duplicate_fault"

            def select(self, requests, layer, query, batch_view):
                del requests, layer, query
                return SelectedPageView(
                    flat_page_ids=batch_view.flat_block_table[[0, 0]],
                    block_table_indptr=torch.tensor([0, 2], dtype=torch.int32),
                    logical_block_ids=batch_view.flat_logical_block_ids[[0, 0]],
                    page_valid_tokens=batch_view.flat_page_valid_tokens[[0, 0]],
                    selection_name=self.name,
                    exact=False,
                    metadata={},
                )

        runtime = make_runtime(pages=2)
        state = runtime.create_request(16, request_id=53)
        key = torch.randn(1, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (1,))
        runtime.selection = DuplicateSelection()
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            runtime.attend((state,), 0, query, (1,), phase="decode")
        self.assertEqual(runtime.profile_stats()["total_pin_count"], 0)
        runtime.close()

    def test_invalid_selected_logical_id_is_rejected_before_pin(self):
        class InvalidSelection:
            name = "invalid_fault"

            def select(self, requests, layer, query, batch_view):
                del requests, layer, query
                return SelectedPageView(
                    flat_page_ids=batch_view.flat_block_table[:1],
                    block_table_indptr=torch.tensor([0, 1], dtype=torch.int32),
                    logical_block_ids=torch.tensor([99], dtype=torch.int32),
                    page_valid_tokens=batch_view.flat_page_valid_tokens[:1],
                    selection_name=self.name,
                    exact=False,
                    metadata={},
                )

        runtime = make_runtime(pages=2)
        state = runtime.create_request(16, request_id=54)
        key = torch.randn(1, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (1,))
        runtime.selection = InvalidSelection()
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(IndexError, "outside request"):
            runtime.attend((state,), 0, query, (1,), phase="decode")
        self.assertEqual(runtime.profile_stats()["total_pin_count"], 0)
        runtime.close()

    def test_pending_attention_fence_holds_pin_until_explicit_wait(self):
        class ManualEvent:
            done = False

            def query(self):
                return self.done

        runtime = make_runtime(pages=2)
        state = runtime.create_request(16, request_id=61)
        key = torch.randn(1, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (1,))
        handle = state.block_table.handles[0]
        event = ManualEvent()
        runtime.page_pool.pin(handle)
        fence = runtime.ownership._fence(
            "attention_layer_0",
            request_id=(state.request_id,),
            source_handles=(handle,),
            cuda_event=event,
        )
        runtime.ownership.attention_pins[0] = [handle]
        runtime.ownership.attention_fences[0] = fence
        runtime.ownership._attention_fence_layers[fence.operation_id] = 0
        runtime.ownership._operation_fences[fence.operation_id] = fence
        descriptor = runtime.page_pool.descriptor(handle)
        self.assertEqual((descriptor.pin_count, descriptor.inflight_compute), (1, 1))
        self.assertFalse(fence.query())
        event.done = True
        runtime.drain_attention_fence(fence.operation_id)
        self.assertEqual((descriptor.pin_count, descriptor.inflight_compute), (0, 0))
        runtime.close()

    def test_attention_exception_records_failed_fence_and_unpins(self):
        runtime = make_runtime(pages=2)
        state = runtime.create_request(16, request_id=62)
        key = torch.randn(1, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (1,))

        def fail(request, phase=None):
            del request, phase
            raise RuntimeError("injected attention failure")

        runtime.dispatcher.execute = fail
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "injected") as raised:
            runtime.attend((state,), 0, query, (1,), phase="decode")
        fence_id = raised.exception.attention_fence_id
        fence = runtime.ownership._operation_fences[fence_id]
        self.assertEqual(fence.status, "failed")
        descriptor = runtime.page_pool.descriptor(state.block_table.handles[0])
        self.assertEqual((descriptor.pin_count, descriptor.inflight_compute), (0, 0))
        runtime.close()


if __name__ == "__main__":
    unittest.main()
