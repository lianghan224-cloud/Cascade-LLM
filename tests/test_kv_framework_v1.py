import math
import unittest

import torch
import torch.nn.functional as F

from layer_streaming import (
    KVCapacityErrorV1,
    KVLifecycleError,
    KVPolicy,
    KVUnsupportedError,
    PageState,
    PagedKVRuntime,
    SelectedPageView,
    default_paged_registry,
)
from layer_streaming.providers.generic_cuda import deterministic_lm_head


def make_policy(backend="reference_paged_exact", dtype="bf16", reuse="request_only"):
    return KVPolicy(
        dtype=dtype,
        page_size=16,
        reuse=reuse,
        attention_backend=backend,
    )


class KVFrameworkV1CPUTest(unittest.TestCase):
    def test_architecture_providers_are_independent_and_unqualified(self):
        registry = default_paged_registry()
        capabilities = registry.capabilities()
        self.assertEqual(capabilities["sm80"]["architectures"], ("sm80",))
        self.assertEqual(capabilities["sm86"]["architectures"], ("sm86",))
        self.assertEqual(capabilities["sm89"]["architectures"], ("sm89",))
        self.assertEqual(capabilities["sm90"]["architectures"], ("sm90",))
        self.assertEqual(
            capabilities["sm86"]["qualification_status"], "smoke_passed"
        )
        for name in ("sm80", "sm89", "sm90"):
            self.assertEqual(
                capabilities[name]["qualification_status"], "unqualified"
            )

    def make_runtime(
        self,
        query_heads=4,
        kv_heads=2,
        layers=1,
        pages=16,
        reuse="request_only",
    ):
        return PagedKVRuntime(
            layer_count=layers,
            num_query_heads=query_heads,
            num_kv_heads=kv_heads,
            head_dim=8,
            page_count=pages,
            page_size=16,
            dtype=torch.bfloat16,
            device="cpu",
            policy=make_policy(reuse=reuse),
            allow_reference=True,
        )

    def test_page_generation_rejects_stale_handle(self):
        runtime = self.make_runtime(pages=2)
        first = runtime.page_pool.allocate(owner_hint=1)
        runtime.page_pool.release(first)
        second = runtime.page_pool.allocate(owner_hint=2)
        self.assertEqual(first.page_id, second.page_id)
        self.assertNotEqual(first.generation, second.generation)
        with self.assertRaisesRegex(KVLifecycleError, "stale"):
            runtime.page_pool.descriptor(first)
        runtime.page_pool.release(second)
        runtime.close()

    def test_ragged_batch_prefill_mha_gqa_and_mqa(self):
        for query_heads, kv_heads in ((2, 2), (4, 2), (4, 1)):
            with self.subTest(query_heads=query_heads, kv_heads=kv_heads):
                torch.manual_seed(31 + kv_heads)
                runtime = self.make_runtime(query_heads, kv_heads)
                requests = (
                    runtime.create_request(64, request_id=11),
                    runtime.create_request(64, request_id=12),
                )
                lengths = (17, 5)
                total = sum(lengths)
                key = torch.randn(total, kv_heads, 8, dtype=torch.bfloat16)
                value = torch.randn_like(key)
                query = torch.randn(total, query_heads, 8, dtype=torch.bfloat16)
                runtime.append(requests, 0, key, value, lengths)
                result = runtime.attend(
                    requests,
                    0,
                    query,
                    lengths,
                    phase="prefill",
                    return_logsumexp=True,
                )
                self.assertEqual(result.output.shape, query.shape)
                self.assertEqual(result.logsumexp.shape, query.shape[:2])
                groups = query_heads // kv_heads
                cursor = 0
                for length in lengths:
                    current_query = query[cursor : cursor + length].transpose(0, 1).unsqueeze(0)
                    current_key = key[cursor : cursor + length].transpose(0, 1).unsqueeze(0)
                    current_value = value[cursor : cursor + length].transpose(0, 1).unsqueeze(0)
                    expected = F.scaled_dot_product_attention(
                        current_query.float(),
                        current_key.float().repeat_interleave(groups, dim=1),
                        current_value.float().repeat_interleave(groups, dim=1),
                        dropout_p=0.0,
                        is_causal=True,
                    ).squeeze(0).transpose(0, 1)
                    torch.testing.assert_close(
                        result.output[cursor : cursor + length].float(),
                        expected,
                        atol=1.7e-2,
                        rtol=1.7e-2,
                    )
                    cursor += length
                batch = runtime.prepare_batch(requests[::-1], lengths[::-1], 0)
                self.assertEqual(batch.request_ids, (12, 11))
                self.assertEqual(batch.sequence_lengths.tolist(), [5, 17])
                runtime.release(requests[0])
                self.assertEqual(runtime.page_pool.allocated_pages, 1)
                runtime.close()

    def test_fork_tail_cow_beam_and_speculative_commit(self):
        runtime = self.make_runtime(query_heads=4, kv_heads=1, layers=2)
        parent = runtime.create_request(64, request_id=1)
        key = torch.randn(3, 1, 8, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        for layer in range(2):
            runtime.append((parent,), layer, key, value, (3,))
        branch = runtime.fork(parent, request_id=2)
        shared = parent.block_table.handles[0]
        self.assertEqual(shared.state, PageState.SHARED)
        self.assertEqual(runtime.page_pool.descriptor(shared).ref_count, 2)
        draft_key = torch.randn(2, 1, 8, dtype=torch.bfloat16)
        draft_value = torch.randn_like(draft_key)
        for layer in range(2):
            runtime.append((branch,), layer, draft_key, draft_value, (2,))
        self.assertNotEqual(
            parent.block_table.handles[0], branch.block_table.handles[0]
        )
        self.assertEqual(parent.sequence_length, 3)
        self.assertEqual(branch.sequence_length, 5)
        runtime.commit_branch(parent, branch)
        self.assertEqual(parent.sequence_length, 5)
        self.assertNotIn(2, runtime._requests)

        beam = runtime.fork(parent, request_id=3)
        runtime.discard_branch(beam)
        self.assertEqual(parent.sequence_length, 5)
        profile = runtime.profile_stats()
        self.assertGreaterEqual(profile["fork_count"], 2)
        self.assertGreaterEqual(profile["cow_count"], 1)
        self.assertGreaterEqual(profile["page_allocations"], 2)
        self.assertEqual(profile["committed_tokens"], 5)
        self.assertEqual(profile["append_tokens"], 10)
        self.assertEqual(profile["attention_accuracy"], "exact")
        self.assertEqual(profile["layout"], "hnd")
        runtime.close()

    def test_chunked_prefill_reads_random_noncontiguous_pages(self):
        torch.manual_seed(55)
        runtime = self.make_runtime(query_heads=4, kv_heads=2, pages=6)
        request = runtime.create_request(64, request_id=21)
        blocker = runtime.create_request(16, request_id=22)
        prefix_key = torch.randn(16, 2, 8, dtype=torch.bfloat16)
        prefix_value = torch.randn_like(prefix_key)
        runtime.append((request,), 0, prefix_key, prefix_value, (16,))
        blocker_kv = torch.zeros(16, 2, 8, dtype=torch.bfloat16)
        runtime.append((blocker,), 0, blocker_kv, blocker_kv, (16,))
        chunk_key = torch.randn(2, 2, 8, dtype=torch.bfloat16)
        chunk_value = torch.randn_like(chunk_key)
        chunk_query = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        runtime.append((request,), 0, chunk_key, chunk_value, (2,))
        self.assertEqual(
            request.block_table.physical_page_ids(),
            (0, 2),
        )
        output = runtime.attend(
            (request,),
            0,
            chunk_query,
            (2,),
            query_positions=(torch.tensor([16, 17]),),
            phase="prefill",
        ).output
        full_key = torch.cat((prefix_key, chunk_key), dim=0).transpose(0, 1)
        full_value = torch.cat((prefix_value, chunk_value), dim=0).transpose(0, 1)
        query = chunk_query.transpose(0, 1)
        scores = torch.matmul(query.float(), full_key.float().repeat_interleave(2, 0).transpose(-1, -2))
        scores *= 1.0 / math.sqrt(8.0)
        mask = torch.arange(18).view(1, -1) <= torch.tensor([16, 17]).view(-1, 1)
        scores.masked_fill_(~mask.unsqueeze(0), -float("inf"))
        expected = torch.matmul(
            torch.softmax(scores, dim=-1),
            full_value.float().repeat_interleave(2, 0),
        ).transpose(0, 1)
        torch.testing.assert_close(
            output.float(), expected, atol=1.7e-2, rtol=1.7e-2
        )
        runtime.close()

    def test_session_and_in_memory_prefix_survive_origin_release(self):
        runtime = self.make_runtime(
            layers=1,
            pages=8,
            reuse="prefix_memory",
        )
        origin = runtime.create_request(
            64, request_id=7, reuse_namespace="model/tokenizer/rope"
        )
        key = torch.randn(16, 2, 8, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        runtime.append((origin,), 0, key, value, (16,))
        hashes = runtime.register_prefix(origin, list(range(16)))
        self.assertEqual(len(hashes), 1)
        session = runtime.session_fork(origin, request_id=8)
        runtime.release(session)
        runtime.release(origin)
        self.assertEqual(runtime.page_pool.allocated_pages, 1)
        reused, match = runtime.reuse_prefix(
            list(range(16)) + [999],
            max_length=64,
            reuse_namespace="model/tokenizer/rope",
            request_id=9,
        )
        self.assertEqual(match.matched_tokens, 16)
        self.assertEqual(reused.sequence_length, 16)
        isolated, isolated_match = runtime.reuse_prefix(
            list(range(16)),
            max_length=64,
            reuse_namespace="other-tenant",
            request_id=10,
        )
        self.assertEqual(isolated_match.matched_tokens, 0)
        runtime.release(isolated)
        runtime.close()

    def test_reuse_policy_does_not_silently_expand(self):
        runtime = self.make_runtime(reuse="request_only")
        state = runtime.create_request(32)
        with self.assertRaisesRegex(KVUnsupportedError, "reuse policy"):
            runtime.session_fork(state)
        with self.assertRaisesRegex(KVUnsupportedError, "reuse policy"):
            runtime.register_prefix(state, [])
        runtime.close()

    def test_admission_and_release_are_predictable(self):
        runtime = self.make_runtime(pages=2)
        first = runtime.create_request(32)
        key = torch.zeros(32, 2, 8, dtype=torch.bfloat16)
        runtime.append((first,), 0, key, key, (32,))
        second = runtime.create_request(32)
        with self.assertRaises(KVCapacityErrorV1):
            runtime.append(
                (second,),
                0,
                key[:1],
                key[:1],
                (1,),
            )
        runtime.release(first)
        runtime.append((second,), 0, key[:1], key[:1], (1,))
        self.assertEqual(second.sequence_length, 1)
        runtime.close()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class KVFrameworkV1CUDATest(unittest.TestCase):
    def test_deterministic_lm_head_is_chunk_independent(self):
        torch.manual_seed(404)
        for dtype in (torch.bfloat16, torch.float16):
            hidden = torch.randn(2, 3, 129, device="cuda", dtype=dtype)
            weight = torch.randn(37, 129, device="cuda", dtype=dtype)
            complete = deterministic_lm_head(hidden, weight)
            chunked = torch.cat(
                [
                    deterministic_lm_head(hidden, chunk)
                    for chunk in weight.split((5, 11, 21))
                ],
                dim=-1,
            )
            self.assertTrue(torch.equal(complete, chunked))
            expected = torch.einsum(
                "...k,nk->...n", hidden.float(), weight.float()
            ).to(dtype)
            error = (complete.float() - expected.float()).abs()
            if dtype == torch.bfloat16:
                self.assertEqual(float(error.max()), 0.0)
            else:
                # Fixed-tree FP32 accumulation and einsum use different
                # reduction orders. This golden has one native FP16 result
                # differing by exactly 2**-18 and no larger error.
                self.assertLessEqual(float(error.max()), 2.0 ** -18)
                self.assertLessEqual(float(error.mean()), 2.0e-8)

    def test_generic_cuda_consumes_sparse_ready_page_metadata(self):
        torch.manual_seed(405)
        runtime = PagedKVRuntime(
            layer_count=1,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=128,
            page_count=4,
            page_size=16,
            dtype=torch.bfloat16,
            device="cuda:0",
            policy=make_policy("generic_cuda"),
        )
        state = runtime.create_request(64)
        key = torch.randn(33, 2, 128, dtype=torch.bfloat16, device="cuda")
        value = torch.randn_like(key)
        runtime.append((state,), 0, key, value, (33,))

        class FirstAndTailSelection:
            name = "test_first_and_tail"

            @staticmethod
            def select(requests, layer, query, batch_view):
                del requests, layer, query
                indices = torch.tensor([0, 2], dtype=torch.long, device="cuda")
                return SelectedPageView(
                    flat_page_ids=batch_view.flat_block_table[indices],
                    block_table_indptr=torch.tensor(
                        [0, 2], dtype=torch.int32, device="cuda"
                    ),
                    logical_block_ids=batch_view.flat_logical_block_ids[indices],
                    page_valid_tokens=batch_view.flat_page_valid_tokens[indices],
                    selection_name="test_first_and_tail",
                    exact=False,
                    metadata={},
                )

        runtime.selection = FirstAndTailSelection()
        query = torch.randn(1, 4, 128, dtype=torch.bfloat16, device="cuda")
        actual = runtime.attend(
            (state,),
            0,
            query,
            (1,),
            query_positions=(torch.tensor([32], device="cuda"),),
            phase="decode",
        ).output
        selected_key = torch.cat((key[:16], key[32:33]), dim=0)
        selected_value = torch.cat((value[:16], value[32:33]), dim=0)
        scores = torch.einsum(
            "thd,shd->ths",
            query.float(),
            selected_key.float().repeat_interleave(2, dim=1),
        ) / math.sqrt(128.0)
        expected = torch.einsum(
            "ths,shd->thd",
            torch.softmax(scores, dim=-1),
            selected_value.float().repeat_interleave(2, dim=1),
        )
        torch.testing.assert_close(
            actual.float(), expected, atol=1.7e-2, rtol=1.7e-2
        )
        page_key, page_value = runtime.store.read_pages(0, (0,))
        runtime.store.write_pages(0, (3,), page_key, page_value)
        copied_key, copied_value = runtime.store.read_pages(0, (3,))
        self.assertTrue(torch.equal(copied_key, page_key))
        self.assertTrue(torch.equal(copied_value, page_value))
        runtime.close()

    def _run_provider(self, backend, query_heads, kv_heads, length, dtype):
        dtype_name = "bf16" if dtype == torch.bfloat16 else "fp16"
        runtime = PagedKVRuntime(
            layer_count=1,
            num_query_heads=query_heads,
            num_kv_heads=kv_heads,
            head_dim=128,
            page_count=16,
            page_size=16,
            dtype=dtype,
            device="cuda:0",
            policy=make_policy(backend, dtype=dtype_name),
        )
        request = runtime.create_request(256)
        torch.manual_seed(71 + length + kv_heads)
        query = torch.randn(1, query_heads, length, 128, device="cuda", dtype=dtype)
        key = torch.randn(1, kv_heads, length, 128, device="cuda", dtype=dtype)
        value = torch.randn_like(key)
        runtime.append((request,), 0, key, value, (length,))
        output = runtime.attend(
            (request,),
            0,
            query,
            (length,),
            phase="prefill",
        ).output
        expected = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(query_heads // kv_heads, dim=1),
            value.repeat_interleave(query_heads // kv_heads, dim=1),
            dropout_p=0.0,
            is_causal=True,
        ).squeeze(0).transpose(0, 1)
        difference = (output.float() - expected.float()).abs()
        tolerance = 1.7e-2 if dtype == torch.bfloat16 else 2.1e-3
        self.assertLessEqual(float(difference.max()), tolerance)
        profile = runtime.profile_stats()
        self.assertEqual(profile["paged_attention_provider"], backend)
        self.assertEqual(profile["workspace_peak_bytes"], 0)
        runtime.close()

    def test_generic_cuda_prefill_decode_mha_gqa_mqa(self):
        for dtype in (torch.bfloat16, torch.float16):
            for query_heads, kv_heads in ((4, 4), (8, 2), (8, 1)):
                with self.subTest(dtype=dtype, q=query_heads, kv=kv_heads):
                    self._run_provider(
                        "generic_cuda", query_heads, kv_heads, 17, dtype
                    )

    def test_sm86_provider_is_real_and_workspace_free(self):
        if torch.cuda.get_device_capability() != (8, 6):
            self.skipTest("SM86 hardware is required")
        self._run_provider("sm86", 8, 2, 33, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
