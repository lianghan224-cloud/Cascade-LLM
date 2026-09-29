import inspect
import unittest

import torch

from layer_streaming.kv import PagedKVRuntime
from layer_streaming.kv.selection import (
    LogicalKVBlockId,
    QuestCPUIndex,
    QuestFlatSelection,
    RGKVGPUScorer,
    TorchTensorizedQuestScorer,
)
from layer_streaming.kv_policy import (
    KVAccuracy,
    KVDataType,
    KVPolicy,
    KVSelectionPolicy,
)


class QuestTensorizedScorerTest(unittest.TestCase):
    def _records(self, count, dimensions=4):
        index = QuestCPUIndex()
        records = []
        for logical in range(count):
            records.append(
                index.build_compact(
                    torch.ones((2, dimensions), dtype=torch.float32),
                    {
                        "logical_block_id": LogicalKVBlockId(
                            "model", "session", "branch", 0, logical
                        ),
                        "token_start": logical * 16,
                        "data_version": logical + 1,
                    },
                )
            )
        return index, tuple(records)

    def test_deterministic_ties_recent_budget_and_logical_order(self):
        _, records = self._records(5)
        scorer = TorchTensorizedQuestScorer()
        scorer.prepare_index((7, 0), records, "cpu")

        result = scorer.select(
            (7, 0),
            torch.ones((1, 2, 4), dtype=torch.float32),
            scored_budget=2,
            recent_window=16,
            page_size=16,
        )

        # All scored pages tie, so stable GPU/CPU tensor ordering chooses the
        # earliest two; the fixed recent page is merged only afterwards.
        self.assertEqual(result.selected_positions.tolist(), [0, 1, 4])
        self.assertEqual(result.scored_budget, 2)
        self.assertEqual(result.recent_budget, 1)
        self.assertEqual(result.total_budget, 3)
        self.assertEqual(result.selected_count, 3)

    def test_workspace_estimate_is_explicit_and_additive(self):
        estimate = TorchTensorizedQuestScorer.estimate_workspace(10, 8)
        self.assertEqual(estimate.index_bytes, 10 * (3 * 8 * 4 + 2 * 8))
        self.assertEqual(estimate.scoring_bytes, 10 * 4)
        self.assertEqual(estimate.topk_bytes, 10 * 8)
        self.assertEqual(estimate.merge_bytes, 10 * 8)
        self.assertEqual(
            estimate.total_bytes,
            estimate.index_bytes + estimate.temporary_bytes,
        )

    def test_tensorized_selection_matches_cpu_reference(self):
        torch.manual_seed(20260809)
        index = QuestCPUIndex()
        records = []
        for logical in range(8):
            records.append(
                index.build_compact(
                    torch.randn(5, 4),
                    {
                        "logical_block_id": LogicalKVBlockId(
                            "model", "session", "branch", 0, logical
                        ),
                        "token_start": logical * 16,
                        "data_version": logical + 1,
                    },
                )
            )
        query = torch.randn(1, 2, 4)
        scorer = TorchTensorizedQuestScorer()
        scorer.prepare_index((3, 0), records, "cpu")
        tensorized = scorer.select(
            (3, 0),
            query,
            scored_budget=3,
            recent_window=32,
            page_size=16,
        )
        cpu = index.select(
            query.float().mean(dim=(0, 1)).tolist(),
            records[:-2],
            budget=3,
            mode="budget",
        )
        expected = sorted(
            item.logical_block_id.logical_block
            for item in cpu.records + tuple(records[-2:])
        )
        self.assertEqual(tensorized.selected_positions.tolist(), expected)

    def test_selection_hot_path_has_no_host_conversion_calls(self):
        scorer_source = inspect.getsource(TorchTensorizedQuestScorer.select)
        adapter_source = inspect.getsource(QuestFlatSelection._select_tensorized)
        for forbidden in (".cpu(", ".tolist(", ".item("):
            self.assertNotIn(forbidden, scorer_source)
            self.assertNotIn(forbidden, adapter_source)


class QuestTensorizedRuntimeTest(unittest.TestCase):
    def _runtime(self, device="cpu", layer_count=1):
        policy = KVPolicy(
            accuracy=KVAccuracy.SPARSE,
            dtype=KVDataType.BF16,
            selection=KVSelectionPolicy.QUEST_FLAT,
            attention_backend="reference_paged_exact",
            page_size=16,
            page_budget=2,
            recent_window=16,
            quest_scorer="torch_tensorized",
        )
        runtime = PagedKVRuntime(
            layer_count=layer_count,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=8,
            page_count=8,
            page_size=16,
            dtype=torch.bfloat16,
            device=device,
            policy=policy,
            allow_reference=True,
        )
        self.assertIsInstance(runtime.selection.scorer, RGKVGPUScorer)
        return runtime

    def test_multilayer_prefill_is_exact_until_index_commit(self):
        runtime = self._runtime(layer_count=2)
        state = runtime.create_request(64)
        key = torch.randn(17, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (17,))
        query = torch.randn(17, 4, 8, dtype=torch.bfloat16)
        batch = runtime.prepare_batch((state,), (17,), 0)
        selected = runtime.selection.select((state,), 0, query, batch)

        self.assertTrue(selected.exact)
        self.assertEqual(int(selected.flat_page_ids.numel()), 2)
        self.assertEqual(
            selected.metadata["requests"][0]["fallback_reason"],
            "rgkv_selection_is_decode_only",
        )
        self.assertEqual(runtime.selection.records, {})

        runtime.append((state,), 1, key, key, (17,))
        self.assertEqual(len(runtime.selection.records), 4)
        self.assertEqual(runtime.selection.scorer.stats()["device_indices"], 2)
        runtime.close()

    def test_explicit_provider_integrates_at_generation_safe_view_boundary(self):
        runtime = self._runtime()
        state = runtime.create_request(64)
        key = torch.randn(33, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, key, key, (33,))
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        batch = runtime.prepare_batch((state,), (1,), 0)
        selected = runtime.selection.select((state,), 0, query, batch)
        handles = selected.resolve_handles((state,), runtime.page_pool)

        self.assertEqual(selected.selection_name, "rgkv")
        self.assertEqual(
            selected.metadata["requests"][0]["scorer_provider"],
            "torch_tensorized",
        )
        self.assertEqual(len(handles), 2)
        self.assertTrue(all(handle.generation > 0 for handle in handles))
        self.assertEqual(runtime.selection.scorer.stats()["queries"], 1)
        profile = runtime.profile_stats()
        self.assertEqual(profile["kv_selection_scorer"], "torch_tensorized")
        self.assertEqual(
            profile["rgkv_scorer_stats"]["qualification_status"],
            "cuda_smoke",
        )
        runtime.close()

    def test_changed_tail_republishes_device_index_before_selection(self):
        runtime = self._runtime()
        state = runtime.create_request(64)
        initial = torch.randn(17, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, initial, initial, (17,))
        builds = runtime.selection.scorer.index_builds
        updates = runtime.selection.scorer.index_updates

        extension = torch.randn(1, 2, 8, dtype=torch.bfloat16)
        runtime.append((state,), 0, extension, extension, (1,))
        self.assertEqual(runtime.selection.scorer.index_builds, builds)
        self.assertEqual(runtime.selection.scorer.index_updates, updates + 1)

        query = torch.randn(1, 4, 8, dtype=torch.bfloat16)
        batch = runtime.prepare_batch((state,), (1,), 0)
        selected = runtime.selection.select((state,), 0, query, batch)
        self.assertEqual(int(selected.flat_page_ids.shape[0]), 2)
        runtime.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_selection_stays_on_query_device(self):
        runtime = self._runtime("cuda:0")
        state = runtime.create_request(64)
        key = torch.randn(33, 2, 8, dtype=torch.bfloat16, device="cuda:0")
        runtime.append((state,), 0, key, key, (33,))
        query = torch.randn(1, 4, 8, dtype=torch.bfloat16, device="cuda:0")
        batch = runtime.prepare_batch((state,), (1,), 0)

        selected = runtime.selection.select((state,), 0, query, batch)

        self.assertEqual(selected.flat_page_ids.device.type, "cuda")
        self.assertEqual(
            runtime.selection.scorer._indices[(state.request_id, 0)]
            .summaries.device.type,
            "cuda",
        )
        runtime.close()


if __name__ == "__main__":
    unittest.main()
