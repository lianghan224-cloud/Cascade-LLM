import inspect
import unittest

import torch

from layer_streaming.kv.page_pool import KVPagePoolV1
from layer_streaming.kv.selection import (
    DEPRECATED_QUEST_APIS,
    QuestCPUIndex,
    QuestFlatSelection,
    RGKVBudget,
    RGKVCPUReferenceScorer,
    RGKVGPUScorer,
    RGKVIndex,
    RGKVPageSummary,
    RGKVSelectionPolicy,
    RGKVSelectionResult,
)


class _Transaction:
    def __init__(self, start, token_count):
        self.start = start
        self.token_count = token_count
        self.completed_layers = set()

    @property
    def end(self):
        return self.start + self.token_count


class _State:
    request_id = 3

    def __init__(self, transaction, handles):
        self.pending_append = transaction
        self.sequence_length = 0
        self.block_table = type("BlockTable", (), {"handles": handles})()


class _Runtime:
    page_size = 4
    layer_count = 1
    device = torch.device("cpu")

    def __init__(self, page_pool):
        self.page_pool = page_pool
        self._metrics = type("Metrics", (), {"rgkv_update_ms": 0.0})()
        self.page_count = page_pool.page_count


class RGKVPublicAPITest(unittest.TestCase):
    def test_public_contract_and_legacy_imports_coexist(self):
        self.assertTrue(issubclass(RGKVIndex, object))
        self.assertTrue(issubclass(RGKVSelectionPolicy, object))
        self.assertTrue(issubclass(RGKVCPUReferenceScorer, object))
        self.assertTrue(issubclass(RGKVGPUScorer, object))
        self.assertTrue(issubclass(RGKVPageSummary, object))
        self.assertTrue(issubclass(RGKVSelectionResult, object))
        self.assertEqual(DEPRECATED_QUEST_APIS["QuestCPUIndex"], "RGKVIndex")
        self.assertEqual(QuestFlatSelection.name, "quest_flat")
        self.assertTrue(callable(QuestCPUIndex))

    def test_gpu_budget_is_total_not_recent_plus_topk(self):
        index = RGKVIndex()
        for logical in range(6):
            index.build_page(
                logical,
                torch.full((2, 4), float(logical + 1)),
                data_epoch=logical + 1,
            )
        scorer = RGKVGPUScorer()
        scorer.prepare_index((3, 0), index, "cpu")
        result = scorer.select(
            (3, 0),
            torch.ones((1, 2, 4)),
            RGKVBudget(total_page_budget=4, recent_pages=2),
        )
        self.assertEqual(result.selected_count, 4)
        self.assertEqual(result.mandatory_recent_pages, 2)
        self.assertEqual(result.relevance_selected_pages, 2)
        self.assertEqual(result.selected_logical_pages.tolist(), [2, 3, 4, 5])

    def test_cpu_gpu_tie_contract_is_score_desc_then_logical_id_ascending(self):
        index = RGKVIndex()
        page_epochs = {}
        for logical in range(6):
            index.build_page(
                logical,
                torch.ones((2, 4)),
                data_epoch=logical + 1,
            )
            page_epochs[logical] = logical + 1
        budget = RGKVBudget(total_page_budget=4, recent_pages=1)
        query = torch.ones((1, 2, 4))
        cpu = RGKVCPUReferenceScorer().select(
            index, query, budget, page_epochs=page_epochs
        )
        gpu_scorer = RGKVGPUScorer()
        gpu_scorer.prepare_index((4, 0), index, "cpu")
        gpu = gpu_scorer.select((4, 0), query, budget)
        self.assertEqual(cpu.selected_logical_pages.tolist(), [0, 1, 2, 5])
        self.assertTrue(
            torch.equal(
                cpu.selected_logical_pages, gpu.selected_logical_pages
            )
        )

    def test_gpu_select_has_no_host_conversion_or_explicit_sync(self):
        source = inspect.getsource(RGKVGPUScorer.select)
        for forbidden in (".cpu(", ".numpy(", ".tolist(", ".item(", "synchronize("):
            self.assertNotIn(forbidden, source)
        self.assertEqual(RGKVGPUScorer.qualification_status, "cuda_smoke")

    def test_gpu_device_index_publish_and_staged_override_are_incremental(self):
        index = RGKVIndex()
        summary = index.build_page(
            0, torch.ones((2, 2, 3)), data_epoch=1
        )
        scorer = RGKVGPUScorer()
        scorer.publish_summary((9, 0), summary, capacity=4)
        self.assertEqual(scorer.stats()["index_updates"], 1)
        self.assertEqual(scorer._indices[(9, 0)].candidate_count, 1)
        override = torch.stack(
            (
                torch.full((2, 3), 2.0),
                torch.full((2, 3), 3.0),
                torch.full((2, 3), 2.5),
            )
        )
        result = scorer.select(
            (9, 0),
            torch.ones((1, 4, 3)),
            RGKVBudget(2, recent_pages=1),
            override_logical=1,
            override_summary=override,
            candidate_count=2,
        )
        self.assertEqual(result.selected_logical_pages.tolist(), [0, 1])
        self.assertEqual(scorer._indices[(9, 0)].candidate_count, 1)

    def test_two_phase_append_uses_only_current_keys_and_committed_epoch(self):
        pool = KVPagePoolV1(
            page_count=2,
            store_id="test",
            dtype="bf16",
            layout="hnd",
        )
        handle = pool.allocate(owner_hint=3, logical_mapping=(3, 0))
        transaction = _Transaction(0, 3)
        state = _State(transaction, [handle])
        runtime = _Runtime(pool)
        policy = RGKVSelectionPolicy(
            RGKVBudget(1), scorer=RGKVGPUScorer()
        )
        keys = torch.tensor(
            [
                [[1.0, 3.0], [3.0, 5.0]],
                [[2.0, 4.0], [4.0, 6.0]],
                [[3.0, 5.0], [5.0, 7.0]],
            ]
        )
        staged = policy.stage_append_layer(
            runtime, state, 0, keys, transaction, (3,)
        )
        self.assertEqual(len(staged), 1)
        self.assertEqual(tuple(staged[0].values.shape), (3, 2, 2))
        transaction.completed_layers.add(0)
        state.pending_append = None
        state.sequence_length = 3
        pool.seal(handle, 3)
        pool.mark_data_updated(handle, version=17)

        published = policy.commit_append(runtime, state, transaction)

        self.assertEqual(len(published), 1)
        self.assertEqual(published[0].data_epoch, 17)
        self.assertEqual(published[0].valid_tokens, 3)
        self.assertEqual(pool.descriptor(handle).index_version, 17)
        self.assertFalse(hasattr(transaction, "_rgkv_staged_layers"))

    def test_append_to_partial_page_combines_without_historical_read(self):
        pool = KVPagePoolV1(
            page_count=2,
            store_id="test",
            dtype="bf16",
            layout="hnd",
        )
        handle = pool.allocate(owner_hint=3, logical_mapping=(3, 0))
        runtime = _Runtime(pool)
        policy = RGKVSelectionPolicy(RGKVBudget(1))

        first = _Transaction(0, 2)
        state = _State(first, [handle])
        policy.stage_append_layer(
            runtime, state, 0, torch.ones((2, 2, 2)), first, (2,)
        )
        first.completed_layers.add(0)
        state.pending_append = None
        state.sequence_length = 2
        pool.seal(handle, 2)
        pool.mark_data_updated(handle, version=2)
        policy.commit_append(runtime, state, first)

        second = _Transaction(2, 1)
        state.pending_append = second
        policy.stage_append_layer(
            runtime,
            state,
            0,
            torch.full((1, 2, 2), 3.0),
            second,
            (1,),
        )
        second.completed_layers.add(0)
        state.pending_append = None
        state.sequence_length = 3
        pool.seal(handle, 3)
        pool.mark_data_updated(handle, version=3)
        summary = policy.commit_append(runtime, state, second)[0]

        self.assertEqual(summary.valid_tokens, 3)
        self.assertEqual(summary.data_epoch, 3)
        self.assertTrue(torch.equal(summary.minimum, torch.ones((2, 2))))
        self.assertTrue(torch.equal(summary.maximum, torch.full((2, 2), 3.0)))

    def test_gqa_scoring_preserves_kv_heads(self):
        index = RGKVIndex()
        keys = torch.tensor(
            [
                [[10.0, 0.0], [0.0, 1.0]],
                [[12.0, 0.0], [0.0, 2.0]],
            ]
        )
        summary = index.build_page(0, keys, data_epoch=1)
        self.assertEqual(tuple(summary.values.shape), (3, 2, 2))
        scorer = RGKVGPUScorer()
        scorer.prepare_index((5, 0), index, "cpu")
        query = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]]
        )
        result = scorer.select((5, 0), query, RGKVBudget(1))
        self.assertEqual(result.selected_logical_pages.tolist(), [0])

    def test_decode_can_select_transaction_local_summary_before_epoch_commit(self):
        pool = KVPagePoolV1(
            page_count=2,
            store_id="test",
            dtype="bf16",
            layout="hnd",
        )
        handle = pool.allocate(owner_hint=3, logical_mapping=(3, 0))
        transaction = _Transaction(0, 1)
        state = _State(transaction, [handle])
        runtime = _Runtime(pool)
        policy = RGKVSelectionPolicy(RGKVBudget(1))
        policy.stage_append_layer(
            runtime,
            state,
            0,
            torch.ones((1, 2, 2)),
            transaction,
            (1,),
        )
        batch = type(
            "BatchView",
            (),
            {
                "batch_size": 1,
                "flat_block_table": torch.tensor([handle.page_id]),
                "flat_logical_block_ids": torch.tensor([0]),
                "flat_page_valid_tokens": torch.tensor([1]),
            },
        )()

        selected = policy.select(
            (state,), 0, torch.ones((1, 4, 2)), batch
        )

        self.assertEqual(selected.logical_block_ids.tolist(), [0])
        self.assertEqual(len(policy.indexes), 0)
        self.assertEqual(pool.descriptor(handle).index_version, 0)

    def test_prefix_reuse_shares_sealed_layer_bundle(self):
        pool = KVPagePoolV1(
            page_count=2,
            store_id="test",
            dtype="bf16",
            layout="hnd",
        )
        handle = pool.allocate(owner_hint=3, logical_mapping=(3, 0))
        transaction = _Transaction(0, 4)
        source = _State(transaction, [handle])
        runtime = _Runtime(pool)
        policy = RGKVSelectionPolicy(RGKVBudget(1))
        policy.stage_append_layer(
            runtime,
            source,
            0,
            torch.ones((4, 2, 2)),
            transaction,
            (4,),
        )
        transaction.completed_layers.add(0)
        source.pending_append = None
        source.sequence_length = 4
        pool.seal(handle, 4)
        pool.mark_data_updated(handle, version=9)
        published = policy.commit_append(runtime, source, transaction)
        target = type(
            "State",
            (),
            {"request_id": 4, "block_table": source.block_table},
        )()

        policy.reuse_request_from_page_metadata(runtime, target)

        self.assertIs(policy.indexes[(4, 0)].get(0), published[0])
        self.assertEqual(policy.indexes[(4, 0)].stats()["prefix_shares"], 1)


if __name__ == "__main__":
    unittest.main()
