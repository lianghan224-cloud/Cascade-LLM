import unittest

import torch

from layer_streaming.kv.selection.rgkv_index import (
    RGKVBudget,
    RGKVIndex,
    RGKVStaleIndexError,
    STALE_INDEX,
)


class RGKVBudgetTest(unittest.TestCase):
    def _index(self, count=8):
        index = RGKVIndex()
        epochs = {}
        for logical in range(count):
            # Page 0 is most relevant, all remaining pages have descending
            # relevance.  Recent pages are deliberately low-scoring.
            value = float(count - logical)
            index.build_page(
                logical,
                torch.full((4, 2, 3), value),
                data_epoch=100 + logical,
                sealed=(logical < count - 1),
            )
            epochs[logical] = 100 + logical
        return index, epochs

    def test_total_budget_includes_recent_pages_and_never_overflows(self):
        index, epochs = self._index()
        result = index.select(
            torch.ones((2, 3)),
            RGKVBudget(total_page_budget=4, recent_pages=2),
            page_epochs=epochs,
        )

        # Relevant top-2 are pages 0/1.  Recent pages 6/7 were excluded from
        # scoring, merged afterwards, then restored to logical order.
        self.assertEqual(result.selected_logical_pages.tolist(), [0, 1, 6, 7])
        self.assertEqual(result.selected_count, 4)
        self.assertEqual(result.mandatory_recent_pages, 2)
        self.assertEqual(result.relevance_selected_pages, 2)
        self.assertLessEqual(result.selected_count, result.total_page_budget)

    def test_ties_are_deterministic_and_output_is_logically_ordered(self):
        index = RGKVIndex()
        epochs = {}
        for logical in (8, 2, 5, 1, 9):
            index.build_page(
                logical,
                torch.ones((2, 1, 2)),
                data_epoch=logical + 20,
            )
            epochs[logical] = logical + 20
        result = index.select(
            torch.ones((1, 2)),
            RGKVBudget(total_page_budget=3, recent_pages=1),
            page_epochs=epochs,
        )
        # Stable ties prefer earliest logical pages 1/2; page 9 is recent.
        self.assertEqual(result.selected_logical_pages.tolist(), [1, 2, 9])

    def test_budget_validation_and_small_candidate_set(self):
        with self.assertRaises(ValueError):
            RGKVBudget(3, recent_pages=4)
        index, epochs = self._index(count=2)
        result = index.select(
            torch.ones((2, 3)),
            RGKVBudget(8, recent_pages=4),
            page_epochs=epochs,
        )
        self.assertEqual(result.selected_logical_pages.tolist(), [0, 1])
        self.assertEqual(result.selected_count, 2)

    def test_zero_primitive_budget_returns_an_empty_selected_set(self):
        index, epochs = self._index(count=3)
        result = index.select(
            torch.ones((2, 3)),
            RGKVBudget(0, recent_pages=0),
            page_epochs=epochs,
        )
        self.assertEqual(result.selected_logical_pages.numel(), 0)
        self.assertEqual(result.selected_scores.numel(), 0)
        self.assertEqual(result.selected_count, 0)
        self.assertEqual(result.total_page_budget, 0)
        self.assertEqual(result.mandatory_recent_pages, 0)
        self.assertEqual(result.relevance_selected_pages, 0)


class RGKVIncrementalIndexTest(unittest.TestCase):
    def test_append_updates_only_changed_tail_without_historical_rows(self):
        index = RGKVIndex()
        first = index.build_page(
            0, torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]), data_epoch=1,
            sealed=True,
        )
        tail = index.build_page(
            1, torch.tensor([[[2.0, 6.0]], [[4.0, 8.0]]]), data_epoch=2,
        )
        first_storage = first.values.data_ptr()

        updated = index.append(
            1, torch.tensor([[[0.0, 10.0]]]), data_epoch=3
        )

        self.assertEqual(index.get(0).values.data_ptr(), first_storage)
        self.assertEqual(updated.valid_tokens, 3)
        torch.testing.assert_close(updated.minimum, torch.tensor([[0.0, 6.0]]))
        torch.testing.assert_close(updated.maximum, torch.tensor([[4.0, 10.0]]))
        torch.testing.assert_close(
            updated.mean, torch.tensor([[(2.0 + 4.0) / 3.0, 8.0]])
        )
        self.assertEqual(index.stats()["append_updates"], 1)
        self.assertTrue(index.get(0).sealed)
        self.assertFalse(tail.sealed)

    def test_sealed_freeze_rollback_cow_and_prefix_contracts(self):
        index = RGKVIndex()
        source = index.build_page(
            0, torch.arange(12, dtype=torch.float32).reshape(3, 2, 2),
            data_epoch=10,
        )
        sealed = index.seal(0, expected_data_epoch=10)
        self.assertTrue(sealed.sealed)
        with self.assertRaisesRegex(Exception, "immutable"):
            index.append(0, torch.ones((1, 2, 2)), data_epoch=11)

        shared = index.share_sealed_prefix(0, 1, data_epoch=10)
        self.assertTrue(shared.sealed)
        self.assertEqual(shared.values.data_ptr(), sealed.values.data_ptr())

        cow = index.cow_clone(0, 2, data_epoch=11)
        self.assertFalse(cow.sealed)
        self.assertNotEqual(cow.values.data_ptr(), source.values.data_ptr())
        rebuilt = index.rollback_tail(
            2,
            torch.full((2, 2, 2), 7.0),
            data_epoch=12,
        )
        self.assertEqual(rebuilt.valid_tokens, 2)
        torch.testing.assert_close(rebuilt.mean, torch.full((2, 2), 7.0))
        self.assertEqual(index.stats()["rollback_rebuilds"], 1)

    def test_epoch_mismatch_is_explicit_stale_index(self):
        index = RGKVIndex()
        index.build_page(0, torch.ones((2, 1, 2)), data_epoch=7)
        with self.assertRaises(RGKVStaleIndexError) as caught:
            index.select(
                torch.ones((1, 2)),
                RGKVBudget(1),
                page_epochs={0: 8},
            )
        self.assertEqual(caught.exception.code, STALE_INDEX)
        self.assertIn(STALE_INDEX, str(caught.exception))
        self.assertEqual(index.stats()["stale_index_count"], 1)

    def test_summary_is_contiguous_and_retains_head_dimension(self):
        index = RGKVIndex()
        summary = index.build_page(
            0, torch.randn(5, 4, 8), data_epoch=3
        )
        self.assertEqual(tuple(summary.values.shape), (3, 4, 8))
        self.assertTrue(summary.values.is_contiguous())
        packed = index.packed_summaries()
        self.assertEqual(tuple(packed.shape), (1, 3, 4, 8))
        self.assertTrue(packed.is_contiguous())


if __name__ == "__main__":
    unittest.main()
