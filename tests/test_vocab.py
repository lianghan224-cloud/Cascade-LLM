import unittest

import torch

from layer_streaming.vocab import deterministic_topk, merge_topk


class DeterministicTopKTest(unittest.TestCase):
    def test_full_topk_breaks_equal_values_by_lower_index(self):
        values = torch.tensor([[3.0, 3.0, 2.0, 3.0]])

        selected_values, selected_indices = deterministic_topk(values, 3)

        self.assertTrue(
            torch.equal(selected_values, torch.tensor([[3.0, 3.0, 3.0]]))
        )
        self.assertTrue(
            torch.equal(selected_indices, torch.tensor([[0, 1, 3]]))
        )

    def test_chunk_merge_uses_the_same_tie_breaking_rule(self):
        current_values = torch.tensor([[3.0, 2.0]])
        current_indices = torch.tensor([[10, 7]])
        candidate_values = torch.tensor([[3.0, 3.0]])
        candidate_indices = torch.tensor([[2, 5]])

        selected_values, selected_indices = merge_topk(
            current_values,
            current_indices,
            candidate_values,
            candidate_indices,
            3,
        )

        self.assertTrue(
            torch.equal(selected_values, torch.tensor([[3.0, 3.0, 3.0]]))
        )
        self.assertTrue(
            torch.equal(selected_indices, torch.tensor([[2, 5, 10]]))
        )


if __name__ == "__main__":
    unittest.main()
