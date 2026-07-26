import unittest

import torch

from layer_streaming.vocab import merge_topk


class VocabTopKTest(unittest.TestCase):
    def test_online_merge_matches_full_topk(self):
        logits = torch.tensor(
            [[[-2.0, 7.0, 1.0, 3.0, 9.0, -1.0, 8.0, 4.0]]]
        )
        values = None
        indices = None
        for start in (0, 3, 6):
            partial = logits[..., start : start + 3]
            local_values, local_indices = torch.topk(
                partial,
                k=min(3, partial.shape[-1]),
                dim=-1,
            )
            values, indices = merge_topk(
                values,
                indices,
                local_values,
                local_indices + start,
                top_k=3,
            )
        expected_values, expected_indices = torch.topk(
            logits,
            k=3,
            dim=-1,
        )
        self.assertTrue(torch.equal(values, expected_values))
        self.assertTrue(torch.equal(indices, expected_indices))


if __name__ == "__main__":
    unittest.main()
