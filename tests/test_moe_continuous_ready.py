import unittest

import torch

from test_moe_backends import (
    GroupedExpertBackend,
    NaiveExpertBackend,
    Router,
    ExpertDispatcher,
    MoEFFN,
    OlmoeForCausalLM,
    tiny_config,
    weights_from_hf,
    moe_config_from_hf,
)


class ContinuousBatchingReadyInterfaceTest(unittest.TestCase):
    def test_cross_request_iteration_token_batch_sizes(self):
        torch.manual_seed(620)
        config = tiny_config(layers=1)
        block = OlmoeForCausalLM(config).model.layers[0].mlp
        values, stacked = weights_from_hf(block)
        naive = MoEFFN(
            Router(moe_config_from_hf(config), block.gate.weight),
            ExpertDispatcher(config.num_experts),
            NaiveExpertBackend(),
            {index: value for index, value in enumerate(values)},
        )
        grouped = MoEFFN(
            Router(moe_config_from_hf(config), block.gate.weight),
            ExpertDispatcher(config.num_experts),
            GroupedExpertBackend(),
            stacked,
        )
        for iteration_tokens in (1, 2, 4, 8, 32):
            # Each row represents one active request's iteration token.
            hidden = torch.randn(iteration_tokens, config.hidden_size)
            expected, expected_routing, expected_plan = naive(hidden)
            actual, actual_routing, actual_plan = grouped(hidden)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            self.assertTrue(
                torch.equal(actual_routing.expert_ids, expected_routing.expert_ids)
            )
            self.assertEqual(actual_plan.token_count, iteration_tokens)
            self.assertEqual(actual_plan.assignment_count, iteration_tokens * 2)
            self.assertEqual(expected_plan.token_count, iteration_tokens)


if __name__ == "__main__":
    unittest.main()
