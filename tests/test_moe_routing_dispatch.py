import unittest

import torch
import torch.nn.functional as F

from layer_streaming.moe import (
    ExpertDispatcher,
    MoEConfig,
    Router,
    RoutingResult,
)


class RouterTest(unittest.TestCase):
    def _config(self, normalize=False, scale=1.0):
        return MoEConfig(
            num_experts=6,
            experts_per_token=2,
            expert_intermediate_size=12,
            router_dtype="float32",
            normalize_topk=normalize,
            routed_scaling_factor=scale,
        )

    def test_router_matches_reference_for_single_and_multiple_tokens(self):
        generator = torch.Generator().manual_seed(19)
        weight = torch.randn(6, 8, generator=generator)
        for token_count in (1, 3, 11):
            hidden = torch.randn(token_count, 8, generator=generator)
            for normalize, scale in ((False, 1.0), (True, 1.0), (True, 1.7)):
                result = Router(self._config(normalize, scale), weight)(hidden)
                logits = F.linear(hidden, weight)
                probabilities = F.softmax(logits, dim=-1, dtype=torch.float32)
                expected_weight, expected_ids = torch.topk(probabilities, 2, dim=-1)
                if normalize:
                    expected_weight /= expected_weight.sum(dim=-1, keepdim=True)
                expected_weight *= scale
                torch.testing.assert_close(result.router_logits, logits)
                self.assertTrue(torch.equal(result.expert_ids, expected_ids))
                torch.testing.assert_close(result.expert_weights, expected_weight)

    def test_router_is_stable_across_layers_and_decode_steps(self):
        generator = torch.Generator().manual_seed(73)
        hidden = torch.randn(1, 8, generator=generator)
        for _layer in range(3):
            weight = torch.randn(6, 8, generator=generator)
            router = Router(self._config(normalize=True), weight)
            for _step in range(4):
                first = router(hidden)
                second = router(hidden)
                self.assertTrue(torch.equal(first.expert_ids, second.expert_ids))
                torch.testing.assert_close(first.expert_weights, second.expert_weights)
                hidden = hidden + 0.01

    def test_routing_result_rejects_inconsistent_shapes(self):
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            RoutingResult(
                expert_ids=torch.zeros((2, 2), dtype=torch.long),
                expert_weights=torch.zeros((2, 1)),
            )


class ExpertDispatcherTest(unittest.TestCase):
    @staticmethod
    def _routing(token_count):
        ids = torch.tensor(
            [[(token * 3) % 5, (token * 3 + 2) % 5] for token in range(token_count)],
            dtype=torch.long,
        )
        weights = torch.arange(1, token_count * 2 + 1, dtype=torch.float32).reshape(
            token_count, 2
        )
        weights /= weights.sum(dim=-1, keepdim=True)
        return RoutingResult(ids, weights)

    @staticmethod
    def _expert_outputs(hidden, plan):
        dispatched = ExpertDispatcher.dispatch_hidden(hidden, plan)
        scale = plan.assignment_expert_ids.to(hidden.dtype)[:, None] + 1.0
        return dispatched * scale

    def test_small_path_matches_direct_weighted_reference(self):
        routing = self._routing(1)
        hidden = torch.tensor([[1.0, -2.0, 4.0]])
        plan = ExpertDispatcher(5)(routing)
        self.assertTrue(plan.small_batch_fast_path)
        actual = ExpertDispatcher.combine(self._expert_outputs(hidden, plan), plan)
        expected = torch.zeros_like(hidden)
        for slot in range(2):
            expected[0] += (
                hidden[0]
                * float(routing.expert_ids[0, slot] + 1)
                * routing.expert_weights[0, slot]
            )
        torch.testing.assert_close(actual, expected)
        buckets = plan.materialize_buckets()
        self.assertEqual(sum(item.token_ids.numel() for item in buckets), 2)

    def test_general_and_small_paths_are_numerically_identical(self):
        generator = torch.Generator().manual_seed(101)
        for token_count in (1, 2, 4, 8, 32):
            routing = self._routing(token_count)
            hidden = torch.randn(token_count, 7, generator=generator)
            dispatcher = ExpertDispatcher(5, small_batch_threshold=4)
            fast = dispatcher(routing)
            general = dispatcher(routing, force_general=True)
            fast_output = dispatcher.combine(self._expert_outputs(hidden, fast), fast)
            general_output = dispatcher.combine(
                self._expert_outputs(hidden, general), general
            )
            torch.testing.assert_close(fast_output, general_output)
            self.assertEqual(fast.assignment_count, token_count * 2)
            self.assertEqual(general.assignment_count, token_count * 2)
            self.assertEqual(
                sum(item.token_ids.numel() for item in general.materialize_buckets()),
                token_count * 2,
            )

    def test_combine_validates_assignment_shape(self):
        plan = ExpertDispatcher(5)(self._routing(2))
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            ExpertDispatcher.combine(torch.zeros(3, 4), plan)


if __name__ == "__main__":
    unittest.main()
