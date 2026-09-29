import copy
import unittest

import torch
from torch import nn
from transformers import OlmoeConfig, OlmoeForCausalLM

from layer_streaming.moe import (
    AdaptiveExpertBackend,
    ExpertDispatcher,
    ExpertWeights,
    GroupedExpertBackend,
    MoEConfig,
    MoEFFN,
    NaiveExpertBackend,
    Router,
    StackedExpertWeights,
)


def tiny_config(layers=2):
    return OlmoeConfig(
        vocab_size=67,
        hidden_size=24,
        intermediate_size=12,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        attention_bias=False,
        tie_word_embeddings=False,
        norm_topk_prob=False,
        torch_dtype="float32",
    )


def weights_from_hf(block):
    values = [
        ExpertWeights(
            expert.gate_proj.weight,
            expert.up_proj.weight,
            expert.down_proj.weight,
        )
        for expert in block.experts
    ]
    return values, StackedExpertWeights.from_experts(values)


def moe_config_from_hf(config):
    return MoEConfig(
        num_experts=config.num_experts,
        experts_per_token=config.num_experts_per_tok,
        expert_intermediate_size=config.intermediate_size,
        router_dtype="float32",
        normalize_topk=config.norm_topk_prob,
    )


class CascadeMoEBlock(nn.Module):
    def __init__(self, hf_block, config, backend):
        super().__init__()
        # Keep parameters registered so model dtype/device movement remains valid.
        self.gate = hf_block.gate
        self.experts = hf_block.experts
        values, stacked = weights_from_hf(self)
        weights = values if isinstance(backend, NaiveExpertBackend) else stacked
        self.runtime = MoEFFN(
            Router(moe_config_from_hf(config), self.gate.weight),
            ExpertDispatcher(hf_block.num_experts),
            backend,
            weights,
        )

    def forward(self, hidden_states):
        output, routing, _plan = self.runtime(hidden_states)
        return output, routing.router_logits


class ExpertBackendTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(997)
        self.config = tiny_config(layers=1)
        self.reference = OlmoeForCausalLM(self.config).model.layers[0].mlp
        self.hidden = torch.randn(2, 5, self.config.hidden_size)

    def _runtime(self, backend, stacked=False):
        values, stacked_values = weights_from_hf(self.reference)
        return MoEFFN(
            Router(moe_config_from_hf(self.config), self.reference.gate.weight),
            ExpertDispatcher(self.config.num_experts),
            backend,
            stacked_values if stacked else {i: value for i, value in enumerate(values)},
        )

    def test_naive_matches_hf_expert_output_router_and_ids(self):
        expected, logits = self.reference(self.hidden)
        actual, routing, _plan = self._runtime(NaiveExpertBackend())(self.hidden)
        torch.testing.assert_close(routing.router_logits, logits)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_grouped_matches_naive_across_batch_sizes(self):
        generator = torch.Generator().manual_seed(251)
        naive = self._runtime(NaiveExpertBackend())
        grouped = self._runtime(GroupedExpertBackend(), stacked=True)
        for tokens in (1, 2, 8, 32):
            hidden = torch.randn(tokens, self.config.hidden_size, generator=generator)
            expected, expected_routing, _ = naive(hidden)
            actual, actual_routing, _ = grouped(hidden)
            self.assertTrue(
                torch.equal(expected_routing.expert_ids, actual_routing.expert_ids)
            )
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_adaptive_backend_keeps_tiny_reference_path(self):
        _values, stacked = weights_from_hf(self.reference)
        backend = AdaptiveExpertBackend(grouped_min_assignments=8)
        runtime = MoEFFN(
            Router(moe_config_from_hf(self.config), self.reference.gate.weight),
            ExpertDispatcher(self.config.num_experts),
            backend,
            stacked,
        )
        one = runtime(torch.randn(1, self.config.hidden_size))[2]
        many = runtime(torch.randn(8, self.config.hidden_size))[2]
        self.assertIs(backend.choose(one, stacked), backend.naive)
        self.assertIs(backend.choose(many, stacked), backend.grouped)


class FunctionalModelTest(unittest.TestCase):
    def test_multi_layer_logits_and_greedy_decode_match_hf(self):
        torch.manual_seed(1234)
        config = tiny_config(layers=3)
        reference = OlmoeForCausalLM(config).eval()
        candidate = copy.deepcopy(reference).eval()
        for layer in candidate.model.layers:
            layer.mlp = CascadeMoEBlock(layer.mlp, config, GroupedExpertBackend())

        input_ids = torch.tensor([[1, 9, 17, 4], [3, 7, 2, 5]])
        with torch.no_grad():
            expected = reference(input_ids).logits
            actual = candidate(input_ids).logits
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        self.assertTrue(torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1)))

        reference_ids = input_ids[:1]
        candidate_ids = input_ids[:1]
        with torch.no_grad():
            for _step in range(5):
                expected_next = reference(reference_ids).logits[:, -1].argmax(-1, keepdim=True)
                actual_next = candidate(candidate_ids).logits[:, -1].argmax(-1, keepdim=True)
                self.assertTrue(torch.equal(actual_next, expected_next))
                reference_ids = torch.cat((reference_ids, expected_next), dim=1)
                candidate_ids = torch.cat((candidate_ids, actual_next), dim=1)
        self.assertTrue(torch.equal(candidate_ids, reference_ids))


if __name__ == "__main__":
    unittest.main()
