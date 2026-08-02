import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from layer_streaming import (
    ExecutionPolicy,
    Llama31DecodeExecutor,
    LlamaModelAdapter,
    PlacementMode,
)


class ResidentState:
    def __init__(self, state):
        self.device = torch.device("cpu")
        self.state = state

    def __getitem__(self, key):
        return self.state[key]


class LlamaReferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.config = LlamaConfig(
            vocab_size=97,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
            torch_dtype="bfloat16",
            bos_token_id=1,
            eos_token_id=2,
        )
        self.model = LlamaForCausalLM(self.config).to(torch.bfloat16).eval()
        self.state = self.model.state_dict()
        self.plan = LlamaModelAdapter().build_plan(
            self.config,
            ExecutionPolicy(
                embedding_mode=PlacementMode.RESIDENT,
                lm_head_mode=PlacementMode.RESIDENT,
                vocab_chunk_bytes=4096,
            ),
        )
        self.executor = Llama31DecodeExecutor(
            self.config,
            ResidentState(self.state),
            top_k=10,
            return_full_logits=True,
            max_cache_length=16,
            kv_block_size=16,
            allow_kv_reference=True,
        )

    def streamed_step(self, input_ids):
        state = self.executor.begin(input_ids)
        for unit in self.plan.units:
            weights = {
                piece.tensor.key: self.state[piece.tensor.key]
                for piece in unit.pieces
            }
            state = self.executor(unit, weights, state)
        return self.executor.finish(state)

    def test_prefill_and_decode_match_hugging_face(self):
        prefill = torch.tensor([[1, 13, 7, 22]], dtype=torch.long)
        decode = torch.tensor([[31]], dtype=torch.long)
        with torch.inference_mode():
            reference_prefill = self.model(prefill, use_cache=True)
            streamed_prefill = self.streamed_step(prefill)
            reference_decode = self.model(
                decode,
                past_key_values=reference_prefill.past_key_values,
                use_cache=True,
            )
            streamed_decode = self.streamed_step(decode)
        self.assertTrue(
            torch.allclose(
                streamed_prefill.logits,
                reference_prefill.logits.float(),
                atol=5e-2,
                rtol=5e-2,
            ),
            (streamed_prefill.logits - reference_prefill.logits.float()).abs().max(),
        )
        self.assertTrue(
            torch.allclose(
                streamed_decode.logits,
                reference_decode.logits.float(),
                atol=5e-2,
                rtol=5e-2,
            ),
            (streamed_decode.logits - reference_decode.logits.float()).abs().max(),
        )
        expected_topk = torch.topk(reference_decode.logits.float(), 10, dim=-1).indices
        self.assertTrue(torch.equal(streamed_decode.topk_indices, expected_topk))

    def tearDown(self):
        self.executor.close()


if __name__ == "__main__":
    unittest.main()
