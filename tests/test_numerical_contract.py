import json
from pathlib import Path
import unittest

from layer_streaming.numerical_contract import evaluate_fused_diagnostic


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures/fused_w8a16_sm86_golden_v1.json"
)


def diagnostic(max_local_relative=0.002, logits_topk=0.9):
    linear = {
        "fused_vs_fallback": {
            "elements_over_allclose_bound": 0,
            "max_abs_error": 0.125,
            "mean_relative_error": max_local_relative,
        },
        "fused_vs_fp32_accumulation": {
            "mean_relative_error": 1.0e-5,
        },
        "weight": {
            "storage_dtype": "int8",
            "compute_dtype": "bfloat16",
            "quantization": {"granularity": "per_channel"},
            "has_storage_padding": False,
            "k_aligned_16": True,
            "n_aligned_8": True,
        },
        "input": {"dtype": "bfloat16"},
        "output": {"dtype": "bfloat16"},
        "scale": {"dtype": "bfloat16"},
    }
    propagation = []
    values = {
        "attention": (0.05, 0.003),
        "mlp": (0.15, 0.01),
        "hidden": (0.25, 0.016),
        "final_norm": (0.5, 0.055),
        "logits": (0.375, 0.057),
    }
    for stage, (maximum, mean) in values.items():
        item = {
            "stage": "decode_0/{}".format(stage),
            "max_abs_error": maximum,
            "mean_abs_error": mean,
        }
        if stage == "logits":
            item.update(
                top1_equal=True, topk_consistency=logits_topk
            )
        propagation.append(item)
    return {
        "provider": "cutlass_sm86_w8a16",
        "provider_abi": 2,
        "model_signature": {
            "model_type": "llama",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "vocab_size": 128256,
        },
        "configuration": {
            "input_ids": [128000, 128006, 882, 220],
            "decode_ids": [128001, 128001],
        },
        "linear_diagnostics": [linear] * 448,
        "error_propagation": (
            [item for item in propagation if "/attention" in item["stage"]] * 64
            + [item for item in propagation if "/mlp" in item["stage"]] * 64
            + [item for item in propagation if "/hidden" in item["stage"]] * 64
            + [item for item in propagation if "/final_norm" in item["stage"]] * 2
            + [item for item in propagation if "/logits" in item["stage"]] * 2
        ),
    }


class NumericalContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_evidence_envelope_passes(self):
        result = evaluate_fused_diagnostic(
            diagnostic(), self.contract
        )
        self.assertTrue(result["passed"], result["violations"])

    def test_local_and_topk_regressions_fail(self):
        result = evaluate_fused_diagnostic(
            diagnostic(max_local_relative=0.01, logits_topk=0.8),
            self.contract,
        )
        self.assertFalse(result["passed"])
        self.assertTrue(
            any("local_mean_relative" in item for item in result["violations"])
        )
        self.assertTrue(
            any("Top-k" in item for item in result["violations"])
        )


if __name__ == "__main__":
    unittest.main()
