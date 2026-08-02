import json
from pathlib import Path
import unittest

import torch

from layer_streaming.attention.paged.numerical_contract import (
    default_paged_numerical_contract,
)
from layer_streaming.kv.reports import (
    KVCompatibilityReport,
    KVNumericalReport,
    KVOwnershipReport,
    KVPerformanceReport,
    KVQualificationReport,
)
from layer_streaming.numerical_contracts import (
    evaluate_logits_generation,
    evaluate_model_quality,
    evaluate_model_stages,
    evaluate_production_stability,
    load_kv_numerical_contract,
)
from layer_streaming.qualification_status import (
    QualificationStatus,
    qualification_status,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "tests/fixtures/kv_numerical_sm86_bf16_abi1_v2.json"


def model_comparison():
    comparisons = {}
    limits = {
        "attention": (0.10, 0.002),
        "hidden": (0.40, 0.020),
        "mlp": (0.30, 0.015),
        "final_norm": (0.80, 0.040),
        "logits": (0.50, 0.050),
    }
    for component, values in limits.items():
        comparisons["layer_0/{}/stage".format(component)] = {
            "max_abs_error": values[0],
            "mean_abs_error": values[1],
            "max_relative_error": values[0],
            "top1_equal": True,
            "topk_consistency": 1.0,
        }
    return {"comparisons": comparisons}


class PagedNumericalContractV2Test(unittest.TestCase):
    def test_l0_l1_compare_against_fp32_and_native_baseline(self):
        reference = torch.tensor(
            [[1.001, -0.499], [0.125, 2.003]], dtype=torch.float32
        )
        baseline = reference.to(torch.bfloat16)
        candidate = baseline.clone()
        contract = default_paged_numerical_contract(
            "sm86", "sm86", "bf16", provider_abi=1
        )
        result = contract.evaluate(
            reference,
            candidate,
            baseline=baseline,
            safety_checks={
                "no_out_of_bounds_access": True,
                "page_block_indices_valid": True,
            },
        )
        self.assertTrue(result["l0"]["passed"])
        self.assertTrue(result["l1"]["passed"])
        self.assertEqual(result["reference"], "fp32_math_attention")
        self.assertIn("p99_abs_error", result["l1"]["candidate_vs_fp32"])

    def test_l0_rejects_missing_memory_safety_evidence(self):
        reference = torch.ones(2, dtype=torch.float32)
        candidate = reference.to(torch.bfloat16)
        result = default_paged_numerical_contract(
            "sm86", "sm86", "bf16"
        ).evaluate(reference, candidate, baseline=candidate)
        self.assertFalse(result["passed"])
        self.assertFalse(result["l0"]["checks"]["no_out_of_bounds_access"])

    def test_l1_rejects_error_above_native_envelope(self):
        reference = torch.tensor([1.001, 2.001], dtype=torch.float32)
        baseline = reference.to(torch.bfloat16)
        candidate = torch.tensor([1.5, 2.5], dtype=torch.bfloat16)
        result = default_paged_numerical_contract(
            "sm86", "sm86", "bf16"
        ).evaluate(
            reference,
            candidate,
            baseline=baseline,
            safety_checks={
                "no_out_of_bounds_access": True,
                "page_block_indices_valid": True,
            },
        )
        self.assertTrue(result["l0"]["passed"])
        self.assertFalse(result["l1"]["passed"])


class ModelNumericalContractV2Test(unittest.TestCase):
    def setUp(self):
        self.contract = load_kv_numerical_contract(CONTRACT)

    def test_l2_l3_and_l4_use_independent_evidence(self):
        comparison = model_comparison()
        l2 = evaluate_model_stages(
            self.contract,
            comparison,
            {
                "production_failed_stage_count": 23,
                "conclusion": {"all_failed_stages_attributed": True},
            },
        )
        self.assertTrue(l2["passed"])
        self.assertEqual(l2["strict_hf_diagnostic_failure_count"], 23)
        l3 = evaluate_logits_generation(
            self.contract,
            comparison,
            {
                "top1_agreement_rate": 1.0,
                "sampled_positions": 1000,
                "all_hf_logits_finite": True,
            },
        )
        self.assertTrue(l3["passed"])
        self.assertFalse(evaluate_model_quality(self.contract)["passed"])
        quality = evaluate_model_quality(
            self.contract,
            {
                "all_finite": True,
                "coverage": {
                    "perplexity_tokens": 64,
                    "short_examples": 8,
                    "long_context_examples": 4,
                    "dialogue_examples": 6,
                },
                "metrics": {
                    "perplexity_relative_degradation": 0.0005,
                    "short_accuracy_drop_pp": 0.1,
                    "long_hit_rate_drop_pp": 0.1,
                    "dialogue_accuracy_drop_pp": 0.1,
                }
            },
        )
        self.assertTrue(quality["passed"])
        quality["coverage"]["perplexity_tokens"] = 63
        self.assertFalse(
            evaluate_model_quality(self.contract, quality)["passed"]
        )

    def test_l5_requires_zero_drift_and_ownership(self):
        soak = {
            "decode_tokens_per_cycle": 1000,
            "resource_drift": {
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
            },
            "acceptance": {
                "all_pages_released": True,
                "long_request_ref_count_exact": True,
                "quiescent_pin_count_zero": True,
            },
            "latency_ms": {"last_vs_first_quartile_ratio": 1.01},
        }
        ownership = {
            "resource_drift": {
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 0,
            },
            "allocator_cache_before_trim": {
                "cuda_allocated_bytes": 0,
                "cuda_reserved_bytes": 2 * 1024 * 1024,
                "steady_cycle_reserved_span_bytes": 0,
            },
            "acceptance": {
                "fork": True,
                "cow": True,
                "prefix": True,
                "speculative": True,
                "allocator_cache_released_after_trim": True,
            }
        }
        performance = [{
            "results": [{
                "supported": True,
                "provider": "sm86",
                "workspace_peak_bytes": 0,
            }]
        }]
        result = evaluate_production_stability(
            self.contract, soak, ownership, performance
        )
        self.assertTrue(result["passed"])
        soak["resource_drift"]["cuda_reserved_bytes"] = 1
        self.assertFalse(
            evaluate_production_stability(
                self.contract, soak, ownership, performance
            )["passed"]
        )


class QualificationSchemaTest(unittest.TestCase):
    def test_shared_status_vocabulary_and_report_serialization(self):
        self.assertEqual(
            qualification_status(QualificationStatus.PRODUCTION), "production"
        )
        with self.assertRaises(ValueError):
            qualification_status("unqualified")
        reports = (
            KVCompatibilityReport("sm86", "sm86", "smoke_passed"),
            KVNumericalReport("sm86", "sm86", "numerically_qualified"),
            KVOwnershipReport("sm86", "sm86", "smoke_passed"),
            KVPerformanceReport("sm86", "sm86", "performance_qualified"),
            KVQualificationReport("sm86", "sm86", "production"),
        )
        for report in reports:
            restored = json.loads(report.to_json())
            self.assertEqual(restored["schema_version"], 1)
            self.assertEqual(restored["status"], report.status)
            round_trip = type(report).from_json(report.to_json())
            self.assertEqual(round_trip, report)


if __name__ == "__main__":
    unittest.main()
