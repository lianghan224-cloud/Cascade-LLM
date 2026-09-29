import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from layer_streaming import KVPolicy, PagedKVRuntime
from layer_streaming.attention.paged import reference as paged_reference


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "qualify_gather_sdpa_prefill.py"
SPEC = importlib.util.spec_from_file_location(
    "qualify_gather_sdpa_prefill", TOOL_PATH
)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class GatherSDPAPrefillQualificationTest(unittest.TestCase):
    def args(self, *extra):
        return TOOL.build_parser().parse_args(list(extra))

    def test_logic_matrix_uses_provider_workspace_contract(self):
        args = TOOL.validate_args(self.args("--mode", "logic"))
        cases = [TOOL.run_logic_case(args, length) for length in args.lengths]
        self.assertEqual(tuple(case["length"] for case in cases), TOOL.REQUIRED_LENGTHS)
        self.assertTrue(all(case["status"] == TOOL.PASS for case in cases))
        estimates = [case["workspace"]["provider_estimate_bytes"] for case in cases]
        self.assertEqual(estimates, sorted(estimates))
        self.assertTrue(all(value > 0 for value in estimates))
        for case in cases:
            self.assertEqual(
                case["routing"]["selected"], TOOL.CANDIDATE_PROVIDER
            )
            self.assertFalse(case["routing"]["execution_observed"])

    def test_workspace_underestimate_is_a_strict_failure(self):
        passed = TOOL.workspace_gate(4096, 4096)
        failed = TOOL.workspace_gate(4096, 4097)
        self.assertTrue(passed["passed"])
        self.assertFalse(passed["planner_underestimated"])
        self.assertFalse(failed["passed"])
        self.assertTrue(failed["planner_underestimated"])

    def test_workspace_formula_covers_explicit_allocations_and_local_guard(self):
        breakdown = paged_reference._gather_sdpa_workspace_breakdown(
            batch_size=1,
            max_sequence_length=128,
            max_query_length=128,
            total_query_tokens=128,
            num_query_heads=8,
            num_kv_heads=2,
            head_dim=128,
            dtype_bytes=2,
            explicit_mask=True,
        )
        explicit_terms = sum(
            breakdown[name]
            for name in (
                "gather_kv_bytes",
                "gqa_expanded_kv_bytes",
                "output_bytes",
                "positions_bytes",
                "mask_bytes",
                "score_matrix_bytes",
                "logsumexp_bytes",
            )
        )
        self.assertEqual(breakdown["explicit_bytes"], explicit_terms)
        self.assertEqual(
            breakdown["total_bytes"],
            explicit_terms + breakdown["sdpa_allocator_guard_bytes"],
        )
        self.assertGreaterEqual(
            breakdown["sdpa_allocator_guard_bytes"], 256 * 1024
        )

    def test_full_prefill_skips_mask_builder_but_chunked_uses_it(self):
        runtime = PagedKVRuntime(
            layer_count=1,
            num_query_heads=4,
            num_kv_heads=2,
            head_dim=8,
            page_count=4,
            page_size=16,
            dtype=torch.bfloat16,
            device="cpu",
            policy=KVPolicy(
                dtype="bf16",
                page_size=16,
                attention_backend=TOOL.CANDIDATE_PROVIDER,
            ),
            prefill_backend=TOOL.CANDIDATE_PROVIDER,
        )
        try:
            state = runtime.create_request(32)
            key = torch.randn(8, 2, 8, dtype=torch.bfloat16)
            value = torch.randn_like(key)
            runtime.append((state,), 0, key, value, (8,))
            full_query = torch.randn(8, 4, 8, dtype=torch.bfloat16)
            with mock.patch.object(
                paged_reference,
                "_build_explicit_causal_mask",
                side_effect=AssertionError("full prefill built a mask"),
            ):
                full = runtime.attend(
                    (state,), 0, full_query, (8,),
                    query_positions=(torch.arange(8),), phase="prefill",
                )
            self.assertEqual(full.provider_metrics["workspace_breakdown"]["mask_bytes"], 0)
            self.assertEqual(
                full.provider_metrics["workspace_breakdown"]["positions_bytes"], 0
            )
            chunk_query = torch.randn(5, 4, 8, dtype=torch.bfloat16)
            original = paged_reference._build_explicit_causal_mask
            with mock.patch.object(
                paged_reference,
                "_build_explicit_causal_mask",
                wraps=original,
            ) as mask_builder:
                chunked = runtime.attend(
                    (state,), 0, chunk_query, (5,),
                    query_positions=(torch.arange(3, 8),), phase="prefill",
                )
            mask_builder.assert_called_once()
            self.assertGreater(
                chunked.provider_metrics["workspace_breakdown"]["mask_bytes"], 0
            )
            self.assertGreater(
                chunked.provider_metrics["workspace_breakdown"]["positions_bytes"], 0
            )
        finally:
            runtime.close()

    def test_qualification_cannot_bypass_exclusive_admission(self):
        args = self.args(
            "--mode", "qualification", "--allow-shared-smoke"
        )
        with self.assertRaisesRegex(ValueError, "cannot bypass"):
            TOOL.validate_args(args)

    def test_real_model_handoff_never_claims_execution(self):
        case = TOOL.real_model_ab_case(self.args("--mode", "logic"))
        self.assertEqual(case["status"], TOOL.SKIPPED)
        self.assertIn("qualify_llama70b_single_request.py", case["suggested_command"])
        self.assertEqual(
            case["providers"],
            [TOOL.REFERENCE_PROVIDER, TOOL.CANDIDATE_PROVIDER],
        )

    def test_synthetic_qualification_can_never_claim_qualified(self):
        args = TOOL.validate_args(self.args("--mode", "qualification"))
        cases = [{
            "case_id": "synthetic",
            "kind": "synthetic_cuda_prefill",
            "status": TOOL.PASS,
        }]
        summary = TOOL.build_summary(args, cases, {"exclusive_snapshot": True})
        self.assertEqual(summary["capability_state"], "QUALIFICATION_READY")
        self.assertNotEqual(summary["capability_state"], "QUALIFIED")
        self.assertFalse(summary["qualification_ready_is_qualified"])

    def test_logic_report_bundle_has_common_four_file_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(
                "--mode", "logic", "--output-dir", directory,
                "--lengths", "128,512,2048,8192",
            )
            summary, cases, paths = TOOL.run(args)
            self.assertEqual(summary["overall"], TOOL.PASS)
            self.assertEqual(
                summary["capability_state"], "LOGIC_VALIDATED"
            )
            self.assertFalse(summary["qualification_ready_is_qualified"])
            self.assertEqual(
                set(paths),
                {"environment.json", "cases.json", "summary.json", "report.md"},
            )
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            cases_document = json.loads(
                Path(paths["cases.json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(cases_document["schema_version"], TOOL.SCHEMA_VERSION)
            self.assertEqual(len(cases_document["cases"]), len(cases))
            handoff = cases_document["cases"][-1]
            self.assertEqual(handoff["case_id"], "real-70b-prefill-ab")
            self.assertEqual(handoff["status"], TOOL.SKIPPED)


if __name__ == "__main__":
    unittest.main()
