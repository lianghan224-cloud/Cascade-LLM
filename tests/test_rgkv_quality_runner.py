import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import qualify_rgkv_quality as runner


def passing_metrics(level):
    return {
        "selection": {
            "ranking_agreement": 0.99,
            "topk_overlap": 0.95,
            "selected_page_agreement": 0.96,
        },
        "attention": {
            "max_abs_error": 0.02,
            "mean_abs_error": 0.005,
            "cosine_similarity": 0.999,
        },
        "logit": {
            "top1_match_rate": 0.98,
            "topk_overlap": 0.95,
            "logit_cosine": 0.999,
            "kl_divergence": 0.01,
        },
        "ppl": {
            "dense_perplexity": 10.0,
            "rgkv_perplexity": 10.4,
        },
        "long_context": {
            "needle_recall_early": 1.0,
            "needle_recall_middle": 1.0,
            "needle_recall_late": 1.0,
            "retrieval_accuracy": 0.95,
            "long_qa_score": 0.92,
            "multi_turn_score": 0.94,
        },
    }[level]


class RGKVQualityRunnerTest(unittest.TestCase):
    def test_plan_has_five_levels_and_required_budget_sweep(self):
        cases = runner.planned_cases()
        self.assertEqual(len(cases), 5 * 4 * 4)
        self.assertEqual({case["level"] for case in cases}, set(runner.QUALITY_LEVELS))
        self.assertEqual({case["budget_ratio"] for case in cases}, set(runner.BUDGET_RATIOS))
        self.assertEqual({case["recent_pages"] for case in cases}, set(runner.RECENT_PAGES))

    def test_missing_evidence_is_not_run_and_never_passes(self):
        cases = runner.aggregate_cases([])
        self.assertTrue(all(case["status"] == runner.NOT_RUN for case in cases))
        self.assertTrue(all(not case["gate"]["passed"] for case in cases))

    def test_supplied_pass_cannot_hide_missing_metrics(self):
        cases = runner.aggregate_cases(
            [{
                "level": "selection",
                "budget_ratio": 0.75,
                "recent_pages": 2,
                "status": "PASS",
                "metrics": {"ranking_agreement": 1.0},
            }]
        )
        first = next(case for case in cases if case["level"] == "selection" and case["budget_ratio"] == 0.75 and case["recent_pages"] == 2)
        self.assertEqual(first["status"], runner.FAIL)
        self.assertEqual(
            set(first["gate"]["missing_metrics"]),
            {"topk_overlap", "selected_page_agreement"},
        )

    def test_nonfinite_out_of_range_and_unknown_status_are_rejected(self):
        nonfinite = runner.evaluate_evidence(
            "selection",
            {
                "ranking_agreement": float("nan"),
                "topk_overlap": 0.95,
                "selected_page_agreement": 0.95,
            },
        )
        self.assertEqual(nonfinite["status"], runner.FAIL)
        self.assertIn("ranking_agreement", nonfinite["gate"]["invalid_metrics"])
        out_of_range = runner.evaluate_evidence(
            "logit",
            {
                "top1_match_rate": 1.2,
                "topk_overlap": 0.95,
                "logit_cosine": 0.99,
                "kl_divergence": -0.1,
            },
        )
        self.assertEqual(out_of_range["status"], runner.FAIL)
        self.assertEqual(
            set(out_of_range["gate"]["invalid_metrics"]),
            {"top1_match_rate", "kl_divergence"},
        )
        with self.assertRaisesRegex(ValueError, "unsupported evidence status"):
            runner.aggregate_cases(
                [{
                    "level": "selection",
                    "budget_ratio": 0.75,
                    "recent_pages": 2,
                    "status": "UNKNOWN",
                    "metrics": passing_metrics("selection"),
                }]
            )

    def test_ppl_degradation_is_recomputed_and_mismatch_fails(self):
        passed = runner.evaluate_evidence(
            "ppl", {"dense_perplexity": 10.0, "rgkv_perplexity": 10.4}
        )
        self.assertEqual(passed["status"], runner.PASS)
        self.assertAlmostEqual(passed["metrics"]["relative_ppl_degradation"], 0.04)
        failed = runner.evaluate_evidence(
            "ppl",
            {
                "dense_perplexity": 10.0,
                "rgkv_perplexity": 10.4,
                "relative_ppl_degradation": 0.01,
            },
        )
        self.assertEqual(failed["status"], runner.FAIL)
        self.assertTrue(failed["gate"]["derived_metric_mismatch"])

    def test_one_complete_configuration_is_identified_but_partial_sweep_is_partial(self):
        records = [
            {
                "level": level,
                "budget_ratio": 0.5,
                "recent_pages": 4,
                "metrics": passing_metrics(level),
            }
            for level in runner.QUALITY_LEVELS
        ]
        args = runner.build_parser().parse_args(["--mode", "aggregate", "--input", "unused"])
        cases = runner.aggregate_cases(records)
        summary = runner.build_summary(args, cases)
        self.assertEqual(summary["status"], runner.PARTIAL)
        self.assertEqual(summary["qualified_configuration_count"], 1)
        self.assertFalse(summary["production_default_allowed"])

    def test_complete_passing_sweep_is_quality_pass_but_not_production_default(self):
        records = [
            {
                "level": level,
                "budget_ratio": ratio,
                "recent_pages": recent,
                "metrics": passing_metrics(level),
            }
            for ratio in runner.BUDGET_RATIOS
            for recent in runner.RECENT_PAGES
            for level in runner.QUALITY_LEVELS
        ]
        args = runner.build_parser().parse_args(
            ["--mode", "aggregate", "--input", "unused"]
        )
        cases = runner.aggregate_cases(records)
        summary = runner.build_summary(args, cases)
        self.assertEqual(summary["status"], runner.PASS)
        self.assertTrue(summary["complete_budget_sweep"])
        self.assertEqual(summary["qualified_configuration_count"], 16)
        self.assertFalse(summary["production_default_allowed"])

    def test_real_70b_is_plan_only(self):
        args = runner.build_parser().parse_args(
            ["--mode", "aggregate", "--target", "real-70b", "--input", "evidence.json"]
        )
        with self.assertRaisesRegex(ValueError, "plan-only"):
            runner.validate_args(args)

    def test_real_70b_plan_writes_blocked_four_file_bundle_without_execution(self):
        environment = {
            "cuda_available": True,
            "selected_physical_gpu": {"uuid": "GPU-test"},
            "selected_gpu_external_compute_processes": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            args = runner.build_parser().parse_args(
                ["--mode", "plan", "--target", "real-70b", "--output-dir", str(output)]
            )
            with mock.patch.object(runner, "capture_cuda_environment", return_value=environment):
                summary, cases, paths = runner.run(args)
            self.assertEqual(summary["status"], runner.BLOCKED)
            self.assertTrue(all(case["status"] == runner.BLOCKED for case in cases))
            self.assertEqual(
                set(paths), {"environment.json", "cases.json", "summary.json", "report.md"}
            )
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            saved_environment = json.loads((output / "environment.json").read_text(encoding="utf-8"))
            self.assertFalse(saved_environment["execution_performed"])
            self.assertFalse(saved_environment["model_weights_opened"])

    def test_duplicate_or_out_of_matrix_evidence_is_rejected(self):
        record = {
            "level": "selection",
            "budget_ratio": 0.75,
            "recent_pages": 2,
            "metrics": passing_metrics("selection"),
        }
        with self.assertRaisesRegex(ValueError, "duplicate"):
            runner.aggregate_cases([record, dict(record)])
        with self.assertRaisesRegex(ValueError, "unsupported recent_pages"):
            runner.aggregate_cases([dict(record, recent_pages=3)])


if __name__ == "__main__":
    unittest.main()
