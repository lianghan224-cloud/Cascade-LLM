from types import SimpleNamespace
import tempfile
import unittest

from tools import run_llama70b_single_gpu_benchmark as matrix_runner
from tools import summarize_llama70b_single_gpu_benchmark as summarizer


class Llama70BSingleGPUBenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            best_granularity="matrix",
            best_slots=2,
            best_prefetch_depth=2,
            best_budget_gib=8,
        )

    def test_four_group_plan_has_requested_matrix(self):
        baseline = matrix_runner.cases_for_stage(self.args, "baseline")
        pipeline = matrix_runner.cases_for_stage(self.args, "pipeline")
        budget = matrix_runner.cases_for_stage(self.args, "budget")
        stability = matrix_runner.cases_for_stage(self.args, "stability")
        self.assertEqual(len(baseline), 12)
        self.assertEqual(
            {(item["prompt"], item["decode"]) for item in baseline},
            {
                (prompt, decode)
                for prompt in (512, 2048, 4096, 8192)
                for decode in (1, 8, 32)
            },
        )
        self.assertEqual(len(pipeline), 18)
        no_prefetch = [item for item in pipeline if item["prefetch_depth"] == 0]
        self.assertEqual(len(no_prefetch), 6)
        self.assertTrue(
            all(item["executable"] for item in no_prefetch)
        )
        self.assertEqual([item["budget_gib"] for item in budget], [0, 4, 8, 12, 16, 20])
        self.assertEqual(stability[0]["repeat"], 5)

    def test_best_configuration_accepts_synchronous_depth_zero(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            args = matrix_runner.parse_args(
                [
                    "--stage",
                    "budget",
                    "--checkpoint",
                    checkpoint,
                    "--best-granularity",
                    "matrix",
                    "--best-slots",
                    "2",
                    "--best-prefetch-depth",
                    "0",
                ]
            )
        self.assertEqual(args.best_prefetch_depth, 0)
        budget = matrix_runner.cases_for_stage(args, "budget")
        self.assertTrue(all(case["prefetch_depth"] == 0 for case in budget))

    @staticmethod
    def synthetic_row(
        stage,
        case_id,
        granularity="matrix",
        slots=2,
        prefetch_depth=2,
        budget_bytes=0,
        tpot_ms=10.0,
        repetition=1,
        output_hash="same",
    ):
        return {
            "stage": stage,
            "case_id": case_id,
            "valid_for_uncontended_comparison": True,
            "gpu_activity_evidence": "CONTINUOUSLY_MONITORED_UNCONTENDED",
            "prompt_tokens": 2048,
            "decode_tokens": 32,
            "repetition": repetition,
            "config": {
                "granularity": granularity,
                "slots": slots,
                "prefetch_depth": prefetch_depth,
                "gpu_resident_weight_budget": budget_bytes,
            },
            "metrics": {
                "tpot_mean_ms": tpot_ms,
                "ttft_ms": 20.0 + repetition,
                "gpu_idle_or_host_overhead_ms": 2.0,
                "h2d_time_ms": 3.0,
                "h2d_bytes_per_generated_token": 4.0,
                "gpu_peak_bytes": 1000.0 + repetition,
                "kv_bytes": 500.0,
                "resident_transformer_weight_bytes": float(budget_bytes),
                "output_hash": output_hash,
            },
        }

    def test_automatic_pipeline_and_budget_selection(self):
        rows = []
        for granularity in ("matrix", "layer"):
            for slots in (1, 2, 3):
                for depth in (0, 1, 2):
                    score = 1.0 if (granularity, slots, depth) == (
                        "layer",
                        3,
                        0,
                    ) else 10.0
                    rows.append(
                        self.synthetic_row(
                            "pipeline",
                            "{}_{}_{}".format(granularity, slots, depth),
                            granularity,
                            slots,
                            depth,
                            tpot_ms=score,
                        )
                    )
        selected = summarizer.select_pipeline_configuration(rows)
        self.assertEqual(
            (selected["granularity"], selected["slots"], selected["prefetch_depth"]),
            ("layer", 3, 0),
        )

        for budget_gib in (0, 4, 8, 12, 16, 20):
            rows.append(
                self.synthetic_row(
                    "budget",
                    "budget_{}".format(budget_gib),
                    "layer",
                    3,
                    0,
                    budget_bytes=budget_gib * 1024 ** 3,
                    tpot_ms=1.0 if budget_gib == 8 else 10.0,
                )
            )
        selected = summarizer.select_budget_configuration(rows)
        self.assertEqual(selected["gpu_resident_weight_budget_gib"], 8.0)

    def test_stability_requires_five_consistent_hashes(self):
        rows = [
            self.synthetic_row(
                "stability",
                "stable_{}".format(repetition),
                repetition=repetition,
            )
            for repetition in range(1, 6)
        ]
        result = summarizer.summarize_stability(rows)
        self.assertTrue(result["complete"])
        self.assertTrue(result["output_consistent"])
        self.assertEqual(result["gpu_peak_drift"]["first_to_last_bytes"], 4.0)
        rows[-1]["metrics"]["output_hash"] = "different"
        self.assertFalse(summarizer.summarize_stability(rows)["output_consistent"])

    def test_complete_matrix_publishes_stable_best_configuration(self):
        rows = []
        for prompt in (512, 2048, 4096, 8192):
            for decode in (1, 8, 32):
                row = self.synthetic_row(
                    "baseline", "baseline_{}_{}".format(prompt, decode)
                )
                row["prompt_tokens"] = prompt
                row["decode_tokens"] = decode
                rows.append(row)
        for granularity in ("matrix", "layer"):
            for slots in (1, 2, 3):
                for depth in (0, 1, 2):
                    rows.append(
                        self.synthetic_row(
                            "pipeline",
                            "pipeline_{}_{}_{}".format(
                                granularity, slots, depth
                            ),
                            granularity,
                            slots,
                            depth,
                            tpot_ms=(
                                1.0
                                if (granularity, slots, depth)
                                == ("layer", 3, 0)
                                else 10.0
                            ),
                        )
                    )
        for budget_gib in (0, 4, 8, 12, 16, 20):
            rows.append(
                self.synthetic_row(
                    "budget",
                    "budget_{}".format(budget_gib),
                    "layer",
                    3,
                    0,
                    budget_bytes=budget_gib * 1024 ** 3,
                    tpot_ms=1.0 if budget_gib == 8 else 10.0,
                )
            )
        for repetition in range(1, 6):
            rows.append(
                self.synthetic_row(
                    "stability",
                    "stable_{}".format(repetition),
                    "layer",
                    3,
                    0,
                    budget_bytes=8 * 1024 ** 3,
                    repetition=repetition,
                )
            )
        summary = summarizer.build_summary(rows, "test-time")
        self.assertEqual(summary["status"], "COMPLETE")
        self.assertEqual(
            summary["best_configuration"][
                "gpu_resident_weight_budget_gib"
            ],
            8.0,
        )
        self.assertTrue(summary["stability"]["output_consistent"])

    def test_uncontended_evidence_rejects_external_boundary(self):
        clean = {
            "gpu_admission": {
                "before": {"selected_gpu_external_compute_processes": 0},
                "after": {"selected_gpu_external_compute_processes": 0},
            },
            "gpu_activity_monitor": {"external_activity_observed": False},
        }
        valid, evidence = summarizer._admission_valid(clean)
        self.assertTrue(valid)
        self.assertEqual(evidence, "CONTINUOUSLY_MONITORED_UNCONTENDED")
        clean["gpu_admission"]["after"][
            "selected_gpu_external_compute_processes"
        ] = 1
        valid, evidence = summarizer._admission_valid(clean)
        self.assertFalse(valid)
        self.assertEqual(evidence, "INVALIDATED_EXTERNAL_GPU_ACTIVITY")


if __name__ == "__main__":
    unittest.main()
