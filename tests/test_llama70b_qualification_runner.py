import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import qualify_llama70b_single_request as runner


class Llama70BQualificationRunnerTest(unittest.TestCase):
    def test_default_matrix_covers_required_axes_and_sensible_long_decode(self):
        cases = runner.planned_cases()
        self.assertEqual(
            {item["prompt_tokens"] for item in cases},
            set(runner.REQUIRED_PROMPT_LENGTHS),
        )
        self.assertEqual(
            {item["decode_tokens"] for item in cases},
            set(runner.REQUIRED_DECODE_LENGTHS),
        )
        self.assertIn(
            (128, 1000),
            {(item["prompt_tokens"], item["decode_tokens"]) for item in cases},
        )

    def test_qualification_defaults_to_five_repetitions(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.mkdir()
            qualification = runner.parse_args(
                ["--mode", "qualification", "--checkpoint", str(checkpoint)]
            )
            smoke = runner.parse_args(
                ["--mode", "smoke", "--checkpoint", str(checkpoint)]
            )
        self.assertEqual(qualification.repeat, 5)
        self.assertEqual(smoke.repeat, 1)

    def test_plan_writes_four_file_bundle_without_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bundle"
            args = runner.parse_args(
                ["--mode", "plan", "--case", "128:32", "--output-dir", str(output)]
            )
            environment = {
                "cuda_available": False,
                "selected_physical_gpu": None,
                "selected_gpu_external_compute_processes": 0,
            }
            with mock.patch.object(runner, "capture_cuda_environment", return_value=environment), mock.patch.object(
                runner, "execute_case", side_effect=AssertionError("plan executed model")
            ):
                summary, cases, paths = runner.run(args)
            self.assertEqual(summary["status"], runner.PLAN_ONLY)
            self.assertEqual(cases[0]["status"], runner.SKIPPED)
            self.assertEqual(
                set(Path(path).name for path in paths.values()),
                {"environment.json", "cases.json", "summary.json", "report.md"},
            )
            for path in paths.values():
                self.assertTrue(Path(path).is_file())
            payload = json.loads((output / "cases.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["cases"][0]["reason"].split(":", 1)[0], "PLAN_ONLY")

    def test_qualification_blocks_shared_gpu_without_running_child(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.mkdir()
            args = runner.parse_args(
                [
                    "--mode", "qualification", "--checkpoint", str(checkpoint),
                    "--case", "1:1", "--output-dir", str(Path(directory) / "output"),
                ]
            )
            environment = {
                "cuda_available": True,
                "selected_physical_gpu": {"index": 0, "uuid": "GPU-test"},
                "selected_gpu_external_compute_processes": 1,
            }
            with mock.patch.object(runner, "capture_cuda_environment", return_value=environment), mock.patch.object(
                runner, "execute_case", side_effect=AssertionError("blocked run executed child")
            ):
                summary, cases, _ = runner.run(args)
            self.assertEqual(summary["status"], runner.BLOCKED_NOT_EXCLUSIVE)
            self.assertEqual(cases[0]["status"], runner.BLOCKED_NOT_EXCLUSIVE)
            self.assertEqual(summary["pass_count"], 0)

    def test_allow_shared_is_always_smoke_only(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.mkdir()
            args = runner.parse_args(
                [
                    "--mode", "qualification", "--allow-shared-smoke",
                    "--checkpoint", str(checkpoint), "--case", "1:1",
                    "--output-dir", str(Path(directory) / "output"),
                ]
            )
            environment = {
                "cuda_available": True,
                "selected_physical_gpu": {"index": 0, "uuid": "GPU-test"},
                "selected_gpu_external_compute_processes": 1,
            }

            def successful_case(_args, case, evidence, **_kwargs):
                return dict(case, status=runner.PASS, evidence_class=evidence, reason=None)

            with mock.patch.object(runner, "capture_cuda_environment", return_value=environment), mock.patch.object(
                runner, "execute_case", side_effect=successful_case
            ):
                summary, cases, _ = runner.run(args)
            self.assertEqual(summary["status"], runner.SMOKE_ONLY)
            self.assertEqual(summary["evidence_class"], runner.SMOKE_ONLY)
            self.assertEqual(cases[0]["evidence_class"], runner.SMOKE_ONLY)

    def test_normalize_report_calculates_percentiles_and_marks_unavailable(self):
        report = {
            "timings": {
                "load_time_seconds": 2.0,
                "time_to_first_token_ms": 40.0,
                "decode_token_latencies_ms": [10.0, 20.0, 30.0],
                "h2d_time_ms": 2.0,
                "unoverlapped_h2d_ms": 1.0,
                "h2d_compute_overlap_ms": 0.5,
            },
            "throughput": {"tokens_per_second": 2.0, "decode_tokens_per_second": 50.0},
            "pipeline": {
                "h2d_bytes": 2000,
                "transfer_slot_reuse_counts": [4, 5],
                "kv": {
                    "kv_pool_total_pages": 10,
                    "kv_pool_peak_pages": 3,
                    "provider_fallback_count": 0,
                    "provider_reference_fallback_count": 0,
                },
            },
            "memory": {
                "gpu_peak_memory_bytes": 100,
                "pinned_bytes": 200,
                "kv_cache_bytes": 300,
                "kv_attention_workspace_bytes": 400,
                "cpu_resident_bytes": 500,
            },
        }
        metrics = runner.normalize_run_report(report)
        self.assertEqual(metrics["performance"]["decode_p50_ms"]["value"], 20.0)
        self.assertEqual(metrics["performance"]["decode_p90_ms"]["value"], 28.0)
        self.assertEqual(
            metrics["weight_streaming"]["effective_pcie_bandwidth_bytes_per_second"]["value"],
            1_000_000.0,
        )
        self.assertFalse(metrics["kv"]["pin_total"]["available"])
        self.assertIn("subprocess", metrics["kv"]["pin_total"]["source"])
        self.assertFalse(metrics["memory"]["cuda_reserved_peak_bytes"]["available"])
        self.assertEqual(
            metrics["memory"]["cpu_resident_arena_bytes"]["note"],
            "arena size, not process RSS",
        )

    def test_build_command_freezes_recommended_dense_configuration(self):
        args = argparse.Namespace(
            python="python", checkpoint=Path("/model"), device="cuda:0",
            weight_store="pinned_staging", granularity="matrix", slots=2,
            prefetch_depth=2, embedding_placement="resident",
            lm_head_placement="streamed", kv_page_size=16,
            kv_attention_backend="generic_cuda", kv_prefill_backend="gather_sdpa_prefill",
            cuda_safety_margin_mib=512, allow_kv_reference=False,
            ignore_memlock_limit=False, gpu_resident_weight_budget=0,
        )
        command = runner.build_command(
            args, {"decode_tokens": 32}, Path("/tmp/report.json"), "prompt"
        )
        self.assertIn("pinned_staging", command)
        self.assertIn("gather_sdpa_prefill", command)
        self.assertIn("streamed", command)
        self.assertEqual(command[-2:], ["--output", "/tmp/report.json"])
        displayed = runner.display_command(command, 16384)
        self.assertEqual(
            displayed[displayed.index("--checkpoint") + 1],
            "$CASCADE_70B_CHECKPOINT",
        )
        self.assertEqual(
            displayed[displayed.index("--prompt") + 1],
            "<generated_exact_16384_token_prompt>",
        )

    def test_checkpoint_path_redaction(self):
        self.assertEqual(
            runner._redact_checkpoint(
                "failed at /private/models/70b/config.json",
                Path("/private/models/70b"),
            ),
            "failed at $CASCADE_70B_CHECKPOINT/config.json",
        )


if __name__ == "__main__":
    unittest.main()
