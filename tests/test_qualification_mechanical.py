import contextlib
from concurrent.futures import Future
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from layer_streaming.kv import KVLifecycleError, KVOperationFence
from tools import qualify_gather_sdpa_prefill as prefill_runner
from tools import qualify_llama70b_single_request as llama70b_runner
from tools import validate_kv_stack as unified_runner
from tools import validate_kv_cuda_async as cuda_harness


class QueryEvent:
    def __init__(self, *, done=False, query_error=None, sync_error=None):
        self.done = bool(done)
        self.query_error = query_error
        self.sync_error = sync_error
        self.synchronize_calls = 0

    def query(self):
        if self.query_error is not None:
            raise self.query_error
        return self.done

    def synchronize(self):
        self.synchronize_calls += 1
        if self.sync_error is not None:
            raise self.sync_error


class QualificationMechanicalTest(unittest.TestCase):
    def test_fence_query_synchronize_and_terminal_state_matrix(self):
        cases = (
            ("complete", QueryEvent(done=True), None, "completed"),
            (
                "query_error",
                QueryEvent(done=True, query_error=RuntimeError("query")),
                KVLifecycleError,
                "failed",
            ),
            (
                "synchronize_error",
                QueryEvent(done=True, sync_error=RuntimeError("sync")),
                KVLifecycleError,
                "failed",
            ),
        )
        for name, event, expected_error, expected_status in cases:
            with self.subTest(name=name):
                fence = KVOperationFence(
                    operation_id="mechanical-{}".format(name),
                    request_id=1,
                    kind="test",
                    cuda_event=event,
                )
                if expected_error is None:
                    self.assertIs(fence.wait(timeout_seconds=0.01), fence)
                    self.assertTrue(fence.query())
                    # Waiting an already completed Fence is intentionally safe.
                    self.assertIs(fence.wait(timeout_seconds=0.01), fence)
                else:
                    with self.assertRaises(expected_error):
                        fence.wait(timeout_seconds=0.01)
                    self.assertTrue(fence.query())
                self.assertEqual(fence.status, expected_status)

    def test_fence_cancel_and_double_cancel_matrix(self):
        for with_future in (False, True):
            with self.subTest(with_future=with_future):
                future = Future() if with_future else None
                fence = KVOperationFence(
                    operation_id="mechanical-cancel-{}".format(with_future),
                    request_id=2,
                    kind="test",
                    io_future=future,
                )
                self.assertIs(fence.cancel(), fence)
                self.assertIs(fence.cancel(), fence)
                self.assertTrue(fence.cancelled)
                self.assertEqual(fence.status, "cancelled")
                if future is not None:
                    self.assertTrue(future.cancelled())
                with self.assertRaisesRegex(KVLifecycleError, "cancelled"):
                    fence.wait(timeout_seconds=0.01)

    def test_attention_double_drain_logic_is_idempotent(self):
        result = cuda_harness._attention_double_drain("cpu")
        self.assertTrue(result["double_drain_idempotent"])
        self.assertEqual(result["final_pin_count"], 0)

    @staticmethod
    def _help_text(parse_args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            with unittest.TestCase().assertRaises(SystemExit) as raised:
                parse_args(["--help"])
        if raised.exception.code != 0:
            raise AssertionError("--help exited nonzero")
        return output.getvalue()

    def test_new_qualification_cli_help_contracts(self):
        cases = (
            (
                cuda_harness.parse_args,
                ("--mode", "--device", "--output-dir", "qualification"),
            ),
            (
                llama70b_runner.parse_args,
                (
                    "--mode",
                    "--checkpoint",
                    "--allow-shared-smoke",
                    "--case",
                    "--kv-prefill-backend",
                ),
            ),
            (
                unified_runner.parse_args,
                (
                    "--mode",
                    "--device",
                    "--checkpoint",
                    "--allow-shared-smoke",
                    "qualification",
                ),
            ),
            (
                prefill_runner.build_parser().parse_args,
                (
                    "--mode",
                    "--device",
                    "--lengths",
                    "--allow-shared-smoke",
                    "--checkpoint",
                    "--output-dir",
                ),
            ),
        )
        for parse_args, expected in cases:
            with self.subTest(parser=parse_args.__module__):
                help_text = self._help_text(parse_args)
                for value in expected:
                    self.assertIn(value, help_text)

    def test_shared_four_file_report_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            args = llama70b_runner.parse_args(
                [
                    "--mode",
                    "plan",
                    "--case",
                    "1:1",
                    "--output-dir",
                    directory,
                ]
            )
            summary, _, paths = llama70b_runner.run(args)
            self.assertEqual(
                {Path(item).name for item in paths.values()},
                {"environment.json", "cases.json", "summary.json", "report.md"},
            )
            environment = json.loads(
                (Path(directory) / "environment.json").read_text(encoding="utf-8")
            )
            cases = json.loads(
                (Path(directory) / "cases.json").read_text(encoding="utf-8")
            )
            persisted_summary = json.loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cases["schema_version"], summary["schema_version"])
            self.assertEqual(
                persisted_summary["schema_version"], summary["schema_version"]
            )
            self.assertEqual(cases["mode"], summary["mode"])
            self.assertIn("cases", cases)
            self.assertNotIn("cases", persisted_summary)
            self.assertIsInstance(environment, dict)
            self.assertTrue((Path(directory) / "report.md").read_text().strip())

    def test_unified_qualification_requires_checkpoint(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                unified_runner.parse_args(["--mode", "qualification"])

    def test_legacy_profile_remains_available(self):
        args = unified_runner.parse_args(["--profile", "logic"])
        self.assertEqual(args.profile, "logic")
        self.assertIsNone(args.mode)
        self.assertIsNone(args.output_dir)

    def test_legacy_profile_report_honors_explicit_output_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            result = unified_runner.result(
                "V00",
                "logic",
                unified_runner.PASS,
                {},
                1,
            )
            _, json_path, markdown_path = unified_runner.write_reports(
                "logic", 1, {}, (result,), 0.0, output_dir=directory
            )
            self.assertEqual(json_path.parent, Path(directory))
            self.assertEqual(markdown_path.parent, Path(directory))
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())

    def test_unified_qualification_blocks_before_component_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint"
            checkpoint.mkdir()
            output = Path(directory) / "bundle"
            args = unified_runner.parse_args(
                [
                    "--mode", "qualification",
                    "--checkpoint", str(checkpoint),
                    "--output-dir", str(output),
                ]
            )
            environment = {
                "cuda_available": True,
                "selected_physical_gpu": {"index": 0, "uuid": "GPU-test"},
                "selected_gpu_external_compute_processes": 1,
            }
            with mock.patch.object(
                unified_runner, "capture_cuda_environment", return_value=environment
            ):
                summary, cases, paths = unified_runner.run_unified(args)
            self.assertEqual(summary["status"], "BLOCKED_NOT_EXCLUSIVE")
            self.assertTrue(cases)
            self.assertTrue(
                all(case["status"] == "BLOCKED_NOT_EXCLUSIVE" for case in cases)
            )
            self.assertEqual(
                {Path(path).name for path in paths.values()},
                {"environment.json", "cases.json", "summary.json", "report.md"},
            )


if __name__ == "__main__":
    unittest.main()
