import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import qualify_rgkv_device_attention as runner


class RGKVDeviceAttentionRunnerTest(unittest.TestCase):
    def test_logic_1000_gate_writes_nonqualified_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            args = runner.parse_args(
                ["--mode", "logic", "--iterations", "1000", "--output-dir", directory]
            )
            summary, cases, paths = runner.run(args)
            self.assertEqual(summary["status"], runner.PASS)
            self.assertEqual(summary["qualification"], runner.NOT_QUALIFIED)
            self.assertFalse(summary["device_tier_miss_batch_started"])
            self.assertEqual(cases[0]["metrics"]["gpu_hit_device_attention_calls"], 1000)
            for name in (
                "selected_metadata_d2h",
                "host_scalar_readbacks",
                "explicit_sync_count",
                "python_attention_wave_count",
                "device_view_fallback_count",
            ):
                self.assertEqual(cases[0]["metrics"][name], 0)
            self.assertEqual(
                {Path(path).name for path in paths.values()},
                {"environment.json", "cases.json", "summary.json", "report.md"},
            )
            persisted = json.loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["qualification"], runner.NOT_QUALIFIED)

    def test_cuda_gate_blocks_before_execution_on_shared_gpu(self):
        environment = {
            "cuda_available": True,
            "selected_physical_gpu": {"index": 0, "uuid": "GPU-test"},
            "selected_gpu_external_compute_processes": 1,
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "capture_cuda_environment", return_value=environment
        ), mock.patch.object(runner, "_cuda_case") as execute:
            summary, cases, _ = runner.run(
                runner.parse_args(
                    ["--mode", "qualification", "--output-dir", directory]
                )
            )
        execute.assert_not_called()
        self.assertEqual(summary["status"], runner.BLOCKED)
        self.assertIn("BLOCKED_NOT_EXCLUSIVE", cases[0]["reason"])


if __name__ == "__main__":
    unittest.main()
