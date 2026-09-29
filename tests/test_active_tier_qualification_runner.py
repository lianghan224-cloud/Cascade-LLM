import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools import validate_active_tier as runner


def environment(*, reserved=False, external=0):
    return {
        "captured_at": "2026-08-11T00:00:00+00:00",
        "cuda_available": True,
        "selected_physical_gpu": {
            "index": 4,
            "uuid": "GPU-test",
            "name": "RTX 3090",
        },
        "selected_gpu_compute_processes": [],
        "selected_gpu_external_compute_processes": external,
        "exclusive_snapshot": external == 0,
        "reservation_evidence_present": reserved,
        "reservation_evidence": {
            "scheduler": "slurm",
            "job_id": "123" if reserved else None,
            "job_gpus": "4" if reserved else None,
        },
    }


class ActiveTierQualificationRunnerTest(unittest.TestCase):
    def test_qualification_blocks_before_cuda_without_reservation(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "capture_cuda_environment",
            return_value=environment(reserved=False),
        ), mock.patch.object(
            runner,
            "run_case",
            side_effect=AssertionError("blocked qualification executed CUDA"),
        ):
            status = runner.main(
                [
                    "--mode",
                    "qualification",
                    "--output-dir",
                    directory,
                ]
            )
            self.assertEqual(status, 2)
            summary = json.loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
            cases = json.loads(
                (Path(directory) / "cases.json").read_text(encoding="utf-8")
            )["cases"]
        self.assertEqual(summary["status"], "BLOCKED_NO_RESERVATION")
        self.assertEqual(summary["capability_state"], "CUDA_SMOKE")
        self.assertFalse(summary["qualification_admitted"])
        self.assertTrue(
            all(
                item["status"] == "BLOCKED"
                and "BLOCKED_NO_RESERVATION" in item["reason"]
                for item in cases
            )
        )

    def test_shared_process_takes_precedence_over_reservation(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner,
            "capture_cuda_environment",
            return_value=environment(reserved=True, external=1),
        ):
            status = runner.main(
                [
                    "--mode",
                    "qualification",
                    "--output-dir",
                    directory,
                ]
            )
            cases = json.loads(
                (Path(directory) / "cases.json").read_text(encoding="utf-8")
            )["cases"]
        self.assertEqual(status, 2)
        self.assertTrue(
            all("BLOCKED_NOT_EXCLUSIVE" in item["reason"] for item in cases)
        )


if __name__ == "__main__":
    unittest.main()
