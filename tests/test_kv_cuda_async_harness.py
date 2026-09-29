import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from layer_streaming.capability_state import CapabilityState
from tools import qualification_common
from tools import validate_kv_cuda_async as harness


def environment(cuda_available=False, *, selected=None, external=0):
    return {
        "captured_at": "2026-08-09T00:00:00+00:00",
        "git_revision": "test",
        "python": "3.10",
        "platform": "test",
        "torch": "test",
        "torch_cuda": None,
        "cuda_available": bool(cuda_available),
        "cuda_device_count": 1 if cuda_available else 0,
        "cuda_visible_devices": None,
        "requested_device": "cuda:0",
        "selected_physical_gpu": selected,
        "selected_gpu_compute_processes": [],
        "selected_gpu_external_compute_processes": int(external),
        "exclusive_snapshot": bool(selected is not None and not external),
        "nvidia_smi_errors": [],
    }


class QualificationCommonTest(unittest.TestCase):
    def test_resolve_physical_gpu_honors_visible_device_order(self):
        inventory = [
            {"index": 0, "uuid": "GPU-zero", "name": "zero"},
            {"index": 3, "uuid": "GPU-three", "name": "three"},
        ]
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,0"}):
            selected, reason = qualification_common.resolve_physical_gpu(
                "cuda:0", inventory
            )
        self.assertIsNone(reason)
        self.assertEqual(selected["index"], 3)

    def test_qualification_admission_blocks_external_process(self):
        admitted, status, reason = qualification_common.qualification_admission(
            environment(
                True,
                selected={"index": 0, "uuid": "GPU-a", "name": "gpu"},
                external=2,
            )
        )
        self.assertFalse(admitted)
        self.assertEqual(status, harness.BLOCKED)
        self.assertIn(harness.BLOCKED_NOT_EXCLUSIVE, reason)

    def test_qualification_admission_requires_scheduler_reservation(self):
        admitted, status, reason = qualification_common.qualification_admission(
            environment(
                True,
                selected={"index": 0, "uuid": "GPU-a", "name": "gpu"},
            ),
            require_reservation=True,
        )
        self.assertFalse(admitted)
        self.assertEqual(status, harness.BLOCKED)
        self.assertIn("BLOCKED_NO_RESERVATION", reason)

        reserved = environment(
            True, selected={"index": 0, "uuid": "GPU-a", "name": "gpu"}
        )
        reserved["reservation_evidence_present"] = True
        admitted, status, reason = qualification_common.qualification_admission(
            reserved, require_reservation=True
        )
        self.assertTrue(admitted)
        self.assertEqual(status, harness.PASS)
        self.assertIsNone(reason)


class CudaAsyncHarnessTest(unittest.TestCase):
    def test_logic_executes_fault_contract_without_cuda_qualification(self):
        with mock.patch.object(
            harness,
            "capture_cuda_environment",
            return_value=environment(False),
        ):
            report = harness.run_validation("logic")
        by_id = {case["case_id"]: case for case in report["cases"]}
        self.assertEqual(report["status"], harness.PASS)
        self.assertEqual(
            report["qualification"], CapabilityState.LOGIC_VALIDATED.value
        )
        self.assertEqual(by_id["B1"]["status"], harness.PASS)
        self.assertEqual(by_id["B2"]["status"], harness.SKIPPED)
        self.assertEqual(by_id["B3"]["status"], harness.PASS)
        self.assertEqual(by_id["B4-QUERY"]["metrics"]["status"], "failed")
        self.assertEqual(by_id["B4-SYNC"]["metrics"]["status"], "failed")
        self.assertEqual(by_id["B4-TIMEOUT"]["status"], harness.PASS)
        self.assertEqual(by_id["B4-CANCEL"]["status"], harness.PASS)
        self.assertEqual(by_id["B4-DRAIN"]["status"], harness.PASS)

    def test_cuda_unavailable_is_skipped_not_passed(self):
        with mock.patch.object(
            harness,
            "capture_cuda_environment",
            return_value=environment(False),
        ):
            report = harness.run_validation("cuda-smoke")
        self.assertEqual(report["counts"][harness.PASS], 0)
        self.assertEqual(
            report["counts"][harness.SKIPPED], len(harness.CASE_NAMES)
        )
        self.assertTrue(
            all(case["status"] == harness.SKIPPED for case in report["cases"])
        )

    def test_qualification_shared_gpu_is_blocked_not_passed(self):
        selected = {"index": 0, "uuid": "GPU-a", "name": "gpu"}
        with mock.patch.object(
            harness,
            "capture_cuda_environment",
            return_value=environment(True, selected=selected, external=1),
        ):
            report = harness.run_validation("qualification")
        self.assertEqual(report["status"], harness.BLOCKED)
        self.assertEqual(report["counts"][harness.PASS], 0)
        self.assertEqual(
            report["qualification"],
            CapabilityState.QUALIFICATION_READY.value,
        )
        self.assertTrue(
            all(case["status"] == harness.BLOCKED for case in report["cases"])
        )
        self.assertTrue(
            all(
                harness.BLOCKED_NOT_EXCLUSIVE in case["reason"]
                for case in report["cases"]
            )
        )

    def test_qualification_idle_snapshot_without_reservation_is_blocked(self):
        selected = {"index": 0, "uuid": "GPU-a", "name": "gpu"}
        with mock.patch.object(
            harness,
            "capture_cuda_environment",
            return_value=environment(True, selected=selected),
        ):
            report = harness.run_validation("qualification")
        self.assertEqual(report["status"], harness.BLOCKED)
        self.assertEqual(report["counts"][harness.PASS], 0)
        self.assertTrue(
            all(
                "BLOCKED_NO_RESERVATION" in case["reason"]
                for case in report["cases"]
            )
        )

    def test_qualification_invalidates_external_activity_at_end_snapshot(self):
        selected = {"index": 0, "uuid": "GPU-a", "name": "gpu"}
        start = environment(True, selected=selected)
        start["reservation_evidence_present"] = True
        start["reservation_evidence"] = {"job_id": "123"}
        end = environment(True, selected=selected, external=1)
        end["reservation_evidence_present"] = True
        end["reservation_evidence"] = {"job_id": "123"}
        passing = [
            harness.make_case(
                case_id,
                harness.PASS,
                CapabilityState.QUALIFIED.value,
            )
            for case_id in harness.CASE_NAMES
        ]
        with mock.patch.object(
            harness,
            "capture_cuda_environment",
            side_effect=[start, end],
        ), mock.patch.object(harness, "_cuda_cases", return_value=passing):
            report = harness.run_validation("qualification")
        self.assertEqual(report["status"], harness.FAIL)
        self.assertEqual(
            report["qualification"],
            CapabilityState.QUALIFICATION_READY.value,
        )
        self.assertTrue(
            all(
                case["reason"] == "INVALIDATED_EXTERNAL_GPU_ACTIVITY"
                for case in report["cases"]
            )
        )
    def test_report_bundle_has_consistent_four_file_schema(self):
        with mock.patch.object(
            harness,
            "capture_cuda_environment",
            return_value=environment(False),
        ):
            report = harness.run_validation("logic")
        with tempfile.TemporaryDirectory() as directory:
            artifacts = harness.write_reports(report, directory)
            self.assertEqual(
                set(artifacts),
                {"environment.json", "cases.json", "summary.json", "report.md"},
            )
            for name in ("environment.json", "cases.json", "summary.json"):
                with (Path(directory) / name).open(encoding="utf-8") as stream:
                    json.load(stream)
            cases = json.loads(
                (Path(directory) / "cases.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (Path(directory) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(cases["schema_version"], summary["schema_version"])
            self.assertEqual(cases["mode"], summary["mode"])
            self.assertNotIn("cases", summary)


if __name__ == "__main__":
    unittest.main()
