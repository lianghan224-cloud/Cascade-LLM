import json
from pathlib import Path
import tempfile
import unittest

from tools.validate_kv_long_stability import (
    NOT_QUALIFIED,
    BLOCKED,
    PASS,
    SKIPPED,
    assess_drift,
    blocked_case,
    run_validation,
    skipped_case,
    write_reports,
)


class KVLongStabilityHarnessTest(unittest.TestCase):
    def test_logic_matrix_closes_resources_and_writes_schema(self):
        report = run_validation(
            "logic",
            session_cycles=(3,),
            decode_lengths=(4,),
            context_lengths=(16,),
            sample_every=1,
        )
        self.assertEqual(report["status"], PASS)
        self.assertEqual(report["qualification"], NOT_QUALIFIED)
        self.assertEqual(report["counts"][PASS], 7)
        for case in report["cases"]:
            self.assertEqual(case["status"], PASS, case.get("reason"))
            self.assertEqual(case["final"]["kv"]["page_allocated"], 0)
            self.assertEqual(case["final"]["kv"]["ref_count"], 0)
            self.assertEqual(case["final"]["kv"]["pin_count"], 0)
            self.assertEqual(case["final"]["kv"]["active_requests"], 0)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_reports(report, directory)
            environment = json.loads(
                Path(paths["environment"]).read_text(encoding="utf-8")
            )
            summary = json.loads(Path(paths["summary"]).read_text(encoding="utf-8"))
            cases = json.loads(Path(paths["cases"]).read_text(encoding="utf-8"))
            markdown = Path(paths["report"]).read_text(encoding="utf-8")
        self.assertEqual(summary["schema_version"], 2)
        self.assertIn("python", environment)
        self.assertNotIn("environment", summary)
        self.assertNotIn("cases", summary)
        self.assertEqual(len(cases["cases"]), 7)
        active = next(
            item
            for item in cases["cases"]
            if item["category"] == "rgkv_active_tier_session_cycle"
        )
        self.assertGreater(active["metrics"]["eviction_count"], 0)
        self.assertGreater(active["metrics"]["prefetch_count"], 0)
        self.assertIn("SKIPPED_WITH_REASON", markdown)
        self.assertIn("does not qualify", markdown)

    def test_skip_is_never_reported_as_pass(self):
        case = skipped_case("LS-X", "unavailable", "cuda_smoke", "no GPU")
        self.assertEqual(case["status"], SKIPPED)
        self.assertEqual(case["qualification"], NOT_QUALIFIED)
        self.assertEqual(case["reason"], "no GPU")

    def test_blocked_is_never_reported_as_skip_or_pass(self):
        case = blocked_case("LS-X", "shared GPU", "cuda", "not exclusive")
        self.assertEqual(case["status"], BLOCKED)
        self.assertEqual(case["reason"], "not exclusive")

    def test_drift_gate_detects_owner_and_thread_growth(self):
        baseline = {
            "kv": {
                "page_total": 2,
                "page_free": 2,
                "page_allocated": 0,
                "ref_count": 0,
                "logical_owner_count": 0,
            },
            "prefix": {"entries": 0, "pages": 0},
            "quest": {"records": 0, "references": 0},
            "system": {"thread_count": 1},
        }
        final = json.loads(json.dumps(baseline))
        final["kv"]["page_free"] = 1
        final["kv"]["page_allocated"] = 1
        final["kv"]["ref_count"] = 1
        final["kv"]["logical_owner_count"] = 1
        final["system"]["thread_count"] = 2
        failures = assess_drift(baseline, final)
        self.assertTrue(any("page_allocated" in item for item in failures))
        self.assertTrue(any("thread count" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
