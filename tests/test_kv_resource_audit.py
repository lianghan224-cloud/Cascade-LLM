import unittest

import torch

from layer_streaming.kv.resource_audit import (
    BLOCKED,
    FAIL,
    PASS,
    SKIPPED_WITH_REASON,
    KVResourceComparison,
    KVResourceSnapshot,
    compare_resource_snapshots,
)
from tools.validate_kv_long_stability import _make_runtime, _append


def snapshot(**changes):
    values = {
        "sample_index": 0,
        "token_index": 0,
        "monotonic_seconds": 0.0,
        "page_total": 8,
        "page_free": 8,
        "page_allocated": 0,
        "page_peak_allocated": 0,
        "ref_total": 0,
        "owner_total": 0,
        "pin_total": 0,
        "inflight_compute": 0,
        "inflight_io": 0,
        "pending_fences": 0,
        "pending_append": 0,
        "prefix_entries": 0,
        "prefix_pages": 0,
        "prefix_bytes": 0,
        "rgkv_records": 0,
        "rgkv_index_bytes": 0,
        "cuda_allocated": 0,
        "cuda_reserved": 0,
        "cpu_rss": 1024,
        "pinned_bytes": 0,
        "thread_count": 1,
        "future_count": 0,
    }
    values.update(changes)
    return KVResourceSnapshot(**values)


class KVResourceAuditTest(unittest.TestCase):
    def test_capture_tracks_owner_and_release(self):
        runtime = _make_runtime("cpu", 32)
        before = KVResourceSnapshot.capture(runtime)
        state = runtime.create_request(32)
        _append(runtime, state, 3)
        during = KVResourceSnapshot.capture(runtime, 1, 3)
        runtime.release(state)
        after = KVResourceSnapshot.capture(runtime, 2, 3)
        self.assertEqual(during.page_allocated, 1)
        self.assertEqual(during.ref_total, during.owner_total)
        self.assertEqual(after.page_allocated, 0)
        self.assertEqual(
            after.metadata["pinned_bytes_source"], "unavailable_reported_zero"
        )
        self.assertEqual(compare_resource_snapshots(before, after).status, PASS)
        runtime.close()

    def test_true_leak_and_expected_growth_are_distinct(self):
        before = snapshot()
        after = snapshot(page_free=7, page_allocated=1, ref_total=1, owner_total=1)
        failed = compare_resource_snapshots(before, after)
        self.assertEqual(failed.status, FAIL)
        self.assertIn("page_allocated", [item.field for item in failed.leaks])
        allowed = compare_resource_snapshots(
            before,
            after,
            expected_growth={
                "page_allocated": 1,
                "ref_total": 1,
                "owner_total": 1,
            },
        )
        # page_free is a lifecycle counter too and must be explicitly declared.
        self.assertEqual(allowed.status, PASS)

    def test_cuda_reserved_cache_is_not_allocated_leak(self):
        before = snapshot(cuda_allocated=100, cuda_reserved=200)
        cached = snapshot(cuda_allocated=100, cuda_reserved=500)
        result = compare_resource_snapshots(
            before, cached, cuda_allocated_tolerance=0
        )
        self.assertEqual(result.status, PASS)
        reserved = next(item for item in result.drift if item.field == "cuda_reserved")
        self.assertEqual(reserved.classification, "allocator_cache")
        strict = compare_resource_snapshots(
            before,
            cached,
            after_empty_cache=cached,
            cuda_allocated_tolerance=0,
        )
        self.assertEqual(strict.status, FAIL)

    def test_snapshot_invariants_are_a_qualification_gate(self):
        before = snapshot()
        mismatched = snapshot(ref_total=1, owner_total=0)
        result = compare_resource_snapshots(
            before,
            mismatched,
            expected_growth={"ref_total": 1},
        )
        self.assertEqual(result.status, FAIL)
        self.assertIn("ref_owner_invariant", result.as_dict()["leak_fields"])

    def test_empty_cache_final_snapshot_is_strict_and_serializable(self):
        before = snapshot(cuda_reserved=100)
        cached = snapshot(cuda_reserved=500)
        cleared = snapshot(cuda_reserved=100)
        result = compare_resource_snapshots(
            before, cached, after_empty_cache=cleared
        )
        self.assertEqual(result.status, PASS)
        self.assertEqual(result.as_dict()["after_empty_cache"]["cuda_reserved"], 100)

    def test_not_run_status_requires_reason(self):
        self.assertEqual(
            KVResourceComparison.not_run(SKIPPED_WITH_REASON, "no CUDA").status,
            SKIPPED_WITH_REASON,
        )
        self.assertEqual(
            KVResourceComparison.not_run(BLOCKED, "GPU is shared").status,
            BLOCKED,
        )
        with self.assertRaises(ValueError):
            KVResourceComparison.not_run(PASS, "not valid")

    def test_parameterized_snapshot_drift_classification(self):
        cases = (
            ("stable", {}, {}, {}, PASS, ()),
            (
                "declared_page_growth",
                {},
                {
                    "page_free": 7,
                    "page_allocated": 1,
                    "ref_total": 1,
                    "owner_total": 1,
                },
                {"page_allocated": 1, "ref_total": 1, "owner_total": 1},
                PASS,
                (),
            ),
            ("pin_leak", {}, {"pin_total": 1, "inflight_compute": 1}, {}, FAIL, ("pin_total",)),
            ("future_leak", {}, {"future_count": 1}, {}, FAIL, ("future_count",)),
            (
                "allocated_within_tolerance",
                {"cuda_allocated": 100},
                {"cuda_allocated": 104},
                {},
                PASS,
                (),
            ),
        )
        for name, before_values, after_values, expected, status, leak_fields in cases:
            with self.subTest(name=name):
                result = compare_resource_snapshots(
                    snapshot(**before_values),
                    snapshot(**after_values),
                    expected_growth=expected,
                    cuda_allocated_tolerance=8,
                )
                self.assertEqual(result.status, status)
                observed = result.as_dict()["leak_fields"]
                for field in leak_fields:
                    self.assertIn(field, observed)

    def test_snapshot_round_trip_and_invalid_compare_inputs(self):
        original = snapshot(sample_index=7, token_index=11)
        self.assertEqual(
            KVResourceSnapshot.from_dict(original.as_dict()), original
        )
        with self.assertRaises(TypeError):
            compare_resource_snapshots({}, original)
        with self.assertRaises(ValueError):
            compare_resource_snapshots(
                original, original, expected_growth={"page_allocated": -1}
            )


if __name__ == "__main__":
    unittest.main()
