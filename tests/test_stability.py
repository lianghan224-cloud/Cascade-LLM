import json
import unittest

import torch

from layer_streaming import (
    KVCacheManager,
    ResourceSnapshot,
    StabilityReport,
    StabilityThresholds,
    analyze_stability,
    capture_resource_snapshot,
)


def snapshot(index, latency, *, allocated=100, reserved=200, pinned=300):
    return ResourceSnapshot(
        sample_index=index,
        token_index=index,
        monotonic_seconds=float(index),
        token_latency_ms=latency,
        cuda_allocated_bytes=allocated,
        cuda_reserved_bytes=reserved,
        pinned_bytes=pinned,
        pageable_bytes=400,
        thread_count=3,
        cascade_thread_count=2,
        event_count=9,
        kv_total_blocks=8,
        kv_allocated_blocks=1,
        kv_active_handles=1,
        queue_depths={"ready_queue": 0},
        queue_capacities={"ready_queue": 2},
    )


class StabilityTest(unittest.TestCase):
    def test_stable_series_passes_and_round_trips(self):
        report = analyze_stability(
            [snapshot(index, 2.0 + (index % 2) * 0.01) for index in range(20)]
        )
        self.assertTrue(report.passed, report.failures)
        restored = StabilityReport.from_dict(
            json.loads(report.to_json())
        )
        self.assertEqual(restored.as_dict(), report.as_dict())

    def test_memory_and_latency_drift_fail(self):
        values = [
            snapshot(index, 1.0, allocated=100)
            for index in range(5)
        ] + [
            snapshot(index, 2.0, allocated=140)
            for index in range(5, 10)
        ]
        report = analyze_stability(
            values,
            thresholds=StabilityThresholds(
                cuda_allocated_drift_bytes=10,
                cuda_reserved_drift_bytes=0,
                pinned_drift_bytes=0,
                thread_count_drift=0,
                event_count_drift=0,
                kv_block_drift=0,
                latency_tail_ratio=1.25,
            ),
            window=3,
        )
        self.assertFalse(report.passed)
        self.assertTrue(
            any("cuda_allocated_bytes" in item for item in report.failures)
        )
        self.assertTrue(
            any("latency tail ratio" in item for item in report.failures)
        )

    def test_kv_resource_snapshot_tracks_release(self):
        manager = KVCacheManager(
            layer_count=1,
            num_key_value_heads=1,
            head_dim=4,
            total_blocks=4,
            block_size=2,
            dtype=torch.float32,
            device="cpu",
        )
        handle = manager.allocate(max_length=3)
        during = capture_resource_snapshot(
            0, 0, kv_manager=manager
        )
        manager.release(handle)
        after = capture_resource_snapshot(
            1, 0, kv_manager=manager
        )
        self.assertEqual(during.kv_allocated_blocks, 2)
        self.assertEqual(during.kv_active_handles, 1)
        self.assertEqual(after.kv_allocated_blocks, 0)
        self.assertEqual(after.kv_active_handles, 0)
        manager.close()
        manager.close()


if __name__ == "__main__":
    unittest.main()
