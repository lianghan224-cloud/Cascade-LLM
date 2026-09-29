import unittest

from layer_streaming.timeline import analyze_copy_compute_timeline


class TimelineAccountingTest(unittest.TestCase):
    def test_overlap_is_counted_once(self):
        result = analyze_copy_compute_timeline(
            copy_intervals=((0, 4), (5, 9)),
            compute_intervals=((2, 7), (8, 10)),
            wall_ms=12,
        )
        self.assertAlmostEqual(result["copy_busy_ms"], 8.0)
        self.assertAlmostEqual(result["compute_busy_ms"], 7.0)
        self.assertAlmostEqual(result["copy_compute_overlap_ms"], 5.0)
        self.assertAlmostEqual(result["copy_only_ms"], 3.0)
        self.assertAlmostEqual(result["compute_only_ms"], 2.0)
        self.assertAlmostEqual(result["gpu_timeline_active_ms"], 10.0)
        self.assertAlmostEqual(
            result["gpu_timeline_idle_or_host_overhead_ms"], 2.0
        )
        self.assertAlmostEqual(result["critical_path_accounted_ms"], 12.0)
        self.assertAlmostEqual(
            result["critical_path_accounting_error_ms"], 0.0
        )

    def test_overlapping_same_stream_intervals_are_merged(self):
        result = analyze_copy_compute_timeline(
            copy_intervals=((1, 5), (3, 7)),
            compute_intervals=(),
            wall_ms=10,
        )
        self.assertAlmostEqual(result["copy_busy_ms"], 6.0)
        self.assertAlmostEqual(result["copy_only_ms"], 6.0)
        self.assertAlmostEqual(
            result["gpu_timeline_idle_or_host_overhead_ms"], 4.0
        )

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            analyze_copy_compute_timeline(((2, 1),), (), 3)


if __name__ == "__main__":
    unittest.main()
