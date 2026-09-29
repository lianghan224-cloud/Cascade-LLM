import tempfile
import unittest

from tools.audit_rgkv_hot_path import build_report, write_report


class RGKVHotPathAuditTest(unittest.TestCase):
    def test_gpu_hit_gate_never_hides_tier_miss_host_bridge(self):
        report = build_report()
        self.assertEqual(report["device_components_status"], "PASS")
        self.assertEqual(
            report["end_to_end_decode_status"],
            "HOST_SYNC_FREE_GPU_HIT_PATH",
        )
        self.assertTrue(report["gpu_hit_path_static_pass"])
        self.assertEqual(report["tier_miss_path_status"], "BLOCKED_HOST_SYNC")
        self.assertFalse(report["production_zero_host_sync"])
        components = [
            item
            for item in report["scopes"]
            if item["category"] == "device_component"
        ]
        self.assertTrue(components)
        self.assertTrue(all(item["status"] == "PASS" for item in components))
        self.assertTrue(
            any(
                item["forbidden_calls"] or item["python_loop_count"]
                for item in report["scopes"]
                if item["category"] == "host_bridge"
            )
        )

    def test_report_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            report, paths = write_report(directory)
            self.assertFalse(report["production_zero_host_sync"])
            self.assertTrue(report["gpu_hit_path_static_pass"])
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
