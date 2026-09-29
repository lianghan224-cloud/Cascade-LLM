import unittest

from layer_streaming.kv.metrics import KVMetrics


class KVMetricsSchemaTest(unittest.TestCase):
    def test_tier_metrics_are_initialized_without_claiming_sampling(self):
        payload = KVMetrics().as_dict()
        self.assertFalse(payload["tier_metrics_sampled"])
        integer_fields = (
            "gpu_kv_capacity_bytes",
            "gpu_kv_used_bytes",
            "gpu_kv_free_bytes",
            "gpu_kv_peak_used_bytes",
            "gpu_kv_peak_used_pages",
            "gpu_kv_capacity_pages",
            "gpu_kv_used_pages",
            "gpu_kv_free_pages",
            "gpu_kv_high_watermark_bytes",
            "gpu_kv_low_watermark_bytes",
            "gpu_kv_high_watermark_pages",
            "gpu_kv_low_watermark_pages",
            "cpu_kv_capacity_bytes",
            "cpu_kv_used_bytes",
            "cpu_kv_reserved_bytes",
            "cpu_kv_free_bytes",
            "cpu_kv_peak_used_bytes",
            "cpu_kv_capacity_pages",
            "cpu_kv_used_pages",
            "cpu_kv_reserved_pages",
            "cpu_kv_free_pages",
            "cpu_kv_high_watermark_bytes",
            "cpu_kv_low_watermark_bytes",
            "gpu_hits",
            "cpu_hits",
            "prefetch_count",
            "prefetch_bytes",
            "prefetch_timeouts",
            "eviction_count",
            "eviction_bytes",
            "h2d_kv_bytes",
            "d2h_kv_bytes",
            "migration_failures",
            "migration_cancellations",
            "authority_changes",
            "tier_version_mismatches",
            "thrashing_count",
            "thrash_window_operations",
            "selected_metadata_d2h",
            "selected_metadata_d2h_bytes",
            "host_scalar_readbacks",
            "explicit_sync_count",
            "python_attention_wave_count",
            "device_view_fallback_count",
            "gpu_hit_device_attention_calls",
        )
        for name in integer_fields:
            self.assertEqual(payload[name], 0, name)
        self.assertEqual(payload["prefetch_wait_ms"], 0.0)
        self.assertFalse(payload["gpu_kv_above_high_watermark"])
        self.assertFalse(payload["gpu_kv_below_low_watermark"])
        self.assertFalse(payload["cpu_kv_above_high_watermark"])
        self.assertFalse(payload["cpu_kv_below_low_watermark"])

    def test_tier_observation_uses_absolute_cumulative_snapshots(self):
        metrics = KVMetrics()
        gpu = {
            "gpu_kv_capacity_bytes": 4096,
            "gpu_kv_used_bytes": 2048,
            "gpu_kv_free_bytes": 2048,
            "gpu_kv_capacity_pages": 4,
            "gpu_kv_used_pages": 2,
            "gpu_kv_free_pages": 2,
            "gpu_kv_peak_used_pages": 3,
            "gpu_kv_peak_used_bytes": 3072,
        }
        cpu = {
            "capacity_bytes": 8192,
            "used_bytes": 2048,
            "reserved_bytes": 1024,
            "free_bytes": 5120,
            "slot_count": 8,
            "free_slots": 5,
            "layer_page_bytes": 1024,
            "high_watermark_bytes": 7168,
            "low_watermark_bytes": 4096,
        }
        migration = {
            "h2d_kv_bytes": 1024,
            "d2h_kv_bytes": 2048,
            "migration_failures": 1,
            "authority_changes": 3,
        }
        operations = {
            "gpu_hits": 7,
            "cpu_hits": 2,
            "prefetch_count": 2,
            "prefetch_bytes": 1024,
            "prefetch_wait_ms": 1.5,
            "eviction_count": 1,
            "thrashing_count": 1,
            "thrash_window_operations": 8,
        }
        metrics.observe_tier(
            gpu_cache=gpu,
            cpu_store=cpu,
            migration=migration,
            operations=operations,
        )
        # Publishing the same cumulative snapshot twice must not double any
        # counter merely because profile_stats() was called twice.
        metrics.observe_tier(
            gpu_cache=gpu,
            cpu_store=cpu,
            migration=migration,
            operations=operations,
        )
        payload = metrics.as_dict()
        self.assertTrue(payload["tier_metrics_sampled"])
        self.assertEqual(payload["gpu_kv_peak_used_pages"], 3)
        self.assertEqual(payload["cpu_kv_used_pages"], 2)
        self.assertEqual(payload["cpu_kv_reserved_pages"], 1)
        self.assertEqual(payload["cpu_kv_free_pages"], 5)
        self.assertEqual(payload["prefetch_count"], 2)
        self.assertEqual(payload["h2d_kv_bytes"], 1024)
        self.assertEqual(payload["authority_changes"], 3)
        self.assertEqual(payload["thrashing_count"], 1)
        self.assertEqual(payload["thrash_window_operations"], 8)

    def test_empty_tier_observation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            KVMetrics().observe_tier()

    def test_decode_execution_audit_metrics_are_explicit_host_deltas(self):
        metrics = KVMetrics()
        metrics.observe_decode_execution(
            selected_metadata_d2h=1,
            selected_metadata_d2h_bytes=128,
            host_scalar_readbacks=2,
            explicit_sync_count=3,
            python_attention_wave_count=16,
            device_view_fallback_count=1,
            gpu_hit_device_attention_calls=4,
        ).observe_decode_execution(
            selected_metadata_d2h_bytes=64,
            gpu_hit_device_attention_calls=1,
        )
        payload = metrics.as_dict()
        self.assertEqual(payload["selected_metadata_d2h"], 1)
        self.assertEqual(payload["selected_metadata_d2h_bytes"], 192)
        self.assertEqual(payload["host_scalar_readbacks"], 2)
        self.assertEqual(payload["explicit_sync_count"], 3)
        self.assertEqual(payload["python_attention_wave_count"], 16)
        self.assertEqual(payload["device_view_fallback_count"], 1)
        self.assertEqual(payload["gpu_hit_device_attention_calls"], 5)
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            metrics.observe_decode_execution(explicit_sync_count=-1)


if __name__ == "__main__":
    unittest.main()
