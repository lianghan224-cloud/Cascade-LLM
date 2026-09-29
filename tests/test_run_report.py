import json
import tempfile
from pathlib import Path
import unittest

from layer_streaming import (
    ExecutionPolicy,
    LlamaModelAdapter,
    MemoryPlanner,
    SystemCapacity,
    build_inference_report,
)

from test_plan import tiny_config


class RunReportTest(unittest.TestCase):
    def test_report_is_json_serializable_and_aggregates_profiles(self):
        config = tiny_config(num_hidden_layers=1, vocab_size=31)
        adapter = LlamaModelAdapter()
        geometry = adapter.build_geometry(config)
        policy = ExecutionPolicy(vocab_chunk_bytes=4096)
        plan = adapter.build_plan(config, policy)
        validation = type(
            "Validation",
            (),
            {"as_dict": lambda self: {"ok": True}},
        )()
        preflight = MemoryPlanner(
            plan,
            geometry,
            policy=policy,
            max_context=16,
            max_prefill_tokens=4,
            cuda_safety_margin_bytes=0,
        ).preflight(
            capacity=SystemCapacity(None, None, None, None),
            raise_on_error=False,
        )
        profiles = [
            {
                "h2d_event_sum_ms": 2.0,
                "compute_event_sum_ms": 3.0,
                "attention_event_sum_ms": 1.0,
                "kv_append_event_sum_ms": 0.1,
                "kv_attention_event_sum_ms": 0.8,
                "mlp_event_sum_ms": 2.0,
                "source_prepare_wait_ms": 0.5,
                "ready_wait_ms": 0.25,
                "free_slot_wait_ms": 0.125,
                "source_queue_max_depth": 2,
                "ready_queue_max_depth": 1,
                "source_queue_capacity": 3,
                "ready_queue_capacity": 3,
                "inference_phase": "prefill",
                "h2d_bytes": 100,
                "weight_h2d_bytes": 90,
                "transfer_slot_reuse_counts": [2, 1],
                "copy_busy_ms": 2.0,
                "compute_busy_ms": 3.0,
                "copy_compute_overlap_ms": 1.0,
                "copy_only_ms": 1.0,
                "compute_only_ms": 2.0,
                "gpu_timeline_idle_or_host_overhead_ms": 0.5,
                "copy_compute_timeline": {"copy_busy_ms": 2.0},
                "transfer_timeline": [{"unit_id": "layer0.q"}],
            },
            {
                "h2d_event_sum_ms": 4.0,
                "compute_event_sum_ms": 5.0,
                "attention_event_sum_ms": 2.0,
                "kv_append_event_sum_ms": 0.2,
                "kv_attention_event_sum_ms": 1.6,
                "mlp_event_sum_ms": 3.0,
                "source_prepare_wait_ms": 1.0,
                "ready_wait_ms": 0.5,
                "free_slot_wait_ms": 0.25,
                "source_queue_max_depth": 3,
                "ready_queue_max_depth": 2,
                "source_queue_capacity": 3,
                "ready_queue_capacity": 3,
                "inference_phase": "decode",
                "h2d_bytes": 100,
                "weight_h2d_bytes": 90,
                "transfer_slot_reuse_counts": [1, 2],
                "copy_busy_ms": 4.0,
                "compute_busy_ms": 5.0,
                "copy_compute_overlap_ms": 2.0,
                "copy_only_ms": 2.0,
                "compute_only_ms": 3.0,
                "gpu_timeline_idle_or_host_overhead_ms": 1.0,
                "copy_compute_timeline": {"copy_busy_ms": 4.0},
                "transfer_timeline": [{"unit_id": "layer0.q"}],
            },
        ]
        report = build_inference_report(
            plan=plan,
            geometry=geometry,
            policy=policy,
            checkpoint="/tmp/tiny",
            validation=validation,
            preflight=preflight,
            checkpoint_load_seconds=1.25,
            prompt_tokens=4,
            generated_tokens=2,
            generation_wall_seconds=0.5,
            time_to_first_token_ms=100.0,
            decode_token_latencies_ms=[50.0],
            runtime_profiles=profiles,
            vocab_profiles=[
                {"wall_ms": 2.0, "embedding_wall_ms": 0.5}
            ],
            gpu_peak_memory_bytes=123,
            cpu_resident_bytes=456,
            pinned_bytes=78,
            kv_cache_bytes=90,
            kv_profiles=[
                {
                    "attention_backend": "dense_paged_reference",
                    "attention_accuracy": "exact",
                    "attention_wall_ms": 1.5,
                    "layout": "hnd",
                    "append_calls": 2,
                    "appended_tokens": 5,
                    "attention_calls": 2,
                    "materialize_calls": 0,
                    "materialized_bytes": 0,
                    "paged_attention_provider": "sm86",
                    "paged_kv_kernel_backend": "sm86_kv_kernel",
                    "paged_provider_bundle": "sm86",
                    "tier_metrics_sampled": True,
                    "gpu_kv_capacity_bytes": 4096,
                    "gpu_kv_used_bytes": 2048,
                    "gpu_kv_free_bytes": 2048,
                    "gpu_kv_capacity_pages": 4,
                    "gpu_kv_used_pages": 2,
                    "gpu_kv_free_pages": 2,
                    "gpu_kv_peak_used_pages": 3,
                    "gpu_kv_high_watermark_bytes": 3072,
                    "gpu_kv_low_watermark_bytes": 1024,
                    "cpu_kv_capacity_bytes": 16384,
                    "cpu_kv_used_bytes": 8192,
                    "gpu_hits": 7,
                    "cpu_hits": 3,
                    "prefetch_count": 3,
                    "prefetch_pages": 4,
                    "prefetch_bytes": 6144,
                    "prefetch_wait_ms": 1.25,
                    "prefetch_timeouts": 1,
                    "eviction_count": 2,
                    "eviction_pages": 3,
                    "eviction_bytes": 4096,
                    "h2d_kv_bytes": 6144,
                    "d2h_kv_bytes": 4096,
                    "migration_failures": 1,
                    "authority_changes": 5,
                    "thrashing_count": 1,
                    "thrash_window_operations": 8,
                    "provider_fallback_count": 1,
                    "provider_reference_fallback_count": 1,
                    "provider_routing_summary": {
                        "total_calls": 2,
                        "fallback_calls": 1,
                        "reference_fallback_calls": 1,
                        "decisions": [
                            {
                                "workload": "full_prefill",
                                "selected": "reference_paged_exact",
                                "attention_backend": "reference_paged_exact",
                                "fallback_reason": "sm86 has no full_prefill kernel",
                                "is_reference": True,
                                "count": 1,
                            },
                            {
                                "workload": "decode",
                                "selected": "sm86",
                                "attention_backend": "sm86",
                                "fallback_reason": None,
                                "count": 1,
                            },
                        ],
                    },
                    "policy": {
                        "accuracy": "exact",
                        "storage": "gpu",
                        "dtype": "bf16",
                        "selection": "none",
                        "reuse": "none",
                        "page_size": 16,
                    },
                }
            ],
            finish_profiles=[
                {
                    "phase": "prefill",
                    "finish_host_enqueue_ms": 0.5,
                    "finish_cuda_ms": 1.25,
                },
                {
                    "phase": "decode",
                    "finish_host_enqueue_ms": 0.25,
                    "finish_cuda_ms": 0.75,
                },
            ],
        )
        payload = json.loads(report.to_json())
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["timings"]["h2d_time_ms"], 6.0)
        self.assertEqual(payload["timings"]["compute_time_ms"], 8.0)
        self.assertEqual(payload["pipeline"]["source_queue_max_depth"], 3)
        self.assertEqual(payload["pipeline"]["h2d_bytes"], 200.0)
        self.assertEqual(
            payload["pipeline"]["transfer_slot_reuse_counts"], [3, 3]
        )
        self.assertEqual(payload["timings"]["h2d_compute_overlap_ms"], 3.0)
        self.assertAlmostEqual(payload["timings"]["kv_append_time_ms"], 0.3)
        self.assertAlmostEqual(
            payload["timings"]["kv_paged_attention_time_ms"], 2.4
        )
        self.assertEqual(payload["timings"]["lm_head_time_ms"], 2.0)
        self.assertEqual(
            payload["timings"]["finish_host_enqueue_time_ms"], 0.75
        )
        self.assertEqual(
            payload["pipeline"]["transfer_timelines"][1]["phase"], "decode"
        )
        self.assertEqual(payload["throughput"]["tokens_per_second"], 4.0)
        self.assertEqual(
            payload["runtime_config"]["weight_format"], "bf16"
        )
        self.assertEqual(
            payload["runtime_config"]["kv_policy"]["accuracy"], "exact"
        )
        self.assertEqual(payload["pipeline"]["kv"]["materialized_bytes"], 0)
        self.assertEqual(
            payload["pipeline"]["kv"]["paged_attention_provider"],
            "sm86",
        )
        self.assertEqual(
            payload["pipeline"]["kv"]["paged_kv_kernel_backend"],
            "sm86_kv_kernel",
        )
        self.assertAlmostEqual(payload["timings"]["kv_attention_time_ms"], 2.4)
        self.assertEqual(
            payload["timings"]["kv_attention_host_dispatch_time_ms"], 1.5
        )
        self.assertEqual(
            payload["pipeline"]["kv"]["provider_fallback_count"], 1
        )
        self.assertEqual(
            payload["pipeline"]["kv"][
                "provider_reference_fallback_count"
            ],
            1,
        )
        self.assertEqual(
            payload["pipeline"]["kv"]["provider_routing_summary"]["total_calls"],
            2,
        )
        tier = payload["pipeline"]["kv"]
        self.assertTrue(tier["tier_metrics_sampled"])
        self.assertEqual(tier["gpu_kv_capacity_pages"], 4)
        self.assertEqual(tier["gpu_kv_free_bytes"], 2048)
        self.assertEqual(tier["cpu_kv_used_bytes"], 8192)
        self.assertEqual(tier["prefetch_wait_ms"], 1.25)
        self.assertEqual(tier["prefetch_pages"], 4)
        self.assertEqual(tier["prefetch_timeouts"], 1)
        self.assertEqual(tier["eviction_pages"], 3)
        self.assertEqual(tier["eviction_bytes"], 4096)
        self.assertEqual(tier["h2d_kv_bytes"], 6144)
        self.assertEqual(tier["d2h_kv_bytes"], 4096)
        self.assertEqual(tier["authority_changes"], 5)
        self.assertEqual(tier["thrashing_count"], 1)
        self.assertEqual(tier["thrash_window_operations"], 8)
        self.assertIn("gpu_weight_budget_bytes", payload["memory"])
        self.assertIn("kv_admission_capacity_bytes", payload["memory"])
        with tempfile.TemporaryDirectory() as directory:
            path = report.write(Path(directory) / "run_report.json")
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
