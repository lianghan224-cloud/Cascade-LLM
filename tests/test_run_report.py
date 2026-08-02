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
                "mlp_event_sum_ms": 2.0,
                "source_prepare_wait_ms": 0.5,
                "ready_wait_ms": 0.25,
                "free_slot_wait_ms": 0.125,
                "source_queue_max_depth": 2,
                "ready_queue_max_depth": 1,
                "source_queue_capacity": 3,
                "ready_queue_capacity": 3,
            },
            {
                "h2d_event_sum_ms": 4.0,
                "compute_event_sum_ms": 5.0,
                "attention_event_sum_ms": 2.0,
                "mlp_event_sum_ms": 3.0,
                "source_prepare_wait_ms": 1.0,
                "ready_wait_ms": 0.5,
                "free_slot_wait_ms": 0.25,
                "source_queue_max_depth": 3,
                "ready_queue_max_depth": 2,
                "source_queue_capacity": 3,
                "ready_queue_capacity": 3,
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
        )
        payload = json.loads(report.to_json())
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["timings"]["h2d_time_ms"], 6.0)
        self.assertEqual(payload["timings"]["compute_time_ms"], 8.0)
        self.assertEqual(payload["pipeline"]["source_queue_max_depth"], 3)
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
        self.assertEqual(payload["timings"]["kv_attention_time_ms"], 1.5)
        with tempfile.TemporaryDirectory() as directory:
            path = report.write(Path(directory) / "run_report.json")
            self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
