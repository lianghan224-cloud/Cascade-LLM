import json
import unittest

from layer_streaming import (
    BENCHMARK_REPORT_SCHEMA_VERSION,
    BenchmarkCase,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkSuiteReport,
    median_metrics,
    metric_variability,
)


def metrics(value):
    values = {
        name: value
        for name in BenchmarkMetrics.__dataclass_fields__
    }
    for name in (
        "weight_h2d_bytes",
        "scale_h2d_bytes",
        "gpu_peak_allocated_bytes",
        "gpu_peak_reserved_bytes",
        "source_queue_max_depth",
        "ready_queue_max_depth",
    ):
        values[name] = int(value)
    return BenchmarkMetrics(**values)


class BenchmarkingTest(unittest.TestCase):
    def test_median_and_schema_round_trip(self):
        case = BenchmarkCase(
            checkpoint="/tmp/model",
            weight_format="bf16",
            backend_requested="bf16_linear",
            granularity="matrix_group",
            weight_store="pinned_staging",
            vocab_mode="streamed",
            slot_count=2,
            prefetch_depth=2,
            prefill_tokens=8,
            decode_tokens=16,
        )
        samples = (metrics(1), metrics(3), metrics(2))
        result = BenchmarkResult(
            case=case,
            status="ok",
            skip_reason=None,
            actual_backends=("bf16_linear",),
            fallback_backends=(),
            samples=samples,
            median=median_metrics(samples),
        )
        self.assertEqual(result.median.ttft_ms, 2.0)
        report = BenchmarkSuiteReport(
            schema_version=BENCHMARK_REPORT_SCHEMA_VERSION,
            synthetic_only=True,
            hardware={"gpu": "test"},
            software={"torch": "test"},
            results=(result,),
        )
        restored = BenchmarkSuiteReport.from_dict(
            json.loads(report.to_json())
        )
        self.assertEqual(restored.as_dict(), report.as_dict())

    def test_unknown_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            BenchmarkSuiteReport.from_dict(
                {
                    "schema_version": 999,
                    "synthetic_only": True,
                    "hardware": {},
                    "software": {},
                    "results": [],
                }
            )

    def test_schema_v1_is_read_with_m6_defaults(self):
        case = BenchmarkCase(
            checkpoint="/tmp/model",
            weight_format="bf16",
            backend_requested="bf16_linear",
            granularity="matrix_group",
            weight_store="pinned_staging",
            vocab_mode="streamed",
            slot_count=2,
            prefetch_depth=2,
            prefill_tokens=8,
            decode_tokens=16,
        )
        payload = BenchmarkSuiteReport(
            schema_version=BENCHMARK_REPORT_SCHEMA_VERSION,
            synthetic_only=True,
            hardware={},
            software={},
            results=(
                BenchmarkResult(
                    case=case,
                    status="ok",
                    skip_reason=None,
                    actual_backends=("bf16_linear",),
                    fallback_backends=(),
                    samples=(metrics(1),),
                    median=metrics(1),
                ),
            ),
        ).as_dict()
        payload["schema_version"] = 1
        for item in payload["results"]:
            item["case"].pop("embedding_placement")
            item["case"].pop("lm_head_placement")
            item["case"].pop("gpu_resident_weight_budget_bytes")
            item.pop("variability")
            for sample in item["samples"] + [item["median"]]:
                for name in (
                    "transformer_weight_h2d_bytes_per_forward",
                    "lm_head_h2d_bytes_per_forward",
                    "effective_pageable_to_pinned_gbps",
                    "effective_h2d_gbps",
                    "source_stall_ratio",
                    "compute_stall_ratio",
                    "overlap_ratio",
                    "resident_weight_bytes",
                    "streamed_weight_bytes",
                    "resident_hit_ratio",
                    "transformer_compute_ms",
                ):
                    sample.pop(name)
        restored = BenchmarkSuiteReport.from_dict(payload)
        self.assertEqual(restored.schema_version, BENCHMARK_REPORT_SCHEMA_VERSION)
        self.assertEqual(restored.results[0].median.resident_weight_bytes, 0)

    def test_variability_reports_cv(self):
        result = metric_variability((metrics(10), metrics(10), metrics(10)))
        self.assertEqual(result["ttft_ms"]["stddev"], 0.0)
        self.assertEqual(
            result["decode_ms_per_token"]["coefficient_of_variation"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
