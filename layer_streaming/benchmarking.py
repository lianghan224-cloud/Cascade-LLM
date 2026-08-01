"""Versioned synthetic benchmark result types and aggregation helpers."""

from dataclasses import MISSING, asdict, dataclass, field
import json
import statistics
from typing import Optional, Tuple


BENCHMARK_REPORT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class BenchmarkCase:
    checkpoint: str
    weight_format: str
    backend_requested: str
    granularity: str
    weight_store: str
    vocab_mode: str
    slot_count: int
    prefetch_depth: int
    prefill_tokens: int
    decode_tokens: int
    embedding_placement: str = "streamed"
    lm_head_placement: str = "streamed"
    gpu_resident_weight_budget_bytes: int = 0

    @property
    def case_id(self):
        return "/".join(
            (
                self.weight_format,
                self.backend_requested,
                self.granularity,
                self.weight_store,
                self.vocab_mode,
                "s{}".format(self.slot_count),
                "p{}".format(self.prefetch_depth),
                "e-{}".format(self.embedding_placement),
                "lm-{}".format(self.lm_head_placement),
                "r{}".format(self.gpu_resident_weight_budget_bytes),
            )
        )

    def as_dict(self):
        result = asdict(self)
        result["case_id"] = self.case_id
        return result

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value.pop("case_id", None)
        value.setdefault(
            "embedding_placement", value.get("vocab_mode", "streamed")
        )
        value.setdefault(
            "lm_head_placement", value.get("vocab_mode", "streamed")
        )
        value.setdefault("gpu_resident_weight_budget_bytes", 0)
        return cls(**value)


@dataclass(frozen=True)
class BenchmarkMetrics:
    pageable_to_pinned_ms: float
    combined_weight_scale_h2d_ms: float
    weight_h2d_bytes: int
    scale_h2d_bytes: int
    dequant_ms: float
    gemm_ms: float
    attention_ms: float
    embedding_ms: float
    lm_head_ms: float
    source_wait_ms: float
    compute_wait_ms: float
    ttft_ms: float
    decode_ms_per_token: float
    gpu_peak_allocated_bytes: int
    gpu_peak_reserved_bytes: int
    source_queue_max_depth: int
    ready_queue_max_depth: int
    transformer_weight_h2d_bytes_per_forward: float = 0.0
    lm_head_h2d_bytes_per_forward: float = 0.0
    effective_pageable_to_pinned_gbps: float = 0.0
    effective_h2d_gbps: float = 0.0
    source_stall_ratio: float = 0.0
    compute_stall_ratio: float = 0.0
    overlap_ratio: float = 0.0
    resident_weight_bytes: int = 0
    streamed_weight_bytes: int = 0
    resident_hit_ratio: float = 0.0
    transformer_compute_ms: float = 0.0
    decode_p50_ms: float = 0.0
    decode_p95_ms: float = 0.0
    decode_p99_ms: float = 0.0
    decode_first_window_ms: float = 0.0
    decode_last_window_ms: float = 0.0
    decode_latency_drift_ratio: float = 0.0
    decode_latency_window_tokens: int = 0

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for name, item in cls.__dataclass_fields__.items():
            if name not in value and item.default is not MISSING:
                value[name] = item.default
        return cls(**value)


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    status: str
    skip_reason: Optional[str]
    actual_backends: Tuple[str, ...]
    fallback_backends: Tuple[str, ...]
    samples: Tuple[BenchmarkMetrics, ...]
    median: Optional[BenchmarkMetrics]
    variability: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "case": self.case.as_dict(),
            "status": self.status,
            "skip_reason": self.skip_reason,
            "actual_backends": list(self.actual_backends),
            "fallback_backends": list(self.fallback_backends),
            "samples": [item.as_dict() for item in self.samples],
            "median": (
                None if self.median is None else self.median.as_dict()
            ),
            "variability": dict(self.variability),
        }

    @classmethod
    def from_dict(cls, value):
        return cls(
            case=BenchmarkCase.from_dict(value["case"]),
            status=value["status"],
            skip_reason=value.get("skip_reason"),
            actual_backends=tuple(value.get("actual_backends", ())),
            fallback_backends=tuple(value.get("fallback_backends", ())),
            samples=tuple(
                BenchmarkMetrics.from_dict(item)
                for item in value.get("samples", ())
            ),
            median=(
                None
                if value.get("median") is None
                else BenchmarkMetrics.from_dict(value["median"])
            ),
            variability=dict(value.get("variability", {})),
        )


@dataclass(frozen=True)
class BenchmarkSuiteReport:
    schema_version: int
    synthetic_only: bool
    hardware: dict
    software: dict
    results: Tuple[BenchmarkResult, ...]
    baseline_comparisons: Tuple[dict, ...] = ()

    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "synthetic_only": self.synthetic_only,
            "hardware": dict(self.hardware),
            "software": dict(self.software),
            "results": [item.as_dict() for item in self.results],
            "baseline_comparisons": [
                dict(item) for item in self.baseline_comparisons
            ],
        }

    def to_json(self, indent=2):
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value):
        version = int(value["schema_version"])
        if version not in {1, BENCHMARK_REPORT_SCHEMA_VERSION}:
            raise ValueError(
                "unsupported benchmark report schema version {}".format(
                    version
                )
            )
        return cls(
            schema_version=BENCHMARK_REPORT_SCHEMA_VERSION,
            synthetic_only=bool(value["synthetic_only"]),
            hardware=dict(value["hardware"]),
            software=dict(value["software"]),
            results=tuple(
                BenchmarkResult.from_dict(item)
                for item in value["results"]
            ),
            baseline_comparisons=tuple(
                dict(item)
                for item in value.get("baseline_comparisons", ())
            ),
        )


def median_metrics(samples):
    samples = tuple(samples)
    if not samples:
        return None
    values = {}
    for name in BenchmarkMetrics.__dataclass_fields__:
        items = [getattr(sample, name) for sample in samples]
        median = statistics.median(items)
        field_type = BenchmarkMetrics.__dataclass_fields__[name].type
        values[name] = int(median) if field_type is int else float(median)
    return BenchmarkMetrics(**values)


def metric_variability(samples):
    """Return sample standard deviation and coefficient of variation."""

    samples = tuple(samples)
    result = {}
    for name in BenchmarkMetrics.__dataclass_fields__:
        values = [float(getattr(sample, name)) for sample in samples]
        mean = statistics.mean(values) if values else 0.0
        stddev = statistics.stdev(values) if len(values) > 1 else 0.0
        result[name] = {
            "mean": mean,
            "stddev": stddev,
            "coefficient_of_variation": (
                0.0 if mean == 0.0 else abs(stddev / mean)
            ),
        }
    return result
