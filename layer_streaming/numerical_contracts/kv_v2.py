"""Model-level Numerical Contract V2 evaluation for Paged KV providers."""

from dataclasses import dataclass
import json
from pathlib import Path


KV_NUMERICAL_CONTRACT_VERSION = 2


@dataclass(frozen=True)
class KVNumericalContractV2:
    name: str
    architecture: str
    provider_name: str
    provider_abi: int
    model_id: str
    kv_dtype: str
    layout: str
    stage_envelopes: dict
    logits_generation: dict
    quality: dict
    stability: dict
    schema_version: int = KV_NUMERICAL_CONTRACT_VERSION

    def __post_init__(self):
        if int(self.schema_version) != KV_NUMERICAL_CONTRACT_VERSION:
            raise ValueError("KV Numerical Contract must use schema V2")
        required = {"attention", "residual", "mlp", "final_norm", "logits"}
        missing = required.difference(self.stage_envelopes)
        if missing:
            raise ValueError(
                "stage envelopes missing {}".format(", ".join(sorted(missing)))
            )

    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "architecture": self.architecture,
            "provider_name": self.provider_name,
            "provider_abi": self.provider_abi,
            "model_id": self.model_id,
            "kv_dtype": self.kv_dtype,
            "layout": self.layout,
            "stage_envelopes": self.stage_envelopes,
            "logits_generation": self.logits_generation,
            "quality": self.quality,
            "stability": self.stability,
        }


def load_kv_numerical_contract(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return KVNumericalContractV2(**payload)


def _component(stage_key):
    raw = stage_key.split("/")[1]
    return "residual" if raw == "hidden" else raw


def evaluate_model_stages(contract, comparison_report, attribution_report):
    """Evaluate L2 without treating HF-SDPA as the numerical truth."""

    comparisons = comparison_report.get("comparisons", {})
    observations = {}
    violations = []
    for component, envelope in contract.stage_envelopes.items():
        items = [
            value
            for key, value in comparisons.items()
            if _component(key) == component
        ]
        observed = {
            "count": len(items),
            "max_abs_error": max(
                (float(item.get("max_abs_error", 0.0)) for item in items),
                default=0.0,
            ),
            "mean_abs_error_max": max(
                (float(item.get("mean_abs_error", 0.0)) for item in items),
                default=0.0,
            ),
            "max_relative_error": max(
                (float(item.get("max_relative_error", 0.0)) for item in items),
                default=0.0,
            ),
        }
        checks = {
            "present": bool(items),
            "max_abs_error": (
                observed["max_abs_error"] <= float(envelope["max_abs_error"])
            ),
            "mean_abs_error": (
                observed["mean_abs_error_max"]
                <= float(envelope["mean_abs_error_max"])
            ),
        }
        observed["checks"] = checks
        observed["envelope"] = envelope
        observed["passed"] = all(checks.values())
        observations[component] = observed
        if not observed["passed"]:
            violations.append("{} exceeds versioned envelope".format(component))

    attribution_complete = bool(
        attribution_report.get("conclusion", {}).get(
            "all_failed_stages_attributed", False
        )
    )
    if not attribution_complete:
        violations.append("one or more strict diagnostic failures are unexplained")
    return {
        "level": "L2",
        "passed": not violations,
        "reference_role": (
            "versioned provider/model envelope; HF SDPA/eager are diagnostics"
        ),
        "strict_hf_diagnostic_failure_count": int(
            attribution_report.get("production_failed_stage_count", 0)
        ),
        "strict_hf_identity_required": False,
        "attribution_complete": attribution_complete,
        "unexplained_spike_count": len(violations),
        "observations": observations,
        "violations": violations,
    }


def evaluate_logits_generation(contract, comparison_report, long_replay):
    rules = contract.logits_generation
    logits = [
        value
        for key, value in comparison_report.get("comparisons", {}).items()
        if _component(key) == "logits"
    ]
    fixed_top1 = all(bool(item.get("top1_equal", False)) for item in logits)
    fixed_top10 = min(
        (float(item.get("topk_consistency", 0.0)) for item in logits),
        default=0.0,
    )
    teacher_forced_top1 = float(long_replay.get("top1_agreement_rate", 0.0))
    checks = {
        "fixed_prompt_top1": fixed_top1,
        "fixed_prompt_top10": (
            fixed_top10 >= float(rules["fixed_prompt_top10_min"])
        ),
        "teacher_forced_top1": (
            teacher_forced_top1
            >= float(rules["teacher_forced_top1_min"])
        ),
        "teacher_forced_coverage": int(
            long_replay.get("sampled_positions", 0)
        )
        >= int(rules.get("teacher_forced_min_positions", 1)),
        "finite": bool(long_replay.get("all_hf_logits_finite", False)),
    }
    return {
        "level": "L3",
        "passed": all(checks.values()),
        "checks": checks,
        "fixed_prompt_count": len(logits),
        "fixed_prompt_top10_min": fixed_top10,
        "teacher_forced_positions": int(long_replay.get("sampled_positions", 0)),
        "teacher_forced_top1": teacher_forced_top1,
    }


def evaluate_model_quality(contract, quality_report=None):
    rules = contract.quality
    if not quality_report:
        return {
            "level": "L4",
            "passed": False,
            "status": "not_run",
            "missing": [
                "perplexity_dataset",
                "short_text_understanding",
                "long_context_retrieval",
                "fixed_dialogue_set",
            ],
        }
    values = quality_report.get("metrics", {})
    coverage = quality_report.get("coverage", {})
    checks = {
        "finite": bool(quality_report.get("all_finite", False)),
        "perplexity_coverage": int(coverage.get("perplexity_tokens", 0))
        >= int(rules.get("perplexity_min_tokens", 1)),
        "short_coverage": int(coverage.get("short_examples", 0))
        >= int(rules.get("short_min_examples", 1)),
        "long_context_coverage": int(
            coverage.get("long_context_examples", 0)
        ) >= int(rules.get("long_min_examples", 1)),
        "dialogue_coverage": int(coverage.get("dialogue_examples", 0))
        >= int(rules.get("dialogue_min_examples", 1)),
        "perplexity": float(values.get("perplexity_relative_degradation", 1.0))
        <= float(rules["perplexity_relative_degradation_max"]),
        "short_text": float(values.get("short_accuracy_drop_pp", 100.0))
        <= float(rules["short_accuracy_drop_pp_max"]),
        "long_context": float(values.get("long_hit_rate_drop_pp", 100.0))
        <= float(rules["long_hit_rate_drop_pp_max"]),
        "dialogue": float(values.get("dialogue_accuracy_drop_pp", 100.0))
        <= float(rules["dialogue_accuracy_drop_pp_max"]),
    }
    return {
        "level": "L4",
        "passed": all(checks.values()),
        "status": "complete",
        "checks": checks,
        "metrics": values,
        "coverage": coverage,
    }


def evaluate_production_stability(
    contract,
    soak_report,
    ownership_report,
    performance_reports,
):
    rules = contract.stability
    acceptance = soak_report.get("acceptance", {})
    ownership = ownership_report.get("acceptance", {})
    ownership_drift = ownership_report.get("resource_drift", {})
    performance_results = [
        item
        for report in performance_reports
        for item in report.get("results", ())
        if item.get("supported", False)
        and item.get("provider") not in {
            "reference_paged_exact",
            "legacy_gather_sdpa_reference",
        }
    ]
    checks = {
        "minimum_tokens": int(soak_report.get("decode_tokens_per_cycle", 0))
        >= int(rules["minimum_generated_tokens"]),
        "cuda_allocated_zero_drift": int(
            soak_report.get("resource_drift", {}).get("cuda_allocated_bytes", -1)
        )
        == 0,
        "cuda_reserved_zero_drift": int(
            soak_report.get("resource_drift", {}).get("cuda_reserved_bytes", -1)
        )
        == 0,
        "pages_released": bool(acceptance.get("all_pages_released", False)),
        "ref_count_exact": bool(
            acceptance.get("long_request_ref_count_exact", False)
        ),
        "pin_count_zero": bool(
            acceptance.get("quiescent_pin_count_zero", False)
        ),
        "workspace_non_linear": bool(performance_results)
        and all(int(item.get("workspace_peak_bytes", -1)) == 0 for item in performance_results),
        "ownership_1000_cycle": all(bool(value) for value in ownership.values()),
        "ownership_cuda_allocated_zero_drift": int(
            ownership_drift.get("cuda_allocated_bytes", -1)
        )
        == 0,
        "ownership_cuda_reserved_zero_drift": int(
            ownership_drift.get("cuda_reserved_bytes", -1)
        )
        == 0,
        "latency_no_sustained_degradation": float(
            soak_report.get("latency_ms", {}).get(
                "last_vs_first_quartile_ratio", float("inf")
            )
        )
        <= float(rules["last_first_quartile_ratio_max"]),
    }
    return {
        "level": "L5",
        "passed": all(checks.values()),
        "checks": checks,
        "tokens": int(soak_report.get("decode_tokens_per_cycle", 0)),
        "latency_ms": soak_report.get("latency_ms", {}),
        "resource_drift": soak_report.get("resource_drift", {}),
        "ownership_resource_drift": ownership_drift,
        "ownership_allocator_cache_before_trim": ownership_report.get(
            "allocator_cache_before_trim", {}
        ),
    }
