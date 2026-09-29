"""Stable JSON run-report schema and profile aggregation."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import platform
import time

import torch

from .hardware import HardwareDetector


RUN_REPORT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class RunReport:
    model: dict
    runtime_config: dict
    timings: dict
    throughput: dict
    memory: dict
    pipeline: dict
    checkpoint_validation: dict
    memory_preflight: dict
    hardware: dict
    schema_version: int = RUN_REPORT_SCHEMA_VERSION
    created_at_unix: float = field(default_factory=time.time)

    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "created_at_unix": self.created_at_unix,
            "model": self.model,
            "runtime_config": self.runtime_config,
            "timings": self.timings,
            "throughput": self.throughput,
            "memory": self.memory,
            "pipeline": self.pipeline,
            "checkpoint_validation": self.checkpoint_validation,
            "memory_preflight": self.memory_preflight,
            "hardware": self.hardware,
        }

    def to_json(self, indent=2):
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version = int(value.get("schema_version", 0))
        if version != RUN_REPORT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported RunReport schema version {}".format(version)
            )
        return cls(**value)

    @classmethod
    def from_json(cls, payload):
        return cls.from_dict(json.loads(payload))

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path


def _sum(profiles, key):
    values = [profile.get(key) for profile in profiles if profile]
    values = [value for value in values if isinstance(value, (int, float))]
    return float(sum(values)) if values else None


def _max(profiles, key):
    values = [profile.get(key) for profile in profiles if profile]
    values = [value for value in values if isinstance(value, (int, float))]
    return max(values) if values else None


def _sum_lists(profiles, key):
    values = [profile.get(key) for profile in profiles if profile]
    values = [value for value in values if isinstance(value, (list, tuple))]
    width = max((len(value) for value in values), default=0)
    return [
        sum(int(value[index]) for value in values if index < len(value))
        for index in range(width)
    ]


def hardware_metadata(device=None):
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if device is not None and torch.cuda.is_available():
        device = torch.device(device)
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "device": str(device),
                "gpu_name": properties.name,
                "gpu_total_memory_bytes": properties.total_memory,
                "gpu_compute_capability": list(
                    torch.cuda.get_device_capability(device)
                ),
            }
        )
    try:
        hardware, runtime = HardwareDetector().detect(
            device if device is not None else 0
        )
        result["hardware_profile"] = hardware.as_dict()
        result["runtime_features"] = runtime.as_dict()
    except BaseException as error:
        # Reporting must not hide a completed inference result. Detection failures
        # stay explicit and retain their exact compatibility status.
        result["hardware_detection_error"] = "{}: {}".format(
            type(error).__name__, error
        )
    return result


def build_inference_report(
    *,
    plan,
    geometry,
    policy,
    checkpoint,
    validation,
    preflight,
    checkpoint_load_seconds,
    prompt_tokens,
    generated_tokens,
    generation_wall_seconds,
    time_to_first_token_ms,
    decode_token_latencies_ms,
    runtime_profiles,
    vocab_profiles,
    gpu_peak_memory_bytes,
    cpu_resident_bytes,
    pinned_bytes,
    kv_cache_bytes,
    device=None,
    kv_profiles=(),
    finish_profiles=(),
):
    runtime_profiles = [item for item in runtime_profiles if item]
    vocab_profiles = [item for item in vocab_profiles if item]
    kv_profiles = [item for item in kv_profiles if item]
    finish_profiles = [item for item in finish_profiles if item]
    kv_profile = kv_profiles[-1] if kv_profiles else {}
    decode_count = len(decode_token_latencies_ms)
    decode_seconds = sum(decode_token_latencies_ms) / 1000.0
    pipeline = {
        "source_wait_time_ms": _sum(
            runtime_profiles, "source_prepare_wait_ms"
        ),
        "ready_wait_time_ms": _sum(runtime_profiles, "ready_wait_ms"),
        "free_slot_wait_time_ms": _sum(
            runtime_profiles, "free_slot_wait_ms"
        ),
        "source_queue_max_depth": _max(
            runtime_profiles, "source_queue_max_depth"
        ),
        "ready_queue_max_depth": _max(
            runtime_profiles, "ready_queue_max_depth"
        ),
        "source_queue_capacity": _max(
            runtime_profiles, "source_queue_capacity"
        ),
        "ready_queue_capacity": _max(
            runtime_profiles, "ready_queue_capacity"
        ),
        "transfer_slot_reuse_counts": _sum_lists(
            runtime_profiles, "transfer_slot_reuse_counts"
        ),
        "h2d_bytes": _sum(runtime_profiles, "h2d_bytes"),
        "weight_h2d_bytes": _sum(runtime_profiles, "weight_h2d_bytes"),
        "copy_compute_timeline": {
            "copy_busy_ms": _sum(runtime_profiles, "copy_busy_ms"),
            "compute_busy_ms": _sum(runtime_profiles, "compute_busy_ms"),
            "overlap_ms": _sum(
                runtime_profiles, "copy_compute_overlap_ms"
            ),
            "copy_only_ms": _sum(runtime_profiles, "copy_only_ms"),
            "compute_only_ms": _sum(runtime_profiles, "compute_only_ms"),
            "idle_or_host_overhead_ms": _sum(
                runtime_profiles, "gpu_timeline_idle_or_host_overhead_ms"
            ),
            "per_forward": [
                {
                    "index": index,
                    "phase": profile.get("inference_phase"),
                    "summary": profile.get("copy_compute_timeline"),
                }
                for index, profile in enumerate(runtime_profiles)
                if profile.get("copy_compute_timeline") is not None
            ],
        },
        "transfer_timelines": [
            {
                "index": index,
                "phase": profile.get("inference_phase"),
                "units": profile.get("transfer_timeline", []),
            }
            for index, profile in enumerate(runtime_profiles)
            if profile.get("transfer_timeline") is not None
        ],
        "staging_timelines": [
            {
                "index": index,
                "phase": profile.get("inference_phase"),
                "units": profile.get("staging_timeline", []),
            }
            for index, profile in enumerate(runtime_profiles)
            if profile.get("staging_timeline") is not None
        ],
        "backends": sorted(
            {
                backend
                for profile in runtime_profiles
                for backend in profile.get("backends", ())
            }
        ),
        "fallback_backends": sorted(
            {
                backend
                for profile in runtime_profiles
                for backend in profile.get("fallback_backends", ())
            }
        ),
        "backend_providers": sorted(
            {
                provider
                for profile in runtime_profiles
                for provider in profile.get("backend_providers", ())
            }
        ),
        "quant_weight_h2d_bytes": _sum(
            runtime_profiles, "quant_weight_h2d_bytes"
        ),
        "scale_h2d_bytes": _sum(
            runtime_profiles, "scale_h2d_bytes"
        ),
        "backend_phases": [
            {
                "phase": profile.get("backend_phase"),
                "backend": profile.get("phase_backend"),
            }
            for profile in runtime_profiles
            if profile.get("backend_phase") is not None
        ],
        "kv": {
            "attention_backend": kv_profile.get(
                "paged_attention_provider",
                kv_profile.get("attention_backend"),
            ),
            "attention_accuracy": kv_profile.get("attention_accuracy"),
            "layout": kv_profile.get("layout"),
            "append_calls": kv_profile.get("append_calls", 0),
            "appended_tokens": kv_profile.get(
                "appended_tokens",
                kv_profile.get(
                    "committed_tokens", kv_profile.get("append_tokens", 0)
                ),
            ),
            "layer_token_writes": kv_profile.get("append_tokens", 0),
            "attention_calls": kv_profile.get("attention_calls", 0),
            "materialize_calls": kv_profile.get("materialize_calls", 0),
            "materialized_bytes": kv_profile.get("materialized_bytes", 0),
            "fork_calls": kv_profile.get(
                "fork_calls", kv_profile.get("fork_count", 0)
            ),
            "cow_page_copies": kv_profile.get(
                "cow_page_copies", kv_profile.get("cow_count", 0)
            ),
            "page_allocations": kv_profile.get("page_allocations", 0),
            "page_releases": kv_profile.get("page_releases", 0),
            "cuda_event_count": kv_profile.get("cuda_event_count", 0),
            "kv_policy_resolved": kv_profile.get(
                "kv_policy_resolved", kv_profile.get("policy")
            ),
            "kv_store": kv_profile.get("kv_store"),
            "kv_dtype": kv_profile.get("kv_dtype"),
            "kv_selection": kv_profile.get("kv_selection"),
            "kv_selection_scorer": kv_profile.get("kv_selection_scorer"),
            "rgkv_scorer_stats": kv_profile.get("rgkv_scorer_stats"),
            "rgkv_index_stats": kv_profile.get("rgkv_index_stats"),
            "rgkv_pages_total": kv_profile.get("rgkv_pages_total", 0),
            "rgkv_pages_selected": kv_profile.get("rgkv_pages_selected", 0),
            "rgkv_selection_ratio": kv_profile.get(
                "rgkv_selection_ratio", 1.0
            ),
            "rgkv_index_bytes": kv_profile.get("rgkv_index_bytes", 0),
            "rgkv_update_ms": kv_profile.get("rgkv_update_ms", 0.0),
            "rgkv_selection_enqueue_ms": kv_profile.get(
                "rgkv_selection_enqueue_ms", 0.0
            ),
            "rgkv_score_ms": kv_profile.get("rgkv_score_ms", 0.0),
            "rgkv_topk_ms": kv_profile.get("rgkv_topk_ms", 0.0),
            "rgkv_timing_sampled": kv_profile.get(
                "rgkv_timing_sampled", False
            ),
            "rgkv_cpu_sync_count": kv_profile.get(
                "rgkv_cpu_sync_count", 0
            ),
            "rgkv_host_authority_page_checks": kv_profile.get(
                "rgkv_host_authority_page_checks", 0
            ),
            "rgkv_stale_index_count": kv_profile.get(
                "rgkv_stale_index_count", 0
            ),
            "kv_reuse": kv_profile.get("kv_reuse"),
            "kv_page_size": kv_profile.get("kv_page_size"),
            "kv_pool_total_pages": kv_profile.get("kv_pool_total_pages", 0),
            "kv_pool_peak_pages": kv_profile.get("pool_peak_pages", 0),
            "kv_shared_pages": kv_profile.get("shared_pages", 0),
            "kv_cow_count": kv_profile.get(
                "cow_count", kv_profile.get("cow_page_copies", 0)
            ),
            "kv_fork_count": kv_profile.get(
                "fork_count", kv_profile.get("fork_calls", 0)
            ),
            "kv_workspace_peak_bytes": kv_profile.get(
                "workspace_peak_bytes", 0
            ),
            # A false sampling flag makes the additive zero defaults explicit:
            # they are initialized schema fields, not a claim that tier I/O
            # was measured during a GPU-only run.
            "tier_metrics_sampled": kv_profile.get(
                "tier_metrics_sampled", False
            ),
            "gpu_kv_capacity_bytes": kv_profile.get(
                "gpu_kv_capacity_bytes", 0
            ),
            "gpu_kv_used_bytes": kv_profile.get("gpu_kv_used_bytes", 0),
            "gpu_kv_free_bytes": kv_profile.get(
                "gpu_kv_free_bytes", kv_profile.get("free_bytes", 0)
            ),
            "gpu_kv_peak_used_bytes": kv_profile.get(
                "gpu_kv_peak_used_bytes", 0
            ),
            "gpu_kv_peak_used_pages": kv_profile.get(
                "gpu_kv_peak_used_pages", 0
            ),
            "gpu_kv_capacity_pages": kv_profile.get(
                "gpu_kv_capacity_pages",
                kv_profile.get("gpu_capacity_pages", 0),
            ),
            "gpu_kv_used_pages": kv_profile.get(
                "gpu_kv_used_pages", kv_profile.get("used_pages", 0)
            ),
            "gpu_kv_free_pages": kv_profile.get(
                "gpu_kv_free_pages", kv_profile.get("free_pages", 0)
            ),
            "gpu_kv_high_watermark_bytes": kv_profile.get(
                "gpu_kv_high_watermark_bytes",
                kv_profile.get("high_watermark_bytes", 0),
            ),
            "gpu_kv_low_watermark_bytes": kv_profile.get(
                "gpu_kv_low_watermark_bytes",
                kv_profile.get("low_watermark_bytes", 0),
            ),
            "gpu_kv_high_watermark_pages": kv_profile.get(
                "gpu_kv_high_watermark_pages",
                kv_profile.get("high_watermark_pages", 0),
            ),
            "gpu_kv_low_watermark_pages": kv_profile.get(
                "gpu_kv_low_watermark_pages",
                kv_profile.get("low_watermark_pages", 0),
            ),
            "gpu_kv_above_high_watermark": kv_profile.get(
                "gpu_kv_above_high_watermark", False
            ),
            "gpu_kv_below_low_watermark": kv_profile.get(
                "gpu_kv_below_low_watermark", False
            ),
            "cpu_kv_capacity_bytes": kv_profile.get(
                "cpu_kv_capacity_bytes", 0
            ),
            "cpu_kv_used_bytes": kv_profile.get("cpu_kv_used_bytes", 0),
            "cpu_kv_reserved_bytes": kv_profile.get(
                "cpu_kv_reserved_bytes", 0
            ),
            "cpu_kv_free_bytes": kv_profile.get("cpu_kv_free_bytes", 0),
            "cpu_kv_peak_used_bytes": kv_profile.get(
                "cpu_kv_peak_used_bytes", 0
            ),
            "cpu_kv_capacity_pages": kv_profile.get(
                "cpu_kv_capacity_pages", 0
            ),
            "cpu_kv_used_pages": kv_profile.get("cpu_kv_used_pages", 0),
            "cpu_kv_reserved_pages": kv_profile.get(
                "cpu_kv_reserved_pages", 0
            ),
            "cpu_kv_free_pages": kv_profile.get("cpu_kv_free_pages", 0),
            "cpu_kv_high_watermark_bytes": kv_profile.get(
                "cpu_kv_high_watermark_bytes", 0
            ),
            "cpu_kv_low_watermark_bytes": kv_profile.get(
                "cpu_kv_low_watermark_bytes", 0
            ),
            "cpu_kv_above_high_watermark": kv_profile.get(
                "cpu_kv_above_high_watermark", False
            ),
            "cpu_kv_below_low_watermark": kv_profile.get(
                "cpu_kv_below_low_watermark", False
            ),
            "gpu_hits": kv_profile.get("gpu_hits", 0),
            "cpu_hits": kv_profile.get("cpu_hits", 0),
            "prefetch_count": kv_profile.get("prefetch_count", 0),
            "prefetch_pages": kv_profile.get(
                "prefetch_pages", kv_profile.get("prefetch_count", 0)
            ),
            "prefetch_bytes": kv_profile.get("prefetch_bytes", 0),
            "prefetch_wait_ms": kv_profile.get("prefetch_wait_ms", 0.0),
            "prefetch_timeouts": kv_profile.get("prefetch_timeouts", 0),
            "eviction_count": kv_profile.get("eviction_count", 0),
            "eviction_pages": kv_profile.get(
                "eviction_pages", kv_profile.get("eviction_count", 0)
            ),
            "eviction_bytes": kv_profile.get("eviction_bytes", 0),
            "h2d_kv_bytes": kv_profile.get("h2d_kv_bytes", 0),
            "d2h_kv_bytes": kv_profile.get("d2h_kv_bytes", 0),
            "migration_failures": kv_profile.get(
                "migration_failures", 0
            ),
            "migration_cancellations": kv_profile.get(
                "migration_cancellations", 0
            ),
            "authority_changes": kv_profile.get("authority_changes", 0),
            "thrashing_count": kv_profile.get("thrashing_count", 0),
            "thrash_window_operations": kv_profile.get(
                "thrash_window_operations", 0
            ),
            "tier_version_mismatches": kv_profile.get(
                "tier_version_mismatches", 0
            ),
            "paged_attention_provider": kv_profile.get(
                "paged_attention_provider"
            ),
            "paged_kv_kernel_backend": kv_profile.get(
                "paged_kv_kernel_backend"
            ),
            "paged_provider_bundle": kv_profile.get(
                "paged_provider_bundle"
            ),
            "paged_prefill_provider": kv_profile.get(
                "paged_prefill_provider"
            ),
            "provider_fallback_reason": kv_profile.get(
                "provider_fallback_reason"
            ),
            "provider_fallback_count": kv_profile.get(
                "provider_fallback_count", 0
            ),
            "provider_reference_fallback_count": kv_profile.get(
                "provider_reference_fallback_count", 0
            ),
            "provider_routing_summary": kv_profile.get(
                "provider_routing_summary",
                {
                    "total_calls": 0,
                    "fallback_calls": 0,
                    "reference_fallback_calls": 0,
                    "decisions": [],
                },
            ),
            "decode_attention_ms": kv_profile.get(
                "decode_attention_ms", 0.0
            ),
            "prefill_attention_ms": kv_profile.get(
                "prefill_attention_ms", 0.0
            ),
        },
    }
    timings = {
        "load_time_seconds": float(checkpoint_load_seconds),
        "generation_wall_seconds": float(generation_wall_seconds),
        "time_to_first_token_ms": float(time_to_first_token_ms),
        "decode_token_latencies_ms": list(decode_token_latencies_ms),
        "pageable_to_pinned_time_ms": _sum(
            runtime_profiles, "staging_event_sum_ms"
        ),
        "pageable_to_pinned_copy_time_ms": _sum(
            runtime_profiles, "pageable_to_pinned_copy_sum_ms"
        ),
        "staging_buffer_wait_time_ms": _sum(
            runtime_profiles, "staging_buffer_wait_sum_ms"
        ),
        "h2d_time_ms": _sum(runtime_profiles, "h2d_event_sum_ms"),
        "h2d_compute_overlap_ms": _sum(
            runtime_profiles, "copy_compute_overlap_ms"
        ),
        "unoverlapped_h2d_ms": _sum(runtime_profiles, "copy_only_ms"),
        "compute_only_ms": _sum(runtime_profiles, "compute_only_ms"),
        "gpu_idle_or_host_overhead_ms": _sum(
            runtime_profiles, "gpu_timeline_idle_or_host_overhead_ms"
        ),
        "compute_time_ms": _sum(
            runtime_profiles, "compute_event_sum_ms"
        ),
        "attention_time_ms": _sum(
            runtime_profiles, "attention_event_sum_ms"
        ),
        "kv_append_time_ms": _sum(
            runtime_profiles, "kv_append_event_sum_ms"
        ),
        "kv_paged_attention_time_ms": _sum(
            runtime_profiles, "kv_attention_event_sum_ms"
        ),
        "mlp_time_ms": _sum(runtime_profiles, "mlp_event_sum_ms"),
        "dequant_time_ms": _sum(
            runtime_profiles, "dequant_event_sum_ms"
        ),
        "gemm_time_ms": _sum(
            runtime_profiles, "gemm_event_sum_ms"
        ),
        "embedding_time_ms": _sum(vocab_profiles, "embedding_wall_ms"),
        "embedding_h2d_time_ms": _sum(
            vocab_profiles, "embedding_h2d_ms"
        ),
        "lm_head_time_ms": (
            _sum(finish_profiles, "finish_cuda_ms")
            if _sum(finish_profiles, "finish_cuda_ms") is not None
            else _sum(vocab_profiles, "wall_ms")
        ),
        "finish_host_enqueue_time_ms": _sum(
            finish_profiles, "finish_host_enqueue_ms"
        ),
        "kv_attention_time_ms": (
            _sum(runtime_profiles, "kv_attention_event_sum_ms")
            if _sum(runtime_profiles, "kv_attention_event_sum_ms") is not None
            else (
                (_sum(kv_profiles, "attention_wall_ms") or 0.0)
                + (_sum(kv_profiles, "decode_attention_ms") or 0.0)
                + (_sum(kv_profiles, "prefill_attention_ms") or 0.0)
            )
        ),
        "kv_attention_host_dispatch_time_ms": (
            (_sum(kv_profiles, "attention_wall_ms") or 0.0)
            + (_sum(kv_profiles, "decode_attention_ms") or 0.0)
            + (_sum(kv_profiles, "prefill_attention_ms") or 0.0)
        ),
    }
    return RunReport(
        model={
            "model_id": plan.model_id,
            "checkpoint": str(checkpoint),
            "model_type": geometry.model_type,
            "geometry": geometry.as_dict(),
        },
        runtime_config={
            "weight_format": policy.weight_format.value,
            "linear_backend_requested": getattr(
                policy, "linear_backend", None
            ),
            "prefill_backend": next(
                (
                    profile.get("prefill_backend")
                    for profile in runtime_profiles
                    if profile.get("prefill_backend") is not None
                ),
                None,
            ),
            "decode_backend": next(
                (
                    profile.get("decode_backend")
                    for profile in runtime_profiles
                    if profile.get("decode_backend") is not None
                ),
                None,
            ),
            "decode_fallback_explicit": any(
                bool(profile.get("decode_fallback_explicit"))
                for profile in runtime_profiles
            ),
            "embedding_dtype": getattr(
                policy, "embedding_dtype", "bfloat16"
            ),
            "lm_head_dtype": getattr(policy, "lm_head_dtype", "bfloat16"),
            "norm_dtype": getattr(policy, "norm_dtype", "bfloat16"),
            "quantization": (
                {
                    name: getattr(policy.quantization, name)
                    for name in policy.quantization.__dataclass_fields__
                }
                if getattr(policy, "quantization", None) is not None
                else None
            ),
            "granularity": policy.granularity.value,
            "cpu_weight_mode": policy.cpu_weight_mode,
            "embedding_mode": policy.embedding_mode.value,
            "lm_head_mode": policy.lm_head_mode.value,
            "lm_head_backend": next(
                (
                    profile.get("lm_head_backend")
                    for profile in vocab_profiles
                    if profile.get("lm_head_backend") is not None
                ),
                (
                    "deterministic_cuda_fp32_accum_native_output"
                    if policy.lm_head_mode.value == "resident"
                    else None
                ),
            ),
            "slot_count": policy.slot_count,
            "prefetch_depth": policy.prefetch_depth,
            "vocab_chunk_bytes": policy.vocab_chunk_bytes,
            "kv_policy": kv_profile.get(
                "kv_policy_resolved", kv_profile.get("policy")
            ),
        },
        timings=timings,
        throughput={
            "prompt_tokens": int(prompt_tokens),
            "generated_tokens": int(generated_tokens),
            "tokens_per_second": (
                generated_tokens / generation_wall_seconds
                if generation_wall_seconds > 0
                else 0.0
            ),
            "decode_tokens_per_second": (
                decode_count / decode_seconds if decode_seconds > 0 else None
            ),
        },
        memory={
            "gpu_peak_memory_bytes": int(gpu_peak_memory_bytes),
            "cpu_resident_bytes": int(cpu_resident_bytes),
            "pinned_bytes": int(pinned_bytes),
            "kv_cache_bytes": int(kv_cache_bytes),
            "gpu_weight_budget_bytes": int(
                getattr(preflight.estimate, "gpu_weight_budget_bytes", 0)
            ),
            "kv_gpu_pool_bytes": int(
                getattr(preflight.estimate, "kv_gpu_pool_bytes", kv_cache_bytes)
            ),
            "kv_cpu_pool_bytes": int(
                getattr(preflight.estimate, "kv_cpu_pool_bytes", 0)
            ),
            "kv_total_context_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_total_context_bytes",
                    preflight.estimate.kv_cache_bytes,
                )
            ),
            "kv_gpu_cache_capacity_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_gpu_cache_capacity_bytes",
                    getattr(preflight.estimate, "kv_gpu_pool_bytes", 0),
                )
            ),
            "kv_cpu_pinned_backing_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_cpu_pinned_backing_bytes",
                    getattr(preflight.estimate, "kv_cpu_pool_bytes", 0),
                )
            ),
            "kv_gpu_migration_slots_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_gpu_migration_slots_bytes",
                    0,
                )
            ),
            "kv_cpu_migration_slots_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_cpu_migration_slots_bytes",
                    0,
                )
            ),
            "kv_layer_page_bytes": int(
                getattr(preflight.estimate, "kv_layer_page_bytes", 0)
            ),
            "kv_gpu_migration_slot_count": int(
                getattr(
                    preflight.estimate,
                    "kv_gpu_migration_slot_count",
                    0,
                )
            ),
            "kv_cpu_migration_slot_count": int(
                getattr(
                    preflight.estimate,
                    "kv_cpu_migration_slot_count",
                    0,
                )
            ),
            "kv_gpu_high_watermark_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_gpu_high_watermark_bytes",
                    0,
                )
            ),
            "kv_gpu_low_watermark_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_gpu_low_watermark_bytes",
                    0,
                )
            ),
            "kv_cpu_high_watermark_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_cpu_high_watermark_bytes",
                    0,
                )
            ),
            "kv_cpu_low_watermark_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_cpu_low_watermark_bytes",
                    0,
                )
            ),
            "kv_nvme_budget_bytes": int(
                getattr(preflight.estimate, "kv_nvme_budget_bytes", 0)
            ),
            "kv_index_bytes": int(
                getattr(preflight.estimate, "kv_index_bytes", 0)
            ),
            "rgkv_index_bytes": int(
                getattr(preflight.estimate, "rgkv_index_bytes", 0)
            ),
            "rgkv_cpu_reference_index_bytes": int(
                getattr(
                    preflight.estimate,
                    "rgkv_cpu_reference_index_bytes",
                    0,
                )
            ),
            "rgkv_gpu_index_bytes": int(
                getattr(preflight.estimate, "rgkv_gpu_index_bytes", 0)
            ),
            "rgkv_build_workspace_bytes": int(
                getattr(
                    preflight.estimate,
                    "rgkv_build_workspace_bytes",
                    0,
                )
            ),
            "rgkv_scoring_workspace_bytes": int(
                getattr(
                    preflight.estimate,
                    "rgkv_scoring_workspace_bytes",
                    0,
                )
            ),
            "rgkv_topk_workspace_bytes": int(
                getattr(
                    preflight.estimate,
                    "rgkv_topk_workspace_bytes",
                    0,
                )
            ),
            "kv_attention_workspace_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_attention_workspace_bytes",
                    0,
                )
            ),
            "kv_prefill_workspace_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_prefill_workspace_bytes",
                    0,
                )
            ),
            "kv_decode_attention_workspace_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_decode_attention_workspace_bytes",
                    0,
                )
            ),
            "kv_admission_required_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_admission_required_bytes",
                    preflight.estimate.kv_cache_bytes,
                )
            ),
            "kv_admission_capacity_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_admission_capacity_bytes",
                    preflight.estimate.kv_cache_bytes,
                )
            ),
            "kv_admission_headroom_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_admission_headroom_bytes",
                    0,
                )
            ),
            "kv_block_table_bytes": int(
                getattr(preflight.estimate, "kv_block_table_bytes", 0)
            ),
            "kv_reserved_page_bytes": int(
                getattr(preflight.estimate, "kv_reserved_page_bytes", 0)
            ),
            "checkpoint_read_bytes": int(plan.host_arena_bytes),
            "dequant_workspace_bytes": int(
                preflight.estimate.dequant_workspace_bytes
            ),
            "gpu_quant_parameter_bytes": int(
                getattr(
                    preflight.estimate,
                    "gpu_quant_parameter_bytes",
                    0,
                )
            ),
        },
        pipeline=pipeline,
        checkpoint_validation=validation.as_dict(),
        memory_preflight=preflight.as_dict(),
        hardware=hardware_metadata(device),
    )
