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
):
    runtime_profiles = [item for item in runtime_profiles if item]
    vocab_profiles = [item for item in vocab_profiles if item]
    kv_profiles = [item for item in kv_profiles if item]
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
            "paged_attention_provider": kv_profile.get(
                "paged_attention_provider"
            ),
            "paged_kv_kernel_backend": kv_profile.get(
                "paged_kv_kernel_backend"
            ),
            "paged_provider_bundle": kv_profile.get(
                "paged_provider_bundle"
            ),
            "provider_fallback_reason": kv_profile.get(
                "provider_fallback_reason"
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
        "h2d_time_ms": _sum(runtime_profiles, "h2d_event_sum_ms"),
        "compute_time_ms": _sum(
            runtime_profiles, "compute_event_sum_ms"
        ),
        "attention_time_ms": _sum(
            runtime_profiles, "attention_event_sum_ms"
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
        "lm_head_time_ms": _sum(vocab_profiles, "wall_ms"),
        "kv_attention_time_ms": (
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
            "kv_gpu_pool_bytes": int(
                getattr(preflight.estimate, "kv_gpu_pool_bytes", kv_cache_bytes)
            ),
            "kv_cpu_pool_bytes": int(
                getattr(preflight.estimate, "kv_cpu_pool_bytes", 0)
            ),
            "kv_nvme_budget_bytes": int(
                getattr(preflight.estimate, "kv_nvme_budget_bytes", 0)
            ),
            "kv_index_bytes": int(
                getattr(preflight.estimate, "kv_index_bytes", 0)
            ),
            "kv_attention_workspace_bytes": int(
                getattr(
                    preflight.estimate,
                    "kv_attention_workspace_bytes",
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
