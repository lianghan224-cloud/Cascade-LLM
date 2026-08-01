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
        # stay explicit and are treated as unqualified by compatibility tooling.
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
):
    runtime_profiles = [item for item in runtime_profiles if item]
    vocab_profiles = [item for item in vocab_profiles if item]
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
            "slot_count": policy.slot_count,
            "prefetch_depth": policy.prefetch_depth,
            "vocab_chunk_bytes": policy.vocab_chunk_bytes,
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
