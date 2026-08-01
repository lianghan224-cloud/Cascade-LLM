"""Deterministic CPU, pinned-memory, and GPU preflight estimates."""

from dataclasses import dataclass
import math
import os
import resource
from typing import Optional, Tuple

import torch

from .adapter import ExecutionPolicy, ModelGeometry, PlacementMode


GIB = 1024 ** 3


class MemoryPreflightError(RuntimeError):
    def __init__(self, result):
        self.result = result
        super().__init__("; ".join(result.errors))


@dataclass(frozen=True)
class SystemCapacity:
    cpu_available_bytes: Optional[int]
    memlock_limit_bytes: Optional[int]
    gpu_free_bytes: Optional[int]
    gpu_total_bytes: Optional[int]


@dataclass(frozen=True)
class MemoryEstimate:
    cpu_checkpoint_arena_bytes: int
    pinned_staging_bytes: int
    estimated_cpu_peak_bytes: int
    gpu_transfer_slots_bytes: int
    resident_parameters_bytes: int
    dequant_workspace_bytes: int
    kv_cache_bytes: int
    embedding_buffer_bytes: int
    lm_head_buffer_bytes: int
    full_logits_bytes: int
    temporary_workspace_bytes: int
    cuda_safety_margin_bytes: int
    estimated_gpu_peak_bytes: int
    cpu_bf16_bytes: int = 0
    cpu_fp16_bytes: int = 0
    cpu_int8_bytes: int = 0
    cpu_int4_packed_bytes: int = 0
    cpu_scale_bytes: int = 0
    cpu_zero_point_bytes: int = 0
    gpu_weight_slot_bytes: int = 0
    gpu_quant_parameter_bytes: int = 0
    gpu_dequant_workspace_bytes: int = 0
    gpu_resident_norm_bytes: int = 0
    gpu_resident_embedding_bytes: int = 0
    gpu_resident_lm_head_bytes: int = 0
    gpu_resident_transformer_bytes: int = 0
    embedding_staging_bytes: int = 0
    lm_head_chunk_bytes: int = 0
    topk_bytes: int = 0

    def as_dict(self):
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class PreflightResult:
    estimate: MemoryEstimate
    capacity: SystemCapacity
    errors: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    suggestions: Tuple[str, ...] = ()

    @property
    def ok(self):
        return not self.errors

    def raise_for_error(self):
        if not self.ok:
            raise MemoryPreflightError(self)
        return self

    def as_dict(self):
        return {
            "ok": self.ok,
            "estimate": self.estimate.as_dict(),
            "capacity": {
                name: getattr(self.capacity, name)
                for name in self.capacity.__dataclass_fields__
            },
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "suggestions": list(self.suggestions),
        }

    def format_text(self):
        estimate = self.estimate
        capacity = self.capacity

        def gib(value):
            return "unknown" if value is None else "{:.2f} GiB".format(value / GIB)

        lines = [
            "CPU checkpoint arena:    {}".format(gib(estimate.cpu_checkpoint_arena_bytes)),
            "  CPU BF16:              {}".format(gib(estimate.cpu_bf16_bytes)),
            "  CPU FP16:              {}".format(gib(estimate.cpu_fp16_bytes)),
            "  CPU INT8:              {}".format(gib(estimate.cpu_int8_bytes)),
            "  CPU INT4 packed:       {}".format(gib(estimate.cpu_int4_packed_bytes)),
            "  CPU scale:             {}".format(gib(estimate.cpu_scale_bytes)),
            "  CPU zero-point:        {}".format(gib(estimate.cpu_zero_point_bytes)),
            "Pinned staging:          {}".format(gib(estimate.pinned_staging_bytes)),
            "Estimated CPU peak:      {}".format(gib(estimate.estimated_cpu_peak_bytes)),
            "GPU transfer slots:      {}".format(gib(estimate.gpu_transfer_slots_bytes)),
            "  GPU quant parameters:  {}".format(gib(estimate.gpu_quant_parameter_bytes)),
            "Resident parameters:     {}".format(gib(estimate.resident_parameters_bytes)),
            "  Resident norm:         {}".format(gib(estimate.gpu_resident_norm_bytes)),
            "  Resident embedding:    {}".format(gib(estimate.gpu_resident_embedding_bytes)),
            "  Resident LM Head:      {}".format(gib(estimate.gpu_resident_lm_head_bytes)),
            "  Resident Transformer:  {}".format(gib(estimate.gpu_resident_transformer_bytes)),
            "Dequant workspace:       {}".format(gib(estimate.dequant_workspace_bytes)),
            "KV cache:                {}".format(gib(estimate.kv_cache_bytes)),
            "Embedding buffer:        {}".format(gib(estimate.embedding_buffer_bytes)),
            "LM Head buffer:          {}".format(gib(estimate.lm_head_buffer_bytes)),
            "Top-k workspace:         {}".format(gib(estimate.topk_bytes)),
            "Full logits:             {}".format(gib(estimate.full_logits_bytes)),
            "Temporary workspace:     {}".format(gib(estimate.temporary_workspace_bytes)),
            "CUDA safety margin:      {}".format(gib(estimate.cuda_safety_margin_bytes)),
            "Estimated GPU peak:      {}".format(gib(estimate.estimated_gpu_peak_bytes)),
            "Available CPU memory:    {}".format(gib(capacity.cpu_available_bytes)),
            "RLIMIT_MEMLOCK:          {}".format(gib(capacity.memlock_limit_bytes)),
            "Available GPU memory:    {}".format(gib(capacity.gpu_free_bytes)),
        ]
        lines.extend("ERROR: " + item for item in self.errors)
        lines.extend("WARNING: " + item for item in self.warnings)
        lines.extend("SUGGESTION: " + item for item in self.suggestions)
        return "\n".join(lines)


@dataclass(frozen=True)
class MetadataPlanResult:
    plan: object
    estimate: MemoryEstimate
    max_host_offset: int
    estimated_shard_count: int
    offsets_are_int64_safe: bool

    def as_dict(self):
        return {
            "model_id": self.plan.model_id,
            "geometry": self.plan.geometry.as_dict(),
            "plan": self.plan.as_dict(),
            "estimate": self.estimate.as_dict(),
            "max_host_offset": self.max_host_offset,
            "estimated_shard_count": self.estimated_shard_count,
            "offsets_are_int64_safe": self.offsets_are_int64_safe,
        }


def _cpu_available_bytes():
    try:
        with open("/proc/meminfo", "r", encoding="ascii") as source:
            for line in source:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (OSError, ValueError):
        return None


def _memlock_limit_bytes():
    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    except (OSError, ValueError):
        return None
    if soft == resource.RLIM_INFINITY:
        return None
    return int(soft)


def collect_system_capacity(device=None):
    gpu_free = None
    gpu_total = None
    if device is not None and torch.cuda.is_available():
        with torch.cuda.device(torch.device(device)):
            gpu_free, gpu_total = torch.cuda.mem_get_info()
    return SystemCapacity(
        cpu_available_bytes=_cpu_available_bytes(),
        memlock_limit_bytes=_memlock_limit_bytes(),
        gpu_free_bytes=gpu_free,
        gpu_total_bytes=gpu_total,
    )


class MemoryPlanner:
    def __init__(
        self,
        plan,
        geometry,
        policy=None,
        max_context=2048,
        max_prefill_tokens=None,
        batch_size=1,
        kv_block_size=16,
        embedding_staging_rows=256,
        return_full_logits=False,
        logits_tokens=None,
        top_k=10,
        temporary_workspace_bytes=0,
        cuda_safety_margin_bytes=512 * 1024 ** 2,
        transformer_placement=None,
    ):
        if not isinstance(geometry, ModelGeometry):
            raise TypeError("geometry must be a ModelGeometry")
        self.plan = plan
        self.geometry = geometry
        self.policy = policy or ExecutionPolicy()
        self.max_context = int(max_context)
        self.max_prefill_tokens = int(
            max_prefill_tokens
            if max_prefill_tokens is not None
            else max_context
        )
        self.batch_size = int(batch_size)
        self.kv_block_size = int(kv_block_size)
        self.embedding_staging_rows = int(embedding_staging_rows)
        self.return_full_logits = bool(return_full_logits)
        self.logits_tokens = int(
            logits_tokens if logits_tokens is not None else self.max_prefill_tokens
        )
        self.top_k = int(top_k)
        self.temporary_workspace_bytes = int(temporary_workspace_bytes)
        self.cuda_safety_margin_bytes = int(cuda_safety_margin_bytes)
        self.transformer_placement = transformer_placement
        for name in (
            "max_context",
            "max_prefill_tokens",
            "batch_size",
            "kv_block_size",
            "embedding_staging_rows",
            "top_k",
        ):
            if getattr(self, name) <= 0:
                raise ValueError("{} must be positive".format(name))
        if self.max_context > geometry.max_position_embeddings:
            raise ValueError(
                "max_context {} exceeds model limit {}".format(
                    self.max_context, geometry.max_position_embeddings
                )
            )

    def estimate(self):
        policy = self.policy
        plan = self.plan
        slot_count = int(policy.slot_count)
        is_generic = hasattr(plan, "regions")
        is_int8 = hasattr(plan, "transfer_slot_bytes")
        if is_generic:
            transfer_slots = slot_count * int(plan.slot_bytes)
            dequant_workspace = slot_count * int(plan.workspace_bytes)
            staging_slot_bytes = int(plan.slot_bytes)
        elif is_int8:
            transfer_slots = slot_count * int(plan.transfer_slot_bytes)
            dequant_workspace = slot_count * int(plan.dequant_workspace_bytes)
            staging_slot_bytes = int(plan.transfer_slot_bytes)
        else:
            transfer_slots = slot_count * int(plan.slot_bytes)
            dequant_workspace = 0
            staging_slot_bytes = int(plan.slot_bytes)

        vocab = getattr(plan, "vocab", None)
        pinned_vocab = 0
        stream_embedding = bool(
            vocab is not None
            and (
                getattr(vocab, "stream_embedding", False)
                or getattr(vocab, "embedding_mode", None) == "streamed"
            )
        )
        stream_lm_head = bool(
            vocab is not None
            and (
                getattr(vocab, "stream_lm_head", False)
                or getattr(vocab, "lm_head_mode", None) == "streamed"
            )
        )
        embedding_dtype_bytes = 2
        lm_head_dtype_bytes = 2
        if is_generic:
            from .specs import DTYPE_BYTES

            embedding_dtype_bytes = DTYPE_BYTES[
                plan.weights[vocab.embedding_name].storage_dtype
            ]
            lm_spec = plan.weights[vocab.lm_head_name]
            if lm_spec.alias_of is not None:
                lm_spec = plan.weights[lm_spec.alias_of]
            lm_head_dtype_bytes = DTYPE_BYTES[lm_spec.storage_dtype]
        embedding_staging = 0
        if stream_embedding:
            embedding_staging = (
                self.embedding_staging_rows
                * self.geometry.hidden_size
                * embedding_dtype_bytes
            )
            pinned_vocab += (
                embedding_staging
            )
        if (
            stream_lm_head
            and policy.cpu_weight_mode == "pinned_staging"
        ):
            pinned_vocab += slot_count * int(vocab.chunk_bytes)
        if policy.cpu_weight_mode == "full_pinned":
            pinned = int(plan.host_arena_bytes) + pinned_vocab
        else:
            pinned = slot_count * staging_slot_bytes + pinned_vocab

        rounded_context = int(
            math.ceil(self.max_context / float(self.kv_block_size))
        ) * self.kv_block_size
        kv_cache = (
            2
            * self.geometry.num_hidden_layers
            * self.batch_size
            * self.geometry.num_key_value_heads
            * rounded_context
            * self.geometry.head_dim
            * 2
        )
        embedding_buffer = (
            self.batch_size
            * self.max_prefill_tokens
            * self.geometry.hidden_size
            * embedding_dtype_bytes
        )
        if policy.lm_head_mode == PlacementMode.STREAMED:
            chunk_rows = vocab.chunk_rows if vocab is not None else 0
            lm_head_buffer = self.batch_size * chunk_rows * 4
        else:
            lm_head_buffer = self.batch_size * self.geometry.vocab_size * 4
        topk_bytes = self.batch_size * self.top_k * (4 + 8)
        full_logits = 0
        if self.return_full_logits:
            full_logits = (
                self.batch_size
                * self.logits_tokens
                * self.geometry.vocab_size
                * 4
            )
            lm_head_buffer = 0
        resident_parameters = int(plan.resident_bytes)
        resident_transformer = 0
        if self.transformer_placement is not None:
            resident_parameters = int(
                self.transformer_placement.resident_arena_bytes
            )
            resident_transformer = int(
                self.transformer_placement.resident_weight_bytes
            )
        gpu_peak = sum(
            (
                transfer_slots,
                resident_parameters,
                dequant_workspace,
                kv_cache,
                embedding_buffer,
                lm_head_buffer,
                topk_bytes,
                full_logits,
                self.temporary_workspace_bytes,
                self.cuda_safety_margin_bytes,
            )
        )
        cpu_peak = (
            int(plan.host_arena_bytes) + pinned
            if policy.cpu_weight_mode == "pinned_staging"
            else max(int(plan.host_arena_bytes), pinned)
        )
        cpu_region_bytes = {
            "bfloat16": 0,
            "float16": 0,
            "int8": 0,
            "int4_packed": 0,
            "scale": 0,
            "zero_point": 0,
        }
        gpu_quant = 0
        resident_norm = 0
        resident_embedding = 0
        resident_lm_head = 0
        if is_generic:
            for name, region in plan.regions.items():
                if name.startswith("scale_"):
                    cpu_region_bytes["scale"] += region.bytes
                elif name.startswith("zero_point_"):
                    cpu_region_bytes["zero_point"] += region.bytes
                elif name in cpu_region_bytes:
                    cpu_region_bytes[name] += region.bytes
            max_quant_unit = max(
                (
                    sum(
                        tensor.storage_bytes
                        for tensor in unit.tensors
                        if plan.weights[tensor.weight_name].role
                        in {"scale", "zero_point"}
                    )
                    for unit in plan.units
                ),
                default=0,
            )
            gpu_quant = slot_count * max_quant_unit
            for placement in plan.resident:
                role = plan.weights[placement.weight_name].role
                if role == "norm":
                    resident_norm += placement.storage_bytes
                elif role == "embedding":
                    resident_embedding += placement.storage_bytes
                elif role == "lm_head":
                    resident_lm_head += placement.storage_bytes
        return MemoryEstimate(
            cpu_checkpoint_arena_bytes=int(plan.host_arena_bytes),
            pinned_staging_bytes=pinned,
            estimated_cpu_peak_bytes=cpu_peak,
            gpu_transfer_slots_bytes=transfer_slots,
            resident_parameters_bytes=resident_parameters,
            dequant_workspace_bytes=dequant_workspace,
            kv_cache_bytes=kv_cache,
            embedding_buffer_bytes=embedding_buffer,
            lm_head_buffer_bytes=lm_head_buffer,
            full_logits_bytes=full_logits,
            temporary_workspace_bytes=self.temporary_workspace_bytes,
            cuda_safety_margin_bytes=self.cuda_safety_margin_bytes,
            estimated_gpu_peak_bytes=gpu_peak,
            cpu_bf16_bytes=cpu_region_bytes["bfloat16"],
            cpu_fp16_bytes=cpu_region_bytes["float16"],
            cpu_int8_bytes=cpu_region_bytes["int8"],
            cpu_int4_packed_bytes=cpu_region_bytes["int4_packed"],
            cpu_scale_bytes=cpu_region_bytes["scale"],
            cpu_zero_point_bytes=cpu_region_bytes["zero_point"],
            gpu_weight_slot_bytes=transfer_slots,
            gpu_quant_parameter_bytes=gpu_quant,
            gpu_dequant_workspace_bytes=dequant_workspace,
            gpu_resident_norm_bytes=resident_norm,
            gpu_resident_embedding_bytes=resident_embedding,
            gpu_resident_lm_head_bytes=resident_lm_head,
            gpu_resident_transformer_bytes=resident_transformer,
            embedding_staging_bytes=embedding_staging,
            lm_head_chunk_bytes=(
                int(vocab.chunk_bytes) if stream_lm_head else 0
            ),
            topk_bytes=topk_bytes,
        )

    @classmethod
    def plan_metadata_only(
        cls,
        config_or_geometry,
        policy=None,
        target_shard_bytes=5 * GIB,
        **planner_options
    ):
        """Build a complete plan and memory estimate without tensor allocation."""

        from .adapter import LlamaModelAdapter, ModelGeometry

        adapter = LlamaModelAdapter()
        if isinstance(config_or_geometry, ModelGeometry):
            geometry = config_or_geometry
            config = geometry.as_dict()
            config["_name_or_path"] = geometry.model_id
        else:
            config = config_or_geometry
            geometry = adapter.build_geometry(config)
        policy = policy or ExecutionPolicy()
        plan = adapter.build_execution_plan(config, policy)
        estimate = cls(
            plan,
            geometry,
            policy=policy,
            **planner_options
        ).estimate()
        max_offset = max(plan.host_offsets.values(), default=0)
        int64_safe = (
            max_offset <= (1 << 63) - 1
            and plan.host_arena_bytes <= (1 << 63) - 1
        )
        if not int64_safe:
            raise OverflowError("metadata plan exceeds signed 64-bit offsets")
        shard_bytes = int(target_shard_bytes)
        if shard_bytes <= 0:
            raise ValueError("target_shard_bytes must be positive")
        shard_count = max(
            1,
            (int(plan.host_arena_bytes) + shard_bytes - 1) // shard_bytes,
        )
        return MetadataPlanResult(
            plan=plan,
            estimate=estimate,
            max_host_offset=max_offset,
            estimated_shard_count=shard_count,
            offsets_are_int64_safe=int64_safe,
        )

    def preflight(self, device=None, capacity=None, raise_on_error=True):
        estimate = self.estimate()
        capacity = capacity or collect_system_capacity(device)
        errors = []
        warnings = []
        suggestions = []
        if (
            capacity.cpu_available_bytes is not None
            and estimate.estimated_cpu_peak_bytes > capacity.cpu_available_bytes
        ):
            errors.append(
                "estimated CPU peak needs {:.2f} GiB, but only {:.2f} GiB of RAM is available".format(
                    estimate.estimated_cpu_peak_bytes / GIB,
                    capacity.cpu_available_bytes / GIB,
                )
            )
        if (
            capacity.memlock_limit_bytes is not None
            and estimate.pinned_staging_bytes > capacity.memlock_limit_bytes
        ):
            errors.append(
                "pinned memory needs {:.2f} GiB, above RLIMIT_MEMLOCK {:.2f} GiB".format(
                    estimate.pinned_staging_bytes / GIB,
                    capacity.memlock_limit_bytes / GIB,
                )
            )
            if self.policy.cpu_weight_mode == "full_pinned":
                suggestions.append(
                    "select cpu_weight_mode=pinned_staging or raise RLIMIT_MEMLOCK"
                )
            else:
                suggestions.append(
                    "reduce slot_count/vocab chunk size or raise RLIMIT_MEMLOCK"
                )
        if (
            capacity.gpu_free_bytes is not None
            and estimate.estimated_gpu_peak_bytes > capacity.gpu_free_bytes
        ):
            errors.append(
                "estimated GPU peak is {:.2f} GiB, but only {:.2f} GiB is free".format(
                    estimate.estimated_gpu_peak_bytes / GIB,
                    capacity.gpu_free_bytes / GIB,
                )
            )
            suggestions.append(
                "reduce max_context or slot_count, stream vocabulary weights, or disable full logits"
            )
        if capacity.gpu_free_bytes is None and device is not None:
            warnings.append("GPU capacity was not available for preflight")
        result = PreflightResult(
            estimate=estimate,
            capacity=capacity,
            errors=tuple(errors),
            warnings=tuple(warnings),
            suggestions=tuple(dict.fromkeys(suggestions)),
        )
        if raise_on_error:
            result.raise_for_error()
        return result
