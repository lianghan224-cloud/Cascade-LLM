"""Orthogonal KV-cache policy and capability contracts.

The policy deliberately separates accuracy, storage, dtype, selection, and
reuse.  Presets are only a user-interface convenience and always expand to a
fully explicit :class:`KVPolicy`.
"""

from dataclasses import dataclass, replace
from enum import Enum


class KVAccuracy(str, Enum):
    EXACT = "exact"
    QUANTIZED = "quantized"
    SPARSE = "sparse"


class KVStoragePolicy(str, Enum):
    GPU = "gpu"
    GPU_CPU = "gpu_cpu"
    GPU_CPU_NVME = "gpu_cpu_nvme"


class KVDataType(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    INT8 = "int8"
    FP8 = "fp8"
    INT4 = "int4"


class KVSelectionPolicy(str, Enum):
    # The CLI spelling is ``none`` because no pages are omitted.  ``dense`` is
    # accepted by ``normalize_selection`` as a configuration alias.
    DENSE = "none"
    QUEST_FLAT = "quest_flat"
    HIERARCHICAL_QUEST = "hierarchical_quest"
    CENTROID_ONLY = "centroid_only"


class KVReusePolicy(str, Enum):
    REQUEST_ONLY = "request_only"
    # Compatibility alias for the D0/D1 spelling. New reports serialize the
    # canonical `request_only` value.
    NONE = "request_only"
    SESSION = "session"
    PREFIX_MEMORY = "prefix_memory"
    PREFIX_PERSISTENT = "prefix_persistent"


class KVLayout(str, Enum):
    HND = "hnd"


def _normalize(value):
    return str(value).strip().lower().replace("-", "_")


def normalize_selection(value):
    if isinstance(value, KVSelectionPolicy):
        return value
    value = _normalize(value)
    if value in {"none", "dense"}:
        return KVSelectionPolicy.DENSE
    return KVSelectionPolicy(value)


def normalize_reuse(value):
    if isinstance(value, KVReusePolicy):
        return value
    value = _normalize(value)
    if value in {"none", "off", "request", "request_only"}:
        return KVReusePolicy.REQUEST_ONLY
    if value in {"memory", "prefix_memory"}:
        return KVReusePolicy.PREFIX_MEMORY
    if value in {"persistent", "prefix_persistent"}:
        return KVReusePolicy.PREFIX_PERSISTENT
    return KVReusePolicy(value)


@dataclass(frozen=True)
class KVPolicy:
    accuracy: KVAccuracy = KVAccuracy.EXACT
    storage: KVStoragePolicy = KVStoragePolicy.GPU
    dtype: KVDataType = KVDataType.BF16
    selection: KVSelectionPolicy = KVSelectionPolicy.DENSE
    reuse: KVReusePolicy = KVReusePolicy.REQUEST_ONLY
    attention_backend: str = "generic_cuda"
    page_size: int = 16
    cpu_budget_bytes: int = 0
    nvme_budget_bytes: int = 0
    page_budget: int = 0
    recent_window: int = 0

    def __post_init__(self):
        object.__setattr__(self, "accuracy", KVAccuracy(self.accuracy))
        object.__setattr__(self, "storage", KVStoragePolicy(self.storage))
        object.__setattr__(self, "dtype", KVDataType(self.dtype))
        object.__setattr__(
            self, "selection", normalize_selection(self.selection)
        )
        object.__setattr__(self, "reuse", normalize_reuse(self.reuse))
        object.__setattr__(
            self,
            "attention_backend",
            _normalize(self.attention_backend),
        )
        for name in (
            "page_size",
            "cpu_budget_bytes",
            "nvme_budget_bytes",
            "page_budget",
            "recent_window",
        ):
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.page_size <= 0:
            raise ValueError("KV page_size must be positive")
        if not self.attention_backend:
            raise ValueError("attention_backend must be explicit")
        for name in (
            "cpu_budget_bytes",
            "nvme_budget_bytes",
            "page_budget",
            "recent_window",
        ):
            if getattr(self, name) < 0:
                raise ValueError("{} must not be negative".format(name))
        self._validate_accuracy_contract()

    def _validate_accuracy_contract(self):
        if self.accuracy == KVAccuracy.EXACT:
            if self.dtype not in {KVDataType.BF16, KVDataType.FP16}:
                raise ValueError(
                    "exact KV requires bf16 or fp16, got {}".format(
                        self.dtype.value
                    )
                )
            if self.selection != KVSelectionPolicy.DENSE:
                raise ValueError("exact KV cannot discard pages")
        elif self.accuracy == KVAccuracy.QUANTIZED:
            if self.dtype not in {
                KVDataType.INT8,
                KVDataType.FP8,
                KVDataType.INT4,
            }:
                raise ValueError("quantized KV requires a low-precision dtype")
            if self.selection != KVSelectionPolicy.DENSE:
                raise ValueError(
                    "quantized accuracy does not include sparse selection"
                )
        elif self.selection == KVSelectionPolicy.DENSE:
            raise ValueError("sparse KV requires an explicit sparse index")

        if (
            self.storage == KVStoragePolicy.GPU
            and (self.cpu_budget_bytes or self.nvme_budget_bytes)
        ):
            raise ValueError("GPU-only KV cannot reserve CPU or NVMe budgets")
        if (
            self.storage == KVStoragePolicy.GPU_CPU
            and self.nvme_budget_bytes
        ):
            raise ValueError("gpu_cpu KV cannot reserve an NVMe budget")
        if (
            self.reuse == KVReusePolicy.PREFIX_PERSISTENT
            and self.storage != KVStoragePolicy.GPU_CPU_NVME
        ):
            raise ValueError(
                "persistent prefix reuse requires gpu_cpu_nvme storage"
            )

    def d1_support_errors(self):
        """Return why this policy cannot run on the current D1 backend."""

        errors = []
        if self.accuracy != KVAccuracy.EXACT:
            errors.append("D1 implements exact KV only")
        if self.storage != KVStoragePolicy.GPU:
            errors.append("D1 implements GPU-resident KV only")
        if self.dtype not in {KVDataType.BF16, KVDataType.FP16}:
            errors.append("D1 implements BF16/FP16 KV only")
        if self.selection != KVSelectionPolicy.DENSE:
            errors.append("D1 implements dense page selection only")
        if self.reuse not in {
            KVReusePolicy.REQUEST_ONLY,
            KVReusePolicy.SESSION,
            KVReusePolicy.PREFIX_MEMORY,
        }:
            errors.append("V1 does not implement persistent prefix reuse")
        return tuple(errors)

    def require_d1_supported(self):
        errors = self.d1_support_errors()
        if errors:
            raise NotImplementedError("; ".join(errors))
        return self

    def as_dict(self):
        return {
            "accuracy": self.accuracy.value,
            "storage": self.storage.value,
            "dtype": self.dtype.value,
            "selection": self.selection.value,
            "reuse": self.reuse.value,
            "attention_backend": self.attention_backend,
            "page_size": self.page_size,
            "cpu_budget_bytes": self.cpu_budget_bytes,
            "nvme_budget_bytes": self.nvme_budget_bytes,
            "page_budget": self.page_budget,
            "recent_window": self.recent_window,
        }

    @property
    def storage_policy(self):
        return self.storage

    @property
    def format_policy(self):
        return self.dtype

    @property
    def selection_policy(self):
        return self.selection

    @property
    def reuse_policy(self):
        return self.reuse


_PRESETS = {
    "performance": KVPolicy(),
    "balanced": KVPolicy(),
    "reuse": KVPolicy(reuse=KVReusePolicy.PREFIX_MEMORY),
    "long_context": KVPolicy(
        accuracy=KVAccuracy.SPARSE,
        selection=KVSelectionPolicy.QUEST_FLAT,
        page_budget=256,
        recent_window=2048,
    ),
}


def expand_kv_preset(name, overrides=None):
    key = _normalize(name)
    if key not in _PRESETS:
        raise ValueError("unknown KV preset {!r}".format(name))
    policy = _PRESETS[key]
    values = dict(overrides or {})
    if "selection" in values:
        values["selection"] = normalize_selection(values["selection"])
    return replace(policy, **values)


def kv_page_pool_bytes(
    *,
    layer_count,
    page_count,
    num_key_value_heads,
    page_size,
    head_dim,
    batch_size=1,
    dtype=KVDataType.BF16,
):
    dtype = KVDataType(dtype)
    bits = {
        KVDataType.BF16: 16,
        KVDataType.FP16: 16,
        KVDataType.INT8: 8,
        KVDataType.FP8: 8,
        KVDataType.INT4: 4,
    }[dtype]
    elements = (
        2
        * int(layer_count)
        * int(page_count)
        * int(batch_size)
        * int(num_key_value_heads)
        * int(page_size)
        * int(head_dim)
    )
    return (elements * bits + 7) // 8


@dataclass(frozen=True)
class KVCapability:
    attention_type: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    supports_mha: bool
    supports_gqa: bool
    supports_mla: bool = False
    supports_mamba_state: bool = False

    @classmethod
    def from_geometry(cls, geometry):
        attention_heads = int(geometry.num_attention_heads)
        kv_heads = int(geometry.num_key_value_heads)
        if attention_heads <= 0 or kv_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if attention_heads % kv_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        return cls(
            attention_type="mha" if attention_heads == kv_heads else "gqa",
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            head_dim=int(geometry.head_dim),
            supports_mha=attention_heads == kv_heads,
            supports_gqa=attention_heads != kv_heads,
        )

    def as_dict(self):
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }
