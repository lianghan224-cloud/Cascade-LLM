"""Provider capability and compatibility request contracts."""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


QUALIFICATION_STATUSES = (
    "unknown",
    "declared",
    "compiled",
    "smoke_passed",
    "qualified",
    "production",
    "unsupported",
    "disabled",
)


@dataclass(frozen=True)
class ProviderCapability:
    provider_name: str
    provider_version: str
    provider_abi: int
    supported_architectures: Tuple[str, ...]
    supported_weight_formats: Tuple[str, ...]
    supported_activation_dtypes: Tuple[str, ...]
    supported_scale_dtypes: Tuple[str, ...]
    supported_group_sizes: Tuple[int, ...]
    supported_phases: Tuple[str, ...]
    min_m: int
    max_m: Optional[int]
    alignment_m: int
    alignment_n: int
    alignment_k: int
    requires_preprocessed_layout: bool
    physical_layout_names: Tuple[str, ...]
    workspace_policy: str
    qualification_status: str
    backend_name: str
    explicit_fallback: bool = False
    requires_extension: bool = False

    def __post_init__(self):
        if not self.provider_name or not self.backend_name:
            raise ValueError("provider_name and backend_name must be non-empty")
        if self.qualification_status not in QUALIFICATION_STATUSES:
            raise ValueError(
                "invalid qualification status {}".format(
                    self.qualification_status
                )
            )
        if self.provider_abi < 0:
            raise ValueError("provider_abi must be non-negative")
        if self.min_m < 1 or any(
            value < 1
            for value in (self.alignment_m, self.alignment_n, self.alignment_k)
        ):
            raise ValueError("M and alignments must be positive")
        if self.max_m is not None and self.max_m < self.min_m:
            raise ValueError("max_m must be >= min_m")

    def as_dict(self):
        value = asdict(self)
        for name in (
            "supported_architectures",
            "supported_weight_formats",
            "supported_activation_dtypes",
            "supported_scale_dtypes",
            "supported_group_sizes",
            "supported_phases",
            "physical_layout_names",
        ):
            value[name] = list(value[name])
        return value


@dataclass(frozen=True)
class CompatibilityRequest:
    phase: str
    backend_requested: str
    weight_format: str
    activation_dtype: str
    scale_dtype: Optional[str]
    group_size: Optional[int]
    m: int
    n: int
    k: int
    physical_layout: Optional[str]
    workspace_limit_bytes: int

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CompatibilityDecision:
    supported: bool
    provider_name: Optional[str]
    status: str
    reasons: Tuple[str, ...]
    warnings: Tuple[str, ...]
    explicit_fallback_available: bool

    def as_dict(self):
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        value["warnings"] = list(self.warnings)
        return value
