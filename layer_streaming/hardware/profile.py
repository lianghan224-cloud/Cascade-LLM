"""Serializable hardware and runtime profiles for compatibility checks."""

from dataclasses import asdict, dataclass
from typing import Optional, Tuple


HARDWARE_PROFILE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HardwareProfile:
    vendor: str
    device_name: str
    device_index: int
    compute_capability_major: int
    compute_capability_minor: int
    architecture: str
    total_memory_bytes: int
    free_memory_bytes: int
    driver_version: Optional[str]
    cuda_runtime_version: Optional[str]
    torch_version: str
    torch_cuda_version: Optional[str]
    supports_fp16: bool
    supports_bf16: bool
    supports_int8_tensor_core: bool
    supports_int4_tensor_core: bool
    unified_addressing: bool
    async_copy_supported: bool
    schema_version: int = HARDWARE_PROFILE_SCHEMA_VERSION

    @property
    def compute_capability(self):
        return (
            int(self.compute_capability_major),
            int(self.compute_capability_minor),
        )

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version = int(value.get("schema_version", 0))
        if version != HARDWARE_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported HardwareProfile schema version {}".format(version)
            )
        return cls(**value)


@dataclass(frozen=True)
class RuntimeFeatureProfile:
    cuda_available: bool
    cuda_graph_available: bool
    pinned_memory_available: bool
    cutlass_extension_loaded: bool
    provider_abi_versions: Tuple[int, ...]
    compiled_architectures: Tuple[str, ...]
    deterministic_mode: bool
    schema_version: int = HARDWARE_PROFILE_SCHEMA_VERSION

    def as_dict(self):
        value = asdict(self)
        value["provider_abi_versions"] = list(self.provider_abi_versions)
        value["compiled_architectures"] = list(self.compiled_architectures)
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version = int(value.get("schema_version", 0))
        if version != HARDWARE_PROFILE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported RuntimeFeatureProfile schema version {}".format(
                    version
                )
            )
        value["provider_abi_versions"] = tuple(
            int(item) for item in value.get("provider_abi_versions", ())
        )
        value["compiled_architectures"] = tuple(
            str(item) for item in value.get("compiled_architectures", ())
        )
        return cls(**value)
