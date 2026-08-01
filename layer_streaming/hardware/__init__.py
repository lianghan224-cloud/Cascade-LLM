"""NVIDIA hardware compatibility layer."""

from .build_metadata import (
    PROVIDER_BUILD_METADATA_SCHEMA_VERSION,
    ProviderBuildMetadata,
)
from .capability import (
    QUALIFICATION_STATUSES,
    CompatibilityDecision,
    CompatibilityRequest,
    ProviderCapability,
)
from .compatibility import CompatibilityResolver
from .detector import (
    HardwareDetector,
    architecture_for_compute_capability,
    fake_hardware_profile,
)
from .profile import (
    HARDWARE_PROFILE_SCHEMA_VERSION,
    HardwareProfile,
    RuntimeFeatureProfile,
)
from .registry import ProviderRegistry, default_provider_registry
from .report import (
    COMPATIBILITY_REPORT_SCHEMA_VERSION,
    CompatibilityReport,
    build_compatibility_report,
)

__all__ = [
    "COMPATIBILITY_REPORT_SCHEMA_VERSION",
    "CompatibilityDecision",
    "CompatibilityReport",
    "CompatibilityRequest",
    "CompatibilityResolver",
    "HARDWARE_PROFILE_SCHEMA_VERSION",
    "HardwareDetector",
    "HardwareProfile",
    "PROVIDER_BUILD_METADATA_SCHEMA_VERSION",
    "ProviderBuildMetadata",
    "ProviderCapability",
    "ProviderRegistry",
    "QUALIFICATION_STATUSES",
    "RuntimeFeatureProfile",
    "architecture_for_compute_capability",
    "build_compatibility_report",
    "default_provider_registry",
    "fake_hardware_profile",
]
