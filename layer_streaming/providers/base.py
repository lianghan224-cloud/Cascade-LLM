"""Provider-facing contracts kept outside the frozen LinearBackend protocol."""

from ..backend_capability import BackendCapability
from ..backends import BackendInfo
from ..hardware.capability import ProviderCapability

__all__ = ["BackendCapability", "BackendInfo", "ProviderCapability"]
