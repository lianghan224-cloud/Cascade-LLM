"""Provider registry compatibility layer."""

from ..backends import (
    register_linear_backend,
    unregister_linear_backend,
)
from ..hardware.registry import ProviderRegistry, default_provider_registry

__all__ = [
    "ProviderRegistry",
    "default_provider_registry",
    "register_linear_backend",
    "unregister_linear_backend",
]
