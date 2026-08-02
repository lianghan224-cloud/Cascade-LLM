"""Explicit paged-provider registry and dispatcher; no silent fallback."""

import threading

import torch

from ...kv.errors import KVProviderError
from .reference import (
    LegacyGatherSDPAReferenceProvider,
    ReferencePagedExactProvider,
)


class PagedAttentionRegistry:
    def __init__(self):
        self._providers = {}
        self._lock = threading.RLock()

    def register(self, provider, replace=False):
        name = str(provider.name)
        with self._lock:
            if name in self._providers and not replace:
                raise ValueError("paged provider already registered: {}".format(name))
            self._providers[name] = provider
        return provider

    def get(self, name):
        with self._lock:
            if name not in self._providers:
                raise KeyError("unknown paged provider {!r}".format(name))
            return self._providers[name]

    def list(self):
        with self._lock:
            return tuple(sorted(self._providers))

    def capabilities(self):
        with self._lock:
            return {
                name: provider.capability().as_dict()
                for name, provider in sorted(self._providers.items())
            }


def detected_architecture(device):
    device = torch.device(device)
    if device.type != "cuda":
        return "cpu"
    major, minor = torch.cuda.get_device_capability(device)
    return "sm{}{}".format(major, minor)


class PagedAttentionDispatcher:
    def __init__(self, registry, provider_name, allow_reference=False):
        self.registry = registry
        self.provider_name = str(provider_name)
        self.allow_reference = bool(allow_reference)
        self.last_decision = None

    @property
    def provider(self):
        return self.registry.get(self.provider_name)

    def validate(self, request, phase=None):
        request.validate()
        provider = self.provider
        if provider.is_reference and not self.allow_reference:
            raise KVProviderError(
                "reference provider {} requires allow_reference=True".format(
                    provider.name
                )
            )
        architecture = detected_architecture(request.query.device)
        reason = provider.capability().unsupported_reason(
            request,
            architecture=architecture,
            phase=phase,
        )
        self.last_decision = {
            "requested": self.provider_name,
            "selected": provider.name,
            "architecture": architecture,
            "supported": reason is None,
            "fallback_reason": None,
            "unsupported_reason": reason,
            "is_reference": bool(provider.is_reference),
        }
        if reason is not None:
            raise KVProviderError(
                "paged provider {} rejected request: {}".format(
                    provider.name, reason
                )
            )
        return provider

    def execute(self, request, phase=None):
        phase = phase or (
            "decode" if request.batch_view.max_query_length == 1 else "prefill"
        )
        provider = self.validate(request, phase=phase)
        return (
            provider.decode(request)
            if phase == "decode"
            else provider.prefill(request)
        )


def default_paged_registry(load_cuda=True):
    registry = PagedAttentionRegistry()
    registry.register(ReferencePagedExactProvider())
    registry.register(LegacyGatherSDPAReferenceProvider())
    if load_cuda:
        try:
            from ...providers.generic_cuda.paged_attention import (
                GenericCUDAPagedAttentionProvider,
            )

            registry.register(GenericCUDAPagedAttentionProvider())
            from ...providers.sm80.paged_attention import SM80PagedAttentionProvider
            from ...providers.sm86.paged_attention import SM86PagedAttentionProvider
            from ...providers.sm89.paged_attention import SM89PagedAttentionProvider
            from ...providers.sm90.paged_attention import SM90PagedAttentionProvider

            for provider_type in (
                SM80PagedAttentionProvider,
                SM86PagedAttentionProvider,
                SM89PagedAttentionProvider,
                SM90PagedAttentionProvider,
            ):
                registry.register(provider_type())
        except ImportError:
            # Registration absence is visible through list/capability and is
            # never converted into a reference fallback.
            pass
    return registry
