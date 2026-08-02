"""Explicit paged-provider registry and dispatcher; no silent fallback."""

import threading

import torch

from ...kv.errors import KVProviderError
from ...kv.kernel_backend import TorchPagedKVKernelBackend
from ...providers.base import PagedProviderBundle
from .reference import (
    LegacyGatherSDPAReferenceBackend,
    ReferencePagedExactBackend,
)


class PagedAttentionRegistry:
    def __init__(self):
        self._providers = {}
        self._lock = threading.RLock()

    def register(self, bundle, replace=False):
        if not isinstance(bundle, PagedProviderBundle):
            raise TypeError("paged registry accepts PagedProviderBundle only")
        name = str(bundle.name)
        with self._lock:
            if name in self._providers and not replace:
                raise ValueError("paged provider already registered: {}".format(name))
            self._providers[name] = bundle
        return bundle

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
            result = {}
            for name, bundle in sorted(self._providers.items()):
                # Keep top-level attention fields for report compatibility and
                # add the explicit bundle split.
                item = bundle.attention_backend.capability().as_dict()
                item["bundle_name"] = bundle.name
                item["attention_backend"] = bundle.attention_backend.name
                item["kv_kernel_backend"] = bundle.kv_kernel_backend.name
                item["kv_kernel_capability"] = (
                    bundle.kv_kernel_backend.capability().as_dict()
                )
                result[name] = item
            return result


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
    def bundle(self):
        return self.registry.get(self.provider_name)

    @property
    def attention_backend(self):
        return self.bundle.attention_backend

    @property
    def kv_kernel_backend(self):
        return self.bundle.kv_kernel_backend

    def validate(self, request, phase=None):
        request.validate()
        backend = self.attention_backend
        if backend.is_reference and not self.allow_reference:
            raise KVProviderError(
                "reference provider {} requires allow_reference=True".format(
                    backend.name
                )
            )
        architecture = detected_architecture(request.query.device)
        reason = backend.capability().unsupported_reason(
            request,
            architecture=architecture,
            phase=phase,
        )
        self.last_decision = {
            "requested": self.provider_name,
            "selected": self.bundle.name,
            "attention_backend": backend.name,
            "kv_kernel_backend": self.kv_kernel_backend.name,
            "architecture": architecture,
            "supported": reason is None,
            "fallback_reason": None,
            "unsupported_reason": reason,
            "is_reference": bool(backend.is_reference),
        }
        if reason is not None:
            raise KVProviderError(
                "paged provider {} rejected request: {}".format(
                    backend.name, reason
                )
            )
        return backend

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
    registry.register(
        PagedProviderBundle(
            name="reference_paged_exact",
            attention_backend=ReferencePagedExactBackend(),
            kv_kernel_backend=TorchPagedKVKernelBackend(),
        )
    )
    registry.register(
        PagedProviderBundle(
            name="legacy_gather_sdpa_reference",
            attention_backend=LegacyGatherSDPAReferenceBackend(),
            kv_kernel_backend=TorchPagedKVKernelBackend(),
        )
    )
    if load_cuda:
        try:
            from ...providers.generic_cuda.paged_attention import (
                GenericCUDAPagedAttentionBackend,
                GenericCUDAPagedKVKernelBackend,
            )

            registry.register(
                PagedProviderBundle(
                    name="generic_cuda",
                    attention_backend=GenericCUDAPagedAttentionBackend(),
                    kv_kernel_backend=GenericCUDAPagedKVKernelBackend(),
                )
            )
            from ...providers.sm80.paged_attention import (
                SM80PagedAttentionBackend,
                SM80PagedKVKernelBackend,
            )
            from ...providers.sm86.paged_attention import (
                SM86PagedAttentionBackend,
                SM86PagedKVKernelBackend,
            )
            from ...providers.sm89.paged_attention import (
                SM89PagedAttentionBackend,
                SM89PagedKVKernelBackend,
            )
            from ...providers.sm90.paged_attention import (
                SM90PagedAttentionBackend,
                SM90PagedKVKernelBackend,
            )

            for name, attention_type, kv_kernel_type in (
                ("sm80", SM80PagedAttentionBackend, SM80PagedKVKernelBackend),
                ("sm86", SM86PagedAttentionBackend, SM86PagedKVKernelBackend),
                ("sm89", SM89PagedAttentionBackend, SM89PagedKVKernelBackend),
                ("sm90", SM90PagedAttentionBackend, SM90PagedKVKernelBackend),
            ):
                registry.register(
                    PagedProviderBundle(
                        name=name,
                        attention_backend=attention_type(),
                        kv_kernel_backend=kv_kernel_type(),
                    )
                )
        except ImportError:
            # Registration absence is visible through list/capability and is
            # never converted into a reference fallback.
            pass
    return registry
