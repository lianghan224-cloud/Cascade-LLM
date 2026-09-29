"""Explicit paged-provider registry and dispatcher; no silent fallback."""

import threading

import torch

from ...kv.errors import KVProviderError, KVUnsupportedError
from .abi import DevicePagedAttentionInput
from ...kv.kernel_backend import TorchPagedKVKernelBackend
from ...providers.base import PagedProviderBundle
from .reference import (
    GatherSDPAPrefillBackend,
    LegacyGatherSDPAReferenceBackend,
    ReferencePagedExactBackend,
)
from .routing import PagedWorkload, classify_paged_workload


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
    def __init__(
        self,
        registry,
        provider_name,
        allow_reference=False,
        prefill_provider_name="reference_paged_exact",
    ):
        self.registry = registry
        self.provider_name = str(provider_name)
        self.allow_reference = bool(allow_reference)
        self.prefill_provider_name = str(prefill_provider_name)
        self.registry.get(self.prefill_provider_name)
        self.last_decision = None
        self._decision_counts = {}
        self._total_calls = 0
        self._fallback_calls = 0
        self._reference_fallback_calls = 0

    def _record_decision(self):
        """Accumulate routing evidence without retaining an unbounded trace."""

        if self.last_decision is None:
            return
        decision = dict(self.last_decision)
        key = (
            decision.get("workload"),
            decision.get("selected"),
            decision.get("attention_backend"),
            decision.get("fallback_reason"),
            bool(decision.get("is_reference", False)),
        )
        self._decision_counts[key] = self._decision_counts.get(key, 0) + 1
        self._total_calls += 1
        if decision.get("fallback_reason") is not None:
            self._fallback_calls += 1
            if decision.get("is_reference", False):
                self._reference_fallback_calls += 1

    def routing_summary(self):
        """Return cumulative, JSON-safe routing counts for the runtime."""

        decisions = []
        for key, count in sorted(
            self._decision_counts.items(),
            key=lambda item: tuple(
                "" if value is None else str(value) for value in item[0]
            ),
        ):
            (
                workload,
                selected,
                attention_backend,
                fallback_reason,
                is_reference,
            ) = key
            decisions.append(
                {
                    "workload": workload,
                    "selected": selected,
                    "attention_backend": attention_backend,
                    "fallback_reason": fallback_reason,
                    "is_reference": bool(is_reference),
                    "count": int(count),
                }
            )
        return {
            "total_calls": int(self._total_calls),
            "fallback_calls": int(self._fallback_calls),
            "reference_fallback_calls": int(
                self._reference_fallback_calls
            ),
            "decisions": decisions,
        }

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
            "workload": (
                phase.value if isinstance(phase, PagedWorkload) else phase
            ),
        }
        if reason is not None:
            raise KVProviderError(
                "paged provider {} rejected request: {}".format(
                    backend.name, reason
                )
            )
        return backend

    def execute(self, request, phase=None):
        workload = classify_paged_workload(request, phase=phase)
        provider = self.attention_backend
        supported_workloads = tuple(
            getattr(provider, "workload_kinds", ())
        )
        if workload.value not in supported_workloads:
            fallback_bundle = self.registry.get(self.prefill_provider_name)
            fallback = fallback_bundle.attention_backend
            if workload.value not in tuple(
                getattr(fallback, "workload_kinds", ())
            ):
                raise KVProviderError(
                    "configured prefill provider {} does not implement {}".format(
                        fallback.name, workload.value
                    )
                )
            request.validate()
            architecture = detected_architecture(request.query.device)
            reason = fallback.capability().unsupported_reason(
                request,
                architecture=architecture,
                phase=workload.capability_phase,
            )
            if reason is not None:
                raise KVProviderError(
                    "correctness fallback {} rejected {}: {}".format(
                        fallback.name, workload.value, reason
                    )
                )
            self.last_decision = {
                "requested": self.provider_name,
                "selected": fallback_bundle.name,
                "attention_backend": fallback.name,
                "kv_kernel_backend": self.kv_kernel_backend.name,
                "architecture": architecture,
                "supported": True,
                "fallback_reason": (
                    "{} has no {} kernel; using {} {}".format(
                        provider.name,
                        workload.value,
                        (
                            "correctness prefill fallback"
                            if fallback.is_reference
                            else "configured prefill provider"
                        ),
                        fallback.name,
                    )
                ),
                "unsupported_reason": None,
                "is_reference": bool(fallback.is_reference),
                "workload": workload.value,
            }
            provider = fallback
        else:
            provider = self.validate(
                request, phase=workload.capability_phase
            )
            self.last_decision["workload"] = workload.value
        result = getattr(provider, workload.backend_method)(request)
        self._record_decision()
        return result

    def execute_device(self, request, *, phase):
        """Execute the strict device-selected Decode ABI without fallback."""

        if not isinstance(request, DevicePagedAttentionInput):
            raise TypeError("device execution requires DevicePagedAttentionInput")
        normalized_phase = (
            phase.value if isinstance(phase, PagedWorkload) else str(phase).lower()
        )
        if normalized_phase != PagedWorkload.DECODE.value:
            raise KVUnsupportedError("UNSUPPORTED_DEVICE_SELECTED_VIEW")
        provider = self.attention_backend
        if (
            provider.is_reference
            or not bool(getattr(provider, "supports_device_selected_view", False))
        ):
            raise KVUnsupportedError("UNSUPPORTED_DEVICE_SELECTED_VIEW")
        provider = self.validate(request, phase="decode")
        self.last_decision["workload"] = PagedWorkload.DECODE.value
        self.last_decision["selected_view"] = "device"
        result = provider.decode_device(request)
        self._record_decision()
        return result


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
            name="gather_sdpa_prefill",
            attention_backend=GatherSDPAPrefillBackend(),
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
