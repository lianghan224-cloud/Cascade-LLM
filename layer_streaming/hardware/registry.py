"""Hardware-aware provider descriptor registry."""

from .capability import ProviderCapability


class ProviderRegistry:
    def __init__(self):
        self._providers = {}

    def register(self, provider, replace=False):
        capability = getattr(provider, "capability", provider)
        if not isinstance(capability, ProviderCapability):
            raise TypeError("provider must expose ProviderCapability")
        name = capability.provider_name
        if name in self._providers and not replace:
            raise ValueError("provider {} is already registered".format(name))
        for existing in self._providers.values():
            if (
                existing.backend_name == capability.backend_name
                and existing.provider_abi == capability.provider_abi
                and set(existing.supported_architectures).intersection(
                    capability.supported_architectures
                )
                and not replace
            ):
                raise ValueError(
                    "provider {} conflicts with {} for backend {} ABI {}".format(
                        name,
                        existing.provider_name,
                        capability.backend_name,
                        capability.provider_abi,
                    )
                )
        self._providers[name] = capability
        return capability

    def get(self, name):
        return self._providers.get(str(name))

    def list_providers(self):
        return tuple(
            self._providers[name] for name in sorted(self._providers)
        )

    def candidates(self, backend_requested):
        requested = str(backend_requested)
        return tuple(
            capability
            for capability in self.list_providers()
            if capability.provider_name == requested
            or capability.backend_name == requested
        )


def _torch_capability(
    provider_name,
    backend_name,
    weight_formats,
    activation_dtypes,
    scale_dtypes=(),
    group_sizes=(),
    explicit_fallback=False,
):
    return ProviderCapability(
        provider_name=provider_name,
        provider_version=str(__import__("torch").__version__),
        provider_abi=0,
        supported_architectures=("sm75", "sm80", "sm86", "sm89", "sm90"),
        supported_weight_formats=tuple(weight_formats),
        supported_activation_dtypes=tuple(activation_dtypes),
        supported_scale_dtypes=tuple(scale_dtypes),
        supported_group_sizes=tuple(group_sizes),
        supported_phases=("prefill", "decode"),
        min_m=1,
        max_m=None,
        alignment_m=1,
        alignment_n=1,
        alignment_k=1,
        requires_preprocessed_layout=False,
        physical_layout_names=("row_major",),
        workspace_policy="dequantized weight workspace" if explicit_fallback else "torch",
        qualification_status="compiled",
        backend_name=backend_name,
        explicit_fallback=bool(explicit_fallback),
        requires_extension=False,
    )


def default_provider_registry():
    registry = ProviderRegistry()
    for capability in (
        _torch_capability(
            "bf16_torch_linear", "bf16_linear", ("bf16",), ("bf16",)
        ),
        _torch_capability(
            "fp16_torch_linear", "fp16_linear", ("fp16",), ("fp16",)
        ),
        _torch_capability(
            "int8_dequant_bf16_fallback",
            "int8_dequant_bf16_fallback",
            ("int8_symmetric_per_channel", "int8_symmetric_per_group"),
            ("bf16",),
            ("bf16", "fp16"),
            (32, 64, 128),
            explicit_fallback=True,
        ),
        _torch_capability(
            "int8_dequant_fp16_fallback",
            "int8_dequant_fp16_fallback",
            ("int8_symmetric_per_channel", "int8_symmetric_per_group"),
            ("fp16",),
            ("bf16", "fp16"),
            (32, 64, 128),
            explicit_fallback=True,
        ),
        _torch_capability(
            "int4_dequant_bf16_fallback",
            "int4_dequant_bf16_fallback",
            ("int4_symmetric_per_group",),
            ("bf16",),
            ("bf16", "fp16"),
            (32, 64, 128),
            explicit_fallback=True,
        ),
        _torch_capability(
            "int4_dequant_fp16_fallback",
            "int4_dequant_fp16_fallback",
            ("int4_symmetric_per_group",),
            ("fp16",),
            ("bf16", "fp16"),
            (32, 64, 128),
            explicit_fallback=True,
        ),
        ProviderCapability(
            provider_name="cutlass_w8a16_sm86_abi2",
            provider_version="2",
            provider_abi=2,
            supported_architectures=("sm86",),
            supported_weight_formats=(
                "int8_symmetric_per_channel",
                "int8_symmetric_per_group",
            ),
            supported_activation_dtypes=("bf16", "fp16"),
            supported_scale_dtypes=("bf16", "fp16"),
            supported_group_sizes=(32, 64, 128),
            supported_phases=("prefill", "decode"),
            min_m=1,
            max_m=None,
            alignment_m=1,
            alignment_n=8,
            alignment_k=16,
            requires_preprocessed_layout=False,
            physical_layout_names=("row_major",),
            workspace_policy="none",
            qualification_status="qualified",
            backend_name="fused_w8a16",
            requires_extension=True,
        ),
    ):
        registry.register(capability)
    for architecture in ("sm80", "sm89", "sm90"):
        registry.register(
            ProviderCapability(
                provider_name="cutlass_w8a16_{}_abi1".format(architecture),
                provider_version="unverified",
                provider_abi=1,
                supported_architectures=(architecture,),
                supported_weight_formats=("int8_symmetric_per_channel",),
                supported_activation_dtypes=("bf16", "fp16"),
                supported_scale_dtypes=("bf16", "fp16"),
                supported_group_sizes=(),
                supported_phases=("prefill", "decode"),
                min_m=1,
                max_m=None,
                alignment_m=1,
                alignment_n=8,
                alignment_k=16,
                requires_preprocessed_layout=False,
                physical_layout_names=("row_major",),
                workspace_policy="unverified",
                qualification_status="declared",
                backend_name="fused_w8a16",
                requires_extension=True,
            )
        )
    return registry
