"""Resolve an explicit backend request before ExecutionPlan execution."""

from .capability import CompatibilityDecision


_QUALIFIED = {
    "numerically_qualified",
    "performance_qualified",
    "production",
}


class CompatibilityResolver:
    def __init__(self, registry):
        self.registry = registry

    def resolve(self, hardware, runtime, request):
        candidates = self.registry.candidates(request.backend_requested)
        if not candidates:
            return CompatibilityDecision(
                supported=False,
                provider_name=None,
                status="unsupported",
                reasons=(
                    "requested backend/provider {} is not registered".format(
                        request.backend_requested
                    ),
                ),
                warnings=(),
                explicit_fallback_available=self._fallback_available(request),
            )
        matching_arch = tuple(
            item
            for item in candidates
            if hardware.architecture in item.supported_architectures
        )
        if not matching_arch:
            return CompatibilityDecision(
                supported=False,
                provider_name=None,
                status="unsupported",
                reasons=(
                    "detected architecture {} is not declared by {}".format(
                        hardware.architecture,
                        ", ".join(item.provider_name for item in candidates),
                    ),
                ),
                warnings=(),
                explicit_fallback_available=self._fallback_available(request),
            )
        capability = matching_arch[0]
        reasons = []
        warnings = []
        if capability.qualification_status == "unsupported":
            reasons.append(
                "provider status is {}".format(capability.qualification_status)
            )
        if not runtime.cuda_available:
            reasons.append("CUDA runtime is unavailable")
        if request.phase not in capability.supported_phases:
            reasons.append("phase {} is unsupported".format(request.phase))
        if request.weight_format not in capability.supported_weight_formats:
            reasons.append(
                "weight format {} is unsupported".format(request.weight_format)
            )
        if request.activation_dtype not in capability.supported_activation_dtypes:
            reasons.append(
                "activation dtype {} is unsupported".format(
                    request.activation_dtype
                )
            )
        if request.activation_dtype == "bf16" and not hardware.supports_bf16:
            reasons.append("hardware does not expose BF16 support")
        if request.activation_dtype == "fp16" and not hardware.supports_fp16:
            reasons.append("hardware does not expose FP16 support")
        if (
            request.scale_dtype is not None
            and request.scale_dtype not in capability.supported_scale_dtypes
        ):
            reasons.append(
                "scale dtype {} is unsupported".format(request.scale_dtype)
            )
        if request.group_size is not None:
            if request.group_size not in capability.supported_group_sizes:
                reasons.append(
                    "group size {} is unsupported".format(request.group_size)
                )
            elif request.k % int(request.group_size):
                reasons.append(
                    "K={} is not divisible by group size {}".format(
                        request.k, request.group_size
                    )
                )
        if request.m < capability.min_m or (
            capability.max_m is not None and request.m > capability.max_m
        ):
            reasons.append("M={} is outside provider range".format(request.m))
        for name, value, alignment in (
            ("M", request.m, capability.alignment_m),
            ("N", request.n, capability.alignment_n),
            ("K", request.k, capability.alignment_k),
        ):
            if value % alignment:
                reasons.append(
                    "{}={} is not aligned to {}".format(name, value, alignment)
                )
        if capability.requires_preprocessed_layout:
            if request.physical_layout not in capability.physical_layout_names:
                reasons.append(
                    "physical layout {} is unsupported".format(
                        request.physical_layout
                    )
                )
        elif (
            request.physical_layout is not None
            and capability.physical_layout_names
            and request.physical_layout not in capability.physical_layout_names
        ):
            reasons.append(
                "physical layout {} is unsupported".format(
                    request.physical_layout
                )
            )
        if request.workspace_limit_bytes < 0:
            reasons.append("workspace limit must be non-negative")
        if capability.requires_extension:
            if hardware.architecture not in runtime.compiled_architectures:
                reasons.append(
                    "compiled architecture missing: {}".format(
                        hardware.architecture
                    )
                )
            if not runtime.cutlass_extension_loaded:
                reasons.append("CUTLASS extension is not loaded")
            if capability.provider_abi not in runtime.provider_abi_versions:
                reasons.append(
                    "provider ABI {} is not loaded".format(
                        capability.provider_abi
                    )
                )
        if capability.qualification_status not in _QUALIFIED:
            warnings.append(
                "provider status is {}; production qualification has not passed".format(
                    capability.qualification_status
                )
            )
        supported = not reasons
        status = capability.qualification_status if supported else "unsupported"
        return CompatibilityDecision(
            supported=supported,
            provider_name=capability.provider_name,
            status=status,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
            explicit_fallback_available=self._fallback_available(request),
        )

    def _fallback_available(self, request):
        return any(
            item.explicit_fallback
            and request.weight_format in item.supported_weight_formats
            and request.activation_dtype in item.supported_activation_dtypes
            for item in self.registry.list_providers()
        )
