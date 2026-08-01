"""Sidecar backend capability and phase-selection contracts.

These types are intentionally not embedded in :class:`ExecutionPlan`.  The
frozen plan schema continues to describe storage and the maximum workspace;
this module validates a user-selected runtime provider before any kernel is
launched.
"""

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

import torch

from .backends import (
    BackendUnavailableError,
    backend_for_weight,
    default_backend_registry,
)
from .specs import normalize_dtype


def weight_format_name(weight):
    """Return the canonical capability format for a WeightSpec."""

    quant = weight.quantization
    if quant is None:
        return "dense_{}".format(weight.storage_dtype)
    if quant.bits == 8:
        return "int8_{}_{}".format(quant.scheme, quant.granularity)
    if quant.bits == 4:
        packing = quant.packing or "int4_pair_uint8"
        return "{}_{}_{}".format(packing, quant.scheme, quant.granularity)
    return "int{}_{}_{}".format(
        quant.bits, quant.scheme, quant.granularity
    )


@dataclass(frozen=True)
class BackendCapability:
    """Static provider limits used during startup qualification.

    An empty ``supported_sms`` tuple means that the backend is not tied to a
    particular CUDA SM (for example a PyTorch reference backend).
    """

    min_m: int
    max_m: Optional[int]
    supported_sms: Tuple[int, ...]
    activation_dtypes: Tuple[str, ...]
    weight_formats: Tuple[str, ...]
    group_sizes: Tuple[int, ...]
    alignment_k: int
    alignment_n: int

    def __post_init__(self):
        if int(self.min_m) < 1:
            raise ValueError("min_m must be positive")
        if self.max_m is not None and int(self.max_m) < int(self.min_m):
            raise ValueError("max_m must be >= min_m")
        if int(self.alignment_k) < 1 or int(self.alignment_n) < 1:
            raise ValueError("backend alignments must be positive")
        object.__setattr__(
            self,
            "activation_dtypes",
            tuple(normalize_dtype(item) for item in self.activation_dtypes),
        )
        object.__setattr__(
            self, "supported_sms", tuple(int(item) for item in self.supported_sms)
        )
        object.__setattr__(
            self, "group_sizes", tuple(int(item) for item in self.group_sizes)
        )

    def as_dict(self):
        return {
            "min_m": self.min_m,
            "max_m": self.max_m,
            "supported_sms": list(self.supported_sms),
            "activation_dtypes": list(self.activation_dtypes),
            "weight_formats": list(self.weight_formats),
            "group_sizes": list(self.group_sizes),
            "alignment_k": self.alignment_k,
            "alignment_n": self.alignment_n,
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for name in (
            "supported_sms",
            "activation_dtypes",
            "weight_formats",
            "group_sizes",
        ):
            value[name] = tuple(value.get(name, ()))
        return cls(**value)

    def unsupported_reason(self, weight, m, sm=None):
        m = int(m)
        n, k = (int(item) for item in weight.logical_shape)
        if m < self.min_m:
            return "M={} is below minimum M={}".format(m, self.min_m)
        if self.max_m is not None and m > self.max_m:
            return "M={} exceeds maximum M={}".format(m, self.max_m)
        if (
            sm is not None
            and self.supported_sms
            and int(sm) not in self.supported_sms
        ):
            return "SM{} is not in supported SMs {}".format(
                sm, self.supported_sms
            )
        if normalize_dtype(weight.compute_dtype) not in self.activation_dtypes:
            return "activation dtype {} is unsupported".format(
                weight.compute_dtype
            )
        storage_format = weight_format_name(weight)
        if storage_format not in self.weight_formats:
            return "weight format {} is unsupported".format(storage_format)
        quant = weight.quantization
        if (
            quant is not None
            and quant.granularity == "per_group"
            and int(quant.group_size) not in self.group_sizes
        ):
            return "group size {} is unsupported".format(quant.group_size)
        if k % self.alignment_k:
            return "K={} is not aligned to {}".format(k, self.alignment_k)
        if n % self.alignment_n:
            return "N={} is not aligned to {}".format(n, self.alignment_n)
        return None


@dataclass(frozen=True)
class BackendQualification:
    backend: str
    supported: bool
    unsupported_reason: Optional[str]
    workspace_bytes: int
    alignment_k: int
    alignment_n: int
    minimum_m: int
    required_sms: Tuple[int, ...]
    m: int
    n: int
    k: int
    activation_dtype: str
    weight_format: str

    def as_dict(self):
        return {
            "backend": self.backend,
            "supported": self.supported,
            "unsupported_reason": self.unsupported_reason,
            "workspace_bytes": self.workspace_bytes,
            "alignment_k": self.alignment_k,
            "alignment_n": self.alignment_n,
            "minimum_m": self.minimum_m,
            "required_sms": list(self.required_sms),
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "activation_dtype": self.activation_dtype,
            "weight_format": self.weight_format,
        }


@dataclass(frozen=True)
class BackendSelection:
    """Explicit prefill/decode provider request."""

    prefill: str
    decode: str

    def __post_init__(self):
        for name in ("prefill", "decode"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError("{} backend cannot be empty".format(name))
            object.__setattr__(self, name, value)

    @property
    def decode_fallback_explicit(self):
        return "fallback" in self.decode

    def as_dict(self):
        return {
            "prefill_backend": self.prefill,
            "decode_backend": self.decode,
            "decode_fallback_explicit": self.decode_fallback_explicit,
        }


@dataclass(frozen=True)
class BackendPhasePlan:
    """Runtime sidecar; it does not alter ExecutionPlan schema v1."""

    selection: BackendSelection
    prefill_backends: Mapping[str, str]
    decode_backends: Mapping[str, str]
    workspace_bytes: int
    device_sm: Optional[int]

    def backends_for_phase(self, phase):
        if phase == "prefill":
            return dict(self.prefill_backends)
        if phase == "decode":
            return dict(self.decode_backends)
        raise ValueError("phase must be prefill or decode")

    def as_dict(self):
        result = self.selection.as_dict()
        result.update(
            {
                "prefill_backends": dict(self.prefill_backends),
                "decode_backends": dict(self.decode_backends),
                "workspace_bytes": int(self.workspace_bytes),
                "device_sm": self.device_sm,
                "execution_plan_schema_version": 1,
            }
        )
        return result


def _sm_for_device(device):
    if device is None:
        return None
    device = torch.device(device)
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        raise BackendUnavailableError(
            "CUDA device {} was requested but CUDA is unavailable".format(device)
        )
    major, minor = torch.cuda.get_device_capability(device)
    return int(major) * 10 + int(minor)


def _fallback_capability(backend):
    formats = []
    storage = backend.info.storage_dtype
    if storage in {"bfloat16", "float16"}:
        formats.append("dense_{}".format(storage))
    elif storage == "int8":
        formats.extend(
            (
                "int8_symmetric_per_channel",
                "int8_symmetric_per_group",
            )
        )
    elif storage == "uint8":
        formats.append("int4_pair_uint8_symmetric_per_group")
    return BackendCapability(
        min_m=1,
        max_m=None,
        supported_sms=(),
        activation_dtypes=(backend.info.activation_dtype,),
        weight_formats=tuple(formats),
        group_sizes=(32, 64, 128),
        alignment_k=1,
        alignment_n=1,
    )


def capability_for_backend(backend):
    capability = getattr(backend, "capability", None)
    if capability is None:
        return _fallback_capability(backend)
    if not isinstance(capability, BackendCapability):
        raise TypeError(
            "backend {} capability must be BackendCapability".format(
                backend.name
            )
        )
    return capability


def qualify_backend(weight, backend_name, m, device=None, registry=None):
    """Qualify one backend/shape without launching a kernel."""

    registry = registry or default_backend_registry()
    try:
        backend = backend_for_weight(
            weight, registry=registry, backend_name=backend_name
        )
        capability = capability_for_backend(backend)
        sm = _sm_for_device(device)
        reason = capability.unsupported_reason(weight, m, sm=sm)
        provider_check = getattr(backend, "unsupported_reason", None)
        if reason is None and callable(provider_check):
            reason = provider_check(weight, int(m), sm=sm)
        workspace = (
            0
            if reason is not None
            else int(backend.workspace_bytes(weight, batch_tokens=int(m)))
        )
    except (BackendUnavailableError, TypeError, ValueError) as error:
        backend = registry.get(str(backend_name))
        capability = (
            capability_for_backend(backend)
            if backend is not None
            else BackendCapability(
                1, None, (), (), (), (), 1, 1
            )
        )
        reason = "{}: {}".format(type(error).__name__, error)
        workspace = 0
    n, k = (int(item) for item in weight.logical_shape)
    return BackendQualification(
        backend=str(backend_name),
        supported=reason is None,
        unsupported_reason=reason,
        workspace_bytes=workspace,
        alignment_k=capability.alignment_k,
        alignment_n=capability.alignment_n,
        minimum_m=capability.min_m,
        required_sms=capability.supported_sms,
        m=int(m),
        n=n,
        k=k,
        activation_dtype=weight.compute_dtype,
        weight_format=weight_format_name(weight),
    )


def build_backend_phase_plan(
    plan,
    selection,
    device=None,
    prefill_m_values=(2, 8, 32, 128, 512),
    decode_m_values=(1,),
):
    """Validate both phases and create a runtime-only dispatch sidecar."""

    if not isinstance(selection, BackendSelection):
        selection = BackendSelection(**dict(selection))
    weights = {
        tensor.weight_name: plan.weights[tensor.weight_name]
        for unit in plan.units
        for tensor in unit.tensors
        if tensor.backend
    }
    phase_backends = {}
    workspace_bytes = 0
    errors = []
    for phase, backend_name, m_values in (
        ("prefill", selection.prefill, prefill_m_values),
        ("decode", selection.decode, decode_m_values),
    ):
        mapping = {}
        for name, weight in sorted(weights.items()):
            mapping[name] = backend_name
            for m in m_values:
                result = qualify_backend(
                    weight, backend_name, m, device=device
                )
                if not result.supported:
                    errors.append(
                        "{} {} M={} N={} K={}: {}".format(
                            phase,
                            name,
                            m,
                            result.n,
                            result.k,
                            result.unsupported_reason,
                        )
                    )
                workspace_bytes = max(
                    workspace_bytes, result.workspace_bytes
                )
        phase_backends[phase] = mapping
    if errors:
        raise BackendUnavailableError(
            "backend phase plan qualification failed:\n- "
            + "\n- ".join(errors)
        )
    if workspace_bytes > int(plan.workspace_bytes):
        raise ValueError(
            "phase backends require {} workspace bytes but ExecutionPlan "
            "reserved {}; build the frozen plan with the largest-workspace "
            "explicit backend".format(workspace_bytes, plan.workspace_bytes)
        )
    return BackendPhasePlan(
        selection=selection,
        prefill_backends=phase_backends["prefill"],
        decode_backends=phase_backends["decode"],
        workspace_bytes=workspace_bytes,
        device_sm=_sm_for_device(device),
    )
