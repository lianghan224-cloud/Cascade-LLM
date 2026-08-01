"""Linear execution backends and reference quantized fallbacks."""

from dataclasses import dataclass
import importlib
import importlib.util
from typing import Dict, Mapping, Protocol

import torch
import torch.nn.functional as F

from .specs import DTYPE_BYTES, WeightSpec, normalize_dtype


_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "int8": torch.int8,
    "uint8": torch.uint8,
}


class BackendUnavailableError(RuntimeError):
    pass


class LinearBackend(Protocol):
    name: str
    is_fallback: bool
    info: object

    def validate(self, weight: WeightSpec, input_dtype: str) -> None:
        ...

    def transfer_bytes(self, weight: WeightSpec) -> int:
        ...

    def workspace_bytes(self, weight: WeightSpec, batch_tokens: int) -> int:
        ...

    def execute(self, x, weight_view, quant_views, workspace):
        ...


@dataclass(frozen=True)
class BackendInfo:
    name: str
    storage_dtype: str
    activation_dtype: str
    output_dtype: str
    requires_dequant: bool
    is_fallback: bool
    supported_gpu_architectures: tuple
    atol: float
    rtol: float

    def as_dict(self):
        return {
            "name": self.name,
            "storage_dtype": self.storage_dtype,
            "activation_dtype": self.activation_dtype,
            "output_dtype": self.output_dtype,
            "requires_dequant": self.requires_dequant,
            "is_fallback": self.is_fallback,
            "supported_gpu_architectures": list(
                self.supported_gpu_architectures
            ),
            "atol": self.atol,
            "rtol": self.rtol,
        }


class _DenseLinearBackend:
    is_fallback = False

    def __init__(self, dtype):
        self.dtype = normalize_dtype(dtype)
        self.name = "{}_linear".format(
            "bf16" if self.dtype == "bfloat16" else "fp16"
        )
        self.info = BackendInfo(
            name=self.name,
            storage_dtype=self.dtype,
            activation_dtype=self.dtype,
            output_dtype=self.dtype,
            requires_dequant=False,
            is_fallback=False,
            supported_gpu_architectures=("cuda", "cpu"),
            atol=1e-3 if self.dtype == "bfloat16" else 5e-4,
            rtol=1e-2 if self.dtype == "bfloat16" else 5e-3,
        )

    def validate(self, weight, input_dtype):
        if weight.alignment < 256 or weight.alignment % 256:
            raise ValueError("{} requires 256-byte weight alignment".format(self.name))
        if weight.quantization is not None:
            raise ValueError("{} requires an unquantized weight".format(self.name))
        if weight.storage_dtype != self.dtype:
            raise ValueError(
                "{} requires {} storage, got {}".format(
                    self.name, self.dtype, weight.storage_dtype
                )
            )
        if normalize_dtype(input_dtype) != self.dtype:
            raise ValueError(
                "{} requires {} activation".format(self.name, self.dtype)
            )

    def transfer_bytes(self, weight):
        return int(weight.storage_nbytes)

    def workspace_bytes(self, weight, batch_tokens):
        del weight, batch_tokens
        return 0

    def execute(self, x, weight_view, quant_views=None, workspace=None):
        del workspace
        if quant_views:
            raise ValueError("{} does not accept quantization tensors".format(self.name))
        if x.dtype != _TORCH_DTYPES[self.dtype]:
            raise ValueError("{} activation dtype mismatch".format(self.name))
        if weight_view.dtype != _TORCH_DTYPES[self.dtype]:
            raise ValueError("{} weight dtype mismatch".format(self.name))
        return F.linear(x, weight_view)


class BF16LinearBackend(_DenseLinearBackend):
    def __init__(self):
        super().__init__("bfloat16")


class FP16LinearBackend(_DenseLinearBackend):
    def __init__(self):
        super().__init__("float16")


class _DequantFallbackBackend:
    is_fallback = True

    def __init__(self, bits, compute_dtype):
        self.bits = int(bits)
        self.compute_dtype = normalize_dtype(compute_dtype)
        prefix = "int{}_dequant_{}_fallback".format(
            self.bits,
            "bf16" if self.compute_dtype == "bfloat16" else "fp16",
        )
        self.name = prefix
        self.info = BackendInfo(
            name=prefix,
            storage_dtype="int8" if bits == 8 else "uint8",
            activation_dtype=self.compute_dtype,
            output_dtype=self.compute_dtype,
            requires_dequant=True,
            is_fallback=True,
            supported_gpu_architectures=("cuda", "cpu"),
            atol=2e-2 if self.compute_dtype == "bfloat16" else 1e-2,
            rtol=2e-2 if self.compute_dtype == "bfloat16" else 1e-2,
        )

    def validate(self, weight, input_dtype):
        if weight.alignment < 256 or weight.alignment % 256:
            raise ValueError("{} requires 256-byte weight alignment".format(self.name))
        quant = weight.quantization
        if quant is None or quant.bits != self.bits:
            raise ValueError("{} requires an INT{} weight".format(self.name, self.bits))
        if quant.scheme != "symmetric":
            raise ValueError("{} currently supports symmetric weights".format(self.name))
        if quant.granularity not in {"per_channel", "per_group"}:
            raise ValueError(
                "{} supports per_channel or per_group weights".format(self.name)
            )
        if quant.axis != 1:
            raise ValueError("{} groups the linear input axis (axis=1)".format(self.name))
        if quant.granularity == "per_group" and quant.group_size not in {
            32,
            64,
            128,
        }:
            raise ValueError("{} supports group sizes 32/64/128".format(self.name))
        if quant.scale_dtype not in {"bfloat16", "float16"}:
            raise ValueError("{} scale must be BF16 or FP16".format(self.name))
        if normalize_dtype(input_dtype) != self.compute_dtype:
            raise ValueError(
                "{} requires {} activation".format(self.name, self.compute_dtype)
            )
        if weight.compute_dtype != self.compute_dtype:
            raise ValueError(
                "{} weight compute dtype is {}".format(
                    self.name, weight.compute_dtype
                )
            )

    def transfer_bytes(self, weight):
        return int(weight.storage_nbytes)

    def workspace_bytes(self, weight, batch_tokens):
        del batch_tokens
        return int(
            weight.logical_numel * DTYPE_BYTES[self.compute_dtype]
        )

    def _quantized_values(self, weight_view, logical_numel):
        if self.bits == 8:
            return weight_view.reshape(-1)[:logical_numel]
        return unpack_int4(weight_view, logical_numel)

    def _expanded_scale(self, scale, logical_shape, quant):
        out_features, in_features = logical_shape
        if quant.granularity == "per_channel":
            expected = (out_features, 1)
            if tuple(scale.shape) != expected:
                raise ValueError(
                    "scale shape {} != expected {}".format(
                        tuple(scale.shape), expected
                    )
                )
            return scale
        expected = (out_features, in_features // int(quant.group_size))
        if tuple(scale.shape) != expected:
            raise ValueError(
                "scale shape {} != expected {}".format(
                    tuple(scale.shape), expected
                )
            )
        return scale.repeat_interleave(int(quant.group_size), dim=1)

    def dequantize(self, weight, weight_view, quant_views, workspace=None):
        self.validate(weight, self.compute_dtype)
        quant_views = quant_views or {}
        if "scale" not in quant_views:
            raise ValueError("{} requires quant_views['scale']".format(self.name))
        logical_shape = tuple(weight.logical_shape)
        raw = self._quantized_values(
            weight_view, weight.logical_numel
        ).reshape(logical_shape)
        scale = self._expanded_scale(
            quant_views["scale"], logical_shape, weight.quantization
        )
        dequantized = raw.to(_TORCH_DTYPES[self.compute_dtype]) * scale.to(
            _TORCH_DTYPES[self.compute_dtype]
        )
        if workspace is not None:
            if workspace.dtype != _TORCH_DTYPES[self.compute_dtype]:
                raise ValueError("dequant workspace dtype mismatch")
            if workspace.numel() < weight.logical_numel:
                raise ValueError(
                    "dequant workspace has {} elements, requires {}".format(
                        workspace.numel(), weight.logical_numel
                    )
                )
            destination = workspace.reshape(-1)[: weight.logical_numel].view(
                logical_shape
            )
            destination.copy_(dequantized)
            return destination
        return dequantized

    def execute(self, x, weight_view, quant_views, workspace):
        if x.dtype != _TORCH_DTYPES[self.compute_dtype]:
            raise ValueError("{} activation dtype mismatch".format(self.name))
        weight = quant_views.get("weight_spec") if quant_views else None
        if weight is None:
            raise ValueError(
                "{} requires quant_views['weight_spec']".format(self.name)
            )
        dense_weight = self.dequantize(
            weight,
            weight_view,
            quant_views,
            workspace=workspace,
        )
        return F.linear(x, dense_weight)


class Int8DequantBF16FallbackBackend(_DequantFallbackBackend):
    def __init__(self):
        super().__init__(8, "bfloat16")


class Int8DequantFP16FallbackBackend(_DequantFallbackBackend):
    def __init__(self):
        super().__init__(8, "float16")


class Int4DequantBF16FallbackBackend(_DequantFallbackBackend):
    def __init__(self):
        super().__init__(4, "bfloat16")


class Int4DequantFP16FallbackBackend(_DequantFallbackBackend):
    def __init__(self):
        super().__init__(4, "float16")


def pack_int4(values):
    """Pack signed values (-8..7), low nibble first, into uint8."""

    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    flat = values.reshape(-1).to(torch.int16)
    if flat.numel() and (int(flat.min()) < -8 or int(flat.max()) > 7):
        raise ValueError("INT4 value must be in [-8, 7]")
    encoded = torch.bitwise_and(flat, 0xF).to(torch.uint8)
    if encoded.numel() % 2:
        encoded = torch.cat(
            (encoded, torch.zeros(1, dtype=torch.uint8, device=encoded.device))
        )
    return encoded[0::2] | (encoded[1::2] << 4)


def unpack_int4(packed, count=None):
    """Unpack uint8 into signed values (-8..7), preserving the device."""

    if not torch.is_tensor(packed):
        packed = torch.as_tensor(packed, dtype=torch.uint8)
    if packed.dtype != torch.uint8:
        raise ValueError("packed INT4 storage must be uint8")
    flat = packed.reshape(-1)
    low = torch.bitwise_and(flat, 0xF)
    high = torch.bitwise_and(flat >> 4, 0xF)
    decoded = torch.stack((low, high), dim=1).reshape(-1).to(torch.int8)
    decoded = torch.where(decoded >= 8, decoded - 16, decoded)
    if count is not None:
        count = int(count)
        if count < 0 or count > decoded.numel():
            raise ValueError(
                "requested {} INT4 values from {} packed values".format(
                    count, decoded.numel()
                )
            )
        decoded = decoded[:count]
    return decoded


class _UnavailableFusedBackend:
    is_fallback = False

    def __init__(
        self,
        name,
        storage_dtype,
        activation_dtype,
        output_dtype,
    ):
        self.name = name
        self.info = BackendInfo(
            name=name,
            storage_dtype=storage_dtype,
            activation_dtype=activation_dtype,
            output_dtype=output_dtype,
            requires_dequant=False,
            is_fallback=False,
            supported_gpu_architectures=(),
            atol=0.0,
            rtol=0.0,
        )

    def _unavailable(self):
        raise BackendUnavailableError(
            "{} is reserved but no fused kernel is registered".format(self.name)
        )

    def validate(self, weight, input_dtype):
        del weight, input_dtype
        self._unavailable()

    def transfer_bytes(self, weight):
        del weight
        self._unavailable()

    def workspace_bytes(self, weight, batch_tokens):
        del weight, batch_tokens
        self._unavailable()

    def execute(self, x, weight_view, quant_views, workspace):
        del x, weight_view, quant_views, workspace
        self._unavailable()


class FusedW8A16Backend(_UnavailableFusedBackend):
    def __init__(self):
        super().__init__("fused_w8a16", "int8", "bfloat16", "bfloat16")


class FusedW8A8Backend(_UnavailableFusedBackend):
    def __init__(self):
        super().__init__("fused_w8a8", "int8", "int8", "bfloat16")


class FusedW4A16Backend(_UnavailableFusedBackend):
    def __init__(self):
        super().__init__("fused_w4a16", "uint8", "bfloat16", "bfloat16")


_REGISTERED_BACKENDS = {}


def register_linear_backend(backend, replace=False):
    """Register an explicit provider without changing checkpoint metadata."""

    name = str(getattr(backend, "name", "")).strip()
    if not name:
        raise ValueError("backend must expose a non-empty name")
    reserved = {
        "fused_w8a16": FusedW8A16Backend,
        "fused_w8a8": FusedW8A8Backend,
        "fused_w4a16": FusedW4A16Backend,
    }
    if name not in reserved:
        raise ValueError(
            "provider registration is limited to frozen fused names: {}".format(
                ", ".join(sorted(reserved))
            )
        )
    for method in (
        "validate",
        "transfer_bytes",
        "workspace_bytes",
        "execute",
    ):
        if not callable(getattr(backend, method, None)):
            raise TypeError(
                "backend {} does not implement {}".format(name, method)
            )
    existing = default_backend_registry().get(name)
    if (
        existing is not None
        and not isinstance(existing, _UnavailableFusedBackend)
        and not replace
    ):
        raise ValueError(
            "backend {} is already registered; pass replace=True explicitly".format(
                name
            )
        )
    if name in _REGISTERED_BACKENDS and not replace:
        raise ValueError("backend {} is already registered".format(name))
    info = getattr(backend, "info", None)
    expected = reserved[name]().info
    if info is None:
        raise TypeError("backend {} must expose BackendInfo".format(name))
    for field_name in (
        "storage_dtype",
        "activation_dtype",
        "output_dtype",
    ):
        if getattr(info, field_name, None) != getattr(expected, field_name):
            raise ValueError(
                "{} {} must be {}, got {}".format(
                    name,
                    field_name,
                    getattr(expected, field_name),
                    getattr(info, field_name, None),
                )
            )
    if bool(getattr(backend, "is_fallback", True)):
        raise ValueError("{} cannot be registered as a fallback".format(name))
    _REGISTERED_BACKENDS[name] = backend
    return backend


def unregister_linear_backend(name):
    """Remove a process-local provider registration."""

    return _REGISTERED_BACKENDS.pop(str(name), None)


def default_backend_registry():
    backends = (
        BF16LinearBackend(),
        FP16LinearBackend(),
        Int8DequantBF16FallbackBackend(),
        Int8DequantFP16FallbackBackend(),
        Int4DequantBF16FallbackBackend(),
        Int4DequantFP16FallbackBackend(),
        FusedW8A16Backend(),
        FusedW8A8Backend(),
        FusedW4A16Backend(),
    )
    result = {backend.name: backend for backend in backends}
    result.update(_REGISTERED_BACKENDS)
    return result


def backend_for_weight(weight, registry=None, backend_name=None):
    if backend_name is not None:
        name = str(backend_name)
    elif weight.quantization is None:
        name = "{}_linear".format(
            "bf16" if weight.compute_dtype == "bfloat16" else "fp16"
        )
    else:
        name = "int{}_dequant_{}_fallback".format(
            weight.quantization.bits,
            "bf16" if weight.compute_dtype == "bfloat16" else "fp16",
        )
    registry = registry or default_backend_registry()
    if name not in registry:
        raise BackendUnavailableError("linear backend {} is not registered".format(name))
    backend = registry[name]
    backend.validate(weight, weight.compute_dtype)
    return backend


def backend_capabilities(device=None):
    """Describe runnable backends and unavailable fused provider candidates."""

    registry = default_backend_registry()
    capability = None
    if (
        device is not None
        and torch.cuda.is_available()
        and torch.device(device).type == "cuda"
    ):
        capability = list(torch.cuda.get_device_capability(device))
    packages = {}
    for package_name in (
        "torchao",
        "bitsandbytes",
        "awq",
        "auto_gptq",
        "vllm",
        "flashinfer",
        "triton",
    ):
        discoverable = bool(importlib.util.find_spec(package_name))
        import_error = None
        if discoverable:
            try:
                importlib.import_module(package_name)
            except BaseException as error:
                import_error = "{}: {}".format(type(error).__name__, error)
        packages[package_name] = {
            "discoverable": discoverable,
            "importable": discoverable and import_error is None,
            "import_error": import_error,
        }
    primitives = {
        name: bool(hasattr(torch.ops.aten, name))
        for name in ("_int_mm", "_scaled_mm", "_weight_int8pack_mm")
    }
    primitive_probes = {}
    if capability is not None:
        probe_device = torch.device(device)

        def probe(name, callback):
            try:
                callback()
                torch.cuda.synchronize(probe_device)
                primitive_probes[name] = {
                    "available": True,
                    "error": None,
                }
            except BaseException as error:
                primitive_probes[name] = {
                    "available": False,
                    "error": "{}: {}".format(
                        type(error).__name__,
                        str(error).splitlines()[0],
                    )[:500],
                }

        probe(
            "weight_int8pack_mm_w8a16_decode",
            lambda: torch.ops.aten._weight_int8pack_mm(
                torch.zeros(
                    (1, 32), dtype=torch.bfloat16, device=probe_device
                ),
                torch.zeros(
                    (16, 32), dtype=torch.int8, device=probe_device
                ),
                torch.ones(
                    (16,), dtype=torch.bfloat16, device=probe_device
                ),
            ),
        )
        probe(
            "int_mm_w8a8_decode_m1",
            lambda: torch.ops.aten._int_mm(
                torch.zeros(
                    (1, 32), dtype=torch.int8, device=probe_device
                ),
                torch.zeros(
                    (32, 16), dtype=torch.int8, device=probe_device
                ),
            ),
        )
    result = {}
    for name, backend in sorted(registry.items()):
        unavailable = isinstance(backend, _UnavailableFusedBackend)
        capability_payload = None
        capability_reason = None
        if not unavailable:
            from .backend_capability import capability_for_backend

            backend_capability = capability_for_backend(backend)
            capability_payload = backend_capability.as_dict()
            if capability is not None and backend_capability.supported_sms:
                device_sm = int(capability[0]) * 10 + int(capability[1])
                if device_sm not in backend_capability.supported_sms:
                    capability_reason = (
                        "device SM{} is not in provider supported SMs {}".format(
                            device_sm, backend_capability.supported_sms
                        )
                    )
        result[name] = {
            "available": not unavailable and capability_reason is None,
            "provider": (
                "builtin"
                if not unavailable and name not in _REGISTERED_BACKENDS
                else (
                    type(backend).__module__
                    if name in _REGISTERED_BACKENDS
                    else None
                )
            ),
            "is_fallback": bool(backend.is_fallback),
            "reason": (
                "no fused provider is registered; fallback is intentionally "
                "not selected"
                if unavailable
                else capability_reason
            ),
            "compute_capability": capability,
            "capability": capability_payload,
            "environment": {
                "packages": packages,
                "torch_aten_primitives": primitives,
                "torch_aten_runtime_probes": primitive_probes,
            } if name.startswith("fused_") else {},
        }
    return result
