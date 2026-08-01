"""Common model weight and quantization metadata.

The classes in this module describe checkpoint storage.  They intentionally do
not own tensors so they can also be used by metadata-only memory planning.
"""

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Optional, Tuple


DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
}

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float": "float32",
    "fp64": "float64",
}

WEIGHT_ROLES = {
    "embedding",
    "lm_head",
    "norm",
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "scale",
    "zero_point",
}


def normalize_dtype(dtype):
    """Return a stable checkpoint dtype name."""

    value = str(getattr(dtype, "value", dtype)).lower()
    if value.startswith("torch."):
        value = value[6:]
    value = _DTYPE_ALIASES.get(value, value)
    if value not in DTYPE_BYTES:
        raise ValueError("unsupported dtype {!r}".format(dtype))
    return value


def shape_numel(shape):
    result = reduce(mul, (int(item) for item in shape), 1)
    if result < 0:
        raise ValueError("shape cannot have a negative element")
    return result


def align_up(value, alignment):
    value = int(value)
    alignment = int(alignment)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class QuantizationSpec:
    bits: int
    scheme: str = "symmetric"
    granularity: str = "per_channel"
    group_size: Optional[int] = None
    scale_dtype: str = "bfloat16"
    zero_point: bool = False
    zero_point_dtype: Optional[str] = None
    packing: Optional[str] = None
    axis: int = 1

    def __post_init__(self):
        bits = int(self.bits)
        scheme = str(self.scheme)
        granularity = str(self.granularity)
        scale_dtype = normalize_dtype(self.scale_dtype)
        packing = self.packing
        if bits not in {4, 8}:
            raise ValueError("quantization bits must be 4 or 8")
        if scheme not in {"symmetric", "asymmetric"}:
            raise ValueError("scheme must be symmetric or asymmetric")
        if granularity not in {"per_tensor", "per_channel", "per_group"}:
            raise ValueError(
                "granularity must be per_tensor, per_channel, or per_group"
            )
        if scale_dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("scale dtype must be BF16, FP16, or FP32")
        if granularity == "per_group":
            if self.group_size is None or int(self.group_size) <= 0:
                raise ValueError("per_group quantization requires group_size")
            group_size = int(self.group_size)
        else:
            if self.group_size is not None:
                raise ValueError(
                    "{} quantization cannot set group_size".format(granularity)
                )
            group_size = None
        if scheme == "symmetric" and self.zero_point:
            raise ValueError("symmetric quantization cannot use a zero-point")
        if scheme == "asymmetric" and not self.zero_point:
            raise ValueError("asymmetric quantization requires a zero-point")
        if self.zero_point:
            if self.zero_point_dtype is None:
                raise ValueError("zero-point dtype is required")
            zero_point_dtype = normalize_dtype(self.zero_point_dtype)
            if zero_point_dtype not in {"int8", "uint8"}:
                raise ValueError("zero-point dtype must be int8 or uint8")
        else:
            if self.zero_point_dtype is not None:
                raise ValueError(
                    "zero_point_dtype cannot be set when zero_point is false"
                )
            zero_point_dtype = None
        if bits == 4:
            packing = packing or "int4_pair_uint8"
            if packing != "int4_pair_uint8":
                raise ValueError("INT4 requires int4_pair_uint8 packing")
        elif packing not in {None, "none"}:
            raise ValueError("INT8 does not support packed storage")
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "granularity", granularity)
        object.__setattr__(self, "group_size", group_size)
        object.__setattr__(self, "scale_dtype", scale_dtype)
        object.__setattr__(self, "zero_point_dtype", zero_point_dtype)
        object.__setattr__(self, "packing", packing)
        object.__setattr__(self, "axis", int(self.axis))

    def validate_logical_shape(self, logical_shape):
        logical_shape = tuple(int(item) for item in logical_shape)
        if len(logical_shape) != 2:
            raise ValueError("quantized linear weight must be rank 2")
        axis = self.axis
        if axis < 0:
            axis += len(logical_shape)
        if axis not in {0, 1}:
            raise ValueError("quantization axis must address a rank-2 dimension")
        if self.granularity == "per_group":
            target = logical_shape[axis]
            if target % int(self.group_size):
                raise ValueError(
                    "group_size {} does not divide axis {} length {}".format(
                        self.group_size, axis, target
                    )
                )
        return logical_shape

    def scale_shape(self, logical_shape):
        logical_shape = self.validate_logical_shape(logical_shape)
        if self.granularity == "per_tensor":
            return (1,)
        if self.granularity == "per_channel":
            # Linear output channels are rows.  Axis selects the grouped input
            # dimension for per-group quantization, not the channel dimension.
            return (logical_shape[0], 1)
        axis = self.axis if self.axis >= 0 else self.axis + 2
        if axis == 1:
            return (logical_shape[0], logical_shape[1] // int(self.group_size))
        return (logical_shape[0] // int(self.group_size), logical_shape[1])

    def storage_shape(self, logical_shape):
        logical_shape = self.validate_logical_shape(logical_shape)
        if self.bits == 8:
            return logical_shape
        return ((shape_numel(logical_shape) + 1) // 2,)

    def as_dict(self):
        return {
            "bits": self.bits,
            "scheme": self.scheme,
            "granularity": self.granularity,
            "group_size": self.group_size,
            "scale_dtype": self.scale_dtype,
            "zero_point": self.zero_point,
            "zero_point_dtype": self.zero_point_dtype,
            "packing": self.packing,
            "axis": self.axis,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class WeightSpec:
    name: str
    logical_shape: Tuple[int, ...]
    storage_shape: Tuple[int, ...]
    storage_dtype: str
    compute_dtype: str
    role: str
    quantization: Optional[QuantizationSpec] = None
    alias_of: Optional[str] = None
    alignment: int = 256

    def __post_init__(self):
        name = str(self.name)
        logical_shape = tuple(int(item) for item in self.logical_shape)
        storage_shape = tuple(int(item) for item in self.storage_shape)
        if not name:
            raise ValueError("weight name cannot be empty")
        if not logical_shape or any(item <= 0 for item in logical_shape):
            raise ValueError("{} has invalid logical shape".format(name))
        if not storage_shape or any(item <= 0 for item in storage_shape):
            raise ValueError("{} has invalid storage shape".format(name))
        storage_dtype = normalize_dtype(self.storage_dtype)
        compute_dtype = normalize_dtype(self.compute_dtype)
        role = str(self.role)
        if role not in WEIGHT_ROLES:
            raise ValueError("{} has unsupported role {!r}".format(name, role))
        alignment = int(self.alignment)
        align_up(0, alignment)
        if self.quantization is None:
            if logical_shape != storage_shape:
                raise ValueError(
                    "{} unquantized logical and storage shapes differ".format(name)
                )
        else:
            expected = self.quantization.storage_shape(logical_shape)
            if storage_shape != expected:
                raise ValueError(
                    "{} storage shape {} != expected {}".format(
                        name, storage_shape, expected
                    )
                )
            expected_dtype = "int8" if self.quantization.bits == 8 else "uint8"
            if storage_dtype != expected_dtype:
                raise ValueError(
                    "{} quantized storage dtype must be {}".format(
                        name, expected_dtype
                    )
                )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "logical_shape", logical_shape)
        object.__setattr__(self, "storage_shape", storage_shape)
        object.__setattr__(self, "storage_dtype", storage_dtype)
        object.__setattr__(self, "compute_dtype", compute_dtype)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "alignment", alignment)

    @property
    def key(self):
        """Compatibility name used by the first-generation adapter."""

        return self.name

    @property
    def shape(self):
        """Checkpoint shape compatibility alias."""

        return self.storage_shape

    @property
    def dtype(self):
        """Checkpoint dtype compatibility alias."""

        return self.storage_dtype

    @property
    def logical_numel(self):
        return shape_numel(self.logical_shape)

    @property
    def storage_numel(self):
        return shape_numel(self.storage_shape)

    @property
    def storage_nbytes(self):
        return self.storage_numel * DTYPE_BYTES[self.storage_dtype]

    def as_dict(self):
        return {
            "name": self.name,
            "logical_shape": list(self.logical_shape),
            "storage_shape": list(self.storage_shape),
            "storage_dtype": self.storage_dtype,
            "compute_dtype": self.compute_dtype,
            "role": self.role,
            "quantization": (
                self.quantization.as_dict()
                if self.quantization is not None
                else None
            ),
            "alias_of": self.alias_of,
            "alignment": self.alignment,
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        quantization = value.get("quantization")
        if quantization is not None:
            value["quantization"] = QuantizationSpec.from_dict(quantization)
        return cls(**value)

    @classmethod
    def dense(
        cls,
        name,
        shape,
        dtype,
        role,
        compute_dtype=None,
        alias_of=None,
        alignment=256,
    ):
        dtype = normalize_dtype(dtype)
        return cls(
            name=name,
            logical_shape=tuple(shape),
            storage_shape=tuple(shape),
            storage_dtype=dtype,
            compute_dtype=compute_dtype or dtype,
            role=role,
            alias_of=alias_of,
            alignment=alignment,
        )

    @classmethod
    def quantized(
        cls,
        name,
        logical_shape,
        quantization,
        compute_dtype,
        role,
        alias_of=None,
        alignment=256,
    ):
        return cls(
            name=name,
            logical_shape=tuple(logical_shape),
            storage_shape=quantization.storage_shape(logical_shape),
            storage_dtype="int8" if quantization.bits == 8 else "uint8",
            compute_dtype=compute_dtype,
            role=role,
            quantization=quantization,
            alias_of=alias_of,
            alignment=alignment,
        )
