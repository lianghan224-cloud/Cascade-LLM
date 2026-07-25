"""Static weight layout and transfer plan for Llama-3.1-8B."""

from dataclasses import dataclass
from enum import Enum
from functools import reduce
from operator import mul
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MIB = 1024 * 1024
BF16_BYTES = 2
ALIGNMENT_BYTES = 256


class Granularity(str, Enum):
    """Smallest independently transferred model unit."""

    MATRIX = "matrix"
    LAYER = "layer"


def _product(values):
    return reduce(mul, values, 1)


def _align(value, alignment):
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class TensorSpec:
    key: str
    shape: Tuple[int, ...]
    resident: bool = False
    alias_of: Optional[str] = None

    @property
    def numel(self):
        return _product(self.shape)

    @property
    def nbytes(self):
        return self.numel * BF16_BYTES


@dataclass(frozen=True)
class UnitPiece:
    tensor: TensorSpec
    unit_offset_elements: int


@dataclass(frozen=True)
class TransferUnit:
    unit_id: str
    layer_index: int
    operation: str
    pieces: Tuple[UnitPiece, ...]
    elements: int
    host_offset_elements: int

    @property
    def nbytes(self):
        return self.elements * BF16_BYTES


@dataclass(frozen=True)
class ResidentPlacement:
    tensor: TensorSpec
    host_offset_elements: int
    device_offset_elements: int


@dataclass(frozen=True)
class ModelPlan:
    model_id: str
    granularity: Granularity
    tensors: Mapping[str, TensorSpec]
    units: Tuple[TransferUnit, ...]
    resident: Tuple[ResidentPlacement, ...]
    aliases: Mapping[str, str]
    host_arena_elements: int
    resident_arena_elements: int
    slot_elements: int
    dtype_bytes: int = BF16_BYTES
    slot_count: int = 2

    @property
    def slot_bytes(self):
        return self.slot_elements * self.dtype_bytes

    @property
    def two_slot_bytes(self):
        return self.slot_count * self.slot_bytes

    @property
    def stream_bytes_per_token(self):
        return sum(unit.nbytes for unit in self.units)

    @property
    def resident_bytes(self):
        return self.resident_arena_elements * self.dtype_bytes

    @property
    def host_arena_bytes(self):
        return self.host_arena_elements * self.dtype_bytes

    def tensor_host_offset(self, key):
        for unit in self.units:
            for piece in unit.pieces:
                if piece.tensor.key == key:
                    return unit.host_offset_elements + piece.unit_offset_elements
        for placement in self.resident:
            if placement.tensor.key == key:
                return placement.host_offset_elements
        alias = self.aliases.get(key)
        if alias is not None:
            return self.tensor_host_offset(alias)
        raise KeyError(key)

    def as_dict(self):
        return {
            "model_id": self.model_id,
            "granularity": self.granularity.value,
            "dtype": "torch.bfloat16",
            "dtype_bytes": self.dtype_bytes,
            "transfer_unit_count": len(self.units),
            "stream_bytes_per_token": self.stream_bytes_per_token,
            "slot_count": self.slot_count,
            "slot_bytes": self.slot_bytes,
            "two_slot_bytes": self.two_slot_bytes,
            "resident_bytes": self.resident_bytes,
            "host_arena_bytes": self.host_arena_bytes,
            "aliases": dict(self.aliases),
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "layer_index": unit.layer_index,
                    "operation": unit.operation,
                    "bytes": unit.nbytes,
                    "pieces": [
                        {
                            "key": piece.tensor.key,
                            "shape": list(piece.tensor.shape),
                            "unit_offset_bytes": (
                                piece.unit_offset_elements * self.dtype_bytes
                            ),
                        }
                        for piece in unit.pieces
                    ],
                }
                for unit in self.units
            ],
            "resident": [
                {
                    "key": placement.tensor.key,
                    "shape": list(placement.tensor.shape),
                    "bytes": placement.tensor.nbytes,
                    "device_offset_bytes": (
                        placement.device_offset_elements * self.dtype_bytes
                    ),
                }
                for placement in self.resident
            ],
        }


def _place_unit(
    unit_id,
    layer_index,
    operation,
    tensors,
    host_cursor,
):
    alignment_elements = ALIGNMENT_BYTES // BF16_BYTES
    pieces = []
    unit_cursor = 0
    for tensor in tensors:
        unit_cursor = _align(unit_cursor, alignment_elements)
        pieces.append(UnitPiece(tensor, unit_cursor))
        unit_cursor += tensor.numel
    unit_elements = _align(unit_cursor, alignment_elements)
    host_cursor = _align(host_cursor, alignment_elements)
    unit = TransferUnit(
        unit_id=unit_id,
        layer_index=layer_index,
        operation=operation,
        pieces=tuple(pieces),
        elements=unit_elements,
        host_offset_elements=host_cursor,
    )
    return unit, host_cursor + unit_elements


def build_llama31_8b_plan(
    granularity=Granularity.MATRIX,
    tie_word_embeddings=False,
):
    """Build a packed BF16 plan for the official Llama-3.1-8B geometry.

    The default checkpoint has separate embedding and LM-head tensors. Set
    ``tie_word_embeddings`` only for a compatible checkpoint that omits
    ``lm_head.weight`` and intentionally aliases it to the token embedding.
    """

    granularity = Granularity(granularity)
    hidden = 4096
    intermediate = 14336
    kv = 1024
    vocab = 128256
    layers = 32
    projection_shapes = (
        ("q_proj", (hidden, hidden)),
        ("k_proj", (kv, hidden)),
        ("v_proj", (kv, hidden)),
        ("o_proj", (hidden, hidden)),
        ("gate_proj", (intermediate, hidden)),
        ("up_proj", (intermediate, hidden)),
        ("down_proj", (hidden, intermediate)),
    )

    specs = {}

    def add(key, shape, resident=False, alias_of=None):
        spec = TensorSpec(
            key=key,
            shape=tuple(shape),
            resident=resident,
            alias_of=alias_of,
        )
        specs[key] = spec
        return spec

    embedding = add(
        "model.embed_tokens.weight",
        (vocab, hidden),
        resident=True,
    )
    if tie_word_embeddings:
        add(
            "lm_head.weight",
            (vocab, hidden),
            resident=True,
            alias_of=embedding.key,
        )
    else:
        add("lm_head.weight", (vocab, hidden), resident=True)

    layer_projections = []
    layer_norms = []
    for layer_index in range(layers):
        prefix = "model.layers.{}".format(layer_index)
        matrices = []
        for name, shape in projection_shapes:
            block = "self_attn" if name in {
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            } else "mlp"
            matrices.append(
                add(
                    "{}.{}.{}.weight".format(prefix, block, name),
                    shape,
                )
            )
        layer_projections.append(tuple(matrices))
        layer_norms.extend(
            [
                add(
                    "{}.input_layernorm.weight".format(prefix),
                    (hidden,),
                    resident=True,
                ),
                add(
                    "{}.post_attention_layernorm.weight".format(prefix),
                    (hidden,),
                    resident=True,
                ),
            ]
        )
    final_norm = add("model.norm.weight", (hidden,), resident=True)

    host_cursor = 0
    units = []
    for layer_index, matrices in enumerate(layer_projections):
        if granularity == Granularity.LAYER:
            unit, host_cursor = _place_unit(
                "layer_{:02d}".format(layer_index),
                layer_index,
                "layer",
                matrices,
                host_cursor,
            )
            units.append(unit)
        else:
            for matrix in matrices:
                operation = matrix.key.rsplit(".", 2)[-2]
                unit, host_cursor = _place_unit(
                    "layer_{:02d}.{}".format(layer_index, operation),
                    layer_index,
                    operation,
                    (matrix,),
                    host_cursor,
                )
                units.append(unit)

    alignment_elements = ALIGNMENT_BYTES // BF16_BYTES
    resident_specs = [embedding]
    if not tie_word_embeddings:
        resident_specs.append(specs["lm_head.weight"])
    resident_specs.extend(layer_norms)
    resident_specs.append(final_norm)
    resident = []
    device_cursor = 0
    for tensor in resident_specs:
        host_cursor = _align(host_cursor, alignment_elements)
        device_cursor = _align(device_cursor, alignment_elements)
        resident.append(
            ResidentPlacement(
                tensor=tensor,
                host_offset_elements=host_cursor,
                device_offset_elements=device_cursor,
            )
        )
        host_cursor += tensor.numel
        device_cursor += tensor.numel

    aliases = {}
    if tie_word_embeddings:
        aliases["lm_head.weight"] = embedding.key
    slot_elements = max(unit.elements for unit in units)
    return ModelPlan(
        model_id="meta-llama/Llama-3.1-8B",
        granularity=granularity,
        tensors=specs,
        units=tuple(units),
        resident=tuple(resident),
        aliases=aliases,
        host_arena_elements=_align(host_cursor, alignment_elements),
        resident_arena_elements=_align(device_cursor, alignment_elements),
        slot_elements=slot_elements,
    )
