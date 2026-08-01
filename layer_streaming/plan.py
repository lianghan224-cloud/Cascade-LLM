"""Static BF16 weight layouts for Llama-family checkpoints."""

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
    MATRIX_GROUP = "matrix_group"
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
class HostPlacement:
    """CPU-only tensor placement inside the shared packed host arena."""

    tensor: TensorSpec
    host_offset_elements: int


@dataclass(frozen=True)
class VocabPlan:
    """Row-contiguous CPU vocabulary layout shared by lookup and projection."""

    embedding_key: str
    lm_head_key: str
    vocab_size: int
    hidden_size: int
    chunk_rows: int
    chunk_elements: int
    chunk_bytes: int
    chunk_count: int
    stream_embedding: bool = True
    stream_lm_head: bool = True


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
    host_only: Tuple[HostPlacement, ...] = ()
    vocab: Optional[VocabPlan] = None
    geometry: object = None

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
        for placement in self.host_only:
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
            "vocab": (
                {
                    "embedding_key": self.vocab.embedding_key,
                    "lm_head_key": self.vocab.lm_head_key,
                    "vocab_size": self.vocab.vocab_size,
                    "hidden_size": self.vocab.hidden_size,
                    "chunk_rows": self.vocab.chunk_rows,
                    "chunk_bytes": self.vocab.chunk_bytes,
                    "chunk_count": self.vocab.chunk_count,
                    "stream_embedding": self.vocab.stream_embedding,
                    "stream_lm_head": self.vocab.stream_lm_head,
                }
                if self.vocab is not None
                else None
            ),
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
            "host_only": [
                {
                    "key": placement.tensor.key,
                    "shape": list(placement.tensor.shape),
                    "bytes": placement.tensor.nbytes,
                    "host_offset_bytes": (
                        placement.host_offset_elements * self.dtype_bytes
                    ),
                }
                for placement in self.host_only
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


def build_llama_plan(
    geometry,
    granularity=Granularity.MATRIX,
    embedding_mode="resident",
    lm_head_mode="resident",
    vocab_chunk_bytes=128 * MIB,
):
    """Build a packed BF16 plan from a validated Llama geometry."""

    granularity = Granularity(granularity)
    embedding_mode = str(getattr(embedding_mode, "value", embedding_mode))
    lm_head_mode = str(getattr(lm_head_mode, "value", lm_head_mode))
    valid_modes = {"resident", "streamed"}
    if embedding_mode not in valid_modes:
        raise ValueError("embedding_mode must be resident or streamed")
    if lm_head_mode not in valid_modes:
        raise ValueError("lm_head_mode must be resident or streamed")

    hidden = int(geometry.hidden_size)
    intermediate = int(geometry.intermediate_size)
    kv = int(geometry.num_key_value_heads) * int(geometry.head_dim)
    vocab = int(geometry.vocab_size)
    layers = int(geometry.num_hidden_layers)
    tie_word_embeddings = bool(geometry.tie_word_embeddings)
    stream_embedding = embedding_mode == "streamed"
    stream_lm_head = lm_head_mode == "streamed"
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
        resident=not stream_embedding,
    )
    if tie_word_embeddings:
        add(
            "lm_head.weight",
            (vocab, hidden),
            resident=not stream_lm_head,
            alias_of=embedding.key,
        )
    else:
        add(
            "lm_head.weight",
            (vocab, hidden),
            resident=not stream_lm_head,
        )

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
        elif granularity == Granularity.MATRIX:
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
        else:
            matrix_groups = (
                ("qkv", matrices[0:3]),
                ("o_proj", matrices[3:4]),
                ("gate_up", matrices[4:6]),
                ("down_proj", matrices[6:7]),
            )
            for operation, group in matrix_groups:
                unit, host_cursor = _place_unit(
                    "layer_{:02d}.{}".format(layer_index, operation),
                    layer_index,
                    operation,
                    group,
                    host_cursor,
                )
                units.append(unit)

    alignment_elements = ALIGNMENT_BYTES // BF16_BYTES
    resident_specs = []
    host_only_specs = []
    if tie_word_embeddings:
        if stream_embedding and stream_lm_head:
            host_only_specs.append(embedding)
        else:
            resident_specs.append(embedding)
    else:
        if stream_embedding:
            host_only_specs.append(embedding)
        else:
            resident_specs.append(embedding)
        lm_head = specs["lm_head.weight"]
        if stream_lm_head:
            host_only_specs.append(lm_head)
        else:
            resident_specs.append(lm_head)
    resident_specs.extend(layer_norms)
    resident_specs.append(final_norm)
    host_only = []
    for tensor in host_only_specs:
        host_cursor = _align(host_cursor, alignment_elements)
        host_only.append(
            HostPlacement(
                tensor=tensor,
                host_offset_elements=host_cursor,
            )
        )
        host_cursor += tensor.numel
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
    vocab_plan = None
    if stream_embedding or stream_lm_head:
        row_bytes = hidden * BF16_BYTES
        requested_chunk_bytes = int(vocab_chunk_bytes)
        if requested_chunk_bytes < row_bytes:
            raise ValueError("vocab chunk must hold at least one row")
        chunk_rows = min(vocab, requested_chunk_bytes // row_bytes)
        chunk_elements = chunk_rows * hidden
        chunk_bytes = chunk_elements * BF16_BYTES
        chunk_count = (vocab + chunk_rows - 1) // chunk_rows
        vocab_plan = VocabPlan(
            embedding_key=embedding.key,
            lm_head_key="lm_head.weight",
            vocab_size=vocab,
            hidden_size=hidden,
            chunk_rows=chunk_rows,
            chunk_elements=chunk_elements,
            chunk_bytes=chunk_bytes,
            chunk_count=chunk_count,
            stream_embedding=stream_embedding,
            stream_lm_head=stream_lm_head,
        )
    slot_elements = max(unit.elements for unit in units)
    if vocab_plan is not None and vocab_plan.stream_lm_head:
        slot_elements = max(slot_elements, vocab_plan.chunk_elements)
    return ModelPlan(
        model_id=str(geometry.model_id),
        granularity=granularity,
        tensors=specs,
        units=tuple(units),
        resident=tuple(resident),
        aliases=aliases,
        host_arena_elements=_align(host_cursor, alignment_elements),
        resident_arena_elements=_align(device_cursor, alignment_elements),
        slot_elements=slot_elements,
        host_only=tuple(host_only),
        vocab=vocab_plan,
        geometry=geometry,
    )
