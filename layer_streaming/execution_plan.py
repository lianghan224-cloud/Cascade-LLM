"""Byte-addressed, mixed-dtype execution plans."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .backends import backend_for_weight
from .plan import Granularity
from .specs import DTYPE_BYTES, WeightSpec, align_up


EXECUTION_PLAN_SCHEMA_VERSION = 1


def storage_region_name(spec):
    if spec.role == "scale":
        return "scale_{}".format(spec.storage_dtype)
    if spec.role == "zero_point":
        return "zero_point_{}".format(spec.storage_dtype)
    if spec.quantization is not None and spec.quantization.bits == 4:
        return "int4_packed"
    return spec.storage_dtype


@dataclass(frozen=True)
class WeightRegionPlan:
    name: str
    storage_dtype: str
    bytes: int
    alignment: int

    def as_dict(self):
        return {
            "name": self.name,
            "storage_dtype": self.storage_dtype,
            "bytes": self.bytes,
            "alignment": self.alignment,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class TransferTensor:
    weight_name: str
    storage_region: str
    host_offset: int
    storage_bytes: int
    device_offset: int
    backend: str
    quant_param_names: Tuple[str, ...] = ()

    def as_dict(self):
        return {
            "weight_name": self.weight_name,
            "storage_region": self.storage_region,
            "host_offset": self.host_offset,
            "storage_bytes": self.storage_bytes,
            "device_offset": self.device_offset,
            "backend": self.backend,
            "quant_param_names": list(self.quant_param_names),
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["quant_param_names"] = tuple(
            value.get("quant_param_names", ())
        )
        return cls(**value)


@dataclass(frozen=True)
class TransferUnit:
    unit_id: str
    layer_id: int
    operation: str
    tensors: Tuple[TransferTensor, ...]
    transfer_bytes: int
    workspace_bytes: int

    @property
    def layer_index(self):
        return self.layer_id

    @property
    def nbytes(self):
        return self.transfer_bytes

    def as_dict(self):
        return {
            "unit_id": self.unit_id,
            "layer_id": self.layer_id,
            "operation": self.operation,
            "tensors": [tensor.as_dict() for tensor in self.tensors],
            "transfer_bytes": self.transfer_bytes,
            "workspace_bytes": self.workspace_bytes,
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value.pop("backends", None)
        value["tensors"] = tuple(
            TransferTensor.from_dict(item)
            for item in value.get("tensors", ())
        )
        return cls(**value)


@dataclass(frozen=True)
class ResidentTensor:
    weight_name: str
    storage_region: str
    host_offset: int
    storage_bytes: int
    device_offset: int
    backend: str
    alias_of: Optional[str] = None

    def as_dict(self):
        return {
            "weight_name": self.weight_name,
            "storage_region": self.storage_region,
            "host_offset": self.host_offset,
            "storage_bytes": self.storage_bytes,
            "device_offset": self.device_offset,
            "backend": self.backend,
            "alias_of": self.alias_of,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class VocabPlacement:
    embedding_name: str
    lm_head_name: str
    embedding_mode: str
    lm_head_mode: str
    embedding_dtype: str
    lm_head_dtype: str
    chunk_rows: int
    chunk_bytes: int

    @property
    def stream_embedding(self):
        return self.embedding_mode == "streamed"

    @property
    def stream_lm_head(self):
        return self.lm_head_mode == "streamed"

    def as_dict(self):
        return {
            "embedding_name": self.embedding_name,
            "lm_head_name": self.lm_head_name,
            "embedding_mode": self.embedding_mode,
            "lm_head_mode": self.lm_head_mode,
            "embedding_dtype": self.embedding_dtype,
            "lm_head_dtype": self.lm_head_dtype,
            "chunk_rows": self.chunk_rows,
            "chunk_bytes": self.chunk_bytes,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class ExecutionPlan:
    model_id: str
    geometry: object
    granularity: Granularity
    weights: Mapping[str, WeightSpec]
    regions: Mapping[str, WeightRegionPlan]
    host_offsets: Mapping[str, int]
    units: Tuple[TransferUnit, ...]
    resident: Tuple[ResidentTensor, ...]
    aliases: Mapping[str, str]
    slot_bytes: int
    workspace_bytes: int
    vocab: VocabPlacement
    slot_count: int = 2

    @property
    def host_arena_bytes(self):
        return sum(region.bytes for region in self.regions.values())

    @property
    def resident_bytes(self):
        return max(
            (
                item.device_offset + item.storage_bytes
                for item in self.resident
            ),
            default=0,
        )

    @property
    def stream_bytes_per_token(self):
        return sum(unit.transfer_bytes for unit in self.units)

    @property
    def shared_weight_savings_bytes(self):
        return sum(
            self.weights[target].storage_nbytes
            for target in self.aliases.values()
            if target in self.weights
        )

    def as_dict(self):
        return {
            "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
            "model_id": self.model_id,
            "geometry": self.geometry.as_dict(),
            "granularity": self.granularity.value,
            "slot_count": self.slot_count,
            "slot_bytes": self.slot_bytes,
            "workspace_bytes": self.workspace_bytes,
            "host_arena_bytes": self.host_arena_bytes,
            "resident_bytes": self.resident_bytes,
            "stream_bytes_per_token": self.stream_bytes_per_token,
            "shared_weight_savings_bytes": self.shared_weight_savings_bytes,
            "regions": {
                name: region.as_dict()
                for name, region in sorted(self.regions.items())
            },
            "weights": {
                name: weight.as_dict()
                for name, weight in sorted(self.weights.items())
            },
            "host_offsets": {
                name: int(offset)
                for name, offset in sorted(self.host_offsets.items())
            },
            "aliases": dict(self.aliases),
            "vocab": self.vocab.as_dict(),
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "layer_id": unit.layer_id,
                    "operation": unit.operation,
                    "transfer_bytes": unit.transfer_bytes,
                    "workspace_bytes": unit.workspace_bytes,
                    "backends": sorted(
                        {
                            tensor.backend
                            for tensor in unit.tensors
                            if tensor.backend
                        }
                    ),
                    "tensors": [
                        tensor.as_dict() for tensor in unit.tensors
                    ],
                }
                for unit in self.units
            ],
            "resident": [
                item.as_dict() for item in self.resident
            ],
        }

    @classmethod
    def from_dict(cls, value):
        from .adapter import ModelGeometry

        value = dict(value)
        version = int(value.pop("schema_version", 0))
        if version != EXECUTION_PLAN_SCHEMA_VERSION:
            raise ValueError(
                "unsupported ExecutionPlan schema version {}".format(version)
            )
        for key in (
            "host_arena_bytes",
            "resident_bytes",
            "stream_bytes_per_token",
            "shared_weight_savings_bytes",
        ):
            value.pop(key, None)
        value["geometry"] = ModelGeometry.from_dict(value["geometry"])
        value["granularity"] = Granularity(value["granularity"])
        value["weights"] = {
            name: WeightSpec.from_dict(item)
            for name, item in value["weights"].items()
        }
        value["regions"] = {
            name: WeightRegionPlan.from_dict(item)
            for name, item in value["regions"].items()
        }
        value["host_offsets"] = {
            name: int(offset)
            for name, offset in value["host_offsets"].items()
        }
        value["units"] = tuple(
            TransferUnit.from_dict(item) for item in value["units"]
        )
        value["resident"] = tuple(
            ResidentTensor.from_dict(item)
            for item in value.get("resident", ())
        )
        value["aliases"] = dict(value["aliases"])
        value["vocab"] = VocabPlacement.from_dict(value["vocab"])
        return cls(**value)


def _operation_for(spec):
    return spec.name.rsplit(".", 2)[-2]


def _layer_for(spec):
    parts = spec.name.split(".")
    if len(parts) < 4 or parts[0:2] != ["model", "layers"]:
        return None
    return int(parts[2])


def _build_region_layout(specs):
    cursors = {}
    offsets = {}
    region_dtypes = {}
    region_alignments = {}
    for spec in specs:
        if spec.alias_of is not None:
            continue
        region = storage_region_name(spec)
        cursor = align_up(cursors.get(region, 0), spec.alignment)
        if cursor > (1 << 63) - 1:
            raise OverflowError("host offset exceeds signed 64-bit range")
        offsets[spec.name] = cursor
        cursors[region] = cursor + int(spec.storage_nbytes)
        region_dtypes[region] = spec.storage_dtype
        region_alignments[region] = max(
            region_alignments.get(region, 1), spec.alignment
        )
    regions = {
        name: WeightRegionPlan(
            name=name,
            storage_dtype=region_dtypes[name],
            bytes=align_up(size, region_alignments[name]),
            alignment=region_alignments[name],
        )
        for name, size in cursors.items()
    }
    return offsets, regions


def build_execution_plan(geometry, weight_specs, policy):
    """Build the generic mixed-dtype plan without allocating model weights."""

    granularity = Granularity(policy.granularity)
    specs = tuple(weight_specs)
    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ValueError("weight specifications contain duplicate names")
    aliases = {
        spec.name: spec.alias_of for spec in specs if spec.alias_of is not None
    }
    for alias, target in aliases.items():
        if target not in by_name:
            raise ValueError("alias {} refers to missing {}".format(alias, target))
    host_offsets, regions = _build_region_layout(specs)

    projection_specs = [
        spec
        for spec in specs
        if spec.role
        in {
            "attention_q",
            "attention_k",
            "attention_v",
            "attention_o",
            "mlp_gate",
            "mlp_up",
            "mlp_down",
        }
    ]
    matrix_by_layer = {}
    for spec in projection_specs:
        layer = _layer_for(spec)
        if layer is None:
            raise ValueError("{} has no Llama layer id".format(spec.name))
        matrix_by_layer.setdefault(layer, {})[_operation_for(spec)] = spec
    expected_operations = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    for layer in range(geometry.num_hidden_layers):
        missing = set(expected_operations).difference(matrix_by_layer.get(layer, {}))
        if missing:
            raise ValueError(
                "layer {} is missing projections {}".format(
                    layer, ", ".join(sorted(missing))
                )
            )

    if granularity == Granularity.MATRIX:
        groups = tuple((name, (name,)) for name in expected_operations)
    elif granularity == Granularity.MATRIX_GROUP:
        groups = (
            ("qkv", ("q_proj", "k_proj", "v_proj")),
            ("o_proj", ("o_proj",)),
            ("gate_up", ("gate_proj", "up_proj")),
            ("down_proj", ("down_proj",)),
        )
    else:
        groups = (("layer", expected_operations),)

    def selected_backend(weight):
        return backend_for_weight(
            weight,
            backend_name=getattr(policy, "linear_backend", None),
        )

    units = []
    for layer in range(geometry.num_hidden_layers):
        for operation, member_names in groups:
            device_cursor = 0
            tensors = []
            workspace_bytes = 0
            for member_name in member_names:
                weight = matrix_by_layer[layer][member_name]
                backend = selected_backend(weight)
                quant_names = []
                if weight.quantization is not None:
                    quant_names.append(weight.name + "_scale")
                    if weight.quantization.zero_point:
                        quant_names.append(weight.name + "_zero_point")
                piece_specs = [weight] + [by_name[name] for name in quant_names]
                workspace_bytes = max(
                    workspace_bytes,
                    backend.workspace_bytes(weight, batch_tokens=1),
                )
                for piece in piece_specs:
                    device_cursor = align_up(device_cursor, piece.alignment)
                    tensors.append(
                        TransferTensor(
                            weight_name=piece.name,
                            storage_region=storage_region_name(piece),
                            host_offset=host_offsets[piece.name],
                            storage_bytes=piece.storage_nbytes,
                            device_offset=device_cursor,
                            backend=backend.name if piece is weight else "",
                            quant_param_names=(
                                tuple(quant_names) if piece is weight else ()
                            ),
                        )
                    )
                    device_cursor += piece.storage_nbytes
            transfer_bytes = align_up(device_cursor, 256)
            units.append(
                TransferUnit(
                    unit_id="layer_{:04d}.{}".format(layer, operation),
                    layer_id=layer,
                    operation=operation,
                    tensors=tuple(tensors),
                    transfer_bytes=transfer_bytes,
                    workspace_bytes=workspace_bytes,
                )
            )

    resident_names = {
        spec.name for spec in specs if spec.role == "norm" and spec.alias_of is None
    }
    embedding = by_name["model.embed_tokens.weight"]
    lm_head = by_name["lm_head.weight"]
    if policy.embedding_mode.value == "resident":
        resident_names.add(embedding.name)
    if policy.lm_head_mode.value == "resident":
        resident_names.add(lm_head.alias_of or lm_head.name)
    resident = []
    resident_cursor = 0
    for name in sorted(resident_names):
        spec = by_name[name]
        resident_cursor = align_up(resident_cursor, spec.alignment)
        backend_name = ""
        if spec.role not in {"norm", "embedding", "lm_head"}:
            backend_name = selected_backend(spec).name
        resident.append(
            ResidentTensor(
                weight_name=name,
                storage_region=storage_region_name(spec),
                host_offset=host_offsets[name],
                storage_bytes=spec.storage_nbytes,
                device_offset=resident_cursor,
                backend=backend_name,
            )
        )
        resident_cursor += spec.storage_nbytes

    chunk_row_bytes = (
        geometry.hidden_size * DTYPE_BYTES[lm_head.storage_dtype]
    )
    if int(policy.vocab_chunk_bytes) < chunk_row_bytes:
        raise ValueError("vocab chunk must hold at least one LM Head row")
    chunk_rows = min(
        geometry.vocab_size,
        int(policy.vocab_chunk_bytes) // chunk_row_bytes,
    )
    vocab = VocabPlacement(
        embedding_name=embedding.name,
        lm_head_name=lm_head.name,
        embedding_mode=policy.embedding_mode.value,
        lm_head_mode=policy.lm_head_mode.value,
        embedding_dtype=embedding.compute_dtype,
        lm_head_dtype=lm_head.compute_dtype,
        chunk_rows=chunk_rows,
        chunk_bytes=chunk_rows * chunk_row_bytes,
    )
    slot_bytes = max(unit.transfer_bytes for unit in units)
    if policy.lm_head_mode.value == "streamed":
        slot_bytes = max(slot_bytes, vocab.chunk_bytes)
    return ExecutionPlan(
        model_id=geometry.model_id,
        geometry=geometry,
        granularity=granularity,
        weights=by_name,
        regions=regions,
        host_offsets=host_offsets,
        units=tuple(units),
        resident=tuple(resident),
        aliases=aliases,
        slot_bytes=slot_bytes,
        workspace_bytes=max(unit.workspace_bytes for unit in units),
        vocab=vocab,
        slot_count=int(policy.slot_count),
    )
