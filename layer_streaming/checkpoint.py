"""Safetensors checkpoint discovery and metadata validation."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import struct
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .specs import DTYPE_BYTES, WeightSpec, normalize_dtype, shape_numel


_SAFETENSORS_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "U8": "uint8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
}
_MAX_HEADER_BYTES = 100 * 1024 * 1024


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key {!r}".format(key))
        result[key] = value
    return result


@dataclass(frozen=True)
class TensorManifest:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    file: str
    data_offsets: Tuple[int, int]

    @property
    def nbytes(self):
        return int(self.data_offsets[1]) - int(self.data_offsets[0])

    def as_dict(self):
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "file": self.file,
            "data_offsets": list(self.data_offsets),
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["shape"] = tuple(value["shape"])
        value["data_offsets"] = tuple(value["data_offsets"])
        return cls(**value)


@dataclass(frozen=True)
class ManifestMismatch:
    name: str
    actual: object
    expected: object


@dataclass
class ManifestValidation:
    checkpoint: str
    missing: Tuple[str, ...] = ()
    unexpected: Tuple[str, ...] = ()
    shape_mismatches: Tuple[ManifestMismatch, ...] = ()
    dtype_mismatches: Tuple[ManifestMismatch, ...] = ()
    errors: Tuple[str, ...] = ()

    @property
    def ok(self):
        return not (
            self.missing
            or self.unexpected
            or self.shape_mismatches
            or self.dtype_mismatches
            or self.errors
        )

    def format_errors(self):
        sections = []
        if self.missing:
            sections.append("missing weights:\n  " + "\n  ".join(self.missing))
        if self.unexpected:
            sections.append(
                "unexpected weights:\n  " + "\n  ".join(self.unexpected)
            )
        if self.shape_mismatches:
            sections.append(
                "shape conflicts:\n  "
                + "\n  ".join(
                    "{}: actual {} != expected {}".format(
                        item.name, item.actual, item.expected
                    )
                    for item in self.shape_mismatches
                )
            )
        if self.dtype_mismatches:
            sections.append(
                "dtype conflicts:\n  "
                + "\n  ".join(
                    "{}: actual {} != expected {}".format(
                        item.name, item.actual, item.expected
                    )
                    for item in self.dtype_mismatches
                )
            )
        if self.errors:
            sections.append("index/shard errors:\n  " + "\n  ".join(self.errors))
        return "\n".join(sections) if sections else "checkpoint is valid"

    def raise_for_error(self):
        if not self.ok:
            raise ValueError(self.format_errors())
        return self


@dataclass(frozen=True)
class CheckpointManifest:
    root: str
    files: Tuple[str, ...]
    tensors: Dict[str, TensorManifest]
    aliases: Dict[str, str] = field(default_factory=dict)
    total_bytes: int = 0
    errors: Tuple[str, ...] = ()
    weight_map: Dict[str, str] = field(default_factory=dict)

    def as_dict(self):
        return {
            "root": self.root,
            "files": list(self.files),
            "tensors": {
                name: tensor.as_dict()
                for name, tensor in sorted(self.tensors.items())
            },
            "aliases": dict(sorted(self.aliases.items())),
            "total_bytes": self.total_bytes,
            "errors": list(self.errors),
            "weight_map": dict(sorted(self.weight_map.items())),
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["files"] = tuple(value.get("files", ()))
        value["errors"] = tuple(value.get("errors", ()))
        value["tensors"] = {
            name: TensorManifest.from_dict(tensor)
            for name, tensor in value.get("tensors", {}).items()
        }
        value["aliases"] = dict(value.get("aliases", {}))
        value["weight_map"] = dict(value.get("weight_map", {}))
        return cls(**value)

    @classmethod
    def from_path(cls, checkpoint, aliases=None):
        checkpoint = Path(checkpoint)
        root, weight_map, files = _discover_files(checkpoint)
        tensors = {}
        errors = []
        file_headers = {}
        total_bytes = 0
        for filename in files:
            if not filename.is_file():
                errors.append("missing shard file {}".format(filename))
                continue
            total_bytes += int(filename.stat().st_size)
            try:
                header = _read_safetensors_header(filename)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                errors.append("cannot read {}: {}".format(filename, error))
                continue
            file_headers[str(filename.resolve())] = header
            for name, metadata in header.items():
                if name in tensors:
                    errors.append("tensor {} occurs in multiple shards".format(name))
                    continue
                tensors[name] = TensorManifest(
                    name=name,
                    shape=metadata["shape"],
                    dtype=metadata["dtype"],
                    file=str(filename),
                    data_offsets=metadata["data_offsets"],
                )
        if weight_map is not None:
            for name, relative_name in sorted(weight_map.items()):
                filename = str((root / relative_name).resolve())
                header = file_headers.get(filename)
                if header is not None and name not in header:
                    errors.append(
                        "index maps {} to {}, but the shard does not contain it".format(
                            name, relative_name
                        )
                    )
            unindexed = set(tensors).difference(weight_map)
            if unindexed:
                errors.append(
                    "shards contain unindexed tensors: {}".format(
                        ", ".join(sorted(unindexed))
                    )
                )
            indexed_missing = set(weight_map).difference(tensors)
            if indexed_missing:
                errors.append(
                    "index references missing tensors: {}".format(
                        ", ".join(sorted(indexed_missing))
                    )
                )
        alias_table = dict(aliases or {})
        for alias, target in sorted(alias_table.items()):
            if alias == target:
                errors.append("alias {} refers to itself".format(alias))
            if target not in tensors and target not in alias_table:
                errors.append(
                    "alias {} refers to missing tensor {}".format(alias, target)
                )
            seen = set()
            cursor = alias
            while cursor in alias_table:
                if cursor in seen:
                    errors.append("cyclic alias involving {}".format(alias))
                    break
                seen.add(cursor)
                cursor = alias_table[cursor]
        return cls(
            root=str(root),
            files=tuple(str(filename) for filename in files),
            tensors=tensors,
            aliases=alias_table,
            total_bytes=int(total_bytes),
            errors=tuple(errors),
            weight_map=dict(weight_map or {}),
        )

    def validate(self, expected_specs, allow_extra=False):
        expected_specs = tuple(expected_specs)
        expected = {
            spec.name: spec for spec in expected_specs if spec.alias_of is None
        }
        physical_names = [
            spec.name for spec in expected_specs if spec.alias_of is None
        ]
        expected_aliases = {
            spec.name: spec.alias_of
            for spec in expected_specs
            if spec.alias_of is not None
        }
        errors = list(self.errors)
        if len(expected) != len(physical_names):
            errors.append("expected weight specifications contain duplicate names")
        for alias, target in sorted(expected_aliases.items()):
            manifest_target = self.aliases.get(alias, target)
            if manifest_target != target:
                errors.append(
                    "alias {} points to {}, expected {}".format(
                        alias, manifest_target, target
                    )
                )
            if target not in expected and target not in self.tensors:
                errors.append(
                    "alias {} refers to missing tensor {}".format(alias, target)
                )
        common = set(expected).intersection(self.tensors)
        shape_mismatches = []
        dtype_mismatches = []
        for name in sorted(common):
            spec = expected[name]
            tensor = self.tensors[name]
            if tuple(tensor.shape) != tuple(spec.storage_shape):
                shape_mismatches.append(
                    ManifestMismatch(
                        name, tuple(tensor.shape), tuple(spec.storage_shape)
                    )
                )
            if tensor.dtype != spec.storage_dtype:
                dtype_mismatches.append(
                    ManifestMismatch(name, tensor.dtype, spec.storage_dtype)
                )
        unexpected = ()
        if not allow_extra:
            unexpected = tuple(sorted(set(self.tensors).difference(expected)))
        return ManifestValidation(
            checkpoint=self.root,
            missing=tuple(sorted(set(expected).difference(self.tensors))),
            unexpected=unexpected,
            shape_mismatches=tuple(shape_mismatches),
            dtype_mismatches=tuple(dtype_mismatches),
            errors=tuple(errors),
        )

    @property
    def shared_bytes(self):
        saved = 0
        for alias, target in self.aliases.items():
            if alias not in self.tensors and target in self.tensors:
                saved += self.tensors[target].nbytes
        return saved


def _read_safetensors_header(path):
    path = Path(path)
    file_size = int(path.stat().st_size)
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors header")
        header_length = struct.unpack("<Q", prefix)[0]
        if header_length > _MAX_HEADER_BYTES:
            raise ValueError(
                "implausible {} byte safetensors header".format(header_length)
            )
        payload = source.read(header_length)
        if len(payload) != header_length:
            raise ValueError("truncated safetensors header")
    header = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_json_object)
    data_bytes = file_size - 8 - int(header_length)
    result = {}
    occupied = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, Mapping):
            raise ValueError("{} metadata is not an object".format(name))
        shape = tuple(int(item) for item in metadata["shape"])
        dtype_code = metadata["dtype"]
        dtype = (
            _SAFETENSORS_DTYPES[dtype_code]
            if dtype_code in _SAFETENSORS_DTYPES
            else normalize_dtype(dtype_code)
        )
        offsets = tuple(int(item) for item in metadata["data_offsets"])
        if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
            raise ValueError("{} has invalid data_offsets {}".format(name, offsets))
        if offsets[1] > data_bytes:
            raise ValueError(
                "{} data offset {} exceeds {} data bytes".format(
                    name, offsets[1], data_bytes
                )
            )
        expected_bytes = shape_numel(shape) * DTYPE_BYTES[dtype]
        if offsets[1] - offsets[0] != expected_bytes:
            raise ValueError(
                "{} occupies {} bytes, expected {}".format(
                    name, offsets[1] - offsets[0], expected_bytes
                )
            )
        occupied.append((offsets[0], offsets[1], name))
        result[name] = {
            "shape": shape,
            "dtype": dtype,
            "data_offsets": offsets,
        }
    previous_end = 0
    for start, end, name in sorted(occupied):
        if start < previous_end:
            raise ValueError("{} overlaps another tensor".format(name))
        if start != previous_end:
            raise ValueError(
                "{} starts at {}, leaving an unreferenced data gap".format(
                    name, start
                )
            )
        previous_end = end
    if previous_end != data_bytes:
        raise ValueError(
            "tensor data covers {} bytes, file contains {} bytes".format(
                previous_end, data_bytes
            )
        )
    return result


def _discover_files(checkpoint):
    if checkpoint.is_file():
        if checkpoint.name.endswith(".index.json"):
            index_path = checkpoint
            root = checkpoint.parent
        elif checkpoint.suffix == ".safetensors":
            return checkpoint.parent, None, (checkpoint,)
        else:
            raise ValueError("unsupported checkpoint file {}".format(checkpoint))
    else:
        root = checkpoint
        index_path = root / "model.safetensors.index.json"
        single = root / "model.safetensors"
        if not index_path.exists():
            if single.exists():
                return root, None, (single,)
            raise FileNotFoundError(
                "expected model.safetensors or model.safetensors.index.json in {}".format(
                    root
                )
            )
    with index_path.open("r", encoding="utf-8") as source:
        index = json.load(source, object_pairs_hook=_unique_json_object)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("{} has no weight_map object".format(index_path))
    files = tuple(sorted({root / str(name) for name in weight_map.values()}))
    return root, weight_map, files
