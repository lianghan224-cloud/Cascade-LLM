"""Semantic weight-object identity layered over frozen physical metadata."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Optional, Tuple


class WeightObjectKind(str, Enum):
    DENSE = "dense"
    EMBEDDING = "embedding"
    LM_HEAD = "lm_head"
    ROUTER = "router"
    SHARED_EXPERT = "shared_expert"
    EXPERT = "expert"


@dataclass(frozen=True)
class WeightObjectKey:
    layer_id: Optional[int]
    kind: WeightObjectKind
    name: str
    expert_id: Optional[int] = None

    def __post_init__(self):
        kind = WeightObjectKind(self.kind)
        name = str(self.name).strip()
        layer_id = None if self.layer_id is None else int(self.layer_id)
        expert_id = None if self.expert_id is None else int(self.expert_id)
        if not name:
            raise ValueError("weight object name cannot be empty")
        if layer_id is not None and layer_id < 0:
            raise ValueError("layer_id cannot be negative")
        if expert_id is not None and expert_id < 0:
            raise ValueError("expert_id cannot be negative")
        if kind == WeightObjectKind.EXPERT:
            if layer_id is None or expert_id is None:
                raise ValueError("routed Expert keys require layer_id and expert_id")
        elif expert_id is not None:
            raise ValueError("expert_id is valid only for routed Expert keys")
        if kind in {
            WeightObjectKind.DENSE,
            WeightObjectKind.ROUTER,
            WeightObjectKind.SHARED_EXPERT,
        } and layer_id is None:
            raise ValueError("{} keys require layer_id".format(kind.value))
        if kind in {WeightObjectKind.EMBEDDING, WeightObjectKind.LM_HEAD} and (
            layer_id is not None
        ):
            raise ValueError("{} keys cannot set layer_id".format(kind.value))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "layer_id", layer_id)
        object.__setattr__(self, "expert_id", expert_id)

    @property
    def sort_key(self):
        return (
            -1 if self.layer_id is None else self.layer_id,
            self.kind.value,
            -1 if self.expert_id is None else self.expert_id,
            self.name,
        )

    def as_dict(self):
        return {
            "layer_id": self.layer_id,
            "kind": self.kind.value,
            "name": self.name,
            "expert_id": self.expert_id,
        }


@dataclass(frozen=True)
class WeightObjectSource:
    tensor_name: str
    storage_region: str
    host_offset: int
    nbytes: int

    def __post_init__(self):
        if not str(self.tensor_name):
            raise ValueError("source tensor_name cannot be empty")
        if not str(self.storage_region):
            raise ValueError("source storage_region cannot be empty")
        if int(self.host_offset) < 0 or int(self.nbytes) <= 0:
            raise ValueError("source offset/size is invalid")
        object.__setattr__(self, "host_offset", int(self.host_offset))
        object.__setattr__(self, "nbytes", int(self.nbytes))


@dataclass(frozen=True)
class WeightObjectRecord:
    key: WeightObjectKey
    tensor_names: Tuple[str, ...]
    storage_dtypes: Tuple[str, ...]
    shapes: Tuple[Tuple[int, ...], ...]
    nbytes: int
    sources: Tuple[WeightObjectSource, ...]

    def __post_init__(self):
        names = tuple(str(item) for item in self.tensor_names)
        dtypes = tuple(str(item) for item in self.storage_dtypes)
        shapes = tuple(tuple(int(value) for value in item) for item in self.shapes)
        sources = tuple(self.sources)
        width = len(names)
        if width < 1 or len(set(names)) != width:
            raise ValueError("weight object tensors must be non-empty and unique")
        if len(dtypes) != width or len(shapes) != width or len(sources) != width:
            raise ValueError("weight object metadata widths do not match")
        if tuple(item.tensor_name for item in sources) != names:
            raise ValueError("source order must match tensor_names")
        if int(self.nbytes) != sum(item.nbytes for item in sources):
            raise ValueError("weight object nbytes does not match sources")
        object.__setattr__(self, "tensor_names", names)
        object.__setattr__(self, "storage_dtypes", dtypes)
        object.__setattr__(self, "shapes", shapes)
        object.__setattr__(self, "nbytes", int(self.nbytes))
        object.__setattr__(self, "sources", sources)

    @property
    def dtype(self):
        return self.storage_dtypes[0] if len(set(self.storage_dtypes)) == 1 else None

    @property
    def shape(self):
        return self.shapes[0] if len(self.shapes) == 1 else self.shapes

    def as_dict(self):
        return {
            "key": self.key.as_dict(),
            "tensor_names": list(self.tensor_names),
            "storage_dtypes": list(self.storage_dtypes),
            "shapes": [list(item) for item in self.shapes],
            "nbytes": self.nbytes,
            "sources": [
                {
                    "tensor_name": item.tensor_name,
                    "storage_region": item.storage_region,
                    "host_offset": item.host_offset,
                    "nbytes": item.nbytes,
                }
                for item in self.sources
            ],
        }


class WeightObjectCatalog:
    """Deterministic semantic index over immutable WeightSpec metadata."""

    def __init__(
        self, records: Iterable[WeightObjectRecord], allow_source_aliases=False
    ):
        ordered = tuple(sorted(tuple(records), key=lambda item: item.key.sort_key))
        by_key = {}
        by_tensor = {}
        for record in ordered:
            if record.key in by_key:
                raise ValueError("duplicate weight object key {}".format(record.key))
            by_key[record.key] = record
            for name in record.tensor_names:
                if name in by_tensor and not allow_source_aliases:
                    raise ValueError("tensor {} belongs to multiple objects".format(name))
                by_tensor.setdefault(name, record.key)
        self._records = ordered
        self._by_key = by_key
        self._by_tensor = by_tensor

    def __len__(self):
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def get(self, key: WeightObjectKey) -> WeightObjectRecord:
        return self._by_key[key]

    def contains(self, key: WeightObjectKey) -> bool:
        return key in self._by_key

    def key_for_tensor(self, tensor_name: str) -> WeightObjectKey:
        return self._by_tensor[str(tensor_name)]

    def iter_layer(self, layer_id: int):
        layer_id = int(layer_id)
        return tuple(item for item in self._records if item.key.layer_id == layer_id)

    def iter_experts(self, layer_id: Optional[int] = None):
        return tuple(
            item
            for item in self._records
            if item.key.kind == WeightObjectKind.EXPERT
            and (layer_id is None or item.key.layer_id == int(layer_id))
        )

    def total_bytes(self, kind: Optional[WeightObjectKind] = None) -> int:
        if kind is not None:
            kind = WeightObjectKind(kind)
        return sum(
            item.nbytes
            for item in self._records
            if kind is None or item.key.kind == kind
        )

    def as_dict(self):
        return {
            "records": [item.as_dict() for item in self._records],
            "total_bytes": self.total_bytes(),
        }

    @classmethod
    def from_dense_plan(cls, plan):
        """Describe a Dense plan without putting catalog lookup on its hot path."""
        from ..execution_plan import storage_region_name

        quant_members = {}
        primary_names = set()
        for unit in plan.units:
            for tensor in unit.tensors:
                if tensor.backend:
                    primary_names.add(tensor.weight_name)
                    quant_members[tensor.weight_name] = tuple(tensor.quant_param_names)
        records = []
        handled = set()
        for name, spec in sorted(plan.weights.items()):
            if spec.alias_of is not None or name in handled:
                continue
            if name in primary_names:
                names = (name,) + quant_members.get(name, ())
            elif spec.role in {"scale", "zero_point"}:
                continue
            else:
                names = (name,)
            handled.update(names)
            if spec.role == "embedding":
                kind = WeightObjectKind.EMBEDDING
                layer_id = None
            elif spec.role == "lm_head":
                kind = WeightObjectKind.LM_HEAD
                layer_id = None
            else:
                kind = WeightObjectKind.DENSE
                parts = name.split(".")
                layer_id = (
                    int(parts[2])
                    if len(parts) > 3 and parts[:2] == ["model", "layers"]
                    else 0
                )
            sources = tuple(
                WeightObjectSource(
                    tensor_name=item,
                    storage_region=storage_region_name(plan.weights[item]),
                    host_offset=int(plan.host_offsets[item]),
                    nbytes=int(plan.weights[item].storage_nbytes),
                )
                for item in names
            )
            records.append(
                WeightObjectRecord(
                    key=WeightObjectKey(layer_id, kind, name),
                    tensor_names=names,
                    storage_dtypes=tuple(plan.weights[item].storage_dtype for item in names),
                    shapes=tuple(plan.weights[item].storage_shape for item in names),
                    nbytes=sum(item.nbytes for item in sources),
                    sources=sources,
                )
            )
        return cls(records)
