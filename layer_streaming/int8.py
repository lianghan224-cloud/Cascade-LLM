"""Mixed-dtype Llama INT8-dequant-to-BF16 fallback runtime.

The compressed checkpoint stores Transformer linear weights as INT8 plus one
BF16 scale per output channel. Embedding, LM head, and normalization weights
remain BF16. This module keeps that representation in CPU memory, transfers
INT8 weights without expansion, and dequantizes one matrix at a time into a
reusable BF16 GPU workspace.  It is intentionally distinct from a fused
W8A8 activation-and-weight kernel.
"""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
import json
from operator import mul
from pathlib import Path
import threading
import time

import torch

from .plan import Granularity, MIB, VocabPlan
from .pipeline import PipelineRuntimeCore


ALIGNMENT_BYTES = 256
DTYPE_BYTES = {"int8": 1, "bfloat16": 2}
TORCH_DTYPES = {
    "int8": torch.int8,
    "bfloat16": torch.bfloat16,
}


def _product(values):
    return reduce(mul, values, 1)


def _align(value, alignment=ALIGNMENT_BYTES):
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class Int8TensorSpec:
    key: str
    shape: tuple
    dtype: str
    resident: bool = False
    alias_of: object = None

    @property
    def numel(self):
        return _product(self.shape)

    @property
    def nbytes(self):
        return self.numel * DTYPE_BYTES[self.dtype]

    @property
    def torch_dtype(self):
        return TORCH_DTYPES[self.dtype]


@dataclass(frozen=True)
class Int8UnitPiece:
    tensor: Int8TensorSpec
    unit_offset_bytes: int


@dataclass(frozen=True)
class Int8TransferUnit:
    unit_id: str
    layer_index: int
    operation: str
    pieces: tuple
    nbytes: int
    host_offset_bytes: int


@dataclass(frozen=True)
class Int8HostPlacement:
    tensor: Int8TensorSpec
    host_offset_bytes: int


@dataclass(frozen=True)
class Int8ResidentPlacement:
    tensor: Int8TensorSpec
    host_offset_bytes: int
    device_offset_elements: int


@dataclass(frozen=True)
class Int8ModelPlan:
    model_id: str
    granularity: Granularity
    tensors: dict
    units: tuple
    resident: tuple
    host_only: tuple
    host_arena_bytes: int
    resident_arena_elements: int
    transfer_slot_bytes: int
    dequant_workspace_elements: int
    vocab: object
    slot_count: int = 2
    geometry: object = None
    alias_map: dict = field(default_factory=dict)

    @property
    def stream_bytes_per_token(self):
        return sum(unit.nbytes for unit in self.units)

    @property
    def resident_bytes(self):
        return self.resident_arena_elements * 2

    @property
    def dequant_workspace_bytes(self):
        return self.dequant_workspace_elements * 2

    @property
    def slot_bytes(self):
        return self.transfer_slot_bytes + self.dequant_workspace_bytes

    @property
    def two_slot_bytes(self):
        return self.slot_count * self.slot_bytes

    @property
    def slot_elements(self):
        # Compatibility surface used by VocabStreamingRuntime. Its device slots
        # are the BF16 dequantization workspaces.
        return self.dequant_workspace_elements

    @property
    def aliases(self):
        return self.alias_map

    def tensor_host_offset_bytes(self, key):
        alias = self.aliases.get(key)
        if alias is not None:
            return self.tensor_host_offset_bytes(alias)
        for unit in self.units:
            for piece in unit.pieces:
                if piece.tensor.key == key:
                    return unit.host_offset_bytes + piece.unit_offset_bytes
        for placement in self.host_only:
            if placement.tensor.key == key:
                return placement.host_offset_bytes
        for placement in self.resident:
            if placement.tensor.key == key:
                return placement.host_offset_bytes
        raise KeyError(key)

    def as_dict(self, include_units=True):
        result = {
            "model_id": self.model_id,
            "granularity": self.granularity.value,
            "checkpoint_format": "compressed-tensors/int-quantized",
            "transfer_unit_count": len(self.units),
            "stream_bytes_per_token": self.stream_bytes_per_token,
            "host_arena_bytes": self.host_arena_bytes,
            "resident_bytes": self.resident_bytes,
            "transfer_slot_bytes": self.transfer_slot_bytes,
            "dequant_workspace_bytes": self.dequant_workspace_bytes,
            "device_bytes_per_slot": self.slot_bytes,
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
            "unit_bytes": {
                "min": min(unit.nbytes for unit in self.units),
                "max": max(unit.nbytes for unit in self.units),
                "sum": sum(unit.nbytes for unit in self.units),
            },
        }
        if include_units:
            result["units"] = [
                {
                    "unit_id": unit.unit_id,
                    "layer_index": unit.layer_index,
                    "operation": unit.operation,
                    "bytes": unit.nbytes,
                    "pieces": [
                        {
                            "key": piece.tensor.key,
                            "shape": list(piece.tensor.shape),
                            "dtype": piece.tensor.dtype,
                            "bytes": piece.tensor.nbytes,
                            "unit_offset_bytes": piece.unit_offset_bytes,
                        }
                        for piece in unit.pieces
                    ],
                }
                for unit in self.units
            ]
        return result


def _place_unit(unit_id, layer_index, operation, tensors, host_cursor):
    unit_cursor = 0
    pieces = []
    for tensor in tensors:
        unit_cursor = _align(unit_cursor)
        pieces.append(Int8UnitPiece(tensor, unit_cursor))
        unit_cursor += tensor.nbytes
    unit_bytes = _align(unit_cursor)
    host_cursor = _align(host_cursor)
    unit = Int8TransferUnit(
        unit_id=unit_id,
        layer_index=layer_index,
        operation=operation,
        pieces=tuple(pieces),
        nbytes=unit_bytes,
        host_offset_bytes=host_cursor,
    )
    return unit, host_cursor + unit_bytes


def build_llama_int8_plan(
    geometry,
    granularity=Granularity.MATRIX,
    embedding_mode="streamed",
    lm_head_mode="streamed",
    vocab_chunk_bytes=128 * MIB,
):
    """Build the explicit INT8-dequant-to-BF16 fallback plan."""

    granularity = Granularity(granularity)
    embedding_mode = str(getattr(embedding_mode, "value", embedding_mode))
    lm_head_mode = str(getattr(lm_head_mode, "value", lm_head_mode))
    if embedding_mode not in {"resident", "streamed"}:
        raise ValueError("embedding_mode must be resident or streamed")
    if lm_head_mode not in {"resident", "streamed"}:
        raise ValueError("lm_head_mode must be resident or streamed")
    stream_embedding = embedding_mode == "streamed"
    stream_lm_head = lm_head_mode == "streamed"
    hidden = int(geometry.hidden_size)
    intermediate = int(geometry.intermediate_size)
    kv = int(geometry.num_key_value_heads) * int(geometry.head_dim)
    vocab = int(geometry.vocab_size)
    layers = int(geometry.num_hidden_layers)
    tie_word_embeddings = bool(geometry.tie_word_embeddings)
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

    def add(key, shape, dtype, resident=False, alias_of=None):
        spec = Int8TensorSpec(
            key=key,
            shape=tuple(shape),
            dtype=dtype,
            resident=resident,
            alias_of=alias_of,
        )
        specs[key] = spec
        return spec

    embedding = add(
        "model.embed_tokens.weight",
        (vocab, hidden),
        "bfloat16",
    )
    if tie_word_embeddings:
        lm_head = add(
            "lm_head.weight",
            (vocab, hidden),
            "bfloat16",
            alias_of=embedding.key,
        )
    else:
        lm_head = add("lm_head.weight", (vocab, hidden), "bfloat16")

    layer_matrices = []
    layer_norms = []
    for layer_index in range(layers):
        prefix = "model.layers.{}".format(layer_index)
        matrices = []
        for name, shape in projection_shapes:
            block = (
                "self_attn"
                if name in {"q_proj", "k_proj", "v_proj", "o_proj"}
                else "mlp"
            )
            key = "{}.{}.{}.weight".format(prefix, block, name)
            weight = add(key, shape, "int8")
            scale = add(
                key + "_scale",
                (shape[0], 1),
                "bfloat16",
            )
            matrices.append((weight, scale))
        layer_matrices.append(tuple(matrices))
        layer_norms.extend(
            (
                add(
                    "{}.input_layernorm.weight".format(prefix),
                    (hidden,),
                    "bfloat16",
                    resident=True,
                ),
                add(
                    "{}.post_attention_layernorm.weight".format(prefix),
                    (hidden,),
                    "bfloat16",
                    resident=True,
                ),
            )
        )
    final_norm = add(
        "model.norm.weight",
        (hidden,),
        "bfloat16",
        resident=True,
    )

    host_cursor = 0
    units = []
    for layer_index, matrices in enumerate(layer_matrices):
        if granularity == Granularity.LAYER:
            pieces = tuple(
                tensor for pair in matrices for tensor in pair
            )
            unit, host_cursor = _place_unit(
                "layer_{:02d}".format(layer_index),
                layer_index,
                "layer",
                pieces,
                host_cursor,
            )
            units.append(unit)
        elif granularity == Granularity.MATRIX:
            for weight, scale in matrices:
                operation = weight.key.rsplit(".", 2)[-2]
                unit, host_cursor = _place_unit(
                    "layer_{:02d}.{}".format(layer_index, operation),
                    layer_index,
                    operation,
                    (weight, scale),
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
                pieces = tuple(
                    tensor for pair in group for tensor in pair
                )
                unit, host_cursor = _place_unit(
                    "layer_{:02d}.{}".format(layer_index, operation),
                    layer_index,
                    operation,
                    pieces,
                    host_cursor,
                )
                units.append(unit)

    host_only_specs = []
    resident_specs = []
    if tie_word_embeddings:
        if stream_embedding and stream_lm_head:
            host_only_specs.append(embedding)
        else:
            resident_specs.append(embedding)
    else:
        (host_only_specs if stream_embedding else resident_specs).append(
            embedding
        )
        (host_only_specs if stream_lm_head else resident_specs).append(lm_head)

    host_only = []
    for tensor in host_only_specs:
        host_cursor = _align(host_cursor)
        host_only.append(Int8HostPlacement(tensor, host_cursor))
        host_cursor += tensor.nbytes

    resident = []
    device_cursor = 0
    for tensor in tuple(resident_specs) + tuple(layer_norms) + (final_norm,):
        host_cursor = _align(host_cursor)
        resident.append(
            Int8ResidentPlacement(
                tensor=tensor,
                host_offset_bytes=host_cursor,
                device_offset_elements=device_cursor,
            )
        )
        host_cursor += tensor.nbytes
        device_cursor += tensor.numel

    vocab_plan = None
    chunk_elements = 0
    if stream_embedding or stream_lm_head:
        row_bytes = hidden * 2
        chunk_rows = min(vocab, int(vocab_chunk_bytes) // row_bytes)
        if chunk_rows < 1:
            raise ValueError("vocabulary chunk must hold at least one row")
        chunk_elements = chunk_rows * hidden
        chunk_bytes = chunk_elements * 2
        vocab_plan = VocabPlan(
            embedding_key=embedding.key,
            lm_head_key=lm_head.key,
            vocab_size=vocab,
            hidden_size=hidden,
            chunk_rows=chunk_rows,
            chunk_elements=chunk_elements,
            chunk_bytes=chunk_bytes,
            chunk_count=(vocab + chunk_rows - 1) // chunk_rows,
            stream_embedding=stream_embedding,
            stream_lm_head=stream_lm_head,
        )
    transfer_slot_bytes = max(unit.nbytes for unit in units)
    dequant_elements = max(
        spec.numel for spec in specs.values() if spec.dtype == "int8"
    )
    if stream_lm_head:
        dequant_elements = max(dequant_elements, chunk_elements)
    return Int8ModelPlan(
        model_id=str(geometry.model_id),
        granularity=granularity,
        tensors=specs,
        units=tuple(units),
        resident=tuple(resident),
        host_only=tuple(host_only),
        host_arena_bytes=_align(host_cursor),
        resident_arena_elements=device_cursor,
        transfer_slot_bytes=transfer_slot_bytes,
        dequant_workspace_elements=dequant_elements,
        vocab=vocab_plan,
        geometry=geometry,
        alias_map=(
            {"lm_head.weight": "model.embed_tokens.weight"}
            if tie_word_embeddings
            else {}
        ),
    )


class Int8WeightStoreMode(str, Enum):
    FULL_PINNED = "full_pinned"
    PINNED_STAGING = "pinned_staging"


def _resolved_checkpoint_files(checkpoint):
    checkpoint = Path(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    single_path = checkpoint / "model.safetensors"
    if not index_path.exists():
        if single_path.exists():
            return None
        raise FileNotFoundError(
            "expected model.safetensors or model.safetensors.index.json in {}".format(
                checkpoint
            )
        )
    with index_path.open("r", encoding="utf-8") as source:
        index = json.load(source)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("{} has no weight_map".format(index_path))
    return {
        key: checkpoint / filename for key, filename in weight_map.items()
    }


class Int8BaseWeightStore:
    def __init__(
        self,
        plan,
        pin_full_arena,
        slot_count=2,
        allocate=True,
        tensor_factory=None,
    ):
        self.plan = plan
        self.pin_full_arena = bool(pin_full_arena)
        self.slot_count = int(slot_count)
        self.tensor_factory = tensor_factory or torch.empty
        self.arena = None
        self._loaded = False
        self._closed = False
        if allocate:
            self.allocate()

    @property
    def mode(self):
        raise NotImplementedError

    @property
    def allocated(self):
        return self.arena is not None

    @property
    def loaded(self):
        return self._loaded

    @property
    def pinned_cpu_bytes(self):
        raise NotImplementedError

    @property
    def arena_is_pinned(self):
        return bool(self.arena is not None and self.arena.is_pinned())

    def allocate(self):
        if self.arena is None:
            self._closed = False
            self.arena = self.tensor_factory(
                self.plan.host_arena_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=self.pin_full_arena,
            )
        return self.arena

    def _target_view(self, key):
        spec = self.plan.tensors[key]
        if spec.alias_of is not None:
            return self._target_view(spec.alias_of)
        start = self.plan.tensor_host_offset_bytes(key)
        raw = self._raw_view(start, spec.nbytes)
        return raw.view(spec.torch_dtype).view(spec.shape)

    def _raw_view(self, start, nbytes):
        return self.arena[start : start + nbytes]

    def tensor(self, key):
        if self.arena is None:
            raise RuntimeError("weight arena has not been allocated")
        return self._target_view(key)

    def unit_source(self, unit, slot_index=None):
        start = unit.host_offset_bytes
        return self._raw_view(start, unit.nbytes)

    def load_checkpoint(self, checkpoint):
        from safetensors import safe_open

        if self.arena is None:
            self.allocate()
        weight_map = _resolved_checkpoint_files(checkpoint)
        required = {
            key
            for key, spec in self.plan.tensors.items()
            if spec.alias_of is None
        }
        if weight_map is None:
            files = [Path(checkpoint) / "model.safetensors"]
        else:
            missing_index = required.difference(weight_map)
            if missing_index:
                raise KeyError(
                    "checkpoint index misses required tensors: {}".format(
                        ", ".join(sorted(missing_index)[:8])
                    )
                )
            files = sorted({weight_map[key] for key in required})
        loaded = set()
        with torch.inference_mode():
            for filename in files:
                with safe_open(
                    str(filename),
                    framework="pt",
                    device="cpu",
                ) as source:
                    available = required.intersection(source.keys())
                    for key in sorted(available):
                        tensor = source.get_tensor(key)
                        target = self._target_view(key)
                        if tuple(tensor.shape) != tuple(target.shape):
                            raise ValueError(
                                "{} shape {} != expected {}".format(
                                    key,
                                    tuple(tensor.shape),
                                    tuple(target.shape),
                                )
                            )
                        if tensor.dtype != target.dtype:
                            raise ValueError(
                                "{} dtype {} != expected {}".format(
                                    key,
                                    tensor.dtype,
                                    target.dtype,
                                )
                            )
                        target.copy_(tensor)
                        loaded.add(key)
        missing = required.difference(loaded)
        if missing:
            raise KeyError(
                "checkpoint misses required tensors: {}".format(
                    ", ".join(sorted(missing)[:8])
                )
            )
        self._loaded = True
        return self

    def close(self):
        if self._closed:
            return
        self.arena = None
        self._loaded = False
        self._closed = True

    def __enter__(self):
        if self.arena is None:
            self.allocate()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def reset_profile(self):
        return None

    def profile_stats(self):
        return {}


class Int8FullPinnedWeightStore(Int8BaseWeightStore):
    """Fully pinned store split at transfer-unit boundaries.

    CUDA host allocation on this machine accepts 64 GiB but rejects the
    checkpoint's 67.7 GiB as one contiguous allocation. Multiple pinned chunks
    avoid that per-allocation ceiling while keeping each H2D source contiguous.
    """

    def __init__(
        self,
        plan,
        slot_count=2,
        allocate=True,
        max_chunk_bytes=32 * 1024**3,
    ):
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.pinned_chunks = []
        super().__init__(
            plan,
            pin_full_arena=True,
            slot_count=slot_count,
            allocate=allocate,
        )

    @property
    def mode(self):
        return Int8WeightStoreMode.FULL_PINNED

    @property
    def pinned_cpu_bytes(self):
        return sum(chunk.numel() for _, chunk in self.pinned_chunks)

    @property
    def arena_is_pinned(self):
        return bool(self.pinned_chunks) and all(
            chunk.is_pinned() for _, chunk in self.pinned_chunks
        )

    def allocate(self):
        if self.pinned_chunks:
            return self.arena
        self._closed = False
        intervals = [
            (unit.host_offset_bytes, unit.host_offset_bytes + unit.nbytes)
            for unit in self.plan.units
        ]
        intervals.extend(
            (
                placement.host_offset_bytes,
                placement.host_offset_bytes + placement.tensor.nbytes,
            )
            for placement in self.plan.host_only
        )
        intervals.extend(
            (
                placement.host_offset_bytes,
                placement.host_offset_bytes + placement.tensor.nbytes,
            )
            for placement in self.plan.resident
        )
        intervals.sort()
        ranges = []
        chunk_start, chunk_end = intervals[0]
        for start, end in intervals[1:]:
            if end - chunk_start > self.max_chunk_bytes:
                ranges.append((chunk_start, chunk_end))
                chunk_start = start
            chunk_end = end
        ranges.append((chunk_start, chunk_end))
        for start, end in ranges:
            chunk = self.tensor_factory(
                end - start,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            self.pinned_chunks.append((start, chunk))
        # Preserve the existing allocation/inspection surface without implying
        # that this first chunk spans the complete logical arena.
        self.arena = self.pinned_chunks[0][1]
        return self.arena

    def _raw_view(self, start, nbytes):
        end = start + nbytes
        for chunk_start, chunk in self.pinned_chunks:
            chunk_end = chunk_start + chunk.numel()
            if start >= chunk_start and end <= chunk_end:
                local_start = start - chunk_start
                return chunk[local_start : local_start + nbytes]
        raise RuntimeError(
            "range [{}, {}) crosses a pinned chunk boundary".format(
                start,
                end,
            )
        )

    def prepare_unit(self, unit, slot_index, reuse_event=None):
        future = Future()
        future.set_result(self.unit_source(unit))
        return future

    def close(self):
        self.pinned_chunks = []
        super().close()


class Int8PinnedStagingWeightStore(Int8BaseWeightStore):
    def __init__(self, plan, slot_count=2, allocate=True):
        self.staging_slots = None
        self._executor = None
        self._profile_lock = threading.Lock()
        self._stage_durations_ms = []
        super().__init__(
            plan,
            pin_full_arena=False,
            slot_count=slot_count,
            allocate=allocate,
        )

    @property
    def mode(self):
        return Int8WeightStoreMode.PINNED_STAGING

    @property
    def pinned_cpu_bytes(self):
        return self.slot_count * self.plan.transfer_slot_bytes

    def allocate(self):
        arena = super().allocate()
        if self.staging_slots is None:
            self.staging_slots = [
                self.tensor_factory(
                    self.plan.transfer_slot_bytes,
                    dtype=torch.uint8,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(self.slot_count)
            ]
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.slot_count,
                thread_name_prefix="int8-weight-stage",
            )
        return arena

    def _stage(self, unit, slot_index, reuse_event):
        started = time.perf_counter()
        if reuse_event is not None:
            reuse_event.synchronize()
        source = self.unit_source(unit)
        target = self.staging_slots[slot_index][: unit.nbytes]
        target.copy_(source)
        duration_ms = (time.perf_counter() - started) * 1000.0
        with self._profile_lock:
            self._stage_durations_ms.append(duration_ms)
        return target

    def prepare_unit(self, unit, slot_index, reuse_event=None):
        return self._executor.submit(
            self._stage,
            unit,
            slot_index,
            reuse_event,
        )

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.staging_slots = None
        super().close()

    def reset_profile(self):
        with self._profile_lock:
            self._stage_durations_ms = []

    def profile_stats(self):
        with self._profile_lock:
            durations = list(self._stage_durations_ms)
        return {
            "staging_copy_count": len(durations),
            "staging_event_sum_ms": sum(durations),
            "staging_event_max_ms": max(durations) if durations else 0.0,
        }


def create_int8_weight_store(
    plan,
    mode=Int8WeightStoreMode.PINNED_STAGING,
    slot_count=2,
    allocate=True,
):
    mode = Int8WeightStoreMode(mode)
    if mode == Int8WeightStoreMode.FULL_PINNED:
        return Int8FullPinnedWeightStore(
            plan,
            slot_count=slot_count,
            allocate=allocate,
        )
    return Int8PinnedStagingWeightStore(
        plan,
        slot_count=slot_count,
        allocate=allocate,
    )


class Int8ResidentDeviceArena:
    def __init__(self, plan, store, device):
        self.plan = plan
        self.store = store
        self.device = torch.device(device)
        self.arena = torch.empty(
            plan.resident_arena_elements,
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.views = {}
        for placement in plan.resident:
            spec = placement.tensor
            start = placement.device_offset_elements
            view = self.arena[start : start + spec.numel].view(spec.shape)
            source = store.tensor(spec.key)
            view.copy_(source, non_blocking=source.is_pinned())
            self.views[spec.key] = view
        for alias, target in plan.aliases.items():
            if target in self.views:
                self.views[alias] = self.views[target]
        torch.cuda.current_stream(self.device).synchronize()

    @property
    def nbytes(self):
        return self.plan.resident_bytes

    def __getitem__(self, key):
        return self.views[key]

    def close(self):
        self.views = {}
        self.arena = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class _DequantizingViews:
    def __init__(self, raw_views, workspace, profile_events=None):
        self.raw_views = raw_views
        self.workspace = workspace
        self.profile_events = profile_events
        self.dequant_events = []

    def __getitem__(self, key):
        source = self.raw_views[key]
        if source.dtype != torch.int8:
            return source
        scale_key = key + "_scale"
        scale = self.raw_views[scale_key]
        target = self.workspace[: source.numel()].view(source.shape)
        if self.profile_events is not None:
            start, end = self.profile_events[key]
            start.record()
        torch.mul(source, scale, out=target)
        if self.profile_events is not None:
            end.record()
            self.dequant_events.append((start, end))
        return target


class Int8DoubleBufferRuntime:
    """INT8 H2D slots plus one reusable BF16 dequant workspace per slot."""

    def __init__(
        self,
        plan,
        store,
        resident,
        device="cuda:0",
        slot_count=2,
        profile=False,
        source_timeout_seconds=120.0,
        prefetch_depth=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.plan = plan
        self.store = store
        self.resident = resident
        self.device = torch.device(device)
        self.slot_count = int(slot_count)
        if self.slot_count < 1:
            raise ValueError("slot_count must be positive")
        self.profile = bool(profile)
        self.source_timeout_seconds = float(source_timeout_seconds)
        if self.source_timeout_seconds <= 0:
            raise ValueError("source_timeout_seconds must be positive")
        self.last_profile = None
        self._closed = False
        self.prefetch_depth = (
            self.slot_count
            if prefetch_depth is None
            else int(prefetch_depth)
        )
        if self.prefetch_depth < 1:
            raise ValueError("prefetch_depth must be positive")
        self._dequant_event_pairs = []
        self._dequant_profile_events = None
        self.transfer_slots = [
            torch.empty(
                plan.transfer_slot_bytes,
                dtype=torch.uint8,
                device=self.device,
            )
            for _ in range(self.slot_count)
        ]
        self.dequant_slots = [
            torch.empty(
                plan.dequant_workspace_elements,
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(self.slot_count)
        ]
        # VocabStreamingRuntime reuses these BF16 workspaces after Transformer
        # execution, so no additional LM-head GPU buffers are allocated.
        self.device_slots = self.dequant_slots
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.compute_stream = torch.cuda.Stream(device=self.device)
        self.coordinator = torch.cuda.current_stream(self.device)
        self.ready_events = [torch.cuda.Event() for _ in self.plan.units]
        self.free_events = [torch.cuda.Event() for _ in self.plan.units]
        self.run_start_event = torch.cuda.Event(enable_timing=False)
        self._profile_events = None
        if self.profile:
            self._dequant_profile_events = {
                key: (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for key, spec in self.plan.tensors.items()
                if spec.dtype == "int8"
            }
            self._profile_events = {
                name: [
                    torch.cuda.Event(enable_timing=True)
                    for _ in self.plan.units
                ]
                for name in (
                    "copy_starts",
                    "copy_ends",
                    "compute_starts",
                    "compute_ends",
                )
            }
            self._profile_events["pipeline_start"] = torch.cuda.Event(
                enable_timing=True
            )
            self._profile_events["pipeline_end"] = torch.cuda.Event(
                enable_timing=True
            )
        self.pipeline = PipelineRuntimeCore(
            plan=self.plan,
            store=self.store,
            slot_count=self.slot_count,
            prefetch_depth=self.prefetch_depth,
            source_timeout_seconds=self.source_timeout_seconds,
            submit_copy=self._submit_copy,
        )

    @property
    def stats(self):
        transfer = self.slot_count * self.plan.transfer_slot_bytes
        dequant = self.slot_count * self.plan.dequant_workspace_bytes
        return {
            "transfer_units": len(self.plan.units),
            "slot_count": self.slot_count,
            "transfer_slot_bytes": self.plan.transfer_slot_bytes,
            "dequant_workspace_bytes": self.plan.dequant_workspace_bytes,
            "device_transfer_slots_bytes": transfer,
            "device_dequant_workspaces_bytes": dequant,
            "device_slots_bytes": transfer + dequant,
            "resident_bytes": self.plan.resident_bytes,
            "weight_gpu_bytes": transfer + dequant + self.plan.resident_bytes,
        }

    def _raw_unit_views(self, unit, slot):
        views = {}
        for piece in unit.pieces:
            spec = piece.tensor
            begin = piece.unit_offset_bytes
            raw = slot[begin : begin + spec.nbytes]
            views[spec.key] = raw.view(spec.torch_dtype).view(spec.shape)
        return views

    def _submit_copy(self, prepared, lease):
        index = prepared.index
        unit = prepared.unit
        transfer_slot = self.transfer_slots[lease.slot_index]
        with torch.cuda.device(self.device):
            if lease.reuse_event is not None:
                self.copy_stream.wait_event(lease.reuse_event)
            with torch.cuda.stream(self.copy_stream):
                if self.profile:
                    self._profile_events["copy_starts"][index].record(
                        self.copy_stream
                    )
                transfer_slot[: unit.nbytes].copy_(
                    prepared.source, non_blocking=True
                )
                if self.profile:
                    self._profile_events["copy_ends"][index].record(
                        self.copy_stream
                    )
                ready_event = self.ready_events[index]
                ready_event.record(self.copy_stream)
        return ready_event

    def _consume_ready(self, ready, compute_unit, state):
        index = ready.index
        unit = ready.unit
        slot_index = ready.device_slot_index
        transfer_slot = self.transfer_slots[slot_index]
        self.compute_stream.wait_event(ready.ready_event)
        with torch.cuda.stream(self.compute_stream):
            if self.profile:
                self._profile_events["compute_starts"][index].record(
                    self.compute_stream
                )
            try:
                raw_views = self._raw_unit_views(unit, transfer_slot)
                views = _DequantizingViews(
                    raw_views,
                    self.dequant_slots[slot_index],
                    profile_events=self._dequant_profile_events,
                )
                state = compute_unit(unit, views, state)
                self._dequant_event_pairs.extend(views.dequant_events)
            finally:
                if self.profile:
                    self._profile_events["compute_ends"][index].record(
                        self.compute_stream
                    )
                free_event = self.free_events[index]
                free_event.record(self.compute_stream)
        return state, free_event

    def run(self, compute_unit, state):
        if self._closed:
            raise RuntimeError("runtime is closed")
        wall_started = time.perf_counter()
        units = self.plan.units
        if not units:
            return state
        copy_starts = []
        copy_ends = []
        compute_starts = []
        compute_ends = []
        pipeline_start = None
        pipeline_end = None
        if self.profile:
            pipeline_start = self._profile_events["pipeline_start"]
            pipeline_end = self._profile_events["pipeline_end"]
            copy_starts = self._profile_events["copy_starts"]
            copy_ends = self._profile_events["copy_ends"]
            compute_starts = self._profile_events["compute_starts"]
            compute_ends = self._profile_events["compute_ends"]
            pipeline_start.record(self.coordinator)
            self.copy_stream.wait_event(pipeline_start)
            self.compute_stream.wait_event(pipeline_start)
        else:
            self.run_start_event.record(self.coordinator)
            self.copy_stream.wait_event(self.run_start_event)
            self.compute_stream.wait_event(self.run_start_event)

        self.store.reset_profile()
        self._dequant_event_pairs = []
        state, last_free_event = self.pipeline.run(
            lambda ready, current: self._consume_ready(
                ready, compute_unit, current
            ),
            state,
            timeout_seconds=self.source_timeout_seconds,
        )
        self.coordinator.wait_event(last_free_event)
        if self.profile:
            pipeline_end.record(self.coordinator)
        self.coordinator.synchronize()
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        if self.profile:
            h2d_durations = [
                start.elapsed_time(end)
                for start, end in zip(copy_starts, copy_ends)
            ]
            compute_durations = [
                start.elapsed_time(end)
                for start, end in zip(compute_starts, compute_ends)
            ]
            h2d_ms = sum(h2d_durations)
            compute_ms = sum(compute_durations)
            attention_operations = {
                "q_proj", "k_proj", "v_proj", "o_proj", "qkv"
            }
            mlp_operations = {
                "gate_proj", "up_proj", "down_proj", "gate_up"
            }
            attention_ms = sum(
                duration
                for unit, duration in zip(units, compute_durations)
                if unit.operation in attention_operations
            )
            mlp_ms = sum(
                duration
                for unit, duration in zip(units, compute_durations)
                if unit.operation in mlp_operations
            )
            dequant_ms = sum(
                start.elapsed_time(end)
                for start, end in self._dequant_event_pairs
            )
            self.last_profile = {
                "wall_ms": wall_ms,
                "host_submit_ms": self.pipeline.last_stats[
                    "pipeline_host_wall_ms"
                ],
                "source_wait_ms": self.pipeline.last_stats[
                    "source_prepare_wait_ms"
                ],
                "gpu_pipeline_ms": pipeline_start.elapsed_time(
                    pipeline_end
                ),
                "h2d_event_sum_ms": h2d_ms,
                "compute_event_sum_ms": compute_ms,
                "attention_event_sum_ms": (
                    None
                    if self.plan.granularity.value == "layer"
                    else attention_ms
                ),
                "mlp_event_sum_ms": (
                    None
                    if self.plan.granularity.value == "layer"
                    else mlp_ms
                ),
                "dequant_event_sum_ms": dequant_ms,
                "h2d_bytes": self.plan.stream_bytes_per_token,
                "h2d_effective_gbps": (
                    self.plan.stream_bytes_per_token / 1e9
                )
                / (h2d_ms / 1000.0),
            }
            self.last_profile.update(self.pipeline.last_stats)
            self.last_profile.update(self.store.profile_stats())
        elif self.pipeline.last_stats:
            self.last_profile = dict(self.pipeline.last_stats)
        return state

    def close(self):
        if self._closed:
            return
        self.pipeline.close()
        self.transfer_slots = []
        self.dequant_slots = []
        self.device_slots = []
        self.ready_events = []
        self.free_events = []
        self.run_start_event = None
        self._profile_events = None
        self._dequant_profile_events = None
        self.copy_stream = None
        self.compute_stream = None
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("runtime is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
