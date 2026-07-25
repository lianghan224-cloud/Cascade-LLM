"""CPU weight stores for full-pinned and reusable pinned-staging modes."""

import json
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import torch

from .plan import ModelPlan


class WeightStoreMode(str, Enum):
    FULL_PINNED = "full_pinned"
    PINNED_STAGING = "pinned_staging"


def _resolved_checkpoint_files(checkpoint):
    checkpoint = Path(checkpoint)
    index_path = checkpoint / "model.safetensors.index.json"
    single_path = checkpoint / "model.safetensors"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as source:
            index = json.load(source)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("{} has no weight_map".format(index_path))
        return {
            key: checkpoint / filename for key, filename in weight_map.items()
        }
    if single_path.exists():
        return None
    raise FileNotFoundError(
        "expected model.safetensors or model.safetensors.index.json in "
        "{}".format(checkpoint)
    )


class BaseWeightStore:
    """Packed CPU arena shared by both memory modes."""

    def __init__(
        self,
        plan,
        pin_full_arena,
        allocate=True,
        tensor_factory=None,
    ):
        if not isinstance(plan, ModelPlan):
            raise TypeError("plan must be a ModelPlan")
        self.plan = plan
        self.pin_full_arena = bool(pin_full_arena)
        self.tensor_factory = tensor_factory or torch.empty
        self.arena = None
        self._loaded = False
        if allocate:
            self.allocate()

    @property
    def mode(self):
        raise NotImplementedError

    @property
    def pinned_cpu_bytes(self):
        raise NotImplementedError

    @property
    def allocated(self):
        return self.arena is not None

    @property
    def loaded(self):
        return self._loaded

    def allocate(self):
        if self.arena is not None:
            return self.arena
        self.arena = self.tensor_factory(
            self.plan.host_arena_elements,
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=self.pin_full_arena,
        )
        return self.arena

    def _target_view(self, key):
        spec = self.plan.tensors[key]
        if spec.alias_of is not None:
            return self._target_view(spec.alias_of)
        offset = self.plan.tensor_host_offset(key)
        return self.arena[offset : offset + spec.numel].view(spec.shape)

    def tensor(self, key):
        if self.arena is None:
            raise RuntimeError("weight arena has not been allocated")
        alias = self.plan.aliases.get(key)
        if alias is not None:
            key = alias
        return self._target_view(key)

    def unit_source(self, unit, slot_index=None):
        if self.arena is None:
            raise RuntimeError("weight arena has not been allocated")
        start = unit.host_offset_elements
        return self.arena[start : start + unit.elements]

    def load_checkpoint(self, checkpoint):
        """Load a local safetensors checkpoint into the packed CPU arena."""

        if self.arena is None:
            self.allocate()
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError(
                "safetensors is required to load a checkpoint"
            ) from error

        weight_map = _resolved_checkpoint_files(checkpoint)
        required = {
            key
            for key, spec in self.plan.tensors.items()
            if spec.alias_of is None
        }
        loaded = set()

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

        with torch.inference_mode():
            for filename in files:
                with safe_open(
                    str(filename),
                    framework="pt",
                    device="cpu",
                ) as source:
                    available = set(source.keys())
                    for key in sorted(required.intersection(available)):
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
                        target.copy_(tensor.to(dtype=torch.bfloat16))
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
        self.arena = None
        self._loaded = False


class FullPinnedWeightStore(BaseWeightStore):
    """All checkpoint weights remain in page-locked CPU memory."""

    def __init__(self, plan, allocate=True, tensor_factory=None):
        super().__init__(
            plan,
            pin_full_arena=True,
            allocate=allocate,
            tensor_factory=tensor_factory,
        )

    @property
    def mode(self):
        return WeightStoreMode.FULL_PINNED

    @property
    def pinned_cpu_bytes(self):
        return self.plan.host_arena_bytes

    def prepare_unit(self, unit, slot_index, reuse_event=None):
        future = Future()
        future.set_result(self.unit_source(unit))
        return future


class PinnedStagingWeightStore(BaseWeightStore):
    """Pageable full store plus two reusable page-locked staging slots."""

    def __init__(
        self,
        plan,
        allocate=True,
        tensor_factory=None,
        worker_count=2,
    ):
        self.staging_slots = None
        self.worker_count = worker_count
        self._executor = None
        super().__init__(
            plan,
            pin_full_arena=False,
            allocate=allocate,
            tensor_factory=tensor_factory,
        )

    @property
    def mode(self):
        return WeightStoreMode.PINNED_STAGING

    @property
    def pinned_cpu_bytes(self):
        return self.plan.two_slot_bytes

    def allocate(self):
        arena = super().allocate()
        if self.staging_slots is None:
            self.staging_slots = [
                self.tensor_factory(
                    self.plan.slot_elements,
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(self.plan.slot_count)
            ]
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.worker_count,
                thread_name_prefix="weight-stage",
            )
        return arena

    def _stage(self, unit, slot_index, reuse_event):
        if reuse_event is not None:
            reuse_event.synchronize()
        source = self.unit_source(unit)
        target = self.staging_slots[slot_index][: unit.elements]
        target.copy_(source)
        return target

    def prepare_unit(self, unit, slot_index, reuse_event=None):
        if self.staging_slots is None or self._executor is None:
            raise RuntimeError("staging store has not been allocated")
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


def create_weight_store(
    plan,
    mode=WeightStoreMode.FULL_PINNED,
    allocate=True,
    tensor_factory=None,
):
    mode = WeightStoreMode(mode)
    if mode == WeightStoreMode.FULL_PINNED:
        return FullPinnedWeightStore(
            plan,
            allocate=allocate,
            tensor_factory=tensor_factory,
        )
    return PinnedStagingWeightStore(
        plan,
        allocate=allocate,
        tensor_factory=tensor_factory,
    )
