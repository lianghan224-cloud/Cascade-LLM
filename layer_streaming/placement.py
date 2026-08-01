"""Runtime-only static Transformer placement plans.

The sidecar deliberately does not modify the frozen ExecutionPlan schema v1.
Only complete Llama layers, starting at layer zero, may become resident.
"""

from dataclasses import dataclass
from typing import Tuple

from .execution_plan import ResidentTensor
from .specs import align_up


@dataclass(frozen=True)
class StaticTransformerPlacement:
    strategy: str
    budget_bytes: int
    resident_layer_ids: Tuple[int, ...]
    resident_unit_ids: Tuple[str, ...]
    streamed_unit_ids: Tuple[str, ...]
    resident_tensors: Tuple[ResidentTensor, ...]
    resident_weight_bytes: int
    streamed_weight_bytes: int
    resident_arena_bytes: int

    @property
    def resident_hit_ratio(self):
        total = self.resident_weight_bytes + self.streamed_weight_bytes
        return 0.0 if total == 0 else self.resident_weight_bytes / total

    def as_dict(self):
        return {
            "strategy": self.strategy,
            "budget_bytes": self.budget_bytes,
            "resident_layer_ids": list(self.resident_layer_ids),
            "resident_unit_ids": list(self.resident_unit_ids),
            "streamed_unit_ids": list(self.streamed_unit_ids),
            "resident_weight_bytes": self.resident_weight_bytes,
            "streamed_weight_bytes": self.streamed_weight_bytes,
            "resident_arena_bytes": self.resident_arena_bytes,
            "resident_hit_ratio": self.resident_hit_ratio,
        }


def build_static_transformer_placement(plan, budget_bytes=0):
    """Select the largest complete prefix of layers within ``budget_bytes``."""

    budget_bytes = int(budget_bytes)
    if budget_bytes < 0:
        raise ValueError("GPU resident Transformer budget cannot be negative")
    units_by_layer = {}
    for unit in plan.units:
        units_by_layer.setdefault(int(unit.layer_id), []).append(unit)
    expected_layers = tuple(range(plan.geometry.num_hidden_layers))
    if tuple(sorted(units_by_layer)) != expected_layers:
        raise ValueError("execution plan does not contain every Transformer layer")

    existing_names = {item.weight_name for item in plan.resident}
    cursor = int(plan.resident_bytes)
    selected_layers = []
    selected_units = []
    resident_tensors = []
    resident_storage_bytes = 0
    all_storage_names = {
        tensor.weight_name
        for unit in plan.units
        for tensor in unit.tensors
    }
    total_storage_bytes = sum(
        int(plan.weights[name].storage_nbytes) for name in all_storage_names
    )

    for layer_id in expected_layers:
        layer_units = tuple(units_by_layer[layer_id])
        layer_names = []
        seen = set(existing_names)
        for unit in layer_units:
            for tensor in unit.tensors:
                if tensor.weight_name not in seen:
                    seen.add(tensor.weight_name)
                    layer_names.append(tensor.weight_name)
        trial_cursor = cursor
        trial = []
        for name in layer_names:
            spec = plan.weights[name]
            trial_cursor = align_up(trial_cursor, spec.alignment)
            trial.append(
                ResidentTensor(
                    weight_name=name,
                    storage_region=next(
                        tensor.storage_region
                        for unit in layer_units
                        for tensor in unit.tensors
                        if tensor.weight_name == name
                    ),
                    host_offset=int(plan.host_offsets[name]),
                    storage_bytes=int(spec.storage_nbytes),
                    device_offset=trial_cursor,
                    backend=next(
                        (
                            tensor.backend
                            for unit in layer_units
                            for tensor in unit.tensors
                            if tensor.weight_name == name and tensor.backend
                        ),
                        "",
                    ),
                )
            )
            trial_cursor += int(spec.storage_nbytes)
        incremental_arena_bytes = trial_cursor - int(plan.resident_bytes)
        if incremental_arena_bytes > budget_bytes:
            break
        selected_layers.append(layer_id)
        selected_units.extend(unit.unit_id for unit in layer_units)
        resident_tensors.extend(trial)
        resident_storage_bytes += sum(item.storage_bytes for item in trial)
        cursor = trial_cursor
        existing_names.update(layer_names)

    selected_unit_ids = tuple(selected_units)
    selected_set = set(selected_unit_ids)
    return StaticTransformerPlacement(
        strategy="prefix_complete_layers",
        budget_bytes=budget_bytes,
        resident_layer_ids=tuple(selected_layers),
        resident_unit_ids=selected_unit_ids,
        streamed_unit_ids=tuple(
            unit.unit_id for unit in plan.units if unit.unit_id not in selected_set
        ),
        resident_tensors=tuple(resident_tensors),
        resident_weight_bytes=resident_storage_bytes,
        streamed_weight_bytes=total_storage_bytes - resident_storage_bytes,
        resident_arena_bytes=cursor,
    )
