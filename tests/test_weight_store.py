import unittest

import torch

from layer_streaming.plan import (
    ModelPlan,
    ResidentPlacement,
    TensorSpec,
    TransferUnit,
    UnitPiece,
)
from layer_streaming.weight_store import (
    FullPinnedWeightStore,
    PinnedStagingWeightStore,
)


def tiny_plan():
    streamed = TensorSpec("stream.weight", (2, 2))
    resident = TensorSpec("resident.weight", (2,), resident=True)
    unit = TransferUnit(
        unit_id="unit",
        layer_index=0,
        operation="q_proj",
        pieces=(UnitPiece(streamed, 0),),
        elements=4,
        host_offset_elements=0,
    )
    placement = ResidentPlacement(
        tensor=resident,
        host_offset_elements=4,
        device_offset_elements=0,
    )
    return ModelPlan(
        model_id="tiny",
        granularity="matrix",
        tensors={
            streamed.key: streamed,
            resident.key: resident,
        },
        units=(unit,),
        resident=(placement,),
        aliases={},
        host_arena_elements=6,
        resident_arena_elements=2,
        slot_elements=4,
    )


class RecordingFactory:
    def __init__(self):
        self.pin_flags = []

    def __call__(self, elements, **kwargs):
        self.pin_flags.append(bool(kwargs.pop("pin_memory")))
        return torch.empty(elements, **kwargs)


class FakeEvent:
    def __init__(self):
        self.synchronized = False

    def synchronize(self):
        self.synchronized = True


class WeightStoreTest(unittest.TestCase):
    def test_full_pinned_requests_one_pinned_arena(self):
        factory = RecordingFactory()
        store = FullPinnedWeightStore(
            tiny_plan(),
            tensor_factory=factory,
        )
        self.assertEqual(factory.pin_flags, [True])
        self.assertEqual(store.pinned_cpu_bytes, 12)
        store.close()

    def test_staging_has_pageable_store_and_two_pinned_slots(self):
        factory = RecordingFactory()
        plan = tiny_plan()
        store = PinnedStagingWeightStore(
            plan,
            tensor_factory=factory,
        )
        self.assertEqual(factory.pin_flags, [False, True, True])
        self.assertEqual(store.pinned_cpu_bytes, plan.two_slot_bytes)
        store.close()

    def test_staging_copies_unit_and_waits_before_reuse(self):
        factory = RecordingFactory()
        plan = tiny_plan()
        store = PinnedStagingWeightStore(
            plan,
            tensor_factory=factory,
        )
        store.arena[:4] = torch.tensor(
            [1.0, 2.0, 3.0, 4.0],
            dtype=torch.bfloat16,
        )
        event = FakeEvent()
        staged = store.prepare_unit(
            plan.units[0],
            0,
            reuse_event=event,
        ).result()
        self.assertTrue(event.synchronized)
        self.assertEqual(staged.float().tolist(), [1.0, 2.0, 3.0, 4.0])
        store.close()


if __name__ == "__main__":
    unittest.main()
