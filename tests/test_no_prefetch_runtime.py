from concurrent.futures import Future
from types import SimpleNamespace
import unittest

from layer_streaming.mixed_runtime import MixedDtypeRuntime


class _Event:
    def __init__(self, name, trace):
        self.name = name
        self.trace = trace

    def synchronize(self):
        self.trace.append(("sync", self.name))


class _Store:
    def __init__(self, trace):
        self.trace = trace

    def prepare_unit(self, unit, slot_index, reuse_event=None):
        self.trace.append(
            (
                "prepare",
                unit.unit_id,
                slot_index,
                None if reuse_event is None else reuse_event.name,
            )
        )
        future = Future()
        future.set_result(unit.unit_id)
        return future


class NoPrefetchRuntimeTest(unittest.TestCase):
    def test_depth_zero_serializes_source_copy_and_compute_on_slot_zero(self):
        trace = []
        runtime = MixedDtypeRuntime.__new__(MixedDtypeRuntime)
        runtime.streamed_units = tuple(
            SimpleNamespace(unit_id="u{}".format(index)) for index in range(3)
        )
        runtime.store = _Store(trace)
        runtime.source_timeout_seconds = 1.0
        runtime.slot_count = 3
        runtime.pipeline = SimpleNamespace(last_stats=None)

        def submit_copy(prepared, lease):
            trace.append(
                (
                    "copy",
                    prepared.unit.unit_id,
                    lease.slot_index,
                    None
                    if lease.reuse_event is None
                    else lease.reuse_event.name,
                )
            )
            return _Event("ready-{}".format(prepared.index), trace)

        def consume(ready, _compute_unit, state):
            trace.append(("compute", ready.unit.unit_id))
            return state + [ready.unit.unit_id], _Event(
                "free-{}".format(ready.index), trace
            )

        runtime._submit_copy = submit_copy
        runtime._consume = consume
        state, final_event = runtime._run_without_prefetch(object(), [])
        self.assertEqual(state, ["u0", "u1", "u2"])
        self.assertEqual(final_event.name, "free-2")
        self.assertEqual(
            runtime.pipeline.last_stats["transfer_slot_reuse_counts"],
            [3, 0, 0],
        )
        self.assertTrue(runtime.pipeline.last_stats["prefetch_disabled"])
        for index in (1, 2):
            prepare_position = trace.index(
                ("prepare", "u{}".format(index), 0, "ready-{}".format(index - 1))
            )
            sync_position = trace.index(("sync", "free-{}".format(index - 1)))
            self.assertLess(sync_position, prepare_position)


if __name__ == "__main__":
    unittest.main()
