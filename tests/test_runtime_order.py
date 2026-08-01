from concurrent.futures import Future
from types import SimpleNamespace
import unittest

from layer_streaming import PipelineRuntimeCore


class ImmediateStore:
    mode = SimpleNamespace(value="full_pinned")

    def prepare_unit(self, unit, slot_index, reuse_event=None):
        del slot_index, reuse_event
        future = Future()
        future.set_result(unit.unit_id)
        return future


class PipelineOrderTest(unittest.TestCase):
    def test_bounded_pipeline_preserves_compute_order(self):
        units = tuple(
            SimpleNamespace(unit_id="unit-{}".format(index))
            for index in range(12)
        )
        plan = SimpleNamespace(units=units)
        copied = []

        def submit_copy(prepared, lease):
            copied.append((prepared.index, lease.slot_index, prepared.source))
            return "ready-{}".format(prepared.index)

        pipeline = PipelineRuntimeCore(
            plan=plan,
            store=ImmediateStore(),
            slot_count=2,
            prefetch_depth=3,
            source_timeout_seconds=2.0,
            submit_copy=submit_copy,
        )
        try:
            def consume(ready, state):
                self.assertEqual(
                    ready.ready_event, "ready-{}".format(ready.index)
                )
                state.append(ready.unit.unit_id)
                return state, "free-{}".format(ready.index)

            state, final_event = pipeline.run(consume, [], 2.0)
            self.assertEqual(state, [unit.unit_id for unit in units])
            self.assertEqual(final_event, "free-11")
            self.assertEqual([item[0] for item in copied], list(range(12)))
            self.assertEqual(
                pipeline.last_stats["source_queue_capacity"], 3
            )
            self.assertLessEqual(
                pipeline.last_stats["source_queue_max_depth"], 3
            )
            self.assertLessEqual(
                pipeline.last_stats["ready_queue_max_depth"], 3
            )
        finally:
            pipeline.close()
        self.assertFalse(pipeline.producer.thread.is_alive())
        self.assertFalse(pipeline.scheduler.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
