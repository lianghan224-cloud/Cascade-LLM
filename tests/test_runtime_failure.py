from concurrent.futures import Future
from types import SimpleNamespace
import unittest

from layer_streaming import (
    PipelineError,
    PipelineRuntimeCore,
    PipelineWorkerError,
)


class FailingStore:
    mode = SimpleNamespace(value="pinned_staging")
    slot_count = 2

    def __init__(self, failed_unit):
        self.failed_unit = failed_unit

    def prepare_unit(self, unit, slot_index, reuse_event=None):
        del slot_index, reuse_event
        future = Future()
        if unit.unit_id == self.failed_unit:
            future.set_exception(ValueError("synthetic source failure"))
        else:
            future.set_result(unit.unit_id)
        return future


class TimeoutStore(FailingStore):
    def prepare_unit(self, unit, slot_index, reuse_event=None):
        del unit, slot_index, reuse_event
        return Future()


class PipelineFailureTest(unittest.TestCase):
    def make_plan(self):
        return SimpleNamespace(
            units=tuple(
                SimpleNamespace(unit_id="unit-{}".format(index))
                for index in range(6)
            )
        )

    def test_worker_failure_reaches_main_thread_and_closes(self):
        pipeline = PipelineRuntimeCore(
            plan=self.make_plan(),
            store=FailingStore("unit-2"),
            slot_count=2,
            prefetch_depth=2,
            source_timeout_seconds=1.0,
            submit_copy=lambda prepared, lease: object(),
        )
        try:
            with self.assertRaises(PipelineWorkerError) as captured:
                pipeline.run(
                    lambda ready, state: (state, object()), [], 1.0
                )
            self.assertIn(
                captured.exception.phase,
                {"source_producer", "h2d_scheduler"},
            )
            self.assertIn("synthetic source failure", str(captured.exception))
            with self.assertRaisesRegex(PipelineError, "after failure"):
                pipeline.run(lambda ready, state: (state, object()), [], 1.0)
        finally:
            pipeline.close()
        self.assertFalse(pipeline.producer.thread.is_alive())
        self.assertFalse(pipeline.scheduler.thread.is_alive())

    def test_compute_callback_failure_marks_runtime_unusable(self):
        pipeline = PipelineRuntimeCore(
            plan=self.make_plan(),
            store=FailingStore("never"),
            slot_count=2,
            prefetch_depth=2,
            source_timeout_seconds=1.0,
            submit_copy=lambda prepared, lease: object(),
        )
        try:
            def fail(ready, state):
                raise RuntimeError("synthetic callback failure")

            with self.assertRaisesRegex(RuntimeError, "callback failure"):
                pipeline.run(fail, [], 1.0)
            with self.assertRaises(PipelineError):
                pipeline.run(fail, [], 1.0)
        finally:
            pipeline.close()

    def test_timeout_closes_workers_and_new_runtime_recovers(self):
        timed_out = PipelineRuntimeCore(
            plan=self.make_plan(),
            store=TimeoutStore("unused"),
            slot_count=2,
            prefetch_depth=2,
            source_timeout_seconds=0.01,
            submit_copy=lambda prepared, lease: object(),
        )
        try:
            with self.assertRaises(PipelineWorkerError) as captured:
                timed_out.run(
                    lambda ready, state: (state, object()), [], 1.0
                )
            self.assertIsInstance(
                captured.exception.original_error, TimeoutError
            )
        finally:
            timed_out.close()
        self.assertEqual(
            timed_out.resource_stats()["worker_thread_count"], 0
        )

        recovered = PipelineRuntimeCore(
            plan=self.make_plan(),
            store=FailingStore("never"),
            slot_count=2,
            prefetch_depth=2,
            source_timeout_seconds=1.0,
            submit_copy=lambda prepared, lease: object(),
        )
        try:
            state, _ = recovered.run(
                lambda ready, state: (state + [ready.index], object()),
                [],
                1.0,
            )
            self.assertEqual(state, list(range(6)))
        finally:
            recovered.close()


if __name__ == "__main__":
    unittest.main()
