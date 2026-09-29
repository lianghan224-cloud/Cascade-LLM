from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from layer_streaming import (
    GenerationChunk,
    GenerationSession,
    GenerationSessionState,
    SamplingConfig,
)


class FakeKVRuntime:
    def __init__(self):
        self.quiesce_calls = 0

    def quiesce(self):
        self.quiesce_calls += 1


class FakeKVCache:
    def __init__(self):
        self.runtime = FakeKVRuntime()
        self.clear_calls = 0
        self.close_calls = 0

    def clear(self):
        self.clear_calls += 1

    def close(self):
        self.close_calls += 1


class FakeExecutor:
    def __init__(self, tokens=(5, 6, 7)):
        self.kv_cache = FakeKVCache()
        self.tokens = tuple(tokens)
        self.finish_calls = 0

    def begin(self, input_ids):
        return SimpleNamespace(input_ids=input_ids)

    def finish(self, state):
        token = self.tokens[min(self.finish_calls, len(self.tokens) - 1)]
        self.finish_calls += 1
        state.topk_values = torch.tensor([[[3.0, 2.0]]])
        state.topk_indices = torch.tensor([[[token, token + 10]]])
        return state


class FakeModelRuntime:
    def __init__(self):
        self.run_calls = 0

    def run(self, executor, state):
        del executor
        self.run_calls += 1
        return state


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    chat_template = "fake-template"

    def __init__(self):
        self.pieces = {
            1: "<bos>",
            2: "<eos>",
            5: "hel",
            6: "lo</",
            7: "stop>",
            8: "tail",
            9: "hello</stop>",
            10: "hello</st",
            11: "op>",
            12: "hello<",
            13: "/stop>",
        }
        self.rendered = None

    def encode(self, text, add_special_tokens=True):
        values = [ord(character) for character in text]
        return ([self.bos_token_id] if add_special_tokens else []) + values

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        del kwargs
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in {
                self.bos_token_id,
                self.eos_token_id,
            }:
                continue
            pieces.append(self.pieces.get(token_id, chr(token_id)))
        return "".join(pieces)

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        if self.rendered is None:
            raise RuntimeError("test did not configure rendered chat")
        return torch.tensor([self.rendered])


class GenerationSessionTest(unittest.TestCase):
    def make_session(self, tokens=(5, 6, 7), eos=(6,)):
        executor = FakeExecutor(tokens)
        model_runtime = FakeModelRuntime()
        session = GenerationSession(
            executor,
            model_runtime,
            SamplingConfig(max_new_tokens=8),
            eos_token_ids=eos,
        )
        return session, executor, model_runtime

    def test_generate_streams_until_eos_without_duplicate_model_step(self):
        session, executor, model_runtime = self.make_session()
        callbacks = []
        output = session.generate(
            torch.tensor([[1, 2, 3]]), callback=callbacks.append
        )
        self.assertEqual(output.tolist(), [[5, 6]])
        self.assertTrue(all(isinstance(item, GenerationChunk) for item in callbacks))
        self.assertEqual([item.token_id for item in callbacks], [5, 6])
        self.assertFalse(callbacks[0].finished)
        self.assertEqual(callbacks[1].finish_reason, "eos_token")
        self.assertEqual(model_runtime.run_calls, 2)
        self.assertEqual(session.state, GenerationSessionState.FINISHED)
        self.assertEqual(session.token_history, [1, 2, 3, 5, 6])
        session.close()
        session.close()
        self.assertEqual(executor.kv_cache.close_calls, 1)

    def test_cancel_reset_and_continuation_cleanup(self):
        session, executor, _ = self.make_session(eos=())
        session.prefill(torch.tensor([[1, 2]]))
        session.decode_one()
        session.cancel()
        self.assertEqual(session.state, GenerationSessionState.CANCELLED)
        self.assertEqual(executor.kv_cache.clear_calls, 1)
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            session.decode_one()
        session.reset()
        self.assertEqual(session.state, GenerationSessionState.NEW)
        session.prefill(torch.tensor([[3, 4]]))
        session.continue_prefill(torch.tensor([[8, 9]]))
        self.assertEqual(session.token_history, [3, 4, 8, 9])
        session.close()

    def test_exception_cancels_and_cleans_request(self):
        session, executor, model_runtime = self.make_session()

        def fail(executor_arg, state):
            del executor_arg, state
            raise RuntimeError("injected model failure")

        model_runtime.run = fail
        with self.assertRaisesRegex(RuntimeError, "injected"):
            session.prefill(torch.tensor([[1]]))
        self.assertEqual(session.state, GenerationSessionState.CANCELLED)
        self.assertEqual(executor.kv_cache.clear_calls, 1)

    def test_cleanup_failure_does_not_mask_model_failure(self):
        session, executor, model_runtime = self.make_session()

        def fail_run(executor_arg, state):
            del executor_arg, state
            raise RuntimeError("primary model failure")

        def fail_clear():
            raise ValueError("secondary cleanup failure")

        model_runtime.run = fail_run
        executor.kv_cache.clear = fail_clear
        with self.assertRaisesRegex(RuntimeError, "primary") as raised:
            session.prefill(torch.tensor([[1]]))
        self.assertEqual(session.state, GenerationSessionState.CANCELLED)
        self.assertEqual(len(raised.exception.kv_cleanup_errors), 1)

    def test_sampling_honors_top_k_and_minimum_before_eos(self):
        executor = FakeExecutor(tokens=(6,))
        runtime = FakeModelRuntime()
        config = SamplingConfig(top_k=1, min_new_tokens=1, max_new_tokens=2)
        session = GenerationSession(
            executor, runtime, config, eos_token_ids=(6,)
        )
        session.prefill(torch.tensor([[1]]))
        captured = {}

        def select(values, indices, **kwargs):
            captured["shape"] = tuple(indices.shape)
            captured["banned"] = kwargs["banned_token_ids"]
            return torch.tensor([[5]])

        with patch("layer_streaming.generation_session.select_next_token", select):
            session.decode_one()
        self.assertEqual(captured["shape"][-1], 1)
        self.assertEqual(captured["banned"], frozenset((6,)))
        session.close()

    def test_tokenizer_text_encode_decode_and_inferred_eos(self):
        tokenizer = FakeTokenizer()
        executor = FakeExecutor(tokens=(2,))
        session = GenerationSession(
            executor,
            FakeModelRuntime(),
            SamplingConfig(top_k=1),
            tokenizer=tokenizer,
        )
        self.assertEqual(session.bos_token_id, 1)
        session.prefill("AB")
        self.assertEqual(session.token_history, [1, 65, 66])
        chunk = tuple(session.stream(max_new_tokens=2))[0]
        self.assertEqual(chunk.finish_reason, "eos_token")
        self.assertEqual(chunk.text_delta, "")
        session.close()

    def test_stop_token_and_cross_token_stop_string(self):
        tokenizer = FakeTokenizer()
        session = GenerationSession(
            FakeExecutor(tokens=(5, 6, 7, 8)),
            FakeModelRuntime(),
            SamplingConfig(top_k=1, max_new_tokens=8),
            tokenizer=tokenizer,
            stop_strings=("</stop>",),
        )
        chunks = tuple(session.stream(torch.tensor([[9]])))
        self.assertEqual([chunk.token_id for chunk in chunks], [5, 6, 7])
        self.assertEqual("".join(chunk.text_delta for chunk in chunks), "hello")
        self.assertEqual(chunks[-1].finish_reason, "stop_string")
        self.assertTrue(chunks[-1].finished)
        session.reset()
        session.stop_token_ids = frozenset((6,))
        chunks = tuple(session.stream(torch.tensor([[9]])))
        self.assertEqual([chunk.token_id for chunk in chunks], [8, 8, 8, 8, 8, 8, 8, 8])
        self.assertEqual(chunks[-1].finish_reason, "length")
        session.close()

    def test_explicit_stop_token_finishes_without_emitting_token_text(self):
        tokenizer = FakeTokenizer()
        session = GenerationSession(
            FakeExecutor(tokens=(5, 6)),
            FakeModelRuntime(),
            SamplingConfig(top_k=1),
            tokenizer=tokenizer,
            stop_token_ids=(6,),
        )
        chunks = tuple(session.stream(torch.tensor([[9]])))
        self.assertEqual([chunk.token_id for chunk in chunks], [5, 6])
        self.assertEqual("".join(chunk.text_delta for chunk in chunks), "hel")
        self.assertEqual(chunks[-1].finish_reason, "stop_token")

    def test_chat_template_and_suffix_only_continuation(self):
        tokenizer = FakeTokenizer()
        tokenizer.rendered = [1, 20, 21]
        session = GenerationSession(
            FakeExecutor(tokens=(5,)),
            FakeModelRuntime(),
            SamplingConfig(top_k=1),
            tokenizer=tokenizer,
            eos_token_ids=(),
        )
        session.prefill_messages([{"role": "user", "content": "one"}])
        session.decode_one()
        tokenizer.rendered = [1, 20, 21, 5, 30, 31]
        session.prefill_messages([{"role": "user", "content": "two"}])
        self.assertEqual(session.token_history, [1, 20, 21, 5, 30, 31])
        submitted = session.model_state.input_ids.tolist()
        self.assertEqual(submitted, [[5, 30, 31]])
        session.close()

    def test_missing_chat_template_and_prefix_mismatch_are_explicit(self):
        tokenizer = FakeTokenizer()
        tokenizer.chat_template = None
        session = GenerationSession(
            FakeExecutor(), FakeModelRuntime(), tokenizer=tokenizer
        )
        with self.assertRaisesRegex(RuntimeError, "no chat template"):
            session.prefill_messages([])
        tokenizer.chat_template = "fake"
        tokenizer.rendered = [1, 10]
        session.prefill_messages([])
        tokenizer.rendered = [1, 11, 12]
        with self.assertRaisesRegex(ValueError, "does not extend"):
            session.prefill_messages([])
        session.close()

    def test_reset_cancel_close_idempotence_and_illegal_transitions(self):
        session, executor, _ = self.make_session(eos=())
        session.reset()
        session.reset()
        self.assertEqual(executor.kv_cache.clear_calls, 0)
        with self.assertRaisesRegex(RuntimeError, "completed prefill"):
            session.decode_one()
        with self.assertRaisesRegex(RuntimeError, "existing context"):
            session.continue_prefill(torch.tensor([[1]]))
        session.cancel()
        session.cancel()
        self.assertEqual(executor.kv_cache.clear_calls, 1)
        session.reset()
        session.reset()
        self.assertEqual(executor.kv_cache.clear_calls, 2)
        session.close()
        session.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.reset()

    def test_stream_callback_failure_cancels_and_quiesces(self):
        session, executor, _ = self.make_session(eos=())

        def fail_callback(chunk):
            del chunk
            raise RuntimeError("consumer failure")

        with self.assertRaisesRegex(RuntimeError, "consumer failure"):
            tuple(
                session.stream(
                    torch.tensor([[1]]),
                    max_new_tokens=2,
                    callback=fail_callback,
                )
            )
        self.assertEqual(session.state, GenerationSessionState.CANCELLED)
        self.assertEqual(executor.kv_cache.runtime.quiesce_calls, 1)
        self.assertEqual(executor.kv_cache.clear_calls, 1)

    def test_close_quiesces_pending_work_before_cache_close(self):
        session, executor, _ = self.make_session(eos=())
        session.prefill(torch.tensor([[1]]))
        session.decode_one()
        session.close()
        self.assertEqual(executor.kv_cache.runtime.quiesce_calls, 1)
        self.assertEqual(executor.kv_cache.close_calls, 1)

    def test_eos_id_normalization_variants(self):
        for eos_value in (2, (2,), [2], frozenset((2,))):
            with self.subTest(eos_value=eos_value):
                session = GenerationSession(
                    FakeExecutor(tokens=(2,)),
                    FakeModelRuntime(),
                    SamplingConfig(top_k=1, max_new_tokens=2),
                    tokenizer=FakeTokenizer(),
                    eos_token_ids=eos_value,
                )
                chunks = tuple(session.stream(torch.tensor([[9]])))
                self.assertEqual([item.token_id for item in chunks], [2])
                self.assertTrue(chunks[0].finished)
                self.assertEqual(chunks[0].finish_reason, "eos_token")
                session.close()

    def test_stop_token_configuration_and_stream_override_variants(self):
        cases = (
            {"configured": 6, "override": None},
            {"configured": (6,), "override": None},
            {"configured": (), "override": 6},
            {"configured": (), "override": [6]},
        )
        for case in cases:
            with self.subTest(**case):
                session = GenerationSession(
                    FakeExecutor(tokens=(5, 6, 8)),
                    FakeModelRuntime(),
                    SamplingConfig(top_k=1, max_new_tokens=4),
                    tokenizer=FakeTokenizer(),
                    eos_token_ids=(),
                    stop_token_ids=case["configured"],
                )
                kwargs = {}
                if case["override"] is not None:
                    kwargs["stop_token_ids"] = case["override"]
                chunks = tuple(session.stream(torch.tensor([[9]]), **kwargs))
                self.assertEqual([item.token_id for item in chunks], [5, 6])
                self.assertEqual(chunks[-1].finish_reason, "stop_token")
                self.assertEqual("".join(item.text_delta for item in chunks), "hel")
                session.close()

    def test_stop_string_is_detected_across_multiple_token_boundaries(self):
        tokenizations = (
            (9,),
            (10, 11),
            (12, 13),
            (5, 6, 7),
        )
        for tokenization in tokenizations:
            with self.subTest(tokenization=tokenization):
                session = GenerationSession(
                    FakeExecutor(tokens=tokenization + (8,)),
                    FakeModelRuntime(),
                    SamplingConfig(top_k=1, max_new_tokens=8),
                    tokenizer=FakeTokenizer(),
                    eos_token_ids=(),
                    stop_strings=("</stop>",),
                )
                chunks = tuple(session.stream(torch.tensor([[1]])))
                self.assertEqual(
                    [item.token_id for item in chunks], list(tokenization)
                )
                self.assertEqual(
                    "".join(item.text_delta for item in chunks), "hello"
                )
                self.assertEqual(chunks[-1].finish_reason, "stop_string")
                session.close()


if __name__ == "__main__":
    unittest.main()
