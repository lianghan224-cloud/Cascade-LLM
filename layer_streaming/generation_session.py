"""Single-request generation lifecycle over the streamed model executor."""

from dataclasses import dataclass
from enum import Enum
import threading

import torch

from .chat import SamplingConfig, first_stop_string, select_next_token


class GenerationSessionState(str, Enum):
    NEW = "new"
    PREFILLED = "prefilled"
    DECODING = "decoding"
    FINISHED = "finished"
    CANCELLED = "cancelled"
    CLOSED = "closed"


@dataclass(frozen=True)
class GenerationChunk:
    """One streamed token and the text that is now safe to publish."""

    token_id: int
    text_delta: str
    finished: bool
    finish_reason: str | None = None


def _normalize_token_ids(value):
    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(item) for item in value)


class GenerationSession:
    """Own one request's generation lifecycle; no batching or scheduling.

    ``prefill`` accepts a batch-one tensor, a flat sequence of token IDs, or
    text when a tokenizer was supplied. Calling it after generation appends a
    continuation to the existing KV cache; it never clears or recomputes the
    preceding context.
    """

    def __init__(
        self,
        executor,
        model_runtime,
        generation_config=None,
        eos_token_ids=None,
        generator=None,
        *,
        tokenizer=None,
        stop_token_ids=(),
        stop_strings=(),
        input_device=None,
    ):
        self.executor = executor
        self.model_runtime = model_runtime
        self.generation_config = (
            generation_config or SamplingConfig()
        ).validate()
        self.tokenizer = tokenizer
        tokenizer_eos = (
            _normalize_token_ids(getattr(tokenizer, "eos_token_id", None))
            if tokenizer is not None
            else ()
        )
        configured_eos = _normalize_token_ids(eos_token_ids)
        self.eos_token_ids = frozenset(
            tokenizer_eos if eos_token_ids is None else configured_eos
        )
        self.stop_token_ids = frozenset(_normalize_token_ids(stop_token_ids))
        self.stop_strings = tuple(str(item) for item in stop_strings if item)
        if self.stop_strings and tokenizer is None:
            raise ValueError("stop_strings require a tokenizer")
        self.generator = generator
        self.input_device = self._resolve_input_device(input_device)
        self.state = GenerationSessionState.NEW
        self.model_state = None
        self.pending_token = None
        self.token_history = []
        self.generated_tokens = []
        self.finish_reason = None
        self._emitted_text_length = 0
        self._stream_active = False
        self._active_eos_token_ids = self.eos_token_ids
        self._active_stop_token_ids = self.stop_token_ids
        self._active_stop_strings = self.stop_strings
        # One model step and its state publication form a lifecycle
        # transaction.  Cancel/reset/close wait for that transaction (and its
        # KV fences) instead of releasing a request between Transformer
        # layers.  RLock is required because an in-step error calls cancel.
        self._lifecycle_lock = threading.RLock()

    @property
    def kv_cache(self):
        return self.executor.kv_cache

    @property
    def kv_runtime(self):
        return self.kv_cache.runtime

    @property
    def bos_token_id(self):
        return getattr(self.tokenizer, "bos_token_id", None)

    def _ensure_open(self):
        if self.state == GenerationSessionState.CLOSED:
            raise RuntimeError("generation session is closed")
        if self.state == GenerationSessionState.CANCELLED:
            raise RuntimeError("generation session is cancelled; reset it first")

    def _resolve_input_device(self, input_device):
        if input_device is not None:
            return torch.device(input_device)
        vocab_runtime = getattr(self.executor, "vocab_runtime", None)
        if bool(getattr(vocab_runtime, "stream_embedding", False)):
            return torch.device("cpu")
        resident = getattr(self.executor, "resident", None)
        if isinstance(resident, dict):
            embedding = resident.get("model.embed_tokens.weight")
            if isinstance(embedding, torch.Tensor):
                return embedding.device
        return None

    def _ensure_not_streaming(self, operation):
        if self._stream_active:
            raise RuntimeError(f"cannot {operation} while stream is active")

    def _run(self, input_ids):
        with torch.inference_mode():
            state = self.executor.begin(input_ids)
            state = self.model_runtime.run(self.executor, state)
            return self.executor.finish(state)

    @staticmethod
    def _attach_cleanup_error(original_error, cleanup_error):
        try:
            current = tuple(getattr(original_error, "kv_cleanup_errors", ()))
            original_error.kv_cleanup_errors = current + (cleanup_error,)
        except BaseException:
            pass

    def _cancel_after_error(self, original_error):
        try:
            self.cancel()
        except BaseException as cleanup_error:
            self._attach_cleanup_error(original_error, cleanup_error)

    def encode(self, text, *, add_special_tokens=True):
        """Encode text through the configured tokenizer."""

        if self.tokenizer is None:
            raise RuntimeError("text input requires a tokenizer")
        token_ids = self.tokenizer.encode(
            str(text), add_special_tokens=bool(add_special_tokens)
        )
        return self._coerce_token_ids(token_ids)

    def decode(self, token_ids, *, skip_special_tokens=False):
        """Decode token IDs through the configured tokenizer."""

        if self.tokenizer is None:
            raise RuntimeError("text decoding requires a tokenizer")
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.reshape(-1).tolist()
        else:
            token_ids = list(token_ids)
        kwargs = {
            "skip_special_tokens": bool(skip_special_tokens),
            "clean_up_tokenization_spaces": False,
        }
        try:
            return str(self.tokenizer.decode(token_ids, **kwargs))
        except TypeError:
            kwargs.pop("clean_up_tokenization_spaces")
            return str(self.tokenizer.decode(token_ids, **kwargs))

    @staticmethod
    def _coerce_token_ids(value):
        if isinstance(value, torch.Tensor):
            input_ids = value
        else:
            input_ids = torch.as_tensor(value, dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("input IDs must be a batch-one tensor or flat sequence")
        if input_ids.numel() == 0:
            raise ValueError("input IDs must not be empty")
        return input_ids.to(dtype=torch.long)

    def _prepare_input_ids(self, value):
        if isinstance(value, str):
            prepared = self.encode(
                value,
                add_special_tokens=self.state == GenerationSessionState.NEW,
            )
        else:
            prepared = self._coerce_token_ids(value)
        if self.input_device is not None:
            prepared = prepared.to(self.input_device)
        return prepared

    def _start_generation_turn(self):
        self.generated_tokens.clear()
        self.finish_reason = None
        self._emitted_text_length = 0

    def prefill(self, input_ids):
        """Prefill an initial prompt or append a continuation to existing KV."""

        with self._lifecycle_lock:
            return self._prefill_locked(input_ids)

    def _prefill_locked(self, input_ids):

        self._ensure_open()
        self._ensure_not_streaming("prefill")
        prepared = self._prepare_input_ids(input_ids)
        continuation = self.state != GenerationSessionState.NEW
        submitted = prepared
        if continuation and self.pending_token is not None:
            submitted = torch.cat(
                (self.pending_token.to(prepared.device), prepared), dim=-1
            )
        try:
            self.model_state = self._run(submitted)
        except BaseException as original_error:
            self._cancel_after_error(original_error)
            raise
        self.token_history.extend(int(item) for item in prepared.reshape(-1).tolist())
        self.pending_token = None
        self._start_generation_turn()
        self.state = GenerationSessionState.PREFILLED
        return self.model_state

    def prefill_messages(self, messages):
        """Render an HF chat template and prefill only its new token suffix.

        For continuation, ``messages`` must describe the complete conversation
        represented by the existing token history plus the newly appended
        turn. A prefix mismatch is rejected instead of silently recomputing KV.
        """

        self._ensure_open()
        self._ensure_not_streaming("prefill messages")
        if self.tokenizer is None:
            raise RuntimeError("prefill_messages requires a tokenizer")
        if not getattr(self.tokenizer, "chat_template", None):
            raise RuntimeError("tokenizer has no chat template")
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        rendered = self._prepare_input_ids(rendered)
        if self.state == GenerationSessionState.NEW:
            return self.prefill(rendered)
        history = rendered.new_tensor(self.token_history).reshape(1, -1)
        if (
            rendered.shape[-1] < history.shape[-1]
            or not torch.equal(rendered[:, : history.shape[-1]], history)
        ):
            raise ValueError(
                "rendered chat does not extend the existing session token prefix"
            )
        suffix = rendered[:, history.shape[-1] :]
        if suffix.numel() == 0:
            raise ValueError("rendered chat contains no continuation tokens")
        return self.prefill(suffix)

    def continue_prefill(self, input_ids):
        """Append text or token IDs without clearing the existing KV cache."""

        self._ensure_open()
        if self.state == GenerationSessionState.NEW:
            raise RuntimeError("continue_prefill requires an existing context")
        return self.prefill(input_ids)

    def _sample(self):
        config = self.generation_config
        values = self.model_state.topk_values[..., : config.top_k]
        indices = self.model_state.topk_indices[..., : config.top_k]
        terminal_ids = self._active_eos_token_ids | self._active_stop_token_ids
        token = select_next_token(
            values,
            indices,
            temperature=config.temperature,
            top_p=config.top_p,
            min_p=config.min_p,
            repetition_penalty=config.repetition_penalty,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            token_history=self.token_history,
            repetition_window=config.repetition_window,
            banned_token_ids=(
                terminal_ids
                if len(self.generated_tokens) < config.min_new_tokens
                else None
            ),
            generator=self.generator,
        ).reshape(1, 1)
        return token

    def _decode_one(self):
        with self._lifecycle_lock:
            return self._decode_one_locked()

    def _decode_one_locked(self):
        self._ensure_open()
        if self.state not in {
            GenerationSessionState.PREFILLED,
            GenerationSessionState.DECODING,
        }:
            raise RuntimeError("decode_one requires a completed prefill")
        try:
            if self.pending_token is not None:
                self.model_state = self._run(self.pending_token)
            token = self._sample()
        except BaseException as original_error:
            self._cancel_after_error(original_error)
            raise
        token_id = int(token.item())
        self.pending_token = token
        self.generated_tokens.append(token_id)
        self.token_history.append(token_id)
        if token_id in self._active_eos_token_ids:
            self.finish_reason = "eos_token"
        elif token_id in self._active_stop_token_ids:
            self.finish_reason = "stop_token"
        else:
            self.finish_reason = None
        self.state = (
            GenerationSessionState.FINISHED
            if self.finish_reason is not None
            else GenerationSessionState.DECODING
        )
        return token

    def decode_one(self):
        self._ensure_not_streaming("decode")
        return self._decode_one()

    def _decoded_generated_text(self, *, omit_last=False):
        if self.tokenizer is None:
            return ""
        token_ids = self.generated_tokens[:-1] if omit_last else self.generated_tokens
        return self.decode(token_ids, skip_special_tokens=True)

    def _safe_text_end(self, text, stop_strings):
        hold = 0
        for stop in stop_strings:
            limit = min(len(text), len(stop) - 1)
            for width in range(1, limit + 1):
                if text.endswith(stop[:width]):
                    hold = max(hold, width)
        return len(text) - hold

    def _make_chunk(self, *, reached_limit):
        token_id = self.generated_tokens[-1]
        reason = self.finish_reason
        full_text = self._decoded_generated_text(omit_last=reason is not None)
        if (
            reason is None
            and len(self.generated_tokens)
            >= self.generation_config.min_new_tokens
        ):
            match = first_stop_string(full_text, self._active_stop_strings)
            if match is not None:
                end, _ = match
                full_text = full_text[:end]
                reason = "stop_string"
                self.finish_reason = reason
                self.state = GenerationSessionState.FINISHED
        if reason is None and reached_limit:
            reason = "length"
            self.finish_reason = reason
            self.state = GenerationSessionState.FINISHED
        safe_end = (
            len(full_text)
            if reason is not None
            else self._safe_text_end(full_text, self._active_stop_strings)
        )
        start = min(self._emitted_text_length, safe_end)
        text_delta = full_text[start:safe_end]
        self._emitted_text_length = safe_end
        return GenerationChunk(
            token_id=token_id,
            text_delta=text_delta,
            finished=reason is not None,
            finish_reason=reason,
        )

    def stream(
        self,
        input_ids=None,
        max_new_tokens=None,
        callback=None,
        *,
        stop_token_ids=None,
        stop_strings=None,
    ):
        """Yield :class:`GenerationChunk` objects for one generation turn."""

        self._ensure_open()
        self._ensure_not_streaming("start another stream")
        if input_ids is not None:
            self.prefill(input_ids)
        if self.state not in {
            GenerationSessionState.PREFILLED,
            GenerationSessionState.DECODING,
        }:
            raise RuntimeError("stream requires a completed prefill")
        limit = int(
            self.generation_config.max_new_tokens
            if max_new_tokens is None
            else max_new_tokens
        )
        if limit < 1:
            raise ValueError("max_new_tokens must be positive")
        active_stop_ids = (
            self.stop_token_ids
            if stop_token_ids is None
            else frozenset(_normalize_token_ids(stop_token_ids))
        )
        active_stop_strings = (
            self.stop_strings
            if stop_strings is None
            else tuple(str(item) for item in stop_strings if item)
        )
        if active_stop_strings and self.tokenizer is None:
            raise ValueError("stop_strings require a tokenizer")
        self._active_eos_token_ids = self.eos_token_ids
        self._active_stop_token_ids = active_stop_ids
        self._active_stop_strings = active_stop_strings
        self._stream_active = True
        try:
            for index in range(limit):
                token = self._decode_one()
                chunk = self._make_chunk(reached_limit=index + 1 == limit)
                if callback is not None:
                    callback(chunk)
                yield chunk
                if chunk.finished:
                    break
        except BaseException as original_error:
            self._cancel_after_error(original_error)
            raise
        finally:
            self._stream_active = False
            self._active_eos_token_ids = self.eos_token_ids
            self._active_stop_token_ids = self.stop_token_ids
            self._active_stop_strings = self.stop_strings

    def generate(self, input_ids=None, max_new_tokens=None, callback=None, **kwargs):
        """Generate tokens and return the traditional ``[1, N]`` tensor."""

        chunks = tuple(
            self.stream(input_ids, max_new_tokens, callback, **kwargs)
        )
        return torch.tensor(
            [[chunk.token_id for chunk in chunks]], dtype=torch.long
        )

    def cancel(self):
        with self._lifecycle_lock:
            return self._cancel_locked()

    def _cancel_locked(self):
        if self.state in {
            GenerationSessionState.CANCELLED,
            GenerationSessionState.CLOSED,
        }:
            return self
        cleanup_errors = []
        try:
            self.kv_runtime.quiesce()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            self.kv_cache.clear()
        except BaseException as error:
            cleanup_errors.append(error)
        finally:
            self.model_state = None
            self.pending_token = None
            self._start_generation_turn()
            self.state = GenerationSessionState.CANCELLED
        if cleanup_errors:
            raise cleanup_errors[0]
        return self

    def reset(self):
        with self._lifecycle_lock:
            return self._reset_locked()

    def _reset_locked(self):
        if self.state == GenerationSessionState.CLOSED:
            raise RuntimeError("generation session is closed")
        if self.state == GenerationSessionState.NEW:
            return self
        self.kv_runtime.quiesce()
        self.kv_cache.clear()
        self.model_state = None
        self.pending_token = None
        self.token_history.clear()
        self._start_generation_turn()
        self.state = GenerationSessionState.NEW
        return self

    def close(self):
        with self._lifecycle_lock:
            return self._close_locked()

    def _close_locked(self):
        if self.state == GenerationSessionState.CLOSED:
            return self
        cleanup_errors = []
        try:
            self.kv_runtime.quiesce()
        except BaseException as error:
            cleanup_errors.append(error)
        try:
            self.kv_cache.close()
        except BaseException as error:
            cleanup_errors.append(error)
        finally:
            self.model_state = None
            self.pending_token = None
            self._start_generation_turn()
            self.state = GenerationSessionState.CLOSED
        if cleanup_errors:
            raise cleanup_errors[0]
        return self

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
