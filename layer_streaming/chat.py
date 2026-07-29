"""Helpers shared by the interactive Llama 3.1 chat frontend."""

from collections import Counter
from dataclasses import dataclass

import torch


@dataclass
class SamplingConfig:
    """Mutable per-session generation settings."""

    temperature: float = 0.0
    top_k: int = 10
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_window: int = 256
    min_new_tokens: int = 0
    max_new_tokens: int = 64

    def validate(self):
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if not 0 <= self.min_p < 1:
            raise ValueError("min_p must be in [0, 1)")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.repetition_window < 0:
            raise ValueError("repetition_window must be non-negative")
        if self.min_new_tokens < 0:
            raise ValueError("min_new_tokens must be non-negative")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.min_new_tokens > self.max_new_tokens:
            raise ValueError(
                "min_new_tokens must not exceed max_new_tokens"
            )
        return self

    def as_dict(self):
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_window": self.repetition_window,
            "min_new_tokens": self.min_new_tokens,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True)
class RenderedChat:
    """A chat prompt after dropping any history that does not fit."""

    messages: tuple
    input_ids: torch.Tensor
    dropped_messages: int


def _normalize_token_ids(value):
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    return tuple(int(item) for item in value)


def collect_stop_token_ids(config, tokenizer):
    """Return all configured Llama end-of-sequence/end-of-turn token IDs."""

    token_ids = set(_normalize_token_ids(getattr(config, "eos_token_id", None)))
    token_ids.update(
        _normalize_token_ids(getattr(tokenizer, "eos_token_id", None))
    )
    for token in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            token_id is not None
            and token_id != getattr(tokenizer, "unk_token_id", None)
        ):
            token_ids.add(int(token_id))
    return frozenset(token_ids)


def select_next_token(
    topk_values,
    topk_indices,
    temperature=0.0,
    top_p=1.0,
    min_p=0.0,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    token_history=None,
    repetition_window=256,
    banned_token_ids=None,
    generator=None,
):
    """Select one token after applying penalties and truncated sampling.

    The input candidates are the exact global top-k produced by the streamed
    LM head. Penalties and top-p/min-p filtering therefore operate within that
    candidate set rather than over a materialized full vocabulary.
    """

    temperature = float(temperature)
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    top_p = float(top_p)
    min_p = float(min_p)
    repetition_penalty = float(repetition_penalty)
    presence_penalty = float(presence_penalty)
    frequency_penalty = float(frequency_penalty)
    repetition_window = int(repetition_window)
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not 0 <= min_p < 1:
        raise ValueError("min_p must be in [0, 1)")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if repetition_window < 0:
        raise ValueError("repetition_window must be non-negative")
    if topk_values.shape != topk_indices.shape:
        raise ValueError("top-k values and indices must have the same shape")
    if topk_values.shape[-1] < 1:
        raise ValueError("top-k tensors must not be empty")

    logits = topk_values.float().clone()
    history = list(token_history or ())
    if repetition_window:
        history = history[-repetition_window:]
    else:
        history = []
    counts = Counter(int(token_id) for token_id in history)
    if counts:
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_indices = topk_indices.reshape(-1, topk_indices.shape[-1])
        for row in range(flat_logits.shape[0]):
            for column, token_id in enumerate(flat_indices[row].tolist()):
                count = counts.get(int(token_id), 0)
                if not count:
                    continue
                value = flat_logits[row, column]
                if repetition_penalty != 1:
                    flat_logits[row, column] = torch.where(
                        value < 0,
                        value * repetition_penalty,
                        value / repetition_penalty,
                    )
                flat_logits[row, column] -= (
                    presence_penalty + frequency_penalty * count
                )

    banned = {int(token_id) for token_id in (banned_token_ids or ())}
    if banned:
        for token_id in banned:
            logits.masked_fill_(topk_indices == token_id, float("-inf"))
    if not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError(
            "all available top-k candidates were banned; increase top_k"
        )

    if temperature == 0:
        positions = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.gather(topk_indices, -1, positions).squeeze(-1)

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if min_p:
        threshold = probabilities.amax(dim=-1, keepdim=True) * min_p
        probabilities = probabilities.masked_fill(
            probabilities < threshold,
            0,
        )
    if top_p < 1:
        sorted_probabilities, sorted_positions = torch.sort(
            probabilities,
            dim=-1,
            descending=True,
        )
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        probabilities = torch.zeros_like(probabilities).scatter(
            -1,
            sorted_positions,
            sorted_probabilities,
        )
    probability_sum = probabilities.sum(dim=-1, keepdim=True)
    if not (probability_sum > 0).all():
        raise ValueError("sampling filters removed every candidate")
    probabilities = probabilities / probability_sum
    width = probabilities.shape[-1]
    sampled = torch.multinomial(
        probabilities.reshape(-1, width),
        num_samples=1,
        generator=generator,
    ).view(probabilities.shape[:-1])
    return torch.gather(
        topk_indices,
        dim=-1,
        index=sampled.unsqueeze(-1),
    ).squeeze(-1)


def first_stop_string(text, stop_strings):
    """Return the earliest custom stop occurrence, preferring longer ties."""

    best = None
    for stop in stop_strings:
        if not stop:
            continue
        position = text.find(stop)
        if position < 0:
            continue
        candidate = (position, -len(stop), stop)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    return best[0], best[2]


def render_chat_prompt(
    tokenizer,
    messages,
    max_context_tokens,
    max_new_tokens,
):
    """Apply the model chat template and prune oldest complete turns.

    The newest user message and optional system message are always preserved.
    """

    max_context_tokens = int(max_context_tokens)
    max_new_tokens = int(max_new_tokens)
    if max_context_tokens < 1:
        raise ValueError("max_context_tokens must be positive")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    prompt_budget = max_context_tokens - max_new_tokens
    if prompt_budget < 1:
        raise ValueError(
            "max_context_tokens must be larger than max_new_tokens"
        )

    working = [dict(message) for message in messages]
    if not working or working[-1].get("role") != "user":
        raise ValueError("chat prompt must end with a user message")
    system_offset = (
        1 if working and working[0].get("role") == "system" else 0
    )
    dropped = 0
    while True:
        input_ids = tokenizer.apply_chat_template(
            working,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if input_ids.shape[-1] <= prompt_budget:
            return RenderedChat(
                messages=tuple(working),
                input_ids=input_ids,
                dropped_messages=dropped,
            )

        # Before the newest user message, history consists of complete
        # user/assistant pairs. Drop one oldest pair and render again.
        removable = len(working) - system_offset - 1
        if removable < 2:
            raise ValueError(
                "the system prompt and newest user message exceed the "
                "available context budget"
            )
        del working[system_offset : system_offset + 2]
        dropped += 2
