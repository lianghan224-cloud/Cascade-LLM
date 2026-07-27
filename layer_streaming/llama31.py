"""Llama-3.1 8B/70B matrix-wise decode executor."""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


LLAMA31_8B_MODEL_ID = "meta-llama/Llama-3.1-8B"
MATRIX_ORDER = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def _rms_norm(hidden_states, weight, epsilon):
    input_dtype = hidden_states.dtype
    variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
    normalized = hidden_states.float() * torch.rsqrt(variance + epsilon)
    return weight * normalized.to(input_dtype)


def _repeat_kv(hidden_states, groups):
    if groups == 1:
        return hidden_states
    return hidden_states.repeat_interleave(groups, dim=1)


class SimpleKVCache:
    """GPU-resident, single-request cache used by the first runtime version."""

    def __init__(self, layer_count=32):
        self.entries = [None] * layer_count

    def sequence_length(self):
        first = self.entries[0]
        return 0 if first is None else first[0].shape[-2]

    def append(self, layer_index, key, value):
        previous = self.entries[layer_index]
        if previous is not None:
            key = torch.cat((previous[0], key), dim=-2)
            value = torch.cat((previous[1], value), dim=-2)
        self.entries[layer_index] = (key, value)
        return key, value

    def clear(self):
        for index in range(len(self.entries)):
            self.entries[index] = None


@dataclass
class DecodeState:
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    layer_values: dict = field(default_factory=dict)
    logits: object = None
    topk_values: object = None
    topk_indices: object = None


class Llama31DecodeExecutor:
    """Stateful callbacks consumed by :class:`DoubleBufferRuntime`.

    This implementation targets an unpadded single request. It supports a
    multi-token initial prefill and subsequent one-token decode calls.
    """

    def __init__(
        self,
        config,
        resident,
        kv_cache=None,
        vocab_runtime=None,
        top_k=10,
        return_full_logits=False,
    ):
        try:
            from transformers.models.llama.modeling_llama import (
                LlamaRotaryEmbedding,
                apply_rotary_pos_emb,
            )
        except ImportError as error:
            raise RuntimeError(
                "transformers is required for the Llama executor"
            ) from error

        geometry = (
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.vocab_size,
        )
        supported = {
            (4096, 14336, 32, 32, 8, 128256),
            (8192, 28672, 80, 64, 8, 128256),
        }
        if geometry not in supported:
            raise ValueError(
                "unsupported Llama-3.1 geometry {}".format(geometry)
            )
        self.config = config
        self.resident = resident
        self.vocab_runtime = vocab_runtime
        self.top_k = int(top_k)
        self.return_full_logits = bool(return_full_logits)
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        self.kv_cache = kv_cache or SimpleKVCache(
            config.num_hidden_layers
        )
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        device = resident.device
        self.rotary = LlamaRotaryEmbedding(config=config, device=device)
        self._apply_rotary_pos_emb = apply_rotary_pos_emb

    def begin(self, input_ids, position_ids=None):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("first runtime supports one unpadded request")
        past = self.kv_cache.sequence_length()
        sequence = input_ids.shape[1]
        if past and sequence != 1:
            raise ValueError(
                "after prefill, only one-token decode calls are supported"
            )
        if position_ids is None:
            position_ids = torch.arange(
                past,
                past + sequence,
                dtype=torch.long,
                device=input_ids.device,
            ).unsqueeze(0)
        if self.vocab_runtime is None:
            hidden = F.embedding(
                input_ids,
                self.resident["model.embed_tokens.weight"],
            )
        else:
            hidden = self.vocab_runtime.embedding(input_ids)
        return DecodeState(
            hidden_states=hidden,
            position_ids=position_ids,
        )

    def _matrix_key(self, layer_index, operation):
        block = "self_attn" if operation in {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        } else "mlp"
        return "model.layers.{}.{}.{}.weight".format(
            layer_index,
            block,
            operation,
        )

    def _start_attention(self, layer_index, state):
        prefix = "model.layers.{}".format(layer_index)
        state.layer_values = {
            "residual": state.hidden_states,
            "normalized": _rms_norm(
                state.hidden_states,
                self.resident[
                    "{}.input_layernorm.weight".format(prefix)
                ],
                self.config.rms_norm_eps,
            ),
        }

    def _attention_output(self, layer_index, state):
        values = state.layer_values
        query = values["q"]
        key = values["k"]
        value = values["v"]
        batch, query_length, _ = query.shape
        query = query.view(
            batch,
            query_length,
            self.config.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.view(
            batch,
            query_length,
            self.config.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch,
            query_length,
            self.config.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        cos, sin = self.rotary(value, state.position_ids)
        query, key = self._apply_rotary_pos_emb(
            query,
            key,
            cos,
            sin,
        )
        had_past = self.kv_cache.entries[layer_index] is not None
        key, value = self.kv_cache.append(
            layer_index,
            key,
            value,
        )
        key = _repeat_kv(key, self.kv_groups)
        value = _repeat_kv(value, self.kv_groups)
        if had_past and query_length != 1:
            raise ValueError("chunked decode with a past cache is unsupported")
        attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=(not had_past and query_length > 1),
        )
        return attention.transpose(1, 2).contiguous().view(
            batch,
            query_length,
            self.config.hidden_size,
        )

    def _execute_matrix(self, layer_index, operation, weight, state):
        values = state.layer_values
        if operation == "q_proj":
            self._start_attention(layer_index, state)
            values = state.layer_values
            values["q"] = F.linear(values["normalized"], weight)
        elif operation == "k_proj":
            values["k"] = F.linear(values["normalized"], weight)
        elif operation == "v_proj":
            values["v"] = F.linear(values["normalized"], weight)
        elif operation == "o_proj":
            attention = self._attention_output(layer_index, state)
            state.hidden_states = values["residual"] + F.linear(
                attention,
                weight,
            )
        elif operation == "gate_proj":
            prefix = "model.layers.{}".format(layer_index)
            state.layer_values = {
                "residual": state.hidden_states,
                "normalized": _rms_norm(
                    state.hidden_states,
                    self.resident[
                        "{}.post_attention_layernorm.weight".format(prefix)
                    ],
                    self.config.rms_norm_eps,
                ),
            }
            state.layer_values["gate"] = F.silu(
                F.linear(state.layer_values["normalized"], weight)
            )
        elif operation == "up_proj":
            state.layer_values["up"] = F.linear(
                state.layer_values["normalized"],
                weight,
            )
        elif operation == "down_proj":
            intermediate = (
                state.layer_values["gate"] * state.layer_values["up"]
            )
            state.hidden_states = state.layer_values["residual"] + F.linear(
                intermediate,
                weight,
            )
            state.layer_values = {}
        else:
            raise ValueError("unknown Llama operation {}".format(operation))
        return state

    def __call__(self, unit, weights, state):
        if unit.operation == "layer":
            for operation in MATRIX_ORDER:
                key = self._matrix_key(unit.layer_index, operation)
                state = self._execute_matrix(
                    unit.layer_index,
                    operation,
                    weights[key],
                    state,
                )
            return state
        if unit.operation == "qkv":
            for operation in ("q_proj", "k_proj", "v_proj"):
                key = self._matrix_key(unit.layer_index, operation)
                state = self._execute_matrix(
                    unit.layer_index,
                    operation,
                    weights[key],
                    state,
                )
            return state
        if unit.operation == "gate_up":
            for operation in ("gate_proj", "up_proj"):
                key = self._matrix_key(unit.layer_index, operation)
                state = self._execute_matrix(
                    unit.layer_index,
                    operation,
                    weights[key],
                    state,
                )
            return state
        key = self._matrix_key(unit.layer_index, unit.operation)
        return self._execute_matrix(
            unit.layer_index,
            unit.operation,
            weights[key],
            state,
        )

    def finish(self, state):
        normalized = _rms_norm(
            state.hidden_states,
            self.resident["model.norm.weight"],
            self.config.rms_norm_eps,
        )
        if self.vocab_runtime is None:
            state.logits = F.linear(
                normalized,
                self.resident["lm_head.weight"],
            ).float()
            state.topk_values, state.topk_indices = torch.topk(
                state.logits[:, -1:, :],
                k=min(self.top_k, self.config.vocab_size),
                dim=-1,
            )
        else:
            result = self.vocab_runtime.lm_head_topk(
                normalized[:, -1:, :],
                top_k=self.top_k,
                return_full_logits=self.return_full_logits,
            )
            state.topk_values = result["values"]
            state.topk_indices = result["indices"]
            state.logits = result["logits"]
        return state
