"""Llama-3.1 8B/70B matrix-wise decode executor."""

from dataclasses import dataclass, field
import math

import torch
import torch.nn.functional as F

from .kv_cache import KVCacheManager, SimpleKVCache


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
    return weight.to(input_dtype) * normalized.to(input_dtype)


def _linear(hidden_states, weight):
    return F.linear(hidden_states.to(weight.dtype), weight)


def _repeat_kv(hidden_states, groups):
    if groups == 1:
        return hidden_states
    return hidden_states.repeat_interleave(groups, dim=1)


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

    This implementation targets an unpadded batch whose requests have the
    same sequence length. It supports a multi-token initial prefill and
    subsequent one-token decode calls.
    """

    def __init__(
        self,
        config,
        resident,
        kv_cache=None,
        vocab_runtime=None,
        top_k=10,
        return_full_logits=False,
        max_cache_length=None,
        kv_block_size=16,
        max_batch_size=1,
        trace_callback=None,
        kv_dtype=None,
        kv_policy=None,
        linear_trace_callback=None,
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

        if config.hidden_size % config.num_attention_heads:
            raise ValueError("hidden size is not divisible by attention heads")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("attention heads are not divisible by KV heads")
        self.config = config
        try:
            from transformers.activations import ACT2FN

            self.activation = ACT2FN[str(getattr(config, "hidden_act", "silu"))]
        except KeyError as error:
            raise ValueError(
                "unsupported Llama hidden_act {!r}".format(config.hidden_act)
            ) from error
        self.resident = resident
        self.vocab_runtime = vocab_runtime
        self.top_k = int(top_k)
        self.return_full_logits = bool(return_full_logits)
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        self._owned_kv_manager = None
        if kv_cache is None:
            if max_cache_length is None:
                raise ValueError(
                    "max_cache_length is required when kv_cache is omitted"
                )
            block_count = int(
                math.ceil(int(max_cache_length) / float(kv_block_size))
            )
            cache_dtype = kv_dtype or getattr(config, "torch_dtype", None)
            if cache_dtype not in {torch.bfloat16, torch.float16}:
                cache_dtype = torch.bfloat16
            self._owned_kv_manager = KVCacheManager(
                layer_count=config.num_hidden_layers,
                num_key_value_heads=config.num_key_value_heads,
                head_dim=config.hidden_size // config.num_attention_heads,
                total_blocks=block_count,
                block_size=kv_block_size,
                max_batch_size=max_batch_size,
                dtype=cache_dtype,
                device=resident.device,
                policy=kv_policy,
            )
            handle = self._owned_kv_manager.allocate(
                max_cache_length,
                batch_size=max_batch_size,
            )
            kv_cache = self._owned_kv_manager.bind(handle)
        self.kv_cache = kv_cache
        self.trace_callback = trace_callback
        self.linear_trace_callback = linear_trace_callback
        self.linear_profile_callback = None
        self.stage_profile_callback = None
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        device = resident.device
        self.rotary = LlamaRotaryEmbedding(config=config, device=device)
        self._apply_rotary_pos_emb = apply_rotary_pos_emb

    def set_linear_profiler(self, callback):
        """Install a runtime-owned CUDA event recorder for Transformer GEMMs."""

        self.linear_profile_callback = callback

    def set_stage_profiler(self, callback):
        """Install a runtime-owned CUDA event recorder for non-GEMM stages."""

        self.stage_profile_callback = callback

    def _profile_stage(self, phase, stage, layer_index=None):
        if self.stage_profile_callback is not None:
            self.stage_profile_callback(phase, stage, layer_index)

    def _linear(self, layer_index, operation, hidden_states, weight):
        key = self._matrix_key(layer_index, operation)
        if self.linear_profile_callback is not None:
            self.linear_profile_callback("start", key)
        output = (
            weight.execute(hidden_states)
            if callable(getattr(weight, "execute", None))
            else _linear(hidden_states, weight)
        )
        if self.linear_profile_callback is not None:
            self.linear_profile_callback("end", key)
        if self.linear_trace_callback is not None:
            # The weight may reference a reusable device slot.  Diagnostics
            # must consume it synchronously and must not retain the view.
            self.linear_trace_callback(
                key, hidden_states, weight, output
            )
        return output

    def _trace(self, stage, layer_index, tensor):
        if self.trace_callback is not None:
            self.trace_callback(stage, layer_index, tensor)

    def begin(self, input_ids, position_ids=None):
        if input_ids.ndim != 2 or input_ids.shape[0] < 1:
            raise ValueError(
                "runtime expects a non-empty [batch, sequence] tensor"
            )
        past = self.kv_cache.sequence_length()
        batch = input_ids.shape[0]
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
            ).unsqueeze(0).expand(batch, -1)
        elif position_ids.shape not in {
            (1, sequence),
            (batch, sequence),
        }:
            raise ValueError(
                "position_ids must have shape [1, sequence] or "
                "[batch, sequence]"
            )
        if (
            self.vocab_runtime is None
            or not self.vocab_runtime.vocab.stream_embedding
        ):
            hidden = F.embedding(
                input_ids,
                self.resident["model.embed_tokens.weight"],
            )
        else:
            hidden = self.vocab_runtime.embedding(input_ids)
        self._trace("embedding", None, hidden)
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
        had_past = self.kv_cache.sequence_length() > 0
        if had_past and query_length != 1:
            raise ValueError("chunked decode with a past cache is unsupported")
        if callable(getattr(self.kv_cache, "attend", None)):
            self.kv_cache.append_only(layer_index, key, value)
            attention = self.kv_cache.attend(
                layer_index,
                query,
                kv_groups=self.kv_groups,
                position_ids=state.position_ids,
            )
        else:
            key, value = self.kv_cache.append(
                layer_index,
                key,
                value,
            )
            key = _repeat_kv(key, self.kv_groups)
            value = _repeat_kv(value, self.kv_groups)
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
            values["q"] = self._linear(
                layer_index, operation, values["normalized"], weight
            )
        elif operation == "k_proj":
            values["k"] = self._linear(
                layer_index, operation, values["normalized"], weight
            )
        elif operation == "v_proj":
            values["v"] = self._linear(
                layer_index, operation, values["normalized"], weight
            )
        elif operation == "o_proj":
            self._profile_stage("start", "attention", layer_index)
            attention = self._attention_output(layer_index, state)
            self._profile_stage("end", "attention", layer_index)
            attention_output = self._linear(
                layer_index, operation, attention, weight
            )
            self._trace("attention", layer_index, attention_output)
            state.hidden_states = (
                values["residual"].to(attention_output.dtype) + attention_output
            )
            self._trace("attention_hidden", layer_index, state.hidden_states)
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
            state.layer_values["gate"] = self.activation(
                self._linear(
                    layer_index,
                    operation,
                    state.layer_values["normalized"],
                    weight,
                )
            )
        elif operation == "up_proj":
            state.layer_values["up"] = self._linear(
                layer_index,
                operation,
                state.layer_values["normalized"],
                weight,
            )
        elif operation == "down_proj":
            intermediate = (
                state.layer_values["gate"] * state.layer_values["up"]
            )
            mlp_output = self._linear(
                layer_index, operation, intermediate, weight
            )
            self._trace("mlp", layer_index, mlp_output)
            state.hidden_states = (
                state.layer_values["residual"].to(mlp_output.dtype)
                + mlp_output
            )
            self._trace("hidden", layer_index, state.hidden_states)
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
        self._trace("final_norm", None, normalized)
        self._profile_stage("start", "lm_head", None)
        try:
            if (
                self.vocab_runtime is None
                or not self.vocab_runtime.vocab.stream_lm_head
            ):
                lm_head_input = (
                    normalized
                    if self.return_full_logits
                    else normalized[:, -1:, :]
                )
                state.logits = _linear(
                    lm_head_input,
                    self.resident["lm_head.weight"],
                ).float()
                state.topk_values, state.topk_indices = torch.topk(
                    state.logits[:, -1:, :],
                    k=min(self.top_k, self.config.vocab_size),
                    dim=-1,
                )
            else:
                lm_head_input = (
                    normalized
                    if self.return_full_logits
                    else normalized[:, -1:, :]
                )
                result = self.vocab_runtime.lm_head_topk(
                    lm_head_input,
                    top_k=self.top_k,
                    return_full_logits=self.return_full_logits,
                )
                state.topk_values = result["values"][:, -1:, :]
                state.topk_indices = result["indices"][:, -1:, :]
                state.logits = result["logits"]
        finally:
            self._profile_stage("end", "lm_head", None)
        if state.logits is not None:
            self._trace("logits", None, state.logits)
        return state

    def close(self):
        if self._owned_kv_manager is not None:
            self._owned_kv_manager.close()
            self._owned_kv_manager = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
