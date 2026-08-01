"""Model configuration, execution policy, and checkpoint validation."""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Tuple

from .plan import Granularity, MIB, build_llama_plan
from .specs import QuantizationSpec, WeightSpec, normalize_dtype


class WeightFormat(str, Enum):
    BF16 = "bf16"
    FP16 = "fp16"
    INT8_DEQUANT_BF16_FALLBACK = "int8_dequant_bf16_fallback"
    INT8_DEQUANT_FP16_FALLBACK = "int8_dequant_fp16_fallback"
    INT4_DEQUANT_BF16_FALLBACK = "int4_dequant_bf16_fallback"
    INT4_DEQUANT_FP16_FALLBACK = "int4_dequant_fp16_fallback"


class PlacementMode(str, Enum):
    RESIDENT = "resident"
    STREAMED = "streamed"


@dataclass(frozen=True)
class ExecutionPolicy:
    """User-selected memory and transfer policies.

    These values deliberately remain explicit.  Automatic policy selection is
    a later benchmarking feature and must not silently override this object.
    """

    granularity: Granularity = Granularity.MATRIX_GROUP
    weight_format: WeightFormat = WeightFormat.BF16
    cpu_weight_mode: str = "pinned_staging"
    embedding_mode: PlacementMode = PlacementMode.STREAMED
    lm_head_mode: PlacementMode = PlacementMode.STREAMED
    slot_count: int = 2
    prefetch_depth: int = 2
    vocab_chunk_bytes: int = 128 * MIB
    embedding_dtype: str = "bfloat16"
    lm_head_dtype: str = "bfloat16"
    norm_dtype: str = "bfloat16"
    quantization: Optional[QuantizationSpec] = None
    linear_backend: Optional[str] = None

    @classmethod
    def from_config(cls, config, **overrides):
        checkpoint_dtype = normalize_dtype(
            _config_value(config, "torch_dtype", "bfloat16") or "bfloat16"
        )
        quant_config = _config_value(config, "quantization_config")
        values = {
            "embedding_dtype": checkpoint_dtype,
            "lm_head_dtype": checkpoint_dtype,
            "norm_dtype": checkpoint_dtype,
        }
        component_dtypes = _config_value(config, "cascade_dtype_config", {}) or {}
        if not isinstance(component_dtypes, Mapping):
            raise ValueError("cascade_dtype_config must be a mapping")
        values.update(
            {
                "embedding_dtype": component_dtypes.get(
                    "embedding", values["embedding_dtype"]
                ),
                "lm_head_dtype": component_dtypes.get(
                    "lm_head", values["lm_head_dtype"]
                ),
                "norm_dtype": component_dtypes.get(
                    "norm", values["norm_dtype"]
                ),
            }
        )
        if quant_config:
            if not isinstance(quant_config, Mapping):
                if hasattr(quant_config, "to_dict"):
                    quant_config = quant_config.to_dict()
                else:
                    raise ValueError("quantization_config must be a mapping")
            bits = int(quant_config["bits"])
            compute_tag = (
                "bf16" if checkpoint_dtype == "bfloat16" else "fp16"
            )
            values["weight_format"] = WeightFormat(
                "int{}_dequant_{}_fallback".format(bits, compute_tag)
            )
            values["quantization"] = QuantizationSpec(
                bits=bits,
                scheme=quant_config.get("scheme", "symmetric"),
                granularity=quant_config.get("granularity", "per_channel"),
                group_size=quant_config.get("group_size"),
                scale_dtype=quant_config.get("scale_dtype", checkpoint_dtype),
                zero_point=bool(quant_config.get("zero_point", False)),
                zero_point_dtype=quant_config.get("zero_point_dtype"),
                packing=quant_config.get("packing"),
                axis=int(quant_config.get("axis", 1)),
            )
        else:
            values["weight_format"] = (
                WeightFormat.BF16
                if checkpoint_dtype == "bfloat16"
                else WeightFormat.FP16
            )
        values.update(overrides)
        return cls(**values)

    def __post_init__(self):
        object.__setattr__(self, "granularity", Granularity(self.granularity))
        object.__setattr__(self, "weight_format", WeightFormat(self.weight_format))
        object.__setattr__(
            self, "embedding_dtype", normalize_dtype(self.embedding_dtype)
        )
        object.__setattr__(
            self, "lm_head_dtype", normalize_dtype(self.lm_head_dtype)
        )
        object.__setattr__(self, "norm_dtype", normalize_dtype(self.norm_dtype))
        object.__setattr__(
            self, "embedding_mode", PlacementMode(self.embedding_mode)
        )
        object.__setattr__(self, "lm_head_mode", PlacementMode(self.lm_head_mode))
        if self.cpu_weight_mode not in {"full_pinned", "pinned_staging"}:
            raise ValueError(
                "cpu_weight_mode must be full_pinned or pinned_staging"
            )
        if int(self.slot_count) < 1:
            raise ValueError("slot_count must be positive")
        if int(self.prefetch_depth) < 1:
            raise ValueError("prefetch_depth must be positive")
        if int(self.vocab_chunk_bytes) < 1:
            raise ValueError("vocab_chunk_bytes must be positive")
        quantization = self.quantization
        bits = {
            WeightFormat.INT8_DEQUANT_BF16_FALLBACK: 8,
            WeightFormat.INT8_DEQUANT_FP16_FALLBACK: 8,
            WeightFormat.INT4_DEQUANT_BF16_FALLBACK: 4,
            WeightFormat.INT4_DEQUANT_FP16_FALLBACK: 4,
        }.get(self.weight_format)
        if bits is None:
            if quantization is not None:
                raise ValueError("dense weight formats cannot set quantization")
        else:
            if quantization is None:
                quantization = QuantizationSpec(
                    bits=bits,
                    granularity="per_channel" if bits == 8 else "per_group",
                    group_size=None if bits == 8 else 64,
                    scale_dtype="bfloat16",
                )
            if quantization.bits != bits:
                raise ValueError(
                    "{} requires a {}-bit QuantizationSpec".format(
                        self.weight_format.value, bits
                    )
                )
            object.__setattr__(self, "quantization", quantization)
        if self.linear_backend is not None:
            backend_name = str(self.linear_backend).strip()
            if not backend_name:
                raise ValueError("linear_backend cannot be empty")
            object.__setattr__(self, "linear_backend", backend_name)


@dataclass(frozen=True)
class ModelGeometry:
    model_id: str
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: Optional[Mapping[str, object]] = None
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"

    def __post_init__(self):
        integer_fields = (
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError("{} must be positive".format(name))
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if float(self.rms_norm_eps) <= 0:
            raise ValueError("rms_norm_eps must be positive")
        if float(self.rope_theta) <= 0:
            raise ValueError("rope_theta must be positive")
        if not str(self.hidden_act):
            raise ValueError("hidden_act cannot be empty")
        if self.rope_scaling is not None:
            if not isinstance(self.rope_scaling, Mapping):
                raise ValueError("rope_scaling must be a mapping or null")
            rope_scaling = dict(self.rope_scaling)
            if "factor" in rope_scaling and float(rope_scaling["factor"]) <= 0:
                raise ValueError("rope_scaling factor must be positive")
            object.__setattr__(self, "rope_scaling", rope_scaling)

    @property
    def kv_width(self):
        return self.num_key_value_heads * self.head_dim

    def as_dict(self):
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**dict(value))


@dataclass(frozen=True)
class TensorMismatch:
    key: str
    actual: object
    expected: object


@dataclass
class ValidationResult:
    checkpoint: str
    missing: Tuple[str, ...] = ()
    unexpected: Tuple[str, ...] = ()
    shape_mismatches: Tuple[TensorMismatch, ...] = ()
    dtype_mismatches: Tuple[TensorMismatch, ...] = ()
    index_errors: Tuple[str, ...] = ()
    shard_files: Tuple[str, ...] = ()
    tensor_count: int = 0

    @property
    def ok(self):
        return not (
            self.missing
            or self.unexpected
            or self.shape_mismatches
            or self.dtype_mismatches
            or self.index_errors
        )

    def as_dict(self):
        def mismatch(item):
            return {
                "key": item.key,
                "actual": item.actual,
                "expected": item.expected,
            }

        return {
            "checkpoint": self.checkpoint,
            "ok": self.ok,
            "tensor_count": self.tensor_count,
            "shard_files": list(self.shard_files),
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "shape_mismatches": [mismatch(item) for item in self.shape_mismatches],
            "dtype_mismatches": [mismatch(item) for item in self.dtype_mismatches],
            "index_errors": list(self.index_errors),
        }

    def format_errors(self):
        sections = []
        if self.missing:
            sections.append("missing weights:\n  " + "\n  ".join(self.missing))
        if self.unexpected:
            sections.append(
                "unexpected weights:\n  " + "\n  ".join(self.unexpected)
            )
        if self.shape_mismatches:
            sections.append(
                "shape conflicts:\n  "
                + "\n  ".join(
                    "{}: actual {} != expected {}".format(
                        item.key, item.actual, item.expected
                    )
                    for item in self.shape_mismatches
                )
            )
        if self.dtype_mismatches:
            sections.append(
                "dtype conflicts:\n  "
                + "\n  ".join(
                    "{}: actual {} != expected {}".format(
                        item.key, item.actual, item.expected
                    )
                    for item in self.dtype_mismatches
                )
            )
        if self.index_errors:
            sections.append("index/shard errors:\n  " + "\n  ".join(self.index_errors))
        return "\n".join(sections) if sections else "checkpoint is valid"

    def raise_for_error(self):
        if not self.ok:
            raise CheckpointValidationError(self)
        return self


class CheckpointValidationError(ValueError):
    def __init__(self, result):
        self.result = result
        super().__init__(result.format_errors())


class ModelAdapter:
    def build_geometry(self, config):
        raise NotImplementedError

    def enumerate_weights(self, config, index=None, policy=None):
        raise NotImplementedError

    def build_transfer_units(self, geometry, policy):
        raise NotImplementedError

    def build_execution_plan(self, config, policy=None):
        raise NotImplementedError

    def validate_checkpoint(self, index, config, policy=None):
        raise NotImplementedError


def _config_value(config, name, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _required_int(config, name):
    value = _config_value(config, name)
    if value is None:
        raise ValueError("config is missing {}".format(name))
    value = int(value)
    if value <= 0:
        raise ValueError("config {} must be positive".format(name))
    return value


class LlamaModelAdapter(ModelAdapter):
    """Adapter for standard Hugging Face Llama weight names."""

    def build_geometry(self, config):
        model_type = str(_config_value(config, "model_type", ""))
        architectures = _config_value(config, "architectures", ()) or ()
        if model_type != "llama" and not any("Llama" in item for item in architectures):
            raise ValueError("expected a Llama config, got {!r}".format(model_type))
        hidden = _required_int(config, "hidden_size")
        attention_heads = _required_int(config, "num_attention_heads")
        kv_heads = int(
            _config_value(config, "num_key_value_heads", attention_heads)
        )
        configured_head_dim = _config_value(config, "head_dim")
        head_dim = (
            int(configured_head_dim)
            if configured_head_dim is not None
            else hidden // attention_heads
        )
        if hidden != attention_heads * head_dim:
            raise ValueError(
                "hidden_size must equal num_attention_heads * head_dim"
            )
        if kv_heads <= 0 or attention_heads % kv_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        model_id = _config_value(config, "_name_or_path") or _config_value(
            config, "name_or_path", "local-llama"
        )
        return ModelGeometry(
            model_id=str(model_id),
            model_type="llama",
            hidden_size=hidden,
            intermediate_size=_required_int(config, "intermediate_size"),
            num_hidden_layers=_required_int(config, "num_hidden_layers"),
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            vocab_size=_required_int(config, "vocab_size"),
            max_position_embeddings=_required_int(
                config, "max_position_embeddings"
            ),
            rms_norm_eps=float(_config_value(config, "rms_norm_eps", 1e-5)),
            rope_theta=float(_config_value(config, "rope_theta", 10000.0)),
            rope_scaling=_config_value(config, "rope_scaling"),
            tie_word_embeddings=bool(
                _config_value(config, "tie_word_embeddings", False)
            ),
            hidden_act=str(_config_value(config, "hidden_act", "silu")),
        )

    def enumerate_weights(self, config, index=None, policy=None):
        del index
        policy = policy or ExecutionPolicy.from_config(config)
        geometry = self.build_geometry(config)
        compute_dtype = {
            WeightFormat.BF16: "bfloat16",
            WeightFormat.FP16: "float16",
            WeightFormat.INT8_DEQUANT_BF16_FALLBACK: "bfloat16",
            WeightFormat.INT8_DEQUANT_FP16_FALLBACK: "float16",
            WeightFormat.INT4_DEQUANT_BF16_FALLBACK: "bfloat16",
            WeightFormat.INT4_DEQUANT_FP16_FALLBACK: "float16",
        }[policy.weight_format]
        quantization = policy.quantization
        if (
            geometry.tie_word_embeddings
            and policy.embedding_dtype != policy.lm_head_dtype
        ):
            raise ValueError(
                "tied Embedding/LM Head must use the same dtype"
            )
        specs = [
            WeightSpec.dense(
                "model.embed_tokens.weight",
                (geometry.vocab_size, geometry.hidden_size),
                policy.embedding_dtype,
                "embedding",
                compute_dtype=policy.embedding_dtype,
            )
        ]
        if not geometry.tie_word_embeddings:
            specs.append(
                WeightSpec.dense(
                    "lm_head.weight",
                    (geometry.vocab_size, geometry.hidden_size),
                    policy.lm_head_dtype,
                    "lm_head",
                    compute_dtype=policy.lm_head_dtype,
                )
            )
        projection_shapes = (
            (
                "self_attn",
                "q_proj",
                "attention_q",
                (geometry.hidden_size, geometry.hidden_size),
            ),
            (
                "self_attn",
                "k_proj",
                "attention_k",
                (geometry.kv_width, geometry.hidden_size),
            ),
            (
                "self_attn",
                "v_proj",
                "attention_v",
                (geometry.kv_width, geometry.hidden_size),
            ),
            (
                "self_attn",
                "o_proj",
                "attention_o",
                (geometry.hidden_size, geometry.hidden_size),
            ),
            (
                "mlp",
                "gate_proj",
                "mlp_gate",
                (geometry.intermediate_size, geometry.hidden_size),
            ),
            (
                "mlp",
                "up_proj",
                "mlp_up",
                (geometry.intermediate_size, geometry.hidden_size),
            ),
            (
                "mlp",
                "down_proj",
                "mlp_down",
                (geometry.hidden_size, geometry.intermediate_size),
            ),
        )
        for layer in range(geometry.num_hidden_layers):
            prefix = "model.layers.{}".format(layer)
            for block, operation, role, shape in projection_shapes:
                key = "{}.{}.{}.weight".format(prefix, block, operation)
                if quantization is None:
                    specs.append(
                        WeightSpec.dense(
                            key,
                            shape,
                            compute_dtype,
                            role,
                            compute_dtype=compute_dtype,
                        )
                    )
                else:
                    specs.append(
                        WeightSpec.quantized(
                            key,
                            shape,
                            quantization,
                            compute_dtype,
                            role,
                        )
                    )
                    scale_shape = quantization.scale_shape(shape)
                    specs.append(
                        WeightSpec.dense(
                            key + "_scale",
                            scale_shape,
                            quantization.scale_dtype,
                            "scale",
                            compute_dtype=compute_dtype,
                        )
                    )
                    if quantization.zero_point:
                        specs.append(
                            WeightSpec.dense(
                                key + "_zero_point",
                                scale_shape,
                                quantization.zero_point_dtype,
                                "zero_point",
                                compute_dtype=compute_dtype,
                            )
                        )
            specs.extend(
                (
                    WeightSpec.dense(
                        prefix + ".input_layernorm.weight",
                        (geometry.hidden_size,),
                        policy.norm_dtype,
                        "norm",
                        compute_dtype=policy.norm_dtype,
                    ),
                    WeightSpec.dense(
                        prefix + ".post_attention_layernorm.weight",
                        (geometry.hidden_size,),
                        policy.norm_dtype,
                        "norm",
                        compute_dtype=policy.norm_dtype,
                    ),
                )
            )
        specs.append(
            WeightSpec.dense(
                "model.norm.weight",
                (geometry.hidden_size,),
                policy.norm_dtype,
                "norm",
                compute_dtype=policy.norm_dtype,
            )
        )
        if geometry.tie_word_embeddings:
            specs.append(
                WeightSpec.dense(
                    "lm_head.weight",
                    (geometry.vocab_size, geometry.hidden_size),
                    policy.embedding_dtype,
                    "lm_head",
                    compute_dtype=policy.lm_head_dtype,
                    alias_of="model.embed_tokens.weight",
                )
            )
        return tuple(specs)

    def build_plan(self, config, policy=None):
        policy = policy or ExecutionPolicy.from_config(config)
        geometry = self.build_geometry(config)
        if policy.weight_format == WeightFormat.BF16:
            return build_llama_plan(
                geometry,
                granularity=policy.granularity,
                embedding_mode=policy.embedding_mode.value,
                lm_head_mode=policy.lm_head_mode.value,
                vocab_chunk_bytes=policy.vocab_chunk_bytes,
            )
        if (
            policy.weight_format != WeightFormat.INT8_DEQUANT_BF16_FALLBACK
            or policy.quantization.granularity != "per_channel"
            or policy.quantization.scale_dtype != "bfloat16"
            or policy.embedding_dtype != "bfloat16"
            or policy.lm_head_dtype != "bfloat16"
            or policy.norm_dtype != "bfloat16"
        ):
            raise ValueError(
                "{} requires the generic execution-plan path".format(
                    policy.weight_format.value
                )
            )
        from .int8 import build_llama_int8_plan

        return build_llama_int8_plan(
            geometry,
            granularity=policy.granularity,
            embedding_mode=policy.embedding_mode.value,
            lm_head_mode=policy.lm_head_mode.value,
            vocab_chunk_bytes=policy.vocab_chunk_bytes,
        )

    def build_execution_plan(self, config, policy=None):
        from .execution_plan import build_execution_plan

        policy = policy or ExecutionPolicy.from_config(config)
        geometry = self.build_geometry(config)
        specs = self.enumerate_weights(config, policy=policy)
        return build_execution_plan(geometry, specs, policy)

    def build_transfer_units(self, geometry, policy):
        if policy.weight_format == WeightFormat.BF16:
            return build_llama_plan(
                geometry,
                granularity=policy.granularity,
                embedding_mode=policy.embedding_mode.value,
                lm_head_mode=policy.lm_head_mode.value,
                vocab_chunk_bytes=policy.vocab_chunk_bytes,
            ).units
        if (
            policy.weight_format != WeightFormat.INT8_DEQUANT_BF16_FALLBACK
            or policy.quantization.granularity != "per_channel"
        ):
            raise ValueError(
                "{} requires the generic execution-plan path".format(
                    policy.weight_format.value
                )
            )
        from .int8 import build_llama_int8_plan

        return build_llama_int8_plan(
            geometry,
            granularity=policy.granularity,
            embedding_mode=policy.embedding_mode.value,
            lm_head_mode=policy.lm_head_mode.value,
            vocab_chunk_bytes=policy.vocab_chunk_bytes,
        ).units

    def validate_checkpoint(self, index, config, policy=None):
        specs = self.enumerate_weights(config, index=index, policy=policy)
        return validate_safetensors_checkpoint(index, specs)


def validate_safetensors_checkpoint(checkpoint, expected_specs):
    """Validate all key/shape/dtype metadata without allocating tensor data."""
    from .checkpoint import CheckpointManifest

    aliases = {
        spec.key: spec.alias_of
        for spec in expected_specs
        if spec.alias_of is not None
    }
    manifest = CheckpointManifest.from_path(checkpoint, aliases=aliases)
    validation = manifest.validate(expected_specs)
    return ValidationResult(
        checkpoint=str(checkpoint),
        missing=validation.missing,
        unexpected=validation.unexpected,
        shape_mismatches=tuple(
            TensorMismatch(item.name, item.actual, item.expected)
            for item in validation.shape_mismatches
        ),
        dtype_mismatches=tuple(
            TensorMismatch(item.name, item.actual, item.expected)
            for item in validation.dtype_mismatches
        ),
        index_errors=validation.errors,
        shard_files=manifest.files,
        tensor_count=len(manifest.tensors),
    )


_MODEL_ADAPTER_FACTORIES = {}


def register_model_adapter(model_type, factory, replace=False):
    """Register an adapter factory without changing the ModelAdapter contract."""

    name = str(model_type).strip().lower()
    if not name:
        raise ValueError("model_type must be non-empty")
    if not callable(factory):
        raise TypeError("adapter factory must be callable")
    if name in _MODEL_ADAPTER_FACTORIES and not replace:
        raise ValueError("model adapter {} is already registered".format(name))
    _MODEL_ADAPTER_FACTORIES[name] = factory


def unregister_model_adapter(model_type):
    return _MODEL_ADAPTER_FACTORIES.pop(str(model_type).strip().lower(), None)


def adapter_for_config(config):
    model_type = str(_config_value(config, "model_type", ""))
    architectures = _config_value(config, "architectures", ()) or ()
    factory = _MODEL_ADAPTER_FACTORIES.get(model_type.lower())
    if factory is not None:
        adapter = factory()
        if not isinstance(adapter, ModelAdapter):
            raise TypeError(
                "adapter factory for {} did not return ModelAdapter".format(
                    model_type
                )
            )
        return adapter
    if model_type == "llama" or any("Llama" in item for item in architectures):
        return LlamaModelAdapter()
    raise ValueError("no ModelAdapter is registered for {!r}".format(model_type))
