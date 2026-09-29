"""MoE model adapters and additive execution-plan sidecars."""

from dataclasses import dataclass
from typing import Mapping

from ..adapter import (
    ExecutionPolicy,
    ModelAdapter,
    ModelGeometry,
    PlacementMode,
    WeightFormat,
    _config_value,
    _required_int,
    validate_safetensors_checkpoint,
)
from ..backends import backend_for_weight
from ..execution_plan import (
    ExecutionPlan,
    ResidentTensor,
    TransferTensor,
    TransferUnit,
    VocabPlacement,
    _build_region_layout,
    storage_region_name,
)
from ..plan import Granularity
from ..specs import DTYPE_BYTES, WeightSpec, align_up
from .config import MoEConfig
from .objects import (
    WeightObjectCatalog,
    WeightObjectKey,
    WeightObjectKind,
    WeightObjectRecord,
    WeightObjectSource,
)


@dataclass(frozen=True)
class MoEExecutionSidecar:
    execution_plan: ExecutionPlan
    moe_config: MoEConfig
    weight_objects: WeightObjectCatalog
    attention_unit_ids_by_layer: Mapping[int, tuple]
    expert_unit_ids_by_layer: Mapping[int, Mapping[int, str]]
    router_names_by_layer: Mapping[int, str]
    q_norm_names_by_layer: Mapping[int, str]
    k_norm_names_by_layer: Mapping[int, str]

    def expert_unit(self, layer_id, expert_id):
        unit_id = self.expert_unit_ids_by_layer[int(layer_id)][int(expert_id)]
        return next(
            item for item in self.execution_plan.units if item.unit_id == unit_id
        )


def _compute_dtype(policy):
    return {
        WeightFormat.BF16: "bfloat16",
        WeightFormat.FP16: "float16",
        WeightFormat.INT8_DEQUANT_BF16_FALLBACK: "bfloat16",
        WeightFormat.INT8_DEQUANT_FP16_FALLBACK: "float16",
        WeightFormat.INT4_DEQUANT_BF16_FALLBACK: "bfloat16",
        WeightFormat.INT4_DEQUANT_FP16_FALLBACK: "float16",
    }[policy.weight_format]


def _append_linear_specs(specs, name, shape, role, policy):
    compute_dtype = _compute_dtype(policy)
    if policy.quantization is None:
        specs.append(
            WeightSpec.dense(
                name, shape, compute_dtype, role, compute_dtype=compute_dtype
            )
        )
        return (name,)
    weight = WeightSpec.quantized(
        name, shape, policy.quantization, compute_dtype, role
    )
    specs.append(weight)
    names = [name]
    scale_name = name + "_scale"
    specs.append(
        WeightSpec.dense(
            scale_name,
            policy.quantization.scale_shape(shape),
            policy.quantization.scale_dtype,
            "scale",
            compute_dtype=compute_dtype,
        )
    )
    names.append(scale_name)
    if policy.quantization.zero_point:
        zero_name = name + "_zero_point"
        specs.append(
            WeightSpec.dense(
                zero_name,
                policy.quantization.scale_shape(shape),
                policy.quantization.zero_point_dtype,
                "zero_point",
                compute_dtype=compute_dtype,
            )
        )
        names.append(zero_name)
    return tuple(names)


class OlmoeModelAdapter(ModelAdapter):
    """Adapter for Hugging Face ``olmoe`` checkpoints.

    Model-specific names terminate here; Router, Dispatcher and ExpertBackend
    consume only the sidecar's semantic objects.
    """

    def build_geometry(self, config):
        model_type = str(_config_value(config, "model_type", ""))
        architectures = _config_value(config, "architectures", ()) or ()
        if model_type != "olmoe" and not any("Olmoe" in item for item in architectures):
            raise ValueError("expected an OLMoE config, got {!r}".format(model_type))
        if bool(_config_value(config, "attention_bias", False)):
            raise ValueError("OLMoE attention bias is not supported")
        hidden = _required_int(config, "hidden_size")
        heads = _required_int(config, "num_attention_heads")
        head_dim = int(_config_value(config, "head_dim", hidden // heads))
        if hidden != heads * head_dim:
            raise ValueError("hidden_size must equal attention heads * head_dim")
        kv_heads = int(_config_value(config, "num_key_value_heads", heads))
        model_id = _config_value(config, "_name_or_path") or _config_value(
            config, "name_or_path", "local-olmoe"
        )
        return ModelGeometry(
            model_id=str(model_id),
            model_type="olmoe",
            hidden_size=hidden,
            intermediate_size=_required_int(config, "intermediate_size"),
            num_hidden_layers=_required_int(config, "num_hidden_layers"),
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            vocab_size=_required_int(config, "vocab_size"),
            max_position_embeddings=_required_int(config, "max_position_embeddings"),
            rms_norm_eps=float(_config_value(config, "rms_norm_eps", 1e-5)),
            rope_theta=float(_config_value(config, "rope_theta", 10000.0)),
            rope_scaling=_config_value(config, "rope_scaling"),
            tie_word_embeddings=bool(_config_value(config, "tie_word_embeddings", False)),
            hidden_act=str(_config_value(config, "hidden_act", "silu")),
        )

    def build_moe_config(self, config, policy=None):
        policy = policy or ExecutionPolicy.from_config(config)
        return MoEConfig(
            num_experts=_required_int(config, "num_experts"),
            experts_per_token=_required_int(config, "num_experts_per_tok"),
            expert_intermediate_size=_required_int(config, "intermediate_size"),
            router_type="linear_topk",
            router_dtype=_compute_dtype(policy),
            normalize_topk=bool(_config_value(config, "norm_topk_prob", False)),
            routed_scaling_factor=1.0,
            has_shared_expert=False,
            num_shared_experts=0,
            expert_weight_layout="gate_up_down",
        )

    def _specs_and_objects(self, config, policy):
        geometry = self.build_geometry(config)
        moe = self.build_moe_config(config, policy)
        specs = []
        objects = []
        embedding = WeightSpec.dense(
            "model.embed_tokens.weight",
            (geometry.vocab_size, geometry.hidden_size),
            policy.embedding_dtype,
            "embedding",
            compute_dtype=policy.embedding_dtype,
        )
        specs.append(embedding)
        objects.append(
            (WeightObjectKey(None, WeightObjectKind.EMBEDDING, embedding.name), (embedding.name,))
        )
        if geometry.tie_word_embeddings:
            if policy.embedding_dtype != policy.lm_head_dtype:
                raise ValueError("tied Embedding/LM Head must use the same dtype")
            lm_head = WeightSpec.dense(
                "lm_head.weight",
                embedding.logical_shape,
                policy.embedding_dtype,
                "lm_head",
                compute_dtype=policy.lm_head_dtype,
                alias_of=embedding.name,
            )
        else:
            lm_head = WeightSpec.dense(
                "lm_head.weight",
                (geometry.vocab_size, geometry.hidden_size),
                policy.lm_head_dtype,
                "lm_head",
                compute_dtype=policy.lm_head_dtype,
            )
        specs.append(lm_head)
        objects.append(
            (
                WeightObjectKey(None, WeightObjectKind.LM_HEAD, lm_head.name),
                (lm_head.alias_of or lm_head.name,),
            )
        )

        for layer in range(geometry.num_hidden_layers):
            prefix = "model.layers.{}".format(layer)
            attention = (
                ("q_proj", "attention_q", (geometry.hidden_size, geometry.hidden_size)),
                ("k_proj", "attention_k", (geometry.kv_width, geometry.hidden_size)),
                ("v_proj", "attention_v", (geometry.kv_width, geometry.hidden_size)),
                ("o_proj", "attention_o", (geometry.hidden_size, geometry.hidden_size)),
            )
            for operation, role, shape in attention:
                name = "{}.self_attn.{}.weight".format(prefix, operation)
                names = _append_linear_specs(specs, name, shape, role, policy)
                objects.append(
                    (WeightObjectKey(layer, WeightObjectKind.DENSE, name), names)
                )
            norm_specs = (
                ("{}.self_attn.q_norm.weight".format(prefix), (geometry.hidden_size,)),
                ("{}.self_attn.k_norm.weight".format(prefix), (geometry.kv_width,)),
                ("{}.input_layernorm.weight".format(prefix), (geometry.hidden_size,)),
                ("{}.post_attention_layernorm.weight".format(prefix), (geometry.hidden_size,)),
            )
            for name, shape in norm_specs:
                specs.append(
                    WeightSpec.dense(
                        name, shape, policy.norm_dtype, "norm", compute_dtype=policy.norm_dtype
                    )
                )
                objects.append(
                    (WeightObjectKey(layer, WeightObjectKind.DENSE, name), (name,))
                )
            router_name = "{}.mlp.gate.weight".format(prefix)
            router_names = _append_linear_specs(
                specs,
                router_name,
                (moe.num_experts, geometry.hidden_size),
                "mlp_gate",
                policy,
            )
            objects.append(
                (WeightObjectKey(layer, WeightObjectKind.ROUTER, router_name), router_names)
            )
            for expert in range(moe.num_experts):
                expert_prefix = "{}.mlp.experts.{}".format(prefix, expert)
                names = []
                names.extend(
                    _append_linear_specs(
                        specs,
                        expert_prefix + ".gate_proj.weight",
                        (moe.expert_intermediate_size, geometry.hidden_size),
                        "mlp_gate",
                        policy,
                    )
                )
                names.extend(
                    _append_linear_specs(
                        specs,
                        expert_prefix + ".up_proj.weight",
                        (moe.expert_intermediate_size, geometry.hidden_size),
                        "mlp_up",
                        policy,
                    )
                )
                names.extend(
                    _append_linear_specs(
                        specs,
                        expert_prefix + ".down_proj.weight",
                        (geometry.hidden_size, moe.expert_intermediate_size),
                        "mlp_down",
                        policy,
                    )
                )
                objects.append(
                    (
                        WeightObjectKey(
                            layer,
                            WeightObjectKind.EXPERT,
                            "L{}/E{}".format(layer, expert),
                            expert_id=expert,
                        ),
                        tuple(names),
                    )
                )
        final_norm = WeightSpec.dense(
            "model.norm.weight",
            (geometry.hidden_size,),
            policy.norm_dtype,
            "norm",
            compute_dtype=policy.norm_dtype,
        )
        specs.append(final_norm)
        # The final norm is global but WeightObjectKey keeps DENSE layer-local.
        # Associate it with the final Transformer layer for deterministic lookup.
        objects.append(
            (
                WeightObjectKey(
                    geometry.num_hidden_layers - 1,
                    WeightObjectKind.DENSE,
                    final_norm.name,
                ),
                (final_norm.name,),
            )
        )
        return tuple(specs), tuple(objects)

    def enumerate_weights(self, config, index=None, policy=None):
        del index
        policy = policy or ExecutionPolicy.from_config(config)
        return self._specs_and_objects(config, policy)[0]

    def build_moe_plan(self, config, policy=None):
        policy = policy or ExecutionPolicy.from_config(config)
        geometry = self.build_geometry(config)
        moe = self.build_moe_config(config, policy)
        specs, object_definitions = self._specs_and_objects(config, policy)
        by_name = {item.name: item for item in specs}
        aliases = {item.name: item.alias_of for item in specs if item.alias_of is not None}
        host_offsets, regions = _build_region_layout(specs)

        def transfer_tensor(name, device_offset, primary=False):
            spec = by_name[name]
            backend = backend_for_weight(
                spec, backend_name=policy.linear_backend
            ) if primary else None
            quant_names = ()
            if primary and spec.quantization is not None:
                quant_names = (name + "_scale",) + (
                    (name + "_zero_point",) if spec.quantization.zero_point else ()
                )
            return TransferTensor(
                weight_name=name,
                storage_region=storage_region_name(spec),
                host_offset=int(host_offsets[name]),
                storage_bytes=int(spec.storage_nbytes),
                device_offset=int(device_offset),
                backend="" if backend is None else backend.name,
                quant_param_names=tuple(quant_names),
            )

        def make_unit(unit_id, layer_id, operation, primary_names):
            cursor = 0
            tensors = []
            workspace = 0
            for primary_name in primary_names:
                primary = by_name[primary_name]
                backend = backend_for_weight(primary, backend_name=policy.linear_backend)
                workspace = max(workspace, backend.workspace_bytes(primary, batch_tokens=1))
                members = [primary_name]
                if primary.quantization is not None:
                    members.append(primary_name + "_scale")
                    if primary.quantization.zero_point:
                        members.append(primary_name + "_zero_point")
                for index, name in enumerate(members):
                    spec = by_name[name]
                    cursor = align_up(cursor, spec.alignment)
                    tensors.append(transfer_tensor(name, cursor, primary=(index == 0)))
                    cursor += spec.storage_nbytes
            return TransferUnit(
                unit_id=unit_id,
                layer_id=int(layer_id),
                operation=str(operation),
                tensors=tuple(tensors),
                transfer_bytes=align_up(cursor, 256),
                workspace_bytes=int(workspace),
            )

        units = []
        attention_ids = {}
        expert_ids = {}
        router_names = {}
        q_norm_names = {}
        k_norm_names = {}
        for layer in range(geometry.num_hidden_layers):
            prefix = "model.layers.{}".format(layer)
            qkv = tuple("{}.self_attn.{}.weight".format(prefix, item) for item in ("q_proj", "k_proj", "v_proj"))
            o_name = "{}.self_attn.o_proj.weight".format(prefix)
            layer_attention = []
            if Granularity(policy.granularity) == Granularity.MATRIX:
                for name in qkv + (o_name,):
                    operation = name.rsplit(".", 2)[-2]
                    unit = make_unit("layer_{:04d}.{}".format(layer, operation), layer, operation, (name,))
                    units.append(unit)
                    layer_attention.append(unit.unit_id)
            else:
                qkv_unit = make_unit("layer_{:04d}.qkv".format(layer), layer, "qkv", qkv)
                o_unit = make_unit("layer_{:04d}.o_proj".format(layer), layer, "o_proj", (o_name,))
                units.extend((qkv_unit, o_unit))
                layer_attention.extend((qkv_unit.unit_id, o_unit.unit_id))
            attention_ids[layer] = tuple(layer_attention)
            router_names[layer] = "{}.mlp.gate.weight".format(prefix)
            q_norm_names[layer] = "{}.self_attn.q_norm.weight".format(prefix)
            k_norm_names[layer] = "{}.self_attn.k_norm.weight".format(prefix)
            expert_ids[layer] = {}
            for expert in range(moe.num_experts):
                expert_prefix = "{}.mlp.experts.{}".format(prefix, expert)
                primary = tuple(
                    expert_prefix + ".{}.weight".format(item)
                    for item in ("gate_proj", "up_proj", "down_proj")
                )
                unit = make_unit(
                    "layer_{:04d}.expert_{:04d}".format(layer, expert),
                    layer,
                    "expert",
                    primary,
                )
                units.append(unit)
                expert_ids[layer][expert] = unit.unit_id

        resident_names = {
            item.name for item in specs if item.role == "norm" and item.alias_of is None
        }
        resident_names.update(router_names.values())
        if policy.embedding_mode == PlacementMode.RESIDENT:
            resident_names.add("model.embed_tokens.weight")
        if policy.lm_head_mode == PlacementMode.RESIDENT:
            resident_names.add(aliases.get("lm_head.weight", "lm_head.weight"))
        resident = []
        cursor = 0
        for name in sorted(resident_names):
            spec = by_name[name]
            cursor = align_up(cursor, spec.alignment)
            backend_name = ""
            if name in router_names.values():
                backend_name = backend_for_weight(spec, backend_name=policy.linear_backend).name
            resident.append(
                ResidentTensor(
                    weight_name=name,
                    storage_region=storage_region_name(spec),
                    host_offset=int(host_offsets[name]),
                    storage_bytes=int(spec.storage_nbytes),
                    device_offset=int(cursor),
                    backend=backend_name,
                )
            )
            cursor += spec.storage_nbytes
        embedding = by_name["model.embed_tokens.weight"]
        lm_head = by_name["lm_head.weight"]
        lm_storage = by_name[lm_head.alias_of] if lm_head.alias_of is not None else lm_head
        row_bytes = geometry.hidden_size * DTYPE_BYTES[lm_storage.storage_dtype]
        if policy.vocab_chunk_bytes < row_bytes:
            raise ValueError("vocab chunk must hold at least one row")
        chunk_rows = min(geometry.vocab_size, policy.vocab_chunk_bytes // row_bytes)
        vocab = VocabPlacement(
            embedding_name=embedding.name,
            lm_head_name=lm_head.name,
            embedding_mode=policy.embedding_mode.value,
            lm_head_mode=policy.lm_head_mode.value,
            embedding_dtype=embedding.compute_dtype,
            lm_head_dtype=lm_head.compute_dtype,
            chunk_rows=int(chunk_rows),
            chunk_bytes=int(chunk_rows * row_bytes),
        )
        # Vocabulary streaming owns its own bounded chunk buffer.  Transformer
        # transfer slots must be sized by attention/Expert objects only; using
        # vocab.chunk_bytes here would pin and allocate 2x128 MiB for OLMoE
        # despite each complete Expert being only 12 MiB.
        slot_bytes = max(item.transfer_bytes for item in units)
        plan = ExecutionPlan(
            model_id=geometry.model_id,
            geometry=geometry,
            granularity=Granularity(policy.granularity),
            weights=by_name,
            regions=regions,
            host_offsets=host_offsets,
            units=tuple(units),
            resident=tuple(resident),
            aliases=aliases,
            slot_bytes=int(slot_bytes),
            workspace_bytes=max((item.workspace_bytes for item in units), default=0),
            vocab=vocab,
            slot_count=int(policy.slot_count),
        )
        records = []
        for key, names in object_definitions:
            sources = tuple(
                WeightObjectSource(
                    tensor_name=name,
                    storage_region=storage_region_name(by_name[name]),
                    host_offset=int(host_offsets[name]),
                    nbytes=int(by_name[name].storage_nbytes),
                )
                for name in names
            )
            records.append(
                WeightObjectRecord(
                    key=key,
                    tensor_names=tuple(names),
                    storage_dtypes=tuple(by_name[name].storage_dtype for name in names),
                    shapes=tuple(by_name[name].storage_shape for name in names),
                    nbytes=sum(item.nbytes for item in sources),
                    sources=sources,
                )
            )
        return MoEExecutionSidecar(
            execution_plan=plan,
            moe_config=moe,
            weight_objects=WeightObjectCatalog(
                records, allow_source_aliases=True
            ),
            attention_unit_ids_by_layer=attention_ids,
            expert_unit_ids_by_layer=expert_ids,
            router_names_by_layer=router_names,
            q_norm_names_by_layer=q_norm_names,
            k_norm_names_by_layer=k_norm_names,
        )

    def build_execution_plan(self, config, policy=None):
        return self.build_moe_plan(config, policy).execution_plan

    def build_transfer_units(self, geometry, policy):
        del geometry, policy
        raise NotImplementedError(
            "OLMoE transfer units require the MoE config; use build_moe_plan"
        )

    def validate_checkpoint(self, index, config, policy=None):
        return validate_safetensors_checkpoint(
            index, self.enumerate_weights(config, index=index, policy=policy)
        )


def moe_adapter_for_config(config):
    model_type = str(_config_value(config, "model_type", "")).lower()
    if model_type == "olmoe":
        return OlmoeModelAdapter()
    raise ValueError("no MoE adapter is registered for {!r}".format(model_type))
