#!/usr/bin/env python3
"""Generate deterministic tiny Llama safetensors and reference weights."""

import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import pack_int4  # noqa: E402


TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=768)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--embedding-dtype", choices=("bf16", "fp16"))
    parser.add_argument("--lm-head-dtype", choices=("bf16", "fp16"))
    parser.add_argument("--norm-dtype", choices=("bf16", "fp16"))
    parser.add_argument(
        "--quantization",
        choices=(
            "none",
            "int8_per_channel",
            "int8_per_group",
            "int4_per_group",
        ),
        default="none",
    )
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=64)
    parser.add_argument(
        "--scale-dtype", choices=("bf16", "fp16"), default="bf16"
    )
    parser.add_argument(
        "--reference-dtype", choices=("bf16", "fp32"), default="fp32"
    )
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--tie-word-embeddings", action="store_true")
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--hidden-act", default="silu")
    return parser.parse_args()


def is_projection(name):
    return (
        (".self_attn." in name or ".mlp." in name)
        and name.endswith(".weight")
    )


def quantize_symmetric(source, bits, granularity, group_size, scale_dtype):
    source = source.detach().float()
    if source.ndim != 2:
        raise ValueError("linear source must be rank 2")
    if granularity == "per_channel":
        scale = source.abs().amax(dim=1, keepdim=True)
    elif granularity == "per_group":
        if source.shape[1] % int(group_size):
            raise ValueError(
                "group_size {} does not divide input dimension {}".format(
                    group_size, source.shape[1]
                )
            )
        grouped = source.view(
            source.shape[0], source.shape[1] // int(group_size), int(group_size)
        )
        scale = grouped.abs().amax(dim=2)
    else:
        raise ValueError("unsupported granularity {}".format(granularity))
    maximum = 127.0 if bits == 8 else 7.0
    scale = torch.clamp(scale / maximum, min=torch.finfo(torch.float32).eps)
    expanded = (
        scale
        if granularity == "per_channel"
        else scale.repeat_interleave(int(group_size), dim=1)
    )
    quantized = torch.round(source / expanded).clamp(-maximum, maximum).to(
        torch.int8
    )
    scale = scale.to(TORCH_DTYPES[scale_dtype])
    if bits == 4:
        return pack_int4(quantized), scale
    return quantized, scale


def build_tensors(config, args):
    model = LlamaForCausalLM(config).float().eval()
    dense_dtype = TORCH_DTYPES[args.dtype]
    embedding_dtype = TORCH_DTYPES[args.embedding_dtype]
    lm_head_dtype = TORCH_DTYPES[args.lm_head_dtype]
    norm_dtype = TORCH_DTYPES[args.norm_dtype]
    reference_dtype = TORCH_DTYPES[args.reference_dtype]
    checkpoint = {}
    reference = {}
    quant_bits = 4 if args.quantization.startswith("int4") else 8
    granularity = (
        "per_group" if args.quantization.endswith("per_group") else "per_channel"
    )
    for name, value in model.state_dict().items():
        if args.tie_word_embeddings and name == "lm_head.weight":
            continue
        source = value.detach().clone()
        reference[name] = source.to(reference_dtype)
        if not is_projection(name):
            if name == "model.embed_tokens.weight":
                target_dtype = embedding_dtype
            elif name == "lm_head.weight":
                target_dtype = lm_head_dtype
            elif "layernorm.weight" in name or name == "model.norm.weight":
                target_dtype = norm_dtype
            else:
                target_dtype = dense_dtype
            checkpoint[name] = source.to(target_dtype)
            continue
        if args.quantization == "none":
            checkpoint[name] = source.to(dense_dtype)
            continue
        quantized, scale = quantize_symmetric(
            source,
            bits=quant_bits,
            granularity=granularity,
            group_size=args.group_size,
            scale_dtype=args.scale_dtype,
        )
        checkpoint[name] = quantized
        checkpoint[name + "_scale"] = scale
    return checkpoint, reference


def tensor_nbytes(tensor):
    return int(tensor.numel() * tensor.element_size())


def save_sharded(tensors, output, shard_count):
    shard_count = int(shard_count)
    if shard_count < 1:
        raise ValueError("shards must be positive")
    if shard_count == 1:
        save_file(tensors, str(output / "model.safetensors"))
        return {"weight_map": {name: "model.safetensors" for name in tensors}}
    if shard_count > len(tensors):
        raise ValueError("shard count cannot exceed tensor count")
    bins = [([], 0) for _ in range(shard_count)]
    for name in sorted(tensors, key=lambda item: tensor_nbytes(tensors[item]), reverse=True):
        index = min(range(shard_count), key=lambda item: bins[item][1])
        bins[index][0].append(name)
        bins[index] = (bins[index][0], bins[index][1] + tensor_nbytes(tensors[name]))
    weight_map = {}
    for index, (names, _) in enumerate(bins, start=1):
        filename = "model-{:05d}-of-{:05d}.safetensors".format(
            index, shard_count
        )
        shard = {name: tensors[name] for name in sorted(names)}
        save_file(shard, str(output / filename))
        weight_map.update({name: filename for name in shard})
    index = {
        "metadata": {
            "total_size": sum(tensor_nbytes(value) for value in tensors.values())
        },
        "weight_map": weight_map,
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return index


def main():
    args = parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be positive")
    args.embedding_dtype = args.embedding_dtype or args.dtype
    args.lm_head_dtype = args.lm_head_dtype or args.dtype
    args.norm_dtype = args.norm_dtype or args.dtype
    if (
        args.tie_word_embeddings
        and args.embedding_dtype != args.lm_head_dtype
    ):
        raise SystemExit(
            "tied Embedding/LM Head must use the same dtype"
        )
    torch.manual_seed(args.seed)
    config = LlamaConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.max_context,
        tie_word_embeddings=args.tie_word_embeddings,
        rope_theta=args.rope_theta,
        hidden_act=args.hidden_act,
        torch_dtype=TORCH_DTYPES[args.dtype],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._name_or_path = "cascade-tiny-synthetic"
    config.cascade_dtype_config = {
        "transformer": args.dtype,
        "embedding": args.embedding_dtype,
        "lm_head": args.lm_head_dtype,
        "norm": args.norm_dtype,
    }
    if args.quantization != "none":
        config.quantization_config = {
            "format": "cascade_symmetric",
            "bits": 4 if args.quantization.startswith("int4") else 8,
            "granularity": (
                "per_group"
                if args.quantization.endswith("per_group")
                else "per_channel"
            ),
            "group_size": (
                args.group_size
                if args.quantization.endswith("per_group")
                else None
            ),
            "scale_dtype": args.scale_dtype,
            "packing": (
                "int4_pair_uint8"
                if args.quantization.startswith("int4")
                else "none"
            ),
            "axis": 1,
            "execution_path": "{}_dequant_{}_fallback".format(
                "int4" if args.quantization.startswith("int4") else "int8",
                args.dtype,
            ),
        }
    args.output.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(args.output)
    checkpoint, reference = build_tensors(config, args)
    save_sharded(checkpoint, args.output, args.shards)
    reference_dir = args.output / "reference"
    reference_dir.mkdir(exist_ok=True)
    save_file(reference, str(reference_dir / "model.safetensors"))
    (reference_dir / "README.txt").write_text(
        "Unquantized {} reference weights for synthetic validation.\n".format(
            args.reference_dtype
        ),
        encoding="utf-8",
    )
    vocabulary = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }
    vocabulary.update(
        {
            "tok_{}".format(index): index
            for index in range(4, args.vocab_size)
        }
    )
    tokenizer_backend = Tokenizer(
        WordLevel(vocabulary, unk_token="<unk>")
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(args.output)
    summary = {
        "output": str(args.output),
        "tensor_count": len(checkpoint),
        "reference_tensor_count": len(reference),
        "checkpoint_bytes": sum(
            tensor_nbytes(value) for value in checkpoint.values()
        ),
        "reference_bytes": sum(
            tensor_nbytes(value) for value in reference.values()
        ),
        "quantization": args.quantization,
        "dtype": args.dtype,
        "embedding_dtype": args.embedding_dtype,
        "lm_head_dtype": args.lm_head_dtype,
        "norm_dtype": args.norm_dtype,
        "shards": args.shards,
        "tied": args.tie_word_embeddings,
    }
    (args.output / "generation_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
