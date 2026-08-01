#!/usr/bin/env python3
"""Create a deterministic tiny BF16 Llama checkpoint for local tests."""

import argparse
from pathlib import Path

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--intermediate-size", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--max-context", type=int, default=256)
    parser.add_argument("--max-shard-size", default="10MB")
    parser.add_argument("--tie-word-embeddings", action="store_true")
    parser.add_argument(
        "--weight-format",
        choices=("bf16", "int8_dequant_bf16_fallback"),
        default="bf16",
    )
    return parser.parse_args()


def main():
    args = parse_args()
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
        torch_dtype="bfloat16",
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.weight_format == "bf16":
        model.save_pretrained(
            args.output,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
    else:
        if args.tie_word_embeddings:
            raise SystemExit(
                "the tiny INT8 generator currently requires untied vocabulary weights"
            )
        quantized = {}
        for key, tensor in model.state_dict().items():
            is_projection = (
                ".self_attn." in key or ".mlp." in key
            ) and key.endswith(".weight")
            if not is_projection:
                quantized[key] = tensor.detach().to(torch.bfloat16).clone()
                continue
            source = tensor.detach().float()
            scale = source.abs().amax(dim=1, keepdim=True) / 127.0
            scale = torch.clamp(scale, min=torch.finfo(torch.float32).eps)
            quantized[key] = torch.round(source / scale).clamp(
                -128, 127
            ).to(torch.int8)
            quantized[key + "_scale"] = scale.to(torch.bfloat16)
        config.quantization_config = {
            "format": "cascade_int8_per_output_channel",
            "execution_path": "int8_dequant_bf16_fallback",
        }
        config.save_pretrained(args.output)
        save_file(quantized, str(args.output / "model.safetensors"))
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
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] }} "
        "{{ message['content'] }} {% endfor %}assistant"
    )
    tokenizer.save_pretrained(args.output)
    print("created deterministic checkpoint at {}".format(args.output))


if __name__ == "__main__":
    main()
