#!/usr/bin/env python3
"""Plan a synthetic Llama geometry without allocating model weights."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    ExecutionPolicy,
    MemoryPlanner,
    PlacementMode,
    QuantizationSpec,
    WeightFormat,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="synthetic-llama")
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--intermediate-size", type=int, required=True)
    parser.add_argument("--attention-heads", type=int, required=True)
    parser.add_argument("--kv-heads", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--model-max-context", type=int, default=131072)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument(
        "--weight-format",
        choices=tuple(item.value for item in WeightFormat),
        default=WeightFormat.BF16.value,
    )
    parser.add_argument(
        "--quant-granularity",
        choices=("per_channel", "per_group"),
        default="per_group",
    )
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--scale-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="pinned_staging",
    )
    parser.add_argument(
        "--granularity",
        choices=("matrix", "matrix_group", "layer"),
        default="matrix_group",
    )
    parser.add_argument(
        "--embedding-mode", choices=("resident", "streamed"), default="streamed"
    )
    parser.add_argument(
        "--lm-head-mode", choices=("resident", "streamed"), default="streamed"
    )
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--target-shard-gib", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    weight_format = WeightFormat(args.weight_format)
    quantization = None
    if "int8" in weight_format.value or "int4" in weight_format.value:
        bits = 8 if "int8" in weight_format.value else 4
        granularity = args.quant_granularity
        if bits == 4 and granularity != "per_group":
            raise SystemExit("INT4 metadata planning requires per_group")
        quantization = QuantizationSpec(
            bits=bits,
            granularity=granularity,
            group_size=args.group_size if granularity == "per_group" else None,
            scale_dtype=args.scale_dtype,
        )
    config = {
        "model_type": "llama",
        "_name_or_path": args.model_id,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "num_hidden_layers": args.layers,
        "num_attention_heads": args.attention_heads,
        "num_key_value_heads": args.kv_heads,
        "vocab_size": args.vocab_size,
        "max_position_embeddings": args.model_max_context,
        "tie_word_embeddings": False,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "hidden_act": "silu",
    }
    policy = ExecutionPolicy(
        granularity=args.granularity,
        weight_format=weight_format,
        cpu_weight_mode=args.weight_store,
        embedding_mode=PlacementMode(args.embedding_mode),
        lm_head_mode=PlacementMode(args.lm_head_mode),
        slot_count=args.slots,
        quantization=quantization,
    )
    result = MemoryPlanner.plan_metadata_only(
        config,
        policy,
        target_shard_bytes=int(args.target_shard_gib * 1024 ** 3),
        max_context=args.max_context,
    )
    rendered = json.dumps(result.as_dict(), indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
