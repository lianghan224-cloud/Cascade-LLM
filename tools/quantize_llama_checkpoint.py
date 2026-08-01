#!/usr/bin/env python3
"""Stream-convert a local Llama safetensors checkpoint to Cascade W8A16."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    ExecutionPolicy,
    adapter_for_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scale-dtype", choices=("bf16", "fp16"), default="bf16"
    )
    return parser.parse_args()


def is_projection(name):
    return (
        (".self_attn." in name or ".mlp." in name)
        and name.endswith(".weight")
    )


def quantize_per_channel(source, scale_dtype):
    source_float = source.float()
    scale = torch.clamp(
        source_float.abs().amax(dim=1, keepdim=True) / 127.0,
        min=torch.finfo(torch.float32).eps,
    )
    quantized = torch.round(source_float / scale).clamp(-127, 127).to(
        torch.int8
    )
    target_dtype = (
        torch.bfloat16 if scale_dtype == "bf16" else torch.float16
    )
    stored_scale = scale.to(target_dtype)
    reconstructed = quantized.float() * stored_scale.float()
    error = (reconstructed - source_float).abs()
    quality = {
        "elements": int(source.numel()),
        "max_abs_error": float(error.max().item()),
        "mean_abs_error": float(error.mean().item()),
    }
    return quantized, stored_scale, quality


def load_index(checkpoint):
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return index, sorted(set(index["weight_map"].values()))
    single = checkpoint / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError("checkpoint has no safetensors weights")
    with safe_open(str(single), framework="pt", device="cpu") as source:
        keys = list(source.keys())
    return {"metadata": {}, "weight_map": {key: single.name for key in keys}}, [
        single.name
    ]


def copy_metadata(source, target):
    ignored = {
        "model.safetensors",
        "model.safetensors.index.json",
        "quantization.json",
        "quantization_report.json",
    }
    for path in source.iterdir():
        if not path.is_file() or path.name in ignored:
            continue
        if path.name.startswith("model-") and path.name.endswith(".safetensors"):
            continue
        shutil.copy2(str(path), str(target / path.name))


def main():
    args = parse_args()
    source_root = args.input.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise SystemExit("output already exists: {}".format(output_root))
    if source_root == output_root:
        raise SystemExit("input and output must differ")
    config = AutoConfig.from_pretrained(source_root, local_files_only=True)
    if config.model_type != "llama":
        raise SystemExit("only Llama-family checkpoints are supported")
    index, shard_names = load_index(source_root)
    output_root.mkdir(parents=True)
    started = time.time()
    output_weight_map = {}
    total_bytes = 0
    tensor_quality = {}
    quantized_count = 0
    try:
        copy_metadata(source_root, output_root)
        for shard_index, shard_name in enumerate(shard_names, 1):
            source_path = source_root / shard_name
            target_path = output_root / shard_name
            tensors = {}
            with safe_open(
                str(source_path), framework="pt", device="cpu"
            ) as source:
                for name in source.keys():
                    value = source.get_tensor(name)
                    if is_projection(name):
                        if value.ndim != 2:
                            raise ValueError(
                                "projection {} is not rank 2".format(name)
                            )
                        weight, scale, quality = quantize_per_channel(
                            value, args.scale_dtype
                        )
                        tensors[name] = weight.contiguous()
                        tensors[name + "_scale"] = scale.contiguous()
                        tensor_quality[name] = quality
                        quantized_count += 1
                    else:
                        tensors[name] = value.contiguous()
            save_file(tensors, str(target_path))
            for name, value in tensors.items():
                output_weight_map[name] = shard_name
                total_bytes += int(value.numel() * value.element_size())
            print(
                "[{}/{}] {}: {} tensors".format(
                    shard_index, len(shard_names), shard_name, len(tensors)
                ),
                flush=True,
            )
            del tensors

        config.quantization_config = {
            "format": "cascade_symmetric",
            "bits": 8,
            "scheme": "symmetric",
            "granularity": "per_channel",
            "group_size": None,
            "scale_dtype": args.scale_dtype,
            "zero_point": False,
            "zero_point_dtype": None,
            "packing": "none",
            "axis": 1,
            "execution_path": "int8_dequant_bf16_fallback",
        }
        config.cascade_dtype_config = {
            "transformer": "bf16",
            "embedding": "bf16",
            "lm_head": "bf16",
            "norm": "bf16",
        }
        config.save_pretrained(output_root)
        output_index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": output_weight_map,
        }
        (output_root / "model.safetensors.index.json").write_text(
            json.dumps(output_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        quantization_document = {
            "schema_version": 1,
            "format": "cascade_symmetric",
            "scope": "transformer_linear_weights",
            "bits": 8,
            "scheme": "symmetric",
            "granularity": "per_channel",
            "axis": 1,
            "scale_dtype": args.scale_dtype,
            "zero_point": False,
            "embedding_dtype": "bf16",
            "lm_head_dtype": "bf16",
            "norm_dtype": "bf16",
            "source_checkpoint": str(source_root),
        }
        (output_root / "quantization.json").write_text(
            json.dumps(quantization_document, indent=2) + "\n",
            encoding="utf-8",
        )
        weighted_mean = sum(
            item["mean_abs_error"] * item["elements"]
            for item in tensor_quality.values()
        ) / max(1, sum(item["elements"] for item in tensor_quality.values()))
        report = {
            "schema_version": 1,
            "source_checkpoint": str(source_root),
            "output_checkpoint": str(output_root),
            "quantized_tensor_count": quantized_count,
            "checkpoint_bytes": total_bytes,
            "duration_seconds": time.time() - started,
            "max_abs_error": max(
                item["max_abs_error"] for item in tensor_quality.values()
            ),
            "weighted_mean_abs_error": weighted_mean,
            "tensors": tensor_quality,
        }
        (output_root / "quantization_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        inferred = ExecutionPolicy.from_config(config)
        validation = adapter_for_config(config).validate_checkpoint(
            output_root, config, inferred
        )
        validation.raise_for_error()
        print(json.dumps({**report, "validation_ok": True}, indent=2))
    except BaseException:
        # Keep the partial directory for forensic inspection; it is never
        # treated as valid because validation/report creation did not finish.
        raise


if __name__ == "__main__":
    main()
