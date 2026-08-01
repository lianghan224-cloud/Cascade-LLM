#!/usr/bin/env python3
"""Validate a real Llama-family checkpoint and optionally compare logits."""

import argparse
from argparse import Namespace
from dataclasses import replace
import json
from pathlib import Path
import sys

import torch
from transformers import AutoConfig, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import (  # noqa: E402
    ExecutionPolicy,
    MemoryPlanner,
    PlacementMode,
    adapter_for_config,
    build_static_transformer_placement,
)
from compare_reference import (  # noqa: E402
    compare,
    parse_byte_size,
    reference_outputs,
    streaming_outputs,
)


ACCEPTANCE_REPORT_SCHEMA_VERSION = 1


def parse_ids(value):
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("token list cannot be empty")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--input-ids", type=parse_ids)
    parser.add_argument("--decode-ids", type=parse_ids)
    parser.add_argument(
        "--backend",
        choices=(
            "checkpoint",
            "bf16_linear",
            "fp16_linear",
            "int8_dequant_bf16_fallback",
            "int8_dequant_fp16_fallback",
            "int4_dequant_bf16_fallback",
            "int4_dequant_fp16_fallback",
            "fused_w8a16",
            "fused_w4a16",
            "fused_w8a8",
        ),
        default="checkpoint",
    )
    parser.add_argument(
        "--provider", choices=("none", "cutlass"), default="none"
    )
    parser.add_argument("--provider-library")
    parser.add_argument(
        "--granularity",
        choices=("matrix", "matrix_group", "layer"),
        default="matrix_group",
    )
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="pinned_staging",
    )
    parser.add_argument(
        "--vocab-mode",
        choices=("resident", "streamed"),
        default="streamed",
    )
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--gpu-resident-weight-budget",
        type=parse_byte_size,
        default=0,
    )
    parser.add_argument(
        "--ignore-memlock-limit",
        action="store_true",
        help=(
            "Treat only RLIMIT_MEMLOCK preflight failures as warnings. "
            "Use this only after verifying that the CUDA pinned allocator works."
        ),
    )
    return parser.parse_args()


def tokenizer_report(tokenizer, geometry):
    special_ids = {
        name: getattr(tokenizer, name, None)
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        )
    }
    invalid = {
        name: value
        for name, value in special_ids.items()
        if value is not None
        and (int(value) < 0 or int(value) >= geometry.vocab_size)
    }
    tokenizer_size = len(tokenizer)
    errors = []
    if tokenizer_size > geometry.vocab_size:
        errors.append(
            "tokenizer size {} exceeds model vocab {}".format(
                tokenizer_size, geometry.vocab_size
            )
        )
    if invalid:
        errors.append(
            "special token ids outside model vocabulary: {}".format(invalid)
        )
    return {
        "ok": not errors,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_size": tokenizer_size,
        "model_vocab_size": geometry.vocab_size,
        "special_token_ids": special_ids,
        "chat_template_present": bool(
            getattr(tokenizer, "chat_template", None)
        ),
        "errors": errors,
    }


def default_ids(tokenizer, geometry):
    bos = tokenizer.bos_token_id
    if bos is None or not 0 <= int(bos) < geometry.vocab_size:
        bos = 1 if geometry.vocab_size > 1 else 0
    candidates = [
        int(bos),
        min(4, geometry.vocab_size - 1),
        min(7, geometry.vocab_size - 1),
    ]
    decode = [min(9, geometry.vocab_size - 1)]
    return candidates, decode


def main():
    args = parse_args()
    if args.provider == "cutlass":
        from layer_streaming.providers.cutlass import (
            load_cutlass_w8a16_provider,
        )

        load_cutlass_w8a16_provider(args.provider_library)
    config = AutoConfig.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    adapter = adapter_for_config(config)
    geometry = adapter.build_geometry(config)
    inferred = ExecutionPolicy.from_config(config)
    policy = ExecutionPolicy.from_config(
        config,
        granularity=args.granularity,
        cpu_weight_mode=args.weight_store,
        embedding_mode=PlacementMode(args.vocab_mode),
        lm_head_mode=PlacementMode(args.vocab_mode),
        slot_count=args.slots,
        prefetch_depth=args.slots,
        linear_backend=(
            None if args.backend == "checkpoint" else args.backend
        ),
    )
    plan = adapter.build_execution_plan(config, policy)
    transformer_placement = build_static_transformer_placement(
        plan, args.gpu_resident_weight_budget
    )
    validation = adapter.validate_checkpoint(
        args.checkpoint, config, policy
    )
    tokenizer_status = tokenizer_report(tokenizer, geometry)
    input_ids, decode_ids = default_ids(tokenizer, geometry)
    input_ids = args.input_ids or input_ids
    decode_ids = args.decode_ids or decode_ids
    for token in input_ids + decode_ids:
        if token < 0 or token >= geometry.vocab_size:
            raise SystemExit(
                "token id {} is outside model vocabulary".format(token)
            )
    total_length = len(input_ids) + len(decode_ids)
    if total_length > geometry.max_position_embeddings:
        raise SystemExit("acceptance sequence exceeds model context limit")
    preflight = MemoryPlanner(
        plan,
        geometry,
        policy=policy,
        max_context=total_length,
        max_prefill_tokens=len(input_ids),
        embedding_staging_rows=len(input_ids),
        return_full_logits=True,
        logits_tokens=len(input_ids),
        top_k=args.top_k,
        transformer_placement=transformer_placement,
    ).preflight(
        None if args.metadata_only else args.device,
        raise_on_error=False,
    )
    if args.ignore_memlock_limit:
        memlock_errors = tuple(
            item
            for item in preflight.errors
            if "RLIMIT_MEMLOCK" in item
        )
        if memlock_errors:
            preflight = replace(
                preflight,
                errors=tuple(
                    item
                    for item in preflight.errors
                    if item not in memlock_errors
                ),
                warnings=preflight.warnings
                + tuple(
                    "explicitly ignored after CUDA pinned-allocation probe: "
                    + item
                    for item in memlock_errors
                ),
            )
    comparison = None
    ok = validation.ok and tokenizer_status["ok"] and preflight.ok
    if not args.metadata_only and ok:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required unless --metadata-only is set")
        device = torch.device(args.device)
        prefill = torch.tensor(
            [input_ids], dtype=torch.long, device=device
        )
        decode = [
            torch.tensor([[token]], dtype=torch.long, device=device)
            for token in decode_ids
        ]
        reference = reference_outputs(
            args.checkpoint,
            config,
            device,
            prefill,
            decode,
            inferred,
        )
        compare_args = Namespace(
            checkpoint=args.checkpoint,
            weight_format="auto",
            backend=args.backend,
            granularity=args.granularity,
            weight_store=args.weight_store,
            embedding_mode=args.vocab_mode,
            lm_head_mode=args.vocab_mode,
            slots=args.slots,
            block_size=16,
            top_k=args.top_k,
            profile=True,
            provider=args.provider,
            provider_library=args.provider_library,
        )
        candidate = streaming_outputs(
            compare_args,
            config,
            device,
            prefill,
            decode,
        )
        comparison_ok, comparisons = compare(
            reference,
            candidate,
            args.atol,
            args.rtol,
            args.top_k,
        )
        comparison = {
            "ok": comparison_ok,
            "atol": args.atol,
            "rtol": args.rtol,
            "outputs": comparisons,
        }
        ok = ok and comparison_ok
    report = {
        "schema_version": ACCEPTANCE_REPORT_SCHEMA_VERSION,
        "ok": ok,
        "checkpoint": str(args.checkpoint.resolve()),
        "synthetic": (
            (args.checkpoint / "generation_manifest.json").is_file()
        ),
        "metadata_only": args.metadata_only,
        "geometry": geometry.as_dict(),
        "weight_format": inferred.weight_format.value,
        "backend_requested": args.backend,
        "planned_backends": sorted(
            {
                tensor.backend
                for unit in plan.units
                for tensor in unit.tensors
                if tensor.backend
            }
        ),
        "transformer_placement": transformer_placement.as_dict(),
        "checkpoint_validation": validation.as_dict(),
        "tokenizer": tokenizer_status,
        "memory_preflight": preflight.as_dict(),
        "input_ids": input_ids,
        "decode_ids": decode_ids,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": ok,
                "synthetic": report["synthetic"],
                "metadata_only": args.metadata_only,
                "weight_format": report["weight_format"],
                "planned_backends": report["planned_backends"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
