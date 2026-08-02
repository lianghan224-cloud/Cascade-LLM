#!/usr/bin/env python3
"""Replay a Cascade long generation through HF and compare sampled Top-k."""

import argparse
import json
from pathlib import Path
import sys
import time

import torch
from transformers import AutoConfig, LlamaForCausalLM


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--soak-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", choices=("single", "balanced"), default="balanced")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--initial-token-id",
        type=int,
        default=None,
        help="Override the first token when an older soak report lacks this field.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    soak = json.loads(args.soak_report.read_text(encoding="utf-8"))
    generated = [int(item) for item in soak["generated_token_ids"]]
    if len(generated) != int(soak["decode_tokens_per_cycle"]):
        raise SystemExit("soak report generated-token count is inconsistent")
    initial = int(
        args.initial_token_id
        if args.initial_token_id is not None
        else soak.get("initial_token_id", 1)
    )
    snapshots = {int(item["token"]): item for item in soak["snapshots"]}
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    load_kwargs = {
        "config": config,
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "attn_implementation": "sdpa",
    }
    if args.device_map == "balanced":
        load_kwargs["device_map"] = "balanced"
    model = LlamaForCausalLM.from_pretrained(
        args.checkpoint,
        **load_kwargs
    ).eval()
    if args.device_map == "single":
        model = model.to(args.device)
    input_device = model.model.embed_tokens.weight.device
    cache = None
    comparisons = []
    started = time.time()
    with torch.inference_mode():
        for token_index in range(1, len(generated) + 1):
            input_token = initial if token_index == 1 else generated[token_index - 2]
            input_ids = torch.tensor([[input_token]], device=input_device)
            output = model(input_ids, past_key_values=cache, use_cache=True)
            cache = output.past_key_values
            if token_index not in snapshots:
                continue
            candidate = snapshots[token_index]
            candidate_indices = torch.tensor(
                candidate["topk_indices"], dtype=torch.long
            )
            count = int(candidate_indices.numel())
            values, indices = torch.topk(
                output.logits[:, -1, :].detach().float().cpu(),
                count,
                dim=-1,
            )
            indices = indices.reshape(-1)
            values = values.reshape(-1)
            candidate_values = torch.tensor(
                candidate["topk_values"], dtype=torch.float32
            )
            intersection = (
                indices.unsqueeze(-1) == candidate_indices.unsqueeze(0)
            ).any(dim=-1).sum().item()
            comparisons.append(
                {
                    "token": token_index,
                    "cascade_top1": int(candidate_indices[0].item()),
                    "hf_top1": int(indices[0].item()),
                    "top1_equal": bool(indices[0] == candidate_indices[0]),
                    "ordered_topk_equal": bool(torch.equal(indices, candidate_indices)),
                    "topk_set_consistency": float(intersection) / float(count),
                    "topk_value_max_abs_error": float(
                        (values - candidate_values).abs().max().item()
                    ),
                    "topk_value_mean_abs_error": float(
                        (values - candidate_values).abs().mean().item()
                    ),
                    "hf_logits_finite": bool(
                        torch.isfinite(output.logits).all().item()
                    ),
                }
            )
    top1_matches = sum(item["top1_equal"] for item in comparisons)
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.resolve()),
        "soak_report": str(args.soak_report),
        "initial_token_id": initial,
        "replayed_tokens": len(generated),
        "sampled_positions": len(comparisons),
        "duration_seconds": time.time() - started,
        "top1_matches": top1_matches,
        "top1_agreement_rate": (
            float(top1_matches) / float(len(comparisons))
            if comparisons else None
        ),
        "minimum_topk_set_consistency": min(
            item["topk_set_consistency"] for item in comparisons
        ),
        "first_top1_mismatch": next(
            (item for item in comparisons if not item["top1_equal"]),
            None,
        ),
        "all_hf_logits_finite": all(
            item["hf_logits_finite"] for item in comparisons
        ),
        "acceptance": {
            "all_sampled_top1_equal": top1_matches == len(comparisons),
            "all_hf_logits_finite": all(
                item["hf_logits_finite"] for item in comparisons
            ),
        },
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
