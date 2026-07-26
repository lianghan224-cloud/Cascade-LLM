#!/usr/bin/env python3
"""Compare saved reference and streaming Llama logits."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    reference = load_file(str(args.reference))["logits"].float()
    candidate = load_file(str(args.candidate))["logits"].float()
    if reference.shape != candidate.shape:
        raise SystemExit(
            "shape mismatch {} != {}".format(
                tuple(reference.shape),
                tuple(candidate.shape),
            )
        )
    reference_meta = json.loads(args.reference_json.read_text())
    candidate_meta = json.loads(args.candidate_json.read_text())
    delta = candidate - reference
    reference_top10 = torch.topk(reference, k=10, dim=-1).indices
    candidate_top10 = torch.topk(candidate, k=10, dim=-1).indices
    overlap = []
    for row in range(reference.shape[0]):
        overlap.append(
            len(
                set(reference_top10[row].tolist()).intersection(
                    candidate_top10[row].tolist()
                )
            )
        )
    cosine = torch.nn.functional.cosine_similarity(
        reference,
        candidate,
        dim=-1,
    )
    report = {
        "schema_version": 1,
        "shape": list(reference.shape),
        "max_abs_error": float(delta.abs().max().item()),
        "mean_abs_error": float(delta.abs().mean().item()),
        "rmse": float(delta.square().mean().sqrt().item()),
        "cosine_similarity_per_step": cosine.tolist(),
        "argmax_match_per_step": (
            reference.argmax(dim=-1) == candidate.argmax(dim=-1)
        ).tolist(),
        "top10_overlap_per_step": overlap,
        "reference_generated_token_ids": reference_meta[
            "generated_token_ids"
        ],
        "candidate_generated_token_ids": candidate_meta[
            "generated_token_ids"
        ],
        "generated_tokens_exact_match": (
            reference_meta["generated_token_ids"]
            == candidate_meta["generated_token_ids"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
