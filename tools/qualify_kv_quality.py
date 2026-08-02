#!/usr/bin/env python3
"""Run the real-model L4 KV provider non-regression micro-suite."""

import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import statistics
import sys
import time

import torch
from transformers import AutoConfig, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming import (  # noqa: E402
    ExecutionPolicy,
    KVPolicy,
    Llama31DecodeExecutor,
    MixedDtypeRuntime,
    MixedResidentDeviceArena,
    MultiDtypeWeightStore,
    PlacementMode,
    adapter_for_config,
)


LABELS = ("A", "B", "C", "D")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate", default="sm86")
    parser.add_argument(
        "--reference", default="legacy_gather_sdpa_reference"
    )
    parser.add_argument("--page-size", type=int, choices=(16, 32), default=16)
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="pinned_staging",
    )
    parser.add_argument("--slots", type=int, choices=(1, 2, 3, 4), default=2)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_chat(tokenizer, messages):
    system = {
        "role": "system",
        "content": "请完成选择题，并且只回答一个大写字母 A、B、C 或 D。",
    }
    return tokenizer.apply_chat_template(
        [system] + list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )


def label_token_ids(tokenizer):
    result = []
    for label in LABELS:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                "quality label {!r} is not a single tokenizer token".format(
                    label
                )
            )
        result.append(encoded[0])
    return tuple(result)


def build_long_examples(suite):
    filler = (
        "背景资料说明：本段内容用于填充上下文，普通记录没有需要回答的校验信息。"
    )
    examples = []
    for index, item in enumerate(suite["long_context"]):
        repetitions = int(item["filler_repetitions"])
        before = repetitions // 2
        after = repetitions - before
        values = [item["value"]] + list(item["distractors"])
        rotation = index % len(values)
        choices = values[rotation:] + values[:rotation]
        correct = choices.index(item["value"])
        context = "\n".join(
            [filler] * before
            + [
                "关键记录：代号 {} 的校验值是 {}。".format(
                    item["record"], item["value"]
                )
            ]
            + [filler] * after
        )
        prompt = (
            "阅读资料并找到关键记录。\n{}\n\n代号 {} 的校验值是什么？\n{}"
        ).format(
            context,
            item["record"],
            "\n".join(
                "{}. {}".format(label, value)
                for label, value in zip(LABELS, choices)
            ),
        )
        examples.append(
            {
                "id": "long_{:02d}".format(index),
                "messages": ({"role": "user", "content": prompt},),
                "answer": LABELS[correct],
                "filler_repetitions": repetitions,
            }
        )
    return examples


class QualityRunner:
    def __init__(
        self,
        config,
        plan,
        resident,
        runtime,
        tokenizer,
        device,
        page_size,
    ):
        self.config = config
        self.plan = plan
        self.resident = resident
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.page_size = int(page_size)
        self.dtype_name = next(
            spec.compute_dtype
            for spec in plan.weights.values()
            if spec.role == "attention_q"
        )

    def executor(self, provider, max_length):
        return Llama31DecodeExecutor(
            self.config,
            self.resident,
            vocab_runtime=None,
            top_k=10,
            return_full_logits=False,
            max_cache_length=max(1, int(max_length)),
            kv_block_size=self.page_size,
            kv_policy=KVPolicy(
                attention_backend=provider,
                page_size=self.page_size,
                dtype=(
                    "bf16" if self.dtype_name == "bfloat16" else "fp16"
                ),
            ),
            allow_kv_reference=provider in {
                "reference_paged_exact",
                "legacy_gather_sdpa_reference",
            },
            kv_dtype=(
                torch.bfloat16
                if self.dtype_name == "bfloat16"
                else torch.float16
            ),
        )

    def forward(self, executor, token_ids):
        ids = torch.tensor(
            [list(token_ids)], dtype=torch.long, device=self.device
        )
        state = executor.finish(
            self.runtime.run(executor, executor.begin(ids))
        )
        logits = state.logits[:, -1, :].detach().float()
        if not bool(torch.isfinite(logits).all().item()):
            raise RuntimeError("quality evaluation produced non-finite logits")
        return logits

    def perplexity(self, provider, passages, eval_tokens):
        total_nll = 0.0
        total_tokens = 0
        details = []
        for index, passage in enumerate(passages):
            ids = self.tokenizer.encode(passage, add_special_tokens=True)
            count = min(int(eval_tokens), max(0, len(ids) - 1))
            if count <= 0:
                raise ValueError("perplexity passage has fewer than two tokens")
            prefix_length = len(ids) - count
            passage_nll = []
            with self.executor(provider, len(ids)) as executor:
                logits = self.forward(executor, ids[:prefix_length])
                for offset, target in enumerate(ids[prefix_length:]):
                    nll = float(
                        (
                            torch.logsumexp(logits[0], dim=-1)
                            - logits[0, int(target)]
                        ).item()
                    )
                    passage_nll.append(nll)
                    total_nll += nll
                    total_tokens += 1
                    if offset + 1 < count:
                        logits = self.forward(executor, (target,))
            details.append(
                {
                    "id": "ppl_{:02d}".format(index),
                    "token_count": count,
                    "mean_nll": statistics.mean(passage_nll),
                    "perplexity": math.exp(statistics.mean(passage_nll)),
                }
            )
            print(
                "quality provider={} perplexity {}/{}".format(
                    provider, index + 1, len(passages)
                ),
                flush=True,
            )
        mean_nll = total_nll / float(total_tokens)
        return {
            "tokens": total_tokens,
            "mean_nll": mean_nll,
            "perplexity": math.exp(mean_nll),
            "examples": details,
        }

    def choices(self, provider, examples, label_ids, category):
        correct = 0
        details = []
        for index, example in enumerate(examples):
            messages = example.get("messages")
            if messages is None:
                messages = (
                    {"role": "user", "content": example["prompt"]},
                )
            prompt = render_chat(self.tokenizer, messages)
            ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            with self.executor(provider, len(ids)) as executor:
                logits = self.forward(executor, ids)[0]
            scores = [float(logits[token].item()) for token in label_ids]
            predicted = LABELS[max(range(len(scores)), key=scores.__getitem__)]
            expected = str(example["answer"])
            passed = predicted == expected
            correct += int(passed)
            details.append(
                {
                    "id": example.get(
                        "id", "{}_{:02d}".format(category, index)
                    ),
                    "prompt_tokens": len(ids),
                    "expected": expected,
                    "predicted": predicted,
                    "correct": passed,
                    "label_logits": dict(zip(LABELS, scores)),
                }
            )
            print(
                "quality provider={} {} {}/{} expected={} predicted={}".format(
                    provider,
                    category,
                    index + 1,
                    len(examples),
                    expected,
                    predicted,
                ),
                flush=True,
            )
        return {
            "examples": len(examples),
            "correct": correct,
            "accuracy": correct / float(len(examples)),
            "results": details,
        }

    def run(self, provider, suite, label_ids):
        started = time.time()
        result = {
            "provider": provider,
            "perplexity": self.perplexity(
                provider,
                suite["perplexity_passages"],
                suite["perplexity_eval_tokens_per_passage"],
            ),
            "short_text": self.choices(
                provider, suite["short_text"], label_ids, "short"
            ),
            "long_context": self.choices(
                provider,
                build_long_examples(suite),
                label_ids,
                "long",
            ),
            "dialogue": self.choices(
                provider, suite["dialogue"], label_ids, "dialogue"
            ),
        }
        result["duration_seconds"] = time.time() - started
        return result


def accuracy_drop(reference, candidate, category):
    return max(
        0.0,
        (
            float(reference[category]["accuracy"])
            - float(candidate[category]["accuracy"])
        ) * 100.0,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    suite = read_json(args.suite)
    if int(suite.get("schema_version", 0)) != 1:
        raise SystemExit("unsupported quality suite schema")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True
    )
    labels = label_token_ids(tokenizer)
    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    inferred = ExecutionPolicy.from_config(config)
    policy = ExecutionPolicy(
        granularity="matrix_group",
        weight_format=inferred.weight_format,
        cpu_weight_mode=args.weight_store,
        embedding_mode=PlacementMode.RESIDENT,
        lm_head_mode=PlacementMode.RESIDENT,
        slot_count=args.slots,
        prefetch_depth=args.slots,
        embedding_dtype=inferred.embedding_dtype,
        lm_head_dtype=inferred.lm_head_dtype,
        norm_dtype=inferred.norm_dtype,
        quantization=inferred.quantization,
        linear_backend=None,
    )
    adapter = adapter_for_config(config)
    plan = adapter.build_execution_plan(config, policy)
    adapter.validate_checkpoint(args.checkpoint, config, policy).raise_for_error()
    with ExitStack() as resources:
        store = resources.enter_context(
            MultiDtypeWeightStore(
                plan,
                args.weight_store,
                staging_slot_count=args.slots,
            )
        )
        store.load_checkpoint(args.checkpoint)
        resident = resources.enter_context(
            MixedResidentDeviceArena(plan, store, device)
        )
        runtime = resources.enter_context(
            MixedDtypeRuntime(
                plan,
                store,
                resident,
                device,
                slot_count=args.slots,
            )
        )
        runner = QualityRunner(
            config,
            plan,
            resident,
            runtime,
            tokenizer,
            device,
            args.page_size,
        )
        reference = runner.run(args.reference, suite, labels)
        candidate = runner.run(args.candidate, suite, labels)

    reference_ppl = float(reference["perplexity"]["perplexity"])
    candidate_ppl = float(candidate["perplexity"]["perplexity"])
    metrics = {
        "perplexity_relative_degradation": max(
            0.0, candidate_ppl / reference_ppl - 1.0
        ),
        "perplexity_relative_change": candidate_ppl / reference_ppl - 1.0,
        "short_accuracy_drop_pp": accuracy_drop(
            reference, candidate, "short_text"
        ),
        "long_hit_rate_drop_pp": accuracy_drop(
            reference, candidate, "long_context"
        ),
        "dialogue_accuracy_drop_pp": accuracy_drop(
            reference, candidate, "dialogue"
        ),
    }
    coverage = {
        "perplexity_tokens": int(candidate["perplexity"]["tokens"]),
        "short_examples": int(candidate["short_text"]["examples"]),
        "long_context_examples": int(candidate["long_context"]["examples"]),
        "dialogue_examples": int(candidate["dialogue"]["examples"]),
    }
    report = {
        "schema_version": 1,
        "suite": suite["name"],
        "suite_description": suite["description"],
        "checkpoint": str(args.checkpoint.resolve()),
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "architecture": "sm{}{}".format(
                *torch.cuda.get_device_capability(device)
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "reference_provider": args.reference,
        "candidate_provider": args.candidate,
        "page_size": args.page_size,
        "coverage": coverage,
        "metrics": metrics,
        "all_finite": all(
            math.isfinite(float(value)) for value in metrics.values()
        ),
        "providers": {
            "reference": reference,
            "candidate": candidate,
        },
        "interpretation": (
            "This suite measures candidate-provider degradation relative to "
            "the explicit gather/SDPA reference; it is not a broad model "
            "capability or population-level accuracy benchmark."
        ),
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
