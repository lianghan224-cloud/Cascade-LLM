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
from layer_streaming.numerical_contracts.kv_v2 import (  # noqa: E402
    evaluate_model_quality,
    load_kv_numerical_contract,
)
from tools.qualification_common import (  # noqa: E402
    BLOCKED_NOT_EXCLUSIVE,
    capture_cuda_environment,
    qualification_admission,
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
        "--candidate-quest-scorer",
        choices=("off", "cpu_reference", "torch_tensorized"),
        default="off",
    )
    parser.add_argument("--candidate-page-budget", type=int, default=0)
    parser.add_argument("--candidate-recent-window", type=int, default=0)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT
        / "tests"
        / "fixtures"
        / "kv_numerical_sm86_bf16_abi1_v2.json",
    )
    parser.add_argument(
        "--mode",
        choices=("cuda-smoke", "qualification"),
        default="cuda-smoke",
    )
    parser.add_argument(
        "--weight-store",
        choices=("full_pinned", "pinned_staging"),
        default="pinned_staging",
    )
    parser.add_argument("--slots", type=int, choices=(1, 2, 3, 4), default=2)
    return parser.parse_args()


def quality_kv_policy(
    provider,
    page_size,
    dtype,
    *,
    quest_scorer="off",
    page_budget=0,
    recent_window=0,
):
    """Build one explicit Dense or Quest quality-run policy."""

    quest_scorer = str(quest_scorer)
    if quest_scorer == "off":
        if int(page_budget) or int(recent_window):
            raise ValueError("Quest budgets require a candidate Quest scorer")
        return KVPolicy(
            attention_backend=provider,
            page_size=page_size,
            dtype=dtype,
        )
    if int(page_budget) <= 0:
        raise ValueError("Quest quality gate requires a positive page budget")
    return KVPolicy(
        accuracy="sparse",
        selection="quest_flat",
        attention_backend=provider,
        page_size=page_size,
        dtype=dtype,
        page_budget=page_budget,
        recent_window=recent_window,
        quest_scorer=quest_scorer,
    )


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

    def executor(self, provider, max_length, kv_options=None):
        kv_options = dict(kv_options or {})
        return Llama31DecodeExecutor(
            self.config,
            self.resident,
            vocab_runtime=None,
            top_k=10,
            return_full_logits=False,
            max_cache_length=max(1, int(max_length)),
            kv_block_size=self.page_size,
            kv_policy=quality_kv_policy(
                provider,
                self.page_size,
                "bf16" if self.dtype_name == "bfloat16" else "fp16",
                **kv_options,
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

    def perplexity(self, provider, passages, eval_tokens, kv_options=None):
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
            with self.executor(provider, len(ids), kv_options) as executor:
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

    def choices(
        self, provider, examples, label_ids, category, kv_options=None
    ):
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
            with self.executor(provider, len(ids), kv_options) as executor:
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

    def run(self, provider, suite, label_ids, kv_options=None):
        started = time.time()
        result = {
            "provider": provider,
            "perplexity": self.perplexity(
                provider,
                suite["perplexity_passages"],
                suite["perplexity_eval_tokens_per_passage"],
                kv_options,
            ),
            "short_text": self.choices(
                provider, suite["short_text"], label_ids, "short", kv_options
            ),
            "long_context": self.choices(
                provider,
                build_long_examples(suite),
                label_ids,
                "long",
                kv_options,
            ),
            "dialogue": self.choices(
                provider,
                suite["dialogue"],
                label_ids,
                "dialogue",
                kv_options,
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
    environment = capture_cuda_environment(ROOT, args.device)
    if args.mode == "qualification":
        admitted, _, reason = qualification_admission(
            environment, require_reservation=True
        )
        if not admitted:
            raise SystemExit(
                "{}: {}".format(BLOCKED_NOT_EXCLUSIVE, reason)
            )
    suite = read_json(args.suite)
    if int(suite.get("schema_version", 0)) != 1:
        raise SystemExit("unsupported quality suite schema")
    candidate_kv_options = {
        "quest_scorer": args.candidate_quest_scorer,
        "page_budget": args.candidate_page_budget,
        "recent_window": args.candidate_recent_window,
    }
    try:
        quality_kv_policy(
            args.candidate,
            args.page_size,
            "bf16",
            **candidate_kv_options,
        )
    except ValueError as error:
        raise SystemExit("Quest quality policy rejected: {}".format(error))
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
        candidate = runner.run(
            args.candidate, suite, labels, candidate_kv_options
        )

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
        "checkpoint": {
            "name": args.checkpoint.name,
            "local_path_redacted": True,
        },
        "evidence_class": (
            "QUALIFICATION" if args.mode == "qualification" else "SMOKE_ONLY"
        ),
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
        "candidate_kv_policy": quality_kv_policy(
            args.candidate,
            args.page_size,
            "bf16" if runner.dtype_name == "bfloat16" else "fp16",
            **candidate_kv_options,
        ).as_dict(),
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
    contract = load_kv_numerical_contract(args.contract)
    report["quality_gate"] = evaluate_model_quality(contract, report)
    report["status"] = (
        "PASS" if report["quality_gate"]["passed"] else "FAIL"
    )
    report["quest_qualified"] = False
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
