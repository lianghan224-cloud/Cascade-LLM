#!/usr/bin/env python3
"""Validate the opt-in tensorized RGKV scorer without model claims."""

import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming.kv import PagedKVRuntime  # noqa: E402
from layer_streaming.kv.selection import (  # noqa: E402
    RGKVBudget,
    RGKVCPUReferenceScorer,
)
from layer_streaming.kv_policy import KVPolicy  # noqa: E402
from tools.qualification_common import (  # noqa: E402
    capture_cuda_environment,
    utc_now,
    write_report_bundle,
)


def _policy():
    return KVPolicy(
        accuracy="sparse",
        selection="rgkv",
        attention_backend="reference_paged_exact",
        page_size=16,
        page_budget=2,
        recent_window=16,
        rgkv_scorer="torch_tensorized",
    )


def _runtime(device, layer_count=1):
    return PagedKVRuntime(
        layer_count=layer_count,
        num_query_heads=4,
        num_kv_heads=2,
        head_dim=8,
        page_count=8,
        page_size=16,
        dtype=torch.bfloat16,
        device=device,
        policy=_policy(),
        allow_reference=True,
    )


def run_selection_case(device):
    torch.manual_seed(20260809)
    runtime = _runtime(device)
    state = runtime.create_request(96)
    key = torch.randn(
        65, 2, 8, dtype=torch.bfloat16, device=torch.device(device)
    )
    runtime.append((state,), 0, key, key, (65,))
    query = torch.randn(
        1, 4, 8, dtype=torch.bfloat16, device=torch.device(device)
    )
    batch = runtime.prepare_batch((state,), (1,), 0)
    selected = runtime.selection.select((state,), 0, query, batch)
    index = runtime.selection.indexes[(state.request_id, 0)]
    page_epochs = {
        logical: runtime.page_pool.descriptor(handle).data_version
        for logical, handle in enumerate(state.block_table.handles)
    }
    cpu = RGKVCPUReferenceScorer().select(
        index,
        query,
        RGKVBudget(total_page_budget=2, recent_pages=1),
        page_epochs=page_epochs,
    )
    expected = cpu.selected_logical_pages.tolist()
    actual = selected.logical_block_ids.detach().cpu().tolist()
    handles = selected.resolve_handles((state,), runtime.page_pool)
    if actual != expected:
        raise AssertionError(
            "tensorized selection {} != CPU reference {}".format(
                actual, expected
            )
        )
    if selected.flat_page_ids.device != torch.device(device):
        raise AssertionError("selected page tensor left the query device")
    if len(handles) != 2 or not all(item.generation > 0 for item in handles):
        raise AssertionError("selected generation-bearing handles are invalid")
    runtime.page_pool.validate_invariants()
    profile = runtime.profile_stats()
    scorer_stats = profile["rgkv_scorer_stats"]
    runtime.close()
    return {
        "selected_logical_pages": actual,
        "cpu_reference_pages": expected,
        "selected_device": str(selected.flat_page_ids.device),
        "scorer_provider": profile["kv_selection_scorer"],
        "scorer_stats": scorer_stats,
        "quality_measured": False,
        "performance_measured": False,
    }


def run_prefill_case(device):
    runtime = _runtime(device, layer_count=2)
    state = runtime.create_request(64)
    key = torch.randn(
        17, 2, 8, dtype=torch.bfloat16, device=torch.device(device)
    )
    runtime.append((state,), 0, key, key, (17,))
    query = torch.randn(
        17, 4, 8, dtype=torch.bfloat16, device=torch.device(device)
    )
    batch = runtime.prepare_batch((state,), (17,), 0)
    selected = runtime.selection.select((state,), 0, query, batch)
    reason = selected.metadata["requests"][0].get("fallback_reason")
    if not selected.exact or reason != "rgkv_selection_is_decode_only":
        raise AssertionError("RGKV Prefill did not preserve exact selection")
    if runtime.selection.records:
        raise AssertionError("RGKV index was published before append commit")
    runtime.append((state,), 1, key, key, (17,))
    records_after_commit = len(runtime.selection.records)
    if records_after_commit != 4:
        raise AssertionError("RGKV index publication did not cover both layers")
    runtime.close()
    return {
        "prefill_exact": True,
        "fallback_reason": reason,
        "records_before_commit": 0,
        "records_after_commit": records_after_commit,
    }


def render_report(summary, cases):
    lines = [
        "# RGKV Tensorized Scorer Validation",
        "",
        "- Mode: `{}`".format(summary["mode"]),
        "- Status: `{}`".format(summary["status"]),
        "- Capability state: `{}`".format(summary["capability_state"]),
        "- Evidence class: `{}`".format(summary["evidence_class"]),
        "- Real-model quality executed: `false`",
        "- Performance benchmark executed: `false`",
        "",
        "| Case | Status | Evidence |",
        "|---|---|---|",
    ]
    for case in cases:
        lines.append(
            "| {} | `{}` | `{}` |".format(
                case["case_id"], case["status"], case["evidence"]
            )
        )
    lines.extend(
        [
            "",
            "CUDA smoke is shared, point-in-time diagnostic evidence only. It does not qualify RGKV quality, latency, throughput, or production readiness.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("logic", "cuda-smoke"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir", default=str(ROOT / "reports" / "rgkv_tensorized")
    )
    args = parser.parse_args(argv)
    environment = capture_cuda_environment(ROOT, args.device)
    if args.mode == "cuda-smoke" and not environment.get("cuda_available"):
        raise SystemExit("CUDA is required for cuda-smoke mode")
    device = "cpu" if args.mode == "logic" else args.device
    evidence = "LOGIC_VALIDATED" if args.mode == "logic" else "SMOKE_ONLY"
    cases = []
    for case_id, runner in (
        ("RGKV-SCORING", run_selection_case),
        ("RGKV-PREFILL-COMMIT", run_prefill_case),
    ):
        try:
            metrics = runner(device)
        except BaseException as error:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "FAIL",
                    "evidence": evidence,
                    "reason": "{}: {}".format(type(error).__name__, error),
                    "metrics": {},
                }
            )
        else:
            cases.append(
                {
                    "case_id": case_id,
                    "status": "PASS",
                    "evidence": evidence,
                    "reason": None,
                    "metrics": metrics,
                }
            )
    passed = sum(item["status"] == "PASS" for item in cases)
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "mode": args.mode,
        "status": "PASS" if passed == len(cases) else "FAIL",
        "capability_state": (
            "LOGIC_VALIDATED" if args.mode == "logic" else "CUDA_SMOKE"
        ),
        "evidence_class": evidence,
        "case_count": len(cases),
        "pass_count": passed,
        "real_model_quality_executed": False,
        "performance_benchmark_executed": False,
        "rgkv_qualified": False,
    }
    paths = write_report_bundle(
        args.output_dir,
        environment,
        cases,
        summary,
        render_report(summary, cases),
    )
    print(paths["report.md"])
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
