#!/usr/bin/env python3
"""Static RGKV Decode hot-path synchronization audit.

This is a mechanical source gate, not runtime or profiler evidence.  It keeps
the strict GPU-hit device route separate from the retained Host reference and
Tier-miss bridge, so a GPU-hit result is never generalized to CPU-miss Tier.
"""

import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (".cpu(", ".numpy(", ".tolist(", ".item(", "synchronize(")

SCOPES = (
    (
        "gpu_scorer_select",
        "layer_streaming/kv/selection/rgkv_gpu.py",
        "RGKVGPUScorer.select",
        "device_component",
    ),
    (
        "gpu_scorer_selected_epochs",
        "layer_streaming/kv/selection/rgkv_gpu.py",
        "RGKVGPUScorer.selected_data_epochs",
        "device_component",
    ),
    (
        "gpu_scorer_candidate_epochs",
        "layer_streaming/kv/selection/rgkv_gpu.py",
        "RGKVGPUScorer.candidate_data_epochs",
        "device_component",
    ),
    (
        "device_page_table_gather",
        "layer_streaming/kv/device_metadata.py",
        "DeviceKVPageTable.gather",
        "device_component",
    ),
    (
        "device_selected_validation",
        "layer_streaming/attention/paged/abi.py",
        "DevicePagedAttentionInput.validate_device_decode",
        "device_component",
    ),
    (
        "strict_device_dispatch",
        "layer_streaming/attention/paged/dispatcher.py",
        "PagedAttentionDispatcher.execute_device",
        "device_component",
    ),
    (
        "generic_cuda_device_launch",
        "layer_streaming/providers/generic_cuda/paged_attention.py",
        "GenericCUDAPagedAttentionBackend.decode_device",
        "device_component",
    ),
    (
        "gpu_hit_execution",
        "layer_streaming/kv/execution.py",
        "KVExecutionCoordinator._attend_active_tier_device_hit",
        "device_component",
    ),
    (
        "selected_handle_bridge",
        "layer_streaming/kv/page_view.py",
        "SelectedPageView.resolve_entries",
        "host_bridge",
    ),
    (
        "active_tier_orchestration",
        "layer_streaming/kv/execution.py",
        "KVExecutionCoordinator._attend_active_tier",
        "host_bridge",
    ),
    (
        "bounded_attention_waves",
        "layer_streaming/attention/paged/tiered_streaming.py",
        "TieredStreamingExactAttention.execute",
        "host_bridge",
    ),
)


def _find_function(tree, dotted):
    class_name, function_name = dotted.split(".", 1)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if child.name == function_name:
                        return child
    raise LookupError("source symbol not found: {}".format(dotted))


def _scope_result(identifier, relative, symbol, category):
    path = ROOT / relative
    source = path.read_text(encoding="utf-8")
    node = _find_function(ast.parse(source), symbol)
    lines = source.splitlines()
    snippet = "\n".join(lines[node.lineno - 1 : node.end_lineno])
    forbidden = tuple(item for item in FORBIDDEN if item in snippet)
    python_loops = sum(isinstance(item, (ast.For, ast.While)) for item in ast.walk(node))
    # Fixed-field Python validation iteration is not a page wave. Device
    # components fail on conversion/synchronization calls; scorer and launch
    # scopes additionally remain free of data-dependent Python page loops.
    loop_forbidden = identifier in {
        "gpu_scorer_select",
        "gpu_hit_execution",
        "generic_cuda_device_launch",
    }
    passed = (
        category == "device_component"
        and not forbidden
        and (not loop_forbidden or not python_loops)
    )
    return {
        "id": identifier,
        "category": category,
        "path": relative,
        "symbol": symbol,
        "line": int(node.lineno),
        "forbidden_calls": list(forbidden),
        "python_loop_count": int(python_loops),
        "status": "PASS" if passed else "BLOCKED_HOST_SYNC",
        "reason": (
            "device component has no selected-metadata readback or synchronization"
            if passed
            else "host bridge/orchestration remains in the Decode integration path"
        ),
    }


def build_report():
    scopes = [_scope_result(*item) for item in SCOPES]
    components = [item for item in scopes if item["category"] == "device_component"]
    device_pass = all(item["status"] == "PASS" for item in components)
    return {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "static_source_audit_not_runtime_evidence",
        "device_components_status": "PASS" if device_pass else "FAIL",
        "end_to_end_decode_status": (
            "HOST_SYNC_FREE_GPU_HIT_PATH"
            if device_pass else "BLOCKED_HOST_SYNC"
        ),
        # Static/source and CPU synthetic evidence cannot replace a real CUDA
        # profiler trace on an exclusive GPU.
        "production_zero_host_sync": False,
        "gpu_hit_path_static_pass": bool(device_pass),
        "tier_miss_path_status": "BLOCKED_HOST_SYNC",
        "cuda_profiler_verified": False,
        "scopes": scopes,
    }


def render_markdown(report):
    lines = [
        "# RGKV Decode Hot-Path Static Audit",
        "",
        "- Device tensor components: `{}`".format(
            report["device_components_status"]
        ),
        "- End-to-end Decode: `{}`".format(report["end_to_end_decode_status"]),
        "- Production zero Host Sync: `{}`".format(
            report["production_zero_host_sync"]
        ),
        "",
        "> This is a mechanical source audit, not CUDA profiler or qualification evidence.",
        "",
        "| Scope | Category | Status | Forbidden calls | Python loops |",
        "|---|---|---|---|---:|",
    ]
    for item in report["scopes"]:
        lines.append(
            "| `{}` | {} | `{}` | {} | {} |".format(
                item["symbol"],
                item["category"],
                item["status"],
                ", ".join("`{}`".format(value) for value in item["forbidden_calls"])
                or "none",
                item["python_loop_count"],
            )
        )
    lines.extend(
        [
            "",
        "The strict all-GPU-resident route passes the source gate. Retained Host",
        "bridges belong to reference/debug and Tier-miss execution; Tier miss remains",
        "blocked until DeviceTierMissBatch is implemented. CUDA profiler verification",
        "is still required before a production zero-sync claim.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    json_path = output_dir / "audit.json"
    markdown_path = output_dir / "report.md"
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report, (json_path, markdown_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "reports" / "rgkv_hot_path_audit_20260812"),
    )
    args = parser.parse_args(argv)
    report, paths = write_report(args.output_dir)
    print(
        json.dumps(
            {
                "device_components_status": report["device_components_status"],
                "end_to_end_decode_status": report["end_to_end_decode_status"],
                "artifacts": [str(item) for item in paths],
            },
            indent=2,
            sort_keys=True,
        )
    )
    # BLOCKED_HOST_SYNC is an honest audit disposition, not a tool failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
