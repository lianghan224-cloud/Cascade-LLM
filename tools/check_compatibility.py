#!/usr/bin/env python3
"""Check every Llama linear shape before execution; never select fallback."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from layer_streaming import ExecutionPolicy, LlamaModelAdapter  # noqa: E402
from layer_streaming.hardware import (  # noqa: E402
    CompatibilityRequest,
    CompatibilityResolver,
    HardwareDetector,
    build_compatibility_report,
    default_provider_registry,
)


def _dtype_tag(value):
    return {"bfloat16": "bf16", "float16": "fp16"}.get(value, value)


def _requests(geometry, policy, backend, phase, m, workspace):
    quant = policy.quantization
    if quant is None:
        weight_format = _dtype_tag(policy.embedding_dtype)
        scale_dtype = None
        group_size = None
    else:
        weight_format = "int{}_{}_{}".format(
            quant.bits, quant.scheme, quant.granularity
        )
        scale_dtype = _dtype_tag(quant.scale_dtype)
        group_size = quant.group_size
    hidden = geometry.hidden_size
    intermediate = geometry.intermediate_size
    kv_width = geometry.kv_width
    shapes = (
        ("attention_q", hidden, hidden),
        ("attention_k", kv_width, hidden),
        ("attention_v", kv_width, hidden),
        ("attention_o", hidden, hidden),
        ("mlp_gate", intermediate, hidden),
        ("mlp_up", intermediate, hidden),
        ("mlp_down", hidden, intermediate),
    )
    return tuple(
        (
            name,
            CompatibilityRequest(
                phase=phase,
                backend_requested=backend,
                weight_format=weight_format,
                activation_dtype=_dtype_tag(
                    "bfloat16"
                    if "bf16" in policy.weight_format.value
                    else "float16"
                ),
                scale_dtype=scale_dtype,
                group_size=group_size,
                m=int(m),
                n=int(n),
                k=int(k),
                physical_layout="row_major",
                workspace_limit_bytes=int(workspace),
            ),
        )
        for name, n, k in shapes
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--workspace-limit-bytes", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-cutlass", action="store_true")
    parser.add_argument("--provider-library")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.load_cutlass:
        from layer_streaming.providers.cutlass import load_cutlass_w8a16_provider

        load_cutlass_w8a16_provider(args.provider_library)
    config_path = args.checkpoint / "config.json"
    if not config_path.is_file():
        raise SystemExit("missing checkpoint config: {}".format(config_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    geometry = LlamaModelAdapter().build_geometry(config)
    policy = ExecutionPolicy.from_config(config)
    hardware, runtime = HardwareDetector().detect(args.device, refresh=True)
    registry = default_provider_registry()
    resolver = CompatibilityResolver(registry)
    named_decisions = tuple(
        (name, resolver.resolve(hardware, runtime, request))
        for name, request in _requests(
            geometry,
            policy,
            args.backend,
            args.phase,
            args.m,
            args.workspace_limit_bytes,
        )
    )
    report = build_compatibility_report(
        hardware,
        runtime,
        registry,
        decisions=tuple(item[1] for item in named_decisions),
        selected_backend=args.backend,
    )
    payload = report.as_dict()
    for target, (_, decision) in zip(payload["decisions"], named_decisions):
        target["matrix"] = _
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if all(item[1].supported for item in named_decisions) else 2


if __name__ == "__main__":
    raise SystemExit(main())
