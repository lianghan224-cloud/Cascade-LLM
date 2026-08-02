#!/usr/bin/env python3
"""Aggregate L0-L5 evidence into the five stable KV report schemas."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layer_streaming.kv.reports import (  # noqa: E402
    KVCompatibilityReport,
    KVNumericalReport,
    KVOwnershipReport,
    KVPerformanceReport,
    KVQualificationReport,
)
from layer_streaming.numerical_contracts import (  # noqa: E402
    evaluate_logits_generation,
    evaluate_model_quality,
    evaluate_model_stages,
    evaluate_production_stability,
    load_kv_numerical_contract,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--model-comparison", type=Path, required=True)
    parser.add_argument("--attribution", type=Path, required=True)
    parser.add_argument("--long-replay", type=Path, required=True)
    parser.add_argument("--soak", type=Path, required=True)
    parser.add_argument("--ownership", type=Path, required=True)
    parser.add_argument("--automated-tests", type=Path, required=True)
    parser.add_argument("--performance", type=Path, nargs="+", required=True)
    parser.add_argument("--quality", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path, report):
    path.write_text(report.to_json() + "\n", encoding="utf-8")


def main():
    args = parse_args()
    contract = load_kv_numerical_contract(args.contract)
    hardware = read(args.hardware)
    comparison = read(args.model_comparison)
    attribution = read(args.attribution)
    long_replay = read(args.long_replay)
    soak = read(args.soak)
    ownership = read(args.ownership)
    automated_tests = read(args.automated_tests)
    performance = [read(path) for path in args.performance]
    quality = None if args.quality is None else read(args.quality)

    provider_item = hardware.get("providers", {}).get(contract.provider_name, {})
    cases = provider_item.get("cases", ())
    l0 = {
        "level": "L0",
        "passed": bool(cases)
        and all(case.get("numerical_contract_v2", {}).get("l0", {}).get("passed", False) for case in cases),
        "case_count": len(cases),
        "checks": [
            case.get("numerical_contract_v2", {}).get("l0", {})
            for case in cases
        ],
    }
    l1 = {
        "level": "L1",
        "passed": bool(cases)
        and all(case.get("numerical_contract_v2", {}).get("l1", {}).get("passed", False) for case in cases),
        "reference": "fp32_math_attention",
        "case_count": len(cases),
        "checks": [
            case.get("numerical_contract_v2", {}).get("l1", {})
            for case in cases
        ],
    }
    l2 = evaluate_model_stages(contract, comparison, attribution)
    l3 = evaluate_logits_generation(contract, comparison, long_replay)
    l4 = evaluate_model_quality(contract, quality)
    l5 = evaluate_production_stability(contract, soak, ownership, performance)
    levels = {item["level"]: item for item in (l0, l1, l2, l3, l4, l5)}

    performance_results = [
        item
        for report in performance
        for item in report.get("results", ())
        if item.get("supported", False)
        and item.get("provider") not in {
            "reference_paged_exact",
            "legacy_gather_sdpa_reference",
        }
    ]
    short_acceptance = performance[0].get("acceptance", {}) if performance else {}
    sm86_speedups = [
        float(item.get("speedup_vs_generic", 0.0))
        for item in performance_results
        if item.get("provider") == contract.provider_name
    ]
    performance_checks = {
        "cases_present": bool(performance_results),
        "workspace_zero": bool(performance_results)
        and all(int(item.get("workspace_peak_bytes", -1)) == 0 for item in performance_results),
        "representative_faster_than_reference": bool(
            short_acceptance.get("production_faster_than_reference", False)
        ),
        # Seven-run microbenchmarks still carry sub-percent host timing noise.
        # Treat <=1% as equivalent, while requiring a positive median gain.
        "sm86_no_material_generic_regression": bool(sm86_speedups)
        and min(sm86_speedups) >= 0.99,
        "sm86_median_improves_generic": bool(sm86_speedups)
        and statistics.median(sm86_speedups) > 1.0,
    }
    performance_passed = all(performance_checks.values())
    numerical_passed = all(levels[name]["passed"] for name in ("L0", "L1", "L2", "L3"))
    automated_tests_passed = bool(automated_tests.get("passed", False))
    production_passed = (
        all(item["passed"] for item in levels.values())
        and performance_passed
        and automated_tests_passed
    )
    if production_passed:
        final_status = "production"
    elif numerical_passed and performance_passed and automated_tests_passed:
        final_status = "performance_qualified"
    elif numerical_passed:
        final_status = "numerically_qualified"
    elif l0["passed"] and l1["passed"]:
        final_status = "smoke_passed"
    else:
        final_status = "compiled"

    common = {
        "provider": contract.provider_name,
        "architecture": contract.architecture,
        "status": final_status,
    }
    compatibility_report = KVCompatibilityReport(
        **common,
        evidence={
            "physical_hardware": hardware.get("hardware", {}),
            "automated_tests": automated_tests,
            "bundle_load_path_present": provider_item.get(
                "bundle_load_path_present", False
            ),
        },
        capability=provider_item.get("declared_capability", {}),
    )
    numerical_report = KVNumericalReport(
        **common,
        evidence={
            "contract": str(args.contract),
            "strict_hf_diagnostic_failures": l2[
                "strict_hf_diagnostic_failure_count"
            ],
        },
        levels=levels,
    )
    performance_report = KVPerformanceReport(
        **common,
        evidence={"input_report_count": len(performance)},
        matrix={
            "passed": performance_passed,
            "checks": performance_checks,
            "case_count": len(performance_results),
        },
    )
    ownership_report = KVOwnershipReport(
        **common,
        evidence={"cycles": ownership.get("cycles", 0)},
        checks=ownership.get("acceptance", {}),
    )
    qualification_report = KVQualificationReport(
        **common,
        evidence={
            "contract_name": contract.name,
            "automated_tests_passed": automated_tests_passed,
            "missing_production_levels": [
                name for name, value in levels.items() if not value["passed"]
            ],
        },
        levels={
            name: {"passed": value["passed"]}
            for name, value in levels.items()
        },
        production_passed=production_passed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "compatibility": ("kv_compatibility_summary.json", compatibility_report),
        "numerical": ("kv_numerical_summary.json", numerical_report),
        "performance": ("kv_performance_summary.json", performance_report),
        "ownership": ("kv_ownership_summary.json", ownership_report),
        "qualification": ("kv_qualification_summary.json", qualification_report),
    }
    for _, (filename, report) in reports.items():
        write_report(args.output_dir / filename, report)

    matrix_rows = []
    for architecture in ("sm80", "sm86", "sm89", "sm90"):
        item = hardware.get("providers", {}).get(architecture, {})
        status = final_status if architecture == contract.architecture else item.get("status", "declared")
        matrix_rows.append(
            {
                "architecture": architecture,
                "physical_hardware": bool(item.get("physical_hardware_available", False)),
                "provider_loaded": bool(item.get("bundle_load_path_present", False)),
                "status": status,
                "numerical": (
                    "passed" if architecture == contract.architecture and numerical_passed else "not_run"
                ),
                "performance": (
                    "passed" if architecture == contract.architecture and performance_passed else "not_run"
                ),
                "production": (
                    "passed" if architecture == contract.architecture and production_passed else "not_passed"
                ),
            }
        )
    matrix = {
        "schema_version": 1,
        "contract_version": contract.schema_version,
        "rows": matrix_rows,
    }
    (args.output_dir / "compatibility_matrix.json").write_text(
        json.dumps(matrix, indent=2) + "\n", encoding="utf-8"
    )
    markdown = [
        "# KV Provider Compatibility Matrix",
        "",
        "| Architecture | Physical | Loaded | Numerical | Performance | Production | Status |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for row in matrix_rows:
        markdown.append(
            "| {architecture} | {physical_hardware} | {provider_loaded} | "
            "{numerical} | {performance} | {production} | `{status}` |".format(
                **row
            )
        )
    (args.output_dir / "compatibility_matrix.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )

    inputs = [
        args.contract,
        args.hardware,
        args.model_comparison,
        args.attribution,
        args.long_replay,
        args.soak,
        args.ownership,
        args.automated_tests,
    ] + list(args.performance)
    if args.quality is not None:
        inputs.append(args.quality)
    manifest = {
        "schema_version": 1,
        "qualification_status": final_status,
        "production_passed": production_passed,
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in inputs
        ],
        "summaries": {
            name: filename for name, (filename, _) in reports.items()
        },
        "compatibility_matrix": "compatibility_matrix.json",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
