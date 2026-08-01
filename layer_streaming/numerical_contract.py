"""Versioned, evidence-based numerical gates for fused diagnostics."""


def _maximum(items, field):
    return max((float(item[field]) for item in items), default=0.0)


def evaluate_fused_diagnostic(report, contract):
    """Return observations and violations for one diagnostic report.

    The contract deliberately uses aggregate error and task-level Top-k gates
    in addition to the elementwise bound.  Relative error at values near zero
    is not used as an end-to-end gate.
    """

    if int(contract.get("schema_version", 0)) != 1:
        raise ValueError("unsupported fused numerical contract schema")
    violations = []
    configuration = report["configuration"]
    cases = contract.get("golden_cases", ())
    if cases and not any(
        list(case["input_ids"]) == list(configuration["input_ids"])
        and list(case["decode_ids"]) == list(configuration["decode_ids"])
        for case in cases
    ):
        violations.append("input/decode IDs are not a registered golden case")
    if report.get("provider") != contract.get("provider"):
        violations.append(
            "provider {} != {}".format(
                report.get("provider"), contract.get("provider")
            )
        )
    if int(report.get("provider_abi", -1)) != int(contract["provider_abi"]):
        violations.append(
            "provider ABI {} != {}".format(
                report.get("provider_abi"), contract["provider_abi"]
            )
        )
    if report.get("model_signature") != contract.get("model_signature"):
        violations.append("model signature does not match golden contract")

    local = report.get("linear_diagnostics", ())
    local_rules = contract["local_linear"]
    observations = {
        "linear_comparison_count": len(local),
        "local_elements_over_allclose_bound_max": max(
            (
                int(item["fused_vs_fallback"]["elements_over_allclose_bound"])
                for item in local
            ),
            default=0,
        ),
        "local_max_abs_error": _maximum(
            [item["fused_vs_fallback"] for item in local], "max_abs_error"
        ),
        "local_mean_relative_error_max": _maximum(
            [item["fused_vs_fallback"] for item in local],
            "mean_relative_error",
        ),
        "fused_vs_fp32_mean_relative_error_max": _maximum(
            [item["fused_vs_fp32_accumulation"] for item in local],
            "mean_relative_error",
        ),
        "storage_padding_count": sum(
            bool(item["weight"]["has_storage_padding"]) for item in local
        ),
        "unaligned_k_count": sum(
            not bool(item["weight"]["k_aligned_16"]) for item in local
        ),
        "unaligned_n_count": sum(
            not bool(item["weight"]["n_aligned_8"]) for item in local
        ),
        "dtype_layout_mismatch_count": sum(
            not (
                item["input"]["dtype"] == local_rules["activation_dtype"]
                and item["output"]["dtype"] == local_rules["output_dtype"]
                and item["weight"]["storage_dtype"]
                == local_rules["weight_storage_dtype"]
                and item["weight"]["compute_dtype"]
                == local_rules["weight_compute_dtype"]
                and item["scale"]["dtype"] == local_rules["scale_dtype"]
                and item["weight"]["quantization"]["granularity"]
                == local_rules["granularity"]
            )
            for item in local
        ),
    }
    checks = (
        ("local_elements_over_allclose_bound_max", "elements_over_allclose_bound_max"),
        ("local_max_abs_error", "max_abs_error"),
        ("local_mean_relative_error_max", "mean_relative_error_max"),
        (
            "fused_vs_fp32_mean_relative_error_max",
            "fused_vs_fp32_mean_relative_error_max",
        ),
        ("storage_padding_count", "storage_padding_count_max"),
        ("unaligned_k_count", "unaligned_k_count_max"),
        ("unaligned_n_count", "unaligned_n_count_max"),
        ("dtype_layout_mismatch_count", "dtype_layout_mismatch_count_max"),
    )
    for observed_name, rule_name in checks:
        if observations[observed_name] > local_rules[rule_name]:
            violations.append(
                "{}={} exceeds {}".format(
                    observed_name,
                    observations[observed_name],
                    local_rules[rule_name],
                )
            )

    propagation = report.get("error_propagation", ())
    observations["stages"] = {}
    for stage, rules in contract["end_to_end_stages"].items():
        items = [
            item
            for item in propagation
            if item["stage"].split("/")[1] == stage
        ]
        stage_observation = {
            "count": len(items),
            "max_abs_error": _maximum(items, "max_abs_error"),
            "mean_abs_error_max": _maximum(items, "mean_abs_error"),
        }
        if stage == "logits":
            stage_observation.update(
                {
                    "top1_all_equal": all(
                        bool(item.get("top1_equal", False)) for item in items
                    ),
                    "topk_consistency_min": min(
                        (
                            float(item.get("topk_consistency", 0.0))
                            for item in items
                        ),
                        default=0.0,
                    ),
                }
            )
        observations["stages"][stage] = stage_observation
        if not items:
            violations.append("stage {} has no observations".format(stage))
            continue
        if len(items) != int(rules["expected_count"]):
            violations.append(
                "{} observation count={} != {}".format(
                    stage, len(items), rules["expected_count"]
                )
            )
        if stage_observation["max_abs_error"] > rules["max_abs_error"]:
            violations.append(
                "{} max_abs_error={} exceeds {}".format(
                    stage,
                    stage_observation["max_abs_error"],
                    rules["max_abs_error"],
                )
            )
        if (
            stage_observation["mean_abs_error_max"]
            > rules["mean_abs_error_max"]
        ):
            violations.append(
                "{} mean_abs_error_max={} exceeds {}".format(
                    stage,
                    stage_observation["mean_abs_error_max"],
                    rules["mean_abs_error_max"],
                )
            )
        if stage == "logits":
            if rules.get("require_top1_equal") and not stage_observation[
                "top1_all_equal"
            ]:
                violations.append("logits Top-1 differs")
            if (
                stage_observation["topk_consistency_min"]
                < rules["topk_consistency_min"]
            ):
                violations.append(
                    "logits Top-k consistency={} is below {}".format(
                        stage_observation["topk_consistency_min"],
                        rules["topk_consistency_min"],
                    )
                )
    if len(local) != int(local_rules["expected_count"]):
        violations.append(
            "linear comparison count={} != {}".format(
                len(local), local_rules["expected_count"]
            )
        )
    return {
        "contract_name": contract["name"],
        "contract_schema_version": contract["schema_version"],
        "passed": not violations,
        "observations": observations,
        "violations": violations,
    }
