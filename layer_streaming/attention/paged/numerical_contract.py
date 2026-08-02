"""Numerical Contract V2 for exact BF16/FP16 paged attention.

FP32 math attention is the primary reference.  Framework/library attention
kernels are comparison baselines, never the sole source of truth.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


PAGED_NUMERICAL_CONTRACT_VERSION = 2


def _dtype_name(dtype):
    return {
        torch.bfloat16: "bf16",
        torch.float16: "fp16",
        torch.float32: "fp32",
    }.get(dtype, str(dtype).replace("torch.", ""))


@dataclass(frozen=True)
class NumericalErrorMetrics:
    max_abs_error: float
    mean_abs_error: float
    p99_abs_error: float
    max_relative_error: float
    cosine_similarity: float

    def as_dict(self):
        return asdict(self)


def numerical_error_metrics(reference_fp32, actual):
    """Measure one output against FP32 without treating near-zero relative
    error as an end-to-end gate.
    """

    reference = reference_fp32.detach().float()
    candidate = actual.detach().float()
    if reference.shape != candidate.shape:
        raise ValueError("numerical metrics require equal shapes")
    if reference.numel() == 0:
        raise ValueError("numerical metrics require non-empty tensors")
    difference = (reference - candidate).abs()
    scale_floor = max(float(reference.abs().max().item()) * 1.0e-6, 1.0e-8)
    relative = difference / reference.abs().clamp_min(scale_floor)
    flat_reference = reference.reshape(-1)
    flat_candidate = candidate.reshape(-1)
    cosine = float(
        F.cosine_similarity(
            flat_reference.unsqueeze(0),
            flat_candidate.unsqueeze(0),
            dim=1,
            eps=1.0e-12,
        ).item()
    )
    return NumericalErrorMetrics(
        max_abs_error=float(difference.max().item()),
        mean_abs_error=float(difference.mean().item()),
        p99_abs_error=float(torch.quantile(difference.reshape(-1), 0.99).item()),
        max_relative_error=float(relative.max().item()),
        cosine_similarity=cosine,
    )


@dataclass(frozen=True)
class PagedNumericalContract:
    architecture: str
    provider_name: str
    provider_abi: int
    kv_dtype: str
    output_dtype: str
    baseline_error_multiplier: float = 2.0
    version: int = PAGED_NUMERICAL_CONTRACT_VERSION

    def __post_init__(self):
        if int(self.version) != PAGED_NUMERICAL_CONTRACT_VERSION:
            raise ValueError("Paged Numerical Contract must use V2")
        if float(self.baseline_error_multiplier) < 1.0:
            raise ValueError("baseline error multiplier must be >= 1")

    def evaluate(self, reference_fp32, actual, baseline=None, safety_checks=None):
        """Evaluate L0/L1.

        `baseline` is the native BF16/FP16 implementation evaluated against
        the same FP32 math reference.  Omitting it produces a diagnostic-only
        result and can never numerically qualify a provider.
        """

        expected_shape = list(reference_fp32.shape)
        actual_shape = list(actual.shape)
        expected_dtype = str(self.output_dtype)
        actual_dtype = _dtype_name(actual.dtype)
        safety_checks = dict(safety_checks or {})
        l0_checks = {
            "shape_correct": expected_shape == actual_shape,
            "dtype_correct": actual_dtype == expected_dtype,
            "no_nan": not bool(torch.isnan(actual.detach().float()).any().item()),
            "no_inf": not bool(torch.isinf(actual.detach().float()).any().item()),
            "no_out_of_bounds_access": bool(
                safety_checks.get("no_out_of_bounds_access", False)
            ),
            "page_block_indices_valid": bool(
                safety_checks.get("page_block_indices_valid", False)
            ),
        }
        l0_passed = all(l0_checks.values())
        result = {
            "contract_version": int(self.version),
            "reference": "fp32_math_attention",
            "l0": {
                "passed": l0_passed,
                "checks": l0_checks,
                "expected_shape": expected_shape,
                "actual_shape": actual_shape,
                "expected_dtype": expected_dtype,
                "actual_dtype": actual_dtype,
            },
            "l1": {
                "passed": False,
                "baseline_present": baseline is not None,
                "error_multiplier_limit": float(self.baseline_error_multiplier),
            },
        }
        if not l0_passed:
            result["passed"] = False
            return result
        candidate_metrics = numerical_error_metrics(reference_fp32, actual)
        result["l1"]["candidate_vs_fp32"] = candidate_metrics.as_dict()
        if baseline is None:
            result["l1"]["reason"] = (
                "V2 qualification requires a native BF16/FP16 baseline"
            )
            result["passed"] = False
            return result
        if list(baseline.shape) != expected_shape:
            result["l1"]["reason"] = "baseline shape mismatch"
            result["passed"] = False
            return result
        baseline_metrics = numerical_error_metrics(reference_fp32, baseline)
        result["l1"]["baseline_vs_fp32"] = baseline_metrics.as_dict()
        multiplier = float(self.baseline_error_multiplier)
        # Max-relative error remains visible but is not gated because values
        # near zero make it unstable.  The other three error statistics and
        # cosine loss are all bounded relative to the native precision path.
        checks = {
            name: getattr(candidate_metrics, name)
            <= multiplier * getattr(baseline_metrics, name) + 1.0e-12
            for name in (
                "max_abs_error",
                "mean_abs_error",
                "p99_abs_error",
            )
        }
        checks["cosine_loss"] = (
            1.0 - candidate_metrics.cosine_similarity
            <= multiplier * (1.0 - baseline_metrics.cosine_similarity) + 1.0e-7
        )
        result["l1"]["checks"] = checks
        result["l1"]["passed"] = all(checks.values())
        result["passed"] = bool(l0_passed and result["l1"]["passed"])
        return result

    def diagnose_pairwise(self, expected, actual):
        """Non-qualifying pairwise metrics for benchmarks and debugging."""

        if list(expected.shape) != list(actual.shape):
            return {
                "diagnostic_only": True,
                "shape_match": False,
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
            }
        return {
            "diagnostic_only": True,
            "shape_match": True,
            "metrics": numerical_error_metrics(expected.float(), actual).as_dict(),
        }

    def as_dict(self):
        return asdict(self)


def default_paged_numerical_contract(
    architecture, provider_name, kv_dtype, provider_abi=1
):
    dtype = str(kv_dtype)
    if dtype not in {"bf16", "fp16"}:
        raise ValueError("no V2 exact paged contract for dtype {}".format(dtype))
    return PagedNumericalContract(
        architecture=str(architecture),
        provider_name=str(provider_name),
        provider_abi=int(provider_abi),
        kv_dtype=dtype,
        output_dtype=dtype,
    )
