"""Architecture/provider keyed paged-attention numerical contracts."""

from dataclasses import dataclass

import torch


PAGED_NUMERICAL_CONTRACT_VERSION = 1


@dataclass(frozen=True)
class PagedNumericalContract:
    architecture: str
    provider_name: str
    provider_abi: int
    kv_dtype: str
    output_dtype: str
    max_abs_error: float
    mean_abs_error: float
    relative_error: float
    version: int = PAGED_NUMERICAL_CONTRACT_VERSION

    def evaluate(self, expected, actual):
        """Evaluate a kernel output without hiding any individual metric."""

        expected = expected.detach().float()
        actual = actual.detach().float()
        if expected.shape != actual.shape:
            return {
                "passed": False,
                "reason": "shape mismatch",
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
            }
        difference = (expected - actual).abs()
        max_abs = float(difference.max().item())
        mean_abs = float(difference.mean().item())
        relative_l2 = float(
            torch.linalg.vector_norm(difference).item()
            / max(torch.linalg.vector_norm(expected).item(), 1.0e-12)
        )
        passed = (
            max_abs <= float(self.max_abs_error)
            and mean_abs <= float(self.mean_abs_error)
            and relative_l2 <= float(self.relative_error)
        )
        return {
            "passed": bool(passed),
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "relative_l2_error": relative_l2,
            "limits": {
                "max_abs_error": float(self.max_abs_error),
                "mean_abs_error": float(self.mean_abs_error),
                "relative_l2_error": float(self.relative_error),
            },
        }

    def as_dict(self):
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


def default_paged_numerical_contract(
    architecture, provider_name, kv_dtype, provider_abi=1
):
    """Return the V1 single-kernel contract.

    These limits gate one attention invocation against the two-pass FP32
    paged reference.  End-to-end model drift is reported separately and may
    not be converted into a kernel qualification pass.
    """

    dtype = str(kv_dtype)
    if dtype == "bf16":
        limits = (1.7e-2, 1.5e-3, 3.0e-3)
    elif dtype == "fp16":
        limits = (2.1e-3, 2.5e-4, 5.0e-4)
    else:
        raise ValueError("no V1 paged contract for dtype {}".format(dtype))
    return PagedNumericalContract(
        architecture=str(architecture),
        provider_name=str(provider_name),
        provider_abi=int(provider_abi),
        kv_dtype=dtype,
        output_dtype=dtype,
        max_abs_error=limits[0],
        mean_abs_error=limits[1],
        relative_error=limits[2],
    )
