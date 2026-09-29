"""One conservative capability vocabulary for qualification reports.

Provider ABI status strings remain frozen for compatibility.  New project
reports use :class:`CapabilityState` and explicitly map legacy declarations so
that a numerical micro-test cannot be mistaken for system qualification.
"""

from dataclasses import asdict, dataclass
from enum import Enum


class CapabilityState(str, Enum):
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    EXPERIMENTAL = "EXPERIMENTAL"
    LOGIC_VALIDATED = "LOGIC_VALIDATED"
    CUDA_SMOKE = "CUDA_SMOKE"
    QUALIFICATION_READY = "QUALIFICATION_READY"
    QUALIFIED = "QUALIFIED"


CAPABILITY_STATES = tuple(item.value for item in CapabilityState)


def capability_state(value):
    if isinstance(value, CapabilityState):
        return value.value
    try:
        return CapabilityState(str(value).upper()).value
    except ValueError:
        raise ValueError(
            "invalid capability state {!r}; expected one of {}".format(
                value, ", ".join(CAPABILITY_STATES)
            )
        )


LEGACY_QUALIFICATION_MAP = {
    "unsupported": CapabilityState.NOT_IMPLEMENTED.value,
    "declared": CapabilityState.EXPERIMENTAL.value,
    "compiled": CapabilityState.EXPERIMENTAL.value,
    "experimental": CapabilityState.EXPERIMENTAL.value,
    "smoke_passed": CapabilityState.CUDA_SMOKE.value,
    # Numerical Provider evidence alone is not system qualification.
    "numerically_qualified": CapabilityState.CUDA_SMOKE.value,
    "performance_qualified": CapabilityState.QUALIFIED.value,
    "production": CapabilityState.QUALIFIED.value,
}


def map_legacy_qualification(value):
    try:
        return LEGACY_QUALIFICATION_MAP[str(value).lower()]
    except KeyError:
        raise ValueError("unknown legacy qualification status {!r}".format(value))


@dataclass(frozen=True)
class CapabilityEntry:
    capability: str
    state: str
    evidence: tuple = ()
    blockers: tuple = ()
    notes: str = ""

    def __post_init__(self):
        object.__setattr__(self, "state", capability_state(self.state))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "blockers", tuple(self.blockers))
        if self.state == CapabilityState.QUALIFIED.value and not self.evidence:
            raise ValueError("QUALIFIED capability requires evidence")

    def as_dict(self):
        return asdict(self)


def dense_kv_capability_matrix():
    """Return the current conservative matrix; qualification tools may promote it."""

    return (
        CapabilityEntry(
            "GPU Dense Page Arena",
            "QUALIFICATION_READY",
            evidence=(
                "reports/kv_qualification_entry/summary.json",
                "reports/kv_qualification_entry_cuda_smoke/summary.json",
            ),
            blockers=("exclusive CUDA long-stability matrix",),
        ),
        CapabilityEntry(
            "Ownership Logic",
            "QUALIFICATION_READY",
            evidence=(
                "reports/kv_qualification_entry/summary.json",
                "reports/kv_long_stability_full_logic/summary.json",
            ),
            blockers=("exclusive CUDA fault injection",),
        ),
        CapabilityEntry(
            "CUDA COW",
            "QUALIFICATION_READY",
            evidence=("reports/kv_cuda_async_cuda_smoke/summary.json",),
            blockers=("qualification-mode execution on exclusive GPU",),
        ),
        CapabilityEntry(
            "Attention Fence",
            "QUALIFICATION_READY",
            evidence=("reports/kv_cuda_async_cuda_smoke/summary.json",),
            blockers=("cross-stream qualification on exclusive GPU",),
        ),
        CapabilityEntry(
            "GenerationSession",
            "LOGIC_VALIDATED",
            evidence=("tests/test_generation_session.py",),
            blockers=("real tokenizer and 70B CUDA smoke",),
        ),
        CapabilityEntry(
            "Gather SDPA Prefill",
            "CUDA_SMOKE",
            evidence=("reports/gather_sdpa_prefill_cuda_smoke/summary.json",),
            blockers=(
                "exclusive CUDA 128/512/2K/8K matrix",
                "real 70B reference/gather A/B",
            ),
        ),
        CapabilityEntry(
            "RGKV Compact Summary Index",
            "LOGIC_VALIDATED",
            evidence=(
                "tests/test_rgkv_index.py",
                "tests/test_rgkv_lifecycle_transactions.py",
            ),
            blockers=("real-model quality qualification",),
        ),
        CapabilityEntry(
            "RGKV Tensorized Scorer",
            "CUDA_SMOKE",
            evidence=(
                "tests/test_rgkv_api.py",
                "reports/quest_tensorized_logic/summary.json",
                "reports/quest_tensorized_cuda_smoke/summary.json",
            ),
            blockers=(
                "real-model quality qualification",
                "exclusive repeated long-context CUDA validation",
            ),
        ),
        CapabilityEntry(
            "RGKV Lifecycle",
            "LOGIC_VALIDATED",
            evidence=(
                "tests/test_rgkv_lifecycle_transactions.py",
                "reports/kv_long_stability_rgkv_active_tier_logic_20260812/summary.json",
            ),
            blockers=(
                "exclusive CUDA append/fence failure matrix",
                "Active Tier fork/COW/prefix/rollback remains unsupported",
            ),
        ),
        CapabilityEntry(
            "Device Selected Page Metadata and Epoch Validation",
            "LOGIC_VALIDATED",
            evidence=(
                "tests/test_device_kv_metadata.py",
                "tests/test_rgkv_active_tier.py",
                "reports/rgkv_hot_path_audit_20260812/audit.json",
            ),
            blockers=(
                "selected-handle/location host bridge still synchronizes",
                "exclusive CUDA stale-epoch validation",
            ),
        ),
        CapabilityEntry(
            "Pinned CPU Layer Page Store",
            "CUDA_SMOKE",
            evidence=(
                "tests/test_pinned_cpu_store.py",
                "reports/kv_active_tier_cuda_smoke/summary.json",
            ),
            blockers=("exclusive repeated CUDA migration validation",),
        ),
        CapabilityEntry(
            "GPU to Pinned CPU Tensor Migration",
            "CUDA_SMOKE",
            evidence=(
                "tests/test_tensor_migration.py",
                "reports/kv_active_tier_cuda_smoke/summary.json",
            ),
            blockers=(
                "exclusive cross-stream fault qualification",
                "real-model long-context validation",
            ),
        ),
        CapabilityEntry(
            "GPU Hot KV Cache and Location Plane",
            "CUDA_SMOKE",
            evidence=(
                "tests/test_gpu_hot_cache.py",
                "reports/kv_active_tier_cuda_smoke/summary.json",
            ),
            blockers=("exclusive long-stability CUDA validation",),
        ),
        CapabilityEntry(
            "Active Tier Prefetch and Eviction",
            "CUDA_SMOKE",
            evidence=(
                "tests/test_active_tier_coordinator.py",
                "reports/kv_active_tier_cuda_smoke/summary.json",
            ),
            blockers=("exclusive cancellation/fault CUDA matrix",),
        ),
        CapabilityEntry(
            "Active GPU to Pinned CPU KV Tier",
            "CUDA_SMOKE",
            evidence=(
                "tests/test_active_tier_runtime.py",
                "reports/kv_active_tier_logic/summary.json",
                "reports/kv_active_tier_cuda_smoke/summary.json",
            ),
            blockers=(
                "exclusive Tiered CUDA qualification",
                "real 70B long-context A/B",
            ),
        ),
        CapabilityEntry(
            "GenerationSession Active Tier",
            "CUDA_SMOKE",
            evidence=(
                "tests/test_generation_session_active_tier.py",
                "reports/kv_active_tier_cuda_smoke/summary.json",
            ),
            blockers=("real-model session CUDA qualification",),
        ),
        CapabilityEntry(
            "RGKV Selected-only Active Tier",
            "LOGIC_VALIDATED",
            evidence=(
                "tests/test_rgkv_active_tier.py",
                "tests/test_generation_session_active_tier.py",
                "reports/kv_long_stability_rgkv_active_tier_logic_20260812/summary.json",
            ),
            blockers=(
                "Decode selected-handle/location host sync remains",
                "exclusive CUDA selected-only prefetch qualification",
                "real 70B quality and E2E performance gates",
            ),
        ),
    )
