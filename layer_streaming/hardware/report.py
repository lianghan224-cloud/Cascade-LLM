"""Compatibility report schema kept inside RunReport.hardware."""

from dataclasses import dataclass, field
import json
from pathlib import Path
import time
from typing import Optional, Tuple

from .capability import CompatibilityDecision
from .profile import HardwareProfile, RuntimeFeatureProfile


COMPATIBILITY_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CompatibilityReport:
    hardware: HardwareProfile
    runtime: RuntimeFeatureProfile
    providers: Tuple[dict, ...]
    decisions: Tuple[CompatibilityDecision, ...]
    selected_backend: Optional[str]
    overall_status: str
    schema_version: int = COMPATIBILITY_REPORT_SCHEMA_VERSION
    created_at_unix: float = field(default_factory=time.time)

    def as_dict(self):
        return {
            "schema_version": self.schema_version,
            "created_at_unix": self.created_at_unix,
            "hardware": self.hardware.as_dict(),
            "runtime": self.runtime.as_dict(),
            "providers": [dict(item) for item in self.providers],
            "decisions": [item.as_dict() for item in self.decisions],
            "selected_backend": self.selected_backend,
            "overall_status": self.overall_status,
        }

    def to_json(self, indent=2):
        return json.dumps(self.as_dict(), indent=indent, ensure_ascii=False)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version = int(value.get("schema_version", 0))
        if version != COMPATIBILITY_REPORT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported CompatibilityReport schema {}".format(version)
            )
        decisions = tuple(
            CompatibilityDecision(
                supported=bool(item["supported"]),
                provider_name=item.get("provider_name"),
                status=str(item["status"]),
                reasons=tuple(item.get("reasons", ())),
                warnings=tuple(item.get("warnings", ())),
                explicit_fallback_available=bool(
                    item.get("explicit_fallback_available", False)
                ),
            )
            for item in value.get("decisions", ())
        )
        return cls(
            hardware=HardwareProfile.from_dict(value["hardware"]),
            runtime=RuntimeFeatureProfile.from_dict(value["runtime"]),
            providers=tuple(value.get("providers", ())),
            decisions=decisions,
            selected_backend=value.get("selected_backend"),
            overall_status=str(value["overall_status"]),
            schema_version=version,
            created_at_unix=float(value.get("created_at_unix", time.time())),
        )

    @classmethod
    def from_json(cls, payload):
        return cls.from_dict(json.loads(payload))


def build_compatibility_report(
    hardware,
    runtime,
    registry,
    decisions=(),
    selected_backend=None,
):
    decisions = tuple(decisions)
    if decisions and all(item.supported for item in decisions):
        if all(item.status in {"qualified", "production"} for item in decisions):
            overall = "qualified"
        else:
            overall = "unqualified"
    elif decisions:
        overall = "unsupported"
    else:
        overall = "unqualified"
    return CompatibilityReport(
        hardware=hardware,
        runtime=runtime,
        providers=tuple(
            item.as_dict() for item in registry.list_providers()
        ),
        decisions=decisions,
        selected_backend=selected_backend,
        overall_status=overall,
    )
