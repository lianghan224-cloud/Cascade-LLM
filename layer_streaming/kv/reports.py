"""Stable, serializable KV qualification report schemas."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from ..qualification_status import qualification_status


KV_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class _KVReport:
    provider: str
    architecture: str
    status: str
    evidence: dict = field(default_factory=dict)
    schema_version: int = KV_REPORT_SCHEMA_VERSION

    def __post_init__(self):
        object.__setattr__(self, "status", qualification_status(self.status))
        if int(self.schema_version) != KV_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported KV report schema")

    def as_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version = int(value.get("schema_version", 0))
        if version != KV_REPORT_SCHEMA_VERSION:
            raise ValueError(
                "unsupported KV report schema {}".format(version)
            )
        return cls(**value)

    @classmethod
    def from_json(cls, payload):
        return cls.from_dict(json.loads(payload))


@dataclass(frozen=True)
class KVCompatibilityReport(_KVReport):
    capability: dict = field(default_factory=dict)


@dataclass(frozen=True)
class KVNumericalReport(_KVReport):
    levels: dict = field(default_factory=dict)


@dataclass(frozen=True)
class KVPerformanceReport(_KVReport):
    matrix: dict = field(default_factory=dict)


@dataclass(frozen=True)
class KVOwnershipReport(_KVReport):
    checks: dict = field(default_factory=dict)


@dataclass(frozen=True)
class KVQualificationReport(_KVReport):
    levels: dict = field(default_factory=dict)
    production_passed: bool = False
