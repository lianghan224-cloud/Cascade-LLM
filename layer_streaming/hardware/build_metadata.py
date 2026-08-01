"""Validated provider build metadata."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Tuple


PROVIDER_BUILD_METADATA_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProviderBuildMetadata:
    provider: str
    provider_version: str
    abi: int
    compiled_architectures: Tuple[str, ...]
    weight_formats: Tuple[str, ...]
    activation_dtypes: Tuple[str, ...]
    build_environment: dict
    status: str = "compiled"
    schema_version: int = PROVIDER_BUILD_METADATA_SCHEMA_VERSION

    def __post_init__(self):
        if not self.provider:
            raise ValueError("provider must be non-empty")
        if self.abi < 0:
            raise ValueError("provider ABI must be non-negative")
        if self.status not in {"declared", "compiled"}:
            raise ValueError("build metadata status must be declared or compiled")
        if self.status == "compiled" and not self.compiled_architectures:
            raise ValueError("compiled metadata requires an architecture")
        for architecture in self.compiled_architectures:
            if architecture not in {"sm75", "sm80", "sm86", "sm89", "sm90"}:
                raise ValueError(
                    "unknown compiled architecture {}".format(architecture)
                )
        required = {"cuda", "compiler", "cutlass"}
        missing = required.difference(self.build_environment)
        if missing:
            raise ValueError(
                "build environment missing {}".format(", ".join(sorted(missing)))
            )

    def as_dict(self):
        value = asdict(self)
        for name in (
            "compiled_architectures",
            "weight_formats",
            "activation_dtypes",
        ):
            value[name] = list(value[name])
        return value

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
        if version != PROVIDER_BUILD_METADATA_SCHEMA_VERSION:
            raise ValueError(
                "unsupported provider metadata schema {}".format(version)
            )
        for name in (
            "compiled_architectures",
            "weight_formats",
            "activation_dtypes",
        ):
            value[name] = tuple(value.get(name, ()))
        return cls(**value)

    @classmethod
    def read(cls, path):
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
