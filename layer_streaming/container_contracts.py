"""Validated Docker bundle and image-manifest schemas."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional, Tuple


PROVIDER_BUNDLE_SCHEMA_VERSION = 1
IMAGE_MANIFEST_SCHEMA_VERSION = 1


def load_versions(path):
    result = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip() or not value.strip():
            raise ValueError("invalid versions.env line {!r}".format(raw_line))
        if key.strip() in result:
            raise ValueError("duplicate version key {}".format(key.strip()))
        result[key.strip()] = value.strip()
    return result


def _safe_relative_path(value, field):
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("{} must be a safe relative path".format(field))
    return str(path)


@dataclass(frozen=True)
class BundleProvider:
    package: str
    source: str
    binary: str
    binary_target: str
    build_metadata: str
    numerical_contract: str
    contract_target: str
    provider_abi: int
    architecture: str
    qualification: str

    def __post_init__(self):
        if not self.package:
            raise ValueError("provider package must be non-empty")
        if self.qualification not in {"performance_qualified", "production"}:
            raise ValueError(
                "installable providers must be performance_qualified or production"
            )
        if self.architecture not in {"sm75", "sm80", "sm86", "sm89", "sm90"}:
            raise ValueError("unsupported provider architecture")
        if int(self.provider_abi) < 1:
            raise ValueError("provider ABI must be positive")
        for field in (
            "source",
            "binary",
            "binary_target",
            "build_metadata",
            "numerical_contract",
            "contract_target",
        ):
            object.__setattr__(
                self, field, _safe_relative_path(getattr(self, field), field)
            )


@dataclass(frozen=True)
class ProviderBundleManifest:
    name: str
    providers: Tuple[BundleProvider, ...]
    declarations: Tuple[dict, ...]
    bundle_version: int = PROVIDER_BUNDLE_SCHEMA_VERSION

    def __post_init__(self):
        if self.bundle_version != PROVIDER_BUNDLE_SCHEMA_VERSION:
            raise ValueError(
                "unsupported provider bundle version {}".format(
                    self.bundle_version
                )
            )
        if not self.name:
            raise ValueError("bundle name must be non-empty")
        packages = [item.package for item in self.providers]
        if len(packages) != len(set(packages)):
            raise ValueError("provider bundle contains duplicate packages")

    def as_dict(self):
        return {
            "bundle_version": self.bundle_version,
            "name": self.name,
            "providers": [asdict(item) for item in self.providers],
            "declarations": [dict(item) for item in self.declarations],
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        return cls(
            name=str(value["name"]),
            providers=tuple(
                BundleProvider(**item) for item in value.get("providers", ())
            ),
            declarations=tuple(value.get("declarations", ())),
            bundle_version=int(value.get("bundle_version", 0)),
        )

    @classmethod
    def read(cls, path):
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )


@dataclass(frozen=True)
class ImageManifest:
    cascade_version: str
    git_commit: str
    execution_plan_schema: int
    run_report_schema: int
    hardware_compatibility_schema: int
    numerical_contract_version: int
    python: str
    torch: str
    cuda_runtime: str
    provider_bundle: str
    providers: Tuple[dict, ...]
    image_type: str
    schema_version: int = IMAGE_MANIFEST_SCHEMA_VERSION

    def as_dict(self):
        value = asdict(self)
        value["providers"] = [dict(item) for item in self.providers]
        return value

    def to_json(self):
        return json.dumps(self.as_dict(), indent=2, ensure_ascii=False)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json() + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        version = int(value.get("schema_version", 0))
        if version != IMAGE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported image manifest version {}".format(version))
        value["providers"] = tuple(value.get("providers", ()))
        return cls(**value)

    @classmethod
    def read(cls, path):
        return cls.from_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )
