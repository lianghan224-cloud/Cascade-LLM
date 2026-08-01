"""Versioned compatibility contract for the frozen M5 core interfaces."""

from dataclasses import fields
import hashlib
import inspect
import json

from .adapter import ModelGeometry
from .backends import LinearBackend, default_backend_registry
from .checkpoint import CheckpointManifest
from .execution_plan import (
    EXECUTION_PLAN_SCHEMA_VERSION,
    ExecutionPlan,
)
from .mixed_runtime import MixedDtypeRuntime
from .reporting import RUN_REPORT_SCHEMA_VERSION, RunReport
from .specs import QuantizationSpec, WeightSpec


CORE_API_VERSION = 1

_FROZEN_DATACLASSES = (
    ModelGeometry,
    WeightSpec,
    QuantizationSpec,
    CheckpointManifest,
    ExecutionPlan,
    RunReport,
)


def _field_names(dataclass_type):
    return [item.name for item in fields(dataclass_type)]


def _parameter_names(callable_object):
    return list(inspect.signature(callable_object).parameters)


def core_api_contract():
    """Return a deterministic, JSON-safe description of the public contract."""

    registry = default_backend_registry()
    return {
        "core_api_version": CORE_API_VERSION,
        "schema_versions": {
            "execution_plan": EXECUTION_PLAN_SCHEMA_VERSION,
            "run_report": RUN_REPORT_SCHEMA_VERSION,
        },
        "dataclass_fields": {
            item.__name__: _field_names(item)
            for item in _FROZEN_DATACLASSES
        },
        "linear_backend": {
            "attributes": ["name", "is_fallback", "info"],
            "methods": {
                name: _parameter_names(getattr(LinearBackend, name))
                for name in (
                    "validate",
                    "transfer_bytes",
                    "workspace_bytes",
                    "execute",
                )
            },
        },
        "mixed_runtime": {
            "class": "MixedDtypeRuntime",
            "alias": "MixedRuntime",
            "methods": {
                "__init__": _parameter_names(MixedDtypeRuntime.__init__),
                "run": _parameter_names(MixedDtypeRuntime.run),
                "close": _parameter_names(MixedDtypeRuntime.close),
            },
            "properties": ["stats"],
        },
        "backend_registry": {
            name: {
                "is_fallback": bool(backend.is_fallback),
                "storage_dtype": backend.info.storage_dtype,
                "activation_dtype": backend.info.activation_dtype,
                "output_dtype": backend.info.output_dtype,
            }
            for name, backend in sorted(registry.items())
        },
    }


def canonical_contract_json():
    return json.dumps(
        core_api_contract(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def core_api_contract_sha256():
    return hashlib.sha256(
        canonical_contract_json().encode("utf-8")
    ).hexdigest()
