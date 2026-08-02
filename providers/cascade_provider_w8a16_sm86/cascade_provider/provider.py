"""Entry-point wrapper around the performance-qualified SM86 binary."""

from dataclasses import dataclass
import json
from pathlib import Path

from layer_streaming.hardware import ProviderCapability
from layer_streaming.providers.cutlass.provider import CutlassW8A16Provider


ROOT = Path(__file__).resolve().parent


def _capability():
    value = json.loads(
        (ROOT / "capability.json").read_text(encoding="utf-8")
    )
    tuple_fields = (
        "supported_architectures",
        "supported_weight_formats",
        "supported_activation_dtypes",
        "supported_scale_dtypes",
        "supported_group_sizes",
        "supported_phases",
        "physical_layout_names",
    )
    for name in tuple_fields:
        value[name] = tuple(value[name])
    return ProviderCapability(**value)


@dataclass(frozen=True)
class Sm86W8A16Plugin:
    name: str = "cascade_provider_w8a16_sm86"
    capability: ProviderCapability = _capability()

    def create_provider(self):
        library = ROOT / "lib" / "libcascade_cutlass_sm86.so"
        if not library.is_file():
            raise FileNotFoundError(
                "performance-qualified SM86 binary is missing: {}".format(library)
            )
        return CutlassW8A16Provider(library=library)


def create_plugin():
    return Sm86W8A16Plugin()
