"""Stable entry-point discovery for providers, adapters and quantizers."""

from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Protocol, Tuple, runtime_checkable

from .adapter import register_model_adapter
from .backends import register_linear_backend
from .hardware.capability import ProviderCapability
from .hardware.registry import register_provider_capability


PROVIDER_ENTRY_POINT = "cascade_llm.providers"
MODEL_ADAPTER_ENTRY_POINT = "cascade_llm.model_adapters"
QUANTIZER_ENTRY_POINT = "cascade_llm.quantizers"


@runtime_checkable
class ProviderPlugin(Protocol):
    name: str
    capability: ProviderCapability

    def create_provider(self):
        ...


@runtime_checkable
class ModelAdapterPlugin(Protocol):
    name: str
    model_types: Tuple[str, ...]

    def create_adapter(self):
        ...


@runtime_checkable
class QuantizationPlugin(Protocol):
    name: str

    def run(self, argv):
        ...


@dataclass(frozen=True)
class PluginLoadRecord:
    group: str
    entry_point: str
    distribution: str
    loaded: bool
    registered_name: str
    error: str

    def as_dict(self):
        return asdict(self)


_QUANTIZERS = {}
_DISCOVERY_CACHE = None


def _entry_points(group):
    discovered = metadata.entry_points()
    if hasattr(discovered, "select"):
        return tuple(discovered.select(group=group))
    return tuple(discovered.get(group, ()))


def _distribution_name(entry_point):
    distribution = getattr(entry_point, "dist", None)
    return str(getattr(distribution, "name", "unknown"))


def _instantiate(entry_point):
    value = entry_point.load()
    return value() if callable(value) else value


def _load_provider(entry_point):
    plugin = _instantiate(entry_point)
    provider = (
        plugin.create_provider()
        if callable(getattr(plugin, "create_provider", None))
        else plugin
    )
    capability = getattr(plugin, "capability", None)
    if not isinstance(capability, ProviderCapability):
        raise TypeError("provider plugin must expose ProviderCapability")
    registered = register_linear_backend(provider)
    register_provider_capability(capability)
    return capability.provider_name or registered.name


def _load_adapter(entry_point):
    plugin = _instantiate(entry_point)
    model_types = tuple(getattr(plugin, "model_types", ()))
    creator = getattr(plugin, "create_adapter", None)
    if not model_types or not callable(creator):
        raise TypeError("adapter plugin must expose model_types/create_adapter")
    for model_type in model_types:
        register_model_adapter(model_type, creator, replace=True)
    return str(getattr(plugin, "name", entry_point.name))


def _load_quantizer(entry_point):
    plugin = _instantiate(entry_point)
    name = str(getattr(plugin, "name", "")).strip()
    if not name or not callable(getattr(plugin, "run", None)):
        raise TypeError("quantizer plugin must expose name/run")
    if name in _QUANTIZERS:
        raise ValueError("quantizer {} is already registered".format(name))
    _QUANTIZERS[name] = plugin
    return name


def discover_plugins(strict=False):
    """Load each installed entry point exactly once and report every failure."""

    global _DISCOVERY_CACHE
    if _DISCOVERY_CACHE is not None:
        if strict:
            failed = [item for item in _DISCOVERY_CACHE if not item.loaded]
            if failed:
                raise RuntimeError(failed[0].error)
        return _DISCOVERY_CACHE
    records = []
    loaders = {
        PROVIDER_ENTRY_POINT: _load_provider,
        MODEL_ADAPTER_ENTRY_POINT: _load_adapter,
        QUANTIZER_ENTRY_POINT: _load_quantizer,
    }
    for group, loader in loaders.items():
        for entry_point in _entry_points(group):
            try:
                registered_name = loader(entry_point)
                record = PluginLoadRecord(
                    group=group,
                    entry_point=entry_point.name,
                    distribution=_distribution_name(entry_point),
                    loaded=True,
                    registered_name=registered_name,
                    error="",
                )
            except BaseException as error:
                record = PluginLoadRecord(
                    group=group,
                    entry_point=entry_point.name,
                    distribution=_distribution_name(entry_point),
                    loaded=False,
                    registered_name="",
                    error="{}: {}".format(type(error).__name__, error),
                )
                if strict:
                    raise RuntimeError(
                        "failed to load {} entry point {}: {}".format(
                            group, entry_point.name, record.error
                        )
                    ) from error
            records.append(record)
    loaded_groups = {
        item.group for item in records if item.loaded
    }
    if MODEL_ADAPTER_ENTRY_POINT not in loaded_groups:
        from .plugin_builtins import create_llama_adapter_plugin

        plugin = create_llama_adapter_plugin()
        for model_type in plugin.model_types:
            register_model_adapter(model_type, plugin.create_adapter, replace=True)
        records.append(
            PluginLoadRecord(
                group=MODEL_ADAPTER_ENTRY_POINT,
                entry_point="llama",
                distribution="cascade-llm-core",
                loaded=True,
                registered_name=plugin.name,
                error="",
            )
        )
    if QUANTIZER_ENTRY_POINT not in loaded_groups:
        from .plugin_builtins import create_int8_per_channel_quantizer

        plugin = create_int8_per_channel_quantizer()
        _QUANTIZERS[plugin.name] = plugin
        records.append(
            PluginLoadRecord(
                group=QUANTIZER_ENTRY_POINT,
                entry_point="int8_per_channel",
                distribution="cascade-llm-core",
                loaded=True,
                registered_name=plugin.name,
                error="",
            )
        )
    _DISCOVERY_CACHE = tuple(records)
    return _DISCOVERY_CACHE


def quantizer_plugins():
    return dict(_QUANTIZERS)
