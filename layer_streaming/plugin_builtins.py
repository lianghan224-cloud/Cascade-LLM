"""Core plugins exposed through the same entry-point protocol as extensions."""

from dataclasses import dataclass
import json
import os
from pathlib import Path

from .adapter import LlamaModelAdapter
from .quantization_cache import (
    checkpoint_digest,
    quantization_cache_key,
    valid_quantization_cache,
    write_quantization_cache_manifest,
)
from .version import __version__


@dataclass(frozen=True)
class LlamaAdapterPlugin:
    name: str = "llama"
    model_types: tuple = ("llama",)

    def create_adapter(self):
        return LlamaModelAdapter()


@dataclass(frozen=True)
class Int8PerChannelQuantizerPlugin:
    name: str = "int8_per_channel"

    def run(self, argv):
        from .cli import run_tool

        argv = list(argv)
        source = _option(argv, "--input")
        if source is None:
            return run_tool("quantize_llama_checkpoint.py", argv)
        output = _option(argv, "--output")
        cache_key = None
        cache_inputs = None
        if output is None:
            source_path = Path(source).resolve()
            config = json.loads(
                (source_path / "config.json").read_text(encoding="utf-8")
            )
            geometry = LlamaModelAdapter().build_geometry(config)
            cache_key, cache_inputs = quantization_cache_key(
                checkpoint_digest(source_path),
                geometry.as_dict(),
                "int8_symmetric_per_channel",
                None,
                _option(argv, "--scale-dtype", "bf16"),
                "row_major",
                2,
                __version__,
            )
            cache_root = Path(
                os.environ.get("CASCADE_CACHE_ROOT", "/cache/cascade")
            )
            output_path = cache_root / "quantized" / cache_key
            manifest_path = output_path / "manifest.json"
            if valid_quantization_cache(manifest_path, cache_key):
                print(
                    json.dumps(
                        {
                            "cache_hit": True,
                            "cache_key": cache_key,
                            "output": str(output_path),
                        },
                        indent=2,
                    )
                )
                return 0
            argv.extend(("--output", str(output_path)))
            output = str(output_path)
        result = run_tool("quantize_llama_checkpoint.py", argv)
        if result == 0 and cache_key is not None:
            write_quantization_cache_manifest(
                Path(output) / "manifest.json", cache_key, cache_inputs
            )
        return result


def _option(argv, name, default=None):
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return default


def create_llama_adapter_plugin():
    return LlamaAdapterPlugin()


def create_int8_per_channel_quantizer():
    return Int8PerChannelQuantizerPlugin()
