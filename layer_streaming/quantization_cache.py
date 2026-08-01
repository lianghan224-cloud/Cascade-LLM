"""Content-addressed quantization cache keys and completion manifests."""

import hashlib
import json
from pathlib import Path


QUANTIZATION_CACHE_SCHEMA_VERSION = 1


def checkpoint_digest(checkpoint, chunk_bytes=8 * 1024 * 1024):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_dir():
        raise ValueError("checkpoint must be a directory")
    candidates = []
    for path in checkpoint.iterdir():
        if path.is_file() and (
            path.suffix in {".json", ".safetensors", ".model"}
            or path.name.startswith("tokenizer")
        ):
            candidates.append(path)
    if not candidates:
        raise ValueError("checkpoint contains no hashable model files")
    digest = hashlib.sha256()
    for path in sorted(candidates, key=lambda item: item.name):
        relative = path.relative_to(checkpoint).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(int(path.stat().st_size).to_bytes(8, "little"))
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(int(chunk_bytes))
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def quantization_cache_key(
    checkpoint_sha256,
    geometry,
    quantization_format,
    group_size,
    scale_dtype,
    physical_layout,
    provider_abi,
    converter_version,
):
    payload = {
        "checkpoint_sha256": str(checkpoint_sha256),
        "geometry": dict(geometry),
        "quantization_format": str(quantization_format),
        "group_size": group_size,
        "scale_dtype": str(scale_dtype),
        "physical_layout": str(physical_layout),
        "provider_abi": int(provider_abi),
        "converter_version": str(converter_version),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256-{}".format(hashlib.sha256(encoded).hexdigest()), payload


def write_quantization_cache_manifest(path, cache_key, inputs):
    path = Path(path)
    payload = {
        "schema_version": QUANTIZATION_CACHE_SCHEMA_VERSION,
        "complete": True,
        "cache_key": str(cache_key),
        "inputs": dict(inputs),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def valid_quantization_cache(path, cache_key):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        int(value.get("schema_version", 0))
        == QUANTIZATION_CACHE_SCHEMA_VERSION
        and bool(value.get("complete"))
        and value.get("cache_key") == cache_key
    )
