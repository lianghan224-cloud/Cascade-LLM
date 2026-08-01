"""Cascade version sourced from docker/versions.env during source builds."""

from importlib import metadata
from pathlib import Path


def _source_version():
    root = Path(__file__).resolve().parents[1]
    path = root / "docker" / "versions.env"
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "CASCADE_VERSION":
            return value.strip()
    return None


def _installed_version():
    try:
        return metadata.version("cascade-llm")
    except metadata.PackageNotFoundError:
        return None


__version__ = _source_version() or _installed_version() or "0+unknown"
