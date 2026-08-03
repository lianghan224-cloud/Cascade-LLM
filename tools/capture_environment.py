#!/usr/bin/env python3
"""Capture a secret-free, machine-readable Cascade-LLM environment receipt."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    "torch",
    "transformers",
    "safetensors",
    "huggingface-hub",
    "tokenizers",
    "accelerate",
    "numpy",
    "psutil",
    "sentencepiece",
    "packaging",
    "pip",
    "setuptools",
    "wheel",
)


def command(args, cwd=None, timeout=10):
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"ok": False, "error": "{}: {}".format(type(error).__name__, error)}
    return {
        "ok": completed.returncode == 0,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def sanitize_remote_url(url):
    """Remove HTTP(S) user information before persisting a Git remote URL."""
    if not url or "://" not in url:
        return url
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = "[{}]".format(hostname)
    netloc = hostname
    if parsed.port is not None:
        netloc = "{}:{}".format(netloc, parsed.port)
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def torch_environment():
    try:
        import torch
    except Exception as error:
        return {"available": False, "error": "{}: {}".format(type(error).__name__, error)}
    result = {
        "available": True,
        "version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cudnn": torch.backends.cudnn.version(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            result["gpus"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
    return result


def git_environment():
    revision = command(("git", "rev-parse", "HEAD"), cwd=ROOT)
    branch = command(("git", "branch", "--show-current"), cwd=ROOT)
    status = command(("git", "status", "--short"), cwd=ROOT)
    remote = command(("git", "remote", "get-url", "origin"), cwd=ROOT)
    return {
        "revision": revision.get("stdout") if revision.get("ok") else None,
        "branch": branch.get("stdout") if branch.get("ok") else None,
        "origin": sanitize_remote_url(remote.get("stdout")) if remote.get("ok") else None,
        "dirty": bool(status.get("stdout")),
        "status": status.get("stdout", "").splitlines(),
    }


def nvidia_environment():
    result = command(
        (
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        )
    )
    if not result.get("ok"):
        return result
    rows = []
    for line in result["stdout"].splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 5:
            rows.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "uuid": fields[2],
                    "driver_version": fields[3],
                    "memory_total_mib": int(fields[4]),
                }
            )
    return {"ok": True, "gpus": rows}


def lock_receipts():
    result = {}
    for relative in ("requirements.lock", "docker/versions.env", "pyproject.toml"):
        path = ROOT / relative
        result[relative] = {
            "exists": path.is_file(),
            "sha256": sha256(path) if path.is_file() else None,
        }
    return result


def build_receipt():
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": sys.version.replace("\n", " "),
            "python_executable": sys.executable,
        },
        "git": git_environment(),
        "packages": package_versions(),
        "torch": torch_environment(),
        "nvidia_smi": nvidia_environment(),
        "locks": lock_receipts(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    rendered = json.dumps(build_receipt(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
