#!/usr/bin/env python3
"""Fail-fast environment check for the real Llama-3.1-8B experiment."""

import json
import os
import shutil
import sys
from pathlib import Path


MODEL_ID = "meta-llama/Llama-3.1-8B"
EXPECTED_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 128256,
}


def bytes_free(path):
    usage = shutil.disk_usage(path)
    return {"total": usage.total, "used": usage.used, "free": usage.free}


def main():
    root = Path(
        os.environ.get(
            "CASCADE_ROOT",
            Path(__file__).resolve().parents[1],
        )
    )
    storage = Path(os.environ.get("CASCADE_STORAGE_ROOT", "/ssd/cascade-llm"))
    model_path = Path(
        os.environ.get(
            "CASCADE_LLAMA31_8B",
            storage / "models" / "Llama-3.1-8B",
        )
    )
    report = {
        "python": sys.version,
        "project_root": str(root),
        "storage_root": str(storage),
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "storage": bytes_free(storage),
        "hf_token_present": False,
        "checkpoint": {"present": model_path.is_dir()},
        "cuda": {"available": False, "devices": []},
        "errors": [],
    }

    try:
        from huggingface_hub import HfFolder

        report["hf_token_present"] = bool(HfFolder.get_token())
    except Exception as error:
        report["errors"].append("huggingface_hub: {}".format(error))

    try:
        import torch

        report["torch"] = torch.__version__
        report["cuda"]["available"] = torch.cuda.is_available()
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report["cuda"]["devices"].append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "capability": [
                        props.major,
                        props.minor,
                    ],
                }
            )
    except Exception as error:
        report["errors"].append("torch: {}".format(error))

    config_path = model_path / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        report["checkpoint"]["config"] = {
            key: config.get(key) for key in EXPECTED_CONFIG
        }
        for key, expected in EXPECTED_CONFIG.items():
            if config.get(key) != expected:
                report["errors"].append(
                    "config {}={} expected {}".format(
                        key,
                        config.get(key),
                        expected,
                    )
                )
    safetensors = (
        sorted(model_path.glob("*.safetensors"))
        if model_path.is_dir()
        else []
    )
    report["checkpoint"]["safetensor_files"] = len(safetensors)
    report["checkpoint"]["safetensor_bytes"] = sum(
        path.stat().st_size for path in safetensors
    )

    if not report["cuda"]["available"]:
        report["errors"].append("CUDA is unavailable in the project venv")
    if report["storage"]["free"] < 30 * 1024 ** 3:
        report["errors"].append("less than 30 GiB free under storage root")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
