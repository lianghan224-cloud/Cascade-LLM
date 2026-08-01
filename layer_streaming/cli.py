"""Stable ``cascade`` command used by both host installs and containers."""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys

from .hardware import (
    HardwareDetector,
    build_compatibility_report,
    default_provider_registry,
)
from .plugins import discover_plugins, quantizer_plugins
from .container_contracts import ImageManifest
from .version import __version__


COMMANDS = (
    "doctor",
    "inspect",
    "validate",
    "run",
    "chat",
    "benchmark",
    "qualify",
    "quantize",
    "shell",
)
BACKEND_ALIASES = {"cutlass_w8a16": "fused_w8a16"}


def project_root():
    configured = os.environ.get("CASCADE_HOME")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[1]


def result_root():
    return Path(os.environ.get("CASCADE_RESULT_ROOT", "/results"))


def _normalize_backend_aliases(argv):
    result = list(argv)
    for index, value in enumerate(result):
        if value in {"--backend", "--prefill-backend", "--decode-backend"}:
            if index + 1 < len(result):
                result[index + 1] = BACKEND_ALIASES.get(
                    result[index + 1], result[index + 1]
                )
    return result


def _ensure_option(argv, option, value):
    if option in argv or any(item.startswith(option + "=") for item in argv):
        return list(argv)
    return list(argv) + [option, str(value)]


def _option_value(argv, option, default=None):
    for index, value in enumerate(argv):
        if value == option and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(option + "="):
            return value.split("=", 1)[1]
    return default


def run_tool(name, argv):
    """Run a legacy tool in-process so discovered plugins remain registered."""

    path = project_root() / "tools" / str(name)
    if not path.is_file():
        raise SystemExit("Cascade tool is missing: {}".format(path))
    tools_path = str(path.parent)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    previous_argv = sys.argv
    sys.argv = [str(path)] + list(argv)
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as error:
        code = error.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 1
    finally:
        sys.argv = previous_argv
    return 0


def _image_manifest():
    configured = os.environ.get("CASCADE_IMAGE_MANIFEST")
    candidates = []
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        (
            Path("/opt/cascade/image-manifest.json"),
            project_root() / "image-manifest.json",
        )
    )
    for path in candidates:
        if path.is_file():
            try:
                manifest = ImageManifest.read(path)
                return str(path), manifest.as_dict(), ""
            except (OSError, ValueError) as error:
                return str(path), None, "{}: {}".format(type(error).__name__, error)
    return None, None, "not running from a manifested image"


def doctor(argv, plugin_records):
    parser = argparse.ArgumentParser(prog="cascade doctor")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args(argv)
    hardware, runtime = HardwareDetector().detect(args.device, refresh=True)
    registry = default_provider_registry()
    compatibility = build_compatibility_report(hardware, runtime, registry)
    manifest_path, manifest, manifest_error = _image_manifest()
    plugin_failures = [item for item in plugin_records if not item.loaded]
    loaded_distributions = {
        item.distribution.lower().replace("_", "-")
        for item in plugin_records
        if item.loaded
    }
    expected_provider_packages = {
        str(item["package"]).lower().replace("_", "-")
        for item in (manifest or {}).get("providers", ())
    }
    missing_provider_packages = sorted(
        expected_provider_packages.difference(loaded_distributions)
    )
    if manifest and manifest.get("cascade_version") != __version__:
        manifest_error = "image/core Cascade version mismatch"
    ok = (
        not plugin_failures
        and not missing_provider_packages
        and not (manifest_path and manifest_error)
        and (runtime.cuda_available or not args.require_cuda)
    )
    payload = {
        "schema_version": 1,
        "ok": ok,
        "mode": args.mode,
        "cascade_version": __version__,
        "image_manifest_path": manifest_path,
        "image_manifest": manifest,
        "image_manifest_error": manifest_error,
        "missing_provider_packages": missing_provider_packages,
        "hardware": hardware.as_dict(),
        "runtime": runtime.as_dict(),
        "compatibility_status": compatibility.overall_status,
        "plugins": [item.as_dict() for item in plugin_records],
    }
    if args.mode == "full":
        payload["providers"] = [
            item.as_dict() for item in registry.list_providers()
        ]
        payload["directories"] = {
            name: {
                "path": value,
                "exists": Path(value).exists(),
                "writable": os.access(value, os.W_OK),
            }
            for name, value in {
                "models": os.environ.get("CASCADE_MODEL_ROOT", "/models"),
                "config": os.environ.get("CASCADE_CONFIG_ROOT", "/config"),
                "cascade_cache": os.environ.get(
                    "CASCADE_CACHE_ROOT", "/cache/cascade"
                ),
                "huggingface_cache": os.environ.get(
                    "HF_HOME", "/cache/huggingface"
                ),
                "results": os.environ.get("CASCADE_RESULT_ROOT", "/results"),
            }.items()
        }
    rendered = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if ok else 1


def _usage():
    return """usage: cascade <command> [arguments]

Stable commands:
  doctor      inspect environment, plugins and image manifest
  inspect     export the hardware compatibility report
  validate    validate checkpoint, tokenizer and memory plan
  run         run Llama-family prefill/decode
  chat        interactive or single-prompt chat
  benchmark   execute the unified benchmark suite
  qualify     create a hardware qualification worksheet
  quantize    convert a checkpoint with a quantization plugin
  shell       open an interactive container shell
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0
    if argv[0] in {"-V", "--version"}:
        print(__version__)
        return 0
    command, arguments = argv[0], argv[1:]
    if command not in COMMANDS:
        print("unknown cascade command: {}".format(command), file=sys.stderr)
        print(_usage(), file=sys.stderr)
        return 2
    strict_plugins = command not in {"doctor", "inspect", "qualify"}
    try:
        plugin_records = discover_plugins(strict=strict_plugins)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1
    if command == "doctor":
        return doctor(arguments, plugin_records)
    if command == "inspect":
        return run_tool("inspect_hardware.py", arguments)
    if command == "validate":
        arguments = _normalize_backend_aliases(arguments)
        backend = _option_value(arguments, "--backend", "checkpoint")
        checkpoint = _option_value(arguments, "--checkpoint")
        if (
            backend != "checkpoint"
            and checkpoint is not None
            and "--metadata-only" not in arguments
        ):
            for phase, m in (("prefill", "8"), ("decode", "1")):
                compatibility_arguments = [
                    "--checkpoint",
                    checkpoint,
                    "--backend",
                    backend,
                    "--phase",
                    phase,
                    "--m",
                    m,
                    "--output",
                    str(result_root() / ("compatibility_{}.json".format(phase))),
                ]
                device = _option_value(arguments, "--device")
                if device is not None:
                    compatibility_arguments.extend(("--device", device))
                result = run_tool(
                    "check_compatibility.py", compatibility_arguments
                )
                if result:
                    return result
        arguments = _ensure_option(
            arguments, "--output", result_root() / "validation.json"
        )
        return run_tool("accept_checkpoint.py", arguments)
    if command == "run":
        arguments = _normalize_backend_aliases(arguments)
        arguments = _ensure_option(
            arguments, "--output", result_root() / "run_report.json"
        )
        return run_tool("run_llama31.py", arguments)
    if command == "chat":
        arguments = _normalize_backend_aliases(arguments)
        return run_tool("chat_llama31_70b_int8.py", arguments)
    if command == "benchmark":
        arguments = _normalize_backend_aliases(arguments)
        arguments = _ensure_option(
            arguments, "--output", result_root() / "benchmark.json"
        )
        return run_tool("benchmark.py", arguments)
    if command == "qualify":
        arguments = _ensure_option(
            arguments, "--output", result_root() / "qualification.json"
        )
        return run_tool("qualify_hardware.py", arguments)
    if command == "quantize":
        plugin_name = "int8_per_channel"
        if arguments[:1] == ["--plugin"]:
            if len(arguments) < 2:
                print("--plugin requires a name", file=sys.stderr)
                return 2
            plugin_name, arguments = arguments[1], arguments[2:]
        plugin = quantizer_plugins().get(plugin_name)
        if plugin is None:
            print(
                "quantization plugin {} is not installed".format(plugin_name),
                file=sys.stderr,
            )
            return 2
        return int(plugin.run(arguments) or 0)
    shell = os.environ.get("SHELL", "/bin/bash")
    os.execvp(shell, [shell])
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
