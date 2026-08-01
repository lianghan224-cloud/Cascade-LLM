"""Build the standalone CUTLASS SM86 provider shared library."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ...hardware import ProviderBuildMetadata


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _cuda_runtime_root(explicit=None):
    if explicit:
        return Path(explicit).resolve()
    try:
        import nvidia.cuda_runtime

        return Path(nvidia.cuda_runtime.__file__).resolve().parent
    except ImportError:
        return None


def _nvcc(explicit=None):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        candidates.append(Path(cuda_home) / "bin" / "nvcc")
    discovered = shutil.which("nvcc")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def parser():
    result = argparse.ArgumentParser()
    result.add_argument(
        "--cutlass-root",
        default=os.environ.get("CUTLASS_ROOT"),
        help="CUTLASS 3.5.1 source root (or set CUTLASS_ROOT).",
    )
    result.add_argument("--nvcc")
    result.add_argument("--cuda-runtime-root")
    result.add_argument(
        "--output",
        default=str(HERE / "_build" / "libcascade_cutlass_sm86.so"),
    )
    result.add_argument("--verbose", action="store_true")
    return result


def main():
    args = parser().parse_args()
    cutlass_root = (
        Path(args.cutlass_root).resolve() if args.cutlass_root else None
    )
    nvcc = _nvcc(args.nvcc)
    runtime_root = _cuda_runtime_root(args.cuda_runtime_root)
    errors = []
    if cutlass_root is None or not (cutlass_root / "include/cutlass").is_dir():
        errors.append(
            "CUTLASS_ROOT must point to a CUTLASS source tree (tested: v3.5.1)"
        )
    if nvcc is None:
        errors.append("nvcc was not found; pass --nvcc or set CUDA_HOME")
    if runtime_root is None or not (
        runtime_root / "include/cuda_runtime.h"
    ).is_file():
        errors.append(
            "CUDA runtime headers were not found; pass --cuda-runtime-root"
        )
    if errors:
        raise SystemExit("\n".join(errors))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    runtime_lib = runtime_root / "lib"
    command = [
        str(nvcc),
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--generate-code=arch=compute_86,code=sm_86",
        "--compiler-options=-fPIC",
        "--shared",
        "--cudart=none",
        "-I{}".format(HERE),
        "-I{}".format(cutlass_root / "include"),
        "-I{}".format(cutlass_root / "tools/util/include"),
        "-I{}".format(runtime_root / "include"),
        str(HERE / "kernels.cu"),
        "-L{}".format(runtime_lib),
        "-Xlinker",
        "-l:libcudart.so.12",
        "-o",
        str(output),
    ]
    if args.verbose:
        print(" ".join(command))
    subprocess.run(command, check=True)
    ProviderBuildMetadata(
        provider="cutlass_w8a16",
        provider_version="2",
        abi=2,
        compiled_architectures=("sm86",),
        weight_formats=(
            "int8_symmetric_per_channel",
            "int8_symmetric_per_group",
        ),
        activation_dtypes=("bf16", "fp16"),
        build_environment={
            "cuda": "unverified",
            "compiler": "unverified",
            "cutlass": "unverified",
        },
        status="compiled",
    ).write(Path(str(output) + ".metadata.json"))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
