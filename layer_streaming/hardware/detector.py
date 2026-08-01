"""One-shot NVIDIA hardware and runtime environment detection."""

from functools import lru_cache
from pathlib import Path
import platform

import torch

from .profile import HardwareProfile, RuntimeFeatureProfile


def architecture_for_compute_capability(major, minor):
    value = "sm{}{}".format(int(major), int(minor))
    if value in {"sm75", "sm80", "sm86", "sm89", "sm90"}:
        return value
    return "unknown"


def _driver_version():
    getter = getattr(getattr(torch, "_C", None), "_cuda_getDriverVersion", None)
    if not callable(getter):
        return None
    try:
        raw = int(getter())
    except BaseException:
        return None
    if raw <= 0:
        return None
    return str(raw)


def _pinned_memory_available(cuda_available):
    if not cuda_available:
        return False
    try:
        torch.empty(1, dtype=torch.uint8, pin_memory=True)
        return True
    except BaseException:
        return False


def _registered_provider_runtime():
    versions = set()
    architectures = set()
    loaded = False
    try:
        from ..backends import default_backend_registry

        for backend in default_backend_registry().values():
            provider_name = str(getattr(backend, "provider_name", ""))
            if not provider_name:
                continue
            loaded = loaded or "cutlass" in provider_name
            version = getattr(backend, "provider_version", None)
            if version is not None:
                versions.add(int(version))
            info = getattr(backend, "info", None)
            architectures.update(
                str(item)
                for item in getattr(
                    info, "supported_gpu_architectures", ()
                )
            )
    except BaseException:
        pass
    try:
        from .build_metadata import ProviderBuildMetadata

        provider_root = Path(__file__).resolve().parents[1] / "providers"
        for metadata_path in provider_root.glob(
            "**/*.so.metadata.json"
        ):
            binary_path = Path(
                str(metadata_path)[: -len(".metadata.json")]
            )
            if not binary_path.is_file():
                continue
            metadata = ProviderBuildMetadata.read(metadata_path)
            if metadata.status != "compiled":
                continue
            versions.add(int(metadata.abi))
            architectures.update(metadata.compiled_architectures)
    except BaseException:
        # Invalid/missing metadata never becomes a positive capability signal.
        pass
    return loaded, tuple(sorted(versions)), tuple(sorted(architectures))


@lru_cache(maxsize=16)
def _detect(device_index):
    cuda_available = bool(torch.cuda.is_available())
    if not cuda_available:
        hardware = HardwareProfile(
            vendor="unknown",
            device_name=platform.machine() or "unknown",
            device_index=int(device_index),
            compute_capability_major=0,
            compute_capability_minor=0,
            architecture="unknown",
            total_memory_bytes=0,
            free_memory_bytes=0,
            driver_version=None,
            cuda_runtime_version=None,
            torch_version=str(torch.__version__),
            torch_cuda_version=(
                str(torch.version.cuda) if torch.version.cuda else None
            ),
            supports_fp16=False,
            supports_bf16=False,
            supports_int8_tensor_core=False,
            supports_int4_tensor_core=False,
            unified_addressing=False,
            async_copy_supported=False,
        )
    else:
        index = int(device_index)
        properties = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except (AttributeError, RuntimeError):
            total_bytes = int(properties.total_memory)
            free_bytes = 0
        hardware = HardwareProfile(
            vendor="nvidia",
            device_name=str(properties.name),
            device_index=index,
            compute_capability_major=int(major),
            compute_capability_minor=int(minor),
            architecture=architecture_for_compute_capability(major, minor),
            total_memory_bytes=int(total_bytes),
            free_memory_bytes=int(free_bytes),
            driver_version=_driver_version(),
            cuda_runtime_version=(
                str(torch.version.cuda) if torch.version.cuda else None
            ),
            torch_version=str(torch.__version__),
            torch_cuda_version=(
                str(torch.version.cuda) if torch.version.cuda else None
            ),
            supports_fp16=(major >= 7),
            supports_bf16=(major >= 8),
            supports_int8_tensor_core=(major >= 7),
            supports_int4_tensor_core=(major >= 7),
            unified_addressing=bool(
                getattr(properties, "unified_addressing", True)
            ),
            async_copy_supported=True,
        )
    extension_loaded, abi_versions, compiled = _registered_provider_runtime()
    runtime = RuntimeFeatureProfile(
        cuda_available=cuda_available,
        cuda_graph_available=bool(
            cuda_available and hasattr(torch.cuda, "CUDAGraph")
        ),
        pinned_memory_available=_pinned_memory_available(cuda_available),
        cutlass_extension_loaded=extension_loaded,
        provider_abi_versions=abi_versions,
        compiled_architectures=compiled,
        deterministic_mode=bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    )
    return hardware, runtime


class HardwareDetector:
    """Detect the selected device once per process/device index."""

    def detect(self, device=0, refresh=False):
        if isinstance(device, torch.device):
            index = device.index if device.index is not None else 0
        elif isinstance(device, str) and device.startswith("cuda"):
            parsed = torch.device(device)
            index = parsed.index if parsed.index is not None else 0
        else:
            index = int(device)
        if refresh:
            _detect.cache_clear()
        return _detect(index)


def fake_hardware_profile(architecture, total_memory_bytes=0):
    """Create an explicitly synthetic profile for pure compatibility tests."""

    architecture = str(architecture).lower()
    if architecture == "unknown":
        major, minor = 0, 0
    elif architecture.startswith("sm") and len(architecture) == 4:
        major, minor = int(architecture[2]), int(architecture[3])
    else:
        raise ValueError("invalid synthetic architecture {}".format(architecture))
    return HardwareProfile(
        vendor="nvidia" if architecture != "unknown" else "unknown",
        device_name="synthetic_{}".format(architecture),
        device_index=0,
        compute_capability_major=major,
        compute_capability_minor=minor,
        architecture=architecture,
        total_memory_bytes=int(total_memory_bytes),
        free_memory_bytes=int(total_memory_bytes),
        driver_version=None,
        cuda_runtime_version=None,
        torch_version=str(torch.__version__),
        torch_cuda_version=None,
        supports_fp16=major >= 7,
        supports_bf16=major >= 8,
        supports_int8_tensor_core=major >= 7,
        supports_int4_tensor_core=major >= 7,
        unified_addressing=major > 0,
        async_copy_supported=major > 0,
    )
