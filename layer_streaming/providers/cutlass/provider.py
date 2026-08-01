"""ctypes wrapper for the CUTLASS SM86 W8A16 shared library."""

import ctypes
from pathlib import Path

import torch

from ...backend_capability import BackendCapability
from ...backends import (
    BackendInfo,
    register_linear_backend,
)
from ...specs import normalize_dtype


_DTYPE_CODE = {
    "bfloat16": 0,
    "float16": 1,
}


def _default_library():
    return Path(__file__).with_name("_build") / "libcascade_cutlass_sm86.so"


class CutlassW8A16Provider:
    """SM86 W8A16 provider with per-channel and groupwise decode paths."""

    name = "fused_w8a16"
    provider_name = "cutlass_sm86_w8a16"
    is_fallback = False
    info = BackendInfo(
        name=name,
        storage_dtype="int8",
        activation_dtype="bfloat16",
        output_dtype="bfloat16",
        requires_dequant=False,
        is_fallback=False,
        supported_gpu_architectures=("sm86",),
        atol=0.08,
        rtol=0.04,
    )
    capability = BackendCapability(
        min_m=1,
        max_m=None,
        supported_sms=(86,),
        activation_dtypes=("bf16", "fp16"),
        weight_formats=(
            "int8_symmetric_per_channel",
            "int8_symmetric_per_group",
        ),
        group_sizes=(32, 64, 128),
        alignment_k=16,
        alignment_n=8,
    )

    def __init__(self, library=None):
        self.library_path = Path(library or _default_library()).resolve()
        if not self.library_path.is_file():
            raise FileNotFoundError(
                "CUTLASS provider library is missing: {}; run "
                "python -m layer_streaming.providers.cutlass.build".format(
                    self.library_path
                )
            )
        self._library = ctypes.CDLL(str(self.library_path))
        self._run = self._library.cascade_cutlass_w8a16_run
        self._run.argtypes = [
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_void_p,
        ]
        self._run.restype = ctypes.c_int32
        self._last_error = self._library.cascade_cutlass_last_error
        self._last_error.argtypes = []
        self._last_error.restype = ctypes.c_char_p
        version = self._library.cascade_cutlass_provider_version
        version.argtypes = []
        version.restype = ctypes.c_int32
        self.provider_version = int(version())
        if self.provider_version != 2:
            raise RuntimeError(
                "unsupported CUTLASS provider ABI {}".format(
                    self.provider_version
                )
            )

    def validate(self, weight, input_dtype):
        if weight.quantization is None or weight.quantization.bits != 8:
            raise ValueError("{} requires INT8 weights".format(self.name))
        quant = weight.quantization
        if quant.scheme != "symmetric" or quant.granularity not in {
            "per_channel",
            "per_group",
        }:
            raise ValueError(
                "{} requires symmetric per-channel/per-group weights".format(
                    self.provider_name
                )
            )
        if (
            quant.granularity == "per_group"
            and quant.group_size not in {32, 64, 128}
        ):
            raise ValueError("{} requires group size 32/64/128".format(self.name))
        if quant.axis != 1:
            raise ValueError("{} requires quantization axis=1".format(self.name))
        if quant.scale_dtype not in {"bfloat16", "float16"}:
            raise ValueError("{} requires BF16/FP16 scales".format(self.name))
        input_dtype = normalize_dtype(input_dtype)
        if input_dtype not in {"bfloat16", "float16"}:
            raise ValueError("{} requires BF16/FP16 activation".format(self.name))
        if normalize_dtype(weight.compute_dtype) != input_dtype:
            raise ValueError("weight and activation compute dtype differ")
        reason = self.capability.unsupported_reason(weight, 1, sm=86)
        if reason is not None:
            raise ValueError(reason)

    def unsupported_reason(self, weight, m, sm=None):
        reason = self.capability.unsupported_reason(weight, m, sm=sm)
        if reason is not None:
            return reason
        if (
            weight.quantization.granularity == "per_group"
            and int(m) != 1
        ):
            return (
                "per-group W8A16 is decode-only (M=1); select an explicit "
                "prefill backend"
            )
        return None

    def transfer_bytes(self, weight):
        self.validate(weight, weight.compute_dtype)
        return int(weight.storage_nbytes)

    def workspace_bytes(self, weight, batch_tokens):
        del batch_tokens
        self.validate(weight, weight.compute_dtype)
        return 0

    def execute(self, x, weight_view, quant_views, workspace):
        del workspace
        if not x.is_cuda or not weight_view.is_cuda:
            raise ValueError("{} requires CUDA tensors".format(self.name))
        major, minor = torch.cuda.get_device_capability(x.device)
        if (major, minor) != (8, 6):
            raise ValueError(
                "{} requires SM86, got SM{}{}".format(
                    self.provider_name, major, minor
                )
            )
        if not x.is_contiguous() or not weight_view.is_contiguous():
            raise ValueError("{} requires contiguous tensors".format(self.name))
        weight = quant_views.get("weight_spec") if quant_views else None
        scale = quant_views.get("scale") if quant_views else None
        if weight is None or scale is None:
            raise ValueError(
                "{} requires weight_spec and scale views".format(self.name)
            )
        self.validate(weight, weight.compute_dtype)
        if not scale.is_cuda or not scale.is_contiguous():
            raise ValueError("scale must be a contiguous CUDA tensor")
        if x.dtype not in {torch.bfloat16, torch.float16}:
            raise ValueError("activation tensor must be BF16 or FP16")
        expected_weight_dtype = torch.int8
        if weight_view.dtype != expected_weight_dtype:
            raise ValueError("weight tensor must be INT8")
        n, k = (int(item) for item in weight.logical_shape)
        if int(x.shape[-1]) != k:
            raise ValueError(
                "activation K={} does not match weight K={}".format(
                    x.shape[-1], k
                )
            )
        if tuple(weight_view.shape) != tuple(weight.storage_shape):
            raise ValueError("weight storage shape mismatch")
        quant = weight.quantization
        expected_scale_shape = (
            (n, 1)
            if quant.granularity == "per_channel"
            else (n, k // int(quant.group_size))
        )
        if tuple(scale.shape) != expected_scale_shape:
            raise ValueError(
                "scale shape {} != expected {}".format(
                    tuple(scale.shape), expected_scale_shape
                )
            )
        activation_dtype = (
            "bfloat16" if x.dtype == torch.bfloat16 else "float16"
        )
        scale_dtype = (
            "bfloat16" if scale.dtype == torch.bfloat16 else "float16"
        )
        if scale.dtype not in {torch.bfloat16, torch.float16}:
            raise ValueError("scale tensor must be BF16 or FP16")
        m = x.numel() // k
        output = torch.empty(
            tuple(x.shape[:-1]) + (n,),
            dtype=x.dtype,
            device=x.device,
        )
        stream = torch.cuda.current_stream(x.device).cuda_stream
        status = self._run(
            _DTYPE_CODE[activation_dtype],
            _DTYPE_CODE[scale_dtype],
            ctypes.c_void_p(x.data_ptr()),
            ctypes.c_void_p(weight_view.data_ptr()),
            ctypes.c_void_p(scale.data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            int(m),
            n,
            k,
            (
                0
                if quant.granularity == "per_channel"
                else int(quant.group_size)
            ),
            ctypes.c_void_p(stream),
        )
        if status:
            message = self._last_error()
            raise RuntimeError(
                "{} failed with status {}: {}".format(
                    self.provider_name,
                    status,
                    (
                        message.decode("utf-8", errors="replace")
                        if message
                        else "unknown error"
                    ),
                )
            )
        return output


def load_cutlass_w8a16_provider(library=None, replace=False):
    """Load and explicitly register the CUTLASS provider."""

    return register_linear_backend(
        CutlassW8A16Provider(library=library), replace=replace
    )
