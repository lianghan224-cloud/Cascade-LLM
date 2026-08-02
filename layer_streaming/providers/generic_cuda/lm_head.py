"""Deterministic BF16/FP16 LM Head used by resident and streamed paths."""

import ctypes
import threading

import torch

from ...kv.errors import KVProviderError
from .lm_head_source import LM_HEAD_CUDA_SOURCE
from .nvrtc import CUDAKernelModule


class DeterministicCUDALMHead:
    """Compute row-wise logits with one fixed FP32 reduction tree.

    A vocabulary row has the same result regardless of the number of other
    rows in its transfer chunk.  This removes the cuBLAS algorithm-selection
    difference between resident and streamed vocabulary layouts.
    """

    name = "deterministic_cuda_fp32_accum_native_output"
    _modules = {}
    _module_lock = threading.RLock()

    @staticmethod
    def _architecture(device):
        major, minor = torch.cuda.get_device_capability(device)
        return "sm{}{}".format(major, minor)

    @classmethod
    def _module(cls, device):
        architecture = cls._architecture(device)
        key = (architecture, 1)
        with cls._module_lock:
            module = cls._modules.get(key)
            if module is None:
                module = CUDAKernelModule(
                    LM_HEAD_CUDA_SOURCE,
                    architecture,
                    (
                        "deterministic_lm_head_bf16",
                        "deterministic_lm_head_fp16",
                    ),
                )
                cls._modules[key] = module
            return module

    def execute(self, hidden_states, weight):
        if not hidden_states.is_cuda or not weight.is_cuda:
            raise KVProviderError("deterministic LM Head requires CUDA tensors")
        if hidden_states.dtype != weight.dtype:
            hidden_states = hidden_states.to(weight.dtype)
        if weight.dtype not in {torch.bfloat16, torch.float16}:
            raise KVProviderError(
                "deterministic LM Head supports BF16 and FP16 weights"
            )
        if weight.ndim != 2 or hidden_states.shape[-1] != weight.shape[-1]:
            raise ValueError("LM Head hidden and weight dimensions do not match")
        hidden = hidden_states.contiguous()
        matrix = weight.contiguous()
        tokens = hidden.numel() // hidden.shape[-1]
        rows, hidden_size = matrix.shape
        if tokens > 65535:
            raise KVProviderError(
                "deterministic LM Head V1 supports at most 65535 tokens per launch"
            )
        output = torch.empty(
            (tokens, rows), dtype=matrix.dtype, device=hidden.device
        )
        kernel = (
            "deterministic_lm_head_bf16"
            if matrix.dtype == torch.bfloat16
            else "deterministic_lm_head_fp16"
        )
        self._module(hidden.device).launch(
            kernel,
            (int(rows), int(tokens), 1),
            (256, 1, 1),
            torch.cuda.current_stream(hidden.device).cuda_stream,
            (
                (ctypes.c_void_p, hidden.data_ptr()),
                (ctypes.c_void_p, matrix.data_ptr()),
                (ctypes.c_void_p, output.data_ptr()),
                (ctypes.c_int, int(tokens)),
                (ctypes.c_int, int(rows)),
                (ctypes.c_int, int(hidden_size)),
            ),
        )
        return output.view(hidden_states.shape[:-1] + (rows,))


_DEFAULT_LM_HEAD = DeterministicCUDALMHead()


def deterministic_lm_head(hidden_states, weight):
    return _DEFAULT_LM_HEAD.execute(hidden_states, weight)
