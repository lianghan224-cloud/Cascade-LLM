"""Generic NVRTC CUDA provider for KV Framework V1."""

import ctypes
import threading
import time

import torch

from ...attention.paged import (
    PAGED_ATTENTION_ABI_VERSION,
    PagedAttentionCapability,
    PagedAttentionBackend,
    PagedAttentionOutput,
    PagedWorkspaceEstimate,
)
from ...kv.errors import KVProviderError
from ...kv.kernel_backend import (
    PAGED_KV_KERNEL_ABI_VERSION,
    PagedKVKernelBackend,
    PagedKVKernelCapability,
)
from .cuda_source import PAGED_ATTENTION_CUDA_SOURCE
from .nvrtc import CUDAKernelModule


class _GenericCUDAKernelSupport:
    """Shared NVRTC module/launch support; it owns no page lifecycle."""

    name = "generic_cuda"
    architectures = ("sm80", "sm86", "sm89", "sm90")
    qualification_status = "compiled"
    supported_head_dims = ()
    _modules = {}
    _module_lock = threading.RLock()

    @staticmethod
    def _architecture(device):
        major, minor = torch.cuda.get_device_capability(device)
        return "sm{}{}".format(major, minor)

    @staticmethod
    def _threads(head_dim):
        head_dim = int(head_dim)
        if head_dim <= 0 or head_dim > 256:
            raise KVProviderError("generic CUDA requires 1 <= head_dim <= 256")
        # The architecture-neutral provider uses a conservative full-warp
        # configuration. Architecture providers may override this launch
        # policy without changing the ABI or CUDA source.
        threads = 1
        while threads < head_dim:
            threads *= 2
        return max(32, min(256, threads * (2 if head_dim >= 128 else 1)))

    def _module(self, device):
        architecture = self._architecture(device)
        if architecture not in self.architectures:
            raise KVProviderError(
                "{} does not support {}".format(self.name, architecture)
            )
        key = (architecture, PAGED_ATTENTION_ABI_VERSION)
        with self._module_lock:
            module = self._modules.get(key)
            if module is None:
                module = CUDAKernelModule(
                    PAGED_ATTENTION_CUDA_SOURCE,
                    architecture,
                    (
                        "append_kv_u16",
                        "paged_attention_bf16",
                        "paged_attention_fp16",
                    ),
                )
                self._modules[key] = module
            return module


class GenericCUDAPagedKVKernelBackend(
    _GenericCUDAKernelSupport,
    PagedKVKernelBackend,
):
    """Append/copy kernel backend; allocation and references stay in runtime."""

    name = "generic_cuda_kv_kernel"

    def capability(self):
        return PagedKVKernelCapability(
            backend_name=self.name,
            backend_version="1",
            backend_abi=PAGED_KV_KERNEL_ABI_VERSION,
            architectures=self.architectures,
            dtypes=("bf16", "fp16"),
            page_sizes=(16, 32),
            head_dims=self.supported_head_dims,
            supports_append=True,
            supports_copy=True,
            qualification_status=self.qualification_status,
        )

    def append_kv(self, append_input):
        append_input.validate()
        if not append_input.key.is_cuda:
            raise KVProviderError("generic CUDA append requires CUDA tensors")
        threads = self._threads(append_input.head_dim)
        key = append_input.key.contiguous()
        value = append_input.value.contiguous()
        stream = torch.cuda.current_stream(key.device).cuda_stream
        module = self._module(key.device)
        module.launch(
            "append_kv_u16",
            (
                int(key.shape[0]),
                int(append_input.num_kv_heads),
                1,
            ),
            (threads, 1, 1),
            stream,
            (
                (ctypes.c_void_p, key.data_ptr()),
                (ctypes.c_void_p, value.data_ptr()),
                (ctypes.c_void_p, append_input.key_pool_view.data_ptr()),
                (ctypes.c_void_p, append_input.value_pool_view.data_ptr()),
                (ctypes.c_void_p, append_input.slot_mapping.page_ids.data_ptr()),
                (ctypes.c_void_p, append_input.slot_mapping.offsets.data_ptr()),
                (ctypes.c_int, int(key.shape[0])),
                (ctypes.c_int, int(append_input.num_kv_heads)),
                (ctypes.c_int, int(append_input.page_size)),
                (ctypes.c_int, int(append_input.head_dim)),
            ),
        )

    def copy_pages(self, store, source_page_ids, target_page_ids, valid_tokens):
        if not (
            len(source_page_ids) == len(target_page_ids) == len(valid_tokens)
        ):
            raise ValueError("copy page lists must have equal length")
        for source, target, valid in zip(
            source_page_ids, target_page_ids, valid_tokens
        ):
            store.copy_page(source, target, valid)


class GenericCUDAPagedAttentionBackend(
    _GenericCUDAKernelSupport,
    PagedAttentionBackend,
):
    """Direct HND-page CUDA attention backend compiled by NVRTC on first use.

    NVRTC compiles device PTX and the CUDA driver launches it directly. No
    Python extension, host compiler, Python headers, nvcc, or full CUDA toolkit
    is required in the runtime image.
    """

    name = "generic_cuda"
    is_reference = False

    def capability(self):
        return PagedAttentionCapability(
            provider_name=self.name,
            provider_version="1",
            provider_abi=PAGED_ATTENTION_ABI_VERSION,
            architectures=self.architectures,
            dtypes=("bf16", "fp16"),
            page_sizes=(16, 32),
            head_dims=self.supported_head_dims,
            supports_mha=True,
            supports_gqa=True,
            supports_mqa=True,
            supports_decode=True,
            supports_prefill=True,
            supports_ragged_batch=True,
            supports_partial_tail=True,
            supports_cuda_graph=False,
            numerical_contract_version=1,
            requires_full_kv_workspace=False,
            requires_full_score_matrix=False,
            qualification_status=self.qualification_status,
        )

    def estimate_workspace(self, request):
        del request
        return PagedWorkspaceEstimate(
            0,
            "register_and_shared_only",
            False,
            False,
            False,
        )

    def _execute(self, request, phase):
        request.validate()
        if not request.query.is_cuda:
            raise KVProviderError("generic CUDA paged attention requires CUDA tensors")
        if request.query.dtype != request.key_pool_view.dtype:
            raise KVProviderError("query and KV pool dtype must match")
        threads = self._threads(request.head_dim)
        query = request.query.contiguous()
        output_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }.get(request.output_dtype)
        if output_dtype != query.dtype:
            raise KVProviderError(
                "generic CUDA V1 requires output dtype to match query dtype"
            )
        output = torch.empty_like(query)
        logsumexp = torch.empty(
            (query.shape[0], query.shape[1]),
            dtype=torch.float32,
            device=query.device,
        )
        batch = request.batch_view
        stream = torch.cuda.current_stream(query.device).cuda_stream
        module = self._module(query.device)
        kernel_name = (
            "paged_attention_bf16"
            if query.dtype == torch.bfloat16
            else "paged_attention_fp16"
        )
        started = time.perf_counter()
        module.launch(
            kernel_name,
            (
                batch.batch_size,
                batch.max_query_length,
                int(request.num_query_heads),
            ),
            (threads, 1, 1),
            stream,
            (
                (ctypes.c_void_p, query.data_ptr()),
                (ctypes.c_void_p, request.key_pool_view.data_ptr()),
                (ctypes.c_void_p, request.value_pool_view.data_ptr()),
                (ctypes.c_void_p, request.flat_block_table.data_ptr()),
                (ctypes.c_void_p, request.logical_block_ids.data_ptr()),
                (ctypes.c_void_p, request.page_valid_tokens.data_ptr()),
                (ctypes.c_void_p, request.block_table_indptr.data_ptr()),
                (ctypes.c_void_p, batch.sequence_lengths.data_ptr()),
                (ctypes.c_void_p, batch.query_indptr.data_ptr()),
                (ctypes.c_void_p, batch.query_positions.data_ptr()),
                (ctypes.c_void_p, output.data_ptr()),
                (ctypes.c_void_p, logsumexp.data_ptr()),
                (ctypes.c_int, batch.batch_size),
                (ctypes.c_int, int(request.num_query_heads)),
                (ctypes.c_int, int(request.num_kv_heads)),
                (ctypes.c_int, int(request.page_size)),
                (ctypes.c_int, int(request.head_dim)),
                (ctypes.c_float, float(request.softmax_scale)),
                (ctypes.c_int, int(bool(request.causal))),
                (ctypes.c_int, int(bool(request.return_logsumexp))),
            ),
        )
        return PagedAttentionOutput(
            output=output,
            logsumexp=(logsumexp if request.return_logsumexp else None),
            provider_metrics={
                "provider": self.name,
                "phase": phase,
                "launch_wall_ms": (time.perf_counter() - started) * 1000.0,
                "workspace_bytes": 0,
                "full_kv_workspace": False,
                "full_score_matrix": False,
                "threads": threads,
                "compiler": "nvrtc",
            },
        )

    def decode(self, request):
        if request.batch_view.max_query_length != 1:
            raise ValueError("decode requires query length one for every request")
        return self._execute(request, "decode")

    def prefill(self, request):
        return self._execute(request, "prefill")


# Compatibility alias.  It is attention-only and no longer exposes append/copy.
GenericCUDAPagedAttentionProvider = GenericCUDAPagedAttentionBackend
