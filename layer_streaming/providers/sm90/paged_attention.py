from ..generic_cuda.paged_attention import (
    GenericCUDAPagedAttentionBackend,
    GenericCUDAPagedKVKernelBackend,
)


class SM90PagedAttentionBackend(GenericCUDAPagedAttentionBackend):
    name = "sm90"
    architectures = ("sm90",)
    qualification_status = "declared"
    supported_head_dims = (64, 80, 96, 128, 256)
    num_warps = 8
    num_stages = 4


class SM90PagedKVKernelBackend(GenericCUDAPagedKVKernelBackend):
    name = "sm90_kv_kernel"
    architectures = ("sm90",)
    qualification_status = "declared"
    supported_head_dims = (64, 80, 96, 128, 256)
