from ..generic_cuda.paged_attention import (
    GenericCUDAPagedAttentionBackend,
    GenericCUDAPagedKVKernelBackend,
)


class SM89PagedAttentionBackend(GenericCUDAPagedAttentionBackend):
    name = "sm89"
    architectures = ("sm89",)
    qualification_status = "unqualified"
    supported_head_dims = (64, 80, 96, 128, 256)
    num_warps = 4
    num_stages = 3


class SM89PagedKVKernelBackend(GenericCUDAPagedKVKernelBackend):
    name = "sm89_kv_kernel"
    architectures = ("sm89",)
    qualification_status = "unqualified"
    supported_head_dims = (64, 80, 96, 128, 256)


SM89PagedAttentionProvider = SM89PagedAttentionBackend
