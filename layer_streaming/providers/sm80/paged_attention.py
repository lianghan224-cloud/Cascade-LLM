from ..generic_cuda.paged_attention import (
    GenericCUDAPagedAttentionBackend,
    GenericCUDAPagedKVKernelBackend,
)


class SM80PagedAttentionBackend(GenericCUDAPagedAttentionBackend):
    name = "sm80"
    architectures = ("sm80",)
    qualification_status = "declared"
    supported_head_dims = (64, 80, 96, 128, 256)
    num_warps = 4
    num_stages = 2


class SM80PagedKVKernelBackend(GenericCUDAPagedKVKernelBackend):
    name = "sm80_kv_kernel"
    architectures = ("sm80",)
    qualification_status = "declared"
    supported_head_dims = (64, 80, 96, 128, 256)
