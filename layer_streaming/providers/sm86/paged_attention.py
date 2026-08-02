from ..generic_cuda.paged_attention import (
    GenericCUDAPagedAttentionBackend,
    GenericCUDAPagedKVKernelBackend,
)


class SM86PagedAttentionBackend(GenericCUDAPagedAttentionBackend):
    name = "sm86"
    architectures = ("sm86",)
    qualification_status = "smoke_passed"
    supported_head_dims = (128,)
    num_warps = 4
    num_stages = 3

    @staticmethod
    def _threads(head_dim):
        if int(head_dim) != 128:
            return GenericCUDAPagedAttentionBackend._threads(head_dim)
        # SM86 Llama GQA specialization: one lane per head dimension avoids
        # the extra generic reduction warp.
        return 128


class SM86PagedKVKernelBackend(GenericCUDAPagedKVKernelBackend):
    name = "sm86_kv_kernel"
    architectures = ("sm86",)
    qualification_status = "smoke_passed"
    supported_head_dims = (128,)


SM86PagedAttentionProvider = SM86PagedAttentionBackend
