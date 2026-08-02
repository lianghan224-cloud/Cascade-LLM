from ..generic_cuda.paged_attention import GenericCUDAPagedAttentionProvider


class SM86PagedAttentionProvider(GenericCUDAPagedAttentionProvider):
    name = "sm86"
    architectures = ("sm86",)
    qualification_status = "smoke_passed"
    supported_head_dims = (128,)
    num_warps = 4
    num_stages = 3

    @staticmethod
    def _threads(head_dim):
        if int(head_dim) != 128:
            return GenericCUDAPagedAttentionProvider._threads(head_dim)
        # SM86 Llama GQA specialization: one lane per head dimension avoids
        # the extra generic reduction warp.
        return 128
