from ..generic_cuda.paged_attention import GenericCUDAPagedAttentionProvider


class SM89PagedAttentionProvider(GenericCUDAPagedAttentionProvider):
    name = "sm89"
    architectures = ("sm89",)
    qualification_status = "unqualified"
    supported_head_dims = (64, 80, 96, 128, 256)
    num_warps = 4
    num_stages = 3
