from ..generic_cuda.paged_attention import GenericCUDAPagedAttentionProvider


class SM90PagedAttentionProvider(GenericCUDAPagedAttentionProvider):
    name = "sm90"
    architectures = ("sm90",)
    qualification_status = "unqualified"
    supported_head_dims = (64, 80, 96, 128, 256)
    num_warps = 8
    num_stages = 4
