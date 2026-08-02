"""Paged attention capability and qualification declarations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PagedAttentionCapability:
    provider_name: str
    provider_version: str
    provider_abi: int
    architectures: tuple
    dtypes: tuple
    page_sizes: tuple
    head_dims: tuple
    supports_mha: bool
    supports_gqa: bool
    supports_mqa: bool
    supports_decode: bool
    supports_prefill: bool
    supports_ragged_batch: bool
    supports_partial_tail: bool
    supports_cuda_graph: bool
    numerical_contract_version: int
    requires_full_kv_workspace: bool
    requires_full_score_matrix: bool
    qualification_status: str

    def unsupported_reason(self, request, architecture=None, phase=None):
        if architecture is not None and architecture not in self.architectures:
            return "architecture {} is not supported".format(architecture)
        if request.kv_dtype not in self.dtypes:
            return "KV dtype {} is not supported".format(request.kv_dtype)
        if int(request.page_size) not in self.page_sizes:
            return "page size {} is not supported".format(request.page_size)
        if self.head_dims and int(request.head_dim) not in self.head_dims:
            return "head dim {} is not supported".format(request.head_dim)
        if request.num_kv_heads == 1 and request.num_query_heads > 1:
            if not self.supports_mqa:
                return "MQA is not supported"
        elif request.num_query_heads == request.num_kv_heads:
            if not self.supports_mha:
                return "MHA is not supported"
        elif not self.supports_gqa:
            return "GQA is not supported"
        if request.batch_view.batch_size > 1 and not self.supports_ragged_batch:
            return "ragged batch is not supported"
        phase = phase or (
            "decode"
            if request.batch_view.max_query_length == 1
            else "prefill"
        )
        if phase == "decode" and not self.supports_decode:
            return "decode is not supported"
        if phase == "prefill" and not self.supports_prefill:
            return "prefill is not supported"
        return None

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}
