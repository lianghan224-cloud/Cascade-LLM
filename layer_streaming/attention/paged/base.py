"""Paged Attention compute backend protocol.

This ABI deliberately excludes page append/copy and every lifecycle operation.
"""

from ...kv.errors import KVUnsupportedError


class PagedAttentionBackend:
    name = "abstract"
    is_reference = False
    schema_version = 1
    workload_kinds = (
        "full_prefill",
        "chunked_prefill",
        "decode",
        "short_suffix",
    )
    supports_device_selected_view = False

    @property
    def provider_abi(self):
        return int(self.capability().provider_abi)

    @property
    def qualification_status(self):
        return self.capability().qualification_status

    def capability(self):
        raise NotImplementedError

    def estimate_workspace(self, request):
        raise NotImplementedError

    def validate_shape(self, shape):
        shape.validate()
        capability = self.capability()
        if not capability.supports_prefill:
            raise KVUnsupportedError(
                "{} does not support prefill".format(self.name)
            )
        if capability.page_sizes and shape.page_size not in capability.page_sizes:
            raise KVUnsupportedError(
                "{} does not support page size {}".format(
                    self.name, shape.page_size
                )
            )
        if capability.dtypes and shape.dtype not in capability.dtypes:
            raise KVUnsupportedError(
                "{} does not support dtype {}".format(
                    self.name, shape.dtype
                )
            )
        if capability.head_dims and shape.head_dim not in capability.head_dims:
            raise KVUnsupportedError(
                "{} does not support head dim {}".format(
                    self.name, shape.head_dim
                )
            )
        return shape

    def estimate_workspace_shape(self, shape):
        """Preflight estimate without allocating request tensors."""

        raise NotImplementedError

    def decode(self, request):
        raise KVUnsupportedError("{} does not implement decode".format(self.name))

    def decode_device(self, request):
        del request
        raise KVUnsupportedError("UNSUPPORTED_DEVICE_SELECTED_VIEW")

    def prefill(self, request):
        raise KVUnsupportedError("{} does not implement prefill".format(self.name))
