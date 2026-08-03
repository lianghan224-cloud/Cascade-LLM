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

    def decode(self, request):
        raise KVUnsupportedError("{} does not implement decode".format(self.name))

    def prefill(self, request):
        raise KVUnsupportedError("{} does not implement prefill".format(self.name))
