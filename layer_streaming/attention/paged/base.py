"""Paged Attention compute backend protocol.

This ABI deliberately excludes page append/copy and every lifecycle operation.
"""

from ...kv.errors import KVUnsupportedError


class PagedAttentionBackend:
    name = "abstract"
    is_reference = False

    def capability(self):
        raise NotImplementedError

    def estimate_workspace(self, request):
        raise NotImplementedError

    def decode(self, request):
        raise KVUnsupportedError("{} does not implement decode".format(self.name))

    def prefill(self, request):
        raise KVUnsupportedError("{} does not implement prefill".format(self.name))


# Import compatibility only.  The frozen V1 contract names the interface
# PagedAttentionBackend and contains no append/copy/lifecycle methods.
PagedAttentionProvider = PagedAttentionBackend
