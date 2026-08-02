"""Paged Attention Provider protocol."""

from ...kv.errors import KVUnsupportedError


class PagedAttentionProvider:
    name = "abstract"
    is_reference = False

    def capability(self):
        raise NotImplementedError

    def estimate_workspace(self, request):
        raise NotImplementedError

    def append_kv(self, append_input):
        raise NotImplementedError

    def copy_pages(self, store, source_page_ids, target_page_ids, valid_tokens):
        if not (
            len(source_page_ids) == len(target_page_ids) == len(valid_tokens)
        ):
            raise ValueError("copy page lists must have equal length")
        for source, target, valid in zip(
            source_page_ids, target_page_ids, valid_tokens
        ):
            store.copy_page(source, target, valid)

    def decode(self, request):
        raise KVUnsupportedError("{} does not implement decode".format(self.name))

    def prefill(self, request):
        raise KVUnsupportedError("{} does not implement prefill".format(self.name))
