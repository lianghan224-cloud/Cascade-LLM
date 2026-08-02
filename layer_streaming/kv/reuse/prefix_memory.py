"""Exact in-memory sealed-prefix reuse provider."""

from .session import SessionReuse
from .base import ReuseCapability


class PrefixMemoryReuse(SessionReuse):
    name = "prefix_memory"

    def capability(self):
        return ReuseCapability(self.name, True, True, False)

    def register_prefix(self, runtime, state, token_ids):
        return runtime._register_prefix_impl(state, token_ids)

    def lookup_prefix(
        self,
        runtime,
        token_ids,
        max_length,
        reuse_namespace="default",
        request_id=None,
    ):
        return runtime._reuse_prefix_impl(
            token_ids,
            max_length,
            reuse_namespace=reuse_namespace,
            request_id=request_id,
        )
