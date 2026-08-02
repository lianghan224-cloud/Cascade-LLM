"""In-memory sealed-prefix ownership separated from KVRuntime coordination."""

from .errors import KVLifecycleError
from .reuse import InMemoryPrefixIndex


class PrefixCache:
    def __init__(self, page_size, page_pool, layer_count, metrics):
        self.page_size = int(page_size)
        self.page_pool = page_pool
        self.layer_count = int(layer_count)
        self.metrics = metrics
        self.index = InMemoryPrefixIndex(self.page_size)
        self.owned_handles = {}

    def register(self, state, token_ids):
        if state.pending_append is not None:
            raise KVLifecycleError("cannot register an uncommitted prefix")
        full_pages = state.sequence_length // self.page_size
        handles = tuple(state.block_table.handles[:full_pages])
        hashes = self.index.register(
            state.reuse_namespace,
            token_ids,
            handles,
        )
        for handle in handles:
            identity = handle.identity()
            if identity not in self.owned_handles:
                self.page_pool.retain(handle)
                self.owned_handles[identity] = handle
        state.token_block_hashes[:] = list(hashes)
        return hashes

    def reuse(
        self,
        runtime,
        token_ids,
        max_length,
        reuse_namespace="default",
        request_id=None,
    ):
        match = self.index.lookup(reuse_namespace, token_ids)
        if not match.page_handles:
            self.metrics.prefix_misses += 1
            return runtime.create_request(
                max_length,
                request_id=request_id,
                reuse_namespace=reuse_namespace,
            ), match
        state = runtime.create_request(
            max_length,
            request_id=request_id,
            reuse_namespace=reuse_namespace,
        )
        try:
            for handle in match.page_handles:
                self.page_pool.retain(handle)
                state.block_table.append(handle)
            state.sequence_length = match.matched_tokens
            state.tail_valid_tokens = self.page_size
            state.layer_lengths[:] = [match.matched_tokens] * self.layer_count
            state.token_block_hashes[:] = list(match.block_hashes)
            state.version += 1
        except BaseException:
            runtime.release(state)
            raise
        self.metrics.prefix_hits += 1
        return state, match

    def close(self):
        handles = tuple(self.owned_handles.values())
        self.page_pool.assert_releasable(handles, "close prefix cache")
        for handle in reversed(handles):
            self.page_pool.release(handle)
        self.owned_handles.clear()

    def __len__(self):
        return len(self.index)
