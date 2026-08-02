"""Stable public adapters over the batch-first PagedKVRuntime API."""


class RequestKVCacheV1:
    """Single-request executor adapter over the batch-first runtime ABI."""

    def __init__(self, runtime, state):
        self.runtime = runtime
        self.state = state
        self.manager = runtime

    @property
    def max_length(self):
        return self.state.block_table.max_length

    @property
    def nbytes(self):
        return self.runtime.store.nbytes

    @property
    def policy(self):
        return self.runtime.policy

    def sequence_length(self):
        return self.state.sequence_length

    def append_only(self, layer_index, key, value):
        token_count = int(key.shape[2])
        return self.runtime.append(
            (self.state,),
            layer_index,
            key,
            value,
            (token_count,),
        )[0]

    def attend(self, layer_index, query, kv_groups=1, position_ids=None):
        groups = self.runtime.num_query_heads // self.runtime.num_kv_heads
        if int(kv_groups) != groups:
            raise ValueError("executor KV group count does not match runtime")
        token_count = int(query.shape[2])
        positions = None
        if position_ids is not None:
            positions = (position_ids.detach().reshape(-1),)
        result = self.runtime.attend(
            (self.state,),
            layer_index,
            query,
            (token_count,),
            query_positions=positions,
            causal=True,
        )
        return result.output.transpose(0, 1).unsqueeze(0)

    def clear(self):
        self.runtime.reset(self.state)

    def close(self):
        self.runtime.release(self.state)

    def profile_stats(self):
        return self.runtime.profile_stats()
