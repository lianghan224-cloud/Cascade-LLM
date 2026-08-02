"""KV V1 runtime metrics with stable report semantics."""

from dataclasses import asdict, dataclass


@dataclass
class KVMetrics:
    pool_peak_pages: int = 0
    shared_pages: int = 0
    cow_count: int = 0
    fork_count: int = 0
    committed_tokens: int = 0
    append_tokens: int = 0
    append_calls: int = 0
    attention_calls: int = 0
    release_count: int = 0
    prefix_hits: int = 0
    prefix_misses: int = 0
    prefill_attention_ms: float = 0.0
    decode_attention_ms: float = 0.0
    workspace_peak_bytes: int = 0

    def as_dict(self):
        return asdict(self)
