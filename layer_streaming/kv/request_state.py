"""Request KV state and append transaction metadata."""

from dataclasses import dataclass, field
from typing import Optional

from .block_table import LogicalBlockTable
from .types import RequestLifecycleState


@dataclass
class PendingAppend:
    start: int
    token_count: int
    slot_page_ids: object
    slot_offsets: object
    completed_layers: set = field(default_factory=set)
    original_block_count: int = 0
    original_layer_lengths: tuple = ()
    allocated_handles: list = field(default_factory=list)
    cow_original: object = None
    cow_replacement: object = None

    @property
    def end(self):
        return self.start + self.token_count


@dataclass
class RequestKVState:
    request_id: int
    block_table: LogicalBlockTable
    sequence_length: int = 0
    tail_valid_tokens: int = 0
    version: int = 0
    parent_request_id: Optional[int] = None
    fork_position: Optional[int] = None
    reuse_namespace: str = "default"
    lifecycle_state: RequestLifecycleState = RequestLifecycleState.ACTIVE
    layer_lengths: list = field(default_factory=list, repr=False)
    pending_append: Optional[PendingAppend] = field(default=None, repr=False)
    token_block_hashes: list = field(default_factory=list, repr=False)

    def ensure_active(self):
        if self.lifecycle_state in {
            RequestLifecycleState.RELEASING,
            RequestLifecycleState.RELEASED,
        }:
            raise RuntimeError("request KV state has been released")

    def as_dict(self):
        return {
            "request_id": int(self.request_id),
            "block_table": self.block_table.as_dict(),
            "sequence_length": int(self.sequence_length),
            "tail_valid_tokens": int(self.tail_valid_tokens),
            "version": int(self.version),
            "parent_request_id": self.parent_request_id,
            "fork_position": self.fork_position,
            "reuse_namespace": self.reuse_namespace,
            "lifecycle_state": self.lifecycle_state.value,
        }
