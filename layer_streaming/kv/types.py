"""Frozen V1 page and request-level value types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .errors import KVLifecycleError


KV_FRAMEWORK_ABI_VERSION = 1
KV_PAGE_FORMAT_VERSION = 1


class PageState(str, Enum):
    FREE = "free"
    ALLOCATED = "allocated"
    ACTIVE = "active"
    SEALED = "sealed"
    SHARED = "shared"
    COPYING = "copying"
    MIGRATING = "migrating"
    EVICT_PENDING = "evict_pending"
    RELEASING = "releasing"


class RequestLifecycleState(str, Enum):
    ACTIVE = "active"
    BRANCH = "branch"
    COMMITTED = "committed"
    RELEASING = "releasing"
    RELEASED = "released"


@dataclass
class PageDescriptor:
    page_id: int
    generation: int
    store_id: str
    dtype: str
    layout: str
    format_version: int = KV_PAGE_FORMAT_VERSION
    state: PageState = PageState.FREE
    valid_tokens: int = 0
    ref_count: int = 0
    pin_count: int = 0
    owner_hint: Optional[int] = None
    last_access_epoch: int = 0
    index_metadata_handle: object = field(default=None, repr=False)
    transfer_event: object = field(default=None, repr=False)

    def __post_init__(self):
        # These V2 lifecycle attributes intentionally remain outside the
        # frozen V1 dataclass field list.  That keeps the published V1 ABI
        # stable while giving the runtime one canonical place for version and
        # in-flight accounting.
        self.data_version = 0
        self.index_version = 0
        self.inflight_compute = 0
        self.inflight_io = 0
        self.dirty = False
        self.error = None
        self.logical_mappings = set()

    def as_dict(self):
        return {
            "page_id": int(self.page_id),
            "generation": int(self.generation),
            "store_id": self.store_id,
            "dtype": self.dtype,
            "layout": self.layout,
            "format_version": int(self.format_version),
            "state": self.state.value,
            "valid_tokens": int(self.valid_tokens),
            "ref_count": int(self.ref_count),
            "pin_count": int(self.pin_count),
            "owner_hint": self.owner_hint,
            "last_access_epoch": int(self.last_access_epoch),
            "data_version": int(self.data_version),
            "index_version": int(self.index_version),
            "inflight_compute": int(self.inflight_compute),
            "inflight_io": int(self.inflight_io),
            "dirty": bool(self.dirty),
            "error": self.error,
            "logical_mapping_count": len(self.logical_mappings),
            "has_index_metadata": self.index_metadata_handle is not None,
            "has_transfer_event": self.transfer_event is not None,
        }


class PageHandle:
    """Stable generation-checked reference to a page descriptor.

    State is exposed through the descriptor so every request sharing a handle
    observes the same lifecycle transition without rewriting its block table.
    """

    __slots__ = ("page_id", "generation", "store_id", "format_id", "_descriptor")

    def __init__(self, descriptor):
        self.page_id = int(descriptor.page_id)
        self.generation = int(descriptor.generation)
        self.store_id = str(descriptor.store_id)
        self.format_id = "{}:{}:{}".format(
            descriptor.dtype,
            descriptor.layout,
            descriptor.format_version,
        )
        self._descriptor = descriptor

    @property
    def state(self):
        if self._descriptor.generation != self.generation:
            raise KVLifecycleError(
                "stale PageHandle state access: page={} handle_generation={} current_generation={}".format(
                    self.page_id,
                    self.generation,
                    self._descriptor.generation,
                )
            )
        return self._descriptor.state

    def identity(self):
        return (
            self.page_id,
            self.generation,
            self.store_id,
            self.format_id,
        )

    def as_dict(self):
        return {
            "page_id": self.page_id,
            "generation": self.generation,
            "store_id": self.store_id,
            "format_id": self.format_id,
            "state": self.state.value,
        }

    def __eq__(self, other):
        return isinstance(other, PageHandle) and self.identity() == other.identity()

    def __hash__(self):
        return hash(self.identity())

    def __repr__(self):
        return "PageHandle(page_id={}, generation={}, store_id={!r}, state={!r})".format(
            self.page_id,
            self.generation,
            self.store_id,
            self.state.value,
        )
