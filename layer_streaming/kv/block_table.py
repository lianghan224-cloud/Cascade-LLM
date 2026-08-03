"""Logical request block tables independent of storage addresses."""

from dataclasses import dataclass, field

from .errors import KVLifecycleError
from .types import PageHandle


@dataclass
class LogicalBlockTable:
    page_size: int
    max_length: int
    handles: list = field(default_factory=list)
    version: int = 0

    def __post_init__(self):
        self.page_size = int(self.page_size)
        self.max_length = int(self.max_length)
        if self.page_size <= 0 or self.max_length <= 0:
            raise ValueError("page_size and max_length must be positive")
        if any(not isinstance(item, PageHandle) for item in self.handles):
            raise TypeError("block table entries must be PageHandle objects")

    def append(self, handle):
        if not isinstance(handle, PageHandle):
            raise TypeError("block table entries must be PageHandle objects")
        if len(self.handles) * self.page_size >= self.max_length:
            raise KVLifecycleError("block table capacity is exhausted")
        self.handles.append(handle)
        self.version += 1

    def replace(self, logical_block, handle):
        if not isinstance(handle, PageHandle):
            raise TypeError("block table entries must be PageHandle objects")
        logical_block = int(logical_block)
        if logical_block < 0 or logical_block >= len(self.handles):
            raise IndexError("logical block is outside the block table")
        self.handles[logical_block] = handle
        self.version += 1

    def truncate(self, count):
        count = int(count)
        if count < 0 or count > len(self.handles):
            raise ValueError("invalid block table truncation")
        removed = self.handles[count:]
        del self.handles[count:]
        if removed:
            self.version += 1
        return tuple(removed)

    def physical_page_ids(self, required_store="gpu"):
        result = []
        for handle in self.handles:
            # Accessing state performs the handle-local generation check before
            # a raw physical ID is compacted for a provider.
            handle.state
            if required_store is not None and handle.store_id != required_store:
                raise KVLifecycleError(
                    "page {} is in store {}, expected {}".format(
                        handle.page_id,
                        handle.store_id,
                        required_store,
                    )
                )
            result.append(handle.page_id)
        return tuple(result)

    def clone_handles(self):
        return list(self.handles)

    def as_dict(self):
        return {
            "page_size": self.page_size,
            "max_length": self.max_length,
            "version": self.version,
            "logical_to_page": [item.as_dict() for item in self.handles],
        }
