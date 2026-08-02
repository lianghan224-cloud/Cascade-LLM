"""Copy-on-write transaction description shared by runtime implementations."""

from dataclasses import dataclass

from .types import PageHandle


@dataclass(frozen=True)
class PageCOWOperation:
    source: PageHandle
    target: PageHandle
    valid_tokens: int
    logical_block: int
