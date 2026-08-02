"""Paged provider workspace contracts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PagedWorkspaceEstimate:
    bytes: int
    policy: str
    scales_with_context: bool
    contains_full_kv: bool
    contains_full_scores: bool

    def validate_production(self):
        if self.scales_with_context or self.contains_full_kv or self.contains_full_scores:
            raise ValueError("production paged workspace violates V1 memory contract")
        return self
