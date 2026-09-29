"""Paged provider workspace contracts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PagedWorkspaceShape:
    batch_size: int
    max_sequence_length: int
    num_query_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    dtype: str
    dtype_bytes: int

    def validate(self):
        for name in (
            "batch_size",
            "max_sequence_length",
            "num_query_heads",
            "num_kv_heads",
            "head_dim",
            "page_size",
            "dtype_bytes",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError("workspace shape {} must be positive".format(name))
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError("workspace shape has invalid GQA head ratio")
        return self


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
