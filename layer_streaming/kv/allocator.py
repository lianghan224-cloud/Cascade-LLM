"""Admission calculations independent of stores and providers."""

from dataclasses import dataclass
import math


def pages_for_tokens(token_count, page_size):
    token_count = int(token_count)
    page_size = int(page_size)
    if token_count < 0 or page_size <= 0:
        raise ValueError("token_count must be non-negative and page_size positive")
    return int(math.ceil(token_count / float(page_size))) if token_count else 0


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    required_pages: int
    available_pages: int
    reserved_pages: int
    reason: object = None


def decide_admission(token_capacity, page_size, free_pages, reserved_pages=0):
    required = pages_for_tokens(token_capacity, page_size)
    available = max(0, int(free_pages) - int(reserved_pages))
    admitted = required <= available
    return AdmissionDecision(
        admitted=admitted,
        required_pages=required,
        available_pages=available,
        reserved_pages=int(reserved_pages),
        reason=(None if admitted else "insufficient_free_pages"),
    )
