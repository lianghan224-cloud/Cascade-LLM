"""Shape-aware Full/Chunked Prefill and Decode routing contract."""

from enum import Enum


class PagedWorkload(str, Enum):
    FULL_PREFILL = "full_prefill"
    CHUNKED_PREFILL = "chunked_prefill"
    DECODE = "decode"
    SHORT_SUFFIX = "short_suffix"

    @property
    def backend_method(self):
        return "decode" if self == PagedWorkload.DECODE else "prefill"

    @property
    def capability_phase(self):
        return "decode" if self == PagedWorkload.DECODE else "prefill"


def classify_paged_workload(request, phase=None):
    """Classify without treating every Q>1 workload as generic prefill."""

    if isinstance(phase, PagedWorkload):
        return phase
    normalized = None if phase is None else str(phase).lower().replace("-", "_")
    explicit = {
        item.value: item for item in PagedWorkload
    }
    if normalized in explicit:
        return explicit[normalized]
    query_lengths = tuple(int(item) for item in request.batch_view.query_lengths.tolist())
    sequence_lengths = tuple(
        int(item) for item in request.batch_view.sequence_lengths.tolist()
    )
    if not query_lengths:
        raise ValueError("paged workload requires a non-empty batch")
    maximum = max(query_lengths)
    if normalized == "decode":
        if maximum != 1:
            raise ValueError("decode phase requires exactly one query token")
        return PagedWorkload.DECODE
    if maximum == 1 and normalized is None:
        return PagedWorkload.DECODE
    if all(query == sequence for query, sequence in zip(query_lengths, sequence_lengths)):
        return PagedWorkload.FULL_PREFILL
    if maximum <= 4:
        return PagedWorkload.SHORT_SUFFIX
    return PagedWorkload.CHUNKED_PREFILL
