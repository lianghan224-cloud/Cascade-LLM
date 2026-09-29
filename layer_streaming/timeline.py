"""Pure helpers for non-overlapping CUDA activity accounting."""

import math


def _merge_intervals(intervals):
    normalized = []
    for start, end in intervals:
        start = float(start)
        end = float(end)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("timeline intervals must be finite")
        if start < 0 or end < start:
            raise ValueError("timeline interval is invalid")
        if end > start:
            normalized.append((start, end))
    normalized.sort()
    merged = []
    for start, end in normalized:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return tuple((start, end) for start, end in merged)


def _duration(intervals):
    return sum(end - start for start, end in intervals)


def _intersection_duration(left, right):
    left = _merge_intervals(left)
    right = _merge_intervals(right)
    left_index = 0
    right_index = 0
    result = 0.0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        result += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return result


def analyze_copy_compute_timeline(copy_intervals, compute_intervals, wall_ms):
    """Return mutually exclusive copy/compute/overlap/idle durations.

    Raw CUDA event sums overlap by design and therefore must not be added to
    host wall time.  This function first merges each stream's intervals, then
    accounts their intersection exactly once.
    """

    wall_ms = float(wall_ms)
    if not math.isfinite(wall_ms) or wall_ms < 0:
        raise ValueError("timeline wall_ms must be finite and non-negative")
    copy = _merge_intervals(copy_intervals)
    compute = _merge_intervals(compute_intervals)
    copy_busy = _duration(copy)
    compute_busy = _duration(compute)
    overlap = _intersection_duration(copy, compute)
    copy_only = copy_busy - overlap
    compute_only = compute_busy - overlap
    active_union = copy_only + compute_only + overlap
    timeline_span = max(
        [0.0]
        + [end for _, end in copy]
        + [end for _, end in compute]
    )
    idle = max(0.0, wall_ms - active_union)
    accounted = copy_only + compute_only + overlap + idle
    return {
        "copy_busy_ms": copy_busy,
        "compute_busy_ms": compute_busy,
        "copy_compute_overlap_ms": overlap,
        "copy_only_ms": copy_only,
        "compute_only_ms": compute_only,
        "gpu_timeline_active_ms": active_union,
        "gpu_timeline_idle_or_host_overhead_ms": idle,
        "gpu_timeline_span_ms": timeline_span,
        "copy_overlap_ratio": 0.0 if copy_busy == 0.0 else overlap / copy_busy,
        "compute_overlap_ratio": (
            0.0 if compute_busy == 0.0 else overlap / compute_busy
        ),
        "gpu_timeline_active_ratio": (
            0.0 if wall_ms == 0.0 else min(1.0, active_union / wall_ms)
        ),
        "critical_path_accounted_ms": accounted,
        "critical_path_accounting_error_ms": wall_ms - accounted,
    }
