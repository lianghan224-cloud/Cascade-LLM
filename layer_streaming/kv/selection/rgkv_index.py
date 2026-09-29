"""Incremental compact index and budget contract for RGKV.

This module owns selection metadata only.  It never retains a PageHandle,
changes a Request reference, or resolves a physical KV location.  One index
normally represents one request layer, so logical page IDs are unique within
the index.

The summary tensor keeps the KV-head/group dimensions intact.  Its leading
axis is ``[minimum, maximum, mean]`` and all remaining axes match one token's
key shape.  Append updates combine summaries algebraically; they never scan
historical key rows.  Rollback is intentionally different: the affected tail
page must be supplied again because a min/max summary cannot remove tokens.
"""

from dataclasses import dataclass, replace
import time

import torch

from ..errors import KVLifecycleError


STALE_INDEX = "STALE_INDEX"


class RGKVStaleIndexError(KVLifecycleError):
    """The published summary does not describe the current page epoch."""

    code = STALE_INDEX


@dataclass(frozen=True)
class RGKVBudget:
    """Hard page budget split between mandatory recent and relevance pages.

    ``recent_pages`` is part of, not additive to, ``total_page_budget``.
    Consequently a selection can never exceed ``total_page_budget``.
    """

    total_page_budget: int
    recent_pages: int = 0

    def __post_init__(self):
        total = int(self.total_page_budget)
        recent = int(self.recent_pages)
        if total < 0:
            raise ValueError("RGKV total_page_budget must not be negative")
        if recent < 0:
            raise ValueError("RGKV recent_pages must not be negative")
        if recent > total:
            raise ValueError(
                "RGKV recent_pages must not exceed total_page_budget"
            )
        object.__setattr__(self, "total_page_budget", total)
        object.__setattr__(self, "recent_pages", recent)

    @property
    def relevance_pages(self):
        return int(self.total_page_budget - self.recent_pages)

    def resolve(self, candidate_count):
        """Return ``(selected_total, recent, relevance)`` for this request."""

        candidate_count = int(candidate_count)
        if candidate_count < 0:
            raise ValueError("RGKV candidate_count must not be negative")
        selected_total = min(self.total_page_budget, candidate_count)
        recent = min(self.recent_pages, selected_total)
        relevance = min(selected_total - recent, candidate_count - recent)
        return selected_total, recent, relevance


@dataclass(frozen=True)
class RGKVPageSummary:
    """One compact, epoch-bound logical-page summary."""

    logical_page_id: int
    data_epoch: int
    valid_tokens: int
    values: torch.Tensor
    sealed: bool = False

    def __post_init__(self):
        logical_page_id = int(self.logical_page_id)
        data_epoch = int(self.data_epoch)
        valid_tokens = int(self.valid_tokens)
        if logical_page_id < 0:
            raise ValueError("RGKV logical_page_id must not be negative")
        if data_epoch < 0:
            raise ValueError("RGKV data_epoch must not be negative")
        if valid_tokens < 0:
            raise ValueError("RGKV valid_tokens must not be negative")
        if not isinstance(self.values, torch.Tensor):
            raise TypeError("RGKV summary values must be a Tensor")
        if self.values.ndim < 2 or int(self.values.shape[0]) != 3:
            raise ValueError(
                "RGKV summary values must have shape [3, ...features]"
            )
        if not self.values.is_contiguous():
            raise ValueError("RGKV summary values must be contiguous")
        if not self.values.is_floating_point():
            raise TypeError("RGKV summary values must be floating point")
        object.__setattr__(self, "logical_page_id", logical_page_id)
        object.__setattr__(self, "data_epoch", data_epoch)
        object.__setattr__(self, "valid_tokens", valid_tokens)
        object.__setattr__(self, "sealed", bool(self.sealed))

    @property
    def minimum(self):
        return self.values[0]

    @property
    def maximum(self):
        return self.values[1]

    @property
    def mean(self):
        return self.values[2]


@dataclass(frozen=True)
class RGKVSelectionResult:
    """Tensor-only selection result in logical order."""

    selected_logical_pages: torch.Tensor
    selected_scores: torch.Tensor
    candidate_count: int
    selected_count: int
    total_page_budget: int
    mandatory_recent_pages: int
    relevance_selected_pages: int

    def __post_init__(self):
        if not isinstance(self.selected_logical_pages, torch.Tensor):
            raise TypeError("selected_logical_pages must be a Tensor")
        if not isinstance(self.selected_scores, torch.Tensor):
            raise TypeError("selected_scores must be a Tensor")
        if self.selected_logical_pages.ndim != 1:
            raise ValueError("selected_logical_pages must be rank one")
        if self.selected_scores.ndim != 1:
            raise ValueError("selected_scores must be rank one")
        if self.selected_logical_pages.numel() != self.selected_scores.numel():
            raise ValueError("selected page and score counts must match")
        if int(self.selected_logical_pages.numel()) != int(self.selected_count):
            raise ValueError("selected_count does not match result tensors")
        if int(self.selected_count) > int(self.total_page_budget):
            raise ValueError("RGKV selection exceeded total_page_budget")


def _page_rows(keys):
    if not isinstance(keys, torch.Tensor):
        keys = torch.as_tensor(keys, dtype=torch.float32)
    if keys.ndim < 2:
        raise ValueError("RGKV page keys must have shape [tokens, ...features]")
    if any(int(size) <= 0 for size in keys.shape[1:]):
        raise ValueError("RGKV key feature dimensions must be non-empty")
    if not keys.is_floating_point():
        keys = keys.float()
    return keys


def _summarize(keys, valid_tokens=None):
    keys = _page_rows(keys)
    available = int(keys.shape[0])
    valid_tokens = available if valid_tokens is None else int(valid_tokens)
    if valid_tokens <= 0 or valid_tokens > available:
        raise ValueError(
            "RGKV valid_tokens must be positive and no larger than key rows"
        )
    rows = keys[:valid_tokens].float()
    return torch.stack(
        (rows.amin(dim=0), rows.amax(dim=0), rows.mean(dim=0)), dim=0
    ).contiguous()


class RGKVIndex:
    """Incremental, epoch-validated compact summary index.

    Lifecycle methods return immutable :class:`RGKVPageSummary` values.  The
    index itself is mutable publication state, but it owns no KV page or
    Request lifetime reference.
    """

    def __init__(self):
        self._pages = {}
        self.page_builds = 0
        self.append_updates = 0
        self.rollback_rebuilds = 0
        self.cow_copies = 0
        self.prefix_shares = 0
        self.stale_index_count = 0
        self.selection_count = 0
        self.selection_ms = 0.0

    def __len__(self):
        return len(self._pages)

    def pages(self):
        return tuple(self._pages[key] for key in sorted(self._pages))

    def get(self, logical_page_id, expected_data_epoch=None):
        logical_page_id = int(logical_page_id)
        try:
            summary = self._pages[logical_page_id]
        except KeyError:
            self._stale(
                "missing RGKV summary for logical page {}".format(
                    logical_page_id
                )
            )
        if expected_data_epoch is not None:
            self._validate_epoch(summary, expected_data_epoch)
        return summary

    def build_page(
        self,
        logical_page_id,
        keys,
        *,
        data_epoch,
        valid_tokens=None,
        sealed=False,
    ):
        """Build and publish one new logical page without touching others."""

        logical_page_id = int(logical_page_id)
        if logical_page_id in self._pages:
            raise KVLifecycleError(
                "RGKV page {} is already indexed".format(logical_page_id)
            )
        keys = _page_rows(keys)
        values = _summarize(keys, valid_tokens)
        resolved_tokens = (
            int(keys.shape[0]) if valid_tokens is None else int(valid_tokens)
        )
        summary = RGKVPageSummary(
            logical_page_id=logical_page_id,
            data_epoch=int(data_epoch),
            valid_tokens=resolved_tokens,
            values=values,
            sealed=sealed,
        )
        self._pages[logical_page_id] = summary
        self.page_builds += 1
        return summary

    def append(self, logical_page_id, appended_keys, *, data_epoch):
        """Update only the changed tail page from its old summary + new rows."""

        current = self.get(logical_page_id)
        if current.sealed:
            raise KVLifecycleError(
                "sealed RGKV page {} is immutable".format(logical_page_id)
            )
        self._require_new_epoch(current, data_epoch)
        appended = _page_rows(appended_keys).float()
        added_tokens = int(appended.shape[0])
        if added_tokens <= 0:
            raise ValueError("RGKV append requires at least one token")
        if tuple(appended.shape[1:]) != tuple(current.values.shape[1:]):
            raise ValueError("RGKV appended key geometry does not match summary")
        added = torch.stack(
            (
                appended.amin(dim=0),
                appended.amax(dim=0),
                appended.mean(dim=0),
            ),
            dim=0,
        )
        old_tokens = int(current.valid_tokens)
        combined_mean = (
            current.mean * float(old_tokens) + added[2] * float(added_tokens)
        ) / float(old_tokens + added_tokens)
        values = torch.stack(
            (
                torch.minimum(current.minimum, added[0]),
                torch.maximum(current.maximum, added[1]),
                combined_mean,
            ),
            dim=0,
        ).contiguous()
        updated = RGKVPageSummary(
            logical_page_id=current.logical_page_id,
            data_epoch=int(data_epoch),
            valid_tokens=old_tokens + added_tokens,
            values=values,
            sealed=False,
        )
        self._pages[current.logical_page_id] = updated
        self.append_updates += 1
        return updated

    def seal(self, logical_page_id, *, expected_data_epoch):
        """Freeze a complete page summary without changing its data epoch."""

        current = self.get(logical_page_id, expected_data_epoch)
        if current.sealed:
            return current
        sealed = replace(current, sealed=True)
        self._pages[current.logical_page_id] = sealed
        return sealed

    def rollback_tail(
        self,
        logical_page_id,
        remaining_keys,
        *,
        data_epoch,
        valid_tokens=None,
    ):
        """Rebuild only the affected mutable tail after token removal."""

        current = self.get(logical_page_id)
        if current.sealed:
            raise KVLifecycleError(
                "sealed RGKV page {} cannot be rolled back in place".format(
                    logical_page_id
                )
            )
        self._require_new_epoch(current, data_epoch)
        remaining_keys = _page_rows(remaining_keys)
        values = _summarize(remaining_keys, valid_tokens)
        resolved_tokens = (
            int(remaining_keys.shape[0])
            if valid_tokens is None
            else int(valid_tokens)
        )
        updated = RGKVPageSummary(
            logical_page_id=current.logical_page_id,
            data_epoch=int(data_epoch),
            valid_tokens=resolved_tokens,
            values=values,
            sealed=False,
        )
        self._pages[current.logical_page_id] = updated
        self.rollback_rebuilds += 1
        return updated

    def cow_clone(
        self,
        source_logical_page_id,
        target_logical_page_id,
        *,
        data_epoch,
        target_keys=None,
        valid_tokens=None,
    ):
        """Publish a mutable COW target, rebuilding only when shape changed."""

        source = self.get(source_logical_page_id)
        target_logical_page_id = int(target_logical_page_id)
        if target_logical_page_id in self._pages:
            raise KVLifecycleError(
                "RGKV COW target {} is already indexed".format(
                    target_logical_page_id
                )
            )
        if int(data_epoch) <= int(source.data_epoch):
            raise KVLifecycleError(
                "RGKV COW target data_epoch must advance beyond source"
            )
        if target_keys is None:
            if valid_tokens is not None and int(valid_tokens) != source.valid_tokens:
                raise KVLifecycleError(
                    "RGKV COW with changed valid_tokens requires target_keys"
                )
            values = source.values.clone().contiguous()
            resolved_tokens = source.valid_tokens
        else:
            target_keys = _page_rows(target_keys)
            values = _summarize(target_keys, valid_tokens)
            resolved_tokens = (
                int(target_keys.shape[0])
                if valid_tokens is None
                else int(valid_tokens)
            )
        target = RGKVPageSummary(
            logical_page_id=target_logical_page_id,
            data_epoch=int(data_epoch),
            valid_tokens=resolved_tokens,
            values=values,
            sealed=False,
        )
        self._pages[target_logical_page_id] = target
        self.cow_copies += 1
        return target

    def share_sealed_prefix(
        self, source_logical_page_id, target_logical_page_id, *, data_epoch
    ):
        """Share immutable summary storage for a fork/prefix logical page."""

        source = self.get(source_logical_page_id, data_epoch)
        if not source.sealed:
            raise KVLifecycleError("only sealed RGKV summaries may be shared")
        target_logical_page_id = int(target_logical_page_id)
        if target_logical_page_id in self._pages:
            raise KVLifecycleError(
                "RGKV prefix target {} is already indexed".format(
                    target_logical_page_id
                )
            )
        target = RGKVPageSummary(
            logical_page_id=target_logical_page_id,
            data_epoch=source.data_epoch,
            valid_tokens=source.valid_tokens,
            values=source.values,
            sealed=True,
        )
        self._pages[target_logical_page_id] = target
        self.prefix_shares += 1
        return target

    def drop(self, logical_page_id):
        return self._pages.pop(int(logical_page_id), None)

    def validate(self, page_epochs):
        """Validate every indexed page against authoritative page epochs."""

        for logical_page_id, summary in self._pages.items():
            if logical_page_id not in page_epochs:
                self._stale(
                    "missing page epoch for RGKV logical page {}".format(
                        logical_page_id
                    )
                )
            self._validate_epoch(summary, page_epochs[logical_page_id])
        return True

    def packed_summaries(self, logical_page_ids=None):
        """Return contiguous ``[page, 3, ...features]`` index storage."""

        logical_page_ids = self._resolve_candidates(logical_page_ids)
        if not logical_page_ids:
            return torch.empty((0, 3, 0), dtype=torch.float32)
        return torch.stack(
            tuple(self._pages[item].values for item in logical_page_ids), dim=0
        ).contiguous()

    def select(self, query, budget, *, page_epochs, logical_page_ids=None):
        """Score non-recent pages, merge mandatory recent, restore order."""

        started = time.perf_counter()
        if not isinstance(budget, RGKVBudget):
            raise TypeError("budget must be RGKVBudget")
        logical_page_ids = self._resolve_candidates(logical_page_ids)
        for logical_page_id in logical_page_ids:
            if logical_page_id not in page_epochs:
                self._stale(
                    "missing page epoch for RGKV logical page {}".format(
                        logical_page_id
                    )
                )
            self._validate_epoch(
                self._pages[logical_page_id], page_epochs[logical_page_id]
            )

        candidate_count = len(logical_page_ids)
        _, recent_count, relevance_count = budget.resolve(candidate_count)
        device = query.device if isinstance(query, torch.Tensor) else torch.device("cpu")
        if not candidate_count or not budget.total_page_budget:
            empty_ids = torch.empty((0,), dtype=torch.int64, device=device)
            empty_scores = torch.empty((0,), dtype=torch.float32, device=device)
            result = RGKVSelectionResult(
                selected_logical_pages=empty_ids,
                selected_scores=empty_scores,
                candidate_count=candidate_count,
                selected_count=0,
                total_page_budget=budget.total_page_budget,
                mandatory_recent_pages=0,
                relevance_selected_pages=0,
            )
            self._record_selection(started)
            return result

        summaries = self.packed_summaries(logical_page_ids).to(device=device)
        query_tensor = torch.as_tensor(query, device=device, dtype=torch.float32)
        feature_shape = tuple(summaries.shape[2:])
        if tuple(query_tensor.shape) != feature_shape:
            if int(query_tensor.numel()) != int(summaries[0, 0].numel()):
                raise ValueError("RGKV query geometry does not match summaries")
            query_tensor = query_tensor.reshape(feature_shape)
        scores = torch.maximum(
            summaries[:, 0] * query_tensor,
            summaries[:, 1] * query_tensor,
        ).flatten(1).sum(dim=1)

        scored_count = candidate_count - recent_count
        if relevance_count:
            ranked = torch.argsort(
                scores[:scored_count], descending=True, stable=True
            )[:relevance_count]
        else:
            ranked = torch.empty((0,), dtype=torch.int64, device=device)
        recent = torch.arange(
            scored_count, candidate_count, dtype=torch.int64, device=device
        )
        selected_positions = torch.sort(torch.cat((ranked, recent))).values
        logical_tensor = torch.tensor(
            logical_page_ids, dtype=torch.int64, device=device
        )
        selected_ids = logical_tensor[selected_positions].contiguous()
        selected_scores = scores[selected_positions].contiguous()
        result = RGKVSelectionResult(
            selected_logical_pages=selected_ids,
            selected_scores=selected_scores,
            candidate_count=candidate_count,
            selected_count=int(selected_positions.numel()),
            total_page_budget=budget.total_page_budget,
            mandatory_recent_pages=recent_count,
            relevance_selected_pages=relevance_count,
        )
        self._record_selection(started)
        return result

    def stats(self):
        index_bytes = sum(
            int(item.values.numel() * item.values.element_size())
            for item in self._pages.values()
        )
        return {
            "pages": len(self._pages),
            "sealed_pages": sum(item.sealed for item in self._pages.values()),
            "index_bytes": index_bytes,
            "page_builds": int(self.page_builds),
            "append_updates": int(self.append_updates),
            "rollback_rebuilds": int(self.rollback_rebuilds),
            "cow_copies": int(self.cow_copies),
            "prefix_shares": int(self.prefix_shares),
            "stale_index_count": int(self.stale_index_count),
            "selections": int(self.selection_count),
            "selection_ms": float(self.selection_ms),
        }

    def _resolve_candidates(self, logical_page_ids):
        if logical_page_ids is None:
            return tuple(sorted(self._pages))
        resolved = tuple(int(item) for item in logical_page_ids)
        if len(set(resolved)) != len(resolved):
            raise ValueError("RGKV candidates must not contain duplicates")
        resolved = tuple(sorted(resolved))
        for logical_page_id in resolved:
            if logical_page_id not in self._pages:
                self._stale(
                    "missing RGKV summary for logical page {}".format(
                        logical_page_id
                    )
                )
        return resolved

    def _validate_epoch(self, summary, expected_data_epoch):
        expected = int(expected_data_epoch)
        if int(summary.data_epoch) != expected:
            self._stale(
                "RGKV {} logical_page={} summary_epoch={} page_epoch={}".format(
                    STALE_INDEX,
                    summary.logical_page_id,
                    summary.data_epoch,
                    expected,
                )
            )

    @staticmethod
    def _require_new_epoch(summary, data_epoch):
        if int(data_epoch) <= int(summary.data_epoch):
            raise KVLifecycleError(
                "RGKV data_epoch must strictly increase: current={} new={}".format(
                    summary.data_epoch, int(data_epoch)
                )
            )

    def _stale(self, message):
        self.stale_index_count += 1
        raise RGKVStaleIndexError(message)

    def _record_selection(self, started):
        self.selection_count += 1
        self.selection_ms += (time.perf_counter() - started) * 1000.0


__all__ = [
    "RGKVBudget",
    "RGKVIndex",
    "RGKVPageSummary",
    "RGKVSelectionResult",
    "RGKVStaleIndexError",
    "STALE_INDEX",
]
