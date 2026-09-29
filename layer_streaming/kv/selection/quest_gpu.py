"""Tensorized Torch scorer for the compact Quest summary ABI.

The provider is explicitly opt-in.  It consumes the frozen CPU FP32
``QuestIndexRecord.compact_summary`` representation while index lifecycle
work is being published, then keeps a packed copy on the query device.  The
selection call therefore performs scoring, ranking, recent-page merging and
logical ordering entirely with tensors on that device.
"""

from dataclasses import dataclass

import torch


def quest_scorer_provider(name):
    """Resolve an explicitly validated Quest scorer policy name."""

    name = str(name)
    if name == "cpu_reference":
        return None
    if name == "torch_tensorized":
        return TorchTensorizedQuestScorer()
    raise ValueError("unknown Quest scorer {!r}".format(name))


@dataclass(frozen=True)
class QuestSelectionWorkspaceEstimate:
    """Persistent and temporary bytes owned by tensorized Quest selection."""

    index_bytes: int
    scoring_bytes: int
    topk_bytes: int
    merge_bytes: int

    @property
    def temporary_bytes(self):
        return int(self.scoring_bytes + self.topk_bytes + self.merge_bytes)

    @property
    def total_bytes(self):
        return int(self.index_bytes + self.temporary_bytes)

    def as_dict(self):
        return {
            "index_bytes": int(self.index_bytes),
            "scoring_bytes": int(self.scoring_bytes),
            "topk_bytes": int(self.topk_bytes),
            "merge_bytes": int(self.merge_bytes),
            "temporary_bytes": self.temporary_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class TensorizedQuestSelectionResult:
    selected_positions: torch.Tensor
    mode: str
    candidate_count: int
    selected_count: int
    scored_budget: int
    recent_budget: int
    total_budget: int

    def as_dict(self):
        return {
            "mode": self.mode,
            "candidate_count": int(self.candidate_count),
            "selected_count": int(self.selected_count),
            "scored_budget": int(self.scored_budget),
            "recent_budget": int(self.recent_budget),
            "total_budget": int(self.total_budget),
            "selected_total": int(self.selected_count),
        }


@dataclass(frozen=True)
class _TensorizedQuestIndex:
    summaries: torch.Tensor
    logical_blocks: torch.Tensor
    data_epochs: torch.Tensor
    dimensions: int
    record_count: int


class TorchTensorizedQuestScorer:
    """Torch tensorized Quest scorer with a device-resident compact index.

    ``prepare_index`` is deliberately separate from ``select``.  Callers
    publish changed compact records during append/COW/rollback and selection
    then has no CPU transfer or scalar extraction in its decode hot path.
    Stable descending argsort supplies deterministic ties because records are
    packed in logical-page order.
    """

    name = "torch_tensorized"
    qualification_status = "experimental"

    def __init__(self):
        self._indices = {}
        self.query_count = 0
        self.candidate_count = 0
        self.selected_count = 0
        self.index_builds = 0

    @staticmethod
    def estimate_workspace(candidate_count, dimensions):
        """Estimate the additive selection allocation without MemoryPlanner.

        The estimate includes the packed FP32 ``[min,max,mean]`` index, two
        int64 metadata vectors, FP32 scores, an int64 stable rank and an
        int64 merge/order vector.  It is intentionally independent of the
        global planner so the Selection Plane can be integrated additively.
        """

        candidate_count = int(candidate_count)
        dimensions = int(dimensions)
        if candidate_count < 0 or dimensions < 0:
            raise ValueError("Quest workspace geometry must not be negative")
        fp32_bytes = torch.empty((), dtype=torch.float32).element_size()
        int64_bytes = torch.empty((), dtype=torch.int64).element_size()
        return QuestSelectionWorkspaceEstimate(
            index_bytes=candidate_count
            * (3 * dimensions * fp32_bytes + 2 * int64_bytes),
            scoring_bytes=candidate_count * fp32_bytes,
            topk_bytes=candidate_count * int64_bytes,
            merge_bytes=candidate_count * int64_bytes,
        )

    def prepare_index(self, key, records, device):
        """Pack current compact records on ``device`` outside selection."""

        records = tuple(records)
        device = torch.device(device)
        if records:
            ordered = tuple(
                sorted(
                    records,
                    key=lambda item: item.logical_block_id.logical_block,
                )
            )
            logical = tuple(
                int(item.logical_block_id.logical_block) for item in ordered
            )
            if logical != tuple(range(len(ordered))):
                raise ValueError(
                    "tensorized Quest index requires contiguous logical pages"
                )
            dimensions = int(ordered[0].dimensions)
            for record in ordered:
                if not record.is_compact:
                    raise TypeError(
                        "tensorized Quest requires compact summary records"
                    )
                if record.index_version != record.data_version:
                    raise RuntimeError("Quest index is stale")
                if int(record.dimensions) != dimensions:
                    raise ValueError("Quest compact dimensions must match")
            host_summaries = torch.stack(
                tuple(item.compact_summary for item in ordered), dim=0
            ).contiguous()
            summaries = host_summaries.to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            logical_blocks = torch.arange(
                len(ordered), dtype=torch.int64, device=device
            )
            data_epochs = torch.tensor(
                tuple(int(item.data_version) for item in ordered),
                dtype=torch.int64,
                device=device,
            )
        else:
            dimensions = 0
            summaries = torch.empty((0, 3, 0), dtype=torch.float32, device=device)
            logical_blocks = torch.empty((0,), dtype=torch.int64, device=device)
            data_epochs = torch.empty((0,), dtype=torch.int64, device=device)
        self._indices[tuple(key)] = _TensorizedQuestIndex(
            summaries=summaries,
            logical_blocks=logical_blocks,
            data_epochs=data_epochs,
            dimensions=dimensions,
            record_count=len(records),
        )
        self.index_builds += 1

    def clone_index(self, source_key, target_key):
        current = self._indices.get(tuple(source_key))
        if current is not None:
            self._indices[tuple(target_key)] = current

    def replace_request(self, source_request_id, target_request_id):
        source_request_id = int(source_request_id)
        target_request_id = int(target_request_id)
        self.drop_request(target_request_id)
        for key in tuple(self._indices):
            if key[0] == source_request_id:
                self._indices[(target_request_id,) + key[1:]] = self._indices.pop(key)

    def drop_index(self, key):
        self._indices.pop(tuple(key), None)

    def drop_request(self, request_id):
        request_id = int(request_id)
        for key in tuple(self._indices):
            if key[0] == request_id:
                self._indices.pop(key, None)

    def select(
        self,
        key,
        query,
        *,
        scored_budget,
        recent_window,
        page_size,
        mode="budget",
    ):
        """Select packed positions without leaving ``query.device``."""

        index = self._indices.get(tuple(key))
        if index is None:
            raise RuntimeError(
                "tensorized Quest index was not prepared before selection"
            )
        if index.summaries.device != query.device:
            raise ValueError("Quest index and query must share a device")
        if index.record_count == 0:
            raise RuntimeError("tensorized Quest selection has no candidates")
        query_vector = query.float().mean(dim=(0, 1))
        if query_vector.shape != (index.dimensions,):
            raise ValueError(
                "query dimension {} does not match index dimension {}".format(
                    int(query_vector.shape[0]), index.dimensions
                )
            )
        candidate_count = index.record_count
        mode = str(mode).lower().replace("topk", "budget")
        if mode == "full":
            selected = index.logical_blocks
            resolved_scored_budget = candidate_count
            requested_scored_budget = candidate_count
            recent_budget = 0
        elif mode == "budget":
            requested_scored_budget = int(scored_budget)
            if requested_scored_budget < 0:
                raise ValueError("Quest scored budget must not be negative")
            resolved_scored_budget = min(requested_scored_budget, candidate_count)
            recent_budget = min(
                candidate_count,
                max(
                    1,
                    (int(recent_window) + int(page_size) - 1)
                    // int(page_size),
                )
                if int(recent_window)
                else 0,
            )
            scored_count = candidate_count - recent_budget
            resolved_scored_budget = min(resolved_scored_budget, scored_count)
            scored_summary = index.summaries[:scored_count]
            scores = torch.maximum(
                scored_summary[:, 0] * query_vector,
                scored_summary[:, 1] * query_vector,
            ).sum(dim=1)
            # Stable ordering makes equal scores prefer the earlier logical
            # page.  Both ranking and truncation remain on the query device.
            ranked = torch.argsort(
                scores, dim=0, descending=True, stable=True
            )[:resolved_scored_budget]
            recent = index.logical_blocks[scored_count:]
            selected = torch.sort(torch.cat((ranked, recent), dim=0)).values
        else:
            raise ValueError("unknown Quest selection mode {!r}".format(mode))
        selected_count = (
            candidate_count
            if mode == "full"
            else resolved_scored_budget + recent_budget
        )
        self.query_count += 1
        self.candidate_count += candidate_count
        self.selected_count += selected_count
        return TensorizedQuestSelectionResult(
            selected_positions=selected,
            mode=mode,
            candidate_count=candidate_count,
            selected_count=selected_count,
            scored_budget=requested_scored_budget,
            recent_budget=recent_budget,
            total_budget=(
                candidate_count
                if mode == "full"
                else requested_scored_budget + recent_budget
            ),
        )

    def stats(self):
        index_bytes = sum(
            item.summaries.numel() * item.summaries.element_size()
            + item.logical_blocks.numel() * item.logical_blocks.element_size()
            + item.data_epochs.numel() * item.data_epochs.element_size()
            for item in self._indices.values()
        )
        return {
            "provider": self.name,
            "qualification_status": self.qualification_status,
            "device_indices": len(self._indices),
            "index_bytes": int(index_bytes),
            "index_builds": int(self.index_builds),
            "queries": int(self.query_count),
            "candidates": int(self.candidate_count),
            "selected": int(self.selected_count),
            "selection_ratio": (
                self.selected_count / float(self.candidate_count)
                if self.candidate_count
                else 1.0
            ),
        }
