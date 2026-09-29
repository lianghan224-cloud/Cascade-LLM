"""Tensorized device-resident scoring and top-k for RGKV."""

from dataclasses import dataclass

import torch

from .rgkv_index import RGKVBudget, RGKVIndex, RGKVSelectionResult


@dataclass(frozen=True)
class RGKVWorkspaceEstimate:
    canonical_index_bytes: int
    packed_index_bytes: int
    scoring_bytes: int
    topk_bytes: int
    merge_bytes: int

    @property
    def index_bytes(self):
        return int(self.canonical_index_bytes + self.packed_index_bytes)

    @property
    def temporary_bytes(self):
        return int(self.scoring_bytes + self.topk_bytes + self.merge_bytes)

    @property
    def total_bytes(self):
        return int(self.index_bytes + self.temporary_bytes)

    def as_dict(self):
        return {
            "canonical_index_bytes": int(self.canonical_index_bytes),
            "packed_index_bytes": int(self.packed_index_bytes),
            "index_bytes": self.index_bytes,
            "scoring_bytes": int(self.scoring_bytes),
            "topk_bytes": int(self.topk_bytes),
            "merge_bytes": int(self.merge_bytes),
            "temporary_bytes": self.temporary_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass
class _DeviceIndex:
    summaries: torch.Tensor
    logical_pages: torch.Tensor
    data_epochs: torch.Tensor
    candidate_count: int
    dimensions: int
    capacity: int


class RGKVGPUScorer:
    """PyTorch tensor scorer whose selection hot path stays on the device."""

    name = "torch_tensorized"
    # Scoring/top-k itself is tensorized, but the current end-to-end runtime
    # still has host-visible SelectedPageView/Tier resolution.  Do not promote
    # the provider on the strength of this component alone.
    qualification_status = "cuda_smoke"

    def __init__(self):
        self._indices = {}
        self.index_builds = 0
        self.index_updates = 0
        self.query_count = 0
        self.candidate_count = 0
        self.selected_count = 0

    @staticmethod
    def estimate_workspace(candidate_count, dimensions):
        candidates = int(candidate_count)
        dimensions = int(dimensions)
        if candidates < 0 or dimensions < 0:
            raise ValueError("RGKV workspace geometry must not be negative")
        return RGKVWorkspaceEstimate(
            canonical_index_bytes=candidates * 3 * dimensions * 4,
            packed_index_bytes=candidates * (3 * dimensions * 4 + 2 * 8),
            scoring_bytes=candidates * 4,
            topk_bytes=candidates * 8,
            merge_bytes=candidates * 8,
        )

    @classmethod
    def estimate_index_bytes(cls, candidate_count, dimensions):
        return cls.estimate_workspace(candidate_count, dimensions).index_bytes

    @staticmethod
    def estimate_build_workspace_bytes(page_size, dimensions):
        return int(page_size) * int(dimensions) * 4

    def prepare_index(self, key, index, device=None):
        """Publish a packed index outside the decode selection call."""

        if not isinstance(index, RGKVIndex):
            raise TypeError("index must be RGKVIndex")
        pages = index.pages()
        if pages:
            source_device = pages[0].values.device
            device = source_device if device is None else torch.device(device)
            summaries = torch.stack(
                tuple(page.values for page in pages), dim=0
            ).contiguous().to(device=device, dtype=torch.float32, non_blocking=True)
            logical_pages = torch.arange(
                len(pages), dtype=torch.int64, device=device
            )
            data_epochs = torch.tensor(
                tuple(page.data_epoch for page in pages),
                dtype=torch.int64,
                device=device,
            )
            dimensions = int(summaries[0, 0].numel())
        else:
            device = torch.device("cpu") if device is None else torch.device(device)
            summaries = torch.empty((0, 3, 0), dtype=torch.float32, device=device)
            logical_pages = torch.empty((0,), dtype=torch.int64, device=device)
            data_epochs = torch.empty((0,), dtype=torch.int64, device=device)
            dimensions = 0
        self._indices[tuple(key)] = _DeviceIndex(
            summaries=summaries,
            logical_pages=logical_pages,
            data_epochs=data_epochs,
            candidate_count=len(pages),
            dimensions=dimensions,
            capacity=len(pages),
        )
        self.index_builds += 1

    def publish_summary(self, key, summary, *, capacity):
        """Publish exactly one changed summary into a preallocated index."""

        key = tuple(key)
        logical = int(summary.logical_page_id)
        values = summary.values.to(dtype=torch.float32, non_blocking=True)
        capacity = int(capacity)
        if capacity <= 0 or logical >= capacity:
            raise ValueError("RGKV device-index capacity is insufficient")
        current = self._indices.get(key)
        if current is None:
            summaries = torch.empty(
                (capacity,) + tuple(values.shape),
                dtype=torch.float32,
                device=values.device,
            )
            logical_pages = torch.arange(
                capacity, dtype=torch.int64, device=values.device
            )
            data_epochs = torch.empty(
                (capacity,), dtype=torch.int64, device=values.device
            )
            current = _DeviceIndex(
                summaries=summaries,
                logical_pages=logical_pages,
                data_epochs=data_epochs,
                candidate_count=0,
                dimensions=int(values[0].numel()),
                capacity=capacity,
            )
            self._indices[key] = current
            self.index_builds += 1
        if current.capacity != capacity or current.summaries.device != values.device:
            raise ValueError("RGKV device-index placement/capacity changed")
        if tuple(current.summaries.shape[1:]) != tuple(values.shape):
            raise ValueError("RGKV device-index summary geometry changed")
        if logical > current.candidate_count:
            raise RuntimeError("RGKV device-index publication is not contiguous")
        current.summaries[logical].copy_(values, non_blocking=True)
        current.data_epochs[logical] = int(summary.data_epoch)
        current.candidate_count = max(current.candidate_count, logical + 1)
        self.index_updates += 1

    def clone_index(self, source_key, target_key):
        current = self._indices.get(tuple(source_key))
        if current is not None:
            self._indices[tuple(target_key)] = _DeviceIndex(
                summaries=current.summaries.clone(),
                logical_pages=current.logical_pages.clone(),
                data_epochs=current.data_epochs.clone(),
                candidate_count=current.candidate_count,
                dimensions=current.dimensions,
                capacity=current.capacity,
            )

    def replace_request(self, source_request_id, target_request_id):
        source_request_id = int(source_request_id)
        target_request_id = int(target_request_id)
        self.drop_request(target_request_id)
        for key in tuple(self._indices):
            if key[0] == source_request_id:
                self._indices[(target_request_id,) + key[1:]] = self._indices.pop(key)

    def drop_index(self, key):
        self._indices.pop(tuple(key), None)

    def truncate_index(self, key, candidate_count):
        current = self._indices.get(tuple(key))
        if current is None:
            return
        candidate_count = int(candidate_count)
        if candidate_count < 0 or candidate_count > current.candidate_count:
            raise ValueError("invalid RGKV device-index truncation")
        current.candidate_count = candidate_count

    def drop_request(self, request_id):
        request_id = int(request_id)
        for key in tuple(self._indices):
            if key[0] == request_id:
                self._indices.pop(key, None)

    def select(
        self,
        key,
        query,
        budget,
        *,
        override_logical=None,
        override_summary=None,
        candidate_count=None,
    ):
        """Score, top-k, merge recent pages and restore logical order."""

        if not isinstance(budget, RGKVBudget):
            raise TypeError("budget must be RGKVBudget")
        index = self._indices.get(tuple(key))
        if index is None and override_summary is None:
            raise RuntimeError("RGKV device index was not prepared")
        if index is not None and index.summaries.device != query.device:
            raise ValueError("RGKV index and query must share a device")
        published_count = 0 if index is None else index.candidate_count
        candidate_count = (
            published_count
            if candidate_count is None
            else int(candidate_count)
        )
        selected_total, recent_count, relevance_count = budget.resolve(
            candidate_count
        )
        feature_shape = (
            tuple(index.summaries.shape[2:])
            if index is not None
            else tuple(override_summary.shape[1:])
        )
        query_float = query.float()
        if len(feature_shape) == 2 and query_float.ndim == 3:
            kv_heads, head_dim = feature_shape
            query_heads = int(query_float.shape[1])
            if (
                int(query_float.shape[2]) != head_dim
                or query_heads % kv_heads
            ):
                raise ValueError("RGKV query/KV head geometry is incompatible")
            query_vector = query_float.reshape(
                int(query_float.shape[0]),
                kv_heads,
                query_heads // kv_heads,
                head_dim,
            ).mean(dim=(0, 2))
        else:
            query_vector = query_float.mean(dim=tuple(range(query_float.ndim - 1)))
        expected_dimensions = (
            index.dimensions
            if index is not None
            else int(override_summary[0].numel())
        )
        if query_vector.numel() != expected_dimensions:
            raise ValueError("RGKV query geometry does not match index")
        query_vector = query_vector.reshape(feature_shape)
        summaries = (
            torch.empty(
                (0, 3) + feature_shape,
                dtype=torch.float32,
                device=query.device,
            )
            if index is None
            else index.summaries[:published_count]
        )
        scores = torch.maximum(
            summaries[:, 0] * query_vector,
            summaries[:, 1] * query_vector,
        ).flatten(1).sum(dim=1)
        if override_summary is not None:
            override_logical = int(override_logical)
            override = override_summary.to(
                device=query.device, dtype=torch.float32, non_blocking=True
            )
            override_score = torch.maximum(
                override[0] * query_vector,
                override[1] * query_vector,
            ).flatten().sum().reshape(1)
            if override_logical < published_count:
                scores = scores.clone()
                scores[override_logical] = override_score[0]
            elif override_logical == published_count:
                scores = torch.cat((scores, override_score), dim=0)
            else:
                raise RuntimeError("RGKV staged summary is not contiguous")
        if int(scores.shape[0]) != candidate_count:
            raise RuntimeError("RGKV candidate count does not match device index")
        scored_count = candidate_count - recent_count
        if relevance_count:
            ranked = torch.argsort(
                scores[:scored_count], descending=True, stable=True
            )[:relevance_count]
        else:
            ranked = torch.empty(
                (0,), dtype=torch.int64, device=query.device
            )
        recent = torch.arange(
            scored_count,
            candidate_count,
            dtype=torch.int64,
            device=query.device,
        )
        selected_positions = torch.sort(
            torch.cat((ranked, recent), dim=0)
        ).values
        selected_pages = selected_positions.contiguous()
        selected_scores = scores[selected_positions].contiguous()
        self.query_count += 1
        self.candidate_count += candidate_count
        self.selected_count += selected_total
        return RGKVSelectionResult(
            selected_logical_pages=selected_pages,
            selected_scores=selected_scores,
            candidate_count=candidate_count,
            selected_count=selected_total,
            total_page_budget=budget.total_page_budget,
            mandatory_recent_pages=recent_count,
            relevance_selected_pages=relevance_count,
        )

    def selected_data_epochs(
        self,
        key,
        selected_logical_pages,
        *,
        override_logical=None,
        override_epoch=None,
    ):
        """Gather selected summary epochs without host scalar extraction."""

        index = self._indices.get(tuple(key))
        if index is None:
            if override_logical != 0 or override_epoch is None:
                raise RuntimeError("RGKV device index has no selected epochs")
            epochs = torch.as_tensor(
                (int(override_epoch),),
                dtype=torch.int64,
                device=selected_logical_pages.device,
            )
        else:
            epochs = index.data_epochs[: index.candidate_count]
            if override_logical is not None:
                override_logical = int(override_logical)
                override_epoch_tensor = torch.as_tensor(
                    (int(override_epoch),),
                    dtype=torch.int64,
                    device=epochs.device,
                )
                if override_logical < int(epochs.shape[0]):
                    positions = torch.as_tensor(
                        (override_logical,),
                        dtype=torch.int64,
                        device=epochs.device,
                    )
                    epochs = epochs.clone().index_copy(
                        0, positions, override_epoch_tensor
                    )
                elif override_logical == int(epochs.shape[0]):
                    epochs = torch.cat((epochs, override_epoch_tensor), dim=0)
                else:
                    raise RuntimeError(
                        "RGKV staged epoch publication is not contiguous"
                    )
        return epochs[selected_logical_pages].contiguous()

    def candidate_data_epochs(
        self,
        key,
        candidate_count,
        *,
        override_logical=None,
        override_epoch=None,
        device=None,
    ):
        """Return every candidate epoch as a device tensor."""

        index = self._indices.get(tuple(key))
        if device is None:
            if index is None:
                raise RuntimeError("device is required without a published index")
            device = index.data_epochs.device
        logical_pages = torch.arange(
            int(candidate_count), dtype=torch.int64, device=device
        )
        return self.selected_data_epochs(
            key,
            logical_pages,
            override_logical=override_logical,
            override_epoch=override_epoch,
        )

    def stats(self):
        index_bytes = sum(
            item.summaries.numel() * item.summaries.element_size()
            + item.logical_pages.numel() * item.logical_pages.element_size()
            + item.data_epochs.numel() * item.data_epochs.element_size()
            for item in self._indices.values()
        )
        return {
            "provider": self.name,
            "qualification_status": self.qualification_status,
            "device_indices": len(self._indices),
            "index_bytes": int(index_bytes),
            "index_builds": int(self.index_builds),
            "index_updates": int(self.index_updates),
            "queries": int(self.query_count),
            "candidates": int(self.candidate_count),
            "selected": int(self.selected_count),
            "selection_ratio": (
                self.selected_count / float(self.candidate_count)
                if self.candidate_count
                else 1.0
            ),
            "scorer_cpu_sync_count": 0,
        }


def rgkv_scorer_provider(name):
    name = str(name)
    if name == "cpu_reference":
        from .rgkv_cpu_reference import RGKVCPUReferenceScorer

        return RGKVCPUReferenceScorer()
    if name in {"torch_tensorized", "rgkv_gpu"}:
        return RGKVGPUScorer()
    raise ValueError("unknown RGKV scorer {!r}".format(name))


__all__ = [
    "RGKVGPUScorer",
    "RGKVWorkspaceEstimate",
    "rgkv_scorer_provider",
]
