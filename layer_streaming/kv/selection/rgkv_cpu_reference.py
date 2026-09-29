"""Deterministic CPU correctness oracle for RGKV page selection."""

import torch

from .rgkv_index import RGKVBudget, RGKVIndex


class RGKVCPUReferenceScorer:
    """Clear, deterministic scorer used as the RGKV quality oracle.

    CPU conversion is deliberate here and must not be confused with the
    production decode provider.  The same hard total budget contract is used
    by both providers.
    """

    name = "cpu_reference"
    qualification_status = "logic_validated"

    def __init__(self):
        self.query_count = 0
        self.cpu_sync_count = 0

    @staticmethod
    def estimate_index_bytes(candidate_count, dimensions):
        return int(candidate_count) * 3 * int(dimensions) * 4

    @staticmethod
    def estimate_build_workspace_bytes(page_size, dimensions):
        return int(page_size) * int(dimensions) * 4

    @staticmethod
    def estimate_workspace(candidate_count, dimensions):
        del dimensions
        candidates = int(candidate_count)
        if candidates < 0:
            raise ValueError("RGKV candidate_count must not be negative")
        return {
            "index_bytes": 0,
            "scoring_bytes": candidates * 4,
            "topk_bytes": candidates * 8,
            "merge_bytes": candidates * 8,
        }

    def select(
        self,
        index,
        query,
        budget,
        *,
        page_epochs,
        logical_page_ids=None,
    ):
        if not isinstance(index, RGKVIndex):
            raise TypeError("index must be RGKVIndex")
        if not isinstance(budget, RGKVBudget):
            raise TypeError("budget must be RGKVBudget")
        if isinstance(query, torch.Tensor):
            if query.device.type != "cpu":
                self.cpu_sync_count += 1
            query = query.detach().float().cpu()
        else:
            query = torch.as_tensor(query, dtype=torch.float32)
        pages = index.pages()
        feature_shape = tuple(pages[0].values.shape[1:]) if pages else ()
        if len(feature_shape) == 2 and query.ndim == 3:
            kv_heads, head_dim = feature_shape
            query_heads = int(query.shape[1])
            if int(query.shape[2]) != head_dim or query_heads % kv_heads:
                raise ValueError("RGKV query/KV head geometry is incompatible")
            query = query.reshape(
                int(query.shape[0]),
                kv_heads,
                query_heads // kv_heads,
                head_dim,
            ).mean(dim=(0, 2))
        else:
            while query.ndim > len(feature_shape):
                query = query.mean(dim=0)
        result = index.select(
            query,
            budget,
            page_epochs=page_epochs,
            logical_page_ids=logical_page_ids,
        )
        self.query_count += 1
        return result

    def stats(self):
        return {
            "provider": self.name,
            "qualification_status": self.qualification_status,
            "queries": int(self.query_count),
            "rgkv_cpu_sync_count": int(self.cpu_sync_count),
        }


__all__ = ["RGKVCPUReferenceScorer"]
