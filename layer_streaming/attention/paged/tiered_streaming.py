"""Exact softmax over a bounded selected tiered-KV working set.

The provider deliberately loads one selected layer page at a time.  It uses a
two-pass softmax (global maximum, then numerator/denominator) so the complete
KV sequence never has to be materialized on the compute device.  The caller's
``page_loader`` owns residency, migration fences, and the compute pin for the
duration of each yielded page.
"""

from contextlib import contextmanager
import time

import torch

from .abi import PagedAttentionOutput


def _output_dtype(name, fallback):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(str(name), fallback)


class TieredStreamingExactAttention:
    """Torch attention whose resident selected working set is one page.

    Dense Selection is exact over the complete context.  RGKV Selection is
    exact over its selected logical pages but intentionally sparse relative to
    the request.  Residency streaming must support both; quality qualification
    remains the Selection policy's responsibility.
    """

    name = "tiered_streaming_exact_torch"

    def execute(self, request, page_loader, page_metadata=None):
        batch = request.batch_view
        if batch.batch_size != 1:
            raise NotImplementedError(
                "tiered streaming exact attention currently supports batch 1"
            )
        page_count = int(request.flat_block_table.numel())
        if page_count <= 0:
            raise RuntimeError("tiered attention selection is empty")
        if page_metadata is not None and len(page_metadata) != page_count:
            raise ValueError("tiered page metadata does not align with selection")

        started = time.perf_counter()
        query = request.query.float()
        query_positions = batch.query_positions.to(
            device=query.device, dtype=torch.long
        )
        query_count, query_heads, head_dim = query.shape
        groups = int(request.num_query_heads) // int(request.num_kv_heads)
        global_max = torch.full(
            (query_count, query_heads),
            -float("inf"),
            dtype=torch.float32,
            device=query.device,
        )

        def scores_for(index, key):
            if page_metadata is None:
                logical = int(request.logical_block_ids[index].item())
                valid = int(request.page_valid_tokens[index].item())
            else:
                logical, valid = page_metadata[index]
                logical, valid = int(logical), int(valid)
            if valid <= 0 or valid > int(request.page_size):
                raise ValueError("selected tiered page has invalid valid_tokens")
            if tuple(key.shape) != (
                int(request.num_kv_heads),
                int(request.page_size),
                int(request.head_dim),
            ):
                raise ValueError("tiered key page has invalid HND geometry")
            expanded = key[:, :valid, :].float().repeat_interleave(groups, dim=0)
            scores = torch.einsum("qhd,htd->qht", query, expanded)
            scores.mul_(float(request.softmax_scale))
            if request.causal:
                key_positions = torch.arange(
                    logical * int(request.page_size),
                    logical * int(request.page_size) + valid,
                    dtype=torch.long,
                    device=query.device,
                )
                visible = key_positions.view(1, 1, -1) <= query_positions.view(
                    -1, 1, 1
                )
                scores.masked_fill_(~visible, -float("inf"))
            return scores, valid

        # Pass 1 establishes one exact normalization maximum across all waves.
        for index in range(page_count):
            with page_loader(index) as payload:
                key, _ = payload
                scores, _ = scores_for(index, key)
                global_max = torch.maximum(global_max, scores.amax(dim=-1))
        if not bool(torch.isfinite(global_max).all()):
            raise RuntimeError("one or more queries have no visible tiered KV page")

        denominator = torch.zeros_like(global_max)
        numerator = torch.zeros(
            (query_count, query_heads, head_dim),
            dtype=torch.float32,
            device=query.device,
        )
        # Pass 2 reloads bounded waves and accumulates the exact softmax sum.
        for index in range(page_count):
            with page_loader(index) as payload:
                key, value = payload
                scores, valid = scores_for(index, key)
                weights = torch.exp(scores - global_max.unsqueeze(-1))
                values = value[:, :valid, :].float().repeat_interleave(
                    groups, dim=0
                )
                denominator.add_(weights.sum(dim=-1))
                numerator.add_(torch.einsum("qht,htd->qhd", weights, values))

        output = (numerator / denominator.unsqueeze(-1)).to(
            _output_dtype(request.output_dtype, request.query.dtype)
        )
        lse = (
            torch.log(denominator) + global_max
            if request.return_logsumexp
            else None
        )
        element_size = query.element_size()
        workspace_bytes = (
            global_max.numel()
            + denominator.numel()
            + numerator.numel()
            + query_count * query_heads * int(request.page_size)
        ) * element_size
        return PagedAttentionOutput(
            output=output,
            logsumexp=lse,
            provider_metrics={
                "provider": self.name,
                "phase": "tiered_streaming",
                "wall_ms": (time.perf_counter() - started) * 1000.0,
                "workspace_bytes": int(workspace_bytes),
                "full_kv_workspace": False,
                "full_score_matrix": False,
                "resident_working_set_pages": 1,
                "streaming_passes": 2,
                "streamed_page_visits": 2 * page_count,
                "selection_exact": bool(request.selected_pages.exact),
            },
        )
