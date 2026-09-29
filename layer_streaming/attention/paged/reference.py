"""Explicit V1 reference providers.

`reference_paged_exact` is a two-pass page-wise FP32 implementation and never
gathers complete KV or constructs a complete score matrix.  The legacy gather
provider remains available only by its explicit diagnostic name.
"""

import math
import time

import torch
import torch.nn.functional as F

from ...kv.errors import KVUnsupportedError
from .abi import PagedAttentionOutput
from .base import PagedAttentionBackend
from .capability import PagedAttentionCapability
from .workspace import PagedWorkspaceEstimate


SDPA_ALLOCATOR_GUARD_MIN_BYTES = 256 * 1024
SDPA_ALLOCATOR_GUARD_NUMERATOR = 1
SDPA_ALLOCATOR_GUARD_DENOMINATOR = 4


def _requires_explicit_causal_mask(
    causal, full_prefill, simple_decode, return_logsumexp
):
    return bool(
        causal
        and (return_logsumexp or not (full_prefill or simple_decode))
    )


def _build_explicit_causal_mask(key_positions, query_positions):
    return key_positions.view(1, 1, 1, -1) <= query_positions.view(
        1, 1, -1, 1
    )


def _gather_sdpa_workspace_breakdown(
    *,
    batch_size,
    max_sequence_length,
    max_query_length,
    total_query_tokens,
    num_query_heads,
    num_kv_heads,
    head_dim,
    dtype_bytes,
    explicit_mask,
    return_logsumexp=False,
):
    """Conservative local peak contract for the gather-SDPA provider.

    The explicit terms mirror allocations in ``_execute``: concatenated K/V,
    GQA-expanded K/V, accumulated output, and the chunked-prefill position/mask
    tensors.  SDPA has backend-dependent opaque storage and the CUDA caching
    allocator rounds transient blocks.  An SM86 length-128 smoke measured
    153,088 bytes beyond the 917,504 explicit full-prefill bytes (16.7%).  The
    provider therefore reserves 25% with a 256 KiB floor.  This is a bounded,
    provider-local guard rather than the planner's unrelated global margin.
    """

    batch_size = int(batch_size)
    sequence = int(max_sequence_length)
    query = int(max_query_length)
    total_query = int(total_query_tokens)
    query_heads = int(num_query_heads)
    kv_heads = int(num_kv_heads)
    dim = int(head_dim)
    element = int(dtype_bytes)
    values = (
        batch_size,
        sequence,
        query,
        total_query,
        query_heads,
        kv_heads,
        dim,
        element,
    )
    if any(value <= 0 for value in values):
        raise ValueError("gather SDPA workspace geometry must be positive")
    # torch.cat materializes native KV before repeat_interleave expands heads.
    gather_kv = sequence * kv_heads * dim * element * 2
    expanded_kv = sequence * query_heads * dim * element * 2
    output = total_query * query_heads * dim * element
    positions = sequence * 8 * 2 if explicit_mask else 0
    mask = query * sequence if explicit_mask else 0
    score_matrix = (
        query * sequence * query_heads * 4
        if return_logsumexp
        else 0
    )
    logsumexp = total_query * query_heads * 4 if return_logsumexp else 0
    explicit = (
        gather_kv
        + expanded_kv
        + output
        + positions
        + mask
        + score_matrix
        + logsumexp
    )
    fractional_guard = (
        explicit * SDPA_ALLOCATOR_GUARD_NUMERATOR
        + SDPA_ALLOCATOR_GUARD_DENOMINATOR
        - 1
    ) // SDPA_ALLOCATOR_GUARD_DENOMINATOR
    guard = max(SDPA_ALLOCATOR_GUARD_MIN_BYTES, fractional_guard)
    return {
        "gather_kv_bytes": int(gather_kv),
        "gqa_expanded_kv_bytes": int(expanded_kv),
        "output_bytes": int(output),
        "positions_bytes": int(positions),
        "mask_bytes": int(mask),
        "score_matrix_bytes": int(score_matrix),
        "logsumexp_bytes": int(logsumexp),
        "explicit_bytes": int(explicit),
        "sdpa_allocator_guard_bytes": int(guard),
        "total_bytes": int(explicit + guard),
        "explicit_mask": bool(explicit_mask),
        "guard_fraction": (
            float(SDPA_ALLOCATOR_GUARD_NUMERATOR)
            / float(SDPA_ALLOCATOR_GUARD_DENOMINATOR)
        ),
        "guard_floor_bytes": int(SDPA_ALLOCATOR_GUARD_MIN_BYTES),
    }


def _output_dtype(name, fallback):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(str(name), fallback)


class ReferencePagedExactBackend(PagedAttentionBackend):
    name = "reference_paged_exact"
    is_reference = True

    def capability(self):
        return PagedAttentionCapability(
            provider_name=self.name,
            provider_version="1",
            provider_abi=1,
            architectures=("cpu", "sm75", "sm80", "sm86", "sm89", "sm90"),
            dtypes=("bf16", "fp16"),
            page_sizes=(16, 32),
            head_dims=(),
            supports_mha=True,
            supports_gqa=True,
            supports_mqa=True,
            supports_decode=True,
            supports_prefill=True,
            supports_ragged_batch=True,
            supports_partial_tail=True,
            supports_cuda_graph=False,
            numerical_contract_version=2,
            requires_full_kv_workspace=False,
            requires_full_score_matrix=False,
            qualification_status="numerically_qualified",
        )

    def estimate_workspace(self, request):
        del request
        return PagedWorkspaceEstimate(0, "page_local", False, False, False)

    def estimate_workspace_shape(self, shape):
        self.validate_shape(shape)
        return PagedWorkspaceEstimate(0, "page_local", False, False, False)

    def _execute(self, request, phase):
        request.validate()
        started = time.perf_counter()
        query = request.query
        batch = request.batch_view
        output = torch.empty_like(
            query,
            dtype=_output_dtype(request.output_dtype, query.dtype),
        )
        logsumexp = (
            torch.empty(
                (query.shape[0], query.shape[1]),
                dtype=torch.float32,
                device=query.device,
            )
            if request.return_logsumexp
            else None
        )
        groups = int(request.num_query_heads) // int(request.num_kv_heads)
        page_size = int(request.page_size)
        for batch_index in range(batch.batch_size):
            query_start = int(batch.query_indptr[batch_index].item())
            query_end = int(batch.query_indptr[batch_index + 1].item())
            block_start = int(request.block_table_indptr[batch_index].item())
            block_end = int(request.block_table_indptr[batch_index + 1].item())
            for query_index in range(query_start, query_end):
                query_position = int(batch.query_positions[query_index].item())
                for query_head in range(int(request.num_query_heads)):
                    kv_head = query_head // groups
                    query_vector = query[query_index, query_head].float()
                    global_max = torch.tensor(
                        -float("inf"), dtype=torch.float32, device=query.device
                    )
                    # Pass one computes the exact global normalization maximum
                    # while keeping only one fixed-size page score vector.
                    for block_index in range(block_start, block_end):
                        logical_block = int(
                            request.logical_block_ids[block_index].item()
                        )
                        page_id = int(request.flat_block_table[block_index].item())
                        valid = int(
                            request.page_valid_tokens[block_index].item()
                        )
                        if valid <= 0:
                            break
                        token_start = logical_block * page_size
                        visible = min(
                            valid,
                            query_position - token_start + 1,
                        ) if request.causal else valid
                        if visible <= 0:
                            continue
                        keys = request.key_pool_view[
                            page_id, kv_head, :visible, :
                        ].float()
                        scores = torch.mv(keys, query_vector) * float(
                            request.softmax_scale
                        )
                        global_max = torch.maximum(global_max, scores.max())
                    if not torch.isfinite(global_max):
                        raise RuntimeError("query has no visible KV page")
                    denominator = torch.zeros((), dtype=torch.float32, device=query.device)
                    numerator = torch.zeros(
                        (int(request.head_dim),),
                        dtype=torch.float32,
                        device=query.device,
                    )
                    # Pass two uses the fixed global maximum, eliminating
                    # page-order rescaling error from the old online backend.
                    for block_index in range(block_start, block_end):
                        logical_block = int(
                            request.logical_block_ids[block_index].item()
                        )
                        page_id = int(request.flat_block_table[block_index].item())
                        valid = int(
                            request.page_valid_tokens[block_index].item()
                        )
                        if valid <= 0:
                            break
                        token_start = logical_block * page_size
                        visible = min(
                            valid,
                            query_position - token_start + 1,
                        ) if request.causal else valid
                        if visible <= 0:
                            continue
                        keys = request.key_pool_view[
                            page_id, kv_head, :visible, :
                        ].float()
                        values = request.value_pool_view[
                            page_id, kv_head, :visible, :
                        ].float()
                        scores = torch.mv(keys, query_vector) * float(
                            request.softmax_scale
                        )
                        weights = torch.exp(scores - global_max)
                        denominator += weights.sum()
                        numerator += torch.mv(values.transpose(0, 1), weights)
                    output[query_index, query_head].copy_(
                        (numerator / denominator).to(output.dtype)
                    )
                    if logsumexp is not None:
                        logsumexp[query_index, query_head] = (
                            torch.log(denominator) + global_max
                        )
        return PagedAttentionOutput(
            output=output,
            logsumexp=logsumexp,
            provider_metrics={
                "provider": self.name,
                "phase": phase,
                "wall_ms": (time.perf_counter() - started) * 1000.0,
                "workspace_bytes": 0,
                "full_kv_workspace": False,
                "full_score_matrix": False,
            },
        )

    def decode(self, request):
        if request.batch_view.max_query_length != 1:
            raise ValueError("decode requires one query token per request")
        return self._execute(request, "decode")

    def prefill(self, request):
        return self._execute(request, "prefill")


class LegacyGatherSDPAReferenceBackend(PagedAttentionBackend):
    name = "legacy_gather_sdpa_reference"
    is_reference = True

    def capability(self):
        base = ReferencePagedExactBackend().capability()
        return PagedAttentionCapability(
            provider_name=self.name,
            provider_version="1",
            provider_abi=base.provider_abi,
            architectures=base.architectures,
            dtypes=base.dtypes,
            page_sizes=base.page_sizes,
            head_dims=base.head_dims,
            supports_mha=True,
            supports_gqa=True,
            supports_mqa=True,
            supports_decode=True,
            supports_prefill=True,
            supports_ragged_batch=True,
            supports_partial_tail=True,
            supports_cuda_graph=False,
            numerical_contract_version=1,
            requires_full_kv_workspace=True,
            requires_full_score_matrix=False,
            qualification_status="experimental",
        )

    def estimate_workspace(self, request):
        batch = request.batch_view
        query_lengths = (
            batch.query_indptr[1:] - batch.query_indptr[:-1]
        )
        sequence_lengths = batch.sequence_lengths
        mask_required = any(
            _requires_explicit_causal_mask(
                request.causal,
                int(query_length) == int(sequence_length),
                int(query_length) == 1,
                request.return_logsumexp,
            )
            for query_length, sequence_length in zip(
                query_lengths.tolist(), sequence_lengths.tolist()
            )
        )
        breakdown = _gather_sdpa_workspace_breakdown(
            batch_size=batch.batch_size,
            max_sequence_length=int(sequence_lengths.max().item()),
            max_query_length=int(query_lengths.max().item()),
            total_query_tokens=int(query_lengths.sum().item()),
            num_query_heads=request.num_query_heads,
            num_kv_heads=request.num_kv_heads,
            head_dim=request.head_dim,
            dtype_bytes=request.key_pool_view.element_size(),
            explicit_mask=mask_required,
            return_logsumexp=request.return_logsumexp,
        )
        return PagedWorkspaceEstimate(
            breakdown["total_bytes"],
            "full_layer_kv_sdpa_guarded",
            True,
            True,
            bool(request.return_logsumexp),
        )

    def estimate_workspace_shape(self, shape):
        self.validate_shape(shape)
        # PagedWorkspaceShape does not carry query length or phase.  Reserve
        # the worst representable query length and an explicit chunked mask;
        # runtime request estimates can omit it for true Full Prefill.
        breakdown = _gather_sdpa_workspace_breakdown(
            batch_size=shape.batch_size,
            max_sequence_length=shape.max_sequence_length,
            max_query_length=shape.max_sequence_length,
            total_query_tokens=(
                int(shape.batch_size) * int(shape.max_sequence_length)
            ),
            num_query_heads=shape.num_query_heads,
            num_kv_heads=shape.num_kv_heads,
            head_dim=shape.head_dim,
            dtype_bytes=shape.dtype_bytes,
            explicit_mask=True,
            return_logsumexp=False,
        )
        return PagedWorkspaceEstimate(
            breakdown["total_bytes"],
            "full_layer_kv_sdpa_guarded",
            True,
            True,
            False,
        )

    def _execute(self, request, phase):
        request.validate()
        outputs = []
        lse_outputs = []
        groups = request.num_query_heads // request.num_kv_heads
        batch = request.batch_view
        request_estimate = self.estimate_workspace(request)
        query_lengths = batch.query_indptr[1:] - batch.query_indptr[:-1]
        sequence_lengths = batch.sequence_lengths
        mask_required = any(
            _requires_explicit_causal_mask(
                request.causal,
                int(query_length) == int(sequence_length),
                int(query_length) == 1,
                request.return_logsumexp,
            )
            for query_length, sequence_length in zip(
                query_lengths.tolist(), sequence_lengths.tolist()
            )
        )
        workspace_breakdown = _gather_sdpa_workspace_breakdown(
            batch_size=batch.batch_size,
            max_sequence_length=int(sequence_lengths.max().item()),
            max_query_length=int(query_lengths.max().item()),
            total_query_tokens=int(query_lengths.sum().item()),
            num_query_heads=request.num_query_heads,
            num_kv_heads=request.num_kv_heads,
            head_dim=request.head_dim,
            dtype_bytes=request.key_pool_view.element_size(),
            explicit_mask=mask_required,
            return_logsumexp=request.return_logsumexp,
        )
        for batch_index in range(batch.batch_size):
            query_start = int(batch.query_indptr[batch_index].item())
            query_end = int(batch.query_indptr[batch_index + 1].item())
            block_start = int(request.block_table_indptr[batch_index].item())
            block_end = int(request.block_table_indptr[batch_index + 1].item())
            sequence_length = int(batch.sequence_lengths[batch_index].item())
            query_length = query_end - query_start
            full_prefill = query_length == sequence_length
            simple_decode = query_length == 1
            build_mask = _requires_explicit_causal_mask(
                request.causal,
                full_prefill,
                simple_decode,
                request.return_logsumexp,
            )
            keys = []
            values = []
            key_positions = [] if build_mask else None
            for block_index in range(block_start, block_end):
                logical = int(request.logical_block_ids[block_index].item())
                valid = int(
                    request.page_valid_tokens[block_index].item()
                )
                page_id = int(request.flat_block_table[block_index].item())
                keys.append(request.key_pool_view[page_id, :, :valid, :])
                values.append(request.value_pool_view[page_id, :, :valid, :])
                if key_positions is not None:
                    key_positions.append(
                        torch.arange(
                            logical * request.page_size,
                            logical * request.page_size + valid,
                            device=request.query.device,
                        )
                    )
            key = torch.cat(keys, dim=1).unsqueeze(0).repeat_interleave(groups, dim=1)
            value = torch.cat(values, dim=1).unsqueeze(0).repeat_interleave(groups, dim=1)
            query = request.query[query_start:query_end].transpose(0, 1).unsqueeze(0)
            positions = batch.query_positions[query_start:query_end]
            mask = (
                _build_explicit_causal_mask(
                    torch.cat(key_positions), positions
                )
                if key_positions is not None
                else None
            )
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=(
                    None if (not request.causal or full_prefill or simple_decode)
                    else mask
                ),
                dropout_p=0.0,
                is_causal=(request.causal and full_prefill and query_length > 1),
                scale=float(request.softmax_scale),
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
            if request.return_logsumexp:
                scores = torch.matmul(query.float(), key.float().transpose(-1, -2))
                scores *= float(request.softmax_scale)
                if request.causal:
                    scores.masked_fill_(~mask, -float("inf"))
                lse_outputs.append(torch.logsumexp(scores, dim=-1).squeeze(0).transpose(0, 1))
        return PagedAttentionOutput(
            output=torch.cat(outputs, dim=0),
            logsumexp=(torch.cat(lse_outputs, dim=0) if lse_outputs else None),
            provider_metrics={
                "provider": self.name,
                "phase": phase,
                "workspace_bytes": int(request_estimate.bytes),
                "workspace_breakdown": workspace_breakdown,
                "full_kv_workspace": True,
                "full_score_matrix": bool(request.return_logsumexp),
            },
        )

    def decode(self, request):
        return self._execute(request, "decode")

    def prefill(self, request):
        return self._execute(request, "prefill")


class GatherSDPAPrefillBackend(LegacyGatherSDPAReferenceBackend):
    """Explicit first-stage Prefill provider using gathered KV and SDPA.

    This provider is intentionally opt-in until real 70B memory and latency
    qualification completes.  Unlike the legacy diagnostic provider it does
    not advertise Decode and is not labeled as a numerical reference path.
    """

    name = "gather_sdpa_prefill"
    is_reference = False
    workload_kinds = ("full_prefill", "chunked_prefill")

    def capability(self):
        base = super().capability()
        return PagedAttentionCapability(
            provider_name=self.name,
            provider_version="1",
            provider_abi=base.provider_abi,
            architectures=base.architectures,
            dtypes=base.dtypes,
            page_sizes=base.page_sizes,
            head_dims=base.head_dims,
            supports_mha=True,
            supports_gqa=True,
            supports_mqa=True,
            supports_decode=False,
            supports_prefill=True,
            supports_ragged_batch=True,
            supports_partial_tail=True,
            supports_cuda_graph=False,
            numerical_contract_version=base.numerical_contract_version,
            requires_full_kv_workspace=True,
            requires_full_score_matrix=False,
            qualification_status="experimental",
        )

    def decode(self, request):
        del request
        raise KVUnsupportedError(
            "gather_sdpa_prefill does not implement decode"
        )


# Compatibility aliases for pre-RC imports.  These aliases expose attention
# methods only; append/copy moved to PagedKVKernelBackend.
