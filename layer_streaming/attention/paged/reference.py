"""Explicit V1 reference providers.

`reference_paged_exact` is a two-pass page-wise FP32 implementation and never
gathers complete KV or constructs a complete score matrix.  The legacy gather
provider remains available only by its explicit diagnostic name.
"""

import math
import time

import torch
import torch.nn.functional as F

from .abi import PagedAttentionOutput
from .base import PagedAttentionProvider
from .capability import PagedAttentionCapability
from .workspace import PagedWorkspaceEstimate


def _output_dtype(name, fallback):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }.get(str(name), fallback)


class ReferencePagedExactProvider(PagedAttentionProvider):
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
            numerical_contract_version=1,
            requires_full_kv_workspace=False,
            requires_full_score_matrix=False,
            qualification_status="qualified_reference",
        )

    def estimate_workspace(self, request):
        del request
        return PagedWorkspaceEstimate(0, "page_local", False, False, False)

    def append_kv(self, append_input):
        append_input.validate()
        for token_index in range(append_input.slot_mapping.token_count):
            page_id = int(append_input.slot_mapping.page_ids[token_index].item())
            offset = int(append_input.slot_mapping.offsets[token_index].item())
            append_input.key_pool_view[page_id, :, offset, :].copy_(
                append_input.key[token_index]
            )
            append_input.value_pool_view[page_id, :, offset, :].copy_(
                append_input.value[token_index]
            )

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


class LegacyGatherSDPAReferenceProvider(PagedAttentionProvider):
    name = "legacy_gather_sdpa_reference"
    is_reference = True

    def capability(self):
        base = ReferencePagedExactProvider().capability()
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
            qualification_status="diagnostic_only",
        )

    def estimate_workspace(self, request):
        elements = (
            int(request.batch_view.sequence_lengths.max().item())
            * int(request.num_kv_heads)
            * int(request.head_dim)
            * 2
        )
        return PagedWorkspaceEstimate(
            elements * request.key_pool_view.element_size(),
            "full_layer_kv",
            True,
            True,
            False,
        )

    def append_kv(self, append_input):
        return ReferencePagedExactProvider().append_kv(append_input)

    def _execute(self, request, phase):
        request.validate()
        outputs = []
        lse_outputs = []
        groups = request.num_query_heads // request.num_kv_heads
        batch = request.batch_view
        workspace_peak = 0
        for batch_index in range(batch.batch_size):
            query_start = int(batch.query_indptr[batch_index].item())
            query_end = int(batch.query_indptr[batch_index + 1].item())
            block_start = int(request.block_table_indptr[batch_index].item())
            block_end = int(request.block_table_indptr[batch_index + 1].item())
            sequence_length = int(batch.sequence_lengths[batch_index].item())
            keys = []
            values = []
            key_positions = []
            for block_index in range(block_start, block_end):
                logical = int(request.logical_block_ids[block_index].item())
                valid = int(
                    request.page_valid_tokens[block_index].item()
                )
                page_id = int(request.flat_block_table[block_index].item())
                keys.append(request.key_pool_view[page_id, :, :valid, :])
                values.append(request.value_pool_view[page_id, :, :valid, :])
                key_positions.append(
                    torch.arange(
                        logical * request.page_size,
                        logical * request.page_size + valid,
                        device=request.query.device,
                    )
                )
            key = torch.cat(keys, dim=1).unsqueeze(0).repeat_interleave(groups, dim=1)
            value = torch.cat(values, dim=1).unsqueeze(0).repeat_interleave(groups, dim=1)
            workspace_peak = max(
                workspace_peak,
                (key.numel() + value.numel()) * key.element_size(),
            )
            query = request.query[query_start:query_end].transpose(0, 1).unsqueeze(0)
            positions = batch.query_positions[query_start:query_end]
            key_positions = torch.cat(key_positions)
            mask = key_positions.view(1, 1, 1, -1) <= positions.view(1, 1, -1, 1)
            query_length = query_end - query_start
            full_prefill = query_length == sequence_length
            simple_decode = query_length == 1
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=(
                    None
                    if (not request.causal or full_prefill or simple_decode)
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
                "workspace_bytes": workspace_peak,
                "full_kv_workspace": True,
                "full_score_matrix": bool(request.return_logsumexp),
            },
        )

    def decode(self, request):
        return self._execute(request, "decode")

    def prefill(self, request):
        return self._execute(request, "prefill")
