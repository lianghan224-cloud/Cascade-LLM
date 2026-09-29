"""Runtime policy for Relevance-Guided KV (RGKV).

RGKV owns selection metadata only.  Page lifetime remains in PagePool and
physical residency remains in the Location Plane.  Append publication is
two-phase so summaries receive the authoritative data epoch assigned by the
Ownership transaction instead of predicting it.
"""

from dataclasses import dataclass
import time

import torch

from .base import KVSelectionPolicyProvider, SelectionCapability
from .rgkv_cpu_reference import RGKVCPUReferenceScorer
from .rgkv_gpu import RGKVGPUScorer
from .rgkv_index import (
    RGKVBudget,
    RGKVIndex,
    RGKVPageSummary,
    RGKVStaleIndexError,
)
from ..device_metadata import (
    DEVICE_PAGE_STALE_EPOCH,
    DEVICE_PAGE_STALE_VALID_TOKENS,
)
from ..page_view import DeviceSelectedPageView, SelectedPageView


@dataclass(frozen=True)
class _StagedPageDelta:
    logical_page: int
    old_valid_tokens: int
    added_tokens: int
    values: torch.Tensor


def _summarize_append(keys):
    rows = keys.float()
    return torch.stack(
        (rows.amin(dim=0), rows.amax(dim=0), rows.mean(dim=0)), dim=0
    ).contiguous()


class _RuntimeRGKVIndex(RGKVIndex):
    """RGKVIndex extension that publishes an already compact append delta."""

    def publish_delta(self, delta, *, data_epoch, sealed):
        logical = int(delta.logical_page)
        current = self._pages.get(logical)
        if current is None:
            if delta.old_valid_tokens:
                raise RGKVStaleIndexError(
                    "STALE_INDEX missing prior summary for partial logical page {}"
                    .format(logical)
                )
            values = delta.values
            valid_tokens = int(delta.added_tokens)
            self.page_builds += 1
        else:
            if current.valid_tokens != int(delta.old_valid_tokens):
                raise RGKVStaleIndexError(
                    "STALE_INDEX logical page {} summary tokens {} != append offset {}"
                    .format(
                        logical, current.valid_tokens, delta.old_valid_tokens
                    )
                )
            self._require_new_epoch(current, data_epoch)
            added = delta.values.to(
                device=current.values.device,
                dtype=current.values.dtype,
                non_blocking=True,
            )
            old_count = int(current.valid_tokens)
            added_count = int(delta.added_tokens)
            combined_mean = (
                current.mean * float(old_count)
                + added[2] * float(added_count)
            ) / float(old_count + added_count)
            values = torch.stack(
                (
                    torch.minimum(current.minimum, added[0]),
                    torch.maximum(current.maximum, added[1]),
                    combined_mean,
                ),
                dim=0,
            ).contiguous()
            valid_tokens = old_count + added_count
            self.append_updates += 1
        summary = RGKVPageSummary(
            logical_page_id=logical,
            data_epoch=int(data_epoch),
            valid_tokens=valid_tokens,
            values=values,
            sealed=bool(sealed),
        )
        self._pages[logical] = summary
        return summary


class RGKVSelectionPolicy(KVSelectionPolicyProvider):
    """Incremental RGKV runtime adapter returning ``SelectedPageView``."""

    name = "rgkv"

    def __init__(self, budget, scorer=None, mode="budget"):
        if isinstance(budget, int):
            budget = RGKVBudget(budget)
        if not isinstance(budget, RGKVBudget):
            raise TypeError("RGKV policy budget must be RGKVBudget")
        self.budget = budget
        self.mode = str(mode)
        self.scorer = scorer or RGKVCPUReferenceScorer()
        if not isinstance(
            self.scorer, (RGKVCPUReferenceScorer, RGKVGPUScorer)
        ):
            raise TypeError("RGKV policy scorer is not an RGKV scorer")
        self.indexes = {}
        self.records = {}
        self._page_pool = None
        self._page_size = None
        self._device_page_table = None
        self.authority_validation_count = 0

    @property
    def scorer_name(self):
        return self.scorer.name

    def capability(self):
        return SelectionCapability(
            name=self.name,
            exact=False,
            implemented=True,
            requires_index=True,
        )

    def estimate_selection_workspace(self, candidate_count, dimensions):
        return self.scorer.estimate_workspace(candidate_count, dimensions)

    def estimate_index_bytes(self, candidate_count, dimensions):
        return self.scorer.estimate_index_bytes(candidate_count, dimensions)

    def stats(self):
        canonical_bytes = sum(
            index.stats()["index_bytes"] for index in self.indexes.values()
        )
        scorer_stats = dict(self.scorer.stats())
        device_bytes = int(scorer_stats.get("index_bytes", 0))
        return {
            "provider": self.name,
            "canonical_index_bytes": int(canonical_bytes),
            "device_index_bytes": device_bytes,
            "index_bytes": int(canonical_bytes + device_bytes),
            "authority_validation_count": int(
                self.authority_validation_count
            ),
            "scorer": scorer_stats,
        }

    def _index(self, request_id, layer, create=False):
        key = (int(request_id), int(layer))
        if create:
            return self.indexes.setdefault(key, _RuntimeRGKVIndex())
        try:
            return self.indexes[key]
        except KeyError:
            raise RGKVStaleIndexError(
                "STALE_INDEX missing request={} layer={} RGKV index".format(
                    key[0], key[1]
                )
            )

    @staticmethod
    def _transaction_staging(transaction, create=False):
        staging = getattr(transaction, "_rgkv_staged_layers", None)
        if staging is None and create:
            staging = {}
            setattr(transaction, "_rgkv_staged_layers", staging)
        return staging

    def stage_append_layer(
        self,
        runtime,
        state,
        layer,
        key_tokens,
        transaction,
        query_lengths=None,
    ):
        """Compact only this append's K tensor; do not read historical KV."""

        self._page_pool = runtime.page_pool
        self._page_size = int(runtime.page_size)
        self._device_page_table = getattr(
            getattr(runtime, "store", None), "device_page_table", None
        )
        layer = int(layer)
        if transaction is not state.pending_append:
            raise RuntimeError("RGKV append staging requires the active transaction")
        if layer in transaction.completed_layers:
            raise RuntimeError("RGKV append layer was staged after completion")
        if not isinstance(key_tokens, torch.Tensor) or key_tokens.ndim != 3:
            raise ValueError(
                "RGKV append keys must be [tokens, kv_heads, head_dim]"
            )
        if int(key_tokens.shape[0]) != int(transaction.token_count):
            raise ValueError("RGKV append keys must contain this request only")
        if query_lengths is not None:
            lengths = tuple(int(item) for item in query_lengths)
            if lengths not in {(), (int(transaction.token_count),)}:
                raise ValueError(
                    "RGKV stage_append_layer requires request-local query_lengths"
                )
        staging = self._transaction_staging(transaction, create=True)
        if layer in staging:
            raise RuntimeError("RGKV append layer was staged twice")
        offset = 0
        deltas = []
        position = int(transaction.start)
        while position < int(transaction.end):
            logical = position // int(runtime.page_size)
            page_offset = position % int(runtime.page_size)
            take = min(
                int(runtime.page_size) - page_offset,
                int(transaction.end) - position,
            )
            page_keys = key_tokens[offset : offset + take]
            values = _summarize_append(page_keys)
            if isinstance(self.scorer, RGKVCPUReferenceScorer):
                # The reference is intentionally host-resident and may block;
                # it is a correctness oracle, never the production hot path.
                if values.device.type != "cpu":
                    self.scorer.cpu_sync_count += 1
                values = values.cpu()
            deltas.append(
                _StagedPageDelta(
                    logical_page=logical,
                    old_valid_tokens=page_offset,
                    added_tokens=take,
                    values=values,
                )
            )
            position += take
            offset += take
        staging[layer] = tuple(deltas)
        return tuple(deltas)

    def _staged_index(self, state, layer):
        """Build a transaction-local read view without publishing an epoch.

        Decode attends after each layer append, while PagePool can publish the
        append's authoritative data epoch only after every layer completes.
        This overlay combines immutable published summaries with this layer's
        compact staged deltas.  It never attaches metadata to a PageHandle and
        is discarded after selection.
        """

        transaction = state.pending_append
        if self._page_size is None:
            raise RuntimeError("RGKV policy has no runtime page geometry")
        staging = self._transaction_staging(transaction)
        if transaction is None or staging is None or int(layer) not in staging:
            raise RGKVStaleIndexError(
                "STALE_INDEX current RGKV layer has no staged append summary"
            )
        published = self.indexes.get((int(state.request_id), int(layer)))
        overlay = _RuntimeRGKVIndex()
        if published is not None:
            overlay._pages.update(published._pages)
        staged_logicals = set()
        for delta in staging[int(layer)]:
            current = overlay._pages.get(int(delta.logical_page))
            synthetic_epoch = 0 if current is None else current.data_epoch + 1
            valid = min(
                self._page_size,
                int(transaction.end)
                - int(delta.logical_page) * self._page_size,
            )
            overlay.publish_delta(
                delta,
                data_epoch=synthetic_epoch,
                sealed=(valid == self._page_size),
            )
            staged_logicals.add(int(delta.logical_page))
        return overlay, frozenset(staged_logicals)

    # Temporary spelling accepted by early integrators.  The explicit public
    # contract is stage_append_layer + commit_append.
    update_append_layer = stage_append_layer

    def commit_append(self, runtime, state, transaction):
        """Publish staged summaries at PagePool's authoritative data epoch."""

        self._page_pool = runtime.page_pool
        staging = self._transaction_staging(transaction)
        expected_layers = set(range(int(runtime.layer_count)))
        if staging is None or set(staging) != expected_layers:
            raise RuntimeError("RGKV append summary transaction is incomplete")
        if int(state.sequence_length) != int(transaction.end):
            raise RuntimeError("RGKV summaries may publish only after KV commit")
        published = []
        page_summaries = {}
        for layer in sorted(staging):
            index = self._index(state.request_id, layer, create=True)
            for delta in staging[layer]:
                handle = state.block_table.handles[delta.logical_page]
                descriptor = runtime.page_pool.descriptor(handle)
                valid = min(
                    int(runtime.page_size),
                    int(state.sequence_length)
                    - delta.logical_page * int(runtime.page_size),
                )
                summary = index.publish_delta(
                    delta,
                    data_epoch=descriptor.data_version,
                    sealed=(valid == int(runtime.page_size)),
                )
                if summary.valid_tokens != valid:
                    raise RGKVStaleIndexError(
                        "STALE_INDEX RGKV valid token count diverged from page"
                    )
                self.records[(state.request_id, layer, delta.logical_page)] = summary
                published.append(summary)
                page_summaries.setdefault(delta.logical_page, {})[layer] = summary
                if isinstance(self.scorer, RGKVGPUScorer):
                    self.scorer.publish_summary(
                        (state.request_id, layer),
                        summary,
                        capacity=runtime.page_count,
                    )
        for logical, summaries in page_summaries.items():
            if set(summaries) != expected_layers:
                raise RuntimeError("RGKV page summary bundle is incomplete")
            handle = state.block_table.handles[logical]
            descriptor = runtime.page_pool.descriptor(handle)
            runtime.page_pool.attach_index(
                handle,
                tuple(summaries[layer] for layer in sorted(summaries)),
                version=descriptor.data_version,
            )
        self.abort_append(state, transaction)
        return tuple(published)

    def abort_append(self, state, transaction):
        del state
        if hasattr(transaction, "_rgkv_staged_layers"):
            delattr(transaction, "_rgkv_staged_layers")

    def fork_request(self, parent, child):
        for key, source in tuple(self.indexes.items()):
            if key[0] != int(parent.request_id):
                continue
            target = _RuntimeRGKVIndex()
            target._pages.update(
                (logical, summary) for logical, summary in source._pages.items()
            )
            self.indexes[(int(child.request_id), key[1])] = target
            for logical, summary in target._pages.items():
                self.records[(child.request_id, key[1], logical)] = summary
            if isinstance(self.scorer, RGKVGPUScorer):
                self.scorer.clone_index(key, (child.request_id, key[1]))

    def release_request(self, state):
        request_id = int(state.request_id)
        for key in tuple(self.indexes):
            if key[0] == request_id:
                self.indexes.pop(key, None)
        for key in tuple(self.records):
            if key[0] == request_id:
                self.records.pop(key, None)
        if isinstance(self.scorer, RGKVGPUScorer):
            self.scorer.drop_request(request_id)

    def commit_branch(self, parent, branch):
        self.release_request(parent)
        branch_id = int(branch.request_id)
        parent_id = int(parent.request_id)
        for key in tuple(self.indexes):
            if key[0] == branch_id:
                self.indexes[(parent_id, key[1])] = self.indexes.pop(key)
        for key in tuple(self.records):
            if key[0] == branch_id:
                self.records[(parent_id, key[1], key[2])] = self.records.pop(key)
        if isinstance(self.scorer, RGKVGPUScorer):
            self.scorer.replace_request(branch_id, parent_id)

    def reuse_request_from_page_metadata(self, runtime, state):
        """Share immutable sealed summaries from authoritative prefix pages."""

        request_id = int(state.request_id)
        try:
            for logical, handle in enumerate(state.block_table.handles):
                descriptor = runtime.page_pool.descriptor(handle)
                bundle = descriptor.index_metadata_handle
                if (
                    descriptor.index_version != descriptor.data_version
                    or not isinstance(bundle, tuple)
                    or len(bundle) != int(runtime.layer_count)
                ):
                    raise RGKVStaleIndexError(
                        "STALE_INDEX prefix page has no current RGKV layer bundle"
                    )
                for layer, summary in enumerate(bundle):
                    if (
                        not isinstance(summary, RGKVPageSummary)
                        or summary.logical_page_id != logical
                        or summary.data_epoch != descriptor.data_version
                        or not summary.sealed
                    ):
                        raise RGKVStaleIndexError(
                            "STALE_INDEX prefix RGKV summary is invalid"
                        )
                    index = self._index(request_id, layer, create=True)
                    index._pages[logical] = summary
                    index.prefix_shares += 1
                    self.records[(request_id, layer, logical)] = summary
                    if isinstance(self.scorer, RGKVGPUScorer):
                        self.scorer.publish_summary(
                            (request_id, layer),
                            summary,
                            capacity=runtime.page_count,
                        )
        except BaseException:
            self.release_request(state)
            raise

    def rollback_request(self, runtime, state, changed_block=None):
        """Drop removed summaries and rebuild only a changed tail page."""

        started = time.perf_counter()
        request_id = int(state.request_id)
        required = len(state.block_table.handles)
        rebuilt_bundle = {}
        for layer in range(int(runtime.layer_count)):
            key = (request_id, layer)
            index = self.indexes.get(key)
            if index is None:
                raise RGKVStaleIndexError(
                    "STALE_INDEX missing RGKV index during rollback"
                )
            for logical in tuple(index._pages):
                if logical >= required:
                    index.drop(logical)
                    self.records.pop((request_id, layer, logical), None)
            if isinstance(self.scorer, RGKVGPUScorer):
                self.scorer.truncate_index(key, required)
            if changed_block is not None:
                logical = int(changed_block)
                handle = state.block_table.handles[logical]
                descriptor = runtime.page_pool.descriptor(handle)
                valid = min(
                    int(runtime.page_size),
                    int(state.sequence_length)
                    - logical * int(runtime.page_size),
                )
                keys, _ = runtime.store.read_pages(layer, (handle.page_id,))
                rows = keys[0, :, :valid, :].transpose(0, 1).contiguous()
                current = index.get(logical)
                if current.sealed:
                    index.drop(logical)
                    summary = index.build_page(
                        logical,
                        rows,
                        data_epoch=descriptor.data_version,
                        valid_tokens=valid,
                        sealed=False,
                    )
                    index.rollback_rebuilds += 1
                else:
                    summary = index.rollback_tail(
                        logical,
                        rows,
                        data_epoch=descriptor.data_version,
                        valid_tokens=valid,
                    )
                self.records[(request_id, layer, logical)] = summary
                rebuilt_bundle[layer] = summary
                if isinstance(self.scorer, RGKVGPUScorer):
                    self.scorer.publish_summary(
                        key, summary, capacity=runtime.page_count
                    )
            if isinstance(self.scorer, RGKVGPUScorer) and not len(index):
                self.scorer.drop_index(key)
        if changed_block is not None:
            if len(rebuilt_bundle) != int(runtime.layer_count):
                raise RuntimeError("RGKV rollback summary bundle is incomplete")
            handle = state.block_table.handles[int(changed_block)]
            descriptor = runtime.page_pool.descriptor(handle)
            runtime.page_pool.attach_index(
                handle,
                tuple(
                    rebuilt_bundle[layer]
                    for layer in range(int(runtime.layer_count))
                ),
                version=descriptor.data_version,
            )
        runtime._metrics.rgkv_update_ms += (
            time.perf_counter() - started
        ) * 1000.0

    def select(self, requests, layer, query, batch_view):
        requests = tuple(requests)
        if int(query.shape[0]) != len(requests):
            return self._select_exact_prefill(batch_view)
        if len(requests) != 1 or int(batch_view.batch_size) != 1:
            raise NotImplementedError("RGKV decode currently supports batch one")
        state = requests[0]
        candidate_count = int(batch_view.flat_block_table.shape[0])
        if state.pending_append is not None:
            index, staged_logicals = self._staged_index(
                state,
                layer,
            )
        else:
            index = self._index(state.request_id, layer)
            staged_logicals = frozenset()
        if len(index) != candidate_count:
            raise RGKVStaleIndexError(
                "STALE_INDEX RGKV index does not match block table"
            )
        if self._page_pool is None:
            raise RuntimeError("RGKV policy has no PagePool authority")
        scorer_key = None
        override_logical = None
        override_summary = None
        override_epoch = None
        if isinstance(self.scorer, RGKVGPUScorer):
            scorer_key = (state.request_id, int(layer))
            if staged_logicals:
                if len(staged_logicals) != 1:
                    raise NotImplementedError(
                        "RGKV Decode staged selection supports one changed tail page"
                    )
                override_logical = next(iter(staged_logicals))
                staged_summary = index.get(override_logical)
                override_summary = staged_summary.values
                override_epoch = int(staged_summary.data_epoch)
        page_epochs = {}
        if (
            isinstance(self.scorer, RGKVGPUScorer)
            and self._device_page_table is not None
        ):
            if batch_view.flat_page_generations is None:
                raise RuntimeError(
                    "RGKV device selection requires expected Page generations"
                )
            expected_candidate_epochs = self.scorer.candidate_data_epochs(
                scorer_key,
                candidate_count,
                override_logical=override_logical,
                override_epoch=override_epoch,
                device=batch_view.flat_block_table.device,
            )
            candidate_metadata = self._device_page_table.gather(
                batch_view.flat_logical_block_ids,
                batch_view.flat_block_table,
                expected_epochs=expected_candidate_epochs,
                expected_generations=batch_view.flat_page_generations,
                expected_valid_tokens=batch_view.flat_page_valid_tokens,
            )
            if override_logical is not None:
                staged_candidate_mask = (
                    batch_view.flat_logical_block_ids == int(override_logical)
                )
                candidate_errors = torch.where(
                    staged_candidate_mask,
                    candidate_metadata.error_states
                    & ~(
                        DEVICE_PAGE_STALE_EPOCH
                        | DEVICE_PAGE_STALE_VALID_TOKENS
                    ),
                    candidate_metadata.error_states,
                )
            else:
                candidate_errors = candidate_metadata.error_states
            # CUDA implements this as a device-side asynchronous assertion;
            # it does not read the result scalar back into Python.  CPU logic
            # tests fail immediately at the same contract point.
            torch._assert_async(
                torch.all(candidate_errors == 0),
                "STALE_INDEX device KV page metadata validation failed",
            )
        else:
            for logical in range(candidate_count):
                if logical in staged_logicals:
                    page_epochs[logical] = index.get(logical).data_epoch
                    continue
                descriptor = self._page_pool.descriptor(
                    state.block_table.handles[logical]
                )
                page_epochs[logical] = int(descriptor.data_version)
                index.get(logical, expected_data_epoch=descriptor.data_version)
            self.authority_validation_count += candidate_count
        if self.mode == "full":
            budget = RGKVBudget(candidate_count, recent_pages=0)
        else:
            budget = self.budget
        if isinstance(self.scorer, RGKVGPUScorer):
            result = self.scorer.select(
                scorer_key,
                query,
                budget,
                override_logical=override_logical,
                override_summary=override_summary,
                candidate_count=candidate_count,
            )
            indices = result.selected_logical_pages
        else:
            result = self.scorer.select(
                index,
                query,
                budget,
                page_epochs=page_epochs,
            )
            indices = result.selected_logical_pages.to(
                device=batch_view.flat_block_table.device
            )
        device = batch_view.flat_block_table.device
        metadata = {
            "mode": self.mode,
            "candidate_count": result.candidate_count,
            "selected_count": result.selected_count,
            "total_page_budget": result.total_page_budget,
            "mandatory_recent_pages": result.mandatory_recent_pages,
            "relevance_selected_pages": result.relevance_selected_pages,
            "scorer_provider": self.scorer.name,
            "index_stats": index.stats(),
            "scorer_stats": self.scorer.stats(),
        }
        flat_page_ids = batch_view.flat_block_table[indices]
        logical_block_ids = batch_view.flat_logical_block_ids[indices]
        page_valid_tokens = batch_view.flat_page_valid_tokens[indices]
        common = dict(
            flat_page_ids=flat_page_ids,
            block_table_indptr=torch.tensor(
                (0, result.selected_count), dtype=torch.int32, device=device
            ),
            logical_block_ids=logical_block_ids,
            page_valid_tokens=page_valid_tokens,
            selection_name=self.name,
            exact=result.selected_count == result.candidate_count,
            metadata={"requests": (metadata,), "index_stats": index.stats()},
        )
        if (
            isinstance(self.scorer, RGKVGPUScorer)
            and self._device_page_table is not None
        ):
            expected_epochs = self.scorer.selected_data_epochs(
                scorer_key,
                indices,
                override_logical=override_logical,
                override_epoch=override_epoch,
            )
            gathered = self._device_page_table.gather(
                logical_block_ids,
                flat_page_ids,
                expected_epochs=expected_epochs,
                expected_generations=batch_view.flat_page_generations[indices],
                expected_valid_tokens=page_valid_tokens,
            )
            staged_mask = (
                torch.zeros_like(logical_block_ids, dtype=torch.bool)
                if override_logical is None
                else logical_block_ids == int(override_logical)
            )
            # A staged tail has no globally committed Page epoch yet.  Its
            # safety comes from the layer Append Fence, so only that selected
            # row is exempt from the committed-epoch comparison.
            error_states = torch.where(
                staged_mask,
                gathered.error_states
                & ~(
                    DEVICE_PAGE_STALE_EPOCH
                    | DEVICE_PAGE_STALE_VALID_TOKENS
                ),
                gathered.error_states,
            )
            expected_generations = batch_view.flat_page_generations[
                indices
            ].contiguous()
            valid_mask = (error_states == 0).contiguous()
            torch._assert_async(
                torch.all(valid_mask),
                "STALE_INDEX selected device KV metadata validation failed",
            )
            return DeviceSelectedPageView(
                **common,
                gpu_physical_slots=gathered.physical_gpu_slots,
                expected_epochs=expected_epochs,
                current_epochs=gathered.data_epochs,
                expected_generations=expected_generations,
                current_generations=gathered.generations,
                location_flags=gathered.location_states,
                valid_mask=valid_mask,
                error_mask=error_states.contiguous(),
                selection_count=result.selected_count,
                staged_tail_mask=staged_mask,
            )
        return SelectedPageView(**common)

    def _select_exact_prefill(self, batch_view):
        requests = []
        for request_index in range(int(batch_view.batch_size)):
            start = int(batch_view.block_table_indptr[request_index].item())
            end = int(batch_view.block_table_indptr[request_index + 1].item())
            requests.append(
                {
                    "mode": "full",
                    "candidate_count": end - start,
                    "selected_count": end - start,
                    "scorer_provider": self.scorer.name,
                    "fallback_reason": "rgkv_selection_is_decode_only",
                }
            )
        return SelectedPageView(
            flat_page_ids=batch_view.flat_block_table,
            block_table_indptr=batch_view.block_table_indptr,
            logical_block_ids=batch_view.flat_logical_block_ids,
            page_valid_tokens=batch_view.flat_page_valid_tokens,
            selection_name=self.name,
            exact=True,
            metadata={"requests": tuple(requests)},
        )


__all__ = ["RGKVSelectionPolicy"]
