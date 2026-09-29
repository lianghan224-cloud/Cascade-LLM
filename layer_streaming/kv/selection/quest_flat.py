"""Runtime adapter over the deterministic Quest CPU reference index."""

from .base import KVSelectionPolicyProvider, SelectionCapability
from .quest_cpu import LogicalKVBlockId, QuestCPUIndex
from ..page_view import SelectedPageView


def _request_key(request_id, layer, logical_block):
    return (int(request_id), int(layer), int(logical_block))


class QuestFlatSelection(KVSelectionPolicyProvider):
    name = "quest_flat"

    def __init__(self, budget=0, recent_window=0, mode="budget", scorer=None):
        self.index = QuestCPUIndex()
        self.budget = int(budget)
        self.recent_window = int(recent_window)
        self.mode = str(mode)
        # CPU reference remains the default.  Tensorized/GPU selection is
        # enabled only by passing an explicit scorer provider.
        self.scorer = scorer
        self.records = {}

    @property
    def scorer_name(self):
        return "cpu_reference" if self.scorer is None else self.scorer.name

    def estimate_selection_workspace(self, candidate_count, dimensions):
        if self.scorer is None:
            return None
        return self.scorer.estimate_workspace(candidate_count, dimensions)

    def capability(self):
        return SelectionCapability(
            name=self.name,
            exact=False,
            implemented=True,
            requires_index=True,
        )

    def build_request_layer(
        self,
        runtime,
        state,
        layer,
        version=None,
        changed_blocks=None,
    ):
        layer = int(layer)
        changed_blocks = (
            None
            if changed_blocks is None
            else {int(item) for item in changed_blocks}
        )
        sequence_length = int(state.layer_lengths[layer])
        required = (sequence_length + runtime.page_size - 1) // runtime.page_size
        if not required:
            if self.scorer is not None:
                self.scorer.drop_index((state.request_id, layer))
            return ()
        rebuild = []
        for logical in range(required):
            key = _request_key(state.request_id, layer, logical)
            if (
                changed_blocks is None
                or logical in changed_blocks
                or key not in self.records
            ):
                rebuild.append(logical)
        keys_by_logical = {}
        if rebuild:
            page_ids = [
                state.block_table.handles[logical].page_id
                for logical in rebuild
            ]
            keys, _ = runtime.store.read_pages(layer, page_ids)
            keys_by_logical = {
                logical: keys[offset]
                for offset, logical in enumerate(rebuild)
            }
        built = []
        for logical in range(required):
            key = _request_key(state.request_id, layer, logical)
            if logical not in keys_by_logical:
                built.append(self.records[key])
                continue
            valid = min(
                runtime.page_size,
                sequence_length - logical * runtime.page_size,
            )
            rows = keys_by_logical[logical][:, :valid, :].float().mean(dim=0)
            block_id = LogicalKVBlockId(
                model_id="runtime",
                session_id=state.reuse_namespace,
                branch_id=str(state.request_id),
                layer=layer,
                logical_block=logical,
            )
            descriptor = runtime.page_pool.descriptor(
                state.block_table.handles[logical]
            )
            record_version = int(
                descriptor.data_version if version is None else version
            )
            record = self.index.build_compact(
                rows,
                {
                    "logical_block_id": block_id,
                    "token_start": logical * runtime.page_size,
                    "data_version": record_version,
                },
            )
            previous = self.records.get(key)
            self.records[key] = record
            if previous is not None:
                self.index.release_ref(previous)
            if descriptor.data_version != record_version:
                runtime.page_pool.mark_data_updated(
                    state.block_table.handles[logical],
                    version=record_version,
                )
            runtime.page_pool.attach_index(
                state.block_table.handles[logical],
                record.record_id.value,
                version=record_version,
            )
            built.append(record)
        if self.scorer is not None:
            self.scorer.prepare_index(
                (state.request_id, layer), built, runtime.device
            )
        return tuple(built)

    def fork_request(self, parent, child):
        layers = set()
        for key, record in tuple(self.records.items()):
            if key[0] != int(parent.request_id):
                continue
            layers.add(key[1])
            child_key = _request_key(child.request_id, key[1], key[2])
            self.records[child_key] = self.index.fork_ref(record)
        if self.scorer is not None:
            for layer in layers:
                self.scorer.clone_index(
                    (parent.request_id, layer), (child.request_id, layer)
                )

    def release_request(self, state):
        if self.scorer is not None:
            self.scorer.drop_request(state.request_id)
        for key in tuple(self.records):
            if key[0] == int(state.request_id):
                self.index.release_ref(self.records.pop(key))

    def commit_branch(self, parent, branch):
        self.release_request(parent)
        branch_records = [
            (key, record)
            for key, record in tuple(self.records.items())
            if key[0] == int(branch.request_id)
        ]
        for key, record in branch_records:
            logical = record.logical_block_id
            parent_logical = LogicalKVBlockId(
                model_id=logical.model_id,
                session_id=logical.session_id,
                branch_id=str(parent.request_id),
                layer=logical.layer,
                logical_block=logical.logical_block,
            )
            self.records[
                _request_key(parent.request_id, key[1], key[2])
            ] = self.index.cow_clone(
                record, logical_block_id=parent_logical
            )
            self.records.pop(key, None)
            self.index.release_ref(record)
        if self.scorer is not None:
            self.scorer.replace_request(branch.request_id, parent.request_id)

    def rollback_request(self, runtime, state, changed_block=None):
        for layer in range(runtime.layer_count):
            if state.layer_lengths[layer] and changed_block is not None:
                self.build_request_layer(
                    runtime,
                    state,
                    layer,
                    changed_blocks=(changed_block,),
                )
        required_by_layer = {
            layer: (state.layer_lengths[layer] + runtime.page_size - 1)
            // runtime.page_size
            for layer in range(runtime.layer_count)
        }
        for key in tuple(self.records):
            if (
                key[0] == int(state.request_id)
                and key[2] >= required_by_layer[key[1]]
            ):
                self.index.release_ref(self.records.pop(key))
        if self.scorer is not None:
            for layer, required in required_by_layer.items():
                if not required:
                    self.scorer.drop_index((state.request_id, layer))

    def select(self, requests, layer, query, batch_view):
        # Quest summaries are published only after the cross-layer append
        # transaction commits.  Full/Chunked Prefill therefore remains exact
        # and does not attempt to consume an absent or pre-commit index.
        # Positive query lengths mean total_query == batch_size only when
        # every request is a one-token Decode.
        requests = tuple(requests)
        if int(query.shape[0]) != len(requests):
            return self._select_exact_prefill(batch_view)
        if self.scorer is not None:
            return self._select_tensorized(requests, layer, query, batch_view)
        return self._select_cpu_reference(requests, layer, query, batch_view)

    def _select_exact_prefill(self, batch_view):
        requests = []
        for request_index in range(batch_view.batch_size):
            start = int(batch_view.block_table_indptr[request_index].item())
            end = int(batch_view.block_table_indptr[request_index + 1].item())
            requests.append(
                {
                    "mode": "full",
                    "candidate_count": end - start,
                    "selected_count": end - start,
                    "scorer_provider": self.scorer_name,
                    "fallback_reason": "quest_sparse_selection_is_decode_only",
                }
            )
        return SelectedPageView(
            flat_page_ids=batch_view.flat_block_table,
            block_table_indptr=batch_view.block_table_indptr,
            logical_block_ids=batch_view.flat_logical_block_ids,
            page_valid_tokens=batch_view.flat_page_valid_tokens,
            selection_name=self.name,
            exact=True,
            metadata={
                "requests": tuple(requests),
                "index_stats": self.index.stats(),
            },
        )

    def _select_tensorized(self, requests, layer, query, batch_view):
        """Run the explicitly enabled tensorized batch-one decode path."""

        import torch

        requests = tuple(requests)
        if len(requests) != 1 or batch_view.batch_size != 1:
            raise NotImplementedError(
                "tensorized Quest selection currently supports batch one"
            )
        state = requests[0]
        result = self.scorer.select(
            (state.request_id, int(layer)),
            query,
            scored_budget=self.budget,
            recent_window=self.recent_window,
            page_size=batch_view.page_size,
            mode=("full" if self.mode == "full" or not self.budget else "budget"),
        )
        if result.candidate_count != int(batch_view.flat_block_table.shape[0]):
            raise RuntimeError(
                "tensorized Quest index does not match the batch block table"
            )
        indices = result.selected_positions
        device = batch_view.flat_block_table.device
        metadata = result.as_dict()
        metadata.update(
            {
                "scorer_provider": self.scorer.name,
                "index_stats": self.index.stats(),
                "scorer_stats": self.scorer.stats(),
            }
        )
        return SelectedPageView(
            flat_page_ids=batch_view.flat_block_table[indices],
            block_table_indptr=torch.tensor(
                (0, result.selected_count), dtype=torch.int32, device=device
            ),
            logical_block_ids=batch_view.flat_logical_block_ids[indices],
            page_valid_tokens=batch_view.flat_page_valid_tokens[indices],
            selection_name=self.name,
            exact=result.mode == "full",
            metadata={"requests": (metadata,), "index_stats": self.index.stats()},
        )

    def _select_cpu_reference(self, requests, layer, query, batch_view):
        import torch

        selected_indices = []
        selected_counts = []
        stats = []
        layer = int(layer)
        for request_index, state in enumerate(requests):
            block_start = int(
                batch_view.block_table_indptr[request_index].item()
            )
            block_end = int(
                batch_view.block_table_indptr[request_index + 1].item()
            )
            candidates = []
            for logical in range(block_end - block_start):
                record = self.records.get(
                    _request_key(state.request_id, layer, logical)
                )
                if record is None:
                    raise RuntimeError(
                        "Quest index missing request={} layer={} logical_block={}".format(
                            state.request_id, layer, logical
                        )
                    )
                if record.index_version != record.data_version:
                    raise RuntimeError("Quest index is stale")
                candidates.append(record)
            query_start = int(batch_view.query_indptr[request_index].item())
            query_end = int(batch_view.query_indptr[request_index + 1].item())
            query_vector = (
                query[query_start:query_end]
                .float()
                .mean(dim=(0, 1))
                .detach()
                .cpu()
                .tolist()
            )
            mode = "full" if self.mode == "full" or not self.budget else "budget"
            recent_records = ()
            if self.recent_window and candidates and mode != "full":
                recent_pages = min(
                    len(candidates),
                    max(
                        1,
                        (self.recent_window + batch_view.page_size - 1)
                        // batch_view.page_size,
                    ),
                )
                recent_records = tuple(candidates[-recent_pages:])
                recent_ids = {
                    item.record_id.value for item in recent_records
                }
                scored_candidates = tuple(
                    item
                    for item in candidates
                    if item.record_id.value not in recent_ids
                )
            else:
                scored_candidates = tuple(candidates)
            scored_result = self.index.select(
                query_vector,
                scored_candidates,
                budget=(
                    len(scored_candidates)
                    if mode == "full"
                    else min(self.budget, len(scored_candidates))
                ),
                mode=mode,
            )
            # Recent pages bypass scoring but remain real selection
            # candidates and selected pages in end-to-end metrics.
            self.index.candidate_count += len(recent_records)
            self.index.selected_count += len(recent_records)
            selected_logical = {
                item.logical_block_id.logical_block
                for item in scored_result.records + recent_records
            }
            logical_order = sorted(selected_logical)
            selected_indices.extend(
                block_start + item for item in logical_order
            )
            selected_counts.append(len(logical_order))
            request_stats = scored_result.as_dict()
            request_stats.update(
                {
                    "scored_budget": (
                        len(candidates) if mode == "full" else self.budget
                    ),
                    "recent_budget": len(recent_records),
                    "total_budget": (
                        len(candidates)
                        if mode == "full"
                        else self.budget + len(recent_records)
                    ),
                    "selected_total": len(logical_order),
                }
            )
            stats.append(request_stats)
        device = batch_view.flat_block_table.device
        indices = torch.tensor(
            selected_indices, dtype=torch.long, device=device
        )
        indptr = [0]
        for count in selected_counts:
            indptr.append(indptr[-1] + count)
        return SelectedPageView(
            flat_page_ids=batch_view.flat_block_table[indices],
            block_table_indptr=torch.tensor(
                indptr, dtype=torch.int32, device=device
            ),
            logical_block_ids=batch_view.flat_logical_block_ids[indices],
            page_valid_tokens=batch_view.flat_page_valid_tokens[indices],
            selection_name=self.name,
            exact=all(item["mode"] == "full" for item in stats),
            metadata={"requests": stats, "index_stats": self.index.stats()},
        )
