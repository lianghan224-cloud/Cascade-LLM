"""Runtime adapter over the deterministic Quest CPU reference index."""

from .base import KVSelectionPolicyProvider, SelectionCapability
from .quest_cpu import LogicalKVBlockId, QuestCPUIndex
from ..page_view import SelectedPageView


def _request_key(request_id, layer, logical_block):
    return (int(request_id), int(layer), int(logical_block))


class QuestFlatSelection(KVSelectionPolicyProvider):
    name = "quest_flat"

    def __init__(self, budget=0, recent_window=0, mode="budget"):
        self.index = QuestCPUIndex()
        self.budget = int(budget)
        self.recent_window = int(recent_window)
        self.mode = str(mode)
        self.records = {}

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
            return ()
        page_ids = [item.page_id for item in state.block_table.handles[:required]]
        keys, _ = runtime.store.read_pages(layer, page_ids)
        built = []
        for logical in range(required):
            key = _request_key(state.request_id, layer, logical)
            if changed_blocks is not None and logical not in changed_blocks:
                existing = self.records.get(key)
                if existing is not None:
                    built.append(existing)
                    continue
            valid = min(
                runtime.page_size,
                sequence_length - logical * runtime.page_size,
            )
            rows = keys[logical, :, :valid, :].float().mean(dim=0)
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
            record = self.index.build(
                rows,
                {
                    "logical_block_id": block_id,
                    "token_start": logical * runtime.page_size,
                    "data_version": record_version,
                },
            )
            self.records[key] = record
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
        return tuple(built)

    def fork_request(self, parent, child):
        for key, record in tuple(self.records.items()):
            if key[0] != int(parent.request_id):
                continue
            child_key = _request_key(child.request_id, key[1], key[2])
            self.records[child_key] = self.index.fork_ref(record)

    def release_request(self, state):
        for key in tuple(self.records):
            if key[0] == int(state.request_id):
                self.records.pop(key, None)

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

    def rollback_request(self, runtime, state, changed_block=None):
        for layer in range(runtime.layer_count):
            if state.layer_lengths[layer] and changed_block is not None:
                self.build_request_layer(
                    runtime,
                    state,
                    layer,
                    version=state.version,
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
                self.records.pop(key, None)

    def select(self, requests, layer, query, batch_view):
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
            budget = (
                len(candidates)
                if mode == "full"
                else min(self.budget, len(candidates))
            )
            result = self.index.select(
                query_vector,
                candidates,
                budget=budget,
                mode=mode,
            )
            selected_logical = {
                item.logical_block_id.logical_block for item in result.records
            }
            if self.recent_window and candidates and mode != "full":
                recent_pages = max(
                    1,
                    (self.recent_window + batch_view.page_size - 1)
                    // batch_view.page_size,
                )
                recent = [
                    item.logical_block_id.logical_block
                    for item in candidates[-recent_pages:]
                ]
                selected_logical.update(recent)
                if len(selected_logical) > budget:
                    selected_logical = set(sorted(selected_logical)[-budget:])
            logical_order = sorted(selected_logical)
            selected_indices.extend(
                block_start + item for item in logical_order
            )
            selected_counts.append(len(logical_order))
            stats.append(result.as_dict())
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
