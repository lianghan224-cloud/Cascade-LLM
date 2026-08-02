"""Exact dense page selection."""

from .base import KVSelectionPolicyProvider, SelectionCapability
from ..page_view import SelectedPageView


class DenseSelection(KVSelectionPolicyProvider):
    name = "dense"

    def capability(self):
        return SelectionCapability(
            name=self.name,
            exact=True,
            implemented=True,
            requires_index=False,
        )

    def select(self, requests, layer, query, batch_view):
        del requests, layer, query
        return SelectedPageView(
            flat_page_ids=batch_view.flat_block_table,
            block_table_indptr=batch_view.block_table_indptr,
            logical_block_ids=batch_view.flat_logical_block_ids,
            page_valid_tokens=batch_view.flat_page_valid_tokens,
            selection_name=self.name,
            exact=True,
            metadata={},
        )
