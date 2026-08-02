"""Hierarchical Quest ABI placeholder; intentionally unsupported."""

from .base import KVSelectionPolicyProvider, SelectionCapability


class HierarchicalQuestSelection(KVSelectionPolicyProvider):
    name = "hierarchical_quest"

    def capability(self):
        return SelectionCapability(
            name=self.name,
            exact=False,
            implemented=False,
            requires_index=True,
        )
