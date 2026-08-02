"""Quest Flat ABI placeholder; intentionally not an implementation."""

from .base import KVSelectionPolicyProvider, SelectionCapability


class QuestFlatSelection(KVSelectionPolicyProvider):
    name = "quest_flat"

    def capability(self):
        return SelectionCapability(
            name=self.name,
            exact=False,
            implemented=False,
            requires_index=True,
        )
