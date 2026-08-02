"""Page-selection policy ABI V1."""

from dataclasses import dataclass

from ..errors import KVUnsupportedError


KV_SELECTION_ABI_VERSION = 1


@dataclass(frozen=True)
class SelectionCapability:
    name: str
    exact: bool
    implemented: bool
    requires_index: bool


class KVSelectionPolicyProvider:
    name = "abstract"

    def capability(self):
        raise NotImplementedError

    def select(self, requests, layer, query, batch_view):
        raise KVUnsupportedError("selection policy {} is unavailable".format(self.name))
