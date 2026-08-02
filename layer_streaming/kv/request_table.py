"""Request-to-block-table registry independent of hardware providers."""

import math

from .block_table import LogicalBlockTable
from .errors import KVCapacityError
from .request_state import RequestKVState
from .types import RequestLifecycleState


class RequestTable:
    """Own request IDs and RequestKVState objects, but never physical pages."""

    def __init__(self, page_size, page_count, layer_count):
        self.page_size = int(page_size)
        self.page_count = int(page_count)
        self.layer_count = int(layer_count)
        self._states = {}
        self._next_request_id = 1

    def create(
        self,
        max_length,
        request_id=None,
        reuse_namespace="default",
        lifecycle_state=RequestLifecycleState.ACTIVE,
    ):
        max_length = int(max_length)
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if int(math.ceil(max_length / float(self.page_size))) > self.page_count:
            raise KVCapacityError("request max_length exceeds total KV capacity")
        if request_id is None:
            request_id = self._next_request_id
            self._next_request_id += 1
        request_id = int(request_id)
        if request_id in self._states:
            raise ValueError("duplicate request_id {}".format(request_id))
        state = RequestKVState(
            request_id=request_id,
            block_table=LogicalBlockTable(self.page_size, max_length),
            reuse_namespace=str(reuse_namespace),
            lifecycle_state=lifecycle_state,
            layer_lengths=[0] * self.layer_count,
        )
        self._states[request_id] = state
        return state

    def request(self, request_id):
        try:
            return self._states[int(request_id)]
        except KeyError:
            raise KeyError("unknown KV request {}".format(request_id))

    def owns(self, state):
        return self._states.get(int(state.request_id)) is state

    def __contains__(self, request_id):
        return int(request_id) in self._states

    def __len__(self):
        return len(self._states)

    def get(self, request_id, default=None):
        return self._states.get(int(request_id), default)

    def pop(self, request_id, default=None):
        return self._states.pop(int(request_id), default)

    def values(self):
        return self._states.values()
