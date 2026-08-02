"""KV reuse-policy ABI V1."""

from dataclasses import dataclass

from ..errors import KVUnsupportedError


KV_REUSE_ABI_VERSION = 1


@dataclass(frozen=True)
class ReuseCapability:
    name: str
    implemented: bool
    cross_request: bool
    persistent: bool


class KVReusePolicyProvider:
    name = "abstract"

    def capability(self):
        raise NotImplementedError

    def fork(self, runtime, state, request_id=None, max_length=None):
        raise KVUnsupportedError(
            "reuse policy {} does not implement session fork".format(self.name)
        )

    def register_prefix(self, runtime, state, token_ids):
        raise KVUnsupportedError(
            "reuse policy {} does not implement prefix registration".format(
                self.name
            )
        )

    def lookup_prefix(
        self,
        runtime,
        token_ids,
        max_length,
        reuse_namespace="default",
        request_id=None,
    ):
        raise KVUnsupportedError(
            "reuse policy {} does not implement prefix lookup".format(self.name)
        )
