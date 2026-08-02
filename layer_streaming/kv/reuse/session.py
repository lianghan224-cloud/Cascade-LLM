from .base import KVReusePolicyProvider, ReuseCapability


class SessionReuse(KVReusePolicyProvider):
    name = "session"

    def capability(self):
        return ReuseCapability(self.name, True, True, False)

    def fork(self, runtime, state, request_id=None, max_length=None):
        return runtime.fork(
            state,
            request_id=request_id,
            max_length=max_length,
            branch=False,
        )
