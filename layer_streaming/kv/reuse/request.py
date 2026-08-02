from .base import KVReusePolicyProvider, ReuseCapability


class RequestOnlyReuse(KVReusePolicyProvider):
    name = "request_only"

    def capability(self):
        return ReuseCapability(self.name, True, False, False)
