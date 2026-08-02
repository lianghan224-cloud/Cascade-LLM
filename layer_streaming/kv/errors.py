"""KV Framework V1 errors."""


class KVError(RuntimeError):
    """Base error for the V1 KV runtime."""


class KVCapacityError(KVError):
    """A request cannot be admitted by the configured page budget."""


class KVLifecycleError(KVError):
    """A page or request lifecycle transition is invalid."""


class KVUnsupportedError(KVError):
    """A declared extension point has no executable implementation."""


class KVProviderError(KVError):
    """A paged-attention provider rejected or failed an operation."""
