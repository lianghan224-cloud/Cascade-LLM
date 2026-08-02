"""KV Store ABI V1."""

from dataclasses import dataclass

from ..errors import KVUnsupportedError


KV_STORE_ABI_VERSION = 1


@dataclass(frozen=True)
class KVStoreCapability:
    store_id: str
    tiers: tuple
    dtypes: tuple
    layouts: tuple
    supports_active_attention: bool
    supports_async_copy: bool
    implemented: bool

    def as_dict(self):
        return {
            name: getattr(self, name) for name in self.__dataclass_fields__
        }


class KVStore:
    store_id = "abstract"

    def capability(self):
        raise NotImplementedError

    def layer_view(self, layer):
        raise KVUnsupportedError("{} does not expose active GPU pages".format(self.store_id))

    def read_pages(self, layer, page_ids, stream=None):
        raise KVUnsupportedError("{} does not implement page reads".format(self.store_id))

    def write_pages(self, layer, page_ids, key, value, stream=None):
        raise KVUnsupportedError("{} does not implement page writes".format(self.store_id))

    def copy_page(self, source_page_id, target_page_id, valid_tokens, stream=None):
        raise KVUnsupportedError("{} does not implement page copy".format(self.store_id))

    @property
    def nbytes(self):
        return 0

    def close(self):
        return None
