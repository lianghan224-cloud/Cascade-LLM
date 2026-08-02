"""In-memory sealed-prefix hash index with namespace isolation."""

from dataclasses import dataclass
import hashlib
import threading


@dataclass(frozen=True)
class PrefixMatch:
    namespace: str
    matched_tokens: int
    block_hashes: tuple
    page_handles: tuple


def chained_block_hash(namespace, parent_hash, token_ids, logical_block):
    digest = hashlib.sha256()
    digest.update(str(namespace).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(parent_hash or "root").encode("ascii"))
    digest.update(b"\0")
    digest.update(str(int(logical_block)).encode("ascii"))
    for token in token_ids:
        digest.update(int(token).to_bytes(8, byteorder="little", signed=True))
    return digest.hexdigest()


class InMemoryPrefixIndex:
    """Index full sealed pages only; the runtime owns page references."""

    def __init__(self, page_size):
        self.page_size = int(page_size)
        self._blocks = {}
        self._lock = threading.RLock()

    def build_hashes(self, namespace, token_ids):
        tokens = [int(item) for item in token_ids]
        full_blocks = len(tokens) // self.page_size
        parent = None
        result = []
        for logical_block in range(full_blocks):
            start = logical_block * self.page_size
            parent = chained_block_hash(
                namespace,
                parent,
                tokens[start : start + self.page_size],
                logical_block,
            )
            result.append(parent)
        return tuple(result)

    def register(self, namespace, token_ids, handles):
        hashes = self.build_hashes(namespace, token_ids)
        if len(handles) < len(hashes):
            raise ValueError("not enough sealed pages for prefix registration")
        with self._lock:
            for index, block_hash in enumerate(hashes):
                self._blocks[(str(namespace), block_hash)] = tuple(
                    handles[: index + 1]
                )
        return hashes

    def lookup(self, namespace, token_ids):
        hashes = self.build_hashes(namespace, token_ids)
        matched = ()
        matched_hashes = ()
        with self._lock:
            for index, block_hash in enumerate(hashes):
                value = self._blocks.get((str(namespace), block_hash))
                if value is None:
                    break
                matched = value
                matched_hashes = hashes[: index + 1]
        return PrefixMatch(
            namespace=str(namespace),
            matched_tokens=len(matched) * self.page_size,
            block_hashes=tuple(matched_hashes),
            page_handles=tuple(matched),
        )

    def remove_handles(self, handles):
        identities = {item.identity() for item in handles}
        with self._lock:
            stale = [
                key
                for key, value in self._blocks.items()
                if any(item.identity() in identities for item in value)
            ]
            for key in stale:
                self._blocks.pop(key, None)

    def __len__(self):
        with self._lock:
            return len(self._blocks)
