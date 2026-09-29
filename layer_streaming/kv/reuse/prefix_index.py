"""In-memory sealed-prefix hash index with namespace isolation."""

from dataclasses import dataclass, replace
import hashlib
import threading


@dataclass(frozen=True)
class PrefixMatch:
    namespace: str
    matched_tokens: int
    block_hashes: tuple
    page_handles: tuple


@dataclass(frozen=True)
class PrefixEntry:
    """One independently evictable prefix-cache index entry.

    ``retained_bytes`` describes the entry's pages.  Physical cache usage is
    calculated from unique handles because chained entries share pages.
    """

    namespace: str
    hash: str
    handles: tuple
    page_count: int
    retained_bytes: int
    created_epoch: int
    last_access_epoch: int
    hit_count: int = 0

    @property
    def key(self):
        return (self.namespace, self.hash)


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

    def __init__(self, page_size, page_bytes=0):
        self.page_size = int(page_size)
        self.page_bytes = int(page_bytes)
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.page_bytes < 0:
            raise ValueError("page_bytes cannot be negative")
        self._blocks = {}
        self._epoch = 0
        self._lock = threading.RLock()

    def _next_epoch(self):
        self._epoch += 1
        return self._epoch

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

    def plan_entries(self, namespace, token_ids, handles):
        """Return a complete prospective index without changing live state."""

        hashes = self.build_hashes(namespace, token_ids)
        if len(handles) < len(hashes):
            raise ValueError("not enough sealed pages for prefix registration")
        namespace = str(namespace)
        with self._lock:
            proposed = dict(self._blocks)
            replacements = 0
            registered_keys = []
            for index, block_hash in enumerate(hashes):
                key = (namespace, block_hash)
                if key in proposed:
                    replacements += 1
                epoch = self._next_epoch()
                entry_handles = tuple(handles[: index + 1])
                proposed[key] = PrefixEntry(
                    namespace=namespace,
                    hash=block_hash,
                    handles=entry_handles,
                    page_count=len(entry_handles),
                    retained_bytes=len(entry_handles) * self.page_bytes,
                    created_epoch=epoch,
                    last_access_epoch=epoch,
                )
                registered_keys.append(key)
        return hashes, proposed, tuple(registered_keys), replacements

    def install_entries(self, entries, page_bytes=None):
        """Atomically replace the live entry table with a prepared table."""

        with self._lock:
            if page_bytes is not None:
                page_bytes = int(page_bytes)
                if page_bytes < 0:
                    raise ValueError("page_bytes cannot be negative")
                self.page_bytes = page_bytes
            self._blocks = {
                key: replace(
                    entry,
                    retained_bytes=entry.page_count * self.page_bytes,
                )
                for key, entry in entries.items()
            }

    def register(self, namespace, token_ids, handles):
        hashes, entries, _, _ = self.plan_entries(namespace, token_ids, handles)
        self.install_entries(entries)
        return hashes

    def plan_register(self, namespace, token_ids, handles):
        """Return hashes and old handles orphaned by an atomic replacement."""

        hashes, proposed, _, _ = self.plan_entries(namespace, token_ids, handles)
        with self._lock:
            referenced_after = {
                handle.identity()
                for entry in proposed.values()
                for handle in entry.handles
            }
            old_handles = {
                handle.identity(): handle
                for entry in self._blocks.values()
                for handle in entry.handles
            }
            stale = tuple(
                handle
                for identity, handle in old_handles.items()
                if identity not in referenced_after
            )
        return hashes, stale

    def lookup(self, namespace, token_ids):
        hashes = self.build_hashes(namespace, token_ids)
        matched_entry = None
        matched_index = -1
        namespace = str(namespace)
        with self._lock:
            # Entries are independently evictable.  A chained descendant hash
            # still authenticates its whole handle tuple when an ancestor entry
            # has already been evicted, so do not stop at the first miss.
            for index, block_hash in enumerate(hashes):
                entry = self._blocks.get((namespace, block_hash))
                if entry is not None:
                    matched_entry = entry
                    matched_index = index
            if matched_entry is not None:
                touched = replace(
                    matched_entry,
                    last_access_epoch=self._next_epoch(),
                    hit_count=matched_entry.hit_count + 1,
                )
                self._blocks[touched.key] = touched
                matched_entry = touched
        matched = () if matched_entry is None else matched_entry.handles
        matched_hashes = () if matched_entry is None else hashes[: matched_index + 1]
        return PrefixMatch(
            namespace=namespace,
            matched_tokens=len(matched) * self.page_size,
            block_hashes=tuple(matched_hashes),
            page_handles=tuple(matched),
        )

    def touch(self, namespace, block_hash):
        key = (str(namespace), str(block_hash))
        with self._lock:
            entry = self._blocks.get(key)
            if entry is None:
                return None
            entry = replace(entry, last_access_epoch=self._next_epoch())
            self._blocks[key] = entry
            return entry

    def get_entry(self, namespace, block_hash):
        with self._lock:
            return self._blocks.get((str(namespace), str(block_hash)))

    def entries(self):
        with self._lock:
            return tuple(self._blocks.values())

    def remove_entry(self, namespace, block_hash):
        with self._lock:
            return self._blocks.pop((str(namespace), str(block_hash)), None)

    def remove_handles(self, handles):
        identities = {item.identity() for item in handles}
        with self._lock:
            stale = [
                key
                for key, entry in self._blocks.items()
                if any(item.identity() in identities for item in entry.handles)
            ]
            for key in stale:
                self._blocks.pop(key, None)

    def referenced_handle_identities(self):
        with self._lock:
            return {
                handle.identity()
                for entry in self._blocks.values()
                for handle in entry.handles
            }

    def __len__(self):
        with self._lock:
            return len(self._blocks)
