"""Bounded in-memory sealed-prefix ownership and LRU eviction."""

import threading

from .errors import KVLifecycleError
from .reuse import InMemoryPrefixIndex
from .types import PageState


class PrefixCache:
    def __init__(
        self,
        page_size,
        page_pool,
        layer_count,
        metrics,
        max_prefix_pages=None,
        max_prefix_bytes=None,
        page_bytes=None,
    ):
        self.page_size = int(page_size)
        self.page_pool = page_pool
        self.layer_count = int(layer_count)
        self.metrics = metrics
        self.max_prefix_pages = self._optional_nonnegative(
            max_prefix_pages, "max_prefix_pages"
        )
        self.max_prefix_bytes = self._optional_nonnegative(
            max_prefix_bytes, "max_prefix_bytes"
        )
        self.page_bytes = None if page_bytes is None else int(page_bytes)
        if self.page_bytes is not None and self.page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        if self.max_prefix_bytes is not None and self.page_bytes is None:
            raise ValueError("page_bytes is required with max_prefix_bytes")
        self.index = InMemoryPrefixIndex(
            self.page_size,
            page_bytes=0 if self.page_bytes is None else self.page_bytes,
        )
        # One Prefix owner is retained per unique physical page, regardless of
        # how many chained PrefixEntry objects refer to it.
        self.owned_handles = {}
        self._evictions = 0
        self._replacements = 0
        self._lock = threading.RLock()
        self._sync_metrics()

    @staticmethod
    def _optional_nonnegative(value, name):
        if value is None:
            return None
        value = int(value)
        if value < 0:
            raise ValueError("{} cannot be negative".format(name))
        return value

    @staticmethod
    def _cache_mapping(handle):
        return ("prefix_cache",) + tuple(handle.identity())

    @staticmethod
    def _request_mapping(state, logical):
        return ("request", int(state.request_id), int(logical))

    @staticmethod
    def _handles_for_entries(entries):
        return {
            handle.identity(): handle
            for entry in entries.values()
            for handle in entry.handles
        }

    def _usage(self, entries):
        pages = len(self._handles_for_entries(entries))
        byte_count = 0 if self.page_bytes is None else pages * self.page_bytes
        return pages, byte_count

    def _over_budget(self, entries):
        pages, byte_count = self._usage(entries)
        return (
            self.max_prefix_pages is not None
            and pages > self.max_prefix_pages
        ) or (
            self.max_prefix_bytes is not None
            and byte_count > self.max_prefix_bytes
        )

    def _entry_fits(self, entry):
        """Return whether one entry can ever fit in the configured cache.

        This admission check is intentionally separate from LRU.  Without it,
        registering a two-page prefix into a one-page cache first evicts the
        useful one-page ancestor and then evicts the still-oversized child,
        leaving the cache empty.
        """

        return not (
            self.max_prefix_pages is not None
            and entry.page_count > self.max_prefix_pages
        ) and not (
            self.max_prefix_bytes is not None
            and entry.page_count * self.page_bytes > self.max_prefix_bytes
        )

    def _plan_budget_evictions(self, entries):
        entries = dict(entries)
        evicted = []
        for key, entry in tuple(entries.items()):
            if not self._entry_fits(entry):
                entries.pop(key)
                evicted.append(entry)
        while entries and self._over_budget(entries):
            key, entry = min(
                entries.items(),
                key=lambda item: (
                    item[1].last_access_epoch,
                    item[1].created_epoch,
                    item[0],
                ),
            )
            entries.pop(key)
            evicted.append(entry)
        return entries, tuple(evicted)

    def _validate_sealed_full_pages(self, handles):
        for handle in handles:
            descriptor = self.page_pool.descriptor(handle)
            if descriptor.state not in {PageState.SEALED, PageState.SHARED}:
                raise KVLifecycleError(
                    "prefix page {} is not sealed".format(handle.page_id)
                )
            if descriptor.valid_tokens != self.page_size:
                raise KVLifecycleError(
                    "prefix page {} is not full: valid_tokens={} page_size={}".format(
                        handle.page_id,
                        descriptor.valid_tokens,
                        self.page_size,
                    )
                )

    def _install_entries(self, proposed):
        """Commit index and Prefix owner changes as one cache transaction."""

        final_handles = self._handles_for_entries(proposed)
        old_handles = dict(self.owned_handles)
        acquire = tuple(
            handle
            for identity, handle in final_handles.items()
            if identity not in old_handles
        )
        release = tuple(
            handle
            for identity, handle in old_handles.items()
            if identity not in final_handles
        )
        # Validate every destructive step before publishing the new index.
        self.page_pool.assert_releasable(release, "evict prefix owner")
        acquired = []
        try:
            for handle in acquire:
                self.page_pool.retain(
                    handle, logical_mapping=self._cache_mapping(handle)
                )
                acquired.append(handle)
            self.index.install_entries(proposed, page_bytes=self.page_bytes)
        except BaseException:
            for handle in reversed(acquired):
                self.page_pool.release(
                    handle, logical_mapping=self._cache_mapping(handle)
                )
            raise
        for handle in reversed(release):
            self.page_pool.release(
                handle, logical_mapping=self._cache_mapping(handle)
            )
        self.owned_handles.clear()
        self.owned_handles.update(final_handles)
        self._sync_metrics()

    def _sync_metrics(self):
        """Keep legacy and additive Prefix metrics available to reporters."""

        pages = len(self.owned_handles)
        byte_count = 0 if self.page_bytes is None else pages * self.page_bytes
        # KVMetrics is intentionally ABI-stable; additive attributes remain
        # directly inspectable even before a report schema revision.
        self.metrics.prefix_entries = len(self.index)
        self.metrics.prefix_pages = pages
        self.metrics.prefix_bytes = byte_count
        self.metrics.prefix_evictions = self._evictions
        self.metrics.prefix_replacements = self._replacements

    def configure_limits(
        self,
        max_prefix_pages=None,
        max_prefix_bytes=None,
        page_bytes=None,
    ):
        """Update cache limits and immediately evict LRU entries to fit."""

        with self._lock:
            max_pages = self._optional_nonnegative(
                max_prefix_pages, "max_prefix_pages"
            )
            max_bytes = self._optional_nonnegative(
                max_prefix_bytes, "max_prefix_bytes"
            )
            next_page_bytes = self.page_bytes if page_bytes is None else int(page_bytes)
            if next_page_bytes is not None and next_page_bytes <= 0:
                raise ValueError("page_bytes must be positive")
            if max_bytes is not None and next_page_bytes is None:
                raise ValueError("page_bytes is required with max_prefix_bytes")
            old = (
                self.max_prefix_pages,
                self.max_prefix_bytes,
                self.page_bytes,
            )
            self.max_prefix_pages = max_pages
            self.max_prefix_bytes = max_bytes
            self.page_bytes = next_page_bytes
            try:
                return self.evict_until_fit()
            except BaseException:
                (
                    self.max_prefix_pages,
                    self.max_prefix_bytes,
                    self.page_bytes,
                ) = old
                raise

    def register(self, state, token_ids):
        with self._lock:
            if state.pending_append is not None:
                raise KVLifecycleError("cannot register an uncommitted prefix")
            hashes = self.index.build_hashes(state.reuse_namespace, token_ids)
            handles = tuple(state.block_table.handles[: len(hashes)])
            if len(handles) < len(hashes):
                raise ValueError("not enough sealed pages for prefix registration")
            self._validate_sealed_full_pages(handles)
            hashes, proposed, _, replacements = self.index.plan_entries(
                state.reuse_namespace,
                token_ids,
                handles,
            )
            proposed, evicted = self._plan_budget_evictions(proposed)
            self._install_entries(proposed)
            self._replacements += replacements
            self._evictions += len(evicted)
            self._sync_metrics()
            state.token_block_hashes[:] = list(hashes)
            return hashes

    def reuse(
        self,
        runtime,
        token_ids,
        max_length,
        reuse_namespace="default",
        request_id=None,
    ):
        with self._lock:
            match = self.index.lookup(reuse_namespace, token_ids)
            if not match.page_handles:
                self.metrics.prefix_misses += 1
                return runtime.create_request(
                    max_length,
                    request_id=request_id,
                    reuse_namespace=reuse_namespace,
                ), match
            state = runtime.create_request(
                max_length,
                request_id=request_id,
                reuse_namespace=reuse_namespace,
            )
            try:
                for logical, handle in enumerate(match.page_handles):
                    self.page_pool.retain(
                        handle,
                        logical_mapping=self._request_mapping(state, logical),
                    )
                    state.block_table.append(handle)
                state.sequence_length = match.matched_tokens
                state.tail_valid_tokens = self.page_size
                state.layer_lengths[:] = [match.matched_tokens] * self.layer_count
                state.token_block_hashes[:] = list(match.block_hashes)
                state.version += 1
            except BaseException:
                runtime.release(state)
                raise
            self.metrics.prefix_hits += 1
            self._sync_metrics()
            return state, match

    def touch(self, namespace, block_hash):
        with self._lock:
            return self.index.touch(namespace, block_hash)

    def evict(self, entry):
        """Evict one PrefixEntry and release only newly orphaned owners."""

        with self._lock:
            key = entry.key if hasattr(entry, "key") else tuple(entry)
            proposed = {item.key: item for item in self.index.entries()}
            current = proposed.get(key)
            if current is None:
                return False
            # A PrefixEntry is a generation-bearing eviction target.  A stale
            # object from before a same-hash replacement must not evict the new
            # owner.  Passing an explicit ``(namespace, hash)`` key remains the
            # administrative "evict current" form.
            if hasattr(entry, "created_epoch") and (
                current.created_epoch != entry.created_epoch
            ):
                return False
            proposed.pop(key)
            self._install_entries(proposed)
            self._evictions += 1
            self._sync_metrics()
            return True

    def evict_until_fit(self):
        with self._lock:
            current = {entry.key: entry for entry in self.index.entries()}
            proposed, evicted = self._plan_budget_evictions(current)
            if evicted:
                self._install_entries(proposed)
                self._evictions += len(evicted)
            self._sync_metrics()
            return len(evicted)

    def close(self):
        self.evict_all()

    def evict_all(self):
        """Drop cache ownership without disturbing active request owners."""

        with self._lock:
            handle_count = len(self.owned_handles)
            entry_count = len(self.index)
            self._install_entries({})
            self._evictions += entry_count
            self._sync_metrics()
            return handle_count

    def stats(self):
        with self._lock:
            pages = len(self.owned_handles)
            return {
                "prefix_entries": len(self.index),
                "prefix_pages": pages,
                "prefix_bytes": 0 if self.page_bytes is None else pages * self.page_bytes,
                "prefix_hits": int(self.metrics.prefix_hits),
                "prefix_misses": int(self.metrics.prefix_misses),
                "prefix_evictions": self._evictions,
                "prefix_replacements": self._replacements,
            }

    def __len__(self):
        return len(self.index)
