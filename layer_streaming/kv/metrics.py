"""KV V1 runtime metrics with stable report semantics."""

from dataclasses import asdict, dataclass


@dataclass
class KVMetrics:
    # ``False`` is deliberate: zero-valued tier counters are schema
    # initialization, not evidence that a tiered run was sampled.  The real
    # migration runtime must set this once it starts publishing observations.
    tier_metrics_sampled: bool = False
    pool_peak_pages: int = 0
    shared_pages: int = 0
    cow_count: int = 0
    fork_count: int = 0
    committed_tokens: int = 0
    append_tokens: int = 0
    append_calls: int = 0
    attention_calls: int = 0
    release_count: int = 0
    prefix_hits: int = 0
    prefix_misses: int = 0
    prefix_entries: int = 0
    prefix_pages: int = 0
    prefix_bytes: int = 0
    prefix_evictions: int = 0
    prefix_replacements: int = 0
    prefill_attention_ms: float = 0.0
    decode_attention_ms: float = 0.0
    workspace_peak_bytes: int = 0
    gpu_kv_capacity_bytes: int = 0
    gpu_kv_used_bytes: int = 0
    gpu_kv_free_bytes: int = 0
    gpu_kv_peak_used_bytes: int = 0
    gpu_kv_peak_used_pages: int = 0
    gpu_kv_capacity_pages: int = 0
    gpu_kv_used_pages: int = 0
    gpu_kv_free_pages: int = 0
    gpu_kv_high_watermark_bytes: int = 0
    gpu_kv_low_watermark_bytes: int = 0
    gpu_kv_high_watermark_pages: int = 0
    gpu_kv_low_watermark_pages: int = 0
    gpu_kv_above_high_watermark: bool = False
    gpu_kv_below_low_watermark: bool = False
    cpu_kv_capacity_bytes: int = 0
    cpu_kv_used_bytes: int = 0
    cpu_kv_reserved_bytes: int = 0
    cpu_kv_free_bytes: int = 0
    cpu_kv_peak_used_bytes: int = 0
    cpu_kv_capacity_pages: int = 0
    cpu_kv_used_pages: int = 0
    cpu_kv_reserved_pages: int = 0
    cpu_kv_free_pages: int = 0
    cpu_kv_high_watermark_bytes: int = 0
    cpu_kv_low_watermark_bytes: int = 0
    cpu_kv_above_high_watermark: bool = False
    cpu_kv_below_low_watermark: bool = False
    gpu_hits: int = 0
    cpu_hits: int = 0
    prefetch_count: int = 0
    prefetch_pages: int = 0
    prefetch_bytes: int = 0
    prefetch_wait_ms: float = 0.0
    prefetch_timeouts: int = 0
    eviction_count: int = 0
    eviction_pages: int = 0
    eviction_bytes: int = 0
    h2d_kv_bytes: int = 0
    d2h_kv_bytes: int = 0
    migration_failures: int = 0
    migration_cancellations: int = 0
    authority_changes: int = 0
    tier_version_mismatches: int = 0
    thrashing_count: int = 0
    thrash_window_operations: int = 0
    rgkv_pages_total: int = 0
    rgkv_pages_selected: int = 0
    rgkv_selection_calls: int = 0
    rgkv_index_bytes: int = 0
    rgkv_update_ms: float = 0.0
    rgkv_selection_enqueue_ms: float = 0.0
    rgkv_score_ms: float = 0.0
    rgkv_topk_ms: float = 0.0
    rgkv_timing_sampled: bool = False
    rgkv_cpu_sync_count: int = 0
    rgkv_host_authority_page_checks: int = 0
    rgkv_stale_index_count: int = 0
    # Dynamic execution-path audit counters.  These are incremented at known
    # Host boundaries; collecting them never reads a CUDA tensor or scalar.
    selected_metadata_d2h: int = 0
    selected_metadata_d2h_bytes: int = 0
    host_scalar_readbacks: int = 0
    explicit_sync_count: int = 0
    python_attention_wave_count: int = 0
    device_view_fallback_count: int = 0
    gpu_hit_device_attention_calls: int = 0

    @property
    def rgkv_selection_ratio(self):
        if not self.rgkv_pages_total:
            return 1.0
        return self.rgkv_pages_selected / float(self.rgkv_pages_total)

    @property
    def gpu_hit_rate(self):
        total = self.gpu_hits + self.cpu_hits
        return self.gpu_hits / float(total) if total else 1.0

    @property
    def cpu_hit_rate(self):
        total = self.gpu_hits + self.cpu_hits
        return self.cpu_hits / float(total) if total else 0.0

    def observe_rgkv_selection(
        self,
        *,
        pages_total,
        pages_selected,
        index_bytes=0,
        selection_enqueue_ms=0.0,
        score_ms=0.0,
        topk_ms=0.0,
        timing_sampled=False,
        cpu_sync_count=0,
        host_authority_page_checks=0,
    ):
        self.rgkv_pages_total += int(pages_total)
        self.rgkv_pages_selected += int(pages_selected)
        self.rgkv_selection_calls += 1
        self.rgkv_index_bytes = max(self.rgkv_index_bytes, int(index_bytes))
        self.rgkv_selection_enqueue_ms += float(selection_enqueue_ms)
        self.rgkv_score_ms += float(score_ms)
        self.rgkv_topk_ms += float(topk_ms)
        self.rgkv_timing_sampled = (
            self.rgkv_timing_sampled or bool(timing_sampled)
        )
        self.rgkv_cpu_sync_count += int(cpu_sync_count)
        self.rgkv_host_authority_page_checks += int(
            host_authority_page_checks
        )
        return self

    def observe_decode_execution(
        self,
        *,
        selected_metadata_d2h=0,
        selected_metadata_d2h_bytes=0,
        host_scalar_readbacks=0,
        explicit_sync_count=0,
        python_attention_wave_count=0,
        device_view_fallback_count=0,
        gpu_hit_device_attention_calls=0,
    ):
        """Accumulate explicit Decode execution-boundary observations.

        Callers provide already-known Host integers.  This helper intentionally
        accepts no tensor, so metrics collection cannot itself introduce a
        scalar readback or CUDA synchronization.
        """

        values = {
            "selected_metadata_d2h": selected_metadata_d2h,
            "selected_metadata_d2h_bytes": selected_metadata_d2h_bytes,
            "host_scalar_readbacks": host_scalar_readbacks,
            "explicit_sync_count": explicit_sync_count,
            "python_attention_wave_count": python_attention_wave_count,
            "device_view_fallback_count": device_view_fallback_count,
            "gpu_hit_device_attention_calls": gpu_hit_device_attention_calls,
        }
        normalized = {}
        for name, value in values.items():
            value = int(value)
            if value < 0:
                raise ValueError("{} must not be negative".format(name))
            normalized[name] = value
        for name, value in normalized.items():
            setattr(self, name, getattr(self, name) + value)
        return self

    def observe_tier(
        self,
        *,
        gpu_cache=None,
        cpu_store=None,
        migration=None,
        operations=None
    ):
        """Publish one cumulative Active-Tier observation.

        Store-specific schemas are normalized here so reporting never has to
        guess whether an unprefixed ``used_bytes`` came from GPU or CPU.  The
        inputs are cumulative snapshots, not deltas; repeated observation is
        therefore idempotent for counters.  Unknown diagnostic fields remain
        available in their owning component and are intentionally ignored.
        """

        snapshots = (gpu_cache, cpu_store, migration, operations)
        if not any(item is not None for item in snapshots):
            raise ValueError("at least one tier observation is required")

        if gpu_cache is not None:
            for name in (
                "gpu_kv_capacity_bytes",
                "gpu_kv_used_bytes",
                "gpu_kv_free_bytes",
                "gpu_kv_peak_used_bytes",
                "gpu_kv_peak_used_pages",
                "gpu_kv_capacity_pages",
                "gpu_kv_used_pages",
                "gpu_kv_free_pages",
                "gpu_kv_high_watermark_bytes",
                "gpu_kv_low_watermark_bytes",
                "gpu_kv_high_watermark_pages",
                "gpu_kv_low_watermark_pages",
                "gpu_kv_above_high_watermark",
                "gpu_kv_below_low_watermark",
            ):
                if name in gpu_cache:
                    setattr(self, name, gpu_cache[name])

        if cpu_store is not None:
            layer_page_bytes = int(cpu_store.get("layer_page_bytes", 0))
            self.cpu_kv_capacity_bytes = int(
                cpu_store.get("capacity_bytes", 0)
            )
            self.cpu_kv_used_bytes = int(cpu_store.get("used_bytes", 0))
            self.cpu_kv_reserved_bytes = int(
                cpu_store.get("reserved_bytes", 0)
            )
            self.cpu_kv_free_bytes = int(cpu_store.get("free_bytes", 0))
            self.cpu_kv_peak_used_bytes = max(
                self.cpu_kv_peak_used_bytes, self.cpu_kv_used_bytes
            )
            self.cpu_kv_capacity_pages = int(cpu_store.get("slot_count", 0))
            self.cpu_kv_free_pages = int(cpu_store.get("free_slots", 0))
            if layer_page_bytes > 0:
                self.cpu_kv_used_pages = (
                    self.cpu_kv_used_bytes // layer_page_bytes
                )
                self.cpu_kv_reserved_pages = (
                    self.cpu_kv_reserved_bytes // layer_page_bytes
                )
            self.cpu_kv_high_watermark_bytes = int(
                cpu_store.get("high_watermark_bytes", 0)
            )
            self.cpu_kv_low_watermark_bytes = int(
                cpu_store.get("low_watermark_bytes", 0)
            )
            self.cpu_kv_above_high_watermark = bool(
                cpu_store.get("above_high_watermark", False)
            )
            self.cpu_kv_below_low_watermark = bool(
                cpu_store.get("below_low_watermark", False)
            )

        if migration is not None:
            for name in (
                "h2d_kv_bytes",
                "d2h_kv_bytes",
                "migration_failures",
                "migration_cancellations",
                "authority_changes",
            ):
                if name in migration:
                    setattr(self, name, int(migration[name]))

        if operations is not None:
            for name in (
                "gpu_hits",
                "cpu_hits",
                "prefetch_count",
                "prefetch_pages",
                "prefetch_bytes",
                "prefetch_wait_ms",
                "prefetch_timeouts",
                "eviction_count",
                "eviction_pages",
                "eviction_bytes",
                "tier_version_mismatches",
                "thrashing_count",
                "thrash_window_operations",
            ):
                if name in operations:
                    setattr(self, name, operations[name])

        self.tier_metrics_sampled = True
        return self

    def as_dict(self):
        result = asdict(self)
        result["rgkv_selection_ratio"] = self.rgkv_selection_ratio
        # Canonical RGKV metric spelling; retain the older cumulative names
        # as compatibility fields in the same schema revision.
        result["rgkv_total_pages"] = int(self.rgkv_pages_total)
        result["rgkv_selected_pages"] = int(self.rgkv_pages_selected)
        result["gpu_hit_rate"] = self.gpu_hit_rate
        result["cpu_hit_rate"] = self.cpu_hit_rate
        return result
