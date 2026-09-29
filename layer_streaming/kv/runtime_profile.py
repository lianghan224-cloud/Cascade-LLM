"""Stable profile composition for the thin PagedKVRuntime shell."""


def build_runtime_profile(runtime):
    pool = runtime.page_pool.profile()
    routing = runtime.dispatcher.routing_summary()
    runtime._metrics.pool_peak_pages = pool["peak_allocated_pages"]
    runtime._metrics.shared_pages = pool["shared_pages"]
    if runtime.active_tier_enabled:
        tier = runtime.active_tier.stats()
        runtime._metrics.observe_tier(
            gpu_cache=runtime.store.stats(),
            cpu_store=runtime.cpu_store.stats(),
            migration=tier,
            operations=tier,
        )
    result = runtime._metrics.as_dict()
    scorer_stats = (
        None
        if getattr(runtime.selection, "scorer", None) is None
        else runtime.selection.scorer.stats()
    )
    rgkv_index_stats = (
        runtime.selection.stats()
        if hasattr(runtime.selection, "stats")
        else None
    )
    result.update(
        {
            "abi_version": runtime.abi_version,
            "kv_policy_resolved": runtime.policy.as_rgkv_dict(),
            "attention_backend": runtime.attention_backend.name,
            "attention_accuracy": runtime.policy.accuracy.value,
            "layout": "hnd",
            "kv_store": runtime.store.store_id,
            "kv_dtype": runtime.policy.dtype.value,
            "kv_selection": runtime.selection.name,
            "kv_selection_scorer": getattr(
                runtime.selection, "scorer_name", None
            ),
            "rgkv_scorer_stats": scorer_stats,
            "rgkv_index_stats": rgkv_index_stats,
            "kv_reuse": runtime.reuse.name,
            "kv_page_size": runtime.page_size,
            "kv_pool_total_pages": runtime.page_count,
            "kv_pool_allocated_pages": pool["allocated_pages"],
            "kv_pool_uuid": runtime.page_pool.pool_uuid,
            "active_requests": len(runtime._requests),
            "prefix_index_entries": len(runtime.prefix_index),
            "paged_attention_provider": runtime.attention_backend.name,
            "paged_kv_kernel_backend": runtime.kv_kernel_backend.name,
            "paged_provider_bundle": runtime.provider_bundle.name,
            "paged_prefill_provider": runtime.dispatcher.prefill_provider_name,
            "provider_fallback_reason": (
                None
                if runtime.dispatcher.last_decision is None
                else runtime.dispatcher.last_decision.get("fallback_reason")
            ),
            "provider_decision": runtime.dispatcher.last_decision,
            "provider_routing_summary": routing,
            "provider_fallback_count": routing["fallback_calls"],
            "provider_reference_fallback_count": routing[
                "reference_fallback_calls"
            ],
            "store_bytes": runtime.store.nbytes,
            "page_state_counts": pool["state_counts"],
            "page_allocations": pool["allocation_count"],
            "page_releases": pool["release_count"],
            "total_ref_count": pool["total_ref_count"],
            "logical_owner_count": pool["logical_owner_count"],
            "max_ref_count": pool["max_ref_count"],
            "total_pin_count": pool["total_pin_count"],
            "max_pin_count": pool["max_pin_count"],
            "cuda_event_count": runtime.ownership.event_count,
            "global_data_epoch": runtime.ownership._data_epoch,
            "pending_append_fences": sum(
                item is not None for item in runtime.ownership.append_fences
            ),
            "pending_attention_fences": sum(
                item is not None for item in runtime.ownership.attention_fences
            ),
            "active_tier_enabled": runtime.active_tier_enabled,
            "prefetch_timeout_seconds": runtime.prefetch_timeout_seconds,
        }
    )
    if runtime.active_tier_enabled:
        result.update(runtime.active_tier.stats())
    return result
