"""Deterministic KV Framework V1 compatibility contract."""

from dataclasses import fields
import hashlib
import inspect
import json

from .attention.paged import (
    PAGED_ATTENTION_ABI_VERSION,
    PagedAttentionCapability,
    PagedAttentionBackend,
    PagedAttentionInput,
    PagedAttentionOutput,
    PagedKVAppendInput,
)
from .kv import (
    KV_FRAMEWORK_ABI_VERSION,
    KV_PAGE_FORMAT_VERSION,
    PAGED_KV_KERNEL_ABI_VERSION,
    LogicalBlockTable,
    PageDescriptor,
    PageState,
    PagedBatchView,
    PagedKVKernelBackend,
    PagedKVKernelCapability,
    RequestKVState,
    SelectedPageView,
    SlotMapping,
)
from .providers.base import PagedProviderBundle
from .kv.reuse import KV_REUSE_ABI_VERSION, KVReusePolicyProvider
from .kv.selection import KV_SELECTION_ABI_VERSION, KVSelectionPolicyProvider
from .kv.stores import KV_STORE_ABI_VERSION, KVStore


KV_FRAMEWORK_CONTRACT_VERSION = 1


def _fields(item):
    return [value.name for value in fields(item)]


def _parameters(item):
    return list(inspect.signature(item).parameters)


def kv_framework_contract():
    return {
        "contract_version": KV_FRAMEWORK_CONTRACT_VERSION,
        "abi_versions": {
            "framework": KV_FRAMEWORK_ABI_VERSION,
            "page_format": KV_PAGE_FORMAT_VERSION,
            "paged_attention": PAGED_ATTENTION_ABI_VERSION,
            "paged_kv_kernel": PAGED_KV_KERNEL_ABI_VERSION,
            "store": KV_STORE_ABI_VERSION,
            "selection": KV_SELECTION_ABI_VERSION,
            "reuse": KV_REUSE_ABI_VERSION,
        },
        "page_layout": "layer,page,kv_head,page_token,head_dim (HND per layer)",
        "page_handle_fields": [
            "page_id",
            "generation",
            "store_id",
            "format_id",
            "state",
        ],
        "page_states": [item.value for item in PageState],
        "dataclass_fields": {
            item.__name__: _fields(item)
            for item in (
                PageDescriptor,
                LogicalBlockTable,
                RequestKVState,
                SlotMapping,
                SelectedPageView,
                PagedBatchView,
                PagedKVAppendInput,
                PagedAttentionInput,
                PagedAttentionOutput,
                PagedAttentionCapability,
                PagedKVKernelCapability,
            )
        },
        "provider_bundle_fields": _fields(PagedProviderBundle),
        "attention_backend_methods": {
            name: _parameters(getattr(PagedAttentionBackend, name))
            for name in (
                "capability",
                "estimate_workspace",
                "decode",
                "prefill",
            )
        },
        "page_kernel_backend_methods": {
            name: _parameters(getattr(PagedKVKernelBackend, name))
            for name in ("capability", "append_kv", "copy_pages")
        },
        "store_methods": {
            name: _parameters(getattr(KVStore, name))
            for name in (
                "capability",
                "layer_view",
                "read_pages",
                "write_pages",
                "copy_page",
                "close",
            )
        },
        "selection_methods": {
            name: _parameters(getattr(KVSelectionPolicyProvider, name))
            for name in ("capability", "select")
        },
        "reuse_methods": {
            name: _parameters(getattr(KVReusePolicyProvider, name))
            for name in (
                "capability",
                "fork",
                "register_prefix",
                "lookup_prefix",
            )
        },
    }


def canonical_kv_contract_json():
    return json.dumps(
        kv_framework_contract(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def kv_framework_contract_sha256():
    return hashlib.sha256(
        canonical_kv_contract_json().encode("utf-8")
    ).hexdigest()
