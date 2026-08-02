"""Cascade-LLM KV Framework V1 public surface."""

from .block_table import LogicalBlockTable
from .api import RequestKVCacheV1
from .batch_state import PAGED_BATCH_ABI_VERSION, PagedBatchView, build_paged_batch_view
from .cow import PageCOWOperation
from .errors import (
    KVCapacityError,
    KVError,
    KVLifecycleError,
    KVProviderError,
    KVUnsupportedError,
)
from .metrics import KVMetrics
from .kernel_backend import (
    PAGED_KV_KERNEL_ABI_VERSION,
    PagedKVKernelBackend,
    PagedKVKernelCapability,
    TorchPagedKVKernelBackend,
)
from .page_pool import KVPagePoolV1
from .page_view import SelectedPageView
from .policy import (
    KVAccuracy,
    KVDataType,
    KVLayout,
    KVPolicy,
    KVReusePolicy,
    KVSelectionPolicy,
    KVStoragePolicy,
)
from .request_state import PendingAppend, RequestKVState
from .runtime import PagedKVRuntime
from .reports import (
    KV_REPORT_SCHEMA_VERSION,
    KVCompatibilityReport,
    KVNumericalReport,
    KVOwnershipReport,
    KVPerformanceReport,
    KVQualificationReport,
)
from .slot_mapping import SlotMapping
from .types import (
    KV_FRAMEWORK_ABI_VERSION,
    KV_PAGE_FORMAT_VERSION,
    PageDescriptor,
    PageHandle,
    PageState,
    RequestLifecycleState,
)

__all__ = [
    "KVCapacityError",
    "KVError",
    "KVLifecycleError",
    "KVProviderError",
    "KVUnsupportedError",
    "KVMetrics",
    "KV_FRAMEWORK_ABI_VERSION",
    "KV_PAGE_FORMAT_VERSION",
    "LogicalBlockTable",
    "KVAccuracy",
    "KVDataType",
    "KVLayout",
    "KVPagePoolV1",
    "KVPolicy",
    "KVReusePolicy",
    "KVSelectionPolicy",
    "KVStoragePolicy",
    "PAGED_BATCH_ABI_VERSION",
    "PAGED_KV_KERNEL_ABI_VERSION",
    "PageCOWOperation",
    "PageDescriptor",
    "PageHandle",
    "PageState",
    "PendingAppend",
    "PagedBatchView",
    "PagedKVKernelBackend",
    "PagedKVKernelCapability",
    "PagedKVRuntime",
    "KV_REPORT_SCHEMA_VERSION",
    "KVCompatibilityReport",
    "KVNumericalReport",
    "KVOwnershipReport",
    "KVPerformanceReport",
    "KVQualificationReport",
    "RequestKVCacheV1",
    "RequestKVState",
    "RequestLifecycleState",
    "SelectedPageView",
    "SlotMapping",
    "TorchPagedKVKernelBackend",
    "build_paged_batch_view",
]
