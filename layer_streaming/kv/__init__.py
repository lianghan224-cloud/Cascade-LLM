"""Cascade-LLM KV Framework V1 public surface."""

from .block_table import LogicalBlockTable
from .api import RequestKVCacheV1
from .batch_state import (
    PAGED_BATCH_ABI_VERSION,
    PAGED_BATCH_V2_ABI_VERSION,
    PagedBatchView,
    PagedBatchViewV2,
    build_paged_batch_view,
)
from .cow import PageCOWOperation
from .errors import (
    KVCapacityError,
    KVError,
    KVLifecycleError,
    KVProviderError,
    KVUnsupportedError,
)
from .metrics import KVMetrics
from .fence import KVOperationFence
from .device_metadata import DeviceKVPageMetadata, DeviceKVPageTable
from .kernel_backend import (
    PAGED_KV_KERNEL_ABI_VERSION,
    PagedKVKernelBackend,
    PagedKVKernelCapability,
    TorchPagedKVKernelBackend,
)
from .page_pool import KVPagePoolV1
from .page_view import DeviceSelectedPageView, SelectedPageView
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
from .resource_audit import (
    KVResourceComparison,
    KVResourceSnapshot,
    ResourceDrift,
    compare_resource_snapshots,
    empty_cuda_cache_and_capture,
)
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
    "KVOperationFence",
    "DeviceKVPageMetadata",
    "DeviceKVPageTable",
    "DeviceSelectedPageView",
    "KVResourceComparison",
    "KVResourceSnapshot",
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
    "PAGED_BATCH_V2_ABI_VERSION",
    "PAGED_KV_KERNEL_ABI_VERSION",
    "PageCOWOperation",
    "PageDescriptor",
    "PageHandle",
    "PageState",
    "PendingAppend",
    "PagedBatchView",
    "PagedBatchViewV2",
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
    "ResourceDrift",
    "SelectedPageView",
    "SlotMapping",
    "TorchPagedKVKernelBackend",
    "build_paged_batch_view",
    "compare_resource_snapshots",
    "empty_cuda_cache_and_capture",
]
