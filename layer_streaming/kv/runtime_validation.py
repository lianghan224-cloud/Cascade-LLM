"""Static runtime/provider compatibility checks kept outside the coordinator."""

from ..attention.paged import detected_architecture
from .errors import KVUnsupportedError


def validate_runtime_provider(runtime, allow_reference=False):
    attention = runtime.dispatcher.attention_backend
    kernel = runtime.dispatcher.kv_kernel_backend
    if attention.is_reference and not allow_reference:
        raise ValueError("reference provider requires allow_reference=True")
    capability = attention.capability()
    kernel_capability = kernel.capability()
    architecture = detected_architecture(runtime.device)
    errors = []
    if architecture not in capability.architectures:
        errors.append("architecture {}".format(architecture))
    if runtime.policy.dtype.value not in capability.dtypes:
        errors.append("dtype {}".format(runtime.policy.dtype.value))
    if runtime.page_size not in capability.page_sizes:
        errors.append("page size {}".format(runtime.page_size))
    if capability.head_dims and runtime.head_dim not in capability.head_dims:
        errors.append("head dim {}".format(runtime.head_dim))
    if runtime.num_kv_heads == 1 and runtime.num_query_heads > 1:
        if not capability.supports_mqa:
            errors.append("MQA")
    elif runtime.num_query_heads == runtime.num_kv_heads:
        if not capability.supports_mha:
            errors.append("MHA")
    elif not capability.supports_gqa:
        errors.append("GQA")
    if architecture not in kernel_capability.architectures:
        errors.append("KV kernel architecture {}".format(architecture))
    if runtime.policy.dtype.value not in kernel_capability.dtypes:
        errors.append("KV kernel dtype {}".format(runtime.policy.dtype.value))
    if runtime.page_size not in kernel_capability.page_sizes:
        errors.append("KV kernel page size {}".format(runtime.page_size))
    if kernel_capability.head_dims and runtime.head_dim not in kernel_capability.head_dims:
        errors.append("KV kernel head dim {}".format(runtime.head_dim))
    if not kernel_capability.supports_append:
        errors.append("KV append")
    if not kernel_capability.supports_copy:
        errors.append("KV page copy")
    if errors:
        raise KVUnsupportedError(
            "provider {} rejected runtime before allocation: {}".format(
                runtime.dispatcher.bundle.name, ", ".join(errors)
            )
        )
