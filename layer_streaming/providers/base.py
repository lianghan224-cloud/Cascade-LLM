"""Provider-facing contracts kept outside the frozen LinearBackend protocol."""

from dataclasses import dataclass

from ..backend_capability import BackendCapability
from ..backends import BackendInfo
from ..hardware.capability import ProviderCapability


@dataclass(frozen=True)
class PagedProviderBundle:
    """Aggregate compute and payload kernels without owning page lifecycle."""

    name: str
    attention_backend: object
    kv_kernel_backend: object

    def __post_init__(self):
        if not self.name:
            raise ValueError("paged provider bundle name cannot be empty")
        if self.attention_backend is None or self.kv_kernel_backend is None:
            raise ValueError("paged provider bundle requires both backends")
        required = {
            "attention backend": (
                self.attention_backend,
                ("capability", "estimate_workspace", "decode", "prefill"),
            ),
            "KV kernel backend": (
                self.kv_kernel_backend,
                ("capability", "append_kv", "copy_pages"),
            ),
        }
        for label, (backend, methods) in required.items():
            missing = tuple(
                name
                for name in methods
                if not callable(getattr(backend, name, None))
            )
            if missing:
                raise TypeError(
                    "{} is missing required methods: {}".format(
                        label, ", ".join(missing)
                    )
                )
        attention_page_ops = {
            "append_kv", "copy_pages"
        }.intersection(dir(self.attention_backend))
        if attention_page_ops:
            raise TypeError(
                "attention backend exposes page kernel methods: {}".format(
                    ", ".join(sorted(attention_page_ops))
                )
            )
        kernel_attention_ops = {
            "decode", "prefill", "estimate_workspace"
        }.intersection(dir(self.kv_kernel_backend))
        if kernel_attention_ops:
            raise TypeError(
                "KV kernel backend exposes attention methods: {}".format(
                    ", ".join(sorted(kernel_attention_ops))
                )
            )
        forbidden = {
            "allocate",
            "release",
            "fork",
            "retain",
            "pin",
            "unpin",
        }
        for backend in (self.attention_backend, self.kv_kernel_backend):
            exposed = forbidden.intersection(dir(backend))
            if exposed:
                raise TypeError(
                    "provider backend exposes lifecycle methods: {}".format(
                        ", ".join(sorted(exposed))
                    )
                )

    @property
    def is_reference(self):
        return bool(getattr(self.attention_backend, "is_reference", False))

    def capability(self):
        return self.attention_backend.capability()

    def capability_report(self):
        return {
            "bundle_name": self.name,
            "attention": self.attention_backend.capability().as_dict(),
            "kv_kernel": self.kv_kernel_backend.capability().as_dict(),
        }

__all__ = [
    "BackendCapability",
    "BackendInfo",
    "PagedProviderBundle",
    "ProviderCapability",
]
