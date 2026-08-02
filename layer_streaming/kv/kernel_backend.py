"""Paged KV payload-kernel ABI, separate from page lifecycle ownership."""

from dataclasses import dataclass


PAGED_KV_KERNEL_ABI_VERSION = 1


@dataclass(frozen=True)
class PagedKVKernelCapability:
    backend_name: str
    backend_version: str
    backend_abi: int
    architectures: tuple
    dtypes: tuple
    page_sizes: tuple
    head_dims: tuple
    supports_append: bool
    supports_copy: bool
    qualification_status: str

    def as_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class PagedKVKernelBackend:
    """Executes payload writes/copies but never owns PageHandle lifecycle."""

    name = "abstract_kv_kernel"

    def capability(self):
        raise NotImplementedError

    def append_kv(self, append_input):
        raise NotImplementedError

    def copy_pages(self, store, source_page_ids, target_page_ids, valid_tokens):
        raise NotImplementedError


class TorchPagedKVKernelBackend(PagedKVKernelBackend):
    """Portable correctness backend for CPU/CUDA payload operations."""

    name = "torch_paged_kv"

    def capability(self):
        return PagedKVKernelCapability(
            backend_name=self.name,
            backend_version="1",
            backend_abi=PAGED_KV_KERNEL_ABI_VERSION,
            architectures=("cpu", "sm75", "sm80", "sm86", "sm89", "sm90"),
            dtypes=("bf16", "fp16"),
            page_sizes=(16, 32),
            head_dims=(),
            supports_append=True,
            supports_copy=True,
            qualification_status="qualified_reference",
        )

    def append_kv(self, append_input):
        append_input.validate()
        pages = append_input.slot_mapping.page_ids
        offsets = append_input.slot_mapping.offsets
        for token_index in range(append_input.slot_mapping.token_count):
            page_id = int(pages[token_index].item())
            offset = int(offsets[token_index].item())
            append_input.key_pool_view[page_id, :, offset, :].copy_(
                append_input.key[token_index]
            )
            append_input.value_pool_view[page_id, :, offset, :].copy_(
                append_input.value[token_index]
            )

    def copy_pages(self, store, source_page_ids, target_page_ids, valid_tokens):
        if not (
            len(source_page_ids) == len(target_page_ids) == len(valid_tokens)
        ):
            raise ValueError("copy page lists must have equal length")
        for source, target, valid in zip(
            source_page_ids, target_page_ids, valid_tokens
        ):
            store.copy_page(source, target, valid)
