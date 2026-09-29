import inspect
import unittest
from dataclasses import replace

import torch

from layer_streaming.attention.paged import (
    DevicePagedAttentionInput,
    PagedAttentionBackend,
    PagedAttentionDispatcher,
    PagedAttentionOutput,
    PagedAttentionRegistry,
    ReferencePagedExactBackend,
)
from layer_streaming.kv.batch_state import PagedBatchView
from layer_streaming.kv.errors import KVUnsupportedError
from layer_streaming.kv.kernel_backend import TorchPagedKVKernelBackend
from layer_streaming.kv.page_view import DeviceSelectedPageView
from layer_streaming.kv.slot_mapping import SlotMapping
from layer_streaming.providers.base import PagedProviderBundle
from layer_streaming.providers.generic_cuda.paged_attention import (
    GenericCUDAPagedAttentionBackend,
)


def _device_request(*, slot_dtype=torch.int32):
    empty = torch.empty((0,), dtype=torch.int32)
    batch = PagedBatchView(
        request_ids=(7,),
        query_indptr=torch.tensor((0, 1), dtype=torch.int32),
        block_table_indptr=torch.tensor((0, 2), dtype=torch.int32),
        flat_block_table=torch.tensor((5, 6), dtype=torch.int32),
        flat_logical_block_ids=torch.tensor((0, 1), dtype=torch.int32),
        flat_page_valid_tokens=torch.tensor((16, 9), dtype=torch.int32),
        sequence_lengths=torch.tensor((25,), dtype=torch.int32),
        query_lengths=torch.tensor((1,), dtype=torch.int32),
        tail_valid_tokens=torch.tensor((9,), dtype=torch.int32),
        slot_mapping=SlotMapping(empty, empty),
        query_positions=torch.tensor((24,), dtype=torch.int32),
        page_size=16,
        layer=0,
    )
    selected = DeviceSelectedPageView(
        flat_page_ids=torch.tensor((5, 6), dtype=torch.int32),
        block_table_indptr=torch.tensor((0, 2), dtype=torch.int32),
        logical_block_ids=torch.tensor((0, 1), dtype=torch.int32),
        page_valid_tokens=torch.tensor((16, 9), dtype=torch.int32),
        selection_name="rgkv",
        exact=False,
        metadata={},
        gpu_physical_slots=torch.tensor((2, 0), dtype=slot_dtype),
        expected_epochs=torch.tensor((11, 12), dtype=torch.int64),
        current_epochs=torch.tensor((11, 12), dtype=torch.int64),
        expected_generations=torch.tensor((3, 4), dtype=torch.int64),
        current_generations=torch.tensor((3, 4), dtype=torch.int64),
        location_flags=torch.tensor((1, 1), dtype=torch.int32),
        valid_mask=torch.ones((2,), dtype=torch.bool),
        error_mask=torch.zeros((2,), dtype=torch.int32),
        selection_count=2,
    )
    return DevicePagedAttentionInput(
        query=torch.ones((1, 2, 4), dtype=torch.float16),
        key_pool_view=torch.zeros((3, 1, 16, 4), dtype=torch.float16),
        value_pool_view=torch.zeros((3, 1, 16, 4), dtype=torch.float16),
        batch_view=batch,
        page_size=16,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=4,
        softmax_scale=0.5,
        causal=True,
        kv_dtype="fp16",
        output_dtype="fp16",
        selected_pages=selected,
    )


class _DeviceBackend(PagedAttentionBackend):
    name = "test_device_attention"
    supports_device_selected_view = True

    def __init__(self):
        self.device_calls = 0

    def capability(self):
        return replace(
            ReferencePagedExactBackend().capability(),
            provider_name=self.name,
            qualification_status="experimental",
        )

    def estimate_workspace(self, request):
        del request
        return None

    def decode(self, request):
        del request
        raise AssertionError("Host decode ABI must not be selected")

    def decode_device(self, request):
        self.device_calls += 1
        return PagedAttentionOutput(request.query.clone(), None, {"device": True})


def _dispatcher(backend, *, allow_reference=False):
    registry = PagedAttentionRegistry()
    bundle = PagedProviderBundle(
        name="configured",
        attention_backend=backend,
        kv_kernel_backend=TorchPagedKVKernelBackend(),
    )
    registry.register(bundle)
    registry.register(
        PagedProviderBundle(
            name="reference_paged_exact",
            attention_backend=ReferencePagedExactBackend(),
            kv_kernel_backend=TorchPagedKVKernelBackend(),
        )
    )
    return PagedAttentionDispatcher(
        registry,
        "configured",
        allow_reference=allow_reference,
        prefill_provider_name="reference_paged_exact",
    )


class DevicePagedAttentionABITest(unittest.TestCase):
    def test_device_validation_requires_int32_physical_slots(self):
        self.assertIsNotNone(_device_request().validate_device_decode())
        with self.assertRaisesRegex(TypeError, "physical slots must use int32"):
            _device_request(slot_dtype=torch.int64).validate_device_decode()

    def test_strict_dispatch_calls_device_entry_only(self):
        backend = _DeviceBackend()
        dispatcher = _dispatcher(backend)
        result = dispatcher.execute_device(_device_request(), phase="decode")
        self.assertEqual(backend.device_calls, 1)
        self.assertTrue(result.provider_metrics["device"])
        self.assertEqual(dispatcher.routing_summary()["fallback_calls"], 0)
        self.assertEqual(dispatcher.last_decision["selected_view"], "device")

    def test_reference_and_non_decode_never_fallback(self):
        reference = _dispatcher(ReferencePagedExactBackend(), allow_reference=True)
        with self.assertRaisesRegex(
            KVUnsupportedError, "^UNSUPPORTED_DEVICE_SELECTED_VIEW$"
        ):
            reference.execute_device(_device_request(), phase="decode")
        self.assertEqual(reference.routing_summary()["total_calls"], 0)

        dispatcher = _dispatcher(_DeviceBackend())
        with self.assertRaisesRegex(
            KVUnsupportedError, "^UNSUPPORTED_DEVICE_SELECTED_VIEW$"
        ):
            dispatcher.execute_device(_device_request(), phase="prefill")
        self.assertEqual(dispatcher.routing_summary()["total_calls"], 0)

    def test_generic_device_entry_passes_physical_slots_directly(self):
        backend = GenericCUDAPagedAttentionBackend()
        request = _device_request()
        captured = {}

        def capture(current, phase, *, physical_blocks, max_query_length):
            captured.update(
                request=current,
                phase=phase,
                physical_blocks=physical_blocks,
                max_query_length=max_query_length,
            )
            return "launched"

        backend._launch_attention = capture
        self.assertEqual(backend.decode_device(request), "launched")
        self.assertIs(
            captured["physical_blocks"],
            request.selected_pages.physical_slot_tensor,
        )
        self.assertEqual(captured["phase"], "decode")
        self.assertEqual(captured["max_query_length"], 1)

    def test_device_entry_sources_have_no_host_readback_or_sync(self):
        source = "\n".join(
            inspect.getsource(item)
            for item in (
                DevicePagedAttentionInput.validate_device_decode,
                PagedAttentionDispatcher.execute_device,
                GenericCUDAPagedAttentionBackend.decode_device,
                GenericCUDAPagedAttentionBackend._launch_attention,
            )
        )
        for forbidden in (
            ".cpu(",
            ".numpy(",
            ".tolist(",
            ".item(",
            "cuda.synchronize",
            "torch.cuda.synchronize",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
