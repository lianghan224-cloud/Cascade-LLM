import inspect
import unittest

import torch

from layer_streaming.kv.device_metadata import (
    DEVICE_PAGE_CPU,
    DEVICE_PAGE_GPU,
    DEVICE_PAGE_INVALID_ID,
    DEVICE_PAGE_STALE_EPOCH,
    DEVICE_PAGE_STALE_GENERATION,
    DEVICE_PAGE_STALE_VALID_TOKENS,
    DeviceKVPageTable,
)
from layer_streaming.kv.page_view import DeviceSelectedPageView, SelectedPageView
from layer_streaming.kv.selection.rgkv import RGKVSelectionPolicy


class DeviceKVPageMetadataTest(unittest.TestCase):
    def test_tensor_gather_and_epoch_validation_stay_device_visible(self):
        table = DeviceKVPageTable(4, device="cpu")
        table.publish_page(
            1,
            generation=3,
            data_epoch=7,
            valid_tokens=16,
            physical_gpu_slot=2,
            location_state=DEVICE_PAGE_GPU,
        )
        table.publish_page(
            2,
            generation=4,
            data_epoch=9,
            valid_tokens=5,
            physical_gpu_slot=-1,
            location_state=DEVICE_PAGE_CPU,
        )
        result = table.gather(
            torch.tensor((0, 1), dtype=torch.int64),
            torch.tensor((1, 2), dtype=torch.int64),
            expected_epochs=torch.tensor((7, 8), dtype=torch.int64),
            expected_generations=torch.tensor((3, 4), dtype=torch.int64),
            expected_valid_tokens=torch.tensor((16, 6), dtype=torch.int32),
        )
        self.assertEqual(result.physical_gpu_slots.tolist(), [2, -1])
        self.assertEqual(result.valid_tokens.tolist(), [16, 5])
        self.assertEqual(
            result.error_states.tolist(),
            [0, DEVICE_PAGE_STALE_EPOCH | DEVICE_PAGE_STALE_VALID_TOKENS],
        )
        self.assertEqual(result.physical_gpu_slots.dtype, torch.int32)

    def test_generation_mismatch_is_device_visible(self):
        table = DeviceKVPageTable(2, device="cpu")
        table.publish_page(
            1,
            generation=4,
            data_epoch=9,
            valid_tokens=5,
            physical_gpu_slot=0,
            location_state=DEVICE_PAGE_GPU,
        )
        result = table.gather(
            torch.tensor((0,), dtype=torch.int32),
            torch.tensor((1,), dtype=torch.int32),
            expected_generations=torch.tensor((3,), dtype=torch.int64),
        )
        self.assertEqual(
            result.error_states.tolist(), [DEVICE_PAGE_STALE_GENERATION]
        )

    def test_invalid_id_is_flagged_without_invalid_indexing(self):
        table = DeviceKVPageTable(2, device="cpu")
        result = table.gather(
            torch.tensor((0, 1), dtype=torch.int64),
            torch.tensor((-1, 5), dtype=torch.int64),
        )
        self.assertEqual(
            result.error_states.tolist(),
            [DEVICE_PAGE_INVALID_ID, DEVICE_PAGE_INVALID_ID],
        )

    def test_device_selected_view_preserves_tensor_contract(self):
        ids = torch.tensor((1, 4), dtype=torch.int32)
        view = DeviceSelectedPageView(
            flat_page_ids=torch.tensor((7, 2), dtype=torch.int32),
            block_table_indptr=torch.tensor((0, 2), dtype=torch.int32),
            logical_block_ids=ids,
            page_valid_tokens=torch.tensor((16, 3), dtype=torch.int32),
            selection_name="rgkv",
            exact=False,
            metadata={},
            gpu_physical_slots=torch.tensor((0, -1), dtype=torch.int32),
            expected_epochs=torch.tensor((11, 13), dtype=torch.int64),
            current_epochs=torch.tensor((11, 12), dtype=torch.int64),
            expected_generations=torch.tensor((2, 8), dtype=torch.int64),
            current_generations=torch.tensor((2, 8), dtype=torch.int64),
            location_flags=torch.tensor(
                (DEVICE_PAGE_GPU, DEVICE_PAGE_CPU), dtype=torch.int32
            ),
            valid_mask=torch.tensor((True, False), dtype=torch.bool),
            error_mask=torch.tensor(
                (0, DEVICE_PAGE_STALE_EPOCH), dtype=torch.int32
            ),
            selection_count=2,
            staged_tail_mask=torch.tensor((False, True)),
        )
        self.assertIs(view.logical_ids_tensor, ids)
        self.assertIs(view.device_epoch_error_state(), view.error_state_tensor)
        self.assertEqual(view.selection_count, 2)
        self.assertEqual(view.gpu_physical_slots.dtype, torch.int32)
        self.assertIs(view.physical_slot_tensor, view.gpu_physical_slots)
        self.assertIs(view.generation_tensor, view.current_generations)

    def test_device_table_hot_methods_have_no_scalar_readback(self):
        sources = "\n".join(
            inspect.getsource(item)
            for item in (
                DeviceKVPageTable.gather,
                DeviceSelectedPageView.device_epoch_error_state,
            )
        )
        for forbidden in (".cpu(", ".numpy(", ".tolist(", ".item(", "synchronize("):
            self.assertNotIn(forbidden, sources)

    def test_rgkv_device_view_assembly_has_no_scalar_readback(self):
        source = inspect.getsource(RGKVSelectionPolicy.select)
        for forbidden in (
            ".cpu(",
            ".numpy(",
            ".tolist(",
            ".item(",
            "synchronize(",
        ):
            self.assertNotIn(forbidden, source)

    def test_host_authority_bridge_uses_one_bulk_transfer_not_scalar_item(self):
        source = inspect.getsource(SelectedPageView.resolve_entries)
        self.assertNotIn(".item(", source)
        self.assertEqual(source.count('.to(device="cpu")'), 1)
        self.assertEqual(source.count(".tolist()"), 1)


if __name__ == "__main__":
    unittest.main()
