import unittest

import torch

from layer_streaming import pack_int4, unpack_int4


class Int4PackingTest(unittest.TestCase):
    def test_low_nibble_first_and_signed_mapping(self):
        values = torch.tensor([-8, -7, -1, 0, 1, 7], dtype=torch.int8)
        packed = pack_int4(values)
        self.assertEqual(packed.tolist(), [0x98, 0x0F, 0x71])
        self.assertTrue(torch.equal(unpack_int4(packed, values.numel()), values))

    def test_odd_element_count_uses_padding_nibble(self):
        values = torch.tensor([-8, 3, 7], dtype=torch.int8)
        packed = pack_int4(values)
        self.assertEqual(packed.numel(), 2)
        self.assertTrue(torch.equal(unpack_int4(packed, 3), values))
        self.assertEqual(unpack_int4(packed).tolist(), [-8, 3, 7, 0])

    def test_out_of_range_and_bad_length_fail(self):
        with self.assertRaisesRegex(ValueError, r"\[-8, 7\]"):
            pack_int4(torch.tensor([8]))
        with self.assertRaisesRegex(ValueError, "requested"):
            unpack_int4(torch.tensor([0], dtype=torch.uint8), 3)


if __name__ == "__main__":
    unittest.main()
