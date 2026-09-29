import unittest

from layer_streaming.capability_state import (
    CAPABILITY_STATES,
    CapabilityEntry,
    dense_kv_capability_matrix,
    map_legacy_qualification,
)


class CapabilityStateTest(unittest.TestCase):
    def test_vocabulary_is_exact_and_legacy_mapping_is_conservative(self):
        self.assertEqual(
            CAPABILITY_STATES,
            (
                "NOT_IMPLEMENTED",
                "EXPERIMENTAL",
                "LOGIC_VALIDATED",
                "CUDA_SMOKE",
                "QUALIFICATION_READY",
                "QUALIFIED",
            ),
        )
        self.assertEqual(
            map_legacy_qualification("numerically_qualified"), "CUDA_SMOKE"
        )
        self.assertEqual(map_legacy_qualification("unsupported"), "NOT_IMPLEMENTED")

    def test_qualified_requires_evidence(self):
        with self.assertRaisesRegex(ValueError, "requires evidence"):
            CapabilityEntry("unsafe claim", "QUALIFIED")

    def test_dense_matrix_does_not_overclaim(self):
        matrix = {item.capability: item for item in dense_kv_capability_matrix()}
        self.assertEqual(matrix["GPU Dense Page Arena"].state, "QUALIFICATION_READY")
        self.assertEqual(matrix["GenerationSession"].state, "LOGIC_VALIDATED")
        self.assertEqual(matrix["Gather SDPA Prefill"].state, "CUDA_SMOKE")
        self.assertEqual(
            matrix["GPU to Pinned CPU Tensor Migration"].state,
            "CUDA_SMOKE",
        )
        self.assertEqual(
            matrix["Active GPU to Pinned CPU KV Tier"].state,
            "CUDA_SMOKE",
        )
        self.assertNotIn("QUALIFIED", {item.state for item in matrix.values()})


if __name__ == "__main__":
    unittest.main()
