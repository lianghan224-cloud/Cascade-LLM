import unittest

from layer_streaming.numerical_contracts import (
    NumericalContractKey,
    NumericalContractRecord,
    NumericalContractRegistry,
)


class NumericalContractRegistryHardwareTest(unittest.TestCase):
    def test_contract_is_exact_by_architecture_and_abi(self):
        registry = NumericalContractRegistry()
        key = NumericalContractKey(
            architecture="sm86",
            provider_name="cutlass_w8a16_sm86_abi2",
            provider_abi=2,
            model_geometry_id="llama31_8b",
            weight_format="int8_symmetric_per_channel",
            activation_dtype="bf16",
            scale_dtype="bf16",
            physical_layout="row_major",
        )
        record = NumericalContractRecord(key, "/contract.json", "qualified")
        registry.register(record)
        self.assertEqual(registry.resolve(key), record)
        self.assertIsNone(
            registry.resolve(
                NumericalContractKey(
                    **dict(key.as_dict(), architecture="sm89")
                )
            )
        )
        self.assertIsNone(
            registry.resolve(
                NumericalContractKey(**dict(key.as_dict(), provider_abi=3))
            )
        )
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(record)


if __name__ == "__main__":
    unittest.main()
