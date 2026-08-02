import unittest

from layer_streaming.hardware import (
    ProviderCapability,
    ProviderRegistry,
    default_provider_registry,
)


class HardwareProviderRegistryTest(unittest.TestCase):
    def test_default_status_boundary(self):
        providers = {
            item.provider_name: item
            for item in default_provider_registry().list_providers()
        }
        self.assertEqual(
            providers["cutlass_w8a16_sm86_abi2"].qualification_status,
            "performance_qualified",
        )
        for architecture in ("sm80", "sm89", "sm90"):
            self.assertEqual(
                providers[
                    "cutlass_w8a16_{}_abi1".format(architecture)
                ].qualification_status,
                "declared",
            )

    def test_duplicate_and_conflict_are_rejected(self):
        registry = ProviderRegistry()
        capability = ProviderCapability(
            provider_name="test_provider",
            provider_version="1",
            provider_abi=1,
            supported_architectures=("sm89",),
            supported_weight_formats=("int8",),
            supported_activation_dtypes=("bf16",),
            supported_scale_dtypes=("bf16",),
            supported_group_sizes=(),
            supported_phases=("decode",),
            min_m=1,
            max_m=1,
            alignment_m=1,
            alignment_n=1,
            alignment_k=1,
            requires_preprocessed_layout=False,
            physical_layout_names=("row_major",),
            workspace_policy="none",
            qualification_status="declared",
            backend_name="test_backend",
        )
        registry.register(capability)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(capability)

    def test_invalid_status_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "qualification status"):
            ProviderCapability(
                provider_name="bad",
                provider_version="1",
                provider_abi=1,
                supported_architectures=("sm89",),
                supported_weight_formats=("int8",),
                supported_activation_dtypes=("bf16",),
                supported_scale_dtypes=("bf16",),
                supported_group_sizes=(),
                supported_phases=("decode",),
                min_m=1,
                max_m=1,
                alignment_m=1,
                alignment_n=1,
                alignment_k=1,
                requires_preprocessed_layout=False,
                physical_layout_names=("row_major",),
                workspace_policy="none",
                qualification_status="unqualified",
                backend_name="bad",
            )


if __name__ == "__main__":
    unittest.main()
