import unittest

from layer_streaming.hardware import (
    CompatibilityReport,
    CompatibilityRequest,
    CompatibilityResolver,
    RuntimeFeatureProfile,
    build_compatibility_report,
    default_provider_registry,
    fake_hardware_profile,
)


class CompatibilityReportTest(unittest.TestCase):
    def test_round_trip_keeps_unqualified_boundary(self):
        hardware = fake_hardware_profile("sm89")
        runtime = RuntimeFeatureProfile(
            cuda_available=True,
            cuda_graph_available=True,
            pinned_memory_available=True,
            cutlass_extension_loaded=False,
            provider_abi_versions=(),
            compiled_architectures=(),
            deterministic_mode=False,
        )
        registry = default_provider_registry()
        request = CompatibilityRequest(
            phase="decode",
            backend_requested="fused_w8a16",
            weight_format="int8_symmetric_per_channel",
            activation_dtype="bf16",
            scale_dtype="bf16",
            group_size=None,
            m=1,
            n=4096,
            k=4096,
            physical_layout="row_major",
            workspace_limit_bytes=0,
        )
        decision = CompatibilityResolver(registry).resolve(
            hardware, runtime, request
        )
        report = build_compatibility_report(
            hardware,
            runtime,
            registry,
            decisions=(decision,),
            selected_backend="fused_w8a16",
        )
        restored = CompatibilityReport.from_json(report.to_json())
        self.assertEqual(restored.overall_status, "unsupported")
        self.assertEqual(restored.hardware.architecture, "sm89")
        self.assertFalse(restored.decisions[0].supported)


if __name__ == "__main__":
    unittest.main()
