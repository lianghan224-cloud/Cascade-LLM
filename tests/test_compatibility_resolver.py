import unittest

from layer_streaming.hardware import (
    CompatibilityRequest,
    CompatibilityResolver,
    RuntimeFeatureProfile,
    default_provider_registry,
    fake_hardware_profile,
)


def runtime(architecture, loaded=False, abi=()):
    return RuntimeFeatureProfile(
        cuda_available=True,
        cuda_graph_available=True,
        pinned_memory_available=True,
        cutlass_extension_loaded=loaded,
        provider_abi_versions=tuple(abi),
        compiled_architectures=(architecture,) if loaded else (),
        deterministic_mode=False,
    )


def request(backend="fused_w8a16", **overrides):
    values = {
        "phase": "decode",
        "backend_requested": backend,
        "weight_format": "int8_symmetric_per_channel",
        "activation_dtype": "bf16",
        "scale_dtype": "bf16",
        "group_size": None,
        "m": 1,
        "n": 4096,
        "k": 4096,
        "physical_layout": "row_major",
        "workspace_limit_bytes": 0,
    }
    values.update(overrides)
    return CompatibilityRequest(**values)


class CompatibilityResolverTest(unittest.TestCase):
    def setUp(self):
        self.registry = default_provider_registry()
        self.resolver = CompatibilityResolver(self.registry)

    def test_only_performance_qualified_loaded_sm86_fused_is_supported(self):
        decision = self.resolver.resolve(
            fake_hardware_profile("sm86"),
            runtime("sm86", loaded=True, abi=(2,)),
            request(),
        )
        self.assertTrue(decision.supported)
        self.assertEqual(decision.status, "performance_qualified")
        self.assertEqual(decision.provider_name, "cutlass_w8a16_sm86_abi2")

    def test_future_architectures_are_declared_but_unavailable(self):
        for architecture in ("sm80", "sm89", "sm90"):
            decision = self.resolver.resolve(
                fake_hardware_profile(architecture),
                runtime(architecture),
                request(),
            )
            self.assertFalse(decision.supported, architecture)
            self.assertEqual(decision.status, "unsupported")
            self.assertTrue(
                any("compiled architecture missing" in item for item in decision.reasons)
            )

    def test_sm75_and_unknown_reject_fused(self):
        for architecture in ("sm75", "unknown"):
            decision = self.resolver.resolve(
                fake_hardware_profile(architecture),
                runtime(architecture),
                request(),
            )
            self.assertFalse(decision.supported)
            self.assertTrue(decision.explicit_fallback_available)

    def test_shape_group_dtype_and_layout_reasons_are_complete(self):
        decision = self.resolver.resolve(
            fake_hardware_profile("sm86"),
            runtime("sm86", loaded=True, abi=(2,)),
            request(
                weight_format="int8_symmetric_per_group",
                activation_dtype="fp32",
                scale_dtype="fp32",
                group_size=48,
                n=4097,
                k=4097,
                physical_layout="column_major",
            ),
        )
        rendered = "\n".join(decision.reasons)
        self.assertIn("activation dtype", rendered)
        self.assertIn("scale dtype", rendered)
        self.assertIn("group size", rendered)
        self.assertIn("N=4097", rendered)
        self.assertIn("K=4097", rendered)
        self.assertIn("physical layout", rendered)

    def test_fallback_must_be_requested_and_is_compiled(self):
        fallback = request(
            backend="int8_dequant_bf16_fallback",
            n=4097,
            k=4096,
        )
        decision = self.resolver.resolve(
            fake_hardware_profile("sm89"), runtime("sm89"), fallback
        )
        self.assertTrue(decision.supported)
        self.assertEqual(decision.status, "compiled")
        fused = self.resolver.resolve(
            fake_hardware_profile("sm89"), runtime("sm89"), request()
        )
        self.assertFalse(fused.supported)
        self.assertNotEqual(fused.provider_name, fallback.backend_requested)


if __name__ == "__main__":
    unittest.main()
