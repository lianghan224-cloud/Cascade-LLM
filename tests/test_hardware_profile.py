import unittest
from unittest import mock

from layer_streaming.hardware import (
    HardwareDetector,
    HardwareProfile,
    RuntimeFeatureProfile,
    architecture_for_compute_capability,
    fake_hardware_profile,
)


class HardwareProfileTest(unittest.TestCase):
    def test_known_and_unknown_architecture_mapping(self):
        for major, minor, expected in (
            (7, 5, "sm75"),
            (8, 0, "sm80"),
            (8, 6, "sm86"),
            (8, 9, "sm89"),
            (9, 0, "sm90"),
            (9, 1, "unknown"),
        ):
            self.assertEqual(
                architecture_for_compute_capability(major, minor), expected
            )

    def test_profiles_round_trip(self):
        profile = fake_hardware_profile("sm89", 24 * 1024 ** 3)
        self.assertEqual(
            HardwareProfile.from_dict(profile.as_dict()), profile
        )
        runtime = RuntimeFeatureProfile(
            cuda_available=True,
            cuda_graph_available=True,
            pinned_memory_available=True,
            cutlass_extension_loaded=False,
            provider_abi_versions=(1, 2),
            compiled_architectures=("sm86", "sm89"),
            deterministic_mode=False,
        )
        self.assertEqual(
            RuntimeFeatureProfile.from_dict(runtime.as_dict()), runtime
        )

    def test_detector_returns_unknown_without_cuda(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            hardware, runtime = HardwareDetector().detect(0, refresh=True)
        self.assertEqual(hardware.architecture, "unknown")
        self.assertFalse(runtime.cuda_available)


if __name__ == "__main__":
    unittest.main()
