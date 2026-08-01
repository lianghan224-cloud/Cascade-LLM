import tempfile
import unittest
from pathlib import Path

from layer_streaming.hardware import ProviderBuildMetadata


class ProviderBuildMetadataTest(unittest.TestCase):
    def test_round_trip_and_large_arch_list(self):
        metadata = ProviderBuildMetadata(
            provider="cutlass_w8a16",
            provider_version="1",
            abi=2,
            compiled_architectures=("sm80", "sm86", "sm89", "sm90"),
            weight_formats=("int8_per_channel",),
            activation_dtypes=("bf16", "fp16"),
            build_environment={
                "cuda": "unverified",
                "compiler": "unverified",
                "cutlass": "unverified",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = metadata.write(Path(directory) / "provider.json")
            self.assertEqual(ProviderBuildMetadata.read(path), metadata)

    def test_compiled_requires_architecture_and_unknown_is_rejected(self):
        base = dict(
            provider="test",
            provider_version="1",
            abi=1,
            weight_formats=("int8",),
            activation_dtypes=("bf16",),
            build_environment={
                "cuda": "unverified",
                "compiler": "unverified",
                "cutlass": "unverified",
            },
        )
        with self.assertRaisesRegex(ValueError, "requires an architecture"):
            ProviderBuildMetadata(compiled_architectures=(), **base)
        with self.assertRaisesRegex(ValueError, "unknown compiled architecture"):
            ProviderBuildMetadata(compiled_architectures=("sm91",), **base)


if __name__ == "__main__":
    unittest.main()
