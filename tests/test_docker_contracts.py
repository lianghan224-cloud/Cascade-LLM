import json
import os
from pathlib import Path
import tempfile
import unittest

import yaml

from layer_streaming.cli import COMMANDS
from layer_streaming.container_contracts import (
    ImageManifest,
    ProviderBundleManifest,
    load_versions,
)


ROOT = Path(__file__).resolve().parents[1]


class DockerContractsTest(unittest.TestCase):
    def test_stable_cli_and_directory_contracts(self):
        self.assertEqual(
            COMMANDS,
            (
                "doctor",
                "inspect",
                "validate",
                "run",
                "chat",
                "benchmark",
                "qualify",
                "quantize",
                "shell",
            ),
        )
        compose = yaml.safe_load((ROOT / "compose.yaml").read_text())
        service = compose["services"]["cascade"]
        rendered_volumes = "\n".join(service["volumes"])
        for target in ("/models:ro", "/config:ro", ":/cache", ":/results"):
            self.assertIn(target, rendered_volumes)
        self.assertTrue(service["read_only"])
        self.assertIn("no-new-privileges:true", service["security_opt"])
        self.assertNotIn("privileged", service)
        self.assertNotIn("ipc", service)

    def test_all_bundles_validate_and_only_sm86_is_installable(self):
        bundle_root = ROOT / "docker/provider-bundles"
        bundles = {
            path.stem: ProviderBundleManifest.read(path)
            for path in bundle_root.glob("*.json")
        }
        self.assertEqual(
            set(bundles),
            {"generic", "qualified", "sm75", "sm80", "sm86", "sm89", "sm90"},
        )
        self.assertFalse(bundles["generic"].providers)
        self.assertEqual(
            bundles["qualified"].providers,
            bundles["sm86"].providers,
        )
        provider = bundles["sm86"].providers[0]
        self.assertEqual(provider.architecture, "sm86")
        self.assertEqual(provider.provider_abi, 2)
        self.assertEqual(provider.qualification, "qualified")
        for architecture in ("sm75", "sm80", "sm89", "sm90"):
            self.assertFalse(bundles[architecture].providers)

    def test_provider_bundle_matches_capability_and_build_metadata(self):
        bundle = ProviderBundleManifest.read(
            ROOT / "docker/provider-bundles/sm86.json"
        )
        provider = bundle.providers[0]
        package_root = ROOT / provider.source / "cascade_provider"
        capability = json.loads(
            (package_root / "capability.json").read_text(encoding="utf-8")
        )
        metadata = json.loads(
            (package_root / "build_metadata.json").read_text(encoding="utf-8")
        )
        self.assertEqual(capability["provider_abi"], provider.provider_abi)
        self.assertEqual(
            capability["supported_architectures"], [provider.architecture]
        )
        self.assertEqual(metadata["abi"], provider.provider_abi)
        self.assertIn(provider.architecture, metadata["compiled_architectures"])

    def test_versions_are_complete_and_dockerfile_has_stable_stages(self):
        versions = load_versions(ROOT / "docker/versions.env")
        required = {
            "CASCADE_VERSION",
            "PYTHON_VERSION",
            "PYTORCH_VERSION",
            "CUDA_VERSION",
            "CUDA_IMAGE_VERSION",
            "TRANSFORMERS_VERSION",
            "PROVIDER_ABI",
            "EXECUTION_PLAN_SCHEMA",
            "RUN_REPORT_SCHEMA",
        }
        self.assertFalse(required.difference(versions))
        dockerfile = (ROOT / "docker/Dockerfile").read_text(encoding="utf-8")
        for stage in ("base", "builder", "runtime-builder", "runtime", "devel"):
            self.assertIn(" AS {}".format(stage), dockerfile)
        self.assertNotIn("if sm86", dockerfile.lower())
        self.assertNotIn("if sm89", dockerfile.lower())

    def test_image_manifest_round_trip(self):
        manifest = ImageManifest(
            cascade_version="0.1.0",
            git_commit="abc123",
            execution_plan_schema=1,
            run_report_schema=2,
            hardware_compatibility_schema=1,
            numerical_contract_version=1,
            python="3.10",
            torch="2.4.1",
            cuda_runtime="12.1",
            provider_bundle="generic",
            providers=(),
            image_type="runtime",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = manifest.write(Path(directory) / "image-manifest.json")
            self.assertEqual(ImageManifest.read(path), manifest)

    def test_container_scripts_are_executable(self):
        paths = (
            ROOT / "docker/entrypoint.sh",
            ROOT / "docker/healthcheck.sh",
            ROOT / "docker/scripts/install-core.sh",
            ROOT / "docker/scripts/install-provider-bundle.sh",
            ROOT / "docker/scripts/validate-image.sh",
            ROOT / "scripts/cascade-docker.sh",
        )
        for path in paths:
            self.assertTrue(os.access(str(path), os.X_OK), str(path))


if __name__ == "__main__":
    unittest.main()
