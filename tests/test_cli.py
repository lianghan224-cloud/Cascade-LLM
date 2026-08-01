from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from layer_streaming import cli
from layer_streaming.container_contracts import ImageManifest
from layer_streaming.hardware import RuntimeFeatureProfile, fake_hardware_profile
from layer_streaming.plugins import discover_plugins, quantizer_plugins


class CascadeCliTest(unittest.TestCase):
    def test_help_and_version(self):
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(["--help"]), 0)
        for command in cli.COMMANDS:
            self.assertIn(command, output.getvalue())
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(["--version"]), 0)
        self.assertEqual(output.getvalue().strip(), "0.1.0")

    def test_builtin_plugins_are_available_and_discovery_is_idempotent(self):
        first = discover_plugins()
        second = discover_plugins()
        self.assertIs(first, second)
        self.assertIn("int8_per_channel", quantizer_plugins())
        self.assertTrue(any(item.registered_name == "llama" for item in first))

    def test_doctor_reads_image_manifest_without_requiring_cuda(self):
        hardware = fake_hardware_profile("unknown")
        runtime = RuntimeFeatureProfile(
            cuda_available=False,
            cuda_graph_available=False,
            pinned_memory_available=False,
            cutlass_extension_loaded=False,
            provider_abi_versions=(),
            compiled_architectures=(),
            deterministic_mode=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "image-manifest.json"
            ImageManifest(
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
            ).write(manifest)
            output = io.StringIO()
            with mock.patch.dict(
                os.environ, {"CASCADE_IMAGE_MANIFEST": str(manifest)}
            ), mock.patch.object(
                cli.HardwareDetector,
                "detect",
                return_value=(hardware, runtime),
            ), redirect_stdout(output):
                self.assertEqual(cli.main(["doctor", "--mode", "quick"]), 0)
            payload = json.loads(output.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["image_manifest"]["cascade_version"], "0.1.0")

    def test_run_alias_and_default_output_are_stable(self):
        calls = []

        def fake_run_tool(name, arguments):
            calls.append((name, tuple(arguments)))
            return 0

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"CASCADE_RESULT_ROOT": directory}
        ), mock.patch.object(cli, "discover_plugins", return_value=()), mock.patch.object(
            cli, "run_tool", side_effect=fake_run_tool
        ):
            result = cli.main(
                [
                    "run",
                    "--checkpoint",
                    "/models/test",
                    "--backend",
                    "cutlass_w8a16",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(calls[0][0], "run_llama31.py")
        self.assertIn("fused_w8a16", calls[0][1])
        self.assertIn(str(Path(directory) / "run_report.json"), calls[0][1])

    def test_validate_runs_prefill_and_decode_compatibility_first(self):
        calls = []

        def fake_run_tool(name, arguments):
            calls.append((name, tuple(arguments)))
            return 0

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"CASCADE_RESULT_ROOT": directory}
        ), mock.patch.object(cli, "discover_plugins", return_value=()), mock.patch.object(
            cli, "run_tool", side_effect=fake_run_tool
        ):
            result = cli.main(
                [
                    "validate",
                    "--checkpoint",
                    "/models/test",
                    "--backend",
                    "fused_w8a16",
                ]
            )
        self.assertEqual(result, 0)
        self.assertEqual(
            [item[0] for item in calls],
            [
                "check_compatibility.py",
                "check_compatibility.py",
                "accept_checkpoint.py",
            ],
        )
        self.assertIn("prefill", calls[0][1])
        self.assertIn("decode", calls[1][1])


if __name__ == "__main__":
    unittest.main()
