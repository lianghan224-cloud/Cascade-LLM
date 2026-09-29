import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_llama31.py"
SPEC = importlib.util.spec_from_file_location("run_llama31_cli", TOOL_PATH)
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class ActiveTierCLITest(unittest.TestCase):
    def parse(self, *arguments):
        argv = ["run_llama31.py", "--checkpoint", "/not-loaded"]
        with mock.patch("sys.argv", argv + list(arguments)):
            return TOOL.parse_args()

    def test_byte_capacity_migration_watermarks_and_timeout_are_explicit(self):
        args = self.parse(
            "--kv-storage", "gpu-cpu",
            "--kv-gpu-hot-budget", "8MiB",
            "--kv-cpu-budget", "64MiB",
            "--kv-gpu-migration-slots", "2MiB",
            "--kv-cpu-migration-slots", "3MiB",
            "--kv-gpu-high-watermark", "7MiB",
            "--kv-gpu-low-watermark", "4MiB",
            "--kv-cpu-high-watermark", "56MiB",
            "--kv-cpu-low-watermark", "32MiB",
            "--kv-prefetch-timeout-ms", "250",
        )
        payload = TOOL.build_active_tier_cli_config(
            args, args.kv_gpu_hot_budget
        )
        self.assertEqual(
            set(payload),
            {
                "schema_version",
                "gpu_hot_capacity_pages_requested",
                "gpu_hot_capacity_bytes",
                "cpu_backing_capacity_bytes",
                "gpu_migration_slots_bytes",
                "cpu_migration_slots_bytes",
                "gpu_high_watermark_bytes",
                "gpu_low_watermark_bytes",
                "cpu_high_watermark_bytes",
                "cpu_low_watermark_bytes",
                "prefetch_timeout_ms",
            },
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["gpu_hot_capacity_bytes"], 8 * 1024 ** 2)
        self.assertEqual(payload["cpu_backing_capacity_bytes"], 64 * 1024 ** 2)
        self.assertEqual(payload["prefetch_timeout_ms"], 250)

    def test_page_capacity_is_resolved_from_checkpoint_geometry(self):
        args = self.parse(
            "--kv-storage", "gpu-cpu",
            "--kv-gpu-hot-pages", "4",
        )
        geometry = SimpleNamespace(
            num_hidden_layers=2,
            num_key_value_heads=1,
            head_dim=8,
        )
        resolved = TOOL.resolve_gpu_hot_budget_bytes(
            args, geometry, "bf16"
        )
        expected = 2 * 4 * 1 * 16 * 8 * 2 * 2
        self.assertEqual(resolved, expected)
        payload = TOOL.build_active_tier_cli_config(args, resolved)
        self.assertEqual(payload["gpu_hot_capacity_pages_requested"], 4)
        self.assertEqual(payload["gpu_hot_capacity_bytes"], expected)

    def test_page_and_byte_capacity_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self.parse(
                "--kv-gpu-hot-pages", "4",
                "--kv-gpu-hot-budget", "8MiB",
            )

    def test_negative_timeout_is_rejected_by_parser(self):
        with self.assertRaises(SystemExit):
            self.parse("--kv-prefetch-timeout-ms", "-1")

    def test_nonzero_timeout_is_forwarded_to_the_executor(self):
        source = TOOL_PATH.read_text(encoding="utf-8")
        self.assertNotIn("PagedKVRuntime does not consume", source)
        self.assertIn(
            "kv_prefetch_timeout_ms=args.kv_prefetch_timeout_ms", source
        )

    def test_zero_weight_prefetch_depth_is_an_explicit_sync_mode(self):
        args = self.parse("--prefetch-depth", "0")
        self.assertEqual(args.prefetch_depth, 0)


if __name__ == "__main__":
    unittest.main()
