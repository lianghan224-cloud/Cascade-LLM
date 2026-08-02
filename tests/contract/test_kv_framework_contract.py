import json
from pathlib import Path
import subprocess
import sys
import unittest

from layer_streaming import (
    KV_FRAMEWORK_CONTRACT_VERSION,
    kv_framework_contract,
    kv_framework_contract_sha256,
    KVPolicy,
    PagedAttentionBackend,
    PagedKVKernelBackend,
    PagedKVRuntime,
)


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/kv_framework_v1.json"
EXPECTED_SHA256 = "c1b8e8cad65c0d0d1779703e24129a832100efbd5024a6cfad38841f8e30c0ed"


class KVFrameworkContractTest(unittest.TestCase):
    def test_v1_contract_is_frozen(self):
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(KV_FRAMEWORK_CONTRACT_VERSION, 1)
        self.assertEqual(kv_framework_contract(), expected)
        self.assertEqual(kv_framework_contract_sha256(), EXPECTED_SHA256)

    def test_production_import_graph_excludes_legacy_and_aliases(self):
        code = """
import sys
import layer_streaming
assert 'layer_streaming.experimental' not in sys.modules
assert not hasattr(layer_streaming, 'KVCacheManager')
assert not hasattr(layer_streaming, 'PagedAttentionProvider')
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runtime_remains_a_thin_coordinator(self):
        project_root = Path(__file__).resolve().parents[2]
        lines = (
            project_root / "layer_streaming/kv/runtime.py"
        ).read_text(encoding="utf-8").splitlines()
        self.assertLessEqual(len(lines), 500)

    def test_frozen_interfaces_expose_versioned_metadata(self):
        policy = KVPolicy()
        self.assertEqual(policy.schema_version, 1)
        self.assertEqual(policy.provider_abi, 1)
        self.assertEqual(policy.qualification_status, "declared")
        self.assertEqual(policy.capability()["schema_version"], 1)
        for interface in (
            PagedKVRuntime,
            PagedKVKernelBackend,
            PagedAttentionBackend,
        ):
            self.assertEqual(interface.schema_version, 1)
            self.assertTrue(hasattr(interface, "capability"))
            self.assertTrue(hasattr(interface, "provider_abi"))
            self.assertTrue(hasattr(interface, "qualification_status"))


if __name__ == "__main__":
    unittest.main()
