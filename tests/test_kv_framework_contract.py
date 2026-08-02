import json
from pathlib import Path
import unittest

from layer_streaming import (
    KV_FRAMEWORK_CONTRACT_VERSION,
    kv_framework_contract,
    kv_framework_contract_sha256,
)


FIXTURE = Path(__file__).parent / "fixtures/kv_framework_v1.json"
EXPECTED_SHA256 = "c1b8e8cad65c0d0d1779703e24129a832100efbd5024a6cfad38841f8e30c0ed"


class KVFrameworkContractTest(unittest.TestCase):
    def test_v1_contract_is_frozen(self):
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(KV_FRAMEWORK_CONTRACT_VERSION, 1)
        self.assertEqual(kv_framework_contract(), expected)
        self.assertEqual(kv_framework_contract_sha256(), EXPECTED_SHA256)


if __name__ == "__main__":
    unittest.main()
