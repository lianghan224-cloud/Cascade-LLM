import json
from pathlib import Path
import unittest

from layer_streaming import (
    KV_FRAMEWORK_CONTRACT_VERSION,
    kv_framework_contract,
    kv_framework_contract_sha256,
)


FIXTURE = Path(__file__).parent / "fixtures/kv_framework_v1.json"
EXPECTED_SHA256 = "0c8e71323062c7d21a18c1cd8703f549421b7b713abef6df796317f6010837dd"


class KVFrameworkContractTest(unittest.TestCase):
    def test_v1_contract_is_frozen(self):
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(KV_FRAMEWORK_CONTRACT_VERSION, 1)
        self.assertEqual(kv_framework_contract(), expected)
        self.assertEqual(kv_framework_contract_sha256(), EXPECTED_SHA256)


if __name__ == "__main__":
    unittest.main()
