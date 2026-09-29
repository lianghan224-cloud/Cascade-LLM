import unittest

from tools import benchmark_rgkv_e2e as runner


class RGKVE2ERunnerTest(unittest.TestCase):
    def test_plan_covers_context_decode_budget_recent_matrix(self):
        cases = runner.planned_cases()
        self.assertEqual(len(cases), 3 * 2 * 4 * 4)
        self.assertEqual({case["context_tokens"] for case in cases}, {2048, 8192, 16384})

    def test_gate_requires_zero_cpu_sync_and_positive_e2e_gain(self):
        metrics = {
            "dense_attention_ms": 5.0,
            "rgkv_score_ms": 1.0,
            "rgkv_topk_ms": 1.0,
            "rgkv_prefetch_ms": 1.0,
            "rgkv_attention_ms": 2.0,
            "dense_decode_ms": 12.0,
            "rgkv_decode_ms": 8.0,
            "dense_h2d_kv_bytes": 100,
            "rgkv_h2d_kv_bytes": 50,
            "rgkv_index_bytes": 20,
            "rgkv_cpu_sync_count": 0,
        }
        status, missing, result = runner.evaluate({"metrics": metrics})
        self.assertEqual(status, "PASS")
        self.assertFalse(missing)
        self.assertEqual(result["e2e_gain_ms"], 4.0)
        metrics["rgkv_cpu_sync_count"] = 1
        self.assertEqual(runner.evaluate({"metrics": metrics})[0], "FAIL")

    def test_missing_component_cannot_pass(self):
        status, missing, _ = runner.evaluate({"metrics": {}})
        self.assertEqual(status, "FAIL")
        self.assertIn("rgkv_prefetch_ms", missing)


if __name__ == "__main__":
    unittest.main()
