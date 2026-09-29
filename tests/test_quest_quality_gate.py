import inspect
import unittest

from layer_streaming.kv_policy import KVAccuracy, KVSelectionPolicy
from tools.qualify_kv_quality import quality_kv_policy


class QuestQualityGateTest(unittest.TestCase):
    def test_dense_reference_policy_remains_exact(self):
        policy = quality_kv_policy(
            "reference_paged_exact", 16, "bf16"
        )
        self.assertEqual(policy.accuracy, KVAccuracy.EXACT)
        self.assertEqual(policy.selection, KVSelectionPolicy.DENSE)
        self.assertEqual(policy.rgkv_scorer, "cpu_reference")

    def test_tensorized_candidate_policy_is_explicit_sparse(self):
        policy = quality_kv_policy(
            "sm86",
            16,
            "bf16",
            quest_scorer="torch_tensorized",
            page_budget=32,
            recent_window=128,
        )
        self.assertEqual(policy.accuracy, KVAccuracy.SPARSE)
        self.assertEqual(policy.selection, KVSelectionPolicy.RGKV)
        self.assertEqual(policy.rgkv_scorer, "torch_tensorized")
        self.assertEqual(policy.page_budget, 32)
        self.assertEqual(policy.recent_window, 128)

    def test_invalid_quality_policy_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, "positive page budget"):
            quality_kv_policy(
                "sm86",
                16,
                "bf16",
                quest_scorer="torch_tensorized",
            )
        with self.assertRaisesRegex(ValueError, "budgets require"):
            quality_kv_policy(
                "sm86", 16, "bf16", page_budget=4
            )

    def test_runner_source_does_not_publish_local_checkpoint_path(self):
        from tools import qualify_kv_quality

        source = inspect.getsource(qualify_kv_quality.main)
        self.assertNotIn("args.checkpoint.resolve()", source)
        self.assertIn("local_path_redacted", source)
        self.assertIn("evaluate_model_quality", source)


if __name__ == "__main__":
    unittest.main()
