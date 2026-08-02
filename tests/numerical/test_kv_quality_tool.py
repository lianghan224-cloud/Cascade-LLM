import json
from pathlib import Path
import unittest

from tools.qualify_kv_quality import build_long_examples


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/kv_quality_suite_v1.json"


class KVQualityToolTest(unittest.TestCase):
    def test_quality_fixture_has_contract_coverage(self):
        suite = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(suite["schema_version"], 1)
        self.assertEqual(
            len(suite["perplexity_passages"])
            * suite["perplexity_eval_tokens_per_passage"],
            64,
        )
        self.assertGreaterEqual(len(suite["short_text"]), 8)
        self.assertGreaterEqual(len(build_long_examples(suite)), 4)
        self.assertGreaterEqual(len(suite["dialogue"]), 6)

    def test_long_context_answer_label_survives_choice_rotation(self):
        suite = json.loads(FIXTURE.read_text(encoding="utf-8"))
        examples = build_long_examples(suite)
        self.assertEqual([item["answer"] for item in examples], ["A", "D", "C", "B"])
        self.assertLess(
            examples[0]["filler_repetitions"],
            examples[-1]["filler_repetitions"],
        )


if __name__ == "__main__":
    unittest.main()
