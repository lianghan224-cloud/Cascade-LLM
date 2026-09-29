import unittest

from layer_streaming.moe import compare_standard_policies


class ExpertPolicyBenchmarkTest(unittest.TestCase):
    def test_standard_matrix_is_deterministic_and_covers_required_patterns(self):
        first = compare_standard_policies(capacity_experts=2, expert_bytes=10)
        second = compare_standard_policies(capacity_experts=2, expert_bytes=10)
        self.assertEqual(first, second)
        self.assertEqual(
            {item.workload for item in first},
            {"uniform", "skewed", "hot_cold", "alternating"},
        )
        self.assertEqual({item.policy for item in first}, {"lru", "lru_frequency"})

    def test_frequency_policy_improves_hot_cold_trace(self):
        results = {
            (item.workload, item.policy): item
            for item in compare_standard_policies(capacity_experts=4, expert_bytes=10)
        }
        lru = results[("hot_cold", "lru")]
        frequency = results[("hot_cold", "lru_frequency")]
        self.assertGreater(frequency.hit_rate, lru.hit_rate)
        self.assertLess(frequency.h2d_bytes, lru.h2d_bytes)
        alternating_lru = results[("alternating", "lru")]
        alternating_frequency = results[("alternating", "lru_frequency")]
        self.assertEqual(alternating_lru.hits, alternating_frequency.hits)


if __name__ == "__main__":
    unittest.main()
