"""Budget and unbiased-prefix accounting for previously larger evaluations."""

import unittest
from collections import Counter

from cosmos_framework.evaluation.robodojo_sr_protocol import evaluation_cells, native_prefix_metrics


class ProtocolTests(unittest.TestCase):
    def test_fifteen_per_family_and_five_per_seed_including_variants(self):
        family, seeded, variants = Counter(), Counter(), Counter()
        for task, seed, quota in evaluation_cells():
            name = task.removesuffix("_random")
            family[name] += quota
            seeded[(name, seed)] += quota
            variants[task] += quota
        self.assertEqual(sum(family.values()), 60)
        self.assertEqual(set(family.values()), {15})
        self.assertEqual(set(seeded.values()), {5})
        self.assertEqual((variants["stack_bowls"], variants["stack_bowls_random"]), (8, 7))
        self.assertEqual((variants["push_T"], variants["push_T_random"]), (8, 7))

    def test_reused_native_history_counts_only_first_fixed_episodes(self):
        native = dict(
            eval_time=12,
            details={str(i): dict(success=i >= 5, score=i / 20, layout_id=100 + i) for i in reversed(range(12))},
        )
        value = native_prefix_metrics(native, 5)
        self.assertEqual(value["episodes"], 5)
        self.assertEqual(value["native_episodes"], 12)
        self.assertEqual(value["successes"], 0)
        self.assertEqual(value["score"], 10)
        self.assertEqual(value["selected_episode_indices"], [0, 1, 2, 3, 4])
        self.assertEqual(value["selected_layout_ids"], [100, 101, 102, 103, 104])
        self.assertEqual(len(native["details"]), 12)

    def test_partial_and_invalid_results(self):
        self.assertEqual(native_prefix_metrics(dict(eval_time=0, details={}), 5)["episodes"], 0)
        with self.assertRaises(ValueError):
            native_prefix_metrics(dict(eval_time=1, details={}), 5)


if __name__ == "__main__":
    unittest.main()
