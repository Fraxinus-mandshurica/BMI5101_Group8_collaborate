"""Synthetic checks for repeated case prediction; no real data or training."""
from pathlib import Path
import random
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hd_prediction as hd
import hd_paper_prediction as paper


class NegativeSetTests(unittest.TestCase):
    def setUp(self):
        self.drugs = ["D%02d" % i for i in range(12)]
        self.diseases = [hd.DISEASE, "OTHER"]
        self.types = dict([(drug, "drug") for drug in self.drugs] +
                          [(disease, "disease") for disease in self.diseases] +
                          [("GENE", "gene")])
        self.positives = [("D00", hd.DISEASE), ("D01", "OTHER")]
        self.universe = [(drug, disease) for drug in sorted(self.drugs)
                         for disease in sorted(self.diseases)
                         if (drug, disease) not in self.positives]
        self.seed0 = random.Random(0).sample(self.universe, len(self.positives))

    def test_sampling_is_deterministic_and_preserves_saved_seed0_order(self):
        first = paper.sample_negative_sets(self.positives, self.seed0, self.types)
        second = paper.sample_negative_sets(self.positives, self.seed0, self.types)
        self.assertEqual(tuple(paper.NEGATIVE_SEEDS), tuple(range(10)))
        self.assertEqual(first, second)
        self.assertEqual(first[0], self.seed0)
        remaining = [pair for pair in self.universe if pair not in self.seed0]
        self.assertEqual(first[1], random.Random(1).sample(remaining, len(self.positives)))

    def test_sets_are_balanced_typed_and_pairwise_disjoint(self):
        samples = paper.sample_negative_sets(self.positives, self.seed0, self.types)
        self.assertEqual(set(samples), set(range(10)))
        observed = set()
        for seed in range(10):
            pairs = samples[seed]
            self.assertEqual(len(pairs), len(self.positives))
            self.assertEqual(len(pairs), len(set(pairs)))
            self.assertFalse(set(pairs) & set(self.positives))
            self.assertFalse(set(pairs) & observed)
            for drug, disease in pairs:
                self.assertEqual(self.types[drug], "drug")
                self.assertEqual(self.types[disease], "disease")
            observed.update(pairs)
        self.assertEqual(len(observed), 10 * len(self.positives))

    def test_seed0_must_match_original_seed0_sample_exactly(self):
        invalid = [list(reversed(self.seed0)), self.seed0[:1],
                   [self.seed0[0], self.seed0[0]],
                   [self.positives[0], self.seed0[1]],
                   [("GENE", hd.DISEASE), self.seed0[1]]]
        for seed0 in invalid:
            with self.subTest(seed0=seed0), self.assertRaises(ValueError):
                paper.sample_negative_sets(self.positives, seed0, self.types)

    def test_sampling_rejects_duplicate_or_wrong_type_positives(self):
        for positives in [self.positives + [self.positives[0]],
                          [(hd.DISEASE, "OTHER"), self.positives[1]],
                          [("missing", hd.DISEASE), self.positives[1]]]:
            with self.subTest(positives=positives), self.assertRaises(ValueError):
                paper.sample_negative_sets(positives, self.seed0, self.types)

    def test_sampling_requires_enough_unknown_pairs_for_ten_sets(self):
        types = {key: value for key, value in self.types.items()
                 if key not in self.drugs[6:]}
        universe = [(drug, disease) for drug in self.drugs[:6]
                    for disease in sorted(self.diseases)
                    if (drug, disease) not in self.positives]
        seed0 = random.Random(0).sample(universe, len(self.positives))
        with self.assertRaises(ValueError):
            paper.sample_negative_sets(self.positives, seed0, types)


class AggregateScoreTests(unittest.TestCase):
    def setUp(self):
        self.drugs = ["D", "C", "B", "A"]
        self.scores = {seed: np.array([0.1 + 0.01 * seed, 0.8, 0.8, 0.95])
                       for seed in range(10)}
        self.positives = [("A", hd.DISEASE), ("B", "OTHER")]
        self.negatives = {seed: [("C", hd.DISEASE)] if seed == 3 else
                         [("D", "OTHER_%d" % seed)] for seed in range(10)}

    def aggregate(self, **changes):
        arguments = dict(drugs=self.drugs, scores_by_seed=self.scores,
                         names={"A": "Drug A"}, positives=self.positives,
                         negative_sets=self.negatives)
        arguments.update(changes)
        return paper.aggregate_scores(**arguments)

    def test_aggregation_uses_mean_sample_sd_and_tied_drug_id_order(self):
        all_rows, candidates, known = self.aggregate()
        self.assertEqual([row["drug"] for row in all_rows], ["A", "B", "C", "D"])
        self.assertEqual([row["drug"] for row in candidates], ["B", "C", "D"])
        self.assertEqual([row["candidate_rank"] for row in candidates], [1, 2, 3])
        self.assertEqual([row["rank_all"] for row in candidates], [2, 3, 4])
        self.assertEqual([row["drug"] for row in known], ["A"])
        self.assertTrue(known[0]["known_in_msi"])
        self.assertEqual(known[0]["drug_name"], "Drug A")
        self.assertFalse(candidates[0]["known_in_msi"])
        row_d = all_rows[-1]
        values = np.array([self.scores[seed][0] for seed in range(10)])
        self.assertAlmostEqual(row_d["mean_score"], float(values.mean()))
        self.assertAlmostEqual(row_d["sd_score"], float(values.std(ddof=1)))
        self.assertNotAlmostEqual(row_d["sd_score"], float(values.std(ddof=0)))
        self.assertTrue(all(row["n_negative_sets"] == 10 for row in all_rows))

    def test_sampled_unknowns_remain_candidates_with_exposure_count(self):
        _, candidates, _ = self.aggregate()
        by_drug = {row["drug"]: row for row in candidates}
        self.assertIn("C", by_drug)
        self.assertEqual(by_drug["C"]["training_negative_count"], 1)
        self.assertAlmostEqual(by_drug["C"]["training_negative_fraction"], 0.1)
        self.assertEqual(by_drug["D"]["training_negative_count"], 0)
        self.assertEqual(by_drug["B"]["training_negative_count"], 0)

    def test_aggregation_rejects_missing_or_extra_seed(self):
        incomplete = {seed: scores for seed, scores in self.scores.items() if seed != 9}
        extra = dict(self.scores)
        extra[10] = self.scores[0]
        for scores in [incomplete, extra]:
            with self.subTest(seeds=sorted(scores)), self.assertRaises(ValueError):
                self.aggregate(scores_by_seed=scores)

    def test_aggregation_rejects_misaligned_duplicate_or_invalid_scores(self):
        invalid = [np.array([0.1, 0.8, 0.95]),
                   np.array([float("nan"), 0.8, 0.8, 0.95]),
                   np.array([float("inf"), 0.8, 0.8, 0.95]),
                   np.array([-0.01, 0.8, 0.8, 0.95]),
                   np.array([1.01, 0.8, 0.8, 0.95])]
        for scores in invalid:
            changed = dict(self.scores)
            changed[0] = scores
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                self.aggregate(scores_by_seed=changed)
        with self.assertRaises(ValueError):
            self.aggregate(drugs=["D", "C", "B", "B"])


if __name__ == "__main__":
    unittest.main()
