"""Synthetic protocol checks; no MSI data are loaded and no model is trained."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "src/hd_prediction.py"
SPEC = importlib.util.spec_from_file_location("hd_prediction", MODULE)
hd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hd)


class HDPredictionTests(unittest.TestCase):
    def setUp(self):
        self.types = {"A": "drug", "B": "drug", "C": "drug", "D": "drug",
                      hd.DISEASE: "disease", "OTHER": "disease"}
        self.positives = [("A", hd.DISEASE), ("B", "OTHER")]
        self.negatives = [("C", hd.DISEASE), ("D", "OTHER")]

    def test_partition_withholds_all_target_labels(self):
        training, held_out = hd.partition_pairs(self.positives, self.negatives, self.types)
        self.assertEqual([(r["drug"], r["disease"], r["label"]) for r in training],
                         [("B", "OTHER", 1), ("D", "OTHER", 0)])
        self.assertEqual([(r["drug"], r["label"]) for r in held_out], [("A", 1), ("C", 0)])
        self.assertEqual({r["pair_index"] for r in training + held_out}, set(range(4)))
        self.assertFalse(any(r["disease"] == hd.DISEASE for r in training))

    def test_partition_rejects_duplicates_overlap_and_wrong_endpoints(self):
        cases = [
            (self.positives + [self.positives[0]], self.negatives),
            (self.positives, self.negatives + [self.negatives[0]]),
            (self.positives, self.negatives + [self.positives[0]]),
            ([("OTHER", hd.DISEASE)], self.negatives),
            ([("A", "B")], self.negatives),
        ]
        for positives, negatives in cases:
            with self.subTest(positives=positives, negatives=negatives), self.assertRaises(ValueError):
                hd.partition_pairs(positives, negatives, self.types)

    def test_features_use_drug_minus_disease(self):
        vectors = {"A": np.array([5.0, 8.0]), "B": np.array([1.0, -2.0]),
                   hd.DISEASE: np.array([2.0, 3.0])}
        actual = hd.pair_features(vectors, [("A", hd.DISEASE), ("B", hd.DISEASE)])
        np.testing.assert_array_equal(actual, np.array([[3, 5], [-1, -5]], dtype=np.float32))
        self.assertEqual(actual.dtype, np.float32)
        vectors["A"][0] = np.nan
        with self.assertRaises(ValueError):
            hd.pair_features(vectors, [("A", hd.DISEASE)])

    def test_ranking_excludes_known_but_keeps_sampled_negatives(self):
        _, held_out = hd.partition_pairs(self.positives, self.negatives, self.types)
        all_rows, candidates, known = hd.rank_drugs(
            ["D", "C", "B", "A"], [0.2, 0.8, 0.8, 0.9], {"A": "Drug A"}, held_out)
        self.assertEqual([r["drug"] for r in all_rows], ["A", "B", "C", "D"])
        self.assertEqual([r["drug"] for r in candidates], ["B", "C", "D"])
        self.assertEqual([r["candidate_rank"] for r in candidates], [1, 2, 3])
        self.assertEqual([r["rank_all"] for r in candidates], [2, 3, 4])
        self.assertEqual([r["drug"] for r in known], ["A"])
        self.assertEqual(known[0]["drug_name"], "Drug A")
        self.assertFalse(candidates[0]["sampled_negative_in_benchmark"])
        self.assertTrue(candidates[1]["sampled_negative_in_benchmark"])
        self.assertFalse(candidates[1]["known_in_msi"])

    def test_ranking_rejects_invalid_or_misaligned_scores(self):
        for drugs, scores in [(["B"], [float("nan")]), (["B"], [1.1]),
                              (["B", "B"], [0.1, 0.2]), (["B"], [])]:
            with self.subTest(drugs=drugs, scores=scores), self.assertRaises(ValueError):
                hd.rank_drugs(drugs, scores, {}, [])

    def test_saved_stage_detects_changed_missing_or_different_run_output(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            artifact = results / "scores.csv"
            artifact.write_text("drug,score\nA,0.2\n", encoding="utf-8")
            self.assertFalse(hd.check_stage(results, "predict", "fingerprint"))
            hd.finish_stage(results, "predict", "fingerprint", [artifact.name])
            self.assertTrue(hd.check_stage(results, "predict", "fingerprint"))
            with self.assertRaises(RuntimeError):
                hd.check_stage(results, "predict", "different")
            artifact.write_text("drug,score\nA,0.9\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                hd.check_stage(results, "predict", "fingerprint")
            artifact.unlink()
            with self.assertRaises(RuntimeError):
                hd.check_stage(results, "predict", "fingerprint")

    def test_open_run_reuses_identity_and_preserves_existing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "run"
            fingerprint = hd.open_run(results, {"seed": 0, "disease": hd.DISEASE})
            original = (results / "run_config.json").read_bytes()
            self.assertEqual(hd.open_run(results, {"disease": hd.DISEASE, "seed": 0}), fingerprint)
            self.assertEqual((results / "run_config.json").read_bytes(), original)
            with self.assertRaises(RuntimeError):
                hd.open_run(results, {"seed": 1, "disease": hd.DISEASE})
            self.assertEqual((results / "run_config.json").read_bytes(), original)

    def test_open_run_preserves_unregistered_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            artifact = results / "existing.csv"
            artifact.write_text("preserve me", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                hd.open_run(results, {"seed": 0})
            self.assertEqual(artifact.read_text(encoding="utf-8"), "preserve me")
            self.assertFalse((results / "run_config.json").exists())


if __name__ == "__main__":
    unittest.main()
