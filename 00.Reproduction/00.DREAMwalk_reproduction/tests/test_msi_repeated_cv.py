"""Pair isolation, historical reuse, fold checkpoints and repeat-level statistics."""
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import msi_repeated_cv as cv


class RepeatedCVTests(unittest.TestCase):
    def test_ten_negative_sets_keep_seed0_and_are_disjoint(self):
        drugs = [f"R{i:02d}" for i in range(12)]
        diseases = [f"D{i:02d}" for i in range(12)]
        positives = list(zip(drugs, diseases))
        candidates = [(r, d) for r in drugs for d in diseases if (r, d) not in positives]
        seed0 = random.Random(0).sample(candidates, len(positives))
        sets = cv.sample_negative_sets(drugs, diseases, positives, seed0, range(10))
        self.assertEqual(sets[0], seed0)
        self.assertEqual(sets, cv.sample_negative_sets(drugs[::-1], diseases[::-1], positives, seed0, range(10)))
        combined = [pair for values in sets.values() for pair in values]
        self.assertEqual(len(combined), 120)
        self.assertEqual(len(set(combined)), 120)
        self.assertFalse(set(combined) & set(positives))
        with self.assertRaisesRegex(ValueError, "canonical"):
            cv.sample_negative_sets(drugs, diseases, positives, seed0[::-1], range(10))
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            cv.sample_negative_sets(drugs, diseases, positives, seed0, range(12))
        with self.assertRaisesRegex(ValueError, "endpoint"):
            cv.sample_negative_sets(drugs, diseases, positives, [("missing", "D00")], range(10))

    def test_folds_test_every_pair_once_with_stratification(self):
        positives = [(f"R{i}", "D0") for i in range(20)]
        negatives = [(f"R{i}", "D1") for i in range(20)]
        rows = cv.assign_folds(positives, negatives, 0)
        self.assertEqual(rows, cv.assign_folds(positives[::-1], negatives, 0))
        self.assertEqual({row["pair_index"] for row in rows}, set(range(40)))
        for fold in range(1, 11):
            test = [row for row in rows if row["test_fold"] == fold]
            self.assertEqual(len(test), 4)
            self.assertEqual(sum(row["label"] for row in test), 2)
        self.assertNotEqual(rows, cv.assign_folds(positives, negatives, 1))
        with self.assertRaisesRegex(ValueError, "Unique balanced"):
            cv.assign_folds(positives, negatives[:-1], 0)

    def historical_fixture(self, root):
        source = root / "legacy"
        source.mkdir()
        embedding = root / "embedding.pkl"
        embedding.write_bytes(b"fixture embedding bytes")
        positives = [(f"R{i}", "D0") for i in range(4)]
        negatives = [(f"R{i}", "D1") for i in range(4)]
        pairs = cv.assign_folds(positives, negatives, 0, 2)
        parameters = dict(seed=0, n_estimators=500)
        cv.write_json(source / "run_config.json", dict(xgboost=parameters))
        cv.write_rows(source / "fold_assignments.csv", pairs, list(pairs[0]))
        rows = []
        for fold in (1, 2):
            rows.append(dict(fold=fold, cv_seed=0, negative_seed=0,
                             n_train=4, n_test=4, n_positive_train=2, n_positive_test=2,
                             accuracy=0.75, auroc=0.8, aupr=0.85, runtime_seconds=1.0))
        cv.write_rows(source / "pp1_fold_results.csv", rows, list(rows[0]))
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        cv.write_json(source / "pp1_evaluation_complete.json", dict(
            embedding_sha256=digest(embedding),
            fold_assignments_sha256=digest(source / "fold_assignments.csv"),
            fold_results_sha256=digest(source / "pp1_fold_results.csv")))
        return source, embedding, pairs, parameters

    def test_historical_reuse_checks_identity_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, embedding, pairs, parameters = self.historical_fixture(root)
            output = root / "new"
            rows = cv.import_seed0(source, embedding, pairs, output, "original", parameters, 2)
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["origin"] == "historical_seed0" for row in rows))
            self.assertFalse((output / "fold01/classifier.json").exists())
            self.assertEqual(rows, cv.import_seed0(source, embedding, pairs, output, "original", parameters, 2))
            with self.assertRaisesRegex(ValueError, "pair order"):
                cv.import_seed0(source, embedding, pairs[::-1], output, "original", parameters, 2)
            with self.assertRaisesRegex(ValueError, "settings differ"):
                cv.import_seed0(source, embedding, pairs, output, "original", dict(parameters, seed=1), 2)
            (output / "fold01/source_manifest.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing outputs"):
                cv.import_seed0(source, embedding, pairs, output, "original", parameters, 2)

    def test_historical_changed_embedding_or_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, embedding, pairs, parameters = self.historical_fixture(root)
            saved = embedding.read_bytes()
            embedding.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "embedding differs"):
                cv.import_seed0(source, embedding, pairs, root / "new", "original", parameters, 2)
            embedding.write_bytes(saved)
            with (source / "pp1_fold_results.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "completion record"):
                cv.import_seed0(source, embedding, pairs, root / "new", "original", parameters, 2)
            self.assertFalse((root / "new").exists())

    def test_fit_uses_only_training_rows_and_resumes_each_fold(self):
        pairs = cv.assign_folds([("R0", "D0"), ("R1", "D0")],
                                [("R0", "D1"), ("R1", "D1")], 3, 2)
        y = np.array([row["label"] for row in pairs])
        x = np.column_stack((np.arange(len(pairs)), y))
        expected_training = {row["pair_index"] for row in pairs if row["test_fold"] != 1}

        class StubClassifier:
            fits = 0

            def __init__(self, **parameters):
                self.parameters = parameters

            def fit(self, features, labels):
                StubClassifier.fits += 1
                if set(features[:, 0]) != expected_training:
                    raise AssertionError("Test examples entered training")

            def predict_proba(self, features):
                scores = np.where(features[:, 1] > 0, 0.8, 0.2)
                return np.column_stack((1 - scores, scores))

            def save_model(self, path):
                Path(path).write_text(json.dumps(self.parameters))

        with tempfile.TemporaryDirectory() as directory, patch("xgboost.XGBClassifier", StubClassifier):
            output = Path(directory)
            row = cv.fit_fold(x, y, pairs, "original", 3, 1, {"seed": 3}, output)
            self.assertEqual(row["auroc"], 1.0)
            self.assertEqual(row, cv.fit_fold(x, y, pairs, "original", 3, 1, {"seed": 3}, output))
            self.assertEqual(StubClassifier.fits, 1)
            predictions = cv.read_rows(output / "predictions.csv")
            self.assertEqual({int(p["pair_index"]) for p in predictions}, set(range(4)) - expected_training)
            (output / "predictions.csv").unlink()
            with self.assertRaisesRegex(ValueError, "missing outputs"):
                cv.fit_fold(x, y, pairs, "original", 3, 1, {"seed": 3}, output)
            (output / "metrics.json").unlink()
            cv.fit_fold(x, y, pairs, "original", 3, 1, {"seed": 3}, output)
            self.assertEqual(StubClassifier.fits, 2)
            with self.assertRaisesRegex(ValueError, "repeat seed"):
                cv.fit_fold(x, y, pairs, "original", 3, 2, {"seed": 0}, output / "other")
            with self.assertRaisesRegex(ValueError, "Misaligned"):
                cv.fit_fold(x, y[::-1], pairs, "original", 3, 2, {"seed": 3}, output / "other")

    def test_summary_uses_repeat_means_and_pairs_by_seed(self):
        fold_rows = []
        values = {("original", 0): [0.4, 0.8], ("original", 1): [0.6, 1.0],
                  ("modified", 0): [0.6, 1.0], ("modified", 1): [0.5, 0.7]}
        for (model, seed), scores in values.items():
            for fold, score in enumerate(scores, 1):
                fold_rows.append(dict(model=model, repeat_seed=seed, fold=fold,
                                      accuracy=score, auroc=score, aupr=score))
        repeats, summary, differences, deltas = cv.summarize(fold_rows[::-1], [0, 1], 2)
        original = next(row for row in summary if row["model"] == "original" and row["metric"] == "auroc")
        self.assertAlmostEqual(original["mean"], 0.7)
        self.assertAlmostEqual(original["SD"], statistics.stdev([0.6, 0.8]))
        self.assertEqual(original["repeats"], 2)
        self.assertAlmostEqual(original["CI95_high"], 0.7 + 12.7062047364 * 0.1, places=7)
        self.assertAlmostEqual(differences[0]["auroc"], 0.2)
        self.assertAlmostEqual(differences[1]["auroc"], -0.2)
        self.assertAlmostEqual(deltas[0]["mean"], 0.0)
        self.assertEqual(len(repeats), 4)
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            cv.summarize(fold_rows[:-1], [0, 1], 2)
        with self.assertRaisesRegex(ValueError, "Exactly one"):
            cv.summarize(fold_rows + [fold_rows[0]], [0, 1], 2)


if __name__ == "__main__":
    unittest.main()
