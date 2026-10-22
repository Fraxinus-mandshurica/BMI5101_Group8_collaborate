"""Tiny fixtures for the saved-embedding control; never fit an estimator."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_gensim_embedding_control as control


class SavedEmbeddingTest(unittest.TestCase):
    def test_real_numeric_gene_id_and_vectors_are_used_without_remapping(self):
        vectors = {"90": object(), "DRUG-A": object()}
        types = {"90": "gene", "DRUG-A": "drug"}
        pairs = [("DRUG-A", "90")]
        with patch.object(control.previous, "load_embedding", return_value=vectors) as loader:
            loaded = control.load_source_vectors(types, pairs)
        loader.assert_called_once_with(control.SOURCE_EMBEDDING, types, pairs)
        self.assertIs(loaded, vectors)
        self.assertIs(loaded["90"], vectors["90"])

    def test_equal_count_with_wrong_node_universe_is_rejected(self):
        vectors = {"WRONG-GENE": object(), "DRUG-A": object()}
        with patch.object(control.previous, "load_embedding", return_value=vectors):
            with self.assertRaises(RuntimeError):
                control.load_source_vectors({"90": "gene", "DRUG-A": "drug"}, [])

    def test_fold_check_rejects_changed_negative_pair_order(self):
        positives = {("drug" + str(i), "positive") for i in range(20)}
        negatives = [("drug" + str(i), "negative") for i in range(20)]
        rows = control.pair_fold_rows(positives, negatives)
        saved = [{key: str(value) for key, value in row.items()} for row in rows]
        with patch.object(control.previous, "read_rows", return_value=saved):
            control.require_saved_folds(rows)
            reordered = control.pair_fold_rows(positives, list(reversed(negatives)))
            with self.assertRaises(RuntimeError):
                control.require_saved_folds(reordered)

    def test_comparison_columns_and_sign_follow_gensim_minus_cpp(self):
        reference = [dict(metric=metric, mean=0.4) for metric in control.METRICS]
        current = [dict(metric=metric, mean=0.6) for metric in control.METRICS]
        before = [dict(fold=i, accuracy=i/20, auroc=i/20, aupr=i/20) for i in range(1, 11)]
        after = [dict(fold=i, accuracy=i/20+0.1, auroc=i/20+0.1, aupr=i/20+0.1)
                 for i in reversed(range(1, 11))]
        summary, deltas = control.comparison_rows(reference, current, before, after)
        self.assertEqual(list(summary[0]), ["metric", "cpp_pp1", "gensim_sg", "gensim_minus_cpp_pp1"])
        self.assertAlmostEqual(summary[0]["gensim_minus_cpp_pp1"], 0.2)
        self.assertEqual(deltas[0]["fold"], 1)
        self.assertAlmostEqual(deltas[0]["gensim_minus_cpp_pp1"], 0.1)


if __name__ == "__main__":
    unittest.main()
