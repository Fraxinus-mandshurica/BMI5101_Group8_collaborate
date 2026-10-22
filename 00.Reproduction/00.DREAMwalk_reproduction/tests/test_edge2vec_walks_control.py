"""Tiny fixtures only: no MSI conversion, model training or evaluation."""
import io
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_edge2vec_walks_control as control


class WalkConversionTest(unittest.TestCase):
    def setUp(self):
        # Numeric order differs from both the real IDs and prefixed token IDs.
        self.nodes = ["GENE-A", "DRUG-A", "DIS-A"]
        self.tokens = {"d19": "DRUG-A", "i2": "DIS-A", "g7": "GENE-A"}
        self.types = {"DRUG-A": "drug", "DIS-A": "disease", "GENE-A": "gene"}
        self.mapping = control.conversion_map(self.nodes, self.tokens, self.types)

    def test_preserves_order_and_real_node_paths_with_historical_serialization(self):
        numeric_walks = "2 0 1\n0 1 0\n1 2 1\n"
        output = io.StringIO()
        counts = control.recode_stream(io.StringIO(numeric_walks), output, self.mapping, 3, 3, 1)
        self.assertEqual(counts, dict(walks=3, tokens=9, nodes=3, walk_length=3))
        self.assertEqual(output.getvalue(), "i2 g7 d19\ng7 d19 g7\nd19 i2 d19")
        original_paths = [[self.nodes[int(token)] for token in line.split()]
                          for line in numeric_walks.splitlines()]
        recovered_paths = [[self.tokens[token] for token in line.split()]
                           for line in output.getvalue().splitlines()]
        self.assertEqual(recovered_paths, original_paths)
        self.assertFalse(output.getvalue().endswith("\n"))

    def test_rejects_bad_indices_lengths_and_start_counts(self):
        for corpus in ("0 9 1\n1 2 0\n2 1 0\n", "0 1\n1 2 0\n2 0 1\n",
                       "0 1 2\n0 1 2\n2 0 1\n"):
            with self.subTest(corpus=corpus), self.assertRaises(ValueError):
                control.recode_stream(io.StringIO(corpus), io.StringIO(), self.mapping, 3, 3, 1)

    def test_rejects_nonbijective_or_wrong_type_token_map(self):
        for tokens in ({"d19": "DRUG-A", "i2": "DIS-A", "g7": "DRUG-A"},
                       {"d19": "DRUG-A", "g2": "DIS-A", "g7": "GENE-A"}):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                control.conversion_map(self.nodes, tokens, self.types)

    def test_binary_copy_preserves_bytes_and_execute_mode_and_refuses_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "HetSG"
            source.write_bytes(b"exact historical binary fixture\x00")
            os.chmod(source, 0o755)
            control.copy_exact(source, target)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)
            control.copy_exact(source, target)
            target.write_bytes(b"different")
            with self.assertRaises(RuntimeError):
                control.copy_exact(source, target)


class PairedComparisonTest(unittest.TestCase):
    def test_aligns_fold_ids_and_computes_current_minus_reference(self):
        baseline = [dict(metric=metric, mean=0.4) for metric in control.METRICS]
        current = [dict(metric=metric, mean=0.6) for metric in control.METRICS]
        reference_folds = [dict(fold=i, accuracy=i/20, auroc=i/20, aupr=i/20) for i in range(1, 11)]
        new_folds = [dict(fold=i, accuracy=i/20+0.1, auroc=i/20+0.1, aupr=i/20+0.1)
                     for i in reversed(range(1, 11))]
        summary, deltas = control.comparison_rows(baseline, current, reference_folds, new_folds)
        self.assertEqual(len(deltas), 30)
        for row in summary:
            self.assertAlmostEqual(row["edge2vec_minus_dreamwalk"], 0.2)
        for row in deltas:
            self.assertAlmostEqual(row["edge2vec_minus_dreamwalk"], 0.1)
        with self.assertRaises(RuntimeError):
            control.comparison_rows(baseline, current, reference_folds, new_folds[:-1] + [new_folds[0]])


if __name__ == "__main__":
    unittest.main()
