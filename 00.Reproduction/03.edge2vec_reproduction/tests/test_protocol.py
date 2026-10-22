"""Small graph checks; no MSI data, embedding training or CV is used."""
from pathlib import Path
import math
import random
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import networkx as nx
import numpy as np
from compat import edge2vec, transition
from run_edge2vec import node_route_to_types


class ProtocolTests(unittest.TestCase):
    def test_backtracking_stays_on_actual_edges(self):
        random.seed(0)
        np.random.seed(0)
        graph = nx.Graph()
        graph.add_edge("0", "1", type=1, weight=1.0, id=1)
        route = edge2vec.edge2vec_walk(graph, 10, "0", np.ones((1, 1)), False, 1, 1)
        self.assertEqual(route, ["0", "1"] * 5)
        self.assertEqual(node_route_to_types(graph, route), ["1"] * 9)

    def test_known_pearson_matrix(self):
        matrix = np.asarray(transition.update_trans_matrix(
            [["1", "1"], ["2", "2"], ["1", "2"]], 2, 4))
        self.assertTrue(np.allclose(np.diag(matrix), 1 / (1 + np.exp(-1))))
        self.assertTrue(np.allclose([matrix[0, 1], matrix[1, 0]], 1 / (1 + np.exp(1))))

    def test_undefined_correlation_preserves_original_fallback(self):
        transition.pearson_fallback_audit.update(constant_pairs=0, nonfinite_pairs=0)
        actual = transition.pearsonr_test([0, 0, 0], [1, 2, 3])
        self.assertAlmostEqual(actual, 1 / (1 + math.exp(1)))
        self.assertEqual(transition.pearson_fallback_audit["constant_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
