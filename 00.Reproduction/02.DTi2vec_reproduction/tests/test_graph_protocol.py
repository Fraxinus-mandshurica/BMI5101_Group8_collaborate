"""Small synthetic correctness checks; no MSI inputs or model training."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_dti2vec import K_GRID, add_training_bridges, knn_union, select_k, target_similarity


class GraphProtocolTests(unittest.TestCase):
    def test_jaccard_with_empty_gene_set(self):
        sets = {"a": {"g1", "g2"}, "b": {"g2", "g3"}, "c": set()}
        scores = target_similarity(["a", "b", "c"], sets)
        np.testing.assert_allclose(scores, [[0, 1/3, 0], [1/3, 0, 0], [0, 0, 0]])

    def test_knn_ties_and_undirected_union(self):
        scores = np.array([[0, .8, .8, 0], [.8, 0, 0, .9],
                           [.8, 0, 0, .9], [0, .9, .9, 0]], dtype=np.float32)
        graph = knn_union(scores, 1).toarray()
        self.assertEqual(graph[0, 2], 0)  # tie chooses node 1; node 2 chooses node 3
        self.assertGreater(graph[0, 1], 0)
        self.assertGreater(graph[2, 3], 0)  # retained from one endpoint's choice
        np.testing.assert_array_equal(graph, graph.T)
        np.testing.assert_array_equal(np.diag(graph), np.zeros(4))

    def test_only_training_positive_bridges_enter_graph(self):
        base = csr_matrix(np.array([[0, .8, 0, 0], [.8, 0, 0, 0],
                                    [0, 0, 0, .6], [0, 0, .6, 0]], dtype=np.float32))
        pairs = [("d0", "i0"), ("d1", "i1"), ("d0", "i1")]
        labels = np.array([1, 1, 0])
        graph = add_training_bridges(base, pairs, labels, [0, 2],
                                     {"d0": 0, "d1": 1, "i0": 2, "i1": 3}).toarray()
        self.assertEqual(graph[0, 2], 1)
        self.assertEqual(graph[2, 0], 1)
        self.assertEqual(graph[1, 3], 0)  # held-out positive
        self.assertEqual(graph[0, 3], 0)  # training negative
        self.assertEqual(base[0, 2], 0)  # shared similarity graph remains unchanged

    def test_k_selection_uses_inner_aupr_and_smallest_tie(self):
        rows = [dict(k=k, aupr=.8 if k in (10, 20) else .7,
                     outer_test_aupr=.99 if k == 50 else .1)
                for k in K_GRID for _ in range(3)]
        chosen, _ = select_k(rows)
        self.assertEqual(chosen, 10)


if __name__ == "__main__":
    unittest.main()
