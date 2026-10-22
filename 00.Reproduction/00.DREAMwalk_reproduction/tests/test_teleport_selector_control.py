"""Tiny selector/spawn fixtures only; no MSI generation, EM or model fitting."""
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_teleport_selector_control as control
import teleport_selector_diagnostic as selector
import networkx as nx
import numpy as np


class SelectorTest(unittest.TestCase):
    def weighted_fixture(self):
        graph = nx.MultiGraph()
        graph.add_edge(0, 1, weight=2)
        graph.add_edge(0, 1, weight=3)
        graph.add_edge(0, 2, weight=5)
        graph.add_edge(0, 3, weight=10)
        return graph

    def test_first_hit_is_not_overwritten_and_random_calls_are_unchanged(self):
        graph = self.weighted_fixture()
        # Integer node IDs give set iteration 1,2,3 in this fixture. Use the
        # observed order rather than assuming how string hashes are ordered.
        order = list(set([1, 2, 3]))
        with patch.object(selector.ge.random, "shuffle") as shuffle, \
                patch.object(selector.ge.np.random, "rand", return_value=0.01) as draw:
            fixed = selector.teleport_operation_fixed(0, graph)
            self.assertEqual(fixed, order[0])
            shuffle.assert_called_once()
            draw.assert_called_once_with()
        with patch.object(selector.ge.random, "shuffle") as shuffle, \
                patch.object(selector.ge.np.random, "rand", return_value=0.01) as draw:
            original = selector.ge._teleport_operation(0, graph)
            self.assertEqual(original, order[-1])
            shuffle.assert_called_once()
            draw.assert_called_once_with()
        self.assertNotEqual(fixed, original)

    def test_parallel_edges_contribute_separately_to_threshold(self):
        graph = self.weighted_fixture()
        self.assertEqual(list(set([1, 2, 3])), [1, 2, 3])
        for fraction, expected in ((0.20, 1), (0.25, 1), (0.30, 2), (0.75, 3)):
            with self.subTest(fraction=fraction), patch.object(selector.ge.random, "shuffle"), \
                    patch.object(selector.ge.np.random, "rand", return_value=fraction):
                self.assertEqual(selector.teleport_operation_fixed(0, graph), expected)

    def test_zero_total_preserves_false_and_consumes_no_numpy_draw(self):
        graph = nx.MultiGraph()
        graph.add_edge(0, 1, weight=0)
        with patch.object(selector.ge.np.random, "rand") as draw:
            self.assertIs(selector.teleport_operation_fixed(0, graph), False)
            draw.assert_not_called()

    def test_real_spawn_worker_installs_and_calls_fixed_selector(self):
        graph = nx.MultiGraph()
        similarity = nx.MultiGraph()
        for a, b in ((0, 1), (1, 2), (0, 2)):
            graph.add_edge(a, b, weight=1, type=1)
            similarity.add_edge(a, b, weight=1)
        # Exactly three length-3 fixture paths, not an MSI walk run. Tau=1
        # forces one actual selector call after each initial molecular edge.
        args = (0, [0, 1, 2], graph, similarity, np.ones((1, 1)), 1, 1, 3, 1)
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            walks, audit = pool.apply_async(selector.worker_walks, args).get(timeout=45)
        self.assertEqual(len(walks), 3)
        self.assertTrue(all(len(path) == 3 for path in walks))
        self.assertNotEqual(audit["pid"], os.getpid())
        self.assertTrue(audit["selector_is_fixed"])
        self.assertEqual(audit["selector_calls"], 3)
        self.assertEqual(audit["selector"], selector.SELECTOR_NAME)
        checked = control.checked_worker_audit([audit], 1, control.sha256(control.SELECTOR_MODULE))
        self.assertEqual(checked["selector_calls"], 3)


class RunnerRoutingTest(unittest.TestCase):
    def test_reuses_matrix_and_worker_route_without_em(self):
        # All walk generation here is stubbed. The only actual upstream walks
        # in this suite are the three tiny spawn paths above.
        nodes = ["DRUG", "DISEASE", "90"]
        types = [dict(node="DRUG", type="drug"), dict(node="DISEASE", type="disease"),
                 dict(node="90", type="gene")]
        graph = SimpleNamespace(nodes=lambda: nodes)
        similarity = SimpleNamespace(number_of_edges=lambda: 1)
        walks = [["DRUG", "DISEASE", "90"], ["DISEASE", "90", "DRUG"], ["90", "DRUG", "DISEASE"]]
        audit = dict(task_id=0, pid=os.getpid()+1, selector=selector.SELECTOR_NAME,
                     selector_is_fixed=True, selector_calls=3,
                     selector_source_sha256=control.sha256(control.SELECTOR_MODULE))
        settings = dict(control.WALK_SETTINGS, num_walks=1, walk_length=3)
        with tempfile.TemporaryDirectory() as directory:
            data, results = Path(directory)/"data", Path(directory)/"results"
            data.mkdir()
            results.mkdir()
            (data/"HetSG").write_bytes(b"never executed")
            control.save_json(data/"token_to_node.json", {"d1": "DRUG", "i2": "DISEASE", "g3": "90"})
            control.save_json(results/"edge_transition_matrix.json", [[0.7]])
            control.save_json(results/"compiler_complete.json", dict(fingerprint="fixture",
                sha256=control.sha256(data/"HetSG"), token_map_sha256=control.sha256(data/"token_to_node.json"),
                matrix_sha256=control.sha256(results/"edge_transition_matrix.json")))
            sim_path = Path(directory)/"similarity.tsv"
            sim_path.write_text("fixture")
            with patch.object(control, "WALK_SETTINGS", settings), patch.object(control, "SOURCE_NODES", 3), \
                    patch.object(control.previous, "read_rows", return_value=types), \
                    patch.object(control.previous, "SIMILARITY", sim_path), \
                    patch.object(selector.ge, "set_seed") as seed, \
                    patch.object(selector.ge, "read_graph", side_effect=[graph, similarity]), \
                    patch.object(selector.ge, "train_edgetype_transition_matrix", side_effect=AssertionError("NO EM")), \
                    patch("parmap.map", return_value=[(walks, audit)]) as mapped:
                control.generate_corpus(data, results, "fixture")
                first = (data/"walks.txt").read_bytes()
                control.generate_corpus(data, results, "fixture")
                self.assertEqual((data/"walks.txt").read_bytes(), first)
                seed.assert_called_once_with(0)
                self.assertIs(mapped.call_args.args[0], selector.worker_walks)
                self.assertEqual(mapped.call_args.args[-1], 0.3)
                np.testing.assert_array_equal(mapped.call_args.args[5], [[0.7]])
                mapped.assert_called_once()
            meta = json.loads((results/"walks_complete.json").read_text())
            self.assertTrue(meta["em_reused"])
            self.assertEqual(meta["selector_calls"], 3)
            self.assertEqual(meta["walks"], 3)

    def test_comparison_sign_and_fold_alignment(self):
        original = [dict(metric=m, mean=0.8) for m in control.METRICS]
        fixed = [dict(metric=m, mean=0.9) for m in control.METRICS]
        rows = control.comparison_rows(original, fixed)
        self.assertAlmostEqual(rows[0]["fixed_minus_original"], 0.1)
        one = [dict(fold=i, accuracy=0.8, auroc=0.8, aupr=0.8) for i in range(1, 11)]
        two = [dict(fold=i, accuracy=0.9, auroc=0.9, aupr=0.9) for i in reversed(range(1, 11))]
        deltas = control.paired_fold_rows(one, two)
        self.assertEqual(deltas[0]["fold"], 1)
        self.assertAlmostEqual(deltas[0]["auroc_fixed_minus_original"], 0.1)
        with self.assertRaises(ValueError):
            control.paired_fold_rows(one, two[:-1])

    def test_worker_audit_requires_actual_child_selector_calls(self):
        audit = dict(task_id=0, pid=os.getpid()+1, selector=selector.SELECTOR_NAME,
                     selector_is_fixed=True, selector_calls=0, selector_source_sha256="sha")
        with self.assertRaises(RuntimeError):
            control.checked_worker_audit([audit], 1, "sha")
        audit["selector_calls"] = 1
        audit["selector_is_fixed"] = False
        with self.assertRaises(RuntimeError):
            control.checked_worker_audit([audit], 1, "sha")


if __name__ == "__main__":
    unittest.main()
