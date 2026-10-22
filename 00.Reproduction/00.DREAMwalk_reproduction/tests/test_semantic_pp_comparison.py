"""Tiny fixtures and mocked upstream calls; no MSI walks or model fitting."""
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_semantic_pp_comparison as control


class SemanticCorpusTest(unittest.TestCase):
    def test_preserves_type_prefixed_paths_and_historical_serialization(self):
        token_map = {"d17": "DRUG", "i2": "DISEASE", "g9": "90"}
        types = {"DRUG": "drug", "DISEASE": "disease", "90": "gene"}
        nodes = control.checked_tokens(token_map, types, types)
        walks = [["DRUG", "DRUG", "90"], ["90", "DISEASE", "DRUG"],
                 ["DISEASE", "DISEASE", "90"]]
        output = io.StringIO()
        meta = control.write_corpus(walks, output, nodes, 3, 3, 1)
        self.assertEqual(output.getvalue(), "d17 d17 g9\ng9 i2 d17\ni2 i2 g9")
        self.assertEqual(meta["adjacent_node_type_pairs"]["d->d"], 1)
        self.assertEqual(meta["adjacent_node_type_pairs"]["i->i"], 1)
        self.assertEqual(meta["tokens"], 9)
        recovered = [[token_map[token] for token in line.split()] for line in output.getvalue().splitlines()]
        self.assertEqual(recovered, walks)

    def test_rejects_wrong_start_counts_and_token_types(self):
        with self.assertRaises(ValueError):
            control.write_corpus([["A", "B"], ["A", "B"]], io.StringIO(), {"A": "d1", "B": "g2"}, 2, 2, 1)
        with self.assertRaises(ValueError):
            control.checked_tokens({"g1": "A"}, {"A": "drug"}, ["A"])

    def test_tau_reaches_upstream_and_completed_corpus_is_reused(self):
        # All generation calls are stubs on a three-node fixture. This tests
        # routing of tau and persistence, without running the upstream walker.
        nodes = ["DRUG", "DISEASE", "90"]
        types = [dict(node="DRUG", type="drug"), dict(node="DISEASE", type="disease"),
                 dict(node="90", type="gene")]
        graph = SimpleNamespace(nodes=lambda: nodes)
        semantic = SimpleNamespace(number_of_edges=lambda: 1)
        walks = [["DRUG", "DRUG", "90"], ["DISEASE", "DISEASE", "90"], ["90", "DRUG", "DISEASE"]]
        matrix = SimpleNamespace(tolist=lambda: [[1]])
        ge = SimpleNamespace(set_seed=Mock(), read_graph=Mock(side_effect=[graph, semantic]),
                             train_edgetype_transition_matrix=Mock(return_value=matrix),
                             generate_DREAMwalk_paths=Mock(return_value=walks))
        package = ModuleType("DREAMwalk")
        package.generate_embeddings = ge
        settings = dict(control.WALK_SETTINGS, num_walks=1, walk_length=3)
        with tempfile.TemporaryDirectory() as directory:
            data, results = Path(directory)/"data", Path(directory)/"results"
            data.mkdir()
            results.mkdir()
            (data/"HetSG").write_bytes(b"binary fixture; never executed")
            control.save_json(data/"token_to_node.json", {"d17": "DRUG", "i2": "DISEASE", "g9": "90"})
            control.save_json(results/"compiler_complete.json", dict(fingerprint="fixture",
                sha256=control.sha256(data/"HetSG"), token_map_sha256=control.sha256(data/"token_to_node.json")))
            similarity = Path(directory)/"similarity.tsv"
            similarity.write_text("semantic fixture")
            with patch.dict(sys.modules, {"DREAMwalk": package}), patch.object(control, "SOURCE_NODES", 3), \
                    patch.object(control, "WALK_SETTINGS", settings), \
                    patch.object(control.previous, "read_rows", return_value=types), \
                    patch.object(control.previous, "SIMILARITY", similarity):
                control.generate_corpus(data, results, "fixture")
                before = (data/"walks.txt").read_bytes()
                control.generate_corpus(data, results, "fixture")
                self.assertEqual((data/"walks.txt").read_bytes(), before)
            ge.set_seed.assert_called_once_with(0)
            ge.generate_DREAMwalk_paths.assert_called_once_with(graph, semantic, matrix, 1, 1, 1, 3, 0.3, 4)
            self.assertEqual(ge.train_edgetype_transition_matrix.call_args.args[0], 5)
            marker = json.loads((results/"walks_complete.json").read_text())
            self.assertEqual(marker["tp_factor"], 0.3)
            self.assertEqual(marker["walks"], 3)


class PpEffectTest(unittest.TestCase):
    def test_signs_of_pp_effect_and_difference_of_effects(self):
        def summary(value):
            return [dict(metric=metric, mean=value) for metric in control.METRICS]
        main, interaction = control.effect_rows(summary(.9), summary(.8), summary(.7), summary(.75))
        self.assertAlmostEqual(main[0]["pp1_minus_pp0"], .1)
        self.assertAlmostEqual(interaction[0]["no_teleport_pp1_minus_pp0"], -.05)
        self.assertAlmostEqual(interaction[0]["pp_effect_difference"], .15)

    def test_fold_alignment_uses_ids_and_not_row_order(self):
        one = [dict(fold=i, accuracy=i/20+.1, auroc=i/20+.1, aupr=i/20+.1) for i in range(1, 11)]
        zero = [dict(fold=i, accuracy=i/20, auroc=i/20, aupr=i/20) for i in reversed(range(1, 11))]
        rows = control.paired_fold_rows(one, zero)
        self.assertAlmostEqual(rows[0]["auroc_pp1_minus_pp0"], .1)
        with self.assertRaises(ValueError):
            control.paired_fold_rows(one, zero[:-1])


if __name__ == "__main__":
    unittest.main()
