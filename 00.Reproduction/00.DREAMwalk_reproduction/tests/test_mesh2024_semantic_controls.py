"""Tiny routing/checkpoint fixtures; no real walks, C++ or model fitting."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import mesh2024_semantic_controls as run


class CorrectedSemanticControlsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.results = self.root / "data", self.root / "results"
        self.data.mkdir()
        self.results.mkdir()
        (self.data / "HetSG").write_bytes(b"original binary fixture")
        self.save(self.data / "token_to_node.json", {"d0": "drug", "i1": "disease"})
        self.save(self.results / "edge_transition_matrix.json", [[1]])
        self.save(self.results / "compiler_complete.json", dict(
            fingerprint="fixture", sha256=run.sha256(self.data / "HetSG"),
            token_map_sha256=run.sha256(self.data / "token_to_node.json"),
            matrix_sha256=run.sha256(self.results / "edge_transition_matrix.json")))
        self.similarity = self.root / "corrected2024.tsv"
        self.similarity.write_text("drug\tdisease\t0.8\n")

    def save(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def test_original_walker_uses_new_graph_and_completed_corpus_is_reused(self):
        graph = SimpleNamespace(nodes=lambda: ["drug", "disease"])
        semantic = SimpleNamespace(number_of_edges=lambda: 1)
        matrix = SimpleNamespace(tolist=lambda: [[1]])
        ge = SimpleNamespace(set_seed=Mock(), read_graph=Mock(side_effect=[graph, semantic]),
                             train_edgetype_transition_matrix=Mock(return_value=matrix),
                             generate_DREAMwalk_paths=Mock(return_value=[["drug", "disease"], ["disease", "drug"]]))
        package = ModuleType("DREAMwalk")
        package.generate_embeddings = ge
        settings = dict(run.WALK_SETTINGS, num_walks=1, walk_length=2)
        old_similarity = run.previous.SIMILARITY
        old_baselines = run.shared.REFERENCE_RESULTS, run.teleport.REFERENCE_RESULTS
        with patch.dict(sys.modules, {"DREAMwalk": package}), \
                patch.object(run, "WALK_SETTINGS", settings), patch.object(run, "SOURCE_NODES", 2), \
                patch.object(run.inputs, "SIMILARITY_PATH", self.similarity), \
                patch.object(run.previous, "read_rows", return_value=[dict(node="drug", type="drug"), dict(node="disease", type="disease")]), \
                contextlib.redirect_stdout(io.StringIO()):
            run.generate_corpus(self.data, self.results, "fixture", "semantic")
            before = (self.data / "walks.txt").read_bytes()
            run.generate_corpus(self.data, self.results, "fixture", "semantic")
        self.assertEqual((self.data / "walks.txt").read_bytes(), before)
        self.assertEqual(ge.read_graph.call_args_list[1].args, (str(self.similarity),))
        ge.generate_DREAMwalk_paths.assert_called_once_with(graph, semantic, matrix, 1, 1, 1, 2, .3, 4)
        self.assertEqual(run.previous.SIMILARITY, old_similarity)
        self.assertEqual((run.shared.REFERENCE_RESULTS, run.teleport.REFERENCE_RESULTS), old_baselines)
        marker = run.read_json(self.results / "walks_complete.json")
        self.assertEqual(marker["similarity_sha256"], run.sha256(self.similarity))
        self.assertFalse(marker["em_reused"])

    def test_selector_uses_new_input_and_saved_matrix_without_retraining_em(self):
        graph = SimpleNamespace(nodes=lambda: ["drug", "disease"])
        semantic = SimpleNamespace(number_of_edges=lambda: 1)
        ge = SimpleNamespace(set_seed=Mock(), read_graph=Mock(side_effect=[graph, semantic]),
                             train_edgetype_transition_matrix=Mock())
        package = ModuleType("DREAMwalk")
        package.generate_embeddings = ge
        selector, pm = ModuleType("teleport_selector_diagnostic"), ModuleType("parmap")
        selector.worker_walks = Mock()
        pm.map = Mock(return_value=[([["drug", "disease"], ["disease", "drug"]], dict(task_id=0))])
        settings = dict(run.WALK_SETTINGS, num_walks=1, walk_length=2)
        checked = dict(audits=[dict(task_id=0)], selector_calls=1)
        with patch.dict(sys.modules, {"DREAMwalk": package, "parmap": pm, "teleport_selector_diagnostic": selector}), \
                patch.object(run, "WALK_SETTINGS", settings), patch.object(run, "SOURCE_NODES", 2), \
                patch.object(run.inputs, "SIMILARITY_PATH", self.similarity), \
                patch.object(run.previous, "read_rows", return_value=[dict(node="drug", type="drug"), dict(node="disease", type="disease")]), \
                patch.object(run.teleport, "checked_worker_audit", return_value=checked), \
                contextlib.redirect_stdout(io.StringIO()):
            run.generate_corpus(self.data, self.results, "fixture", "teleport")
        ge.train_edgetype_transition_matrix.assert_not_called()
        self.assertEqual(ge.read_graph.call_args_list[1].args, (str(self.similarity),))
        self.assertIs(pm.map.call_args.args[0], selector.worker_walks)
        self.assertEqual(pm.map.call_args.args[4], semantic)
        marker = run.read_json(self.results / "walks_complete.json")
        self.assertTrue(marker["em_reused"])

    def test_old_section7_cannot_satisfy_new_section8_prerequisite(self):
        old_results = self.root / "results/msi_seed0_semantic_skipgram_pp_comparison"
        old_results.mkdir(parents=True)
        self.save(old_results / "run_config.json", {})
        self.save(old_results / "pp0_evaluation_complete.json", {})
        with patch.object(run, "ROOT", self.root):
            with self.assertRaisesRegex(RuntimeError, "Complete notebook Section 7 pp=0"):
                run.semantic_reference()

    def test_new_section8_prepares_artifacts_from_new_section7(self):
        source_data = self.root / "data/embeddings" / run.SEMANTIC_RUN_NAME
        source_results = self.root / "results" / run.SEMANTIC_RUN_NAME
        source_data.mkdir(parents=True)
        source_results.mkdir(parents=True)
        (source_data / "HetSG").write_bytes(b"new-section7-original-binary")
        self.save(source_data / "token_to_node.json", {"d1": "new-map"})
        self.save(source_results / "edge_transition_matrix.json", [[2]])
        self.save(source_results / "compiler_complete.json", dict(sha256="new-original"))
        (source_results / "fold_assignments.csv").write_text("new-folds\n")
        # Use empty targets for the independent selector run.
        for path in self.data.iterdir():
            if path.is_file():
                path.unlink()
        for path in self.results.iterdir():
            if path.is_file():
                path.unlink()
        with patch.object(run, "ROOT", self.root), patch.object(run, "semantic_reference") as require, \
                contextlib.redirect_stdout(io.StringIO()):
            run.prepare(self.data, self.results, "fixture", "teleport")
        require.assert_called_once()
        self.assertEqual((self.data / "HetSG").read_bytes(), (source_data / "HetSG").read_bytes())
        self.assertEqual((self.results / "edge_transition_matrix.json").read_bytes(),
                         (source_results / "edge_transition_matrix.json").read_bytes())
        marker = run.read_json(self.results / "compiler_complete.json")
        self.assertIn(run.SEMANTIC_RUN_NAME, marker["reused_from"])

    def test_new_baseline_validation_reads_only_corrected_section7(self):
        data = self.root / "data/embeddings" / run.SEMANTIC_RUN_NAME
        results = self.root / "results" / run.SEMANTIC_RUN_NAME
        data.mkdir(parents=True)
        results.mkdir(parents=True)
        for name in ("HetSG", "token_to_node.json"):
            (data / name).write_bytes(name.encode())
        for name in ("walks_complete.json", "compiler_complete.json", "pp0_embedding_complete.json",
                     "pp0_evaluation_complete.json", "edge_transition_matrix.json"):
            self.save(results / name, {})
        cohort, provenance = dict(cohort="fixed"), dict(similarity="corrected2024")
        self.save(results / "run_config.json", dict(fingerprint="new-section7", sources={},
                  walks=run.WALK_SETTINGS, semantic=run.SEMANTIC_SETTINGS, selector="upstream",
                  semantic_input_provenance=provenance, reference=cohort))
        embedded = dict(pp=0, sha256="embedding", corpus_sha256="new-corpus",
                        binary_sha256=run.sha256(data / "HetSG"),
                        token_map_sha256=run.sha256(data / "token_to_node.json"),
                        command=["./HetSG", "-train", "walks.txt", "-output", "pp0", "-pp", "0"] + run.paired.SG_ARGS)
        evaluated = dict(sha256="new-summary", fold_assignments_sha256="fixed-folds")
        with patch.object(run, "ROOT", self.root), \
                patch.object(run.inputs, "artifact_identity", return_value=provenance), \
                patch.object(run.original, "reference_identity", return_value=cohort), \
                patch.object(run, "require_corpus", return_value=dict(sha256="new-corpus")) as corpus, \
                patch.object(run.shared, "require_evaluation", return_value=(embedded, evaluated)) as evaluation:
            identity = run.semantic_reference()
        corpus.assert_called_once_with(data, results, "new-section7", "semantic")
        evaluation.assert_called_once_with(0, data, results, "new-section7")
        self.assertEqual(identity["run_name"], run.SEMANTIC_RUN_NAME)
        self.assertEqual(identity["artifacts"]["summary"], "new-summary")


if __name__ == "__main__":
    unittest.main()
