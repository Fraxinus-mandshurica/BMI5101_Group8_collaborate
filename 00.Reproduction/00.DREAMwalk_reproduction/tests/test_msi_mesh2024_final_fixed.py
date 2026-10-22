"""Small mocked integration checks; no scientific MSI run is started."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import run_msi_seed0_mesh2024_final_fixed as run


class Mesh2024FinalFixedTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data, self.results = self.root / "data", self.root / "results"
        self.data.mkdir()
        self.results.mkdir()
        self.fingerprint = "fixture"

    def save(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def marker(self, path, name, **details):
        value = dict(fingerprint=self.fingerprint, sha256=run.sha256(path), **details)
        self.save(self.results / name, value)
        return value

    def test_preparation_reuses_exact_patched_binary_map_and_saved_folds(self):
        source_data, source_results = self.root / "reference_data", self.root / "reference_results"
        source_data.mkdir()
        source_results.mkdir()
        patched_data = self.root / "patched"
        patched_data.mkdir()
        (patched_data / "HetSG").write_bytes(b"exact-patched-binary")
        (source_data / "token_to_node.json").write_text('{"d0":"drug","i1":"disease"}')
        (source_results / "fold_assignments.csv").write_text("pair_index,test_fold\n0,1\n")
        patch_info = run.patch_identity()
        reference = dict(mode="reuse_verified_patched_binary")
        self.save(self.results / "run_config.json", dict(
            fingerprint=self.fingerprint, patch=patch_info, compiler_reference=reference))
        with patch.object(run.fixed, "ROOT", self.root), \
                patch.object(run.fixed, "REFERENCE_DATA", source_data), \
                patch.object(run.fixed, "REFERENCE_RESULTS", source_results), \
                patch.object(run.fixed, "FILTERED_DATA", patched_data), \
                patch.object(run.fixed, "inspect_compiler_reference", return_value=reference), \
                patch.object(run.fixed, "require_prepared"), \
                patch.object(run.fixed.subprocess, "run") as compile_cpp, \
                contextlib.redirect_stdout(io.StringIO()):
            run.prepare(self.data, self.results, self.fingerprint)
        compile_cpp.assert_not_called()
        self.assertEqual((self.data / "HetSG").read_bytes(), (patched_data / "HetSG").read_bytes())
        self.assertEqual((self.data / "token_to_node.json").read_bytes(),
                         (source_data / "token_to_node.json").read_bytes())
        self.assertEqual((self.results / "fold_assignments.csv").read_bytes(),
                         (source_results / "fold_assignments.csv").read_bytes())
        self.assertEqual(run.sha256(self.data / run.fixed.PATCHED_NAME), patch_info["patched_source_sha256"])

    def test_2024_semantic_input_is_passed_to_fixed_worker_with_original_em(self):
        import numpy as np
        self.save(self.data / "token_to_node.json", {"d0": "drug", "i1": "disease"})
        semantic_path = self.root / "mesh2024_fixed.tsv"
        semantic_path.write_text("drug\tdisease\t0.8\n")
        graph, semantic, ge = MagicMock(), MagicMock(), MagicMock()
        graph.nodes.return_value = ["drug", "disease"]
        semantic.number_of_edges.return_value = 1
        ge.read_graph.side_effect = [graph, semantic]
        matrix = np.ones((1, 1))
        ge.train_edgetype_transition_matrix.return_value = matrix
        package = types.ModuleType("DREAMwalk")
        package.generate_embeddings = ge
        selector = types.ModuleType("teleport_selector_diagnostic")
        selector.worker_walks = MagicMock()
        task_walks = [["drug", "disease"] * 5, ["disease", "drug"] * 5]
        audits = [dict(task_id=index) for index in range(100)]
        pm = types.ModuleType("parmap")
        pm.map = MagicMock(return_value=[(task_walks, audit) for audit in audits])
        checked = dict(audits=audits, selector_calls=100)
        compiler = dict(token_map_sha256=run.sha256(self.data / "token_to_node.json"))
        rows = [dict(node="drug", type="drug"), dict(node="disease", type="disease")]
        with patch.dict(sys.modules, {"DREAMwalk": package, "parmap": pm,
                                      "teleport_selector_diagnostic": selector}), \
                patch.object(run.inputs, "SIMILARITY_PATH", semantic_path), \
                patch.object(run, "SOURCE_NODES", 2), \
                patch.object(run, "require_prepared", return_value=compiler), \
                patch.object(run.previous, "read_rows", return_value=rows), \
                patch.object(run.fixed.teleport, "checked_worker_audit", return_value=checked) as check_audit, \
                contextlib.redirect_stdout(io.StringIO()):
            run.generate_corpus(self.data, self.results, self.fingerprint)
        ge.set_seed.assert_called_once_with(0)
        self.assertEqual(ge.read_graph.call_args_list[1].args, (str(semantic_path),))
        ge.train_edgetype_transition_matrix.assert_called_once_with(
            5, graph, str(run.previous.GRAPH), "\t", 10, 1, 1)
        self.assertEqual(pm.map.call_args.args, (
            selector.worker_walks, range(100), ["drug", "disease"], graph, semantic, matrix, 1, 1, 10, 0.3))
        self.assertEqual(pm.map.call_args.kwargs, dict(pm_pbar=False, pm_processes=4))
        check_audit.assert_called_once_with(audits, 100, run.sha256(run.SELECTOR_MODULE))
        marker = run.read_json(self.results / "walks_complete.json")
        self.assertEqual(marker["walks"], 200)
        self.assertEqual(marker["similarity_sha256"], run.sha256(semantic_path))
        self.assertFalse(marker["em_reused"])
        self.assertFalse((self.data / "walks.txt.tmp").exists())

    def test_embedding_is_persisted_before_separate_evaluation(self):
        import numpy  # Load before mocking the C++ subprocess launch.
        (self.data / "HetSG").write_bytes(b"patched-cpp")
        (self.data / "walks.txt").write_text("d0 i1\ni1 d0")
        self.save(self.data / "token_to_node.json", {"d0": "drug", "i1": "disease"})
        self.marker(self.data / "HetSG", "compiler_complete.json")
        corpus = self.marker(self.data / "walks.txt", "walks_complete.json",
                             token_map_sha256=run.sha256(self.data / "token_to_node.json"))

        def fake_cpp(command, cwd, check):
            self.assertEqual(command, run.fixed.positive_filter.training_command())
            self.assertEqual(cwd, self.data)
            self.assertTrue(check)
            vector = " ".join(["0.1"] * 128)
            (self.data / "pp1.txt").write_text("2 128\nd0 " + vector + "\ni1 " + vector + "\n")

        with patch.object(sys, "argv", ["runner", "embedding"]), \
                patch.object(run, "context", return_value=(self.data, self.results, self.fingerprint)), \
                patch.object(run, "require_corpus", return_value=corpus), \
                patch.object(run.paired.subprocess, "run", side_effect=fake_cpp), \
                patch.object(run, "evaluate") as evaluate, \
                contextlib.redirect_stdout(io.StringIO()):
            run.main()
        evaluate.assert_not_called()
        self.assertTrue((self.data / "pp1.pkl").is_file())
        self.assertFalse((self.data / "pp1.pkl.tmp").exists())
        marker = run.read_json(self.results / "pp1_embedding_complete.json")
        self.assertEqual(marker["pp"], 1)
        self.assertEqual(marker["binary_sha256"], run.sha256(self.data / "HetSG"))

    def test_evaluation_uses_same_negative_order_and_saved_fold_ids(self):
        import numpy as np
        from sklearn.model_selection import StratifiedKFold
        positives = {(f"drug{index}", f"disease{index}") for index in range(10)}
        negatives = [(f"drug{index}", f"disease{(index+1)%10}") for index in range(10)]
        negative_path = self.root / "negatives.csv"
        run.previous.write_csv(negative_path, [dict(drug=a, disease=b) for a, b in negatives], ["drug", "disease"])
        pairs, labels = sorted(positives) + negatives, np.asarray([1]*10 + [0]*10)
        assignment = np.empty(20, dtype=np.int32)
        for fold, (_, test) in enumerate(StratifiedKFold(10, shuffle=True, random_state=0).split(
                np.zeros(20), labels), 1):
            assignment[test] = fold
        rows = [dict(pair_index=index, drug=a, disease=b, label=int(labels[index]),
                     test_fold=int(assignment[index])) for index, (a, b) in enumerate(pairs)]
        reference = self.root / "reference"
        reference.mkdir()
        for path in (reference / "fold_assignments.csv", self.results / "fold_assignments.csv"):
            run.previous.write_csv(path, rows, list(rows[0]))
        fold_rows = [dict(fold=fold, accuracy=0.6, auroc=0.7, aupr=0.8) for fold in range(1, 11)]
        vectors = {"fixture": "vectors"}
        with patch.object(run, "REFERENCE_RESULTS", reference), \
                patch.object(run, "require_embedding", return_value=dict(sha256="embedding")), \
                patch.object(run.previous, "NEGATIVES", negative_path), \
                patch.object(run.previous, "base_inputs", return_value=({}, [], [], positives)), \
                patch.object(run.previous, "load_embedding", return_value=vectors) as load, \
                patch.object(run.previous, "evaluate", return_value=fold_rows) as fit:
            run.evaluate(self.data, self.results, self.fingerprint)
        load.assert_called_once_with(self.data / "pp1.pkl", {}, sorted(positives) + negatives)
        fit.assert_called_once_with(vectors, positives, negatives, 0, "mesh2024_final_fixed_pp1")
        self.assertEqual((self.results / "fold_assignments.csv").read_bytes(),
                         (reference / "fold_assignments.csv").read_bytes())
        marker = run.read_json(self.results / "pp1_evaluation_complete.json")
        self.assertEqual(marker["fold_assignments_sha256"], run.sha256(reference / "fold_assignments.csv"))

    def test_comparison_does_not_require_completed_2021_experiment(self):
        missing2021 = self.root / "not_run_2021"
        baseline = self.root / "baseline"
        baseline.mkdir()
        (baseline / "pp1_summary.csv").write_text("metric,mean\naccuracy,0.7\nauroc,0.7\naupr,0.7\n")
        paper = self.root / "paper.csv"
        paper.write_text("metric,mean\naccuracy,0.9\nauroc,0.9\naupr,0.9\n")
        summary = self.results / "pp1_summary.csv"
        summary.write_text("metric,mean\naccuracy,0.8\nauroc,0.8\naupr,0.8\n")
        evaluated = dict(sha256=run.sha256(summary), fold_assignments_sha256="shared-folds")
        with patch.object(run, "FINAL2021_RESULTS", missing2021), \
                patch.object(run, "ORIGINAL2024_RESULTS", baseline), \
                patch.object(run, "PAPER_REFERENCE", paper), \
                patch.object(run, "require_evaluation", return_value=evaluated), \
                patch.object(run, "original_summary", return_value=dict.fromkeys(run.METRICS, 0.7)), \
                contextlib.redirect_stdout(io.StringIO()):
            run.compare(self.data, self.results, self.fingerprint)
        rows = run.previous.read_rows(self.results / "final_fixed_comparison.csv", ",")
        for row in rows:
            self.assertAlmostEqual(float(row["final_minus_original2024"]), 0.1)
            self.assertEqual(row["final_mesh2021_fixed_pp1"], "")
        self.assertEqual(run.read_json(self.results / "comparison_complete.json")["optional_final2021"]["status"],
                         "not_completed")


if __name__ == "__main__":
    unittest.main()
