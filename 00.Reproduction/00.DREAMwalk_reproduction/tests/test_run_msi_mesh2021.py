"""Mocked MeSH-year routing, persistence and cohort checks; no MSI training."""
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
import run_msi_mesh2021 as run


class Mesh2021RunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data, self.results = self.root / "data", self.root / "results"
        self.data.mkdir()
        self.results.mkdir()
        self.fingerprint = "fixture-fingerprint"

    def write_json(self, path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def marker(self, output, name, **extra):
        value = dict(fingerprint=self.fingerprint, sha256=run.sha256(output), **extra)
        self.write_json(self.results / name, value)
        return value

    def corpus_fixture(self):
        (self.data / "HetSG").write_bytes(b"fixture-binary")
        (self.data / "walks.txt").write_text("d0 i1\ni1 d0", encoding="utf-8")
        self.write_json(self.data / "token_to_node.json", {"d0": "drug", "i1": "disease"})
        self.marker(self.data / "HetSG", "compiler_complete.json")
        self.marker(self.data / "walks.txt", "walks_complete.json",
                    token_map_sha256=run.sha256(self.data / "token_to_node.json"))

    def test_original_walker_receives_new_semantic_graph_and_accepted_parameters(self):
        import numpy as np
        self.write_json(self.data / "token_to_node.json", {"d0": "drug", "i1": "disease"})
        semantic_path = self.root / "mesh2021.tsv"
        semantic_path.write_text("drug\tdisease\t0.8\n")
        graph, semantic = MagicMock(), MagicMock()
        graph.nodes.return_value = ["drug", "disease"]
        semantic.number_of_edges.return_value = 1
        ge = MagicMock()
        ge.read_graph.side_effect = [graph, semantic]
        matrix = np.ones((1, 1))
        ge.train_edgetype_transition_matrix.return_value = matrix
        ge.generate_DREAMwalk_paths.return_value = [
            ["drug", "disease"] * 5, ["disease", "drug"] * 5] * 100
        package = types.ModuleType("DREAMwalk")
        package.generate_embeddings = ge
        compiler = dict(token_map_sha256=run.sha256(self.data / "token_to_node.json"))
        rows = [dict(node="drug", type="drug"), dict(node="disease", type="disease")]
        with patch.dict(sys.modules, {"DREAMwalk": package}), \
                patch.object(run.inputs, "SIMILARITY_PATH", semantic_path), \
                patch.object(run, "SOURCE_NODES", 2), \
                patch.object(run.paired, "completed", return_value=False), \
                patch.object(run.paired, "require_checkpoint", return_value=compiler), \
                patch.object(run.previous, "read_rows", return_value=rows), \
                contextlib.redirect_stdout(io.StringIO()):
            run.generate_corpus(self.data, self.results, self.fingerprint)
        ge.set_seed.assert_called_once_with(0)
        self.assertEqual(ge.read_graph.call_args_list[1].args, (str(semantic_path),))
        self.assertEqual(ge.read_graph.call_args_list[1].kwargs,
                         dict(weighted=True, directed=False, delimiter="\t"))
        ge.train_edgetype_transition_matrix.assert_called_once_with(
            5, graph, str(run.previous.GRAPH), "\t", 10, 1, 1)
        ge.generate_DREAMwalk_paths.assert_called_once_with(
            graph, semantic, matrix, 1, 1, 100, 10, 0.3, 4)
        marker = json.loads((self.results / "walks_complete.json").read_text())
        self.assertEqual(marker["similarity_sha256"], run.sha256(semantic_path))
        self.assertEqual(marker["walks"], 200)
        self.assertNotEqual(str(semantic_path), str(run.previous.SIMILARITY))

    def test_embedding_phase_saves_original_pp1_before_any_evaluation(self):
        import numpy  # Import before mocking the process launch used by C++.
        self.corpus_fixture()

        def fake_cpp(command, cwd, check):
            self.assertEqual(command, ["./HetSG", "-train", "walks.txt", "-output", "pp1", "-pp", "1"]
                             + run.paired.SG_ARGS)
            self.assertEqual(cwd, self.data)
            self.assertTrue(check)
            vector = " ".join(["0.1"] * 128)
            (self.data / "pp1.txt").write_text("2 128\nd0 " + vector + "\ni1 " + vector + "\n")

        with patch.object(sys, "argv", ["runner", "embedding"]), \
                patch.object(run, "context", return_value=(self.data, self.results, self.fingerprint)), \
                patch.object(run.paired.subprocess, "run", side_effect=fake_cpp), \
                patch.object(run, "evaluate") as evaluate, \
                contextlib.redirect_stdout(io.StringIO()):
            run.main()
        evaluate.assert_not_called()
        output = self.data / "pp1.pkl"
        self.assertTrue(output.is_file())
        self.assertFalse((self.data / "pp1.pkl.tmp").exists())
        marker = json.loads((self.results / "pp1_embedding_complete.json").read_text())
        self.assertEqual(marker["sha256"], run.sha256(output))
        self.assertEqual(marker["pp"], 1)
        self.assertEqual(marker["nodes"], 2)

    def evaluation_fixture(self):
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
        (self.data / "pp1.pkl").write_bytes(b"fixture-embedding")
        self.marker(self.data / "pp1.pkl", "pp1_embedding_complete.json")
        return positives, negatives, negative_path, reference

    def test_evaluation_retains_exact_saved_negative_order_and_fold_ids(self):
        positives, negatives, negative_path, reference = self.evaluation_fixture()
        fold_rows = [dict(fold=fold, accuracy=0.6, auroc=0.7, aupr=0.8) for fold in range(1, 11)]
        vectors = {"fixture": "vectors"}
        with patch.object(run, "REFERENCE_RESULTS", reference), \
                patch.object(run.previous, "NEGATIVES", negative_path), \
                patch.object(run.previous, "base_inputs", return_value=({}, [], [], positives)), \
                patch.object(run.previous, "load_embedding", return_value=vectors) as load, \
                patch.object(run.previous, "evaluate", return_value=fold_rows) as fit:
            run.evaluate(self.data, self.results, self.fingerprint)
        load.assert_called_once_with(self.data / "pp1.pkl", {}, sorted(positives) + negatives)
        fit.assert_called_once_with(vectors, positives, negatives, 0, "mesh2021_original_pp1")
        self.assertEqual((self.results / "fold_assignments.csv").read_bytes(),
                         (reference / "fold_assignments.csv").read_bytes())
        marker = json.loads((self.results / "pp1_evaluation_complete.json").read_text())
        self.assertEqual(marker["fold_assignments_sha256"], run.sha256(reference / "fold_assignments.csv"))

    def test_mismatched_saved_fold_blocks_evaluation_before_fit(self):
        positives, _, negative_path, reference = self.evaluation_fixture()
        path = reference / "fold_assignments.csv"
        rows = run.previous.read_rows(path, ",")
        rows[0]["test_fold"] = "999"
        run.previous.write_csv(path, rows, list(rows[0]))
        with patch.object(run, "REFERENCE_RESULTS", reference), \
                patch.object(run.previous, "NEGATIVES", negative_path), \
                patch.object(run.previous, "base_inputs", return_value=({}, [], [], positives)), \
                patch.object(run.previous, "evaluate") as fit:
            with self.assertRaisesRegex(RuntimeError, "Pair/fold assignments before fitting"):
                run.evaluate(self.data, self.results, self.fingerprint)
        fit.assert_not_called()
        self.assertFalse((self.results / "pp1_summary.csv").exists())

    def test_comparison_uses_matching_original_pp1_baseline(self):
        self.corpus_fixture()
        baseline = self.root / "baseline2024"
        baseline.mkdir()
        old_rows = [dict(metric=metric, mean=0.7) for metric in run.METRICS]
        new_rows = [dict(metric=metric, mean=0.8) for metric in run.METRICS]
        for path, rows in ((baseline / "pp1_summary.csv", old_rows),
                           (self.results / "pp1_summary.csv", new_rows)):
            run.previous.write_csv(path, rows, ["metric", "mean"])
        self.write_json(baseline / "run_config.json", dict(
            fingerprint="old", walks=run.WALK_SETTINGS, skipgram_args=run.paired.SG_ARGS))
        self.write_json(baseline / "pp1_embedding_complete.json", dict(
            sha256="embedding2024", binary_sha256=run.sha256(self.data / "HetSG"),
            token_map_sha256=run.sha256(self.data / "token_to_node.json")))
        self.write_json(baseline / "pp1_evaluation_complete.json", dict(
            fingerprint="old", sha256=run.sha256(baseline / "pp1_summary.csv"),
            embedding_sha256="embedding2024", fold_assignments_sha256="same-folds"))
        evaluation = dict(sha256=run.sha256(self.results / "pp1_summary.csv"),
                          fold_assignments_sha256="same-folds")
        with patch.object(run, "MESH2024_RESULTS", baseline), \
                patch.object(run, "require_evaluation", return_value=evaluation), \
                contextlib.redirect_stdout(io.StringIO()):
            run.compare(self.data, self.results, self.fingerprint)
        rows = run.previous.read_rows(self.results / "mesh_year_comparison.csv", ",")
        self.assertEqual({row["metric"] for row in rows}, set(run.METRICS))
        for row in rows:
            self.assertAlmostEqual(float(row["mesh2021_minus_mesh2024"]), 0.1)


if __name__ == "__main__":
    unittest.main()
