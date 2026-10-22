"""Small provenance and protocol fixtures; never train walks, embeddings or XGB."""
import ast
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_hetionet_original as run


class NegativeAndFoldTest(unittest.TestCase):
    def test_balanced_saved_candidates_exclude_every_known_ctd_and_non_target_types(self):
        types = {"D%d" % i: "drug" for i in range(5)}
        types.update({"I%d" % i: "disease" for i in range(4)})
        types.update({"G": "gene", "P": "etc"})
        positives = [("D0", "I0"), ("D1", "I1"), ("D2", "I2")]
        raw = set(positives) | {("D3", "I3"), ("D4", "I0")}
        negatives = run.sample_negatives(types, positives, raw)
        self.assertEqual(len(negatives), len(positives))
        self.assertEqual(len(set(negatives)), len(negatives))
        self.assertFalse(set(negatives) & raw)
        self.assertTrue(all(types[a] == "drug" and types[b] == "disease"
                            for a, b in negatives))
        self.assertEqual(negatives, run.sample_negatives(types, positives, raw))

    def test_one_saved_test_fold_per_pair_and_both_classes_in_each_fold(self):
        positives = [("D%d" % i, "I%d" % i) for i in range(20)]
        negatives = [("D%d" % i, "N%d" % i) for i in range(20)]
        rows = run.fold_rows(positives, negatives)
        self.assertEqual([r["pair_index"] for r in rows], list(range(40)))
        self.assertEqual([(r["drug"], r["disease"]) for r in rows], positives + negatives)
        self.assertEqual(rows, run.fold_rows(positives, negatives))
        for fold in range(1, 11):
            heldout = [r for r in rows if r["test_fold"] == fold]
            self.assertEqual(len(heldout), 4)
            self.assertEqual(sorted(r["label"] for r in heldout), [0, 0, 1, 1])

    def test_saved_fold_mismatch_stops_before_embedding_load_or_classifier_fit(self):
        positives = [("D%d" % i, "I%d" % i) for i in range(10)]
        negatives = [("D%d" % i, "N%d" % i) for i in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run.save_csv(root / "negative_pairs.csv",
                         [dict(drug=a, disease=b) for a, b in negatives], ["drug", "disease"])
            wrong = run.fold_rows(positives, negatives)
            wrong[0]["test_fold"] = 99
            run.save_csv(root / "fold_assignments.csv", wrong, list(wrong[0]))
            with patch.object(run, "context", return_value=(root, root, "fixture")), \
                    patch.object(run, "completed", side_effect=[False, True]), \
                    patch.object(run, "input_data", return_value=({}, {}, positives, set(positives))), \
                    patch.object(run, "sample_negatives", return_value=negatives), \
                    patch.object(run, "load_embedding") as load, \
                    patch("xgboost.XGBClassifier") as classifier:
                with self.assertRaisesRegex(RuntimeError, "Saved fold IDs before fitting"):
                    run.evaluate("hetionet_fixture")
                load.assert_not_called()
                classifier.assert_not_called()
            self.assertFalse((root / "evaluation_started.json").exists())

    def test_partial_rows_must_match_frozen_fold_counts_and_finite_metrics(self):
        assignments = [dict(test_fold=fold) for fold in range(1, 11) for _ in range(2)]
        good = dict(fold="1", accuracy="0.5", auroc="0.7", aupr="0.8",
                    n_train="18", n_test="2", seconds="0.1")
        self.assertEqual(run.validate_saved_folds([good], assignments), [1])
        self.assertEqual(run.validate_saved_folds([], assignments), [])
        for changes in ({"fold": "0"}, {"fold": "11"}, {"fold": "1.5"},
                        {"accuracy": "nan"}, {"auroc": "inf"}, {"aupr": "1.01"},
                        {"accuracy": "-0.01"}, {"n_train": "17"}, {"n_test": "3"},
                        {"seconds": "-1"}, {"seconds": "nan"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "Invalid partial"):
                run.validate_saved_folds([dict(good, **changes)], assignments)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            run.validate_saved_folds([good, good], assignments)
        del good["auroc"]
        with self.assertRaisesRegex(ValueError, "Invalid partial"):
            run.validate_saved_folds([good], assignments)


class CheckpointIdentityTest(unittest.TestCase):
    def test_config_and_pair_hashes_are_required_before_run_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "results/hetionet_fixture"
            result.mkdir(parents=True)
            pair = result / "negative_pairs.csv"
            pair.write_text("drug,disease\nD,I\n")
            current = {"settings": {"seed": 0}, "inputs": {"graph": "fixture"}}
            config = dict(fingerprint=run.fingerprint(current),
                          pair_artifacts={pair.name: run.sha256(pair)})
            run.save_json(result / "run_config.json", config)
            with patch.object(run, "ROOT", root), patch.object(run, "identity", return_value=current):
                self.assertEqual(run.context("hetionet_fixture")[2], config["fingerprint"])
                pair.write_text("drug,disease\nD,changed\n")
                with self.assertRaisesRegex(RuntimeError, "Saved negative_pairs.csv"):
                    run.context("hetionet_fixture")
            with patch.object(run, "ROOT", root), \
                    patch.object(run, "identity", return_value={"settings": {"seed": 1}}):
                with self.assertRaisesRegex(RuntimeError, "Frozen input/code/settings identity"):
                    run.context("hetionet_fixture")

    def test_completed_marker_rejects_changed_artifact_or_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact, marker = root / "embedding.pkl", root / "complete.json"
            artifact.write_bytes(b"fixture")
            self.assertFalse(run.completed(artifact, marker, "fp"))
            run.save_json(marker, dict(fingerprint="fp", sha256=run.sha256(artifact)))
            self.assertTrue(run.completed(artifact, marker, "fp"))
            with self.assertRaisesRegex(RuntimeError, "Checkpoint identity"):
                run.completed(artifact, marker, "different")
            artifact.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Checkpoint embedding.pkl"):
                run.completed(artifact, marker, "fp")

    def test_msi_names_and_unregistered_output_are_refused(self):
        for name in ("msi_seed0", "hetionet_../other", "../hetionet_run", "hetionet_run/file"):
            with self.assertRaises(ValueError):
                run.run_paths(name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data/embeddings/hetionet_fixture"
            data.mkdir(parents=True)
            (data / "partial.pkl").write_bytes(b"partial")
            with patch.object(run, "ROOT", root), \
                    patch.object(run.inputs, "prepare_inputs"), \
                    patch.object(run, "identity", return_value={"settings": "fixture"}), \
                    patch.object(run, "sample_negatives") as sampling:
                with self.assertRaisesRegex(RuntimeError, "Unregistered run outputs"):
                    run.prepare("hetionet_fixture")
                sampling.assert_not_called()


class OriginalEmbeddingTest(unittest.TestCase):
    def test_original_call_persists_valid_embedding_before_any_evaluation(self):
        import numpy as np
        import DREAMwalk.generate_embeddings as upstream
        original_selector = upstream._teleport_operation
        original_sg = upstream.HeterogeneousSG
        types = {"Compound::D": "drug", "Disease::I": "disease"}
        vectors = {node: np.zeros(128, dtype=np.float32) for node in types}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, result = root / "data", root / "results"
            data.mkdir()
            result.mkdir()
            similarity = data / "similarity_graph.tsv"
            similarity.write_text("fixture semantic graph")
            paths = {key: root / key for key in ("graph", "nodetypes")}
            calls = []

            def fake_save(**kwargs):
                calls.append(kwargs)
                # Only a fixture write: no EM, random walks, C++ or fitting.
                self.assertEqual(upstream._teleport_operation, original_selector)
                self.assertEqual(upstream.HeterogeneousSG, original_sg)
                self.assertTrue((Path.cwd() / "DREAMwalk").is_symlink())
                self.assertFalse((data / "hetionet_seed0_original.pkl").exists())
                with open(kwargs["outputf"], "wb") as stream:
                    pickle.dump(vectors, stream)

            old_cwd = Path.cwd()
            with patch.object(run, "ROOT", root), \
                    patch.object(run, "context", return_value=(data, result, "fp")), \
                    patch.object(run, "completed", side_effect=[False, True]), \
                    patch.object(run, "input_data", return_value=(paths, types, [], set())), \
                    patch.object(run.platform, "system", return_value="Linux"), \
                    patch.object(upstream, "save_embedding_files", side_effect=fake_save), \
                    patch.object(run, "evaluate") as evaluation:
                run.embedding("hetionet_fixture")
                evaluation.assert_not_called()
            self.assertEqual(Path.cwd(), old_cwd)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], dict(netf=str(paths["graph"]), sim_netf=str(similarity),
                             nodetypef=str(paths["nodetypes"]), outputf=calls[0]["outputf"],
                             directed=False, weighted=True, net_delimiter="\t",
                             tp_factor=.3, seed=0, num_walks=100, walk_length=10,
                             dimension=128, window_size=4, workers=4, em_max_iter=5, p=1, q=1))
            output = data / "hetionet_seed0_original.pkl"
            self.assertEqual(set(run.load_embedding(output, types)), set(types))
            metadata = run.read_json(result / "embedding_complete.json")
            self.assertEqual(metadata["sha256"], run.sha256(output))
            self.assertEqual(metadata["expected_walks"], 200)
            self.assertEqual(metadata["implementation"], "unmodified save_embedding_files")
            self.assertFalse((result / "summary.csv").exists())

    def test_source_has_main_guard_and_does_not_import_msi_or_diagnostic_runners(self):
        tree = ast.parse(Path(run.__file__).read_text())
        imports = []
        for item in ast.walk(tree):
            if isinstance(item, ast.ImportFrom):
                imports.append(item.module or "")
            elif isinstance(item, ast.Import):
                imports.extend(alias.name for alias in item.names)
        self.assertFalse(any("msi" in name or "diagnostic" in name or "control" in name
                             for name in imports))
        self.assertTrue(any(isinstance(item, ast.If)
                            and ast.dump(item.test) == ast.dump(ast.parse('__name__ == "__main__"').body[0].value)
                            for item in tree.body))


if __name__ == "__main__":
    unittest.main()
