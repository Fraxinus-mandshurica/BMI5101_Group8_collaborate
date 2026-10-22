"""Small synthetic fixtures only. Never start MSI preprocessing or training."""
import ast
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse
import torch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_seal as run
from seal_model import DGCNN, collate_graphs, drnl, enclosing_subgraph, pooling_k

torch.set_num_threads(1)


def adjacency(size, edges):
    rows, cols = [], []
    for a, b in edges:
        rows.extend((a, b))
        cols.extend((b, a))
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(size, size))


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProtocolTests(unittest.TestCase):
    def test_drnl_matches_pinned_original_with_disconnected_nodes(self):
        source = ROOT / "vendor/SEAL_upstream/Python/util_functions.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "node_label")
        namespace = dict(np=np, ssp=sparse)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        for graph in [adjacency(6, [(0, 2), (1, 2), (2, 3), (3, 4)]),
                      adjacency(4, [(0, 2), (1, 3)]), adjacency(2, [])]:
            with np.errstate(all="ignore"):
                original = namespace["node_label"](graph)
            np.testing.assert_array_equal(drnl(graph), original)

    def test_extract_keeps_endpoints_and_masks_target_edge(self):
        graph = adjacency(6, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)])
        nodes, edges, labels = enclosing_subgraph(graph, 0, 1, hops=1)
        np.testing.assert_array_equal(nodes, [0, 1, 2, 3])
        self.assertNotIn((0, 1), set(map(tuple, edges.T)))
        self.assertNotIn((1, 0), set(map(tuple, edges.T)))
        np.testing.assert_array_equal(labels[:2], [1, 1])
        self.assertEqual(graph[0, 1], 1)

    def test_port_matches_original_dgcnn_and_classifier(self):
        """Replace only original native sparse construction, then compare forward."""
        graphs = [adjacency(4, [(0, 2), (1, 2), (2, 3)]), adjacency(2, [])]
        graph_list = [types.SimpleNamespace(num_nodes=g.shape[0], degs=np.asarray(g.sum(1)).ravel().tolist()) for g in graphs]

        class NativeReplacement:
            @staticmethod
            def PrepareSparseMatrices(items):
                matrix = sparse.block_diag(graphs).tocoo()
                n2n = torch.sparse_coo_tensor(np.asarray([matrix.row, matrix.col]),
                                               torch.tensor(matrix.data, dtype=torch.float32), matrix.shape)
                e2n = torch.empty((matrix.shape[0], 0)).to_sparse()
                subg = torch.empty((len(graphs), matrix.shape[0])).to_sparse()
                return n2n, e2n, subg

        native = types.ModuleType("gnn_lib")
        native.GNNLIB = NativeReplacement
        with patch.dict(sys.modules, {"gnn_lib": native}):
            utilities = load_file("original_pytorch_util", ROOT / "vendor/pytorch_DGCNN_upstream/lib/pytorch_util.py")
            with patch.dict(sys.modules, {"pytorch_util": utilities}):
                original = load_file("original_dgcnn", ROOT / "vendor/pytorch_DGCNN_upstream/DGCNN_embedding.py")
                head = load_file("original_mlp", ROOT / "vendor/pytorch_DGCNN_upstream/mlp_dropout.py")
                port = DGCNN(7, 10).eval()
                old = original.DGCNN(0, 7, 0, k=10).eval()
                mlp = head.MLPClassifier(old.dense_dim, 128, 2, with_dropout=True).eval()
                for a, b in zip(old.conv_params, port.convs):
                    a.load_state_dict(b.state_dict())
                old.conv1d_params1.load_state_dict(port.conv1.state_dict())
                old.conv1d_params2.load_state_dict(port.conv2.state_dict())
                mlp.h1_weights.load_state_dict(port.hidden.state_dict())
                mlp.h2_weights.load_state_dict(port.output.state_dict())
                features = torch.randn(6, 7)
                combined = sparse.block_diag(graphs).tocoo()
                edges = torch.tensor(np.asarray([combined.row, combined.col]), dtype=torch.long)
                with torch.no_grad():
                    expected = mlp(old(graph_list, features, None))
                    observed = port(features, edges, [4, 2])
                torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)

    def test_collate_preserves_drnl_and_global_feature_alignment(self):
        features = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        graphs = [(np.asarray([2, 0]), np.empty((2, 0), dtype=int), np.asarray([1, 1]), 1),
                  (np.asarray([3, 1]), np.asarray([[0, 1], [1, 0]]), np.asarray([1, 1]), 0)]
        x, edges, sizes, targets = collate_graphs(graphs, features, 3)
        torch.testing.assert_close(x[:, 3:], features[[2, 0, 3, 1]])
        self.assertEqual(sizes, [2, 2])
        self.assertEqual(targets.tolist(), [1, 0])
        self.assertEqual(edges.tolist(), [[2, 3], [3, 2]])

    def test_saved_pairs_are_balanced_disjoint_and_ordered(self):
        positives = [dict(drug=f"d{i}", disease=f"c{i}") for i in range(10)]
        negatives = [dict(drug=f"d{i}", disease=f"c{(i + 1) % 10}") for i in range(10)]
        node_types = {**{f"d{i}": "drug" for i in range(10)}, **{f"c{i}": "disease" for i in range(10)}}
        folds = [dict(pair_index=i, drug=p["drug"], disease=p["disease"], label=int(i < 10), test_fold=1 + i % 2)
                 for i, p in enumerate(positives + negatives)]
        run.check_pairs(positives, negatives, folds, node_types, expected_count=10, n_folds=2)
        with self.assertRaises(ValueError):
            run.check_pairs(positives, positives, folds, node_types, expected_count=10, n_folds=2)
        with self.assertRaises(ValueError):
            run.check_pairs(positives, negatives, folds[::-1], node_types, expected_count=10, n_folds=2)
        partitions = run.validation_partitions(folds, n_folds=2, fraction=0.2)
        for fold, part in partitions.items():
            self.assertFalse(set(part["train"]) & set(part["validation"]))
            self.assertFalse((set(part["train"]) | set(part["validation"])) & set(part["test"]))
            self.assertEqual(set(part["train"] + part["validation"] + part["test"]), set(range(20)))
            self.assertTrue(all(int(folds[i]["test_fold"]) == int(fold) for i in part["test"]))

    def test_original_pooling_percentile_and_minimum(self):
        self.assertEqual(pooling_k([2, 3, 4, 20, 100]), 10)
        self.assertEqual(pooling_k([20, 30, 40, 50, 60]), 40)
        with self.assertRaises(ValueError):
            pooling_k([])

    def test_checkpoint_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact, marker = directory / "file", directory / "complete.json"
            artifact.write_text("complete")
            run.mark(marker, "same", {"file": artifact})
            self.assertTrue(run.complete(marker, "same", {"file": artifact}))
            with self.assertRaises(RuntimeError):
                run.complete(marker, "different", {"file": artifact})
            artifact.write_text("changed")
            with self.assertRaises(RuntimeError):
                run.complete(marker, "same", {"file": artifact})

    def test_tiny_fold_saves_best_before_test_and_resumes(self):
        """Two toy epochs on six-node graphs; never read project scientific data."""
        with tempfile.TemporaryDirectory() as temporary:
            data, results = Path(temporary) / "data", Path(temporary) / "results"
            data.mkdir(); results.mkdir()
            graph = adjacency(4, [(0, 2), (1, 2), (2, 3)])
            sparse.save_npz(data / "graph.npz", graph)
            np.save(data / "node_features.npy", np.arange(32, dtype=np.float32).reshape(4, 8) / 32)
            run.save_json(data / "nodes.json", ["d", "c", "g", "e"])
            pairs = [dict(pair_index=i, drug="d", disease="c", label=int(i < 10), test_fold=1 + i % 10) for i in range(20)]
            run.save_csv(results / "fold_assignments.csv", pairs)
            run.save_json(results / "partitions.json", run.validation_partitions(pairs))
            run.mark(results / "prepare_complete.json", "toy", run.prepared_artifacts(data, results))
            path = data / "subgraphs.sqlite"
            with sqlite3.connect(path) as db:
                db.execute("CREATE TABLE graphs(pair_id INTEGER PRIMARY KEY,payload BLOB)")
                nodes, edges, labels = enclosing_subgraph(graph, 0, 1)
                stream = io.BytesIO()
                np.savez_compressed(stream, nodes=nodes, edges=edges, labels=labels)
                db.executemany("INSERT INTO graphs VALUES(?,?)", [(i, stream.getvalue()) for i in range(20)])
            run.mark(results / "subgraphs_complete.json", "toy", {"database": path}, label_count=int(labels.max()) + 1, sortpooling_k=10)
            settings = dict(run.CONFIG, batch_size=2, max_epochs=2, patience=1, threads=1, latent_dims=[4, 4, 4, 1], hidden_dim=8)
            real_score = run.score
            calls = []

            def checked_score(model, db, indices, *args):
                if indices == run.read_json(results / "partitions.json")["1"]["test"]:
                    self.assertTrue((results / "fold01/best.pt").is_file())
                    calls.append("heldout")
                return real_score(model, db, indices, *args)

            with patch.dict(run.CONFIG, settings, clear=True), patch.object(run, "score", checked_score), contextlib.redirect_stdout(io.StringIO()):
                run.train(data, results, "toy", 1)
                before = run.sha256(results / "fold01/test_predictions.csv")
                run.train(data, results, "toy", 1)
                self.assertEqual(run.sha256(results / "fold01/test_predictions.csv"), before)
                self.assertEqual(calls, ["heldout"])
                # Remove completion only: resume final epoch state and recover the same predictions.
                (results / "fold01/complete.json").unlink()
                run.train(data, results, "toy", 1)
                self.assertEqual(run.sha256(results / "fold01/test_predictions.csv"), before)
                self.assertEqual(calls, ["heldout", "heldout"])
            self.assertEqual(len(run.read_csv(results / "fold01/test_predictions.csv")), 2)
            with self.assertRaises(RuntimeError):
                run.summarize(data, results, "toy")

    def test_pinned_source_and_reference_hashes(self):
        for entry in run.read_json(ROOT / "vendor/manifest.json"):
            self.assertEqual(run.sha256(ROOT / entry["path"]), entry["sha256"])
        rows = run.read_csv(ROOT / "reference/paper_seal_fig2a_rows.csv")
        self.assertEqual(len(rows), 10)
        self.assertAlmostEqual(sum(float(row["auroc"]) for row in rows) / 10, 0.7936975331042188)

    def test_notebook_runs_from_repo_and_notebook_directory_with_every_phase_mocked(self):
        notebook = run.read_json(ROOT / "notebooks/MSI_SEAL_seed0_local.ipynb")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "SEAL notebook", "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
        for launch in ("repo", "notebook"):
            with tempfile.TemporaryDirectory() as temporary:
                repo = Path(temporary)
                reproduction = repo / "00.Reproduction"
                project = reproduction / "04.SEAL_reproduction"
                for relative in ["src", "notebooks", "reference", "environment/.conda-env/bin"]:
                    (project / relative).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / "src/run_seal.py", project / "src/run_seal.py")
                shutil.copyfile(ROOT / "reference/paper_seal_fig2a_summary.csv", project / "reference/paper_seal_fig2a_summary.csv")
                (project / "environment/.conda-env/bin/python").touch()
                for path in run.INPUTS.values():
                    relative = path.relative_to(ROOT.parent)
                    fake = reproduction / relative
                    fake.parent.mkdir(parents=True, exist_ok=True)
                    fake.touch()
                calls = []
                results = project / "results/msi_seed0"

                class Process:
                    pid = 999999

                    def __init__(self, command, **kwargs):
                        phase = command[3]
                        self.stdout = []
                        calls.append((phase, command[-1] if "--fold" in command else None))
                        results.mkdir(parents=True, exist_ok=True)
                        if phase == "prepare":
                            run.save_json(results / "prepare_complete.json", dict(nodes=29959, molecular_rows=478728,
                                simple_edges=1, positives=5926, negatives=5926, node_feature_dimension=132, treatment_edges_added=0))
                        elif phase == "subgraphs":
                            run.save_json(results / "subgraphs_complete.json", dict(pairs=11852, max_nodes=10, label_count=3, sortpooling_k=10))
                        elif phase == "train":
                            fold = int(command[-1])
                            folder = results / f"fold{fold:02d}"
                            run.save_csv(folder / "history.csv", [dict(epoch=1, validation_loss=0.5)])
                            run.save_json(folder / "metrics.json", dict(fold=fold, auroc=0.5))
                        elif phase == "summarize":
                            run.save_csv(results / "fold_results.csv", [dict(fold=1, auroc=0.5)])
                            run.save_csv(results / "summary.csv", [dict(metric="auroc", mean=0.5)])
                        elif phase == "compare":
                            run.save_csv(results / "paper_comparison.csv", [dict(metric="auroc", local_minus_paper=0.0)])

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def wait(self, **kwargs):
                        return 0

                display_module = types.ModuleType("IPython.display")
                display_module.display = lambda *args, **kwargs: None
                display_module.HTML = lambda value: value
                previous = Path.cwd()
                try:
                    os.chdir(repo if launch == "repo" else project / "notebooks")
                    with patch("subprocess.run", return_value=types.SimpleNamespace(returncode=0, stdout="fixture environment", stderr="")), \
                         patch("subprocess.Popen", Process), \
                         patch.dict(sys.modules, {"IPython.display": display_module}), \
                         contextlib.redirect_stdout(io.StringIO()):
                        namespace = {}
                        for cell in notebook["cells"]:
                            if cell["cell_type"] == "code":
                                exec(compile("".join(cell["source"]), "SEAL notebook", "exec"), namespace)
                finally:
                    os.chdir(previous)
                self.assertEqual([name for name, _ in calls], ["config", "prepare", "subgraphs"] + ["train"] * 10 + ["summarize", "compare"])
                self.assertEqual([int(fold) for name, fold in calls if name == "train"], list(range(1, 11)))


if __name__ == "__main__":
    unittest.main()
