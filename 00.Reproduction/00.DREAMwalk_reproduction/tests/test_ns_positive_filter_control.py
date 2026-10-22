"""Small source/provenance fixtures; no full walks, trainer or classifier fit."""
import difflib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_ns_positive_filter_control as control


class PositiveOutputPatchTest(unittest.TestCase):
    def setUp(self):
        # Build the fixture from released code without a completed MSI experiment.
        source = control.baseline.CPP_SOURCE.read_text(encoding="utf-8")
        self.original = control.baseline.patch_source(source)[0]

    def test_only_one_line_is_added_and_pp0_and_output_pool_are_unchanged(self):
        original = self.original
        modified, difference = control.patch_source(original)
        changes = list(difflib.ndiff(original.splitlines(), modified.splitlines()))
        self.assertEqual([line[2:] for line in changes if line.startswith("- ")], [])
        self.assertEqual([line[2:] for line in changes if line.startswith("+ ")],
                         ["              if (target == word) continue;"])
        start = original.index("            if (pp) {\n")
        old_end = original.index("            } else {\n", start)
        new_end = modified.index("            } else {\n", start)
        self.assertEqual(original[:start], modified[:start])
        self.assertEqual(original[old_end:], modified[new_end:])
        self.assertEqual(modified[start:new_end].count("vocab[word].word[0]"), 5)
        self.assertNotIn("vocab[last_word].word[0]", modified[start:new_end])
        for statement in ("l1 = last_word * layer1_size;", "target = word;", "label = 1;"):
            self.assertEqual(original.count(statement), modified.count(statement))
        self.assertIn("+++ HeterogeneousSG_output_type_reject_positive.cpp", difference)

    def test_rejection_skips_this_draw_without_resampling_or_filtering_other_pairs(self):
        original = self.original
        modified, _ = control.patch_source(original)
        self.assertIn("if (negative > 0) for (d = 0; d < negative + 1; d++)", modified)
        branch = modified[modified.index("            if (pp) {\n"):]
        branch = branch[:branch.index("            } else {\n")]
        self.assertTrue(branch.endswith("              if (target == word) continue;\n"))
        # The sole inserted continue targets the unchanged outer d++ loop.
        # No retry loop, d decrement, known-positive list or window scan is added.
        self.assertNotIn("while", branch)
        self.assertNotIn("d--", branch)
        self.assertNotIn("--d", branch)
        self.assertNotIn("for (", branch)
        self.assertEqual(modified.count("next_random ="), original.count("next_random ="))
        self.assertEqual(modified.count("if (target == word) continue;"),
                         original.count("if (target == word) continue;") + 1)

    def test_patch_rejects_already_modified_and_input_type_sources(self):
        original = self.original
        modified, _ = control.patch_source(original)
        with self.assertRaises(ValueError):
            control.patch_source(modified)
        with self.assertRaises(ValueError):
            control.patch_source(original.replace("vocab[word].word[0]", "vocab[last_word].word[0]"))
        with self.assertRaises(ValueError):
            control.patch_source(original.replace("            if (pp) {", "            if (pp == 1) {"))


class FilterControlProtocolTest(unittest.TestCase):
    def test_public_parameters_and_training_arguments_are_retained(self):
        self.assertEqual(control.training_command(), control.baseline.training_command())
        for key in ("pp", "tp_factor", "seed", "num_walks", "walk_length", "dimension", "window_size",
                    "workers", "epochs", "negative", "sample", "folds", "cv_seed"):
            self.assertEqual(control.RUN_SETTINGS[key], control.baseline.RUN_SETTINGS[key])
        self.assertTrue(control.RUN_SETTINGS["reject_positive_output"])
        self.assertEqual(control.RUN_SETTINGS["rejection_policy"], "skip_without_resampling")

    def test_completed_baseline_run_names_are_refused_before_writing(self):
        with patch.object(control, "save_json") as write:
            for name in (control.REFERENCE_NAME, control.baseline.REFERENCE_NAME):
                with self.assertRaisesRegex(ValueError, "completed baseline"):
                    control.context(name)
            write.assert_not_called()

    def test_fold_mismatch_stops_before_embedding_load_or_classifier_fit(self):
        positives = {(f"D{i}", f"I{i}") for i in range(10)}
        negatives = [dict(drug=f"D{i}", disease=f"N{i}") for i in range(10)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def fixture_rows(path, delimiter):
                return negatives if path == control.previous.NEGATIVES else [dict(wrong="fold assignments")]
            with patch.object(control, "require_embedding", return_value=dict(sha256="fixture")), \
                    patch.object(control.paired, "completed", return_value=False), \
                    patch.object(control.previous, "base_inputs", return_value=({}, [], [], positives)), \
                    patch.object(control.previous, "read_rows", side_effect=fixture_rows), \
                    patch.object(control.previous, "load_embedding") as load, \
                    patch.object(control.previous, "evaluate") as fit:
                with self.assertRaisesRegex(RuntimeError, "before fitting"):
                    control.evaluate(root, root, "fixture")
                load.assert_not_called()
                fit.assert_not_called()
            self.assertFalse((root / "fold_assignments.csv").exists())

    def test_comparison_pairs_folds_and_reports_filtered_minus_unfiltered(self):
        old_summary = [dict(metric=metric, mean=.4) for metric in control.METRICS]
        new_summary = [dict(metric=metric, mean=.6) for metric in control.METRICS]
        old = [dict(fold=i, accuracy=i/20, auroc=i/20, aupr=i/20) for i in range(1, 11)]
        new = [dict(fold=i, accuracy=i/20+.1, auroc=i/20+.1, aupr=i/20+.1)
               for i in reversed(range(1, 11))]
        summary, deltas = control.comparison_rows(old_summary, new_summary, old, new)
        self.assertEqual([row["fold"] for row in deltas], list(range(1, 11)))
        for row in summary:
            self.assertAlmostEqual(row["output_type_unfiltered"], .4)
            self.assertAlmostEqual(row["output_type_filtered"], .6)
            self.assertAlmostEqual(row["filtered_minus_unfiltered"], .2)
        for row in deltas:
            self.assertAlmostEqual(row["auroc_filtered_minus_unfiltered"], .1)
        with self.assertRaises(ValueError):
            control.comparison_rows(old_summary, new_summary, old, new[:-1]+[new[0]])

    def test_baseline_inspection_is_read_only_and_detects_artifact_changes(self):
        """Inspect tiny saved artifacts; never create the new run or invoke trainers."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, results = root / "data", root / "results"
            data.mkdir()
            results.mkdir()
            for name in ("walks.txt", "token_to_node.json", "HetSG", "pp1.pkl", "patch.diff",
                         control.baseline.PATCHED_NAME):
                (data / name).write_text(name, encoding="utf-8")
            for name in ("pp1_summary.csv", "pp1_fold_results.csv", "fold_assignments.csv",
                         "ns_pool_comparison.csv", "paired_fold_deltas.csv"):
                (results / name).write_text(name, encoding="utf-8")
            digest = lambda name: control.sha256(data / name)
            result_digest = lambda name: control.sha256(results / name)
            source_identity = dict(original_source_sha256="upstream fixture",
                                   patched_source_sha256=digest(control.baseline.PATCHED_NAME),
                                   patch_diff_sha256=digest("patch.diff"))
            reference_identity = dict(inputs={"negatives": "saved negative fixture"}, artifacts=dict(
                corpus=digest("walks.txt"), token_map=digest("token_to_node.json"),
                fold_assignments=result_digest("fold_assignments.csv")))
            config = dict(fingerprint="saved fixture", runner_sha256=control.sha256(Path(control.baseline.__file__)),
                          settings=control.baseline.RUN_SETTINGS, reference=reference_identity, patch=source_identity,
                          skipgram_args=control.paired.SG_ARGS, compile_flags=control.paired.COMPILE_FLAGS,
                          xgboost=control.previous.xgb_settings(), upstream_commit=control.paired.PINNED_COMMIT)
            markers = {
                "walks_complete.json": dict(sha256=digest("walks.txt"), token_map_sha256=digest("token_to_node.json")),
                "compiler_complete.json": dict(source_identity, sha256=digest("HetSG"),
                    command=["g++", control.baseline.PATCHED_NAME, "-o", "HetSG.build"] + control.paired.COMPILE_FLAGS),
                "pp1_embedding_complete.json": dict(sha256=digest("pp1.pkl"), corpus_sha256=digest("walks.txt"),
                    token_map_sha256=digest("token_to_node.json"), binary_sha256=digest("HetSG"), pp=1,
                    command=control.training_command()),
                "pp1_evaluation_complete.json": dict(sha256=result_digest("pp1_summary.csv"),
                    embedding_sha256=digest("pp1.pkl"), fold_results_sha256=result_digest("pp1_fold_results.csv"),
                    fold_assignments_sha256=result_digest("fold_assignments.csv")),
                "comparison_complete.json": dict(sha256=result_digest("ns_pool_comparison.csv"),
                    paired_fold_deltas_sha256=result_digest("paired_fold_deltas.csv"))}
            (results / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            for name, marker in markers.items():
                (results / name).write_text(json.dumps(dict(marker, fingerprint="saved fixture")), encoding="utf-8")
            before = {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            with patch.object(control, "ROOT", root), patch.object(control, "REFERENCE_DATA", data), \
                    patch.object(control, "REFERENCE_RESULTS", results), \
                    patch.object(control, "CPP_SOURCE", data / control.baseline.PATCHED_NAME), \
                    patch.object(control.baseline, "inspect_reference", return_value=reference_identity), \
                    patch.object(control.baseline, "patch_identity", return_value=source_identity), \
                    patch.object(control.paired, "train") as train, \
                    patch.object(control.previous, "evaluate") as fit:
                inspected = control.inspect_reference()
                self.assertEqual(inspected["artifacts"]["binary"], digest("HetSG"))
                self.assertEqual(before, {str(path): path.read_bytes() for path in root.rglob("*") if path.is_file()})
                (results / "pp1_fold_results.csv").write_text("changed fixture", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "fold results"):
                    control.inspect_reference()
                train.assert_not_called()
                fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
