"""Small source/provenance fixtures; never run C++ training or classifier fit."""
import difflib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_ns_output_type_control as control


class EndpointPatchTest(unittest.TestCase):
    def test_patch_changes_only_five_pp1_endpoint_references(self):
        original = control.CPP_SOURCE.read_text(encoding="utf-8")
        patched, difference = control.patch_source(original)
        changes = list(difflib.ndiff(original.splitlines(), patched.splitlines()))
        removed = [line[2:] for line in changes if line.startswith("- ")]
        added = [line[2:] for line in changes if line.startswith("+ ")]
        self.assertEqual(len(removed), 5)
        self.assertEqual(len(added), 5)
        self.assertEqual([line.replace("vocab[last_word]", "vocab[word]") for line in removed], added)
        # Input-vector selection and the positive output must retain their roles.
        for statement in ("l1 = last_word * layer1_size;", "target = word;", "label = 1;"):
            self.assertEqual(patched.count(statement), original.count(statement))
        pp = "            if (pp) {\n"
        start = original.index(pp)
        branch_end = original.index("            } else {\n", start)
        new_end = patched.index("            } else {\n", patched.index(pp))
        self.assertEqual(original[:start], patched[:start])
        self.assertEqual(original[branch_end:], patched[new_end:])
        self.assertNotIn("if (target == word)", patched[start:new_end])
        self.assertIn("if (target == word) continue;", patched[new_end:])
        self.assertEqual(original, control.CPP_SOURCE.read_text(encoding="utf-8"))
        self.assertIn("--- vendor/DREAMwalk_upstream/DREAMwalk/HeterogeneousSG.cpp", difference)
        self.assertIn("+++ HeterogeneousSG_output_type.cpp", difference)

    def test_patch_refuses_unrecognized_or_already_modified_source(self):
        original = control.CPP_SOURCE.read_text(encoding="utf-8")
        patched, _ = control.patch_source(original)
        with self.assertRaises(ValueError):
            control.patch_source(patched)
        with self.assertRaises(ValueError):
            control.patch_source(original.replace("            if (pp) {", "            if (pp == 1) {"))
        with self.assertRaises(ValueError):
            control.patch_source(original.replace("            if (pp) {\n",
                "            if (pp) {\n              if (target == word) continue;\n"))

    def test_written_source_and_corpus_are_never_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "saved_walks.txt", root / "walks.txt"
            source.write_bytes(b"d1 g2 i3\ng2 d1 i3")
            control.copy_exact(source, target, allow_hardlink=True)
            self.assertEqual(source.read_bytes(), target.read_bytes())
            control.copy_exact(source, target, allow_hardlink=True)
            target.unlink()  # Break the link before modifying this fixture.
            target.write_bytes(b"different")
            with self.assertRaises(RuntimeError):
                control.copy_exact(source, target)
            output = root / "source.cpp"
            control.write_exact(output, b"original source")
            with self.assertRaises(RuntimeError):
                control.write_exact(output, b"modified source")


class ControlProtocolTest(unittest.TestCase):
    def test_reference_training_arguments_are_retained(self):
        self.assertEqual(control.training_command(),
                         ["./HetSG", "-train", "walks.txt", "-output", "pp1", "-pp", "1"] + control.paired.SG_ARGS)
        self.assertEqual(control.RUN_SETTINGS["tp_factor"], 0)
        self.assertEqual(control.RUN_SETTINGS["negative"], 5)
        self.assertEqual(control.RUN_SETTINGS["sample"], 0.001)

    def test_fold_mismatch_stops_before_loading_embedding_or_fitting(self):
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

    def test_comparison_pairs_fold_ids_and_reports_output_minus_input(self):
        reference = [dict(metric=metric, mean=.4) for metric in control.METRICS]
        current = [dict(metric=metric, mean=.6) for metric in control.METRICS]
        old = [dict(fold=i, accuracy=i/20, auroc=i/20, aupr=i/20) for i in range(1, 11)]
        new = [dict(fold=i, accuracy=i/20+.1, auroc=i/20+.1, aupr=i/20+.1)
               for i in reversed(range(1, 11))]
        summary, deltas = control.comparison_rows(reference, current, old, new)
        self.assertEqual([row["fold"] for row in deltas], list(range(1, 11)))
        for row in summary:
            self.assertAlmostEqual(row["output_minus_input"], .2)
        for row in deltas:
            self.assertAlmostEqual(row["auroc_output_minus_input"], .1)
        with self.assertRaises(ValueError):
            control.comparison_rows(reference, current, old, new[:-1]+[new[0]])


if __name__ == "__main__":
    unittest.main()
