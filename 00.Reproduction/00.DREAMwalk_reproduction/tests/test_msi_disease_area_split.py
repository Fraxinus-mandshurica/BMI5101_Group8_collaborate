"""Scientific split/sampling invariants and a synthetic checkpoint check."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import msi_disease_area_split as area


class DiseaseAreaTests(unittest.TestCase):
    def test_all_tree_positions_and_unmapped_exclusion(self):
        rows = [dict(msi_cui="D1", usable_for_primary_hierarchy="True",
                     mesh_tree_numbers="C04.123;C10.456;C04.789"),
                dict(msi_cui="D2", usable_for_primary_hierarchy="False",
                     mesh_tree_numbers="C04.123")]
        self.assertEqual(area.category_map(rows), {"D1": {"C04", "C10"}, "D2": set()})
        self.assertEqual(rows[0]["mesh_tree_numbers"], "C04.123;C10.456;C04.789")

    def test_multicategory_priority_prevents_training_exposure(self):
        categories = {f"D{i}": {f"C{i:02d}"} for i in range(10)}
        allocated, _ = area.split_categories(categories, 0)
        train = next(c for c, p in allocated.items() if p == "train")
        validation = next(c for c, p in allocated.items() if p == "validation")
        test = next(c for c, p in allocated.items() if p == "test")
        categories.update(MIXED_TEST={train, validation, test}, MIXED_VALIDATION={train, validation})
        allocated, diseases = area.split_categories(categories, 0)
        self.assertEqual(diseases["MIXED_TEST"], "test")
        self.assertEqual(diseases["MIXED_VALIDATION"], "validation")
        self.assertEqual(list(allocated.values()).count("train"), 8)
        for disease, partition in diseases.items():
            memberships = {allocated[c] for c in categories[disease]}
            if partition == "train":
                self.assertEqual(memberships, {"train"})
            if partition == "validation":
                self.assertNotIn("test", memberships)
        self.assertEqual(area.split_categories(categories, 0), (allocated, diseases))

    def test_negatives_are_disjoint_unknown_pairs(self):
        drugs, diseases = [f"R{i}" for i in range(10)], [f"D{i}" for i in range(10)]
        positives = {(drug, diseases[i]) for i, drug in enumerate(drugs)}
        used = set()
        for seed in range(4):
            negatives = area.sample_negatives(drugs, diseases, positives, used, 10, seed)
            self.assertEqual(len(set(negatives)), 10)
            self.assertFalse(set(negatives) & (positives | used))
            self.assertEqual(negatives, area.sample_negatives(drugs, diseases, positives, used, 10, seed))
            used.update(negatives)
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            area.sample_negatives(["R"], ["D"], {("R", "D")}, set(), 1, 0)

    def test_primary_assignment_is_fixed_and_order_independent(self):
        categories = {"D1": {"C04", "C10"}, "D2": {"F03"}, "D3": {"C10", "F03"}}
        primary = area.select_primary_categories(categories, 0)
        self.assertEqual(primary, area.select_primary_categories(dict(reversed(list(categories.items()))), 0))
        self.assertTrue(all(category in categories[disease] for disease, category in primary.items()))
        self.assertEqual(primary["D2"], "F03")
        with self.assertRaisesRegex(ValueError, "category for every"):
            area.select_primary_categories({"D1": set()})

    def test_primary_split_relaxes_category_overlap_without_duplicate_diseases(self):
        categories = {f"D{i}": {f"C{i:02d}"} for i in range(10)}
        allocated, _ = area.split_categories(categories, 0)
        train = next(c for c, p in allocated.items() if p == "train")
        test = next(c for c, p in allocated.items() if p == "test")
        categories["MIXED"] = {train, test}
        primary = {disease: sorted(values)[0] for disease, values in categories.items()}
        primary["MIXED"] = train
        strict_allocated, strict = area.split_categories(categories, 0)
        primary_allocated, relaxed = area.split_categories(categories, 0, primary)
        self.assertEqual(strict_allocated, primary_allocated)
        self.assertEqual(strict["MIXED"], "test")
        self.assertEqual(relaxed["MIXED"], "train")
        groups = [{d for d, p in relaxed.items() if p == partition} for partition in area.PARTITIONS]
        self.assertEqual(set.union(*groups), set(categories))
        self.assertTrue(all(not groups[i] & groups[j] for i in range(3) for j in range(i)))
        with self.assertRaisesRegex(ValueError, "Primary categories"):
            area.split_categories(categories, 0, {"MIXED": train})
        with self.assertRaisesRegex(ValueError, "Primary categories"):
            area.split_categories(categories, 0, dict(primary, MIXED="C99"))

    def test_primary_repeat_changes_only_partitions(self):
        categories = {f"D{i}": {f"C{i:02d}"} for i in range(10)}
        categories["MIXED"] = {"C00", "C01", "C02"}
        positives = [("R1", disease) for disease in sorted(categories)]
        negatives = [("R2", disease) for disease in sorted(categories)]
        cohort = dict(categories=categories, drugs=["R1", "R2"],
                      positives=positives, all_positives=set(positives))
        primary = area.select_primary_categories(categories, 0)
        with patch.object(area, "sample_negatives", return_value=negatives):
            strict = area.make_repeat(cohort, 0, set())
            relaxed = area.make_repeat(cohort, 0, set(), primary)
        self.assertEqual(strict[0], relaxed[0])
        self.assertEqual(strict[3], relaxed[3])
        self.assertEqual([(r["drug"], r["disease"], r["label"]) for r in strict[2]],
                         [(r["drug"], r["disease"], r["label"]) for r in relaxed[2]])
        self.assertTrue(all(row["partition"] == relaxed[0][primary[row["disease"]]]
                            for row in relaxed[2]))

    def test_cohort_keeps_embeddings_scope_but_excludes_unclassified_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            area.write_text(root / "data/processed/msi_nodetypes.tsv",
                            "node\ttype\nR1\tdrug\nR2\tdrug\nD1\tdisease\nD2\tdisease\n")
            area.write_text(root / "data/processed/msi_positive_pairs.tsv",
                            "drug\tdisease\tlabel\nR1\tD1\t1\nR2\tD2\t1\n")
            mapping = [dict(msi_cui="D1", mesh_descriptor_id="M1", mapping_status="USABLE",
                            usable_for_primary_hierarchy="True", mesh_tree_numbers="C04.1"),
                       dict(msi_cui="D2", mesh_descriptor_id="", mapping_status="UNMAPPED",
                            usable_for_primary_hierarchy="False", mesh_tree_numbers="")]
            area.write_rows(root / "results/msi_mesh2024_demo_roots_inputs/mapping_audit.csv",
                            mapping, list(mapping[0]))
            cohort = area.load_cohort(root)
            self.assertEqual(cohort["drugs"], ["R1", "R2"])
            self.assertEqual(cohort["positives"], [("R1", "D1")])
            self.assertEqual(cohort["excluded_positives"], [("R2", "D2")])
            self.assertIn(("R2", "D2"), cohort["all_positives"])

    def test_resume_and_incomplete_checkpoint(self):
        import numpy as np
        vectors = {"R": np.array([1., 0.]), "S": np.array([0., 1.]), "D": np.zeros(2)}
        pairs = [dict(drug=drug, disease="D", label=label, partition=partition)
                 for partition in area.PARTITIONS for drug, label in [("R", 1), ("S", 0)]]

        class StubClassifier:
            fits = 0

            def __init__(self, **parameters):
                self.parameters = parameters

            def fit(self, x, y):
                StubClassifier.fits += 1
                self.training_rows = len(y)
                if len(y) != 2:
                    raise AssertionError("Validation/test labels entered training")

            def predict_proba(self, x):
                scores = np.where(x[:, 0] > 0, 0.8, 0.2)
                return np.column_stack((1 - scores, scores))

            def save_model(self, path):
                Path(path).write_text(json.dumps(self.parameters))

        with tempfile.TemporaryDirectory() as directory, patch("xgboost.XGBClassifier", StubClassifier):
            root = Path(directory)
            area.open_run(root, {"scientific_setting": 1})
            output = root / "original/repeat00"
            rows = area.fit_repeat(vectors, pairs, output, "original", 0, {"seed": 0})
            self.assertEqual({row["partition"] for row in rows}, {"validation", "test"})
            self.assertTrue(all(row["auroc"] == 1.0 for row in rows))
            self.assertEqual(area.fit_repeat(vectors, pairs, output, "original", 0, {"seed": 0}), rows)
            self.assertEqual(StubClassifier.fits, 1)
            with self.assertRaisesRegex(ValueError, "changed"):
                area.open_run(root, {"scientific_setting": 2})
            (output / "predictions.csv").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                area.fit_repeat(vectors, pairs, output, "original", 0, {"seed": 0})

    def test_paired_summary_matches_seeds_and_excludes_validation(self):
        rows = []
        for name, seed, value in [("modified", 1, 0.75), ("original", 0, 0.6),
                                  ("modified", 0, 0.7), ("original", 1, 0.8)]:
            for partition in ("test", "validation"):
                score = value if partition == "test" else 0.99
                rows.append(dict(model=name, repeat_seed=seed, partition=partition,
                                 accuracy=score, auroc=score, aupr=score))
        deltas, summary = area.paired_deltas(rows)
        self.assertAlmostEqual(deltas[0]["auroc"], 0.1)
        self.assertAlmostEqual(deltas[1]["auroc"], -0.05)
        self.assertAlmostEqual(summary[0]["mean_modified_minus_original"], 0.025)
        means = {(row["model"], row["partition"], row["metric"]): row["mean"]
                 for row in area.summarize(rows)}
        self.assertAlmostEqual(means[("original", "test", "auroc")], 0.7)
        self.assertAlmostEqual(means[("modified", "test", "auroc")], 0.725)


if __name__ == "__main__":
    unittest.main()
