"""Tiny fixtures for the category-layer input correction; never preprocess MSI."""
import contextlib
import csv
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import msi_mesh2021_demo_roots as inputs
import msi_mesh2021_inputs as historical


MESH = """*NEWRECORD
UI = D1
MN = C01

*NEWRECORD
UI = D2
MN = C01.100

*NEWRECORD
UI = D3
MN = C01.200
MN = C02.100

*NEWRECORD
UI = D4
MN = C02

*NEWRECORD
UI = D5
MN = A01

*NEWRECORD
UI = D6
MN = E01

*NEWRECORD
UI = D7
MN = G01

*NEWRECORD
UI = D8
MN = F01

"""


def write_csv(path, rows, columns, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


class CategoryLayerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        settings = dict(ROOT=self.root, MESH_PATH=self.root / "data/d2021.bin",
                        HIERARCHY_PATH=self.root / "processed/new_hierarchy.csv",
                        SIMILARITY_PATH=self.root / "processed/new_similarity.tsv",
                        AUDIT_DIR=self.root / "audits/new",
                        GRAPH_PATH=self.root / "processed/graph.tsv",
                        NODETYPES_PATH=self.root / "processed/nodetypes.tsv",
                        MAPPING_PATH=self.root / "source/mapping.csv",
                        ATC_PATH=self.root / "source/atc.csv")
        for name, value in settings.items():
            patcher = patch.object(inputs, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        inputs.MESH_PATH.parent.mkdir(parents=True)
        inputs.MESH_PATH.write_text(MESH, encoding="ascii")
        patcher = patch.object(inputs, "EXPECTED_MESH_SHA256", inputs.sha256(inputs.MESH_PATH))
        patcher.start()
        self.addCleanup(patcher.stop)
        types = [("drug1", "drug"), ("drug2", "drug"), ("gene", "gene")]
        types += [(f"CUI{index}", "disease") for index in range(1, 7)]
        write_csv(inputs.NODETYPES_PATH,
                  [dict(node=node, type=kind) for node, kind in types], ["node", "type"], "\t")
        inputs.GRAPH_PATH.write_text("".join(f"{node}\tgene\t1\t1.0\t{index}\n"
                                    for index, (node, _) in enumerate(types) if node != "gene"))
        mapping = [dict(msi_cui=f"CUI{index}", all_descriptor_ids=descriptor,
                        all_supplementary_ids="", n_unexpected_ids=0)
                   for index, descriptor in enumerate(("D2", "D3", "D5", "D6", "D7", "D8"), 1)]
        write_csv(inputs.MAPPING_PATH, mapping, list(mapping[0]))
        self.atc_edges = {("drug1", "A"), ("A", "drug"), ("drug2", "C"), ("C", "drug")}
        write_csv(inputs.ATC_PATH, [dict(child=child, parent=parent)
                                   for child, parent in sorted(self.atc_edges)], ["child", "parent"])
        self.historical_artifacts = [self.root / "processed/msi_mesh2021_semantic_hierarchy.csv",
                                     self.root / "processed/msi_mesh2021_semantic_similarity.tsv"]
        for path in self.historical_artifacts:
            path.write_text("frozen historical fixture\n")
        self.old_global_paths = {name: getattr(historical, name) for name in settings}
        self.original_input_hashes = {path: inputs.sha256(path) for path in (
            inputs.MESH_PATH, inputs.GRAPH_PATH, inputs.NODETYPES_PATH,
            inputs.MAPPING_PATH, inputs.ATC_PATH, *self.historical_artifacts)}

    def build_hierarchy(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return inputs.build_hierarchy()

    def test_all_used_letters_inserted_without_atc_merging_and_exact_paths_preserved(self):
        manifest = self.build_hierarchy()
        self.assertEqual(manifest["summary"]["mesh_category_ids"],
                         ["A_dis", "C_dis", "E_dis", "F_dis", "G_dis"])
        import pandas as pd
        tree = inputs._upstream_similarity_module()._generate_tree(
            pd.read_csv(inputs.HIERARCHY_PATH),
            ["drug1", "drug2"] + [f"CUI{index}" for index in range(1, 7)])
        self.assertEqual({tuple(row) for row in tree["disease"]["CUI2"]}, {
            ("CUI2", "D3", "D3@C01.200", "D1@C01", "C_dis", "disease"),
            ("CUI2", "D3", "D3@C02.100", "D4@C02", "C_dis", "disease")})
        for index, letter in ((3, "A"), (4, "E"), (5, "G"), (6, "F")):
            self.assertEqual(tree["disease"][f"CUI{index}"][0][-2:], [letter + "_dis", "disease"])
        self.assertEqual(tree["drug"]["drug1"], [["drug1", "A", "drug"]])
        self.assertEqual(tree["drug"]["drug2"], [["drug2", "C", "drug"]])
        with inputs.HIERARCHY_PATH.open(newline="") as stream:
            edges = {(row["child"], row["parent"]) for row in csv.DictReader(stream)}
        self.assertTrue(self.atc_edges <= edges)
        self.assertNotIn(("D1@C01", "disease"), edges)
        self.assertIn(("D1@C01", "C_dis"), edges)
        old_edges, old_paths, _ = historical._disease_paths(
            {f"CUI{index}": descriptor for index, descriptor in
             enumerate(("D2", "D3", "D5", "D6", "D7", "D8"), 1)}, inputs.read_mesh(inputs.MESH_PATH))
        for cui, paths in tree["disease"].items():
            self.assertEqual({tuple(path[:-2] + path[-1:]) for path in paths}, old_paths[cui])

    def test_original_similarity_generator_and_checkpoint_do_not_modify_old_artifacts(self):
        self.build_hierarchy()
        upstream = inputs._upstream_similarity_module()
        with patch.object(upstream, "save_sim_graph", wraps=upstream.save_sim_graph) as save:
            with contextlib.redirect_stdout(io.StringIO()):
                first = inputs.build_similarity()
                second = inputs.build_similarity()
                artifacts = inputs.artifact_identity()
            self.assertEqual(save.call_count, 1)
            self.assertEqual(save.call_args.kwargs,
                             dict(cutoff=0.4, weighted=True, directed=False, net_delimiter="\t"))
            self.assertEqual(save.call_args.args[:2],
                             (str(inputs.GRAPH_PATH), str(inputs.HIERARCHY_PATH)))
        self.assertEqual(first, second)
        self.assertEqual(artifacts["similarity"]["sha256"], inputs.sha256(inputs.SIMILARITY_PATH))
        self.assertEqual({name: getattr(historical, name) for name in self.old_global_paths},
                         self.old_global_paths)
        for path, digest in self.original_input_hashes.items():
            self.assertEqual(inputs.sha256(path), digest)
        identity = inputs.input_identity()
        self.assertIn("shared_pure_helpers_source", identity["sources"])
        self.assertIn("upstream_demo_hierarchy", identity["sources"])

    def test_changed_source_or_output_is_rejected_and_preserved(self):
        self.build_hierarchy()
        old_hierarchy = inputs.HIERARCHY_PATH.read_bytes()
        inputs.ATC_PATH.write_text(inputs.ATC_PATH.read_text() + "B,drug\n")
        with self.assertRaisesRegex(RuntimeError, "fingerprint changed"):
            self.build_hierarchy()
        self.assertEqual(inputs.HIERARCHY_PATH.read_bytes(), old_hierarchy)

    def test_unmanifested_existing_output_is_not_overwritten(self):
        inputs.HIERARCHY_PATH.parent.mkdir(parents=True, exist_ok=True)
        inputs.HIERARCHY_PATH.write_text("old unfinished output\n")
        with self.assertRaisesRegex(RuntimeError, "Unmanifested"):
            self.build_hierarchy()
        self.assertEqual(inputs.HIERARCHY_PATH.read_text(), "old unfinished output\n")


if __name__ == "__main__":
    unittest.main()
