"""Small year-revalidation/path/provenance fixtures; no real MSI preprocessing."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
import msi_mesh2021_inputs as inputs


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

"""


def write_csv(path, rows, columns, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


class Mesh2021InputsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        settings = dict(ROOT=self.root, MESH_PATH=self.root / "data/d2021.bin",
                        HIERARCHY_PATH=self.root / "processed/hierarchy.csv",
                        SIMILARITY_PATH=self.root / "processed/similarity.tsv",
                        AUDIT_DIR=self.root / "audits",
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
        types += [(f"C{index}", "disease") for index in range(1, 8)]
        write_csv(inputs.NODETYPES_PATH,
                  [dict(node=node, type=kind) for node, kind in types], ["node", "type"], "\t")
        inputs.GRAPH_PATH.write_text("".join(f"{node}\tgene\t1\t1.0\t{index}\n"
                                    for index, (node, _) in enumerate(types) if node != "gene"))
        descriptors = ("D2", "D3", "D404", "D5", "", "D2;D404", "D2")
        mapping = [dict(msi_cui=f"C{index}", all_descriptor_ids=value,
                        all_supplementary_ids="C888" if index == 5 else "",
                        n_unexpected_ids=int(index == 7),
                        usable_for_primary_hierarchy="False")
                   for index, value in enumerate(descriptors, 1)]
        write_csv(inputs.MAPPING_PATH, mapping, list(mapping[0]))
        write_csv(inputs.ATC_PATH, [dict(child="drug1", parent="A"),
                                   dict(child="drug2", parent="A"),
                                   dict(child="A", parent="drug")], ["child", "parent"])

    def build_hierarchy(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return inputs.build_hierarchy()

    def test_year_revalidation_ignores_old_flag_and_does_not_infer_missing_ids(self):
        manifest = self.build_hierarchy()
        self.assertEqual(manifest["summary"]["hierarchy_covered_diseases"], 2)
        self.assertEqual(manifest["summary"]["uncovered_diseases"], 5)
        with (inputs.AUDIT_DIR / "mapping_audit.csv").open(newline="") as stream:
            rows = {row["msi_cui"]: row for row in csv.DictReader(stream)}
        self.assertEqual(rows["C1"]["mapping_status"], "DIRECT_UNIQUE_USABLE")
        self.assertEqual(rows["C3"]["mapping_status"], "DIRECT_DESCRIPTOR_NOT_IN_2021")
        self.assertEqual(rows["C4"]["mapping_status"], "DIRECT_DESCRIPTOR_NO_TREE")
        self.assertEqual(rows["C5"]["mapping_status"], "SUPPLEMENTARY_UNRESOLVED")
        self.assertEqual(rows["C6"]["mapping_status"], "DIRECT_MULTIPLE_DESCRIPTOR")
        self.assertEqual(rows["C7"]["mapping_status"], "UNEXPECTED")
        with (inputs.AUDIT_DIR / "descriptor_candidates.csv").open(newline="") as stream:
            candidates = list(csv.DictReader(stream))
        self.assertTrue(any(row["mesh_descriptor_id"] == "D404"
                            and row["descriptor_validation"] == "ABSENT_FROM_MESH2021"
                            for row in candidates))

    def test_multitree_descriptor_has_exact_paths_under_original_upstream_merger(self):
        self.build_hierarchy()
        import pandas as pd
        tree = inputs._upstream_similarity_module()._generate_tree(
            pd.read_csv(inputs.HIERARCHY_PATH), ["C1", "C2", "drug1", "drug2"])
        self.assertEqual({tuple(row) for row in tree["disease"]["C2"]}, {
            ("C2", "D3", "D3@C01.200", "D1@C01", "disease"),
            ("C2", "D3", "D3@C02.100", "D4@C02", "disease")})
        self.assertEqual({tuple(row) for row in tree["drug"]["drug1"]},
                         {("drug1", "A", "drug")})
        self.assertEqual(tree["disease"]["C1"], [["C1", "D2", "D2@C01.100", "D1@C01", "disease"]])

    def test_similarity_uses_original_generator_exact_arguments_and_checkpoints(self):
        self.build_hierarchy()
        upstream = inputs._upstream_similarity_module()
        with patch.object(upstream, "save_sim_graph", wraps=upstream.save_sim_graph) as save:
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = inputs.build_similarity()
            self.assertEqual(save.call_count, 1)
            self.assertEqual(save.call_args.kwargs,
                             dict(cutoff=0.4, weighted=True, directed=False, net_delimiter="\t"))
            self.assertEqual(save.call_args.args[:2],
                             (str(inputs.GRAPH_PATH), str(inputs.HIERARCHY_PATH)))
            with contextlib.redirect_stdout(io.StringIO()):
                repeated = inputs.build_similarity()
                identity = inputs.artifact_identity()
            self.assertEqual(save.call_count, 1)
        self.assertEqual(manifest, repeated)
        self.assertEqual(identity["similarity"]["sha256"], inputs.sha256(inputs.SIMILARITY_PATH))
        self.assertEqual(manifest["summary"]["cutoff"], 0.4)
        self.assertFalse(inputs.MESH_PATH.parent.joinpath("MRCONSO.RRF").exists())

    def test_changed_source_and_changed_output_do_not_overwrite_checkpoint(self):
        first = self.build_hierarchy()
        old_hierarchy = inputs.HIERARCHY_PATH.read_bytes()
        inputs.ATC_PATH.write_text(inputs.ATC_PATH.read_text() + "B,drug\n")
        with self.assertRaisesRegex(RuntimeError, "fingerprint changed"):
            self.build_hierarchy()
        self.assertEqual(inputs.HIERARCHY_PATH.read_bytes(), old_hierarchy)
        self.assertEqual(json.loads((inputs.AUDIT_DIR / "hierarchy_manifest.json").read_text()), first)

    def test_changed_artifact_cannot_be_reused(self):
        self.build_hierarchy()
        inputs.HIERARCHY_PATH.write_text("child,parent\nwrong,disease\n")
        with self.assertRaisesRegex(RuntimeError, "missing or changed"):
            self.build_hierarchy()


if __name__ == "__main__":
    unittest.main()
