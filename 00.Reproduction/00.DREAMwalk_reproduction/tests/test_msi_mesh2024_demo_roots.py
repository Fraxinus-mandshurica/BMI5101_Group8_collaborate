"""Small independent fixtures; do not prepare or train the actual MSI run."""
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
import msi_mesh2024_demo_roots as inputs
import msi_mesh2021_inputs as historical


MESH = """*NEWRECORD
UI = D1
MN = C01

*NEWRECORD
UI = D2
MN = C01.100
MN = C02.100

*NEWRECORD
UI = D3
MN = C02

*NEWRECORD
UI = D4
MN = A01

*NEWRECORD
UI = D5
MN = F01

*NEWRECORD
UI = D6
MN = G01

*NEWRECORD
UI = D7
MN = E01

"""


def write_csv(path, rows, columns, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


class Mesh2024CategoryInputsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        settings = dict(ROOT=self.root, MESH_PATH=self.root / "data/d2024.bin",
                        HIERARCHY_PATH=self.root / "new/hierarchy.csv",
                        SIMILARITY_PATH=self.root / "new/similarity.tsv",
                        AUDIT_DIR=self.root / "audits/new",
                        GRAPH_PATH=self.root / "source/graph.tsv",
                        NODETYPES_PATH=self.root / "source/nodetypes.tsv",
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
                   for index, descriptor in enumerate(("D2", "D4", "D5", "D6", "D7", "D999"), 1)]
        write_csv(inputs.MAPPING_PATH, mapping, list(mapping[0]))
        self.atc_edges = {("drug1", "A"), ("A", "drug"), ("drug2", "C"), ("C", "drug")}
        write_csv(inputs.ATC_PATH, [dict(child=c, parent=p) for c, p in sorted(self.atc_edges)],
                  ["child", "parent"])
        self.historical_artifacts = [self.root / "source/msi_semantic_hierarchy.csv",
                                     self.root / "source/msi_mesh2021_semantic_hierarchy.csv"]
        for path in self.historical_artifacts:
            path.write_text("historical fixture remains unchanged\n")
        self.old_globals = {name: getattr(historical, name) for name in settings}
        self.input_hashes = {path: inputs.sha256(path) for path in (
            inputs.MESH_PATH, inputs.GRAPH_PATH, inputs.NODETYPES_PATH,
            inputs.MAPPING_PATH, inputs.ATC_PATH, *self.historical_artifacts)}

    def build_hierarchy(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return inputs.build_hierarchy()

    def test_true_paths_all_letter_roots_and_correct_2024_mapping_labels(self):
        manifest = self.build_hierarchy()
        self.assertEqual(manifest["input_identity"]["mesh_year"], 2024)
        self.assertEqual(manifest["summary"]["hierarchy_covered_diseases"], 5)
        self.assertEqual(manifest["summary"]["mesh_category_ids"],
                         ["A_dis", "C_dis", "E_dis", "F_dis", "G_dis"])
        self.assertEqual(manifest["summary"]["shared_drug_disease_hierarchy_node_ids"], 0)
        import pandas as pd
        tree = inputs._variant().upstream_similarity_module()._generate_tree(
            pd.read_csv(inputs.HIERARCHY_PATH),
            ["drug1", "drug2"] + [f"CUI{index}" for index in range(1, 7)])
        self.assertEqual({tuple(path) for path in tree["disease"]["CUI1"]}, {
            ("CUI1", "D2", "D2@C01.100", "D1@C01", "C_dis", "disease"),
            ("CUI1", "D2", "D2@C02.100", "D3@C02", "C_dis", "disease")})
        self.assertNotIn("CUI6", tree["disease"])
        self.assertEqual(tree["drug"]["drug1"], [["drug1", "A", "drug"]])
        self.assertEqual(tree["drug"]["drug2"], [["drug2", "C", "drug"]])
        with (inputs.AUDIT_DIR / "mapping_audit.csv").open(newline="") as stream:
            audit = {row["msi_cui"]: row for row in csv.DictReader(stream)}
        self.assertEqual(audit["CUI6"]["mapping_status"], "DIRECT_DESCRIPTOR_NOT_IN_2024")
        with (inputs.AUDIT_DIR / "descriptor_candidates.csv").open(newline="") as stream:
            candidates = {row["msi_cui"]: row for row in csv.DictReader(stream)}
        self.assertEqual(candidates["CUI6"]["descriptor_validation"], "ABSENT_FROM_MESH2024")
        with (inputs.AUDIT_DIR / "hierarchy_namespace_audit.csv").open(newline="") as stream:
            namespace = {row["hierarchy_node"]: row["semantic_branch"] for row in csv.DictReader(stream)}
        self.assertEqual(namespace["C"], "drug")
        self.assertEqual(namespace["C_dis"], "disease")

    def test_original_similarity_checkpoint_preserves_historical_inputs_and_globals(self):
        self.build_hierarchy()
        upstream = inputs._variant().upstream_similarity_module()
        with patch.object(upstream, "save_sim_graph", wraps=upstream.save_sim_graph) as save:
            with contextlib.redirect_stdout(io.StringIO()):
                first = inputs.build_similarity()
                second = inputs.build_similarity()
                artifacts = inputs.artifact_identity()
        self.assertEqual(save.call_count, 1)
        self.assertEqual(save.call_args.args[:2],
                         (str(inputs.GRAPH_PATH), str(inputs.HIERARCHY_PATH)))
        self.assertEqual(save.call_args.kwargs,
                         dict(cutoff=0.4, weighted=True, directed=False, net_delimiter="\t"))
        self.assertEqual(first, second)
        self.assertEqual(artifacts["similarity"]["sha256"], inputs.sha256(inputs.SIMILARITY_PATH))
        for path, digest in self.input_hashes.items():
            self.assertEqual(inputs.sha256(path), digest)
        self.assertEqual({name: getattr(historical, name) for name in self.old_globals}, self.old_globals)
        sources = inputs.input_identity()["sources"]
        self.assertIn("configurable_helpers_source", sources)
        self.assertIn("shared_pure_helpers_source", sources)

    def test_changed_input_rejected_and_old_output_preserved(self):
        self.build_hierarchy()
        old_hierarchy = inputs.HIERARCHY_PATH.read_bytes()
        inputs.ATC_PATH.write_text(inputs.ATC_PATH.read_text() + "B,drug\n")
        with self.assertRaisesRegex(RuntimeError, "fingerprint changed"):
            self.build_hierarchy()
        self.assertEqual(inputs.HIERARCHY_PATH.read_bytes(), old_hierarchy)


if __name__ == "__main__":
    unittest.main()
