"""Small native-ID, molecular-filtering and real-DAG fixtures; no model fits."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import hetionet_inputs as inputs


OBO = """format-version: 1.2
[Term]
id: DOID:4
name: disease
[Term]
id: DOID:10
name: first parent
is_a: DOID:4 ! disease
[Term]
id: DOID:11
name: second parent
is_a: DOID:4 ! disease
[Term]
id: DOID:1
name: two-parent disease
alt_id: DOID:99
is_a: DOID:10 ! first parent
is_a: DOID:11 ! second parent
[Term]
id: DOID:2
name: one-parent disease
is_a: DOID:10 ! first parent
[Typedef]
id: unrelated_relation
"""


def node(identifier, kind):
    return dict(id=identifier, kind=kind, name=identifier)


class MolecularFilteringTest(unittest.TestCase):
    def test_node_types_produce_four_distinct_upstream_training_prefixes(self):
        sys.path.insert(0, str(ROOT / "vendor/DREAMwalk_upstream"))
        from DREAMwalk.HeterogeneousSG import _prep_hetSG_walks
        leaves = ["Compound::DB1", "Disease::DOID:1", "Gene::1", "Pathway::PW1"]
        with tempfile.TemporaryDirectory() as directory:
            nodetypes = Path(directory) / "node_types.tsv"
            inputs._write_rows(nodetypes, zip(leaves, [value for _, value in inputs.NODE_TYPES]),
                               header=("node", "type"), delimiter="\t")
            annotated = _prep_hetSG_walks([leaves], {node: str(i) for i, node in enumerate(leaves)}, str(nodetypes))
            self.assertEqual(annotated, ["d0 i1 g2 e3"])

    def test_retains_row_order_parallel_relations_and_native_ids_without_dda(self):
        nodes = {row["id"]: row for row in (
            node("Compound::DB1", "Compound"), node("Gene::1", "Gene"),
            node("Disease::DOID:1", "Disease"), node("Pathway::PW1", "Pathway"),
            node("Compound::isolated", "Compound"), node("Anatomy::A1", "Anatomy"))}
        edges = [dict(source=s, metaedge=r, target=t) for s, r, t in (
            ("Compound::DB1", "CuG", "Gene::1"),
            ("Compound::DB1", "CtD", "Disease::DOID:1"),
            ("Compound::DB1", "CbG", "Gene::1"),
            ("Disease::DOID:1", "DaG", "Gene::1"),
            ("Gene::1", "GpPW", "Pathway::PW1"),
            ("Compound::isolated", "CtD", "Disease::DOID:1"),
            ("Anatomy::A1", "AeG", "Gene::1"))]
        retained, positives, active, counts, unique_pairs = inputs._select_molecular(nodes, edges)
        self.assertEqual([row[2] for row in retained], [2, 1, 4, 10])
        self.assertEqual([row[4] for row in retained], [0, 1, 2, 3])
        self.assertEqual(unique_pairs, 3)
        self.assertEqual(len(positives), 2)
        self.assertNotIn("Compound::isolated", active)
        self.assertNotIn("Anatomy::A1", active)
        self.assertEqual(retained[0][:2], ("Compound::DB1", "Gene::1"))
        self.assertFalse(any({row[0], row[1]} == set(positives[0]) for row in retained))

    def test_self_loop_is_reported_instead_of_creating_or_silently_removing_it(self):
        nodes = {"Gene::1": node("Gene::1", "Gene")}
        with self.assertRaisesRegex(ValueError, "self-loops"):
            inputs._select_molecular(nodes, [dict(source="Gene::1", target="Gene::1", metaedge="GiG")])


class NativeHierarchyTest(unittest.TestCase):
    def fixture(self):
        nodes = {row["id"]: row for row in (
            node("Compound::DB1", "Compound"), node("Compound::DB2", "Compound"),
            node("Disease::DOID:99", "Disease"), node("Disease::DOID:2", "Disease"))}
        drugs = [dict(drugbank_id="DB1", atc_codes="A01AA01|B02BB02"),
                 dict(drugbank_id="DB2", atc_codes="")]
        return nodes, drugs

    def test_exact_alt_id_and_atc_prefixes_preserve_all_real_paths(self):
        nodes, drugs = self.fixture()
        edges, paths, diseases, atc, coverage = inputs._semantic_hierarchy(nodes, set(nodes), drugs, OBO)
        self.assertEqual(len(paths["Disease::DOID:99"]), 2)
        self.assertEqual(len(paths["Disease::DOID:2"]), 1)
        self.assertEqual(len(paths["Compound::DB1"]), 2)
        self.assertNotIn("Compound::DB2", paths)
        self.assertIn(("Disease::DOID:99", "DO::DOID:1"), edges)
        self.assertIn(("DO::DOID:4", "disease"), edges)
        self.assertIn(("ATC::A01AA01", "ATC::A01AA"), edges)
        self.assertEqual(next(row for row in diseases if row[0] == "Disease::DOID:99")[1:4],
                         ("DOID:99", "DOID:1", "alt_id"))
        self.assertEqual(coverage["active_drug_mapped"], 1)
        self.assertEqual(coverage["active_do_paths"], 3)
        self.assertEqual(len(atc), 2)

    def test_upstream_merge_matches_valid_dag_paths_exactly(self):
        import pandas as pd
        sys.path.insert(0, str(ROOT / "vendor/DREAMwalk_upstream"))
        from DREAMwalk.generate_similarity_net import _generate_tree
        nodes, drugs = self.fixture()
        edges, expected, _, _, _ = inputs._semantic_hierarchy(nodes, set(nodes), drugs, OBO)
        actual = _generate_tree(pd.DataFrame(edges, columns=["child", "parent"]), list(nodes))
        flattened = {leaf: {tuple(path) for path in paths}
                     for leaves in actual.values() for leaf, paths in leaves.items()}
        self.assertEqual(flattened, expected)

    def test_isolates_have_mapping_audit_but_no_hierarchy_membership(self):
        nodes, drugs = self.fixture()
        active = {"Compound::DB1", "Disease::DOID:2"}
        edges, paths, audit, _, coverage = inputs._semantic_hierarchy(nodes, active, drugs, OBO)
        self.assertNotIn("Disease::DOID:99", paths)
        self.assertFalse(any(row[0] == "Disease::DOID:99" for row in edges))
        self.assertEqual(next(row for row in audit if row[0] == "Disease::DOID:99")[-1], False)
        self.assertEqual(coverage["disease_mapped"], 2)
        self.assertEqual(coverage["active_disease_mapped"], 1)

    def test_cycles_missing_terms_and_external_roots_stop_preparation(self):
        with self.assertRaisesRegex(ValueError, "Cycle"):
            inputs._disease_paths("DOID:1", {"DOID:1": {"DOID:2"}, "DOID:2": {"DOID:1"}})
        with self.assertRaisesRegex(ValueError, "Missing"):
            inputs._disease_paths("DOID:1", {})
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            inputs._disease_paths("DOID:9", {"DOID:9": set()})


class SourceAndAtomicTest(unittest.TestCase):
    def test_source_hash_mismatch_is_not_silently_redownloaded(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = inputs.paths(directory)["raw_dir"]
            raw.mkdir(parents=True)
            (raw / inputs.SOURCES[0].name).write_text("wrong source")
            with patch.object(inputs, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "Pinned source"):
                    inputs.download_sources(directory)
                network.assert_not_called()

    def test_identical_atomic_writes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            inputs._write_rows(path, [("a", "b")], header=("child", "parent"))
            before = path.stat().st_mtime_ns
            inputs._write_rows(path, [("a", "b")], header=("child", "parent"))
            self.assertEqual(path.read_bytes(), b"child,parent\na,b\n")
            self.assertEqual(path.stat().st_mtime_ns, before)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
