"""Small BRITE parsing and exact hierarchy-path fixtures; no model training."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import kegg_semantics as semantic


ATC = """+F\tDrug
AA ALIMENTARY TRACT
B  A01 STOMATOLOGICAL PREPARATIONS
C    A01A STOMATOLOGICAL PREPARATIONS
D      A01AA Caries prophylactic agents
E        A01AA01 Sodium fluoride
F          D00001 Sodium fluoride
F          C00001 Non-drug compound
AB BLOOD
B  B01 ANTITHROMBOTIC AGENTS
C    B01A ANTITHROMBOTIC AGENTS
D      B01AA Vitamin K antagonists
E        B01AA01 Dicoumarol
F          D00001 Same native drug, different ATC assignment
F          D00002 Another drug
#Last updated: fixture
"""

ICD = """+H\tDisease
A01 First chapter
B  First grouping
C    1A00 First disease term
D      H00001 Disease one
C    1A01 Another term
D      H00002 Disease two
A02 Second chapter
B  Second grouping
C    1A00 Same code in a different tree position
D      H00001 Disease one
D      H00003 Disease three
"""


class KeggSemanticTest(unittest.TestCase):
    def build(self):
        return semantic.build_hierarchy(ATC, ICD, {
            "dr:D00001": "drug", "dr:D99999": "drug", "ds:H00001": "disease",
            "ds:H00002": "disease", "ds:H99999": "disease", "hsa:1": "gene"})

    def test_native_leaves_atc_prefixes_and_unmapped_nodes(self):
        result = self.build()
        self.assertIn(("dr:D00001", "ATC::A01AA01"), result.edges)
        self.assertIn(("dr:D00001", "ATC::B01AA01"), result.edges)
        self.assertIn(("ATC::A01AA01", "ATC::A01AA"), result.edges)
        self.assertIn(("ATC::A", "drug"), result.edges)
        self.assertEqual(len(result.expected_paths["dr:D00001"]), 2)
        self.assertNotIn("dr:D00002", result.expected_paths)
        self.assertFalse(any("C00001" in value for edge in result.edges for value in edge))
        self.assertFalse(any("99999" in value for edge in result.edges for value in edge))
        self.assertEqual(result.stats["active_drug_unmapped"], 1)
        self.assertEqual(result.stats["active_disease_unmapped"], 1)
        self.assertEqual(sum(row["mapping_status"] == "unmapped" for row in result.audit), 2)

    def test_repeated_icd_code_keeps_exact_positions(self):
        result = self.build()
        paths = result.expected_paths["ds:H00001"]
        self.assertEqual(len(paths), 2)
        code_rows = [row for row in result.audit if row["node_type"] == "ontology" and row["source_code"] == "1A00"]
        self.assertEqual(len(code_rows), 2)
        self.assertNotEqual(code_rows[0]["node"], code_rows[1]["node"])
        self.assertTrue(all(path[0] == "ds:H00001" and path[-1] == "disease" for path in paths))

    def test_upstream_merger_has_no_fabricated_cross_position_paths(self):
        import pandas as pd
        sys.path.insert(0, str(ROOT / "vendor/DREAMwalk_upstream"))
        from DREAMwalk.generate_similarity_net import _generate_tree
        result = self.build()
        leaves = list(result.expected_paths)
        actual = _generate_tree(pd.DataFrame(result.edges, columns=["child", "parent"]), leaves)
        paths = {leaf: {tuple(path) for path in values}
                 for group in actual.values() for leaf, values in group.items()}
        self.assertEqual(paths, result.expected_paths)

    def test_parser_rejects_missing_parent_level(self):
        with self.assertRaisesRegex(ValueError, "skips a parent"):
            semantic.build_hierarchy(ATC, "A01 Chapter\nC H00001 Broken hierarchy", {})

    def test_parser_rejects_invalid_atc_parent(self):
        broken = ATC.replace("A01AA01 Sodium fluoride", "A02AA01 Sodium fluoride")
        with self.assertRaisesRegex(ValueError, "official prefix"):
            semantic.build_hierarchy(broken, ICD, {})

    def test_native_graph_types_are_checked(self):
        with self.assertRaisesRegex(ValueError, "native KEGG disease ID"):
            semantic.build_hierarchy(ATC, ICD, {"H00001": "disease"})

    def test_deterministic_independent_of_active_input_order(self):
        first = self.build()
        other = semantic.build_hierarchy(ATC, ICD, dict(reversed(list({
            "dr:D00001": "drug", "dr:D99999": "drug", "ds:H00001": "disease",
            "ds:H00002": "disease", "ds:H99999": "disease", "hsa:1": "gene"}.items()))))
        self.assertEqual(first, other)


if __name__ == "__main__":
    unittest.main()
