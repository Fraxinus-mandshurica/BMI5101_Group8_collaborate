"""Small KEGG native-ID, record-field and frozen-source fixtures; no model fits."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import kegg_inputs as inputs


CARDS = """ENTRY       H00001                      Disease
NAME        disease
PATHWAY     hsa00010  metabolism
            hsa00020  second metabolism
DIS_PATHWAY hsa05211  cancer
DRUG        Drug one [DR:D00001]
///
ENTRY       hsa00010                    Pathway
DRUG        D00001  First drug
            D00002  Second drug
DISEASE     H00001  First disease
REL_PATHWAY hsa00020  Another pathway
REFERENCE   PMID:123
  TITLE     Ignore unrelated identifiers D99999 H99999 map99999
///
ENTRY       map00010                    Pathway
REL_PATHWAY map00020  Same explicit reference relation
///
"""


class KeggRecordTest(unittest.TestCase):
    def test_reference_citation_legacy_encoding_does_not_break_ascii_entry_ids(self):
        data = b"ENTRY       H01475 Disease\nREFERENCE   Fran\xe7ois\n///\n"
        text, encoding = inputs._decode_text(data)
        self.assertEqual(encoding, "iso-8859-1")
        self.assertIn("H01475", inputs._flat_records(text))
        self.assertEqual(text.encode(encoding), data)
        self.assertEqual(inputs._decode_text("UTF-8 \u00e7".encode())[1], "utf-8")

    def test_field_continuations_and_human_reference_normalization(self):
        records = inputs._flat_records(CARDS)
        self.assertEqual(set(records), {"H00001", "hsa00010", "map00010"})
        links = list(inputs._card_links(records))
        self.assertIn(("drug_pathway", "dr:D00002", "path:map00010", "hsa00010:DRUG"), links)
        self.assertIn(("disease_pathway", "ds:H00001", "path:map05211", "H00001:DIS_PATHWAY"), links)
        self.assertEqual(sum(row[0] == "disease_pathway" for row in links), 4)
        self.assertEqual(sum(row[0] == "pathway_pathway" for row in links), 2)
        self.assertFalse(any("99999" in value for row in links for value in row))
        self.assertFalse(any(row[0] == "drug_disease" for row in links))

    def test_records_require_explicit_terminators(self):
        with self.assertRaisesRegex(ValueError, "terminator"):
            inputs._flat_records("ENTRY       H00001 Disease\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            inputs._flat_records("ENTRY       H00001 Disease\n///\nENTRY       H00001 Disease\n///\n")

    def test_list_ids_and_link_orientation_are_not_guessed(self):
        self.assertEqual(inputs._list_rows("D00001\tName\n", "drug")[0]["id"], "dr:D00001")
        self.assertEqual(inputs._list_rows("H00001\tName\n", "disease")[0]["id"], "ds:H00001")
        self.assertEqual(inputs._list_rows("hsa:123\tCDS\t1\tGene name\n", "gene")[0]["id"], "hsa:123")
        self.assertEqual(inputs._list_rows("hsa00010\tName\nmap00010\tName\n", "etc")[0]["id"], "path:map00010")
        self.assertEqual(list(inputs._bulk_links("hsa:1\tpath:hsa00010\n", "gene", "etc")), [("hsa:1", "path:map00010")])
        with self.assertRaisesRegex(ValueError, "endpoint types"):
            list(inputs._bulk_links("dr:D00001\tds:H00001\n", "disease", "drug"))
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            inputs._list_rows("ko00010\tNonhuman\n", "etc")

    def test_batches_never_exceed_ten_and_names_depend_on_actual_ids(self):
        identifiers = ["H%05d" % number for number in range(1, 24)]
        jobs = inputs._batch_jobs(identifiers, "disease")
        self.assertEqual([len(endpoint[4:].split("+")) for _, endpoint in jobs], [10, 10, 3])
        self.assertEqual(jobs, inputs._batch_jobs(reversed(identifiers), "disease"))
        self.assertNotEqual(jobs[0][0], inputs._batch_jobs(identifiers[1:], "disease")[0][0])


class KeggMolecularTest(unittest.TestCase):
    def fixture(self):
        types = {"dr:D00001": "drug", "ds:H00001": "disease", "hsa:1": "gene",
                 "path:map00010": "etc", "path:map00020": "etc", "dr:D00002": "drug"}
        nodes = {node: dict(id=node, type=kind) for node, kind in types.items()}
        rows = [("disease_gene", "ds:H00001", "hsa:1", "bulk"),
                ("drug_gene", "dr:D00001", "hsa:1", "bulk"),
                ("gene_pathway", "hsa:1", "path:map00010", "bulk"),
                ("pathway_pathway", "path:map00010", "path:map00020", "hsa:REL_PATHWAY"),
                ("pathway_pathway", "path:map00010", "path:map00020", "map:REL_PATHWAY"),
                ("pathway_pathway", "path:map00020", "path:map00010", "reciprocal:REL_PATHWAY"),
                ("disease_pathway", "ds:H00001", "path:map00010", "PATHWAY"),
                ("drug_pathway", "dr:D00001", "path:map00010", "DRUG")]
        return nodes, rows

    def test_exact_relation_deduplication_keeps_reciprocal_explicit_links(self):
        nodes, rows = self.fixture()
        graph, active, counts, origins, absent = inputs._select_molecular(nodes, rows)
        self.assertEqual([row[2] for row in graph], [1, 2, 3, 4, 4, 5, 6])
        self.assertEqual([row[4] for row in graph], list(range(7)))
        self.assertEqual(counts["pathway_pathway"], 2)
        self.assertEqual(len(origins), 8)
        self.assertNotIn("dr:D00002", active)
        self.assertFalse(absent)
        self.assertFalse(any(row[:2] == ("dr:D00001", "ds:H00001") for row in graph))

    def test_missing_endpoints_are_audited_without_inventing_nodes_or_edges(self):
        nodes, rows = self.fixture()
        rows.append(("drug_gene", "dr:D99999", "hsa:1", "orphan source"))
        graph, active, counts, _, absent = inputs._select_molecular(nodes, rows)
        self.assertNotIn("dr:D99999", active)
        self.assertEqual(absent[0][-1], "dr:D99999")
        self.assertEqual(len(graph), 7)

    def test_self_loop_halts_instead_of_silently_deleting_or_adding_one(self):
        nodes, rows = self.fixture()
        with self.assertRaisesRegex(ValueError, "self-loops"):
            inputs._select_molecular(nodes, rows + [("pathway_pathway", "path:map00010", "path:map00010", "REL_PATHWAY")])


class KeggCacheTest(unittest.TestCase):
    def create(self, directory):
        cache = inputs.FrozenRestCache(directory)
        name, data = "fixture.txt", b"fixed source\n"
        path = Path(directory) / name
        path.write_bytes(data)
        cache.index["sources"][name] = dict(url=inputs.REST + "list/drug", bytes=len(data),
                                             sha256=hashlib.sha256(data).hexdigest(), acquired_utc="fixed")
        inputs._write_json(cache.index_path, cache.index)
        return cache, name

    def test_modified_cached_source_fails_without_network_or_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, name = self.create(directory)
            (Path(directory) / name).write_bytes(b"new content")
            with patch.object(inputs, "urlopen") as network:
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    cache.fetch(name, "list/drug")
                network.assert_not_called()

    def test_completed_snapshot_does_not_expand_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            cache, name = self.create(directory)
            cache.index["completed_utc"] = "fixed"
            with patch.object(inputs, "urlopen") as network:
                self.assertEqual(cache.fetch(name, "list/drug"), Path(directory) / name)
                with self.assertRaisesRegex(ValueError, "lacks required source"):
                    cache.fetch("missing.txt", "list/disease")
                network.assert_not_called()

    def test_identical_writes_do_not_change_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            inputs._write_json(path, {"fixed": 1})
            before = path.stat().st_mtime_ns
            inputs._write_json(path, {"fixed": 1})
            self.assertEqual(before, path.stat().st_mtime_ns)
            self.assertEqual(json.loads(path.read_text()), {"fixed": 1})


if __name__ == "__main__":
    unittest.main()
