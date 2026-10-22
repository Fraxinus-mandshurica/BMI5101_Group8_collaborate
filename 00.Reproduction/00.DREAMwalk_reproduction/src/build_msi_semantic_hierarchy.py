"""Build path-faithful DREAMwalk hierarchy from accepted ATC and MeSH audits.

MeSH internal nodes are actual Descriptor UIs qualified by their NLM Tree
Number. This prevents DREAMwalk's child/parent merge from recombining branches
that belong to different tree positions of the same Descriptor. The disease
leaf still maps first to its actual, accepted D-prefixed Descriptor UI.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from map_msi_cui_to_mesh_umls import read_mesh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/processed/msi_semantic_hierarchy_final.csv"
SUMMARY = ROOT / "results/msi_final_hierarchy_summary.csv"
POSITION_AUDIT = ROOT / "results/msi_final_mesh_position_nodes.csv"
ATC = ROOT / "results/msi_atc_hierarchy_edges.csv"
MAPPING = ROOT / "results/msi_cui_mesh_umls_mapping.csv"
NODETYPES = ROOT / "data/processed/msi_nodetypes.tsv"


def read_csv(path, delimiter=","):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    mesh, mesh_stats = read_mesh()
    position_to_descriptor = {position: descriptor for descriptor, positions in mesh.items() for position in positions}
    assert len(position_to_descriptor) == mesh_stats["mesh2024_unique_tree_numbers"]
    mapping = read_csv(MAPPING)
    mapped = {r["msi_cui"]: r["mesh_descriptor_id"] for r in mapping if r["usable_for_primary_hierarchy"] == "True"}
    assert len(mapping) == 840 and len(mapped) == 748
    types = {r["node"]: r["type"] for r in read_csv(NODETYPES, "\t")}
    drugs = {n for n, t in types.items() if t == "drug"}
    diseases = {n for n, t in types.items() if t == "disease"}
    assert len(drugs) == 1661 and len(diseases) == 840 and set(mapped) <= diseases
    drug_edges = {(r["child"], r["parent"]) for r in read_csv(ATC)}
    assert len(drug_edges) == 7095
    disease_edges = set()
    expected = defaultdict(set)
    positions_used = {}
    for cui, descriptor in mapped.items():
        assert descriptor in mesh and mesh[descriptor]
        for tree in mesh[descriptor]:
            chain = [cui, descriptor]
            cursor = tree
            while True:
                parent_descriptor = position_to_descriptor.get(cursor)
                assert parent_descriptor, f"Missing MeSH descriptor at {cursor}"
                node = parent_descriptor + "@" + cursor
                positions_used[node] = (parent_descriptor, cursor)
                chain.append(node)
                if "." not in cursor:
                    break
                cursor = cursor.rsplit(".", 1)[0]
            chain.append("disease")
            disease_edges.update(zip(chain, chain[1:]))
            expected[cui].add(tuple(chain))
    assert len(expected) == 748
    # No artificial membership for the 92 unmapped diseases.
    assert not any(cui in {child for child, _ in disease_edges} for cui in diseases-set(mapped))
    combined = drug_edges | disease_edges
    assert len(combined) == len(drug_edges)+len(disease_edges)
    assert all(child != parent for child, parent in combined)
    assert not any(child in drugs and parent == "disease" for child, parent in combined)
    assert not any(child in diseases and parent == "drug" for child, parent in combined)
    # Use actual upstream path expansion, not only our own graph traversal.
    upstream = ROOT / "vendor/DREAMwalk_upstream"
    sys.path.insert(0, str(upstream))
    import pandas as pd
    from DREAMwalk.generate_similarity_net import _generate_tree
    hierarchy_df = pd.DataFrame(sorted(combined), columns=["child", "parent"])
    tree = _generate_tree(hierarchy_df, list(drugs | diseases))
    assert set(tree) == {"drug", "disease"}
    for cui, wanted in expected.items():
        actual = {tuple(path) for path in tree["disease"].get(cui, [])}
        assert actual == wanted, f"MeSH path mismatch for {cui}: expected {len(wanted)}, got {len(actual)}"
    assert not (set(tree["disease"]) & (diseases-set(mapped)))
    mapped_drugs = {node for node in tree["drug"] if node in drugs}
    assert len(mapped_drugs) == 1328
    assert not (set(tree["drug"]) & (drugs-mapped_drugs))
    assert sum(len(v) for v in tree["drug"].values()) == 2448
    assert sum(len(v) for v in tree["disease"].values()) == sum(len(v) for v in expected.values())
    write_csv(OUT, [dict(child=c, parent=p) for c, p in sorted(combined)], ["child", "parent"])
    write_csv(POSITION_AUDIT, [dict(hierarchy_node=n, mesh_descriptor_id=d, mesh_tree_number=t)
                               for n, (d, t) in sorted(positions_used.items())],
              ["hierarchy_node", "mesh_descriptor_id", "mesh_tree_number"])
    summary = dict(total_graph_drugs=len(drugs), hierarchy_covered_drugs=len(mapped_drugs),
                   uncovered_drugs=len(drugs-mapped_drugs), drug_atc_assignments=2448,
                   drug_hierarchy_paths=sum(len(v) for v in tree["drug"].values()),
                   total_graph_diseases=len(diseases), hierarchy_covered_diseases=len(mapped),
                   uncovered_diseases=len(diseases-set(mapped)), disease_mesh_mappings=len(mapped),
                   disease_hierarchy_paths=sum(len(v) for v in tree["disease"].values()),
                   drug_hierarchy_edges=len(drug_edges), disease_hierarchy_edges=len(disease_edges),
                   total_hierarchy_edges=len(combined), unique_mesh_position_nodes=len(positions_used),
                   tree_strategy="source_descriptor_UI_qualified_by_MeSH2024_tree_number",
                   mesh2024_sha256=mesh_stats["mesh2024_sha256"],
                   hierarchy_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest())
    write_csv(SUMMARY, [summary], list(summary))
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
