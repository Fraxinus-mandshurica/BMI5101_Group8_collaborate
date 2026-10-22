"""Prepare DREAMwalk inputs from the official MSI files, keeping labels separate."""

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/msi_raw"
OUT = ROOT / "data/processed"
EDGES = [
    "1_drug_to_protein.tsv",
    "2_indication_to_protein.tsv",
    "3_protein_to_protein.tsv",
    "4_protein_to_biological_function.tsv",
    "5_biological_function_to_biological_function.tsv",
]
TYPES = {"drug": "drug", "indication": "disease", "protein": "gene", "biological_function": "etc"}


def read_tsv(path):
    with path.open(newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def prepare_full():
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = {}
    with (OUT / "msi_graph.tsv").open("w") as graph:
        edge_id = 0
        for relation, filename in enumerate(EDGES, 1):
            for row in read_tsv(SOURCE / filename):
                a, b = row["node_1"], row["node_2"]
                for node, kind in ((a, row["node_1_type"]), (b, row["node_2_type"])):
                    if node in nodes and nodes[node] != TYPES[kind]:
                        raise ValueError(f"Conflicting type for {node}")
                    nodes[node] = TYPES[kind]
                graph.write(f"{a}\t{b}\t{relation}\t1.0\t{edge_id}\n")
                edge_id += 1
    with (OUT / "msi_nodetypes.tsv").open("w") as f:
        f.write("node\ttype\n")
        for node, kind in sorted(nodes.items()):
            f.write(f"{node}\t{kind}\n")
    positives = set()
    for row in read_tsv(SOURCE / "6_drug_indication_df.tsv"):
        pair = (row["drug"], row["indication"])
        assert pair not in positives
        assert nodes.get(pair[0]) == "drug" and nodes.get(pair[1]) == "disease"
        positives.add(pair)
    with (OUT / "msi_positive_pairs.tsv").open("w") as f:
        f.write("drug\tdisease\tlabel\n")
        for a, b in sorted(positives):
            f.write(f"{a}\t{b}\t1\n")
    # The five graph relations have no drug-disease relation at all.
    assert edge_id == 478728 and len(nodes) == 29959 and len(positives) == 5926
    return nodes, positives


def atc_hierarchy(nodes):
    """Generate only public MSI drug→ATC ancestry; no fabricated MeSH links."""
    links = set()
    covered = set()
    for row in read_tsv(SOURCE / "7_drug_classification_df.tsv"):
        drug = row["db_id"]
        if drug not in nodes:
            continue
        code = row["atc_code"]
        if not code:
            continue
        covered.add(drug)
        links.add((drug, code))
        chain = [code, row["level_4"], row["level_3"], row["level_2"], row["level_1"], "drug"]
        links.update((a, b) for a, b in zip(chain, chain[1:]) if a and b and a != b)
    with (OUT / "msi_atc_hierarchy.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["child", "parent"])
        writer.writerows(sorted(links))
    return covered


def main():
    nodes, positives = prepare_full()
    covered = atc_hierarchy(nodes)
    print(f"Prepared {len(nodes)} nodes, {len(positives)} separate labels, {len(covered)} ATC-covered graph drugs")


if __name__ == "__main__":
    main()
