"""Count the public MSI v1.0.0 source files without changing their contents."""

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/msi_raw"
FILES = [
    "1_drug_to_protein.tsv",
    "2_indication_to_protein.tsv",
    "3_protein_to_protein.tsv",
    "4_protein_to_biological_function.tsv",
    "5_biological_function_to_biological_function.tsv",
]


def rows(name):
    with (DATA / name).open(newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def main():
    nodes = {}
    edges = {}
    for name in FILES:
        count = 0
        for row in rows(name):
            count += 1
            for suffix in ("1", "2"):
                node = row[f"node_{suffix}"]
                kind = row[f"node_{suffix}_type"]
                if node in nodes and nodes[node] != kind:
                    raise ValueError(f"ID collision: {node}: {nodes[node]} / {kind}")
                nodes[node] = kind
        edges[name] = count
    pairs = [(r["drug"], r["indication"]) for r in rows("6_drug_indication_df.tsv")]
    assert len(pairs) == len(set(pairs)), "Repeated treatment pair"
    drugs = {n for n, t in nodes.items() if t == "drug"}
    diseases = {n for n, t in nodes.items() if t == "indication"}
    missing_drugs = {d for d, _ in pairs} - drugs
    missing_diseases = {i for _, i in pairs} - diseases
    classes = list(rows("7_drug_classification_df.tsv"))
    result = {
        "source_node_total": len(nodes),
        "source_nodes_by_type": dict(Counter(nodes.values())),
        "source_edge_total_excluding_treatments": sum(edges.values()),
        "source_edges_by_file": edges,
        "treatment_rows": len(pairs),
        "treatment_unique": len(set(pairs)),
        "missing_treatment_drugs": len(missing_drugs),
        "missing_treatment_diseases": len(missing_diseases),
        "atc_rows": len(classes),
        "atc_drugs": len({r["db_id"] for r in classes}),
        "atc_graph_drugs": len({r["db_id"] for r in classes} & drugs),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    (ROOT / "results/msi_source_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
