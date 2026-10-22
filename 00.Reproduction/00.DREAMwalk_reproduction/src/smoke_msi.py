"""Reduced MSI end-to-end feasibility test. Its metrics are not paper comparable."""

import csv
import os
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/DREAMwalk_upstream"
sys.path.insert(0, str(UPSTREAM))
from DREAMwalk.generate_similarity_net import save_sim_graph
from DREAMwalk.generate_embeddings import save_embedding_files
from DREAMwalk.utils import read_graph

SOURCE = ROOT / "data/msi_raw"
OUT = ROOT / "logs/smoke"
OUT.mkdir(parents=True, exist_ok=True)


def read_tsv(name):
    with (SOURCE / name).open(newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def write_inputs():
    random.seed(0)
    atc = {r["db_id"] for r in read_tsv("7_drug_classification_df.tsv")}
    all_positive = {(r["drug"], r["indication"]) for r in read_tsv("6_drug_indication_df.tsv")}
    selected_positive = random.sample(sorted(p for p in all_positive if p[0] in atc), 80)
    drugs = {a for a, _ in selected_positive}
    diseases = {b for _, b in selected_positive}
    chosen = []
    proteins = set()
    for relation, name in ((1, "1_drug_to_protein.tsv"), (2, "2_indication_to_protein.tsv")):
        for row in read_tsv(name):
            if row["node_1"] in (drugs if relation == 1 else diseases):
                chosen.append((row["node_1"], row["node_2"], relation))
                proteins.add(row["node_2"])
    # Bound the smoke graph while retaining the selected drug/disease neighborhoods.
    for row in read_tsv("3_protein_to_protein.tsv"):
        if row["node_1"] in proteins and row["node_2"] in proteins:
            chosen.append((row["node_1"], row["node_2"], 3))
    functions = set()
    for row in read_tsv("4_protein_to_biological_function.tsv"):
        if row["node_1"] in proteins:
            chosen.append((row["node_1"], row["node_2"], 4))
            functions.add(row["node_2"])
    for row in read_tsv("5_biological_function_to_biological_function.tsv"):
        if row["node_1"] in functions and row["node_2"] in functions:
            chosen.append((row["node_1"], row["node_2"], 5))
    node_types = {d: "drug" for d in drugs}
    node_types.update({d: "disease" for d in diseases})
    node_types.update({p: "gene" for p in proteins})
    node_types.update({f: "etc" for f in functions})
    with (OUT / "graph.tsv").open("w") as f:
        for i, (a, b, relation) in enumerate(chosen):
            f.write(f"{a}\t{b}\t{relation}\t1.0\t{i}\n")
    with (OUT / "nodetypes.tsv").open("w") as f:
        f.write("node\ttype\n")
        for node, kind in sorted(node_types.items()):
            f.write(f"{node}\t{kind}\n")
    links = set()
    for row in read_tsv("7_drug_classification_df.tsv"):
        if row["db_id"] in drugs:
            chain = [row["db_id"], row["atc_code"], row["level_4"], row["level_3"], row["level_2"], row["level_1"], "drug"]
            links.update((a, b) for a, b in zip(chain, chain[1:]) if a and b and a != b)
    with (OUT / "atc.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("child", "parent"))
        w.writerows(sorted(links))
    assert all(node_types[a] == "drug" and node_types[b] == "disease" for a, b in selected_positive)
    assert not any(node_types[a] == "drug" and node_types[b] == "disease" for a, b, _ in chosen)
    return selected_positive, all_positive, drugs, diseases, node_types, chosen


def main():
    start = time.monotonic()
    positives, known, drugs, diseases, node_types, edges = write_inputs()
    print(f"SMOKE TEST — NOT PAPER-COMPARABLE: {len(node_types)} nodes, {len(edges)} edges, {len(positives)} positives", flush=True)
    graph = read_graph(str(OUT / "graph.tsv"))
    assert set(graph.nodes()) == set(node_types)
    save_sim_graph(str(OUT / "graph.tsv"), str(OUT / "atc.csv"), str(OUT / "sim.tsv"), cutoff=0.4)
    sim = read_graph(str(OUT / "sim.tsv"))
    assert set(sim.nodes()) <= drugs and sim.number_of_edges() > 0
    save_embedding_files(str(OUT / "graph.tsv"), str(OUT / "sim.tsv"), str(OUT / "embedding.pkl"),
                         str(OUT / "nodetypes.tsv"), tp_factor=0.3, seed=0,
                         num_walks=1, walk_length=4, dimension=16, window_size=4,
                         workers=4, em_max_iter=1, p=1, q=1)
    import pickle
    with (OUT / "embedding.pkl").open("rb") as f:
        embeddings = pickle.load(f)
    assert all(a in embeddings and b in embeddings for a, b in positives)
    candidates = [(a, b) for a in sorted(drugs) for b in sorted(diseases) if (a, b) not in known and a in embeddings and b in embeddings]
    rng = random.Random(0)
    negatives = rng.sample(candidates, len(positives))
    assert not set(negatives) & known and len(set(negatives)) == len(positives)
    pairs = positives + negatives
    X = np.asarray([embeddings[a] - embeddings[b] for a, b in pairs])
    y = np.asarray([1] * len(positives) + [0] * len(negatives))
    train, test = train_test_split(np.arange(len(y)), test_size=0.25, random_state=0, stratify=y)
    assert set(y[train]) == set(y[test]) == {0, 1}
    clf = XGBClassifier(base_score=0.5, booster="gbtree", eval_metric="error", objective="binary:logistic",
                        gamma=0, learning_rate=0.1, max_depth=6, n_estimators=500,
                        tree_method="auto", min_child_weight=4, subsample=0.8, colsample_bytree=0.9,
                        scale_pos_weight=1, max_delta_step=1, seed=0, n_jobs=4)
    clf.fit(X[train], y[train])
    pred = clf.predict_proba(X[test])[:, 1]
    metrics = {"accuracy": accuracy_score(y[test], pred.round()),
               "auroc": roc_auc_score(y[test], pred), "aupr": average_precision_score(y[test], pred)}
    assert all(np.isfinite(v) for v in metrics.values())
    print(f"sim_edges={sim.number_of_edges()} embedding_nodes={len(embeddings)} metrics={metrics}", flush=True)
    elapsed = time.monotonic() - start
    parent_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"elapsed_seconds={elapsed:.1f} peak_rss_bytes={parent_rss}", flush=True)
    result_file = ROOT / "results/smoke_test_runs.csv"
    with result_file.open(newline="") as f:
        run_number = sum(1 for _ in csv.DictReader(f)) + 1
    with result_file.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([run_number, 0, len(node_types), len(edges), sim.number_of_edges(),
                         len(positives), len(negatives), 4, 1, 4, 16, 4, 1, 0.4, 0.3,
                         len(train), len(test), metrics["accuracy"], metrics["auroc"],
                         metrics["aupr"], round(elapsed, 1), parent_rss, "", ""])


if __name__ == "__main__":
    main()
