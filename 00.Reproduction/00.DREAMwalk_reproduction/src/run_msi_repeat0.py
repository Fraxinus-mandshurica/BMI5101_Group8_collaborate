"""First full MSI DREAMwalk run, seed/repeat zero only.

Run phases separately: prepare, similarity, embedding, evaluate. No other seed
is accepted. Upstream similarity, walk, transition and Skip-gram code is used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/DREAMwalk_upstream"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "src"))
GRAPH = ROOT / "data/processed/msi_graph.tsv"
NODETYPES = ROOT / "data/processed/msi_nodetypes.tsv"
POSITIVES = ROOT / "data/processed/msi_positive_pairs.tsv"
HIERARCHY = ROOT / "data/processed/msi_semantic_hierarchy_final.csv"
SIMILARITY = ROOT / "data/processed/msi_similarity_graph_final.txt"
EMBEDDING = ROOT / "data/embeddings/msi_seed_0.pkl"
NEGATIVES = ROOT / "results/msi_repeat0_negative_pairs.csv"
SIM_SUMMARY = ROOT / "results/msi_semantic_similarity_summary.csv"
CONFIG = ROOT / "results/msi_repeat0_run_config.json"
TIMING = ROOT / "results/msi_repeat0_timing.json"
FOLDS = ROOT / "results/msi_fig2a_fold_results.csv"
SUMMARY = ROOT / "results/msi_fig2a_summary.csv"
SEED = 0
N_FOLDS = 5
WORKERS = 4
CUTOFF = 0.4
TP_FACTOR = 0.3
NUM_WALKS = 100
WALK_LENGTH = 10
DIMENSION = 128
WINDOW = 4
EM_MAX_ITER = 5


def read_rows(path, delimiter="\t"):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def swap_snapshot():
    try:
        swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True, timeout=10).strip()
        vm = subprocess.check_output(["vm_stat"], text=True, timeout=10)
        pageouts = next((line.strip() for line in vm.splitlines() if line.startswith("Pageouts:")), "")
        return {"swapusage": swap, "pageouts": pageouts}
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"swapusage": "unavailable", "pageouts": "unavailable"}


def update_timing(**items):
    data = json.loads(TIMING.read_text()) if TIMING.exists() else {}
    data.update(items)
    TIMING.write_text(json.dumps(data, indent=2) + "\n")


def base_inputs():
    types = {r["node"]: r["type"] for r in read_rows(NODETYPES)}
    drugs = sorted(n for n, t in types.items() if t == "drug")
    diseases = sorted(n for n, t in types.items() if t == "disease")
    positives = {(r["drug"], r["disease"]) for r in read_rows(POSITIVES)}
    assert len(types) == 29959 and len(drugs) == 1661 and len(diseases) == 840
    assert len(positives) == 5926
    assert all(types[a] == "drug" and types[b] == "disease" for a, b in positives)
    return types, drugs, diseases, positives


def check_graph_leakage(types, positives):
    observed = set()
    relations = Counter()
    with GRAPH.open(encoding="utf-8") as stream:
        for line in stream:
            a, b, rel, weight, edge_id = line.rstrip("\n").split("\t")
            observed.update((a, b))
            relations[int(rel)] += 1
            assert not ({types[a], types[b]} == {"drug", "disease"}), f"DDA graph edge {a},{b}"
            assert (a, b) not in positives and (b, a) not in positives
    assert len(observed) == 29959 and sum(relations.values()) == 478728
    assert dict(relations) == {1: 8568, 2: 25212, 3: 387626, 4: 34777, 5: 22545}
    return dict(nodes=len(observed), edges=sum(relations.values()), relations=dict(relations),
                known_treatment_edges_in_graph=0)


def sample_negatives(drugs, diseases, positives):
    candidates = [(a, b) for a in drugs for b in diseases if (a, b) not in positives]
    assert len(candidates) == 1661 * 840 - 5926
    selected = random.Random(SEED).sample(candidates, len(positives))
    assert len(selected) == len(set(selected)) == 5926 and not set(selected) & positives
    rows = [dict(drug=a, disease=b) for a, b in selected]
    if NEGATIVES.exists():
        old = read_rows(NEGATIVES, ",")
        assert old == rows, "Saved repeat-0 negatives differ from deterministic seed-0 sample"
    else:
        write_csv(NEGATIVES, rows, ["drug", "disease"])
    return selected


def phase_prepare():
    started = time.monotonic()
    types, drugs, diseases, positives = base_inputs()
    graph = check_graph_leakage(types, positives)
    negatives = sample_negatives(drugs, diseases, positives)
    update_timing(preparation_seconds=time.monotonic() - started)
    print(json.dumps({"graph": graph, "positives": len(positives), "negatives": len(negatives),
                      "negative_sha256": sha256(NEGATIVES)}, indent=2), flush=True)


def analyze_similarity():
    types, drugs, diseases, positives = base_inputs()
    drug_mapped = {r["drug_id"] for r in read_rows(ROOT / "results/msi_atc_mapping_audit.csv", ",") if r["in_atc_source"] == "True"}
    disease_mapped = {r["msi_cui"] for r in read_rows(ROOT / "results/msi_cui_mesh_umls_mapping.csv", ",") if r["usable_for_primary_hierarchy"] == "True"}
    assert len(drug_mapped) == 1328 and len(disease_mapped) == 748
    weights = defaultdict(list)
    neighbors = defaultdict(lambda: defaultdict(set))
    pair_seen = set()
    with SIMILARITY.open(encoding="utf-8") as stream:
        for line in stream:
            a, b, edge_type, weight, edge_id = line.rstrip("\n").split("\t")
            val = float(weight)
            assert math.isfinite(val) and CUTOFF < val <= 1 and a != b
            assert a in types and b in types and types[a] == types[b] and types[a] in ("drug", "disease")
            kind = types[a]
            assert a in (drug_mapped if kind == "drug" else disease_mapped)
            assert b in (drug_mapped if kind == "drug" else disease_mapped)
            key = (kind, min(a, b), max(a, b))
            assert key not in pair_seen, f"Duplicate semantic pair {key}"
            pair_seen.add(key)
            weights[kind].append(val)
            neighbors[kind][a].add(b)
            neighbors[kind][b].add(a)
    result = []
    for kind, mapped in (("drug", drug_mapped), ("disease", disease_mapped)):
        values = weights[kind]
        assert values, f"No retained {kind} similarities"
        result.append(dict(semantic_type=kind, hierarchy_covered_graph_nodes=len(mapped),
                           possible_evaluated_pairs=len(mapped)*(len(mapped)-1)//2,
                           retained_undirected_pairs=len(values),
                           similarity_min=min(values), similarity_median=statistics.median(values),
                           similarity_mean=statistics.mean(values), similarity_max=max(values),
                           graph_nodes_with_semantic_neighbor=len(neighbors[kind]),
                           mapped_nodes_with_zero_retained_neighbors=len(mapped-set(neighbors[kind])),
                           unmapped_graph_nodes_with_semantic_edge=0,
                           saved_edge_directionality="one_row_per_undirected_pair",
                           cutoff=CUTOFF))
    write_csv(SIM_SUMMARY, result, list(result[0]))
    assert sum(int(r["retained_undirected_pairs"]) for r in result) == len(pair_seen)
    return result


def phase_similarity():
    assert HIERARCHY.is_file(), "Final hierarchy is missing"
    types, drugs, diseases, positives = base_inputs()
    check_graph_leakage(types, positives)
    from DREAMwalk.generate_similarity_net import save_sim_graph
    started = time.monotonic()
    print("Starting official full semantic similarity generation", flush=True)
    save_sim_graph(str(GRAPH), str(HIERARCHY), str(SIMILARITY), cutoff=CUTOFF, directed=False)
    elapsed = time.monotonic() - started
    rows = analyze_similarity()
    update_timing(semantic_similarity_seconds=elapsed, semantic_similarity_peak_rss_bytes=peak_rss_bytes())
    print(json.dumps({"semantic_summary": rows, "semantic_seconds": elapsed,
                      "similarity_sha256": sha256(SIMILARITY)}, indent=2), flush=True)


def xgb_settings():
    return dict(base_score=0.5, booster="gbtree", eval_metric="error",
                objective="binary:logistic", gamma=0, learning_rate=0.1,
                max_depth=6, n_estimators=500, tree_method="auto",
                min_child_weight=4, subsample=0.8, colsample_bytree=0.9,
                scale_pos_weight=1, max_delta_step=1, seed=SEED, n_jobs=WORKERS)


def write_config():
    types, drugs, diseases, positives = base_inputs()
    graph = check_graph_leakage(types, positives)
    neg = read_rows(NEGATIVES, ",")
    assert len(neg) == len({(r["drug"], r["disease"]) for r in neg}) == 5926
    assert not {(r["drug"], r["disease"]) for r in neg} & positives
    sim_rows = read_rows(SIM_SUMMARY, ",")
    counts = {r["semantic_type"]: int(r["retained_undirected_pairs"]) for r in sim_rows}
    config = dict(graph=dict(msi_source="MSI_v1.0.0", **graph, known_treatment_pairs=len(positives),
                             all_treatment_edges_excluded=True, graph_sha256=sha256(GRAPH)),
                  drug_semantics=dict(source="public_MSI_ATC", mapped=1328, unmapped=333,
                                      undirected_semantic_pairs=counts["drug"]),
                  disease_semantics=dict(source="licensed_MRCONSO_SAB_MSH_SDUI", hierarchy="NLM_MeSH_2024",
                                         mapped=748, unmapped=92, undirected_semantic_pairs=counts["disease"],
                                         hierarchy_representation="descriptor_UI_plus_MeSH_tree_position"),
                  dreamwalk=dict(cutoff=CUTOFF, tp_factor=TP_FACTOR, num_walks=NUM_WALKS,
                                 walk_length=WALK_LENGTH, dimension=DIMENSION, window_size=WINDOW,
                                 p=1, q=1, em_max_iter=EM_MAX_ITER,
                                 em_edge_sample_fraction=0.01, em_convergence_threshold=0.01,
                                 workers=WORKERS, hierarchy_sha256=sha256(HIERARCHY),
                                 similarity_sha256=sha256(SIMILARITY),
                                 similarity_directed=False),
                  evaluation=dict(repeat=0, seed=SEED, folds=N_FOLDS,
                                  negative_sampling="random.Random(0).sample(sorted drug×disease candidates excluding all known positives, 5926); future repeats must exclude saved negatives",
                                  negative_pairs_sha256=sha256(NEGATIVES),
                                  feature="drug_embedding - disease_embedding",
                                  xgboost=xgb_settings()),
                  caveats=["Figure 2a teleport factor not published; 0.3 from demo/Figure 4",
                           "Main Methods walk length 4 conflicts with Supplementary Method/released code 10",
                           "Paper says max_depth below six; released code uses six",
                           "C++ Skip-gram and multiprocessing may remain nondeterministic"])
    CONFIG.write_text(json.dumps(config, indent=2) + "\n")
    print("FINAL PRE-RUN CONFIGURATION\n" + json.dumps(config, indent=2), flush=True)
    return config


def phase_embedding():
    assert SIMILARITY.is_file() and SIM_SUMMARY.is_file() and NEGATIVES.is_file()
    analyze_similarity()  # fail before any full embedding if graph checks changed
    config = write_config()
    from DREAMwalk import generate_embeddings as ge
    stage = {}
    actual_walks = {}

    def timed(name, fn):
        def wrapper(*args, **kwargs):
            print("Starting " + name, flush=True)
            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            stage[name] = time.monotonic() - t0
            if name == "random_walk_generation":
                actual_walks["count"] = len(result)
                print("Actual walks:", len(result), flush=True)
            print(name + " seconds:", round(stage[name], 2), flush=True)
            return result
        return wrapper

    ge.train_edgetype_transition_matrix = timed("edge_transition_training", ge.train_edgetype_transition_matrix)
    ge.generate_DREAMwalk_paths = timed("random_walk_generation", ge.generate_DREAMwalk_paths)
    ge.HeterogeneousSG = timed("heterogeneous_skipgram", ge.HeterogeneousSG)
    work = ROOT / "logs/full_embedding"
    work.mkdir(parents=True, exist_ok=True)
    link = work / "DREAMwalk"
    if not link.exists():
        link.symlink_to(UPSTREAM / "DREAMwalk", target_is_directory=True)
    os.environ["CPATH"] = "/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk/usr/include/c++/v1"
    os.chdir(work)
    EMBEDDING.parent.mkdir(parents=True, exist_ok=True)
    swap_before = swap_snapshot()
    started = time.monotonic()
    print("Starting formal repeat-0 embedding", flush=True)
    ge.save_embedding_files(netf=str(GRAPH), sim_netf=str(SIMILARITY), outputf=str(EMBEDDING),
                            nodetypef=str(NODETYPES), tp_factor=TP_FACTOR, seed=SEED,
                            num_walks=NUM_WALKS, walk_length=WALK_LENGTH,
                            dimension=DIMENSION, window_size=WINDOW, workers=WORKERS,
                            em_max_iter=EM_MAX_ITER, p=1, q=1)
    elapsed = time.monotonic() - started
    expected = config["graph"]["nodes"] * NUM_WALKS
    assert actual_walks.get("count") == expected, (actual_walks, expected)
    with EMBEDDING.open("rb") as stream:
        embedding = pickle.load(stream)
    import numpy as np
    types, drugs, diseases, positives = base_inputs()
    assert len(embedding) == len(types) == 29959
    assert all(np.asarray(v).shape == (DIMENSION,) and np.isfinite(v).all() for v in embedding.values())
    assert all(a in embedding and b in embedding for a, b in positives)
    update_timing(**{name + "_seconds": value for name, value in stage.items()},
                  embedding_total_seconds=elapsed,
                  expected_walks=expected, actual_walks=actual_walks["count"],
                  embedding_nodes=len(embedding), embedding_dimension=DIMENSION,
                  embedding_peak_rss_bytes=peak_rss_bytes(),
                  swap_before=swap_before, swap_after=swap_snapshot())
    print("EMBEDDING COMPLETE", json.dumps({"seconds": elapsed, "walks": actual_walks["count"],
                                             "nodes": len(embedding), "dimension": DIMENSION,
                                             "peak_rss_bytes": peak_rss_bytes()}, indent=2), flush=True)


def phase_evaluate():
    assert EMBEDDING.is_file() and CONFIG.is_file()
    import numpy as np
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    types, drugs, diseases, positives = base_inputs()
    check_graph_leakage(types, positives)
    negatives = [(r["drug"], r["disease"]) for r in read_rows(NEGATIVES, ",")]
    assert len(negatives) == len(set(negatives)) == len(positives) == 5926
    assert not set(negatives) & positives
    with EMBEDDING.open("rb") as stream:
        embedding = pickle.load(stream)
    pairs = sorted(positives) + negatives
    assert all(a in embedding and b in embedding for a, b in pairs)
    started_features = time.monotonic()
    x = np.asarray([embedding[a] - embedding[b] for a, b in pairs], dtype=np.float32)
    y = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int8)
    assert x.shape == (11852, 128) and np.isfinite(x).all()
    feature_seconds = time.monotonic() - started_features
    random.seed(SEED)
    np.random.seed(SEED)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    seen_test = set()
    started_cv = time.monotonic()
    for fold, (train, test) in enumerate(splitter.split(x, y), start=1):
        assert not set(train) & set(test)
        assert not seen_test & set(test)
        seen_test.update(test)
        assert set(y[train]) == set(y[test]) == {0, 1}
        t0 = time.monotonic()
        model = XGBClassifier(**xgb_settings())
        model.fit(x[train], y[train])
        prob = model.predict_proba(x[test])[:, 1]
        row = dict(repeat=0, fold=fold, seed=SEED,
                   accuracy=accuracy_score(y[test], prob.round()),
                   auroc=roc_auc_score(y[test], prob),
                   aupr=average_precision_score(y[test], prob),
                   n_train=len(train), n_test=len(test),
                   n_positive_train=int(y[train].sum()),
                   n_positive_test=int(y[test].sum()),
                   runtime_seconds=time.monotonic() - t0)
        assert all(math.isfinite(row[key]) for key in ("accuracy", "auroc", "aupr"))
        rows.append(row)
        print("FOLD", fold, json.dumps(row), flush=True)
    assert seen_test == set(range(len(pairs))) and len(rows) == 5
    write_csv(FOLDS, rows, ["repeat", "fold", "seed", "accuracy", "auroc", "aupr", "n_train", "n_test",
                            "n_positive_train", "n_positive_test", "runtime_seconds"])
    summary = []
    for metric in ("accuracy", "auroc", "aupr"):
        vals = [float(row[metric]) for row in rows]
        summary.append(dict(metric=metric, mean=statistics.mean(vals), SD=statistics.stdev(vals),
                            min=min(vals), max=max(vals), n=len(vals)))
    write_csv(SUMMARY, summary, ["metric", "mean", "SD", "min", "max", "n"])
    update_timing(feature_construction_seconds=feature_seconds,
                  xgboost_five_fold_seconds=time.monotonic()-started_cv,
                  evaluation_peak_rss_bytes=peak_rss_bytes(),
                  final_swap_snapshot=swap_snapshot())
    print("REPEAT-0 SUMMARY", json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "similarity", "embedding", "evaluate"))
    args = parser.parse_args()
    {"prepare": phase_prepare, "similarity": phase_similarity,
     "embedding": phase_embedding, "evaluate": phase_evaluate}[args.phase]()


if __name__ == "__main__":
    main()
