"""Local MSI DTi2vec reconstruction with training-only K selection.

Phases are launched explicitly by the notebook; importing never runs an
experiment. Only new-project directories receive generated files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import pickle
import platform
import random
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

# Shared upstream modules are read-only inputs; imports must not write caches
# back into the already completed DREAMwalk project.
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT.parent / "00.DREAMwalk_reproduction"
UPSTREAM = SHARED / "vendor/DREAMwalk_upstream"
INPUTS = {
    "graph": SHARED / "data/processed/msi_graph.tsv",
    "types": SHARED / "data/processed/msi_nodetypes.tsv",
    "positives": SHARED / "data/processed/msi_positive_pairs.tsv",
    "negatives": SHARED / "results/msi_repeat0_negative_pairs.csv",
    "hierarchy": SHARED / "data/processed/msi_semantic_hierarchy_final.csv",
}
PINS = {"numpy": "1.23.3", "scipy": "1.9.1", "scikit-learn": "1.1.3",
        "networkx": "2.8.8", "pandas": "1.4.4", "gensim": "4.3.2",
        "smart_open": "6.4.0", "xgboost": "1.7.4"}
K_GRID = (5, 10, 20, 50)
SG = dict(vector_size=128, window=4, workers=4, epochs=1, sg=1, hs=0,
          negative=5, sample=0.001, alpha=0.025, min_alpha=0.0001,
          ns_exponent=0.75, min_count=0, sorted_vocab=1, batch_words=10000,
          shrink_windows=True, seed=0)
XGB = dict(base_score=0.5, booster="gbtree", eval_metric="error",
           objective="binary:logistic", gamma=0, learning_rate=0.1,
           max_depth=6, n_estimators=500, tree_method="auto", min_child_weight=4,
           subsample=0.8, colsample_bytree=0.9, scale_pos_weight=1,
           max_delta_step=1, seed=0, n_jobs=4)
SETTINGS = dict(seed=0, k_grid=list(K_GRID), outer_folds=10, inner_folds=3,
                k_selection="mean inner average_precision; ties choose smallest K",
                similarity_fusion="0.5 * Jaccard + 0.5 * semantic; missing semantic=0",
                semantic_cutoff=0.0, knn="top-K positive scores per row; undirected union",
                graph_nodes="1661 drugs + 840 diseases; gene nodes only define target sets",
                graph_edges="KNN similarity edges + current training-positive DDA bridges weight=1",
                num_walks=100, walk_length=10, p=1, q=1,
                walk_sampler="weighted first-order node2vec equivalent for p=q=1, disk corpus",
                skipgram=SG, xgboost=XGB, feature="concatenate drug and disease embeddings (256D)",
                scaling="MinMaxScaler fit only on current training features",
                pair_order="sorted positives followed by original saved negative order",
                scope="one seed-0 outer ten-fold repeat; 120 inner fits + 10 final fits",
                choices=["Arithmetic-mean fusion and KNN symmetrization are explicit reconstruction choices.",
                         "Bang et al. did not publish the MSI DTi2vec baseline code or K.",
                         "Three inner folds limit local compute; outer ten-fold procedure is shared.",
                         "Standard Gensim Skip-gram has no DREAMwalk pp parameter.",
                         "XGBoost uses common DREAMwalk evaluation settings, not original DTi2Vec defaults.",
                         "No treatment labels enter similarity calculation; only training labels enter graph bridges.",
                         "Four-thread Skip-gram may vary between runs."])
METRICS = ("accuracy", "auroc", "aupr")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_rows(path, delimiter="\t"):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def context(name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("RUN_NAME accepts letters, digits, underscore and hyphen")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Use the notebook-managed Python 3.8 environment")
    packages = {p: version(p) for p in PINS}
    if packages != PINS:
        raise RuntimeError(f"Environment differs from fixed pins: {packages}")
    sources = [Path(__file__), UPSTREAM / "DREAMwalk/generate_similarity_net.py",
               UPSTREAM / "DREAMwalk/utils.py"]
    identity = dict(settings=SETTINGS, packages=packages,
                    inputs={k: sha256(v) for k, v in INPUTS.items()},
                    sources={str(p): sha256(p) for p in sources})
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    data, results = ROOT / "data" / name, ROOT / "results" / name
    data.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    manifest = results / "run_config.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("Inputs/code/settings changed; use a new RUN_NAME to preserve existing output")
    else:
        save_json(manifest, dict(identity, fingerprint=fingerprint,
                  input_paths={k: str(v) for k, v in INPUTS.items()}, python=sys.version,
                  platform=platform.platform(), pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def load_inputs():
    import numpy as np
    types = {r["node"]: r["type"] for r in read_rows(INPUTS["types"])}
    drugs = sorted(node for node, kind in types.items() if kind == "drug")
    diseases = sorted(node for node, kind in types.items() if kind == "disease")
    nodes = drugs + diseases
    positives = sorted({(r["drug"], r["disease"]) for r in read_rows(INPUTS["positives"])})
    negatives = [(r["drug"], r["disease"]) for r in read_rows(INPUTS["negatives"], ",")]
    if len(drugs) != 1661 or len(diseases) != 840 or len(set(nodes)) != 2501:
        raise ValueError("Unexpected drug/disease node vocabulary")
    if len(positives) != 5926 or len(negatives) != 5926 or len(set(negatives)) != 5926 or set(positives) & set(negatives):
        raise ValueError("Unexpected shared DDA sample set")
    pairs = positives + negatives
    if any(types[a] != "drug" or types[b] != "disease" for a, b in pairs):
        raise ValueError("Pair endpoints must be drugs and diseases")
    return types, drugs, diseases, nodes, pairs, np.asarray([1]*5926 + [0]*5926, dtype=np.int8)


def reuse(path, marker, fingerprint):
    if not marker.exists():
        return False
    info = json.loads(marker.read_text())
    if info["fingerprint"] != fingerprint or not path.exists() or sha256(path) != info["sha256"]:
        raise RuntimeError(f"Incomplete or changed checkpoint: {path}; use a new RUN_NAME")
    print("Reusing:", path, flush=True)
    return True


def target_similarity(nodes, gene_sets):
    """Binary gene membership -> Jaccard; empty-set scores and diagonal are zero."""
    import numpy as np
    from scipy.sparse import csr_matrix
    genes = sorted({g for node in nodes for g in gene_sets[node]})
    index = {g: i for i, g in enumerate(genes)}
    rows, cols = [], []
    for i, node in enumerate(nodes):
        for gene in sorted(gene_sets[node]):
            rows.append(i)
            cols.append(index[gene])
    memberships = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                             shape=(len(nodes), len(genes)))
    overlap = (memberships @ memberships.T).toarray()
    counts = np.asarray(memberships.sum(axis=1)).ravel()
    union = counts[:, None] + counts[None, :] - overlap
    scores = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    np.fill_diagonal(scores, 0)
    return scores


def knn_union(scores, k):
    """Select top-K positive neighbors; tie-break by sorted node index."""
    import numpy as np
    from scipy.sparse import csr_matrix
    rows, cols, values = [], [], []
    for i in range(len(scores)):
        candidates = np.flatnonzero(scores[i] > 0)
        candidates = candidates[candidates != i]
        order = np.lexsort((candidates, -scores[i, candidates]))
        selected = candidates[order[:k]]
        rows.extend([i] * len(selected))
        cols.extend(selected.tolist())
        values.extend(scores[i, selected].tolist())
    directed = csr_matrix((values, (rows, cols)), shape=scores.shape, dtype=np.float32)
    return directed.maximum(directed.T).tocsr()


def prepare(data, results, fingerprint):
    import numpy as np
    from scipy.sparse import block_diag, save_npz
    artifact, marker = data / "similarities.npz", results / "prepare_complete.json"
    if reuse(artifact, marker, fingerprint):
        info = json.loads(marker.read_text())
        for k in K_GRID:
            path = data / f"graph_knn_k{k}.npz"
            if not path.exists() or sha256(path) != info["knn_sha256"][str(k)]:
                raise RuntimeError(f"Prepared KNN graph missing or changed: {path}")
        if sha256(data / "nodes.json") != info["nodes_sha256"]:
            raise RuntimeError("Saved node vocabulary changed")
        return
    started = time.monotonic()
    types, drugs, diseases, nodes, _, _ = load_inputs()
    gene_sets = {node: set() for node in nodes}
    with INPUTS["graph"].open() as stream:
        for line in stream:
            a, b, rel, weight, edge_id = line.rstrip("\n").split("\t")
            if rel in ("1", "2"):
                if a in gene_sets and types[b] == "gene":
                    gene_sets[a].add(b)
                elif b in gene_sets and types[a] == "gene":
                    gene_sets[b].add(a)
                else:
                    raise ValueError("Unexpected endpoint types for target association")
    print("Calculating target Jaccard similarity from shared MSI gene sets", flush=True)
    jt_drug = target_similarity(drugs, gene_sets)
    jt_disease = target_similarity(diseases, gene_sets)
    # Same hierarchy and formula as DREAMwalk, with no teleport cutoff before KNN.
    sys.path.insert(0, str(UPSTREAM))
    from DREAMwalk.generate_similarity_net import save_sim_graph
    semantic_file = data / "semantic_raw.tsv"
    print("Calculating raw semantic scores from existing MeSH/ATC hierarchy (cutoff=0)", flush=True)
    save_sim_graph(str(INPUTS["graph"]), str(INPUTS["hierarchy"]), str(semantic_file),
                   cutoff=0.0, weighted=True, directed=False)
    sem_drug = np.zeros_like(jt_drug)
    sem_disease = np.zeros_like(jt_disease)
    drug_index, disease_index = {n: i for i, n in enumerate(drugs)}, {n: i for i, n in enumerate(diseases)}
    with semantic_file.open() as stream:
        for line in stream:
            a, b, kind, weight, edge_id = line.rstrip("\n").split("\t")
            index, matrix = (drug_index, sem_drug) if types[a] == "drug" else (disease_index, sem_disease)
            i, j = index[a], index[b]
            matrix[i, j] = matrix[j, i] = float(weight)
    fused_drug = (jt_drug + sem_drug) * 0.5
    fused_disease = (jt_disease + sem_disease) * 0.5
    np.savez_compressed(artifact, target_drug=jt_drug, target_disease=jt_disease,
                        semantic_drug=sem_drug, semantic_disease=sem_disease,
                        fused_drug=fused_drug, fused_disease=fused_disease)
    save_json(data / "nodes.json", nodes)
    knn_sha, counts = {}, {}
    for k in K_GRID:
        graph = block_diag((knn_union(fused_drug, k), knn_union(fused_disease, k)), format="csr")
        path = data / f"graph_knn_k{k}.npz"
        save_npz(path, graph)
        knn_sha[str(k)], counts[str(k)] = sha256(path), int(graph.nnz // 2)
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(artifact),
                           knn_sha256=knn_sha, knn_undirected_edges=counts,
                           nodes_sha256=sha256(data / "nodes.json"),
                           semantic_raw_sha256=sha256(semantic_file),
                           drugs=len(drugs), diseases=len(diseases), seconds=time.monotonic()-started))
    print("SIMILARITY PREPARATION COMPLETE", flush=True)


def add_training_bridges(base, pairs, labels, train_indices, node_index):
    """Only training positives become drug-disease graph edges."""
    graph = base.tolil(copy=True)
    for i in train_indices:
        if labels[i] == 1:
            a, b = pairs[i]
            graph[node_index[a], node_index[b]] = 1.0
            graph[node_index[b], node_index[a]] = 1.0
    return graph.tocsr()


def stream_walks(graph, path):
    import numpy as np
    order = list(range(graph.shape[0]))
    neighbors, cumulative = [], []
    for i in order:
        begin, end = graph.indptr[i:i+2]
        neighbors.append(graph.indices[begin:end])
        weights = graph.data[begin:end]
        cdf = np.cumsum(weights, dtype=np.float64)
        if len(cdf):
            cdf /= cdf[-1]
            cdf[-1] = 1.0
        cumulative.append(cdf)
    shuffle_rng, walk_rng = random.Random(0), np.random.RandomState(0)
    tokens = [str(i) for i in order]
    with path.open("w", encoding="utf-8") as stream:
        for iteration in range(100):
            shuffle_rng.shuffle(order)
            for start in order:
                current = start
                route = [tokens[current]]
                for step in range(9):
                    if not len(neighbors[current]):
                        break
                    chosen = np.searchsorted(cumulative[current], walk_rng.random_sample(), side="right")
                    current = int(neighbors[current][chosen])
                    route.append(tokens[current])
                stream.write(" ".join(route) + "\n")


def indices_digest(indices):
    return hashlib.sha256(json.dumps([int(i) for i in indices]).encode()).hexdigest()


def fit_case(case, k, train, validation, base_graph, nodes, pairs, labels, data, results, fingerprint):
    import numpy as np
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    from sklearn.preprocessing import MinMaxScaler
    from xgboost import XGBClassifier
    case_data, case_results = data / case, results / case
    case_data.mkdir(parents=True, exist_ok=True)
    case_results.mkdir(parents=True, exist_ok=True)
    embedding = case_data / "embedding.pkl"
    marker = case_results / "embedding_complete.json"
    metric_path = case_results / "metrics.json"
    identity = dict(fingerprint=fingerprint, k=int(k), train_indices_sha256=indices_digest(train),
                    validation_indices_sha256=indices_digest(validation))
    if set(map(int, train)) & set(map(int, validation)):
        raise ValueError("Training and validation/test partitions overlap")
    if metric_path.exists():
        metrics = json.loads(metric_path.read_text())
        if any(metrics[key] != value for key, value in identity.items()):
            raise RuntimeError("Cached case partitions or configuration changed")
        if not reuse(embedding, marker, fingerprint):
            raise RuntimeError("Cached case embedding missing")
        print("Reusing completed fit:", case, flush=True)
        return metrics
    started = time.monotonic()
    print("FIT", case, "K", k, "train", len(train), "held-out", len(validation), flush=True)
    if reuse(embedding, marker, fingerprint):
        info = json.loads(marker.read_text())
        if any(info[key] != value for key, value in identity.items()):
            raise RuntimeError("Cached embedding was built for different partitions")
        with embedding.open("rb") as stream:
            vectors = pickle.load(stream)
    else:
        if embedding.exists():
            raise RuntimeError(f"Embedding has no completion record; preserve for recovery: {embedding}")
        node_index = {n: i for i, n in enumerate(nodes)}
        graph = add_training_bridges(base_graph, pairs, labels, train, node_index)
        corpus = case_data / "walks.txt"
        walk_start = time.monotonic()
        stream_walks(graph, corpus)
        walk_seconds = time.monotonic() - walk_start
        sg_start = time.monotonic()
        model = Word2Vec(sentences=LineSentence(str(corpus)), **SG)
        vectors = {node: model.wv[str(i)].copy() for i, node in enumerate(nodes)}
        if any(v.shape != (128,) or not np.isfinite(v).all() for v in vectors.values()):
            raise ValueError("Invalid embedding output")
        temporary = embedding.with_name("embedding.pkl.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(vectors, stream, protocol=4)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(embedding)
        save_json(marker, dict(identity, sha256=sha256(embedding), nodes=len(vectors),
                               graph_undirected_edges=int(graph.nnz//2),
                               training_positive_bridges=int(labels[train].sum()),
                               walks=100*len(nodes), walk_seconds=walk_seconds,
                               skipgram_seconds=time.monotonic()-sg_start))
        # Embedding is durable. Corpus is reproducible from this fit's saved
        # partition identities and is not needed for the requested evaluation.
        corpus.unlink()
        del graph, model
        print("Embedding saved:", embedding, flush=True)

    def features(indices):
        return np.asarray([np.concatenate((vectors[pairs[i][0]], vectors[pairs[i][1]]))
                           for i in indices], dtype=np.float32)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(features(train))
    x_held_out = scaler.transform(features(validation))
    classifier = XGBClassifier(**XGB)
    classifier.fit(x_train, labels[train])
    probabilities = classifier.predict_proba(x_held_out)[:, 1]
    metrics = dict(identity, accuracy=float(accuracy_score(labels[validation], probabilities.round())),
                   auroc=float(roc_auc_score(labels[validation], probabilities)),
                   aupr=float(average_precision_score(labels[validation], probabilities)),
                   n_train=len(train), n_held_out=len(validation),
                   held_out_indices=[int(i) for i in validation],
                   probabilities=[float(p) for p in probabilities], seconds=time.monotonic()-started)
    save_json(metric_path, metrics)
    print("RESULT", case, {key: metrics[key] for key in METRICS}, flush=True)
    return metrics


def select_k(inner_rows):
    """Pick by inner AUPR only, with a deterministic smallest-K tie-break."""
    scores = {k: statistics.mean(r["aupr"] for r in inner_rows if r["k"] == k) for k in K_GRID}
    winner = max(scores, key=lambda k: (scores[k], -k))
    return winner, scores


def benchmark(data, results, fingerprint):
    import numpy as np
    from scipy.sparse import load_npz
    from sklearn.model_selection import StratifiedKFold
    summary_path = results / "summary.csv"
    marker = results / "benchmark_complete.json"
    if reuse(summary_path, marker, fingerprint):
        return
    prepare(data, results, fingerprint)
    _, _, _, nodes, pairs, labels = load_inputs()
    bases = {k: load_npz(data / f"graph_knn_k{k}.npz") for k in K_GRID}
    logging.basicConfig(level=logging.WARNING)
    outer = list(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(np.zeros(len(pairs)), labels))
    assignments = np.empty(len(pairs), dtype=np.int32)
    for fold, (_, test) in enumerate(outer, 1):
        assignments[test] = fold
    save_csv(results / "fold_assignments.csv", [dict(pair_index=i, drug=a, disease=b,
              label=int(labels[i]), test_fold=int(assignments[i])) for i, (a, b) in enumerate(pairs)])
    inner_rows, selected_rows, folds, predictions = [], [], [], []
    for fold, (train, test) in enumerate(outer, 1):
        print(f"OUTER FOLD {fold}/10: select K on training partitions only", flush=True)
        inner_splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=0).split(np.zeros(len(train)), labels[train]))
        this_inner = []
        for k in K_GRID:
            for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits, 1):
                case = f"outer{fold:02d}/inner{inner_fold:02d}/k{k:03d}"
                measured = fit_case(case, k, train[inner_train], train[inner_valid], bases[k],
                                    nodes, pairs, labels, data, results, fingerprint)
                row = dict(outer_fold=fold, inner_fold=inner_fold, k=k,
                           **{m: measured[m] for m in METRICS}, seconds=measured["seconds"])
                this_inner.append(row)
                inner_rows.append(row)
                save_csv(results / "inner_cv_scores.csv", inner_rows)
        chosen, scores = select_k(this_inner)
        selected_rows.append(dict(outer_fold=fold, selected_k=chosen,
                             **{f"k{k}_mean_inner_aupr": scores[k] for k in K_GRID}))
        save_csv(results / "selected_k.csv", selected_rows)
        print("Selected K:", chosen, "inner scores:", scores, flush=True)
        measured = fit_case(f"outer{fold:02d}/final/k{chosen:03d}", chosen, train, test,
                            bases[chosen], nodes, pairs, labels, data, results, fingerprint)
        folds.append(dict(seed=0, fold=fold, selected_k=chosen,
                     **{m: measured[m] for m in METRICS}, n_train=len(train), n_test=len(test),
                     n_positive_train=int(labels[train].sum()), n_positive_test=int(labels[test].sum()),
                     seconds=measured["seconds"]))
        predictions.extend(dict(pair_index=i, fold=fold, drug=pairs[i][0], disease=pairs[i][1],
                           label=int(labels[i]), probability=p)
                           for i, p in zip(measured["held_out_indices"], measured["probabilities"]))
        save_csv(results / "fold_results.csv", folds)
        save_csv(results / "oof_predictions.csv", predictions)
    if len(predictions) != len(pairs) or len({r["pair_index"] for r in predictions}) != len(pairs):
        raise RuntimeError("Outer test predictions do not cover each pair exactly once")
    summary = [dict(metric=m, mean=statistics.mean(r[m] for r in folds),
                   SD=statistics.stdev(r[m] for r in folds), n_folds=10, n_repeats=1)
               for m in METRICS]
    save_csv(summary_path, summary)
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(summary_path),
                           outer_folds=10, inner_folds=3, fit_cases=130, oof_pairs=len(predictions)))
    print("BENCHMARK COMPLETE\n" + json.dumps(summary, indent=2), flush=True)


def compare(data, results, fingerprint):
    summary = results / "summary.csv"
    if not reuse(summary, results / "benchmark_complete.json", fingerprint):
        raise RuntimeError("Run the benchmark phase first")
    reference = {r["metric"]: r for r in read_rows(ROOT / "reference/paper_dti2vec_fig2a_summary.csv", ",")}
    rows = []
    for row in read_rows(summary, ","):
        paper = reference[row["metric"]]
        rows.append(dict(metric=row["metric"], local_seed0_nestedcv_mean=float(row["mean"]),
                         paper_10repeat_mean=float(paper["mean"]),
                         local_minus_paper=float(row["mean"])-float(paper["mean"]),
                         local_fold_SD=float(row["SD"]), paper_repeat_SD=float(paper["SD"])))
    path = results / "paper_comparison.csv"
    save_csv(path, rows)
    save_json(results / "comparison_complete.json", dict(fingerprint=fingerprint, sha256=sha256(path)))
    print("DTi2vec PAPER COMPARISON\n" + json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "prepare", "benchmark", "compare"))
    parser.add_argument("--run-name", default="msi_seed0")
    args = parser.parse_args()
    data, results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    else:
        globals()[args.phase](data, results, fingerprint)


if __name__ == "__main__":
    main()
