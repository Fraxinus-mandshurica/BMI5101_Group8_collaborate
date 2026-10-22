"""Local MSI node2vec reconstruction: one seed, saved negatives, ten folds.

No experiment executes on import. The notebook launches one phase at a time.
p=q=1 and unit weights make node2vec exactly a uniform neighbor walk in
distribution; the sampler avoids quadratic second-order alias tables. This is
a Python 3 reconstruction, not Bang et al.'s unpublished baseline pipeline.
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
import warnings
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "00.DREAMwalk_reproduction"
INPUTS = {
    "graph": SOURCE / "data/processed/msi_graph.tsv",
    "nodetypes": SOURCE / "data/processed/msi_nodetypes.tsv",
    "positives": SOURCE / "data/processed/msi_positive_pairs.tsv",
    "negatives": SOURCE / "results/msi_repeat0_negative_pairs.csv",
}
PINS = {"numpy": "1.23.3", "scipy": "1.9.1", "scikit-learn": "1.1.3",
        "gensim": "4.3.2", "smart_open": "6.4.0"}
CONFIG = {
    "seed": 0, "p": 1.0, "q": 1.0, "num_walks": 100, "walk_length": 10,
    "vector_size": 128, "window": 4, "workers": 4, "epochs": 1,
    "sg": 1, "hs": 0, "negative": 5, "sample": 0.001,
    "alpha": 0.025, "min_alpha": 0.0001, "ns_exponent": 0.75,
    "min_count": 0, "sorted_vocab": 1, "batch_words": 10000,
    "shrink_windows": True,
    "graph_policy": "undirected simple unit-weight graph; all treatments excluded",
    "walk_sampler": "uniform neighbor sampling specialized to p=q=1; disk corpus",
    "pair_feature": "drug_embedding - disease_embedding",
    "classifier": {"penalty": "l2", "C": 1.0, "solver": "lbfgs",
                   "fit_intercept": True, "class_weight": None,
                   "max_iter": 1000, "tol": 0.0001, "random_state": 0},
    "cv": {"n_splits": 10, "shuffle": True, "random_state": 0},
    "pair_order": "sorted unique positives followed by existing negative CSV order",
    "fold_origin": "deterministically regenerated using existing local ten-fold procedure",
    "aupr_definition": "sklearn.metrics.average_precision_score",
    "implementation_status": "documented reconstruction; not exact author baseline code",
    "unpublished_choices": [
        "Pair subtraction follows released DREAMwalk prediction code; baseline operator is unpublished.",
        "LogisticRegression C=1, L2, lbfgs; max_iter=1000 is a numerical convergence ceiling.",
        "Gensim 4.3.2 standard Skip-gram, one epoch from node2vec reference CLI default.",
        "Other SG settings use explicit Gensim/reference defaults; author baseline SG implementation is unavailable.",
        "Gensim dynamic window/min_alpha differ from released DREAMwalk C++ details.",
        "Four training threads may remain nondeterministic; seed 0 is not a bitwise guarantee.",
        "One repeat with ten folds is not the paper's ten repeats with ten folds.",
    ],
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def save_csv(path, rows):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_rows(path, delimiter="\t"):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def context(run_name):
    if CONFIG["p"] != 1.0 or CONFIG["q"] != 1.0:
        raise ValueError("This fixed MSI sampler supports only p=q=1")
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("Use letters, numbers, underscore or hyphen for run-name")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Run with the notebook-managed Python 3.8 environment")
    installed = {name: version(name) for name in PINS}
    if installed != PINS:
        raise RuntimeError(f"Environment differs from pins: {installed}")
    inputs = {name: {"path": str(path), "sha256": digest(path)} for name, path in INPUTS.items()}
    identity = {"config": CONFIG, "inputs": {k: v["sha256"] for k, v in inputs.items()},
                "packages": installed, "runner_sha256": digest(__file__)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    results = ROOT / "results" / run_name
    data = ROOT / "data" / run_name
    results.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    manifest_path = results / "run_config.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["fingerprint"] != fingerprint:
            raise RuntimeError("Inputs/code/settings changed. Choose a new RUN_NAME; existing output is preserved.")
    else:
        save_json(manifest_path, dict(identity, fingerprint=fingerprint, input_locations=inputs,
                                     python=sys.version, platform=platform.platform(),
                                     pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                                     created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def graph():
    # Numeric tokens also support source node identifiers containing whitespace.
    neighbors = {}
    rows = 0
    with INPUTS["graph"].open() as stream:
        for line in stream:
            a, b, relation, weight, edge_id = line.rstrip("\n").split("\t")
            if float(weight) != 1.0:
                raise ValueError("This sampler requires the existing unit-weight MSI graph")
            neighbors.setdefault(a, set()).add(b)
            neighbors.setdefault(b, set()).add(a)
            rows += 1
    nodes = list(neighbors)  # deterministic input-file insertion order
    index = {node: i for i, node in enumerate(nodes)}
    adjacency = [[index[n] for n in sorted(neighbors[node])] for node in nodes]
    if len(nodes) != 29959 or rows != 478728:
        raise ValueError(f"Unexpected existing MSI input size: {len(nodes)} nodes, {rows} rows")
    return nodes, adjacency


def reuse(path, marker, fingerprint):
    if not marker.exists():
        return False
    info = json.loads(marker.read_text())
    if info["fingerprint"] != fingerprint or not path.exists() or digest(path) != info["sha256"]:
        raise RuntimeError(f"Incomplete or changed checkpoint: {path}; use a new RUN_NAME")
    print(f"Reusing completed checkpoint: {path}", flush=True)
    return True


def walks(data, results, fingerprint):
    import numpy as np
    path, marker = data / "walks.txt", results / "walks_complete.json"
    if reuse(path, marker, fingerprint):
        if not (data / "nodes.json").is_file():
            raise RuntimeError("Saved node-token map missing; use a new RUN_NAME")
        if digest(data / "nodes.json") != json.loads(marker.read_text())["nodes_sha256"]:
            raise RuntimeError("Saved node-token map changed; use a new RUN_NAME")
        return
    started = time.monotonic()
    nodes, adjacency = graph()
    tokens = [str(i) for i in range(len(nodes))]
    order = list(range(len(nodes)))
    shuffle_rng, walk_rng = random.Random(0), np.random.RandomState(0)
    temp = path.with_name("walks.txt.tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as stream:
        for iteration in range(CONFIG["num_walks"]):
            shuffle_rng.shuffle(order)
            for start in order:
                current = start
                route = [tokens[current]]
                for step in range(CONFIG["walk_length"] - 1):
                    adjacent = adjacency[current]
                    if not adjacent:
                        break
                    current = adjacent[int(walk_rng.random_sample() * len(adjacent))]
                    route.append(tokens[current])
                stream.write(" ".join(route) + "\n")
                count += 1
            print(f"Walk round {iteration + 1}/100; {count:,} walks; {time.monotonic()-started:.1f}s", flush=True)
    if count != 2995900:
        raise RuntimeError(f"Unexpected walk count {count}")
    temp.replace(path)
    save_json(data / "nodes.json", nodes)
    save_json(marker, dict(fingerprint=fingerprint, sha256=digest(path),
                           nodes_sha256=digest(data / "nodes.json"), walks=count,
                           nodes=len(nodes), seconds=time.monotonic()-started))


def embedding(data, results, fingerprint):
    path = data / "msi_node2vec_seed0.pkl"
    marker = results / "embedding_complete.json"
    if reuse(path, marker, fingerprint):
        return
    if not (results / "walks_complete.json").exists():
        raise RuntimeError("Run the walks section first")
    walks(data, results, fingerprint)  # reuse existing complete corpus, not a new sample
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    import numpy as np
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    nodes = json.loads((data / "nodes.json").read_text())
    started = time.monotonic()
    kwargs = {key: CONFIG[key] for key in (
        "vector_size", "window", "workers", "epochs", "sg", "hs", "negative", "sample",
        "alpha", "min_alpha", "ns_exponent", "min_count", "sorted_vocab", "batch_words",
        "shrink_windows", "seed")}
    model = Word2Vec(sentences=LineSentence(str(data / "walks.txt")), **kwargs)
    vectors = {node: model.wv[str(i)].copy() for i, node in enumerate(nodes)}
    if len(vectors) != 29959 or any(v.shape != (128,) or not np.isfinite(v).all() for v in vectors.values()):
        raise RuntimeError("Incomplete embedding output")
    # Persist before evaluation; a successful atomic rename leaves a usable embedding.
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        pickle.dump(vectors, stream, protocol=4)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    save_json(marker, dict(fingerprint=fingerprint, sha256=digest(path), nodes=len(vectors),
                           dimensions=128, seconds=time.monotonic()-started))
    print(f"Embedding saved: {path}", flush=True)


def evaluate(data, results, fingerprint):
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    summary_path = results / "summary.csv"
    marker = results / "evaluation_complete.json"
    if reuse(summary_path, marker, fingerprint):
        return
    emb_path = data / "msi_node2vec_seed0.pkl"
    if not (results / "embedding_complete.json").exists() or not reuse(emb_path, results / "embedding_complete.json", fingerprint):
        raise RuntimeError("Run the embedding section first")
    with emb_path.open("rb") as stream:
        vectors = pickle.load(stream)
    positive_set = {(r["drug"], r["disease"]) for r in read_rows(INPUTS["positives"])}
    negatives = [(r["drug"], r["disease"]) for r in read_rows(INPUTS["negatives"], ",")]
    if len(positive_set) != 5926 or len(negatives) != 5926 or len(set(negatives)) != 5926 or positive_set & set(negatives):
        raise ValueError("Unexpected saved pair set")
    pairs = sorted(positive_set) + negatives
    x = np.asarray([vectors[a] - vectors[b] for a, b in pairs], dtype=np.float32)
    y = np.asarray([1]*5926 + [0]*5926, dtype=np.int8)
    splitter = StratifiedKFold(**CONFIG["cv"])
    splits = list(splitter.split(x, y))
    assignments = np.empty(len(pairs), dtype=np.int32)
    for fold, (_, test) in enumerate(splits, 1):
        assignments[test] = fold
    save_csv(results / "fold_assignments.csv", [
        dict(pair_index=i, drug=a, disease=b, label=int(y[i]), test_fold=int(assignments[i]))
        for i, (a, b) in enumerate(pairs)])
    rows, prediction_rows = [], []
    started = time.monotonic()
    for fold, (train, test) in enumerate(splits, 1):
        t0 = time.monotonic()
        classifier = LogisticRegression(**CONFIG["classifier"])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x[train], y[train])
        messages = [str(w.message) for w in captured]
        for message in messages:
            print(f"Fold {fold} warning: {message}", flush=True)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
        probabilities = classifier.predict_proba(x[test])[:, 1]
        row = dict(seed=0, fold=fold, accuracy=float(accuracy_score(y[test], probabilities >= 0.5)),
                   auroc=float(roc_auc_score(y[test], probabilities)),
                   aupr=float(average_precision_score(y[test], probabilities)),
                   n_train=len(train), n_test=len(test), converged=converged,
                   iterations=int(classifier.n_iter_.max()), warnings=" | ".join(messages),
                   seconds=time.monotonic()-t0)
        rows.append(row)
        prediction_rows.extend(dict(pair_index=int(i), fold=fold, label=int(y[i]), probability=float(p))
                               for i, p in zip(test, probabilities))
        save_csv(results / "fold_results.csv", rows)
        save_csv(results / "oof_predictions.csv", prediction_rows)
        print(json.dumps(row), flush=True)
    summary = []
    for metric in ("accuracy", "auroc", "aupr"):
        values = [r[metric] for r in rows]
        summary.append(dict(metric=metric, mean=statistics.mean(values), SD=statistics.stdev(values),
                            n_folds=10, n_repeats=1, all_folds_converged=all(r["converged"] for r in rows)))
    save_csv(summary_path, summary)
    reference = read_rows(ROOT / "reference/paper_node2vec_fig2a_summary.csv", ",")
    save_csv(results / "paper_comparison.csv", [
        dict(metric=local["metric"], local_seed0_10fold_mean=local["mean"],
             paper_10repeat_mean=float(paper["mean"]),
             local_minus_paper=local["mean"]-float(paper["mean"]),
             paper_SD_between_repeats=float(paper["SD"]),
             local_SD_between_folds=local["SD"], all_folds_converged=local["all_folds_converged"])
        for local, paper in zip(summary, reference)])
    save_json(marker, dict(fingerprint=fingerprint, sha256=digest(summary_path),
                           seconds=time.monotonic()-started, all_folds_converged=all(r["converged"] for r in rows)))
    print("Evaluation complete\n" + json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "walks", "embedding", "evaluate"))
    parser.add_argument("--run-name", default="msi_seed0")
    args = parser.parse_args()
    data, results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    else:
        globals()[args.phase](data, results, fingerprint)


if __name__ == "__main__":
    main()
