"""Local MSI edge2vec baseline using a pinned, documented upstream port.

Importing this module does not read MSI inputs or run an experiment. The
notebook calls each phase separately; completed stages are reusable.
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

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "00.DREAMwalk_reproduction"
UPSTREAM_COMMIT = "b9250bd17112d203c2a5b496b75b4b6b259da9ed"
INPUTS = {
    "graph": SOURCE / "data/processed/msi_graph.tsv",
    "nodetypes": SOURCE / "data/processed/msi_nodetypes.tsv",
    "positives": SOURCE / "data/processed/msi_positive_pairs.tsv",
    "negatives": SOURCE / "results/msi_repeat0_negative_pairs.csv",
}
PINS = {"numpy": "1.23.3", "scipy": "1.9.1", "scikit-learn": "1.1.3",
        "networkx": "2.8.8", "pandas": "1.4.4", "gensim": "4.3.2",
        "smart_open": "6.4.0"}
CONFIG = {
    "seed": 0, "p": 1.0, "q": 1.0, "directed": False,
    "edge_type_count": 5, "em_iterations": 5,
    "em_walks_per_node_per_iteration": 1, "em_walk_length": 10,
    "transition_metric": "sigmoid(Pearson correlation of per-walk edge-type counts)",
    "undefined_pearson_fallback": "original convention: correlation -1 -> sigmoid(-1); count all fallbacks",
    "num_walks": 100, "walk_length": 10,
    "vector_size": 128, "window": 4, "workers": 4, "epochs": 1,
    "sg": 1, "hs": 0, "negative": 5, "sample": 0.001,
    "alpha": 0.025, "min_alpha": 0.0001, "ns_exponent": 0.75,
    "min_count": 0, "sorted_vocab": 1, "batch_words": 10000,
    "shrink_windows": True,
    "graph_policy": "shared undirected MSI molecular graph, original five edge types; no DDA edges",
    "em_sampler": "one official node walk per node per EM iteration, converted to adjacent edge-type IDs",
    "walk_sampler": "ported official edge2vec.edge2vec_walk; sequential sampling, disk corpus",
    "token_map": "numeric string tokens in shared input-file node insertion order",
    "pair_feature": "drug_embedding - disease_embedding",
    "classifier": {"penalty": "l2", "C": 1.0, "solver": "lbfgs",
                   "fit_intercept": True, "class_weight": None,
                   "max_iter": 1000, "tol": 0.0001, "random_state": 0},
    "cv": {"n_splits": 10, "shuffle": True, "random_state": 0},
    "pair_order": "sorted unique positives followed by existing seed-0 negative CSV order",
    "fold_origin": "deterministically regenerated using shared local ten-fold procedure",
    "aupr_definition": "sklearn.metrics.average_precision_score",
    "upstream": {"repository": "https://github.com/RoyZhengGao/edge2vec", "commit": UPSTREAM_COMMIT},
    "implementation_status": "documented Python 3/Gensim 4 port and Bang protocol adapter; not unpublished author MSI baseline code",
    "recorded_adaptations": [
        "Python 2 print syntax converted; unused matplotlib import removed; Gensim size/iter renamed vector_size/epochs.",
        "Pearson routine's stray Wilcoxon overwrite removed to implement Bang's specified Pearson+sigmoid formula.",
        "Return-to-previous-node branch corrected to assign next, the variable appended by official node walk.",
        "Bang's one walk per node replaces original transition CLI's edge-started walks and 1000-edge cap.",
        "Five EM iterations follow original edge2vec CLI default; Bang does not state this iteration count.",
        "One SG epoch matches our node2vec comparator, overriding original edge2vec CLI's five-epoch default.",
        "LR C=1/L2/lbfgs and pair subtraction reuse local node2vec protocol; baseline details are unpublished.",
        "Undefined Pearson correlation preserves original fallback -1 and is counted per iteration.",
        "Four SG threads may remain nondeterministic; seed 0 is not a bitwise reproducibility guarantee.",
        "One repeat with ten folds is compared with paper's ten repeats with ten folds.",
    ],
}


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot save a CSV without rows")
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def read_rows(path, delimiter="\t"):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def source_hashes():
    paths = [Path(__file__), ROOT / "src/compat/edge2vec.py",
             ROOT / "src/compat/transition.py", ROOT / "src/compat/__init__.py",
             ROOT / "vendor/edge2vec_upstream/edge2vec.py",
             ROOT / "vendor/edge2vec_upstream/transition.py",
             ROOT / "vendor/edge2vec_upstream/UPSTREAM_SNAPSHOT.json"]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}


def context(run_name):
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("Use letters, numbers, underscore or hyphen for run-name")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Run with the notebook-managed Python 3.8 environment")
    installed = {name: version(name) for name in PINS}
    if installed != PINS:
        raise RuntimeError(f"Environment differs from pins: {installed}")
    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Shared DREAMwalk input files missing: " + ", ".join(missing))
    inputs = {name: {"path": str(path), "sha256": digest(path)} for name, path in INPUTS.items()}
    identity = {"config": CONFIG, "inputs": {key: item["sha256"] for key, item in inputs.items()},
                "packages": installed, "source_sha256": source_hashes()}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    data, results = ROOT / "data" / run_name, ROOT / "results" / run_name
    data.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    manifest = results / "run_config.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("Inputs/code/settings changed. Choose a new RUN_NAME; existing output is preserved.")
    else:
        save_json(manifest, dict(identity, fingerprint=fingerprint, input_locations=inputs,
                                 python=sys.version, platform=platform.platform(),
                                 pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                                 created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def output_key(path):
    return path.relative_to(ROOT).as_posix()


def checkpoint(marker, fingerprint, outputs, **info):
    save_json(marker, dict(fingerprint=fingerprint,
                           outputs_sha256={output_key(path): digest(path) for path in outputs}, **info))


def completed(marker, fingerprint):
    if not marker.is_file():
        return False
    info = json.loads(marker.read_text())
    if info["fingerprint"] != fingerprint:
        raise RuntimeError(f"Changed stage configuration: {marker}; choose a new RUN_NAME")
    for relative, expected in info["outputs_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f"Incomplete or changed checkpoint: {path}; choose a new RUN_NAME")
    print(f"Reusing completed stage: {marker.name}", flush=True)
    return True


def require_stage(stage, results, fingerprint):
    if not completed(results / f"{stage}_complete.json", fingerprint):
        raise RuntimeError(f"Run the {stage} notebook section first")


def seed_sampling():
    import numpy as np
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])


def atomic_pickle(path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        pickle.dump(value, stream, protocol=4)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def load_graph(data):
    with (data / "graph.pkl").open("rb") as stream:
        return pickle.load(stream)


def prepare(data, results, fingerprint):
    marker = results / "prepare_complete.json"
    if completed(marker, fingerprint):
        return
    from compat import edge2vec as official
    started = time.monotonic()
    nodes, node_index = [], {}
    relations = {kind: 0 for kind in range(1, 6)}
    typed_path = data / "typed_graph.tsv"
    temp = typed_path.with_name(typed_path.name + ".tmp")
    with INPUTS["graph"].open(encoding="utf-8") as source, temp.open("w", encoding="utf-8") as target:
        for row_number, line in enumerate(source, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 5:
                raise ValueError(f"MSI graph row {row_number} must have five columns")
            a, b, relation, weight, edge_id = fields
            kind = int(relation)
            if kind not in relations or float(weight) != 1.0:
                raise ValueError(f"Unexpected MSI edge type or weight on row {row_number}")
            relations[kind] += 1
            for node in (a, b):
                if node not in node_index:
                    node_index[node] = len(nodes)
                    nodes.append(node)
            target.write(f"{node_index[a]}\t{node_index[b]}\t{kind}\t{weight}\t{int(edge_id)}\n")
    if len(nodes) != 29959 or sum(relations.values()) != 478728:
        raise ValueError(f"Unexpected shared MSI dimensions: {len(nodes)} nodes, {relations}")
    expected = {1: 8568, 2: 25212, 3: 387626, 4: 34777, 5: 22545}
    if relations != expected:
        raise ValueError(f"Shared edge-type counts differ: {relations}")
    temp.replace(typed_path)
    graph = official.read_graph(str(typed_path), weighted=True, directed=False)
    if graph.number_of_nodes() != len(nodes) or graph.number_of_edges() != 478728:
        raise ValueError("Official graph reader changed the shared graph dimensions")
    save_json(data / "nodes.json", nodes)
    atomic_pickle(data / "graph.pkl", graph)
    checkpoint(marker, fingerprint, [typed_path, data / "nodes.json", data / "graph.pkl"],
               nodes=len(nodes), edges=graph.number_of_edges(), edge_type_counts=relations,
               treatments_in_graph=0, seconds=time.monotonic()-started)
    print(f"Prepared {len(nodes):,} nodes and {graph.number_of_edges():,} typed molecular edges", flush=True)


def node_route_to_types(graph, route):
    return [str(graph[a][b]["type"]) for a, b in zip(route, route[1:])]


def transition(data, results, fingerprint):
    marker = results / "transition_complete.json"
    if completed(marker, fingerprint):
        return
    require_stage("prepare", results, fingerprint)
    import numpy as np
    from compat import edge2vec as official_walk
    from compat import transition as official_transition
    seed_sampling()
    graph = load_graph(data)
    nodes = list(graph.nodes())
    matrix = official_transition.initialize_edge_type_matrix(CONFIG["edge_type_count"])
    started, rows = time.monotonic(), []
    outputs = []
    for iteration in range(1, CONFIG["em_iterations"] + 1):
        random.shuffle(nodes)
        routes = []
        t0 = time.monotonic()
        for count, node in enumerate(nodes, 1):
            route = official_walk.edge2vec_walk(graph, CONFIG["em_walk_length"], node,
                                                matrix, CONFIG["directed"], CONFIG["p"], CONFIG["q"])
            routes.append(node_route_to_types(graph, route))
            if count % 5000 == 0:
                print(f"EM {iteration}/{CONFIG['em_iterations']}: {count:,}/{len(nodes):,} node walks", flush=True)
        official_transition.pearson_fallback_audit.update(constant_pairs=0, nonfinite_pairs=0)
        matrix = np.asarray(official_transition.update_trans_matrix(
            routes, CONFIG["edge_type_count"], 4), dtype=np.float64)
        if matrix.shape != (5, 5) or not np.isfinite(matrix).all() or (matrix <= 0).any():
            raise RuntimeError("Transition matrix must be finite, positive and 5 by 5")
        if not np.allclose(matrix, matrix.T, atol=1e-12):
            raise RuntimeError("Pearson transition matrix must be symmetric")
        iteration_path = data / f"transition_matrix_iter{iteration:02d}.txt"
        np.savetxt(iteration_path, matrix, fmt="%.17g")
        outputs.append(iteration_path)
        row = dict(iteration=iteration, node_walks=len(routes),
                   edge_type_tokens=sum(map(len, routes)),
                   constant_pairs=official_transition.pearson_fallback_audit["constant_pairs"],
                   nonfinite_pairs=official_transition.pearson_fallback_audit["nonfinite_pairs"],
                   min_weight=float(matrix.min()), max_weight=float(matrix.max()),
                   seconds=time.monotonic()-t0)
        rows.append(row)
        save_csv(results / "transition_iterations.csv", rows)
        print(json.dumps(row), flush=True)
        print(matrix, flush=True)
    path = data / "transition_matrix.txt"
    temp = path.with_name(path.name + ".tmp")
    np.savetxt(temp, matrix, fmt="%.17g")
    temp.replace(path)
    outputs.extend([path, results / "transition_iterations.csv"])
    checkpoint(marker, fingerprint, outputs, iterations=len(rows), seconds=time.monotonic()-started,
               total_constant_pair_fallbacks=sum(row["constant_pairs"] for row in rows),
               total_nonfinite_pair_fallbacks=sum(row["nonfinite_pairs"] for row in rows))


def walks(data, results, fingerprint):
    marker = results / "walks_complete.json"
    if completed(marker, fingerprint):
        return
    require_stage("prepare", results, fingerprint)
    require_stage("transition", results, fingerprint)
    import numpy as np
    from compat import edge2vec as official
    seed_sampling()
    graph = load_graph(data)
    matrix = np.loadtxt(data / "transition_matrix.txt")
    nodes = list(graph.nodes())
    path = data / "walks.txt"
    temp = path.with_name(path.name + ".tmp")
    count, tokens, started = 0, 0, time.monotonic()
    with temp.open("w", encoding="utf-8") as stream:
        for iteration in range(CONFIG["num_walks"]):
            random.shuffle(nodes)
            for node in nodes:
                route = official.edge2vec_walk(graph, CONFIG["walk_length"], node,
                                               matrix, CONFIG["directed"], CONFIG["p"], CONFIG["q"])
                stream.write(" ".join(route) + "\n")
                count += 1
                tokens += len(route)
                if count % 5000 == 0:
                    print(f"Walks: {count:,}/2,995,900; {time.monotonic()-started:.1f}s", flush=True)
            print(f"Walk round {iteration + 1}/{CONFIG['num_walks']}; {count:,} walks; {time.monotonic()-started:.1f}s", flush=True)
    if count != 2995900 or tokens != count * CONFIG["walk_length"]:
        raise RuntimeError(f"Unexpected MSI walk corpus dimensions: {count:,} walks, {tokens:,} tokens")
    temp.replace(path)
    checkpoint(marker, fingerprint, [path], walks=count, tokens=tokens,
               nodes=graph.number_of_nodes(), seconds=time.monotonic()-started)


def embedding(data, results, fingerprint):
    path = data / "msi_edge2vec_seed0.pkl"
    marker = results / "embedding_complete.json"
    if completed(marker, fingerprint):
        return
    if path.exists():
        raise RuntimeError("Embedding exists without completion marker; preserved. Choose a new RUN_NAME.")
    require_stage("prepare", results, fingerprint)
    require_stage("walks", results, fingerprint)
    import numpy as np
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    nodes = json.loads((data / "nodes.json").read_text())
    started = time.monotonic()
    kwargs = {key: CONFIG[key] for key in (
        "vector_size", "window", "workers", "epochs", "sg", "hs", "negative", "sample",
        "alpha", "min_alpha", "ns_exponent", "min_count", "sorted_vocab", "batch_words",
        "shrink_windows", "seed")}
    model = Word2Vec(sentences=LineSentence(str(data / "walks.txt")), **kwargs)
    vectors = {node: model.wv[str(index)].copy() for index, node in enumerate(nodes)}
    if len(vectors) != 29959 or any(vector.shape != (128,) or not np.isfinite(vector).all()
                                    for vector in vectors.values()):
        raise RuntimeError("Incomplete embedding output")
    atomic_pickle(path, vectors)
    checkpoint(marker, fingerprint, [path], nodes=len(vectors), dimensions=128,
               seconds=time.monotonic()-started)
    print(f"Embedding saved before evaluation: {path}", flush=True)


def evaluate(data, results, fingerprint):
    marker = results / "evaluation_complete.json"
    if completed(marker, fingerprint):
        return
    require_stage("embedding", results, fingerprint)
    import numpy as np
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    with (data / "msi_edge2vec_seed0.pkl").open("rb") as stream:
        vectors = pickle.load(stream)
    positive_set = {(row["drug"], row["disease"]) for row in read_rows(INPUTS["positives"])}
    negatives = [(row["drug"], row["disease"]) for row in read_rows(INPUTS["negatives"], ",")]
    if len(positive_set) != 5926 or len(negatives) != 5926 or len(set(negatives)) != 5926 or positive_set & set(negatives):
        raise ValueError("Unexpected saved positive/negative pair set")
    pairs = sorted(positive_set) + negatives
    x = np.asarray([vectors[a] - vectors[b] for a, b in pairs], dtype=np.float32)
    y = np.asarray([1] * 5926 + [0] * 5926, dtype=np.int8)
    splits = list(StratifiedKFold(**CONFIG["cv"]).split(x, y))
    assignments = np.empty(len(pairs), dtype=np.int32)
    for fold, (_, test) in enumerate(splits, 1):
        assignments[test] = fold
    save_csv(results / "fold_assignments.csv", [
        dict(pair_index=index, drug=a, disease=b, label=int(y[index]), test_fold=int(assignments[index]))
        for index, (a, b) in enumerate(pairs)])
    rows, predictions, started = [], [], time.monotonic()
    for fold, (train, test) in enumerate(splits, 1):
        t0 = time.monotonic()
        classifier = LogisticRegression(**CONFIG["classifier"])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", ConvergenceWarning)
            classifier.fit(x[train], y[train])
        messages = [str(item.message) for item in captured]
        for message in messages:
            print(f"Fold {fold} warning: {message}", flush=True)
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in captured)
        probabilities = classifier.predict_proba(x[test])[:, 1]
        row = dict(seed=0, fold=fold, accuracy=float(accuracy_score(y[test], probabilities >= 0.5)),
                   auroc=float(roc_auc_score(y[test], probabilities)),
                   aupr=float(average_precision_score(y[test], probabilities)),
                   n_train=len(train), n_test=len(test), converged=converged,
                   iterations=int(classifier.n_iter_.max()), warnings=" | ".join(messages),
                   seconds=time.monotonic()-t0)
        rows.append(row)
        predictions.extend(dict(pair_index=int(index), fold=fold, label=int(y[index]), probability=float(probability))
                           for index, probability in zip(test, probabilities))
        save_csv(results / "fold_results.csv", rows)
        save_csv(results / "oof_predictions.csv", predictions)
        print(json.dumps(row), flush=True)
    summary = []
    for metric in ("accuracy", "auroc", "aupr"):
        values = [row[metric] for row in rows]
        summary.append(dict(metric=metric, mean=statistics.mean(values), SD=statistics.stdev(values),
                            n_folds=10, n_repeats=1,
                            all_folds_converged=all(row["converged"] for row in rows)))
    path = results / "summary.csv"
    save_csv(path, summary)
    checkpoint(marker, fingerprint, [path, results / "fold_assignments.csv", results / "fold_results.csv",
                                     results / "oof_predictions.csv"],
               seconds=time.monotonic()-started,
               all_folds_converged=all(row["converged"] for row in rows))
    print("Evaluation complete\n" + json.dumps(summary, indent=2), flush=True)


def compare(data, results, fingerprint):
    marker = results / "comparison_complete.json"
    if completed(marker, fingerprint):
        return
    require_stage("evaluation", results, fingerprint)
    local_rows = read_rows(results / "summary.csv", ",")
    reference_path = ROOT / "reference/paper_edge2vec_fig2a_summary.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Paper comparison table missing: {reference_path}")
    reference = {row["metric"]: row for row in read_rows(reference_path, ",")}
    output = []
    for local in local_rows:
        paper = reference[local["metric"]]
        output.append(dict(metric=local["metric"], local_seed0_10fold_mean=float(local["mean"]),
                           paper_10repeat_mean=float(paper["mean"]),
                           local_minus_paper=float(local["mean"])-float(paper["mean"]),
                           paper_SD_between_repeats=float(paper["SD"]),
                           local_SD_between_folds=float(local["SD"]),
                           all_folds_converged=local["all_folds_converged"]))
    path = results / "paper_comparison.csv"
    save_csv(path, output)
    checkpoint(marker, fingerprint, [path], reference_sha256=digest(reference_path))
    print(json.dumps(output, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "prepare", "transition", "walks", "embedding", "evaluate", "compare"))
    parser.add_argument("--run-name", default="msi_seed0")
    args = parser.parse_args()
    data, results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    else:
        globals()[args.phase](data, results, fingerprint)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, FileNotFoundError) as error:
        print(f"edge2vec phase failed: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
