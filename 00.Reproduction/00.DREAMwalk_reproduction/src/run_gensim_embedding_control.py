"""Evaluate the saved edge2vec Gensim embedding with DREAMwalk's evaluator.

This control compares the complete embedding stages on the same real-node
walk sequences, including their different negative-sampling strategies.
Only config, evaluation and comparison are available; no training on import.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_edge2vec_walks_control as crosswalk

previous = crosswalk.previous
paired = crosswalk.paired
read_json = crosswalk.read_json
require_equal = crosswalk.require_equal
sha256 = previous.sha256
save_json = crosswalk.save_json
RUN_SETTINGS = dict(seed=0, dimension=128, folds=10, cv_seed=0,
                    source_walks=2995900, source_nodes=29959)
EDGE_ROOT = ROOT.parent / "03.edge2vec_reproduction"
EDGE_DATA = EDGE_ROOT / "data/msi_seed0"
EDGE_RESULTS = EDGE_ROOT / "results/msi_seed0"
SOURCE_EMBEDDING = EDGE_DATA / "msi_edge2vec_seed0.pkl"
REFERENCE_NAME = "msi_seed0_edge2vec_walks_pp1"
REFERENCE_DATA = ROOT / "data/embeddings" / REFERENCE_NAME
REFERENCE_RESULTS = ROOT / "results" / REFERENCE_NAME
METRICS = ("accuracy", "auroc", "aupr")


def inspect_sources():
    """Read-only binding of the two embeddings to their common source walks."""
    edge_config = read_json(EDGE_RESULTS / "run_config.json")
    edge_walks = read_json(EDGE_RESULTS / "walks_complete.json")
    edge_prepare = read_json(EDGE_RESULTS / "prepare_complete.json")
    edge_embedding = read_json(EDGE_RESULTS / "embedding_complete.json")
    ref_config = read_json(REFERENCE_RESULTS / "run_config.json")
    ref_walks = read_json(REFERENCE_RESULTS / "walks_complete.json")
    ref_embedding = read_json(REFERENCE_RESULTS / "pp1_embedding_complete.json")
    ref_evaluation = read_json(REFERENCE_RESULTS / "pp1_evaluation_complete.json")
    for marker in (edge_walks, edge_prepare, edge_embedding):
        require_equal(marker["fingerprint"], edge_config["fingerprint"], "Gensim source checkpoint")
    for marker in (ref_walks, ref_embedding, ref_evaluation):
        require_equal(marker["fingerprint"], ref_config["fingerprint"], "C++ reference checkpoint")
    paths = dict(graph=previous.GRAPH, nodetypes=previous.NODETYPES,
                 positives=previous.POSITIVES, negatives=previous.NEGATIVES)
    inputs = {key: sha256(path) for key, path in paths.items()}
    for key, digest in inputs.items():
        require_equal(digest, edge_config["inputs"][key], "Gensim source " + key)
        require_equal(digest, ref_config["sources"]["inputs"][key], "C++ source " + key)
    require_equal(ref_config["sources"]["edge_fingerprint"], edge_config["fingerprint"], "Walk source identity")
    artifact_paths = dict(gensim_embedding=SOURCE_EMBEDDING,
                          edge_walks=EDGE_DATA / "walks.txt", edge_nodes=EDGE_DATA / "nodes.json",
                          cpp_walks=REFERENCE_DATA / "walks.txt",
                          cpp_summary=REFERENCE_RESULTS / "pp1_summary.csv",
                          cpp_folds=REFERENCE_RESULTS / "fold_assignments.csv",
                          cpp_fold_results=REFERENCE_RESULTS / "pp1_fold_results.csv")
    artifacts = {key: sha256(path) for key, path in artifact_paths.items()}
    require_equal(artifacts["gensim_embedding"],
                  edge_embedding["outputs_sha256"]["data/msi_seed0/msi_edge2vec_seed0.pkl"], "Saved Gensim embedding")
    require_equal(artifacts["edge_walks"], edge_walks["outputs_sha256"]["data/msi_seed0/walks.txt"], "Saved edge2vec walks")
    require_equal(artifacts["edge_nodes"], edge_prepare["outputs_sha256"]["data/msi_seed0/nodes.json"], "Saved edge2vec node map")
    require_equal(ref_walks["source_corpus_sha256"], artifacts["edge_walks"], "C++ real-node walk source")
    require_equal(ref_walks["source_nodes_sha256"], artifacts["edge_nodes"], "C++ walk source node map")
    require_equal(artifacts["cpp_walks"], ref_walks["sha256"], "Converted C++ corpus")
    require_equal(artifacts["cpp_walks"], ref_embedding["corpus_sha256"], "C++ embedding corpus")
    require_equal(artifacts["cpp_summary"], ref_evaluation["sha256"], "C++ summary")
    require_equal(artifacts["cpp_folds"], ref_evaluation["fold_assignments_sha256"], "C++ fold IDs")
    require_equal(artifacts["cpp_fold_results"], ref_evaluation["fold_results_sha256"], "C++ fold results")
    require_equal(ref_embedding["pp"], 1, "C++ reference pp")
    require_equal(ref_config["xgboost"], previous.xgb_settings(), "XGBoost parameters")
    require_equal(ref_config["feature"], "drug_embedding - disease_embedding", "Pair feature")
    require_equal(ref_config["settings"]["folds"], RUN_SETTINGS["folds"], "CV folds")
    require_equal(ref_config["settings"]["cv_seed"], RUN_SETTINGS["cv_seed"], "CV seed")
    require_equal(sha256(Path(crosswalk.__file__)), ref_config["runner_sha256"], "Completed C++ control code")
    for marker in (edge_embedding, ref_embedding):
        require_equal(marker["nodes"], RUN_SETTINGS["source_nodes"], "Embedding node count")
        require_equal(marker["dimensions"], RUN_SETTINGS["dimension"], "Embedding dimension")
    require_equal(edge_walks["walks"], RUN_SETTINGS["source_walks"], "Source walk count")
    # Scientific evaluation dependencies remain the same as in the completed
    # DREAMwalk reference. Importing Gensim is neither required nor attempted.
    original_config = read_json(crosswalk.REFERENCE_RESULTS / "run_config.json")
    for relative, digest in original_config["sources"].items():
        require_equal(sha256(ROOT / relative), digest, relative)
    sg_keys = ("vector_size", "window", "workers", "epochs", "sg", "hs", "negative",
               "sample", "alpha", "min_alpha", "ns_exponent", "min_count", "sorted_vocab",
               "batch_words", "shrink_windows", "seed")
    records = dict(edge_config=EDGE_RESULTS / "run_config.json",
                   edge_walks=EDGE_RESULTS / "walks_complete.json",
                   edge_prepare=EDGE_RESULTS / "prepare_complete.json",
                   edge_embedding=EDGE_RESULTS / "embedding_complete.json",
                   cpp_config=REFERENCE_RESULTS / "run_config.json",
                   cpp_walks=REFERENCE_RESULTS / "walks_complete.json",
                   cpp_embedding=REFERENCE_RESULTS / "pp1_embedding_complete.json",
                   cpp_evaluation=REFERENCE_RESULTS / "pp1_evaluation_complete.json")
    return dict(inputs=inputs, artifacts=artifacts,
                records={key: sha256(path) for key, path in records.items()},
                source_paths={key: str(path.relative_to(ROOT.parent)) for key, path in artifact_paths.items()},
                gensim_settings={key: edge_config["config"][key] for key in sg_keys},
                cpp_skipgram_args=ref_config["skipgram_args"],
                edge_fingerprint=edge_config["fingerprint"], cpp_fingerprint=ref_config["fingerprint"])


def context(run_name):
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                           for c in run_name):
        raise ValueError("RUN_NAME must contain only letters, digits, underscore or hyphen")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Use the existing DREAMwalk Python 3.8 interpreter")
    packages = {name: version(name) for name in paired.PINS}
    require_equal(packages, paired.PINS, "DREAMwalk environment")
    commit = subprocess.check_output(["git", "-C", str(paired.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    require_equal(commit, paired.PINNED_COMMIT, "Upstream commit")
    identity = dict(settings=RUN_SETTINGS, xgboost=previous.xgb_settings(),
                    feature="drug_embedding - disease_embedding", sources=inspect_sources(),
                    runner_sha256=sha256(Path(__file__)), packages=packages, upstream_commit=commit)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    results = ROOT / "results" / run_name
    config = results / "run_config.json"
    if config.exists():
        require_equal(read_json(config)["fingerprint"], fingerprint,
                      "Run identity; preserve outputs and use a new RUN_NAME if inputs/code changed")
    else:
        if results.exists() and any(results.iterdir()):
            raise RuntimeError("Unregistered results exist; preserve them and use a new RUN_NAME")
        results.mkdir(parents=True, exist_ok=True)
        save_json(config, dict(identity, fingerprint=fingerprint,
                  analysis="DIAGNOSTIC EMBEDDING-STAGE CONTROL; not a paper benchmark",
                  changed_stage="saved Gensim SG replaces C++ heterogeneous SG pp=1; same real-node walks",
                  negative_sampling=dict(gensim="global frequency-based NS",
                                         cpp_reference="released heterogeneous pp=1 branch"),
                  pair_order="sorted positives + original saved seed-0 negative order",
                  fold_origin="deterministically regenerated and checked against saved C++ reference IDs",
                  notes=["Reuse the saved pickle directly, with its original real-node keys and vector scale.",
                         "No new walks, Skip-gram training, embedding copy or Gensim import.",
                         "This compares embedding implementations and NS strategies as a whole.",
                         "Gensim has no pp flag; the formal DREAMwalk pp=1 setting is retained.",
                         "No PYTHONHASHSEED or multiprocessing start-method override."],
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return results, fingerprint


def pair_fold_rows(positives, negatives):
    """Same ordered pairs and deterministic ten folds as the existing evaluator."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    if len(positives) != len(negatives) or len(negatives) != len(set(negatives)) or set(negatives) & set(positives):
        raise ValueError("Expected equal-sized disjoint positive/negative sets with unique negatives")
    pairs = sorted(positives) + negatives
    labels = np.asarray([1]*len(positives) + [0]*len(negatives))
    assignments = np.empty(len(pairs), dtype=np.int32)
    for fold, (_, test) in enumerate(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(
            np.zeros(len(pairs)), labels), 1):
        assignments[test] = fold
    return [dict(pair_index=i, drug=a, disease=b, label=int(labels[i]), test_fold=int(assignments[i]))
            for i, (a, b) in enumerate(pairs)]


def require_saved_folds(rows):
    require_equal([{key: str(value) for key, value in row.items()} for row in rows],
                  previous.read_rows(REFERENCE_RESULTS / "fold_assignments.csv", ","), "Pair/fold IDs")


def load_source_vectors(types, pairs):
    vectors = previous.load_embedding(SOURCE_EMBEDDING, types, pairs)
    # Some real gene IDs are numeric strings. Never reinterpret these pickle
    # keys as the numeric walk tokens; the source already resolved those IDs.
    require_equal(set(vectors), set(types), "Embedding node universe")
    return vectors


def evaluate(results, fingerprint):
    output, marker = results / "summary.csv", results / "evaluation_complete.json"
    if paired.completed(output, marker, fingerprint):
        require_evaluation(results, fingerprint)
        return
    if output.exists():
        raise RuntimeError("Summary exists without a completion marker; preserve it for recovery")
    types, _, _, positives = previous.base_inputs()
    negatives = [(r["drug"], r["disease"]) for r in previous.read_rows(previous.NEGATIVES, ",")]
    rows = pair_fold_rows(positives, negatives)
    require_saved_folds(rows)
    fold_path = results / "fold_assignments.csv"
    if fold_path.exists():
        require_equal(previous.read_rows(fold_path, ","),
                      [{key: str(value) for key, value in row.items()} for row in rows], "Existing fold IDs")
    else:
        previous.write_csv(fold_path, rows, list(rows[0]))
    require_equal(sha256(fold_path), sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Fold CSV bytes")
    vectors = load_source_vectors(types, sorted(positives) + negatives)
    started = time.monotonic()
    print("Evaluating saved Gensim vectors with the unchanged DREAMwalk XGBoost evaluator", flush=True)
    folds = previous.evaluate(vectors, positives, negatives, 0, "no_teleport")
    for row in folds:
        row.update(label="edge2vec_gensim_xgb", embedding="saved_edge2vec_Gensim_SG")
    summary = previous.metric_summary(folds, 0, "edge2vec_gensim_xgb")
    fold_results = results / "fold_results.csv"
    previous.write_csv(fold_results, folds, list(folds[0]))
    previous.write_csv(output, summary, list(summary[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output),
              source_embedding_sha256=sha256(SOURCE_EMBEDDING),
              fold_assignments_sha256=sha256(fold_path), fold_results_sha256=sha256(fold_results),
              seconds=time.monotonic()-started))


def require_evaluation(results, fingerprint):
    meta = paired.require_checkpoint(results / "summary.csv", results / "evaluation_complete.json", fingerprint)
    require_equal(meta["source_embedding_sha256"], sha256(SOURCE_EMBEDDING), "Evaluated embedding")
    require_equal(meta["fold_assignments_sha256"], sha256(results / "fold_assignments.csv"), "Saved fold IDs")
    require_equal(meta["fold_assignments_sha256"], sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Paired fold IDs")
    require_equal(meta["fold_results_sha256"], sha256(results / "fold_results.csv"), "Saved fold metrics")
    return meta


def comparison_rows(reference, current, reference_folds, current_folds):
    # Reuse the established metric/fold alignment and arithmetic, then give the
    # columns names appropriate to the embedding-stage control.
    old_summary, old_deltas = crosswalk.comparison_rows(reference, current, reference_folds, current_folds)

    def rename(row):
        result = {"fold": row["fold"]} if "fold" in row else {}
        result.update(metric=row["metric"], cpp_pp1=row["dreamwalk_walks_pp1"],
                      gensim_sg=row["edge2vec_walks_pp1"],
                      gensim_minus_cpp_pp1=row["edge2vec_minus_dreamwalk"])
        return result

    return [rename(row) for row in old_summary], [rename(row) for row in old_deltas]


def compare(results, fingerprint):
    require_evaluation(results, fingerprint)
    output, marker = results / "embedding_trainer_comparison.csv", results / "comparison_complete.json"
    if paired.completed(output, marker, fingerprint):
        require_equal(sha256(results / "paired_fold_deltas.csv"), read_json(marker)["paired_fold_deltas_sha256"],
                      "Paired fold deltas")
        return
    summary, deltas = comparison_rows(
        previous.read_rows(REFERENCE_RESULTS / "pp1_summary.csv", ","),
        previous.read_rows(results / "summary.csv", ","),
        previous.read_rows(REFERENCE_RESULTS / "pp1_fold_results.csv", ","),
        previous.read_rows(results / "fold_results.csv", ","))
    previous.write_csv(output, summary, list(summary[0]))
    delta_path = results / "paired_fold_deltas.csv"
    previous.write_csv(delta_path, deltas, list(deltas[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output),
              paired_fold_deltas_sha256=sha256(delta_path),
              fold_assignments_sha256=sha256(results / "fold_assignments.csv"), reference_run=REFERENCE_NAME))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "evaluate", "compare"))
    parser.add_argument("--run-name", default="msi_seed0_edge2vec_gensim_xgb")
    args = parser.parse_args()
    results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    elif args.phase == "evaluate":
        evaluate(results, fingerprint)
    else:
        compare(results, fingerprint)


if __name__ == "__main__":
    main()
