"""User-run control: saved edge2vec walks -> DREAMwalk pp=1 -> XGBoost.

Replace the whole walk-generation stage, retaining the completed DREAMwalk
pp=1 control's binary, node-token bijection, SG arguments and evaluation.
No walks, training or evaluation run on import.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_skipgram_pp_comparison as paired

previous = paired.previous
save_json = paired.save_json
sha256 = previous.sha256
RUN_SETTINGS = dict(pp=1, seed=0, num_walks=100, walk_length=10,
                    dimension=128, window_size=4, workers=4, epochs=1,
                    folds=10, cv_seed=0, source_walks=2995900, source_nodes=29959)
REFERENCE_NAME = "msi_seed0_skipgram_pp_comparison"
REFERENCE_DATA = ROOT / "data/embeddings" / REFERENCE_NAME
REFERENCE_RESULTS = ROOT / "results" / REFERENCE_NAME
EDGE_ROOT = ROOT.parent / "03.edge2vec_reproduction"
EDGE_DATA = EDGE_ROOT / "data/msi_seed0"
EDGE_RESULTS = EDGE_ROOT / "results/msi_seed0"
METRICS = ("accuracy", "auroc", "aupr")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual, expected, description):
    if actual != expected:
        raise RuntimeError(description + " differs from the saved control")


def inspect_sources():
    """Read-only audit of the artifacts actually reused by this control."""
    edge_config = read_json(EDGE_RESULTS / "run_config.json")
    ref_config = read_json(REFERENCE_RESULTS / "run_config.json")
    edge_prepare = read_json(EDGE_RESULTS / "prepare_complete.json")
    edge_walks = read_json(EDGE_RESULTS / "walks_complete.json")
    ref_walks = read_json(REFERENCE_RESULTS / "walks_complete.json")
    ref_compiler = read_json(REFERENCE_RESULTS / "compiler_complete.json")
    ref_embedding = read_json(REFERENCE_RESULTS / "pp1_embedding_complete.json")
    ref_evaluation = read_json(REFERENCE_RESULTS / "pp1_evaluation_complete.json")
    for marker in (edge_prepare, edge_walks):
        require_equal(marker["fingerprint"], edge_config["fingerprint"], "edge2vec checkpoint")
    for marker in (ref_walks, ref_compiler, ref_embedding, ref_evaluation):
        require_equal(marker["fingerprint"], ref_config["fingerprint"], "DREAMwalk checkpoint")
    inputs = dict(graph=previous.GRAPH, nodetypes=previous.NODETYPES,
                  positives=previous.POSITIVES, negatives=previous.NEGATIVES)
    input_hashes = {key: sha256(path) for key, path in inputs.items()}
    for key, digest in input_hashes.items():
        require_equal(digest, edge_config["inputs"][key], "edge2vec " + key)
        require_equal(digest, ref_config["inputs"][key], "DREAMwalk " + key)
    for key, value in dict(seed=0, p=1, q=1, num_walks=100, walk_length=10,
                           em_iterations=5, em_walk_length=10).items():
        require_equal(edge_config["config"][key], value, "edge2vec " + key)
    require_equal(ref_config["walks"], paired.WALKS, "Reference walk recipe")
    require_equal(ref_config["skipgram_args"], paired.SG_ARGS, "Skip-gram arguments")
    require_equal(ref_config["xgboost"], previous.xgb_settings(), "XGBoost settings")
    require_equal(ref_config["folds"], 10, "CV folds")
    require_equal(ref_config["cv_seed"], 0, "CV seed")
    require_equal(ref_embedding["pp"], 1, "Reference pp")
    require_equal(ref_embedding["command"],
                  ["./HetSG", "-train", "walks.txt", "-output", "pp1", "-pp", "1"] + paired.SG_ARGS,
                  "Reference training command")
    for marker, keys in ((edge_walks, dict(walks=2995900, tokens=29959000, nodes=29959)),
                         (ref_walks, dict(walks=2995900, walk_length=10, nodes=29959))):
        for key, value in keys.items():
            require_equal(marker[key], value, "Source " + key)
    artifacts = {
        "edge_walks": EDGE_DATA / "walks.txt", "edge_nodes": EDGE_DATA / "nodes.json",
        "token_map": REFERENCE_DATA / "token_to_node.json", "binary": REFERENCE_DATA / "HetSG",
        "reference_summary": REFERENCE_RESULTS / "pp1_summary.csv",
        "reference_folds": REFERENCE_RESULTS / "fold_assignments.csv",
        "reference_fold_results": REFERENCE_RESULTS / "pp1_fold_results.csv",
    }
    artifact_hashes = {key: sha256(path) for key, path in artifacts.items()}
    require_equal(artifact_hashes["edge_walks"],
                  edge_walks["outputs_sha256"]["data/msi_seed0/walks.txt"], "Source corpus")
    require_equal(artifact_hashes["edge_nodes"],
                  edge_prepare["outputs_sha256"]["data/msi_seed0/nodes.json"], "Source node map")
    require_equal(artifact_hashes["binary"], ref_compiler["sha256"], "Reference binary")
    require_equal(artifact_hashes["binary"], ref_embedding["binary_sha256"], "Embedding binary")
    require_equal(artifact_hashes["token_map"], ref_walks["token_map_sha256"], "Reference map")
    require_equal(artifact_hashes["token_map"], ref_embedding["token_map_sha256"], "Embedding map")
    require_equal(ref_walks["sha256"], ref_embedding["corpus_sha256"], "Reference corpus provenance")
    require_equal(artifact_hashes["reference_summary"], ref_evaluation["sha256"], "Reference summary")
    require_equal(artifact_hashes["reference_folds"], ref_evaluation["fold_assignments_sha256"],
                  "Reference fold assignments")
    # Reused scientific functions and upstream sources must still match the
    # completed reference. The new runner itself is bound below, separately.
    for relative, digest in ref_config["sources"].items():
        require_equal(sha256(ROOT / relative), digest, relative)
    records = {"edge_config": EDGE_RESULTS / "run_config.json",
               "edge_prepare": EDGE_RESULTS / "prepare_complete.json",
               "edge_walks_record": EDGE_RESULTS / "walks_complete.json",
               "reference_config": REFERENCE_RESULTS / "run_config.json",
               "reference_walks_record": REFERENCE_RESULTS / "walks_complete.json",
               "reference_compiler": REFERENCE_RESULTS / "compiler_complete.json",
               "reference_embedding": REFERENCE_RESULTS / "pp1_embedding_complete.json",
               "reference_evaluation": REFERENCE_RESULTS / "pp1_evaluation_complete.json"}
    return dict(inputs=input_hashes, artifacts=artifact_hashes,
                records={key: sha256(path) for key, path in records.items()},
                source_paths={key: str(path.relative_to(ROOT.parent))
                              for key, path in dict(artifacts, **records).items()},
                edge_fingerprint=edge_config["fingerprint"],
                reference_fingerprint=ref_config["fingerprint"])


def context(run_name):
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                           for c in run_name):
        raise ValueError("RUN_NAME must contain only letters, digits, underscore or hyphen")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Use the existing DREAMwalk Python 3.8 interpreter")
    packages = {name: version(name) for name in paired.PINS}
    require_equal(packages, paired.PINS, "DREAMwalk environment")
    commit = subprocess.check_output(
        ["git", "-C", str(paired.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    require_equal(commit, paired.PINNED_COMMIT, "Upstream commit")
    identity = dict(settings=RUN_SETTINGS, skipgram_args=paired.SG_ARGS,
                    xgboost=previous.xgb_settings(),
                    feature="drug_embedding - disease_embedding", sources=inspect_sources(),
                    runner_sha256=sha256(Path(__file__)), packages=packages, upstream_commit=commit)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    data, results = ROOT / "data/embeddings" / run_name, ROOT / "results" / run_name
    config = results / "run_config.json"
    if config.exists():
        require_equal(read_json(config)["fingerprint"], fingerprint,
                      "Run identity; preserve outputs and use a new RUN_NAME if inputs/code changed")
    else:
        if (data.exists() and any(data.iterdir())) or (results.exists() and any(results.iterdir())):
            raise RuntimeError("Unregistered outputs exist; preserve them and use a new RUN_NAME")
        data.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        save_json(config, dict(identity, fingerprint=fingerprint,
                  analysis="DIAGNOSTIC WALK-SOURCE CONTROL; not a paper benchmark",
                  changed_stage="saved edge2vec EM + walks replace saved DREAMwalk EM + walks",
                  pair_order="sorted positives + original saved seed-0 negative order",
                  fold_origin="deterministically regenerated and checked against saved reference IDs",
                  notes=["Numeric tokens are recoded using the identical historical DREAMwalk node bijection.",
                         "Identical C++ binary and pp=1 arguments; no new EM, walks or negative pairs.",
                         "Corpus-dependent vocabulary order, frequencies and SG negative pools are rebuilt.",
                         "Four C++ threads retain their existing nondeterminism.",
                         "No PYTHONHASHSEED or multiprocessing start-method override."],
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def conversion_map(nodes, token_to_node, types):
    """Numeric edge2vec index -> historical type-prefixed DREAMwalk token."""
    if not isinstance(nodes, list) or len(nodes) != len(set(nodes)):
        raise ValueError("edge2vec nodes.json must be a unique ordered list")
    if len(set(token_to_node.values())) != len(token_to_node):
        raise ValueError("DREAMwalk token map is not a bijection")
    if set(nodes) != set(token_to_node.values()) or set(nodes) != set(types):
        raise ValueError("Node universes differ")
    prefixes = {"drug": "d", "disease": "i", "gene": "g", "etc": "e"}
    for token, node in token_to_node.items():
        if types[node] not in prefixes or not token.startswith(prefixes[types[node]]) or not token[1:].isdigit():
            raise ValueError("Incorrect node-type prefix: " + token)
    inverse = {node: token for token, node in token_to_node.items()}
    return {str(i): inverse[node] for i, node in enumerate(nodes)}


def recode_stream(source, output, mapping, walk_length, expected_walks, walks_per_node):
    """Preserve each walk; match historical serialization (no final newline)."""
    count = 0
    seen, starts = set(), Counter()
    for line in source:
        fields = line.split()
        if len(fields) != walk_length:
            raise ValueError("Unexpected walk length at line " + str(count + 1))
        try:
            converted = [mapping[token] for token in fields]
        except KeyError as error:
            raise ValueError("Unknown numeric node token: " + str(error)) from error
        if count:
            output.write("\n")
        output.write(" ".join(converted))
        starts[fields[0]] += 1
        seen.update(fields)
        count += 1
    if count != expected_walks or seen != set(mapping):
        raise ValueError("Unexpected corpus size or missing nodes")
    if set(starts) != set(mapping) or any(n != walks_per_node for n in starts.values()):
        raise ValueError("Unexpected number of walks starting at each node")
    return dict(walks=count, tokens=count * walk_length, nodes=len(seen), walk_length=walk_length)


def copy_exact(source, target):
    if target.exists():
        require_equal(sha256(target), sha256(source), "Existing copied artifact")
        return
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    require_equal(sha256(temporary), sha256(source), "Copied artifact")
    temporary.replace(target)


def prepare(data, results, fingerprint):
    started = time.monotonic()
    copy_exact(REFERENCE_DATA / "token_to_node.json", data / "token_to_node.json")
    copy_exact(REFERENCE_DATA / "HetSG", data / "HetSG")
    compiler_marker = results / "compiler_complete.json"
    if not paired.completed(data / "HetSG", compiler_marker, fingerprint):
        save_json(compiler_marker, dict(fingerprint=fingerprint, sha256=sha256(data / "HetSG"),
                  reused_from=str((REFERENCE_DATA / "HetSG").relative_to(ROOT)),
                  original_compiler=read_json(REFERENCE_RESULTS / "compiler_complete.json")))
    corpus, marker = data / "walks.txt", results / "walks_complete.json"
    if paired.completed(corpus, marker, fingerprint):
        require_equal(sha256(data / "token_to_node.json"), read_json(marker)["token_map_sha256"],
                      "Prepared token map")
        return
    if corpus.exists() or (data / "pp1.pkl").exists():
        raise RuntimeError("Unregistered corpus/embedding exists; preserve it for recovery")
    nodes = read_json(EDGE_DATA / "nodes.json")
    types = {row["node"]: row["type"] for row in previous.read_rows(previous.NODETYPES)}
    mapping = conversion_map(nodes, read_json(data / "token_to_node.json"), types)
    require_equal(len(mapping), RUN_SETTINGS["source_nodes"], "Vocabulary size")
    temporary = data / "walks.txt.tmp"
    print("Recoding saved edge2vec walks; no EM or walk generation", flush=True)
    with (EDGE_DATA / "walks.txt").open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
        counts = recode_stream(source, output, mapping, RUN_SETTINGS["walk_length"],
                               RUN_SETTINGS["source_walks"], RUN_SETTINGS["num_walks"])
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(corpus)
    save_json(marker, dict(counts, fingerprint=fingerprint, sha256=sha256(corpus),
              token_map_sha256=sha256(data / "token_to_node.json"),
              source_corpus_sha256=sha256(EDGE_DATA / "walks.txt"),
              source_nodes_sha256=sha256(EDGE_DATA / "nodes.json"),
              serialization="newline between walks; no final newline (historical DREAMwalk format)",
              seconds=time.monotonic()-started))
    print("CONVERTED CORPUS SAVED:", corpus, flush=True)


def evaluate(data, results, fingerprint):
    """Use exactly the existing ten-fold fitting/metrics; label this new corpus."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    output, marker = results / "pp1_summary.csv", results / "pp1_evaluation_complete.json"
    if paired.completed(output, marker, fingerprint):
        meta = read_json(marker)
        require_equal(sha256(results / "fold_assignments.csv"), meta["fold_assignments_sha256"], "Saved folds")
        require_equal(sha256(results / "pp1_fold_results.csv"), meta["fold_results_sha256"], "Saved fold results")
        return
    paired.require_checkpoint(data / "pp1.pkl", results / "pp1_embedding_complete.json", fingerprint)
    types, _, _, positives = previous.base_inputs()
    negatives = [(r["drug"], r["disease"]) for r in previous.read_rows(previous.NEGATIVES, ",")]
    pairs = sorted(positives) + negatives
    labels = np.asarray([1]*len(positives) + [0]*len(negatives))
    assignments = np.empty(len(pairs), dtype=np.int32)
    for fold, (_, test) in enumerate(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(
            np.zeros(len(pairs)), labels), 1):
        assignments[test] = fold
    pair_rows = [dict(pair_index=i, drug=a, disease=b, label=int(labels[i]), test_fold=int(assignments[i]))
                 for i, (a, b) in enumerate(pairs)]
    # Check actual pair IDs and folds BEFORE fitting the classifier.
    expected = previous.read_rows(REFERENCE_RESULTS / "fold_assignments.csv", ",")
    require_equal([{k: str(v) for k, v in row.items()} for row in pair_rows], expected, "Pair/fold assignments")
    fold_path = results / "fold_assignments.csv"
    previous.write_csv(fold_path, pair_rows, list(pair_rows[0]))
    require_equal(sha256(fold_path), sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Fold CSV bytes")
    vectors = previous.load_embedding(data / "pp1.pkl", types, pairs)
    started = time.monotonic()
    rows = previous.evaluate(vectors, positives, negatives, 0, "no_teleport")
    for row in rows:
        row.update(label="edge2vec_walks_pp1", embedding="edge2vec_walks_DREAMwalk_pp1", pp=1)
    summary = previous.metric_summary(rows, 0, "edge2vec_walks_pp1")
    folds_path = results / "pp1_fold_results.csv"
    previous.write_csv(folds_path, rows, list(rows[0]))
    previous.write_csv(output, summary, list(summary[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output), pp=1,
              fold_assignments_sha256=sha256(fold_path), fold_results_sha256=sha256(folds_path),
              seconds=time.monotonic()-started))


def comparison_rows(reference, current, reference_folds, current_folds):
    baseline = {r["metric"]: float(r["mean"]) for r in reference}
    control = {r["metric"]: float(r["mean"]) for r in current}
    require_equal(set(baseline), set(METRICS), "Reference metrics")
    require_equal(set(control), set(METRICS), "Control metrics")
    paper = dict(accuracy=None, auroc=0.905, aupr=0.912)
    summary = [dict(metric=metric, dreamwalk_walks_pp1=baseline[metric],
                    edge2vec_walks_pp1=control[metric],
                    edge2vec_minus_dreamwalk=control[metric]-baseline[metric],
                    paper_no_teleport_approx=paper[metric]) for metric in METRICS]
    by_fold = {int(r["fold"]): r for r in reference_folds}
    current_by_fold = {int(r["fold"]): r for r in current_folds}
    if (len(reference_folds) != 10 or len(current_folds) != 10
            or set(by_fold) != set(range(1, 11)) or set(current_by_fold) != set(by_fold)):
        raise RuntimeError("Expected exactly ten matching folds")
    deltas = []
    for fold in sorted(by_fold):
        for metric in METRICS:
            a, b = float(by_fold[fold][metric]), float(current_by_fold[fold][metric])
            deltas.append(dict(fold=fold, metric=metric, dreamwalk_walks_pp1=a,
                               edge2vec_walks_pp1=b, edge2vec_minus_dreamwalk=b-a))
    return summary, deltas


def compare(data, results, fingerprint):
    output, marker = results / "walk_source_comparison.csv", results / "comparison_complete.json"
    if paired.completed(output, marker, fingerprint):
        require_equal(sha256(results / "paired_fold_deltas.csv"), read_json(marker)["paired_fold_deltas_sha256"],
                      "Paired fold deltas")
        return
    embedding = paired.require_checkpoint(data / "pp1.pkl", results / "pp1_embedding_complete.json", fingerprint)
    evaluation = paired.require_checkpoint(results / "pp1_summary.csv", results / "pp1_evaluation_complete.json", fingerprint)
    original = read_json(REFERENCE_RESULTS / "pp1_embedding_complete.json")
    for key in ("pp", "binary_sha256", "token_map_sha256", "command", "nodes", "dimensions"):
        require_equal(embedding[key], original[key], "Paired embedding " + key)
    fold_hash = sha256(results / "fold_assignments.csv")
    require_equal(fold_hash, evaluation["fold_assignments_sha256"], "Current fold assignments")
    require_equal(fold_hash, sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Paired fold IDs")
    require_equal(sha256(results / "pp1_fold_results.csv"), evaluation["fold_results_sha256"], "Current fold results")
    summary, deltas = comparison_rows(
        previous.read_rows(REFERENCE_RESULTS / "pp1_summary.csv", ","),
        previous.read_rows(results / "pp1_summary.csv", ","),
        previous.read_rows(REFERENCE_RESULTS / "pp1_fold_results.csv", ","),
        previous.read_rows(results / "pp1_fold_results.csv", ","))
    previous.write_csv(output, summary, list(summary[0]))
    delta_path = results / "paired_fold_deltas.csv"
    previous.write_csv(delta_path, deltas, list(deltas[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output),
              paired_fold_deltas_sha256=sha256(delta_path), fold_assignments_sha256=fold_hash,
              reference_run=REFERENCE_NAME, pp=1))
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "prepare", "embedding", "evaluate", "compare"))
    parser.add_argument("--run-name", default="msi_seed0_edge2vec_walks_pp1")
    args = parser.parse_args()
    data, results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    elif args.phase == "prepare":
        prepare(data, results, fingerprint)
    elif args.phase == "embedding":
        paired.train(1, data, results, fingerprint)
    elif args.phase == "evaluate":
        evaluate(data, results, fingerprint)
    else:
        compare(data, results, fingerprint)


if __name__ == "__main__":
    main()
