"""User-started original MSI DREAMwalk with MeSH2021, tau=.3 and pp=1.

Only the MeSH-derived semantic input is rebuilt. The released EM, walker,
teleport selector and C++ negative-sampling implementation remain unchanged.
Importing this module does not build inputs, train embeddings or evaluate.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import multiprocessing
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import msi_mesh2021_inputs as inputs
import run_semantic_pp_comparison as shared

paired = shared.paired
previous = paired.previous
sha256 = previous.sha256
save_json = paired.save_json
read_json = shared.read_json
require_equal = shared.require_equal
WALK_SETTINGS = dict(tp_factor=0.3, seed=0, num_walks=100, walk_length=10,
                     p=1, q=1, workers=4, em_max_iter=5)
SEMANTIC_SETTINGS = dict(mesh_year=2021, cutoff=0.4)
PP = 1
RUN_NAME = "msi_seed0_mesh2021_original"
SOURCE_NODES = 29959
REFERENCE_NAME = "msi_seed0_skipgram_pp_comparison"
REFERENCE_DATA = ROOT / "data/embeddings" / REFERENCE_NAME
REFERENCE_RESULTS = ROOT / "results" / REFERENCE_NAME
MESH2024_RESULTS = ROOT / "results/msi_seed0_semantic_skipgram_pp_comparison"
METRICS = ("accuracy", "auroc", "aupr")


def reference_identity():
    """Check only the historical artifacts needed by this pp=1 experiment."""
    config = read_json(REFERENCE_RESULTS / "run_config.json")
    compiler = read_json(REFERENCE_RESULTS / "compiler_complete.json")
    corpus = read_json(REFERENCE_RESULTS / "walks_complete.json")
    evaluation = read_json(REFERENCE_RESULTS / "pp1_evaluation_complete.json")
    common = dict(graph=previous.GRAPH, nodetypes=previous.NODETYPES,
                  positives=previous.POSITIVES, negatives=previous.NEGATIVES)
    for key, path in common.items():
        require_equal(sha256(path), config["inputs"][key], "Shared " + key)
    require_equal(config["skipgram_args"], paired.SG_ARGS, "Skip-gram arguments")
    require_equal(config["xgboost"], previous.xgb_settings(), "XGBoost settings")
    require_equal(config["folds"], 10, "CV folds")
    require_equal(config["cv_seed"], 0, "CV seed")
    for marker in (compiler, corpus, evaluation):
        require_equal(marker["fingerprint"], config["fingerprint"], "Reference checkpoint")
    artifacts = dict(binary=sha256(REFERENCE_DATA / "HetSG"),
                     token_map=sha256(REFERENCE_DATA / "token_to_node.json"),
                     fold_assignments=sha256(REFERENCE_RESULTS / "fold_assignments.csv"))
    require_equal(artifacts["binary"], compiler["sha256"], "Original C++ binary")
    require_equal(artifacts["token_map"], corpus["token_map_sha256"], "Original token map")
    require_equal(artifacts["fold_assignments"], evaluation["fold_assignments_sha256"], "Saved fold IDs")
    # In particular, do not compare the NEW semantic graph to the historical
    # config's MeSH2024 similarity hash. That is the intended changed input.
    return dict(run_name=REFERENCE_NAME, fingerprint=config["fingerprint"],
                artifacts=artifacts,
                records={name: sha256(REFERENCE_RESULTS / name) for name in
                         ("run_config.json", "compiler_complete.json", "walks_complete.json",
                          "pp1_evaluation_complete.json")})


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
    # Input construction owns its own completion records and verifies their
    # raw inputs and outputs. This read-only check never builds a graph.
    input_identity = inputs.artifact_identity()
    paths = dict(graph=previous.GRAPH, similarity=inputs.SIMILARITY_PATH,
                 hierarchy=inputs.HIERARCHY_PATH, mesh=inputs.MESH_PATH,
                 nodetypes=previous.NODETYPES, positives=previous.POSITIVES,
                 negatives=previous.NEGATIVES)
    sources = [Path(__file__), Path(inputs.__file__), Path(shared.__file__), Path(paired.__file__),
               Path(previous.__file__), ROOT / "src/run_msi_repeat0.py",
               ROOT / "src/run_no_teleport_length4.py",
               paired.UPSTREAM / "DREAMwalk/generate_embeddings.py",
               paired.UPSTREAM / "DREAMwalk/utils.py",
               paired.UPSTREAM / "DREAMwalk/HeterogeneousSG.cpp"]
    identity = dict(walks=WALK_SETTINGS, semantic=SEMANTIC_SETTINGS, pp=PP,
                    skipgram_args=paired.SG_ARGS, xgboost=previous.xgb_settings(),
                    folds=10, cv_seed=0, feature="drug_embedding - disease_embedding",
                    inputs={key: sha256(path) for key, path in paths.items()},
                    semantic_input_provenance=input_identity, reference=reference_identity(),
                    sources={str(path.relative_to(ROOT)): sha256(path) for path in sources},
                    packages=packages, upstream_commit=commit)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    data, results = ROOT / "data/embeddings" / run_name, ROOT / "results" / run_name
    config = results / "run_config.json"
    if config.exists():
        require_equal(read_json(config)["fingerprint"], fingerprint,
                      "Run identity; preserve outputs and use a new RUN_NAME if inputs/code changed")
    else:
        if ((data.exists() and any(data.iterdir())) or (results.exists() and any(results.iterdir()))):
            raise RuntimeError("Unregistered outputs exist; preserve them and use a new RUN_NAME")
        data.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        save_json(config, dict(identity, fingerprint=fingerprint,
                  analysis="ORIGINAL UPSTREAM DREAMwalk; MeSH2021 semantic input",
                  input_paths={key: str(path.relative_to(ROOT)) for key, path in paths.items()},
                  pair_order="sorted positives + original saved seed-0 negative order",
                  fold_origin="deterministically regenerated and checked against saved no-teleport IDs",
                  notes=["MeSH2021 semantic hierarchy/similarity is separate from historical MeSH2024 outputs.",
                         "Molecular graph, ATC, audited disease mapping, seed-0 negatives and CV are reused.",
                         "The exact historical original C++ binary and node-token map are copied.",
                         "Original edge-type EM, teleport selector and pp=1 negative sampling are unchanged.",
                         "Upstream set_seed(0) is unchanged; no hashseed restoration or start-method override.",
                         "Worker Python/NumPy RNG state is not asserted; four C++ threads retain nondeterminism.",
                         "Walks are regenerated for the changed semantic input; cross-year comparison is not a paired corpus.",
                         "Tau=.3 is the accepted local setting, not a claimed Figure2a published value."],
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  multiprocessing_start_method=multiprocessing.get_context().get_start_method(),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def prepare(data, results, fingerprint):
    shared.copy_exact(REFERENCE_DATA / "HetSG", data / "HetSG")
    shared.copy_exact(REFERENCE_DATA / "token_to_node.json", data / "token_to_node.json")
    shared.copy_exact(REFERENCE_RESULTS / "fold_assignments.csv", results / "fold_assignments.csv")
    marker = results / "compiler_complete.json"
    if not paired.completed(data / "HetSG", marker, fingerprint):
        save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(data / "HetSG"),
                  token_map_sha256=sha256(data / "token_to_node.json"),
                  reused_from=str((REFERENCE_DATA / "HetSG").relative_to(ROOT)),
                  original_compiler=read_json(REFERENCE_RESULTS / "compiler_complete.json")))
    print("Original C++ binary, token map and saved fold IDs ready", flush=True)


def generate_corpus(data, results, fingerprint):
    from DREAMwalk import generate_embeddings as ge
    corpus, marker = data / "walks.txt", results / "walks_complete.json"
    if paired.completed(corpus, marker, fingerprint):
        require_equal(sha256(data / "token_to_node.json"), read_json(marker)["token_map_sha256"], "Saved map")
        return
    compiler = paired.require_checkpoint(data / "HetSG", results / "compiler_complete.json", fingerprint)
    map_path = data / "token_to_node.json"
    require_equal(sha256(map_path), compiler["token_map_sha256"], "Prepared map")
    if corpus.exists() or (data / "pp1.pkl").exists() or (results / "pp1_embedding_complete.json").exists():
        raise RuntimeError("Existing corpus/embedding prevents regeneration; preserve outputs and use a new RUN_NAME")
    started = time.monotonic()
    ge.set_seed(WALK_SETTINGS["seed"])
    graph = ge.read_graph(str(previous.GRAPH), weighted=True, directed=False, delimiter="\t")
    semantic = ge.read_graph(str(inputs.SIMILARITY_PATH), weighted=True, directed=False, delimiter="\t")
    if semantic.number_of_edges() == 0:
        raise RuntimeError("MeSH2021 semantic graph is empty")
    types = {row["node"]: row["type"] for row in previous.read_rows(previous.NODETYPES)}
    node_to_token = shared.checked_tokens(read_json(map_path), types, graph.nodes())
    require_equal(len(node_to_token), SOURCE_NODES, "MSI node count")
    t0 = time.monotonic()
    print("Starting unchanged edge-type EM: 5 iterations, length=10", flush=True)
    matrix = ge.train_edgetype_transition_matrix(WALK_SETTINGS["em_max_iter"], graph,
                                                str(previous.GRAPH), "\t", WALK_SETTINGS["walk_length"],
                                                WALK_SETTINGS["p"], WALK_SETTINGS["q"])
    em_seconds = time.monotonic() - t0
    t0 = time.monotonic()
    print("Starting original MeSH2021 walks: tau=.3, 100 x 29,959, length=10", flush=True)
    walks = ge.generate_DREAMwalk_paths(graph, semantic, matrix, WALK_SETTINGS["p"], WALK_SETTINGS["q"],
                                       WALK_SETTINGS["num_walks"], WALK_SETTINGS["walk_length"],
                                       WALK_SETTINGS["tp_factor"], WALK_SETTINGS["workers"])
    walk_seconds = time.monotonic() - t0
    temporary = data / "walks.txt.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        counts = shared.write_corpus(walks, stream, node_to_token, WALK_SETTINGS["walk_length"],
                                     SOURCE_NODES * WALK_SETTINGS["num_walks"], WALK_SETTINGS["num_walks"])
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(corpus)
    save_json(results / "edge_transition_matrix.json", matrix.tolist())
    save_json(marker, dict(counts, fingerprint=fingerprint, sha256=sha256(corpus),
              token_map_sha256=sha256(map_path), tp_factor=WALK_SETTINGS["tp_factor"],
              similarity_sha256=sha256(inputs.SIMILARITY_PATH), similarity_edges=semantic.number_of_edges(),
              serialization="newline between walks; no final newline",
              em_seconds=em_seconds, walk_seconds=walk_seconds, total_seconds=time.monotonic()-started))
    print("MESH2021 CORPUS SAVED:", corpus, flush=True)


def evaluate(data, results, fingerprint):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    output, marker = results / "pp1_summary.csv", results / "pp1_evaluation_complete.json"
    if paired.completed(output, marker, fingerprint):
        require_evaluation(data, results, fingerprint)
        return
    embedding = paired.require_checkpoint(data / "pp1.pkl", results / "pp1_embedding_complete.json", fingerprint)
    types, _, _, positives = previous.base_inputs()
    negatives = [(row["drug"], row["disease"]) for row in previous.read_rows(previous.NEGATIVES, ",")]
    pairs = sorted(positives) + negatives
    labels = np.asarray([1]*len(positives) + [0]*len(negatives))
    assignments = np.empty(len(pairs), dtype=np.int32)
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(pairs)), labels), 1):
        assignments[test] = fold
    pair_rows = [dict(pair_index=i, drug=a, disease=b, label=int(labels[i]), test_fold=int(assignments[i]))
                 for i, (a, b) in enumerate(pairs)]
    expected = previous.read_rows(REFERENCE_RESULTS / "fold_assignments.csv", ",")
    serialized = [{key: str(value) for key, value in row.items()} for row in pair_rows]
    require_equal(serialized, expected, "Pair/fold assignments before fitting")
    fold_path = results / "fold_assignments.csv"
    require_equal(previous.read_rows(fold_path, ","), expected, "Prepared pair/fold assignments")
    require_equal(sha256(fold_path), sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Fold CSV bytes")
    vectors = previous.load_embedding(data / "pp1.pkl", types, pairs)
    started = time.monotonic()
    rows = previous.evaluate(vectors, positives, negatives, 0, "mesh2021_original_pp1")
    for row in rows:
        row.update(analysis="ORIGINAL UPSTREAM DREAMwalk; MeSH2021 semantic input",
                   label="mesh2021_original_pp1", embedding="mesh2021_tau_0.3_pp1", pp=PP)
    summary = previous.metric_summary(rows, 0, "mesh2021_original_pp1")
    for row in summary:
        row["analysis"] = "ORIGINAL UPSTREAM DREAMwalk; MeSH2021 semantic input"
    fold_results = results / "pp1_fold_results.csv"
    previous.write_csv(fold_results, rows, list(rows[0]))
    previous.write_csv(output, summary, list(summary[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output), pp=PP,
              embedding_sha256=embedding["sha256"], fold_assignments_sha256=sha256(fold_path),
              fold_results_sha256=sha256(fold_results), seconds=time.monotonic()-started))


def require_evaluation(data, results, fingerprint):
    meta = paired.require_checkpoint(results / "pp1_summary.csv", results / "pp1_evaluation_complete.json", fingerprint)
    embedding = paired.require_checkpoint(data / "pp1.pkl", results / "pp1_embedding_complete.json", fingerprint)
    require_equal(meta["embedding_sha256"], embedding["sha256"], "Evaluated embedding")
    require_equal(meta["fold_results_sha256"], sha256(results / "pp1_fold_results.csv"), "Saved fold metrics")
    require_equal(meta["fold_assignments_sha256"], sha256(results / "fold_assignments.csv"), "Saved fold IDs")
    require_equal(meta["fold_assignments_sha256"], sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Reference fold IDs")
    return meta


def compare(data, results, fingerprint):
    evaluation = require_evaluation(data, results, fingerprint)
    old_config = read_json(MESH2024_RESULTS / "run_config.json")
    old_meta = read_json(MESH2024_RESULTS / "pp1_evaluation_complete.json")
    old_embedding = read_json(MESH2024_RESULTS / "pp1_embedding_complete.json")
    require_equal(old_config["walks"], WALK_SETTINGS, "MeSH2024 walk settings")
    require_equal(old_config["skipgram_args"], paired.SG_ARGS, "MeSH2024 Skip-gram settings")
    require_equal(old_meta["fingerprint"], old_config["fingerprint"], "MeSH2024 result fingerprint")
    require_equal(old_meta["sha256"], sha256(MESH2024_RESULTS / "pp1_summary.csv"), "MeSH2024 summary")
    require_equal(old_meta["embedding_sha256"], old_embedding["sha256"], "MeSH2024 evaluated embedding")
    require_equal(old_meta["fold_assignments_sha256"], evaluation["fold_assignments_sha256"], "Cross-year fold IDs")
    require_equal(old_embedding["binary_sha256"], sha256(data / "HetSG"), "Cross-year C++ binary")
    require_equal(old_embedding["token_map_sha256"], sha256(data / "token_to_node.json"), "Cross-year token map")
    summaries = [{row["metric"]: float(row["mean"]) for row in previous.read_rows(path, ",")}
                 for path in (MESH2024_RESULTS / "pp1_summary.csv", results / "pp1_summary.csv")]
    if any(set(values) != set(METRICS) for values in summaries):
        raise ValueError("Expected accuracy, auroc and aupr summaries")
    old, new = summaries
    rows = [dict(metric=metric, mesh2024_pp1=old[metric], mesh2021_pp1=new[metric],
                 mesh2021_minus_mesh2024=new[metric]-old[metric]) for metric in METRICS]
    output = results / "mesh_year_comparison.csv"
    previous.write_csv(output, rows, list(rows[0]))
    save_json(results / "comparison_complete.json", dict(fingerprint=fingerprint, sha256=sha256(output),
              mesh2024_summary_sha256=old_meta["sha256"], mesh2021_summary_sha256=evaluation["sha256"],
              shared_fold_assignments_sha256=evaluation["fold_assignments_sha256"],
              note="Same settings/binary/map/pairs/folds; separately generated walk corpora and four-thread training."))
    print("MESH YEAR COMPARISON\n" + json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("hierarchy", "similarity", "config", "prepare", "walks",
                                          "embedding", "evaluate", "compare"))
    parser.add_argument("--run-name", default=RUN_NAME)
    args = parser.parse_args()
    if args.phase == "hierarchy":
        inputs.build_hierarchy()
        return
    if args.phase == "similarity":
        inputs.build_similarity()
        return
    data, results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    elif args.phase == "prepare":
        prepare(data, results, fingerprint)
    elif args.phase == "walks":
        generate_corpus(data, results, fingerprint)
    elif args.phase == "embedding":
        # This unchanged helper atomically saves pp1.pkl and its completion
        # record immediately. Evaluation is a separate later notebook cell.
        paired.train(PP, data, results, fingerprint)
    elif args.phase == "evaluate":
        evaluate(data, results, fingerprint)
    else:
        compare(data, results, fingerprint)


if __name__ == "__main__":
    main()
