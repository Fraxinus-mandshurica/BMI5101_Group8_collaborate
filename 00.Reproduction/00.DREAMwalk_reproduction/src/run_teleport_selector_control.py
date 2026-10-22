"""User-run first-hit teleport-selector diagnostic, pp=0 and tau=.3.

Reuse the completed semantic control's EM matrix, binary, node map and pp=0
baseline. Only the fixed-selector corpus, its embedding and evaluation are new.
The public vendor implementation and all historical outputs remain untouched.
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
import run_semantic_pp_comparison as semantic

paired = semantic.paired
previous = semantic.previous
save_json, read_json, sha256 = semantic.save_json, semantic.read_json, semantic.sha256
require_equal = semantic.require_equal
WALK_SETTINGS = dict(tp_factor=0.3, seed=0, num_walks=100, walk_length=10,
                     p=1, q=1, workers=4, em_max_iter=5)
RUN_SETTINGS = dict(pp=0, tp_factor=0.3, seed=0, dimension=128, window_size=4,
                    epochs=1, workers=4, folds=10, cv_seed=0,
                    walk_length=10, num_walks=100)
SEMANTIC_SETTINGS = dict(cutoff=0.4)
SOURCE_NODES = 29959
RUN_NAME = "msi_seed0_teleport_selector_control"
REFERENCE_NAME = "msi_seed0_semantic_skipgram_pp_comparison"
REFERENCE_DATA = ROOT / "data/embeddings" / REFERENCE_NAME
REFERENCE_RESULTS = ROOT / "results" / REFERENCE_NAME
METRICS = ("accuracy", "auroc", "aupr")
SELECTOR_MODULE = ROOT / "src/teleport_selector_diagnostic.py"


def inspect_reference():
    """Read and verify baseline identities without running any scientific stage."""
    config = read_json(REFERENCE_RESULTS / "run_config.json")
    require_equal(config["walks"], WALK_SETTINGS, "Semantic baseline walks")
    require_equal(config["semantic"], SEMANTIC_SETTINGS, "Semantic baseline cutoff")
    require_equal(config["skipgram_args"], paired.SG_ARGS, "Baseline Skip-gram arguments")
    require_equal(config["xgboost"], previous.xgb_settings(), "Baseline XGBoost")
    require_equal(config["reference"], semantic.inspect_reference(), "Baseline shared-input reference")
    for relative, digest in config["sources"].items():
        require_equal(sha256(ROOT / relative), digest, "Baseline source " + relative)
    fingerprint = config["fingerprint"]
    corpus = paired.require_checkpoint(REFERENCE_DATA / "walks.txt",
                                      REFERENCE_RESULTS / "walks_complete.json", fingerprint)
    compiler = paired.require_checkpoint(REFERENCE_DATA / "HetSG",
                                        REFERENCE_RESULTS / "compiler_complete.json", fingerprint)
    embedding, evaluation = semantic.require_evaluation(0, REFERENCE_DATA, REFERENCE_RESULTS, fingerprint)
    artifacts = dict(corpus=corpus["sha256"], binary=compiler["sha256"],
                     token_map=sha256(REFERENCE_DATA / "token_to_node.json"),
                     matrix=sha256(REFERENCE_RESULTS / "edge_transition_matrix.json"),
                     embedding=embedding["sha256"], summary=evaluation["sha256"],
                     fold_results=evaluation["fold_results_sha256"],
                     fold_assignments=evaluation["fold_assignments_sha256"])
    require_equal(embedding["pp"], 0, "Baseline pp mode")
    require_equal(embedding["command"], ["./HetSG", "-train", "walks.txt", "-output", "pp0", "-pp", "0"]
                  + paired.SG_ARGS, "Baseline training command")
    for key, value in (("corpus_sha256", artifacts["corpus"]),
                       ("binary_sha256", artifacts["binary"]),
                       ("token_map_sha256", artifacts["token_map"])):
        require_equal(embedding[key], value, "Baseline embedding " + key)
    require_equal(corpus["token_map_sha256"], artifacts["token_map"], "Baseline corpus map")
    require_equal(corpus["tp_factor"], WALK_SETTINGS["tp_factor"], "Baseline tau")
    require_equal(corpus["similarity_sha256"], sha256(previous.SIMILARITY), "Baseline semantic input")
    return dict(name=REFERENCE_NAME, fingerprint=fingerprint, artifacts=artifacts,
                records={name: sha256(REFERENCE_RESULTS / name) for name in
                         ("run_config.json", "walks_complete.json", "compiler_complete.json",
                          "pp0_embedding_complete.json", "pp0_evaluation_complete.json")})


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
    sources = [Path(__file__), SELECTOR_MODULE, Path(semantic.__file__), Path(paired.__file__),
               Path(previous.__file__), ROOT / "src/run_msi_repeat0.py", ROOT / "src/run_no_teleport_length4.py",
               paired.UPSTREAM / "DREAMwalk/generate_embeddings.py",
               paired.UPSTREAM / "DREAMwalk/utils.py", paired.UPSTREAM / "DREAMwalk/HeterogeneousSG.cpp"]
    identity = dict(walks=WALK_SETTINGS, run_settings=RUN_SETTINGS, semantic=SEMANTIC_SETTINGS,
                    skipgram_args=paired.SG_ARGS, xgboost=previous.xgb_settings(), folds=10, cv_seed=0,
                    feature="drug_embedding - disease_embedding", reference=inspect_reference(),
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
                  analysis="DIAGNOSTIC TELEPORT SELECTOR CHANGE; not the original benchmark implementation",
                  comparison="fixed first-hit selector versus saved upstream selector; pp=0 in both",
                  changed_logic="Return immediately at first cumulative-threshold hit instead of breaking inner loop",
                  fold_origin="deterministically regenerated and checked against saved semantic baseline IDs",
                  notes=["Reuse the baseline trained EM matrix; do not rerun EM or the baseline.",
                         "Keep original shuffle, set iteration and NumPy draw; do not sort candidates differently.",
                         "Each spawned worker installs the fixed selector before calling unchanged upstream _parmap_walks.",
                         "The fixed corpus is newly sampled; it is not a paired RNG replay of baseline walks.",
                         "Python/NumPy worker RNG states are not asserted; four-thread SG remains nondeterministic.",
                         "Original set_seed(0) and multiprocessing start method are unchanged.",
                         "Tau=.3 is the accepted local setting, not a claimed published Figure2a value."],
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  multiprocessing_start_method=multiprocessing.get_context().get_start_method(),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def prepare(data, results, fingerprint):
    semantic.copy_exact(REFERENCE_DATA / "HetSG", data / "HetSG")
    semantic.copy_exact(REFERENCE_DATA / "token_to_node.json", data / "token_to_node.json")
    semantic.copy_exact(REFERENCE_RESULTS / "edge_transition_matrix.json", results / "edge_transition_matrix.json")
    semantic.copy_exact(REFERENCE_RESULTS / "fold_assignments.csv", results / "fold_assignments.csv")
    marker = results / "compiler_complete.json"
    if not paired.completed(data / "HetSG", marker, fingerprint):
        save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(data / "HetSG"),
                  token_map_sha256=sha256(data / "token_to_node.json"),
                  matrix_sha256=sha256(results / "edge_transition_matrix.json"),
                  reused_from=str(REFERENCE_DATA.relative_to(ROOT)),
                  original_compiler=read_json(REFERENCE_RESULTS / "compiler_complete.json")))
    print("Reference EM matrix, unchanged pp=0 binary, token map and folds ready", flush=True)


def checked_worker_audit(audits, expected_tasks, selector_sha256):
    from teleport_selector_diagnostic import SELECTOR_NAME
    if len(audits) != expected_tasks or {row["task_id"] for row in audits} != set(range(expected_tasks)):
        raise RuntimeError("Missing/duplicate worker task audits")
    for row in audits:
        require_equal(row["selector"], SELECTOR_NAME, "Worker selector identity")
        require_equal(row["selector_is_fixed"], True, "Worker-installed selector")
        require_equal(row["selector_source_sha256"], selector_sha256, "Worker selector source")
        if row["pid"] == os.getpid() or row["selector_calls"] < 0:
            raise RuntimeError("Expected fixed selector evidence from child workers")
    if sum(row["selector_calls"] for row in audits) <= 0:
        raise RuntimeError("No worker actually executed the fixed teleport selector")
    return dict(tasks=len(audits), worker_pids=sorted({row["pid"] for row in audits}),
                selector_calls=sum(row["selector_calls"] for row in audits),
                selector_source_sha256=selector_sha256, audits=audits,
                rng_state_claim="No assertion about Python/NumPy RNG state inside workers")


def generate_corpus(data, results, fingerprint):
    import numpy as np
    import parmap
    from DREAMwalk import generate_embeddings as ge
    import teleport_selector_diagnostic as selector
    corpus, marker = data / "walks.txt", results / "walks_complete.json"
    if paired.completed(corpus, marker, fingerprint):
        meta = read_json(marker)
        require_equal(sha256(data / "token_to_node.json"), meta["token_map_sha256"], "Saved map")
        require_equal(sha256(results / "worker_selector_audit.json"), meta["worker_audit_sha256"], "Worker audit")
        require_equal(sha256(results / "edge_transition_matrix.json"), meta["matrix_sha256"], "Reused matrix")
        return
    compiler = paired.require_checkpoint(data / "HetSG", results / "compiler_complete.json", fingerprint)
    require_equal(sha256(data / "token_to_node.json"), compiler["token_map_sha256"], "Prepared map")
    require_equal(sha256(results / "edge_transition_matrix.json"), compiler["matrix_sha256"], "Prepared matrix")
    if corpus.exists() or (data / "pp0.pkl").exists() or (results / "pp0_embedding_complete.json").exists():
        raise RuntimeError("Existing corpus/embedding prevents regeneration; preserve outputs and use a new RUN_NAME")
    started = time.monotonic()
    ge.set_seed(WALK_SETTINGS["seed"])
    graph = ge.read_graph(str(previous.GRAPH), weighted=True, directed=False, delimiter="\t")
    similarity = ge.read_graph(str(previous.SIMILARITY), weighted=True, directed=False)
    if WALK_SETTINGS["tp_factor"] <= 0 or similarity.number_of_edges() == 0:
        raise RuntimeError("This selector control requires active teleport and nonempty semantic input")
    types = {row["node"]: row["type"] for row in previous.read_rows(previous.NODETYPES)}
    node_to_token = semantic.checked_tokens(read_json(data / "token_to_node.json"), types, graph.nodes())
    require_equal(len(node_to_token), SOURCE_NODES, "MSI node count")
    matrix = np.asarray(read_json(results / "edge_transition_matrix.json"), dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
        raise ValueError("Invalid saved edge-transition matrix")
    print("Reusing trained EM matrix; generating fixed-selector tau=.3 walks only", flush=True)
    t0 = time.monotonic()
    # Match upstream generate_DREAMwalk_paths scheduling/flattening. Only the
    # worker entry point differs; it installs the patch in its own interpreter.
    task_results = parmap.map(selector.worker_walks, range(WALK_SETTINGS["num_walks"]),
                              list(graph.nodes()), graph, similarity, matrix,
                              WALK_SETTINGS["p"], WALK_SETTINGS["q"], WALK_SETTINGS["walk_length"],
                              WALK_SETTINGS["tp_factor"], pm_pbar=False,
                              pm_processes=min(WALK_SETTINGS["num_walks"], WALK_SETTINGS["workers"]))
    walk_seconds = time.monotonic() - t0
    audits = checked_worker_audit([item[1] for item in task_results], WALK_SETTINGS["num_walks"],
                                 sha256(SELECTOR_MODULE))
    walks = (walk for task_walks, _ in task_results for walk in task_walks)
    temporary = data / "walks.txt.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        counts = semantic.write_corpus(walks, stream, node_to_token, WALK_SETTINGS["walk_length"],
                                       SOURCE_NODES * WALK_SETTINGS["num_walks"], WALK_SETTINGS["num_walks"])
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(corpus)
    audit_path = results / "worker_selector_audit.json"
    save_json(audit_path, dict(audits, fingerprint=fingerprint))
    save_json(marker, dict(counts, fingerprint=fingerprint, sha256=sha256(corpus),
              token_map_sha256=sha256(data / "token_to_node.json"),
              matrix_sha256=sha256(results / "edge_transition_matrix.json"),
              worker_audit_sha256=sha256(audit_path), selector_calls=audits["selector_calls"],
              tp_factor=WALK_SETTINGS["tp_factor"], similarity_sha256=sha256(previous.SIMILARITY),
              similarity_edges=similarity.number_of_edges(),
              serialization="newline between walks; no final newline", em_reused=True,
              em_reference=str((REFERENCE_RESULTS / "edge_transition_matrix.json").relative_to(ROOT)),
              walk_seconds=walk_seconds, total_seconds=time.monotonic()-started))
    print("FIXED-SELECTOR CORPUS SAVED:", corpus, flush=True)


def evaluate(data, results, fingerprint):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    output, marker = results / "pp0_summary.csv", results / "pp0_evaluation_complete.json"
    if paired.completed(output, marker, fingerprint):
        require_evaluation(data, results, fingerprint)
        return
    embedding = paired.require_checkpoint(data / "pp0.pkl", results / "pp0_embedding_complete.json", fingerprint)
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
    expected = previous.read_rows(REFERENCE_RESULTS / "fold_assignments.csv", ",")
    require_equal([{k: str(v) for k, v in row.items()} for row in pair_rows], expected,
                  "Pair/fold assignments BEFORE fitting")
    fold_path = results / "fold_assignments.csv"
    require_equal(previous.read_rows(fold_path, ","), expected, "Prepared fold assignments")
    require_equal(sha256(fold_path), sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Fold CSV bytes")
    vectors = previous.load_embedding(data / "pp0.pkl", types, pairs)
    started = time.monotonic()
    rows = previous.evaluate(vectors, positives, negatives, 0, "fixed_teleport_selector_pp0")
    for row in rows:
        row.update(label="fixed_teleport_selector_pp0", embedding="fixed_teleport_tau_0.3_pp0", pp=0)
    summary = previous.metric_summary(rows, 0, "fixed_teleport_selector_pp0")
    fold_results = results / "pp0_fold_results.csv"
    previous.write_csv(fold_results, rows, list(rows[0]))
    previous.write_csv(output, summary, list(summary[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output), pp=0,
              embedding_sha256=embedding["sha256"], fold_assignments_sha256=sha256(fold_path),
              fold_results_sha256=sha256(fold_results), seconds=time.monotonic()-started))


def require_evaluation(data, results, fingerprint):
    embedding, evaluation = semantic.require_evaluation(0, data, results, fingerprint)
    require_equal(embedding["pp"], 0, "Fixed selector pp")
    require_equal(embedding["command"], ["./HetSG", "-train", "walks.txt", "-output", "pp0", "-pp", "0"]
                  + paired.SG_ARGS, "Fixed selector training command")
    return embedding, evaluation


def comparison_rows(original, fixed):
    original_values = {r["metric"]: float(r["mean"]) for r in original}
    fixed_values = {r["metric"]: float(r["mean"]) for r in fixed}
    require_equal(set(original_values), set(METRICS), "Original metrics")
    require_equal(set(fixed_values), set(METRICS), "Fixed metrics")
    return [dict(metric=metric, original_selector_pp0=original_values[metric],
                 fixed_selector_pp0=fixed_values[metric],
                 fixed_minus_original=fixed_values[metric]-original_values[metric]) for metric in METRICS]


def paired_fold_rows(original, fixed):
    by_mode = [{int(row["fold"]): row for row in rows} for rows in (original, fixed)]
    if any(len(rows) != 10 for rows in (original, fixed)) or any(set(rows) != set(range(1, 11)) for rows in by_mode):
        raise ValueError("Expected ten matching folds")
    return [dict(fold=fold, **{metric + "_fixed_minus_original":
                 float(by_mode[1][fold][metric])-float(by_mode[0][fold][metric]) for metric in METRICS})
            for fold in range(1, 11)]


def compare(data, results, fingerprint):
    fixed_embedding, fixed_evaluation = require_evaluation(data, results, fingerprint)
    reference_config = read_json(REFERENCE_RESULTS / "run_config.json")
    original_embedding, original_evaluation = semantic.require_evaluation(
        0, REFERENCE_DATA, REFERENCE_RESULTS, reference_config["fingerprint"])
    for key in ("binary_sha256", "token_map_sha256"):
        require_equal(fixed_embedding[key], original_embedding[key], "Comparison " + key)
    require_equal(fixed_evaluation["fold_assignments_sha256"], original_evaluation["fold_assignments_sha256"],
                  "Comparison folds")
    corpus = paired.require_checkpoint(data / "walks.txt", results / "walks_complete.json", fingerprint)
    require_equal(corpus["sha256"], fixed_embedding["corpus_sha256"], "Fixed corpus")
    require_equal(sha256(data / "HetSG"), fixed_embedding["binary_sha256"], "Fixed binary")
    require_equal(sha256(data / "token_to_node.json"), fixed_embedding["token_map_sha256"], "Fixed token map")
    require_equal(corpus["matrix_sha256"], sha256(REFERENCE_RESULTS / "edge_transition_matrix.json"), "Shared EM matrix")
    audit_path = results / "worker_selector_audit.json"
    require_equal(corpus["worker_audit_sha256"], sha256(audit_path), "Fixed worker audit")
    checked_worker_audit(read_json(audit_path)["audits"], WALK_SETTINGS["num_walks"], sha256(SELECTOR_MODULE))
    output, marker = results / "teleport_selector_comparison.csv", results / "comparison_complete.json"
    if paired.completed(output, marker, fingerprint):
        require_equal(read_json(marker)["paired_fold_deltas_sha256"], sha256(results / "paired_fold_deltas.csv"),
                      "Saved paired deltas")
        return
    rows = comparison_rows(previous.read_rows(REFERENCE_RESULTS / "pp0_summary.csv", ","),
                           previous.read_rows(results / "pp0_summary.csv", ","))
    deltas = paired_fold_rows(previous.read_rows(REFERENCE_RESULTS / "pp0_fold_results.csv", ","),
                              previous.read_rows(results / "pp0_fold_results.csv", ","))
    previous.write_csv(output, rows, list(rows[0]))
    previous.write_csv(results / "paired_fold_deltas.csv", deltas, list(deltas[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output),
              paired_fold_deltas_sha256=sha256(results / "paired_fold_deltas.csv"),
              original_summary_sha256=original_evaluation["sha256"], fixed_summary_sha256=fixed_evaluation["sha256"],
              original_corpus_sha256=original_embedding["corpus_sha256"], fixed_corpus_sha256=corpus["sha256"],
              shared_binary_sha256=fixed_embedding["binary_sha256"],
              shared_token_map_sha256=fixed_embedding["token_map_sha256"],
              shared_matrix_sha256=corpus["matrix_sha256"], worker_audit_sha256=sha256(audit_path),
              shared_fold_assignments_sha256=fixed_evaluation["fold_assignments_sha256"]))
    print("TELEPORT SELECTOR COMPARISON COMPLETE\n" + json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "prepare", "walks", "embedding", "evaluate", "compare"))
    parser.add_argument("--run-name", default=RUN_NAME)
    args = parser.parse_args()
    data, results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    elif args.phase == "prepare":
        prepare(data, results, fingerprint)
    elif args.phase == "walks":
        generate_corpus(data, results, fingerprint)
    elif args.phase == "embedding":
        paired.train(0, data, results, fingerprint)
    elif args.phase == "evaluate":
        evaluate(data, results, fingerprint)
    else:
        compare(data, results, fingerprint)


if __name__ == "__main__":
    main()
