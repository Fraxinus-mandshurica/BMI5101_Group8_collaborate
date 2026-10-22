"""User-run pp=1/pp=0 comparison on one persisted semantic DREAMwalk corpus.

Reuse the no-teleport control's C++ binary, node-token map and evaluator.
Generate the new tau=.3 corpus once using the unchanged upstream functions.
No scientific computation runs on import.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.metadata import version
import json
import multiprocessing
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
WALK_SETTINGS = dict(tp_factor=0.3, seed=0, num_walks=100, walk_length=10,
                     p=1, q=1, workers=4, em_max_iter=5)
SEMANTIC_SETTINGS = dict(cutoff=0.4)
SOURCE_NODES = 29959
REFERENCE_NAME = "msi_seed0_skipgram_pp_comparison"
REFERENCE_DATA = ROOT / "data/embeddings" / REFERENCE_NAME
REFERENCE_RESULTS = ROOT / "results" / REFERENCE_NAME
METRICS = ("accuracy", "auroc", "aupr")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual, expected, description):
    if actual != expected:
        raise RuntimeError(description + " differs from the saved control")


def inspect_reference():
    """Read-only identity of the no-teleport artifacts reused for this control."""
    config = read_json(REFERENCE_RESULTS / "run_config.json")
    compiler = read_json(REFERENCE_RESULTS / "compiler_complete.json")
    corpus = read_json(REFERENCE_RESULTS / "walks_complete.json")
    inputs = dict(graph=previous.GRAPH, similarity=previous.SIMILARITY,
                  nodetypes=previous.NODETYPES, positives=previous.POSITIVES,
                  negatives=previous.NEGATIVES)
    input_hashes = {key: sha256(path) for key, path in inputs.items()}
    for key, digest in input_hashes.items():
        require_equal(digest, config["inputs"][key], "Shared " + key)
    require_equal(config["skipgram_args"], paired.SG_ARGS, "Skip-gram arguments")
    require_equal(config["xgboost"], previous.xgb_settings(), "XGBoost settings")
    require_equal(config["walks"], paired.WALKS, "No-teleport reference settings")
    require_equal(config["folds"], 10, "CV folds")
    require_equal(config["cv_seed"], 0, "CV seed")
    formal = read_json(ROOT / "results/msi_repeat0_run_config.json")
    require_equal(formal["dreamwalk"]["cutoff"], SEMANTIC_SETTINGS["cutoff"], "Saved semantic cutoff")
    require_equal(formal["dreamwalk"]["similarity_sha256"], input_hashes["similarity"], "Saved semantic graph")
    artifacts = dict(binary=sha256(REFERENCE_DATA / "HetSG"),
                     token_map=sha256(REFERENCE_DATA / "token_to_node.json"),
                     fold_assignments=sha256(REFERENCE_RESULTS / "fold_assignments.csv"))
    require_equal(artifacts["binary"], compiler["sha256"], "Historical binary")
    require_equal(artifacts["token_map"], corpus["token_map_sha256"], "Historical token map")
    for marker in (compiler, corpus):
        require_equal(marker["fingerprint"], config["fingerprint"], "Reference checkpoint")
    for pp in (1, 0):
        embedding = read_json(REFERENCE_RESULTS / f"pp{pp}_embedding_complete.json")
        evaluation = read_json(REFERENCE_RESULTS / f"pp{pp}_evaluation_complete.json")
        for marker in (embedding, evaluation):
            require_equal(marker["fingerprint"], config["fingerprint"], "Reference pp checkpoint")
        require_equal(embedding["pp"], pp, "Reference pp mode")
        require_equal(embedding["binary_sha256"], artifacts["binary"], "Reference embedding binary")
        require_equal(embedding["token_map_sha256"], artifacts["token_map"], "Reference embedding map")
        require_equal(embedding["corpus_sha256"], corpus["sha256"], "Reference embedding corpus")
        require_equal(evaluation["fold_assignments_sha256"], artifacts["fold_assignments"], "Reference fold IDs")
        artifacts[f"pp{pp}_summary"] = sha256(REFERENCE_RESULTS / f"pp{pp}_summary.csv")
        require_equal(artifacts[f"pp{pp}_summary"], evaluation["sha256"], "Reference pp summary")
    for relative, digest in config["sources"].items():
        require_equal(sha256(ROOT / relative), digest, relative)
    records = ["run_config.json", "compiler_complete.json", "walks_complete.json",
               "pp1_embedding_complete.json", "pp0_embedding_complete.json",
               "pp1_evaluation_complete.json", "pp0_evaluation_complete.json"]
    return dict(inputs=input_hashes, reference_artifacts=artifacts,
                reference_fingerprint=config["fingerprint"],
                reference_records={name: sha256(REFERENCE_RESULTS / name) for name in records})


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
    sources = [Path(__file__), Path(paired.__file__), Path(previous.__file__),
               ROOT / "src/run_msi_repeat0.py", ROOT / "src/run_no_teleport_length4.py",
               paired.UPSTREAM / "DREAMwalk/generate_embeddings.py",
               paired.UPSTREAM / "DREAMwalk/utils.py",
               paired.UPSTREAM / "DREAMwalk/HeterogeneousSG.cpp"]
    identity = dict(walks=WALK_SETTINGS, semantic=SEMANTIC_SETTINGS,
                    skipgram_args=paired.SG_ARGS, xgboost=previous.xgb_settings(),
                    folds=10, cv_seed=0, feature="drug_embedding - disease_embedding",
                    reference=inspect_reference(),
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
                  analysis="DIAGNOSTIC SEMANTIC-CORPUS PP COMPARISON",
                  comparison="same semantic corpus, same token map, same binary; pp=1 versus pp=0",
                  semantic_input="existing reviewed similarity graph at cutoff .4; no rebuilding or rethresholding",
                  pair_order="sorted positives + original saved seed-0 negative order",
                  fold_origin="deterministically regenerated and checked against saved no-teleport IDs",
                  notes=["A new tau=.3 corpus is required because the historical semantic walks were not retained.",
                         "Reuse the exact no-teleport binary and node-token map.",
                         "Both modes keep identical node-type-prefixed tokens, including pp=0.",
                         "Original EM, teleport selection and both negative-sampling branches are unchanged.",
                         "Upstream set_seed(0) is unchanged; no hashseed restoration or start-method override.",
                         "Worker Python/NumPy RNG state is not asserted; four C++ threads retain nondeterminism.",
                         "The within-semantic pp contrast is paired; comparison across tau uses separately sampled corpora.",
                         "Tau=.3 follows the accepted local semantic configuration, not a claimed Figure2a published value."],
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  multiprocessing_start_method=multiprocessing.get_context().get_start_method(),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def copy_exact(source, target):
    if target.exists():
        require_equal(sha256(target), sha256(source), "Copied artifact")
        return
    temporary = target.with_name(target.name + ".tmp")
    shutil.copy2(source, temporary)
    require_equal(sha256(temporary), sha256(source), "Copied artifact")
    temporary.replace(target)


def prepare(data, results, fingerprint):
    copy_exact(REFERENCE_DATA / "HetSG", data / "HetSG")
    copy_exact(REFERENCE_DATA / "token_to_node.json", data / "token_to_node.json")
    marker = results / "compiler_complete.json"
    if not paired.completed(data / "HetSG", marker, fingerprint):
        save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(data / "HetSG"),
                  token_map_sha256=sha256(data / "token_to_node.json"),
                  reused_from=str((REFERENCE_DATA / "HetSG").relative_to(ROOT)),
                  original_compiler=read_json(REFERENCE_RESULTS / "compiler_complete.json")))
    print("Historical C++ binary and token map ready", flush=True)


def checked_tokens(token_to_node, types, graph_nodes):
    if len(token_to_node) != len(set(token_to_node.values())):
        raise ValueError("Token map must be a bijection")
    require_equal(set(token_to_node.values()), set(types), "Token-map node universe")
    require_equal(set(graph_nodes), set(types), "Graph node universe")
    prefixes = dict(drug="d", disease="i", gene="g", etc="e")
    for token, node in token_to_node.items():
        if types[node] not in prefixes or not token.startswith(prefixes[types[node]]) or not token[1:].isdigit():
            raise ValueError("Incorrect type-prefixed token: " + token)
    return {node: token for token, node in token_to_node.items()}


def write_corpus(walks, stream, node_to_token, walk_length, expected_walks, walks_per_node):
    starts = Counter()
    type_pairs = Counter()
    count = 0
    for walk in walks:
        if len(walk) != walk_length:
            raise ValueError("Unexpected semantic walk length at path " + str(count + 1))
        tokens = [node_to_token[node] for node in walk]
        if count:
            stream.write("\n")
        stream.write(" ".join(tokens))
        starts[walk[0]] += 1
        type_pairs.update(a[0] + "->" + b[0] for a, b in zip(tokens, tokens[1:]))
        count += 1
    if count != expected_walks or set(starts) != set(node_to_token) or any(n != walks_per_node for n in starts.values()):
        raise ValueError("Unexpected semantic corpus size/start-node counts")
    return dict(walks=count, tokens=count * walk_length, nodes=len(node_to_token), walk_length=walk_length,
                adjacent_node_type_pairs=dict(type_pairs))


def generate_corpus(data, results, fingerprint):
    from DREAMwalk import generate_embeddings as ge
    corpus, marker = data / "walks.txt", results / "walks_complete.json"
    if paired.completed(corpus, marker, fingerprint):
        require_equal(sha256(data / "token_to_node.json"), read_json(marker)["token_map_sha256"], "Saved map")
        return
    compiler = paired.require_checkpoint(data / "HetSG", results / "compiler_complete.json", fingerprint)
    map_path = data / "token_to_node.json"
    require_equal(sha256(map_path), compiler["token_map_sha256"], "Prepared map")
    if corpus.exists() or any((data / f"pp{pp}.pkl").exists() or
                             (results / f"pp{pp}_embedding_complete.json").exists() for pp in (1, 0)):
        raise RuntimeError("Existing corpus/embedding prevents regeneration; preserve outputs and use a new RUN_NAME")
    started = time.monotonic()
    ge.set_seed(WALK_SETTINGS["seed"])
    graph = ge.read_graph(str(previous.GRAPH), weighted=True, directed=False, delimiter="\t")
    semantic = ge.read_graph(str(previous.SIMILARITY), weighted=True, directed=False)
    if WALK_SETTINGS["tp_factor"] <= 0 or semantic.number_of_edges() == 0:
        raise RuntimeError("This control requires active semantic teleport and a nonempty semantic graph")
    types = {row["node"]: row["type"] for row in previous.read_rows(previous.NODETYPES)}
    node_to_token = checked_tokens(read_json(map_path), types, graph.nodes())
    require_equal(len(node_to_token), SOURCE_NODES, "MSI node count")
    t0 = time.monotonic()
    print("Starting unchanged edge-type EM: 5 iterations, length=10", flush=True)
    matrix = ge.train_edgetype_transition_matrix(WALK_SETTINGS["em_max_iter"], graph,
                                                str(previous.GRAPH), "\t", WALK_SETTINGS["walk_length"],
                                                WALK_SETTINGS["p"], WALK_SETTINGS["q"])
    em_seconds = time.monotonic() - t0
    t0 = time.monotonic()
    print("Starting ONE semantic corpus: tau=.3, 100 x 29,959 walks, length=10", flush=True)
    walks = ge.generate_DREAMwalk_paths(graph, semantic, matrix, WALK_SETTINGS["p"], WALK_SETTINGS["q"],
                                       WALK_SETTINGS["num_walks"], WALK_SETTINGS["walk_length"],
                                       WALK_SETTINGS["tp_factor"], WALK_SETTINGS["workers"])
    walk_seconds = time.monotonic() - t0
    temporary = data / "walks.txt.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        counts = write_corpus(walks, stream, node_to_token, WALK_SETTINGS["walk_length"],
                              SOURCE_NODES * WALK_SETTINGS["num_walks"], WALK_SETTINGS["num_walks"])
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(corpus)
    save_json(results / "edge_transition_matrix.json", matrix.tolist())
    save_json(marker, dict(counts, fingerprint=fingerprint, sha256=sha256(corpus),
              token_map_sha256=sha256(map_path), tp_factor=WALK_SETTINGS["tp_factor"],
              similarity_sha256=sha256(previous.SIMILARITY), similarity_edges=semantic.number_of_edges(),
              serialization="newline between walks; no final newline",
              em_seconds=em_seconds, walk_seconds=walk_seconds, total_seconds=time.monotonic()-started))
    print("SEMANTIC SHARED CORPUS SAVED:", corpus, flush=True)


def evaluate(pp, data, results, fingerprint):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    output, marker = results / f"pp{pp}_summary.csv", results / f"pp{pp}_evaluation_complete.json"
    if paired.completed(output, marker, fingerprint):
        require_evaluation(pp, data, results, fingerprint)
        return
    embedding = paired.require_checkpoint(data / f"pp{pp}.pkl", results / f"pp{pp}_embedding_complete.json", fingerprint)
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
    serialized = [{key: str(value) for key, value in row.items()} for row in pair_rows]
    require_equal(serialized, expected, "Pair/fold assignments before fitting")
    fold_path = results / "fold_assignments.csv"
    if fold_path.exists():
        require_equal(previous.read_rows(fold_path, ","), expected, "Existing pair/fold assignments")
    else:
        previous.write_csv(fold_path, pair_rows, list(pair_rows[0]))
    require_equal(sha256(fold_path), sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Fold CSV bytes")
    vectors = previous.load_embedding(data / f"pp{pp}.pkl", types, pairs)
    started = time.monotonic()
    rows = previous.evaluate(vectors, positives, negatives, 0, "semantic_shared_walks")
    for row in rows:
        row.update(label=f"semantic_same_walks_pp{pp}", embedding=f"semantic_tau_0.3_pp{pp}", pp=pp)
    summary = previous.metric_summary(rows, 0, f"semantic_same_walks_pp{pp}")
    fold_results = results / f"pp{pp}_fold_results.csv"
    previous.write_csv(fold_results, rows, list(rows[0]))
    previous.write_csv(output, summary, list(summary[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output), pp=pp,
              embedding_sha256=embedding["sha256"], fold_assignments_sha256=sha256(fold_path),
              fold_results_sha256=sha256(fold_results), seconds=time.monotonic()-started))


def require_evaluation(pp, data, results, fingerprint):
    meta = paired.require_checkpoint(results / f"pp{pp}_summary.csv", results / f"pp{pp}_evaluation_complete.json", fingerprint)
    embedding = paired.require_checkpoint(data / f"pp{pp}.pkl", results / f"pp{pp}_embedding_complete.json", fingerprint)
    require_equal(meta["embedding_sha256"], embedding["sha256"], "Evaluated embedding")
    require_equal(meta["fold_results_sha256"], sha256(results / f"pp{pp}_fold_results.csv"), "Saved fold metrics")
    require_equal(meta["fold_assignments_sha256"], sha256(results / "fold_assignments.csv"), "Saved fold IDs")
    require_equal(meta["fold_assignments_sha256"], sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Reference fold IDs")
    return embedding, meta


def effect_rows(semantic_one, semantic_zero, no_teleport_one, no_teleport_zero):
    def values(rows):
        result = {row["metric"]: float(row["mean"]) for row in rows}
        require_equal(set(result), set(METRICS), "Summary metrics")
        return result
    one, zero, old_one, old_zero = map(values, (semantic_one, semantic_zero, no_teleport_one, no_teleport_zero))
    main, interaction = [], []
    for metric in METRICS:
        current_effect = one[metric]-zero[metric]
        previous_effect = old_one[metric]-old_zero[metric]
        main.append(dict(metric=metric, semantic_pp1=one[metric], semantic_pp0=zero[metric],
                         pp1_minus_pp0=current_effect))
        interaction.append(dict(metric=metric, no_teleport_pp1=old_one[metric], no_teleport_pp0=old_zero[metric],
                                no_teleport_pp1_minus_pp0=previous_effect,
                                semantic_pp1=one[metric], semantic_pp0=zero[metric],
                                semantic_pp1_minus_pp0=current_effect,
                                pp_effect_difference=current_effect-previous_effect))
    return main, interaction


def paired_fold_rows(one, zero):
    by_mode = [{int(row["fold"]): row for row in rows} for rows in (one, zero)]
    if any(len(rows) != 10 for rows in (one, zero)) or any(set(rows) != set(range(1, 11)) for rows in by_mode):
        raise ValueError("Expected exactly ten matching folds in both modes")
    return [dict(fold=fold, **{metric + "_pp1_minus_pp0":
                float(by_mode[0][fold][metric])-float(by_mode[1][fold][metric]) for metric in METRICS})
            for fold in range(1, 11)]


def compare(data, results, fingerprint):
    embeddings, evaluations = {}, {}
    for pp in (1, 0):
        embeddings[pp], evaluations[pp] = require_evaluation(pp, data, results, fingerprint)
        require_equal(embeddings[pp]["command"],
                      ["./HetSG", "-train", "walks.txt", "-output", f"pp{pp}", "-pp", str(pp)] + paired.SG_ARGS,
                      "Training command")
    for key in ("corpus_sha256", "binary_sha256", "token_map_sha256"):
        require_equal(embeddings[1][key], embeddings[0][key], "Paired " + key)
    require_equal(evaluations[1]["fold_assignments_sha256"], evaluations[0]["fold_assignments_sha256"], "Paired folds")
    corpus = paired.require_checkpoint(data / "walks.txt", results / "walks_complete.json", fingerprint)
    require_equal(corpus["sha256"], embeddings[1]["corpus_sha256"], "Shared semantic corpus")
    require_equal(sha256(data / "HetSG"), embeddings[1]["binary_sha256"], "Shared binary")
    require_equal(sha256(data / "token_to_node.json"), embeddings[1]["token_map_sha256"], "Shared token map")
    output, marker = results / "pp_comparison.csv", results / "comparison_complete.json"
    if paired.completed(output, marker, fingerprint):
        meta = read_json(marker)
        require_equal(meta["paired_fold_deltas_sha256"], sha256(results / "paired_fold_deltas.csv"), "Paired deltas")
        require_equal(meta["pp_by_teleport_sha256"], sha256(results / "pp_by_teleport.csv"), "PP-by-teleport table")
        return
    main, interaction = effect_rows(
        previous.read_rows(results / "pp1_summary.csv", ","), previous.read_rows(results / "pp0_summary.csv", ","),
        previous.read_rows(REFERENCE_RESULTS / "pp1_summary.csv", ","),
        previous.read_rows(REFERENCE_RESULTS / "pp0_summary.csv", ","))
    deltas = paired_fold_rows(previous.read_rows(results / "pp1_fold_results.csv", ","),
                              previous.read_rows(results / "pp0_fold_results.csv", ","))
    previous.write_csv(output, main, list(main[0]))
    previous.write_csv(results / "paired_fold_deltas.csv", deltas, list(deltas[0]))
    previous.write_csv(results / "pp_by_teleport.csv", interaction, list(interaction[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output),
              paired_fold_deltas_sha256=sha256(results / "paired_fold_deltas.csv"),
              pp_by_teleport_sha256=sha256(results / "pp_by_teleport.csv"),
              shared_corpus_sha256=corpus["sha256"], shared_binary_sha256=embeddings[1]["binary_sha256"],
              shared_token_map_sha256=embeddings[1]["token_map_sha256"],
              shared_fold_assignments_sha256=evaluations[1]["fold_assignments_sha256"]))
    print("SEMANTIC PAIRED COMPARISON COMPLETE\n" + json.dumps(main, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "prepare", "walks", "pp1", "evaluate-pp1", "pp0", "evaluate-pp0", "compare"))
    parser.add_argument("--run-name", default="msi_seed0_semantic_skipgram_pp_comparison")
    args = parser.parse_args()
    data, results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    elif args.phase == "prepare":
        prepare(data, results, fingerprint)
    elif args.phase == "walks":
        generate_corpus(data, results, fingerprint)
    elif args.phase in ("pp1", "pp0"):
        paired.train(int(args.phase[-1]), data, results, fingerprint)
    elif args.phase.startswith("evaluate-"):
        evaluate(int(args.phase[-1]), data, results, fingerprint)
    else:
        compare(data, results, fingerprint)


if __name__ == "__main__":
    main()
