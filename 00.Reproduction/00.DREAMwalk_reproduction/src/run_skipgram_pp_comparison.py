"""Paired pp=1/pp=0 diagnostic on ONE persisted DREAMwalk walk corpus.

The upstream sources are unmodified. Compile once, then invoke the same C++
binary with exactly the same prefixed tokens and settings except -pp.
No experiment executes on import. All phases are launched by the user notebook.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import diagnose_msi_repeat0 as previous
from run_no_teleport_length4 import PINS, PINNED_COMMIT, save_json, completed

UPSTREAM = previous.UPSTREAM
WALKS = dict(tp_factor=0, seed=0, num_walks=100, walk_length=10,
             p=1, q=1, workers=4, em_max_iter=5)
SG_ARGS = ["-min-count", "0", "-size", "128", "-iter", "1",
           "-window", "4", "-threads", "4"]
COMPILE_FLAGS = ["-lm", "-pthread", "-O3", "-march=native", "-Wall",
                 "-funroll-loops", "-Wno-unused-result"]
SDK_HEADERS = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk/usr/include/c++/v1")


def context(name):
    if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("RUN_NAME must contain only letters, digits, underscore or hyphen")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Use the existing DREAMwalk Python 3.8 environment")
    packages = {p: version(p) for p in PINS}
    if packages != PINS:
        raise RuntimeError(f"DREAMwalk environment pins differ: {packages}")
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Unexpected upstream commit: {commit}")
    inputs = dict(graph=previous.GRAPH, similarity=previous.SIMILARITY,
                  nodetypes=previous.NODETYPES, positives=previous.POSITIVES,
                  negatives=previous.NEGATIVES, historical_length10=previous.CONTROL_SUMMARY)
    sources = [Path(__file__), ROOT / "src/run_no_teleport_length4.py",
               Path(previous.__file__), ROOT / "src/run_msi_repeat0.py",
               UPSTREAM / "DREAMwalk/generate_embeddings.py",
               UPSTREAM / "DREAMwalk/utils.py", UPSTREAM / "DREAMwalk/HeterogeneousSG.cpp"]
    identity = dict(walks=WALKS, skipgram_args=SG_ARGS, compile_flags=COMPILE_FLAGS,
                    comparison="same corpus, same token map, same binary; only -pp 1 versus -pp 0",
                    xgboost=previous.xgb_settings(), folds=10, cv_seed=0,
                    inputs={k: previous.sha256(p) for k, p in inputs.items()},
                    sources={str(p.relative_to(ROOT)): previous.sha256(p) for p in sources},
                    packages=packages, upstream_commit=commit)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    results = ROOT / "results" / name
    data = ROOT / "data/embeddings" / name
    results.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    config = results / "run_config.json"
    if config.exists():
        if json.loads(config.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("Inputs/code/settings changed; use a new RUN_NAME")
    else:
        if any(data.iterdir()):
            raise RuntimeError(f"Existing unregistered data at {data}; preserve it and use a new RUN_NAME")
        save_json(config, dict(identity, fingerprint=fingerprint,
                  analysis="DIAGNOSTIC ABLATION; not an author benchmark or a claimed bug fix",
                  input_paths={k: str(p) for k, p in inputs.items()},
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  multiprocessing_start_method=multiprocessing.get_context().get_start_method(),
                  notes=["One newly generated corpus, not the deleted historical length-10 corpus.",
                         "Both modes consume identical type-prefixed tokens, including pp=0.",
                         "pp switches the entire negative-sampling branch, including rejection behavior.",
                         "Four-thread training may vary; worker RNG state is not asserted.",
                         "No hashseed restoration or multiprocessing start-method override.",
                         "Paper no-teleport AUROC/AUPR values are approximate contextual references."],
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def require_checkpoint(path, marker, fingerprint):
    if not completed(path, marker, fingerprint):
        raise RuntimeError(f"Complete the prior notebook stage first: {marker.name}")
    return json.loads(marker.read_text())


def compile_binary(data, results, fingerprint):
    binary, marker = data / "HetSG", results / "compiler_complete.json"
    if completed(binary, marker, fingerprint):
        return
    if not SDK_HEADERS.is_dir():
        raise FileNotFoundError(f"Previously used compiler headers unavailable: {SDK_HEADERS}")
    env = dict(os.environ, CPATH=str(SDK_HEADERS))
    command = ["g++", str(UPSTREAM / "DREAMwalk/HeterogeneousSG.cpp"), "-o", "HetSG.build"] + COMPILE_FLAGS
    print("Compiling the unchanged upstream C++ once", flush=True)
    subprocess.run(command, cwd=data, env=env, check=True)
    (data / "HetSG.build").replace(binary)
    save_json(marker, dict(fingerprint=fingerprint, sha256=previous.sha256(binary), command=command,
                           compiler=subprocess.check_output(["g++", "--version"], text=True),
                           cpath=str(SDK_HEADERS)))


def generate_corpus(data, results, fingerprint):
    from DREAMwalk import generate_embeddings as ge
    corpus, marker = data / "walks.txt", results / "walks_complete.json"
    if completed(corpus, marker, fingerprint):
        meta = json.loads(marker.read_text())
        if previous.sha256(data / "token_to_node.json") != meta["token_map_sha256"]:
            raise RuntimeError("Saved token map changed")
        return
    require_checkpoint(data / "HetSG", results / "compiler_complete.json", fingerprint)
    if any((results / f"pp{pp}_embedding_complete.json").exists() for pp in (1, 0)):
        raise RuntimeError("Existing embeddings cannot be paired with a regenerated corpus; use a new RUN_NAME")
    started = time.monotonic()
    # Same upstream call sequence as save_embedding_files, stopped before SG.
    ge.set_seed(0)
    graph = ge.read_graph(str(previous.GRAPH), weighted=True, directed=False, delimiter="\t")
    semantic = ge.read_graph(str(previous.SIMILARITY), weighted=True, directed=False)
    t0 = time.monotonic()
    print("Starting edge-type transition training (length=10)", flush=True)
    matrix = ge.train_edgetype_transition_matrix(5, graph, str(previous.GRAPH), "\t", 10, 1, 1)
    em_seconds = time.monotonic() - t0
    t0 = time.monotonic()
    print("Starting full no-teleport walks: 100 x 29,959, length=10", flush=True)
    walks = ge.generate_DREAMwalk_paths(graph, semantic, matrix, 1, 1, 100, 10, 0, 4)
    walk_seconds = time.monotonic() - t0
    if len(walks) != 2995900:
        raise RuntimeError(f"Expected 2,995,900 walks; got {len(walks)}")
    # Match upstream's set-based numbering and type prefixes, but create the map
    # only once and retain exactly the same bytes for both SG training processes.
    node_to_id = {node: str(i) for i, node in enumerate(set(graph.nodes()))}
    types = {r["node"]: r["type"] for r in previous.read_rows(previous.NODETYPES)}
    prefixes = {"drug": "d", "disease": "i", "gene": "g", "etc": "e"}
    tokens = {node: prefixes.get(types.get(node), "e") + index for node, index in node_to_id.items()}
    if len(tokens) != 29959:
        raise RuntimeError("Unexpected graph vocabulary size")
    temp = data / "walks.txt.tmp"
    with temp.open("w", encoding="utf-8") as stream:
        for i, walk in enumerate(walks):
            if i:
                stream.write("\n")
            stream.write(" ".join(tokens[node] for node in walk))
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(corpus)
    save_json(data / "token_to_node.json", {token: node for node, token in tokens.items()})
    save_json(results / "edge_transition_matrix.json", matrix.tolist())
    save_json(marker, dict(fingerprint=fingerprint, sha256=previous.sha256(corpus),
                           token_map_sha256=previous.sha256(data / "token_to_node.json"),
                           walks=len(walks), nodes=len(tokens), walk_length=10,
                           em_seconds=em_seconds, walk_seconds=walk_seconds,
                           total_seconds=time.monotonic()-started))
    print("SHARED CORPUS SAVED:", corpus, flush=True)


def parse_vectors(path, token_to_node):
    import numpy as np
    vectors = {}
    with path.open() as stream:
        header = stream.readline().split()
        if len(header) != 2 or int(header[1]) != 128:
            raise ValueError("Invalid C++ embedding header")
        row_count = 0
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            row_count += 1
            if fields[0] == "</s>":
                continue
            node = token_to_node[fields[0]]
            if node in vectors or len(fields) != 129:
                raise ValueError("Duplicate node or incorrect vector dimension")
            vectors[node] = np.asarray(fields[1:], dtype=np.float32)
        if row_count != int(header[0]):
            raise ValueError("Truncated C++ embedding output")
    if set(vectors) != set(token_to_node.values()) or any(not np.isfinite(v).all() for v in vectors.values()):
        raise ValueError("Missing or non-finite node embeddings")
    return vectors


def train(pp, data, results, fingerprint):
    output, marker = data / f"pp{pp}.pkl", results / f"pp{pp}_embedding_complete.json"
    if completed(output, marker, fingerprint):
        return
    if output.exists():
        raise RuntimeError(f"Embedding exists without completion record; preserve for recovery: {output}")
    corpus_meta = require_checkpoint(data / "walks.txt", results / "walks_complete.json", fingerprint)
    compiler_meta = require_checkpoint(data / "HetSG", results / "compiler_complete.json", fingerprint)
    map_path = data / "token_to_node.json"
    if previous.sha256(map_path) != corpus_meta["token_map_sha256"]:
        raise RuntimeError("Saved token map changed")
    # Short relative filenames avoid upstream's fixed-size C string buffers.
    # No Python wrapper removes type prefixes when pp=0.
    command = ["./HetSG", "-train", "walks.txt", "-output", f"pp{pp}", "-pp", str(pp)] + SG_ARGS
    started = time.monotonic()
    print("Training:", " ".join(command), flush=True)
    subprocess.run(command, cwd=data, check=True)
    vectors = parse_vectors(data / f"pp{pp}.txt", json.loads(map_path.read_text()))
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(vectors, stream, protocol=4)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    save_json(marker, dict(fingerprint=fingerprint, sha256=previous.sha256(output), pp=pp,
                           corpus_sha256=corpus_meta["sha256"], binary_sha256=compiler_meta["sha256"],
                           token_map_sha256=corpus_meta["token_map_sha256"],
                           command=command, nodes=len(vectors), dimensions=128,
                           seconds=time.monotonic()-started))
    print("EMBEDDING SAVED:", output, flush=True)


def evaluate(pp, data, results, fingerprint):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    output, marker = results / f"pp{pp}_summary.csv", results / f"pp{pp}_evaluation_complete.json"
    if completed(output, marker, fingerprint):
        return
    require_checkpoint(data / f"pp{pp}.pkl", results / f"pp{pp}_embedding_complete.json", fingerprint)
    types, _, _, positives = previous.base_inputs()
    negatives = [(r["drug"], r["disease"]) for r in previous.read_rows(previous.NEGATIVES, ",")]
    pairs = sorted(positives) + negatives
    vectors = previous.load_embedding(data / f"pp{pp}.pkl", types, pairs)
    labels = np.asarray([1]*len(positives) + [0]*len(negatives))
    assignments = np.empty(len(pairs), dtype=np.int32)
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(pairs)), labels), 1):
        assignments[test] = fold
    pair_rows = [dict(pair_index=i, drug=a, disease=b, label=int(labels[i]), test_fold=int(assignments[i]))
                 for i, (a, b) in enumerate(pairs)]
    fold_path = results / "fold_assignments.csv"
    if fold_path.exists():
        old = previous.read_rows(fold_path, ",")
        if old != [{k: str(v) for k, v in row.items()} for row in pair_rows]:
            raise RuntimeError("Pair/fold assignments differ between branches")
    else:
        previous.write_csv(fold_path, pair_rows, list(pair_rows[0]))
    started = time.monotonic()
    rows = previous.evaluate(vectors, positives, negatives, 0, "no_teleport")
    for row in rows:
        row.update(label=f"same_walks_pp{pp}", embedding=f"no_teleport_length10_pp{pp}", pp=pp)
    summary = previous.metric_summary(rows, 0, f"same_walks_pp{pp}")
    previous.write_csv(results / f"pp{pp}_fold_results.csv", rows, list(rows[0]))
    previous.write_csv(output, summary, list(summary[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=previous.sha256(output), pp=pp,
                           fold_assignments_sha256=previous.sha256(fold_path),
                           seconds=time.monotonic()-started))


def compare(data, results, fingerprint):
    summaries, embeddings, evaluations = {}, {}, {}
    for pp in (1, 0):
        summary = results / f"pp{pp}_summary.csv"
        evaluations[pp] = require_checkpoint(summary, results / f"pp{pp}_evaluation_complete.json", fingerprint)
        embeddings[pp] = require_checkpoint(data / f"pp{pp}.pkl", results / f"pp{pp}_embedding_complete.json", fingerprint)
        summaries[pp] = {r["metric"]: float(r["mean"]) for r in previous.read_rows(summary, ",")}
    for key in ("corpus_sha256", "binary_sha256", "token_map_sha256"):
        if embeddings[1][key] != embeddings[0][key]:
            raise RuntimeError(f"Comparison inputs differ: {key}")
    if evaluations[1]["fold_assignments_sha256"] != evaluations[0]["fold_assignments_sha256"]:
        raise RuntimeError("Fold assignments differ")
    history = {r["metric"]: float(r["mean"]) for r in previous.read_rows(previous.CONTROL_SUMMARY, ",")}
    paper = {"accuracy": None, "auroc": 0.905, "aupr": 0.912}
    rows = [dict(metric=metric, same_walks_pp1=summaries[1][metric], same_walks_pp0=summaries[0][metric],
                 pp0_minus_pp1=summaries[0][metric]-summaries[1][metric],
                 historical_length10_pp1=history[metric], paper_no_teleport_approx=paper[metric])
            for metric in previous.METRICS]
    previous.write_csv(results / "pp_comparison.csv", rows, list(rows[0]))
    one = previous.read_rows(results / "pp1_fold_results.csv", ",")
    zero = previous.read_rows(results / "pp0_fold_results.csv", ",")
    paired = [dict(fold=a["fold"], **{metric + "_pp0_minus_pp1": float(b[metric])-float(a[metric])
                                    for metric in previous.METRICS}) for a, b in zip(one, zero)]
    previous.write_csv(results / "paired_fold_deltas.csv", paired, list(paired[0]))
    save_json(results / "comparison_complete.json", dict(fingerprint=fingerprint,
              sha256=previous.sha256(results / "pp_comparison.csv"),
              shared_corpus_sha256=embeddings[1]["corpus_sha256"],
              shared_binary_sha256=embeddings[1]["binary_sha256"],
              shared_fold_assignments_sha256=evaluations[1]["fold_assignments_sha256"]))
    print("PAIRED COMPARISON COMPLETE\n" + json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "compile", "walks", "pp1", "pp0", "evaluate-pp1", "evaluate-pp0", "compare"))
    parser.add_argument("--run-name", default="msi_seed0_skipgram_pp_comparison")
    args = parser.parse_args()
    data, results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    elif args.phase == "compile":
        compile_binary(data, results, fingerprint)
    elif args.phase == "walks":
        generate_corpus(data, results, fingerprint)
    elif args.phase in ("pp1", "pp0"):
        train(int(args.phase[-1]), data, results, fingerprint)
    elif args.phase.startswith("evaluate-"):
        evaluate(int(args.phase[-1]), data, results, fingerprint)
    else:
        compare(data, results, fingerprint)


if __name__ == "__main__":
    main()
