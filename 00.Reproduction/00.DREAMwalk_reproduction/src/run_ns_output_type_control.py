"""User-run diagnostic: output-type NS on the saved no-teleport corpus.

Keep the upstream source read-only. Change only the five pool/type endpoint
references inside pp=1 in a separate C++ source copy. This is a formula-motivated
diagnostic alternative, not a claimed DREAMwalk implementation correction.
No EM, walks, embedding training or classifier fitting runs on import.
"""
from __future__ import annotations

import argparse
import difflib
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
RUN_NAME = "msi_seed0_ns_output_type_control"
RUN_SETTINGS = dict(pp=1, tp_factor=0, seed=0, num_walks=100, walk_length=10,
                    dimension=128, window_size=4, workers=4, epochs=1,
                    negative=5, sample=0.001, folds=10, cv_seed=0,
                    source_walks=2995900, source_nodes=29959)
REFERENCE_NAME = "msi_seed0_skipgram_pp_comparison"
REFERENCE_DATA = ROOT / "data/embeddings" / REFERENCE_NAME
REFERENCE_RESULTS = ROOT / "results" / REFERENCE_NAME
CPP_SOURCE = paired.UPSTREAM / "DREAMwalk/HeterogeneousSG.cpp"
PATCHED_NAME = "HeterogeneousSG_output_type.cpp"
METRICS = ("accuracy", "auroc", "aupr")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_equal(actual, expected, description):
    if actual != expected:
        raise RuntimeError(description + " differs from the saved control")


def patch_source(source):
    """Modify only pp=1's four pool selectors and its matching type check.

    In this C++ implementation syn0's input is last_word; syn1neg's positive
    output is word. Do not change the vector roles, pp=0, the positive target,
    or collision/rejection handling.
    """
    start_marker = "            if (pp) {\n"
    end_marker = "            } else {\n"
    if source.count(start_marker) != 1:
        raise ValueError("Expected exactly one pp=1 negative-sampling branch")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    branch = source[start:end]
    endpoint = "vocab[last_word].word[0]"
    if branch.count(endpoint) != 5:
        raise ValueError("Expected four type-pool selectors and one type check")
    if "if (target == word)" in branch:
        raise ValueError("Upstream pp=1 rejection behavior changed")
    replacement = branch.replace(endpoint, "vocab[word].word[0]")
    patched = source[:start] + replacement + source[end:]
    difference = "".join(difflib.unified_diff(
        source.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile="vendor/DREAMwalk_upstream/DREAMwalk/HeterogeneousSG.cpp",
        tofile=PATCHED_NAME))
    return patched, difference


def patch_identity():
    original = CPP_SOURCE.read_bytes()
    patched, difference = patch_source(original.decode("utf-8"))
    return dict(original_source_sha256=hashlib.sha256(original).hexdigest(),
                patched_source_sha256=hashlib.sha256(patched.encode("utf-8")).hexdigest(),
                patch_diff_sha256=hashlib.sha256(difference.encode("utf-8")).hexdigest(),
                replacements=5, original_pool="input node type (last_word)",
                diagnostic_pool="positive context/output node type (word)",
                positive_output_filter="unchanged: no target == word filter in pp=1")


def inspect_reference():
    """Validate only read-only reference artifacts; never rerun its trainer."""
    config = read_json(REFERENCE_RESULTS / "run_config.json")
    corpus = read_json(REFERENCE_RESULTS / "walks_complete.json")
    compiler = read_json(REFERENCE_RESULTS / "compiler_complete.json")
    embedding = read_json(REFERENCE_RESULTS / "pp1_embedding_complete.json")
    evaluation = read_json(REFERENCE_RESULTS / "pp1_evaluation_complete.json")
    for marker in (corpus, compiler, embedding, evaluation):
        require_equal(marker["fingerprint"], config["fingerprint"], "Reference checkpoint")
    for key, value in (("walks", paired.WALKS), ("skipgram_args", paired.SG_ARGS),
                       ("compile_flags", paired.COMPILE_FLAGS),
                       ("xgboost", previous.xgb_settings()), ("folds", 10), ("cv_seed", 0)):
        require_equal(config[key], value, "Reference " + key)
    inputs = dict(graph=previous.GRAPH, similarity=previous.SIMILARITY,
                  nodetypes=previous.NODETYPES, positives=previous.POSITIVES,
                  negatives=previous.NEGATIVES)
    input_hashes = {name: sha256(path) for name, path in inputs.items()}
    for name, digest in input_hashes.items():
        require_equal(digest, config["inputs"][name], "Reference input " + name)
    paths = dict(corpus=REFERENCE_DATA / "walks.txt", token_map=REFERENCE_DATA / "token_to_node.json",
                 binary=REFERENCE_DATA / "HetSG", embedding=REFERENCE_DATA / "pp1.pkl",
                 summary=REFERENCE_RESULTS / "pp1_summary.csv",
                 fold_results=REFERENCE_RESULTS / "pp1_fold_results.csv",
                 fold_assignments=REFERENCE_RESULTS / "fold_assignments.csv")
    artifacts = {name: sha256(path) for name, path in paths.items()}
    for actual, expected, label in (
            (artifacts["corpus"], corpus["sha256"], "corpus"),
            (artifacts["corpus"], embedding["corpus_sha256"], "embedding corpus"),
            (artifacts["token_map"], corpus["token_map_sha256"], "corpus map"),
            (artifacts["token_map"], embedding["token_map_sha256"], "embedding map"),
            (artifacts["binary"], compiler["sha256"], "binary"),
            (artifacts["binary"], embedding["binary_sha256"], "embedding binary"),
            (artifacts["embedding"], embedding["sha256"], "embedding"),
            (artifacts["summary"], evaluation["sha256"], "summary"),
            (artifacts["fold_assignments"], evaluation["fold_assignments_sha256"], "fold assignments")):
        require_equal(actual, expected, "Reference " + label)
    require_equal(embedding["pp"], 1, "Reference pp")
    require_equal(embedding["command"], training_command(), "Reference training arguments")
    for key, value in (("walks", 2995900), ("nodes", 29959), ("walk_length", 10)):
        require_equal(corpus[key], value, "Reference corpus " + key)
    for relative, digest in config["sources"].items():
        require_equal(sha256(ROOT / relative), digest, "Historical source " + relative)
    records = ["run_config.json", "walks_complete.json", "compiler_complete.json",
               "pp1_embedding_complete.json", "pp1_evaluation_complete.json"]
    return dict(fingerprint=config["fingerprint"], inputs=input_hashes, artifacts=artifacts,
                records={name: sha256(REFERENCE_RESULTS / name) for name in records},
                source_paths={name: str(path.relative_to(ROOT)) for name, path in paths.items()})


def training_command():
    return ["./HetSG", "-train", "walks.txt", "-output", "pp1", "-pp", "1"] + paired.SG_ARGS


def context(run_name=RUN_NAME):
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                           for c in run_name):
        raise ValueError("RUN_NAME must contain only letters, digits, underscore or hyphen")
    if run_name == REFERENCE_NAME:
        raise ValueError("The diagnostic must not use the historical reference RUN_NAME")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Use the existing DREAMwalk Python 3.8 interpreter")
    packages = {name: version(name) for name in paired.PINS}
    require_equal(packages, paired.PINS, "DREAMwalk environment")
    commit = subprocess.check_output(["git", "-C", str(paired.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    require_equal(commit, paired.PINNED_COMMIT, "Upstream commit")
    identity = dict(settings=RUN_SETTINGS, reference=inspect_reference(), patch=patch_identity(),
                    skipgram_args=paired.SG_ARGS, compile_flags=paired.COMPILE_FLAGS,
                    compiler_headers=str(paired.SDK_HEADERS), xgboost=previous.xgb_settings(),
                    feature="drug_embedding - disease_embedding", packages=packages,
                    runner_sha256=sha256(Path(__file__)), upstream_commit=commit)
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
                  analysis="DIAGNOSTIC NS POOL ENDPOINT CHANGE; not a formal benchmark implementation",
                  comparison="output-type pp=1 minus original input-type pp=1 on identical no-teleport walks",
                  changed_stage="five pp=1 pool/type endpoint references: last_word -> word",
                  pair_order="sorted positives + original saved seed-0 negative order",
                  fold_origin="deterministically regenerated and checked against saved reference IDs before fitting",
                  notes=["Reuse the existing tau=0 corpus, node-token map and original pp=1 results.",
                         "Compile only an independent source copy; upstream vendor sources remain unchanged.",
                         "Do not add positive-output rejection or change the pp=0 branch.",
                         "Both binaries retain frequency^0.75, negative=5, sample=.001 and the same SG arguments.",
                         "Binary hashes intentionally differ; corpus, token-map and fold hashes must match.",
                         "No new EM, walks, link negatives or classifier settings.",
                         "Four C++ threads retain their existing nondeterminism."],
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def write_exact(path, content):
    if path.exists():
        require_equal(path.read_bytes(), content, "Existing " + path.name)
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def copy_exact(source, target, allow_hardlink=False):
    if target.exists():
        require_equal(sha256(target), sha256(source), "Existing " + target.name)
        return
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    if allow_hardlink:
        try:
            # Neither this module nor the trainer writes to the saved corpus.
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
    else:
        shutil.copy2(source, temporary)
    require_equal(sha256(temporary), sha256(source), "Copied " + target.name)
    temporary.replace(target)


def prepare(data, results, fingerprint):
    config = read_json(results / "run_config.json")
    require_equal(config["fingerprint"], fingerprint, "Run configuration")
    require_equal(patch_identity(), config["patch"], "Endpoint patch identity")
    patched, difference = patch_source(CPP_SOURCE.read_bytes().decode("utf-8"))
    write_exact(data / PATCHED_NAME, patched.encode("utf-8"))
    write_exact(data / "patch.diff", difference.encode("utf-8"))
    copy_exact(REFERENCE_DATA / "token_to_node.json", data / "token_to_node.json")
    copy_exact(REFERENCE_DATA / "walks.txt", data / "walks.txt", allow_hardlink=True)
    for name, path in (("corpus", data / "walks.txt"), ("token_map", data / "token_to_node.json")):
        require_equal(sha256(path), config["reference"]["artifacts"][name], "Prepared " + name)
    corpus_marker = results / "walks_complete.json"
    if not paired.completed(data / "walks.txt", corpus_marker, fingerprint):
        old = read_json(REFERENCE_RESULTS / "walks_complete.json")
        save_json(corpus_marker, dict(fingerprint=fingerprint, sha256=sha256(data / "walks.txt"),
                  token_map_sha256=sha256(data / "token_to_node.json"), walks=old["walks"],
                  nodes=old["nodes"], walk_length=old["walk_length"], tau=0,
                  reused_from=str((REFERENCE_DATA / "walks.txt").relative_to(ROOT))))
    binary, marker = data / "HetSG", results / "compiler_complete.json"
    if paired.completed(binary, marker, fingerprint):
        require_prepared(data, results, fingerprint)
        return
    if binary.exists():
        raise RuntimeError("Unregistered binary exists; preserve it and use a new RUN_NAME")
    if not paired.SDK_HEADERS.is_dir():
        raise FileNotFoundError("Previously used compiler headers unavailable: " + str(paired.SDK_HEADERS))
    command = ["g++", PATCHED_NAME, "-o", "HetSG.build"] + paired.COMPILE_FLAGS
    print("Compiling independent output-type NS source; vendor remains unchanged", flush=True)
    subprocess.run(command, cwd=data, env=dict(os.environ, CPATH=str(paired.SDK_HEADERS)), check=True)
    (data / "HetSG.build").replace(binary)
    save_json(marker, dict(config["patch"], fingerprint=fingerprint, sha256=sha256(binary),
              command=command, compiler=subprocess.check_output(["g++", "--version"], text=True),
              cpath=str(paired.SDK_HEADERS)))
    require_prepared(data, results, fingerprint)


def require_prepared(data, results, fingerprint):
    config = read_json(results / "run_config.json")
    require_equal(config["fingerprint"], fingerprint, "Run configuration")
    compiler = paired.require_checkpoint(data / "HetSG", results / "compiler_complete.json", fingerprint)
    corpus = paired.require_checkpoint(data / "walks.txt", results / "walks_complete.json", fingerprint)
    for key, path in (("original_source_sha256", CPP_SOURCE),
                      ("patched_source_sha256", data / PATCHED_NAME),
                      ("patch_diff_sha256", data / "patch.diff")):
        require_equal(sha256(path), config["patch"][key], "Prepared source " + key)
        require_equal(compiler[key], config["patch"][key], "Compiled source " + key)
    for name, digest in (("corpus", corpus["sha256"]),
                         ("token_map", sha256(data / "token_to_node.json"))):
        require_equal(digest, config["reference"]["artifacts"][name], "Shared " + name)
    require_equal(corpus["token_map_sha256"], sha256(data / "token_to_node.json"), "Prepared corpus map")
    return corpus, compiler


def embedding(data, results, fingerprint):
    require_prepared(data, results, fingerprint)
    paired.train(1, data, results, fingerprint)
    require_embedding(data, results, fingerprint)
    # paired.train atomically persists pp1.pkl and its hash before this phase ends.
    print("Output-type embedding persisted; evaluation can run independently", flush=True)


def require_embedding(data, results, fingerprint):
    corpus, compiler = require_prepared(data, results, fingerprint)
    meta = paired.require_checkpoint(data / "pp1.pkl", results / "pp1_embedding_complete.json", fingerprint)
    for key, expected in (("corpus_sha256", corpus["sha256"]),
                          ("binary_sha256", compiler["sha256"]),
                          ("token_map_sha256", corpus["token_map_sha256"]),
                          ("command", training_command()), ("pp", 1)):
        require_equal(meta[key], expected, "Diagnostic embedding " + key)
    return meta


def require_evaluation(data, results, fingerprint):
    embedded = require_embedding(data, results, fingerprint)
    meta = paired.require_checkpoint(results / "pp1_summary.csv", results / "pp1_evaluation_complete.json", fingerprint)
    require_equal(meta["embedding_sha256"], embedded["sha256"], "Evaluated embedding")
    require_equal(meta["fold_results_sha256"], sha256(results / "pp1_fold_results.csv"), "Fold results")
    folds = sha256(results / "fold_assignments.csv")
    require_equal(folds, meta["fold_assignments_sha256"], "Saved fold IDs")
    require_equal(folds, sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Reference fold IDs")
    return embedded, meta


def evaluate(data, results, fingerprint):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    embedded = require_embedding(data, results, fingerprint)
    output, marker = results / "pp1_summary.csv", results / "pp1_evaluation_complete.json"
    if paired.completed(output, marker, fingerprint):
        require_evaluation(data, results, fingerprint)
        return
    types, _, _, positives = previous.base_inputs()
    negatives = [(row["drug"], row["disease"]) for row in previous.read_rows(previous.NEGATIVES, ",")]
    pairs = sorted(positives) + negatives
    labels = np.asarray([1]*len(positives) + [0]*len(negatives))
    assignments = np.empty(len(pairs), dtype=np.int32)
    for fold, (_, test) in enumerate(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(
            np.zeros(len(pairs)), labels), 1):
        assignments[test] = fold
    rows = [dict(pair_index=i, drug=a, disease=b, label=int(labels[i]), test_fold=int(assignments[i]))
            for i, (a, b) in enumerate(pairs)]
    expected = previous.read_rows(REFERENCE_RESULTS / "fold_assignments.csv", ",")
    require_equal([{key: str(value) for key, value in row.items()} for row in rows], expected,
                  "Pair/fold assignments before fitting")
    fold_path = results / "fold_assignments.csv"
    if fold_path.exists():
        require_equal(previous.read_rows(fold_path, ","), expected, "Existing fold assignments")
    else:
        previous.write_csv(fold_path, rows, list(rows[0]))
    require_equal(sha256(fold_path), sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Fold CSV bytes")
    vectors = previous.load_embedding(data / "pp1.pkl", types, pairs)
    started = time.monotonic()
    rows = previous.evaluate(vectors, positives, negatives, 0, "no_teleport")
    for row in rows:
        row.update(label="no_teleport_output_type_pp1", embedding="no_teleport_output_type_pp1", pp=1)
    summary = previous.metric_summary(rows, 0, "no_teleport_output_type_pp1")
    fold_results = results / "pp1_fold_results.csv"
    previous.write_csv(fold_results, rows, list(rows[0]))
    previous.write_csv(output, summary, list(summary[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output), pp=1,
              embedding_sha256=embedded["sha256"], fold_assignments_sha256=sha256(fold_path),
              fold_results_sha256=sha256(fold_results), seconds=time.monotonic()-started))


def comparison_rows(reference, current, reference_folds, current_folds):
    baseline, control = [{row["metric"]: float(row["mean"]) for row in rows}
                         for rows in (reference, current)]
    for values in (baseline, control):
        require_equal(set(values), set(METRICS), "Summary metrics")
    summary = [dict(metric=metric, input_type_pp1=baseline[metric], output_type_pp1=control[metric],
                    output_minus_input=control[metric]-baseline[metric]) for metric in METRICS]
    by_fold = [{int(row["fold"]): row for row in rows} for rows in (reference_folds, current_folds)]
    if any(len(rows) != 10 for rows in (reference_folds, current_folds)) or any(
            set(rows) != set(range(1, 11)) for rows in by_fold):
        raise ValueError("Expected exactly ten matching folds")
    deltas = [dict(fold=fold, **{metric + "_output_minus_input":
                 float(by_fold[1][fold][metric])-float(by_fold[0][fold][metric]) for metric in METRICS})
              for fold in range(1, 11)]
    return summary, deltas


def compare(data, results, fingerprint):
    embedded, evaluation = require_evaluation(data, results, fingerprint)
    reference = inspect_reference()
    config = read_json(results / "run_config.json")
    require_equal(reference, config["reference"], "Reference identity")
    original = read_json(REFERENCE_RESULTS / "pp1_embedding_complete.json")
    for key in ("corpus_sha256", "token_map_sha256", "command", "pp", "nodes", "dimensions"):
        require_equal(embedded[key], original[key], "Paired embedding " + key)
    require_equal(evaluation["fold_assignments_sha256"], reference["artifacts"]["fold_assignments"], "Paired folds")
    require_equal(sha256(data / "HetSG"), embedded["binary_sha256"], "Diagnostic binary")
    # Different binary hashes are intentional: only the endpoint patch differs.
    output, marker = results / "ns_pool_comparison.csv", results / "comparison_complete.json"
    if paired.completed(output, marker, fingerprint):
        require_equal(read_json(marker)["paired_fold_deltas_sha256"], sha256(results / "paired_fold_deltas.csv"),
                      "Saved paired deltas")
        return
    summary, deltas = comparison_rows(
        previous.read_rows(REFERENCE_RESULTS / "pp1_summary.csv", ","),
        previous.read_rows(results / "pp1_summary.csv", ","),
        previous.read_rows(REFERENCE_RESULTS / "pp1_fold_results.csv", ","),
        previous.read_rows(results / "pp1_fold_results.csv", ","))
    previous.write_csv(output, summary, list(summary[0]))
    delta_path = results / "paired_fold_deltas.csv"
    previous.write_csv(delta_path, deltas, list(deltas[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output),
              paired_fold_deltas_sha256=sha256(delta_path), reference_run=REFERENCE_NAME,
              shared_corpus_sha256=embedded["corpus_sha256"], shared_token_map_sha256=embedded["token_map_sha256"],
              shared_fold_assignments_sha256=evaluation["fold_assignments_sha256"],
              original_binary_sha256=original["binary_sha256"], diagnostic_binary_sha256=embedded["binary_sha256"],
              patched_source_sha256=config["patch"]["patched_source_sha256"],
              patch_diff_sha256=config["patch"]["patch_diff_sha256"]))
    print("NS POOL COMPARISON COMPLETE\n" + json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "prepare", "embedding", "evaluate", "compare"))
    parser.add_argument("--run-name", default=RUN_NAME)
    args = parser.parse_args()
    data, results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    else:
        globals()[args.phase](data, results, fingerprint)


if __name__ == "__main__":
    main()
