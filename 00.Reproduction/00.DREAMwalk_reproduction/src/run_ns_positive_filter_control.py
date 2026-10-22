"""User-run NS collision control on the completed output-type, tau=0 corpus.

Add one positive-output rejection to an independent copy of the completed
output-type pp=1 source. Skip a colliding draw without resampling. Preserve
the original sources, saved corpus, embeddings, negatives and evaluation.
No compilation, model training or evaluation runs on import.
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
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_ns_output_type_control as baseline

paired = baseline.paired
previous = paired.previous
save_json = baseline.save_json
sha256 = previous.sha256
read_json = baseline.read_json
require_equal = baseline.require_equal
write_exact = baseline.write_exact
copy_exact = baseline.copy_exact
RUN_NAME = "msi_seed0_ns_positive_filter_control"
RUN_SETTINGS = dict(pp=1, tp_factor=0, seed=0, num_walks=100, walk_length=10,
                    dimension=128, window_size=4, workers=4, epochs=1,
                    negative=5, sample=0.001, folds=10, cv_seed=0,
                    reject_positive_output=True,
                    rejection_policy="skip_without_resampling")
REFERENCE_NAME = baseline.RUN_NAME
REFERENCE_DATA = ROOT / "data/embeddings" / REFERENCE_NAME
REFERENCE_RESULTS = ROOT / "results" / REFERENCE_NAME
CPP_SOURCE = REFERENCE_DATA / baseline.PATCHED_NAME
PATCHED_NAME = "HeterogeneousSG_output_type_reject_positive.cpp"
METRICS = ("accuracy", "auroc", "aupr")


def patch_source(source):
    """Add only the current-positive-output check inside the pp=1 branch.

    This check sits in the existing fixed-count draw loop. Its continue skips
    this draw; it does not decrement d, retry, or inspect other positive pairs.
    The pp=0 branch and all five output-type endpoint references stay intact.
    """
    start_marker = "            if (pp) {\n"
    end_marker = "            } else {\n"
    if source.count(start_marker) != 1:
        raise ValueError("Expected exactly one pp=1 negative-sampling branch")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    branch = source[start:end]
    if branch.count("vocab[word].word[0]") != 5 or "vocab[last_word].word[0]" in branch:
        raise ValueError("Expected the completed output-type pp=1 source")
    if "if (target == word)" in branch:
        raise ValueError("The pp=1 source already rejects the positive output")
    lines = branch.splitlines(keepends=True)
    anchors = [index for index, line in enumerate(lines)
               if line.strip() == "if (vocab[word].word[0] != vocab[target].word[0]) continue;"]
    if len(anchors) != 1:
        raise ValueError("Expected exactly one output-type rejection check")
    lines.insert(anchors[0] + 1, "              if (target == word) continue;\n")
    patched = source[:start] + "".join(lines) + source[end:]
    difference = "".join(difflib.unified_diff(
        source.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile="data/embeddings/" + REFERENCE_NAME + "/" + baseline.PATCHED_NAME,
        tofile=PATCHED_NAME))
    return patched, difference


def patch_identity():
    original = CPP_SOURCE.read_bytes()
    patched, difference = patch_source(original.decode("utf-8"))
    return dict(reference_source_sha256=hashlib.sha256(original).hexdigest(),
                patched_source_sha256=hashlib.sha256(patched.encode("utf-8")).hexdigest(),
                patch_diff_sha256=hashlib.sha256(difference.encode("utf-8")).hexdigest(),
                added_lines=1, pool="positive context/output node type (word)",
                positive_output_filter="if (target == word) continue;",
                rejection_policy="skip_without_resampling")


def training_command():
    return ["./HetSG", "-train", "walks.txt", "-output", "pp1", "-pp", "1"] + paired.SG_ARGS


def inspect_reference():
    """Inspect completed baselines read-only; do not create or rerun any run."""
    config = read_json(REFERENCE_RESULTS / "run_config.json")
    require_equal(config["runner_sha256"], sha256(Path(baseline.__file__)), "Output-type runner")
    require_equal(config["settings"], baseline.RUN_SETTINGS, "Output-type settings")
    require_equal(config["reference"], baseline.inspect_reference(), "Output-type reference")
    require_equal(config["patch"], baseline.patch_identity(), "Output-type patch identity")
    for key, value in (("skipgram_args", paired.SG_ARGS), ("compile_flags", paired.COMPILE_FLAGS),
                       ("xgboost", previous.xgb_settings()), ("upstream_commit", paired.PINNED_COMMIT)):
        require_equal(config[key], value, "Output-type " + key)
    markers = {name: read_json(REFERENCE_RESULTS / name) for name in (
        "walks_complete.json", "compiler_complete.json", "pp1_embedding_complete.json",
        "pp1_evaluation_complete.json", "comparison_complete.json")}
    for marker in markers.values():
        require_equal(marker["fingerprint"], config["fingerprint"], "Output-type checkpoint")
    corpus, compiler = markers["walks_complete.json"], markers["compiler_complete.json"]
    embedded, evaluated = markers["pp1_embedding_complete.json"], markers["pp1_evaluation_complete.json"]
    paths = dict(corpus=REFERENCE_DATA / "walks.txt", token_map=REFERENCE_DATA / "token_to_node.json",
                 source=CPP_SOURCE, patch_diff=REFERENCE_DATA / "patch.diff", binary=REFERENCE_DATA / "HetSG",
                 embedding=REFERENCE_DATA / "pp1.pkl", summary=REFERENCE_RESULTS / "pp1_summary.csv",
                 fold_results=REFERENCE_RESULTS / "pp1_fold_results.csv",
                 fold_assignments=REFERENCE_RESULTS / "fold_assignments.csv",
                 comparison=REFERENCE_RESULTS / "ns_pool_comparison.csv",
                 paired_deltas=REFERENCE_RESULTS / "paired_fold_deltas.csv")
    artifacts = {name: sha256(path) for name, path in paths.items()}
    comparisons = (
        (artifacts["corpus"], corpus["sha256"], "corpus"),
        (artifacts["corpus"], embedded["corpus_sha256"], "embedding corpus"),
        (artifacts["token_map"], corpus["token_map_sha256"], "corpus map"),
        (artifacts["token_map"], embedded["token_map_sha256"], "embedding map"),
        (artifacts["source"], config["patch"]["patched_source_sha256"], "source"),
        (artifacts["patch_diff"], config["patch"]["patch_diff_sha256"], "patch diff"),
        (artifacts["binary"], compiler["sha256"], "binary"),
        (artifacts["binary"], embedded["binary_sha256"], "embedding binary"),
        (artifacts["embedding"], embedded["sha256"], "embedding"),
        (artifacts["embedding"], evaluated["embedding_sha256"], "evaluated embedding"),
        (artifacts["summary"], evaluated["sha256"], "summary"),
        (artifacts["fold_results"], evaluated["fold_results_sha256"], "fold results"),
        (artifacts["fold_assignments"], evaluated["fold_assignments_sha256"], "fold assignments"),
        (artifacts["fold_assignments"], config["reference"]["artifacts"]["fold_assignments"], "paired folds"),
        (artifacts["corpus"], config["reference"]["artifacts"]["corpus"], "shared corpus"),
        (artifacts["token_map"], config["reference"]["artifacts"]["token_map"], "shared map"),
        (artifacts["comparison"], markers["comparison_complete.json"]["sha256"], "comparison"),
        (artifacts["paired_deltas"], markers["comparison_complete.json"]["paired_fold_deltas_sha256"], "deltas"))
    for actual, expected, label in comparisons:
        require_equal(actual, expected, "Completed output-type " + label)
    for key in ("original_source_sha256", "patched_source_sha256", "patch_diff_sha256"):
        require_equal(compiler[key], config["patch"][key], "Compiled output-type source " + key)
    require_equal(embedded["command"], training_command(), "Output-type training arguments")
    require_equal(embedded["pp"], 1, "Output-type pp")
    require_equal(compiler["command"], ["g++", baseline.PATCHED_NAME, "-o", "HetSG.build"] +
                  paired.COMPILE_FLAGS, "Output-type compile arguments")
    records = ["run_config.json"] + list(markers)
    return dict(fingerprint=config["fingerprint"], inputs=config["reference"]["inputs"],
                artifacts=artifacts, records={name: sha256(REFERENCE_RESULTS / name) for name in records},
                source_paths={name: str(path.relative_to(ROOT)) for name, path in paths.items()})


def context(run_name=RUN_NAME):
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                           for c in run_name):
        raise ValueError("RUN_NAME must contain only letters, digits, underscore or hyphen")
    if run_name in (REFERENCE_NAME, baseline.REFERENCE_NAME):
        raise ValueError("The diagnostic must not use a completed baseline RUN_NAME")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Use the existing DREAMwalk Python 3.8 interpreter")
    packages = {name: version(name) for name in paired.PINS}
    require_equal(packages, paired.PINS, "DREAMwalk environment")
    commit = subprocess.check_output(["git", "-C", str(paired.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    require_equal(commit, paired.PINNED_COMMIT, "Upstream commit")
    source_paths = [Path(__file__), Path(baseline.__file__), Path(paired.__file__), Path(previous.__file__),
                    ROOT / "src/run_no_teleport_length4.py", ROOT / "src/run_msi_repeat0.py"]
    identity = dict(settings=RUN_SETTINGS, reference=inspect_reference(), patch=patch_identity(),
                    skipgram_args=paired.SG_ARGS, compile_flags=paired.COMPILE_FLAGS,
                    compiler_headers=str(paired.SDK_HEADERS), xgboost=previous.xgb_settings(),
                    feature="drug_embedding - disease_embedding", packages=packages,
                    sources={str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
                    upstream_commit=commit)
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
                  analysis="DIAGNOSTIC CURRENT-POSITIVE-OUTPUT REJECTION; not a formal benchmark implementation",
                  comparison="output-type filtered pp=1 minus completed output-type unfiltered pp=1",
                  changed_stage="one pp=1 line: if (target == word) continue;",
                  pair_order="sorted positives + original saved seed-0 negative order",
                  fold_origin="deterministically regenerated and checked against saved reference IDs before fitting",
                  notes=["Reuse completed output-type baseline; do not rerun its trainer or evaluation.",
                         "Corpus and token map are reused as read-only inputs; no new EM or walks.",
                         "Reject only the current positive output, not all known pairs or window contexts.",
                         "A rejected draw is skipped; do not retry or replenish five negative attempts.",
                         "Retain the output-type pool, frequency^0.75, negative=5, sample=.001 and all SG parameters.",
                         "Compile an independent source copy; upstream and historical runners remain unchanged.",
                         "The pp=0 branch is byte-for-byte unchanged.",
                         "Persist the embedding before evaluation; keep negatives, folds and XGBoost unchanged.",
                         "Four C++ threads retain their existing nondeterminism."],
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def prepare(data, results, fingerprint):
    config = read_json(results / "run_config.json")
    require_equal(config["fingerprint"], fingerprint, "Run configuration")
    require_equal(inspect_reference(), config["reference"], "Completed output-type baseline")
    require_equal(patch_identity(), config["patch"], "Positive-output rejection patch identity")
    patched, difference = patch_source(CPP_SOURCE.read_text(encoding="utf-8"))
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
    print("Compiling independent positive-output rejection control", flush=True)
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
    for key, path in (("reference_source_sha256", CPP_SOURCE), ("patched_source_sha256", data / PATCHED_NAME),
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
    print("Filtered output-type embedding persisted; evaluation can run independently", flush=True)


def require_embedding(data, results, fingerprint):
    corpus, compiler = require_prepared(data, results, fingerprint)
    meta = paired.require_checkpoint(data / "pp1.pkl", results / "pp1_embedding_complete.json", fingerprint)
    for key, expected in (("corpus_sha256", corpus["sha256"]), ("binary_sha256", compiler["sha256"]),
                          ("token_map_sha256", corpus["token_map_sha256"]),
                          ("command", training_command()), ("pp", 1)):
        require_equal(meta[key], expected, "Filtered embedding " + key)
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
        row.update(label="no_teleport_output_type_filtered_pp1",
                   embedding="no_teleport_output_type_filtered_pp1", pp=1)
    summary = previous.metric_summary(rows, 0, "no_teleport_output_type_filtered_pp1")
    fold_results = results / "pp1_fold_results.csv"
    previous.write_csv(fold_results, rows, list(rows[0]))
    previous.write_csv(output, summary, list(summary[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output), pp=1,
              embedding_sha256=embedded["sha256"], fold_assignments_sha256=sha256(fold_path),
              fold_results_sha256=sha256(fold_results), seconds=time.monotonic()-started))


def comparison_rows(reference, current, reference_folds, current_folds):
    unfiltered, filtered = [{row["metric"]: float(row["mean"]) for row in rows}
                            for rows in (reference, current)]
    for values in (unfiltered, filtered):
        require_equal(set(values), set(METRICS), "Summary metrics")
    summary = [dict(metric=metric, output_type_unfiltered=unfiltered[metric],
                    output_type_filtered=filtered[metric],
                    filtered_minus_unfiltered=filtered[metric]-unfiltered[metric]) for metric in METRICS]
    by_fold = [{int(row["fold"]): row for row in rows} for rows in (reference_folds, current_folds)]
    if any(len(rows) != 10 for rows in (reference_folds, current_folds)) or any(
            set(rows) != set(range(1, 11)) for rows in by_fold):
        raise ValueError("Expected exactly ten matching folds")
    deltas = [dict(fold=fold, **{metric + "_filtered_minus_unfiltered":
                 float(by_fold[1][fold][metric])-float(by_fold[0][fold][metric]) for metric in METRICS})
              for fold in range(1, 11)]
    return summary, deltas


def compare(data, results, fingerprint):
    embedded, evaluation = require_evaluation(data, results, fingerprint)
    reference = inspect_reference()
    config = read_json(results / "run_config.json")
    require_equal(reference, config["reference"], "Completed baseline identity")
    original = read_json(REFERENCE_RESULTS / "pp1_embedding_complete.json")
    for key in ("corpus_sha256", "token_map_sha256", "command", "pp", "nodes", "dimensions"):
        require_equal(embedded[key], original[key], "Paired embedding " + key)
    require_equal(evaluation["fold_assignments_sha256"], reference["artifacts"]["fold_assignments"], "Paired folds")
    output, marker = results / "ns_positive_filter_comparison.csv", results / "comparison_complete.json"
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
              reference_binary_sha256=original["binary_sha256"], diagnostic_binary_sha256=embedded["binary_sha256"],
              patched_source_sha256=config["patch"]["patched_source_sha256"],
              patch_diff_sha256=config["patch"]["patch_diff_sha256"],
              rejection_policy="skip_without_resampling"))
    print("NS POSITIVE-OUTPUT FILTER COMPARISON COMPLETE\n" + json.dumps(summary, indent=2), flush=True)


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
