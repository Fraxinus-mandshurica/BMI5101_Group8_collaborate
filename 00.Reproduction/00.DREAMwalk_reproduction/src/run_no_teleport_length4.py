"""One local no-teleport length-4 diagnostic, launched by the notebook.

Uses unmodified upstream generation and the existing ten-fold evaluator.
Only walk_length changes from the historical length-10 no-teleport control.
Importing this module never launches walks, embedding, or evaluation.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import diagnose_msi_repeat0 as previous

UPSTREAM = previous.UPSTREAM
PINNED_COMMIT = "2f93723c3fbfa899d0f23c02e3459de8bfac0581"
PARAMETERS = dict(tp_factor=0, seed=0, num_walks=100, walk_length=4,
                  dimension=128, window_size=4, workers=4, em_max_iter=5,
                  p=1, q=1, directed=False, weighted=True)
PINS = {"numpy": "1.23.3", "scipy": "1.9.1", "scikit-learn": "1.1.3",
        "networkx": "2.8.8", "pandas": "1.4.4", "tqdm": "4.64.1",
        "parmap": "1.6.0", "xgboost": "1.7.4"}


def save_json(path, data):
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def context(run_name):
    import hashlib
    import subprocess
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("Use letters, digits, underscore and hyphen in RUN_NAME")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Select the existing DREAMwalk Python 3.8 interpreter in the notebook configuration")
    packages = {name: version(name) for name in PINS}
    if packages != PINS:
        raise RuntimeError(f"The existing DREAMwalk environment pins are required: {packages}")
    head = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if head != PINNED_COMMIT:
        raise RuntimeError(f"Unexpected upstream commit: {head}")
    paths = dict(graph=previous.GRAPH, similarity=previous.SIMILARITY,
                 nodetypes=previous.NODETYPES, positives=previous.POSITIVES,
                 negatives=previous.NEGATIVES, baseline_summary=previous.CONTROL_SUMMARY)
    # Bind checkpoints to this single configuration; never mix changed settings
    # with an already generated embedding or overwrite the historical control.
    identity = dict(parameters=PARAMETERS, xgboost=previous.xgb_settings(),
                    folds=10, cv_seed=0, feature="drug_embedding - disease_embedding",
                    input_sha256={k: previous.sha256(p) for k, p in paths.items()},
                    source_sha256={name: previous.sha256(path) for name, path in {
                        "runner": Path(__file__), "evaluator": Path(previous.__file__),
                        "base_runner": ROOT / "src/run_msi_repeat0.py",
                        "upstream_walks": UPSTREAM / "DREAMwalk/generate_embeddings.py",
                        "upstream_skipgram": UPSTREAM / "DREAMwalk/HeterogeneousSG.cpp",
                        "upstream_skipgram_wrapper": UPSTREAM / "DREAMwalk/HeterogeneousSG.py",
                    }.items()}, packages=packages, upstream_commit=head)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    result = ROOT / "results" / run_name
    result.mkdir(parents=True, exist_ok=True)
    embedding = ROOT / "data/embeddings" / f"{run_name}.pkl"
    embedding.parent.mkdir(parents=True, exist_ok=True)
    config_path = result / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("Inputs/code/settings changed; use a new RUN_NAME to preserve existing results")
    else:
        if embedding.exists():
            raise RuntimeError(f"Unregistered embedding already exists: {embedding}; use a new RUN_NAME")
        save_json(config_path, dict(identity, fingerprint=fingerprint,
                  analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                  changed_parameter="walk_length: 10 -> 4 in BOTH EM and final walks",
                  pair_order="sorted positives + original saved seed-0 negative order",
                  fold_origin="deterministically regenerated from existing ten-fold implementation",
                  input_paths={k: str(p) for k, p in paths.items()},
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  multiprocessing_start_method=multiprocessing.get_context().get_start_method(),
                  note="No start-method or seed-wrapper override. Upstream set_seed(0) is unchanged. Worker Python/NumPy RNG state is not asserted.",
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return result, embedding, fingerprint


def completed(path, marker, fingerprint):
    if not marker.exists():
        return False
    meta = json.loads(marker.read_text())
    if meta["fingerprint"] != fingerprint or not path.exists() or meta["sha256"] != previous.sha256(path):
        raise RuntimeError(f"Checkpoint changed or missing: {path}")
    print("Reusing completed output:", path, flush=True)
    return True


def embed(result, output, fingerprint):
    from DREAMwalk import generate_embeddings as ge
    marker = result / "embedding_complete.json"
    if completed(output, marker, fingerprint):
        return
    if output.exists():
        # A prior interrupt may have happened after the final rename but before
        # the completion record; preserve it instead of silently training again.
        raise RuntimeError(f"Embedding exists without completion record: {output}; preserve it for recovery")
    sdk_headers = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk/usr/include/c++/v1")
    if not sdk_headers.is_dir():
        raise FileNotFoundError(f"Previously used compiler headers are unavailable: {sdk_headers}")
    os.environ["CPATH"] = str(sdk_headers)
    work_root = ROOT / "logs" / result.name
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="embedding_work_", dir=str(work_root)))
    (work / "DREAMwalk").symlink_to(UPSTREAM / "DREAMwalk", target_is_directory=True)
    temporary_embedding = work / "embedding_pending.pkl"
    stage = {}

    def timed(name, original):
        def wrapper(*args, **kwargs):
            print("Starting", name, flush=True)
            started = time.monotonic()
            value = original(*args, **kwargs)
            stage[name + "_seconds"] = time.monotonic() - started
            if name == "random_walk_generation":
                stage["actual_walks"] = len(value)
                if len(value) != 2995900:
                    raise RuntimeError(f"Unexpected full walk count: {len(value)}")
                print("Actual walks:", len(value), flush=True)
            print(name, "seconds:", round(stage[name + "_seconds"], 2), flush=True)
            return value
        return wrapper

    ge.train_edgetype_transition_matrix = timed("edge_transition_training", ge.train_edgetype_transition_matrix)
    ge.generate_DREAMwalk_paths = timed("random_walk_generation", ge.generate_DREAMwalk_paths)
    ge.HeterogeneousSG = timed("heterogeneous_skipgram", ge.HeterogeneousSG)
    os.chdir(work)
    started = time.monotonic()
    ge.save_embedding_files(netf=str(previous.GRAPH), sim_netf=str(previous.SIMILARITY),
                            nodetypef=str(previous.NODETYPES), outputf=str(temporary_embedding),
                            **PARAMETERS)
    types, _, _, positives = previous.base_inputs()
    vectors = previous.load_embedding(temporary_embedding, types, sorted(positives))
    # Save the expensive result before any classifier training is started.
    temporary_embedding.replace(output)
    save_json(marker, dict(fingerprint=fingerprint, sha256=previous.sha256(output),
                          nodes=len(vectors), dimensions=128, **stage,
                          embedding_total_seconds=time.monotonic()-started,
                          embedding_path=str(output), work_directory=str(work)))
    print("EMBEDDING SAVED:", output, flush=True)


def evaluate(result, embedding, fingerprint):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    summary_path = result / "summary.csv"
    marker = result / "evaluation_complete.json"
    if completed(summary_path, marker, fingerprint):
        return
    if not completed(embedding, result / "embedding_complete.json", fingerprint):
        raise RuntimeError("Run the embedding section first")
    types, _, _, positives = previous.base_inputs()
    negatives = [(r["drug"], r["disease"]) for r in previous.read_rows(previous.NEGATIVES, ",")]
    pairs = sorted(positives) + negatives
    vectors = previous.load_embedding(embedding, types, pairs)
    # Save the same split IDs for convenient subsequent comparison.
    labels = np.array([1] * len(positives) + [0] * len(negatives))
    assignments = np.empty(len(pairs), dtype=np.int32)
    for fold, (_, test) in enumerate(StratifiedKFold(n_splits=10, shuffle=True, random_state=0).split(np.zeros(len(pairs)), labels), 1):
        assignments[test] = fold
    pair_rows = [dict(pair_index=i, drug=a, disease=b, label=int(labels[i]), test_fold=int(assignments[i]))
                 for i, (a, b) in enumerate(pairs)]
    previous.write_csv(result / "fold_assignments.csv", pair_rows, list(pair_rows[0]))
    started = time.monotonic()
    # Exact existing XGBoost estimator, probability rounding, metrics and folds.
    rows = previous.evaluate(vectors, positives, negatives, 0, "no_teleport")
    for row in rows:
        row.update(label="no_teleport_length4", embedding="no_teleport_tau_0_length4", walk_length=4)
    summary = previous.metric_summary(rows, 0, "no_teleport_length4")
    previous.write_csv(result / "fold_results.csv", rows, list(rows[0]))
    previous.write_csv(summary_path, summary, list(summary[0]))
    baseline = {r["metric"]: float(r["mean"]) for r in previous.read_rows(previous.CONTROL_SUMMARY, ",")}
    paper = {"accuracy": None, "auroc": 0.905, "aupr": 0.912}
    comparison = []
    for row in summary:
        metric, current = row["metric"], row["mean"]
        comparison.append(dict(metric=metric, no_teleport_length10=baseline[metric],
                               no_teleport_length4=current, length4_minus_length10=current-baseline[metric],
                               paper_no_teleport_approx=paper[metric],
                               length4_minus_paper_approx=current-paper[metric] if paper[metric] is not None else None))
    previous.write_csv(result / "comparison_length4_vs_length10.csv", comparison, list(comparison[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=previous.sha256(summary_path),
                          folds=10, seed=0, seconds=time.monotonic()-started))
    print("EVALUATION COMPLETE\n" + json.dumps(comparison, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "embedding", "evaluate"))
    parser.add_argument("--run-name", default="msi_seed0_no_teleport_length4")
    args = parser.parse_args()
    result, output, fingerprint = context(args.run_name)
    if args.phase == "config":
        print((result / "run_config.json").read_text(), flush=True)
    elif args.phase == "embedding":
        embed(result, output, fingerprint)
    else:
        evaluate(result, output, fingerprint)


if __name__ == "__main__":
    main()
