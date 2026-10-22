"""DIAGNOSTIC / SENSITIVITY ANALYSIS: one full hash-seed-1 MSI run.

The optional environment restoration preserves the *startup* hash seed for
macOS spawn workers after unchanged upstream set_seed(0) resets os.environ.
No DREAMwalk walk, transition, teleport, or Skip-gram algorithm is changed.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import platform
import random
import resource
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/DREAMwalk_upstream"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "src"))

from run_msi_repeat0 import (  # noqa: E402
    GRAPH, NODETYPES, POSITIVES, NEGATIVES, SIMILARITY, HIERARCHY,
    EMBEDDING, CONFIG, FOLDS, SUMMARY, base_inputs, check_graph_leakage,
    read_rows, sha256, xgb_settings,
)
from diagnose_msi_repeat0 import evaluate, metric_summary, write_csv, write_json, load_embedding  # noqa: E402

RESULTS = ROOT / "results"
NEW_EMBEDDING = ROOT / "data/embeddings/msi_seed0_hashseed1.pkl"
RUN_CONFIG = RESULTS / "diagnostic_hashseed1_run_config.json"
TIMING = RESULTS / "diagnostic_hashseed1_timing.json"
NEW_FOLDS = RESULTS / "diagnostic_hashseed1_fold_results.csv"
NEW_SUMMARY = RESULTS / "diagnostic_hashseed1_summary.csv"
COMPARISON = RESULTS / "diagnostic_hashseed_full_comparison.csv"
EMBEDDING_COMPARISON = RESULTS / "diagnostic_hashseed_embedding_comparison.csv"
HASH0_FOLDS = RESULTS / "diagnostic_repeat0_10fold.csv"
HASH0_SUMMARY = RESULTS / "diagnostic_negative_sampling.csv"
FORMAL_FILES = (GRAPH, NODETYPES, POSITIVES, NEGATIVES, SIMILARITY,
                HIERARCHY, EMBEDDING, CONFIG, FOLDS, SUMMARY, HASH0_FOLDS, HASH0_SUMMARY)
METRICS = ("accuracy", "auroc", "aupr")


def baseline_hashes():
    return {str(path.relative_to(ROOT)): sha256(path) for path in FORMAL_FILES}


def check_baseline_hashes(expected):
    assert baseline_hashes() == expected, "A formal or hash0 baseline file changed"


def worker_fingerprint(_):
    ids = ["DB00985", "DB01076", "C0038362", "C0040997", "DB00001", "C0000737"]
    return dict(pid=os.getpid(), pythonhashseed=os.environ.get("PYTHONHASHSEED"),
                set_iteration=list(set(ids)))


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def checked_pairs():
    types, drugs, diseases, positives = base_inputs()
    graph_check = check_graph_leakage(types, positives)
    negatives = [(r["drug"], r["disease"]) for r in read_rows(NEGATIVES, ",")]
    assert len(negatives) == len(set(negatives)) == len(positives) == 5926
    assert not set(negatives) & positives
    return types, positives, negatives, graph_check


def phase_embedding(preserve_worker_hashseed):
    assert os.environ.get("PYTHONHASHSEED") == "1", "Start a fresh interpreter with PYTHONHASHSEED=1"
    assert not NEW_EMBEDDING.exists(), "Hashseed-1 embedding exists; refusing a second full run"
    assert not RUN_CONFIG.exists() and not TIMING.exists()
    from DREAMwalk import generate_embeddings as ge

    before = baseline_hashes()
    types, positives, negatives, graph_check = checked_pairs()
    formal = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert formal["dreamwalk"]["tp_factor"] == 0.3
    assert formal["dreamwalk"]["num_walks"] == 100
    assert formal["dreamwalk"]["walk_length"] == 10
    assert formal["dreamwalk"]["dimension"] == 128
    assert formal["dreamwalk"]["window_size"] == 4
    assert formal["dreamwalk"]["workers"] == 4
    assert formal["dreamwalk"]["em_max_iter"] == 5
    assert formal["dreamwalk"]["cutoff"] == 0.4
    assert formal["dreamwalk"]["p"] == formal["dreamwalk"]["q"] == 1
    assert formal["dreamwalk"]["em_edge_sample_fraction"] == 0.01
    assert formal["dreamwalk"]["em_convergence_threshold"] == 0.01
    assert formal["graph"]["graph_sha256"] == sha256(GRAPH)
    assert formal["dreamwalk"]["similarity_sha256"] == sha256(SIMILARITY)
    assert formal["dreamwalk"]["hierarchy_sha256"] == sha256(HIERARCHY)
    assert formal["evaluation"]["negative_pairs_sha256"] == sha256(NEGATIVES)
    assert graph_check["known_treatment_edges_in_graph"] == 0
    startup = dict(pythonhashseed=os.environ["PYTHONHASHSEED"],
                   python_version=sys.version, pid=os.getpid(),
                   parent_python_random_seed=0, parent_numpy_seed=0,
                   multiprocessing_start_method=mp.get_start_method())
    assert startup["multiprocessing_start_method"] == "spawn"

    original_set_seed = ge.set_seed
    def set_seed_with_worker_env(seed):
        assert seed == 0
        original_set_seed(seed)
        if preserve_worker_hashseed:
            # Parent hash secret was fixed at interpreter startup. This only
            # restores the same value for future spawn-worker startups.
            os.environ["PYTHONHASHSEED"] = startup["pythonhashseed"]
        print("POST-SET-SEED WORKER ENV", os.environ.get("PYTHONHASHSEED"), flush=True)
    ge.set_seed = set_seed_with_worker_env

    # Verify what a freshly spawned worker will actually inherit. This probe
    # occurs before save_embedding_files, which reseeds Python/NumPy itself.
    set_seed_with_worker_env(0)
    with mp.Pool(processes=2) as pool:
        worker_checks = pool.map(worker_fingerprint, range(8))
    expected_worker_hash = "1" if preserve_worker_hashseed else "0"
    assert all(check["pythonhashseed"] == expected_worker_hash for check in worker_checks)
    os.environ["CPATH"] = "/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk/usr/include/c++/v1"
    work = ROOT / "logs/diagnostic_hashseed1_work"
    work.mkdir(parents=True, exist_ok=True)
    link = work / "DREAMwalk"
    if not link.exists():
        link.symlink_to(UPSTREAM / "DREAMwalk", target_is_directory=True)

    run_config = dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                      purpose="full-model startup PYTHONHASHSEED sensitivity, not formal repeat 1",
                      execution_wrapper_change=(
                          "DIAGNOSTIC EXECUTION WRAPPER CHANGE: call unchanged upstream set_seed(0), "
                          "then restore PYTHONHASHSEED=1 in parent os.environ before macOS spawn workers; "
                          "do not alter Python/NumPy seed results or upstream source" if preserve_worker_hashseed
                          else "none; unchanged upstream set_seed(0) resets future spawn workers to hash seed 0"),
                      startup=startup,
                      preserve_hashseed_for_spawn_workers=preserve_worker_hashseed,
                      worker_startup_hashseed=expected_worker_hash,
                      worker_preflight=worker_checks,
                      worker_python_numpy_rng_state_verified=False,
                      baseline_hashes=before,
                      graph=graph_check,
                      accepted_semantic_pairs=dict(drug=27530, disease=25573),
                      model_parameters=formal["dreamwalk"],
                      evaluation=dict(positive_pairs=5926, negative_pairs=5926,
                                      negative_file_sha256=sha256(NEGATIVES),
                                      cv_folds=10, cv_seed=0, xgboost_seed=0,
                                      xgboost_settings=xgb_settings(),
                                      feature="drug_embedding - disease_embedding"))
    write_json(RUN_CONFIG, run_config)
    print("DIAGNOSTIC FULL HASHSEED-1 CONFIG", json.dumps(run_config, indent=2), flush=True)

    stage = {}
    actual_walks = {}
    def timed(name, fn):
        def wrapper(*args, **kwargs):
            print("Starting", name, flush=True)
            t0 = time.monotonic()
            result = fn(*args, **kwargs)
            stage[name + "_seconds"] = time.monotonic() - t0
            if name == "random_walk_generation":
                actual_walks["count"] = len(result)
                print("Actual walks:", len(result), flush=True)
            print(name, "seconds:", round(stage[name + "_seconds"], 2), flush=True)
            return result
        return wrapper
    ge.train_edgetype_transition_matrix = timed("edge_transition_training", ge.train_edgetype_transition_matrix)
    ge.generate_DREAMwalk_paths = timed("random_walk_generation", ge.generate_DREAMwalk_paths)
    ge.HeterogeneousSG = timed("heterogeneous_skipgram", ge.HeterogeneousSG)

    os.chdir(work)
    started = time.monotonic()
    print("Starting DIAGNOSTIC hashseed-1 full semantic embedding", flush=True)
    ge.save_embedding_files(netf=str(GRAPH), sim_netf=str(SIMILARITY),
                            outputf=str(NEW_EMBEDDING), nodetypef=str(NODETYPES),
                            tp_factor=0.3, seed=0, num_walks=100, walk_length=10,
                            dimension=128, window_size=4, workers=4,
                            em_max_iter=5, p=1, q=1)
    elapsed = time.monotonic() - started
    assert actual_walks.get("count") == 2995900
    assert os.environ.get("PYTHONHASHSEED") == expected_worker_hash
    embedding = load_embedding(NEW_EMBEDDING, types, sorted(positives) + negatives)
    import numpy as np
    nonfinite_vectors = sum(not np.isfinite(np.asarray(vector)).all() for vector in embedding.values())
    assert nonfinite_vectors == 0 and len(embedding) == 29959
    timing = dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                  startup=startup, worker_startup_hashseed=expected_worker_hash,
                  expected_walks=2995900, actual_walks=actual_walks["count"],
                  embedding_nodes=len(embedding), embedding_dimension=128,
                  nonfinite_vector_count=nonfinite_vectors,
                  total_embedding_seconds=elapsed,
                  peak_parent_rss_bytes=peak_rss_bytes(),
                  output_embedding_sha256=sha256(NEW_EMBEDDING),
                  **stage)
    write_json(TIMING, timing)
    check_baseline_hashes(before)
    print("DIAGNOSTIC HASHSEED-1 EMBEDDING COMPLETE", json.dumps(timing, indent=2), flush=True)


def phase_evaluate():
    assert NEW_EMBEDDING.exists() and RUN_CONFIG.exists() and TIMING.exists()
    assert not NEW_FOLDS.exists() and not NEW_SUMMARY.exists(), "Refusing to overwrite diagnostic evaluation"
    before = baseline_hashes()
    types, positives, negatives, graph_check = checked_pairs()
    embedding = load_embedding(NEW_EMBEDDING, types, sorted(positives) + negatives)
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    rows = evaluate(embedding, positives, negatives, 0, "hashseed1_semantic_tau_0.3")
    summary = metric_summary(rows, 0, "hashseed1_semantic_tau_0.3")
    write_csv(NEW_FOLDS, rows, list(rows[0]))
    write_csv(NEW_SUMMARY, summary, list(summary[0]))
    check_baseline_hashes(before)
    print("DIAGNOSTIC HASHSEED-1 10-FOLD SUMMARY", json.dumps(summary, indent=2), flush=True)


def phase_compare():
    assert NEW_FOLDS.exists() and NEW_SUMMARY.exists()
    before = baseline_hashes()
    hash0 = {r["metric"]: r for r in read_rows(HASH0_SUMMARY, ",") if r["negative_seed"] == "0"}
    hash1 = {r["metric"]: r for r in read_rows(NEW_SUMMARY, ",")}
    assert set(hash0) == set(hash1) == set(METRICS)
    paper = dict(accuracy=0.8770983, auroc=0.9440605, aupr=0.9467667)
    rows = []
    for metric in METRICS:
        a = float(hash0[metric]["mean"])
        b = float(hash1[metric]["mean"])
        rows.append(dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                         metric=metric, semantic_hashseed0_10fold=a,
                         semantic_hashseed1_10fold=b, hash1_minus_hash0=b-a,
                         paper_semantic=paper[metric],
                         hash0_fold_SD=float(hash0[metric]["SD"]),
                         hash1_fold_SD=float(hash1[metric]["SD"])))
    write_csv(COMPARISON, rows, list(rows[0]))
    check_baseline_hashes(before)
    print("DIAGNOSTIC FULL HASH-SEED COMPARISON", json.dumps(rows, indent=2), flush=True)


def phase_compare_embeddings():
    import numpy as np
    assert NEW_EMBEDDING.exists() and not EMBEDDING_COMPARISON.exists()
    before = baseline_hashes()
    types, positives, negatives, graph_check = checked_pairs()
    a = load_embedding(EMBEDDING, types, sorted(positives) + negatives)
    b = load_embedding(NEW_EMBEDDING, types, sorted(positives) + negatives)
    values = []
    groups = []
    for node in sorted(types):
        va, vb = np.asarray(a[node], dtype=np.float64), np.asarray(b[node], dtype=np.float64)
        denominator = np.linalg.norm(va) * np.linalg.norm(vb)
        assert denominator > 0
        similarity = float(np.dot(va, vb) / denominator)
        assert np.isfinite(similarity)
        values.append(similarity)
        groups.append(types[node])
    values = np.asarray(values)
    groups = np.asarray(groups)
    rows = []
    for kind in ("all", "drug", "disease", "gene", "etc"):
        selected = values if kind == "all" else values[groups == kind]
        assert len(selected) > 0
        rows.append(dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                         caution="raw independent Skip-gram coordinates are unaligned",
                         node_type=kind, n=len(selected),
                         cosine_mean=float(np.mean(selected)),
                         cosine_median=float(np.median(selected)),
                         cosine_SD=float(np.std(selected, ddof=1)),
                         cosine_min=float(np.min(selected)),
                         cosine_max=float(np.max(selected))))
    write_csv(EMBEDDING_COMPARISON, rows, list(rows[0]))
    check_baseline_hashes(before)
    print("UNALIGNED EMBEDDING COSINE COMPARISON", json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("embedding", "evaluate", "compare", "compare-embeddings"))
    parser.add_argument("--preserve-worker-hashseed", action="store_true")
    args = parser.parse_args()
    if args.phase == "embedding":
        phase_embedding(args.preserve_worker_hashseed)
    elif args.phase == "evaluate":
        phase_evaluate()
    elif args.phase == "compare":
        phase_compare()
    else:
        phase_compare_embeddings()


if __name__ == "__main__":
    main()
