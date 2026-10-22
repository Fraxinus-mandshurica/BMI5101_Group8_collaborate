"""User-started combined MSI diagnostic: MeSH2024, tau=.3, seed0, pp=1.

Use demo-aligned MeSH letter roots, first-hit teleport selection, and
positive-output-type negative sampling with current-positive rejection.
The historical MeSH2021 runner and all of its artifacts remain unchanged.
Importing this module does not prepare scientific inputs or start training.
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
import msi_mesh2024_demo_roots as inputs
import run_msi_seed0_final_fixed as fixed

# These helpers depend only on the unchanged MSI cohort, C++ patch and saved
# reference artifacts. They do not consult the historical semantic provider.
# In particular, no imported module globals or historical source are modified.
shared, paired, previous = fixed.shared, fixed.paired, fixed.previous
sha256, save_json = fixed.sha256, fixed.save_json
read_json, require_equal = fixed.read_json, fixed.require_equal
patch_source, patch_identity = fixed.patch_source, fixed.patch_identity
inspect_compiler_reference = fixed.inspect_compiler_reference
prepare, require_prepared = fixed.prepare, fixed.require_prepared
original_summary = fixed.original_summary
RUN_NAME = "msi_seed0_mesh2024_final_fixed"
RUN_SETTINGS = dict(pp=1, tp_factor=0.3, seed=0, num_walks=100, walk_length=10,
                    dimension=128, window_size=4, workers=4, epochs=1,
                    negative=5, sample=0.001, folds=10, cv_seed=0,
                    ns_pool="positive_output_type", reject_positive_output=True,
                    rejection_policy="skip_without_resampling",
                    teleport_selector="first_hit")
WALK_SETTINGS = dict(tp_factor=0.3, seed=0, num_walks=100, walk_length=10,
                     p=1, q=1, workers=4, em_max_iter=5)
SEMANTIC_SETTINGS = dict(mesh_year=2024, cutoff=0.4, letter_roots=True)
SOURCE_NODES = fixed.SOURCE_NODES
SELECTOR_MODULE, CPP_SOURCE = fixed.SELECTOR_MODULE, fixed.CPP_SOURCE
REFERENCE_RESULTS, REFERENCE_DATA = fixed.REFERENCE_RESULTS, fixed.REFERENCE_DATA
ORIGINAL2024_RESULTS = fixed.ORIGINAL2024_RESULTS
FINAL2021_RESULTS = ROOT / "results" / fixed.RUN_NAME
PAPER_REFERENCE, METRICS = fixed.PAPER_REFERENCE, fixed.METRICS
ANALYSIS = "COMBINED DIAGNOSTIC: MeSH2024 letter roots, first-hit teleport, output-type NS with positive rejection"
LABEL = "mesh2024_final_fixed_pp1"


def context(run_name=RUN_NAME):
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("RUN_NAME must contain only letters, digits, underscore or hyphen")
    historical_names = (fixed.RUN_NAME, fixed.original.RUN_NAME, fixed.original.REFERENCE_NAME,
                        fixed.positive_filter.RUN_NAME, fixed.output_pool.RUN_NAME,
                        fixed.teleport.RUN_NAME, "msi_seed0_semantic_skipgram_pp_comparison")
    if run_name in historical_names:
        raise ValueError("Preserve historical runs; choose a separate RUN_NAME")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Use the existing DREAMwalk Python 3.8 interpreter")
    packages = {name: version(name) for name in paired.PINS}
    require_equal(packages, paired.PINS, "DREAMwalk environment")
    commit = subprocess.check_output(["git", "-C", str(paired.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    require_equal(commit, paired.PINNED_COMMIT, "Upstream commit")
    semantic_identity, patch = inputs.artifact_identity(), patch_identity()
    paths = dict(graph=previous.GRAPH, nodetypes=previous.NODETYPES,
                 positives=previous.POSITIVES, negatives=previous.NEGATIVES,
                 mesh=inputs.MESH_PATH, hierarchy=inputs.HIERARCHY_PATH,
                 similarity=inputs.SIMILARITY_PATH)
    sources = [Path(__file__), Path(inputs.__file__), Path(fixed.__file__),
               Path(fixed.original.__file__), Path(fixed.output_pool.__file__),
               Path(fixed.positive_filter.__file__), Path(fixed.teleport.__file__),
               SELECTOR_MODULE, Path(shared.__file__), Path(paired.__file__), Path(previous.__file__),
               ROOT / "src/run_msi_repeat0.py", ROOT / "src/run_no_teleport_length4.py",
               paired.UPSTREAM / "DREAMwalk/generate_embeddings.py",
               paired.UPSTREAM / "DREAMwalk/utils.py", CPP_SOURCE]
    identity = dict(settings=RUN_SETTINGS, walks=WALK_SETTINGS, semantic=SEMANTIC_SETTINGS,
                    patch=patch, compiler_reference=inspect_compiler_reference(patch),
                    reference=fixed.original.reference_identity(), semantic_input_provenance=semantic_identity,
                    skipgram_args=paired.SG_ARGS, compile_flags=paired.COMPILE_FLAGS,
                    xgboost=previous.xgb_settings(), folds=10, cv_seed=0,
                    feature="drug_embedding - disease_embedding", packages=packages, upstream_commit=commit,
                    inputs={name: sha256(path) for name, path in paths.items()},
                    sources={str(path.relative_to(ROOT)): sha256(path) for path in sources})
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
        save_json(config, dict(identity, fingerprint=fingerprint, analysis=ANALYSIS,
                  input_paths={name: str(path.relative_to(ROOT)) for name, path in paths.items()},
                  pair_order="sorted positives + original saved seed-0 negative order",
                  fold_origin="deterministically regenerated and checked against saved original fold IDs",
                  notes=["Combined diagnostic changes; not an isolated causal contrast or the released benchmark implementation.",
                         "Use independent MeSH2024 hierarchy/similarity with author-demo letter roots.",
                         "Reuse MSI molecular graph, ATC, mapping candidates, link-level negatives and fold IDs.",
                         "Train unchanged upstream edge-type EM afresh before generating fixed teleport walks.",
                         "Every spawn worker installs and reports actual use of the first-hit selector.",
                         "NS uses positive output type; skip only a current-positive collision, without resampling.",
                         "Reuse an exact verified patched C++ binary, or compile the independent patched source.",
                         "Upstream vendor sources and historical MeSH2021 sources/artifacts remain unchanged.",
                         "Original set_seed(0) is unchanged; no start-method override or hashseed restoration.",
                         "Worker Python/NumPy RNG states are not asserted; four-thread C++ training retains nondeterminism.",
                         "Persist embedding atomically before the separate evaluation phase.",
                         "Tau=.3 is the accepted local setting, not a claimed published Figure2a value."],
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  multiprocessing_start_method=multiprocessing.get_context().get_start_method(),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def generate_corpus(data, results, fingerprint):
    import parmap
    from DREAMwalk import generate_embeddings as ge
    import teleport_selector_diagnostic as selector
    corpus, marker = data / "walks.txt", results / "walks_complete.json"
    compiler = require_prepared(data, results, fingerprint)
    if paired.completed(corpus, marker, fingerprint):
        require_corpus(data, results, fingerprint)
        return
    if corpus.exists() or (data / "pp1.pkl").exists() or (results / "pp1_embedding_complete.json").exists():
        raise RuntimeError("Existing corpus/embedding prevents regeneration; preserve outputs and use a new RUN_NAME")
    started = time.monotonic()
    ge.set_seed(WALK_SETTINGS["seed"])
    graph = ge.read_graph(str(previous.GRAPH), weighted=True, directed=False, delimiter="\t")
    semantic = ge.read_graph(str(inputs.SIMILARITY_PATH), weighted=True, directed=False, delimiter="\t")
    if WALK_SETTINGS["tp_factor"] <= 0 or semantic.number_of_edges() == 0:
        raise RuntimeError("This combined run requires active teleport and nonempty MeSH2024 semantic input")
    types = {row["node"]: row["type"] for row in previous.read_rows(previous.NODETYPES)}
    node_to_token = shared.checked_tokens(read_json(data / "token_to_node.json"), types, graph.nodes())
    require_equal(len(node_to_token), SOURCE_NODES, "MSI node count")
    t0 = time.monotonic()
    print("Starting unchanged edge-type EM: 5 iterations, length=10", flush=True)
    matrix = ge.train_edgetype_transition_matrix(WALK_SETTINGS["em_max_iter"], graph,
                                                str(previous.GRAPH), "\t", WALK_SETTINGS["walk_length"],
                                                WALK_SETTINGS["p"], WALK_SETTINGS["q"])
    em_seconds = time.monotonic() - t0
    save_json(results / "edge_transition_matrix.json", matrix.tolist())
    t0 = time.monotonic()
    print("Starting fixed MeSH2024 walks: tau=.3, 100 x 29,959, length=10", flush=True)
    # The importable worker installs the first-hit patch in each spawn process.
    task_results = parmap.map(selector.worker_walks, range(WALK_SETTINGS["num_walks"]),
                              list(graph.nodes()), graph, semantic, matrix,
                              WALK_SETTINGS["p"], WALK_SETTINGS["q"], WALK_SETTINGS["walk_length"],
                              WALK_SETTINGS["tp_factor"], pm_pbar=False,
                              pm_processes=min(WALK_SETTINGS["num_walks"], WALK_SETTINGS["workers"]))
    walk_seconds = time.monotonic() - t0
    audits = fixed.teleport.checked_worker_audit([item[1] for item in task_results], WALK_SETTINGS["num_walks"],
                                                 sha256(SELECTOR_MODULE))
    walks = (walk for task_walks, _ in task_results for walk in task_walks)
    temporary = data / "walks.txt.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        counts = shared.write_corpus(walks, stream, node_to_token, WALK_SETTINGS["walk_length"],
                                     SOURCE_NODES * WALK_SETTINGS["num_walks"], WALK_SETTINGS["num_walks"])
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(corpus)
    audit_path = results / "worker_selector_audit.json"
    save_json(audit_path, dict(audits, fingerprint=fingerprint))
    save_json(marker, dict(counts, fingerprint=fingerprint, sha256=sha256(corpus),
              token_map_sha256=compiler["token_map_sha256"],
              matrix_sha256=sha256(results / "edge_transition_matrix.json"),
              worker_audit_sha256=sha256(audit_path), selector_calls=audits["selector_calls"],
              tp_factor=WALK_SETTINGS["tp_factor"], similarity_sha256=sha256(inputs.SIMILARITY_PATH),
              similarity_edges=semantic.number_of_edges(), em_reused=False,
              serialization="newline between walks; no final newline", em_seconds=em_seconds,
              walk_seconds=walk_seconds, total_seconds=time.monotonic()-started))
    print("FINAL MESH2024 COMBINED CORPUS SAVED:", corpus, flush=True)


def require_corpus(data, results, fingerprint):
    require_prepared(data, results, fingerprint)
    corpus = paired.require_checkpoint(data / "walks.txt", results / "walks_complete.json", fingerprint)
    require_equal(corpus["token_map_sha256"], sha256(data / "token_to_node.json"), "Corpus token map")
    require_equal(corpus["similarity_sha256"], sha256(inputs.SIMILARITY_PATH), "Corpus MeSH2024 semantic input")
    require_equal(corpus["matrix_sha256"], sha256(results / "edge_transition_matrix.json"), "Corpus EM matrix")
    audit_path = results / "worker_selector_audit.json"
    require_equal(corpus["worker_audit_sha256"], sha256(audit_path), "Saved worker selector audit")
    fixed.teleport.checked_worker_audit(read_json(audit_path)["audits"], WALK_SETTINGS["num_walks"], sha256(SELECTOR_MODULE))
    return corpus


def embedding(data, results, fingerprint):
    require_corpus(data, results, fingerprint)
    paired.train(1, data, results, fingerprint)
    require_embedding(data, results, fingerprint)
    print("MeSH2024 pp1.pkl persisted; evaluation is a separate later notebook cell", flush=True)


def require_embedding(data, results, fingerprint):
    corpus = require_corpus(data, results, fingerprint)
    meta = paired.require_checkpoint(data / "pp1.pkl", results / "pp1_embedding_complete.json", fingerprint)
    for key, expected in (("corpus_sha256", corpus["sha256"]), ("binary_sha256", sha256(data / "HetSG")),
                          ("token_map_sha256", corpus["token_map_sha256"]), ("pp", 1),
                          ("command", fixed.positive_filter.training_command())):
        require_equal(meta[key], expected, "Final MeSH2024 embedding " + key)
    return meta


def evaluate(data, results, fingerprint):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    output, marker = results / "pp1_summary.csv", results / "pp1_evaluation_complete.json"
    if paired.completed(output, marker, fingerprint):
        require_evaluation(data, results, fingerprint)
        return
    embedded = require_embedding(data, results, fingerprint)
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
    require_equal([{key: str(value) for key, value in row.items()} for row in pair_rows], expected,
                  "Pair/fold assignments before fitting")
    fold_path = results / "fold_assignments.csv"
    require_equal(previous.read_rows(fold_path, ","), expected, "Prepared original pair/fold assignments")
    require_equal(sha256(fold_path), sha256(REFERENCE_RESULTS / "fold_assignments.csv"), "Original fold CSV bytes")
    vectors = previous.load_embedding(data / "pp1.pkl", types, pairs)
    started = time.monotonic()
    rows = previous.evaluate(vectors, positives, negatives, 0, LABEL)
    for row in rows:
        row.update(analysis=ANALYSIS, label=LABEL, embedding="mesh2024_final_fixed_tau_0.3_pp1", pp=1)
    summary = previous.metric_summary(rows, 0, LABEL)
    for row in summary:
        row["analysis"] = ANALYSIS
    fold_results = results / "pp1_fold_results.csv"
    previous.write_csv(fold_results, rows, list(rows[0]))
    previous.write_csv(output, summary, list(summary[0]))
    save_json(marker, dict(fingerprint=fingerprint, sha256=sha256(output), pp=1,
              embedding_sha256=embedded["sha256"], fold_assignments_sha256=sha256(fold_path),
              fold_results_sha256=sha256(fold_results), seconds=time.monotonic()-started))


def require_evaluation(data, results, fingerprint):
    embedded = require_embedding(data, results, fingerprint)
    meta = fixed.original.require_evaluation(data, results, fingerprint)
    require_equal(meta["embedding_sha256"], embedded["sha256"], "Final MeSH2024 evaluated embedding")
    return meta


def compare(data, results, fingerprint):
    meta = require_evaluation(data, results, fingerprint)
    final = {row["metric"]: float(row["mean"]) for row in previous.read_rows(results / "pp1_summary.csv", ",")}
    old2024 = original_summary(ORIGINAL2024_RESULTS, meta["fold_assignments_sha256"])
    paper = {row["metric"]: float(row["mean"]) for row in previous.read_rows(PAPER_REFERENCE, ",")}
    # The 2024 experiment is independent of completion of the 2021 experiment.
    old2021 = None
    optional_marker = FINAL2021_RESULTS / "pp1_evaluation_complete.json"
    if optional_marker.is_file():
        old2021 = original_summary(FINAL2021_RESULTS, meta["fold_assignments_sha256"])
        old_config = read_json(FINAL2021_RESULTS / "run_config.json")
        require_equal(old_config["semantic"], dict(mesh_year=2021, cutoff=0.4, letter_roots=True),
                      "Optional completed MeSH2021 final semantic settings")
        require_equal(old_config["settings"], RUN_SETTINGS, "Optional completed MeSH2021 final model settings")
        require_equal(old_config["patch"], patch_identity(), "Optional completed MeSH2021 final patch")
    for values in (final, old2024, paper) + (() if old2021 is None else (old2021,)):
        require_equal(set(values), set(METRICS), "Comparison metrics")
    rows = [dict(metric=metric, original_mesh2024_pp1=old2024[metric],
                 final_mesh2024_fixed_pp1=final[metric],
                 final_minus_original2024=final[metric]-old2024[metric],
                 final_mesh2021_fixed_pp1="" if old2021 is None else old2021[metric],
                 mesh2024_minus_mesh2021_final="" if old2021 is None else final[metric]-old2021[metric],
                 paper_figure2a_mean=paper[metric], final_minus_paper=final[metric]-paper[metric])
            for metric in METRICS]
    output = results / "final_fixed_comparison.csv"
    previous.write_csv(output, rows, list(rows[0]))
    optional = dict(status="not_completed", run_name=fixed.RUN_NAME)
    if old2021 is not None:
        optional.update(status="completed", summary_sha256=sha256(FINAL2021_RESULTS / "pp1_summary.csv"),
                        evaluation_sha256=sha256(optional_marker))
    save_json(results / "comparison_complete.json", dict(fingerprint=fingerprint, sha256=sha256(output),
              final_summary_sha256=meta["sha256"], shared_fold_assignments_sha256=meta["fold_assignments_sha256"],
              original2024_summary_sha256=sha256(ORIGINAL2024_RESULTS / "pp1_summary.csv"),
              optional_final2021=optional, paper_reference_sha256=sha256(PAPER_REFERENCE),
              note="Combined semantic-input, selector and NS changes; not a single-variable attribution. Paper values are published means, not a seed-0 run."))
    print("FINAL MESH2024 COMBINED COMPARISON\n" + json.dumps(rows, indent=2), flush=True)


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
    phases = dict(prepare=prepare, walks=generate_corpus, embedding=embedding,
                  evaluate=evaluate, compare=compare)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    else:
        phases[args.phase](data, results, fingerprint)


if __name__ == "__main__":
    main()
