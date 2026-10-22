"""Notebook controls on corrected MeSH2024 inputs; historical runners stay frozen.

Original pp=1/0 share one corpus. The selector control reuses that run's pp=0
baseline and EM matrix. Importing this module starts no scientific work.
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

import msi_mesh2024_demo_roots as inputs
import run_msi_mesh2021 as original
import run_semantic_pp_comparison as shared
import run_teleport_selector_control as teleport

ROOT = Path(__file__).resolve().parents[1]
paired, previous = shared.paired, shared.previous
sha256, read_json = shared.sha256, shared.read_json
save_json, require_equal = shared.save_json, shared.require_equal
WALK_SETTINGS = dict(tp_factor=0.3, seed=0, num_walks=100, walk_length=10,
                     p=1, q=1, workers=4, em_max_iter=5)
SEMANTIC_SETTINGS = dict(mesh_year=2024, cutoff=0.4, letter_roots=True)
SEMANTIC_RUN_NAME = "msi_seed0_semantic_pp_mesh2024_roots"
TELEPORT_RUN_NAME = "msi_seed0_teleport_mesh2024_roots"
REFERENCE_DATA, REFERENCE_RESULTS = original.REFERENCE_DATA, original.REFERENCE_RESULTS
SELECTOR_MODULE = ROOT / "src/teleport_selector_diagnostic.py"
SOURCE_NODES = 29959


def semantic_reference():
    """Require the new original-selector pp=0 baseline before the fixed control."""
    data = ROOT / "data/embeddings" / SEMANTIC_RUN_NAME
    results = ROOT / "results" / SEMANTIC_RUN_NAME
    config_path = results / "run_config.json"
    required = (config_path, results / "pp0_evaluation_complete.json")
    if any(not path.is_file() for path in required):
        raise RuntimeError("Complete notebook Section 7 pp=0 with corrected MeSH2024 inputs first: "
                           + SEMANTIC_RUN_NAME)
    config = read_json(config_path)
    require_equal(config["walks"], WALK_SETTINGS, "New Section 7 walks")
    require_equal(config["semantic"], SEMANTIC_SETTINGS, "New Section 7 semantic settings")
    require_equal(config["selector"], "upstream", "New Section 7 original selector")
    require_equal(config["semantic_input_provenance"], inputs.artifact_identity(), "New Section 7 input provenance")
    require_equal(config["reference"], original.reference_identity(), "New Section 7 common cohort")
    for relative, digest in config["sources"].items():
        require_equal(sha256(ROOT / relative), digest, "New Section 7 source " + relative)
    fingerprint = config["fingerprint"]
    corpus = require_corpus(data, results, fingerprint, "semantic")
    embedding, evaluation = shared.require_evaluation(0, data, results, fingerprint)
    require_equal(embedding["pp"], 0, "New Section 7 baseline pp")
    require_equal(embedding["command"], ["./HetSG", "-train", "walks.txt", "-output", "pp0", "-pp", "0"]
                  + paired.SG_ARGS, "New Section 7 baseline training command")
    for key, digest in (("corpus_sha256", corpus["sha256"]),
                        ("binary_sha256", sha256(data / "HetSG")),
                        ("token_map_sha256", sha256(data / "token_to_node.json"))):
        require_equal(embedding[key], digest, "New Section 7 baseline " + key)
    return dict(run_name=SEMANTIC_RUN_NAME, fingerprint=fingerprint,
                artifacts=dict(corpus=corpus["sha256"], binary=embedding["binary_sha256"],
                               token_map=embedding["token_map_sha256"],
                               matrix=sha256(results / "edge_transition_matrix.json"),
                               embedding=embedding["sha256"], summary=evaluation["sha256"],
                               fold_assignments=evaluation["fold_assignments_sha256"]),
                records={name: sha256(results / name) for name in
                         ("run_config.json", "walks_complete.json", "compiler_complete.json",
                          "pp0_embedding_complete.json", "pp0_evaluation_complete.json")})


def context(run_name, control):
    if control not in ("semantic", "teleport"):
        raise ValueError("Unknown control")
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("RUN_NAME must contain only letters, digits, underscore or hyphen")
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Use the existing DREAMwalk Python 3.8 interpreter")
    packages = {name: version(name) for name in paired.PINS}
    require_equal(packages, paired.PINS, "DREAMwalk environment")
    commit = subprocess.check_output(["git", "-C", str(paired.UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    require_equal(commit, paired.PINNED_COMMIT, "Upstream commit")
    reference = original.reference_identity() if control == "semantic" else semantic_reference()
    paths = dict(graph=previous.GRAPH, nodetypes=previous.NODETYPES,
                 positives=previous.POSITIVES, negatives=previous.NEGATIVES,
                 hierarchy=inputs.HIERARCHY_PATH, similarity=inputs.SIMILARITY_PATH, mesh=inputs.MESH_PATH)
    source_paths = [Path(__file__), ROOT / "src/run_semantic_mesh2024_controls.py",
                    ROOT / "src/run_teleport_mesh2024_control.py", Path(inputs.__file__),
                    ROOT / "src/mesh_hierarchy_variant.py", Path(original.__file__),
                    Path(original.inputs.__file__), Path(shared.__file__), Path(teleport.__file__),
                    Path(paired.__file__), Path(previous.__file__), SELECTOR_MODULE,
                    ROOT / "src/run_no_teleport_length4.py", ROOT / "src/run_msi_repeat0.py",
                    paired.UPSTREAM / "DREAMwalk/generate_embeddings.py",
                    paired.UPSTREAM / "DREAMwalk/utils.py", paired.UPSTREAM / "DREAMwalk/HeterogeneousSG.cpp"]
    identity = dict(walks=WALK_SETTINGS, semantic=SEMANTIC_SETTINGS,
                    selector="upstream" if control == "semantic" else "first_hit", control=control,
                    skipgram_args=paired.SG_ARGS, xgboost=previous.xgb_settings(),
                    folds=10, cv_seed=0, feature="drug_embedding - disease_embedding",
                    inputs={key: sha256(path) for key, path in paths.items()},
                    semantic_input_provenance=inputs.artifact_identity(), reference=reference,
                    sources={str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
                    packages=packages, upstream_commit=commit)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    data, results = ROOT / "data/embeddings" / run_name, ROOT / "results" / run_name
    config_path = results / "run_config.json"
    if config_path.exists():
        require_equal(read_json(config_path)["fingerprint"], fingerprint,
                      "Run identity; preserve outputs and use a new RUN_NAME if inputs/code changed")
    else:
        if (data.exists() and any(data.iterdir())) or (results.exists() and any(results.iterdir())):
            raise RuntimeError("Unregistered outputs exist; preserve them and use a new RUN_NAME")
        data.mkdir(parents=True, exist_ok=True)
        results.mkdir(parents=True, exist_ok=True)
        save_json(config_path, dict(identity, fingerprint=fingerprint,
                  analysis="MeSH2024 demo-root semantic pp control" if control == "semantic" else
                           "DIAGNOSTIC first-hit selector; MeSH2024 demo roots; pp=0",
                  input_paths={key: str(path.relative_to(ROOT)) for key, path in paths.items()},
                  pair_order="sorted positives + original saved seed-0 negative order",
                  fold_origin="deterministically regenerated and checked against saved no-teleport IDs",
                  notes=["Original C++ pp branches are unchanged; no output-type or collision-filter patch.",
                         "Section 7 uses one original-selector corpus for pp=1 and pp=0.",
                         "Section 8 reuses Section 7 corrected-input pp=0 baseline and EM matrix.",
                         "Upstream set_seed(0) is unchanged; no hashseed restoration or start-method override.",
                         "Worker Python/NumPy RNG state is not asserted; four C++ threads retain nondeterminism.",
                         "The selector-control corpus is newly sampled, not a paired RNG replay.",
                         "Embedding is persisted before the separate evaluation phase."],
                  python=sys.version, platform=platform.platform(),
                  startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
                  multiprocessing_start_method=multiprocessing.get_context().get_start_method(),
                  created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def prepare(data, results, fingerprint, control):
    source_data, source_results = REFERENCE_DATA, REFERENCE_RESULTS
    if control == "teleport":
        semantic_reference()
        source_data = ROOT / "data/embeddings" / SEMANTIC_RUN_NAME
        source_results = ROOT / "results" / SEMANTIC_RUN_NAME
        shared.copy_exact(source_results / "edge_transition_matrix.json", results / "edge_transition_matrix.json")
    for name in ("HetSG", "token_to_node.json"):
        shared.copy_exact(source_data / name, data / name)
    shared.copy_exact(source_results / "fold_assignments.csv", results / "fold_assignments.csv")
    marker = results / "compiler_complete.json"
    if not paired.completed(data / "HetSG", marker, fingerprint):
        meta = dict(fingerprint=fingerprint, sha256=sha256(data / "HetSG"),
                    token_map_sha256=sha256(data / "token_to_node.json"),
                    reused_from=str(source_data.relative_to(ROOT)),
                    original_compiler=read_json(source_results / "compiler_complete.json"))
        if control == "teleport":
            meta["matrix_sha256"] = sha256(results / "edge_transition_matrix.json")
        save_json(marker, meta)
    print("Original binary, token map and fixed folds ready; control:", control, flush=True)


def require_corpus(data, results, fingerprint, control):
    compiler = paired.require_checkpoint(data / "HetSG", results / "compiler_complete.json", fingerprint)
    corpus = paired.require_checkpoint(data / "walks.txt", results / "walks_complete.json", fingerprint)
    require_equal(corpus["token_map_sha256"], sha256(data / "token_to_node.json"), "Corpus token map")
    require_equal(corpus["token_map_sha256"], compiler["token_map_sha256"], "Prepared token map")
    require_equal(corpus["similarity_sha256"], sha256(inputs.SIMILARITY_PATH), "Corrected MeSH2024 semantic input")
    require_equal(corpus["matrix_sha256"], sha256(results / "edge_transition_matrix.json"), "Corpus EM matrix")
    require_equal(corpus["tp_factor"], WALK_SETTINGS["tp_factor"], "Corpus tau")
    if control == "teleport":
        require_equal(corpus["matrix_sha256"], compiler["matrix_sha256"], "Reused EM matrix")
        audit_path = results / "worker_selector_audit.json"
        require_equal(corpus["worker_audit_sha256"], sha256(audit_path), "Worker selector audit")
        teleport.checked_worker_audit(read_json(audit_path)["audits"], WALK_SETTINGS["num_walks"], sha256(SELECTOR_MODULE))
    return corpus


def generate_corpus(data, results, fingerprint, control):
    from DREAMwalk import generate_embeddings as ge
    corpus, marker = data / "walks.txt", results / "walks_complete.json"
    if paired.completed(corpus, marker, fingerprint):
        require_corpus(data, results, fingerprint, control)
        return
    compiler = paired.require_checkpoint(data / "HetSG", results / "compiler_complete.json", fingerprint)
    require_equal(sha256(data / "token_to_node.json"), compiler["token_map_sha256"], "Prepared map")
    if corpus.exists() or any((data / f"pp{pp}.pkl").exists() or
                             (results / f"pp{pp}_embedding_complete.json").exists() for pp in (1, 0)):
        raise RuntimeError("Existing corpus/embedding prevents regeneration; preserve outputs and use a new RUN_NAME")
    started = time.monotonic()
    ge.set_seed(WALK_SETTINGS["seed"])
    graph = ge.read_graph(str(previous.GRAPH), weighted=True, directed=False, delimiter="\t")
    similarity = ge.read_graph(str(inputs.SIMILARITY_PATH), weighted=True, directed=False, delimiter="\t")
    if WALK_SETTINGS["tp_factor"] <= 0 or similarity.number_of_edges() == 0:
        raise RuntimeError("Active teleport requires nonempty corrected MeSH2024 semantic input")
    types = {row["node"]: row["type"] for row in previous.read_rows(previous.NODETYPES)}
    node_to_token = shared.checked_tokens(read_json(data / "token_to_node.json"), types, graph.nodes())
    require_equal(len(node_to_token), SOURCE_NODES, "MSI node count")
    t0 = time.monotonic()
    if control == "semantic":
        matrix = ge.train_edgetype_transition_matrix(WALK_SETTINGS["em_max_iter"], graph,
                                                    str(previous.GRAPH), "\t", WALK_SETTINGS["walk_length"],
                                                    WALK_SETTINGS["p"], WALK_SETTINGS["q"])
        save_json(results / "edge_transition_matrix.json", matrix.tolist())
    else:
        import numpy as np
        require_equal(sha256(results / "edge_transition_matrix.json"), compiler["matrix_sha256"], "Prepared matrix")
        matrix = np.asarray(read_json(results / "edge_transition_matrix.json"), dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.isfinite(matrix).all():
            raise ValueError("Invalid saved edge-transition matrix")
    em_seconds = time.monotonic() - t0
    t0 = time.monotonic()
    audits = None
    if control == "semantic":
        walks = ge.generate_DREAMwalk_paths(graph, similarity, matrix, WALK_SETTINGS["p"], WALK_SETTINGS["q"],
                                           WALK_SETTINGS["num_walks"], WALK_SETTINGS["walk_length"],
                                           WALK_SETTINGS["tp_factor"], WALK_SETTINGS["workers"])
    else:
        import parmap
        import teleport_selector_diagnostic as selector
        task_results = parmap.map(selector.worker_walks, range(WALK_SETTINGS["num_walks"]),
                                  list(graph.nodes()), graph, similarity, matrix,
                                  WALK_SETTINGS["p"], WALK_SETTINGS["q"], WALK_SETTINGS["walk_length"],
                                  WALK_SETTINGS["tp_factor"], pm_pbar=False,
                                  pm_processes=min(WALK_SETTINGS["num_walks"], WALK_SETTINGS["workers"]))
        audits = teleport.checked_worker_audit([item[1] for item in task_results], WALK_SETTINGS["num_walks"],
                                              sha256(SELECTOR_MODULE))
        walks = (walk for task_walks, _ in task_results for walk in task_walks)
    walk_seconds = time.monotonic() - t0
    temporary = data / "walks.txt.tmp"
    with temporary.open("w", encoding="utf-8") as stream:
        counts = shared.write_corpus(walks, stream, node_to_token, WALK_SETTINGS["walk_length"],
                                     SOURCE_NODES * WALK_SETTINGS["num_walks"], WALK_SETTINGS["num_walks"])
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(corpus)
    meta = dict(counts, fingerprint=fingerprint, sha256=sha256(corpus),
                token_map_sha256=sha256(data / "token_to_node.json"),
                matrix_sha256=sha256(results / "edge_transition_matrix.json"),
                tp_factor=WALK_SETTINGS["tp_factor"], similarity_sha256=sha256(inputs.SIMILARITY_PATH),
                similarity_edges=similarity.number_of_edges(), selector=control,
                serialization="newline between walks; no final newline", em_reused=control == "teleport",
                em_seconds=em_seconds, walk_seconds=walk_seconds, total_seconds=time.monotonic()-started)
    if audits is not None:
        audit_path = results / "worker_selector_audit.json"
        save_json(audit_path, dict(audits, fingerprint=fingerprint))
        meta.update(worker_audit_sha256=sha256(audit_path), selector_calls=audits["selector_calls"])
    save_json(marker, meta)
    print("CORRECTED MESH2024 CORPUS SAVED:", corpus, flush=True)


def compare_teleport(data, results, fingerprint):
    reference = semantic_reference()
    base_data = ROOT / "data/embeddings" / SEMANTIC_RUN_NAME
    base_results = ROOT / "results" / SEMANTIC_RUN_NAME
    fixed_embedding, fixed_evaluation = shared.require_evaluation(0, data, results, fingerprint)
    original_embedding, original_evaluation = shared.require_evaluation(0, base_data, base_results, reference["fingerprint"])
    corpus = require_corpus(data, results, fingerprint, "teleport")
    for key in ("binary_sha256", "token_map_sha256"):
        require_equal(fixed_embedding[key], original_embedding[key], "Comparison " + key)
    require_equal(fixed_embedding["corpus_sha256"], corpus["sha256"], "Fixed embedding corpus")
    require_equal(fixed_evaluation["fold_assignments_sha256"], original_evaluation["fold_assignments_sha256"], "Comparison folds")
    require_equal(corpus["matrix_sha256"], reference["artifacts"]["matrix"], "Shared new Section 7 EM matrix")
    rows = teleport.comparison_rows(previous.read_rows(base_results / "pp0_summary.csv", ","),
                                    previous.read_rows(results / "pp0_summary.csv", ","))
    deltas = teleport.paired_fold_rows(previous.read_rows(base_results / "pp0_fold_results.csv", ","),
                                      previous.read_rows(results / "pp0_fold_results.csv", ","))
    output = results / "teleport_selector_comparison.csv"
    previous.write_csv(output, rows, list(rows[0]))
    previous.write_csv(results / "paired_fold_deltas.csv", deltas, list(deltas[0]))
    save_json(results / "comparison_complete.json", dict(fingerprint=fingerprint, sha256=sha256(output),
              paired_fold_deltas_sha256=sha256(results / "paired_fold_deltas.csv"),
              original_summary_sha256=original_evaluation["sha256"], fixed_summary_sha256=fixed_evaluation["sha256"],
              reference_run=SEMANTIC_RUN_NAME, shared_matrix_sha256=corpus["matrix_sha256"],
              shared_similarity_sha256=sha256(inputs.SIMILARITY_PATH)))
    print(json.dumps(rows, indent=2), flush=True)


def main(control):
    phases = ("hierarchy", "similarity", "config", "prepare", "walks", "pp1", "evaluate-pp1", "pp0", "evaluate-pp0", "compare")
    if control == "teleport":
        phases = ("config", "prepare", "walks", "embedding", "evaluate", "compare")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=phases)
    parser.add_argument("--run-name", default=SEMANTIC_RUN_NAME if control == "semantic" else TELEPORT_RUN_NAME)
    args = parser.parse_args()
    if args.phase in ("hierarchy", "similarity"):
        build = inputs.build_hierarchy if args.phase == "hierarchy" else inputs.build_similarity
        print(json.dumps(build(), indent=2), flush=True)
        return
    data, results, fingerprint = context(args.run_name, control)
    if args.phase == "config":
        print((results / "run_config.json").read_text(), flush=True)
    elif args.phase == "prepare":
        prepare(data, results, fingerprint, control)
    elif args.phase == "walks":
        generate_corpus(data, results, fingerprint, control)
    elif args.phase in ("pp1", "pp0", "embedding"):
        pp = 0 if args.phase == "embedding" else int(args.phase[-1])
        require_corpus(data, results, fingerprint, control)
        paired.train(pp, data, results, fingerprint)
    elif args.phase.startswith("evaluate"):
        pp = 0 if args.phase == "evaluate" else int(args.phase[-1])
        require_corpus(data, results, fingerprint, control)
        shared.evaluate(pp, data, results, fingerprint)
    elif control == "semantic":
        require_corpus(data, results, fingerprint, control)
        shared.compare(data, results, fingerprint)
    else:
        compare_teleport(data, results, fingerprint)
