"""DIAGNOSTIC / SENSITIVITY ANALYSIS of the formal repeat-0 result.

Each phase writes only diagnostic outputs. The accepted graph, semantics,
labels, formal embedding, and formal result files are read-only inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import random
import resource
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/DREAMwalk_upstream"
sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(ROOT / "src"))

from run_msi_repeat0 import (  # noqa: E402
    GRAPH, NODETYPES, POSITIVES, NEGATIVES, SIMILARITY, EMBEDDING,
    HIERARCHY, CONFIG, FOLDS, SUMMARY, base_inputs, check_graph_leakage,
    read_rows, sha256, xgb_settings,
)

RESULTS = ROOT / "results"
DIAG_EMBEDDING = ROOT / "data/embeddings/msi_seed0_no_teleport.pkl"
SEMANTIC_FOLDS = RESULTS / "diagnostic_repeat0_10fold.csv"
NEGATIVE_SETS = RESULTS / "diagnostic_negative_sets.csv"
NEGATIVE_FOLDS = RESULTS / "diagnostic_negative_sampling_folds.csv"
NEGATIVE_SUMMARY = RESULTS / "diagnostic_negative_sampling.csv"
NEGATIVE_ACROSS = RESULTS / "diagnostic_negative_sampling_across_seeds.csv"
TRACE = RESULTS / "diagnostic_teleport_trace_summary.json"
SELECTION_PROBE = RESULTS / "diagnostic_teleport_selection_probe.json"
HASH_PROBES = {seed: RESULTS / f"diagnostic_teleport_hashseed{seed}.json" for seed in (0, 1)}
HASH_COMPARISON = RESULTS / "diagnostic_teleport_hashseed_comparison.csv"
HASH_SUMMARY = RESULTS / "diagnostic_teleport_hashseed_summary.json"
CONTROL_FOLDS = RESULTS / "diagnostic_no_teleport_fold_results.csv"
CONTROL_SUMMARY = RESULTS / "diagnostic_no_teleport_summary.csv"
COMPARISON = RESULTS / "diagnostic_repeat0_comparison.csv"
CONTROL_TIMING = RESULTS / "diagnostic_no_teleport_timing.json"
FORMAL_FILES = (GRAPH, NODETYPES, POSITIVES, NEGATIVES, SIMILARITY,
                HIERARCHY, EMBEDDING, CONFIG, FOLDS, SUMMARY)
METRICS = ("accuracy", "auroc", "aupr")
N_FOLDS = 10
SEED = 0


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def formal_hashes():
    return {str(path.relative_to(ROOT)): sha256(path) for path in FORMAL_FILES}


def verify_formal_files(before):
    assert formal_hashes() == before, "A formal input/output changed during diagnostics"


def checked_inputs():
    types, drugs, diseases, positives = base_inputs()
    check_graph_leakage(types, positives)
    negatives = [(r["drug"], r["disease"]) for r in read_rows(NEGATIVES, ",")]
    assert len(negatives) == len(set(negatives)) == len(positives) == 5926
    assert not set(negatives) & positives
    return types, drugs, diseases, positives, negatives


def load_embedding(path, types, pairs):
    import numpy as np
    with path.open("rb") as stream:
        embedding = pickle.load(stream)
    assert len(embedding) == len(types) == 29959
    assert all(np.asarray(v).shape == (128,) and np.isfinite(v).all()
               for v in embedding.values())
    assert all(a in embedding and b in embedding for a, b in pairs)
    return embedding


def evaluate(embedding, positives, negatives, negative_seed, label):
    import numpy as np
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier

    assert len(negatives) == len(set(negatives)) == len(positives) == 5926
    assert not set(negatives) & positives
    pairs = sorted(positives) + negatives
    x = np.asarray([embedding[a] - embedding[b] for a, b in pairs], dtype=np.float32)
    y = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int8)
    assert x.shape == (11852, 128) and np.isfinite(x).all()
    random.seed(SEED)
    np.random.seed(SEED)
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    rows = []
    seen_test = set()
    for fold, (train, test) in enumerate(splitter.split(x, y), 1):
        assert not set(train) & set(test)
        assert not seen_test & set(test)
        seen_test.update(test)
        assert set(y[train]) == set(y[test]) == {0, 1}
        start = time.monotonic()
        model = XGBClassifier(**xgb_settings())
        model.fit(x[train], y[train])
        prob = model.predict_proba(x[test])[:, 1]
        row = dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS", label=label,
                   embedding="semantic_tau_0.3" if label != "no_teleport" else "no_teleport_tau_0",
                   negative_seed=negative_seed, fold=fold, cv_seed=SEED,
                   accuracy=accuracy_score(y[test], prob.round()),
                   auroc=roc_auc_score(y[test], prob),
                   aupr=average_precision_score(y[test], prob),
                   n_train=len(train), n_test=len(test),
                   n_positive_train=int(y[train].sum()),
                   n_positive_test=int(y[test].sum()),
                   runtime_seconds=time.monotonic() - start)
        assert all(math.isfinite(row[metric]) for metric in METRICS)
        rows.append(row)
        print(label, "negative seed", negative_seed, "fold", fold,
              {key: round(row[key], 6) for key in METRICS}, flush=True)
    assert len(rows) == N_FOLDS and seen_test == set(range(11852))
    return rows


def metric_summary(rows, negative_seed, label):
    result = []
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        result.append(dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS", label=label,
                           negative_seed=negative_seed, metric=metric,
                           mean=statistics.mean(values), SD=statistics.stdev(values),
                           min=min(values), max=max(values), n=len(values)))
    return result


def phase_evaluate_semantic():
    before = formal_hashes()
    types, drugs, diseases, positives, saved_negatives = checked_inputs()
    embedding = load_embedding(EMBEDDING, types, sorted(positives) + saved_negatives)
    candidates = [(a, b) for a in drugs for b in diseases if (a, b) not in positives]
    assert random.Random(0).sample(candidates, len(positives)) == saved_negatives
    all_fold_rows = []
    summary_rows = []
    negative_set_rows = []
    started = time.monotonic()
    for negative_seed in range(5):
        negatives = (saved_negatives if negative_seed == 0 else
                     random.Random(negative_seed).sample(candidates, len(positives)))
        assert len(negatives) == len(set(negatives)) == 5926 and not set(negatives) & positives
        negative_set_rows.extend(dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                                      negative_seed=negative_seed, drug=a, disease=b)
                                 for a, b in negatives)
        rows = evaluate(embedding, positives, negatives, negative_seed, "semantic_tau_0.3")
        all_fold_rows.extend(rows)
        summary_rows.extend(metric_summary(rows, negative_seed, "semantic_tau_0.3"))
        if negative_seed == 0:
            write_csv(SEMANTIC_FOLDS, rows, list(rows[0]))
        print("NEGATIVE-SET RESULT", negative_seed,
              {r["metric"]: r["mean"] for r in summary_rows[-3:]}, flush=True)
    write_csv(NEGATIVE_SETS, negative_set_rows, list(negative_set_rows[0]))
    write_csv(NEGATIVE_FOLDS, all_fold_rows, list(all_fold_rows[0]))
    write_csv(NEGATIVE_SUMMARY, summary_rows, list(summary_rows[0]))
    across = []
    for metric in METRICS:
        values = [r["mean"] for r in summary_rows if r["metric"] == metric]
        assert len(values) == 5
        across.append(dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS", metric=metric,
                           mean=statistics.mean(values), SD=statistics.stdev(values),
                           min=min(values), max=max(values), n_negative_sets=5))
    write_csv(NEGATIVE_ACROSS, across, list(across[0]))
    verify_formal_files(before)
    print("SEMANTIC EVALUATION COMPLETE", json.dumps({
        "seconds": time.monotonic() - started,
        "same_pairs_10fold": {r["metric"]: r["mean"] for r in summary_rows[:3]},
        "negative_sampling_across_seeds": across}, indent=2), flush=True)


def phase_trace():
    import numpy as np
    from DREAMwalk import generate_embeddings as ge
    from DREAMwalk.generate_similarity_net import _generate_tree, _ic_from_tree, _simJC_from_tree
    from DREAMwalk.utils import read_graph
    import pandas as pd

    before = formal_hashes()
    types, drugs, diseases, positives, saved_negatives = checked_inputs()
    ge.set_seed(SEED)
    started = time.monotonic()
    graph = read_graph(str(GRAPH), weighted=True, directed=False)
    sim_graph = read_graph(str(SIMILARITY), weighted=True, directed=False)
    assert set(sim_graph) <= set(graph)
    transition = ge.train_edgetype_transition_matrix(5, graph, str(GRAPH), "\t", 10, 1, 1)
    assert np.isfinite(transition).all()
    print("TRACE transition matrix trained", round(time.monotonic() - started, 2), flush=True)

    calls = []
    original = ge._teleport_operation
    def observed_teleport(cur, g_sim):
        destination = original(cur, g_sim)
        links = g_sim.get_edge_data(cur, destination) if destination is not False else None
        weights = [float(edge["weight"]) for edge in links.values()] if links else []
        calls.append((cur, destination, weights))
        return destination

    ge._teleport_operation = observed_teleport
    try:
        starts = random.Random(SEED).sample(list(graph.nodes()), 12000)
        paths = [ge._DREAMwalker(node, graph, sim_graph, transition, 1, 1, 10, 0.3)
                 for node in starts]
    finally:
        ge._teleport_operation = original
    eligible = sum(cur in sim_graph and graph.degree(cur) > 0
                   for path in paths for cur in path[1:-1])
    decisions = len(calls)
    total_steps = sum(len(path) - 1 for path in paths)
    assert eligible > 1000 and decisions > 100
    assert 0.25 < decisions / eligible < 0.35, (decisions, eligible)
    breakdown = Counter()
    chosen_weights = []
    for source, destination, weights in calls:
        assert source in sim_graph and destination in sim_graph and destination in graph
        assert types[source] in ("drug", "disease")
        if types[source] == "drug" and types[destination] == "drug":
            breakdown["drug_to_drug"] += 1
        elif types[source] == "disease" and types[destination] == "disease":
            breakdown["disease_to_disease"] += 1
        else:
            breakdown["invalid_cross_type"] += 1
        assert len(weights) == 1 and 0.4 < weights[0] <= 1
        chosen_weights.append(weights[0])
    assert breakdown["invalid_cross_type"] == 0
    assert sum(breakdown.values()) == decisions

    # Inspect high-weight pairs against original hierarchy paths and the
    # upstream similarity function. This tests structural consistency only.
    hierarchy_df = pd.read_csv(HIERARCHY)
    tree = _generate_tree(hierarchy_df, list(graph.nodes()))
    ic = _ic_from_tree(tree, list(graph.nodes()))
    examples = []
    for kind in ("drug", "disease"):
        edges = []
        for a, b, data in sim_graph.edges(data=True):
            if types[a] == kind and types[b] == kind:
                edges.append((float(data["weight"]), a, b))
        used = set()
        selected = []
        for weight, a, b in sorted(edges, reverse=True):
            if a not in used and b not in used:
                selected.append((weight, a, b))
                used.update((a, b))
            if len(selected) == 3:
                break
        assert len(selected) == 3
        for weight, a, b in selected:
            recalculated = _simJC_from_tree(a, b, tree[kind], ic[kind])
            assert abs(weight - recalculated) < 1e-12
            paths_a = tree[kind][a]
            paths_b = tree[kind][b]
            common = sorted(set().union(*(set(p) for p in paths_a)) &
                            set().union(*(set(p) for p in paths_b)) - {kind})
            assert common
            examples.append(dict(type=kind, node_a=a, node_b=b, similarity=weight,
                                 path_count_a=len(paths_a), path_count_b=len(paths_b),
                                 example_path_a=paths_a[0], example_path_b=paths_b[0],
                                 common_hierarchy_nodes=common))

    summary = dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                   sampled_walks=len(paths), sampled_start_types=dict(Counter(types[n] for n in starts)),
                   total_walk_steps=total_steps, eligible_semantic_node_decisions=eligible,
                   teleport_decisions=decisions,
                   conditional_teleport_decision_rate=decisions / eligible,
                   teleport_fraction_of_all_steps=decisions / total_steps,
                   drug_to_drug_teleports=breakdown["drug_to_drug"],
                   disease_to_disease_teleports=breakdown["disease_to_disease"],
                   invalid_cross_type_teleports=breakdown["invalid_cross_type"],
                   teleports_outside_molecular_graph=0,
                   selected_similarity_weight_distribution=dict(
                       min=min(chosen_weights), median=statistics.median(chosen_weights),
                       mean=statistics.mean(chosen_weights), max=max(chosen_weights),
                       n=len(chosen_weights)),
                   high_weight_ontology_path_examples=examples,
                   trace_seconds=time.monotonic() - started,
                   method="unmodified upstream DREAMwalker; wrapper counts original teleport calls and destinations")
    write_json(TRACE, summary)
    verify_formal_files(before)
    print("TELEPORT TRACE COMPLETE", json.dumps({key: summary[key] for key in (
        "sampled_walks", "total_walk_steps", "eligible_semantic_node_decisions",
        "teleport_decisions", "conditional_teleport_decision_rate",
        "drug_to_drug_teleports", "disease_to_disease_teleports",
        "invalid_cross_type_teleports", "teleports_outside_molecular_graph",
        "selected_similarity_weight_distribution", "trace_seconds")}, indent=2), flush=True)


def phase_probe_selection():
    """Repeatedly call the unmodified upstream selector on fixed sources."""
    from DREAMwalk.generate_embeddings import _teleport_operation
    from DREAMwalk.utils import read_graph

    before = formal_hashes()
    types, drugs, diseases, positives, saved_negatives = checked_inputs()
    sim_graph = read_graph(str(SIMILARITY), weighted=True, directed=False)
    random.seed(0)
    import numpy as np
    np.random.seed(0)
    examples = []
    for kind in ("drug", "disease"):
        candidates = sorted(n for n in sim_graph if types[n] == kind and 10 <= sim_graph.degree(n) <= 30)
        assert len(candidates) >= 5
        for source in random.Random(0).sample(candidates, 5):
            counts = Counter(_teleport_operation(source, sim_graph) for _ in range(200))
            assert set(counts) <= set(sim_graph.neighbors(source))
            weights = {neighbor: sum(float(e["weight"]) for e in sim_graph[source][neighbor].values())
                       for neighbor in sim_graph.neighbors(source)}
            total_weight = sum(weights.values())
            assert total_weight > 0
            examples.append(dict(type=kind, source=source, neighbor_count=len(weights),
                                 calls=200, unique_selected_destinations=len(counts),
                                 observed_destinations=dict(counts),
                                 selected_destination_weights={n: weights[n] for n in counts},
                                 selected_destination_weight_share={n: weights[n]/total_weight for n in counts},
                                 max_single_neighbor_weight_share=max(weights.values())/total_weight))
    result = dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                  method="200 direct calls per source to unchanged upstream _teleport_operation",
                  probes=examples,
                  sources_with_one_selected_destination=sum(e["unique_selected_destinations"] == 1 for e in examples),
                  total_sources=len(examples),
                  source_code_observation="The upstream loop breaks only its inner edge loop after threshold; later neighbors can overwrite the selected destination.")
    write_json(SELECTION_PROBE, result)
    verify_formal_files(before)
    print("TELEPORT SELECTION PROBE", json.dumps(result, indent=2), flush=True)


def phase_probe_selection_hash(calls_per_source, output):
    """Probe the unchanged selector in a fresh, startup-hash-seeded process."""
    import numpy as np
    from DREAMwalk.generate_embeddings import _teleport_operation
    from DREAMwalk.utils import read_graph

    hash_seed = os.environ.get("PYTHONHASHSEED")
    assert hash_seed in ("0", "1"), "Launch a fresh interpreter with PYTHONHASHSEED=0 or 1"
    assert calls_per_source == 1000
    output = output.resolve()
    assert output == HASH_PROBES[int(hash_seed)].resolve()
    assert not output.exists(), "Refusing to overwrite an existing hash-seed probe"
    original = json.loads(SELECTION_PROBE.read_text(encoding="utf-8"))
    sources = [probe["source"] for probe in original["probes"]]
    assert len(sources) == len(set(sources)) == 10
    random.seed(0)
    np.random.seed(0)
    sim_graph = read_graph(str(SIMILARITY), weighted=True, directed=False)
    hash_order_ids = sources + ["DB00001", "DB00002", "C0000737", "C0000744"]
    hash_order_probe = list(set(hash_order_ids))
    results = []
    for original_probe in original["probes"]:
        source = original_probe["source"]
        assert source in sim_graph
        weights = {neighbor: sum(float(edge["weight"]) for edge in sim_graph[source][neighbor].values())
                   for neighbor in sim_graph.neighbors(source)}
        total_weight = sum(weights.values())
        assert len(weights) == original_probe["neighbor_count"] and total_weight > 0
        counts = Counter(_teleport_operation(source, sim_graph) for _ in range(calls_per_source))
        assert sum(counts.values()) == calls_per_source and set(counts) <= set(weights)
        expected = {neighbor: weight / total_weight for neighbor, weight in weights.items()}
        top_observed = min(counts, key=lambda neighbor: (-counts[neighbor], neighbor))
        top_expected = min(expected, key=lambda neighbor: (-expected[neighbor], neighbor))
        tv_expected = 0.5 * sum(abs(counts.get(neighbor, 0) / calls_per_source - probability)
                                for neighbor, probability in expected.items())
        results.append(dict(type=original_probe["type"], source=source,
                            semantic_neighbor_count=len(weights), total_semantic_weight=total_weight,
                            calls=calls_per_source, observed_counts=dict(counts),
                            unique_destinations=len(counts), top_destination=top_observed,
                            top_destination_frequency=counts[top_observed] / calls_per_source,
                            expected_weight_probabilities=expected,
                            expected_top_destination=top_expected,
                            expected_top_probability=expected[top_expected],
                            observed_frequency_of_expected_top=counts.get(top_expected, 0) / calls_per_source,
                            tv_observed_vs_expected_weighted=tv_expected))
    data = dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                method="unchanged upstream _teleport_operation direct calls in a fresh interpreter",
                startup=dict(pythonhashseed=hash_seed, python_version=sys.version,
                             process_id=os.getpid(), python_random_seed=0, numpy_seed=0,
                             hash_order_fixed_ids=hash_order_ids,
                             hash_order_set_iteration=hash_order_probe),
                semantic_graph_sha256=sha256(SIMILARITY),
                original_source_probe_sha256=sha256(SELECTION_PROBE),
                calls_per_source=calls_per_source, source_ids=sources, probes=results)
    write_json(output, data)
    print("HASH-SEED TELEPORT PROBE COMPLETE", json.dumps(dict(
        startup=data["startup"], source_ids=sources,
        top_destinations={r["source"]: r["top_destination"] for r in results}), indent=2), flush=True)


def phase_compare_selection_hash():
    probes = {seed: json.loads(HASH_PROBES[seed].read_text(encoding="utf-8")) for seed in (0, 1)}
    a, b = probes[0], probes[1]
    assert a["startup"]["pythonhashseed"] == "0" and b["startup"]["pythonhashseed"] == "1"
    assert a["startup"]["process_id"] != b["startup"]["process_id"]
    assert all(p["startup"]["python_random_seed"] == p["startup"]["numpy_seed"] == 0
               for p in (a, b))
    assert a["calls_per_source"] == b["calls_per_source"] == 1000
    assert a["source_ids"] == b["source_ids"]
    assert a["semantic_graph_sha256"] == b["semantic_graph_sha256"] == sha256(SIMILARITY)
    assert a["original_source_probe_sha256"] == b["original_source_probe_sha256"] == sha256(SELECTION_PROBE)
    assert a["startup"]["hash_order_fixed_ids"] == b["startup"]["hash_order_fixed_ids"]
    hash_order_changed = a["startup"]["hash_order_set_iteration"] != b["startup"]["hash_order_set_iteration"]
    assert hash_order_changed, "The fixed-list set iteration order did not differ between startup hash seeds"
    rows = []
    for p0, p1 in zip(a["probes"], b["probes"]):
        assert p0["source"] == p1["source"] and p0["type"] == p1["type"]
        assert p0["semantic_neighbor_count"] == p1["semantic_neighbor_count"]
        assert abs(p0["total_semantic_weight"] - p1["total_semantic_weight"]) < 1e-12
        assert p0["expected_weight_probabilities"] == p1["expected_weight_probabilities"]
        c0, c1 = p0["observed_counts"], p1["observed_counts"]
        observed_union = set(c0) | set(c1)
        observed_intersection = set(c0) & set(c1)
        tv = 0.5 * sum(abs(c0.get(node, 0)/1000 - c1.get(node, 0)/1000)
                       for node in observed_union)
        assert -1e-12 <= tv <= 1 + 1e-12
        rows.append(dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                         source=p0["source"], type=p0["type"],
                         semantic_neighbor_count=p0["semantic_neighbor_count"],
                         total_semantic_weight=p0["total_semantic_weight"],
                         hash0_top_destination=p0["top_destination"],
                         hash0_top_frequency=p0["top_destination_frequency"],
                         hash0_unique_destinations=p0["unique_destinations"],
                         hash1_top_destination=p1["top_destination"],
                         hash1_top_frequency=p1["top_destination_frequency"],
                         hash1_unique_destinations=p1["unique_destinations"],
                         top_destination_changed=p0["top_destination"] != p1["top_destination"],
                         observed_destination_overlap_count=len(observed_intersection),
                         observed_destination_jaccard=len(observed_intersection)/len(observed_union),
                         tv_hash0_vs_hash1=tv,
                         expected_top_destination=p0["expected_top_destination"],
                         expected_top_probability=p0["expected_top_probability"],
                         hash0_observed_frequency_of_expected_top=p0["observed_frequency_of_expected_top"],
                         hash1_observed_frequency_of_expected_top=p1["observed_frequency_of_expected_top"],
                         tv_hash0_vs_expected_weighted=p0["tv_observed_vs_expected_weighted"],
                         tv_hash1_vs_expected_weighted=p1["tv_observed_vs_expected_weighted"]))
    write_csv(HASH_COMPARISON, rows, list(rows[0]))
    tvs = [r["tv_hash0_vs_hash1"] for r in rows]
    changed = sum(r["top_destination_changed"] for r in rows)
    category = ("A: large hash-order effect" if changed >= 5 and statistics.median(tvs) >= 0.25 else
                "B: little hash-order effect" if changed == 0 and max(tvs) < 0.1 else
                "C: mixed or source-specific effect")
    summary = dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                   source_count=len(rows), calls_per_source=1000,
                   startup_process_ids={str(seed): probes[seed]["startup"]["process_id"] for seed in (0, 1)},
                   startup_pythonhashseeds=["0", "1"], python_random_seed=0, numpy_seed=0,
                   fixed_id_set_iteration_order_changed=hash_order_changed,
                   top_destination_changed_sources=changed,
                   tv_hash0_vs_hash1=dict(mean=statistics.mean(tvs), median=statistics.median(tvs),
                                          min=min(tvs), max=max(tvs)),
                   source_counts_by_tv_threshold={"ge_0.1": sum(v >= 0.1 for v in tvs),
                                                  "ge_0.25": sum(v >= 0.25 for v in tvs),
                                                  "ge_0.5": sum(v >= 0.5 for v in tvs)},
                   tv_vs_expected_weighted=dict(
                       hash0_mean=statistics.mean(r["tv_hash0_vs_expected_weighted"] for r in rows),
                       hash1_mean=statistics.mean(r["tv_hash1_vs_expected_weighted"] for r in rows),
                       hash0_median=statistics.median(r["tv_hash0_vs_expected_weighted"] for r in rows),
                       hash1_median=statistics.median(r["tv_hash1_vs_expected_weighted"] for r in rows)),
                   category=category)
    write_json(HASH_SUMMARY, summary)
    print("HASH-SEED TELEPORT COMPARISON", json.dumps(summary, indent=2), flush=True)


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if platform.system() == "Darwin" else value * 1024


def phase_embed_no_teleport():
    from DREAMwalk import generate_embeddings as ge
    import numpy as np

    before = formal_hashes()
    types, drugs, diseases, positives, saved_negatives = checked_inputs()
    assert SEMANTIC_FOLDS.exists() and NEGATIVE_ACROSS.exists() and TRACE.exists(), "Parts A and B must finish first"
    assert not DIAG_EMBEDDING.exists(), "Diagnostic control embedding already exists; refusing overwrite"
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
    work = ROOT / "logs/diagnostic_no_teleport_work"
    work.mkdir(parents=True, exist_ok=True)
    link = work / "DREAMwalk"
    if not link.exists():
        link.symlink_to(UPSTREAM / "DREAMwalk", target_is_directory=True)
    os.environ["CPATH"] = "/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk/usr/include/c++/v1"
    os.chdir(work)
    started = time.monotonic()
    print("DIAGNOSTIC / SENSITIVITY ANALYSIS: starting no-teleport full embedding", flush=True)
    ge.save_embedding_files(netf=str(GRAPH), sim_netf=str(SIMILARITY),
                            outputf=str(DIAG_EMBEDDING), nodetypef=str(NODETYPES),
                            tp_factor=0, seed=0, num_walks=100, walk_length=10,
                            dimension=128, window_size=4, workers=4,
                            em_max_iter=5, p=1, q=1)
    elapsed = time.monotonic() - started
    assert actual_walks.get("count") == 2995900
    embedding = load_embedding(DIAG_EMBEDDING, types, sorted(positives) + saved_negatives)
    timing = dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS",
                  control="no_teleport_tau_0", seed=0, workers=4,
                  input_graph_sha256=before[str(GRAPH.relative_to(ROOT))],
                  input_similarity_sha256=before[str(SIMILARITY.relative_to(ROOT))],
                  embedding_sha256=sha256(DIAG_EMBEDDING),
                  expected_walks=2995900, actual_walks=actual_walks["count"],
                  embedding_nodes=len(embedding), embedding_dimension=128,
                  embedding_total_seconds=elapsed, peak_parent_rss_bytes=peak_rss_bytes(),
                  **stage)
    write_json(CONTROL_TIMING, timing)
    verify_formal_files(before)
    print("NO-TELEPORT EMBEDDING COMPLETE", json.dumps(timing, indent=2), flush=True)


def phase_evaluate_no_teleport():
    before = formal_hashes()
    types, drugs, diseases, positives, saved_negatives = checked_inputs()
    assert DIAG_EMBEDDING.exists()
    embedding = load_embedding(DIAG_EMBEDDING, types, sorted(positives) + saved_negatives)
    rows = evaluate(embedding, positives, saved_negatives, 0, "no_teleport")
    summary = metric_summary(rows, 0, "no_teleport")
    write_csv(CONTROL_FOLDS, rows, list(rows[0]))
    write_csv(CONTROL_SUMMARY, summary, list(summary[0]))
    verify_formal_files(before)
    print("NO-TELEPORT EVALUATION COMPLETE", json.dumps(summary, indent=2), flush=True)


def phase_compare():
    before = formal_hashes()
    five_fold = {r["metric"]: float(r["mean"]) for r in read_rows(SUMMARY, ",")}
    semantic = {r["metric"]: float(r["mean"]) for r in read_rows(NEGATIVE_SUMMARY, ",")
                if r["negative_seed"] == "0"}
    control = {r["metric"]: float(r["mean"]) for r in read_rows(CONTROL_SUMMARY, ",")}
    assert set(five_fold) == set(semantic) == set(control) == set(METRICS)
    paper_semantic = dict(accuracy=0.8770983, auroc=0.9440605, aupr=0.9467667)
    paper_control = dict(accuracy=None, auroc=0.905, aupr=0.912)
    rows = []
    for metric in METRICS:
        rows.append(dict(analysis="DIAGNOSTIC / SENSITIVITY ANALYSIS", metric=metric,
                         formal_semantic_5fold=five_fold[metric],
                         same_embedding_same_pairs_semantic_10fold=semantic[metric],
                         fold_count_delta_10fold_minus_5fold=semantic[metric] - five_fold[metric],
                         same_pairs_no_teleport_10fold=control[metric],
                         semantic_minus_no_teleport=semantic[metric] - control[metric],
                         paper_semantic=paper_semantic[metric],
                         paper_no_teleport_approx=paper_control[metric] if paper_control[metric] is not None else ""))
    write_csv(COMPARISON, rows, list(rows[0]))
    verify_formal_files(before)
    print("DIAGNOSTIC COMPARISON", json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("evaluate-semantic", "trace", "probe-selection",
                                          "probe-selection-hash", "compare-selection-hash",
                                          "embed-no-teleport", "evaluate-no-teleport", "compare"))
    parser.add_argument("--calls-per-source", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.phase == "probe-selection-hash":
        assert args.output is not None
        phase_probe_selection_hash(args.calls_per_source, args.output)
        return
    if args.phase == "compare-selection-hash":
        phase_compare_selection_hash()
        return
    {"evaluate-semantic": phase_evaluate_semantic, "trace": phase_trace,
     "probe-selection": phase_probe_selection,
     "embed-no-teleport": phase_embed_no_teleport,
     "evaluate-no-teleport": phase_evaluate_no_teleport,
     "compare": phase_compare}[args.phase]()


if __name__ == "__main__":
    main()
