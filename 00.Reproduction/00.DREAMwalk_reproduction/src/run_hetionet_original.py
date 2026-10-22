"""User-started Hetionet run of the unmodified DREAMwalk v1.0.0 pipeline.

Dataset preparation is separate from model execution. Importing this module
does not download data, generate walks, train embeddings, or fit classifiers.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/DREAMwalk_upstream"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(UPSTREAM))
import hetionet_inputs as inputs

PINNED_COMMIT = "2f93723c3fbfa899d0f23c02e3459de8bfac0581"
RUN_NAME = "hetionet_seed0_original"
SETTINGS = dict(tp_factor=0.3, cutoff=0.4, seed=0, num_walks=100,
                walk_length=10, dimension=128, window_size=4, workers=4,
                em_max_iter=5, p=1, q=1, pp=1, epochs=1, negative=5,
                sample=0.001, folds=10, cv_seed=0)
PINS = {"numpy": "1.23.3", "scipy": "1.9.1", "scikit-learn": "1.1.3",
        "networkx": "2.8.8", "pandas": "1.4.4", "tqdm": "4.64.1",
        "parmap": "1.6.0", "xgboost": "1.7.4"}
METRICS = ("accuracy", "auroc", "aupr")
NEGATIVE_POLICY = ("random.Random(0).sample over sorted active drug x disease pairs, "
                   "excluding all raw CtD positives; CpD is a different relation, "
                   "not added to CtD labels or exclusions; sample saved once")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def rows(path, delimiter=","):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def save_csv(path, values, columns):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def same(actual, expected, label):
    if actual != expected:
        raise RuntimeError(label + " differs; preserve existing outputs and use a new run name")


def run_paths(name):
    if not name.startswith("hetionet_") or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("Use a separate hetionet_ run name with letters, digits, underscore or hyphen")
    result = ROOT / "results" / name
    data = ROOT / "data/embeddings" / name
    return data, result


def environment():
    if sys.version_info[:2] != (3, 8):
        raise RuntimeError("Use environment/.conda-env/bin/python (the existing Python 3.8 environment)")
    packages = {name: version(name) for name in PINS}
    same(packages, PINS, "Package pins")
    commit = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    same(commit, PINNED_COMMIT, "Upstream commit")
    # Check tracked source, including staged changes, before calling original code.
    subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--exit-code", "HEAD", "--"],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return dict(packages=packages, upstream_commit=commit)


def xgb_settings():
    # Same released constructor and local MSI evaluator; no parameter search.
    return dict(base_score=0.5, booster="gbtree", eval_metric="error",
                objective="binary:logistic", gamma=0, learning_rate=0.1,
                max_depth=6, n_estimators=500, tree_method="auto",
                min_child_weight=4, subsample=0.8, colsample_bytree=0.9,
                scale_pos_weight=1, max_delta_step=1, seed=0, n_jobs=4)


def input_data():
    paths = inputs.paths(ROOT)
    types = {r["node"]: r["type"] for r in rows(paths["nodetypes"], "\t")}
    positives = sorted({(r["drug"], r["disease"]) for r in rows(paths["positives"])})
    raw = {(r["drug"], r["disease"]) for r in rows(paths["raw_positives"])}
    same(len(types), 21896, "Incident molecular nodes")
    same(len(positives), 751, "Evaluable CtD positives")
    same(len(raw), 755, "Raw CtD positives")
    if not set(positives) <= raw or any(a not in types or b not in types for a, b in positives):
        raise ValueError("Positive labels have missing graph nodes")
    return paths, types, positives, raw


def sample_negatives(types, positives, all_positives):
    drugs = sorted(n for n, kind in types.items() if kind == "drug")
    diseases = sorted(n for n, kind in types.items() if kind == "disease")
    candidates = [(a, b) for a in drugs for b in diseases if (a, b) not in all_positives]
    return random.Random(SETTINGS["seed"]).sample(candidates, len(positives))


def fold_rows(positives, negatives):
    import numpy as np
    from sklearn.model_selection import StratifiedKFold
    pairs = list(positives) + list(negatives)
    labels = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int8)
    assigned = {}
    splitter = StratifiedKFold(n_splits=SETTINGS["folds"], shuffle=True, random_state=SETTINGS["cv_seed"])
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(pairs)), labels), 1):
        assigned.update({int(i): fold for i in test})
    return [dict(pair_index=i, drug=a, disease=b, label=int(labels[i]), test_fold=assigned[i])
            for i, (a, b) in enumerate(pairs)]


def identity():
    env = environment()
    paths, _, _, _ = input_data()
    source_paths = [Path(__file__), Path(inputs.__file__)] + [
        UPSTREAM / "DREAMwalk" / name for name in
        ("generate_embeddings.py", "generate_similarity_net.py", "HeterogeneousSG.py",
         "HeterogeneousSG.cpp", "utils.py")]
    return dict(settings=SETTINGS, xgboost=xgb_settings(), negative_policy=NEGATIVE_POLICY,
                feature="drug_embedding - disease_embedding",
                inputs={key: sha256(paths[key]) for key in
                        ("graph", "nodetypes", "positives", "raw_positives", "hierarchy", "manifest")},
                sources={str(path.relative_to(ROOT)): sha256(path) for path in source_paths}, **env)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def context(name):
    data, result = run_paths(name)
    config = read_json(result / "run_config.json")
    same(config["fingerprint"], fingerprint(identity()), "Frozen input/code/settings identity")
    for filename, digest in config["pair_artifacts"].items():
        same(sha256(result / filename), digest, "Saved " + filename)
    return data, result, config["fingerprint"]


def prepare(name):
    inputs.prepare_inputs(ROOT)
    current = identity()
    fp = fingerprint(current)
    data, result = run_paths(name)
    if (result / "run_config.json").exists():
        context(name)
        print("Prepared inputs and saved negative/fold IDs reused", flush=True)
        return
    if ((data.exists() and any(data.iterdir())) or (result.exists() and any(result.iterdir()))):
        raise RuntimeError("Unregistered run outputs exist; preserve them and use a new hetionet_ run name")
    data.mkdir(parents=True, exist_ok=True)
    result.mkdir(parents=True, exist_ok=True)
    paths, types, positives, raw = input_data()
    negatives = sample_negatives(types, positives, raw)
    pairs = [dict(drug=a, disease=b) for a, b in negatives]
    save_csv(result / "negative_pairs.csv", pairs, ["drug", "disease"])
    folds = fold_rows(positives, negatives)
    save_csv(result / "fold_assignments.csv", folds, list(folds[0]))
    with gzip.open(paths["raw_dir"] / "hetionet-v1.0-edges.sif.gz", "rt") as stream:
        palliative = {(r["source"], r["target"]) for r in csv.DictReader(stream, delimiter="\t")
                      if r["metaedge"] == "CpD"}
    save_json(result / "run_config.json", dict(current, fingerprint=fp,
              analysis="ORIGINAL UPSTREAM DREAMwalk; Hetionet reconstructed inputs; one seed, ten folds",
              pair_artifacts={name: sha256(result / name) for name in
                              ("negative_pairs.csv", "fold_assignments.csv")},
              counts=dict(raw_positives=755, evaluable_positives=751,
                          excluded_no_molecular_embedding=4, negatives=len(negatives),
                          sampled_negatives_also_CpD=len(set(negatives) & palliative)),
              python=sys.version, platform=platform.platform(),
              startup_pythonhashseed=os.environ.get("PYTHONHASHSEED", "unset"),
              multiprocessing_start_method=multiprocessing.get_context().get_start_method(),
              notes=["Unmodified original teleport selector and input-type pp=1 negative sampling.",
                     "No startup hashseed override; upstream set_seed(0) is unchanged.",
                     "Worker RNG states and four-thread Skip-gram are not asserted deterministic.",
                     "Table 3 retains CtD, not CpD. Link negatives are generated locally and saved.",
                     "Four raw CtD labels lack molecular endpoints; exact author handling is unavailable.",
                     "Historical same-source DO/DrugBank snapshots are not claimed as author exact semantic inputs.",
                     "Tau .3 follows the accepted demo setting; length10 follows supplement/released code.",
                     "One seed x ten folds is compared with the paper's ten repeat means."],
              created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    print(json.dumps(read_json(result / "run_config.json")["counts"], indent=2), flush=True)


def completed(path, marker, fp):
    if not marker.exists():
        return False
    meta = read_json(marker)
    same(meta["fingerprint"], fp, "Checkpoint identity")
    if not path.is_file():
        raise FileNotFoundError(path)
    same(sha256(path), meta["sha256"], "Checkpoint " + path.name)
    print("Reusing completed output:", path, flush=True)
    return True


def similarity(name):
    from DREAMwalk.generate_similarity_net import save_sim_graph
    data, result, fp = context(name)
    output, marker = data / "similarity_graph.tsv", result / "similarity_complete.json"
    if completed(output, marker, fp):
        return
    if output.exists():
        raise RuntimeError("Semantic output exists without completion record; preserve it before retrying")
    paths, types, _, _ = input_data()
    pending = output.with_name("similarity_pending.tsv")
    started = time.monotonic()
    save_sim_graph(networkf=str(paths["graph"]), hierf=str(paths["hierarchy"]), outputf=str(pending),
                   cutoff=SETTINGS["cutoff"], weighted=True, directed=False, net_delimiter="\t")
    counts = dict(drug=0, disease=0)
    with pending.open() as stream:
        for line in stream:
            a, b, _, weight, _ = line.rstrip("\n").split("\t")
            if (a == b or a not in types or b not in types or types[a] != types[b]
                    or types[a] not in counts or not SETTINGS["cutoff"] < float(weight) <= 1):
                raise ValueError("Unexpected semantic edge: " + line)
            counts[types[a]] += 1
    if not all(counts.values()):
        raise ValueError("Expected nonempty drug and disease semantic graphs")
    pending.replace(output)
    save_json(marker, dict(fingerprint=fp, sha256=sha256(output),
                          semantic_pairs=counts, seconds=time.monotonic() - started))
    print("Original semantic graph saved:", counts, flush=True)


def embedding_parameters():
    return {key: SETTINGS[key] for key in ("tp_factor", "seed", "num_walks", "walk_length",
            "dimension", "window_size", "workers", "em_max_iter", "p", "q")}


def load_embedding(path, types):
    import numpy as np
    with path.open("rb") as stream:
        vectors = pickle.load(stream)
    if set(vectors) != set(types):
        raise ValueError("Embedding node IDs differ from molecular graph")
    if not all(np.asarray(v).shape == (SETTINGS["dimension"],) and np.isfinite(v).all()
               for v in vectors.values()):
        raise ValueError("Embedding has invalid dimensions or values")
    return vectors


def embedding(name):
    from DREAMwalk.generate_embeddings import save_embedding_files
    data, result, fp = context(name)
    output, marker = data / "hetionet_seed0_original.pkl", result / "embedding_complete.json"
    if completed(output, marker, fp):
        return
    if output.exists():
        raise RuntimeError("Embedding exists without completion record; preserve it for recovery")
    sim = data / "similarity_graph.tsv"
    if not completed(sim, result / "similarity_complete.json", fp):
        raise RuntimeError("Run the similarity cell first")
    paths, types, _, _ = input_data()
    if platform.system() == "Darwin":
        headers = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX15.5.sdk/usr/include/c++/v1")
        if not headers.is_dir():
            raise FileNotFoundError("Existing macOS compiler adaptation is missing: " + str(headers))
        os.environ["CPATH"] = str(headers)
    work_root = ROOT / "logs" / name
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="original_embedding_", dir=str(work_root)))
    (work / "DREAMwalk").symlink_to(UPSTREAM / "DREAMwalk", target_is_directory=True)
    pending = work / "embedding_pending.pkl"
    started = time.monotonic()
    old_cwd = Path.cwd()
    try:
        os.chdir(work)
        # Direct upstream call. No function replacement, selector fix, NS patch,
        # hashseed restoration, start-method override, or RNG reinitialization.
        save_embedding_files(netf=str(paths["graph"]), sim_netf=str(sim),
                             nodetypef=str(paths["nodetypes"]), outputf=str(pending),
                             directed=False, weighted=True, net_delimiter="\t",
                             **embedding_parameters())
    finally:
        os.chdir(old_cwd)
    load_embedding(pending, types)
    # Persist the expensive artifact immediately. Evaluation is a separate phase.
    pending.replace(output)
    save_json(marker, dict(fingerprint=fp, sha256=sha256(output),
                          similarity_sha256=sha256(sim), nodes=len(types),
                          expected_walks=len(types) * SETTINGS["num_walks"],
                          dimensions=SETTINGS["dimension"], seconds=time.monotonic() - started,
                          implementation="unmodified save_embedding_files", work_directory=str(work)))
    print("EMBEDDING SAVED before evaluation:", output, flush=True)


def validate_saved_folds(done, assignments):
    """Refuse malformed partial rows before resuming an interrupted evaluation."""
    fold_sizes = {fold: sum(int(r["test_fold"]) == fold for r in assignments)
                  for fold in range(1, SETTINGS["folds"] + 1)}
    done_ids = []
    for row in done:
        try:
            fold = int(row["fold"])
            if fold not in fold_sizes or fold in done_ids:
                raise ValueError("duplicate or out-of-range fold ID")
            metrics = [float(row[key]) for key in METRICS]
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in metrics):
                raise ValueError("nonfinite or out-of-range metric")
            if (int(row["n_test"]) != fold_sizes[fold]
                    or int(row["n_train"]) != len(assignments) - fold_sizes[fold]):
                raise ValueError("sample counts differ from saved fold assignments")
            seconds = float(row["seconds"])
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError("invalid elapsed time")
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError("Invalid partial fold results: " + str(error)) from error
        done_ids.append(fold)
    return done_ids


def evaluate(name):
    import numpy as np
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    from xgboost import XGBClassifier
    data, result, fp = context(name)
    output, marker = result / "summary.csv", result / "evaluation_complete.json"
    if completed(output, marker, fp):
        meta = read_json(marker)
        for filename, digest in meta["artifacts"].items():
            same(sha256(result / filename), digest, "Evaluation " + filename)
        same(meta["embedding_sha256"], sha256(data / "hetionet_seed0_original.pkl"), "Evaluated embedding")
        return
    embed = data / "hetionet_seed0_original.pkl"
    if not completed(embed, result / "embedding_complete.json", fp):
        raise RuntimeError("Run the embedding cell first")
    _, types, positives, raw = input_data()
    negatives = [(r["drug"], r["disease"]) for r in rows(result / "negative_pairs.csv")]
    same(negatives, sample_negatives(types, positives, raw), "Saved negative pair IDs")
    expected = fold_rows(positives, negatives)
    expected_strings = [{key: str(value) for key, value in r.items()} for r in expected]
    same(rows(result / "fold_assignments.csv"), expected_strings, "Saved fold IDs before fitting")
    vectors = load_embedding(embed, types)
    x = np.asarray([vectors[r["drug"]] - vectors[r["disease"]] for r in expected], dtype=np.float32)
    y = np.asarray([r["label"] for r in expected], dtype=np.int8)
    fold_ids = np.asarray([r["test_fold"] for r in expected])
    partial, state = result / "fold_results.csv", result / "evaluation_started.json"
    eval_identity = dict(fingerprint=fp, embedding_sha256=sha256(embed))
    if state.exists():
        same(read_json(state), eval_identity, "Partial evaluation identity")
    elif partial.exists():
        raise RuntimeError("Unregistered fold results exist; preserve them before evaluation")
    else:
        save_json(state, eval_identity)
    done = rows(partial) if partial.exists() else []
    done_ids = validate_saved_folds(done, expected)
    random.seed(0)
    np.random.seed(0)
    for fold in range(1, SETTINGS["folds"] + 1):
        if fold in done_ids:
            print("Reusing evaluated fold", fold, flush=True)
            continue
        train, test = fold_ids != fold, fold_ids == fold
        started = time.monotonic()
        model = XGBClassifier(**xgb_settings())
        model.fit(x[train], y[train])
        prob = model.predict_proba(x[test])[:, 1]
        row = dict(fold=fold, accuracy=float(accuracy_score(y[test], prob.round())),
                   auroc=float(roc_auc_score(y[test], prob)),
                   aupr=float(average_precision_score(y[test], prob)),
                   n_train=int(train.sum()), n_test=int(test.sum()),
                   seconds=time.monotonic() - started)
        if not all(math.isfinite(row[m]) for m in METRICS):
            raise ValueError("Nonfinite fold metrics")
        done.append(row)
        done.sort(key=lambda r: int(r["fold"]))
        save_csv(partial, done, list(row))
        print("Fold", fold, {m: round(row[m], 6) for m in METRICS}, flush=True)
    summary = []
    for metric in METRICS:
        values = [float(r[metric]) for r in done]
        summary.append(dict(metric=metric, mean=statistics.mean(values),
                            SD=statistics.stdev(values), n=len(values)))
    save_csv(output, summary, list(summary[0]))
    save_json(marker, dict(eval_identity, sha256=sha256(output),
                          artifacts={name: sha256(result / name) for name in
                                     ("fold_results.csv", "fold_assignments.csv", "negative_pairs.csv")}))
    print("Ten-fold evaluation saved:", output, flush=True)


def compare(name):
    data, result, fp = context(name)
    if not completed(result / "summary.csv", result / "evaluation_complete.json", fp):
        raise RuntimeError("Run the evaluation cell first")
    evaluation = read_json(result / "evaluation_complete.json")
    for filename, digest in evaluation["artifacts"].items():
        same(sha256(result / filename), digest, "Evaluation " + filename)
    same(evaluation["embedding_sha256"], sha256(data / "hetionet_seed0_original.pkl"), "Evaluated embedding")
    # Reference file is small and versioned; values come from Figure2c Source Data.
    reference_file = ROOT / "references/hetionet_fig2c_dreamwalk.csv"
    reference = rows(reference_file)
    same([r["metric"] for r in reference], list(METRICS), "Paper reference metrics")
    local = {r["metric"]: float(r["mean"]) for r in rows(result / "summary.csv")}
    values = [dict(metric=r["metric"], local_seed0_10fold=local[r["metric"]],
                   paper_10repeat_mean=float(r["mean"]),
                   local_minus_paper=local[r["metric"]] - float(r["mean"])) for r in reference]
    save_csv(result / "paper_comparison.csv", values, list(values[0]))
    save_json(result / "comparison_complete.json", dict(fingerprint=fp,
              sha256=sha256(result / "paper_comparison.csv"),
              evaluation_sha256=sha256(result / "summary.csv"),
              reference_sha256=sha256(reference_file),
              reference="Figure2c original DREAMwalk ten-repeat Source Data means",
              input_limit="751 evaluable CtD positives versus 755 raw; ontology snapshots reconstructed"))
    print(json.dumps(values, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("environment", "download", "prepare", "similarity",
                                          "embedding", "evaluate", "compare"))
    parser.add_argument("--run-name", default=RUN_NAME)
    args = parser.parse_args()
    run_paths(args.run_name)
    if args.phase == "environment":
        print(json.dumps(dict(environment(), settings=SETTINGS, xgboost=xgb_settings()), indent=2))
    elif args.phase == "download":
        inputs.download_sources(ROOT)
    else:
        globals()[args.phase](args.run_name)


if __name__ == "__main__":
    main()
