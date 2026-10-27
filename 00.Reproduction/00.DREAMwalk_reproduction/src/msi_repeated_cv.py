"""Repeated random-pair CV on fixed MSI embeddings; no work runs on import."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from msi_disease_area_split import (METRICS, features, load_vectors, open_run,
                                    read_rows, write_json, write_rows)


def sample_negative_sets(drugs, diseases, positives, seed0_negatives, seeds):
    """Keep the saved seed-0 order; draw later sets from remaining unknown pairs."""
    seeds = list(seeds)
    known = set(positives)
    if not seeds or seeds[0] != 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Unique repeat seeds starting with 0 are required")
    if not known or len(known) != len(positives):
        raise ValueError("Positive pairs must be nonempty and unique")
    drugs, diseases = sorted(drugs), sorted(diseases)
    if len(set(drugs)) != len(drugs) or len(set(diseases)) != len(diseases):
        raise ValueError("Node IDs must be unique")
    drug_set, disease_set = set(drugs), set(diseases)
    if any(drug not in drug_set or disease not in disease_set
           for drug, disease in list(positives) + list(seed0_negatives)):
        raise ValueError("Unexpected drug/disease endpoint")
    candidates = [(drug, disease) for drug in drugs for disease in diseases
                  if (drug, disease) not in known]
    if len(candidates) < len(positives) * len(seeds):
        raise ValueError("Insufficient unknown pairs for disjoint negative sets")
    if random.Random(0).sample(candidates, len(positives)) != list(seed0_negatives):
        raise ValueError("Saved seed-0 negatives differ from the canonical sample")
    result = {0: list(seed0_negatives)}
    selected = set(seed0_negatives)
    for seed in seeds[1:]:
        candidates = [pair for pair in candidates if pair not in selected]
        result[seed] = random.Random(seed).sample(candidates, len(positives))
        selected.update(result[seed])
    return result


def assign_folds(positives, negatives, seed, n_folds=10):
    """Sorted positives followed by saved negative order, as in seed-0 evaluation."""
    pairs = sorted(positives) + list(negatives)
    if (len(set(pairs)) != len(pairs) or len(positives) != len(negatives)
            or n_folds < 2 or len(positives) < n_folds):
        raise ValueError("Unique balanced pairs and enough examples for all folds are required")
    labels = np.array([1] * len(positives) + [0] * len(negatives), dtype=np.int32)
    assigned = np.zeros(len(pairs), dtype=np.int32)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(pairs)), labels), 1):
        assigned[test] = fold
    return [dict(pair_index=i, drug=drug, disease=disease, label=int(labels[i]),
                 test_fold=int(assigned[i])) for i, (drug, disease) in enumerate(pairs)]


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fold_counts(pairs, fold):
    test = [row for row in pairs if int(row["test_fold"]) == fold]
    if not test or len(test) == len(pairs):
        raise ValueError("Fold must contain both training and test examples")
    positive_test = sum(int(row["label"]) for row in test)
    return dict(n_train=len(pairs) - len(test), n_test=len(test),
                n_positive_train=sum(int(row["label"]) for row in pairs) - positive_test,
                n_positive_test=positive_test)


def _completed_fold(output, model_name, seed, fold):
    output = Path(output)
    marker = output / "metrics.json"
    if not marker.exists():
        return None
    row = json.loads(marker.read_text())
    if (row["model"], row["repeat_seed"], row["fold"]) != (model_name, seed, fold):
        raise ValueError("Saved fold identity differs: " + str(output))
    required = (["source_manifest.json"] if row["origin"] == "historical_seed0"
                else ["classifier.json", "predictions.csv"])
    if not all((output / name).is_file() for name in required):
        raise ValueError("Completed fold has missing outputs: " + str(output))
    return row


def import_seed0(source, embedding_path, pairs, output, model_name, parameters, n_folds=10):
    """Reuse verified historical metrics; no historical classifier is invented."""
    source, output = Path(source), Path(output)
    paths = {name: source / name for name in
             ("run_config.json", "fold_assignments.csv", "pp1_fold_results.csv",
              "pp1_evaluation_complete.json")}
    marker = json.loads(paths["pp1_evaluation_complete.json"].read_text())
    config = json.loads(paths["run_config.json"].read_text())
    if config["xgboost"] != parameters or parameters["seed"] != 0:
        raise ValueError("Historical seed-0 classifier settings differ")
    if marker["embedding_sha256"] != _sha256(embedding_path):
        raise ValueError("Historical embedding differs")
    if (marker["fold_results_sha256"] != _sha256(paths["pp1_fold_results.csv"])
            or marker["fold_assignments_sha256"] != _sha256(paths["fold_assignments.csv"])):
        raise ValueError("Historical fold artifacts differ from the completion record")
    expected = [{key: str(value) for key, value in row.items()} for row in pairs]
    if read_rows(paths["fold_assignments.csv"]) != expected:
        raise ValueError("Historical seed-0 pair order or fold IDs differ")
    rows = read_rows(paths["pp1_fold_results.csv"])
    if len(rows) != n_folds or {int(row["fold"]) for row in rows} != set(range(1, n_folds + 1)):
        raise ValueError("Historical fold results are incomplete")
    manifest = {name: dict(path=str(path), sha256=_sha256(path)) for name, path in paths.items()}
    result = []
    for old in sorted(rows, key=lambda row: int(row["fold"])):
        fold = int(old["fold"])
        counts = _fold_counts(pairs, fold)
        if (int(old["cv_seed"]) != 0 or int(old["negative_seed"]) != 0
                or any(int(old[key]) != value for key, value in counts.items())):
            raise ValueError("Historical fold counts or seeds differ")
        metrics = {metric: float(old[metric]) for metric in METRICS}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in metrics.values()):
            raise ValueError("Invalid historical metrics")
        row = dict(model=model_name, repeat_seed=0, fold=fold, classifier_seed=0,
                   origin="historical_seed0", **metrics, **counts,
                   runtime_seconds=float(old["runtime_seconds"]))
        directory = output / f"fold{fold:02d}"
        completed = _completed_fold(directory, model_name, 0, fold)
        if completed is not None:
            if completed != row:
                raise ValueError("Imported seed-0 metrics differ; preserve the saved run")
        else:
            write_json(directory / "source_manifest.json", manifest)
            write_json(directory / "metrics.json", row)
        result.append(row)
    return result


def fit_fold(x, y, pairs, model_name, seed, fold, parameters, output):
    """Fit one fold, then save the classifier, test predictions and completion record."""
    from xgboost import XGBClassifier
    output = Path(output)
    completed = _completed_fold(output, model_name, seed, fold)
    if completed is not None:
        return completed
    if parameters["seed"] != seed:
        raise ValueError("Classifier seed must equal the repeat seed")
    y = np.asarray(y)
    test = np.array([int(row["test_fold"]) == fold for row in pairs])
    train = ~test
    if (len(x) != len(y) or len(y) != len(pairs)
            or not np.array_equal(y, [int(row["label"]) for row in pairs])
            or set(y[train]) != {0, 1} or set(y[test]) != {0, 1}):
        raise ValueError("Misaligned features/labels or invalid train/test classes")
    started = time.monotonic()
    model = XGBClassifier(**parameters)
    model.fit(x[train], y[train])
    probability = model.predict_proba(x[test])[:, 1]
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid test probabilities")
    row = dict(model=model_name, repeat_seed=seed, fold=fold, classifier_seed=seed,
               origin="new_fit", accuracy=float(accuracy_score(y[test], probability.round())),
               auroc=float(roc_auc_score(y[test], probability)),
               aupr=float(average_precision_score(y[test], probability)),
               **_fold_counts(pairs, fold), runtime_seconds=time.monotonic() - started)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "classifier.tmp.json"
    model.save_model(str(temporary))
    temporary.replace(output / "classifier.json")
    predictions = [dict(pair_index=pair["pair_index"], drug=pair["drug"], disease=pair["disease"],
                        label=int(pair["label"]), score=float(score))
                   for pair, score in zip((p for p in pairs if int(p["test_fold"]) == fold), probability)]
    write_rows(output / "predictions.csv", predictions,
               ["pair_index", "drug", "disease", "label", "score"])
    write_json(output / "metrics.json", row)
    return row


def mean_intervals(rows):
    """Mean, sample SD and Student-t 95% CI across repeat-level values."""
    from scipy.stats import t
    result = []
    for model in sorted({row["model"] for row in rows}):
        subset = [row for row in rows if row["model"] == model]
        if len(subset) < 2 or len({row["repeat_seed"] for row in subset}) != len(subset):
            raise ValueError("At least two distinct repeats per model are required")
        for metric in METRICS:
            values = [float(row[metric]) for row in subset]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("Nonfinite repeat metric")
            mean, sd = statistics.mean(values), statistics.stdev(values)
            margin = float(t.ppf(0.975, len(values) - 1)) * sd / math.sqrt(len(values))
            result.append(dict(model=model, metric=metric, mean=mean, SD=sd,
                               CI95_low=mean - margin, CI95_high=mean + margin,
                               repeats=len(values)))
    return result


def summarize(fold_rows, seeds, n_folds=10):
    """Average folds within repeats, then summarize repeats and paired model changes."""
    seeds = list(seeds)
    expected = {(model, seed, fold) for model in ("original", "modified")
                for seed in seeds for fold in range(1, n_folds + 1)}
    keys = [(row["model"], row["repeat_seed"], row["fold"]) for row in fold_rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Exactly one result for every model, repeat and fold is required")
    repeats = []
    for model in ("original", "modified"):
        for seed in seeds:
            subset = [row for row in fold_rows if row["model"] == model and row["repeat_seed"] == seed]
            repeats.append(dict(model=model, repeat_seed=seed, folds=n_folds,
                                **{metric: statistics.mean(float(row[metric]) for row in subset)
                                   for metric in METRICS}))
    indexed = {(row["model"], row["repeat_seed"]): row for row in repeats}
    differences = [dict(model="modified_minus_original", repeat_seed=seed,
                        **{metric: indexed[("modified", seed)][metric] - indexed[("original", seed)][metric]
                           for metric in METRICS}) for seed in seeds]
    return repeats, mean_intervals(repeats), differences, mean_intervals(differences)
