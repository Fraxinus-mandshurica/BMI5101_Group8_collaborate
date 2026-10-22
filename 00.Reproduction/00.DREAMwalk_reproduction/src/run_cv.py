"""Review-gated DREAMwalk MSI random-split evaluation; never runs on import."""

import argparse
import csv
import pickle
import random
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
FOLD_COLUMNS = ["repeat", "fold", "seed", "accuracy", "auroc", "aupr", "n_train", "n_test",
                "n_positive_train", "n_positive_test", "runtime_seconds"]


def read_rows(path):
    with path.open(newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def inputs():
    node_types = {r["node"]: r["type"] for r in read_rows(ROOT / "data/processed/msi_nodetypes.tsv")}
    drugs = sorted(n for n, t in node_types.items() if t == "drug")
    diseases = sorted(n for n, t in node_types.items() if t == "disease")
    positives = [(r["drug"], r["disease"]) for r in read_rows(ROOT / "data/processed/msi_positive_pairs.tsv")]
    assert len(positives) == len(set(positives)) == 5926
    assert all(node_types.get(a) == "drug" and node_types.get(b) == "disease" for a, b in positives)
    return drugs, diseases, sorted(positives)


def negatives_for_repeats(drugs, diseases, positives, seeds):
    known = set(positives)
    candidates = [(a, b) for a in drugs for b in diseases if (a, b) not in known]
    rng = random.Random(seeds[0])
    selected = rng.sample(candidates, len(positives) * len(seeds))
    assert not set(selected) & known and len(set(selected)) == len(selected)
    for i, seed in enumerate(seeds):
        yield seed, selected[i * len(positives):(i + 1) * len(positives)]


def classifier(seed, workers):
    return XGBClassifier(base_score=0.5, booster="gbtree", eval_metric="error",
                         objective="binary:logistic", gamma=0, learning_rate=0.1,
                         max_depth=6, n_estimators=500, tree_method="auto",
                         min_child_weight=4, subsample=0.8, colsample_bytree=0.9,
                         scale_pos_weight=1, max_delta_step=1, seed=seed,
                         n_jobs=workers)


def evaluate(embedding_pattern, seeds=(0, 1, 2, 3, 4), n_folds=5, workers=4):
    """Requires separately reviewed full-MSI embeddings for each repeat seed."""
    drugs, diseases, positives = inputs()
    fold_rows = []
    for repeat, (seed, negatives) in enumerate(negatives_for_repeats(drugs, diseases, positives, seeds)):
        random.seed(seed)
        np.random.seed(seed)
        embedding_path = Path(embedding_pattern.format(seed=seed))
        with embedding_path.open("rb") as f:
            embedding = pickle.load(f)
        pairs = positives + negatives
        assert all(a in embedding and b in embedding for a, b in pairs), "Missing embedding for evaluation pair"
        x = np.asarray([embedding[a] - embedding[b] for a, b in pairs], dtype=np.float32)
        y = np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int8)
        assert y.sum() == len(positives) and len(y) == 2 * len(positives)
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(x, y), 1):
            started = time.monotonic()
            assert set(y[train]) == set(y[test]) == {0, 1}
            model = classifier(seed, workers)
            model.fit(x[train], y[train])
            probability = model.predict_proba(x[test])[:, 1]
            row = dict(repeat=repeat + 1, fold=fold, seed=seed,
                       accuracy=accuracy_score(y[test], probability.round()),
                       auroc=roc_auc_score(y[test], probability),
                       aupr=average_precision_score(y[test], probability),
                       n_train=len(train), n_test=len(test),
                       n_positive_train=int(y[train].sum()),
                       n_positive_test=int(y[test].sum()),
                       runtime_seconds=time.monotonic() - started)
            assert all(np.isfinite(row[k]) for k in ("accuracy", "auroc", "aupr"))
            fold_rows.append(row)
    result_dir = ROOT / "results"
    with (result_dir / "msi_fig2a_fold_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FOLD_COLUMNS)
        writer.writeheader()
        writer.writerows(fold_rows)
    with (result_dir / "msi_fig2a_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "SD", "min", "max", "n"])
        writer.writeheader()
        for metric in ("accuracy", "auroc", "aupr"):
            values = np.asarray([row[metric] for row in fold_rows])
            writer.writerow(dict(metric=metric, mean=values.mean(), SD=values.std(ddof=1),
                                 min=values.min(), max=values.max(), n=len(values)))
    return fold_rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", required=True, help="Path pattern containing {seed}, e.g. data/embeddings/msi_seed_{seed}.pkl")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()
    evaluate(args.embeddings, tuple(range(args.repeats)), args.folds, args.workers)
