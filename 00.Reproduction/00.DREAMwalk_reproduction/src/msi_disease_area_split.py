"""Paired disease-area evaluation on existing MSI embeddings.

Importing this module does not prepare outputs or fit a classifier.
"""
from __future__ import annotations

from collections import Counter
import csv
import io
import json
from pathlib import Path
import pickle
import random
import re


PARTITIONS = ("train", "validation", "test")
METRICS = ("accuracy", "auroc", "aupr")


def read_rows(path, delimiter=","):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_rows(path, rows, columns):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    write_text(path, stream.getvalue())


def category_map(mapping_rows):
    """Use every highest-level tree category of an accepted direct mapping."""
    result = {}
    for row in mapping_rows:
        disease = row["msi_cui"]
        if disease in result:
            raise ValueError("Duplicate disease in MeSH mapping: " + disease)
        categories = set()
        if str(row["usable_for_primary_hierarchy"]).lower() == "true":
            for tree in filter(None, row["mesh_tree_numbers"].split(";")):
                if not re.fullmatch(r"[A-Z]\d{2}(?:\.\d+)*", tree):
                    raise ValueError("Invalid MeSH tree number: " + tree)
                categories.add(tree.split(".")[0])
        result[disease] = categories
    return result


def load_cohort(root):
    root = Path(root)
    nodes = read_rows(root / "data/processed/msi_nodetypes.tsv", "\t")
    drugs = sorted(row["node"] for row in nodes if row["type"] == "drug")
    drug_set = set(drugs)
    diseases = {row["node"] for row in nodes if row["type"] == "disease"}
    raw = read_rows(root / "data/processed/msi_positive_pairs.tsv", "\t")
    positives = {(row["drug"], row["disease"]) for row in raw}
    if len(positives) != len(raw):
        raise ValueError("Duplicate positive pairs")
    if any(drug not in drug_set or disease not in diseases for drug, disease in positives):
        raise ValueError("Positive pair endpoint is absent from the expected node type")
    mapping = read_rows(root / "results/msi_mesh2024_demo_roots_inputs/mapping_audit.csv")
    all_categories = category_map(mapping)
    if set(all_categories) != diseases:
        raise ValueError("MeSH mapping and molecular disease IDs differ")
    categories = {disease: values for disease, values in all_categories.items() if values}
    counts = Counter(disease for _, disease in positives)
    audit = [dict(disease=row["msi_cui"], mesh_descriptor_id=row["mesh_descriptor_id"],
                  mapping_status=row["mapping_status"],
                  categories=";".join(sorted(all_categories[row["msi_cui"]])),
                  category_count=len(all_categories[row["msi_cui"]]),
                  in_evaluation=bool(all_categories[row["msi_cui"]]),
                  positive_pairs=counts[row["msi_cui"]]) for row in mapping]
    return dict(drugs=drugs, categories=categories, all_positives=positives,
                positives=sorted(pair for pair in positives if pair[1] in categories),
                excluded_positives=sorted(pair for pair in positives if pair[1] not in categories),
                audit=audit)


def select_primary_categories(categories, seed=0):
    """Choose one fixed evaluation category per disease without consulting labels."""
    if not categories or any(not values for values in categories.values()):
        raise ValueError("A category for every disease is required")
    rng = random.Random(seed)
    return {disease: rng.choice(sorted(values)) for disease, values in sorted(categories.items())}


def split_categories(categories, seed, primary_categories=None):
    """Allocate all categories 8:1:1; use all memberships or one fixed primary."""
    names = sorted({value for values in categories.values() for value in values})
    if len(names) < 3 or any(not values for values in categories.values()):
        raise ValueError("At least three categories and a category for every disease are required")
    memberships = categories
    if primary_categories is not None:
        if set(primary_categories) != set(categories) or any(
                primary_categories[disease] not in values for disease, values in categories.items()):
            raise ValueError("Primary categories must assign every disease one of its accepted categories")
        memberships = {disease: {category} for disease, category in primary_categories.items()}
    random.Random(seed).shuffle(names)
    n_test = max(1, round(len(names) * 0.1))
    n_validation = max(1, round(len(names) * 0.1))
    category_split = {name: "train" for name in names[:-(n_test + n_validation)]}
    category_split.update({name: "validation" for name in names[-(n_test + n_validation):-n_test]})
    category_split.update({name: "test" for name in names[-n_test:]})
    disease_split = {}
    for disease, values in sorted(memberships.items()):
        partitions = {category_split[value] for value in values}
        disease_split[disease] = ("test" if "test" in partitions else
                                  "validation" if "validation" in partitions else "train")
    for disease, partition in disease_split.items():
        partitions = {category_split[value] for value in memberships[disease]}
        if partition == "train" and partitions != {"train"}:
            raise AssertionError("Held-out category entered training")
        if partition == "validation" and "test" in partitions:
            raise AssertionError("Test category entered validation")
    return category_split, disease_split


def sample_negatives(drugs, diseases, positives, used_negatives, count, seed):
    """Uniform global sampling without replacement or cross-repeat overlap."""
    drugs, diseases = sorted(drugs), sorted(diseases)
    drug_set, disease_set = set(drugs), set(diseases)
    forbidden = set(positives) | set(used_negatives)
    blocked = sum(drug in drug_set and disease in disease_set for drug, disease in forbidden)
    size = len(drugs) * len(diseases)
    if count < 0 or count > size - blocked:
        raise ValueError("Insufficient unknown pairs for disjoint negative sampling")
    rng, selected = random.Random(seed), []
    while len(selected) < count:
        index = rng.randrange(size)
        pair = (drugs[index // len(diseases)], diseases[index % len(diseases)])
        if pair not in forbidden:
            forbidden.add(pair)
            selected.append(pair)
    return selected


def make_repeat(cohort, seed, used_negatives, primary_categories=None):
    category_split, disease_split = split_categories(cohort["categories"], seed, primary_categories)
    negatives = sample_negatives(cohort["drugs"], cohort["categories"],
                                 cohort["all_positives"], used_negatives,
                                 len(cohort["positives"]), seed)
    pairs = [dict(drug=drug, disease=disease, label=label,
                  partition=disease_split[disease])
             for label, values in ((1, cohort["positives"]), (0, negatives))
             for drug, disease in values]
    summary = []
    for partition in PARTITIONS:
        subset = [row for row in pairs if row["partition"] == partition]
        labels = Counter(row["label"] for row in subset)
        if set(labels) != {0, 1}:
            raise ValueError(f"Seed {seed}, {partition}: both classes are required; no automatic split search")
        summary.append(dict(repeat_seed=seed, partition=partition,
                            allocated_categories=sum(value == partition for value in category_split.values()),
                            diseases=sum(value == partition for value in disease_split.values()),
                            positives=labels[1], negatives=labels[0], pairs=len(subset),
                            positive_fraction=labels[1] / len(subset)))
    return category_split, disease_split, pairs, negatives, summary


def open_run(results, config):
    """Guard scientific inputs/settings; notebook wording is not a cache key."""
    results = Path(results)
    path = results / "run_config.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Scientific inputs/settings changed; preserve results and choose a new RUN_NAME")
    elif results.exists() and any(results.iterdir()):
        raise ValueError("Unregistered outputs exist; choose a new RUN_NAME")
    else:
        write_json(path, config)


def save_repeat(results, seed, categories, category_split, disease_split, pairs):
    directory = Path(results) / "splits" / f"repeat{seed:02d}"
    write_rows(directory / "category_split.csv",
               [dict(category=key, partition=value) for key, value in sorted(category_split.items())],
               ["category", "partition"])
    write_rows(directory / "disease_split.csv",
               [dict(disease=key, categories=";".join(sorted(categories[key])), partition=value)
                for key, value in sorted(disease_split.items())],
               ["disease", "categories", "partition"])
    write_rows(directory / "pairs.csv", pairs, ["drug", "disease", "label", "partition"])


def load_vectors(path, required_nodes, dimension=128):
    import numpy as np
    with Path(path).open("rb") as stream:
        vectors = pickle.load(stream)
    for node in required_nodes:
        if node not in vectors:
            raise ValueError("Embedding is missing node: " + node)
        value = np.asarray(vectors[node])
        if value.shape != (dimension,) or not np.isfinite(value).all():
            raise ValueError("Invalid embedding vector: " + node)
    return vectors


def features(vectors, pairs):
    import numpy as np
    x = np.asarray([vectors[row["drug"]] - vectors[row["disease"]] for row in pairs], dtype=np.float32)
    y = np.asarray([int(row["label"]) for row in pairs], dtype=np.int32)
    return x, y


def evaluate(vectors, pairs, model):
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
    x, y = features(vectors, pairs)
    scores = model.predict_proba(x)[:, 1]
    metrics = dict(accuracy=float(accuracy_score(y, scores >= 0.5)),
                   auroc=float(roc_auc_score(y, scores)),
                   aupr=float(average_precision_score(y, scores)))
    predictions = [dict(row, score=float(score)) for row, score in zip(pairs, scores)]
    return metrics, predictions


def fit_repeat(vectors, pairs, output, model_name, seed, parameters):
    """One fixed-parameter fit; validation is reported, never used for selection."""
    from xgboost import XGBClassifier
    output = Path(output)
    marker = output / "metrics.json"
    if marker.exists():
        if not all((output / name).is_file() for name in ("classifier.json", "predictions.csv")):
            raise ValueError("Completed fit has missing outputs: " + str(output))
        return json.loads(marker.read_text())
    training = [row for row in pairs if row["partition"] == "train"]
    x, y = features(vectors, training)
    model = XGBClassifier(**parameters)
    model.fit(x, y)
    rows, predictions = [], []
    for partition in ("validation", "test"):
        subset = [row for row in pairs if row["partition"] == partition]
        metrics, predicted = evaluate(vectors, subset, model)
        rows.append(dict(model=model_name, repeat_seed=seed, partition=partition,
                         positives=sum(int(row["label"]) for row in subset),
                         negatives=sum(not int(row["label"]) for row in subset), **metrics))
        predictions.extend(predicted)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "classifier.tmp.json"
    model.save_model(str(temporary))
    temporary.replace(output / "classifier.json")
    write_rows(output / "predictions.csv", predictions,
               ["drug", "disease", "label", "partition", "score"])
    write_json(marker, rows)
    return rows


def summarize(rows):
    import statistics
    result = []
    for model in sorted({row["model"] for row in rows}):
        for partition in ("validation", "test"):
            subset = [row for row in rows if row["model"] == model and row["partition"] == partition]
            for metric in METRICS:
                values = [row[metric] for row in subset]
                result.append(dict(model=model, partition=partition, metric=metric,
                                   mean=statistics.mean(values), SD=statistics.stdev(values) if len(values) > 1 else 0.0,
                                   repeats=len(values)))
    return result


def paired_deltas(rows):
    import statistics
    indexed = {(row["model"], row["repeat_seed"]): row for row in rows if row["partition"] == "test"}
    seeds = sorted({seed for _, seed in indexed})
    differences, summary = [], []
    for seed in seeds:
        original, modified = indexed[("original", seed)], indexed[("modified", seed)]
        differences.append(dict(repeat_seed=seed, **{metric: modified[metric] - original[metric] for metric in METRICS}))
    for metric in METRICS:
        values = [row[metric] for row in differences]
        summary.append(dict(metric=metric, mean_modified_minus_original=statistics.mean(values),
                            SD=statistics.stdev(values) if len(values) > 1 else 0.0,
                            repeats=len(values)))
    return differences, summary
