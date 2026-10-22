"""I/O and pair handling for the user-run HD prediction notebook.

Importing this module never fits a model or writes reproduction artifacts.
"""
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle

import numpy as np

DISEASE = "C0020179"
EMBEDDING_RUN = "msi_seed0_mesh2024_final_fixed"
PINS = {"numpy": "1.23.3", "scikit-learn": "1.1.3", "xgboost": "1.7.4"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path, delimiter=","):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_rows(path, rows):
    if not rows:
        raise ValueError("Refuse to save an empty case-study table")
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def partition_pairs(positives, negatives, types, disease=DISEASE):
    """Keep ALL target-disease labels out of downstream training."""
    if types.get(disease) != "disease":
        raise ValueError("Target disease is missing from MSI")
    if len(set(positives)) != len(positives) or len(set(negatives)) != len(negatives):
        raise ValueError("Duplicate labelled pairs")
    if set(positives) & set(negatives):
        raise ValueError("Positive/negative pair overlap")
    rows = []
    for label, pairs in ((1, positives), (0, negatives)):
        for drug, indication in pairs:
            if types.get(drug) != "drug" or types.get(indication) != "disease":
                raise ValueError("Unexpected drug/disease endpoint")
            rows.append(dict(pair_index=len(rows), drug=drug, disease=indication, label=label))
    training = [row for row in rows if row["disease"] != disease]
    held_out = [row for row in rows if row["disease"] == disease]
    if {row["label"] for row in training} != {0, 1} or not held_out:
        raise ValueError("Both training classes and held-out target labels are required")
    return training, held_out


def pair_features(vectors, pairs):
    x = np.asarray([vectors[a] - vectors[b] for a, b in pairs], dtype=np.float32)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Invalid pair features")
    return x


def load_inputs(dreamwalk):
    """Read the completed corrected embedding and its original labelled cohort."""
    dreamwalk = Path(dreamwalk)
    results = dreamwalk / "results" / EMBEDDING_RUN
    paths = dict(
        embedding=dreamwalk / "data/embeddings" / EMBEDDING_RUN / "pp1.pkl",
        embedding_config=results / "run_config.json",
        embedding_complete=results / "pp1_embedding_complete.json",
        graph=dreamwalk / "data/processed/msi_graph.tsv",
        nodetypes=dreamwalk / "data/processed/msi_nodetypes.tsv",
        positives=dreamwalk / "data/processed/msi_positive_pairs.tsv",
        negatives=dreamwalk / "results/msi_repeat0_negative_pairs.csv",
        drug_names=dreamwalk / "data/msi_raw/1_drug_to_protein.tsv",
        dda_names=dreamwalk / "data/msi_raw/6_drug_indication_df.tsv",
    )
    hashes = {name: sha256(path) for name, path in paths.items()}
    config, complete = read_json(paths["embedding_config"]), read_json(paths["embedding_complete"])
    if complete["fingerprint"] != config["fingerprint"] or complete["sha256"] != hashes["embedding"]:
        raise ValueError("Embedding does not match its completion record")
    for key in ("graph", "nodetypes", "positives", "negatives"):
        if config["inputs"][key] != hashes[key]:
            raise ValueError("Shared input differs from completed run: " + key)
    expected = dict(tp_factor=0.3, pp=1, seed=0, dimension=128,
                    ns_pool="positive_output_type", reject_positive_output=True,
                    teleport_selector="first_hit")
    if any(config["settings"].get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected embedding variant")
    if config["semantic"]["mesh_year"] != 2024 or config["feature"] != "drug_embedding - disease_embedding":
        raise ValueError("Unexpected MeSH version or feature operator")
    packages = {name: version(name) for name in PINS}
    if packages != PINS:
        raise RuntimeError("Select the existing DREAMwalk Python 3.8 interpreter: " + repr(packages))
    types = {row["node"]: row["type"] for row in read_rows(paths["nodetypes"], "\t")}
    positives = sorted((row["drug"], row["disease"]) for row in read_rows(paths["positives"], "\t"))
    negatives = [(row["drug"], row["disease"]) for row in read_rows(paths["negatives"])]
    if len(types) != 29959 or len(positives) != 5926 or len(negatives) != 5926:
        raise ValueError("Unexpected MSI cohort size")
    # Only load the local embedding after its recorded digest has matched.
    with paths["embedding"].open("rb") as stream:
        vectors = pickle.load(stream)
    if set(vectors) != set(types) or any(
            np.asarray(v).shape != (128,) or not np.isfinite(v).all() for v in vectors.values()):
        raise ValueError("Embedding must cover every MSI node with finite 128D vectors")
    training, held_out = partition_pairs(positives, negatives, types)
    names = {}
    for row in read_rows(paths["drug_names"], "\t"):
        if row["node_1_name"].strip():
            names.setdefault(row["node_1"], row["node_1_name"].strip())
    for row in read_rows(paths["dda_names"], "\t"):
        if row["drug_name"].strip():
            names.setdefault(row["drug"], row["drug_name"].strip())
    provenance = dict(
        embedding_run=EMBEDDING_RUN, embedding_fingerprint=config["fingerprint"],
        settings=config["settings"], semantic=config["semantic"],
        xgboost=config["xgboost"], feature=config["feature"], packages=packages,
        inputs={name: dict(path=str(path.relative_to(dreamwalk)), sha256=hashes[name])
                for name, path in paths.items()},
    )
    return vectors, types, names, training, held_out, provenance


def rank_drugs(drugs, scores, names, held_out, disease=DISEASE):
    if len(drugs) != len(set(drugs)) or len(drugs) != len(scores):
        raise ValueError("Drug IDs and prediction scores must align uniquely")
    if any(not np.isfinite(score) or not 0 <= score <= 1 for score in scores):
        raise ValueError("Invalid association scores")
    positives = {row["drug"] for row in held_out if row["label"] == 1}
    negatives = {row["drug"] for row in held_out if row["label"] == 0}
    if not (positives | negatives) <= set(drugs):
        raise ValueError("A labelled target drug is missing from candidates")
    ranked = [dict(drug=drug, drug_name=names.get(drug) or drug, disease=disease,
                   disease_name="Huntington disease", association_score=float(score),
                   known_in_msi=drug in positives,
                   sampled_negative_in_benchmark=drug in negatives)
              for drug, score in zip(drugs, scores)]
    ranked.sort(key=lambda row: (-row["association_score"], row["drug"]))
    candidates, known = [], []
    for rank, row in enumerate(ranked, 1):
        row["rank_all"] = rank
        if row["known_in_msi"]:
            known.append(dict(row))
        else:
            candidates.append(dict(row, candidate_rank=len(candidates) + 1))
    return ranked, candidates, known


def open_run(results, identity):
    """Preserve saved runs if inputs or notebook code change."""
    results = Path(results)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    config = results / "run_config.json"
    if config.exists():
        if read_json(config)["fingerprint"] != fingerprint:
            raise RuntimeError("Case configuration changed; preserve outputs and choose a new RUN_NAME")
    else:
        if results.exists() and any(results.iterdir()):
            raise RuntimeError("Unregistered outputs exist; choose a new RUN_NAME")
        results.mkdir(parents=True, exist_ok=True)
        save_json(config, dict(identity, fingerprint=fingerprint))
    return fingerprint


def check_stage(results, stage, fingerprint):
    marker = Path(results) / (stage + "_complete.json")
    if not marker.exists():
        return False
    saved = read_json(marker)
    if saved["fingerprint"] != fingerprint or any(
            not (Path(results) / name).is_file() or sha256(Path(results) / name) != digest
            for name, digest in saved["outputs"].items()):
        raise RuntimeError("Saved case-study output changed: " + stage)
    return True


def finish_stage(results, stage, fingerprint, files):
    save_json(Path(results) / (stage + "_complete.json"), dict(
        fingerprint=fingerprint, outputs={name: sha256(Path(results) / name) for name in files}))
