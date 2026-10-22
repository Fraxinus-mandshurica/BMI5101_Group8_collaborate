"""MSI SEAL baseline. Importing this file starts no experiment.

Supplementary Method 1 supplies node2vec + node-type features and patience=10.
The pinned original SEAL/DGCNN supplies the other architecture/training defaults.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import pickle
import platform
import random
import shutil
import sqlite3
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "00.DREAMwalk_reproduction"
NODE2VEC = ROOT.parent / "01.node2vec_reproduction"
INPUTS = {
    "graph": SOURCE / "data/processed/msi_graph.tsv",
    "types": SOURCE / "data/processed/msi_nodetypes.tsv",
    "positives": SOURCE / "data/processed/msi_positive_pairs.tsv",
    "negatives": SOURCE / "results/msi_repeat0_negative_pairs.csv",
    "folds": NODE2VEC / "results/msi_seed0/fold_assignments.csv",
    "node2vec": NODE2VEC / "data/msi_seed0/msi_node2vec_seed0.pkl",
    "node2vec_complete": NODE2VEC / "results/msi_seed0/embedding_complete.json",
    "node2vec_config": NODE2VEC / "results/msi_seed0/run_config.json",
}
PINS = {"torch": "2.4.0", "numpy": "1.26.2", "scipy": "1.11.4", "scikit-learn": "1.3.2"}
CONFIG = {
    "seed": 0, "folds": 10, "validation_fraction": 0.1,
    "device": "cpu", "threads": 4,
    "hops": 1, "max_nodes_per_hop": None,
    "latent_dims": [32, 32, 32, 1], "sortpooling_percentile": 0.6,
    "hidden_dim": 128, "dropout": 0.5, "batch_size": 50,
    "optimizer": "Adam", "learning_rate": 0.0001, "weight_decay": 0.0,
    "max_epochs": 50, "patience": 10, "selection_metric": "validation loss",
    "train_drop_last": True,
    "node2vec_dimension": 128, "node_types": ["drug", "disease", "gene", "etc"],
    "features": "DRNL one-hot + existing 128D node2vec + 4D node-type one-hot",
    "graph_policy": "shared undirected simple molecular graph; no DDA label edges added",
    "node2vec_policy": "reuse completed seed0 molecular-only node2vec; no new walks or sampling",
    "outer_folds": "exact saved node2vec pair order and fold IDs",
    "validation_policy": "stratified 10% of each outer training fold; seed0; outer test untouched",
    "structural_vocabulary": "DRNL label maximum and SortPooling percentile over the fixed subgraph corpus, as original SEAL",
    "negative_policy": "reuse existing 5926 seed0 link negatives, excluding known positives",
    "aupr_definition": "sklearn average_precision_score",
    "implementation_status": "documented CPU port of original SEAL/DGCNN plus MSI benchmark adapter",
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_csv(path, delimiter=","):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def context(run_name):
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("RUN_NAME allows only letters, digits, underscore and hyphen")
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use the notebook-managed Python 3.11 interpreter")
    actual = {name: version(name) for name in PINS}
    if actual != PINS:
        raise RuntimeError(f"Environment differs from pins: {actual}")
    for path in INPUTS.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed MSI/node2vec input: {path}")
    manifest_path = ROOT / "vendor/manifest.json"
    vendor = read_json(manifest_path)
    for item in vendor:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("Pinned upstream source changed: " + item["path"])
    sources = [Path(__file__), ROOT / "src/seal_model.py", manifest_path,
               ROOT / "reference/paper_seal_fig2a_summary.csv"]
    identity = dict(config=CONFIG, packages=actual,
                    inputs={key: sha256(path) for key, path in INPUTS.items()},
                    sources={str(path.relative_to(ROOT)): sha256(path) for path in sources},
                    upstream=vendor)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    data, results = ROOT / "data" / run_name, ROOT / "results" / run_name
    config_path = results / "run_config.json"
    if config_path.exists() and read_json(config_path)["fingerprint"] != fingerprint:
        raise RuntimeError("Inputs/code/parameters changed. Use a new RUN_NAME; old outputs are preserved.")
    if not config_path.exists() and (data.exists() or results.exists()):
        if any(p.exists() and any(p.iterdir()) for p in (data, results)):
            raise RuntimeError("Nonempty run has no matching configuration; use a new RUN_NAME")
    data.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        save_json(config_path, dict(identity, fingerprint=fingerprint, python=sys.version,
                                   platform=platform.platform(),
                                   input_paths={key: str(path) for key, path in INPUTS.items()},
                                   created_at=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    return data, results, fingerprint


def complete(marker, fingerprint, artifacts=None):
    marker = Path(marker)
    if not marker.exists():
        return False
    info = read_json(marker)
    if info["fingerprint"] != fingerprint:
        raise RuntimeError("Checkpoint configuration mismatch: " + str(marker))
    if artifacts is not None:
        for name, path in artifacts.items():
            if not Path(path).is_file() or sha256(path) != info["artifacts"][name]:
                raise RuntimeError("Checkpoint artifact changed: " + str(path))
    return True


def mark(marker, fingerprint, artifacts, **metadata):
    save_json(marker, dict(fingerprint=fingerprint,
                           artifacts={key: sha256(path) for key, path in artifacts.items()}, **metadata))


def check_pairs(positives, negatives, folds, types, expected_count=5926, n_folds=10):
    positive_pairs = sorted((row["drug"], row["disease"]) for row in positives)
    negative_pairs = [(row["drug"], row["disease"]) for row in negatives]
    if (len(positive_pairs) != expected_count or len(negative_pairs) != expected_count
            or len(set(positive_pairs)) != expected_count or len(set(negative_pairs)) != expected_count
            or set(positive_pairs) & set(negative_pairs)):
        raise ValueError("Positive/negative pairs must be unique, balanced and disjoint")
    expected = positive_pairs + negative_pairs
    if len(folds) != len(expected):
        raise ValueError("Saved fold table has the wrong pair count")
    for i, ((drug, disease), row) in enumerate(zip(expected, folds)):
        label = int(i < len(positive_pairs))
        if (types.get(drug) != "drug" or types.get(disease) != "disease"
                or (row["drug"], row["disease"]) != (drug, disease)
                or int(row["label"]) != label or int(row["pair_index"]) != i
                or not 1 <= int(row["test_fold"]) <= n_folds):
            raise ValueError("Saved fold IDs do not match shared pair order/types")
    if {int(row["test_fold"]) for row in folds} != set(range(1, n_folds + 1)):
        raise ValueError("Missing outer folds")
    return folds


def validation_partitions(pairs, n_folds=10, fraction=0.1):
    import numpy as np
    from sklearn.model_selection import train_test_split

    labels = np.asarray([int(row["label"]) for row in pairs])
    fold_ids = np.asarray([int(row["test_fold"]) for row in pairs])
    output = {}
    for fold in range(1, n_folds + 1):
        outer_train, test = np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold)
        train, validation = train_test_split(outer_train, test_size=fraction,
                                             stratify=labels[outer_train], random_state=0)
        if set(train) & set(validation) or (set(train) | set(validation)) & set(test):
            raise ValueError("Training/validation/test overlap")
        output[str(fold)] = dict(train=train.tolist(), validation=validation.tolist(), test=test.tolist())
    return output


def prepared_artifacts(data, results):
    return {"graph": data / "graph.npz", "features": data / "node_features.npy",
            "nodes": data / "nodes.json", "pairs": results / "fold_assignments.csv",
            "partitions": results / "partitions.json"}


def require_prepared(data, results, fingerprint):
    if not complete(results / "prepare_complete.json", fingerprint, prepared_artifacts(data, results)):
        raise RuntimeError("Run prepare first")


def prepare(data, results, fingerprint):
    import numpy as np
    from scipy import sparse

    artifacts = prepared_artifacts(data, results)
    if complete(results / "prepare_complete.json", fingerprint, artifacts):
        print("Reusing graph, node features, negative pairs and exact outer folds", flush=True)
        return
    types = {row["node"]: row["type"] for row in read_csv(INPUTS["types"], "\t")}
    nodes = sorted(types)
    if len(nodes) != 29959 or set(types.values()) != set(CONFIG["node_types"]):
        raise ValueError("Unexpected shared MSI node types/count")
    index = {node: i for i, node in enumerate(nodes)}
    rows, cols = [], []
    line_count = 0
    with INPUTS["graph"].open(encoding="utf-8") as stream:
        for line in stream:
            a, b, relation, weight, edge_id = line.rstrip("\n").split("\t")
            if (a == b or a not in index or b not in index or float(weight) != 1
                    or {types[a], types[b]} == {"drug", "disease"}):
                raise ValueError("Unexpected molecular edge or DDA label edge")
            rows.extend((index[a], index[b]))
            cols.extend((index[b], index[a]))
            line_count += 1
    if line_count != 478728:
        raise ValueError("Unexpected molecular row count")
    graph = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(nodes), len(nodes)))
    graph.sum_duplicates()
    graph.data[:] = 1
    graph.sort_indices()
    if (graph != graph.T).nnz or np.any(np.diff(graph.indptr) == 0):
        raise ValueError("Expected undirected MSI graph without isolates")
    embedding_marker = read_json(INPUTS["node2vec_complete"])
    if embedding_marker["sha256"] != sha256(INPUTS["node2vec"]):
        raise ValueError("Existing node2vec checkpoint has changed")
    node2vec_config = read_json(INPUTS["node2vec_config"])
    if node2vec_config["inputs"]["graph"] != sha256(INPUTS["graph"]):
        raise ValueError("node2vec was not trained on the shared molecular graph")
    with INPUTS["node2vec"].open("rb") as stream:
        vectors = pickle.load(stream)
    if set(vectors) != set(nodes):
        raise ValueError("node2vec node vocabulary differs from MSI")
    embedding = np.asarray([vectors[node] for node in nodes], dtype=np.float32)
    if embedding.shape != (len(nodes), 128) or not np.isfinite(embedding).all():
        raise ValueError("node2vec must contain one finite 128D vector per molecular node")
    type_ids = [CONFIG["node_types"].index(types[node]) for node in nodes]
    features = np.concatenate([embedding, np.eye(4, dtype=np.float32)[type_ids]], axis=1)
    folds = check_pairs(read_csv(INPUTS["positives"], "\t"), read_csv(INPUTS["negatives"]),
                        read_csv(INPUTS["folds"]), types)
    partitions = validation_partitions(folds)
    sparse.save_npz(artifacts["graph"], graph)
    np.save(artifacts["features"], features, allow_pickle=False)
    save_json(artifacts["nodes"], nodes)
    shutil.copyfile(INPUTS["folds"], artifacts["pairs"])
    save_json(artifacts["partitions"], partitions)
    mark(results / "prepare_complete.json", fingerprint, artifacts,
         nodes=len(nodes), molecular_rows=line_count, simple_edges=graph.nnz // 2,
         positives=5926, negatives=5926, node_feature_dimension=132,
         treatment_edges_added=0, existing_embedding_reused=True)
    print("Prepared 29,959 nodes, 132D features, 11,852 labeled pairs and exact saved ten-fold IDs", flush=True)


def decode_graph(payload):
    import numpy as np

    with np.load(io.BytesIO(payload), allow_pickle=False) as item:
        return item["nodes"].astype(np.int64), item["edges"].astype(np.int64), item["labels"].astype(np.int64)


def subgraphs(data, results, fingerprint):
    import numpy as np
    from scipy.sparse import load_npz
    from seal_model import enclosing_subgraph, pooling_k

    require_prepared(data, results, fingerprint)
    path = data / "subgraphs.sqlite"
    if complete(results / "subgraphs_complete.json", fingerprint, {"database": path}):
        print("Reusing complete enclosing subgraph corpus", flush=True)
        return
    graph = load_npz(data / "graph.npz").tocsr()
    index = {node: i for i, node in enumerate(read_json(data / "nodes.json"))}
    pairs = read_csv(results / "fold_assignments.csv")
    started = time.monotonic()
    # SQLite keeps the full corpus on disk and commits progress every 100 pairs.
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS metadata (fingerprint TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS graphs (pair_id INTEGER PRIMARY KEY, nodes INTEGER, edges INTEGER, max_label INTEGER, payload BLOB)")
        metadata = db.execute("SELECT fingerprint FROM metadata").fetchall()
        if not metadata:
            db.execute("INSERT INTO metadata VALUES (?)", (fingerprint,))
            db.commit()
        elif metadata != [(fingerprint,)]:
            raise ValueError("Partial subgraph cache belongs to a different configuration")
        existing = {row[0] for row in db.execute("SELECT pair_id FROM graphs")}
        if not existing <= set(range(len(pairs))):
            raise ValueError("Subgraph cache has unexpected pair IDs")
        for i, pair in enumerate(pairs):
            if i in existing:
                continue
            nodes, edges, labels = enclosing_subgraph(graph, index[pair["drug"]], index[pair["disease"]], CONFIG["hops"])
            stream = io.BytesIO()
            np.savez_compressed(stream, nodes=nodes.astype(np.int32), edges=edges.astype(np.int32), labels=labels.astype(np.int32))
            db.execute("INSERT INTO graphs VALUES (?, ?, ?, ?, ?)",
                       (i, len(nodes), edges.shape[1], int(labels.max()), stream.getvalue()))
            if (i + 1) % 100 == 0:
                db.commit()
                print(f"Subgraphs {i + 1:,}/{len(pairs):,}; {time.monotonic() - started:.1f}s", flush=True)
        db.commit()
        records = db.execute("SELECT pair_id,nodes,edges,max_label FROM graphs ORDER BY pair_id").fetchall()
    if [row[0] for row in records] != list(range(len(pairs))):
        raise ValueError("Subgraph cache is incomplete")
    sizes = [row[1] for row in records]
    label_count = max(row[3] for row in records) + 1
    k = pooling_k(sizes, CONFIG["sortpooling_percentile"])
    save_csv(results / "subgraph_sizes.csv", [dict(pair_index=i, nodes=n, directed_edges=e, max_drnl=z) for i, n, e, z in records])
    mark(results / "subgraphs_complete.json", fingerprint, {"database": path},
         pairs=len(records), max_nodes=max(sizes), label_count=label_count,
         sortpooling_k=k, node_count_percentile=0.6, seconds=time.monotonic() - started)
    print(f"Saved {len(records):,} subgraphs; SortPooling k={k}; DRNL classes={label_count}", flush=True)


def save_torch(path, value):
    import torch

    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_torch(path):
    import torch

    # Local run checkpoints contain optimizer/Python/NumPy RNG state as well as tensors.
    return torch.load(path, map_location="cpu", weights_only=False)


def batches(db, indices, pairs, features, label_count, batch_size, drop_last=False):
    from seal_model import collate_graphs

    stop = len(indices) - len(indices) % batch_size if drop_last else len(indices)
    for start in range(0, stop, batch_size):
        selected = indices[start:min(start + batch_size, stop)]
        graphs = []
        for i in selected:
            row = db.execute("SELECT payload FROM graphs WHERE pair_id=?", (int(i),)).fetchone()
            if row is None:
                raise ValueError("Missing cached subgraph: " + str(i))
            graphs.append((*decode_graph(row[0]), int(pairs[i]["label"])))
        yield selected, collate_graphs(graphs, features, label_count)


def score(model, db, indices, pairs, features, label_count):
    import numpy as np
    import torch
    from torch.nn import functional as F

    model.eval()
    scores, labels, loss, count = [], [], 0.0, 0
    with torch.no_grad():
        for selected, (x, edge_index, sizes, y) in batches(db, indices, pairs, features, label_count, CONFIG["batch_size"]):
            output = model(x, edge_index, sizes)
            loss += float(F.nll_loss(output, y, reduction="sum"))
            count += len(selected)
            scores.extend(output[:, 1].exp().tolist())
            labels.extend(y.tolist())
    if not count or not math.isfinite(loss) or not np.isfinite(scores).all():
        raise ValueError("Invalid evaluation loss/probabilities")
    return loss / count, np.asarray(labels), np.asarray(scores)


def metrics(labels, scores):
    import numpy as np
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    return dict(accuracy=float(accuracy_score(labels, np.asarray(scores) > 0.5)),
                auroc=float(roc_auc_score(labels, scores)),
                aupr=float(average_precision_score(labels, scores)))


def fold_artifacts(directory):
    return {name: directory / name for name in ("best.pt", "test_predictions.csv", "metrics.json", "history.csv")}


def train(data, results, fingerprint, fold):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from seal_model import DGCNN

    if fold not in range(1, 11):
        raise ValueError("Fold must be in 1..10")
    require_prepared(data, results, fingerprint)
    if not complete(results / "subgraphs_complete.json", fingerprint, {"database": data / "subgraphs.sqlite"}):
        raise RuntimeError("Run subgraphs first")
    directory = results / f"fold{fold:02d}"
    directory.mkdir(exist_ok=True)
    if complete(directory / "complete.json", fingerprint, fold_artifacts(directory)):
        print(f"Fold {fold}: reusing saved model and held-out predictions", flush=True)
        return
    torch.set_num_threads(CONFIG["threads"])
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    pairs = read_csv(results / "fold_assignments.csv")
    partition = read_json(results / "partitions.json")[str(fold)]
    info = read_json(results / "subgraphs_complete.json")
    features = torch.from_numpy(np.load(data / "node_features.npy", allow_pickle=False))
    model = DGCNN(features.shape[1] + info["label_count"], info["sortpooling_k"],
                  tuple(CONFIG["latent_dims"]), CONFIG["hidden_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    history, order, best_loss, best_epoch, stale, start_epoch = [], list(partition["train"]), float("inf"), 0, 0, 1
    last = directory / "last.pt"
    if last.exists():
        checkpoint = load_torch(last)
        if checkpoint["fingerprint"] != fingerprint or checkpoint["fold"] != fold:
            raise ValueError("Epoch checkpoint does not match this fold/configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        history, order = checkpoint["history"], checkpoint["order"]
        best_loss, best_epoch, stale = checkpoint["best_loss"], checkpoint["best_epoch"], checkpoint["stale"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Fold {fold}: resume at epoch {start_epoch}", flush=True)
    started = time.monotonic()
    with sqlite3.connect((data / "subgraphs.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        for epoch in range(start_epoch, CONFIG["max_epochs"] + 1):
            if stale >= CONFIG["patience"]:
                break
            random.shuffle(order)
            model.train()
            total_loss, used = 0.0, 0
            for batch_number, (_, (x, edge_index, sizes, y)) in enumerate(
                    batches(db, order, pairs, features, info["label_count"],
                            CONFIG["batch_size"], CONFIG["train_drop_last"]), 1):
                optimizer.zero_grad(set_to_none=True)
                loss = F.nll_loss(model(x, edge_index, sizes), y)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach()) * len(y)
                used += len(y)
                if batch_number % 20 == 0:
                    print(f"Fold {fold}/10 | epoch {epoch} | training batch {batch_number} | "
                          f"loss {total_loss / used:.5f}", flush=True)
            if not used:
                raise ValueError("Training fold is smaller than the original batch size")
            val_loss, val_y, val_scores = score(model, db, partition["validation"], pairs, features, info["label_count"])
            if val_loss <= best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                save_torch(directory / "best.pt", dict(fingerprint=fingerprint, fold=fold,
                           epoch=epoch, validation_loss=val_loss, model=model.state_dict(), config=CONFIG,
                           label_count=info["label_count"], sortpooling_k=info["sortpooling_k"]))
            else:
                stale += 1
            history.append(dict(epoch=epoch, train_loss=total_loss / used, training_samples=used,
                                validation_loss=val_loss, validation_auroc=metrics(val_y, val_scores)["auroc"],
                                best_epoch=best_epoch, epochs_without_improvement=stale))
            save_torch(last, dict(fingerprint=fingerprint, fold=fold, epoch=epoch,
                       model=model.state_dict(), optimizer=optimizer.state_dict(), history=history, order=order,
                       best_loss=best_loss, best_epoch=best_epoch, stale=stale,
                       python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state()))
            save_csv(directory / "history.csv", history)
            print(f"Fold {fold}/10 | epoch {epoch}/50 | train {total_loss / used:.5f} | "
                  f"validation {val_loss:.5f} | best {best_epoch} | patience {stale}/10 | "
                  f"{time.monotonic() - started:.1f}s", flush=True)
        best = load_torch(directory / "best.pt")
        if best["fingerprint"] != fingerprint or best["fold"] != fold:
            raise ValueError("Best checkpoint mismatch")
        model.load_state_dict(best["model"])
        test_loss, y, probabilities = score(model, db, partition["test"], pairs, features, info["label_count"])
    predictions = [dict(pair_index=i, drug=pairs[i]["drug"], disease=pairs[i]["disease"],
                        label=int(pairs[i]["label"]), test_fold=fold, probability=float(probability))
                   for i, probability in zip(partition["test"], probabilities)]
    save_csv(directory / "test_predictions.csv", predictions)
    result = dict(fold=fold, **metrics(y, probabilities), test_loss=test_loss, best_epoch=best["epoch"],
                  validation_loss=best["validation_loss"], n_train=len(partition["train"]),
                  n_validation=len(partition["validation"]), n_test=len(partition["test"]))
    save_json(directory / "metrics.json", result)
    save_csv(directory / "history.csv", history)
    mark(directory / "complete.json", fingerprint, fold_artifacts(directory), fold=fold)
    print(json.dumps(result, indent=2), flush=True)


def summarize(data, results, fingerprint):
    pairs = read_csv(results / "fold_assignments.csv")
    folds, predictions = [], []
    for fold in range(1, 11):
        directory = results / f"fold{fold:02d}"
        if not complete(directory / "complete.json", fingerprint, fold_artifacts(directory)):
            raise RuntimeError(f"Fold {fold} has not completed; no partial ten-fold mean is reported")
        result = read_json(directory / "metrics.json")
        fold_predictions = read_csv(directory / "test_predictions.csv")
        for row in fold_predictions:
            i = int(row["pair_index"])
            if (not 0 <= i < len(pairs) or int(row["test_fold"]) != fold
                    or int(pairs[i]["test_fold"]) != fold
                    or (row["drug"], row["disease"], row["label"]) != (pairs[i]["drug"], pairs[i]["disease"], pairs[i]["label"])):
                raise ValueError("Held-out predictions do not match the saved fold table")
        measured = metrics([int(row["label"]) for row in fold_predictions], [float(row["probability"]) for row in fold_predictions])
        if any(abs(measured[key] - result[key]) > 1e-12 for key in measured):
            raise ValueError("Saved fold metrics do not match predictions")
        folds.append(result)
        predictions.extend(fold_predictions)
    if sorted(int(row["pair_index"]) for row in predictions) != list(range(len(pairs))):
        raise ValueError("OOF predictions must cover every labeled pair exactly once")
    summary = [dict(metric=key, mean=statistics.mean(row[key] for row in folds),
                    SD=statistics.stdev(row[key] for row in folds), n=10) for key in ("accuracy", "auroc", "aupr")]
    save_csv(results / "fold_results.csv", folds)
    save_csv(results / "oof_predictions.csv", sorted(predictions, key=lambda row: int(row["pair_index"])))
    save_csv(results / "summary.csv", summary)
    mark(results / "evaluation_complete.json", fingerprint,
         {name: results / name for name in ("fold_results.csv", "oof_predictions.csv", "summary.csv")})
    print(json.dumps(summary, indent=2), flush=True)


def compare(data, results, fingerprint):
    artifacts = {name: results / name for name in ("fold_results.csv", "oof_predictions.csv", "summary.csv")}
    if not complete(results / "evaluation_complete.json", fingerprint, artifacts):
        raise RuntimeError("Complete all ten held-out folds and summarize first")
    paper = {row["metric"]: row for row in read_csv(ROOT / "reference/paper_seal_fig2a_summary.csv")}
    rows = []
    for row in read_csv(results / "summary.csv"):
        key, local, reference = row["metric"], float(row["mean"]), float(paper[row["metric"]]["mean"])
        rows.append(dict(metric=key, local_seed0_10fold_mean=local, paper_10repeat_mean=reference,
                         local_minus_paper=local - reference,
                         local_fold_SD=float(row["SD"]), paper_repeat_SD=float(paper[key]["SD"])))
    save_csv(results / "paper_comparison.csv", rows)
    mark(results / "comparison_complete.json", fingerprint, {"comparison": results / "paper_comparison.csv"})
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("config", "prepare", "subgraphs", "train", "summarize", "compare"))
    parser.add_argument("--run-name", default="msi_seed0")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    data, results, fingerprint = context(args.run_name)
    if args.phase == "config":
        print(json.dumps(read_json(results / "run_config.json"), indent=2), flush=True)
    elif args.phase == "train":
        if args.fold is None:
            parser.error("train requires --fold 1..10")
        train(data, results, fingerprint, args.fold)
    else:
        globals()[args.phase](data, results, fingerprint)


if __name__ == "__main__":
    main()
