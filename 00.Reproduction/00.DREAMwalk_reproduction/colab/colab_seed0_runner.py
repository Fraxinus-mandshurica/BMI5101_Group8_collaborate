"""Frozen MSI semantic seed-0 Colab runner. Invoke with the pinned Python 3.8.

Commands are deliberately separate: verify, embed, integrity, evaluate.
Nothing here starts an embedding unless the explicit ``embed`` command runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
import pickle
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/DREAMwalk_upstream"
INPUT = ROOT / "input"
EVAL = ROOT / "evaluation"
RESULTS = ROOT / "results"
MANIFEST = ROOT / "manifest.json"
GRAPH = INPUT / "msi_graph.tsv"
SIM = INPUT / "msi_similarity_graph_final.txt"
TYPES = INPUT / "msi_nodetypes.tsv"
POSITIVES = EVAL / "msi_positive_pairs.tsv"
NEGATIVES = EVAL / "msi_repeat0_negative_pairs.csv"
EMBEDDING = RESULTS / "msi_seed0_colab_linux.pkl"
FOLDS = RESULTS / "colab_seed0_fold_results.csv"
SUMMARY = RESULTS / "colab_seed0_summary.json"
CONFIG = RESULTS / "colab_seed0_run_config.json"
INTEGRITY = RESULTS / "colab_embedding_integrity.json"
PINS = {"python": "3.8", "networkx": "2.8.8", "pandas": "1.4.4",
        "scikit-learn": "1.1.3", "scipy": "1.9.1", "numpy": "1.23.3",
        "tqdm": "4.64.1", "parmap": "1.6.0", "xgboost": "1.7.4"}
RELATION_COUNTS = {1: 8568, 2: 25212, 3: 387626, 4: 34777, 5: 22545}
TYPE_COUNTS = {"drug": 1661, "disease": 840, "gene": 17660, "etc": 9798}
RELATION_TYPES = {1: ("drug", "gene"), 2: ("disease", "gene"),
                  3: ("gene", "gene"), 4: ("gene", "etc"), 5: ("etc", "etc")}
UPSTREAM_FILES = ("DREAMwalk/generate_embeddings.py", "DREAMwalk/HeterogeneousSG.py",
                  "DREAMwalk/HeterogeneousSG.cpp", "DREAMwalk/utils.py", "environment.yaml")
EXPECTED_COMMIT = "2f93723c3fbfa899d0f23c02e3459de8bfac0581"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_rows(path, delimiter="\t"):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def verify_manifest():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["upstream_commit"] == EXPECTED_COMMIT
    for item in manifest["files"]:
        path = ROOT / item["bundled_relative_path"]
        assert path.is_file(), "Missing bundle input: {}".format(path)
        assert path.stat().st_size == item["size_bytes"], "Size mismatch: {}".format(path)
        assert sha256(path) == item["sha256"], "SHA256 mismatch: {}".format(path)
    write_json(RESULTS / "colab_input_manifest.json", manifest)
    return manifest


def verify_upstream(manifest):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(UPSTREAM), text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain=v1"], cwd=str(UPSTREAM), text=True).strip()
    assert commit == EXPECTED_COMMIT, "Unexpected upstream commit: {}".format(commit)
    assert not status, "Bundled upstream checkout is modified: {}".format(status)
    hashes = {name: sha256(UPSTREAM / name) for name in UPSTREAM_FILES}
    assert hashes == manifest["upstream_relevant_sha256"], "Upstream source hash mismatch"
    return {"commit": commit, "git_status_porcelain": status, "relevant_sha256": hashes}


def verify_versions():
    from importlib import metadata
    assert sys.version_info[:2] == (3, 8), "Expected Python 3.8, got {}".format(sys.version)
    versions = {name: metadata.version(name) for name in PINS if name != "python"}
    versions["python"] = "{}.{}.{}".format(*sys.version_info[:3])
    for name, expected in PINS.items():
        actual = versions[name]
        assert actual.startswith("3.8.") if name == "python" else actual == expected, \
            "{} version mismatch: expected {}, got {}".format(name, expected, actual)
    return versions


def verify_inputs():
    assignments = read_rows(TYPES)
    nodes = {}
    for row in assignments:
        assert row["node"] not in nodes, "Duplicate node type assignment"
        assert row["type"] in TYPE_COUNTS, "Unknown type"
        nodes[row["node"]] = row["type"]
    assert len(nodes) == 29959
    assert dict(Counter(nodes.values())) == TYPE_COUNTS
    positives = {(r["drug"], r["disease"]) for r in read_rows(POSITIVES)}
    assert len(positives) == 5926
    assert all(nodes[a] == "drug" and nodes[b] == "disease" for a,b in positives)
    negatives = [(r["drug"],r["disease"]) for r in read_rows(NEGATIVES, ",")]
    assert len(negatives) == len(set(negatives)) == 5926
    assert not set(negatives) & positives
    graph_nodes, relations, ids = set(), Counter(), set()
    edge_count = 0
    with GRAPH.open(encoding="utf-8") as stream:
        for line in stream:
            a,b,kind,weight,eid = line.rstrip("\n").split("\t")
            kind,weight,eid = int(kind),float(weight),int(eid)
            assert kind in RELATION_TYPES and {nodes[a],nodes[b]} == set(RELATION_TYPES[kind])
            assert math.isfinite(weight) and weight > 0
            assert (a,b) not in positives and (b,a) not in positives
            assert eid not in ids, "Duplicate edge ID"
            ids.add(eid); graph_nodes.update((a,b)); relations[kind] += 1; edge_count += 1
    assert graph_nodes == set(nodes) and edge_count == 478728
    assert dict(relations) == RELATION_COUNTS
    semantic_types, semantic_pairs = Counter(), set()
    with SIM.open(encoding="utf-8") as stream:
        for line in stream:
            a,b,kind,weight,eid = line.rstrip("\n").split("\t")
            assert a in nodes and b in nodes and a != b
            assert nodes[a] == nodes[b] and nodes[a] in ("drug","disease")
            assert math.isfinite(float(weight)) and 0.4 < float(weight) <= 1
            key = (min(a,b),max(a,b))
            assert key not in semantic_pairs, "Duplicate semantic pair"
            semantic_pairs.add(key);semantic_types[nodes[a]] += 1
    assert dict(semantic_types) == {"drug":27530,"disease":25573}
    return {"node_type_counts":dict(Counter(nodes.values())),
            "molecular_edge_rows":edge_count,"molecular_relation_counts":dict(relations),
            "semantic_edge_counts":dict(semantic_types),"known_treatment_pairs":len(positives),
            "known_treatment_edges_in_graph":0,"negative_pairs":len(negatives)}


def command_verify():
    RESULTS.mkdir(exist_ok=True)
    manifest = verify_manifest()
    upstream = verify_upstream(manifest)
    versions = verify_versions()
    inputs = verify_inputs()
    start_method = mp.get_start_method()
    config = {"status":"VERIFIED — FULL RUN NOT YET STARTED", "created_utc":utc_now(),
              "model":"DREAMwalk","network":"MSI","seed":0,"similarity_cutoff":0.4,
              "tp_factor":0.3,"num_walks":100,"walk_length":10,"dimension":128,
              "window_size":4,"p":1,"q":1,"em_max_iter":5,"workers":4,
              "weighted":True,"directed":False,"expected_walks":2995900,
              "external_pythonhashseed_set_by_notebook":False,
              "multiprocessing_start_method":start_method,"runtime_os":platform.platform(),
              "architecture":platform.machine(),"environment_package_versions":versions,
              "upstream":upstream,"inputs":inputs,
              "input_sha256":{i["bundled_relative_path"]:i["sha256"] for i in manifest["files"]
                              if i["role"] in ("molecular_graph","semantic_graph","node_types","positive_pairs","negative_pairs")},
              "fold_membership":"Deterministically regenerated: StratifiedKFold(n_splits=10, shuffle=True, random_state=0) on sorted positives then saved negatives; explicit per-pair IDs were not saved locally.",
              "xgboost":{"base_score":0.5,"booster":"gbtree","eval_metric":"error",
                         "objective":"binary:logistic","gamma":0,"learning_rate":0.1,
                         "max_depth":6,"n_estimators":500,"tree_method":"auto",
                         "min_child_weight":4,"subsample":0.8,"colsample_bytree":0.9,
                         "scale_pos_weight":1,"max_delta_step":1,"seed":0,"n_jobs":4},
              "reconstruction_caveats":["Historical semantic mappings were reconstructed.",
                  "Original paper negative IDs and folds are unavailable.",
                  "Figure 2a does not explicitly specify tp_factor; 0.3 is best-supported.",
                  "Original Ubuntu 18.04 compiler/runtime details differ from Colab Linux."]}
    write_json(CONFIG,config)
    print(json.dumps({"verified":True,"upstream_commit":upstream["commit"],
                      "versions":versions,"inputs":inputs,"start_method":start_method},indent=2),flush=True)


def command_embed():
    assert CONFIG.is_file(), "Run verify first"
    assert not EMBEDDING.exists(), "Output already exists; refusing to overwrite"
    sys.path.insert(0,str(UPSTREAM))
    from DREAMwalk import generate_embeddings as ge
    started = utc_now(); tic=time.monotonic()
    print("FULL EMBEDDING START",started,"expected walks",2995900,flush=True)
    os.chdir(str(UPSTREAM))  # Required by upstream's relative C++ compilation path.
    ge.save_embedding_files(netf=str(GRAPH),sim_netf=str(SIM),outputf=str(EMBEDDING),
                            nodetypef=str(TYPES),tp_factor=0.3,seed=0,
                            directed=False,weighted=True,em_max_iter=5,
                            num_walks=100,walk_length=10,workers=4,
                            dimension=128,window_size=4,p=1,q=1,net_delimiter="\t")
    result={"started_utc":started,"ended_utc":utc_now(),
            "elapsed_seconds":time.monotonic()-tic,"embedding_sha256":sha256(EMBEDDING),
            "expected_walks":2995900,"embedding_path":str(EMBEDDING)}
    write_json(RESULTS/"colab_embedding_runtime.json",result)
    print("FULL EMBEDDING FINISHED",json.dumps(result),flush=True)


def norm_summary(values):
    return {"count":len(values),"mean":statistics.mean(values),
            "sd":statistics.pstdev(values),"min":min(values),"max":max(values)}


def command_integrity():
    import numpy as np
    assert EMBEDDING.is_file(), "Embedding missing"
    types = {r["node"]:r["type"] for r in read_rows(TYPES)}
    with EMBEDDING.open("rb") as stream:
        emb=pickle.load(stream)
    assert set(emb)==set(types), "Embedding node set differs from graph nodes"
    norm_by_type={kind:[] for kind in TYPE_COUNTS}
    dtypes=Counter();nan_count=inf_count=zero_count=0
    for node,vector in emb.items():
        array=np.asarray(vector)
        assert array.shape==(128,),"Embedding dimension mismatch: {}".format(node)
        dtypes[str(array.dtype)]+=1
        nan_count+=int(np.isnan(array).sum());inf_count+=int(np.isinf(array).sum())
        if np.isfinite(array).all():
            norm=float(np.linalg.norm(array));zero_count+=norm==0
            norm_by_type[types[node]].append(norm)
    result={"embedding_sha256":sha256(EMBEDDING),"nodes":len(emb),
            "dimension":128,"missing_nodes":0,"extra_nodes":0,
            "dtypes":dict(dtypes),"nan_values":nan_count,"infinite_values":inf_count,
            "zero_vectors":zero_count,"norm_overall":norm_summary([v for group in norm_by_type.values() for v in group]),
            "norm_by_type":{k:norm_summary(v) for k,v in norm_by_type.items()}}
    write_json(INTEGRITY,result)
    assert len(emb)==29959 and nan_count==0 and inf_count==0 and zero_count==0
    print(json.dumps(result,indent=2),flush=True)


def command_evaluate():
    import numpy as np
    from sklearn.metrics import accuracy_score,average_precision_score,roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    assert INTEGRITY.is_file(), "Embedding integrity must pass before evaluation"
    with EMBEDDING.open("rb") as stream: emb=pickle.load(stream)
    positives={(r["drug"],r["disease"]) for r in read_rows(POSITIVES)}
    negatives=[(r["drug"],r["disease"]) for r in read_rows(NEGATIVES,",")]
    assert len(positives)==len(negatives)==5926 and not positives.intersection(negatives)
    pairs=sorted(positives)+negatives
    x=np.asarray([emb[drug]-emb[disease] for drug,disease in pairs],dtype=np.float32)
    y=np.asarray([1]*len(positives)+[0]*len(negatives),dtype=np.int8)
    assert x.shape==(11852,128) and np.isfinite(x).all()
    random.seed(0);np.random.seed(0)
    config=json.loads(CONFIG.read_text())
    settings=config["xgboost"]
    splitter=StratifiedKFold(n_splits=10,shuffle=True,random_state=0)
    rows=[];seen=set()
    for fold,(train,test) in enumerate(splitter.split(x,y),1):
        assert not seen.intersection(test)
        seen.update(test)
        tic=time.monotonic()
        model=XGBClassifier(**settings);model.fit(x[train],y[train])
        prob=model.predict_proba(x[test])[:,1]
        row={"fold":fold,"cv_seed":0,"xgboost_seed":0,"n_train":len(train),"n_test":len(test),
             "n_positive_train":int(y[train].sum()),"n_positive_test":int(y[test].sum()),
             "accuracy":accuracy_score(y[test],prob.round()),
             "auroc":roc_auc_score(y[test],prob),
             "aupr":average_precision_score(y[test],prob),
             "runtime_seconds":time.monotonic()-tic}
        rows.append(row)
        print("FOLD",fold,"accuracy",row["accuracy"],"AUROC",row["auroc"],"AUPR",row["aupr"],flush=True)
    assert seen==set(range(len(pairs)))
    with FOLDS.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    metrics={key:{"mean":statistics.mean(row[key] for row in rows),
                  "sd":statistics.stdev(row[key] for row in rows)} for key in ("accuracy","auroc","aupr")}
    local={"accuracy":0.8037473762105009,"auroc":0.8842840659774147,"aupr":0.8931381698854646}
    paper={"accuracy":0.8770983,"auroc":0.9440605,"aupr":0.9467667}
    comparison={key:{"local_semantic_seed0_10fold":local[key],"colab_semantic_seed0_10fold":metrics[key]["mean"],
                     "paper_figure2a_mean":paper[key],"colab_minus_local":metrics[key]["mean"]-local[key],
                     "colab_minus_paper":metrics[key]["mean"]-paper[key]} for key in metrics}
    summary={"analysis":"ONE Colab/Linux semantic seed-0 embedding, 10-fold DDA evaluation",
             "embedding_sha256":sha256(EMBEDDING),"fold_assignment_source":"Deterministically regenerated from existing local implementation; no saved per-pair fold IDs.",
             "metrics":metrics,"comparison":comparison,
             "reconstruction_caveats":config["reconstruction_caveats"]}
    write_json(SUMMARY,summary)
    config["status"]="EVALUATED";config["completed_utc"]=utc_now();config["embedding_sha256"]=summary["embedding_sha256"]
    write_json(CONFIG,config)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("verify","embed","integrity","evaluate"))
    args=parser.parse_args()
    {"verify":command_verify,"embed":command_embed,"integrity":command_integrity,"evaluate":command_evaluate}[args.command]()


if __name__=="__main__":main()
