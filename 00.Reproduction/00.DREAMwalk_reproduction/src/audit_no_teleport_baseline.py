"""Cheap, read-only-input diagnostics for the MSI no-teleport baseline."""
from __future__ import annotations

import csv
import json
import math
import pickle
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vendor/DREAMwalk_upstream"), str(ROOT / "src")]
from run_msi_repeat0 import GRAPH, NODETYPES, POSITIVES, NEGATIVES, read_rows, sha256, xgb_settings

OUT = ROOT / "results"
REL = {1: "drug-protein", 2: "disease-protein", 3: "protein-protein", 4: "protein-GO", 5: "GO-GO"}


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def csvout(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def stats(values):
    values = list(values)
    return {"n": len(values), "min": min(values), "max": max(values),
            "mean": statistics.mean(values), "sd": statistics.pstdev(values),
            "median": statistics.median(values)}


def graph_and_types():
    from DREAMwalk.utils import read_graph
    types, assignments = {}, Counter()
    for row in read_rows(NODETYPES):
        assignments[row["node"]] += 1
        types[row["node"]] = row["type"]
    node_set, rel, weights, ids = set(), Counter(), Counter(), []
    exact, directed, unordered = Counter(), Counter(), Counter()
    loops = 0
    for line in GRAPH.open():
        a, b, t, weight, eid = line.rstrip("\n").split("\t")
        t, weight, eid = int(t), float(weight), int(eid)
        node_set.update((a,b)); rel[t] += 1; weights[weight] += 1; ids.append(eid)
        exact[(a,b,t,weight)] += 1
        directed[(a,b,t,weight)] += 1
        unordered[(min(a,b),max(a,b))] += 1
        loops += a == b
    positives = {(r["drug"],r["disease"]) for r in read_rows(POSITIVES)}
    treatment = sum((a,b) in positives or (b,a) in positives for a,b in unordered)
    G = read_graph(str(GRAPH), weighted=True, directed=False, delimiter="\t")
    loaded_rel = Counter(d["type"] for _,_,d in G.edges(data=True))
    degree = {t: stats(G.degree(n) for n in node_set if types.get(n)==t) for t in sorted(set(types.values()))}
    ga = {"source": str(GRAPH.relative_to(ROOT)), "sha256": sha256(GRAPH),
          "raw_unique_nodes":len(node_set), "raw_edge_rows":sum(rel.values()),
          "relation_mapping":REL, "raw_relation_counts":rel,
          "duplicate_exact_rows_excluding_id":sum(v-1 for v in exact.values()),
          "parallel_extra_rows_same_undirected_pair":sum(v-1 for v in unordered.values()),
          "undirected_pairs_with_parallel_rows":sum(v>1 for v in unordered.values()),
          "exact_reverse_direction_row_pairs":sum(min(v,directed.get((b,a,t,w),0)) for (a,b,t,w),v in directed.items() if a<b),
          "weight_distribution":{str(k):v for k,v in sorted(weights.items())},
          "unique_weight_values":sorted(weights), "nonfinite_weight_count":sum(v for k,v in weights.items() if not math.isfinite(k)),
          "unique_edge_ids":len(set(ids)), "duplicate_edge_ids":len(ids)-len(set(ids)),
          "edge_id_min":min(ids),"edge_id_max":max(ids), "self_loops":loops,
          "known_treatment_pairs":len(positives),"known_treatment_pairs_in_graph":treatment,
          "loaded_graph_class":type(G).__name__,"loaded_directed":G.is_directed(),
          "loaded_multigraph":G.is_multigraph(),"loaded_nodes":G.number_of_nodes(),
          "loaded_edges":G.number_of_edges(),"loaded_relation_counts":loaded_rel,
          "loaded_isolated_nodes":sum(d==0 for _,d in G.degree()), "degree_by_node_type":degree}
    save("diagnostic_no_teleport_graph_audit.json", ga)
    ta = {"source":str(NODETYPES.relative_to(ROOT)),"sha256":sha256(NODETYPES),
          "assignment_rows":sum(assignments.values()),"unique_nodes":len(types),
          "counts":Counter(types.values()),"duplicated_assignments":sum(v-1 for v in assignments.values()),
          "unknown_labels":sorted(set(types.values())-{"drug","disease","gene","etc"}),
          "graph_nodes_absent_from_type_file":sorted(node_set-set(types)),
          "type_file_nodes_absent_from_graph":sorted(set(types)-node_set),
          "upstream_prefix_mapping":{"drug":"d","disease":"i","gene":"g","etc":"e"}}
    save("diagnostic_no_teleport_node_type_audit.json",ta)
    return G,types,ga,ta


def em_and_walks(G, types, nwalks=20000):
    import networkx as nx
    import numpy as np
    from scipy import stats as spstats
    from DREAMwalk import generate_embeddings as ge
    from DREAMwalk.utils import set_seed
    set_seed(0)
    matrix=ge._init_edge_transition_matrix(str(GRAPH),"\t")
    history=[]; matrices=[matrix.copy()]
    for iteration in range(1,6):
        paths=ge._sample_edge_paths(G,matrix,10,1,1)
        counts={i:[Counter(p).get(i,0) for p in paths] for i in range(1,6)}
        raw=np.array([[spstats.mstats.pearsonr(counts[i],counts[j])[0] for j in range(1,6)] for i in range(1,6)],dtype=float)
        new=ge._update_trans_matrix(paths,matrix)
        assert np.allclose(np.nan_to_num(1/(1+np.exp(-raw))),new)
        diff=float(np.nan_to_num(np.absolute((new-matrix)/matrix)).mean())
        history.append({"iteration":iteration,"sampled_edges":int(G.number_of_edges()*0.01),
                        "generated_edge_paths":len(paths),"path_length_distribution":dict(Counter(map(len,paths))),
                        "nonfinite_raw_pearson_entries":int((~np.isfinite(raw)).sum()),
                        "zero_entries_after_nan_to_num":int((new==0).sum()),
                        "convergence_metric":diff,"matrix":new.tolist()})
        matrices.append(new.copy()); matrix=new
        print("EM",iteration,"diff",diff,flush=True)
        if diff<0.01: break
    csvout("diagnostic_edge_transition_matrix.csv",[
        {"from_relation":i,**{f"to_{j}":float(matrix[i-1,j-1]) for j in range(1,6)}} for i in range(1,6)])
    csvout("diagnostic_edge_transition_em_history.csv",[
        {"iteration":i,"convergence_metric":None if i==0 else history[i-1]["convergence_metric"],
         "sampled_edges":None if i==0 else history[i-1]["sampled_edges"],
         "generated_edge_paths":None if i==0 else history[i-1]["generated_edge_paths"],
         "path_length_distribution":None if i==0 else json.dumps(history[i-1]["path_length_distribution"]),
         "nonfinite_raw_pearson_entries":None if i==0 else history[i-1]["nonfinite_raw_pearson_entries"],
         "matrix":json.dumps(m.tolist())} for i,m in enumerate(matrices)])
    # This is a separate, sequential diagnostic sample, never a training corpus.
    random.seed(0); np.random.seed(0)
    # One start per sampled node mirrors the full run's equal walks per graph node.
    starts=random.sample(sorted(G.nodes()),nwalks)
    random.shuffle(starts)
    original=ge._network_traverse
    recorded=[]
    def trace(*args):
        result=original(*args)
        recorded.append(result[1]);return result
    ge._network_traverse=trace
    path_lengths=Counter(); positions=defaultdict(Counter); node_freq=Counter();
    edges=Counter(); trans=defaultdict(Counter); node_trans=Counter(); ambiguous_first=0; full=0
    paths_containing=Counter()
    try:
        for idx,start in enumerate(starts):
            recorded.clear()
            path=ge._DREAMwalker(start,G,nx.empty_graph(),matrix,1,1,10,0)
            path_lengths[len(path)]+=1; full+=len(path)==10
            paths_containing.update({types[node] for node in path})
            for pos,node in enumerate(path):
                kind=types[node];positions[pos+1][kind]+=1;node_freq[kind]+=1
            for a,b in zip(path,path[1:]):node_trans[f"{types[a]}->{types[b]}"]+=1
            if len(path)>1:
                first_types={d["type"] for d in G[path[0]][path[1]].values()}
                if len(first_types)==1:
                    path_edge_types=[first_types.pop()]+list(recorded)
                    assert len(path_edge_types)==len(path)-1
                    edges.update(path_edge_types)
                    for a,b in zip(path_edge_types,path_edge_types[1:]):trans[a][b]+=1
                else:ambiguous_first+=1
            if (idx+1)%3000==0:print("walks",idx+1,flush=True)
    finally:ge._network_traverse=original
    rows={str(i):{str(j):trans[i][j]/sum(trans[i].values()) if sum(trans[i].values()) else None for j in range(1,6)} for i in range(1,6)}
    result={"analysis":"DIAGNOSTIC SAMPLE, not formal embedding", "walk_count":len(starts),
            "start_nodes_per_type":Counter(types[n] for n in starts),"path_length_distribution":path_lengths,
            "fraction_full_length_10":full/len(starts),"node_type_by_position":positions,
            "node_type_occurrences":node_freq,"node_type_occurrence_fraction":{t:c/sum(node_freq.values()) for t,c in node_freq.items()},
            "edge_type_occurrences":edges,"edge_type_fraction":{str(t):c/sum(edges.values()) for t,c in edges.items()},
            "ambiguous_first_edge_type_paths":ambiguous_first,
            "observed_edge_type_transition_counts":trans,"observed_row_normalized_edge_type_transitions":rows,
            "learned_matrix":matrix.tolist(),"consecutive_node_type_transitions":node_trans,
            "ppi_node_fraction_by_position":{str(p):c["gene"]/sum(c.values()) for p,c in positions.items()},
            "drug_occurrence_rate_per_walk":node_freq["drug"]/len(starts),
            "disease_occurrence_rate_per_walk":node_freq["disease"]/len(starts),
            "fraction_paths_containing_type":{t:c/len(starts) for t,c in paths_containing.items()},
            "em_iterations":len(history),"em_history":history}
    save("diagnostic_molecular_walk_summary.json",result)
    return result


def embedding_and_skipgram(types):
    import numpy as np
    path=ROOT/"data/embeddings/msi_seed0_no_teleport.pkl"
    with path.open("rb") as f:emb=pickle.load(f)
    all_norms={t:[] for t in ("drug","disease","gene","etc")}
    nan=inf=zero=badshape=0; dimensions=Counter();dtypes=Counter()
    for n,v in emb.items():
        a=np.asarray(v);dimensions[str(a.shape)]+=1;dtypes[str(a.dtype)]+=1
        if a.shape!=(128,):badshape+=1;continue
        nan+=int(np.isnan(a).sum());inf+=int(np.isinf(a).sum())
        if np.isfinite(a).all():
            norm=float(np.linalg.norm(a));zero+=norm==0
            if n in types:all_norms[types[n]].append(norm)
    result={"source":str(path.relative_to(ROOT)),"sha256":sha256(path),"embedding_count":len(emb),
            "dimensions":dimensions,"dtypes":dtypes,"missing_graph_nodes":len(set(types)-set(emb)),
            "extra_nodes":len(set(emb)-set(types)),"nan_values":nan,"infinite_values":inf,
            "zero_vectors":zero,"bad_shape_vectors":badshape,
            "norm_overall":stats(v for group in all_norms.values() for v in group),
            "norm_by_type":{t:stats(v) for t,v in all_norms.items()}}
    save("diagnostic_no_teleport_embedding_integrity.json",result)
    sg={"actual_command":"./HetSG -train tmp_walkfile -output tmp_outputf -pp 1 -min-count 0 -size 128 -iter 1 -window 4 -threads 4",
        "compiler":"g++ DREAMwalk/HeterogeneousSG.cpp -o HetSG -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result",
        "effective_settings":{"pp":1,"size":128,"window":4,"iter":1,"negative":5,"hs":0,"initial_alpha":0.025,"sample":0.001,"min_count":0,"threads":4,"cbow":0},
        "negative_sampling":"C++ NodeType partitions 1e8-entry global unigram table by first prefix character. For each context token last_word, d/v_table, i/i_table, g/f_table, e/a_table; candidate first character is checked. Tables are weighted multisets, not one entry per node.",
        "context_pool_distinct_node_counts_in_full_vocab_inferred_from_complete_embedding":Counter(types.values()),
        "actual_weighted_table_lengths":"Unavailable: temporary full walk corpus and C++ in-memory tables were removed after training.",
        "caveat":"C++ uses context token last_word to choose the negative pool; rejected candidates are skipped, not resampled. It does not explicitly exclude positive target in pp branch. The dynamic-window offset is disabled (b=0), so the effective context window reaches all four positions on both sides when available.",
        "paper_vs_code":"Paper wording says maximum depth below six; released XGBoost constructor uses six. Skip-gram settings above come from released code; no documented invocation mismatch found."}
    save("diagnostic_skipgram_audit.json",sg)
    return emb,result,sg


def depth(emb):
    import numpy as np
    from sklearn.metrics import accuracy_score,roc_auc_score,average_precision_score
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    positives={(r["drug"],r["disease"]) for r in read_rows(POSITIVES)}
    negatives=[(r["drug"],r["disease"]) for r in read_rows(NEGATIVES,",")]
    pairs=sorted(positives)+negatives
    x=np.asarray([emb[a]-emb[b] for a,b in pairs],dtype=np.float32)
    y=np.asarray([1]*len(positives)+[0]*len(negatives),dtype=np.int8)
    old={int(r["fold"]):r for r in read_rows(OUT/"diagnostic_no_teleport_fold_results.csv",",")}
    folds=[]
    random.seed(0);np.random.seed(0)
    for fold,(train,test) in enumerate(StratifiedKFold(n_splits=10,shuffle=True,random_state=0).split(x,y),1):
        settings=xgb_settings();settings["max_depth"]=5
        model=XGBClassifier(**settings);model.fit(x[train],y[train])
        p=model.predict_proba(x[test])[:,1]
        vals={"accuracy":accuracy_score(y[test],p.round()),"auroc":roc_auc_score(y[test],p),"aupr":average_precision_score(y[test],p)}
        row={"fold":fold,"n_train":len(train),"n_test":len(test)}
        for metric in vals:
            v6=float(old[fold][metric]);row[f"depth6_{metric}"]=v6
            row[f"depth5_{metric}"]=vals[metric];row[f"difference_5_minus_6_{metric}"]=vals[metric]-v6
        folds.append(row);print("depth5 fold",fold,vals,flush=True)
    csvout("diagnostic_xgb_depth_folds.csv",folds)
    summary=[]
    for metric in ("accuracy","auroc","aupr"):
        v6=[r[f"depth6_{metric}"] for r in folds];v5=[r[f"depth5_{metric}"] for r in folds]
        summary.append({"metric":metric,"depth6_mean":statistics.mean(v6),"depth5_mean":statistics.mean(v5),
                        "difference_5_minus_6":statistics.mean(v5)-statistics.mean(v6),
                        "depth6_sd":statistics.stdev(v6),"depth5_sd":statistics.stdev(v5)})
    csvout("diagnostic_xgb_depth_summary.csv",summary)
    return summary


if __name__=="__main__":
    phase=sys.argv[1]
    if phase=="graph":
        G,types,ga,ta=graph_and_types()
        print(json.dumps({"graph":ga,"types":ta},default=str),flush=True)
    elif phase=="emwalk":
        from DREAMwalk.utils import read_graph
        types={r["node"]:r["type"] for r in read_rows(NODETYPES)}
        G=read_graph(str(GRAPH),weighted=True,directed=False,delimiter="\t")
        result=em_and_walks(G,types)
        print(json.dumps({"em_iterations":result["em_iterations"],"full_length":result["fraction_full_length_10"]}),flush=True)
    elif phase=="embedding":
        types={r["node"]:r["type"] for r in read_rows(NODETYPES)}
        emb,integrity,sg=embedding_and_skipgram(types)
        print(json.dumps(integrity,default=str),flush=True)
    elif phase=="depth":
        with (ROOT/"data/embeddings/msi_seed0_no_teleport.pkl").open("rb") as f:emb=pickle.load(f)
        print(json.dumps(depth(emb)),flush=True)
    else:raise SystemExit("Expected graph, emwalk, embedding, or depth")
