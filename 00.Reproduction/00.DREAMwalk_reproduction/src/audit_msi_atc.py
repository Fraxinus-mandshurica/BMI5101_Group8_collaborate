"""Audit public MSI ATC annotations and DREAMwalk child-parent hierarchy.

Creates count/audit tables only. Does not generate a similarity network.
"""
from __future__ import annotations

import csv
import hashlib
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/msi_raw/7_drug_classification_df.tsv"
DRUG_EDGES = ROOT / "data/msi_raw/1_drug_to_protein.tsv"
NODETYPES = ROOT / "data/processed/msi_nodetypes.tsv"
OLD_HIERARCHY = ROOT / "data/processed/msi_atc_hierarchy.csv"
OUT = ROOT / "results"
ATC_PATTERN = re.compile(r"[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}")
DB_PATTERN = re.compile(r"DB[0-9]{5}")
LEVELS = (1, 3, 4, 5, 7)


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def ancestors(code):
    return tuple(code[:n] for n in LEVELS)


def parent_edges(codes):
    return {(code[:child], code[:parent]) for code in codes
            for child, parent in ((7, 5), (5, 4), (4, 3), (3, 1))}


def level_edge_counts(codes):
    edges = parent_edges(codes)
    return {f"atc_level_{child}_to_{parent}_edges": sum(len(a) == child and len(b) == parent for a, b in edges)
            for child, parent in ((7, 5), (5, 4), (4, 3), (3, 1))}


def counts(rows):
    assignments = {(r["db_id"], r["atc_code"]) for r in rows}
    codes = {code for _, code in assignments}
    terms = {a for code in codes for a in ancestors(code)}
    edges = parent_edges(codes)
    memberships = {(drug, term) for drug, code in assignments for term in ancestors(code)}
    return dict(rows=len(rows), drugs=len({r["db_id"] for r in rows}),
                unique_observed_atc_codes=len(codes), unique_hierarchy_terms=len(terms),
                unique_parent_child_edges=len(edges), drug_atc_assignments=len(assignments),
                terminal_drug_atc_assignments=len(assignments),
                all_drug_hierarchy_memberships=len(memberships),
                level_1_terms=sum(len(x) == 1 for x in terms),
                level_2_terms=sum(len(x) == 3 for x in terms),
                level_3_terms=sum(len(x) == 4 for x in terms),
                level_4_terms=sum(len(x) == 5 for x in terms),
                level_5_terms=sum(len(x) == 7 for x in terms),
                **level_edge_counts(codes))


def check_acyclic(edges, graph_drugs):
    parents = defaultdict(set)
    for child, parent in edges:
        assert child != parent
        parents[child].add(parent)
    assert all(len(vals) == 1 for child, vals in parents.items() if child not in graph_drugs)
    for node in parents:
        stack = [(node, frozenset())]
        while stack:
            cur, visited = stack.pop()
            assert cur not in visited, f"Cycle at {node}"
            stack.extend((parent, visited | {cur}) for parent in parents.get(cur, ()))
    return parents


def make_comparison(all_counts, graph_counts, old_nodes, old_edges, full_edges):
    paper_terms, paper_assoc = 4647, 4543
    hypotheses = [
        ("all_source_hierarchy_terms", paper_terms, all_counts["unique_hierarchy_terms"],
         "Five ATC levels from all 3,987 public source rows; exact numerical match."),
        ("graph_reachable_atc_terms", paper_terms, graph_counts["unique_hierarchy_terms"],
         "Only ATC terms assigned to the 1,328 covered MSI graph drugs."),
        ("previous_graph_hierarchy_all_nodes", paper_terms, old_nodes,
         "Includes graph drug IDs and root; not an ATC-term definition."),
        ("all_source_atc_parent_child_edges", paper_assoc, all_counts["unique_parent_child_edges"],
         "All four structural ATC parent-child levels."),
        ("all_source_edges_excluding_level2_to_level1", paper_assoc,
         all_counts["unique_parent_child_edges"]-all_counts["atc_level_3_to_1_edges"],
         "Exact numerical match; association definition not explicitly published."),
        ("graph_reachable_atc_parent_child_edges", paper_assoc, graph_counts["unique_parent_child_edges"],
         "Four ATC structural levels reachable from graph drugs."),
        ("graph_drug_atc_assignments", paper_assoc, graph_counts["drug_atc_assignments"],
         "Unique graph drug-to-terminal ATC pairs; different object type."),
        ("all_source_drug_atc_assignments", paper_assoc, all_counts["drug_atc_assignments"],
         "Unique source-file drug-to-terminal ATC pairs; different object type."),
        ("graph_all_ancestor_memberships", paper_assoc, graph_counts["all_drug_hierarchy_memberships"],
         "Unique graph drug-to-any-ATC-term memberships; not hierarchy edges."),
        ("previous_graph_hierarchy_all_links", paper_assoc, old_edges,
         "Includes graph drug membership, ATC edges and 14 root links."),
        ("full_source_taxonomy_plus_graph_drug_links", paper_assoc, full_edges,
         "Intended DREAMwalk child-parent input; includes graph drug membership and root links."),
    ]
    return [dict(definition=name, paper_value=paper, reconstructed_value=value,
                 difference=value-paper, interpretation=meaning)
            for name, paper, value, meaning in hypotheses]


def main():
    raw = SOURCE.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    source = read_tsv(SOURCE)
    graph = {r["node_1"] for r in read_tsv(DRUG_EDGES)}
    prepared = {r["node"] for r in read_tsv(NODETYPES) if r["type"] == "drug"}
    assert graph == prepared and len(graph) == 1661
    assert all(DB_PATTERN.fullmatch(r["db_id"]) for r in source)
    assert len(source) == 3987
    columns = list(source[0])
    assert columns == ["db_id", "atc_code", "level_4", "level_4_name", "level_3", "level_3_name",
                       "level_2", "level_2_name", "level_1", "level_1_name"]
    missing = sum(not r["atc_code"] for r in source)
    malformed = [r for r in source if not ATC_PATTERN.fullmatch(r["atc_code"])]
    mismatched_ancestors = [r for r in source if any(r[f"level_{level}"] != r["atc_code"][:size]
                             for level, size in ((1, 1), (2, 3), (3, 4), (4, 5)))]
    assignments = [(r["db_id"], r["atc_code"]) for r in source]
    duplicate_assignments = len(assignments)-len(set(assignments))
    assert missing == 0 and not malformed and not mismatched_ancestors and duplicate_assignments == 0, (
        "ATC source requires review before hierarchy construction")
    source_drugs = {r["db_id"] for r in source}
    graph_rows = [r for r in source if r["db_id"] in graph]
    graph_assignments = {(r["db_id"], r["atc_code"]) for r in graph_rows}
    graph_by_drug = defaultdict(set)
    for drug, code in graph_assignments:
        graph_by_drug[drug].add(code)
    all_counts, graph_counts = counts(source), counts(graph_rows)
    assert all_counts["drugs"] == 2379 and graph_counts["drugs"] == 1328
    uncovered = graph-source_drugs
    assert len(uncovered) == 333
    source_normalized = {x.strip().upper() for x in source_drugs}
    assert not ({x.strip().upper() for x in uncovered} & source_normalized)
    multiplicity = Counter(len(vals) for vals in graph_by_drug.values())
    mean_codes = statistics.mean(map(len, graph_by_drug.values()))
    median_codes = statistics.median(map(len, graph_by_drug.values()))
    atc_codes_all = {r["atc_code"] for r in source}
    taxonomy_edges = parent_edges(atc_codes_all)
    root_edges = {(code[0], "drug") for code in atc_codes_all}
    membership_edges = set(graph_assignments)
    typed_edges = {(a, b, "ATC_PARENT_CHILD") for a, b in taxonomy_edges}
    typed_edges |= {(a, b, "ATC_ROOT") for a, b in root_edges}
    typed_edges |= {(a, b, "DRUG_ATC_ASSIGNMENT") for a, b in membership_edges}
    hierarchy_edges = {(a, b) for a, b, _ in typed_edges}
    assert len(hierarchy_edges) == len(typed_edges)
    parent_map = check_acyclic(hierarchy_edges, graph)
    # Each assigned terminal code leads through exactly 5 ATC levels to drug root.
    for drug in graph:
        codes = graph_by_drug[drug]
        if codes:
            assert drug in parent_map and parent_map[drug] == codes
            for code in codes:
                cur, path = drug, [drug]
                while cur in parent_map:
                    cur = next(iter(parent_map[cur])) if cur != drug else code
                    path.append(cur)
                assert path == [drug, code, code[:5], code[:4], code[:3], code[:1], "drug"]
        else:
            assert drug not in parent_map
    old_rows = list(csv.DictReader(OLD_HIERARCHY.open(newline="", encoding="utf-8")))
    old_edges = {(r["child"], r["parent"]) for r in old_rows}
    old_nodes = {x for edge in old_edges for x in edge}
    graph_codes = {r["atc_code"] for r in graph_rows}
    expected_old = parent_edges(graph_codes) | {(code[0], "drug") for code in graph_codes} | membership_edges
    assert old_edges == expected_old and len(old_edges) == 5554 and len(old_nodes) == 4435
    audit = []
    for drug in sorted(graph):
        codes = sorted(graph_by_drug[drug])
        if codes:
            status, note = "ATC_MAPPED", "exact_source_drug_id"
        elif DB_PATTERN.fullmatch(drug):
            status, note = "NO_ATC_SOURCE_ROW_DB_ID", "DB-format graph ID absent from public ATC source"
        else:
            status, note = "NO_ATC_SOURCE_ROW_NON_DB_ID", "graph ID outside source DB ID namespace"
        audit.append(dict(drug_id=drug, in_atc_source=bool(codes), n_atc_codes=len(codes),
                          atc_codes=";".join(codes), mapping_status=status, notes=note))
    counts_by_status = Counter(r["mapping_status"] for r in audit)
    summary = dict(total_msi_drugs=len(graph), mapped_msi_drugs=graph_counts["drugs"],
                   mapped_percent=round(100*graph_counts["drugs"]/len(graph), 2),
                   unmapped_msi_drugs=len(uncovered), unmapped_percent=round(100*len(uncovered)/len(graph), 2),
                   unmapped_db_format_absent=counts_by_status["NO_ATC_SOURCE_ROW_DB_ID"],
                   unmapped_non_db_identifier=counts_by_status["NO_ATC_SOURCE_ROW_NON_DB_ID"],
                   source_rows=all_counts["rows"], source_drugs=all_counts["drugs"],
                   source_drugs_not_in_graph=len(source_drugs-graph),
                   source_drug_atc_assignments=all_counts["drug_atc_assignments"],
                   drug_atc_assignment_count=graph_counts["drug_atc_assignments"],
                   terminal_drug_atc_assignments=graph_counts["terminal_drug_atc_assignments"],
                   all_drug_hierarchy_memberships=graph_counts["all_drug_hierarchy_memberships"],
                   unique_observed_atc_codes=all_counts["unique_observed_atc_codes"],
                   unique_hierarchy_terms=all_counts["unique_hierarchy_terms"],
                   unique_parent_child_edges=all_counts["unique_parent_child_edges"],
                   graph_unique_observed_atc_codes=graph_counts["unique_observed_atc_codes"],
                   graph_unique_hierarchy_terms=graph_counts["unique_hierarchy_terms"],
                   graph_unique_parent_child_edges=graph_counts["unique_parent_child_edges"],
                   source_all_drug_hierarchy_memberships=all_counts["all_drug_hierarchy_memberships"],
                   graph_drugs_one_atc=multiplicity[1], graph_drugs_two_atc=multiplicity[2],
                   graph_drugs_three_atc=multiplicity[3],
                   graph_drugs_more_than_three_atc=sum(v for n, v in multiplicity.items() if n > 3),
                   max_atc_codes_per_graph_drug=max(multiplicity),
                   mean_atc_codes_per_mapped_drug=round(mean_codes, 4),
                   median_atc_codes_per_mapped_drug=median_codes,
                   missing_atc_values=missing, malformed_atc_codes=len(malformed),
                   duplicate_source_assignments=duplicate_assignments,
                   mismatched_source_ancestor_columns=len(mismatched_ancestors),
                   atc_code_length_distribution="7:"+str(len(source)),
                   previous_hierarchy_nodes=len(old_nodes), previous_hierarchy_links=len(old_edges),
                   intended_hierarchy_edges=len(hierarchy_edges),
                   public_source_sha256=sha,
                   **{f"all_source_{k}": v for k, v in level_edge_counts(atc_codes_all).items()},
                   **{f"graph_{k}": v for k, v in level_edge_counts(graph_codes).items()})
    assert len(audit) == 1661 and len({r["drug_id"] for r in audit}) == 1661
    assert sum(r["n_atc_codes"] for r in audit) == len(graph_assignments)
    assert sum(r["n_atc_codes"] > 0 for r in audit) == 1328
    assert sum(r["n_atc_codes"] == 0 for r in audit) == 333
    assert sum(multiplicity.values()) == 1328
    assert summary["unmapped_db_format_absent"] + summary["unmapped_non_db_identifier"] == 333
    assert len(hierarchy_edges) == (all_counts["unique_parent_child_edges"]+14+len(graph_assignments))
    comparison = make_comparison(all_counts, graph_counts, len(old_nodes), len(old_edges), len(hierarchy_edges))
    write_csv(OUT / "msi_atc_mapping_audit.csv", audit, list(audit[0]))
    write_csv(OUT / "msi_atc_uncovered_drugs.csv", [r for r in audit if not r["in_atc_source"]], list(audit[0]))
    write_csv(OUT / "msi_atc_mapping_summary.csv", [summary], list(summary))
    write_csv(OUT / "msi_atc_hierarchy_edges.csv", [dict(child=a, parent=b) for a, b in sorted(hierarchy_edges)], ["child", "parent"])
    write_csv(OUT / "msi_atc_hierarchy_edge_types.csv", [dict(child=a, parent=b, edge_type=t) for a, b, t in sorted(typed_edges)], ["child", "parent", "edge_type"])
    write_csv(OUT / "msi_atc_paper_comparison.csv", comparison, list(comparison[0]))
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
