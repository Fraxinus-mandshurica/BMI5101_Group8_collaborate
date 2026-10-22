"""Prepare pinned native Hetionet inputs for the unchanged DREAMwalk pipeline.

This module performs data I/O only. It never generates walks, trains a model,
or evaluates embeddings. Molecular rows retain their official source order;
the upstream MultiDiGraph reader decides how parallel and reverse edges behave.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import NamedTuple
from urllib.request import Request, urlopen


class Source(NamedTuple):
    name: str
    url: str
    sha256: str
    size: int


_HETIO = "https://raw.githubusercontent.com/hetio/hetionet/4933ca17639cdcf146f91cc44e28128cbfc6fd24/hetnet/tsv/"
SOURCES = (
    Source("hetionet-v1.0-nodes.tsv", _HETIO + "hetionet-v1.0-nodes.tsv",
           "3ad9eb86c031df8bc7d5efefa6edc93562182eb9a6a71e5129d3f18b9d42c50e", 2509984),
    Source("hetionet-v1.0-edges.sif.gz", _HETIO + "hetionet-v1.0-edges.sif.gz",
           "611f0411ac666be4e0270e54bbca67407e14720bfc3099311f0f5eec25c5a947", 12401407),
    Source("HumanDO.obo", "https://raw.githubusercontent.com/dhimmel/disease-ontology/75050ea2d4f60e745d3f3578ae03560a2cc0e444/download/HumanDO.obo",
           "220921d830b1b8c71054e5e3fc73f0774ce35e2406f4e33eda18021a50535364", 4014997),
    Source("drugbank-slim.tsv", "https://raw.githubusercontent.com/dhimmel/drugbank/3e87872db5fca5ac427ce27464ab945c0ceb4ec6/data/drugbank-slim.tsv",
           "cfb847f197862f6642484d6bb4ed96473d23235873a0ee4d4a9ebd8dc63d3613", 441087),
)

# Supplementary Table 3 order. Numeric IDs are contiguous, as upstream requires.
EDGE_TYPES = ("CbG", "CuG", "CdG", "DaG", "DuG", "DdG", "Gr>G", "GiG", "GcG", "GpPW")
# Use names recognized by upstream HeterogeneousSG.type2meta: d/i/g/e.
NODE_TYPES = (("Compound", "drug"), ("Disease", "disease"),
              ("Gene", "gene"), ("Pathway", "etc"))
REFERENCE_EDGE_COUNTS = (11571, 18756, 21102, 12623, 7731, 7623,
                         265672, 147164, 61690, 84372)


def paths(root):
    root = Path(root).resolve()
    processed = root / "data/processed"
    audit = root / "results/hetionet_inputs"
    return dict(graph=processed / "hetionet_molecular.tsv",
                nodetypes=processed / "hetionet_nodetypes.tsv",
                positives=processed / "hetionet_positives.csv",
                raw_positives=processed / "hetionet_positives_raw.csv",
                hierarchy=processed / "hetionet_semantic_hierarchy.csv",
                manifest=audit / "manifest.json", audit_dir=audit,
                raw_dir=root / "data/hetionet_raw")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _atomic_text(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            yield stream
        if path.exists() and sha256(path) == sha256(temporary):
            temporary.unlink()
        else:
            os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_rows(path, rows, header=None, delimiter=","):
    with _atomic_text(path) as stream:
        writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
        if header is not None:
            writer.writerow(header)
        writer.writerows(rows)


def _write_json(path, value):
    with _atomic_text(path) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def _validate_source(path, source):
    if path.stat().st_size != source.size or sha256(path) != source.sha256:
        raise ValueError("Pinned source does not match: " + str(path))


def download_sources(root):
    """Download four fixed public sources atomically, or verify local copies."""
    raw = paths(root)["raw_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    result = {}
    for source in SOURCES:
        destination = raw / source.name
        if destination.exists():
            _validate_source(destination, source)
        else:
            descriptor, name = tempfile.mkstemp(prefix=source.name + ".", dir=str(raw))
            temporary = Path(name)
            try:
                request = Request(source.url, headers={"User-Agent": "DREAMwalk-reproduction"})
                with os.fdopen(descriptor, "wb") as output, urlopen(request, timeout=60) as response:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                _validate_source(temporary, source)
                os.replace(str(temporary), str(destination))
            finally:
                if temporary.exists():
                    temporary.unlink()
        result[source.name] = destination
    return result


def _select_molecular(nodes, edges):
    """Keep ten molecular relations, including their original order and direction."""
    allowed = dict(NODE_TYPES)
    type_ids = {name: index + 1 for index, name in enumerate(EDGE_TYPES)}
    retained, positives, active, counts = [], [], set(), Counter()
    undirected_pairs = set()
    self_loops = 0
    for row in edges:
        source, target, relation = row["source"], row["target"], row["metaedge"]
        if source not in nodes or target not in nodes:
            raise ValueError("An edge endpoint is absent from the official nodes table")
        if relation == "CtD":
            if nodes[source]["kind"] != "Compound" or nodes[target]["kind"] != "Disease":
                raise ValueError("Unexpected CtD endpoint types")
            positives.append((source, target))
        if relation not in type_ids:
            continue
        if nodes[source]["kind"] not in allowed or nodes[target]["kind"] not in allowed:
            raise ValueError("Unexpected molecular endpoint types")
        if source == target:
            self_loops += 1
        retained.append((source, target, type_ids[relation], 1, len(retained)))
        counts[relation] += 1
        active.update((source, target))
        undirected_pairs.add(tuple(sorted((source, target))))
    if self_loops:
        raise ValueError("Official molecular input contains self-loops; do not silently remove them")
    if len(positives) != len(set(positives)):
        raise ValueError("Duplicate official treatment pairs")
    return retained, positives, active, counts, len(undirected_pairs)


def _parse_obo(text):
    """Read native is_a edges and exact alt_id aliases; never use external xrefs."""
    terms, aliases, parents = {}, {}, {}
    stanza, fields = None, defaultdict(list)

    def finish():
        if stanza != "Term" or not fields.get("id"):
            return
        node = fields["id"][0]
        terms[node] = dict(fields)
        parents[node] = set(value.split(" ! ", 1)[0].strip() for value in fields.get("is_a", []))
        for alias in fields.get("alt_id", []):
            if alias in aliases and aliases[alias] != node:
                raise ValueError("Ambiguous Disease Ontology alias: " + alias)
            aliases[alias] = node

    for line in text.splitlines():
        if line.startswith("[") and line.endswith("]"):
            finish()
            stanza, fields = line[1:-1], defaultdict(list)
        elif stanza == "Term" and ": " in line:
            key, value = line.split(": ", 1)
            fields[key].append(value)
    finish()
    if "DOID:4" not in terms or parents["DOID:4"]:
        raise ValueError("Expected Disease Ontology root DOID:4")
    return terms, aliases, parents


def _disease_paths(node, parents, visiting=()):
    if node in visiting:
        raise ValueError("Cycle in Disease Ontology hierarchy")
    if node not in parents:
        raise ValueError("Missing Disease Ontology term: " + node)
    if not parents[node]:
        if node != "DOID:4":
            raise ValueError("Unexpected Disease Ontology root: " + node)
        return [(node,)]
    return [(node,) + path for parent in sorted(parents[node])
            for path in _disease_paths(parent, parents, visiting + (node,))]


def _semantic_hierarchy(nodes, active, drugbank_rows, obo_text):
    terms, aliases, parents = _parse_obo(obo_text)
    compounds = {row["drugbank_id"]: row for row in drugbank_rows}
    edges, expected_paths, do_audit, atc_audit = set(), {}, [], []
    coverage = Counter()
    for node in sorted(nodes):
        kind = nodes[node]["kind"]
        if kind not in ("Compound", "Disease"):
            continue
        identifier = node.split("::", 1)[1]
        is_active = node in active
        if kind == "Compound":
            if identifier not in compounds:
                raise ValueError("Compound missing from same-source DrugBank slim: " + identifier)
            codes = sorted(filter(None, compounds[identifier]["atc_codes"].split("|")))
            if len(codes) != len(set(codes)):
                raise ValueError("Duplicate ATC assignment for " + identifier)
            for code in codes:
                if re.fullmatch(r"[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}", code) is None:
                    raise ValueError("Invalid full ATC code: " + code)
                atc_audit.append((node, code, is_active))
                if is_active:
                    chain = (node,) + tuple("ATC::" + code[:size] for size in (7, 5, 4, 3, 1)) + ("drug",)
                    expected_paths.setdefault(node, set()).add(chain)
                    edges.update(zip(chain, chain[1:]))
            coverage["drug_total"] += 1
            coverage["drug_mapped"] += bool(codes)
            if is_active:
                coverage["active_drug_total"] += 1
                coverage["active_drug_mapped"] += bool(codes)
                coverage["active_atc_assignments"] += len(codes)
        else:
            canonical = aliases.get(identifier, identifier)
            if canonical not in terms or terms[canonical].get("is_obsolete") == ["true"]:
                raise ValueError("Disease missing or obsolete in fixed DO snapshot: " + identifier)
            resolution = "alt_id" if canonical != identifier else "exact_id"
            do_audit.append((node, identifier, canonical, resolution, is_active))
            chains = _disease_paths(canonical, parents)
            coverage["disease_total"] += 1
            coverage["disease_mapped"] += 1
            coverage["disease_alt_id"] += resolution == "alt_id"
            if is_active:
                coverage["active_disease_total"] += 1
                coverage["active_disease_mapped"] += 1
                coverage["active_do_paths"] += len(chains)
                for path in chains:
                    chain = (node,) + tuple("DO::" + term for term in path) + ("disease",)
                    expected_paths.setdefault(node, set()).add(chain)
                    edges.update(zip(chain, chain[1:]))
    coverage["hierarchy_edges"] = len(edges)
    coverage["drug_hierarchy_edges"] = sum(parent == "drug" or parent.startswith("ATC::") for _, parent in edges)
    coverage["disease_hierarchy_edges"] = sum(parent == "disease" or parent.startswith("DO::") for _, parent in edges)
    return sorted(edges), expected_paths, do_audit, atc_audit, dict(coverage)


def prepare_inputs(root):
    """Verify downloaded sources and write deterministic graph/hierarchy audits.

    Native identifiers are retained. Nodes isolated after molecular filtering
    are audited and omitted, because the unchanged edge reader cannot add them.
    Neither self-loops nor treatment edges are introduced to retain isolates.
    """
    root, output = Path(root).resolve(), paths(root)
    source_records = []
    for source in SOURCES:
        source_path = output["raw_dir"] / source.name
        _validate_source(source_path, source)
        source_records.append(dict(source._asdict(), path=str(source_path.relative_to(root))))
    with (output["raw_dir"] / SOURCES[0].name).open(encoding="utf-8", newline="") as stream:
        node_rows = list(csv.DictReader(stream, delimiter="\t"))
    nodes = {row["id"]: row for row in node_rows}
    if len(nodes) != len(node_rows):
        raise ValueError("Duplicate native Hetionet node IDs")
    with gzip.open(output["raw_dir"] / SOURCES[1].name, "rt", encoding="utf-8", newline="") as stream:
        graph_rows, positives, active, counts, simple_edges = _select_molecular(
            nodes, csv.DictReader(stream, delimiter="\t"))
    active_positives = [pair for pair in positives if set(pair) <= active]
    excluded = [(drug, disease, ";".join(node for node in (drug, disease) if node not in active))
                for drug, disease in positives if not {drug, disease} <= active]
    node_type_map = dict(NODE_TYPES)
    isolates = [(node, nodes[node]["name"], nodes[node]["kind"], node_type_map[nodes[node]["kind"]])
                for node in sorted(nodes) if nodes[node]["kind"] in node_type_map and node not in active]
    with (output["raw_dir"] / SOURCES[3].name).open(encoding="utf-8", newline="") as stream:
        drugbank_rows = list(csv.DictReader(stream, delimiter="\t"))
    hierarchy, expected_paths, do_audit, atc_audit, semantic = _semantic_hierarchy(
        nodes, active, drugbank_rows, (output["raw_dir"] / SOURCES[2].name).read_text(encoding="utf-8"))
    active_counts = Counter(node_type_map[nodes[node]["kind"]] for node in active)
    expected = (len(nodes) == 47031 and tuple(counts[k] for k in EDGE_TYPES) == REFERENCE_EDGE_COUNTS
                and len(graph_rows) == 638304 and simple_edges == 634482
                and dict(active_counts) == dict(drug=1440, disease=134, gene=18500, etc=1822)
                and len(positives) == 755 and len(active_positives) == 751
                and semantic["drug_mapped"] == 1255 and semantic["disease_mapped"] == 137)
    if not expected:
        raise ValueError("Pinned Hetionet preparation does not match the audited reference counts")
    _write_rows(output["graph"], graph_rows, delimiter="\t")
    # Official node table order is preserved for the active node-type table.
    _write_rows(output["nodetypes"], ((row["id"], node_type_map[row["kind"]])
                                      for row in node_rows if row["id"] in active),
                header=("node", "type"), delimiter="\t")
    _write_rows(output["raw_positives"], positives, header=("drug", "disease"))
    _write_rows(output["positives"], active_positives, header=("drug", "disease"))
    _write_rows(output["hierarchy"], hierarchy, header=("child", "parent"))
    audit = output["audit_dir"]
    _write_rows(audit / "isolated_nodes.csv", isolates, header=("node", "name", "kind", "type"))
    _write_rows(audit / "excluded_positive_pairs.csv", excluded, header=("drug", "disease", "absent_endpoints"))
    _write_rows(audit / "do_mapping.csv", do_audit,
                header=("node", "source_doid", "canonical_doid", "resolution", "active_graph"))
    _write_rows(audit / "atc_assignments.csv", atc_audit, header=("node", "atc_code", "active_graph"))
    _write_json(audit / "hierarchy_paths.json", {node: [list(path) for path in sorted(values)]
                                                for node, values in sorted(expected_paths.items())})
    files = [output[key] for key in ("graph", "nodetypes", "positives", "raw_positives", "hierarchy")]
    files += [audit / name for name in ("isolated_nodes.csv", "excluded_positive_pairs.csv",
                                        "do_mapping.csv", "atc_assignments.csv", "hierarchy_paths.json")]
    manifest = dict(schema_version=1, sources=source_records,
                    molecular=dict(original_nodes=len(nodes), selected_nodes=sum(row["kind"] in node_type_map for row in node_rows),
                                   active_nodes=len(active), active_node_counts=dict(sorted(active_counts.items())),
                                   isolated_nodes=len(isolates), isolated_node_counts=dict(sorted(Counter(row[3] for row in isolates).items())),
                                   edge_rows=len(graph_rows), unique_undirected_pairs=simple_edges,
                                   additional_parallel_or_reverse_rows=len(graph_rows) - simple_edges,
                                   edge_types=[dict(id=i + 1, metaedge=name, rows=counts[name]) for i, name in enumerate(EDGE_TYPES)],
                                   row_order="official edge file order; no sorting or deduplication",
                                   weight=1, treatment_edges_in_graph=0, synthetic_self_loops=0),
                    positives=dict(raw=len(positives), active=len(active_positives), excluded=len(excluded),
                                   source_relation="CtD", excluded_pairs=[dict(drug=d, disease=i, absent_endpoints=a) for d, i, a in excluded]),
                    semantic=semantic,
                    ontology_policy="native DO is_a DAG plus exact alt_id aliases; no UMLS or MeSH mapping",
                    ic_policy="unchanged upstream occurrence counts over every valid root path",
                    source_limit="same-source Hetionet ontology snapshots; exact DREAMwalk paper snapshots were not released",
                    artifacts={str(path.relative_to(root)): dict(sha256=sha256(path), bytes=path.stat().st_size) for path in files})
    _write_json(output["manifest"], manifest)
    return manifest
