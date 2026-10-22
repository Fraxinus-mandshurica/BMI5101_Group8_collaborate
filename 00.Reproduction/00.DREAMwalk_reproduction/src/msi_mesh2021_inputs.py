"""Separate MeSH2021 inputs for the original MSI DREAMwalk experiment.

Reuse the audited direct UMLS Descriptor candidates and the accepted ATC edges.
Revalidate the candidates against d2021.bin; never reuse the MeSH2024 usability
flag. No UMLS rescan, mapping inference, graph construction or training occurs
on import. The original upstream similarity implementation is called explicitly
by build_similarity(). Historical MeSH2024 artifacts are never overwritten.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "data/mesh/2021/d2021.bin"
HIERARCHY_PATH = ROOT / "data/processed/msi_mesh2021_semantic_hierarchy.csv"
SIMILARITY_PATH = ROOT / "data/processed/msi_mesh2021_semantic_similarity.tsv"
AUDIT_DIR = ROOT / "results/msi_mesh2021_inputs"
GRAPH_PATH = ROOT / "data/processed/msi_graph.tsv"
NODETYPES_PATH = ROOT / "data/processed/msi_nodetypes.tsv"
MAPPING_PATH = ROOT / "results/msi_cui_mesh_umls_mapping.csv"
ATC_PATH = ROOT / "results/msi_atc_hierarchy_edges.csv"
UPSTREAM = ROOT / "vendor/DREAMwalk_upstream"
EXPECTED_MESH_SHA256 = "9c32bf0eeab933ee2e097e105ae6b8ca57b254b4eadc7b50486e9c3faec3571c"
CUTOFF = 0.4
UPSTREAM_COMMIT = "2f93723c3fbfa899d0f23c02e3459de8bfac0581"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def _file_identity(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required MeSH2021 input is missing: {path}")
    return dict(path=_name(path), sha256=sha256(path), bytes=path.stat().st_size)


def input_identity():
    """Fingerprint fixed sources without constructing any scientific artifact."""
    mesh = _file_identity(MESH_PATH)
    if mesh["sha256"] != EXPECTED_MESH_SHA256:
        raise ValueError("d2021.bin differs from the user-provided MeSH2021 file; "
                         "preserve this experiment and use a separate input identity")
    sources = dict(mesh=mesh, direct_umls_mapping=_file_identity(MAPPING_PATH),
                   unchanged_atc_hierarchy=_file_identity(ATC_PATH),
                   molecular_graph=_file_identity(GRAPH_PATH),
                   nodetypes=_file_identity(NODETYPES_PATH),
                   preparation_source=_file_identity(Path(__file__)),
                   upstream_similarity_source=_file_identity(
                       UPSTREAM / "DREAMwalk/generate_similarity_net.py"),
                   upstream_utils_source=_file_identity(UPSTREAM / "DREAMwalk/utils.py"))
    return dict(schema=1, mesh_year=2021, upstream_commit=UPSTREAM_COMMIT,
                cutoff=CUTOFF, directed=False, weighted=True,
                hierarchy_strategy="descriptor_UI_plus_MeSH_tree_position",
                mapping_policy="same_direct_unique_descriptor_policy_revalidated_against_2021",
                sources=sources)


def _rows(path, delimiter=","):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def _atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(value)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv(path, rows, columns):
    import io
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def _manifest_path(stage):
    return AUDIT_DIR / (stage + "_manifest.json")


def _reuse(stage, identity, paths):
    path = _manifest_path(stage)
    if not path.exists():
        present = [str(value) for value in paths if Path(value).exists()]
        if present:
            raise RuntimeError("Unmanifested MeSH2021 outputs already exist; preserve or "
                               "review them before rebuilding: " + ", ".join(present))
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["input_identity"] != identity:
        raise RuntimeError(f"{stage} input/source fingerprint changed; preserve the saved "
                           "MeSH2021 artifacts and use a separate experiment")
    for output in manifest["artifacts"].values():
        target = ROOT / output["path"]
        if not target.is_file() or sha256(target) != output["sha256"]:
            raise RuntimeError(f"Saved {stage} artifact is missing or changed: {target}")
    print(f"Reusing completed MeSH2021 {stage}: {path}", flush=True)
    return manifest


def _save_manifest(stage, identity, paths, summary):
    manifest = dict(schema=1, stage=stage,
                    created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    input_identity=identity,
                    artifacts={name: _file_identity(path) for name, path in paths.items()},
                    summary=summary)
    _atomic_text(_manifest_path(stage), json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(stage=stage, **summary), indent=2), flush=True)
    return manifest


def read_mesh(path):
    """Read native NLM ASCII Descriptor records, retaining every tree position."""
    records, current = {}, {"ui": None, "trees": []}

    def finish():
        ui, trees = current["ui"], current["trees"]
        if ui is not None:
            if not re.fullmatch(r"D\d+", ui) or ui in records:
                raise ValueError(f"Invalid or duplicate MeSH Descriptor UI: {ui}")
            if len(trees) != len(set(trees)):
                raise ValueError(f"Repeated tree position for {ui}")
            if any(not re.fullmatch(r"[A-Z]\d+(?:\.\d+)*", item) for item in trees):
                raise ValueError(f"Invalid MeSH Tree Number for {ui}")
            records[ui] = tuple(trees)
        elif trees:
            raise ValueError("MeSH tree record has no Descriptor UI")
        current.update(ui=None, trees=[])

    with Path(path).open("rb") as stream:
        for raw in stream:
            line = raw.rstrip(b"\r\n")
            if not line:
                finish()
            elif line.startswith(b"UI = "):
                if current["ui"] is not None:
                    raise ValueError("MeSH Descriptor records lack a blank separator")
                current["ui"] = line[5:].decode("ascii")
            elif line.startswith(b"MN = "):
                current["trees"].append(line[5:].decode("ascii"))
        finish()
    if not records:
        raise ValueError("No NLM Descriptor records were found")
    return records


def _mapping_for_year(rows, mesh, diseases):
    """Use the accepted unique-direct-Descriptor policy, with new year checks."""
    if len(rows) != len({row["msi_cui"] for row in rows}):
        raise ValueError("Duplicate CUI in the saved direct-UMLS mapping audit")
    if {row["msi_cui"] for row in rows} != diseases:
        raise ValueError("Direct-UMLS mapping CUI set differs from MSI disease nodes")
    mapped, audit, candidates = {}, [], []
    for row in sorted(rows, key=lambda value: value["msi_cui"]):
        cui = row["msi_cui"]
        descriptors = sorted(set(filter(None, row["all_descriptor_ids"].split(";"))))
        supplementary = row.get("all_supplementary_ids", "")
        if any(not re.fullmatch(r"D\d+", value) for value in descriptors):
            raise ValueError(f"Invalid direct Descriptor candidate for {cui}")
        if int(row["n_unexpected_ids"]) > 0:
            status = "UNEXPECTED"
        elif len(descriptors) > 1:
            status = "DIRECT_MULTIPLE_DESCRIPTOR"
        elif not descriptors:
            status = "SUPPLEMENTARY_UNRESOLVED" if supplementary else "NO_MSH_MAPPING"
        elif descriptors[0] not in mesh:
            status = "DIRECT_DESCRIPTOR_NOT_IN_2021"
        elif not mesh[descriptors[0]]:
            status = "DIRECT_DESCRIPTOR_NO_TREE"
        else:
            status = "DIRECT_UNIQUE_USABLE"
            mapped[cui] = descriptors[0]
        descriptor = descriptors[0] if len(descriptors) == 1 else ""
        audit.append(dict(msi_cui=cui, mapping_status=status,
                          all_descriptor_ids=";".join(descriptors),
                          all_supplementary_ids=supplementary,
                          n_unexpected_ids=int(row["n_unexpected_ids"]),
                          mesh_descriptor_id=descriptor,
                          mesh_tree_numbers=";".join(mesh.get(descriptor, ())),
                          usable_for_primary_hierarchy=cui in mapped))
        for value in descriptors:
            state = ("ABSENT_FROM_MESH2021" if value not in mesh else
                     "VALID_DESCRIPTOR_NO_TREE" if not mesh[value] else
                     "VALID_DESCRIPTOR_WITH_TREE")
            candidates.append(dict(msi_cui=cui, mesh_descriptor_id=value,
                                   descriptor_validation=state,
                                   mesh_tree_numbers=";".join(mesh.get(value, ()))))
    return mapped, audit, candidates


def _disease_paths(mapped, mesh):
    position_to_descriptor = {}
    for descriptor, positions in mesh.items():
        for position in positions:
            if position in position_to_descriptor:
                raise ValueError(f"More than one Descriptor owns MeSH tree position {position}")
            position_to_descriptor[position] = descriptor
    edges, expected, positions_used = set(), defaultdict(set), {}
    for cui, descriptor in sorted(mapped.items()):
        for tree_number in mesh[descriptor]:
            chain, cursor = [cui, descriptor], tree_number
            while True:
                if cursor not in position_to_descriptor:
                    raise ValueError(f"Missing MeSH Descriptor at ancestor position {cursor}")
                ancestor = position_to_descriptor[cursor]
                node = ancestor + "@" + cursor
                chain.append(node)
                positions_used[node] = (ancestor, cursor)
                if "." not in cursor:
                    break
                cursor = cursor.rsplit(".", 1)[0]
            chain.append("disease")
            edges.update(zip(chain, chain[1:]))
            expected[cui].add(tuple(chain))
    return edges, expected, positions_used


def _upstream_similarity_module():
    sys.path.insert(0, str(UPSTREAM))
    from DREAMwalk import generate_similarity_net
    return generate_similarity_net


def build_hierarchy():
    """Build only the independent MeSH2021 hierarchy and mapping audit files."""
    identity = input_identity()
    paths = dict(hierarchy=HIERARCHY_PATH,
                 mapping_audit=AUDIT_DIR / "mapping_audit.csv",
                 descriptor_candidates=AUDIT_DIR / "descriptor_candidates.csv",
                 position_nodes=AUDIT_DIR / "mesh_position_nodes.csv",
                 summary=AUDIT_DIR / "hierarchy_summary.csv")
    cached = _reuse("hierarchy", identity, paths.values())
    if cached is not None:
        return cached
    mesh = read_mesh(MESH_PATH)
    types = {row["node"]: row["type"] for row in _rows(NODETYPES_PATH, "\t")}
    diseases = {node for node, kind in types.items() if kind == "disease"}
    drugs = {node for node, kind in types.items() if kind == "drug"}
    mapped, audit, candidates = _mapping_for_year(_rows(MAPPING_PATH), mesh, diseases)
    disease_edges, expected, positions = _disease_paths(mapped, mesh)
    drug_edges = {(row["child"], row["parent"]) for row in _rows(ATC_PATH)}
    combined = drug_edges | disease_edges
    if len(combined) != len(drug_edges) + len(disease_edges):
        raise ValueError("ATC and disease hierarchy unexpectedly share an edge")
    if any(child == parent for child, parent in combined):
        raise ValueError("Hierarchy contains a self edge")
    import pandas as pd
    upstream = _upstream_similarity_module()
    tree = upstream._generate_tree(pd.DataFrame(sorted(combined), columns=["child", "parent"]),
                                   list(drugs | diseases))
    if set(tree) - {"drug", "disease"}:
        raise ValueError("Hierarchy contains an unsupported terminal root")
    actual = {cui: {tuple(path) for path in rows}
              for cui, rows in tree.get("disease", {}).items()}
    if actual != dict(expected):
        raise ValueError("Upstream expansion changed or recombined MeSH2021 paths")
    covered_drugs = set(tree.get("drug", {})) & drugs
    _write_csv(HIERARCHY_PATH, [dict(child=c, parent=p) for c, p in sorted(combined)],
               ["child", "parent"])
    _write_csv(paths["mapping_audit"], audit, ["msi_cui", "mapping_status", "all_descriptor_ids",
               "all_supplementary_ids", "n_unexpected_ids", "mesh_descriptor_id",
               "mesh_tree_numbers", "usable_for_primary_hierarchy"])
    _write_csv(paths["descriptor_candidates"], candidates,
               ["msi_cui", "mesh_descriptor_id", "descriptor_validation", "mesh_tree_numbers"])
    _write_csv(paths["position_nodes"], [dict(hierarchy_node=node, mesh_descriptor_id=d,
               mesh_tree_number=t) for node, (d, t) in sorted(positions.items())],
               ["hierarchy_node", "mesh_descriptor_id", "mesh_tree_number"])
    summary = dict(mesh_descriptor_records=len(mesh),
                   mesh_descriptors_with_tree=sum(bool(tree) for tree in mesh.values()),
                   total_graph_diseases=len(diseases), hierarchy_covered_diseases=len(mapped),
                   uncovered_diseases=len(diseases) - len(mapped),
                   total_graph_drugs=len(drugs), hierarchy_covered_drugs=len(covered_drugs),
                   uncovered_drugs=len(drugs) - len(covered_drugs),
                   drug_hierarchy_edges=len(drug_edges), disease_hierarchy_edges=len(disease_edges),
                   total_hierarchy_edges=len(combined), mesh_position_nodes=len(positions),
                   disease_hierarchy_paths=sum(map(len, expected.values())),
                   drug_hierarchy_paths=sum(map(len, tree.get("drug", {}).values())),
                   mapping_status_counts=dict(Counter(row["mapping_status"] for row in audit)),
                   molecular_graph_rebuilt=False, atc_hierarchy_changed=False,
                   umls_rescanned=False)
    csv_summary = dict(summary)
    csv_summary["mapping_status_counts"] = json.dumps(summary["mapping_status_counts"], sort_keys=True)
    _write_csv(paths["summary"], [csv_summary], list(csv_summary))
    return _save_manifest("hierarchy", identity, paths, summary)


def build_similarity():
    """Call the original upstream similarity generator at cutoff 0.4 explicitly."""
    identity = input_identity()
    hierarchy_manifest = _reuse("hierarchy", identity, [HIERARCHY_PATH])
    if hierarchy_manifest is None:
        raise FileNotFoundError("Run build_hierarchy() before build_similarity()")
    identity = dict(identity, hierarchy_sha256=sha256(HIERARCHY_PATH),
                    hierarchy_manifest_sha256=sha256(_manifest_path("hierarchy")))
    paths = dict(similarity=SIMILARITY_PATH, summary=AUDIT_DIR / "similarity_summary.csv")
    cached = _reuse("similarity", identity, paths.values())
    if cached is not None:
        return cached
    SIMILARITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=SIMILARITY_PATH.name + ".",
                                            dir=str(SIMILARITY_PATH.parent))
    os.close(descriptor)
    try:
        _upstream_similarity_module().save_sim_graph(
            str(GRAPH_PATH), str(HIERARCHY_PATH), temporary,
            cutoff=CUTOFF, weighted=True, directed=False, net_delimiter="\t")
        types = {row["node"]: row["type"] for row in _rows(NODETYPES_PATH, "\t")}
        counts, nodes = Counter(), defaultdict(set)
        with open(temporary, newline="", encoding="utf-8") as stream:
            for index, row in enumerate(csv.reader(stream, delimiter="\t")):
                if len(row) != 5:
                    raise ValueError("Upstream semantic output must contain five columns")
                first, second, _, weight, edge_index = row
                kind = types.get(first)
                if kind not in {"drug", "disease"} or types.get(second) != kind:
                    raise ValueError("Semantic edge has a missing node or mixed node types")
                if first == second or not CUTOFF < float(weight) <= 1:
                    raise ValueError("Semantic edge violates upstream cutoff or self-edge policy")
                if int(edge_index) != index:
                    raise ValueError("Upstream semantic edge indices are not consecutive")
                counts[kind] += 1
                nodes[kind].update((first, second))
        os.replace(temporary, SIMILARITY_PATH)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    summary = dict(cutoff=CUTOFF, saved_edge_directionality="one_row_per_undirected_pair",
                   semantic_pairs_by_type=dict(counts),
                   nodes_with_semantic_edges_by_type={kind: len(values)
                                                     for kind, values in nodes.items()},
                   total_semantic_pairs=sum(counts.values()),
                   columns=["node_1", "node_2", "semantic_type", "weight", "edge_index"])
    summary_rows = [dict(semantic_type=kind, retained_undirected_pairs=counts[kind],
                         nodes_with_semantic_edges=len(nodes[kind]), cutoff=CUTOFF,
                         saved_edge_directionality="one_row_per_undirected_pair")
                    for kind in ("drug", "disease")]
    _write_csv(paths["summary"], summary_rows, ["semantic_type", "retained_undirected_pairs",
               "nodes_with_semantic_edges", "cutoff", "saved_edge_directionality"])
    return _save_manifest("similarity", identity, paths, summary)


def artifact_identity():
    """Validate saved hierarchy/similarity manifests and return compact identities."""
    identity = input_identity()
    hierarchy = _reuse("hierarchy", identity, [HIERARCHY_PATH])
    if hierarchy is None:
        raise FileNotFoundError("MeSH2021 hierarchy has not been built")
    sim_identity = dict(identity, hierarchy_sha256=sha256(HIERARCHY_PATH),
                        hierarchy_manifest_sha256=sha256(_manifest_path("hierarchy")))
    similarity = _reuse("similarity", sim_identity, [SIMILARITY_PATH])
    if similarity is None:
        raise FileNotFoundError("MeSH2021 semantic similarity has not been built")
    return dict(hierarchy=_file_identity(HIERARCHY_PATH),
                similarity=_file_identity(SIMILARITY_PATH),
                hierarchy_manifest=_file_identity(_manifest_path("hierarchy")),
                similarity_manifest=_file_identity(_manifest_path("similarity")))
