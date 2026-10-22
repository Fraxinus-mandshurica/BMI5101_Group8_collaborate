"""Independent MeSH2021 hierarchy with the category layer shown by author demo.

The frozen historical MeSH2021 input module supplies only pure parsing/I/O
helpers. Its paths and artifacts are never changed. This variant inserts one
namespaced letter category per used MeSH tree (for example C_dis and A_dis).
ATC A/C/etc therefore retain their original drug-only meaning. No preparation,
similarity computation or model execution occurs on import.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import tempfile

import msi_mesh2021_inputs as historical


ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "data/mesh/2021/d2021.bin"
HIERARCHY_PATH = ROOT / "data/processed/msi_mesh2021_demo_roots_hierarchy.csv"
SIMILARITY_PATH = ROOT / "data/processed/msi_mesh2021_demo_roots_similarity.tsv"
AUDIT_DIR = ROOT / "results/msi_mesh2021_demo_roots_inputs"
GRAPH_PATH = ROOT / "data/processed/msi_graph.tsv"
NODETYPES_PATH = ROOT / "data/processed/msi_nodetypes.tsv"
MAPPING_PATH = ROOT / "results/msi_cui_mesh_umls_mapping.csv"
ATC_PATH = ROOT / "results/msi_atc_hierarchy_edges.csv"
UPSTREAM = ROOT / "vendor/DREAMwalk_upstream"
EXPECTED_MESH_SHA256 = historical.EXPECTED_MESH_SHA256
CUTOFF = 0.4
UPSTREAM_COMMIT = historical.UPSTREAM_COMMIT

# These helpers do not read or mutate the historical module's output paths.
sha256 = historical.sha256
read_mesh = historical.read_mesh
_rows = historical._rows
_atomic_text = historical._atomic_text
_write_csv = historical._write_csv
_mapping_for_year = historical._mapping_for_year


def _file_identity(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required MeSH2021 category-layer input is missing: {path}")
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        name = str(path)
    return dict(path=name, sha256=sha256(path), bytes=path.stat().st_size)


def input_identity():
    """Freeze the sources used by this new input variant without rebuilding."""
    mesh = _file_identity(MESH_PATH)
    if mesh["sha256"] != EXPECTED_MESH_SHA256:
        raise ValueError("d2021.bin differs from the accepted user-provided file")
    return dict(
        schema=1, mesh_year=2021, cutoff=CUTOFF, directed=False, weighted=True,
        upstream_commit=UPSTREAM_COMMIT,
        hierarchy_strategy="descriptor_UI_plus_MeSH_tree_position_and_demo_letter_categories",
        category_policy="every_used_first_letter_plus_dis_suffix_before_disease_root",
        mapping_policy="same_direct_unique_descriptor_policy_revalidated_against_2021",
        sources=dict(
            mesh=mesh, direct_umls_mapping=_file_identity(MAPPING_PATH),
            unchanged_atc_hierarchy=_file_identity(ATC_PATH),
            molecular_graph=_file_identity(GRAPH_PATH),
            nodetypes=_file_identity(NODETYPES_PATH),
            preparation_source=_file_identity(Path(__file__)),
            shared_pure_helpers_source=_file_identity(Path(historical.__file__)),
            upstream_demo_hierarchy=_file_identity(UPSTREAM / "demo/demo_hierarchy.csv"),
            upstream_similarity_source=_file_identity(UPSTREAM / "DREAMwalk/generate_similarity_net.py"),
            upstream_utils_source=_file_identity(UPSTREAM / "DREAMwalk/utils.py")))


def _manifest_path(stage):
    return AUDIT_DIR / (stage + "_manifest.json")


def _reuse(stage, identity, paths):
    path = _manifest_path(stage)
    if not path.exists():
        present = [str(value) for value in paths if Path(value).exists()]
        if present:
            raise RuntimeError("Unmanifested category-layer outputs already exist: " + ", ".join(present))
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["input_identity"] != identity:
        raise RuntimeError(f"{stage} input/source fingerprint changed; preserve this experiment")
    for output in manifest["artifacts"].values():
        target = ROOT / output["path"]
        if not target.is_file() or sha256(target) != output["sha256"]:
            raise RuntimeError(f"Saved {stage} artifact is missing or changed: {target}")
    print(f"Reusing completed MeSH2021 category-layer {stage}: {path}", flush=True)
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


def _disease_paths(mapped, mesh):
    """Preserve each real MeSH path and add its one-letter category ancestor."""
    edges, expected, positions = historical._disease_paths(mapped, mesh)
    revised_edges, revised_expected = set(), defaultdict(set)
    for cui, paths in expected.items():
        for path in paths:
            top_position = positions[path[-2]][1]
            # The namespace is required: bare A/C/etc are already ATC nodes.
            category = top_position[0] + "_dis"
            revised = tuple(path[:-1]) + (category, "disease")
            revised_expected[cui].add(revised)
            revised_edges.update(zip(revised, revised[1:]))
    return revised_edges, revised_expected, positions


def _upstream_similarity_module():
    # Historical helper only selects the unchanged vendored upstream module.
    return historical._upstream_similarity_module()


def build_hierarchy():
    """Build the separate category-complete MeSH2021 hierarchy and audits."""
    identity = input_identity()
    paths = dict(hierarchy=HIERARCHY_PATH,
                 mapping_audit=AUDIT_DIR / "mapping_audit.csv",
                 descriptor_candidates=AUDIT_DIR / "descriptor_candidates.csv",
                 position_nodes=AUDIT_DIR / "mesh_position_nodes.csv",
                 categories=AUDIT_DIR / "mesh_category_nodes.csv",
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
    disease_nodes = {node for edge in disease_edges for node in edge}
    drug_nodes = {node for edge in drug_edges for node in edge}
    if disease_nodes & drug_nodes:
        raise ValueError("ATC and MeSH hierarchies share node IDs: " +
                         ", ".join(sorted(disease_nodes & drug_nodes)))
    combined = drug_edges | disease_edges
    if any(child == parent for child, parent in combined):
        raise ValueError("Hierarchy contains a self edge")
    import pandas as pd
    tree = _upstream_similarity_module()._generate_tree(
        pd.DataFrame(sorted(combined), columns=["child", "parent"]), sorted(drugs | diseases))
    if set(tree) - {"drug", "disease"}:
        raise ValueError("Hierarchy contains an unsupported terminal root")
    actual = {cui: {tuple(path) for path in rows}
              for cui, rows in tree.get("disease", {}).items()}
    if actual != dict(expected):
        raise ValueError("Upstream expansion changed or recombined MeSH2021 paths")
    covered_drugs = set(tree.get("drug", {})) & drugs
    categories = sorted(child for child, parent in disease_edges if parent == "disease")
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
    _write_csv(paths["categories"], [dict(hierarchy_node=node, mesh_tree_category=node[0],
               parent="disease") for node in categories],
               ["hierarchy_node", "mesh_tree_category", "parent"])
    summary = dict(mesh_descriptor_records=len(mesh),
                   mesh_descriptors_with_tree=sum(bool(trees) for trees in mesh.values()),
                   total_graph_diseases=len(diseases), hierarchy_covered_diseases=len(mapped),
                   uncovered_diseases=len(diseases) - len(mapped),
                   total_graph_drugs=len(drugs), hierarchy_covered_drugs=len(covered_drugs),
                   uncovered_drugs=len(drugs) - len(covered_drugs),
                   drug_hierarchy_edges=len(drug_edges), disease_hierarchy_edges=len(disease_edges),
                   total_hierarchy_edges=len(combined), mesh_position_nodes=len(positions),
                   disease_hierarchy_nodes=len(disease_nodes),
                   unique_mesh_descriptor_uis=len({descriptor for descriptor, _ in positions.values()}),
                   max_disease_path_nodes=max((len(path) for paths in expected.values()
                                               for path in paths), default=0),
                   mesh_category_nodes=len(categories), mesh_category_ids=categories,
                   disease_hierarchy_paths=sum(map(len, expected.values())),
                   drug_hierarchy_paths=sum(map(len, tree.get("drug", {}).values())),
                   mapping_status_counts=dict(Counter(row["mapping_status"] for row in audit)),
                   molecular_graph_rebuilt=False, atc_hierarchy_changed=False,
                   umls_rescanned=False)
    csv_summary = dict(summary)
    for key in ("mapping_status_counts", "mesh_category_ids"):
        csv_summary[key] = json.dumps(summary[key], sort_keys=True)
    _write_csv(paths["summary"], [csv_summary], list(csv_summary))
    return _save_manifest("hierarchy", identity, paths, summary)


def build_similarity():
    """Use the original similarity generator, with the corrected input only."""
    identity = input_identity()
    if _reuse("hierarchy", identity, [HIERARCHY_PATH]) is None:
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
    """Validate both new checkpoints and expose their compact identities."""
    identity = input_identity()
    if _reuse("hierarchy", identity, [HIERARCHY_PATH]) is None:
        raise FileNotFoundError("Category-complete MeSH2021 hierarchy has not been built")
    sim_identity = dict(identity, hierarchy_sha256=sha256(HIERARCHY_PATH),
                        hierarchy_manifest_sha256=sha256(_manifest_path("hierarchy")))
    if _reuse("similarity", sim_identity, [SIMILARITY_PATH]) is None:
        raise FileNotFoundError("Category-complete MeSH2021 similarity has not been built")
    return dict(hierarchy=_file_identity(HIERARCHY_PATH),
                similarity=_file_identity(SIMILARITY_PATH),
                hierarchy_manifest=_file_identity(_manifest_path("hierarchy")),
                similarity_manifest=_file_identity(_manifest_path("similarity")))
