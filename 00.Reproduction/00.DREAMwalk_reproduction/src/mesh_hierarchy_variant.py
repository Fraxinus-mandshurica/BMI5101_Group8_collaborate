"""Configurable, separate MeSH input snapshots with author-demo categories.

This module borrows pure parsers from the frozen MeSH2021 implementation. It
never changes that module's globals, sources or output artifacts. Instances
receive every scientific path explicitly; importing this module does no work.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import tempfile

import msi_mesh2021_inputs as historical


sha256 = historical.sha256
read_mesh = historical.read_mesh
_rows = historical._rows
_atomic_text = historical._atomic_text
_write_csv = historical._write_csv


def mapping_for_year(rows, mesh, diseases, year):
    """Retain the accepted direct-unique policy and label the actual year."""
    mapped, audit, candidates = historical._mapping_for_year(rows, mesh, diseases)
    for row in audit:
        if row["mapping_status"] == "DIRECT_DESCRIPTOR_NOT_IN_2021":
            row["mapping_status"] = f"DIRECT_DESCRIPTOR_NOT_IN_{year}"
    for row in candidates:
        if row["descriptor_validation"] == "ABSENT_FROM_MESH2021":
            row["descriptor_validation"] = f"ABSENT_FROM_MESH{year}"
    return mapped, audit, candidates


def disease_paths(mapped, mesh):
    """Insert one category per true MeSH path, without merging ATC letters."""
    _, expected, positions = historical._disease_paths(mapped, mesh)
    edges, revised = set(), defaultdict(set)
    for cui, paths in expected.items():
        for path in paths:
            top_tree = positions[path[-2]][1]
            category = top_tree[0] + "_dis"
            chain = tuple(path[:-1]) + (category, "disease")
            revised[cui].add(chain)
            edges.update(zip(chain, chain[1:]))
    return edges, revised, positions


@dataclass(frozen=True)
class MeshHierarchyVariant:
    root: Path
    mesh_year: int
    mesh_path: Path
    expected_mesh_sha256: str
    hierarchy_path: Path
    similarity_path: Path
    audit_dir: Path
    graph_path: Path
    nodetypes_path: Path
    mapping_path: Path
    atc_path: Path
    upstream: Path
    provider_source: Path
    upstream_commit: str
    cutoff: float = 0.4

    def file_identity(self, path):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Required MeSH{self.mesh_year} input is missing: {path}")
        try:
            name = str(path.relative_to(self.root))
        except ValueError:
            name = str(path)
        return dict(path=name, sha256=sha256(path), bytes=path.stat().st_size)

    def input_identity(self):
        mesh = self.file_identity(self.mesh_path)
        if mesh["sha256"] != self.expected_mesh_sha256:
            raise ValueError(f"d{self.mesh_year}.bin differs from the accepted raw file")
        return dict(
            schema=1, mesh_year=self.mesh_year, cutoff=self.cutoff,
            directed=False, weighted=True, upstream_commit=self.upstream_commit,
            hierarchy_strategy="descriptor_UI_plus_MeSH_tree_position_and_demo_letter_categories",
            category_policy="every_used_first_letter_plus_dis_suffix_before_disease_root",
            mapping_policy=f"same_direct_unique_descriptor_policy_revalidated_against_{self.mesh_year}",
            sources=dict(
                mesh=mesh, direct_umls_mapping=self.file_identity(self.mapping_path),
                unchanged_atc_hierarchy=self.file_identity(self.atc_path),
                molecular_graph=self.file_identity(self.graph_path),
                nodetypes=self.file_identity(self.nodetypes_path),
                preparation_source=self.file_identity(self.provider_source),
                configurable_helpers_source=self.file_identity(Path(__file__)),
                shared_pure_helpers_source=self.file_identity(Path(historical.__file__)),
                upstream_demo_hierarchy=self.file_identity(self.upstream / "demo/demo_hierarchy.csv"),
                upstream_similarity_source=self.file_identity(
                    self.upstream / "DREAMwalk/generate_similarity_net.py"),
                upstream_utils_source=self.file_identity(self.upstream / "DREAMwalk/utils.py")))

    def manifest_path(self, stage):
        return self.audit_dir / (stage + "_manifest.json")

    def reuse(self, stage, identity, paths):
        path = self.manifest_path(stage)
        if not path.exists():
            present = [str(value) for value in paths if Path(value).exists()]
            if present:
                raise RuntimeError("Unmanifested MeSH variant outputs already exist: " + ", ".join(present))
            return None
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["input_identity"] != identity:
            raise RuntimeError(f"{stage} input/source fingerprint changed; preserve this experiment")
        for output in manifest["artifacts"].values():
            target = self.root / output["path"]
            if not target.is_file() or sha256(target) != output["sha256"]:
                raise RuntimeError(f"Saved {stage} artifact is missing or changed: {target}")
        print(f"Reusing completed MeSH{self.mesh_year} category-layer {stage}: {path}", flush=True)
        return manifest

    def save_manifest(self, stage, identity, paths, summary):
        manifest = dict(schema=1, stage=stage,
                        created_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        input_identity=identity,
                        artifacts={name: self.file_identity(path) for name, path in paths.items()},
                        summary=summary)
        _atomic_text(self.manifest_path(stage), json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(dict(stage=stage, **summary), indent=2), flush=True)
        return manifest

    def upstream_similarity_module(self):
        # Select the unchanged public implementation; no historical path is altered.
        import sys
        sys.path.insert(0, str(self.upstream))
        from DREAMwalk import generate_similarity_net
        return generate_similarity_net

    def build_hierarchy(self):
        identity = self.input_identity()
        paths = dict(hierarchy=self.hierarchy_path,
                     mapping_audit=self.audit_dir / "mapping_audit.csv",
                     descriptor_candidates=self.audit_dir / "descriptor_candidates.csv",
                     position_nodes=self.audit_dir / "mesh_position_nodes.csv",
                     categories=self.audit_dir / "mesh_category_nodes.csv",
                     namespace_audit=self.audit_dir / "hierarchy_namespace_audit.csv",
                     summary=self.audit_dir / "hierarchy_summary.csv")
        cached = self.reuse("hierarchy", identity, paths.values())
        if cached is not None:
            return cached
        mesh = read_mesh(self.mesh_path)
        types = {row["node"]: row["type"] for row in _rows(self.nodetypes_path, "\t")}
        diseases = {node for node, kind in types.items() if kind == "disease"}
        drugs = {node for node, kind in types.items() if kind == "drug"}
        mapped, audit, candidates = mapping_for_year(
            _rows(self.mapping_path), mesh, diseases, self.mesh_year)
        disease_edges, expected, positions = disease_paths(mapped, mesh)
        drug_edges = {(row["child"], row["parent"]) for row in _rows(self.atc_path)}
        disease_nodes = {node for edge in disease_edges for node in edge}
        drug_nodes = {node for edge in drug_edges for node in edge}
        if disease_nodes & drug_nodes:
            raise ValueError("ATC and MeSH hierarchies share node IDs: " +
                             ", ".join(sorted(disease_nodes & drug_nodes)))
        combined = drug_edges | disease_edges
        if any(child == parent for child, parent in combined):
            raise ValueError("Hierarchy contains a self edge")
        import pandas as pd
        tree = self.upstream_similarity_module()._generate_tree(
            pd.DataFrame(sorted(combined), columns=["child", "parent"]), sorted(drugs | diseases))
        if set(tree) - {"drug", "disease"}:
            raise ValueError("Hierarchy contains an unsupported terminal root")
        actual = {cui: {tuple(path) for path in rows}
                  for cui, rows in tree.get("disease", {}).items()}
        if actual != dict(expected):
            raise ValueError(f"Upstream expansion changed or recombined MeSH{self.mesh_year} paths")
        covered_drugs = set(tree.get("drug", {})) & drugs
        categories = sorted(child for child, parent in disease_edges if parent == "disease")
        _write_csv(self.hierarchy_path, [dict(child=c, parent=p) for c, p in sorted(combined)],
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
        namespace_rows = [dict(hierarchy_node=node, semantic_branch=branch)
                          for branch, nodes in (("drug", drug_nodes), ("disease", disease_nodes))
                          for node in sorted(nodes)]
        _write_csv(paths["namespace_audit"], namespace_rows, ["hierarchy_node", "semantic_branch"])
        summary = dict(mesh_year=self.mesh_year, mesh_descriptor_records=len(mesh),
                       mesh_descriptors_with_tree=sum(bool(trees) for trees in mesh.values()),
                       total_graph_diseases=len(diseases), hierarchy_covered_diseases=len(mapped),
                       uncovered_diseases=len(diseases) - len(mapped),
                       total_graph_drugs=len(drugs), hierarchy_covered_drugs=len(covered_drugs),
                       uncovered_drugs=len(drugs) - len(covered_drugs),
                       drug_hierarchy_edges=len(drug_edges), disease_hierarchy_edges=len(disease_edges),
                       total_hierarchy_edges=len(combined), mesh_position_nodes=len(positions),
                       disease_hierarchy_nodes=len(disease_nodes),
                       unique_mesh_descriptor_uis=len({d for d, _ in positions.values()}),
                       max_disease_path_nodes=max((len(path) for paths in expected.values()
                                                   for path in paths), default=0),
                       mesh_category_nodes=len(categories), mesh_category_ids=categories,
                       disease_hierarchy_paths=sum(map(len, expected.values())),
                       drug_hierarchy_paths=sum(map(len, tree.get("drug", {}).values())),
                       mapping_status_counts=dict(Counter(row["mapping_status"] for row in audit)),
                       shared_drug_disease_hierarchy_node_ids=0,
                       molecular_graph_rebuilt=False, atc_hierarchy_changed=False, umls_rescanned=False)
        csv_summary = dict(summary)
        for key in ("mapping_status_counts", "mesh_category_ids"):
            csv_summary[key] = json.dumps(summary[key], sort_keys=True)
        _write_csv(paths["summary"], [csv_summary], list(csv_summary))
        return self.save_manifest("hierarchy", identity, paths, summary)

    def build_similarity(self):
        identity = self.input_identity()
        if self.reuse("hierarchy", identity, [self.hierarchy_path]) is None:
            raise FileNotFoundError("Run build_hierarchy() before build_similarity()")
        identity = dict(identity, hierarchy_sha256=sha256(self.hierarchy_path),
                        hierarchy_manifest_sha256=sha256(self.manifest_path("hierarchy")))
        paths = dict(similarity=self.similarity_path, summary=self.audit_dir / "similarity_summary.csv")
        cached = self.reuse("similarity", identity, paths.values())
        if cached is not None:
            return cached
        self.similarity_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=self.similarity_path.name + ".",
                                                dir=str(self.similarity_path.parent))
        os.close(descriptor)
        try:
            self.upstream_similarity_module().save_sim_graph(
                str(self.graph_path), str(self.hierarchy_path), temporary,
                cutoff=self.cutoff, weighted=True, directed=False, net_delimiter="\t")
            types = {row["node"]: row["type"] for row in _rows(self.nodetypes_path, "\t")}
            counts, nodes = Counter(), defaultdict(set)
            with open(temporary, newline="", encoding="utf-8") as stream:
                for index, row in enumerate(csv.reader(stream, delimiter="\t")):
                    if len(row) != 5:
                        raise ValueError("Upstream semantic output must contain five columns")
                    first, second, _, weight, edge_index = row
                    kind = types.get(first)
                    if kind not in {"drug", "disease"} or types.get(second) != kind:
                        raise ValueError("Semantic edge has a missing node or mixed node types")
                    if first == second or not self.cutoff < float(weight) <= 1:
                        raise ValueError("Semantic edge violates upstream cutoff or self-edge policy")
                    if int(edge_index) != index:
                        raise ValueError("Upstream semantic edge indices are not consecutive")
                    counts[kind] += 1
                    nodes[kind].update((first, second))
            os.replace(temporary, self.similarity_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        summary = dict(cutoff=self.cutoff, saved_edge_directionality="one_row_per_undirected_pair",
                       semantic_pairs_by_type=dict(counts),
                       nodes_with_semantic_edges_by_type={kind: len(values)
                                                         for kind, values in nodes.items()},
                       total_semantic_pairs=sum(counts.values()),
                       columns=["node_1", "node_2", "semantic_type", "weight", "edge_index"])
        summary_rows = [dict(semantic_type=kind, retained_undirected_pairs=counts[kind],
                             nodes_with_semantic_edges=len(nodes[kind]), cutoff=self.cutoff,
                             saved_edge_directionality="one_row_per_undirected_pair")
                        for kind in ("drug", "disease")]
        _write_csv(paths["summary"], summary_rows, ["semantic_type", "retained_undirected_pairs",
                   "nodes_with_semantic_edges", "cutoff", "saved_edge_directionality"])
        return self.save_manifest("similarity", identity, paths, summary)

    def artifact_identity(self):
        identity = self.input_identity()
        if self.reuse("hierarchy", identity, [self.hierarchy_path]) is None:
            raise FileNotFoundError(f"Category-complete MeSH{self.mesh_year} hierarchy has not been built")
        sim_identity = dict(identity, hierarchy_sha256=sha256(self.hierarchy_path),
                            hierarchy_manifest_sha256=sha256(self.manifest_path("hierarchy")))
        if self.reuse("similarity", sim_identity, [self.similarity_path]) is None:
            raise FileNotFoundError(f"Category-complete MeSH{self.mesh_year} similarity has not been built")
        return dict(hierarchy=self.file_identity(self.hierarchy_path),
                    similarity=self.file_identity(self.similarity_path),
                    hierarchy_manifest=self.file_identity(self.manifest_path("hierarchy")),
                    similarity_manifest=self.file_identity(self.manifest_path("similarity")))
