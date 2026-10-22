"""MeSH2024 MSI inputs with the full letter-category layer in author demo.

Independent artifacts for the final corrected-code experiment. The existing
MeSH2024 inputs and all MeSH2021 inputs remain untouched. Explicitly call the
two build functions from the notebook; importing this provider does no work.
"""
from __future__ import annotations

from pathlib import Path

from mesh_hierarchy_variant import (
    MeshHierarchyVariant, disease_paths as _disease_paths, mapping_for_year,
    read_mesh, sha256,
)


ROOT = Path(__file__).resolve().parents[1]
MESH_PATH = ROOT / "data/mesh/2024/d2024.bin"
HIERARCHY_PATH = ROOT / "data/processed/msi_mesh2024_demo_roots_hierarchy.csv"
SIMILARITY_PATH = ROOT / "data/processed/msi_mesh2024_demo_roots_similarity.tsv"
AUDIT_DIR = ROOT / "results/msi_mesh2024_demo_roots_inputs"
GRAPH_PATH = ROOT / "data/processed/msi_graph.tsv"
NODETYPES_PATH = ROOT / "data/processed/msi_nodetypes.tsv"
MAPPING_PATH = ROOT / "results/msi_cui_mesh_umls_mapping.csv"
ATC_PATH = ROOT / "results/msi_atc_hierarchy_edges.csv"
UPSTREAM = ROOT / "vendor/DREAMwalk_upstream"
EXPECTED_MESH_SHA256 = "c96077dcfda566b010af0444f9f928f85e09c928fa88f17bcb7c0a07759db248"
CUTOFF = 0.4
UPSTREAM_COMMIT = "2f93723c3fbfa899d0f23c02e3459de8bfac0581"


def _variant():
    return MeshHierarchyVariant(
        root=ROOT, mesh_year=2024, mesh_path=MESH_PATH,
        expected_mesh_sha256=EXPECTED_MESH_SHA256, hierarchy_path=HIERARCHY_PATH,
        similarity_path=SIMILARITY_PATH, audit_dir=AUDIT_DIR, graph_path=GRAPH_PATH,
        nodetypes_path=NODETYPES_PATH, mapping_path=MAPPING_PATH, atc_path=ATC_PATH,
        upstream=UPSTREAM, provider_source=Path(__file__),
        upstream_commit=UPSTREAM_COMMIT, cutoff=CUTOFF)


def _mapping_for_year(rows, mesh, diseases):
    return mapping_for_year(rows, mesh, diseases, 2024)


def input_identity():
    return _variant().input_identity()


def build_hierarchy():
    return _variant().build_hierarchy()


def build_similarity():
    return _variant().build_similarity()


def artifact_identity():
    return _variant().artifact_identity()
