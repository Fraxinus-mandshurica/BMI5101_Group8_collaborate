"""Local licensed-UMLS MSI CUI to MeSH 2024 descriptor audit.

Reads MRCONSO in a single streaming pass. Never writes UMLS terms or RRF rows.
Requires local licensed data/umls/MRCONSO.RRF; no network access.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MRCONSO = ROOT / "data/umls/MRCONSO.RRF"
MESH = ROOT / "data/mesh/2024/d2024.bin"
MYDISEASE = ROOT / "results/msi_cui_mesh_mapping.csv"
OUT = ROOT / "results"
RRF_FIELDS = ("CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF")
RRF_INDEX = {name: i for i, name in enumerate(RRF_FIELDS)}


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_msi():
    path = ROOT / "data/msi_raw/2_indication_to_protein.tsv"
    with path.open(newline="", encoding="utf-8") as stream:
        data = list(csv.DictReader(stream, delimiter="\t"))
    cuis = {row["node_1"] for row in data}
    assert len(cuis) == 840, f"Expected 840 unique CUIs, got {len(cuis)}"
    assert all(re.fullmatch(r"C\d{7}", cui) for cui in cuis)
    assert all(row["node_1_type"] == "indication" for row in data)
    return sorted(cuis)


def read_mesh():
    """NLM ASCII MeSH: blank-line-separated descriptor records, repeated MN."""
    records = {}
    sha = hashlib.sha256()
    total = 0
    current_ui = None
    current_mn = []

    def finish():
        nonlocal total, current_ui, current_mn
        if current_ui is not None:
            total += 1
            assert re.fullmatch(r"D\d+", current_ui), current_ui
            assert current_ui not in records, f"Duplicate MeSH UI {current_ui}"
            assert len(current_mn) == len(set(current_mn)), current_ui
            records[current_ui] = tuple(current_mn)
        current_ui, current_mn = None, []

    with MESH.open("rb") as stream:
        for raw in stream:
            sha.update(raw)
            line = raw.rstrip(b"\r\n")
            if not line:
                finish()
            elif line.startswith(b"UI = "):
                current_ui = line[5:].decode("ascii")
            elif line.startswith(b"MN = "):
                current_mn.append(line[5:].decode("ascii"))
        finish()
    assert total == len(records)
    assert all(re.fullmatch(r"[A-Z]\d+(?:\.\d+)*", tree) for vals in records.values() for tree in vals)
    stats = dict(mesh2024_descriptor_records=total,
                 mesh2024_unique_descriptor_uis=len(records),
                 mesh2024_descriptors_with_tree=sum(bool(v) for v in records.values()),
                 mesh2024_descriptors_without_tree=sum(not v for v in records.values()),
                 mesh2024_unique_tree_numbers=len({t for v in records.values() for t in v}),
                 mesh2024_max_tree_positions=max(map(len, records.values())),
                 mesh2024_sha256=sha.hexdigest())
    return records, stats


def read_mrconso(cuis):
    """Official 18-column RRF, with its mandatory final delimiter."""
    sought = {c.encode("ascii") for c in cuis}
    atoms = defaultdict(list)
    sha = hashlib.sha256()
    line_count = 0
    matching_cui_rows = 0
    malformed = 0
    with MRCONSO.open("rb") as stream:
        for raw in stream:
            sha.update(raw)
            line_count += 1
            if not raw.startswith(b"C"):
                continue
            parts = raw.rstrip(b"\r\n").split(b"|")
            if len(parts) != 19 or parts[-1] != b"":
                malformed += 1
                continue
            if parts[0] not in sought:
                continue
            matching_cui_rows += 1
            if parts[RRF_INDEX["SAB"]] != b"MSH":
                continue
            atoms[parts[0].decode("ascii")].append({
                key: parts[RRF_INDEX[key]].decode("utf-8")
                for key in ("LAT", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI", "TTY", "SUPPRESS")
            })
    assert malformed == 0, f"Malformed MRCONSO rows: {malformed}"
    stats = dict(mrconso_sha256=sha.hexdigest(), mrconso_rows=line_count,
                 mrconso_rows_with_msi_cui=matching_cui_rows,
                 mrconso_msh_atoms_for_msi=sum(map(len, atoms.values())))
    return atoms, stats


def id_kind(value):
    if re.fullmatch(r"D\d+", value):
        return "D"
    if re.fullmatch(r"C\d+", value):
        return "C"
    return "OTHER"


def direct_status(atoms, desc, supp, unexpected):
    if not atoms:
        return "NO_MSH_ATOM"
    if unexpected:
        return "UNEXPECTED"
    if desc and supp:
        return "DESCRIPTOR_PLUS_SUPPLEMENTARY"
    if len(desc) > 1:
        return "MULTIPLE_DESCRIPTORS"
    if len(desc) == 1:
        return "UNIQUE_DESCRIPTOR"
    if supp:
        return "SUPPLEMENTARY_ONLY"
    return "MSH_ATOM_NO_USABLE_ID"


def hierarchy_status(descriptor, mesh):
    if descriptor not in mesh:
        return "DESCRIPTOR_NOT_FOUND_IN_MESH2024"
    if not mesh[descriptor]:
        return "VALID_DESCRIPTOR_NO_TREE"
    return "VALID_DESCRIPTOR_WITH_TREE"


def final_status(direct, desc, supp, mesh):
    if direct == "UNEXPECTED":
        return "UNEXPECTED"
    if len(desc) > 1:
        return "DIRECT_MULTIPLE_DESCRIPTOR"
    if len(desc) == 1:
        state = hierarchy_status(next(iter(desc)), mesh)
        return {"VALID_DESCRIPTOR_WITH_TREE": "DIRECT_UNIQUE_USABLE",
                "VALID_DESCRIPTOR_NO_TREE": "DIRECT_DESCRIPTOR_NO_TREE",
                "DESCRIPTOR_NOT_FOUND_IN_MESH2024": "DIRECT_DESCRIPTOR_NOT_IN_2024"}[state]
    if supp:
        return "SUPPLEMENTARY_UNRESOLVED"
    return "NO_MSH_MAPPING"


def build_mapping(cuis, atoms_by_cui, mesh):
    rows, candidates = [], []
    for cui in cuis:
        atoms = atoms_by_cui[cui]
        all_ids = {a["SDUI"] for a in atoms if a["SDUI"]}
        active_atoms = [a for a in atoms if a["LAT"] == "ENG" and a["SUPPRESS"] == "N"]
        active_ids = {a["SDUI"] for a in active_atoms if a["SDUI"]}
        desc = {x for x in all_ids if id_kind(x) == "D"}
        supp = {x for x in all_ids if id_kind(x) == "C"}
        other = {x for x in all_ids if id_kind(x) == "OTHER"}
        direct = direct_status(atoms, desc, supp, other)
        final = final_status(direct, desc, supp, mesh)
        unique = next(iter(desc)) if len(desc) == 1 else ""
        usable = final == "DIRECT_UNIQUE_USABLE"
        notes = []
        if all_ids != active_ids:
            notes.append("all_vs_active_english_ids_differ")
        if any(not a["SDUI"] for a in atoms):
            notes.append("some_msh_atoms_without_sdui")
        if supp and desc:
            notes.append("supplementary_ids_retained_separately;direct_descriptor_takes_priority")
        if final == "SUPPLEMENTARY_UNRESOLVED":
            notes.append("c2024_and_mrrel_unavailable")
        row = dict(msi_cui=cui, direct_mapping_status=direct, mapping_status=final,
                   mapping_route="MRCONSO_SAB_MSH_SDUI" if desc else ("MSH_SUPPLEMENTARY_UNRESOLVED" if supp else "NONE"),
                   mesh_descriptor_id=unique, mesh_supplementary_id=";".join(sorted(supp)),
                   all_descriptor_ids=";".join(sorted(desc)), all_supplementary_ids=";".join(sorted(supp)),
                   n_msh_atoms=len(atoms), n_active_english_atoms=len(active_atoms),
                   n_distinct_sdui=len(all_ids), n_distinct_descriptor_ids=len(desc),
                   n_distinct_supplementary_ids=len(supp), n_unexpected_ids=len(other),
                   descriptor_in_mesh2024=(unique in mesh) if unique else "",
                   mesh_tree_position_count=len(mesh.get(unique, ())) if unique else 0,
                   mesh_tree_numbers=";".join(mesh.get(unique, ())) if unique else "",
                   usable_for_primary_hierarchy=usable, notes=";".join(notes))
        rows.append(row)
        for value in sorted(all_ids):
            kind = id_kind(value)
            tree = mesh.get(value, ()) if kind == "D" else ()
            candidates.append(dict(msi_cui=cui, mesh_id=value, id_kind=kind,
                                   n_msh_atoms_for_id=sum(a["SDUI"] == value for a in atoms),
                                   n_active_english_atoms_for_id=sum(a["SDUI"] == value for a in active_atoms),
                                   descriptor_validation=hierarchy_status(value, mesh) if kind == "D" else "NOT_DESCRIPTOR",
                                   mesh_tree_position_count=len(tree), mesh_tree_numbers=";".join(tree),
                                   source="MRCONSO.SAB=MSH;SDUI"))
    return rows, candidates


def read_mydisease():
    grouped = defaultdict(set)
    with MYDISEASE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["mesh_id"]:
                grouped[row["msi_cui"]].add(row["mesh_id"])
    return grouped


def compare(rows, mydisease, mesh):
    comparison = []
    for row in rows:
        cui = row["msi_cui"]
        umls_all = set(filter(None, row["all_descriptor_ids"].split(";")))
        my_all = mydisease[cui]
        my_desc = {x for x in my_all if id_kind(x) == "D"}
        umls_unique = len(umls_all) == 1
        my_unique = len(my_all) == 1
        if len(umls_all) > 1 or len(my_all) > 1:
            category = "AMBIGUOUS_COMPARISON"
        elif umls_all and my_desc:
            category = "SAME_DESCRIPTOR" if umls_all == my_desc else "DIFFERENT_DESCRIPTOR"
        elif umls_all:
            category = "UMLS_ONLY"
        elif my_desc:
            category = "MYDISEASE_ONLY"
        else:
            category = "BOTH_UNMAPPED"
        comparison.append(dict(msi_cui=cui, comparison_status=category,
                               umls_mapped=bool(umls_all or row["all_supplementary_ids"]),
                               umls_unique_descriptor=umls_unique,
                               umls_ambiguous=len(umls_all) > 1,
                               umls_hierarchy_usable=row["usable_for_primary_hierarchy"],
                               umls_descriptor_ids=";".join(sorted(umls_all)),
                               mydisease_mapped=bool(my_all), mydisease_unique=my_unique,
                               mydisease_ambiguous=len(my_all) > 1,
                               mydisease_hierarchy_usable=my_unique and next(iter(my_all), "") in mesh and bool(mesh.get(next(iter(my_all), ""))),
                               mydisease_mesh_ids=";".join(sorted(my_all)),
                               notes="descriptor_comparison_only;2024_tree_for_both"))
    return comparison


def validate(rows, candidates, summary, mesh, cuis, comparison):
    assert len(rows) == 840 and len({r["msi_cui"] for r in rows}) == 840
    assert {r["msi_cui"] for r in rows} == set(cuis)
    assert len(comparison) == 840
    assert all(r["mesh_descriptor_id"].startswith("D") and r["mesh_descriptor_id"] in mesh and len(mesh[r["mesh_descriptor_id"]]) > 0
               for r in rows if r["usable_for_primary_hierarchy"])
    assert all(r["mesh_tree_numbers"].split(";") == list(mesh[r["mesh_descriptor_id"]])
               for r in rows if r["usable_for_primary_hierarchy"])
    assert all(not r["mesh_id"].startswith("C") or r["descriptor_validation"] == "NOT_DESCRIPTOR" for r in candidates)
    assert summary["final_unique_usable_mapping"] + summary["final_ambiguous_mapping"] + summary["final_unusable_or_unmapped"] == 840
    assert sum(r["mapping_status"] == "DIRECT_UNIQUE_USABLE" for r in rows) == summary["final_unique_usable_mapping"]
    assert sum(r["n_msh_atoms"] for r in rows) == summary["mrconso_msh_atoms_for_msi"]


def main():
    if not MRCONSO.is_file():
        raise SystemExit("Missing data/umls/MRCONSO.RRF")
    cuis = read_msi()
    mesh, mesh_stats = read_mesh()
    atoms, conso_stats = read_mrconso(cuis)
    rows, candidates = build_mapping(cuis, atoms, mesh)
    mydisease = read_mydisease()
    comparison = compare(rows, mydisease, mesh)
    states = Counter(r["mapping_status"] for r in rows)
    direct = Counter(r["direct_mapping_status"] for r in rows)
    unique_direct_ids = {r["mesh_id"] for r in candidates if r["id_kind"] == "D"}
    missing = {d for d in unique_direct_ids if d not in mesh}
    no_tree = {d for d in unique_direct_ids if d in mesh and not mesh[d]}
    comp = Counter(r["comparison_status"] for r in comparison)
    summary = dict(total_msi_diseases=len(cuis), msh_atom_present=sum(r["n_msh_atoms"] > 0 for r in rows),
                   direct_unique_descriptor=sum(r["n_distinct_descriptor_ids"] == 1 for r in rows),
                   direct_multiple_descriptors=sum(r["n_distinct_descriptor_ids"] > 1 for r in rows),
                   supplementary_only=direct["SUPPLEMENTARY_ONLY"],
                   descriptor_plus_supplementary=direct["DESCRIPTOR_PLUS_SUPPLEMENTARY"],
                   msh_atom_no_usable_id=direct["MSH_ATOM_NO_USABLE_ID"], no_msh_atom=direct["NO_MSH_ATOM"],
                   unexpected_direct=direct["UNEXPECTED"],
                   supplementary_resolved_unique=0, supplementary_resolved_multiple=0,
                   supplementary_unresolved=states["SUPPLEMENTARY_UNRESOLVED"],
                   valid_descriptor_in_mesh2024=len(unique_direct_ids & mesh.keys()),
                   valid_descriptor_with_tree=sum(d in mesh and bool(mesh[d]) for d in unique_direct_ids),
                   distinct_umls_descriptor_ids=len(unique_direct_ids),
                   distinct_umls_descriptor_ids_absent_from_mesh2024=len(missing),
                   distinct_umls_descriptor_ids_without_tree=len(no_tree),
                   cui_with_direct_descriptor_absent_from_mesh2024=sum(any(d in missing for d in filter(None, r["all_descriptor_ids"].split(";"))) for r in rows),
                   cui_with_direct_descriptor_without_tree=sum(any(d in no_tree for d in filter(None, r["all_descriptor_ids"].split(";"))) for r in rows),
                   final_unique_usable_mapping=states["DIRECT_UNIQUE_USABLE"],
                   final_ambiguous_mapping=states["DIRECT_MULTIPLE_DESCRIPTOR"] + states["SUPPLEMENTARY_RESOLVED_MULTIPLE"],
                   final_unusable_or_unmapped=len(cuis)-states["DIRECT_UNIQUE_USABLE"]-states["DIRECT_MULTIPLE_DESCRIPTOR"]-states["SUPPLEMENTARY_RESOLVED_MULTIPLE"],
                   final_usable_hierarchy_coverage_percent=round(100*states["DIRECT_UNIQUE_USABLE"]/len(cuis), 2),
                   all_vs_active_english_sdui_disagreement=sum("all_vs_active_english_ids_differ" in r["notes"] for r in rows),
                   mydisease_mapped=sum(r["mydisease_mapped"] for r in comparison),
                   mydisease_unique=sum(r["mydisease_unique"] for r in comparison),
                   mydisease_ambiguous=sum(r["mydisease_ambiguous"] for r in comparison),
                   mydisease_2024_hierarchy_usable=sum(r["mydisease_hierarchy_usable"] for r in comparison),
                   comparison_same_descriptor=comp["SAME_DESCRIPTOR"], comparison_different_descriptor=comp["DIFFERENT_DESCRIPTOR"],
                   comparison_umls_only=comp["UMLS_ONLY"], comparison_mydisease_only=comp["MYDISEASE_ONLY"],
                   comparison_both_unmapped=comp["BOTH_UNMAPPED"], comparison_ambiguous=comp["AMBIGUOUS_COMPARISON"],
                   umls_release="UNKNOWN_MRSAB_OR_RELEASE_METADATA_NOT_AVAILABLE",
                   umls_msh_source_version="UNKNOWN_MRSAB_NOT_AVAILABLE", mesh_descriptor_version="2024",
                   supplementary_resolution="UNAVAILABLE_C2024_AND_MRREL_NOT_PRESENT",
                   **mesh_stats, **conso_stats)
    validate(rows, candidates, summary, mesh, cuis, comparison)
    ambiguous = [r for r in rows if r["mapping_status"] != "DIRECT_UNIQUE_USABLE"]
    write_csv(OUT / "msi_cui_mesh_umls_mapping.csv", rows, list(rows[0]))
    write_csv(OUT / "msi_cui_mesh_umls_candidates.csv", candidates, list(candidates[0]))
    write_csv(OUT / "msi_cui_mesh_umls_mapping_summary.csv", [summary], list(summary))
    write_csv(OUT / "msi_cui_mesh_umls_ambiguities.csv", ambiguous, list(rows[0]))
    write_csv(OUT / "msi_cui_mesh_umls_vs_mydisease.csv", comparison, list(comparison[0]))
    print(json.dumps(summary, indent=2))
    print("FINAL_STATUS", json.dumps(states, indent=2))
    print("DIRECT_STATUS", json.dumps(direct, indent=2))


if __name__ == "__main__":
    main()
