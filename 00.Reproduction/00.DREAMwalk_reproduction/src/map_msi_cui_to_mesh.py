"""Audit exact public-source CUI/MeSH cross-references in MyDisease.info.

Run --validate before --full. Raw projected API replies are cached under data/.
The UMLS-licensed ``umls.mesh`` field is never requested or used.
"""
import argparse
import csv
import json
import re
import time
import gzip
import xml.etree.ElementTree as ET
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/mydisease_cache"
SCOPES = "mondo.xrefs.umls,mondo.xrefs.umls_cui,disease_ontology.xrefs.umls_cui"
FIELDS = "mondo.mondo,mondo.xrefs.umls,mondo.xrefs.umls_cui,mondo.xrefs.mesh,ctd.mesh,disease_ontology.xrefs.umls_cui,disease_ontology.xrefs.mesh"
VALIDATE = ["C0002395", "C0030567", "C0242422", "C0030569", "C0011849", "C0011854", "C0687720", "C0004096", "C0151740", "C2973725", "C0028840", "C0020545", "C0678222", "C0006142", "C1134719", "C0003873", "C0010674", "C0024141", "C0024143", "C0020179", "C0149931", "C0014544", "C1260899", "C3544321", "C0751967"]


def as_list(value):
    if value is None or value == {}:
        return []
    return value if isinstance(value, list) else [value]


def nested(obj, *path):
    for key in path:
        obj = obj.get(key, {}) if isinstance(obj, dict) else {}
    return obj


def cui_sources(record, cui):
    found = []
    for path in ("umls", "umls_cui"):
        if any(cui in as_list(nested(m, "xrefs", path)) for m in as_list(record.get("mondo"))):
            found.append("mondo.xrefs." + path)
    if any(cui in as_list(nested(d, "xrefs", "umls_cui")) for d in as_list(record.get("disease_ontology"))):
        found.append("disease_ontology.xrefs.umls_cui")
    return found


def query_batch(cuis, cache_name):
    CACHE.mkdir(exist_ok=True)
    dest = CACHE / cache_name
    if dest.exists():
        return json.loads(dest.read_text())
    body = urllib.parse.urlencode({"q": ",".join(cuis), "scopes": SCOPES, "fields": FIELDS, "returnall": "true"}).encode()
    req = urllib.request.Request("https://mydisease.info/v1/query", data=body, headers={"User-Agent": "DREAMwalk-MSI-public-mapping-audit/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                data = json.load(response)
            if not isinstance(data, list):
                raise ValueError(f"Unexpected batch response: {type(data)}")
            dest.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
            return data
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))


def records_for(item):
    if item.get("notfound"):
        return []
    # BioThings returnall mode nests all matching records under hits.
    return item.get("hits", []) if isinstance(item.get("hits"), list) else [item]


def source_mesh(record):
    return [
        ("MONDO", "mondo.xrefs.mesh", [v for m in as_list(record.get("mondo")) for v in as_list(nested(m, "xrefs", "mesh"))]),
        ("CTD", "ctd.mesh", [v for c in as_list(record.get("ctd")) for v in as_list(c.get("mesh"))]),
        ("DISEASE_ONTOLOGY", "disease_ontology.xrefs.mesh", [v for d in as_list(record.get("disease_ontology")) for v in as_list(nested(d, "xrefs", "mesh"))]),
    ]


def compact_inspect(cuis, data):
    by_cui = defaultdict(list)
    for item in data:
        by_cui[item.get("query")].extend(records_for(item))
    for cui in cuis:
        recs = by_cui[cui]
        print(cui, "records", len(recs))
        for rec in recs:
            match = cui_sources(rec, cui)
            mapping = {field: as_list(val) for _, field, val in source_mesh(rec) if val}
            print("  ", rec.get("_id"), "exact", match, "mesh", mapping)


def get_cuis():
    path = ROOT / "data/msi_raw/2_indication_to_protein.tsv"
    with path.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    cuis = sorted({row["node_1"] for row in rows})
    assert len(cuis) == 840 and all(re.fullmatch(r"C\d{7}", c) for c in cuis)
    return cuis


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--validate", action="store_true")
    argp.add_argument("--full", action="store_true")
    args = argp.parse_args()
    if args.validate == args.full:
        argp.error("choose exactly one of --validate or --full")
    all_cuis = get_cuis()
    cuis = [c for c in VALIDATE if c in all_cuis] if args.validate else all_cuis
    data = []
    for i in range(0, len(cuis), 40):
        batch = cuis[i:i+40]
        name = ("validation" if args.validate else "full") + f"_{i//40:03d}.json"
        result = query_batch(batch, name)
        data.extend(result)
        print(f"batch {i//40+1}: {len(batch)} requested, {len(result)} response entries", flush=True)
        if not args.validate:
            time.sleep(.25)
    if args.validate:
        compact_inspect(cuis, data)
        return
    # Mapping table construction follows validated response semantics.
    build_outputs(cuis, data)


def build_outputs(cuis, data):
    by_cui = defaultdict(list)
    for item in data:
        by_cui[item.get("query")].extend(records_for(item))
    missing_response = sorted(set(cuis) - set(by_cui) - {i.get("query") for i in data if i.get("notfound")})
    if missing_response:
        raise ValueError(f"API omitted {len(missing_response)} queries, e.g. {missing_response[:5]}")
    hierarchy = load_mesh_hierarchy()
    rows, details = [], {}
    for cui in cuis:
        hits = [(r, cui_sources(r, cui)) for r in by_cui[cui]]
        hits = [(r, f) for r, f in hits if f]
        evidence = []
        for rec, exact_fields in hits:
            for source, field, values in source_mesh(rec):
                # Each source field must describe the matching source object, not
                # an unrelated item inside a merged list.
                if source == "MONDO":
                    values = [v for m in as_list(rec.get("mondo")) if any(cui in as_list(nested(m, "xrefs", p)) for p in ("umls", "umls_cui")) for v in as_list(nested(m, "xrefs", "mesh"))]
                    relation = "direct_mondo_xrefs"
                elif source == "DISEASE_ONTOLOGY":
                    values = [v for d in as_list(rec.get("disease_ontology")) if cui in as_list(nested(d, "xrefs", "umls_cui")) for v in as_list(nested(d, "xrefs", "mesh"))]
                    relation = "direct_disease_ontology_xrefs"
                else:
                    relation = "same_aggregated_record_as_exact_cui"
                for mesh in sorted(set(values)):
                    if isinstance(mesh, str) and re.fullmatch(r"[DC]\d+", mesh):
                        evidence.append((rec.get("_id", ""), mesh, source, field, relation, ";".join(exact_fields)))
        evidence = sorted(set(evidence))
        mesh_ids = {e[1] for e in evidence}
        sources = {e[2] for e in evidence}
        record_ids = {e[0] for e in evidence}
        if not hits:
            status = "NO_QUERY_HIT"
        elif not mesh_ids:
            status = "UNMAPPED"
        elif len(mesh_ids) > 1:
            status = "CONFLICTING"
        elif len(record_ids) > 1:
            status = "MULTIPLE_CONCORDANT"
        elif len(sources) > 1:
            status = "UNIQUE_CONCORDANT"
        else:
            status = "UNIQUE_SINGLE_SOURCE"
        details[cui] = dict(status=status, mesh_ids=mesh_ids, sources=sources,
                            records=len(hits), evidence=evidence)
        if evidence:
            for record_id, mesh, source, field, relation, exact_fields in evidence:
                rows.append(dict(msi_cui=cui, query_hit=True, mydisease_record_id=record_id,
                                 mesh_id=mesh, mapping_source=source, mapping_field=field,
                                 mapping_relation=relation, mapping_status=status,
                                 mesh_in_hierarchy=mesh in hierarchy,
                                 notes="exact CUI: " + exact_fields))
        else:
            rows.append(dict(msi_cui=cui, query_hit=bool(hits), mydisease_record_id=";".join(sorted({r.get("_id", "") for r, _ in hits})),
                             mesh_id="", mapping_source="", mapping_field="", mapping_relation="",
                             mapping_status=status, mesh_in_hierarchy="", notes=";".join(sorted({f for _, ff in hits for f in ff}))))
    out = ROOT / "results"
    write_csv(out / "msi_cui_mesh_mapping.csv", rows)
    status_counts = Counter(d["status"] for d in details.values())
    mesh_all = {m for d in details.values() for m in d["mesh_ids"]}
    hit_count = sum(d["records"] > 0 for d in details.values())
    any_mesh = sum(bool(d["mesh_ids"]) for d in details.values())
    pairs = [({e[1] for e in d["evidence"] if e[2] == "MONDO"},
              {e[1] for e in d["evidence"] if e[2] == "CTD"}) for d in details.values()]
    mondo_ctd_agree = sum(bool(m and c and m == c) for m, c in pairs)
    mondo_ctd_disagree = sum(bool(m and c and m != c) for m, c in pairs)
    summary = dict(total_msi_diseases=len(cuis), query_hits=hit_count, query_hit_percent=round(100*hit_count/len(cuis), 2),
                   diseases_with_any_mesh=any_mesh, mesh_coverage_percent=round(100*any_mesh/len(cuis), 2),
                   unique_concordant=status_counts["UNIQUE_CONCORDANT"], unique_single_source=status_counts["UNIQUE_SINGLE_SOURCE"],
                   multiple_concordant=status_counts["MULTIPLE_CONCORDANT"], conflicting=status_counts["CONFLICTING"],
                   unmapped=status_counts["UNMAPPED"], no_query_hit=status_counts["NO_QUERY_HIT"],
                   exactly_one_candidate_mesh=sum(len(d["mesh_ids"]) == 1 for d in details.values()),
                   multiple_candidate_mesh=sum(len(d["mesh_ids"]) > 1 for d in details.values()),
                   no_candidate_mesh=sum(not d["mesh_ids"] for d in details.values()),
                   mondo_supported=sum("MONDO" in d["sources"] for d in details.values()),
                   ctd_supported=sum("CTD" in d["sources"] for d in details.values()),
                   disease_ontology_supported=sum("DISEASE_ONTOLOGY" in d["sources"] for d in details.values()),
                   mondo_ctd_agree=mondo_ctd_agree, mondo_ctd_disagree=mondo_ctd_disagree,
                   distinct_mesh_ids=len(mesh_all), mesh_ids_in_hierarchy=len(mesh_all & hierarchy),
                   mesh_ids_absent_from_hierarchy=len(mesh_all - hierarchy),
                   mesh_hierarchy_coverage_percent=round(100*len(mesh_all & hierarchy)/len(mesh_all), 2) if mesh_all else 0,
                   mapped_cuis_with_hierarchy_mesh=sum(bool(d["mesh_ids"] & hierarchy) for d in details.values()),
                   unambiguous_cuis_with_hierarchy_mesh=sum(len(d["mesh_ids"]) == 1 and bool(d["mesh_ids"] & hierarchy) for d in details.values()),
                   unambiguous_cuis_without_hierarchy_mesh=sum(len(d["mesh_ids"]) == 1 and not bool(d["mesh_ids"] & hierarchy) for d in details.values()),
                   mesh_hierarchy_version="2026", mydisease_build_version="20260901", api_access_date="2026-09-28")
    write_csv(out / "msi_cui_mesh_mapping_summary.csv", [summary])
    problems = [dict(msi_cui=c, mapping_status=d["status"], candidate_mesh_ids=";".join(sorted(d["mesh_ids"])), record_count=d["records"])
                for c, d in details.items() if d["status"] in ("CONFLICTING", "MULTIPLE_CONCORDANT")]
    write_csv(out / "msi_cui_mesh_mapping_ambiguities.csv", problems,
              ["msi_cui", "mapping_status", "candidate_mesh_ids", "record_count"])
    print(json.dumps(summary, indent=2))
    print("absent hierarchy", sorted(mesh_all - hierarchy))
    print("ambiguities", problems)


def write_csv(path, rows, columns=None):
    columns = columns or list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)


def load_mesh_hierarchy():
    path = ROOT / "data/mesh_cache_desc2026.gz"
    ids = set()
    with gzip.open(path, "rb") as stream:
        for _, elem in ET.iterparse(stream, events=("end",)):
            if elem.tag == "DescriptorRecord":
                ui = elem.findtext("DescriptorUI")
                if ui and elem.find("TreeNumberList/TreeNumber") is not None:
                    ids.add(ui)
                elem.clear()
    return ids


if __name__ == "__main__":
    main()
