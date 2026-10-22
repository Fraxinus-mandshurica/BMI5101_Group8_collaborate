"""Freeze public KEGG REST inputs and construct an auditable molecular graph.

The author did not publish the KEGG benchmark snapshot. This module reconstructs
the six relations in Supplementary Table 4 from current explicit KEGG records.
It performs data preparation only, never EM, walks, training, or evaluation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import ssl
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from kegg_semantics import AUDIT_COLUMNS, build_hierarchy


REST = "https://rest.kegg.jp/"
EDGE_TYPES = ("disease_gene", "drug_gene", "gene_pathway", "pathway_pathway",
              "disease_pathway", "drug_pathway")
REFERENCE_EDGE_COUNTS = (6319, 11860, 43226, 2129, 2573, 10274)
REFERENCE_NODE_COUNTS = dict(drug=6008, disease=1963, gene=14496, etc=461)
STATIC_SOURCES = (
    ("info_kegg.txt", "info/kegg"),
    ("list_drug.tsv", "list/drug"),
    ("list_disease.tsv", "list/disease"),
    ("list_hsa.tsv", "list/hsa"),
    ("list_pathway.tsv", "list/pathway"),
    ("list_pathway_hsa.tsv", "list/pathway/hsa"),
    ("link_hsa_disease.tsv", "link/hsa/disease"),
    ("link_hsa_drug.tsv", "link/hsa/drug"),
    ("link_pathway_hsa.tsv", "link/pathway/hsa"),
    ("link_drug_disease.tsv", "link/drug/disease"),
    ("br08303_atc.txt", "get/br:br08303"),
    ("br08403_icd11.txt", "get/br:br08403"),
)


def paths(root):
    root = Path(root).resolve()
    processed, audit = root / "data/processed", root / "results/kegg_inputs"
    return dict(graph=processed / "kegg_molecular.tsv",
                nodetypes=processed / "kegg_nodetypes.tsv",
                positives=processed / "kegg_positives.csv",
                raw_positives=processed / "kegg_positives_raw.csv",
                hierarchy=processed / "kegg_semantic_hierarchy.csv",
                manifest=audit / "manifest.json", audit_dir=audit,
                raw_dir=root / "data/kegg_raw")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utcnow():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _decode_text(data):
    """KEGG reference citations can contain historical ISO-8859-1 bytes.

    Native IDs and field names are ASCII. Preserve every source byte in the
    cache and decode display text with an explicit lossless fallback.
    """
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("iso-8859-1"), "iso-8859-1"


def _read_text(path):
    return _decode_text(Path(path).read_bytes())[0]


def _tls_context():
    # Moving the project can leave OpenSSL's compiled-in conda CA path stale.
    # Load the actual environment bundle while retaining certificate checks.
    candidates = (Path(sys.prefix) / "ssl/cert.pem", Path("/etc/ssl/cert.pem"))
    for candidate in candidates:
        if candidate.is_file():
            return ssl.create_default_context(cafile=str(candidate))
    return ssl.create_default_context()


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8"))


def _write_rows(path, rows, header=None, delimiter=","):
    import io
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter=delimiter, lineterminator="\n")
    if header is not None:
        writer.writerow(header)
    writer.writerows(rows)
    _atomic_bytes(path, stream.getvalue().encode("utf-8"))


class FrozenRestCache:
    """Persistent source hashes; resume missing downloads without refreshing data.

    All workers share one limiter (at most 2.5 starts/second). Each successful
    response and its metadata are written before the next notebook stage.
    """
    def __init__(self, raw_dir):
        self.root = Path(raw_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "sources.json"
        self.lock = threading.Lock()
        self.rate_lock = threading.Lock()
        self.next_request = 0.0
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text())
            if self.index.get("schema_version") != 1:
                raise ValueError("Unsupported KEGG source manifest")
        else:
            self.index = dict(schema_version=1, created_utc=_utcnow(), sources={},
                              policy="freeze first successful response; never auto-refresh cached source")

    def validate(self, name, url=None):
        record = self.index["sources"].get(name)
        if record is None:
            return False
        path = self.root / name
        if url is not None and record["url"] != url:
            raise ValueError("KEGG cache URL changed: " + name)
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError("Frozen KEGG source does not match its SHA256: " + name)
        return True

    def _wait_turn(self):
        with self.rate_lock:
            delay = self.next_request - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self.next_request = time.monotonic() + 0.4

    def fetch(self, name, endpoint):
        url = REST + endpoint
        with self.lock:
            if self.validate(name, url):
                return self.root / name
            if self.index.get("completed_utc"):
                raise ValueError("Frozen KEGG snapshot lacks required source: " + name)
        destination = self.root / name
        # An interrupted response can leave an unindexed file. Fetch it again;
        # it is not yet part of the frozen source manifest.
        last_error = None
        for attempt in range(5):
            self._wait_turn()
            try:
                request = Request(url, headers={"User-Agent": "DREAMwalk-reproduction/KEGG-inputs"})
                with urlopen(request, timeout=90, context=_tls_context()) as response:
                    data = response.read()
                if not data.strip() or data.lstrip().startswith((b"<!DOCTYPE", b"<html")):
                    raise ValueError("Empty or HTML response for KEGG source: " + url)
                if endpoint.startswith("get/") and not endpoint.startswith("get/br:"):
                    requested = set(endpoint[4:].split("+"))
                    actual = set(_flat_records(_decode_text(data)[0]))
                    # Entry values use hsa/map/H prefixes, without ds:/path:.
                    if requested != actual:
                        raise ValueError("Incomplete KEGG GET batch: %s; missing %s" % (url, sorted(requested - actual)))
                break
            except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
                last_error = error
                if isinstance(error, HTTPError) and error.code not in (408, 429, 500, 502, 503, 504):
                    raise
                if attempt == 4:
                    raise RuntimeError("Unable to retrieve KEGG source: " + url) from last_error
                time.sleep(min(2 ** attempt, 16))
        _atomic_bytes(destination, data)
        with self.lock:
            self.index["sources"][name] = dict(url=url, sha256=hashlib.sha256(data).hexdigest(),
                                               bytes=len(data), acquired_utc=_utcnow(),
                                               text_encoding=_decode_text(data)[1])
            _write_json(self.index_path, self.index)
        return destination

    def complete(self):
        with self.lock:
            for name in self.index["sources"]:
                self.validate(name)
            self.index.setdefault("completed_utc", _utcnow())
            self.index["kegg_release_info"] = _read_text(self.root / "info_kegg.txt")
            _write_json(self.index_path, self.index)


def _list_rows(text, kind):
    """Keep native database IDs; normalize REST's bare list identifiers."""
    prefixes = dict(drug="dr:", disease="ds:", gene="hsa:", etc="path:")
    patterns = dict(drug=r"D[0-9]{5}", disease=r"H[0-9]{5}",
                    gene=r"[0-9]+", etc=r"(?:map|hsa)[0-9]{5}")
    rows = []
    seen = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        identifier = fields[0].strip()
        bare = identifier.split(":", 1)[-1]
        if re.fullmatch(patterns[kind], bare) is None:
            raise ValueError("Unexpected KEGG list identifier: " + identifier)
        if kind == "etc":
            bare = "map" + bare[-5:]
        native = prefixes[kind] + bare
        if native not in seen:
            rows.append(dict(id=native, name=fields[-1], type=kind))
            seen.add(native)
    return rows


def _flat_records(text):
    """Read KEGG top-level fixed-width fields, preserving continuations."""
    records, fields, key = {}, None, None
    for line in text.splitlines():
        if line.startswith("///"):
            if fields is not None:
                identifier = fields["ENTRY"][0].split()[0]
                if identifier in records:
                    raise ValueError("Duplicate KEGG flat-file entry: " + identifier)
                records[identifier] = fields
            fields, key = None, None
            continue
        tag, value = line[:12].strip(), line[12:].strip()
        if tag == "ENTRY":
            if fields is not None:
                raise ValueError("KEGG record missing /// terminator")
            fields, key = defaultdict(list), tag
        elif fields is None:
            if line.strip():
                raise ValueError("KEGG flat-file content outside ENTRY record")
            continue
        elif line[:12].strip() and not line.startswith(" "):
            key = tag
        # Subfields such as '  AUTHORS' belong to a separate ignored field.
        elif tag:
            key = "_subfield_" + tag
        if key:
            fields[key].append(value)
    if fields is not None:
        raise ValueError("KEGG record missing /// terminator")
    return records


def _batch_jobs(identifiers, family):
    identifiers = sorted(set(identifiers))
    jobs = []
    for start in range(0, len(identifiers), 10):
        batch = identifiers[start:start + 10]
        # Stable names include actual IDs, preventing accidental cache reuse
        # when a source list or chunk boundary differs.
        token = hashlib.sha256("+".join(batch).encode()).hexdigest()[:12]
        name = "cards/%s_%s_%s.txt" % (family, batch[0], token)
        jobs.append((name, "get/" + "+".join(batch)))
    return jobs


def download_sources(root):
    """Acquire one frozen current KEGG snapshot; subsequent calls verify/reuse it."""
    output = paths(root)
    cache = FrozenRestCache(output["raw_dir"])
    for name, endpoint in STATIC_SOURCES:
        cache.fetch(name, endpoint)
    raw = output["raw_dir"]
    disease_ids = [row["id"].split(":", 1)[1] for row in _list_rows(_read_text(raw / "list_disease.tsv"), "disease")]
    map_ids = [row["id"].split(":", 1)[1] for row in _list_rows(_read_text(raw / "list_pathway.tsv"), "etc")]
    hsa_ids = [line.split("\t", 1)[0].split(":", 1)[-1]
               for line in _read_text(raw / "list_pathway_hsa.tsv").splitlines() if line.strip()]
    if not disease_ids or not map_ids or any(re.fullmatch(r"hsa[0-9]{5}", value) is None for value in hsa_ids):
        raise ValueError("Incomplete KEGG source lists")
    jobs = _batch_jobs(disease_ids, "disease") + _batch_jobs(map_ids + hsa_ids, "pathway")
    pending = [(name, endpoint) for name, endpoint in jobs if not cache.validate(name, REST + endpoint)]
    print("KEGG raw cards: %d batches cached, %d to download (10 records/batch, at most 2.5 requests/s)" % (len(jobs) - len(pending), len(pending)), flush=True)
    if pending:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(cache.fetch, name, endpoint): name for name, endpoint in pending}
            for completed, future in enumerate(as_completed(futures), 1):
                future.result()
                if completed % 10 == 0 or completed == len(pending):
                    print("KEGG raw card downloads: %d/%d completed" % (completed, len(pending)), flush=True)
    cache.complete()
    return {name: raw / name for name in cache.index["sources"]}


def _native(identifier, kind):
    value = identifier.split(":", 1)[-1]
    if kind == "etc":
        if re.fullmatch(r"(?:hsa|map)[0-9]{5}", value) is None:
            raise ValueError("Unexpected pathway identifier: " + identifier)
        return "path:map" + value[-5:]
    prefix = dict(drug="dr:", disease="ds:", gene="hsa:")[kind]
    pattern = dict(drug=r"D[0-9]{5}", disease=r"H[0-9]{5}", gene=r"[0-9]+")[kind]
    if re.fullmatch(pattern, value) is None:
        raise ValueError("Unexpected %s identifier: %s" % (kind, identifier))
    return prefix + value


def _bulk_links(text, left_type, right_type):
    prefixes = dict(drug="dr:", disease="ds:", gene="hsa:", etc="path:")
    for line in text.splitlines():
        if not line.strip():
            continue
        values = line.split("\t")
        if len(values) != 2:
            raise ValueError("Invalid KEGG link row")
        if not values[0].startswith(prefixes[left_type]) or not values[1].startswith(prefixes[right_type]):
            raise ValueError("Unexpected KEGG LINK endpoint types: " + line)
        yield _native(values[0], left_type), _native(values[1], right_type)


def _card_links(records):
    """Extract explicit record fields only; never infer treatment by shared genes."""
    for identifier, fields in records.items():
        if re.fullmatch(r"H[0-9]{5}", identifier):
            disease = _native(identifier, "disease")
            for field in ("PATHWAY", "DIS_PATHWAY"):
                for value in fields.get(field, []):
                    for pathway in re.findall(r"\b(?:hsa|map)[0-9]{5}\b", value):
                        yield "disease_pathway", disease, _native(pathway, "etc"), identifier + ":" + field
        elif re.fullmatch(r"(?:hsa|map)[0-9]{5}", identifier):
            pathway = _native(identifier, "etc")
            for value in fields.get("REL_PATHWAY", []):
                for target in re.findall(r"\b(?:hsa|map)[0-9]{5}\b", value):
                    yield "pathway_pathway", pathway, _native(target, "etc"), identifier + ":REL_PATHWAY"
            for value in fields.get("DRUG", []):
                for drug in re.findall(r"\bD[0-9]{5}\b", value):
                    yield "drug_pathway", _native(drug, "drug"), pathway, identifier + ":DRUG"
            for value in fields.get("DISEASE", []):
                for disease in re.findall(r"\bH[0-9]{5}\b", value):
                    yield "disease_pathway", _native(disease, "disease"), pathway, identifier + ":DISEASE"


def _select_molecular(nodes, relation_rows):
    """Deterministic six-relation graph; preserve reciprocal pathway records."""
    type_ids = {name: number + 1 for number, name in enumerate(EDGE_TYPES)}
    pairs = {name: set() for name in EDGE_TYPES}
    origin_audit, absent, self_loops = [], [], []
    for relation, source, target, origin in relation_rows:
        if relation not in pairs:
            raise ValueError("Unexpected molecular relation: " + relation)
        if source not in nodes or target not in nodes:
            absent.append((relation, source, target, origin, "|".join(value for value in (source, target) if value not in nodes)))
            continue
        if source == target:
            self_loops.append((relation, source, target, origin))
            continue
        origin_audit.append((relation, source, target, origin))
        pairs[relation].add((source, target))
    if self_loops:
        raise ValueError("KEGG explicit relation contains self-loops; stop rather than silently changing it: " + str(self_loops[:5]))
    rows, active, counts = [], set(), Counter()
    for relation in EDGE_TYPES:
        for source, target in sorted(pairs[relation]):
            rows.append((source, target, type_ids[relation], 1, len(rows)))
            active.update((source, target))
            counts[relation] += 1
    return rows, active, counts, sorted(set(origin_audit)), absent


def prepare_inputs(root):
    """Write graph, benchmark positives, native hierarchy, and source audit."""
    root = Path(root).resolve()
    output = paths(root)
    raw = output["raw_dir"]
    if not (raw / "sources.json").exists():
        raise FileNotFoundError("Run download_sources first to freeze KEGG raw data")
    cache = FrozenRestCache(raw)
    if not cache.index.get("completed_utc"):
        raise ValueError("KEGG raw acquisition is incomplete; rerun the download stage")
    for name in cache.index["sources"]:
        cache.validate(name)
    node_rows = []
    for name, kind in (("list_drug.tsv", "drug"), ("list_disease.tsv", "disease"),
                       ("list_hsa.tsv", "gene"), ("list_pathway.tsv", "etc")):
        node_rows.extend(_list_rows(_read_text(raw / name), kind))
    nodes = {row["id"]: row for row in node_rows}
    relation_rows = []
    for name, relation, left, right in (
            ("link_hsa_disease.tsv", "disease_gene", "disease", "gene"),
            ("link_hsa_drug.tsv", "drug_gene", "drug", "gene"),
            ("link_pathway_hsa.tsv", "gene_pathway", "gene", "etc")):
        relation_rows.extend((relation, source, target, name) for source, target in _bulk_links(_read_text(raw / name), left, right))
    for name in sorted(cache.index["sources"]):
        if name.startswith("cards/"):
            relation_rows.extend(_card_links(_flat_records(_read_text(raw / name))))
    graph, active, counts, origins, absent = _select_molecular(nodes, relation_rows)
    if not graph or any(counts[relation] == 0 for relation in EDGE_TYPES):
        raise ValueError("A required Table 4 relation is empty; review acquisition before training")
    positives = sorted(set((drug, disease) for disease, drug in _bulk_links(_read_text(raw / "link_drug_disease.tsv"), "disease", "drug")))
    retained, excluded = [], []
    for drug, disease in positives:
        missing = [node for node in (drug, disease) if node not in active]
        if missing:
            excluded.append((drug, disease, "|".join(missing)))
        else:
            retained.append((drug, disease))
    active_types = {node: nodes[node]["type"] for node in sorted(active)}
    semantic = build_hierarchy(_read_text(raw / "br08303_atc.txt"),
                               _read_text(raw / "br08403_icd11.txt"), active_types)
    _write_rows(output["graph"], graph, delimiter="\t")
    _write_rows(output["nodetypes"], ((row["id"], row["type"]) for row in node_rows if row["id"] in active),
                header=("node", "type"), delimiter="\t")
    _write_rows(output["raw_positives"], positives, header=("drug", "disease"))
    _write_rows(output["positives"], retained, header=("drug", "disease"))
    _write_rows(output["hierarchy"], semantic.edges, header=("child", "parent"))
    audit = output["audit_dir"]
    isolates = [(row["id"], row["name"], row["type"]) for row in node_rows if row["id"] not in active]
    _write_rows(audit / "isolated_nodes.csv", isolates, header=("node", "name", "type"))
    _write_rows(audit / "excluded_positive_pairs.csv", excluded, header=("drug", "disease", "absent_endpoints"))
    _write_rows(audit / "molecular_relation_sources.csv", origins, header=("relation", "source", "target", "source_field"))
    _write_rows(audit / "unlisted_molecular_endpoints.csv", absent, header=("relation", "source", "target", "source_field", "unlisted_endpoints"))
    pathway_normalization = []
    for name in ("list_pathway.tsv", "list_pathway_hsa.tsv"):
        for line in _read_text(raw / name).splitlines():
            if line.strip():
                identifier = line.split("\t", 1)[0]
                pathway_normalization.append((name, identifier, _native(identifier, "etc")))
    _write_rows(audit / "pathway_id_normalization.csv", pathway_normalization,
                header=("source_list", "native_identifier", "canonical_identifier"))
    _write_rows(audit / "semantic_mapping.csv", ([row.get(column, "") for column in AUDIT_COLUMNS] for row in semantic.audit), header=AUDIT_COLUMNS)
    _write_json(audit / "hierarchy_paths.json", {node: [list(path) for path in sorted(values)] for node, values in sorted(semantic.expected_paths.items())})
    _write_rows(audit / "paper_table4_counts.csv",
                [("nodes", kind, REFERENCE_NODE_COUNTS[kind], sum(row["type"] == kind for row in node_rows), sum(value == kind for value in active_types.values())) for kind in ("drug", "disease", "gene", "etc")]
                + [("edges", relation, reference, counts[relation], counts[relation]) for relation, reference in zip(EDGE_TYPES, REFERENCE_EDGE_COUNTS)]
                + [("labels", "drug_disease", 2272, len(positives), len(retained))],
                header=("category", "item", "paper", "current_selected", "current_active"))
    artifacts = [output[key] for key in ("graph", "nodetypes", "positives", "raw_positives", "hierarchy")]
    artifacts += sorted(path for path in audit.glob("*") if path.is_file() and path.name != "manifest.json")
    active_counts = Counter(active_types.values())
    pairs = {tuple(sorted((row[0], row[1]))) for row in graph}
    manifest = dict(schema_version=1, dataset="KEGG current REST reconstruction",
                    source_limit="Author's frozen KEGG benchmark inputs were not released; current API counts are not forced to 2023 Table 4",
                    sources={name: dict(record) for name, record in sorted(cache.index["sources"].items())},
                    snapshot_created_utc=cache.index["created_utc"], snapshot_completed_utc=cache.index["completed_utc"],
                    kegg_release_info=cache.index["kegg_release_info"],
                    molecular=dict(original_nodes=len(nodes), selected_nodes=len(nodes), active_nodes=len(active),
                                   active_node_counts=dict(sorted(active_counts.items())), isolated_nodes=len(isolates),
                                   isolated_node_counts=dict(sorted(Counter(row[2] for row in isolates).items())),
                                   edge_rows=len(graph), unique_undirected_pairs=len(pairs),
                                   additional_parallel_or_reverse_rows=len(graph) - len(pairs),
                                   edge_types=[dict(id=index + 1, metaedge=relation, rows=counts[relation]) for index, relation in enumerate(EDGE_TYPES)],
                                   row_order="Table 4 relation order then native source/target sort; duplicate identical directed relation removed",
                                   pathway_normalization="path:hsaNNNNN and path:mapNNNNN -> path:mapNNNNN",
                                   relation_sources="Explicit KEGG LINK + disease PATHWAY/DIS_PATHWAY + map/hsa pathway REL_PATHWAY/DRUG/DISEASE fields",
                                   unlisted_endpoint_rows=len(absent), treatment_edges_in_graph=0, synthetic_self_loops=0, weight=1),
                    positives=dict(raw=len(positives), active=len(retained), excluded=len(excluded),
                                   source_relation="KEGG explicit link/drug/disease", inferred_by_shared_targets=False,
                                   excluded_pairs=[dict(drug=drug, disease=disease, absent_endpoints=missing) for drug, disease, missing in excluded]),
                    semantic=semantic.stats,
                    ontology_policy="KEGG BRITE ATC br08303 and ICD-11 br08403; exact native identifiers; preserve each explicit hierarchy position",
                    ic_policy="unchanged upstream occurrence counts over every valid root path",
                    artifacts={str(path.relative_to(root)): dict(sha256=sha256(path), bytes=path.stat().st_size) for path in artifacts})
    _write_json(output["manifest"], manifest)
    return manifest
