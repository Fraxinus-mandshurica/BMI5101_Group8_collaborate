"""Parse official KEGG BRITE ATC and ICD-11 hierarchies without model changes.

These are input adapters only. BRITE br08303 supplies native KEGG DRUG to ATC
assignments; br08403 supplies native KEGG DISEASE leaves and ICD-11 ancestors.
The current KEGG ATC assignment source differs from the DrugBank assignment
source named in the DREAMwalk paper. Record the source snapshot in the caller.

ICD-11 internal nodes are qualified by their complete BRITE ancestor path.
This preserves every published path even if a code or label is repeated in a
different position. No WHO API credentials or text/name matching are used.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import html
import json
import re
from typing import NamedTuple


AUDIT_COLUMNS = ("node", "node_type", "source", "source_code", "label", "path", "mapping_status")
_ATC_LENGTHS = (1, 3, 4, 5, 7)
_ATC_CODE = re.compile(r"[A-Z](?:[0-9]{2}(?:[A-Z](?:[A-Z](?:[0-9]{2})?)?)?)?\Z")
_ICD_CODE = re.compile(r"(?:[0-9]{2}|[0-9A-Z]{4}(?:\.[0-9A-Z]+)?)\Z")


class HierarchyResult(NamedTuple):
    edges: list
    audit: list
    stats: dict
    expected_paths: dict


def _brite_rows(text):
    """Yield (level, normalized payload, parent payload tuple) in source order."""
    stack = []
    previous_leaf = None
    for lineno, line in enumerate(text.splitlines(), 1):
        if re.match(r"^[A-Z]", line) is None:
            continue
        level = ord(line[0]) - ord("A")
        payload = re.sub(r"<[^>]+>", "", html.unescape(line[1:])).strip()
        payload = re.sub(r"\s+", " ", payload)
        if not payload:
            raise ValueError("Empty BRITE hierarchy row at line {}".format(lineno))
        if previous_leaf is not None and level > previous_leaf:
            raise ValueError("Native BRITE leaf unexpectedly has children at line {}".format(lineno))
        if level > len(stack):
            raise ValueError("BRITE hierarchy skips a parent level at line {}".format(lineno))
        stack = stack[:level]
        parents = tuple(stack)
        yield level, payload, parents
        # A native D/H leaf cannot become the parent of the next row.
        is_leaf = re.match(r"[DH][0-9]{5}(?: |$)", payload) is not None
        previous_leaf = level if is_leaf else None
        stack.append(payload)


def _atc_assignments(text):
    assignments, labels = defaultdict(set), {}
    source_rows = 0
    for level, payload, parents in _brite_rows(text):
        token = payload.split(" ", 1)[0]
        if re.fullmatch(r"D[0-9]{5}", token):
            if level != 5 or len(parents) != 5:
                raise ValueError("Expected a KEGG drug below all five ATC levels: " + token)
            codes = tuple(parent.split(" ", 1)[0] for parent in parents)
            if tuple(map(len, codes)) != _ATC_LENGTHS or any(_ATC_CODE.fullmatch(code) is None for code in codes):
                raise ValueError("Invalid five-level ATC ancestry for " + token)
            if any(codes[-1][:size] != code for size, code in zip(_ATC_LENGTHS, codes)):
                raise ValueError("ATC parent does not match its official prefix: " + token)
            assignments["dr:" + token].add(codes)
            source_rows += 1
        elif level < 5:
            if _ATC_CODE.fullmatch(token) is None or len(token) != _ATC_LENGTHS[level]:
                raise ValueError("Unexpected ATC hierarchy term: " + payload)
            label = payload.split(" ", 1)[1] if " " in payload else ""
            if token in labels and labels[token] != label:
                raise ValueError("Conflicting ATC labels for " + token)
            labels[token] = label
        # Other terminal records, e.g. Cxxxxx compounds, are not KEGG DRUG.
    if not assignments:
        raise ValueError("BRITE ATC input has no native KEGG drug assignments")
    return assignments, labels, source_rows


def _icd_assignments(text):
    assignments, labels = defaultdict(set), {}
    source_rows = 0
    for _, payload, parents in _brite_rows(text):
        token = payload.split(" ", 1)[0]
        if re.fullmatch(r"H[0-9]{5}", token):
            if not parents:
                raise ValueError("KEGG disease has no ICD-11 ancestors: " + token)
            assignments["ds:" + token].add(parents)
            source_rows += 1
        else:
            position = parents + (payload,)
            labels[position] = payload
    if not assignments:
        raise ValueError("BRITE ICD-11 input has no native KEGG disease assignments")
    return assignments, labels, source_rows


def _icd_position_id(position):
    serialized = json.dumps(position, ensure_ascii=False, separators=(",", ":"))
    return "ICD11pos::" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _audit(node, node_type, source, source_code, label, path, status="mapped"):
    return dict(zip(AUDIT_COLUMNS, (node, node_type, source, source_code, label,
                                   json.dumps(path, ensure_ascii=False), status)))


def build_hierarchy(atc_text, icd_text, active_types):
    """Return (edges, audit, stats, expected_paths) for active molecular nodes.

    ``active_types`` maps exact native graph IDs (dr:Dxxxxx / ds:Hxxxxx) to
    upstream node types (drug / disease / gene / etc). Unmapped drug/disease
    nodes remain in the molecular graph but get no fabricated semantic edges.
    ``expected_paths`` contains complete leaf-to-root chains for comparison
    with the unchanged upstream hierarchy merger.
    """
    atc, atc_labels, atc_rows = _atc_assignments(atc_text)
    icd, _, icd_rows = _icd_assignments(icd_text)
    edges, expected, audit = set(), {}, []
    used_atc, used_icd = set(), set()
    stats = Counter()
    for node, kind in sorted(active_types.items()):
        if kind not in ("drug", "disease"):
            continue
        prefix, pattern = ("dr:", r"D[0-9]{5}") if kind == "drug" else ("ds:", r"H[0-9]{5}")
        if not node.startswith(prefix) or re.fullmatch(pattern, node[len(prefix):]) is None:
            raise ValueError("Expected native KEGG {} ID, received {}".format(kind, node))
        stats["active_" + kind + "_total"] += 1
        source, mappings = ("br08303", atc.get(node, set())) if kind == "drug" else ("br08403", icd.get(node, set()))
        stats["active_" + kind + "_mapped"] += bool(mappings)
        stats["active_" + kind + "_unmapped"] += not bool(mappings)
        if not mappings:
            audit.append(_audit(node, kind, source, node[len(prefix):], "", [], "unmapped"))
            continue
        for path in sorted(mappings):
            if kind == "drug":
                internals = tuple("ATC::" + code for code in reversed(path))
                chain = (node,) + internals + ("drug",)
                used_atc.update(path)
                assignment = path[-1]
            else:
                positions = tuple(path[:index + 1] for index in range(len(path)))
                internals = tuple(_icd_position_id(position) for position in reversed(positions))
                chain = (node,) + internals + ("disease",)
                used_icd.update(positions)
                token = path[-1].split(" ", 1)[0]
                assignment = token if _ICD_CODE.fullmatch(token) else ""
            expected.setdefault(node, set()).add(chain)
            edges.update(zip(chain, chain[1:]))
            audit.append(_audit(node, kind, source, assignment, "", path))
        stats["active_" + kind + "_paths"] += len(mappings)
    for code in sorted(used_atc):
        audit.append(_audit("ATC::" + code, "ontology", "br08303", code, atc_labels[code],
                            [code[:size] for size in _ATC_LENGTHS if size <= len(code)]))
    for position in sorted(used_icd):
        token = position[-1].split(" ", 1)[0]
        code = token if _ICD_CODE.fullmatch(token) else ""
        label = position[-1][len(token):].strip() if code else position[-1]
        audit.append(_audit(_icd_position_id(position), "ontology", "br08403", code, label, position))
    stats.update(dict(source_atc_drugs=len(atc), source_atc_assignment_rows=atc_rows,
                      source_icd_diseases=len(icd), source_icd_assignment_rows=icd_rows,
                      atc_internal_nodes=len(used_atc), icd_internal_positions=len(used_icd),
                      hierarchy_edges=len(edges), mapped_graph_nodes=len(expected)))
    # Emit zero counts as well, so the manifest has a stable coverage schema.
    for kind in ("drug", "disease"):
        for suffix in ("total", "mapped", "unmapped", "paths"):
            stats.setdefault("active_" + kind + "_" + suffix, 0)
    return HierarchyResult(sorted(edges), audit, dict(stats), expected)
