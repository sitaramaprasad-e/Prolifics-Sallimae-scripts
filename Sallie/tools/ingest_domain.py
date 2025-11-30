# Here is the spec for what to create in Neo4j using the rules‑portal graph routes.
#
# Node labels:
# - LogicStep   (one per business rule / decision)
# - CodeFunction
# - CodeFile
#    + sourcePath
#    + rootDir
# - Parameter
# - DomainType
#   + attributes
# - Domain
#    + diagramPath
#    + ingestionTimestamp
#
# Relationship types:
# - IMPLEMENTED_BY (LogicStep -> CodeFunction)
# - PART_OF       (CodeFunction -> CodeFile), (DomainType -> Domain)
# - INPUT         (LogicStep -> Parameter for inputs)
# - OUTPUT        (LogicStep -> Parameter for outputs)
# - LINKS_TO      (Parameter -> Parameter for hierarchy links between steps)
# - OPERATES_ON (LogicStep -> DomainType)
# - RELATED_TO (DomainType -> DomainType)
#
# When ingesting, we populate the above structure, creating domain types and domains, and linking logic steps to domain types.
#
# The ingestion code talks to the rules‑portal Express server via the following
# HTTP routes exposed in `graphRoutes.js`. The base URL is controlled by the
# environment variable `RULES_PORTAL_BASE_URL` and defaults to
# `http://localhost:3201`:
#
#   GET  /api/graph/status
#       - Lightweight health check.
#       - Returns `{ up: boolean, message: string, ... }`.
#       - Used once per run to decide whether Neo4j graph population is enabled.
#
#   POST /api/graph/node
#       - Create a new node with one or more labels.
#       - Request body: `{ "label"?: string, "labels"?: string[], "properties"?: object }`.
#       - Response body on success: `{ success: true, node: { identity, labels, properties } }`.
#       - We always send `labels` (array) and `properties` from the ingestor.
#
#   POST /api/graph/relationship
#       - Create a relationship between two existing nodes by internal Neo4j id.
#       - Request body: `{ "fromId": number, "toId": number, "type": string, "properties"?: object }`.
#       - Response body on success: `{ success: true, relationship: { identity, type, properties } }`.
#
#   POST /api/graph/run
#       - Execute arbitrary Cypher (primarily for debugging / ad‑hoc queries).
#       - Request body: `{ "query": string, "parameters"?: object }`.
#       - Not required for the standard ingestion flow, which uses the
#         `/api/graph/node` and `/api/graph/relationship` routes.
#
#   GET  /api/graph/related
#       - Fetch neighbourhood around a given node id.
#       - Request query params: `nodeId`, optional `type`, `direction`.
#       - Mainly useful for UI / exploration; not used directly by ingestion.
#
# In summary, rule ingestion is responsible for:
#   1. Creating/merging Message and Sequence nodes.
#   2. Wiring them together with SEQUENCED_BY and PART_OF relationships using the graph routes above.
#!/usr/bin/env python3

# Here is the spec for what to create in Neo4j using the rules‑portal graph routes.
#
# Node labels:
# - LogicStep   (one per business rule / decision)
# - CodeFunction
# - CodeFile
#    + sourcePath
#    + rootDir
# - Parameter
# - DomainType
#   + attributes
# - Domain
#    + diagramPath
#    + ingestionTimestamp
#
# Relationship types:
# - IMPLEMENTED_BY (LogicStep -> CodeFunction)
# - PART_OF       (CodeFunction -> CodeFile), (DomainType -> Domain)
# - INPUT         (LogicStep -> Parameter for inputs)
# - OUTPUT        (LogicStep -> Parameter for outputs)
# - LINKS_TO      (Parameter -> Parameter for hierarchy links between steps)
# - OPERATES_ON   (LogicStep -> DomainType)
# - RELATED_TO    (DomainType -> DomainType)
#
# When ingesting, we populate the above structure, creating domain types and the
# containing Domain node, wiring DomainType nodes to the Domain via PART_OF, and
# linking DomainTypes to each other via RELATED_TO relationships derived from the
# relationships section of the PlantUML diagram. We also attempt to link existing
# LogicStep nodes to DomainTypes via OPERATES_ON relationships using the rule
# notes declared on each domain type (e.g. "note top of Account : Rules:\nE8B· Should …").
#
# The ingestion code talks to the rules‑portal Express server via the following
# HTTP routes exposed in `graphRoutes.js`. The base URL defaults to
# `http://localhost:443` and is typically overridden from the spec file via
# the `graph-url` property.
#
#   GET  /api/graph/status
#       - Lightweight health check.
#       - Returns `{ up: boolean, message: string, ... }`.
#       - Used once per run to decide whether Neo4j graph population is enabled.
#
#   POST /api/graph/node
#       - Create a new node with one or more labels.
#       - Request body: `{ "label"?: string, "labels"?: string[], "properties"?: object }`.
#       - Response body on success: `{ success: true, node: { identity, labels, properties } }`.
#       - We always send `labels` (array) and `properties` from the ingestor.
#
#   POST /api/graph/relationship
#       - Create a relationship between two existing nodes by internal Neo4j id.
#       - Request body: `{ "fromId": number, "toId": number, "type": string, "properties"?: object }`.
#       - Response body on success: `{ success: true, relationship: { identity, type, properties } }`.
#
#   POST /api/graph/run
#       - Execute arbitrary Cypher (primarily for debugging / ad‑hoc queries).
#       - Request body: `{ "query": string, "parameters"?: object }`.
#       - Not required for the standard ingestion flow, which uses the
#         `/api/graph/node` and `/api/graph/relationship` routes.
#
#   GET  /api/graph/related
#       - Fetch neighbourhood around a given node id.
#       - Request query params: `nodeId`, optional `type`, `direction`.
#       - Mainly useful for UI / exploration; not used directly by ingestion.
#
# In summary, domain ingestion is responsible for:
#   1. Parsing the PlantUML domain model report (`domain_model_report.wsd`).
#   2. Creating DomainType nodes (for classes with `<<DomainType>>` and enums).
#   3. Creating a Domain node for the diagram and wiring DomainType -> Domain
#      via PART_OF relationships.
#   4. Creating RELATED_TO relationships between DomainTypes based on the
#      relationships section of the diagram.
#   5. Linking existing LogicStep nodes to DomainTypes via OPERATES_ON using the
#      rule notes (e.g. `note top of Account : Rules:\nE8B· Should …`).

import warnings

# Suppress the LibreSSL/OpenSSL compatibility warning from urllib3 v2
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
    module="urllib3",
)

# ===== Implementation =====
import os
import sys
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

def _normalise_entity_name(name: str) -> str:
    """Normalise a domain/entity name so 'Claim Line' matches 'ClaimLine', etc."""
    return re.sub(r"\s+", "", name or "").strip()

import requests
import logging

# ===== TRACE logger setup (mirrors ingest_sequence.py) =====
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def _trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)


logging.Logger.trace = _trace  # type: ignore[attr-defined]


def _setup_logger(verbose: bool = True) -> None:
    level = TRACE_LEVEL_NUM if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


LOG = logging.getLogger("ingest_domain")
_setup_logger(verbose=True)

# ===== Base URL =====
# Default base URL for the rules‑portal graph API. This is typically overridden
# via the `graph-url` value in the selected spec file inside `main()`.
RULES_PORTAL_BASE_URL = "http://localhost:443"
# Always run graph calls without TLS certificate verification (insecure mode).
GRAPH_VERIFY = False


# ===== HTTP helpers =====
def _get(path: str) -> Optional[dict]:
    url = RULES_PORTAL_BASE_URL + path
    try:
        resp = requests.get(url, timeout=8, verify=GRAPH_VERIFY)
        if resp.status_code == 200:
            return resp.json()
        LOG.warning("[warn] GET %s returned %s: %s", url, resp.status_code, resp.text)
    except Exception as e:  # noqa: BLE001
        LOG.warning("[warn] GET %s failed: %s", url, e)
    return None


def _post(path: str, payload: dict) -> Optional[dict]:
    url = RULES_PORTAL_BASE_URL + path
    try:
        resp = requests.post(url, json=payload, timeout=12, verify=GRAPH_VERIFY)
        if resp.status_code == 200:
            return resp.json()
        LOG.warning("[warn] POST %s returned %s: %s", url, resp.status_code, resp.text)
    except Exception as e:  # noqa: BLE001
        LOG.warning("[warn] POST %s failed: %s", url, e)
    return None


def run_cypher(query: str, parameters: Optional[dict] = None) -> Optional[list]:
    payload = {"query": query, "parameters": parameters or {}}
    resp = _post("/api/graph/run", payload)
    if resp is None:
        LOG.warning("[warn] run_cypher failed for query: %s", query)
        return None
    if "results" in resp:
        return resp["results"]
    return resp


def _to_int_id(value: Any) -> Optional[int]:
    """Convert a Neo4j identity into a plain int.

    Accepts either an int, or an object like {"low": 123, "high": 0}.
    """

    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        low = value.get("low")
        high = value.get("high", 0)
        if isinstance(low, int) and isinstance(high, int):
            return int(low) + (int(high) << 32)
    try:
        return int(value)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None


def create_node(labels: List[str], properties: Dict[str, Any]) -> Optional[int]:
    payload = {"labels": labels, "properties": properties}
    resp = _post("/api/graph/node", payload)
    if not resp or not resp.get("success"):
        LOG.warning("[warn] create_node failed: %s", resp)
        return None
    node = resp.get("node")
    if not node:
        LOG.warning("[warn] create_node: no node in response: %s", resp)
        return None
    return _to_int_id(node.get("identity"))


def create_relationship(
    from_id: int,
    to_id: int,
    rel_type: str,
    properties: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    payload = {
        "fromId": from_id,
        "toId": to_id,
        "type": rel_type,
        "properties": properties or {},
    }
    resp = _post("/api/graph/relationship", payload)
    if not resp or not resp.get("success"):
        LOG.warning("[warn] create_relationship failed: %s", resp)
        return None
    rel = resp.get("relationship")
    if not rel:
        LOG.warning("[warn] create_relationship: no relationship in response: %s", resp)
        return None
    return _to_int_id(rel.get("identity"))


def _normalize_cypher_results(results: Any) -> List[Any]:
    """Best‑effort normalisation of graph API results into a simple row list."""

    if isinstance(results, list):
        return results
    if isinstance(results, dict):
        for key in ("results", "rows", "data", "records"):
            val = results.get(key)
            if isinstance(val, list):
                return val
        LOG.warning("[warn] Unexpected run_cypher result format (dict): %r", results)
        return []
    LOG.warning("[warn] Unexpected run_cypher result type: %r", type(results))
    return []


# ===== Spec discovery & load (mirrors ingest_sequence.py) =====


def _spec_dir() -> str:
    # This script lives in <repo>/tools; specs are under <repo>/tools/spec
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec")


def _find_spec_files() -> List[str]:
    d = _spec_dir()
    if not os.path.isdir(d):
        return []
    return [
        os.path.join(d, f)
        for f in sorted(os.listdir(d))
        if f.endswith(".json")
        and (f.startswith("source_") or f.startswith("sources_"))
        and os.path.isfile(os.path.join(d, f))
    ]


def _choose_spec(specs: List[str]) -> Optional[str]:
    if not specs:
        LOG.error(
            "[error] No spec files found under tools/spec (expected 'source_*.json' or 'sources_*.json').",
        )
        return None
    print("\n=== Select a spec file for domain ingestion ===")
    for i, path in enumerate(specs, 1):
        print(f"  {i}. {os.path.basename(path)}")
    print("Choose a spec by number (or Ctrl+C to cancel).")
    while True:
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(specs):
            return specs[idx - 1]
        print("Out of range. Try again.")


def _load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _choose_pair(spec: Dict[str, Any]) -> Optional[int]:
    """Prompt the user to choose a single path‑pair index (1‑based) from the spec."""

    pairs = spec.get("path-pairs", [])
    if not pairs:
        LOG.error("[error] Spec has no 'path-pairs' entries.")
        return None
    print("\n=== Select a path-pair for domain ingestion ===")
    for i, p in enumerate(pairs, 1):
        src = p.get("source-path", "?")
        outp = p.get("output-path", "?")
        team = p.get("team", "-")
        comp = p.get("component", "-")
        print(f"  {i}. {src} -> {outp} (team={team}, component={comp})")
    print("Choose a pair by number (or Ctrl+C to cancel).")
    while True:
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(pairs):
            return idx
        print("Out of range. Try again.")


# ===== Domain model parser =====


class DomainTypeSpec(Dict[str, Any]):
    """Convenience type alias for a parsed DomainType."""


def _parse_domain_model(path: str) -> Tuple[List[DomainTypeSpec], List[Dict[str, str]]]:
    """Parse the PlantUML domain model file.

    Returns a tuple of (domain_types, relationships) where:
      * domain_types is a list of dicts with keys:
          - name: str
          - kind: "class" | "enum"
          - attributes: List[str]
          - logics: List[{code, name, raw}] (may be empty)
      * relationships is a list of dicts with keys:
          - from: str
          - to: str
          - label: str
    """

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:  # noqa: BLE001
        LOG.error("[error] Could not read domain model file %s: %s", path, e)
        return [], []

    domain_types: Dict[str, DomainTypeSpec] = {}
    relationships: List[Dict[str, str]] = []

    class_rx = re.compile(r"^\s*class\s+(\w+)\s*<<DomainType>>\s*\{")
    enum_rx = re.compile(r"^\s*enum\s+(\w+)\s*\{")

    i = 0
    n = len(lines)

    # ---- First pass: classes and enums ----
    while i < n:
        line = lines[i]

        m_class = class_rx.match(line)
        m_enum = enum_rx.match(line)

        if m_class:
            name = m_class.group(1)
            attrs: List[str] = []
            i += 1
            while i < n and "}" not in lines[i]:
                raw = lines[i].strip()
                if raw and not raw.startswith("note ") and not raw.startswith("'"):
                    # Attribute lines typically look like: "+Id: String"
                    attrs.append(raw.lstrip("+"))
                i += 1
            domain_types[name] = DomainTypeSpec(
                name=name,
                kind="class",
                attributes=attrs,
                logics=[],
            )
        elif m_enum:
            name = m_enum.group(1)
            values: List[str] = []
            i += 1
            while i < n and "}" not in lines[i]:
                raw = lines[i].strip()
                if raw and not raw.startswith("'"):
                    values.append(raw)
                i += 1
            domain_types[name] = DomainTypeSpec(
                name=name,
                kind="enum",
                attributes=values,
                logics=[],
            )
        else:
            i += 1

    # ---- Second pass: notes (Logics) and relationships ----
    note_inline_rx = re.compile(r"^\s*note\s+top\s+of\s+(\w+)\s*:\s*(.*)$")
    note_block_rx = re.compile(r"^\s*note\s+(top|right|left|bottom)\s+of\s+(\w+)\s*$")
    rel_rx = re.compile(
        r"^\s*(\w+)\s+\"[^\"]*\"\s+[-o]+-\s+\"[^\"]*\"\s+(\w+)\s*:?(.*)$",
    )

    i = 0
    while i < n:
        line = lines[i]

        # Inline note style: "note top of Account : Logics:\nE8B· ...,4AF· ..."
        m_inline = note_inline_rx.match(line)
        if m_inline:
            dt_name = m_inline.group(1)
            rest = m_inline.group(2) or ""
            if "Logics:" in rest:
                logics_part = rest.split("Logics:", 1)[1].strip()
                logics_text = logics_part.replace("\\n", "\n")
                flat = logics_text.replace("\n", ",")
                pieces = [p.strip().strip("\"") for p in flat.split(",")]

                dt = domain_types.setdefault(
                    dt_name,
                    DomainTypeSpec(name=dt_name, kind="class", attributes=[], logics=[]),
                )
                if "logics" not in dt:
                    dt["logics"] = []

                for item in pieces:
                    if not item:
                        continue
                    m_logic = re.match(r"^([^\s·•]+)\s*[·•]\s*(.+)$", item)
                    if m_logic:
                        code = m_logic.group(1).strip()
                        rname = m_logic.group(2).strip()
                    else:
                        code = None
                        rname = item
                    dt["logics"].append({"code": code, "name": rname, "raw": item})
            i += 1
            continue

        # Block note style:
        #   note right of Patient
        #     Logics:
        #     2B0· Standardize Gender,
        #     41F· Calculate Age At First Claim
        #   end note
        m_block = note_block_rx.match(line)
        if m_block:
            dt_name = m_block.group(2)
            dt = domain_types.setdefault(
                dt_name,
                DomainTypeSpec(name=dt_name, kind="class", attributes=[], logics=[]),
            )
            if "logics" not in dt:
                dt["logics"] = []

            i += 1
            in_logics = False
            while i < n and "end note" not in lines[i]:
                stripped = lines[i].strip()
                if not stripped:
                    i += 1
                    continue
                # Detect the start of a "Logics" section in a case-insensitive way,
                # allowing for an optional trailing colon (e.g. "Logics" or "Logics:").
                lower = stripped.lower()
                if lower == "logics" or lower.startswith("logics:"):
                    # Start of logics section inside this note
                    in_logics = True
                    i += 1
                    continue
                if in_logics:
                    # Strip any trailing comma
                    item = stripped.rstrip(",").strip().strip("\"")
                    if item:
                        m_logic = re.match(r"^([^\s·•]+)\s*[·•]\s*(.+)$", item)
                        if m_logic:
                            code = m_logic.group(1).strip()
                            rname = m_logic.group(2).strip()
                        else:
                            code = None
                            rname = item
                        dt["logics"].append({"code": code, "name": rname, "raw": item})
                i += 1
            # Skip the "end note" line
            if i < n and "end note" in lines[i]:
                i += 1
            continue

        # Relationships between domain types (unchanged)
        m_rel = rel_rx.match(line)
        if m_rel:
            from_name = m_rel.group(1)
            to_name = m_rel.group(2)
            label = (m_rel.group(3) or "").strip()
            relationships.append({"from": from_name, "to": to_name, "label": label})

        i += 1

    LOG.info(
        "[info] Parsed %d DomainType(s) and %d relationship(s) from %s",
        len(domain_types),
        len(relationships),
        path,
    )

    return list(domain_types.values()), relationships


# ===== Helper: parse twin .txt domain description file =====
def _parse_domain_description_file(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse a Markdown-style domain description file that matches the domain_model_report.wsd.

    Returns a mapping:
      {
        "<DomainTypeName>": {
          "description": "<entity description>",
          "attributes": {
             "<AttrName>": "<attribute description>",
             ...
          }
        },
        ...
      }

    The parser looks for sections like:

      ### **Patient**
      Represents individuals...

      **Attributes:**
      - PatientKey: Unique system identifier for the patient
      - SourcePatientID: Original patient identifier from source system

    and similarly for enums under "## Enumerations".
    """
    entities: Dict[str, Dict[str, Any]] = {}
    current_entity: Optional[str] = None
    collecting_entity_desc = False
    collecting_attrs = False
    entity_desc_parts: List[str] = []

    # Regexes for section and attribute detection
    entity_header_rx = re.compile(r"^###\s+\*\*(.+?)\*\*")
    attrs_header_rx = re.compile(r"^\s*\*\*Attributes:\*\*")
    relationships_header_rx = re.compile(r"^\s*\*\*Relationships:\*\*")
    enum_header_rx = re.compile(r"^##\s+Enumerations\s*$")

    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:  # noqa: BLE001
        LOG.warning("[warn] Could not read domain description file %s: %s", path, e)
        return entities

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip("\n")

        # Detect entity header like "### **Patient**"
        m_ent = entity_header_rx.match(line)
        if m_ent:
            # Flush previous entity description if any
            if current_entity is not None and entity_desc_parts:
                desc = " ".join(p.strip() for p in entity_desc_parts if p.strip())
                key = _normalise_entity_name(current_entity)
                entities.setdefault(key, {})["description"] = desc
            # Heading text, e.g. "Claim Line"
            current_entity = m_ent.group(1).strip()
            key = _normalise_entity_name(current_entity)
            entities.setdefault(key, {}).setdefault("attributes", {})
            entity_desc_parts = []
            collecting_entity_desc = True
            collecting_attrs = False
            i += 1
            continue

        # Attributes header
        if attrs_header_rx.match(line):
            # Finish entity description collection
            if current_entity is not None and entity_desc_parts:
                desc = " ".join(p.strip() for p in entity_desc_parts if p.strip())
                key = _normalise_entity_name(current_entity)
                entities.setdefault(key, {})["description"] = desc
            collecting_entity_desc = False
            collecting_attrs = True
            i += 1
            continue

        # Relationships header or new section – stop collecting attributes
        if relationships_header_rx.match(line) or line.startswith("## ") or line.startswith("### "):
            collecting_attrs = False
            collecting_entity_desc = False
            # Do not consume the line here for headers – allow re-processing if it's another entity
            if entity_header_rx.match(line):
                continue
            i += 1
            continue

        # Collect entity description text: first paragraph(s) after the entity header until Attributes/Relationships/next header.
        # If we encounter a "(note" line, we treat that (and anything that follows) as out of scope for the description.
        if collecting_entity_desc and current_entity is not None:
            stripped = line.strip()
            if stripped.lower().startswith("(note"):
                # Stop collecting description when we hit the logics/note section.
                collecting_entity_desc = False
                i += 1
                continue
            # Skip empty lines
            if stripped:
                entity_desc_parts.append(stripped)
            i += 1
            continue

        # Collect attributes as "- Name: Description"
        if collecting_attrs and current_entity is not None:
            stripped = line.strip()
            if stripped.startswith("- "):
                # Attribute line
                attr_body = stripped[2:].strip()
                # Split on first ":" to separate name and description
                if ":" in attr_body:
                    attr_name, attr_desc = attr_body.split(":", 1)
                    attr_name = attr_name.strip()
                    attr_desc = attr_desc.strip()
                else:
                    attr_name = attr_body.strip()
                    attr_desc = ""
                key = _normalise_entity_name(current_entity)
                ent = entities.setdefault(key, {})
                attrs_map = ent.setdefault("attributes", {})
                attrs_map[attr_name] = attr_desc
                i += 1
                continue
            # Stop collecting attributes when we hit a blank line or another section
            if not stripped or stripped.startswith("**Relationships:**") or stripped.startswith("### ") or stripped.startswith("## "):
                collecting_attrs = False
            i += 1
            continue

        i += 1

    # Flush last entity description, if any
    if current_entity is not None and entity_desc_parts:
        desc = " ".join(p.strip() for p in entity_desc_parts if p.strip())
        key = _normalise_entity_name(current_entity)
        entities.setdefault(key, {})["description"] = desc

    LOG.info("[info] Parsed descriptions for %d domain types from %s", len(entities), path)
    return entities


# ===== LogicStep lookup (for OPERATES_ON) =====


def _find_logic_step_id(logic_code: Optional[str], name: Optional[str]) -> Optional[int]:
    """Lookup a LogicStep id for a given rule using short code + name.

    The matching strategy is:
      * Derive a 3-hex short code from the LogicStep `id` property using
        `toUpper(right(replace(ls.id, '-', ''), 3))`, and require it to equal
        the provided `logic_code` (case-insensitive).
      * Require an exact, case-insensitive match between `ls.name` and the
        provided `name`.

    This reflects the convention that the domain model logic code (e.g. '2B0')
    is the last 3 hex characters of the LogicStep UUID, and we also harden the
    match by checking the logic name.
    """

    short = (logic_code or "").strip().upper()
    name = (name or "").strip()
    if not short and not name:
        return None

    query = """
    MATCH (ls:LogicStep)
    WITH ls, toUpper(right(replace(ls.id, '-', ''), 3)) AS shortCode
    WHERE
      ($short <> '' AND shortCode = $short)
      AND
      ($name <> '' AND toLower(ls.name) = toLower($name))
    RETURN id(ls) AS id,
           coalesce(ls.name, '') AS name
    LIMIT 1
    """

    params = {"short": short, "name": name}
    results = run_cypher(query, params)
    if not results:
        return None

    rows = _normalize_cypher_results(results)
    if not rows:
        return None

    row = rows[0]
    ls_id: Optional[int] = None
    display_name: Optional[str] = None

    if isinstance(row, dict):
        if "id" in row:
            ls_id = _to_int_id(row.get("id"))
            display_name = str(row.get("name") or "")
        elif isinstance(row.get("row"), (list, tuple)) and row["row"]:
            ls_id = _to_int_id(row["row"][0])
            if len(row["row"]) > 1:
                display_name = str(row["row"][1])
    elif isinstance(row, (list, tuple)) and row:
        ls_id = _to_int_id(row[0])
        if len(row) > 1:
            display_name = str(row[1])

    if ls_id is not None:
        LOG.info(
            "[info] Matched logic '%s' (code=%s) to LogicStep id=%s name='%s'",
            name or short,
            short or "",
            ls_id,
            display_name or "",
        )
    return ls_id


# ===== Helper to delete existing Domain and DomainTypes for diagramPath =====
def _delete_existing_domain(diagram_path: str):
    """
    Delete any existing Domain and DomainType nodes for the given diagram_path.
    """
    cypher = """
    MATCH (d:Domain {diagramPath: $diagramPath})
    OPTIONAL MATCH (dt:DomainType)-[:PART_OF]->(d)
    WITH collect(DISTINCT d) AS domains, collect(DISTINCT dt) AS dts
    FOREACH (x IN dts | DETACH DELETE x)
    FOREACH (d IN domains | DETACH DELETE d)
    RETURN size(domains) AS domainsDeleted, size(dts) AS domainTypesDeleted
    """
    params = {"diagramPath": diagram_path}
    results = run_cypher(cypher, params)
    rows = _normalize_cypher_results(results)
    if not rows:
        LOG.info("[info] No existing Domain node found for diagramPath=%s; nothing to delete", diagram_path)
        return
    row = rows[0]
    domains_deleted: Any = 0
    domain_types_deleted: Any = 0

    if isinstance(row, dict):
        domains_deleted = row.get("domainsDeleted", 0)
        domain_types_deleted = row.get("domainTypesDeleted", 0)
    elif isinstance(row, (list, tuple)):
        if len(row) > 0:
            domains_deleted = row[0]
        if len(row) > 1:
            domain_types_deleted = row[1]

    # Normalise potential Neo4j integer structures (e.g. {"low": 1, "high": 0}) to plain ints
    try:
        domains_deleted_int = _to_int_id(domains_deleted) or 0
    except Exception:  # noqa: BLE001
        domains_deleted_int = 0

    try:
        domain_types_deleted_int = _to_int_id(domain_types_deleted) or 0
    except Exception:  # noqa: BLE001
        domain_types_deleted_int = 0

    LOG.info(
        "[info] Deleted %d existing Domain node(s) and %d DomainType node(s) for diagramPath=%s",
        domains_deleted_int,
        domain_types_deleted_int,
        diagram_path,
    )


# ===== Main entry point =====


def main() -> None:
    global RULES_PORTAL_BASE_URL
    _setup_logger(verbose=True)

    domain_name: Optional[str] = None

    # Determine domain model path either from CLI or spec selection
    if len(sys.argv) == 2:
        domain_path = sys.argv[1]
        LOG.info("[info] Using explicit domain model path from CLI.")
    elif len(sys.argv) == 1:
        specs = _find_spec_files()
        if not specs:
            LOG.error("[error] No spec files found in %s", _spec_dir())
            sys.exit(1)
        spec_path = _choose_spec(specs)
        if not spec_path:
            sys.exit(1)
        LOG.info("[info] Loading spec from %s", spec_path)
        spec = _load_spec(spec_path)

        graph_url = spec.get("graph-url")
        if graph_url:
            RULES_PORTAL_BASE_URL = str(graph_url).rstrip("/")
            LOG.info("[info] Using graph URL from spec: %s", RULES_PORTAL_BASE_URL)

        pair_index = _choose_pair(spec)
        if pair_index is None:
            sys.exit(1)

        pairs = spec.get("path-pairs", [])
        pair = pairs[pair_index - 1]

        component = pair.get("component") or ""
        if component:
            domain_name = f"{component} Domain"
        LOG.info("[info] Selected component='%s'", component)

        root_dir = os.path.expanduser(spec.get("root-directory", os.getcwd()))
        out_rel = pair.get("output-path") or ""

        output_dir = os.path.join(root_dir, out_rel) if out_rel else root_dir
        domain_path = os.path.join(output_dir, "domain_model_report", "domain_model_report.wsd")

        LOG.info("[info] Selected rootDir='%s'", root_dir)
        LOG.info("[info] Selected outputDir='%s'", output_dir)
        LOG.info("[info] Domain model will be read from '%s'", domain_path)
    else:
        print("Usage: python ingest_domain.py [path/to/domain_model_report.wsd]")
        sys.exit(1)

    abs_domain_path = os.path.abspath(domain_path)
    LOG.info("[info] Domain model path: %s", abs_domain_path)
    if not os.path.isfile(abs_domain_path):
        LOG.error("[error] Domain model file not found: %s", abs_domain_path)
        sys.exit(1)

    # Contact graph API
    LOG.info("[info] Contacting rules-portal graph API at %s", RULES_PORTAL_BASE_URL)
    status = _get("/api/graph/status")
    if not status or not status.get("up", False):
        LOG.error("[error] rules-portal graph API is not up or unreachable.")
        sys.exit(1)

    # Parse domain model
    domain_types, relationships = _parse_domain_model(abs_domain_path)
    # Attempt to parse twin .txt description file next to the .wsd
    description_map: Dict[str, Dict[str, Any]] = {}
    txt_path = os.path.splitext(abs_domain_path)[0] + ".txt"
    if os.path.isfile(txt_path):
        LOG.info("[info] Loading domain descriptions from %s", txt_path)
        description_map = _parse_domain_description_file(txt_path)
    else:
        LOG.info("[info] No domain description file found at %s (skipping descriptions)", txt_path)
    if not domain_types:
        LOG.warning("[warn] No DomainType entries parsed from file: %s", abs_domain_path)
        sys.exit(0)

    _delete_existing_domain(abs_domain_path)

    ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # Create Domain node
    domain_props: Dict[str, Any] = {
        "diagramPath": abs_domain_path,
        "ingestionTimestamp": ts,
    }
    if domain_name:
        domain_props["name"] = domain_name

    domain_id = create_node(["Domain"], domain_props)
    if domain_id is None:
        LOG.error("[error] Failed to create Domain node.")
        sys.exit(1)

    # Create DomainType nodes and PART_OF relationships
    dt_name_to_id: Dict[str, int] = {}
    for dt in domain_types:
        name = dt.get("name") or ""
        if not name:
            continue
        kind = dt.get("kind", "class")
        raw_attributes = dt.get("attributes", []) or []

        # Look up description and attribute descriptions from the .txt file (if available)
        norm_name = _normalise_entity_name(name)
        desc_entry = description_map.get(norm_name)
        if desc_entry is None:
            LOG.trace(
                "[trace] No description entry found for DomainType '%s' (normalised='%s') in description_map",
                name,
                norm_name,
            )
            desc_entry = {}
        entity_description = desc_entry.get("description")
        if not entity_description:
            LOG.trace(
                "[trace] No entity description found for DomainType '%s' (will omit 'description' property)",
                name,
            )
        attr_descs: Dict[str, str] = desc_entry.get("attributes", {}) or {}

        # Enrich each attribute string with "-<description>" where we can match by attribute name
        enriched_attributes: List[str] = []
        for attr in raw_attributes:
            attr_str = str(attr)
            # Strip common UML prefixes/suffixes to get the attribute name (before colon, if present)
            tmp = attr_str.lstrip("+#-").strip()
            attr_name = tmp.split(":", 1)[0].strip()
            desc = attr_descs.get(attr_name)
            if desc:
                enriched_attributes.append(f"{attr_str} - {desc}")
            else:
                LOG.trace(
                    "[trace] No attribute description found for '%s' on DomainType '%s' (raw='%s')",
                    attr_name,
                    name,
                    attr_str,
                )
                enriched_attributes.append(attr_str)

        props: Dict[str, Any] = {
            "name": name,
            "kind": kind,
            "diagramPath": abs_domain_path,
        }
        if entity_description:
            props["description"] = entity_description

        # Store attributes as JSON string for easier inspection/searching
        try:
            props["attributes"] = json.dumps(enriched_attributes, ensure_ascii=False)
        except TypeError:
            props["attributes"] = str(enriched_attributes)

        dt_id = create_node(["DomainType"], props)
        if dt_id is None:
            LOG.warning("[warn] Failed to create DomainType node for '%s'", name)
            continue
        dt_name_to_id[name] = dt_id

        rel_id = create_relationship(dt_id, domain_id, "PART_OF", {})
        if rel_id is None:
            LOG.warning(
                "[warn] Failed to create PART_OF relationship for DomainType '%s' -> Domain %d",
                name,
                domain_id,
            )

    LOG.info("[info] Created %d DomainType node(s)", len(dt_name_to_id))

    # Create RELATED_TO relationships between DomainTypes (bidirectional for easy traversal)
    related_created = 0
    for rel in relationships:
        from_name = rel.get("from")
        to_name = rel.get("to")
        label = rel.get("label", "")
        if not from_name or not to_name:
            continue
        from_id = dt_name_to_id.get(from_name)
        to_id = dt_name_to_id.get(to_name)
        if from_id is None or to_id is None:
            LOG.trace(
                "[trace] Skipping RELATED_TO for '%s' -> '%s' (one or both DomainTypes missing)",
                from_name,
                to_name,
            )
            continue

        props = {"label": label} if label else {}

        if create_relationship(from_id, to_id, "RELATED_TO", props) is not None:
            related_created += 1
        if create_relationship(to_id, from_id, "RELATED_TO", props) is not None:
            related_created += 1

    LOG.info("[info] Created %d RELATED_TO relationship(s) (including reverse links)", related_created)

    # Link LogicSteps to DomainTypes via OPERATES_ON based on rules in notes
    operates_created = 0
    cache: Dict[Tuple[Optional[str], Optional[str]], Optional[int]] = {}

    for dt in domain_types:
        dt_name = dt.get("name")
        dt_id = dt_name_to_id.get(dt_name)
        if dt_id is None:
            continue
        logics = dt.get("logics") or []
        if not logics:
            continue

        for logic in logics:
            code = logic.get("code")
            rname = logic.get("name")
            key = (code, rname)
            if key not in cache:
                cache[key] = _find_logic_step_id(code, rname)
            ls_id = cache[key]
            if ls_id is None:
                LOG.info(
                    "[info] No LogicStep match found for logic '%s' (code=%s)",
                    rname or code or "",
                    code or "",
                )
                continue

            props = {
                "logicCode": code,
                "logicName": rname,
                "domainType": dt_name,
                "diagramPath": abs_domain_path,
                "ingestionTimestamp": ts,
            }
            if create_relationship(ls_id, dt_id, "OPERATES_ON", props) is not None:
                operates_created += 1

    LOG.info("[info] Created %d OPERATES_ON relationship(s)", operates_created)
    LOG.info(
        "[info] Domain ingestion complete: %d DomainType nodes, %d RELATED_TO, %d OPERATES_ON",
        len(dt_name_to_id),
        related_created,
        operates_created,
    )


if __name__ == "__main__":
    main()