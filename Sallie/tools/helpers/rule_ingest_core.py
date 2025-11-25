
import os
import re
import uuid
import logging
import requests
from datetime import datetime
from typing import Optional, List, Dict, Any, Set, Tuple


# Here is the spec for what to create in Neo4j using the rules‑portal graph routes.
#
# Node labels:
# - LogicStep   (one per business rule / decision)
# - CodeFunction
# - CodeFile
#   + sourcePath
#   + rootDir
# - Parameter
#
# Relationship types:
# - IMPLEMENTED_BY (LogicStep -> CodeFunction)
# - PART_OF       (CodeFunction -> CodeFile)
# - INPUT         (LogicStep -> Parameter for inputs)
# - OUTPUT        (LogicStep -> Parameter for outputs)
# - LINKS_TO      (Parameter -> Parameter for hierarchy links between steps)
#
# When ingesting, we populate the above structure from the rule definitions and DMN
# inputs/outputs, plus the cross‑step `links` that describe how outputs from one
# step feed into inputs of another.
#
# The ingestion code talks to the rules‑portal Express server via the following
# HTTP routes exposed in `graphRoutes.js`. The base URL is controlled by the
# environment variable `RULES_PORTAL_BASE_URL` and defaults to
# `http://localhost:3000`:
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
#   1. Creating/merging LogicStep, CodeFunction, CodeFile and Parameter nodes.
#   2. Wiring them together with IMPLEMENTED_BY, PART_OF, INPUT, OUTPUT and
#      LINKS_TO relationships using the graph routes above.

# --- Neo4j / graph API configuration (lazy, best‑effort) ---
GRAPH_BASE_URL = os.getenv("RULES_PORTAL_BASE_URL", "http://localhost:443").rstrip("/")
GRAPH_ENABLED_ENV = os.getenv("RULES_PORTAL_GRAPH_ENABLED", "true").strip().lower()
GRAPH_ENABLED = GRAPH_ENABLED_ENV in {"1", "true", "yes", "on"}

# --- Run-scoped override to forcibly disable graph calls ---
GRAPH_FORCE_DISABLED = False

def set_graph_disabled(disabled: bool = True) -> None:
    """Override graph population for the current process/run."""
    global GRAPH_FORCE_DISABLED
    GRAPH_FORCE_DISABLED = bool(disabled)

_GRAPH_STATUS_CHECKED = False
_GRAPH_AVAILABLE = False

# Simple in‑memory caches so we reuse nodes instead of creating duplicates
_LOGICSTEP_CACHE: Dict[str, Any] = {}
_CODEFUNC_CACHE: Dict[Tuple[str, str], Any] = {}
_CODEFILE_CACHE: Dict[str, Any] = {}
_PARAM_CACHE: Dict[Any, Any] = {}


def _graph_is_enabled(logger: logging.Logger) -> bool:
    """Return True if graph population is enabled and the backend is reachable.

    This is lazy: we only hit /api/graph/status once per process and then cache
    the result. All failures are logged as warnings but do not stop ingestion.
    """
    global _GRAPH_STATUS_CHECKED, _GRAPH_AVAILABLE

    if GRAPH_FORCE_DISABLED:
        return False

    if not GRAPH_ENABLED:
        return False

    if _GRAPH_STATUS_CHECKED:
        return _GRAPH_AVAILABLE

    _GRAPH_STATUS_CHECKED = True
    status_url = f"{GRAPH_BASE_URL}/api/graph/status"
    try:
        resp = requests.get(status_url, timeout=2)
        if resp.ok:
            data = resp.json()
            _GRAPH_AVAILABLE = bool(data.get("up"))
            if not _GRAPH_AVAILABLE:
                logger.warning(
                    "[warn] Neo4j graph API reported up=false at %s: %s",
                    status_url,
                    data.get("message"),
                )
        else:
            logger.warning(
                "[warn] Neo4j graph status check failed (%s): HTTP %s",
                status_url,
                resp.status_code,
            )
            _GRAPH_AVAILABLE = False
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "[warn] Neo4j graph status check error at %s: %s",
            status_url,
            exc,
        )
        _GRAPH_AVAILABLE = False

    return _GRAPH_AVAILABLE


def _graph_create_node(labels: List[str], properties: Dict[str, Any], logger: logging.Logger,
                       cache: Optional[Dict[Any, Any]] = None, cache_key: Any = None) -> Optional[Any]:
    """Create a node via /api/graph/node, with optional caching by cache_key.

    Returns the Neo4j internal id (identity) on success, or None on failure.
    """
    if cache is not None and cache_key is not None:
        existing = cache.get(cache_key)
        if existing is not None:
            return existing

    if not labels:
        return None

    url = f"{GRAPH_BASE_URL}/api/graph/node"
    payload = {"labels": labels, "properties": properties or {}}

    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            logger.warning("[warn] Neo4j /api/graph/node returned success=false: %s", data)
            return None
        node = (data.get("node") or {})
        identity = node.get("identity")
        if cache is not None and cache_key is not None and identity is not None:
            cache[cache_key] = identity
        return identity
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[warn] Failed to create graph node %s: %s", properties, exc)
        return None


def _graph_create_relationship(from_id: Any, to_id: Any, rel_type: str,
                               properties: Optional[Dict[str, Any]],
                               logger: logging.Logger) -> None:
    """Create a relationship via /api/graph/relationship (best‑effort)."""
    if from_id is None or to_id is None or not rel_type:
        return

    url = f"{GRAPH_BASE_URL}/api/graph/relationship"
    payload = {
        "fromId": from_id,
        "toId": to_id,
        "type": rel_type,
        "properties": properties or {},
    }

    try:
        resp = requests.post(url, json=payload, timeout=5)
        # We log non‑2xx but do not raise to keep ingestion resilient.
        if not resp.ok:
            logger.warning(
                "[warn] Failed to create relationship %s -> %s (%s): HTTP %s",
                from_id,
                to_id,
                rel_type,
                resp.status_code,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "[warn] Exception while creating relationship %s -> %s (%s): %s",
            from_id,
            to_id,
            rel_type,
            exc,
        )



# --- Helper: best-effort lookup of existing LogicStep node by id and name ---
def _graph_find_logicstep_node(rule_id: str, name: Optional[str], logger: logging.Logger) -> Optional[Any]:
    """
    Best-effort lookup of an existing LogicStep node by id and name.

    Returns the Neo4j internal identity if found, or None otherwise.
    """
    if not rule_id:
        return None

    # If graph is globally disabled/unavailable, skip lookup
    if not _graph_is_enabled(logger):
        return None

    url = f"{GRAPH_BASE_URL}/api/graph/run"
    query = (
        "MATCH (ls:LogicStep) "
        "WHERE ls.id = $id "
        "AND toLower(ls.name) = toLower($name) "
        "RETURN id(ls) AS identity "
        "LIMIT 1"
    )
    payload = {
        "query": query,
        "parameters": {
            "id": rule_id,
            "name": (name or ""),
        },
    }

    try:
        resp = requests.post(url, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            logger.trace(
                "[trace] LogicStep lookup returned success=false for id=%s name='%s': %s",
                rule_id,
                name,
                data,
            )
            return None
        records = data.get("records") or []
        if not records:
            return None
        first = records[0] or {}
        identity = first.get("identity")
        if identity is None:
            logger.trace(
                "[trace] LogicStep lookup returned record without identity for id=%s name='%s': %s",
                rule_id,
                name,
                first,
            )
            return None
        return identity
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "[warn] Failed to lookup LogicStep in graph for id=%s name='%s': %s",
            rule_id,
            name,
            exc,
        )
        return None


def _ensure_logicstep_node(rule: Dict[str, Any], logger: logging.Logger) -> Tuple[Optional[Any], bool]:
    """Ensure there is a LogicStep node for this rule.

    Returns a tuple of (identity, existed) where:
      • identity is the Neo4j internal id (or None on failure)
      • existed is True if we matched an existing node (cache or graph),
        and False if we had to create a new node.
    """
    rule_id = rule.get("id")
    if not rule_id:
        return None, False

    # Check in-process cache first
    cached = _LOGICSTEP_CACHE.get(rule_id)
    if cached is not None:
        # From our perspective, this "existed" already in this run
        return cached, True

    # Best-effort lookup in existing graph to avoid duplicate LogicStep nodes
    existing_id = _graph_find_logicstep_node(rule_id, rule.get("rule_name"), logger)
    if existing_id is not None:
        _LOGICSTEP_CACHE[rule_id] = existing_id
        return existing_id, True

    # Fall back to creating a new LogicStep node
    props = {
        "id": rule_id,
        "name": rule.get("rule_name"),
        "kind": rule.get("kind") or "",
        "owner": rule.get("owner") or "",
        "component": rule.get("component") or "",
        "timestamp": rule.get("timestamp") or "",
    }
    identity = _graph_create_node(["LogicStep"], props, logger, _LOGICSTEP_CACHE, rule_id)
    return identity, False


def _ensure_codefile_node(
    path: str,
    logger: logging.Logger,
    root_dir: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Optional[Any]:
    if not path:
        return None
    norm_path = os.path.normpath(path)
    cached = _CODEFILE_CACHE.get(norm_path)
    if cached is not None:
        return cached
    props = {"path": norm_path, "name": os.path.basename(norm_path)}
    if source_path:
        props["sourcePath"] = source_path
    if root_dir:
        props["rootDir"] = root_dir
    return _graph_create_node(["CodeFile"], props, logger, _CODEFILE_CACHE, norm_path)


def _ensure_codefunc_node(name: str, file_path: Optional[str], logger: logging.Logger) -> Optional[Any]:
    if not name:
        return None
    key_path = os.path.normpath(file_path) if file_path else ""
    key = (key_path, name)
    cached = _CODEFUNC_CACHE.get(key)
    if cached is not None:
        return cached
    props = {"name": name}
    if key_path:
        props["filePath"] = key_path
    return _graph_create_node(["CodeFunction"], props, logger, _CODEFUNC_CACHE, key)


def _ensure_param_node(rule_id: str, name: str, direction: str, typ: str, logger: logging.Logger) -> Optional[Any]:
    if not rule_id or not name:
        return None
    key = (rule_id, name)
    cached = _PARAM_CACHE.get(key)
    if cached is not None:
        return cached

    dir_norm = (direction or "").lower()
    prefix = ""
    if dir_norm == "input":
        prefix = "in: "
    elif dir_norm == "output":
        prefix = "out: "
    display_name = f"{prefix}{name}" if prefix else name

    props = {
        "stepId": rule_id,
        "name": display_name,
        "direction": direction or "",
        "type": typ or "",
    }
    return _graph_create_node(["Parameter"], props, logger, _PARAM_CACHE, key)


def _populate_graph_for_rule(rule: Dict[str, Any], logger: logging.Logger) -> None:
    """Populate the Neo4j knowledge graph for a single rule (best‑effort).

    This function is pure side‑effect: it never raises and never affects the
    core ingestion path if Neo4j is unavailable or the API fails.
    """
    if not _graph_is_enabled(logger):
        return

    rule_id = rule.get("id")
    if not rule_id:
        return

    logic_id, existed = _ensure_logicstep_node(rule, logger)
    # If the LogicStep already existed in the graph, we assume its CodeFile,
    # CodeFunction and Parameter nodes (and relationships) are already present.
    # In that case, skip re-creating them to avoid duplicates.
    if logic_id is None:
        return
    if existed:
        return

    # Code function and file
    code_function = rule.get("code_function") or ""
    code_file = rule.get("code_file") or ""
    code_file_root_dir = rule.get("rootDir") or rule.get("root_dir")
    code_file_source_path = rule.get("sourcePath") or rule.get("source_path")

    file_id = (
        _ensure_codefile_node(
            code_file,
            logger,
            root_dir=code_file_root_dir,
            source_path=code_file_source_path,
        )
        if code_file
        else None
    )
    func_id = _ensure_codefunc_node(code_function, code_file, logger) if code_function else None

    if logic_id is not None and func_id is not None:
        _graph_create_relationship(logic_id, func_id, "IMPLEMENTED_BY", {}, logger)
    if func_id is not None and file_id is not None:
        _graph_create_relationship(func_id, file_id, "PART_OF", {}, logger)

    # DMN inputs/outputs become Parameter nodes
    dmn_inputs = rule.get("dmn_inputs") or []
    dmn_outputs = rule.get("dmn_outputs") or []

    for p in dmn_inputs:
        pname = str(p.get("name") or "").strip()
        if not pname:
            continue
        ptype = str(p.get("type") or "")
        pid = _ensure_param_node(rule_id, pname, "input", ptype, logger)
        if logic_id is not None and pid is not None:
            _graph_create_relationship(logic_id, pid, "INPUT", {}, logger)

    for p in dmn_outputs:
        pname = str(p.get("name") or "").strip()
        if not pname:
            continue
        ptype = str(p.get("type") or "")
        pid = _ensure_param_node(rule_id, pname, "output", ptype, logger)
        if logic_id is not None and pid is not None:
            _graph_create_relationship(logic_id, pid, "OUTPUT", {}, logger)

    # Cross‑rule links (Parameter -> Parameter)
    links = rule.get("links") or []
    for link in links:
        from_step_id = link.get("from_step_id")
        from_output = link.get("from_output")
        to_input = link.get("to_input")

        if not from_step_id or not from_output or not to_input:
            continue

        # Source parameter: output of the upstream rule
        src_param_id = _ensure_param_node(
            str(from_step_id),
            str(from_output),
            "output",
            "",
            logger,
        )

        # Target parameter: input of the current rule
        dst_param_id = _ensure_param_node(
            rule_id,
            str(to_input),
            "input",
            "",
            logger,
        )

        if src_param_id is not None and dst_param_id is not None:
            _graph_create_relationship(
                src_param_id,
                dst_param_id,
                "LINKS_TO",
                {"kind": link.get("kind") or ""},
                logger,
            )


def _dedupe_key(rule_name, timestamp):
    """Build a stable key from rule_name and timestamp with normalization."""
    if rule_name is None or timestamp is None:
        return None
    rn = str(rule_name).strip()
    ts = str(timestamp).strip()
    # Normalize timestamp to drop microseconds
    if "." in ts:
        ts = ts.split(".")[0] + "Z"
    return (rn, ts)


def heading_text(line: str) -> str:
    """Extract clean heading text from a markdown heading line.
    Removes leading/trailing '#' and surrounding whitespace/markers.
    Also strips leading 'Rule Name:' / 'Name:' labels.
    """
    s = (line or "").strip()
    # remove leading hashes and spaces
    s = re.sub(r"^\s*#{1,6}\s*", "", s)
    # remove trailing hashes and spaces
    s = re.sub(r"\s*#{1,6}\s*$", "", s)
    s = s.strip(" *-\t")
    # If the heading still includes a leading label like "Rule Name:", strip it.
    s = re.sub(r"^(?:Rule\s+Name|Name):\s*", "", s, flags=re.IGNORECASE)
    return s


def _normalize_rule_doc_text(t: str) -> str:
    # Normalize various "Rule Name" heading formats to a consistent "## " heading
    t = re.sub(r"#{2,6}\s*\d+\.\s*\*\*Rule Name:\*\*\s*", "## ", t)   # e.g., "### 1. **Rule Name:**"
    t = re.sub(r"#{2,6}\s*\*\*Rule Name:\*\*\s*", "## ", t)            # e.g., "### **Rule Name:**"
    t = re.sub(r"#{2,6}\s*\d+\.\s*Rule Name:\s*", "## ", t)            # e.g., "### 1. Rule Name:"
    t = re.sub(r"#{2,6}\s*Rule Name:\s*", "## ", t)                    # e.g., "### Rule Name:"

    # Newer "Name:" heading variants → also normalize to "## "
    t = re.sub(r"#{2,6}\s*\d+\.\s*\*\*Name:\*\*\s*", "## ", t)         # e.g., "### 1. **Name:**"
    t = re.sub(r"#{2,6}\s*\*\*Name:\*\*\s*", "## ", t)                 # e.g., "### **Name:**"
    t = re.sub(r"#{2,6}\s*\d+\.\s*Name:\s*", "## ", t)                 # e.g., "### 1. Name:"
    t = re.sub(r"#{2,6}\s*Name:\s*", "## ", t)                         # e.g., "### Name:"

    t = re.sub(r"\n---+\n", "\n", t)                                   # Remove separators

    # Also handle non-heading inline forms
    t = re.sub(r"(?m)^\s*\*\*Rule Name:\*\*\s*", "## ", t)
    t = re.sub(r"(?m)^\s*Rule Name:\s*", "## ", t)
    t = re.sub(r"(?m)^\s*\*\*Name:\*\*\s*", "## ", t)
    t = re.sub(r"(?m)^\s*Name:\s*", "## ", t)

    return t


def _parse_dmn_io_decl(line: str) -> dict:
    """
    Parse a single DMN Inputs/Outputs line of the form:
      - Name : Type [Allowed Values: v1, v2, "v 3"]
    Returns a dict with keys: name, type, and optional allowedValues (list).
    Backwards compatible with lines that omit type or allowed values.
    """
    s = (line or "").strip()
    # Strip bullet markers if present
    s = s.lstrip("-*").strip()
    # Try full pattern with Allowed Values
    m = re.match(
        r'^(?P<name>[^:]+?)\s*:\s*(?P<type>[^\[\]]+?)(?:\s*\[\s*Allowed\s*Values\s*:\s*(?P<values>.+?)\s*\])?\s*$',
        s,
        flags=re.IGNORECASE,
    )
    if m:
        name = m.group("name").strip().strip("`")
        typ = (m.group("type") or "").strip().strip("`")
        out = {"name": name, "type": typ}
        vals = m.group("values")
        if vals:
            parts = []
            for tok in vals.split(","):
                t = tok.strip()
                if (len(t) >= 2) and ((t[0] == t[-1]) and t[0] in {'"', "'"}):
                    t = t[1:-1]
                if t:
                    parts.append(t)
            if parts:
                out["allowedValues"] = parts
        return out
    # Fallbacks for legacy forms:
    if ":" in s:
        name, typ = s.split(":", 1)
        return {"name": name.strip().strip("`"), "type": typ.strip().strip("`")}
    field = s.strip().strip("`")
    return {"name": field, "type": ""}


def _has_material_change(existing: dict, new_fields: dict) -> bool:
    """
    Determine whether DMN-relevant fields have materially changed.
    Compares hit policy, inputs, outputs (including allowedValues), and table text.
    Returns True if different; False if effectively the same or if existing is missing.
    """
    if not isinstance(existing, dict):
        return True

    def _norm_io(lst):
        out = []
        for x in (lst or []):
            if not isinstance(x, dict):
                continue
            out.append(
                {
                    "name": str(x.get("name", "")),
                    "type": str(x.get("type", "")),
                    "allowedValues": [str(v) for v in (x.get("allowedValues") or [])],
                }
            )
        return out

    ex_hp = str(existing.get("dmn_hit_policy", "") or "")
    ex_in = _norm_io(existing.get("dmn_inputs"))
    ex_out = _norm_io(existing.get("dmn_outputs"))
    ex_tbl = str(existing.get("dmn_table", "") or "").strip()

    new_hp = str(new_fields.get("dmn_hit_policy", "") or "")
    new_in = _norm_io(new_fields.get("dmn_inputs"))
    new_out = _norm_io(new_fields.get("dmn_outputs"))
    new_tbl = str(new_fields.get("dmn_table", "") or "").strip()

    if ex_hp != new_hp:
        return True
    if ex_in != new_in:
        return True
    if ex_out != new_out:
        return True
    if ex_tbl != new_tbl:
        return True
    return False


def add_rules_from_text(
    doc_text: str,
    section_timestamp: str,
    *,
    existing_by_name: Dict[str, dict],
    name_to_id: Dict[str, str],
    seen: Set[Tuple[str, str]],
    force_load: bool,
    logger: logging.Logger,
    ingest_owner: str,
    ingest_component: str,
    audit_add,
    updated_count: int = 0,
    new_count: int = 0,
    considered_count: int = 0,
    allowed_names: Optional[Set[str]] = None,
    root_dir: Optional[str] = None,
    source_path: Optional[str] = None,
) -> Tuple[List[dict], int, int, int]:
    """Parse markdown rule sections and return new/updated rule records plus updated counters."""
    t = _normalize_rule_doc_text(doc_text or "")
    rule_sections = re.split(r"(?m)^\s{0,3}#{1,6}\s+", t.strip())[1:]
    new_rules: List[dict] = []
    name_counts: Dict[str, int] = {}
    # Track duplicates within this document by (base rule name, code file)
    seen_name_codefile: Set[Tuple[str, str]] = set()

    for idx, section in enumerate(rule_sections, start=1):
        considered_count += 1
        try:
            lines = section.strip().splitlines()
            logger.trace("[trace] section %d: first line='%s'", idx, (lines[0] if lines else ""))
            rule_name = heading_text(lines[0] if lines else "")
            if not rule_name or rule_name.lower() in {"rule name", "rule-name"}:
                rn_match = re.search(r"\*\*Rule Name:\*\*\s*(.+)", section)
                if rn_match:
                    rule_name = rn_match.group(1).strip()

            if not rule_name or rule_name.lower() in {"rule name", "rule-name"}:
                rn2 = re.search(r"\*\*Name:\*\*\s*(.+)", section)
                if rn2:
                    rule_name = rn2.group(1).strip()

            logger.trace("[trace] section %d: parsed rule_name (pre-suffix)='%s'", idx, rule_name)

            # Preserve the base heading text before any suffixing
            base_name = rule_name

            # New optional Kind field (Decision | BKM)
            kind = ""
            m_kind = re.search(r"\*\*Kind:\*\*\s*([A-Za-z]+)", section)
            if m_kind:
                kind = m_kind.group(1).strip()

            # code_file, code_lines, code_function
            code_file = ""
            code_lines: Optional[List[int]] = None

            m_codefile_inline = re.search(
                r"\*\*Code\s*Block:\*\*\s*`?([^`\n]+)`?", section, re.IGNORECASE
            )
            if m_codefile_inline:
                code_file = m_codefile_inline.group(1).strip()
            else:
                m_codefile_fileline = re.search(
                    r"(?mi)^\s*File:\s*`?([^`\n]+)`?", section
                )
                if m_codefile_fileline:
                    code_file = m_codefile_fileline.group(1).strip()
            if code_file:
                code_file = code_file.replace("`", "").strip()

            m_codelines = re.search(
                r"\bLine(?:s)?\s*:??\s*(\d+)(?:\s*[\-\u2013\u2014]\s*(\d+))?",
                section,
                re.IGNORECASE,
            )
            if m_codelines:
                try:
                    start_line = int(m_codelines.group(1))
                    end_line = m_codelines.group(2)
                    if end_line is not None:
                        end_line = int(end_line)
                    else:
                        end_line = start_line
                    code_lines = [start_line, end_line]
                except Exception:
                    code_lines = None

            code_function = ""
            m_codefunc = re.search(
                r"(?mi)^\s*Function\b\s*[:\-\u2013\u2014]*\s*`?([^`\n]+)`?", section
            )
            if m_codefunc:
                code_function = m_codefunc.group(1).strip()
                code_function = re.sub(
                    r"^[\s:;\-\u2013\u2014]+", "", code_function
                ).strip()

            # If we have both a base name and a code file, treat (name, code_file) duplicates
            # within the same document as true duplicates and skip them.
            if base_name and code_file:
                cf_key = code_file or ""
                dup_key = (base_name, cf_key)
                if dup_key in seen_name_codefile:
                    logger.info(
                        "[info] Skipping duplicate rule '%s' in same code file '%s' within document",
                        base_name,
                        cf_key,
                    )
                    audit_add(
                        "rule",
                        path=cf_key,
                        source="input_doc",
                        decision="rejected",
                        reason="duplicate name in same code file",
                        tests={"base_name": base_name},
                        derived={},
                    )
                    continue
                seen_name_codefile.add(dup_key)

            # Now make rule names unique within this document by appending a counter,
            # but only after we've handled same-file duplicates.
            rule_name = base_name
            if rule_name:
                count = name_counts.get(rule_name, 0) + 1
                name_counts[rule_name] = count
                if count > 1:
                    # First occurrence keeps the plain name; subsequent ones get " (2)", " (3)", etc.
                    rule_name = f"{rule_name} ({count})"

            logger.trace("[trace] section %d: final rule_name='%s'", idx, rule_name)

            # If we are restricting to an allowed set of names, apply it after suffixing
            if allowed_names is not None and rule_name not in allowed_names:
                logger.trace(
                    "[trace] section %d: rule_name '%s' not in allowed_names (size=%s); skipping",
                    idx,
                    rule_name,
                    len(allowed_names),
                )
                continue

            # Purpose
            purpose_match = re.search(
                r"\*\*(?:Rule\s+)?Purpose:\*\*\s*\n?(.*?)(?=\n\*\*(?:Rule\s+)?Spec|\n\*\*Specification|\n\*\*Code Block|\n\*\*Example|$)",
                section,
                re.DOTALL | re.IGNORECASE,
            )
            rule_purpose = purpose_match.group(1).strip() if purpose_match else ""

            # Spec
            spec_marker = re.search(
                r"\*\*(?:Rule\s+)?Spec:\*\*|\*\*Specification:\*\*", section, re.IGNORECASE
            )
            if spec_marker:
                start = spec_marker.end()
                next_marker = re.search(
                    r"\n\*\*(Code Block|Example):\*\*|\n(?:\*{0,2}\s*)?DMN\s*:\s*(?:\*{0,2})?",
                    section[start:],
                    re.DOTALL | re.IGNORECASE,
                )
                end = next_marker.start() + start if next_marker else len(section)
                rule_spec = section[start:end].strip()
            else:
                rule_spec = ""

            code_match = re.search(r"```[a-zA-Z]*\n(.*?)```", section, re.DOTALL)
            code_block = code_match.group(1).strip() if code_match else ""

            example_match = re.search(
                r"\*\*Example:\*\*\s*\n?(.*?)(?=\n(?:\*{0,2}\s*)?DMN\s*:\s*(?:\*{0,2})?|\n## |\Z)",
                section,
                re.DOTALL | re.IGNORECASE,
            )
            example = example_match.group(1).strip() if example_match else ""
            if example:
                example = re.split(
                    r"\n(?:\*{0,2}\s*)?DMN\s*:\s*(?:\*{0,2})?", example, flags=re.IGNORECASE
                )[0].strip()

            # Dedupe key for this section
            k = _dedupe_key(rule_name, section_timestamp)
            logger.trace("[trace] section %d: dedupe_key=%s", idx, k)

            # DMN
            dmn_hit_policy = ""
            dmn_inputs: List[dict] = []
            dmn_outputs: List[dict] = []
            dmn_table = ""

            dmn_match = re.search(
                r"(?:^|\n)(?:\*{0,2}\s*)?DMN\s*:\s*\n?(.*?)(?=\n## |\Z)",
                section,
                re.DOTALL | re.IGNORECASE,
            )
            if dmn_match:
                raw_dmn = dmn_match.group(1).strip()
                m_code = re.search(r"```.*?\n(.*?)```", raw_dmn, re.DOTALL)
                dmn_body = m_code.group(1).strip() if m_code else raw_dmn
                dmn_body = re.sub(r"`+", "", dmn_body)
                dmn_body = re.sub(r"\*\*", "", dmn_body)

                m_hp = re.search(r"Hit\s*Policy\s*:\s*([A-Za-z_]+)", dmn_body, re.IGNORECASE)
                if m_hp:
                    dmn_hit_policy = m_hp.group(1).strip()

                m_inputs = re.search(
                    r"Inputs\s*:\s*\n(?P<block>(?:\s*[-*]\s*.*(?:\n|$))+)",
                    dmn_body,
                    re.IGNORECASE,
                )
                if m_inputs:
                    for ln in m_inputs.group("block").splitlines():
                        ln = ln.strip()
                        if not (ln.startswith("-") or ln.startswith("*")):
                            continue
                        parsed = _parse_dmn_io_decl(ln)
                        dmn_inputs.append(parsed)

                m_outputs = re.search(
                    r"Outputs\s*:\s*\n(?P<block>(?:\s*[-*]\s*.*(?:\n|$))+)",
                    dmn_body,
                    re.IGNORECASE,
                )
                if m_outputs:
                    for ln in m_outputs.group("block").splitlines():
                        ln = ln.strip()
                        if not (ln.startswith("-") or ln.startswith("*")):
                            continue
                        parsed = _parse_dmn_io_decl(ln)
                        dmn_outputs.append(parsed)

                lines2 = [ln.rstrip() for ln in dmn_body.splitlines()]
                table_lines: List[str] = []
                in_table = False
                for ln in lines2:
                    if ("|" in ln) or ("+" in ln) or re.search(r"-{2,}", ln):
                        table_lines.append(ln.rstrip())
                        in_table = True
                    else:
                        if in_table:
                            break
                dmn_table = "\n".join(table_lines).strip()

            # Dedup by (rule_name, timestamp)
            if k in seen and not force_load:
                logger.trace(
                    "[trace] section %d: dedupe key %s already seen and force_load=False; skipping",
                    idx,
                    k,
                )
                audit_add(
                    "rule",
                    path="section",
                    source="input_doc",
                    decision="rejected",
                    reason="duplicate or older",
                    tests={"key": str(k), "force": force_load},
                    derived={},
                )
                continue

            existing = existing_by_name.get(rule_name)
            if existing:
                old_ts = existing.get("timestamp")
                logger.trace(
                    "[trace] section %d: existing rule found for '%s' (id=%s, ts=%s)",
                    idx,
                    rule_name,
                    existing.get("id"),
                    old_ts,
                )
                new_fields = {
                    "dmn_hit_policy": dmn_hit_policy,
                    "dmn_inputs": dmn_inputs,
                    "dmn_outputs": dmn_outputs,
                    "dmn_table": dmn_table,
                }
                is_material_change = _has_material_change(existing, new_fields)
                logger.trace(
                    "[trace] section %d: material_change=%s (old_ts=%s, new_ts=%s)",
                    idx,
                    is_material_change,
                    old_ts,
                    section_timestamp,
                )
                if (
                    not force_load
                    and old_ts
                    and old_ts >= section_timestamp
                    and not is_material_change
                ):
                    audit_add(
                        "rule",
                        path="section",
                        source="input_doc",
                        decision="rejected",
                        reason="duplicate or older",
                        tests={"key": str(k), "force": force_load, "material_change": False},
                        derived={},
                    )
                    continue
                if (
                    not force_load
                    and old_ts
                    and old_ts >= section_timestamp
                    and is_material_change
                ):
                    logger.info(
                        "[info] material change detected for '%s' — updating despite same/older timestamp (allowedValues or DMN changed)",
                        rule_name,
                    )
                rule_id = (
                    existing.get("id")
                    or name_to_id.get(rule_name)
                    or str(uuid.uuid4())
                )
                name_to_id[rule_name] = rule_id
                updated_count += 1
                logger.trace(
                    "[trace] section %d: updating existing rule id=%s name='%s'",
                    idx,
                    rule_id,
                    rule_name,
                )
            else:
                rule_id = name_to_id.get(rule_name) or str(uuid.uuid4())
                name_to_id[rule_name] = rule_id
                new_count += 1
                logger.trace(
                    "[trace] section %d: creating NEW rule id=%s name='%s'",
                    idx,
                    rule_id,
                    rule_name,
                )

            seen.add(k)


            # Links
            links: List[dict] = []
            m_links = re.search(
                r"(?mi)^\s*\*\*Links:\*\*\s*\n(?P<body>.*?)(?=\n## |\Z)",
                section,
                re.DOTALL,
            )
            if m_links:
                body = m_links.group("body").strip()
                if body.lower() == "none":
                    links = []
                else:
                    for ln in [x.strip() for x in body.splitlines() if x.strip()]:
                        mm = re.match(
                            r"^(.+?)\.(.+?)\s*->\s*(.+?)\s*\[kind=([^\]]+)\]\s*$",
                            ln,
                            flags=re.IGNORECASE,
                        )
                        if mm:
                            _from_step = mm.group(1).strip()
                            _from_output = mm.group(2).strip()
                            rhs = mm.group(3).strip()
                            _kind = mm.group(4).strip()

                            _from_id = name_to_id.get(_from_step)
                            if not _from_id:
                                ex = existing_by_name.get(_from_step)
                                if isinstance(ex, dict):
                                    _from_id = ex.get("id")

                            if _from_id:
                                links.append(
                                    {
                                        "from_step_id": _from_id,
                                        "from_output": _from_output,
                                        "to_input": rhs,
                                        "kind": _kind,
                                    }
                                )
                            if not _from_id:
                                logger.trace(
                                    "[trace] skipped link: could not resolve id for from_step='%s' (owner rule='%s')",
                                    _from_step,
                                    rule_name,
                                )
                        else:
                            links.append({"raw": ln})

            rule_rec: Dict[str, Any] = {
                "rule_name": rule_name,
                "rule_purpose": rule_purpose,
                "rule_spec": rule_spec,
                "code_block": code_block,
                "code_file": code_file,
                "code_lines": code_lines,
                "code_function": code_function,
                "example": example,
                "dmn_hit_policy": dmn_hit_policy,
                "dmn_inputs": dmn_inputs,
                "dmn_outputs": dmn_outputs,
                "dmn_table": dmn_table,
                "timestamp": section_timestamp,
                "id": rule_id,
                "owner": ingest_owner,
                "component": ingest_component,
                "kind": kind,
                "links": links,
            }
            # Attach optional file-origin metadata so CodeFile nodes can receive rootDir/sourcePath
            if root_dir:
                rule_rec["rootDir"] = root_dir
            if source_path:
                rule_rec["sourcePath"] = source_path

            existing = existing_by_name.get(rule_name)
            if existing and existing.get("archived") is not None:
                if existing.get("archived") is True:
                    rule_rec["archived"] = False
                    logger.info(
                        "[info] Unarchived rule on update: %s", rule_name
                    )
                else:
                    rule_rec["archived"] = existing["archived"]

            if force_load and existing:
                for src_key in (
                    "rule_category",
                    "category",
                    "category_id",
                    "categoryId",
                ):
                    val = existing.get(src_key)
                    if val not in (None, "", []):
                        rule_rec["rule_category"] = val
                        break

            # Populate Neo4j knowledge graph for this rule (best‑effort, non‑fatal)
            try:
                _populate_graph_for_rule(rule_rec, logger)
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(
                    "[warn] Graph population failed for rule '%s': %s",
                    rule_name,
                    e,
                )

            audit_add(
                "rule",
                path=rule_rec.get("code_file") or "",
                source="input_doc",
                decision=("updated" if existing else "new"),
                reason=("updated" if existing else "new"),
                tests={"has_code_block": bool(code_block)},
                derived={
                    "id": rule_id,
                    "rule_name": rule_name,
                    "kind": kind,
                    "links": len(links),
                },
            )
            logger.info(
                "[info] ✓ %s rule: %s",
                ("updated" if existing else "new"),
                rule_name,
            )
            new_rules.append(rule_rec)

            logger.trace(
                "[trace] section %d: appended rule id=%s name='%s' kind='%s'",
                idx,
                rule_id,
                rule_name,
                kind,
            )
        except Exception as e:
            logger.error(
                "[error] Failed to parse a rule section: %s", e
            )

    logger.info(
        "[summary] add_rules_from_text: considered=%d updated=%d new=%d emitted=%d",
        considered_count,
        updated_count,
        new_count,
        len(new_rules),
    )
    return new_rules, updated_count, new_count, considered_count