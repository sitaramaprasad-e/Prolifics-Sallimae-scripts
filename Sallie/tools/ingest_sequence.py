# Here is the spec for what to create in Neo4j using the rules‑portal graph routes.
#
# Node labels:
# - LogicStep   (one per logic)
# - CodeFunction
# - CodeFile
#    + sourcePath
#    + rootDir
# - Parameter
# - Message
#   + stepNumber
#   + fromParticipant
#   + toParticipant
#   + functionName
# - Sequence
#    + diagramPath
#    + ingestionTimestamp
#
# Relationship types:
# - IMPLEMENTED_BY (LogicStep -> CodeFunction)
# - PART_OF       (CodeFunction -> CodeFile), (Message -> Sequence)
# - INPUT         (LogicStep -> Parameter for inputs)
# - OUTPUT        (LogicStep -> Parameter for outputs)
# - LINKS_TO      (Parameter -> Parameter for hierarchy links between steps)
# - SEQUENCED_BY (CodeFunction -> Message)
#
# When ingesting, we populate the above structure, creating messages and sequences, and linking logic steps to messages and code functions to messages.
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
# In summary, sequence ingestion is responsible for:
#   1. Creating Message and Sequence nodes for the parsed sequence diagram.
#   2. Wiring Message nodes to Sequence via PART_OF and to CodeFunction via SEQUENCED_BY using the graph routes above.

#!/usr/bin/env python3
import warnings
# Suppress the LibreSSL/OpenSSL compatibility warning from urllib3 v2
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
    module="urllib3"
)

# ===== Implementation =====
import os
import sys
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
import logging

# ===== TRACE logger setup (same as ingest_logics.py) =====
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def _trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)

logging.Logger.trace = _trace

def _setup_logger(verbose: bool = True):
    level = TRACE_LEVEL_NUM if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

LOG = logging.getLogger("ingest_sequence")
_setup_logger(verbose=True)

# ===== Base URL =====
# Default base URL for the rules-portal graph API. This is typically overridden
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
        LOG.warning(f"[warn] GET {url} returned {resp.status_code}: {resp.text}")
    except Exception as e:
        LOG.warning(f"[warn] GET {url} failed: {e}")
    return None

def _post(path: str, payload: dict) -> Optional[dict]:
    url = RULES_PORTAL_BASE_URL + path
    try:
        resp = requests.post(url, json=payload, timeout=12, verify=GRAPH_VERIFY)
        if resp.status_code == 200:
            return resp.json()
        LOG.warning(f"[warn] POST {url} returned {resp.status_code}: {resp.text}")
    except Exception as e:
        LOG.warning(f"[warn] POST {url} failed: {e}")
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

def create_node(labels: list, properties: dict) -> Optional[int]:
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

def create_relationship(from_id: int, to_id: int, rel_type: str, properties: Optional[dict] = None) -> Optional[int]:
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

def _to_int_id(value) -> Optional[int]:
    # Accepts: int, or dict like {"low": 123, "high": 0}
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        low = value.get("low")
        high = value.get("high", 0)
        if isinstance(low, int) and isinstance(high, int):
            # Handles 53-bit Neo4j ints, but we expect small numbers
            return int(low) + (int(high) << 32)
    try:
        return int(value)
    except Exception:
        return None

# ===== Helper: Delete existing Sequence/Message for a diagramPath =====
def _delete_existing_sequence(diagram_path: str) -> None:
    """
    Delete any existing Sequence and associated Message nodes for the given diagramPath.
    """
    LOG.info("[info] Checking for existing Sequence/Message set for diagramPath=%s", diagram_path)
    cypher = """
    MATCH (s:Sequence {diagramPath: $diagramPath})
    OPTIONAL MATCH (m:Message)-[:PART_OF]->(s)
    WITH DISTINCT s, m
    DETACH DELETE s, m
    RETURN count(DISTINCT s) AS sequencesDeleted, count(DISTINCT m) AS messagesDeleted
    """
    params = {"diagramPath": diagram_path}
    results = run_cypher(cypher, params)
    if not results or not isinstance(results, (list, tuple)) or len(results) == 0:
        LOG.info("[info] Nothing to delete for diagramPath=%s", diagram_path)
        return
    row = results[0]
    sequences_deleted = None
    messages_deleted = None
    # Try dict form first
    if isinstance(row, dict):
        sequences_deleted = row.get("sequencesDeleted")
        messages_deleted = row.get("messagesDeleted")
        # Try Neo4j-like shape: { "row": [a, b], ... }
        if (sequences_deleted is None or messages_deleted is None) and isinstance(row.get("row"), (list, tuple)) and len(row["row"]) >= 2:
            sequences_deleted = row["row"][0]
            messages_deleted = row["row"][1]
    elif isinstance(row, (list, tuple)) and len(row) >= 2:
        sequences_deleted = row[0]
        messages_deleted = row[1]
    # Normalize to int, fallback to 0
    try:
        sequences_deleted_int = _to_int_id(sequences_deleted)
        if sequences_deleted_int is None:
            sequences_deleted_int = 0
    except Exception:
        sequences_deleted_int = 0
    try:
        messages_deleted_int = _to_int_id(messages_deleted)
        if messages_deleted_int is None:
            messages_deleted_int = 0
    except Exception:
        messages_deleted_int = 0
    LOG.info(
        "[info] Deleted %d existing Sequence node(s) and %d Message node(s) for diagramPath=%s",
        sequences_deleted_int,
        messages_deleted_int,
        diagram_path,
    )

#
# ===== Spec discovery & load (mirrors pcpt_pipeline) =====

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
        LOG.error("[error] No spec files found under tools/spec (expected 'source_*.json' or 'sources_*.json').")
        return None
    print("\n=== Select a spec file for sequence ingestion ===")
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
    """Prompt the user to choose a single path-pair index (1-based) from the spec."""
    pairs = spec.get("path-pairs", [])
    if not pairs:
        LOG.error("[error] Spec has no 'path-pairs' entries.")
        return None
    print("\n=== Select a path-pair for sequence ingestion ===")
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

# ===== Sequence message parser =====
def parse_sequence_messages(path: str) -> List[Dict[str, Any]]:
    # Example line: "ActorA -> ActorB: SomeFunction(params)"
    msg_rx = re.compile(r"^\s*([\w.]+)\s*[-=]+>\s*([\w.]+)\s*:\s*(.+\S)\s*$")
    messages = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        LOG.error(f"[error] Could not read file {path}: {e}")
        return []
    step = 1
    for line in lines:
        m = msg_rx.match(line)
        if not m:
            continue
        from_part, to_part, raw_text = m.group(1), m.group(2), m.group(3)
        fn = raw_text.strip()
        if " - " in fn:
            fn = fn.split(" - ", 1)[0].strip()
        # Do not strip off parentheses for function names, keep e.g. "SearchGlobalAcc(event)"
        functionName = fn
        messages.append({
            "stepNumber": step,
            "fromParticipant": from_part,
            "toParticipant": to_part,
            "functionName": functionName,
            "rawText": raw_text.strip(),
        })
        step += 1
    return messages

# ===== CodeFunction lookup =====
def load_code_functions_for_source(source_path: Optional[str]) -> List[Dict[str, Any]]:
    """Return candidate CodeFunctions scoped to the given source_path, where sourcePath
    is a property on CodeFile. We find LogicSteps whose implementation ultimately lives
    in CodeFiles with that sourcePath, via:
        (ls:LogicStep)-[:IMPLEMENTED_BY]->(f:CodeFunction)-[:PART_OF]->(cf:CodeFile {sourcePath: $src})
    We fall back to all CodeFunctions when source_path is None or empty.
    Each item is a dict with keys: id (int) and label (str).
    """
    if source_path:
        LOG.info("[info] Loading CodeFunctions for CodeFile.sourcePath='%s' via LogicStep/IMPLEMENTED_BY/PART_OF…", source_path)
        query = """
        MATCH (ls:LogicStep)-[:IMPLEMENTED_BY]->(f:CodeFunction)-[:PART_OF]->(cf:CodeFile {sourcePath: $src})
        RETURN DISTINCT id(f) AS id, coalesce(f.label, f.name) AS label
        """
        params = {"src": source_path}
    else:
        LOG.info("[info] Loading ALL CodeFunctions (no sourcePath provided)…")
        query = """
        MATCH (f:CodeFunction)
        RETURN id(f) AS id, coalesce(f.label, f.name) AS label
        """
        params = {}
    results = run_cypher(query, params)
    if not results:
        LOG.info("[info] No CodeFunction candidates found for sourcePath='%s'", source_path)
        return []

    # Normalize results into a list of row-like objects
    rows: List[Any]
    if isinstance(results, dict):
        # Try common keys used by the graph API
        if isinstance(results.get("results"), list):
            rows = results["results"]
        elif isinstance(results.get("rows"), list):
            rows = results["rows"]
        elif isinstance(results.get("data"), list):
            rows = results["data"]
        elif isinstance(results.get("records"), list):
            rows = results["records"]
        else:
            LOG.warning("[warn] Unexpected run_cypher result format (dict): %r", results)
            return []
    elif isinstance(results, list):
        rows = results
    else:
        LOG.warning("[warn] Unexpected run_cypher result type: %r", type(results))
        return []

    candidates: List[Dict[str, Any]] = []
    for row in rows:
        id_val = None
        label_val = None

        if isinstance(row, dict):
            # Standard shape: { "id": ..., "label": ... }
            id_val = row.get("id")
            label_val = row.get("label")
            # Neo4j-like shape: { "row": [id, label], ... }
            if (label_val is None) and isinstance(row.get("row"), (list, tuple)) and len(row["row"]) >= 2:
                id_val = row["row"][0]
                label_val = row["row"][1]
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            id_val, label_val = row[0], row[1]
        else:
            LOG.trace("[trace] Skipping unexpected row format in CodeFunction load: %r", row)
            continue

        cid = _to_int_id(id_val)
        label = str(label_val or "").strip()
        if cid is None or not label:
            continue
        candidates.append({"id": cid, "label": label})
    LOG.info("[info] Loaded %d candidate CodeFunction(s) for sourcePath='%s'", len(candidates), source_path)
    return candidates

# ===== CodeFunction lookup =====

def _substring_distance(pattern: str, text: str) -> Optional[int]:
    """
    Compute the minimum character-wise Hamming distance between `pattern`
    and any substring of `text` of the same length.

    Returns:
      - An integer >= 0 representing the smallest distance found, or
      - None if pattern or text are empty or pattern is longer than text.

    Distance here is simply the count of differing characters at each position.
    """
    if not pattern or not text:
        return None

    pattern_norm = pattern.lower()
    text_norm = text.lower()

    if len(pattern_norm) > len(text_norm):
        return None

    min_dist: Optional[int] = None
    plen = len(pattern_norm)
    tlen = len(text_norm)

    for start in range(0, tlen - plen + 1):
        window = text_norm[start:start + plen]
        dist = 0
        for a, b in zip(pattern_norm, window):
            if a != b:
                dist += 1
                # Small optimisation: if distance already worse than any known min, break
                if min_dist is not None and dist > min_dist:
                    break
        if min_dist is None or dist < min_dist:
            min_dist = dist
            if min_dist == 0:
                # Can't do better than an exact match
                break

    return min_dist

def _symmetric_substring_distance(a: str, b: str) -> Optional[int]:
    """
    Wrapper around _substring_distance that always uses the shorter string
    as the pattern and the longer string as the text. This lets us handle
    cases where either the CodeFunction label or the message functionName
    is longer than the other.
    """
    a_norm = (a or "").strip()
    b_norm = (b or "").strip()
    if not a_norm or not b_norm:
        return None

    if len(a_norm) <= len(b_norm):
        return _substring_distance(a_norm, b_norm)
    else:
        return _substring_distance(b_norm, a_norm)

def find_code_function_id(function_name: str, candidates: List[Dict[str, Any]]) -> Optional[int]:
    """Choose the CodeFunction whose label most closely matches function_name,
    restricted to the provided candidates.

    Matching rule:
      - Use a symmetric substring distance so the shorter of label/function_name is always the pattern.
      - If the best distance is <= 3, treat it as a valid match.
      - If nothing meets that threshold but there is exactly one candidate,
        fall back to that single candidate.

    This allows the CodeFunction label to appear as a mostly-full substring
    within the message function name, with up to 3 differing characters.
    """
    fn_raw = (function_name or "").strip()
    if not fn_raw:
        LOG.trace("[trace] Empty functionName, cannot link to CodeFunction")
        return None
    if not candidates:
        LOG.trace("[trace] No CodeFunction candidates available for functionName '%s'", fn_raw)
        return None

    best_id: Optional[int] = None
    best_label: str = ""
    best_dist: Optional[int] = None

    for cand in candidates:
        label = str(cand.get("label") or "").strip()
        cid = _to_int_id(cand.get("id"))
        if not label or cid is None:
            continue

        dist = _symmetric_substring_distance(label, fn_raw)
        if dist is None:
            continue

        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_id = cid
            best_label = label

            # Early exit on perfect match
            if best_dist == 0:
                break

    # If we found a candidate within an acceptable distance, use it
    if best_id is not None and best_dist is not None and best_dist <= 3:
        LOG.info(
            "[info] Matched functionName '%s' to CodeFunction id=%s (label='%s') with substring_distance=%d",
            fn_raw,
            best_id,
            best_label,
            best_dist,
        )
        return best_id

    # No fallback: if no candidate met the substring-distance criteria, return None.

    LOG.trace("[trace] No CodeFunction candidate met substring-distance criteria for functionName '%s'", fn_raw)
    return None

# ===== Main entry point =====
def main():
    global RULES_PORTAL_BASE_URL
    _setup_logger(verbose=True)
    # Parse args / interactive spec selection
    source_path: Optional[str] = None
    sequence_name: Optional[str] = None

    if len(sys.argv) == 3:
        # Legacy mode: explicit sequence path and sourcePath
        seq_path = sys.argv[1]
        source_path = sys.argv[2]
        LOG.info("[info] Using explicit sequence path and sourcePath from CLI.")
    elif len(sys.argv) == 2:
        # Legacy mode: explicit sequence path only
        seq_path = sys.argv[1]
        LOG.info("[info] Using explicit sequence path from CLI.")
    elif len(sys.argv) == 1:
        # New default mode: choose spec and path-pair interactively
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
        # --- Begin: component/sequence_name logic ---
        pair = pairs[pair_index - 1]

        component = pair.get("component") or ""
        if component:
            sequence_name = f"{component} Sequence"
        LOG.info("[info] Selected component='%s'", component)
        # --- End: component/sequence_name logic ---

        root_dir = os.path.expanduser(spec.get("root-directory", os.getcwd()))
        src_rel = pair.get("source-path") or ""
        out_rel = pair.get("output-path") or ""

        # Determine absolute output folder and sequence file path
        output_dir = os.path.join(root_dir, out_rel) if out_rel else root_dir
        seq_path = os.path.join(output_dir, "sequence_report", "sequence_report.wsd")

        source_path = src_rel or None

        LOG.info("[info] Selected rootDir='%s'", root_dir)
        LOG.info("[info] Selected sourcePath='%s'", source_path)
        LOG.info("[info] Selected outputDir='%s'", output_dir)
        LOG.info("[info] Sequence diagram will be read from '%s'", seq_path)
    else:
        print("Usage: python ingest_sequence.py [sequence_report/sequence_report.wsd] [sourcePath]")
        sys.exit(1)

    # Allow SOURCE_PATH env var as a fallback / override when not explicitly passed
    env_source = os.environ.get("SOURCE_PATH")
    if env_source and not source_path:
        source_path = env_source

    abs_seq_path = os.path.abspath(seq_path)
    LOG.info(f"[info] Sequence diagram path: {abs_seq_path}")
    if not os.path.isfile(abs_seq_path):
        LOG.error(f"[error] Sequence diagram file not found: {abs_seq_path}")
        sys.exit(1)
    messages = parse_sequence_messages(abs_seq_path)
    if not messages:
        LOG.warning(f"[warn] No messages found in file: {abs_seq_path}")
        sys.exit(0)
    # Check graph API status
    LOG.info("[info] Contacting rules-portal graph API at %s", RULES_PORTAL_BASE_URL)
    status = _get("/api/graph/status")
    if not status or not status.get("up", False):
        LOG.error("[error] rules-portal graph API is not up or unreachable.")
        sys.exit(1)
    # Delete any existing Sequence/Message set for this diagramPath before re-ingesting
    _delete_existing_sequence(abs_seq_path)

    # Load candidate CodeFunctions scoped by sourcePath (if provided)
    code_function_candidates = load_code_functions_for_source(source_path)

    ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    diagram_path = abs_seq_path
    sequence_props = {
        "diagramPath": diagram_path,
        "ingestionTimestamp": ts,
    }
    if sequence_name:
        sequence_props["name"] = sequence_name
    sequence_id = create_node(["Sequence"], sequence_props)
    if sequence_id is None:
        LOG.error("[error] Failed to create Sequence node.")
        sys.exit(1)
    ingested = 0
    for message in messages:
        # First, try to find a matching CodeFunction for this message step.
        cf_id = find_code_function_id(message["functionName"], code_function_candidates)

        message_props = {
            "stepNumber": message["stepNumber"],
            "fromParticipant": message["fromParticipant"],
            "toParticipant": message["toParticipant"],
            "functionName": message["functionName"],
            "name": f"{message['stepNumber']}: {message['functionName']}",
        }
        message_id = create_node(["Message"], message_props)
        if message_id is None:
            LOG.warning(
                "[warn] Failed to create Message node for step %d",
                message["stepNumber"],
            )
            continue

        LOG.info(
            "[info] Created Message node step %d: %s",
            message["stepNumber"],
            message["functionName"],
        )

        rel_id = create_relationship(message_id, sequence_id, "PART_OF", {})
        if rel_id is None:
            LOG.warning(
                "[warn] Failed to create PART_OF relationship for Message %d -> Sequence %d",
                message_id,
                sequence_id,
            )

        # Only create SEQUENCED_BY if we found a matching CodeFunction
        if cf_id is not None:
            rel2_id = create_relationship(
                cf_id,
                message_id,
                "SEQUENCED_BY",
                {
                    "diagramPath": diagram_path,
                    "ingestionTimestamp": ts,
                },
            )
            if rel2_id is None:
                LOG.info(
                    "[info] Failed to create SEQUENCED_BY relationship from CodeFunction %d to Message %d",
                    cf_id,
                    message_id,
                )
        else:
            LOG.trace(
                "[trace] No matching CodeFunction/LogicStep for step %d functionName='%s'; Message is PART_OF Sequence only",
                message["stepNumber"],
                message["functionName"],
            )

        ingested += 1

    LOG.info(
        "[info] Ingested %d messages into Sequence node id %s.",
        ingested,
        sequence_id,
    )

if __name__ == "__main__":
    main()