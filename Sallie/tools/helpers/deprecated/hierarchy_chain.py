# helpers/hierarchy_chain.py

# Here is the spec for what is in Neo4j accessible via rules‑portal graph routes.
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
# - PART_OF       (CodeFunction -> CodeFile), (DomainType -> Domain), (Message -> Sequence)
# - INPUT         (LogicStep -> Parameter for inputs)
# - OUTPUT        (LogicStep -> Parameter for outputs)
# - LINKS_TO      (Parameter -> Parameter for hierarchy links between steps)
# - OPERATES_ON (LogicStep -> DomainType)
# - RELATED_TO (DomainType -> DomainType)
# - SEQUENCED_BY (LogicStep -> Message)
#
# You can access all of this to work out the things you need to work out.
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

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, Optional
import os
import requests

from ..hierarchy_common import (
    step_header,
    ensure_dir,
    load_json,
    write_json,
)

def _is_businessy_name(name: str) -> bool:
    """
    Very simple heuristic for “business outcome-like” decision names.
    You can refine this over time.
    """
    if not name:
        return False
    lowered = name.lower().strip()

    prefixes = [
        "is ",
        "should ",
        "determine ",
        "calculate ",
        "can ",
        "must ",
        "has ",
    ]
    suffixes = [
        " valid",
        " eligibility",
        " eligible",
        " status",
        " decision",
        " allowed",
        " required",
    ]

    if any(lowered.startswith(p) for p in prefixes):
        return True
    if any(lowered.endswith(s) for s in suffixes):
        return True
    # Fallback: contains words that look like a business outcome
    keywords = ["valid", "status", "eligibility", "approval", "route", "result"]
    if any(k in lowered for k in keywords):
        return True

    return False


# === Neo4j HTTP helpers and graph query helpers ===

def _rules_portal_base_url() -> str:
    """Return the Rules Portal base URL from env or default."""
    return os.environ.get("RULES_PORTAL_BASE_URL", "http://localhost:3201")


def _run_cypher(query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Call the rules-portal /api/graph/run endpoint with a Cypher query.

    Returns the parsed JSON response, or raises RuntimeError on failure.
    """
    base_url = _rules_portal_base_url().rstrip("/")
    url = f"{base_url}/api/graph/run"
    payload = {"query": query, "parameters": parameters or {}}

    try:
        resp = requests.post(url, json=payload, timeout=30)
    except Exception as exc:  # network error
        raise RuntimeError(f"Failed to call {url}: {exc}") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"Graph run failed with status {resp.status_code}: {resp.text}")

    data = resp.json()
    if not data.get("success", False):
        raise RuntimeError(f"Graph run returned success=false: {data}")

    return data

def _neo4j_fetch_dataflow(rule_id: str) -> Dict[str, Any]:
    """Fetch immediate and transitive dataflow neighbours for a LogicStep by id.

    Returns a dict with keys:
      - upstream: immediate upstream rule ids
      - downstream: immediate downstream rule ids
      - upstream_all: all upstream rule ids (transitive)
      - downstream_all: all downstream rule ids (transitive)
    """
    # Immediate neighbours
    query_immediate = """
    MATCH (s:LogicStep {id: $ruleId})
    OPTIONAL MATCH (u:LogicStep)-[:OUTPUT]->(:Parameter)<-[:INPUT]-(s)
    WITH s, COLLECT(DISTINCT u.id) AS upstream
    OPTIONAL MATCH (s)-[:OUTPUT]->(:Parameter)-[:INPUT]->(d:LogicStep)
    RETURN upstream, COLLECT(DISTINCT d.id) AS downstream
    """

    data_immediate = _run_cypher(query_immediate, {"ruleId": rule_id})
    recs = data_immediate.get("records", [])
    if recs:
        row = recs[0]
        upstream = row.get("upstream") or []
        downstream = row.get("downstream") or []
    else:
        upstream, downstream = [], []

    # Transitive upstream (dataflow before this rule)
    query_up = """
    MATCH (s:LogicStep {id: $ruleId})
    OPTIONAL MATCH path = (u:LogicStep)-[:OUTPUT]->(:Parameter)<-[:INPUT*1..]-(s)
    RETURN [x IN COLLECT(DISTINCT u) WHERE x.id IS NOT NULL | x.id] AS upstream_all
    """
    data_up = _run_cypher(query_up, {"ruleId": rule_id})
    recs_up = data_up.get("records", [])
    upstream_all: List[str] = []
    if recs_up:
        upstream_all = recs_up[0].get("upstream_all") or []

    # Transitive downstream (dataflow after this rule)
    query_down = """
    MATCH (s:LogicStep {id: $ruleId})
    OPTIONAL MATCH path = (s)-[:OUTPUT]->(:Parameter)-[:INPUT*1..]->(d:LogicStep)
    RETURN [x IN COLLECT(DISTINCT d) WHERE x.id IS NOT NULL | x.id] AS downstream_all
    """
    data_down = _run_cypher(query_down, {"ruleId": rule_id})
    recs_down = data_down.get("records", [])
    downstream_all: List[str] = []
    if recs_down:
        downstream_all = recs_down[0].get("downstream_all") or []

    upstream = [u for u in upstream if u]
    downstream = [d for d in downstream if d]
    upstream_all = [u for u in upstream_all if u]
    downstream_all = [d for d in downstream_all if d]

    return {
        "upstream": sorted(set(upstream)),
        "downstream": sorted(set(downstream)),
        "upstream_all": sorted(set(upstream_all)),
        "downstream_all": sorted(set(downstream_all)),
    }


def _neo4j_fetch_sequence_neighbours(rule_id: str) -> Dict[str, Any]:
    """For a LogicStep id, find rules before/after it in Sequences by Message.stepNumber."""
    # Rules before this one in the same sequence(s)
    query_before = """
    MATCH (s:LogicStep {id: $ruleId})-[:SEQUENCED_BY]->(m:Message)-[:PART_OF]->(seq:Sequence)
    MATCH (other:LogicStep)-[:SEQUENCED_BY]->(om:Message)-[:PART_OF]->(seq)
    WHERE om.stepNumber < m.stepNumber
    RETURN COLLECT(DISTINCT other.id) AS before_ids
    """
    data_before = _run_cypher(query_before, {"ruleId": rule_id})
    recs_b = data_before.get("records", [])
    before_ids: List[str] = []
    if recs_b:
        before_ids = recs_b[0].get("before_ids") or []

    # Rules after this one in the same sequence(s)
    query_after = """
    MATCH (s:LogicStep {id: $ruleId})-[:SEQUENCED_BY]->(m:Message)-[:PART_OF]->(seq:Sequence)
    MATCH (other:LogicStep)-[:SEQUENCED_BY]->(om:Message)-[:PART_OF]->(seq)
    WHERE om.stepNumber > m.stepNumber
    RETURN COLLECT(DISTINCT other.id) AS after_ids
    """
    data_after = _run_cypher(query_after, {"ruleId": rule_id})
    recs_a = data_after.get("records", [])
    after_ids: List[str] = []
    if recs_a:
        after_ids = recs_a[0].get("after_ids") or []

    before_ids = [b for b in before_ids if b]
    after_ids = [a for a in after_ids if a]

    return {
        "before": sorted(set(before_ids)),
        "after": sorted(set(after_ids)),
    }


# New per-rule analysis function (Neo4j-backed)
def _analyse_logic(
    rid: str,
    logic_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Return per-rule analysis using Neo4j graph data.

    For each rule (LogicStep id) we query the graph to derive:
      - dataflow upstream/downstream (immediate + transitive)
      - sequence-based before/after neighbours
      - simple businessy-name heuristic

    We keep a numbered 'checks' list so the rest of the pipeline and UI
    can remain stable.
    """
    logic = logic_by_id.get(rid, {})
    name = logic.get("name") or logic.get("name") or ""

    # Defaults in case graph calls fail
    upstream: List[str] = []
    downstream: List[str] = []
    upstream_all: List[str] = []
    downstream_all: List[str] = []
    seq_before: List[str] = []
    seq_after: List[str] = []

    # Call Neo4j for dataflow info
    try:
        df = _neo4j_fetch_dataflow(rid)
        upstream = df.get("upstream", [])
        downstream = df.get("downstream", [])
        upstream_all = df.get("upstream_all", [])
        downstream_all = df.get("downstream_all", [])
    except Exception as exc:
        print(f"[WARN] Dataflow analysis failed for rule {rid}: {exc}")

    # Call Neo4j for sequence info
    try:
        seq = _neo4j_fetch_sequence_neighbours(rid)
        seq_before = seq.get("before", [])
        seq_after = seq.get("after", [])
    except Exception as exc:
        print(f"[WARN] Sequence analysis failed for rule {rid}: {exc}")

    # Derived flags (still simple but now graph-backed)
    is_terminal = len(downstream) == 0 and len(upstream) > 0
    is_source = len(upstream) == 0 and len(downstream) > 0
    is_isolated = len(upstream) == 0 and len(downstream) == 0
    has_multiple_predecessors = len(upstream) >= 2
    has_multiple_successors = len(downstream) >= 2
    is_businessy = _is_businessy_name(name)
    is_mid_chain = (not is_source) and (not is_terminal) and (len(upstream) > 0 or len(downstream) > 0)

    checks = [
        {"num": 1, "label": "Is terminal in dataflow (no outgoing edges)", "value": is_terminal},
        {"num": 2, "label": "Is source in dataflow (no incoming edges)", "value": is_source},
        {"num": 3, "label": "Is isolated in dataflow (no incoming or outgoing edges)", "value": is_isolated},
        {"num": 4, "label": "Has multiple predecessors (multi-BD convergence candidate)", "value": has_multiple_predecessors},
        {"num": 5, "label": "Has multiple successors (fan-out candidate)", "value": has_multiple_successors},
        {"num": 6, "label": "Is mid-chain in dataflow (has both predecessors and successors)", "value": is_mid_chain},
        {"num": 7, "label": "Has businessy decision name", "value": is_businessy},
        {"num": 8, "label": "Has rules before it in sequence diagrams", "value": len(seq_before) > 0},
        {"num": 9, "label": "Has rules after it in sequence diagrams", "value": len(seq_after) > 0},
    ]

    return {
        "id": rid,
        "name": name,
        "checks": checks,
        # dataflow neighbours
        "predecessors": upstream,
        "successors": downstream,
        "upstream_all": upstream_all,
        "downstream_all": downstream_all,
        # sequence neighbours
        "sequence_before": seq_before,
        "sequence_after": seq_after,
    }


def _print_human_report(logics_analyses: List[Dict[str, Any]]) -> str:
    """Print a human-readable summary of the per-rule analysis.

    This is intended for humans looking at the console, not for machines.
    Returns the report string.
    """
    if not logics_analyses:
        text = "No rules to report on.\n"
        print(text)
        return text

    lines = []
    # Build an id -> name map to resolve predecessors/successors nicely
    id_to_name: Dict[str, str] = {r.get("id", ""): (r.get("name") or "") for r in logics_analyses}

    def _short_id(rid: str) -> str:
        rid = rid or ""
        return rid[-6:] if len(rid) > 6 else rid

    lines.append("===== Decision Chain Analysis Report =====\n")
    lines.append(f"Total rules analysed: {len(logics_analyses)}\n")

    for idx, r in enumerate(logics_analyses, start=1):
        rid = r.get("id", "")
        name = r.get("name") or "<unnamed>"
        preds = r.get("predecessors", []) or []
        succs = r.get("successors", []) or []

        upstream_all = r.get("upstream_all", []) or []
        downstream_all = r.get("downstream_all", []) or []

        seq_before = r.get("sequence_before", []) or []
        seq_after = r.get("sequence_after", []) or []

        pred_labels = [id_to_name.get(p, p) or p for p in preds]
        succ_labels = [id_to_name.get(s, s) or s for s in succs]

        upstream_labels = [id_to_name.get(p, p) or p for p in upstream_all]
        downstream_labels = [id_to_name.get(s, s) or s for s in downstream_all]

        seq_before_labels = [id_to_name.get(p, p) or p for p in seq_before]
        seq_after_labels = [id_to_name.get(s, s) or s for s in seq_after]

        lines.append(f"{idx}. {name} [{_short_id(rid)}]")
        if pred_labels:
            lines.append(f"   • Inputs from: {', '.join(pred_labels)}")
        else:
            lines.append("   • Inputs from: (none)")

        if succ_labels:
            lines.append(f"   • Outputs to: {', '.join(succ_labels)}")
        else:
            lines.append("   • Outputs to: (none)")

        if upstream_labels:
            lines.append(f"   • All logics before (upstream): {', '.join(upstream_labels)}")
        else:
            lines.append("   • All logics before (upstream): (none)")

        if downstream_labels:
            lines.append(f"   • All logics after (downstream): {', '.join(downstream_labels)}")
        else:
            lines.append("   • All logics after (downstream): (none)")

        if seq_before_labels:
            lines.append(f"   • All logics before (sequence): {', '.join(seq_before_labels)}")
        else:
            lines.append("   • All logics before (sequence): (none)")

        if seq_after_labels:
            lines.append(f"   • All logics after (sequence): {', '.join(seq_after_labels)}")
        else:
            lines.append("   • All logics after (sequence): (none)")

        # Print the numbered checks with a tick/cross
        for check in r.get("checks", []):
            label = check.get("label", "")
            num = check.get("num", 0)
            value = bool(check.get("value", False))
            mark = "✓" if value else "✗"
            lines.append(f"   {num}. {mark} {label}")

        lines.append("")  # Blank line between rules

    report = "\n".join(lines)
    print(report)
    return report


def run_chain_compose(
    model_info: Dict[str, Any],
    spec_info: Dict[str, Any],
    model_home_prompted: Path,
    compose_mode: str,
    skip_generate: bool,
    keep_going: bool = False,
) -> Dict[str, Any]:
    """
    'chain' compose mode – PER-LOGIC ANALYSIS STUB.

    This helper:
      • Loads the logics for the selected model only (no DB / Neo4j writes).
      • Builds a simple dataflow graph based on DMN inputs/outputs.
      • For each logic, reports a checklist of discrete properties relevant to chaining/DRD decisions, without clustering.
      • Writes results to a JSON file under `chain_output_dummy/chain_analysis.json` as a list of rules and their discrete checks.
      • Does NOT change business_rules.json or models.json.
    """

    step_header("CHAIN-1", "run_chain_compose CALLED (analysis stub)", {"compose_mode": compose_mode})

    selected_model = model_info.get("selected_model", {})
    selected_model_name = selected_model.get("name")
    selected_model_id = selected_model.get("id")
    logics_path_str = model_info.get("logics_out_path")

    print("\n=== Incoming Arguments (CHAIN STUB) ===")
    print(f"Model name: {selected_model_name}")
    print(f"Model id:   {selected_model_id}")
    print(f"logics_out_path: {logics_path_str}")
    print(f"spec_info: {spec_info}")
    print(f"model_home_prompted: {model_home_prompted}")
    print(f"compose_mode: {compose_mode}")
    print(f"skip_generate: {skip_generate}")
    print(f"keep_going: {keep_going}")
    print("======================================\n")

    if not logics_path_str:
        step_header("CHAIN-2", "No logics_out_path provided; nothing to analyse", {})
        return {
            "result": "no_rules",
            "reason": "logics_out_path missing in model_info",
        }

    logics_path = Path(logics_path_str)
    if not logics_path.exists():
        step_header("CHAIN-2", "logics_for_model.json not found; nothing to analyse", {"path": str(logics_path)})
        return {
            "result": "no_rules",
            "reason": f"rules file not found at {logics_path}",
        }

    step_header("CHAIN-2", "Load logics_for_model.json", {"path": str(logics_path)})
    rules = load_json(logics_path)
    if not isinstance(rules, list):
        step_header("CHAIN-2", "logics_for_model.json is not a list", {})
        return {
            "result": "invalid_rules_shape",
            "reason": "logics_for_model.json must be a list",
        }

    # Prepare a simple id -> rule map for Neo4j-backed analysis
    step_header("CHAIN-3", "Prepare logics for Neo4j-backed chain analysis", {})
    logic_by_id: Dict[str, Dict[str, Any]] = {}
    for r in rules:
        if isinstance(r, dict) and r.get("id"):
            logic_by_id[r["id"]] = r

    all_ids = sorted(logic_by_id.keys())

    logics_analyses: List[Dict[str, Any]] = []
    for idx, rid in enumerate(all_ids, start=1):
        analysis = _analyse_logic(rid, logic_by_id)
        logics_analyses.append(analysis)
        # Print summary for this rule
        step_header(f"CHAIN-RULE-{idx}", f"Logics analysis: {analysis['name']} ({analysis['id']})", {})
        for check in analysis["checks"]:
            print(f"  {check['num']}) {check['label']}: {check['value']}")
        print()

    # Print a consolidated human-readable report to the console and capture as markdown
    human_report = _print_human_report(logics_analyses)

    # Write everything out
    output_dir = Path("chain_output_dummy").expanduser().resolve()
    ensure_dir(output_dir)
    analysis_path = output_dir / "chain_analysis.json"

    payload = {
        "model": {
            "id": selected_model_id,
            "name": selected_model_name,
        },
        "logic_count": len(logics_analyses),
        "logics": logics_analyses,
    }
    write_json(analysis_path, payload)

    # Write markdown report
    md_path = output_dir / "chain_analysis.md"
    md_path.write_text(human_report, encoding="utf-8")
    print(f"→ Wrote chain analysis markdown to {md_path}")

    step_header("CHAIN-4", "Analysis complete (stub)", {"analysis_path": str(analysis_path)})
    print(f"→ Wrote chain analysis to {analysis_path}\n")

    return {
        "result": "ok",
        "analysis_path": str(analysis_path),
        "compose_mode": compose_mode,
        "logic_count": len(logics_analyses),
    }