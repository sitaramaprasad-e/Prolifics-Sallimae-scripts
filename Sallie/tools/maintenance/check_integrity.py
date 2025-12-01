#!/usr/bin/env python3
"""
checkIntegrity.py

Quick integrity and summary check for the model-home JSONs:
- business_rules.json (logics and links between them)
- models.json (models, hierarchies, and their references back to logics)

This script is intentionally read-only. It does NOT modify any files.

It will:
1. Load business_rules.json and models.json from a given model-home directory.
2. Support both legacy list-only formats and the new versioned root formats:
   - business_rules.json:
       [ { ...logic... }, ... ]
     or
       { "version": N, "logics": [ { ...logic... }, ... ] }
   - models.json:
       [ { ...model... }, ... ]
     or
       { "version": N, "models": [ { ...model... }, ... ] }
3. Compute and print:
   - Counts of logics, archived vs active, links between logics, dangling link targets.
   - Counts of models, logic references per model, and missing logic references.
   - Counts of hierarchies, and whether each hierarchy top refers to an existing logic.
   - Simple cross-check summaries between models/hierarchies and logics.

Usage:
  python checkIntegrity.py
    (prompts for model-home)

  python checkIntegrity.py --model-home /path/to/.model

Exit code:
  0  – script ran successfully (even if issues are found; they are reported, not enforced)
  1  – fatal error (missing files, unreadable JSON, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

try:
    import requests  # type: ignore
except ImportError:
    requests = None  # type: ignore

# Suppress noisy OpenSSL/LibreSSL compatibility warnings from urllib3 in some
# environments; they do not affect functionality of these simple HTTP calls.
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
    module="urllib3",
)


@dataclass
class LogicSummary:
    total: int
    archived: int
    active: int
    total_links: int
    links_with_from_logic: int
    links_missing_from_logic: int
    logics_with_links_json: int
    logics_without_links_json: int
    # Category integrity
    logics_with_category: int
    logics_without_category: int
    logics_with_unknown_category: int
    unknown_categories: Dict[str, int]
    # Existing integrity info
    extra_logic_fields: Dict[str, int]
    extra_link_fields: Dict[str, int]
    duplicate_logic_ids: Dict[str, int]
    short_code_name_collisions: Dict[str, int]
    duplicate_logic_names: Dict[str, int]
    # New link-shape integrity
    links_missing_required_fields: int
    links_with_blank_required_fields: int
def _load_category_names(model_home: str) -> set[str]:
    """
    Load category names from rule_categories.json.

    Supports both legacy and new shapes:
    - Legacy: { "ruleCategories": [ { "name": ... }, ... ] }
    - New:    { "categories":     [ { "name": ... }, ... ] }

    Returns a set of non-empty category names. If rule_categories.json is
    missing or malformed, returns an empty set and prints a warning.
    """
    path = Path(model_home) / "rule_categories.json"
    if not path.exists():
        _eprint(f"WARNING: rule_categories.json not found in {model_home}; skipping category reference checks.")
        return set()

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        _eprint(f"WARNING: Failed to load rule_categories.json: {exc}")
        return set()

    if not isinstance(data, dict):
        _eprint("WARNING: rule_categories.json is not an object; skipping category reference checks.")
        return set()

    if isinstance(data.get("categories"), list):
        cats = data.get("categories") or []
    elif isinstance(data.get("ruleCategories"), list):
        cats = data.get("ruleCategories") or []
    else:
        _eprint("WARNING: rule_categories.json has no categories/ruleCategories array; skipping category reference checks.")
        return set()

    names: set[str] = set()
    for cat in cats:
        if not isinstance(cat, dict):
            continue
        name = cat.get("name")
        if isinstance(name, str):
            name = name.strip()
            if name:
                names.add(name)

    return names


@dataclass
class ModelSummary:
    total_models: int
    total_model_logic_refs: int
    unique_model_logic_refs: int
    missing_model_logic_refs: int
    total_hierarchies: int
    hierarchies_with_top: int
    hierarchies_missing_top: int
    hierarchy_tops_missing_logic: int
    extra_model_fields: Dict[str, int]
    extra_hierarchy_fields: Dict[str, int]



@dataclass
class KGSummary:
    total_logic_steps: int
    logic_steps_with_id: int
    logic_steps_without_id: int
    logic_ids_with_step: int
    logic_ids_without_step: int
    steps_with_missing_logic: int
    total_supports_rels: int
    logic_ids_with_supports: int
    supports_cycles_count: int
    mutual_support_pairs_count: int
    graph_url: Optional[str]
    # Detailed lists for drill-down
    logic_ids_without_step_list: List[str]
    steps_with_missing_logic_list: List[str]
    supports_cycle_edges: List[Tuple[str, str]]
    mutual_support_pairs: List[Tuple[str, str]]
    # OK-to-miss structural logics (Decision (Top-Level) / Decision (Composite)) not present in KG
    ok_missing_logic_count: int
    ok_missing_logic_ids: List[str]

# Default base URL for the graph API (matches RULES_PORTAL_BASE_URL with /api/graph)
DEFAULT_GRAPH_BASE_URL = "http://localhost:443/api/graph"

# Toggle TLS certificate verification for KG HTTP calls. In many dev and some
# internal environments we use self-signed or private CAs, so by default we
# disable verification here. You can override this by setting the
# RULES_PORTAL_VERIFY environment variable to "true"/"1"/"yes" to enable
# verification.
GRAPH_VERIFY = False
env_verify = os.getenv("RULES_PORTAL_VERIFY")
if env_verify is not None:
    GRAPH_VERIFY = env_verify.strip().lower() in {"1", "true", "yes", "on"}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


# --- Spec selection helpers ---
def _spec_dir() -> str:
    """
    Locate the default spec directory for this repo.

    This script lives in <repo>/tools/maintenance; specs are under <repo>/tools/spec,
    the same convention used by ingest_domain.py and other tools.
    """
    tools_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(tools_dir, "spec")


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
        return None
    print("\n=== Select a spec file for KG checks (optional) ===")
    for i, path in enumerate(specs, 1):
        print(f"  {i}. {os.path.basename(path)}")
    print("Choose a spec by number, or press ENTER to skip.")
    while True:
        choice = input("Enter number (or ENTER to skip): ").strip()
        if not choice:
            return None
        if not choice.isdigit():
            print("Please enter a valid number or press ENTER to skip.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(specs):
            return specs[idx - 1]
        print("Out of range. Try again.")


def _load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _prompt_model_home() -> str:
    """
    Prompt the user for model-home if not provided via CLI.

    This mirrors the pattern in other tools: user can paste or type the model-home
    path (usually the .model directory under the repo).
    """
    print("Enter model-home path (directory containing models.json and business_rules.json)")
    print("Press ENTER to use default: ~/.model")
    typed = input("> ").strip()
    if not typed:
        return os.path.expanduser("~/.model")
    return typed


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_logics(model_home: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Load business_rules.json in a way that supports both legacy list-only and new
    versioned-root structures.

    Returns:
      (logics_list, logics_by_id)
    """
    rules_path = os.path.join(model_home, "business_rules.json")
    if not os.path.isfile(rules_path):
        raise FileNotFoundError(f"business_rules.json not found at {rules_path}")

    data = _load_json(rules_path)

    if isinstance(data, list):
        logics = data
    elif isinstance(data, dict) and isinstance(data.get("logics"), list):
        logics = data["logics"]
    else:
        raise ValueError(
            f"business_rules.json is not in a supported format: "
            f"expected list or {{version, logics}} object, got {type(data).__name__}"
        )

    logics_by_id: Dict[str, Dict[str, Any]] = {}
    for logic in logics:
        if not isinstance(logic, dict):
            continue
        lid = logic.get("id")
        if isinstance(lid, str) and lid:
            logics_by_id[lid] = logic

    return logics, logics_by_id


def _load_models(model_home: str) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Load models.json in a way that supports both legacy list-only and new
    versioned-root structures.

    Returns:
      (models_list, models_by_id)
    """
    models_path = os.path.join(model_home, "models.json")
    if not os.path.isfile(models_path):
        raise FileNotFoundError(f"models.json not found at {models_path}")

    data = _load_json(models_path)

    if isinstance(data, list):
        models = data
    elif isinstance(data, dict) and isinstance(data.get("models"), list):
        models = data["models"]
    else:
        raise ValueError(
            f"models.json is not in a supported format: "
            f"expected list or {{version, models}} object, got {type(data).__name__}"
        )

    models_by_id: Dict[str, Dict[str, Any]] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        mid = model.get("id")
        if isinstance(mid, str) and mid:
            models_by_id[mid] = model

    return models, models_by_id


def _load_kg_summary(
    graph_base_url: Optional[str],
    logics_by_id: Dict[str, Dict[str, Any]],
) -> Optional[KGSummary]:
    """
    Optionally load a summary of LogicStep nodes from the Knowledge Graph and
    cross-check them against logics_by_id.

    If graph_base_url is None, or the requests library is unavailable, or the
    /status endpoint reports the graph as down, this returns None and the KG
    section is simply omitted from the summary.
    """
    if not graph_base_url:
        return None
    if requests is None:  # type: ignore
        _eprint("requests library not available; skipping KG checks.")
        return None

    # Normalize graph_base_url: it may be a bare host (http://localhost:443)
    # or already include /api/graph.
    base = graph_base_url.rstrip("/")
    if not base.endswith("/api/graph"):
        # If the caller passed just the host, append the correct KG path.
        base = base + "/api/graph"

    status_url = f"{base}/status"
    run_url = f"{base}/run"

    try:
        resp = requests.get(status_url, timeout=5, verify=GRAPH_VERIFY)
        resp.raise_for_status()
        status_payload = resp.json()
    except Exception as exc:
        _eprint(f"WARNING: Failed to reach graph status at {status_url}: {exc}")
        return None

    if not isinstance(status_payload, dict) or not status_payload.get("up"):
        _eprint(f"WARNING: Graph status at {status_url} reports up = False; skipping KG checks.")
        return None

    # Pull all LogicStep ids from the KG.
    query = "MATCH (n:LogicStep) RETURN n.id AS logicId"
    try:
        resp = requests.post(
            run_url,
            json={"query": query, "parameters": {}},
            timeout=15,
            verify=GRAPH_VERIFY,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        _eprint(f"WARNING: Failed to query LogicStep nodes via {run_url}: {exc}")
        return None

    records = payload.get("records", [])
    if not isinstance(records, list):
        _eprint("WARNING: Unexpected /run payload shape; 'records' is not a list. Skipping KG checks.")
        return None

    total_logic_steps = len(records)
    logic_steps_with_id = 0
    logic_steps_without_id = 0
    step_ids: set[str] = set()

    for rec in records:
        if not isinstance(rec, dict):
            continue
        logic_id = rec.get("logicId")
        if logic_id is None:
            logic_steps_without_id += 1
            continue
        # Neo4j driver might surface values as nested objects; we only care
        # about string-ish values here.
        if isinstance(logic_id, (int, float)):
            logic_id = str(logic_id)
        if not isinstance(logic_id, str):
            logic_steps_without_id += 1
            continue

        logic_steps_with_id += 1
        if logic_id:
            step_ids.add(logic_id)

    logic_ids = set(logics_by_id.keys())

    # Base vs top-level/composite distinction:
    # - Base logics (Decision, BKM, etc.) are REQUIRED to have a LogicStep in the KG.
    # - Decision (Top-Level) and Decision (Composite) are allowed to be absent from the KG.
    base_logic_ids: set[str] = set()
    for lid, logic in logics_by_id.items():
        if not isinstance(logic, dict):
            continue
        kind = (logic.get("kind") or "").strip()
        if kind in ("Decision (Top-Level)", "Decision (Composite)"):
            # These are "structural" decisions; it's OK if they have no LogicStep.
            continue
        base_logic_ids.add(lid)

    logic_ids_with_step = len(logic_ids & step_ids)
    missing_logic_ids = sorted(list(base_logic_ids - step_ids))

    # Structural (top-level / composite) logics are allowed to be absent from KG.
    # We still count them for reporting, but they are not treated as integrity issues.
    structural_logic_ids = logic_ids - base_logic_ids
    ok_missing_logic_ids = sorted(list(structural_logic_ids - step_ids))

    extra_step_ids = sorted(list(step_ids - logic_ids))
    logic_ids_without_step = len(missing_logic_ids)
    steps_with_missing_logic = len(extra_step_ids)

    logic_ids_without_step_list = missing_logic_ids
    steps_with_missing_logic_list = extra_step_ids
    ok_missing_logic_count = len(ok_missing_logic_ids)

    # Count SUPPORTS relationships between LogicStep nodes and track which logic ids participate
    total_supports_rels = 0
    logic_ids_with_supports_set: set[str] = set()
    supports_query = (
        "MATCH (a:LogicStep)-[r:SUPPORTS]->(b:LogicStep) "
        "RETURN count(r) AS supportsCount, "
        "collect(DISTINCT a.id) AS fromIds, "
        "collect(DISTINCT b.id) AS toIds"
    )
    try:
        resp2 = requests.post(
            run_url,
            json={"query": supports_query, "parameters": {}},
            timeout=15,
            verify=GRAPH_VERIFY,
        )
        resp2.raise_for_status()
        payload2 = resp2.json()
        records2 = payload2.get("records", [])
        if isinstance(records2, list) and records2:
            rec0 = records2[0]
            if isinstance(rec0, dict):
                supports_val = rec0.get("supportsCount")
                # Neo4j integers may come through as plain numbers or {low, high}
                if isinstance(supports_val, dict) and "low" in supports_val:
                    supports_val = supports_val.get("low")
                if isinstance(supports_val, (int, float)):
                    total_supports_rels = int(supports_val)

                # Collect distinct logic ids that participate in SUPPORTS relationships
                for key in ("fromIds", "toIds"):
                    arr = rec0.get(key)
                    if not isinstance(arr, list):
                        continue
                    for v in arr:
                        val = v
                        if isinstance(val, dict) and "low" in val:
                            val = val.get("low")
                        if isinstance(val, (int, float)):
                            val = str(val)
                        if isinstance(val, str) and val:
                            logic_ids_with_supports_set.add(val)
    except Exception as exc:
        _eprint(f"WARNING: Failed to count SUPPORTS relationships via {run_url}: {exc}")

    # Cycle detection over SUPPORTS edges
    supports_cycles_count = 0
    mutual_support_pairs_count = 0
    supports_cycle_edges: set[Tuple[str, str]] = set()
    mutual_support_pairs: set[Tuple[str, str]] = set()

    edges_query = (
        "MATCH (a:LogicStep)-[:SUPPORTS]->(b:LogicStep) "
        "RETURN a.id AS fromId, b.id AS toId"
    )
    try:
        resp3 = requests.post(
            run_url,
            json={"query": edges_query, "parameters": {}},
            timeout=30,
            verify=GRAPH_VERIFY,
        )
        resp3.raise_for_status()
        payload3 = resp3.json()
        records3 = payload3.get("records", [])

        # Build edge set and adjacency list for cycle detection
        from typing import Dict as _DictType, List as _ListType, Set as _SetType

        edges: _SetType[tuple[str, str]] = set()
        adj: _DictType[str, _ListType[str]] = {}

        if isinstance(records3, list):
            for rec in records3:
                if not isinstance(rec, dict):
                    continue
                u = rec.get("fromId")
                v = rec.get("toId")

                # Normalise potential Neo4j integer/map representations
                if isinstance(u, dict) and "low" in u:
                    u = u.get("low")
                if isinstance(v, dict) and "low" in v:
                    v = v.get("low")
                if isinstance(u, (int, float)):
                    u = str(u)
                if isinstance(v, (int, float)):
                    v = str(v)

                if not (isinstance(u, str) and isinstance(v, str)):
                    continue
                if not u or not v:
                    continue

                edges.add((u, v))
                adj.setdefault(u, []).append(v)

        # Count mutually-supporting pairs A↔B
        for (u, v) in edges:
            if (v, u) in edges and u < v:
                mutual_support_pairs_count += 1
                mutual_support_pairs.add((u, v))

        # Detect cycles using DFS over the directed SUPPORTS graph
        visited: _SetType[str] = set()
        in_stack: _SetType[str] = set()

        def _dfs(node: str) -> None:
            nonlocal supports_cycles_count
            visited.add(node)
            in_stack.add(node)
            for neigh in adj.get(node, []):
                if neigh not in visited:
                    _dfs(neigh)
                elif neigh in in_stack:
                    # Found a back-edge indicating a cycle. We count occurrences;
                    # this is a heuristic count rather than a canonical cycle set.
                    supports_cycles_count += 1
                    supports_cycle_edges.add((node, neigh))
            in_stack.remove(node)

        for node in adj.keys():
            if node not in visited:
                _dfs(node)
    except Exception as exc:
        _eprint(f"WARNING: Failed to fetch SUPPORTS edges for cycle detection via {run_url}: {exc}")

    logic_ids_with_supports = len(set(logics_by_id.keys()) & logic_ids_with_supports_set)

    return KGSummary(
        total_logic_steps=total_logic_steps,
        logic_steps_with_id=logic_steps_with_id,
        logic_steps_without_id=logic_steps_without_id,
        logic_ids_with_step=logic_ids_with_step,
        logic_ids_without_step=logic_ids_without_step,
        steps_with_missing_logic=steps_with_missing_logic,
        total_supports_rels=total_supports_rels,
        logic_ids_with_supports=logic_ids_with_supports,
        supports_cycles_count=supports_cycles_count,
        mutual_support_pairs_count=mutual_support_pairs_count,
        graph_url=base,
        logic_ids_without_step_list=logic_ids_without_step_list,
        steps_with_missing_logic_list=steps_with_missing_logic_list,
        supports_cycle_edges=sorted(list(supports_cycle_edges)),
        mutual_support_pairs=sorted(list(mutual_support_pairs)),
        ok_missing_logic_count=ok_missing_logic_count,
        ok_missing_logic_ids=ok_missing_logic_ids,
    )
# === Issue collection and drill-down helpers ===

def _collect_issues(
    logic_summary: LogicSummary,
    model_summary: ModelSummary,
    kg_summary: Optional[KGSummary],
) -> List[Tuple[str, str]]:
    issues: List[Tuple[str, str]] = []

    if logic_summary.links_missing_from_logic:
        issues.append(
            (
                "links_missing_from_logic",
                "some links have 'from' references that do not resolve to logics",
            )
        )
    if logic_summary.links_missing_required_fields:
        issues.append(
            (
                "links_missing_required_fields",
                "some links are missing one or more of the required fields from_output, to_input, kind, from_logic_id",
            )
        )
    if logic_summary.links_with_blank_required_fields:
        issues.append(
            (
                "links_with_blank_required_fields",
                "some links have required fields (from_output, to_input, kind, from_logic_id) present but blank or non-string",
            )
        )
    if model_summary.missing_model_logic_refs:
        issues.append(
            (
                "missing_model_logic_refs",
                "some model logic IDs do not resolve to logics",
            )
        )
    if model_summary.hierarchy_tops_missing_logic:
        issues.append(
            (
                "hierarchy_tops_missing_logic",
                "some hierarchy tops do not resolve to logics",
            )
        )
    if model_summary.hierarchies_missing_top:
        issues.append(
            (
                "hierarchies_missing_top",
                "some hierarchies are missing a topId/topDecisionId",
            )
        )
    if logic_summary.extra_logic_fields:
        issues.append(
            (
                "extra_logic_fields",
                "some logics contain unexpected fields not in the standard schema",
            )
        )
    if logic_summary.extra_link_fields:
        issues.append(
            (
                "extra_link_fields",
                "some link objects contain unexpected fields not in the standard schema",
            )
        )
    if model_summary.extra_model_fields:
        issues.append(
            (
                "extra_model_fields",
                "some models contain unexpected fields not in the standard schema",
            )
        )
    if model_summary.extra_hierarchy_fields:
        issues.append(
            (
                "extra_hierarchy_fields",
                "some hierarchy objects contain unexpected fields not in the standard schema",
            )
        )
    if logic_summary.duplicate_logic_ids:
        issues.append(
            (
                "duplicate_logic_ids",
                "some logic IDs are duplicated in business_rules.json",
            )
        )
    if logic_summary.short_code_name_collisions:
        issues.append(
            (
                "short_code_name_collisions",
                "some logics share the same short code (last 3 hex chars of id) and name, which may cause UI ambiguity",
            )
        )
    if logic_summary.duplicate_logic_names:
        issues.append(
            (
                "duplicate_logic_names",
                "some logics share the same name, which may indicate duplicated or ambiguous rules",
            )
        )
    if logic_summary.logics_with_unknown_category:
        issues.append(
            (
                "logics_with_unknown_category",
                "some logics reference a category name that does not exist in rule_categories.json",
            )
        )
    if kg_summary is not None:
        if kg_summary.logic_ids_without_step:
            issues.append(
                (
                    "kg_logic_ids_without_step",
                    "some base logics (non-top-level / non-composite) in business_rules.json do not have a corresponding LogicStep node in the Knowledge Graph",
                )
            )
        if kg_summary.steps_with_missing_logic:
            issues.append(
                (
                    "kg_steps_with_missing_logic",
                    "some LogicStep nodes in the Knowledge Graph have ids that do not match any logic in business_rules.json",
                )
            )
        if kg_summary.supports_cycles_count:
            issues.append(
                (
                    "kg_supports_cycles",
                    "some LogicStep nodes participate in cycles of SUPPORTS relationships in the Knowledge Graph",
                )
            )
        if kg_summary.mutual_support_pairs_count:
            issues.append(
                (
                    "kg_mutual_support_pairs",
                    "some LogicStep pairs mutually SUPPORT each other (bidirectional SUPPORTS) in the Knowledge Graph",
                )
            )

    return issues


def _print_issue_details(
    issue_key: str,
    model_home: str,
    logics: List[Dict[str, Any]],
    models: List[Dict[str, Any]],
    logics_by_id: Dict[str, Dict[str, Any]],
    logic_summary: LogicSummary,
    model_summary: ModelSummary,
    kg_summary: Optional[KGSummary],
    category_names: Optional[set[str]],
) -> None:
    """
    Print detail for a specific issue key. This is used by the interactive
    drill-down after the main summary.
    """
    # Links with missing 'from' logic
    if issue_key == "links_missing_from_logic":
        print("Links whose 'from' reference does not resolve to a logic:")
        found = False
        for logic in logics:
            if not isinstance(logic, dict):
                continue
            lid = logic.get("id")
            name = logic.get("name") or ""
            links = logic.get("links")
            if not isinstance(links, list):
                continue
            for link in links:
                if not isinstance(link, dict):
                    continue
                from_logic_id = link.get("from_logic_id")
                from_step_id = link.get("from_step_id")
                effective_from = None
                if isinstance(from_logic_id, str) and from_logic_id:
                    effective_from = from_logic_id
                elif isinstance(from_step_id, str) and from_step_id:
                    effective_from = from_step_id
                if not effective_from:
                    continue
                if effective_from in logics_by_id:
                    continue
                found = True
                kind = link.get("kind", "")
                from_output = link.get("from_output", "")
                to_input = link.get("to_input", "")
                print(
                    f"  - Logic {lid} ({name}): "
                    f"from_logic_id={from_logic_id}, from_step_id={from_step_id}, "
                    f"kind={kind}, from_output={from_output}, to_input={to_input}"
                )
        if not found:
            print("  (No detailed offending links found.)")
        return

    # Links missing required fields
    if issue_key == "links_missing_required_fields":
        print("Links missing one or more required fields (from_output, to_input, kind, from_logic_id):")
        found = False
        required_keys = ("from_output", "to_input", "kind", "from_logic_id")
        for logic in logics:
            if not isinstance(logic, dict):
                continue
            lid = logic.get("id")
            name = logic.get("name") or ""
            links = logic.get("links")
            if not isinstance(links, list):
                continue
            for link in links:
                if not isinstance(link, dict):
                    continue
                missing = [k for k in required_keys if k not in link]
                if not missing:
                    continue
                found = True
                print(
                    f"  - Logic {lid} ({name}): missing {', '.join(missing)} "
                    f"(link kind={link.get('kind')!r}, from_output={link.get('from_output')!r}, "
                    f"to_input={link.get('to_input')!r}, from_logic_id={link.get('from_logic_id')!r})"
                )
        if not found:
            print("  (No links with missing required fields found on re-scan.)")
        return

    # Links with blank required fields
    if issue_key == "links_with_blank_required_fields":
        print("Links with blank/non-string required fields (from_output, to_input, kind, from_logic_id):")
        found = False
        required_keys = ("from_output", "to_input", "kind", "from_logic_id")
        for logic in logics:
            if not isinstance(logic, dict):
                continue
            lid = logic.get("id")
            name = logic.get("name") or ""
            links = logic.get("links")
            if not isinstance(links, list):
                continue
            for link in links:
                if not isinstance(link, dict):
                    continue
                blanks = []
                for k in required_keys:
                    val = link.get(k)
                    if not isinstance(val, str) or not val.strip():
                        blanks.append(k)
                if not blanks:
                    continue
                found = True
                print(
                    f"  - Logic {lid} ({name}): blank/non-string {', '.join(blanks)} "
                    f"(link kind={link.get('kind')!r}, from_output={link.get('from_output')!r}, "
                    f"to_input={link.get('to_input')!r}, from_logic_id={link.get('from_logic_id')!r})"
                )
        if not found:
            print("  (No links with blank required fields found on re-scan.)")
        return

    # Missing model logic references
    if issue_key == "missing_model_logic_refs":
        print("Model logic references that do not resolve to a logic:")
        found = False
        for model in models:
            if not isinstance(model, dict):
                continue
            mid = model.get("id")
            mname = model.get("name") or ""
            logic_ids = _get_model_logic_ids(model)
            for lid in logic_ids:
                if lid not in logics_by_id:
                    found = True
                    print(f"  - Model {mid} ({mname}): missing logic id {lid}")
        if not found:
            print("  (No missing model logic references found.)")
        return

    # Hierarchy tops missing logic
    if issue_key == "hierarchy_tops_missing_logic":
        print("Hierarchies whose topId/topDecisionId does not resolve to a logic:")
        found = False
        for model in models:
            if not isinstance(model, dict):
                continue
            mid = model.get("id")
            mname = model.get("name") or ""
            for h in _get_hierarchies(model):
                if not isinstance(h, dict):
                    continue
                top_id = _get_hierarchy_top_id(h)
                if top_id and top_id not in logics_by_id:
                    found = True
                    hname = h.get("name") or ""
                    print(
                        f"  - Model {mid} ({mname}), hierarchy '{hname}': "
                        f"topId={top_id} (no matching logic)"
                    )
        if not found:
            print("  (No hierarchy tops with missing logic found.)")
        return

    # Hierarchies missing a top
    if issue_key == "hierarchies_missing_top":
        print("Hierarchies that do not specify a topId/topDecisionId:")
        found = False
        for model in models:
            if not isinstance(model, dict):
                continue
            mid = model.get("id")
            mname = model.get("name") or ""
            for h in _get_hierarchies(model):
                if not isinstance(h, dict):
                    continue
                top_id = _get_hierarchy_top_id(h)
                if not top_id:
                    found = True
                    hname = h.get("name") or ""
                    print(
                        f"  - Model {mid} ({mname}), hierarchy '{hname}' has no topId/topDecisionId"
                    )
        if not found:
            print("  (No hierarchies without tops found.)")
        return

    # Extra/legacy schema fields
    if issue_key == "extra_logic_fields":
        print("Unexpected logic fields (field: count of logics):")
        if not logic_summary.extra_logic_fields:
            print("  (None)")
            return
        for key, count in sorted(logic_summary.extra_logic_fields.items()):
            print(f"  {key}: {count}")
        return

    if issue_key == "extra_link_fields":
        print("Unexpected link fields (field: count of links):")
        if not logic_summary.extra_link_fields:
            print("  (None)")
            return
        for key, count in sorted(logic_summary.extra_link_fields.items()):
            print(f"  {key}: {count}")
        return

    if issue_key == "extra_model_fields":
        print("Unexpected model fields (field: count of models):")
        if not model_summary.extra_model_fields:
            print("  (None)")
            return
        for key, count in sorted(model_summary.extra_model_fields.items()):
            print(f"  {key}: {count}")
        return

    if issue_key == "extra_hierarchy_fields":
        print("Unexpected hierarchy fields (field: count of hierarchies):")
        if not model_summary.extra_hierarchy_fields:
            print("  (None)")
            return
        for key, count in sorted(model_summary.extra_hierarchy_fields.items()):
            print(f"  {key}: {count}")
        return

    # Duplicate IDs and short-code collisions
    if issue_key == "duplicate_logic_ids":
        print("Duplicate logic IDs (id: count of occurrences):")
        if not logic_summary.duplicate_logic_ids:
            print("  (None)")
            return
        for lid, count in sorted(logic_summary.duplicate_logic_ids.items()):
            print(f"  {lid}: {count}")
        return

    if issue_key == "short_code_name_collisions":
        print('Short code + name collisions ("SHORT | Name": count of logics):')
        if not logic_summary.short_code_name_collisions:
            print("  (None)")
            return
        for key, count in sorted(logic_summary.short_code_name_collisions.items()):
            print(f"  {key}: {count}")
        return

    if issue_key == "duplicate_logic_names":
        print("Duplicate logic names (name: count of occurrences):")
        if not logic_summary.duplicate_logic_names:
            print("  (None)")
            return
        for name, count in sorted(logic_summary.duplicate_logic_names.items()):
            print(f"  {name}: {count}")
        return

    # Unknown categories
    if issue_key == "logics_with_unknown_category":
        print("Logics referencing categories not found in rule_categories.json:")
        if not category_names:
            print("  (Category names could not be loaded; only aggregate counts are available.)")
            if not logic_summary.unknown_categories:
                print("  (No unknown category names recorded.)")
            else:
                print("  Unknown category names (name: count of logics):")
                for name, count in sorted(logic_summary.unknown_categories.items()):
                    print(f"    {name}: {count}")
            return

        found = False
        for logic in logics:
            if not isinstance(logic, dict):
                continue
            cat = logic.get("category")
            if cat is None:
                continue
            cat_name = str(cat).strip()
            if not cat_name or cat_name in category_names:
                continue
            found = True
            lid = logic.get("id")
            name = logic.get("name") or ""
            print(f"  - Logic {lid} ({name}): category='{cat_name}' (unknown)")
        if not found:
            print("  (No logics with unknown categories found.)")
        return

    # KG-related issues
    if issue_key == "kg_logic_ids_without_step":
        print("Base logics present in business_rules.json but missing a LogicStep node in the KG:")
        if kg_summary is None:
            print("  (Knowledge Graph summary not available.)")
            return
        if not kg_summary.logic_ids_without_step_list:
            print("  (None)")
            return
        for lid in kg_summary.logic_ids_without_step_list:
            logic = logics_by_id.get(lid)
            name = (logic or {}).get("name") or ""
            print(f"  - {lid} ({name})")
        return

    if issue_key == "kg_steps_with_missing_logic":
        print("LogicStep nodes in the KG whose ids do not match any logic in business_rules.json:")
        if kg_summary is None:
            print("  (Knowledge Graph summary not available.)")
            return
        if not kg_summary.steps_with_missing_logic_list:
            print("  (None)")
            return

        # Try to look up LogicStep names and kinds for better reporting.
        name_by_id: Dict[str, Optional[tuple[Optional[str], Optional[str]]]] = {}
        if requests is not None and kg_summary.graph_url:
            base = kg_summary.graph_url.rstrip("/")
            run_url = f"{base}/run"
            ids_param = kg_summary.steps_with_missing_logic_list
            # Updated query to also return n.kind AS kind
            query = (
                "MATCH (n:LogicStep) "
                "WHERE n.id IN $ids "
                "RETURN n.id AS id, n.name AS name, n.kind AS kind"
            )
            try:
                resp = requests.post(
                    run_url,
                    json={"query": query, "parameters": {"ids": ids_param}},
                    timeout=15,
                    verify=GRAPH_VERIFY,
                )
                resp.raise_for_status()
                payload = resp.json()
                records = payload.get("records", [])
                if isinstance(records, list):
                    for rec in records:
                        if not isinstance(rec, dict):
                            continue
                        rid = rec.get("id")
                        rname = rec.get("name")
                        rkind = rec.get("kind")
                        # Normalise potential Neo4j integer/map representations
                        if isinstance(rid, dict) and "low" in rid:
                            rid = rid.get("low")
                        if isinstance(rid, (int, float)):
                            rid = str(rid)
                        # name normalization
                        if isinstance(rname, dict) and "low" in rname:
                            rname = rname.get("low")
                        if isinstance(rname, (int, float)):
                            rname = str(rname)
                        if isinstance(rname, str):
                            rname = rname.strip() or None
                        else:
                            rname = None
                        # kind normalization
                        if isinstance(rkind, dict) and "low" in rkind:
                            rkind = rkind.get("low")
                        if isinstance(rkind, (int, float)):
                            rkind = str(rkind)
                        if isinstance(rkind, str):
                            rkind = rkind.strip() or None
                        else:
                            rkind = None
                        if isinstance(rid, str):
                            name_by_id[rid] = (rname, rkind)
            except Exception as exc:  # noqa: BLE001
                _eprint(f"WARNING: Failed to look up LogicStep names/kinds via KG: {exc}")

        for sid in kg_summary.steps_with_missing_logic_list:
            name_kind = name_by_id.get(sid)
            if name_kind:
                name, kind = name_kind
                if name and kind:
                    print(f"  - {sid} ({name}, {kind})")
                elif name:
                    print(f"  - {sid} ({name})")
                elif kind:
                    print(f"  - {sid} ({kind})")
                else:
                    print(f"  - {sid}")
            else:
                print(f"  - {sid}")
        return

    if issue_key == "kg_supports_cycles":
        print("Back-edges indicating cycles in SUPPORTS relationships (fromId -> toId):")
        if kg_summary is None:
            print("  (Knowledge Graph summary not available.)")
            return
        if not kg_summary.supports_cycle_edges:
            if kg_summary.supports_cycles_count:
                print("  (Cycles were counted, but no specific edges were captured.)")
            else:
                print("  (None)")
            return
        for (u, v) in kg_summary.supports_cycle_edges:
            print(f"  - {u} -> {v}")
        return

    if issue_key == "kg_mutual_support_pairs":
        print("Mutually-supporting LogicStep id pairs (A ↔ B):")
        if kg_summary is None:
            print("  (Knowledge Graph summary not available.)")
            return
        if not kg_summary.mutual_support_pairs:
            if kg_summary.mutual_support_pairs_count:
                print("  (Mutual pairs were counted, but no specific pairs were captured.)")
            else:
                print("  (None)")
            return
        for (a, b) in kg_summary.mutual_support_pairs:
            print(f"  - {a} <-> {b}")
        return

    # Fallback for unknown keys
    print(f"(No detail handler implemented for issue key '{issue_key}'.)")


def _summarize_logics(
    logics: List[Dict[str, Any]],
    logics_by_id: Dict[str, Dict[str, Any]],
    category_names: Optional[set[str]] = None,
) -> LogicSummary:
    total = len(logics)
    archived = 0
    total_links = 0
    links_with_from_logic = 0
    links_missing_from_logic = 0
    logics_with_links_json = 0

    logics_with_category = 0
    logics_without_category = 0
    logics_with_unknown_category = 0
    unknown_categories: Dict[str, int] = {}

    allowed_logic_keys = {
        "code_block",
        "code_file",
        "code_lines",
        "code_function",
        "example",
        "dmn_hit_policy",
        "dmn_inputs",
        "dmn_outputs",
        "dmn_table",
        "timestamp",
        "id",
        "owner",
        "component",
        "kind",
        "links",
        "rootDir",
        "sourcePath",
        "business_area",
        "ai_categorized",
        "ai_categorised",
        "category_explanation",
        "category",
        "name",
        "purpose",
        "spec",
        "archived",
    }

    allowed_link_keys = {
        "from_output",
        "to_input",
        "kind",
        "from_logic_id",
    }

    extra_logic_fields: Dict[str, int] = {}
    extra_link_fields: Dict[str, int] = {}

    id_counts: Dict[str, int] = {}
    short_code_name_counts: Dict[str, int] = {}
    name_counts: Dict[str, int] = {}

    links_missing_required_fields = 0
    links_with_blank_required_fields = 0

    required_link_keys = {
        "from_output",
        "to_input",
        "kind",
        "from_logic_id",
    }

    for logic in logics:
        if not isinstance(logic, dict):
            continue

        # Archived flag
        if logic.get("archived") is True:
            archived += 1

        # Schema check for logic fields: report any unexpected keys
        logic_keys = set(logic.keys())
        unexpected_logic_keys = logic_keys - allowed_logic_keys
        for key in unexpected_logic_keys:
            extra_logic_fields[key] = extra_logic_fields.get(key, 0) + 1

        # Track duplicate IDs and short-code+name collisions
        lid = logic.get("id")
        name = logic.get("name")
        if isinstance(lid, str) and lid:
            id_counts[lid] = id_counts.get(lid, 0) + 1

            cleaned = lid.replace("-", "")
            if len(cleaned) < 3:
                short = cleaned.upper()
            else:
                short = cleaned[-3:].upper()

            if isinstance(name, str) and name:
                key = f"{short} | {name}"
                short_code_name_counts[key] = short_code_name_counts.get(key, 0) + 1

        # Track duplicate names
        name = logic.get("name")
        if isinstance(name, str) and name.strip():
            name_counts[name] = name_counts.get(name, 0) + 1

        # Category presence and reference integrity
        cat_val = logic.get("category")
        if cat_val is None or (isinstance(cat_val, str) and not cat_val.strip()):
            logics_without_category += 1
        else:
            logics_with_category += 1
            if category_names:
                cat_name = str(cat_val).strip()
                if cat_name and cat_name not in category_names:
                    logics_with_unknown_category += 1
                    unknown_categories[cat_name] = unknown_categories.get(cat_name, 0) + 1

        links = logic.get("links")
        if isinstance(links, list) and len(links) > 0:
            logics_with_links_json += 1
        else:
            # No valid links list; skip link-level processing
            continue

        for link in links:
            if not isinstance(link, dict):
                continue

            total_links += 1

            # Schema check for link fields: report any unexpected keys
            link_keys = set(link.keys())
            unexpected_link_keys = link_keys - allowed_link_keys
            for key in unexpected_link_keys:
                extra_link_fields[key] = extra_link_fields.get(key, 0) + 1

            # Required fields: each link should have these 4 fields, none empty
            missing_any_required = False
            blank_any_required = False
            for req_key in required_link_keys:
                if req_key not in link:
                    missing_any_required = True
                    continue
                val = link.get(req_key)
                if not isinstance(val, str) or not val.strip():
                    blank_any_required = True

            if missing_any_required:
                links_missing_required_fields += 1
            if blank_any_required:
                links_with_blank_required_fields += 1

            # From-logic reference (for post-migration data this is from_logic_id;
            # older data may still have from_step_id – we treat missing resolution
            # to a logic as "missing from logic" for summary purposes).
            from_logic_id = link.get("from_logic_id")
            from_step_id = link.get("from_step_id")

            effective_from = None
            if isinstance(from_logic_id, str) and from_logic_id:
                effective_from = from_logic_id
            elif isinstance(from_step_id, str) and from_step_id:
                # We don't know how to resolve step IDs to logics here,
                # so we just count them as "present but not directly resolvable".
                effective_from = from_step_id

            if effective_from:
                if effective_from in logics_by_id:
                    links_with_from_logic += 1
                else:
                    links_missing_from_logic += 1

    duplicate_logic_ids: Dict[str, int] = {
        logic_id: count for logic_id, count in id_counts.items() if count > 1
    }
    short_code_name_collisions: Dict[str, int] = {
        key: count for key, count in short_code_name_counts.items() if count > 1
    }
    duplicate_logic_names: Dict[str, int] = {
        name: count for name, count in name_counts.items() if count > 1
    }

    active = total - archived

    logics_without_links_json = total - logics_with_links_json

    return LogicSummary(
        total=total,
        archived=archived,
        active=active,
        total_links=total_links,
        links_with_from_logic=links_with_from_logic,
        links_missing_from_logic=links_missing_from_logic,
        logics_with_links_json=logics_with_links_json,
        logics_without_links_json=logics_without_links_json,
        logics_with_category=logics_with_category,
        logics_without_category=logics_without_category,
        logics_with_unknown_category=logics_with_unknown_category,
        unknown_categories=unknown_categories,
        extra_logic_fields=extra_logic_fields,
        extra_link_fields=extra_link_fields,
        duplicate_logic_ids=duplicate_logic_ids,
        short_code_name_collisions=short_code_name_collisions,
        duplicate_logic_names=duplicate_logic_names,
        links_missing_required_fields=links_missing_required_fields,
        links_with_blank_required_fields=links_with_blank_required_fields,
    )


def _get_model_logic_ids(model: Dict[str, Any]) -> List[str]:
    """
    For integrity checking we treat logicIds as canonical. Older models may
    still have businessLogicIds; if logicIds is missing or empty and
    businessLogicIds is present, we include those for completeness.
    """
    logic_ids = model.get("logicIds")
    ids: List[str] = []

    if isinstance(logic_ids, list):
        ids = [v for v in logic_ids if isinstance(v, str) and v]

    if not ids:
        legacy = model.get("businessLogicIds")
        if isinstance(legacy, list):
            ids = [v for v in legacy if isinstance(v, str) and v]

    return ids


def _get_hierarchies(model: Dict[str, Any]) -> List[Dict[str, Any]]:
    hierarchies = model.get("hierarchies")
    if isinstance(hierarchies, list):
        return [h for h in hierarchies if isinstance(h, dict)]
    return []


def _get_hierarchy_top_id(h: Dict[str, Any]) -> Optional[str]:
    """
    Support both topId and topDecisionId (older data).
    """
    for key in ("topId", "topDecisionId"):
        top = h.get(key)
        if isinstance(top, str) and top:
            return top
    return None


def _summarize_models_and_hierarchies(
    models: List[Dict[str, Any]],
    logics_by_id: Dict[str, Dict[str, Any]],
) -> ModelSummary:
    total_models = len(models)
    total_model_logic_refs = 0
    model_logic_ref_set: set[str] = set()
    missing_model_logic_refs = 0

    total_hierarchies = 0
    hierarchies_with_top = 0
    hierarchies_missing_top = 0
    hierarchy_tops_missing_logic = 0

    # Allowed schema based on current examples (plus legacy support)
    allowed_model_keys = {
        "id",
        "name",
        "description",
        "logicIds",
        "businessLogicIds",  # legacy support
        "hierarchies",
        "status",
    }

    allowed_hierarchy_keys = {
        "topId",
        "topDecisionId",  # legacy support
        "name",
        "description",
        "useGraph",
    }

    extra_model_fields: Dict[str, int] = {}
    extra_hierarchy_fields: Dict[str, int] = {}

    for model in models:
        if not isinstance(model, dict):
            continue

        # Schema check for model fields
        model_keys = set(model.keys())
        unexpected_model_keys = model_keys - allowed_model_keys
        for key in unexpected_model_keys:
            extra_model_fields[key] = extra_model_fields.get(key, 0) + 1

        # Model → logics references
        logic_ids = _get_model_logic_ids(model)
        total_model_logic_refs += len(logic_ids)
        for lid in logic_ids:
            model_logic_ref_set.add(lid)
            if lid not in logics_by_id:
                missing_model_logic_refs += 1

        # Hierarchies per model
        hierarchies = _get_hierarchies(model)
        total_hierarchies += len(hierarchies)

        for h in hierarchies:
            if not isinstance(h, dict):
                continue

            # Schema check for hierarchy fields
            h_keys = set(h.keys())
            unexpected_h_keys = h_keys - allowed_hierarchy_keys
            for key in unexpected_h_keys:
                extra_hierarchy_fields[key] = extra_hierarchy_fields.get(key, 0) + 1

            top_id = _get_hierarchy_top_id(h)
            if top_id:
                hierarchies_with_top += 1
                if top_id not in logics_by_id:
                    hierarchy_tops_missing_logic += 1
            else:
                hierarchies_missing_top += 1

    unique_model_logic_refs = len(model_logic_ref_set)

    return ModelSummary(
        total_models=total_models,
        total_model_logic_refs=total_model_logic_refs,
        unique_model_logic_refs=unique_model_logic_refs,
        missing_model_logic_refs=missing_model_logic_refs,
        total_hierarchies=total_hierarchies,
        hierarchies_with_top=hierarchies_with_top,
        hierarchies_missing_top=hierarchies_missing_top,
        hierarchy_tops_missing_logic=hierarchy_tops_missing_logic,
        extra_model_fields=extra_model_fields,
        extra_hierarchy_fields=extra_hierarchy_fields,
    )


def _print_summary(
    model_home: str,
    logic_summary: LogicSummary,
    model_summary: ModelSummary,
    kg_summary: Optional[KGSummary],
) -> List[Tuple[str, str]]:
    print()
    print("=== Integrity Summary ===")
    print(f"Model home: {model_home}")
    print()

    print("Summary Statistics")
    print("------------------")
    print()

    print("Logics")
    print("------")
    print(f"Total logics:                {logic_summary.total}")
    print(f"Active (not archived):       {logic_summary.active}")
    print(f"Archived:                    {logic_summary.archived}")
    print()
    print(f"Total links:                        {logic_summary.total_links}")
    print(f"Links with resolvable 'from' logic: {logic_summary.links_with_from_logic}")
    print(f"Links with missing 'from' logic:    {logic_summary.links_missing_from_logic}")
    print(f"Logics with links in JSON:          {logic_summary.logics_with_links_json}")
    print(f"Logics with no links in JSON:       {logic_summary.logics_without_links_json}")
    print(f"Links missing required fields:      {logic_summary.links_missing_required_fields}")
    print(f"Links with blank required fields:   {logic_summary.links_with_blank_required_fields}")
    print(f"Logics with category:               {logic_summary.logics_with_category}")
    print(f"Logics with no category:            {logic_summary.logics_without_category}")
    print(f"Logics with unknown category:       {logic_summary.logics_with_unknown_category}")
    print()

    if logic_summary.unknown_categories:
        print("Unknown categories (name: count of logics):")
        for name, count in sorted(logic_summary.unknown_categories.items()):
            print(f"  {name}: {count}")
        print()

    if logic_summary.extra_logic_fields:
        print("Unexpected logic fields (field: count of logics):")
        for key, count in sorted(logic_summary.extra_logic_fields.items()):
            print(f"  {key}: {count}")
        print()

    if logic_summary.extra_link_fields:
        print("Unexpected link fields (field: count of links):")
        for key, count in sorted(logic_summary.extra_link_fields.items()):
            print(f"  {key}: {count}")
        print()

    if logic_summary.duplicate_logic_ids:
        print("Duplicate logic IDs (id: count of occurrences):")
        for lid, count in sorted(logic_summary.duplicate_logic_ids.items()):
            print(f"  {lid}: {count}")
        print()

    if logic_summary.short_code_name_collisions:
        print('Short code + name collisions ("SHORT | Name": count of logics):')
        for key, count in sorted(logic_summary.short_code_name_collisions.items()):
            print(f"  {key}: {count}")
        print()

    if logic_summary.duplicate_logic_names:
        print("Duplicate logic names (name: count of logics):")
        for name, count in sorted(logic_summary.duplicate_logic_names.items()):
            print(f"  {name}: {count}")
        print()

    print("Models & Hierarchies")
    print("--------------------")
    print(f"Total models:                        {model_summary.total_models}")
    print(f"Total logic refs from models:        {model_summary.total_model_logic_refs}")
    print(f"Unique logic refs from models:       {model_summary.unique_model_logic_refs}")
    print(f"Missing logic refs (no such logic):  {model_summary.missing_model_logic_refs}")
    print()
    print(f"Total hierarchies:                   {model_summary.total_hierarchies}")
    print(f"Hierarchies with a top:              {model_summary.hierarchies_with_top}")
    print(f"Hierarchies missing a top:           {model_summary.hierarchies_missing_top}")
    print(f"Hierarchy tops missing logic:        {model_summary.hierarchy_tops_missing_logic}")
    print()

    if model_summary.extra_model_fields:
        print("Unexpected model fields (field: count of models):")
        for key, count in sorted(model_summary.extra_model_fields.items()):
            print(f"  {key}: {count}")
        print()

    if model_summary.extra_hierarchy_fields:
        print("Unexpected hierarchy fields (field: count of hierarchies):")
        for key, count in sorted(model_summary.extra_hierarchy_fields.items()):
            print(f"  {key}: {count}")
        print()

    if kg_summary is not None:
        print("Knowledge Graph (LogicStep)")
        print("---------------------------")
        if kg_summary.graph_url:
            print(f"Graph base URL:                     {kg_summary.graph_url}")
        print(f"Total LogicStep nodes:                  {kg_summary.total_logic_steps}")
        print(f"LogicStep nodes with id:                {kg_summary.logic_steps_with_id}")
        print(f"LogicStep nodes without id:             {kg_summary.logic_steps_without_id}")
        print(f"Logics with a LogicStep:                {kg_summary.logic_ids_with_step}")
        print(f"Base logics missing in KG (NOT OK):     {kg_summary.logic_ids_without_step}")
        print(f"Top/Composite missing in KG (OK):       {kg_summary.ok_missing_logic_count}")
        print(f"Total logics missing in KG:             {kg_summary.logic_ids_without_step + kg_summary.ok_missing_logic_count}")
        print(f"LogicSteps with no JSON logic:          {kg_summary.steps_with_missing_logic}")
        print(f"SUPPORTS relationships (LS→LS):         {kg_summary.total_supports_rels}")
        print(f"Logics with SUPPORTS links in KG:       {kg_summary.logic_ids_with_supports}")
        print(f"SUPPORTS cycles detected:               {kg_summary.supports_cycles_count}")
        print(f"Mutual SUPPORTS pairs:                  {kg_summary.mutual_support_pairs_count}")
        print()

    issues = _collect_issues(logic_summary, model_summary, kg_summary)

    print("Detected Issues")
    print("---------------")
    if not issues:
        print("No obvious integrity issues detected.")
    else:
        for i, (key, description) in enumerate(issues, start=1):
            # determine count for this issue type
            if key == "links_missing_from_logic":
                count = logic_summary.links_missing_from_logic
            elif key == "links_missing_required_fields":
                count = logic_summary.links_missing_required_fields
            elif key == "links_with_blank_required_fields":
                count = logic_summary.links_with_blank_required_fields
            elif key == "missing_model_logic_refs":
                count = model_summary.missing_model_logic_refs
            elif key == "hierarchy_tops_missing_logic":
                count = model_summary.hierarchy_tops_missing_logic
            elif key == "hierarchies_missing_top":
                count = model_summary.hierarchies_missing_top
            elif key == "extra_logic_fields":
                count = sum(logic_summary.extra_logic_fields.values())
            elif key == "extra_link_fields":
                count = sum(logic_summary.extra_link_fields.values())
            elif key == "extra_model_fields":
                count = sum(model_summary.extra_model_fields.values())
            elif key == "extra_hierarchy_fields":
                count = sum(model_summary.extra_hierarchy_fields.values())
            elif key == "duplicate_logic_ids":
                count = sum(logic_summary.duplicate_logic_ids.values())
            elif key == "short_code_name_collisions":
                count = sum(logic_summary.short_code_name_collisions.values())
            elif key == "duplicate_logic_names":
                count = sum(logic_summary.duplicate_logic_names.values())
            elif key == "logics_with_unknown_category":
                count = logic_summary.logics_with_unknown_category
            elif key == "kg_logic_ids_without_step" and kg_summary:
                count = kg_summary.logic_ids_without_step
            elif key == "kg_steps_with_missing_logic" and kg_summary:
                count = kg_summary.steps_with_missing_logic
            elif key == "kg_supports_cycles" and kg_summary:
                count = kg_summary.supports_cycles_count
            elif key == "kg_mutual_support_pairs" and kg_summary:
                count = kg_summary.mutual_support_pairs_count
            else:
                count = 0

            print(f"  {i}. {description} (count={count})")
    print()

    return issues


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check basic integrity and summary stats for models.json and business_rules.json."
    )
    parser.add_argument(
        "--model-home",
        help="Path to the model-home directory (containing models.json and business_rules.json). "
        "If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--graph-base-url",
        help=(
            "Base URL for the graph routes "
            f"(default: {DEFAULT_GRAPH_BASE_URL}). "
            "If set to an empty string, KG checks are disabled. "
            "When non-empty, the script will summarize LogicStep nodes in the KG "
            "and check integrity of links between JSON logics and LogicSteps."
        ),
    )
    args = parser.parse_args(argv)

    model_home = args.model_home or _prompt_model_home()
    if not model_home:
        _eprint("No model-home provided. Aborting.")
        return 1

    model_home = os.path.abspath(os.path.expanduser(model_home))
    if not os.path.isdir(model_home):
        _eprint(f"Model-home is not a directory: {model_home}")
        return 1

    # --- Optional spec selection for KG URL (mirrors ingest_domain.py) ---
    # If the user has not explicitly provided --graph-base-url, allow them
    # to pick a spec file from tools/spec and honour its "graph-url" value.
    if args.graph_base_url is None:
        spec_files = _find_spec_files()
        if spec_files:
            spec_path = _choose_spec(spec_files)
            if spec_path:
                try:
                    spec = _load_spec(spec_path)
                    spec_graph_url = spec.get("graph-url")
                    if isinstance(spec_graph_url, str) and spec_graph_url.strip():
                        args.graph_base_url = spec_graph_url.strip()
                        _eprint(f"Using graph-url from spec: {args.graph_base_url}")
                except Exception as exc:  # noqa: BLE001
                    _eprint(f"WARNING: Failed to load spec from {spec_path}: {exc}")

    try:
        logics, logics_by_id = _load_logics(model_home)
        models, _ = _load_models(model_home)
    except Exception as exc:
        _eprint(f"ERROR: {exc}")
        return 1

    category_names = _load_category_names(model_home)

    kg_summary: Optional[KGSummary] = None
    # If the user passes an explicit value, use it; otherwise, default to the
    # dev server base URL. An explicit empty string disables KG checks.
    graph_base_url = args.graph_base_url if args.graph_base_url is not None else DEFAULT_GRAPH_BASE_URL
    graph_base_url = graph_base_url.strip()
    if graph_base_url:
        kg_summary = _load_kg_summary(graph_base_url, logics_by_id)

    logic_summary = _summarize_logics(logics, logics_by_id, category_names)
    model_summary = _summarize_models_and_hierarchies(models, logics_by_id)

    issues = _print_summary(model_home, logic_summary, model_summary, kg_summary)

    # Interactive drill-down into issues, if any were detected.
    if issues:
        while True:
            print("Enter an issue number for details, 'a' for all, or press ENTER to finish.")
            choice = input("> ").strip().lower()
            if choice in {"", "q", "quit", "exit"}:
                break
            if choice == "a":
                for idx, (key, description) in enumerate(issues, start=1):
                    print()
                    print(f"Details for issue {idx}: {description}")
                    _print_issue_details(
                        key,
                        model_home,
                        logics,
                        models,
                        logics_by_id,
                        logic_summary,
                        model_summary,
                        kg_summary,
                        category_names,
                    )
                print()
                continue
            try:
                idx = int(choice)
            except ValueError:
                print("Invalid input. Please enter a number, 'a' for all, or press ENTER to finish.")
                continue
            if not (1 <= idx <= len(issues)):
                print(f"Invalid issue number. Please enter a value between 1 and {len(issues)}.")
                continue
            key, description = issues[idx - 1]
            print()
            print(f"Details for issue {idx}: {description}")
            _print_issue_details(
                key,
                model_home,
                logics,
                models,
                logics_by_id,
                logic_summary,
                model_summary,
                kg_summary,
                category_names,
            )
            print()

    # We always return 0 as this is a reporting-only tool; issues are printed but
    # do not cause a non-zero exit code.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
