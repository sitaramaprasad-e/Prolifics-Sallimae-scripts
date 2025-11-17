#!/usr/bin/env python3
import warnings
# Suppress the LibreSSL/OpenSSL compatibility warning from urllib3 v2
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
    module="urllib3"
)
import json
import os
from pathlib import Path
from datetime import datetime
import zipfile
import argparse
import requests
from typing import Optional, Dict

parser = argparse.ArgumentParser()
parser.add_argument("--report-only", action="store_true")
args = parser.parse_args()
report_only = args.report_only

# --- Neo4j / graph API configuration (lazy, best-effort) ---
GRAPH_BASE_URL = os.getenv("RULES_PORTAL_BASE_URL", "http://localhost:443").rstrip("/")
GRAPH_ENABLED_ENV = os.getenv("RULES_PORTAL_GRAPH_ENABLED", "true").strip().lower()
GRAPH_ENABLED = GRAPH_ENABLED_ENV in {"1", "true", "yes", "on"}

_GRAPH_STATUS_CHECKED = False
_GRAPH_AVAILABLE = False


def graph_is_enabled() -> bool:
    """Return True if graph population is enabled and the backend is reachable.

    We reuse the same env vars as rule_ingest_core and call /api/graph/status
    once per process. Failures are logged but do not stop this script.
    """
    global _GRAPH_STATUS_CHECKED, _GRAPH_AVAILABLE

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
                print(
                    f"[WARN] Neo4j graph API reported up=false at {status_url}: {data.get('message')}"
                )
        else:
            print(
                f"[WARN] Neo4j graph status check failed ({status_url}): HTTP {resp.status_code}"
            )
            _GRAPH_AVAILABLE = False
    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARN] Neo4j graph status check error at {status_url}: {e}")
        _GRAPH_AVAILABLE = False

    return _GRAPH_AVAILABLE


def graph_run_cypher(query: str, parameters: Optional[Dict] = None) -> Optional[Dict]:
    """Execute a Cypher statement via POST /api/graph/run (best-effort).

    Any failures are logged to stdout but do not stop this script.
    """
    if not graph_is_enabled():
        return None

    url = f"{GRAPH_BASE_URL}/api/graph/run"
    payload = {"query": query, "parameters": parameters or {}}

    try:
        resp = requests.post(url, json=payload, timeout=10)
        if not resp.ok:
            print(
                f"[WARN] Graph run failed HTTP {resp.status_code}: {resp.text[:200]}"
            )
            return None
        return resp.json()
    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARN] Graph run error: {e}")
        return None

def main():
    default_home = Path(os.path.expanduser("~/.model"))
    user_input = input(f"Enter model home directory [{default_home}]: ").strip()
    model_home = Path(user_input) if user_input else default_home

    business_rules_path = model_home / "business_rules.json"
    if not business_rules_path.exists():
        print(f"[ERROR] business_rules.json not found at {business_rules_path}")
        return

    # Create a timestamped ZIP backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_backup_path = business_rules_path.parent / f"business_rules_backup_{timestamp}.zip"
    try:
        with zipfile.ZipFile(zip_backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(business_rules_path, arcname="business_rules.json")
        print(f"[INFO] ZIP backup created: {zip_backup_path}")
    except Exception as e:
        print(f"[WARN] Could not create ZIP backup: {e}")

    # Load and filter rules
    with open(business_rules_path, "r", encoding="utf-8") as f:
        try:
            rules = json.load(f)
            if not isinstance(rules, list):
                print(f"[ERROR] business_rules.json format invalid: expected a list.")
                return
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON: {e}")
            return

    archived_rules = [r for r in rules if r.get("archived", False)]

    before_count = len(rules)
    # Remove archived rules (existing behavior)
    filtered = [r for r in rules if not r.get("archived", False)]
    removed_count = before_count - len(filtered)

    # Precompute IDs for archived rules (those we are deleting)
    archived_ids = set()
    for r in archived_rules:
        rid = r.get("id")
        if isinstance(rid, str) and rid.strip():
            archived_ids.add(rid.strip())

    # --- Dangling link cleanup ---
    # Build a set of ALL rule IDs (including archived).
    all_ids = set()
    for r in rules:
        rid = r.get("id")
        if isinstance(rid, str) and rid.strip():
            all_ids.add(rid.strip())

    def _clean_links(rule_obj, all_ids, archived_ids, dry_run=False):
        links = rule_obj.get("links")
        if not isinstance(links, list):
            return [], 0
        removed_links = []
        for l in links:
            if not isinstance(l, dict):
                removed_links.append(l)
                continue
            fsid = (l.get("from_step_id") or "").strip()
            # Always treat links whose source ID does not exist in all_ids as dangling
            if not fsid or fsid not in all_ids:
                removed_links.append(l)
                continue
            # When we are actually deleting archived rules (not report-only),
            # also remove links whose source points at an archived rule ID,
            # because those rules will no longer exist after this script runs.
            if not dry_run and fsid in archived_ids:
                removed_links.append(l)
        if not dry_run:
            rule_obj["links"] = [l for l in links if l not in removed_links]
        return removed_links, len(removed_links)

    total_links_removed = 0
    for r in filtered:
        try:
            removed_links, count = _clean_links(r, all_ids, archived_ids, dry_run=report_only)
            total_links_removed += count
            if report_only and count > 0:
                if "id" in r:
                    print(f"[REPORT] Rule {r['id']} has {count} dangling link(s):")
                    for dl in removed_links:
                        print(f"         from_step_id={dl.get('from_step_id')} to_input={dl.get('to_input')}")
        except Exception:
            # Be robust; if links are malformed, skip cleaning that rule
            pass

    if report_only:
        print("[INFO] --report-only mode: no changes written.")
        print(f"[INFO] {len(archived_rules)} rule(s) would be deleted:")
        for r in archived_rules:
            print(f"       {r.get('id')}  {r.get('rule_name')}")
        print(
            f"[INFO] {total_links_removed} link(s) would be cleaned up where the from_step_id no longer matches an existing rule id or would refer to a rule being deleted."
        )
        return

    # --- Neo4j graph cleanup for deleted rules ---
    # We use DETACH DELETE so that relationships are removed automatically
    # when their endpoint nodes are deleted.
    if graph_is_enabled() and archived_rules:
        deleted_rule_ids = {rid for rid in archived_ids}

        # Collect candidate code functions and files from the rules being deleted
        code_functions = set()
        code_files = set()
        for r in archived_rules:
            cf = r.get("code_function")
            if isinstance(cf, str) and cf.strip():
                code_functions.add(cf.strip())
            cfile = r.get("code_file")
            if isinstance(cfile, str) and cfile.strip():
                code_files.add(cfile.strip())

        print(
            f"[INFO] Cleaning Neo4j graph for {len(deleted_rule_ids)} archived rule(s) "
            f"(base={GRAPH_BASE_URL})..."
        )

        # 1) Delete LogicStep nodes for each archived rule id (and all their relationships)
        #    and then delete any Parameter nodes scoped to that rule id.
        for rid in sorted(deleted_rule_ids):
            # Delete the LogicStep for this rule
            graph_run_cypher(
                "MATCH (l:LogicStep {ruleId: $ruleId}) DETACH DELETE l",
                {"ruleId": rid},
            )

            # Delete any Parameter nodes belonging to this rule
            graph_run_cypher(
                "MATCH (p:Parameter {ruleId: $ruleId}) DETACH DELETE p",
                {"ruleId": rid},
            )

        # 2) Delete CodeFunction nodes that are no longer referenced by any LogicStep
        for func_name in sorted(code_functions):
            graph_run_cypher(
                """
                MATCH (f:CodeFunction {name: $name})
                OPTIONAL MATCH (l:LogicStep)-[:IMPLEMENTED_BY]->(f)
                WITH f, count(l) AS c
                WHERE c = 0
                DETACH DELETE f
                """,
                {"name": func_name},
            )

        # 3) Delete CodeFile nodes that are no longer referenced by any CodeFunction
        for path in sorted(code_files):
            graph_run_cypher(
                """
                MATCH (cf:CodeFile {path: $path})
                OPTIONAL MATCH (f:CodeFunction)-[:PART_OF]->(cf)
                WITH cf, count(f) AS c
                WHERE c = 0
                DETACH DELETE cf
                """,
                {"path": path},
            )

    # Write updated rules including cleaned links
    with open(business_rules_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Deleted {removed_count} archived rule(s).")
    print(
        f"[INFO] Removed {total_links_removed} dangling link(s) where from_step_id no longer matches an existing rule id."
    )
    print(f"[INFO] {len(filtered)} rule(s) remain in {business_rules_path}")

if __name__ == "__main__":
    main()