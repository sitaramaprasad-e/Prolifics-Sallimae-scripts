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
from typing import Optional, Dict, Any, List

parser = argparse.ArgumentParser()
parser.add_argument("--report-only", action="store_true")
args = parser.parse_args()
report_only = args.report_only

# --- Neo4j / graph API configuration (lazy, best-effort) ---
GRAPH_BASE_URL = os.getenv("RULES_PORTAL_BASE_URL", "http://localhost:443").rstrip("/")
GRAPH_ENABLED_ENV = os.getenv("RULES_PORTAL_GRAPH_ENABLED", "true").strip().lower()
GRAPH_ENABLED = GRAPH_ENABLED_ENV in {"1", "true", "yes", "on"}

# Toggle TLS certificate verification for KG HTTP calls. In many dev and some
# internal environments we use self-signed or private CAs, so by default we
# disable verification here. You can override this by setting the
# RULES_PORTAL_VERIFY environment variable to "true"/"1"/"yes" to enable
# verification.
GRAPH_VERIFY = False
env_verify = os.getenv("RULES_PORTAL_VERIFY")
if env_verify is not None:
    GRAPH_VERIFY = env_verify.strip().lower() in {"1", "true", "yes", "on"}

_GRAPH_STATUS_CHECKED = False
_GRAPH_AVAILABLE = False
_GRAPH_CLEANUP_ERRORS = 0


def graph_is_enabled() -> bool:
    """Return True if graph population is enabled and the backend is reachable.

    We reuse the same env vars as logic_ingest_core and call /api/graph/status
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
        resp = requests.get(status_url, timeout=2, verify=GRAPH_VERIFY)
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
    global _GRAPH_CLEANUP_ERRORS
    if not graph_is_enabled():
        return None

    url = f"{GRAPH_BASE_URL}/api/graph/run"
    payload = {"query": query, "parameters": parameters or {}}

    try:
        resp = requests.post(url, json=payload, timeout=10, verify=GRAPH_VERIFY)
        if not resp.ok:
            print(
                f"[WARN] Graph run failed HTTP {resp.status_code}: {resp.text[:200]}"
            )
            _GRAPH_CLEANUP_ERRORS += 1
            return None
        return resp.json()
    except Exception as e:  # pragma: no cover - defensive
        print(f"[WARN] Graph run error: {e}")
        _GRAPH_CLEANUP_ERRORS += 1
        return None


# --- Spec helper functions for selecting a spec and extracting graph-url ---
def _spec_dir() -> str:
    """
    Locate the default spec directory for this repo.

    This script lives in <repo>/tools/maintenance; specs are under <repo>/tools/spec,
    the same convention used by other tools.
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
    print("\n=== Select a spec file for KG cleanup (optional) ===")
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

def main():
    global GRAPH_BASE_URL

    graph_cleanup_performed = False

    # Optional spec selection for KG URL (mirrors other tools). If the user
    # chooses a spec with a "graph-url" value, we prefer that over the default
    # RULES_PORTAL_BASE_URL.
    spec_files = _find_spec_files()
    if spec_files:
        spec_path = _choose_spec(spec_files)
        if spec_path:
            try:
                spec = _load_spec(spec_path)
                spec_graph_url = spec.get("graph-url")
                if isinstance(spec_graph_url, str) and spec_graph_url.strip():
                    GRAPH_BASE_URL = spec_graph_url.strip().rstrip("/")
                    print(f"[INFO] Using graph-url from spec: {GRAPH_BASE_URL}")
            except Exception as e:
                print(f"[WARN] Failed to load spec from {spec_path}: {e}")

    default_home = Path(os.path.expanduser("~/.model"))
    user_input = input(f"Enter model home directory [{default_home}]: ").strip()
    model_home = Path(user_input) if user_input else default_home

    logics_path = model_home / "business_rules.json"
    if not logics_path.exists():
        print(f"[ERROR] business_rules.json not found at {logics_path}")
        return

    # Create a timestamped ZIP backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_backup_path = logics_path.parent / f"business_rules_backup_{timestamp}.zip"
    try:
        with zipfile.ZipFile(zip_backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(logics_path, arcname="business_rules.json")
        print(f"[INFO] ZIP backup created: {zip_backup_path}")
    except Exception as e:
        print(f"[WARN] Could not create ZIP backup: {e}")

    # Load and filter logic (support legacy list and rooted {"version", "logics"})
    br_root: Any = None
    br_version: Optional[int] = None
    try:
        with open(logics_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        return

    if isinstance(raw, dict) and isinstance(raw.get("logics"), list):
        br_root = raw
        if isinstance(raw.get("version"), int):
            br_version = raw["version"]
        logics = raw["logics"]
    elif isinstance(raw, list):
        logics = raw
    else:
        print(
            "[ERROR] business_rules.json format invalid: expected a list or an object with a 'logics' array."
        )
        return

    archived_logics = [r for r in logics if r.get("archived", False)]

    before_count = len(logics)
    # Remove archived logics (existing behavior)
    filtered = [r for r in logics if not r.get("archived", False)]
    removed_count = before_count - len(filtered)

    # Precompute IDs for archived logics (those we are deleting)
    archived_ids = set()
    for r in archived_logics:
        rid = r.get("id")
        if isinstance(rid, str) and rid.strip():
            archived_ids.add(rid.strip())

    # --- Dangling link cleanup ---
    # Build a set of ALL logic IDs (including archived).
    all_ids = set()
    for r in logics:
        rid = r.get("id")
        if isinstance(rid, str) and rid.strip():
            all_ids.add(rid.strip())

    def _clean_links(logic_obj, all_ids, archived_ids, dry_run=False):
        links = logic_obj.get("links")
        if not isinstance(links, list):
            return [], 0
        removed_links = []
        for l in links:
            if not isinstance(l, dict):
                removed_links.append(l)
                continue
            fsid = (l.get("from_logic_id") or "").strip()
            # Always treat links whose source ID does not exist in all_ids as dangling
            if not fsid or fsid not in all_ids:
                removed_links.append(l)
                continue
            # When we are actually deleting archived logics (not report-only),
            # also remove links whose source points at an archived logic ID,
            # because those logics will no longer exist after this script runs.
            if not dry_run and fsid in archived_ids:
                removed_links.append(l)
        if not dry_run:
            logic_obj["links"] = [l for l in links if l not in removed_links]
        return removed_links, len(removed_links)

    total_links_removed = 0
    for r in filtered:
        try:
            removed_links, count = _clean_links(r, all_ids, archived_ids, dry_run=report_only)
            total_links_removed += count
            if report_only and count > 0:
                if "id" in r:
                    print(f"[REPORT] Logic {r['id']} has {count} dangling link(s):")
                    for dl in removed_links:
                        print(f"         from_logic_id={dl.get('from_logic_id')} to_input={dl.get('to_input')}")
        except Exception:
            # Be robust; if links are malformed, skip cleaning that rule
            pass

    # --- Model logicIds dangling cleanup ---
    # Build a set of remaining (non-archived) logic IDs.
    remaining_logic_ids = set()
    for r in filtered:
        rid = r.get("id")
        if isinstance(rid, str) and rid.strip():
            remaining_logic_ids.add(rid.strip())

    models_path = model_home / "models.json"
    models_root: Any = None
    models_version: Optional[int] = None
    models = None
    dangling_model_refs_total = 0
    models_with_dangling = 0

    if models_path.exists():
        try:
            with open(models_path, "r", encoding="utf-8") as mf:
                raw_models = json.load(mf)
            if isinstance(raw_models, dict) and isinstance(raw_models.get("models"), list):
                models_root = raw_models
                if isinstance(raw_models.get("version"), int):
                    models_version = raw_models["version"]
                models = raw_models["models"]
            elif isinstance(raw_models, list):
                models = raw_models
            else:
                print(
                    f"[WARN] models.json format invalid at {models_path}: expected a list or an object with a 'models' array."
                )
                models = None
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse models.json at {models_path}: {e}")
            models = None
    else:
        print(f"[INFO] models.json not found at {models_path} (skipping model dangling check).")

    if models:
        for m in models:
            model_id = m.get("id")
            model_name = m.get("name")
            bl_ids = m.get("logicIds")
            if not isinstance(bl_ids, list):
                continue

            cleaned_ids = []
            dangling_ids = []

            for bid in bl_ids:
                # We only treat string IDs that are non-empty and present in remaining_logic_ids as valid.
                if not isinstance(bid, str) or not bid.strip():
                    dangling_ids.append(bid)
                    continue
                bid_str = bid.strip()
                if bid_str not in remaining_logic_ids:
                    dangling_ids.append(bid_str)
                else:
                    cleaned_ids.append(bid_str)

            if dangling_ids:
                models_with_dangling += 1
                dangling_model_refs_total += len(dangling_ids)
                if report_only:
                    print(
                        f"[REPORT] Model {model_id} ({model_name}) has {len(dangling_ids)} "
                        f"dangling logicId(s) (logic(s) no longer exist): {', '.join(map(str, dangling_ids))}"
                    )
                else:
                    # Apply the cleanup only when not in report-only mode.
                    m["logicIds"] = cleaned_ids

    if report_only:
        print("[INFO] --report-only mode: no changes written.")
        print(f"[INFO] {len(archived_logics)} logic(s) would be deleted:")
        for r in archived_logics:
            print(f"       {r.get('id')}  {r.get('name')}")
        print(
            f"[INFO] {total_links_removed} link(s) would be cleaned up where the from_logic_id no longer matches an existing logic id or would refer to a logic being deleted."
        )
        return

    # --- Neo4j graph cleanup for deleted logics ---
    # We use DETACH DELETE so that relationships are removed automatically
    # when their endpoint nodes are deleted.
    if archived_logics:
        if graph_is_enabled():
            deleted_logic_ids = {rid for rid in archived_ids}

            # Collect candidate code functions and files from the rules being deleted
            code_functions = set()
            code_files = set()
            for r in archived_logics:
                cf = r.get("code_function")
                if isinstance(cf, str) and cf.strip():
                    code_functions.add(cf.strip())
                cfile = r.get("code_file")
                if isinstance(cfile, str) and cfile.strip():
                    code_files.add(cfile.strip())

            print(
                f"[INFO] Cleaning Neo4j graph for {len(deleted_logic_ids)} archived logic(s) "
                f"(base={GRAPH_BASE_URL})..."
            )

            # 1) Delete LogicStep nodes for each archived logic id (and all their relationships)
            for rid in sorted(deleted_logic_ids):
                # Delete any Message nodes sequenced by this LogicStep, then delete the LogicStep
                graph_run_cypher(
                    """
                    MATCH (l:LogicStep {id: $logicId})
                    OPTIONAL MATCH (l)-[:SEQUENCED_BY]->(m:Message)
                    DETACH DELETE m, l
                    """,
                    {"logicId": rid},
                )

                # Delete any Parameter nodes belonging to this logic
                graph_run_cypher(
                    "MATCH (p:Parameter {logicId: $logicId}) DETACH DELETE p",
                    {"logicId": rid},
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

            # 4) Delete Message nodes that are no longer linked to any LogicStep
            graph_run_cypher(
                """
                MATCH (m:Message)
                WHERE NOT EXISTS { MATCH (m)<-[:SEQUENCED_BY]-(:LogicStep) }
                DETACH DELETE m
                """,
                {},
            )

            # 5) Delete Sequence nodes that no longer have any Message nodes at all
            graph_run_cypher(
                """
                MATCH (s:Sequence)
                WHERE NOT EXISTS { MATCH (s)<-[:PART_OF]-(:Message) }
                DETACH DELETE s
                """,
                {},
            )

            # 6) Delete Domain and DomainType nodes that are no longer linked to any LogicStep.
            #    For any Domain node where none of its DomainType children are the target
            #    of an OPERATES_ON relationship from a LogicStep, delete that Domain and
            #    all of its DomainType nodes.
            graph_run_cypher(
                """
                MATCH (d:Domain)
                WHERE NOT EXISTS {
                  MATCH (d)--(dt:DomainType)<-[:OPERATES_ON]-(:LogicStep)
                }
                WITH collect(d) AS doms
                UNWIND doms AS d
                MATCH (d)--(dt:DomainType)
                DETACH DELETE dt, d
                """,
                {},
            )
            graph_cleanup_performed = True
        else:
            print(
                "[WARN] Graph is disabled or unreachable; archived logics will not be removed from the KG."
            )

    # Decide whether we need to ask for confirmation based on graph cleanup status.
    need_confirm = False
    if archived_logics:
        if not graph_cleanup_performed:
            # Either graph is disabled/unreachable, or we skipped cleanup.
            need_confirm = True
            print(
                "[WARN] Graph cleanup was not performed; archived logics will remain in the KG if any exist there."
            )
        elif _GRAPH_CLEANUP_ERRORS > 0:
            # Cleanup was attempted but some Cypher calls failed.
            need_confirm = True
            print(
                f"[WARN] Graph cleanup encountered {_GRAPH_CLEANUP_ERRORS} error(s); some KG nodes/relationships may remain."
            )
        else:
            # Cleanup ran and no HTTP/transport errors were observed.
            print(
                "[INFO] Graph cleanup completed without HTTP errors; proceeding to update JSON."
            )

    if need_confirm:
        confirm = input(
            "\nProceed with deleting archived logics from JSON and updating models anyway? [y/N]: "
        ).strip().lower()
        if confirm not in ("y", "yes"):
            print("[INFO] Cancelled by user. JSON files were not modified.")
            return

    # If we modified any models, write them back out now (after confirmation).
    if models and dangling_model_refs_total > 0:
        try:
            # Best-effort optimistic concurrency using models_version
            if models_path.exists() and models_version is not None:
                try:
                    with open(models_path, "r", encoding="utf-8") as mf:
                        cur_raw_models = json.load(mf)
                    if isinstance(cur_raw_models, dict) and isinstance(cur_raw_models.get("models"), list):
                        cur_version = cur_raw_models.get("version")
                        if isinstance(cur_version, int) and cur_version != models_version:
                            print(
                                f"[ERROR] models.json version changed on disk (expected {models_version}, found {cur_version}); "
                                f"aborting write to avoid overwriting concurrent changes."
                            )
                            return
                except Exception as ex:
                    print(
                        f"[WARN] Could not re-read models.json for concurrency check; proceeding anyway: {ex}"
                    )

            # Persist using rooted {"version", "models"} where possible
            if isinstance(models_root, dict):
                current_version = models_root.get("version")
                if not isinstance(current_version, int):
                    current_version = models_version or 0
                new_version = current_version + 1
                models_root["models"] = models
                models_root["version"] = new_version
                to_write = models_root
            else:
                # Legacy list-only models.json; upgrade to rooted structure on write
                new_version = (models_version or 0) + 1
                to_write = {"version": new_version, "models": models}

            with open(models_path, "w", encoding="utf-8") as mf:
                json.dump(to_write, mf, indent=2, ensure_ascii=False)
            print(
                f"[INFO] Removed {dangling_model_refs_total} dangling logicId(s) "
                f"from {models_with_dangling} model(s) in {models_path}"
            )
        except Exception as e:
            print(f"[WARN] Failed to write cleaned models.json to {models_path}: {e}")

    # Write updated logics including cleaned links, using rooted {"version", "logics"} and a best-effort optimistic check
    try:
        if logics_path.exists() and br_version is not None:
            try:
                with open(logics_path, "r", encoding="utf-8") as f:
                    cur_raw = json.load(f)
                if isinstance(cur_raw, dict) and isinstance(cur_raw.get("logics"), list):
                    cur_version = cur_raw.get("version")
                    if isinstance(cur_version, int) and cur_version != br_version:
                        print(
                            f"[ERROR] business_rules.json version changed on disk (expected {br_version}, found {cur_version}); "
                            f"aborting write to avoid overwriting concurrent changes."
                        )
                        return
            except Exception as ex:
                print(
                    f"[WARN] Could not re-read business_rules.json for concurrency check; proceeding anyway: {ex}"
                )

        if isinstance(br_root, dict):
            current_version = br_root.get("version")
            if not isinstance(current_version, int):
                current_version = br_version or 0
            new_version = current_version + 1
            br_root["logics"] = filtered
            br_root["version"] = new_version
            to_write = br_root
        else:
            # Legacy list-only business_rules.json; upgrade to rooted structure on write
            new_version = (br_version or 0) + 1
            to_write = {"version": new_version, "logics": filtered}

        with open(logics_path, "w", encoding="utf-8") as f:
            json.dump(to_write, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[ERROR] Failed to write updated business_rules.json: {e}")
        return

    print(f"[INFO] Deleted {removed_count} archived logic(s).")
    print(
        f"[INFO] Removed {total_links_removed} dangling link(s) where from_logic_id no longer matches an existing logic id."
    )
    print(f"[INFO] {len(filtered)} logic(s) remain in {logics_path}")

if __name__ == "__main__":
    main()