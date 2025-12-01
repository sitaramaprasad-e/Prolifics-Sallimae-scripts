#!/usr/bin/env python3
# categorize_logics.py
import os
import json
import shutil
import subprocess
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Optional
import re
import glob
import time

# Reusable pcpt helpers
from helpers.call_pcpt import build_output_path, clean_previous_outputs, run_pcpt_for_logic

# ----------------------------
# Model home prompt (defaults to home dir)
# ----------------------------

def _prompt_model_home() -> str:
    try:
        resp = input("Enter model home path (default='~'): ").strip()
    except EOFError:
        # Non-interactive (e.g., piped/cron) – fall back to default
        resp = ""
    if not resp:
        resp = "~"
    return os.path.expanduser(resp)

BASE_HOME = os.path.abspath(os.path.expanduser(_prompt_model_home()))
MODEL_HOME = os.path.join(BASE_HOME, ".model")
os.makedirs(MODEL_HOME, exist_ok=True)

# ----------------------------
# Configuration
# ----------------------------
CATEGORIES_JSON = os.path.join(MODEL_HOME, "rule_categories.json")
BUSINESS_RULES_JSON = os.path.join(MODEL_HOME, "business_rules.json")
EXECUTIONS_JSON = os.path.join(MODEL_HOME, "executions.json")
LOG_DIR = os.path.expanduser("~/.pcpt/log")
LOG_SUBDIR = os.path.join(LOG_DIR, "categorise_logic")

TMP_DIR = ".tmp/logic_categorization"
DYNAMIC_LOGIC_FILE = os.path.join(TMP_DIR, "logic.json")

# Per your exact invocation shape:
OUTPUT_DIR_ARG = "docs"
OUTPUT_FILE_ARG = "categorise-logic/categorise-logic.md"
PROMPT_NAME = "categorise-logic.templ"


# ----------------------------
# Normalization for latest business rules format
# ----------------------------
FILE_PREFIX_RE = re.compile(r"^\s*File:\s*", re.IGNORECASE)

def _ensure_logic_id(rule: Dict[str, Any]) -> None:
    """Guarantee each rule has an immutable `id` field in-place."""
    if not rule.get("id"):
        rule["id"] = str(uuid.uuid4())

def _normalize_logic_inplace(logic: Dict[str, Any]) -> None:
    """Normalize fields expected in the latest schema without discarding extras."""
    _ensure_logic_id(logic)
    # Normalize code_file to strip any leading 'File: '
    cf = logic.get("code_file")
    if isinstance(cf, str):
        logic["code_file"] = FILE_PREFIX_RE.sub("", cf).strip()
    # Ensure presence of new optional keys so downstream tooling can rely on them
    for k, default in (
        ("category", None),
        ("business_area", None),
        ("owner", None),
    ):
        logic.setdefault(k, default)
    # Ensure timestamp is a string if present
    ts = logic.get("timestamp")
    if ts is not None:
        logic["timestamp"] = str(ts)

# ----------------------------
# Execution logging helpers
# ----------------------------
def _utc_now_str() -> str:
    """Return a UTC timestamp string in ISO 8601 format with 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _safe_abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    try:
        return os.path.abspath(path)
    except Exception:
        return path

def _load_executions() -> List[Dict[str, Any]]:
    if not os.path.exists(EXECUTIONS_JSON):
        return []
    try:
        data = load_json(EXECUTIONS_JSON)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def _save_executions(executions: List[Dict[str, Any]]) -> None:
    dump_json(EXECUTIONS_JSON, executions)

def _ensure_execution_record(
    executions: List[Dict[str, Any]],
    exec_type: str,
    log_path: str,
    input_artifacts: List[str],
    output_artifact: str,
    logic_ids: List[str],
) -> str:
    """
    Ensure a single execution record in the new schema:

    {
      "id": "...",
      "type": "Categorize Rules",
      "created_at": "2025-08-29T12:55:50Z",
      "log_path": ".../categorise_logic/run_categorise_YYYYMMDD_HHMMSS.log",
      "input_artifacts": [".../.model/business_rules.json"],
      "output_artifact": ".../docs/categorise-logic/categorise-logic.md",
      "logic_ids": ["..."]
    }
    """
    log_abs = _safe_abs(log_path)

    # Try to reuse an existing execution keyed on the absolute log path
    for exec_obj in executions:
        if _safe_abs(exec_obj.get("log_path")) == log_abs:
            # Normalize to new schema
            exec_obj["type"] = exec_type or exec_obj.get("type") or "Categorize Logics"

            prev_inputs = set(exec_obj.get("input_artifacts", []))
            exec_obj["input_artifacts"] = sorted(prev_inputs.union({_safe_abs(a) for a in input_artifacts if a}))

            if output_artifact:
                exec_obj["output_artifact"] = _safe_abs(output_artifact)

            prev_logic_ids = set(exec_obj.get("logic_ids", []))
            exec_obj["logic_ids"] = sorted(prev_logic_ids.union({rid for rid in logic_ids if rid}))

            # Drop old fields from previous schema if present
            exec_obj.pop("artifacts", None)
            exec_obj.pop("output_report_path", None)

            return exec_obj.get("id") or ""

    # Create new execution
    new_id = str(uuid.uuid4())
    new_exec = {
        "id": new_id,
        "type": exec_type,
        "created_at": _utc_now_str(),
        "log_path": log_abs,
        "input_artifacts": sorted({_safe_abs(a) for a in input_artifacts if a}),
        "output_artifact": _safe_abs(output_artifact),
        "logic_ids": sorted({rid for rid in logic_ids if rid}),
    }
    executions.append(new_exec)
    return new_id

def _write_execution_log(
    logic: Dict[str, Any],
    dynamic_logic_file: str,
    cmd: List[str],
    output_report_path: str,
    selected_category: Optional[Dict[str, Any]],
) -> str:
    os.makedirs(LOG_SUBDIR, exist_ok=True)
    # Build a filename that includes timestamp and a slugged logic name
    raw_name = str(logic.get("name") or logic.get("name") or "unnamed_logic")
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in raw_name)[:60].strip("-_")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_filename = f"log_categorise_{ts}_{slug or 'rule'}.log"
    log_path = os.path.join(LOG_SUBDIR, log_filename)

    # Read the dynamic payload (if available) to include in the log
    try:
        dynamic_payload = load_json(dynamic_logic_file)
        dynamic_payload_str = json.dumps(dynamic_payload, indent=2, ensure_ascii=False)
    except Exception:
        dynamic_payload_str = "(unable to read dynamic logic payload)"

    selected_block = json.dumps(selected_category, indent=2, ensure_ascii=False) if selected_category else "(none)"

    lines = [
        f"Created At: {_utc_now_str()}",
        f"Logic Name: {raw_name}",
        f"Logic ID: {str(logic.get('id') or '')}",
        f"Dynamic Logic File: {_safe_abs(dynamic_logic_file)}",
        f"Output Report Path: {_safe_abs(output_report_path)}",
        "Command:",
        "  " + " ".join(cmd),
        "",
        "Selected Category:",
        selected_block,
        "",
        "Dynamic Payload:",
        dynamic_payload_str,
        "",
    ]
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines))
    except Exception:
        # Best-effort; if logging fails, still return a path (may be non-existent)
        pass

    return log_path

def _start_run_log() -> str:
    os.makedirs(LOG_SUBDIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_log = os.path.join(LOG_SUBDIR, f"run_categorise_{ts}.log")
    header = [
        f"Run Started: {_utc_now_str()}",
        "Type: categorize_logics",
        ""
    ]
    with open(run_log, "w", encoding="utf-8") as f:
        f.write("\n".join(header))
    return run_log
def _append_run_log(run_log: str, lines: List[str]) -> None:
    try:
        with open(run_log, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines and not lines[-1].endswith("\n") else ""))
    except Exception:
        pass

 # ----------------------------
# Console helpers (enhanced output)
# ----------------------------
def _log(msg: str, header: bool = False) -> None:
    """Print timestamped log messages, with optional header formatting."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    if header:
        print(f"\n\033[95m[{ts}] ╔══════════════════════════════════════════════════════════╗\033[0m")
        print(f"\033[95m[{ts}] ║ {msg}\033[0m")
        print(f"\033[95m[{ts}] ╚══════════════════════════════════════════════════════════╝\033[0m")
    else:
        print(f"\033[90m[{ts}]\033[0m {msg}")

def _log_cmd(cmd: List[str]) -> None:
    """Pretty-print a shell command in bold."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\033[90m[{ts}]\033[0m $ \033[1m{' '.join(cmd)}\033[0m")

# ----------------------------
# Utilities
# ----------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def ensure_paths() -> None:
    os.makedirs(TMP_DIR, exist_ok=True)
    # Make sure rule_categories.json exists
    if not os.path.exists(CATEGORIES_JSON):
        raise FileNotFoundError(f"Missing {CATEGORIES_JSON}")
    # Make sure business_rules.json exists
    if not os.path.exists(BUSINESS_RULES_JSON):
        raise FileNotFoundError(f"Missing {BUSINESS_RULES_JSON}")

    # Stage rule categories into TMP so both inputs live under a single mount point
    categories_tmp_path = os.path.join(TMP_DIR, "rule_categories.json")
    try:
        shutil.copyfile(CATEGORIES_JSON, categories_tmp_path)
    except Exception as e:
        raise RuntimeError(f"Failed to stage categories into {TMP_DIR}: {e}")


# ----------------------------
# Category filtering helpers
# ----------------------------
def _norm_team(val: Optional[str]) -> str:
    return (val or "").strip().lower()

def _is_leaf_category(cat: Dict[str, Any]) -> bool:
    """Return True if this category entry looks like a leaf (not a group).

    Heuristics:
    - If it has an explicit group indicator (type/isGroup/kind), treat as group.
    - If it has nested children, treat as group.
    Otherwise, assume it is a leaf category.
    """
    if not isinstance(cat, dict):
        return False

    type_val = str(cat.get("type", "")).strip().lower()
    kind_val = str(cat.get("kind", "")).strip().lower()

    if type_val in {"group", "category_group", "grouping"}:
        return False
    if kind_val in {"group", "category_group", "grouping"}:
        return False
    if bool(cat.get("isGroup")) or bool(cat.get("is_group")):
        return False

    # If this category has nested children/categories, treat it as a group
    if isinstance(cat.get("children"), list) and cat["children"]:
        return False
    if isinstance(cat.get("categories"), list) and cat["categories"]:
        return False

    # Otherwise, we treat it as a leaf
    return True

def filter_categories_for_logic(rule: Dict[str, Any]) -> str:
    """Create a filtered copy of rule_categories.json that includes only
    categories with no team OR a team matching the rule's owner (team),
    and only leaf categories (not groups).
    Returns the path to the filtered categories file under TMP_DIR.
    """
    categories_src = CATEGORIES_JSON
    categories_dst = os.path.join(TMP_DIR, "categories.filtered.json")

    try:
        data = load_json(categories_src)
    except Exception as e:
        raise RuntimeError(f"Failed to read {categories_src}: {e}")

    owner_team = _norm_team(rule.get("owner"))

    # The file is expected to be an object with a "categories" array.
    # We preserve all other keys as-is and only filter the array.
    if isinstance(data, dict) and isinstance(data.get("categories"), list):
        filtered = []
        for cat in data["categories"]:
            if not isinstance(cat, dict):
                continue

            # Only keep leaf categories, never groups
            if not _is_leaf_category(cat):
                continue

            team_val = _norm_team(cat.get("team"))
            # keep when no team is specified or it's a match (case-insensitive)
            if team_val == "" or team_val == owner_team:
                filtered.append(cat)
        # Replace with filtered list (even if empty — that's intentional)
        data["categories"] = filtered
    else:
        # If structure is unexpected, do not filter to avoid masking data
        pass

    dump_json(categories_dst, data)
    return categories_dst

def is_missing_category(rule: Dict[str, Any]) -> bool:
    cat = rule.get("category")
    return cat is None or (isinstance(cat, str) and cat.strip() == "")


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ensure_paths()
    _log("Start: Categorize Logics", header=True)
    # Clean all prior outputs that match the suffixed/unsuffixed patterns
    clean_previous_outputs(OUTPUT_DIR_ARG, OUTPUT_FILE_ARG)
    _log("Step 1: Clean previous outputs", header=True)
    executions: List[Dict[str, Any]] = _load_executions()

    run_log_path = _start_run_log()
    categorized_logic_ids: List[str] = []
    per_logic_log_paths: List[str] = []

    # Backup current business rules (for safety)
    backup_path = f"{BUSINESS_RULES_JSON}.bak"
    try:
        shutil.copyfile(BUSINESS_RULES_JSON, backup_path)
        print(f"🧷 Backup created: {backup_path}")
    except Exception as e:
        print(f"⚠️ Could not create backup ({backup_path}): {e}")

    # Load business rules, supporting both legacy list and rooted {version, logics} shapes
    original_version: Optional[int] = None
    root: Optional[Dict[str, Any]] = None

    data = load_json(BUSINESS_RULES_JSON)
    if isinstance(data, list):
        # Legacy shape: top-level array of logics
        logics: List[Dict[str, Any]] = data
    elif isinstance(data, dict) and isinstance(data.get("logics"), list):
        # New shape: { "version": N, "logics": [...] }
        root = data
        logics = data["logics"]
        v = root.get("version")
        if isinstance(v, int):
            original_version = v
    else:
        raise ValueError(f"{BUSINESS_RULES_JSON} is not in a supported format")

    assigned_ids = 0
    for r in logics:
        if isinstance(r, dict):
            had_id = bool(r.get("id"))
            _normalize_logic_inplace(r)
            if not had_id and r.get("id"):
                assigned_ids += 1

    total = len(logics)
    _log(f"Step 2: Scan {total} logic(s)", header=True)
    # Precompute target rules and total
    target_logics = [
        r for r in logics
        if isinstance(r, dict) and is_missing_category(r) and not r.get("archived", False)
    ]
    total_targets = len(target_logics)
    unchanged_with_category = len([r for r in logics if isinstance(r, dict) and not is_missing_category(r)])
    _log(f"Step 3: Categorize {total_targets} logic(s) needing category", header=True)
    skipped = 0
    categorized = 0

    for running_index, logic in enumerate(target_logics, start=1):
        # Build a one-item array as your "dynamically generated" logic.json
        single_rule_payload = [logic]

        # Write dynamic file
        dump_json(DYNAMIC_LOGIC_FILE, single_rule_payload)

        # Build a categories file filtered by the rule's owner/team
        categories_filtered_path = filter_categories_for_logic(logic)

        # Construct the command for logging purposes (matches run_pcpt_for_logic)
        cmd_for_log = [
            "pcpt.sh",
            "run-custom-prompt",
            "--input-file",
            categories_filtered_path,
            "--input-file2",
            DYNAMIC_LOGIC_FILE,
            "--output",
            OUTPUT_DIR_ARG,
        ]
        if total_targets > 0:
            cmd_for_log.extend(["--index", str(running_index), "--total", str(total_targets)])
        cmd_for_log.extend([DYNAMIC_LOGIC_FILE, PROMPT_NAME])
        _log_cmd(cmd_for_log)

        # Run pcpt.sh on that dynamic file
        result = run_pcpt_for_logic(
            DYNAMIC_LOGIC_FILE,
            categories_filtered_path,
            OUTPUT_DIR_ARG,
            OUTPUT_FILE_ARG,
            PROMPT_NAME,
            index=running_index,
            total=total_targets,
        )
        if not result or "selectedCategory" not in result:
            print(f"⚠️ No selectedCategory returned for logic '{logic.get('name','(unnamed)')}'. Skipping.")
            skipped += 1
            continue

        selected = result["selectedCategory"]
        name = selected.get("name")
        explanation = selected.get("explanation")

        if not name:
            print(f"⚠️ selectedCategory missing 'name' for logic '{logic.get('name','(unnamed)')}'. Skipping.")
            skipped += 1
            continue

        # Update rule fields in-place
        logic["category"] = name
        # Only add the two fields you requested:
        logic["ai_categorized"] = True
        logic["category_explanation"] = explanation or ""

        categorized += 1
        print(f"✅ Categorized: {logic.get('name','(unnamed)')} → {name}")

        # --- Execution logging per categorized logic ---
        # 1) Write a log file capturing inputs/outputs for this categorization
        current_output_path = build_output_path(OUTPUT_DIR_ARG, OUTPUT_FILE_ARG, index=running_index, total=total_targets)
        log_path = _write_execution_log(
            logic=logic,
            dynamic_logic_file=DYNAMIC_LOGIC_FILE,
            cmd=cmd_for_log,
            output_report_path=current_output_path,
            selected_category=selected,
        )

        per_logic_log_paths.append(log_path)
        rid = str(logic.get("id") or "")
        if rid:
            categorized_logic_ids.append(rid)
        # Also append a short entry to the run log for visibility
        _append_run_log(run_log_path, [
            f"Categorized: {logic.get('name','(unnamed)')} -> {name}",
            f"  Logic ID: {rid or '(none)'}",
            f"  Per-Logic Log: {log_path}",
            ""
        ])

    # --- One execution per entire run (for run-log display only) ---
    OUTPUT_PARENT_DIR = os.path.join(OUTPUT_DIR_ARG, os.path.dirname(OUTPUT_FILE_ARG))  # e.g., docs/categorise-logic
    run_artifacts = [
        CATEGORIES_JSON,
        BUSINESS_RULES_JSON,
        OUTPUT_PARENT_DIR,
    ]
    _append_run_log(run_log_path, [
        "──────── Run Summary ────────",
        f"Total logics scanned: {total}",
        f"Already had category: {unchanged_with_category}",
        f"Newly AI-categorized: {categorized}",
        f"Skipped (no/invalid output): {skipped}",
        "Logics included in this run (IDs):",
        ", ".join(categorized_logic_ids) if categorized_logic_ids else "(none)",
        "",
        "Per-logic logs:",
        *(per_logic_log_paths if per_logic_log_paths else ["(none)"]),
        "",
        f"Output Reports Dir: {OUTPUT_PARENT_DIR}",
        f"Run Completed: {_utc_now_str()}",
    ])
    _ensure_execution_record(
        executions=executions,
        exec_type="Categorize Logics",
        log_path=run_log_path,
        input_artifacts=[BUSINESS_RULES_JSON],
        output_artifact=OUTPUT_PARENT_DIR,
        logic_ids=categorized_logic_ids,
    )

    _save_executions(executions)
    print(f"🧾 Executions updated: {EXECUTIONS_JSON}")

    # Persist changes
    _log("Summary", header=True)
    print(f"Total logics:                  {total}")
    print(f"Already had category:         {unchanged_with_category}")
    print(f"Newly AI-categorized:         {categorized}")
    print(f"Skipped (no/invalid output):  {skipped}")
    print(f"IDs assigned (backfill):       {assigned_ids}")

    # Optimistic concurrency check for rooted, versioned files
    if root is not None and original_version is not None:
        try:
            current = load_json(BUSINESS_RULES_JSON)
            if isinstance(current, dict) and isinstance(current.get("logics"), list):
                cur_v = current.get("version")
                if isinstance(cur_v, int) and cur_v != original_version:
                    print(
                        f"⚠️ Version conflict for {BUSINESS_RULES_JSON}: "
                        f"expected version {original_version}, found {cur_v}; "
                        "aborting without writing changes."
                    )
                    return
        except Exception as e:
            print(f"⚠️ Failed optimistic concurrency check for {BUSINESS_RULES_JSON}: {e}")
            return

    # Write back, preserving structure and bumping version for rooted files
    if root is not None:
        current_version = root.get("version")
        if isinstance(current_version, int):
            root["version"] = current_version + 1
        else:
            root["version"] = 1
        root["logics"] = logics
        dump_json(BUSINESS_RULES_JSON, root)
    else:
        # Legacy list-only shape
        dump_json(BUSINESS_RULES_JSON, logics)

    print(f"📄 Updated file: {BUSINESS_RULES_JSON}")

if __name__ == "__main__":
    main()