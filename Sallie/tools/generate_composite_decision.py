#!/usr/bin/env python3
"""
Generate a composed decision report by selecting a model and its rules, then
invoking pcpt.sh run-custom-prompt via the helper wrapper.

Spec implemented:
0) Prompt for location of model home (default to ~/.model) where models.json and business_rules.json live.
1) Read models.json, prompt to select one, and write selected model details to a temp file under .tmp/generate_composite_decision.
2) Export all rules for the selected model (cross-reference using rule UUIDs) to a temp file in the same temp dir.
3) Look for sources_*.json files under tools/spec and allow selection of one.
4) List the pairs in the selected file and allow selection of one; gather source_path, output_path, filter path, and mode.
5) Construct and run a call to pcpt_run_custom_prompt, plugging in source_path, the rules temp file, and the model temp file.
6) Print concise step outputs and stream pcpt output.
"""
# --- import path bootstrap: allow `from tools...` when run as tools/pcpt_pipeline.py ---
from __future__ import annotations
import os as _os, sys as _sys
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# --- end import path bootstrap ---

import json
import os
import sys
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.call_pcpt import pcpt_run_custom_prompt


# ---------------------------
# Console helpers
# ---------------------------

def prompt_with_default(message: str, default: str) -> str:
    try:
        entered = input(f"{message} [{default}]: ").strip()
    except EOFError:
        entered = ""
    return entered or default


def choose_from_list(title: str, items: List[str], default_index: int = 1) -> int:
    """
    Present a 1-based menu of items and return the chosen 1-based index.
    default_index is 1-based.
    """
    if not items:
        raise ValueError(f"No items to choose from for: {title}")
    print(f"\n{title}")
    for i, item in enumerate(items, start=1):
        print(f"  {i}. {item}")
    while True:
        try:
            raw = input(f"Choose 1-{len(items)} [{default_index}]: ").strip()
        except EOFError:
            raw = ""
        if not raw:
            return default_index
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(items):
                return idx
        print("Invalid selection, please try again.")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(p: Path, data: Any) -> None:
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def eprint(*args, **kwargs) -> None:
    print(*args, file=sys.stderr, **kwargs)


# ---------------------------
# Merge helpers for composed decision
# ---------------------------

def _safe_backup_json(path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    backup = path.with_suffix(path.suffix + f".bak-{ts}")
    try:
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except FileNotFoundError:
        # Nothing to back up; that's fine.
        pass
    return backup

def _normalize_rule_for_compare(rule: dict) -> dict:
    # Exclude volatile keys for duplicate detection
    exclude = {"id", "timestamp", "archived"}
    return {k: v for k, v in rule.items() if k not in exclude}

def _load_top_rule_from_report(report_path: Path) -> dict:
    """Load the top-level rule JSON from a report file which may be pure JSON or Markdown containing a JSON block."""
    raw = report_path.read_text(encoding="utf-8").strip()
    # Try full JSON first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start:end+1]
            return json.loads(snippet)
        raise

def merge_top_level_rule_into_model_home(model_home: Path, output_path: Path, selected_model_id: str) -> None:
    """Merge the generated top-level decision back into business_rules.json and models.json safely.

    Steps:
    1) Locate the composed-decision report under output_path.
    2) Parse the JSON for the new rule.
    3) Generate a new UUID; add timestamp and archived fields if missing.
    4) Backup and merge into business_rules.json (skip if equivalent by content).
    5) Backup and update models.json to include the new rule id in the selected model's businessLogicIds.
    """
    # Candidate report locations (support both md/json and nested folder)
    candidates = [
        output_path / "composed-decision-report" / "composed-decision-report.md",
        output_path / "composed-decision-report.md",
        output_path / "composed-decision-report" / "composed-decision-report.json",
        output_path / "composed-decision-report.json",
    ]
    report_file = next((p for p in candidates if p.exists()), None)
    if report_file is None:
        eprint(f"[WARN] Expected composed decision report not found in: {output_path}")
        return

    try:
        new_rule = _load_top_rule_from_report(report_file)
    except Exception as ex:
        eprint(f"[WARN] Could not parse top-level rule JSON from report {report_file.name}: {ex}")
        return

    # Enrich with system fields
    new_rule_id = str(uuid.uuid4())
    new_rule["id"] = new_rule_id
    if "timestamp" not in new_rule:
        new_rule["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if "archived" not in new_rule:
        new_rule["archived"] = False

    # --- business_rules.json merge ---
    br_path = (model_home / "business_rules.json").resolve()
    business_rules = []
    if br_path.exists():
        try:
            business_rules = load_json(br_path)
            if not isinstance(business_rules, list):
                eprint("[WARN] business_rules.json is not a list; initializing a new list.")
                business_rules = []
        except Exception as ex:
            eprint(f"[WARN] Failed reading existing business_rules.json: {ex}; initializing a new list.")
            business_rules = []

    # Duplicate detection (exclude volatile keys)
    new_norm = _normalize_rule_for_compare(new_rule)
    exists = any(_normalize_rule_for_compare(r) == new_norm for r in business_rules if isinstance(r, dict))
    if exists:
        eprint("[INFO] Top-level rule already present in business_rules.json (by content). Skipping add.")
    else:
        _safe_backup_json(br_path)
        business_rules.append(new_rule)
        write_json(br_path, business_rules)
        print(f"Added new rule to business_rules.json with id {new_rule_id}")

    # --- models.json merge ---
    models_path = (model_home / "models.json").resolve()
    models = []
    if models_path.exists():
        try:
            models = load_json(models_path)
            if not isinstance(models, list):
                eprint("[WARN] models.json is not a list; initializing a new list.")
                models = []
        except Exception as ex:
            eprint(f"[WARN] Failed reading models.json: {ex}; initializing a new list.")
            models = []

    sel_idx = None
    for idx, m in enumerate(models):
        if isinstance(m, dict) and m.get("id") == selected_model_id:
            sel_idx = idx
            break

    if sel_idx is None:
        eprint(f"[WARN] Selected model id {selected_model_id} not found in models.json; cannot append businessLogicIds.")
        return

    model_obj = models[sel_idx]
    ids = model_obj.get("businessLogicIds")
    if not isinstance(ids, list):
        ids = []
    if new_rule_id not in ids:
        _safe_backup_json(models_path)
        ids.append(new_rule_id)
        model_obj["businessLogicIds"] = ids
        models[sel_idx] = model_obj
        write_json(models_path, models)
        print(f"Appended new rule id to model {selected_model_id} in models.json")
    else:
        print("Rule id already present in selected model; no change to models.json")


# ---------------------------
# Paths and resolution
# ---------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # project root
TMP_DIR = REPO_ROOT / ".tmp" / "generate_composite_decision"
SPEC_DIR = REPO_ROOT / "tools" / "spec"
TEMPLATES_DIR = REPO_ROOT / "prompts"
COMPOSED_TEMPLATE = TEMPLATES_DIR / "composed-decision-report.templ"


def resolve_optional_path(candidate: Optional[str], base_candidates: List[Path]) -> Optional[str]:
    """
    Try to resolve a possibly relative path string against a list of base directories.
    Returns a string path (absolute) if found, else returns the original candidate.
    If candidate is None/empty -> returns None.
    """
    if not candidate:
        return None
    cand_path = Path(candidate).expanduser()
    if cand_path.is_absolute() and cand_path.exists():
        return str(cand_path)
    # Try under provided bases
    for base in base_candidates:
        p = (base / candidate).expanduser()
        if p.exists():
            return str(p)
    # Fallback to original string
    return candidate


# ---------------------------
# Core steps
# ---------------------------

def step_select_model(model_home: Path) -> Dict[str, Any]:
    print("\nSTEP 1: Load models and select one")
    models_path = model_home / "models.json"
    rules_path = model_home / "business_rules.json"

    if not models_path.exists():
        eprint(f"ERROR: models.json not found at {models_path}")
        sys.exit(1)
    if not rules_path.exists():
        eprint(f"ERROR: business_rules.json not found at {rules_path}")
        sys.exit(1)

    models = load_json(models_path)
    if not isinstance(models, list) or not models:
        eprint("ERROR: models.json should be a non-empty list.")
        sys.exit(1)

    menu = [f"{m.get('name','(unnamed)')}  –  {m.get('id','')}" for m in models]
    sel_idx = choose_from_list("Select a model:", menu, default_index=1)
    selected_model = models[sel_idx - 1]

    # Temp write selected model
    ensure_dir(TMP_DIR)
    selected_model_path = TMP_DIR / "selected_model.json"
    # Remove businessLogicIds from temp model file (not needed downstream)
    model_copy = dict(selected_model)
    model_copy.pop("businessLogicIds", None)
    write_json(selected_model_path, model_copy)
    print(f"→ Wrote selected model to {selected_model_path}")

    # Cross-reference rules
    print("STEP 2: Export rules belonging to the selected model")
    rules_all = load_json(rules_path)
    if not isinstance(rules_all, list):
        eprint("ERROR: business_rules.json must be a list.")
        sys.exit(1)

    wanted_ids = selected_model.get("businessLogicIds") or []
    wanted_set = set(wanted_ids)
    # Keep order according to model's list
    id_to_rule = {r.get("id"): r for r in rules_all if isinstance(r, dict) and r.get("id")}
    filtered_rules: List[Dict[str, Any]] = [id_to_rule[rid] for rid in wanted_ids if rid in id_to_rule]

    missing = [rid for rid in wanted_ids if rid not in id_to_rule]
    if missing:
        eprint(f"WARNING: {len(missing)} rule ids listed in the model were not found in business_rules.json")

    # Strip fields not needed in composed decision temp output
    cleaned_rules = []
    for r in filtered_rules:
        rc = dict(r)
        for k in ["timestamp", "doc_rule_id", "business_area", "doc_match_score", "id", "archived"]:
            rc.pop(k, None)
        cleaned_rules.append(rc)

    rules_out_path = TMP_DIR / "rules_for_model.json"
    write_json(rules_out_path, cleaned_rules)
    print(f"→ Wrote {len(cleaned_rules)} rules to {rules_out_path}")

    return {
        "selected_model": selected_model,
        "selected_model_path": str(selected_model_path),
        "rules_out_path": str(rules_out_path),
    }


def step_select_sources_spec() -> Dict[str, Any]:
    print("\nSTEP 3: Select a sources spec file (tools/spec/sources_*.json)")
    specs = sorted(SPEC_DIR.glob("sources_*.json"))
    if not specs:
        eprint(f"ERROR: No sources_*.json files found under {SPEC_DIR}")
        sys.exit(1)

    menu = [f"{p.name}" for p in specs]
    sel_idx = choose_from_list("Select a sources file:", menu, default_index=1)
    chosen = specs[sel_idx - 1]
    data = load_json(chosen)

    # Normalize keys and validate
    if not isinstance(data, dict) or "path-pairs" not in data:
        eprint("ERROR: Spec file should be an object containing 'path-pairs'.")
        sys.exit(1)

    root_dir = data.get("root-directory") or ""
    model_home_from_spec = data.get("model-home") or ""
    path_pairs = data.get("path-pairs") or []
    if not isinstance(path_pairs, list) or not path_pairs:
        eprint("ERROR: No 'path-pairs' defined in the selected spec.")
        sys.exit(1)

    print("\nSTEP 4: Select a path-pair from the spec")
    pair_menu = []
    for pair in path_pairs:
        src = pair.get("source-path", "")
        outp = pair.get("output-path", "")
        team = pair.get("team", "")
        comp = pair.get("component", "")
        pair_menu.append(f"{src} → {outp}  [{team} / {comp}]")
    pair_idx = choose_from_list("Select a pair:", pair_menu, default_index=1)
    chosen_pair = path_pairs[pair_idx - 1]

    return {
        "spec_path": str(chosen),
        "spec_dir": str(chosen.parent),
        "root_directory": root_dir,
        "model_home": model_home_from_spec,
        "pair": chosen_pair,
    }


def step_run_pcpt(
    model_info: Dict[str, Any],
    spec_info: Dict[str, Any],
    model_home_prompted: Path,
) -> None:
    print("\nSTEP 5: Run PCPT (run-custom-prompt)")

    # Resolve root-directory from spec; if relative, interpret relative to spec file directory
    spec_dir = Path(spec_info["spec_dir"])
    root_dir_str = spec_info.get("root_directory") or ""
    root_dir = Path(root_dir_str).expanduser()
    if not root_dir.is_absolute():
        root_dir = (spec_dir / root_dir).resolve()

    pair = spec_info["pair"]
    src_rel = pair.get("source-path", "")
    out_rel = pair.get("output-path", "")
    filt_rel = "" #pair.get("filter")
    mode = pair.get("mode")

    source_path = (root_dir / src_rel).resolve()
    output_path = (root_dir / out_rel).resolve()

    # Try to resolve filter in likely places: alongside spec file, under repo root, then under root-dir
    filter_path = resolve_optional_path(
        filt_rel,
        base_candidates=[spec_dir, REPO_ROOT, root_dir],
    )

    # Inputs for the prompt (temp files created earlier)
    rules_file = model_info["rules_out_path"]
    model_file = model_info["selected_model_path"]

    # Ensure template exists
    if not COMPOSED_TEMPLATE.exists():
        eprint(f"ERROR: Template not found: {COMPOSED_TEMPLATE}")
        sys.exit(1)

    # Make sure output directory exists
    ensure_dir(output_path)

    # Streaming: pcpt_run_custom_prompt uses subprocess.run with check=True (streams to console).
    print(f"→ Source: {src_rel}")
    print(f"→ Output: {out_rel}")
    if filt_rel:
        print(f"→ Filter: {filt_rel}")
    if mode:
        print(f"→ Mode:   {mode}")
    print(f"→ Input 1 (rules): {rules_file}")
    print(f"→ Input 2 (model): {model_file}")
    print(f"→ Template: {COMPOSED_TEMPLATE.name}")

    pcpt_run_custom_prompt(
        source_path=str(source_path),
        custom_prompt_template=COMPOSED_TEMPLATE.name,
        input_file=str(rules_file),
        input_file2=str(model_file),
        output_dir_arg=str(output_path),
        filter_path=filter_path,
        mode=mode,
    )

    # STEP 6: Merge the generated top-level rule back into business_rules.json and models.json
    try:
        selected_model = model_info.get("selected_model") or {}
        sel_model_id = selected_model.get("id")
        if not sel_model_id:
            # Fallback to temp file
            sel_model_path = Path(model_info.get("selected_model_path", ""))
            if sel_model_path and sel_model_path.exists():
                sel_model = load_json(sel_model_path)
                sel_model_id = sel_model.get("id") if isinstance(sel_model, dict) else None
        if not sel_model_id:
            eprint("[WARN] Could not determine selected model id for merge; skipping Step 6.")
        else:
            print("\nSTEP 6: Merge back to model home")
            print(f"→ Model home: {model_home_prompted}")
            print(f"→ Output path: {output_path}")
            merge_top_level_rule_into_model_home(Path(model_home_prompted), Path(output_path), sel_model_id)
    except Exception as ex:
        eprint(f"[WARN] Merge step failed: {ex}")

    print("\nSTEP 7: Done ✔")
    print("Composed decision report generated and merged.")


def main() -> None:
    print("=== Generate Composite Decision ===")

    # Step 0: Prompt for model home (default ~/.model)
    default_model_home = str(Path("~/.model").expanduser())
    model_home_str = prompt_with_default(
        "Enter model home (contains models.json & business_rules.json)",
        default_model_home,
    )
    model_home = Path(model_home_str).expanduser().resolve()

    # Create temp dir
    ensure_dir(TMP_DIR)

    # Steps 1-2: Select model and export its rules
    model_info = step_select_model(model_home)

    # Step 3-4: Select spec and pair
    spec_info = step_select_sources_spec()

    # Step 5-6: Run pcpt and finish
    step_run_pcpt(model_info, spec_info, model_home)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)