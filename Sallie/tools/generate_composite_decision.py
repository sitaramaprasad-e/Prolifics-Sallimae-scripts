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
import hashlib
import os
import sys
import subprocess
import uuid
import argparse
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

# --- DMN snapshot/compare helpers ---
def _dmn_snapshot(rule: dict) -> dict:
    def _norm_io(lst):
        out = []
        for x in (lst or []):
            if not isinstance(x, dict):
                continue
            out.append({
                "name": x.get("name", ""),
                "type": x.get("type", ""),
                "allowedValues": list(x.get("allowedValues", []) or []),
            })
        return out
    return {
        "hitPolicy": rule.get("dmn_hit_policy", ""),
        "inputs": _norm_io(rule.get("dmn_inputs") or []),
        "outputs": _norm_io(rule.get("dmn_outputs") or []),
        "table": (rule.get("dmn_table") or "").strip(),
    }

def _has_dmn_material_change(old_rule: dict, new_rule: dict) -> bool:
    """Return True if DMN-relevant fields differ (including allowedValues)."""
    return _dmn_snapshot(old_rule) != _dmn_snapshot(new_rule)


# --- Helper: content fingerprint for rules ---
def _content_fingerprint(rule: dict) -> str:
    try:
        norm = _normalize_rule_for_compare(rule)
        payload = json.dumps(norm, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]
    except Exception:
        return ""

def _load_rules_from_report(report_path: Path) -> List[dict]:
    """Load one or more rule JSON objects from a report file.
    
    The report may be:
    - Pure JSON: a dict (single rule), a list of dicts (multiple), or a dict with key "rules".
    - Markdown with an embedded JSON object/array.
    Returns a list of rule dicts (possibly length 1).
    """
    raw = report_path.read_text(encoding="utf-8").strip()

    def _as_rule_list(obj):
        # Normalize parsed JSON into a list of rule dicts
        if isinstance(obj, dict) and "rules" in obj and isinstance(obj["rules"], list):
            return [r for r in obj["rules"] if isinstance(r, dict)]
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        raise ValueError("Unexpected JSON structure in report.")

    # 1) Try full-document JSON first
    try:
        return _as_rule_list(json.loads(raw))
    except json.JSONDecodeError:
        pass

    # 2) Try fenced code blocks first (```json ... ``` or ``` ... ```)
    import re
    fenced_blocks = re.findall(r"```(?:json|JSON)?\s*([\s\S]*?)```", raw)
    for block in fenced_blocks:
        s = block.strip()
        if not s:
            continue
        try:
            return _as_rule_list(json.loads(s))
        except Exception:
            continue

    # 3) Heuristic: extract the first bracketed JSON array/object in the document
    start_brace = raw.find("{")
    start_bracket = raw.find("[")

    # Prefer whichever opens first
    candidates = []
    if start_bracket != -1:
        end_bracket = raw.rfind("]")
        if end_bracket > start_bracket:
            candidates.append(raw[start_bracket:end_bracket+1] if (start_brace != -1 and start_brace < start_bracket) else raw[start_bracket if start_brace != -1 else start_bracket : end_bracket+1])
            candidates.append(raw[start_bracket:end_bracket+1])
    if start_brace != -1:
        end_brace = raw.rfind("}")
        if end_brace > start_brace:
            candidates.append(raw[start_brace:end_brace+1])

    for snippet in candidates:
        try:
            return _as_rule_list(json.loads(snippet))
        except Exception:
            continue

    # 4) If nothing worked, raise the original error
    raise ValueError("Could not locate valid JSON rules in report.")

def merge_generated_rules_into_model_home(model_home: Path, output_path: Path, selected_model_id: str, template_base: Optional[str] = None) -> None:
    """Merge one or more generated rules back into business_rules.json and models.json safely.

    Steps:
    1) Locate the composed-decision report under output_path (supports single and multi templates).
    2) Parse JSON for one or more rules.
    3) For each rule, generate a new UUID; add timestamp and archived fields if missing.
    4) Backup and merge into business_rules.json (skip if equivalent by content).
    5) Backup and update models.json to include new rule ids in the selected model's businessLogicIds.
    """
    # Candidate report locations (dynamic from template name; support md/json and nested folder)
    bases: List[str] = []
    if template_base:
        bases.append(template_base)
    # Also try legacy defaults as fallbacks
    for legacy in ("composed-decision-report", "multi-composed-decision-report"):
        if legacy not in bases:
            bases.append(legacy)
    candidates: List[Path] = []
    for b in bases:
        candidates.extend([
            output_path / b / f"{b}.md",
            output_path / f"{b}.md",
            output_path / b / f"{b}.json",
            output_path / f"{b}.json",
        ])
    report_file = next((p for p in candidates if p.exists()), None)
    if report_file is None:
        # Fallback: scan output_path and its immediate subdirs for the most recent plausible report (*.md/*.json)
        def _iter_candidates(root: Path) -> List[Path]:
            found: List[Path] = []
            try:
                for p in root.iterdir():
                    if p.is_file() and p.suffix.lower() in {".md", ".json"}:
                        found.append(p)
                    elif p.is_dir():
                        # one level deep
                        for q in p.iterdir():
                            if q.is_file() and q.suffix.lower() in {".md", ".json"}:
                                found.append(q)
            except Exception:
                pass
            return found
        pool = _iter_candidates(output_path)
        if not pool:
            eprint(f"[WARN] No report files (*.md/*.json) found under: {output_path}")
            return
        # Sort most recent first
        pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        # Try each until one parses as rules
        for cand in pool:
            try:
                _ = _load_rules_from_report(cand)
                report_file = cand
                break
            except Exception:
                continue
        if report_file is None:
            eprint(f"[WARN] Could not locate a parseable composed decision report under: {output_path}")
            return

    try:
        new_rules = _load_rules_from_report(report_file)
    except Exception as ex:
        eprint(f"[WARN] Could not parse rule JSON from report {report_file.name}: {ex}")
        return

    # Enrich and prepare merge
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in new_rules:
        if "id" in r:
            # Ensure we don't carry over prior ids
            r.pop("id", None)
        r["id"] = str(uuid.uuid4())
        r.setdefault("timestamp", now_ts)
        r.setdefault("archived", False)

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

    # Lookup by name for update-in-place behavior
    existing_by_name = {}
    for idx, r in enumerate(business_rules):
        if isinstance(r, dict):
            rn = (r.get("rule_name") or r.get("name") or "").strip()
            if rn:
                existing_by_name[rn] = {"idx": idx, "rule": r}

    # Build lookup of existing rules by fingerprint
    existing_by_fp = {}
    for idx, r in enumerate(business_rules):
        if not isinstance(r, dict):
            continue
        fp = _content_fingerprint(r)
        if fp:
            existing_by_fp.setdefault(fp, []).append({
                "idx": idx,
                "id": r.get("id"),
                "name": r.get("rule_name") or r.get("name") or "(unnamed)",
            })
    skipped_details: List[Dict[str, Any]] = []
    added_ids: List[str] = []
    for new_rule in new_rules:
        rn = (new_rule.get("rule_name") or new_rule.get("name") or "").strip()
        # If an existing rule with same name exists, prefer update-in-place when DMN changed (incl. allowedValues)
        ex = existing_by_name.get(rn)
        if ex:
            old = ex["rule"]
            if _has_dmn_material_change(old, new_rule):
                # Preserve stable fields from existing rule, update DMN fields from new_rule
                preserved_id = old.get("id")
                preserved_archived = old.get("archived", False)
                # Keep new timestamp already set on new_rule
                new_rule["id"] = preserved_id or new_rule.get("id") or str(uuid.uuid4())
                new_rule["archived"] = preserved_archived
                business_rules[ex["idx"]] = new_rule
                # Update lookups
                existing_by_name[rn] = {"idx": ex["idx"], "rule": new_rule}
                print(f"[INFO] Updated rule by name with DMN changes (incl. allowedValues): '{rn}'")
                continue
            # If no material DMN change, fall back to duplicate-by-content handling below

        new_norm = _normalize_rule_for_compare(new_rule)
        fp_new = _content_fingerprint(new_rule)
        exists = any(_normalize_rule_for_compare(r) == new_norm for r in business_rules if isinstance(r, dict))
        if exists:
            # Try to resolve which existing entry matched, for better diagnostics
            matches = existing_by_fp.get(fp_new) or []
            # Fall back to a lightweight search by name if fingerprint wasn’t available
            if not matches:
                rn2 = (new_rule.get("rule_name") or new_rule.get("name") or "").strip()
                if rn2:
                    for idx2, r in enumerate(business_rules):
                        if isinstance(r, dict) and (r.get("rule_name") == rn2 or r.get("name") == rn2):
                            matches.append({
                                "idx": idx2,
                                "id": r.get("id"),
                                "name": r.get("rule_name") or r.get("name") or "(unnamed)",
                            })
                            break
            detail = {
                "new_name": new_rule.get("rule_name") or new_rule.get("name") or "(unnamed)",
                "fingerprint": fp_new,
                "matches": matches,
            }
            skipped_details.append(detail)
            # Print a concise inline message for immediate visibility
            if matches:
                first = matches[0]
                eprint(f"[INFO] Skipping duplicate by content: '{detail['new_name']}' → existing id={first.get('id')} (fp={fp_new})")
            else:
                eprint(f"[INFO] Skipping duplicate by content: '{detail['new_name']}' (fp={fp_new})")
            continue
        business_rules.append(new_rule)
        added_ids.append(new_rule["id"])

    if skipped_details:
        print(f"Skipped {len(skipped_details)} duplicate rule(s) by content:")
        for d in skipped_details:
            name = d.get("new_name")
            fp = d.get("fingerprint") or ""
            matches = d.get("matches") or []
            if matches:
                tgt = matches[0]
                print(f"  • {name}  (fp={fp})  → existing id={tgt.get('id')}")
            else:
                print(f"  • {name}  (fp={fp})")

    if added_ids:
        _safe_backup_json(br_path)
        write_json(br_path, business_rules)
        print(f"Added {len(added_ids)} rule(s) to business_rules.json")
    else:
        print("No new rules added to business_rules.json (all duplicates).")

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

    if added_ids:
        model_obj = models[sel_idx]
        ids = model_obj.get("businessLogicIds")
        if not isinstance(ids, list):
            ids = []
        # Append unique ids
        for rid in added_ids:
            if rid not in ids:
                ids.append(rid)
        _safe_backup_json(models_path)
        model_obj["businessLogicIds"] = ids
        models[sel_idx] = model_obj
        write_json(models_path, models)
        print(f"Appended {len(added_ids)} new rule id(s) to model {selected_model_id} in models.json")
    else:
        print("No changes to models.json (no new rule ids).")


# ---------------------------
# Paths and resolution
# ---------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # project root
TMP_DIR = REPO_ROOT / ".tmp" / "generate_composite_decision"
SPEC_DIR = REPO_ROOT / "tools" / "spec"
TEMPLATES_DIR = REPO_ROOT / "prompts"
COMPOSED_TEMPLATE_ONE = TEMPLATES_DIR / "composed-decision-report.templ"
COMPOSED_TEMPLATE_NEXT = TEMPLATES_DIR / "multi-composed-decision-report.templ"

def _resolve_template_path(mode: str) -> Path:
    """
    mode: 'top'  -> single composed decision template
          'next' -> multi composed decision template (can emit multiple rules)
    """
    m = (mode or "top").strip().lower()
    if m == "next":
        return COMPOSED_TEMPLATE_NEXT
    return COMPOSED_TEMPLATE_ONE


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
    compose_mode: str,
    skip_generate: bool,
) -> None:
    print("\nSTEP 5: Run PCPT (run-custom-prompt)" if not skip_generate else "\nSTEP 5: Skip generate – using existing report for ingest")

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
    pcpt_mode = "multi"  # force multi mode; ignore spec-provided mode

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

    # Select template
    template_path = _resolve_template_path(compose_mode)
    if not template_path.exists():
        eprint(f"ERROR: Template not found: {template_path}")
        sys.exit(1)
    template_base = template_path.stem

    # Make sure output directory exists
    ensure_dir(output_path)

    # Streaming: pcpt_run_custom_prompt uses subprocess.run with check=True (streams to console).
    print(f"→ Source: {src_rel}")
    print(f"→ Output: {out_rel}")
    if filt_rel:
        print(f"→ Filter: {filt_rel}")
    if pcpt_mode:
        print(f"→ Mode:   {pcpt_mode}")
    print(f"→ Input 1 (rules): {rules_file}")
    print(f"→ Input 2 (model): {model_file}")
    print(f"→ Template: {template_path.name}")
    print(f"→ Compose Mode: {compose_mode}")

    if not skip_generate:
        pcpt_run_custom_prompt(
            source_path=str(source_path),
            custom_prompt_template=template_path.name,
            input_file=str(rules_file),
            input_file2=str(model_file),
            output_dir_arg=str(output_path),
            filter_path=filter_path,
            mode=pcpt_mode,
        )
    else:
        print("[INFO] --skip-generate supplied: not calling pcpt; proceeding to merge from existing report.")

    # STEP 6: Merge the generated rule(s) back into business_rules.json and models.json
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
            merge_generated_rules_into_model_home(Path(model_home_prompted), Path(output_path), sel_model_id, template_base)
    except Exception as ex:
        eprint(f"[WARN] Merge step failed: {ex}")

    print("\nSTEP 7: Done ✔")
    print("Composed decision report generated and merged.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a composed decision report (single or multi).")
    parser.add_argument(
        "--mode",
        choices=["top", "next"],
        help=("Select template behavior: 'top' uses composed-decision-report.templ, "
              "'next' uses multi-composed-decision-report.templ (may emit multiple rules). If omitted, you'll be prompted.")
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help=(
            "Skip running PCPT and go straight to merge (ingest) using an existing composed decision report "
            "under the selected output path."
        ),
    )
    args = parser.parse_args()
    compose_mode = args.mode
    skip_generate = args.skip_generate
    if not compose_mode:
        choice = choose_from_list(
            "Select compose mode:",
            [
                "top  – single composed decision template",
                "next – multi composed decision template (can emit multiple rules)",
            ],
            default_index=1,
        )
        compose_mode = "top" if choice == 1 else "next"

    print("=== Generate Composite Decision ===")

    # Create temp dir
    ensure_dir(TMP_DIR)

    # Step 3-4: Select spec and pair (first, so we can check for model-home in spec)
    spec_info = step_select_sources_spec()

    # Determine model_home_str from spec, or prompt as fallback
    model_home_from_spec = (spec_info.get("model_home") or "").strip()
    if model_home_from_spec:
        # Ensure the spec's model-home points to the actual model directory (~/.model by default)
        expanded = Path(model_home_from_spec).expanduser()
        if expanded.name != ".model":
            expanded = expanded / ".model"
        model_home_str = str(expanded)
        print("STEP 0: Using model home from spec")
        print(f"→ Model home (spec): {model_home_from_spec}")
        print(f"→ Resolved model home: {model_home_str}")
    else:
        default_model_home = str(Path("~/.model").expanduser())
        model_home_str = prompt_with_default(
            "Enter model home (contains models.json & business_rules.json)",
            default_model_home,
        )

    model_home = Path(model_home_str).expanduser().resolve()

    # Steps 1-2: Select model and export its rules
    model_info = step_select_model(model_home)

    # Step 5-6: Run pcpt and finish
    step_run_pcpt(model_info, spec_info, model_home, compose_mode, skip_generate)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)