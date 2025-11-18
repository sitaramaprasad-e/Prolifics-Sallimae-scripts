# --- import path bootstrap: allow `from tools...` when run as tools/pcpt_pipeline.py ---
from __future__ import annotations
import os as _os, sys as _sys
_THIS_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
# --- end import path bootstrap ---

import warnings

# Suppress the LibreSSL/OpenSSL compatibility warning from urllib3 v2
warnings.filterwarnings(
    "ignore",
    message="urllib3 v2 only supports OpenSSL 1.1.1+",
    module="urllib3",
)

import json
import sys
import uuid
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from helpers.call_pcpt import pcpt_run_custom_prompt

# Simple ANSI styling (disabled if not a TTY)
ANSI_BOLD = "\033[1m" if sys.stdout.isatty() else ""
ANSI_DIM = "\033[2m" if sys.stdout.isatty() else ""
ANSI_BLUE = "\033[34m" if sys.stdout.isatty() else ""
ANSI_CYAN = "\033[36m" if sys.stdout.isatty() else ""
ANSI_MAGENTA = "\033[35m" if sys.stdout.isatty() else ""
ANSI_YELLOW = "\033[33m" if sys.stdout.isatty() else ""
ANSI_RESET = "\033[0m" if sys.stdout.isatty() else ""

# ---------------------------
# Paths and resolution
# ---------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # project root
TMP_DIR = REPO_ROOT / ".tmp" / "generate_hierarchy"
TMP_HIER_DIR = REPO_ROOT / ".tmp" / "hierarchy"
TO_PCPT_DIR = TMP_DIR
SPEC_DIR = REPO_ROOT / "spec"
TEMPLATES_DIR = REPO_ROOT / ".." / "prompts"

HIERARCHY_TEMPLATE_TOP = "suggest-decisions-top"
HIERARCHY_TEMPLATE_SELECTED_TOP = "suggest-decisions-selected-top"
HIERARCHY_TEMPLATE_COMP = "suggest-decisions-comp"
HIERARCHY_TEMPLATE_MIM = "suggest-decisions-mim"

HIERARCHY_TEMPLATE_TOP_DIR = TEMPLATES_DIR / (HIERARCHY_TEMPLATE_TOP + ".templ")
HIERARCHY_TEMPLATE_SELECTED_TOP_DIR = TEMPLATES_DIR / (HIERARCHY_TEMPLATE_SELECTED_TOP + ".templ")
HIERARCHY_TEMPLATE_COMP_DIR = TEMPLATES_DIR / (HIERARCHY_TEMPLATE_COMP + ".templ")
HIERARCHY_TEMPLATE_MIM_DIR = TEMPLATES_DIR / (HIERARCHY_TEMPLATE_MIM + ".templ")
SUGGEST_HIERARCHY_TEMPLATE_DIR = TEMPLATES_DIR / "suggest-hierarchy.templ"
SUGGEST_HIERARCHY_MINIMAL_TEMPLATE_DIR = TEMPLATES_DIR / "suggest-hierarchy-minimal.templ"

# --- Helper: content fingerprint for rules ---
def _content_fingerprint(rule: dict) -> str:
    try:
        norm = _normalize_rule_for_compare(rule)
        payload = json.dumps(norm, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]
    except Exception:
        return ""
        
# ---------------------------
# Helper: Prepare rules file for PCPT (convert from_step_id UUIDs to from_step names)
# ---------------------------
def prepare_rules_file_for_pcpt(rules_path: Path) -> Path:
    """
    Create a PCPT-specific copy of the rules JSON where any link
    with from_step_id set to a rule UUID is converted to from_step
    with the rule name. The copy is written under TO_PCPT_DIR,
    leaving the original file untouched.
    """
    ensure_dir(TO_PCPT_DIR)
    dest = TO_PCPT_DIR / f"to_pcpt_{rules_path.name}"

    # Best-effort: if anything goes wrong, just copy unchanged.
    try:
        data = load_json(rules_path)
    except Exception as ex:
        eprint(f"[WARN] prepare_rules_file_for_pcpt: failed to load {rules_path}: {ex}; copying unchanged.")
        try:
            shutil.copy2(rules_path, dest)
        except Exception as ex2:
            eprint(f"[WARN] prepare_rules_file_for_pcpt: failed to copy {rules_path} → {dest}: {ex2}")
        return dest

    try:
        # If it's our typical list-of-rules structure, transform links.
        if isinstance(data, list):
            # Map rule id → rule name
            id_to_name: Dict[str, str] = {}
            for r in data:
                if not isinstance(r, dict):
                    continue
                rid = (r.get("id") or "").strip()
                rn = (r.get("rule_name") or r.get("name") or "").strip()
                if rid and rn:
                    id_to_name[rid] = rn

            # Walk links and swap from_step_id → from_step when it matches a rule id
            for r in data:
                if not isinstance(r, dict):
                    continue
                links = r.get("links") or []
                if not isinstance(links, list):
                    continue
                for l in links:
                    if not isinstance(l, dict):
                        continue
                    fsid = (l.get("from_step_id") or "").strip()
                    if fsid and fsid in id_to_name:
                        l["from_step"] = id_to_name[fsid]
                        l.pop("from_step_id", None)

            write_json(dest, data)
        else:
            # Not the list-of-rules shape; just copy through.
            shutil.copy2(rules_path, dest)
    except Exception as ex:
        eprint(f"[WARN] prepare_rules_file_for_pcpt: error transforming {rules_path}: {ex}; copying unchanged.")
        try:
            shutil.copy2(rules_path, dest)
        except Exception as ex2:
            eprint(f"[WARN] prepare_rules_file_for_pcpt: failed to copy {rules_path} → {dest}: {ex2}")

    return dest

def merge_generated_rules_into_model_home(
    model_home: Path,
    output_path: Path,
    selected_model_id: str,
    template_base: Optional[str] = None,
    restrict_ids: Optional[Set[str]] = None,
    restrict_names: Optional[Set[str]] = None,
    hierarchy_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Merge one or more generated rules back into business_rules.json and models.json safely.

    Steps:
    1) Locate the composed-decision report under output_path (supports single and multi templates).
    2) Parse JSON for one or more rules.
    3) For each rule, generate a new UUID; add timestamp and archived fields if missing.
    4) Backup and merge into business_rules.json (skip if equivalent by content).
    5) Backup and update models.json to include new rule ids in the selected model's businessLogicIds.

    When running in MIM mode, you can pass `hierarchy_meta` to attach `hierarchy_name` and `hierarchy_description`
    onto the Top‑Level decision. These attributes are optional and only set if present.
    """
    # Optional hierarchy metadata (used in MIM mode): maps top decision id/name -> {hierarchy_name, hierarchy_description}
    meta_by_id: Dict[str, Dict[str, str]] = {}
    meta_by_name_cf: Dict[str, Dict[str, str]] = {}
    if hierarchy_meta and isinstance(hierarchy_meta, dict):
        by_id = hierarchy_meta.get("by_id") or {}
        by_name = hierarchy_meta.get("by_name") or {}
        if isinstance(by_id, dict):
            meta_by_id = {str(k).strip(): v for k, v in by_id.items() if str(k).strip()}
        if isinstance(by_name, dict):
            meta_by_name_cf = {str(k).casefold().strip(): v for k, v in by_name.items() if str(k).strip()}

    def _apply_hierarchy_meta(rule_obj: dict, incoming_name: str | None = None):
        """If hierarchy metadata matches this rule by id or name, attach optional fields.
        Does not error if metadata missing. Overwrites existing values only if provided.
        """
        try:
            rid_local = (rule_obj.get("id") or "").strip()
            rn_cf = (incoming_name or rule_obj.get("rule_name") or rule_obj.get("name") or "").casefold()
            meta = None
            if rid_local and rid_local in meta_by_id:
                meta = meta_by_id[rid_local]
            elif rn_cf and rn_cf in meta_by_name_cf:
                meta = meta_by_name_cf[rn_cf]
            if meta and isinstance(meta, dict):
                hn = (meta.get("hierarchy_name") or "").strip()
                hd = (meta.get("hierarchy_description") or "").strip()
                if hn:
                    rule_obj["hierarchy_name"] = hn
                if hd:
                    rule_obj["hierarchy_description"] = hd
        except Exception:
            pass
    # Candidate report locations (dynamic from template name; support md/json and nested folder)
    bases: List[str] = []
    if template_base:
        bases.append(template_base)

    # Also try legacy defaults as fallbacks
    for legacy in (HIERARCHY_TEMPLATE_TOP, HIERARCHY_TEMPLATE_COMP):
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
    # Also support numbered outputs when PCPT is called with total/index, e.g. meet-in-the-middle-decision-report-1of5-.md
    numbered_matches: List[Path] = []
    for b in bases:
        # Search both directly under output_path and under a subfolder named after the base
        for pat in [
            output_path / b / f"{b}-*of*-*.md",
            output_path / f"{b}-*of*-*.md",
            output_path / b / f"{b}-*of*-*.json",
            output_path / f"{b}-*of*-*.json",
        ]:
            # Path.glob only supports patterns on the last component; use glob on the parent
            parent = pat.parent
            pattern = pat.name
            try:
                for match in parent.glob(pattern):
                    if match.is_file():
                        numbered_matches.append(match)
            except Exception:
                continue
    # If we found any numbered matches, prefer the most recent one
    if numbered_matches:
        try:
            numbered_matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception:
            pass
        report_file = numbered_matches[0]
    else:
        report_file = None
    report_file = report_file or next((p for p in candidates if p.exists()), None)
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

    # Optional per-hierarchy restriction: only process rules that match the provided ids/names,
    # Always include new rules (rule_id == "" or __source_id_blank True), restrict existing by id/name.
    if restrict_ids or restrict_names:
        # Normalize provided hierarchy scope
        _ids = {i.strip() for i in (restrict_ids or set()) if isinstance(i, str) and i.strip()}
        _names_cf = {n.strip().casefold() for n in (restrict_names or set()) if isinstance(n, str) and n.strip()}
        before_len = len(new_rules)

        def _matches(rule: dict) -> bool:
            """Hierarchy filter:
            - NEW rules (no id in the PCPT output) are ALWAYS included.
            - EXISTING rules are included only if their id or name is within the current hierarchy scope.
            
            NOTE: This check runs BEFORE enrichment, so we must infer newness from raw fields.
            """
            rid_raw = str(rule.get("id", "")).strip()
            # A rule is considered NEW pre-enrichment if it has no id at all.
            is_new_pre_enrich = (rid_raw == "")
            if is_new_pre_enrich:
                return True

            # Existing: match by id or name against the hierarchy's ids/names
            rid = (rule.get("id") or "").strip()
            rn  = (rule.get("rule_name") or rule.get("name") or "").strip()
            rn_cf = rn.casefold()
            if rid and rid in _ids:
                return True
            if rn and rn_cf in _names_cf:
                return True
            return False

        filtered = []
        dropped = []
        for r in new_rules:
            if isinstance(r, dict) and _matches(r):
                filtered.append(r)
            else:
                dropped.append(r)
        new_rules = filtered
        print(f"[TRACE] Restricting merge to hierarchy scope: {len(new_rules)}/{before_len} rule(s)."
              f"  (ids={len(_ids)}, names={len(_names_cf)})")
        if dropped:
            dropped_names = [ (d.get('rule_name') or d.get('name') or '(unnamed)') for d in dropped if isinstance(d, dict) ]
            print(f"[TRACE] Excluded outside-scope rule(s): {', '.join(dropped_names[:5])}{' …' if len(dropped_names)>5 else ''}")
        if not new_rules:
            eprint("[WARN] No rules matched the hierarchy filter; skipping merge for this hierarchy.")
            return

    # Detect MIM mode by template name
    is_mim_mode = (template_base or "").strip().lower() == HIERARCHY_TEMPLATE_MIM

    # Enrich and prepare merge
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in new_rules:
        # Treat 'id' from PCPT as the authoritative original identifier.
        orig_id = (r.get("id") or "").strip()
        if orig_id:
            # Existing rule: normalize into 'id' so downstream lookups work, and flag as not-new
            r["id"] = orig_id
            r["__source_orig_id"] = orig_id
            r["__source_id_blank"] = False
        else:
            # Truly new rule requested by PCPT (no id present) → create a fresh UUID
            r["id"] = str(uuid.uuid4())
            r["__source_orig_id"] = ""
            r["__source_id_blank"] = True
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

    # --- Limit scope: allowed rule IDs in the selected model ---
    models_path = (model_home / "models.json").resolve()
    models = []
    if models_path.exists():
        try:
            models = load_json(models_path)
        except Exception:
            models = []
    allowed_ids_in_model = set()
    model_obj = None
    for m in (models if isinstance(models, list) else []):
        if isinstance(m, dict) and m.get("id") == selected_model_id:
            model_obj = m
            break
    if model_obj:
        ids = model_obj.get("businessLogicIds")
        if isinstance(ids, list):
            allowed_ids_in_model = set(str(i) for i in ids if isinstance(i, str))
        else:
            allowed_ids_in_model = set()
    else:
        allowed_ids_in_model = set()
    if not allowed_ids_in_model:
        print("[TRACE] Name→ID resolution limited to model scope: no ids found for selected model.")

    # --- Post-process incoming links: map from_step (name or uuid) → from_step_id using existing+incoming name→id ---
    def _pp_looks_like_uuid(s: str) -> bool:
        s = (s or "").strip()
        return len(s) == 36 and s.count("-") == 4

    # Build name→id map from existing business_rules, but only for rules in allowed_ids_in_model
    _pp_name_to_id = {}
    try:
        for _r in (business_rules or []):
            if isinstance(_r, dict):
                _rid = (_r.get("id") or "").strip()
                _rn = (_r.get("rule_name") or _r.get("name") or "").strip()
                if _rid and _rn and _rid in allowed_ids_in_model:
                    _pp_name_to_id[_rn.casefold()] = _rid
    except Exception:
        pass
    # Also include names from the incoming rules themselves so that links can resolve
    # to newly-created decisions in this same merge batch (before they're in models.json).
    try:
        for _r in (new_rules or []):
            if not isinstance(_r, dict):
                continue
            _rid = (_r.get("id") or "").strip()
            _rn = (_r.get("rule_name") or _r.get("name") or "").strip()
            if _rid and _rn:
                # Do not overwrite an explicit existing mapping; only fill gaps
                _pp_name_to_id.setdefault(_rn.casefold(), _rid)
    except Exception:
        pass

    # Normalize links in all incoming rules: set from_step_id and drop from_step
    try:
        for _r in (new_rules or []):
            if not isinstance(_r, dict):
                continue
            _links = _r.get("links") or []
            if not isinstance(_links, list):
                continue
            # NOTE: Upstream PCPT may return links where `from_step_id` contains a NAME instead of a UUID.
            # We defensively correct this by resolving names to ids using the aggregated name→id map.
            for _l in _links:
                if not isinstance(_l, dict):
                    continue
                _fs = (_l.get("from_step") or "").strip()
                _fsid = (_l.get("from_step_id") or "").strip()

                # If PCPT emitted a name in from_step, map to id.
                if _fs and not _fsid:
                    if _pp_looks_like_uuid(_fs):
                        _l["from_step_id"] = _fs
                    else:
                        _mid = _pp_name_to_id.get(_fs.casefold(), "")
                        if _mid:
                            _l["from_step_id"] = _mid

                # NEW: Guard for bad inputs where from_step_id itself is actually a NAME.
                # Sometimes upstream tools write the decision NAME into from_step_id.
                # We correct that here by checking if the value matches any known rule name
                # and swapping it for the corresponding UUID.
                _fsid = (_l.get("from_step_id") or "").strip()
                if _fsid and not _pp_looks_like_uuid(_fsid):
                    _mid = _pp_name_to_id.get(_fsid.casefold(), "")
                    if _mid:
                        _l["from_step_id"] = _mid

                # Always drop from_step to enforce id‑only persistence
                if "from_step" in _l:
                    _l.pop("from_step", None)
    except Exception:
        pass

    # Helpers for MIM mode and kind classification
    def _norm_kind(val: Optional[str]) -> str:
        return (val or "").strip().casefold()

    # def _is_composite(rule: dict) -> bool:
    #     return _norm_kind(rule.get("Kind") or rule.get("kind")) == "decision (composite)".lower()

    # def _is_top_level(rule: dict) -> bool:
    #     return _norm_kind(rule.get("Kind") or rule.get("kind")) == "decision (top-level)".lower()


    def _merge_links_in_place_generic(existing_rule: dict, incoming_rule: dict, existing_by_id_map: dict) -> int:
        """
        Merge unique links from incoming_rule into existing_rule. Returns number of links added.

        This shared helper normalizes incoming links so that:
        - from_step or from_step_id that contains a NAME is mapped to the correct UUID using existing_by_id_map.
        - 'from_step' is dropped; only 'from_step_id' persists.
        - Duplicate links (by from_step_id, from_output, to_input, kind) are ignored.
        """
        existing_links = existing_rule.get("links") or []
        if not isinstance(existing_links, list):
            existing_links = []
        incoming_links = incoming_rule.get("links") or []
        if not isinstance(incoming_links, list):
            incoming_links = []

        def _looks_like_uuid(s: str) -> bool:
            s = (s or "").strip()
            return len(s) == 36 and s.count('-') == 4

        # Build a name->id map from existing rules (best‑effort)
        name_to_id = {}
        try:
            for _rid, _entry in (existing_by_id_map or {}).items():
                rn = (_entry.get("rule", {}).get("rule_name") or _entry.get("rule", {}).get("name") or "").strip()
                if _rid and rn:
                    name_to_id[rn.casefold()] = _rid
        except Exception:
            pass

        # Normalize incoming links in-place
        for _l in incoming_links:
            if not isinstance(_l, dict):
                continue
            fs = (_l.get("from_step") or "").strip()
            fsid = (_l.get("from_step_id") or "").strip()

            # Map name/id in from_step → from_step_id
            if not fsid and fs:
                if _looks_like_uuid(fs):
                    _l["from_step_id"] = fs
                else:
                    _id = name_to_id.get(fs.casefold(), "")
                    if _id:
                        _l["from_step_id"] = _id

            # Guard: from_step_id might actually be a NAME; resolve it
            fsid = (_l.get("from_step_id") or "").strip()
            if fsid and not _looks_like_uuid(fsid):
                _id = name_to_id.get(fsid.casefold(), "")
                if _id:
                    _l["from_step_id"] = _id

            # Enforce id‑only persistence
            if "from_step" in _l:
                _l.pop("from_step", None)

        # Filter out links whose from_step_id is not resolvable within the in-scope map
        filtered_incoming = []
        for _l in incoming_links:
            if not isinstance(_l, dict):
                continue
            _fsid = (_l.get("from_step_id") or "").strip()
            if not _fsid or _fsid not in (existing_by_id_map or {}):
                # Skip links that cannot be bound to an in-scope producer
                continue
            filtered_incoming.append(_l)
        incoming_links = filtered_incoming

        # Deduplicate and merge
        def _k(link: dict) -> tuple:
            return (
                (link.get("from_step_id") or "").strip(),
                (link.get("from_output") or "").strip(),
                (link.get("to_input") or "").strip(),
                (link.get("kind") or "").strip(),
            )

        seen = {_k(l) for l in existing_links if isinstance(l, dict)}
        added = 0
        for l in incoming_links:
            if not isinstance(l, dict):
                continue
            key = _k(l)
            if key in seen:
                continue
            existing_links.append(l)
            seen.add(key)
            added += 1

        existing_rule["links"] = existing_links
        return added

    skipped_details: List[Dict[str, Any]] = []
    added_ids: List[str] = []
    ensure_model_ids: set[str] = set()
    updated_top_links = 0
    # Track whether simple (non-MIM) mode performed any link-only updates
    simple_links_updated = False

    # Precompute lookups used by both paths
    existing_by_name = {}
    for idx, r in enumerate(business_rules):
        if isinstance(r, dict):
            rn0 = (r.get("rule_name") or r.get("name") or "").strip()
            kind0 = _norm_kind(r.get("Kind") or r.get("kind"))
            if rn0:
                key = f"{rn0}||{kind0}"
                existing_by_name[key] = {"idx": idx, "rule": r}
    # NEW: Precompute lookup by id
    existing_by_id = {}
    for idx, r in enumerate(business_rules):
        if isinstance(r, dict):
            rid0 = (r.get("id") or "").strip()
            if rid0:
                existing_by_id[rid0] = {"idx": idx, "rule": r}

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

    # def _process_as_add_update(rule: dict) -> None:
    #     nonlocal skipped_details, added_ids, business_rules, existing_by_name, ensure_model_ids, existing_by_id
    #     # Drop any internal flags before comparison/persistence
    #     rule.pop("__source_id_blank", None)
    #     rn = (rule.get("rule_name") or rule.get("name") or "").strip()
    #     rid = (rule.get("id") or "").strip()
    #     ex = None
    #     if rid:
    #         ex = existing_by_id.get(rid)
    #     if not ex and rn:           
    #         ex = existing_by_name.get(rn)

    #     # If existing rule found and there are no DMN material changes,
    #     # check for full content duplicate, else update in place with non-DMN changes
    #     if ex and not _has_dmn_material_change(ex["rule"], rule):
    #         new_norm = _normalize_rule_for_compare(rule)
    #         ex_norm = _normalize_rule_for_compare(ex["rule"])
    #         if new_norm == ex_norm:
    #             fp_new = _content_fingerprint(rule)
    #             matches = []
    #             matches.append({"idx": ex["idx"], "id": ex["rule"].get("id"), "name": rn})
    #             skipped_details.append({"new_name": rn or "(unnamed)", "fingerprint": fp_new, "matches": matches})
    #             if matches:
    #                 first = matches[0]
    #                 eprint(f"[INFO] Skipping duplicate by content: '{rn}' → existing id={first.get('id')} (fp={fp_new})")
    #             else:
    #                 eprint(f"[INFO] Skipping duplicate by content: '{rn}' (fp={fp_new})")
    #             return
    #         # Otherwise, update in place (preserving id and archived)
    #         preserved_id = ex["rule"].get("id")
    #         preserved_archived = ex["rule"].get("archived", False)
    #         rule["id"] = preserved_id or rid or str(uuid.uuid4())
    #         rule["archived"] = preserved_archived
    #         business_rules[ex["idx"]] = rule
    #         existing_by_name[rn] = {"idx": ex["idx"], "rule": rule}
    #         if rule.get("id"):
    #             ensure_model_ids.add(rule["id"])
    #             existing_by_id[rule["id"]] = {"idx": ex["idx"], "rule": rule}
    #         print(f"[INFO] Updated rule by {'id' if rid else 'name'} with non-DMN changes")
    #         return

    #     if ex and _has_dmn_material_change(ex["rule"], rule):
    #         preserved_id = ex["rule"].get("id")
    #         was_archived = ex["rule"].get("archived", False)
    #         rule["id"] = preserved_id or rid or str(uuid.uuid4())
    #         if was_archived:
    #             rule["archived"] = False
    #             print(f"[INFO] Unarchived rule due to update: '{rn}' (id={rule['id']})")
    #         else:
    #             rule["archived"] = ex["rule"].get("archived", False)
    #         business_rules[ex["idx"]] = rule
    #         existing_by_name[rn] = {"idx": ex["idx"], "rule": rule}
    #         if rule.get("id"):
    #             ensure_model_ids.add(rule["id"])
    #             existing_by_id[rule["id"]] = {"idx": ex["idx"], "rule": rule}
    #         print(f"[INFO] Updated rule by {'id' if rid else 'name'} with DMN changes (incl. allowedValues): '{rn}'")
    #         return
    #     new_norm = _normalize_rule_for_compare(rule)
    #     exists = any(_normalize_rule_for_compare(r) == new_norm for r in business_rules if isinstance(r, dict))
    #     if exists:
    #         fp_new = _content_fingerprint(rule)
    #         matches = []
    #         if ex:
    #             matches.append({"idx": ex["idx"], "id": ex["rule"].get("id"), "name": rn})
    #         skipped_details.append({"new_name": rn or "(unnamed)", "fingerprint": fp_new, "matches": matches})
    #         if matches:
    #             first = matches[0]
    #             eprint(f"[INFO] Skipping duplicate by content: '{rn}' → existing id={first.get('id')} (fp={fp_new})")
    #         else:
    #             eprint(f"[INFO] Skipping duplicate by content: '{rn}' (fp={fp_new})")
    #         return
    #     business_rules.append(rule)
    #     added_ids.append(rule["id"])
    #     if rule.get("id"):
    #         existing_by_id[rule["id"]] = {"idx": len(business_rules)-1, "rule": rule}

    if is_mim_mode:
        # === New strict MIM semantics ===
        # 1) Normalize incoming rule ids from 'id' only.
        for r in new_rules:
            if isinstance(r, dict):
                rid = (r.get("id") or "").strip()
                # Normalize into "id" so downstream paths are consistent
                r["id"] = rid

        # Build quick lookups for existing rules by id and (name,kind) (was computed earlier as existing_by_id/name)
        # Recompute here to be safe if code changes above in future.
        existing_by_name = {}
        existing_by_id = {}
        for idx, r in enumerate(business_rules):
            if not isinstance(r, dict):
                continue
            rid0 = (r.get("id") or "").strip()
            rn0 = (r.get("rule_name") or r.get("name") or "").strip()
            kind0 = _norm_kind(r.get("Kind") or r.get("kind"))
            if rn0:
                key = f"{rn0}||{kind0}"
                existing_by_name[key] = {"idx": idx, "rule": r}
            if rid0:
                existing_by_id[rid0] = {"idx": idx, "rule": r}
        # Filtered id→entry map for only rules in selected model
        existing_by_id_model = {rid: entry for rid, entry in (existing_by_id or {}).items() if rid in allowed_ids_in_model}
        # Broader map for link resolution during this merge: include all rules, not only those already in the model
        existing_by_id_for_links = dict(existing_by_id)

        # New: Kind detector (does not mutate), for overlay
        def _detect_incoming_kind(incoming_rule: dict) -> str:
            """Return normalized incoming kind (does not mutate existing rules)."""
            return (incoming_rule.get("Kind") or incoming_rule.get("kind") or "").strip()

        created_ids: List[str] = []
        updated_ids: List[str] = []
        created_name_to_id: dict[str, str] = {}
        # No longer accumulate Top-Level decisions for overlay storage on the model

        for incoming in new_rules:
            if not isinstance(incoming, dict):
                continue

            # Normalize ID handling (use 'id' only)
            incoming_id_raw = incoming.get("id") or ""
            incoming_id = (incoming_id_raw or "").strip()
            incoming_name = (incoming.get("rule_name") or incoming.get("name") or "").strip()

            # Determine newness strictly from the enrichment flag. Items with an existing 'id' (even if rule_id is empty) are EXISTING.
            is_new = bool(incoming.get("__source_id_blank", False))
            print(f"[TRACE] MIM classify: {(incoming.get('rule_name') or incoming.get('name') or '(unnamed)')} → {'NEW' if is_new else 'EXISTING'} (id='{incoming_id}')")

            if is_new:
                # --- CREATE path ---
                # Ensure we only create once; avoid duplicates by content+name
                norm_incoming = _normalize_rule_for_compare(incoming)
                duplicate = any(_normalize_rule_for_compare(r) == norm_incoming for r in business_rules if isinstance(r, dict))
                if duplicate:
                    eprint(f"[INFO] MIM/Create: Skipping duplicate new rule by content: '{incoming_name or '(unnamed)'}'")
                    continue

                # Reuse generated uuid if already set earlier; otherwise generate now
                new_id = (incoming.get("id") or "").strip() or str(uuid.uuid4())
                incoming["id"] = new_id
                # Record created name→id mapping
                if incoming_name:
                    created_name_to_id[incoming_name.casefold()] = new_id
                incoming["archived"] = False
                incoming.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                # (No longer apply hierarchy metadata to rule)

                business_rules.append(incoming)
                created_ids.append(new_id)
                if incoming_name:
                    existing_by_name[incoming_name] = {"idx": len(business_rules)-1, "rule": incoming}
                existing_by_id[new_id] = {"idx": len(business_rules)-1, "rule": incoming}
                # Ensure newly created rules are available to link resolution in this merge
                existing_by_id_for_links[new_id] = {"idx": len(business_rules)-1, "rule": incoming}
                print(f"[INFO] MIM/Create: Created new decision/rule '{incoming_name or '(unnamed)'}' (id={new_id}).")
                continue

            # --- UPDATE path (links only; no Kind mutation). If not found locally, skip to avoid implicit create. ---
            ex = existing_by_id.get(incoming_id)
            if not ex and incoming_name:
                inc_kind = _detect_incoming_kind(incoming)
                key_name_kind = f"{incoming_name}||{_norm_kind(inc_kind)}"
                ex = existing_by_name.get(key_name_kind)
            if not ex:
                eprint(f"[WARN] MIM/Update: Rule '{incoming_name or '(unnamed)'}' with id={incoming_id} not found; skipping update to avoid unintended create.")
                continue

            # Do not mutate Kind on existing rules (overlay approach). Only merge links.
            inc_kind = _detect_incoming_kind(incoming)
            added_links = _merge_links_in_place_generic(ex["rule"], incoming, existing_by_id_for_links)
            if ex["rule"].get("archived", False) and added_links:
                ex["rule"]["archived"] = False
                print(f"[INFO] MIM/Update: Unarchived '{incoming_name or '(unnamed)'}' due to link changes.")

            # (No longer apply hierarchy metadata to rule)

            if added_links:
                business_rules[ex["idx"]] = ex["rule"]
                updated_ids.append(ex["rule"].get("id") or "")
                print(f"[INFO] MIM/Update: Updated '{incoming_name or '(unnamed)'}' (+{added_links} link(s)).")
            else:
                print(f"[TRACE] MIM/Update: No link changes for '{incoming_name or '(unnamed)'}'.")

        # Persist if there were changes
        if created_ids or updated_ids:
            # Strip internal flags before writing
            for rr in business_rules:
                if isinstance(rr, dict):
                    rr.pop("__source_id_blank", None)
                    rr.pop("__source_orig_id", None)
            _safe_backup_json(br_path)
            write_json(br_path, business_rules)
            if created_ids:
                print(f"Added {len(created_ids)} new rule(s) to business_rules.json")
            if updated_ids:
                print(f"Updated {len([x for x in updated_ids if x])} existing rule(s) in business_rules.json")

        # Ensure models.json contains any newly created ids
        for rid in created_ids:
            if rid:
                ensure_model_ids.add(rid)

        # Upgrade hierarchy_meta names → ids using just-created rules (ensures topDecisionId gets written)
        if hierarchy_meta and isinstance(hierarchy_meta, dict) and created_name_to_id:
            by_name = hierarchy_meta.get("by_name") or {}
            if isinstance(by_name, dict) and by_name:
                for tname, payload in list(by_name.items()):
                    if not isinstance(payload, dict):
                        continue
                    key = (str(tname) or "").strip().casefold()
                    tid = created_name_to_id.get(key)
                    if tid:
                        hierarchy_meta.setdefault("by_id", {})
                        hierarchy_meta["by_id"][tid] = payload
                        del by_name[tname]
                hierarchy_meta["by_name"] = by_name

        # --- models.json merge with Hierarchies only (ditch overlay) ---
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

        # Merge hierarchy records into model_obj['hierarchies']
        model_obj = models[sel_idx]
        def _merge_hierarchies_into_model(model_obj: dict, hierarchy_meta: Optional[Dict[str, Any]]) -> bool:
            """
            Merge hierarchy records into model_obj['hierarchies'] without duplicates.
            Each record structure:
              {
                "topDecisionId": "<id or ''>",
                "name": "<hierarchy_name>",
                "description": "<hierarchy_description>"
              }

            Rules:
            - Prefer a single record per hierarchy "name".
            - If we get both an id and a name for the same hierarchy in the same call,
              update the same record instead of appending a second one.
            - If a matching record exists (by id OR by hierarchy name, case‑insensitive),
              update missing fields rather than creating a new record.

            Returns True if the list changed.
            """
            if not hierarchy_meta or not isinstance(hierarchy_meta, dict):
                return False

            by_id = hierarchy_meta.get("by_id") or {}
            by_name = hierarchy_meta.get("by_name") or {}
            if not isinstance(by_id, dict):
                by_id = {}
            if not isinstance(by_name, dict):
                by_name = {}

            # Start from current list (normalize)
            existing = model_obj.get("hierarchies")
            if not isinstance(existing, list):
                existing = []

            # Fast indexes over the live 'existing' list (kept up to date as we modify/append)
            def _norm(s: str) -> str:
                return (s or "").strip()

            def _ncf(s: str) -> str:
                return _norm(s).casefold()

            def _rebuild_indexes():
                idx_by_id = {}
                idx_by_hname = {}
                for i, item in enumerate(existing):
                    if not isinstance(item, dict):
                        continue
                    tid = _norm(item.get("topDecisionId", ""))
                    hname = _norm(item.get("name", ""))
                    if tid:
                        idx_by_id[tid] = i
                    if hname:
                        idx_by_hname[_ncf(hname)] = i
                return idx_by_id, idx_by_hname

            idx_by_id, idx_by_hname = _rebuild_indexes()

            changed = False

            def _ensure_record(tid: str, tname: str, hname: str, hdesc: str) -> None:
                """
                Find or create a record for this hierarchy. Match order:
                1) by topDecisionId (exact)
                2) by hierarchy 'name' (case-insensitive)
                Then update missing fields on the found record; otherwise append a new record.
                """
                nonlocal idx_by_id, idx_by_hname, changed

                tid = _norm(tid)
                tname = _norm(tname)
                hname = _norm(hname)
                hdesc = _norm(hdesc)

                # Try match by id first
                i = idx_by_id.get(tid) if tid else None
                # Else match by hierarchy name
                if i is None and hname:
                    i = idx_by_hname.get(_ncf(hname))

                if i is not None:
                    rec = existing[i]
                    # Update only if values are missing/empty
                    if tid and not _norm(rec.get("topDecisionId", "")):
                        rec["topDecisionId"] = tid
                        changed = True
                    # Always prefer the latest non-empty description
                    if hdesc and _norm(rec.get("description", "")) != hdesc:
                        rec["description"] = hdesc
                        changed = True
                    # Ensure the canonical hierarchy name is set
                    if hname and _norm(rec.get("name", "")) != hname:
                        rec["name"] = hname
                        changed = True
                else:
                    # Create new
                    rec = {
                        "topDecisionId": tid,
                        "name": hname,
                        "description": hdesc,
                    }
                    existing.append(rec)
                    changed = True

                # Rebuild indexes after any mutation
                idx_by_id, idx_by_hname = _rebuild_indexes()

            # 1) Ingest id-keyed entries
            for tid, payload in by_id.items():
                if not isinstance(payload, dict):
                    continue
                hname = _norm(payload.get("hierarchy_name", ""))
                if not hname:
                    continue
                hdesc = _norm(payload.get("hierarchy_description", ""))
                _ensure_record(_norm(str(tid)), "", hname, hdesc)

            # 2) Ingest name-keyed entries (may enrich the same records)
            for tname, payload in by_name.items():
                if not isinstance(payload, dict):
                    continue
                hname = _norm(payload.get("hierarchy_name", ""))
                if not hname:
                    continue
                hdesc = _norm(payload.get("hierarchy_description", ""))
                _ensure_record("", _norm(str(tname)), hname, hdesc)

            if changed:
                model_obj["hierarchies"] = existing
            return changed

        # Build to_append: created ids first
        to_append_created = list(created_ids)

        # Preserve only those ensured ids that are already in this model (no cross-add)
        model_obj = models[sel_idx]
        ids = model_obj.get("businessLogicIds")
        if not isinstance(ids, list):
            ids = []
        filtered_ensure: list[str] = []
        for rid in ensure_model_ids:
            if rid in ids and rid not in to_append_created:
                filtered_ensure.append(rid)

        to_append = list(to_append_created) + filtered_ensure

        # Always attempt hierarchy merge (even if no new businessLogicIds)
        overlay_changed = False  # overlay is deprecated, always False
        hierarchy_changed = _merge_hierarchies_into_model(model_obj, hierarchy_meta)

        # Remove the deprecated overlay list from the model object
        model_obj.pop("topLevelDecisionIds", None)

        # Append new/ensured ids to businessLogicIds with (name,kind) de-dup (one per model per name+kind)
        if to_append:
            # id→(name_cf, kind_norm) map from current business_rules
            id_to_name_kind = {}
            for r in business_rules:
                if isinstance(r, dict):
                    rid0 = (r.get("id") or "").strip()
                    rn0 = (r.get("rule_name") or r.get("name") or "").strip()
                    if rid0 and rn0:
                        kind0 = _norm_kind(r.get("Kind") or r.get("kind"))
                        id_to_name_kind[rid0] = (rn0.casefold(), kind0)

            # Swap-by-(name,kind) for newly created ids
            for new_id in to_append_created:
                if not new_id:
                    continue
                new_key = id_to_name_kind.get(new_id)
                if not new_key:
                    continue
                ids = [existing_id for existing_id in ids
                       if id_to_name_kind.get(existing_id) != new_key or existing_id == new_id]

            before_len = len(ids)
            for rid in to_append:
                if rid and rid not in ids:
                    ids.append(rid)

            if len(ids) != before_len:
                model_obj["businessLogicIds"] = ids
                ids_changed = True
            else:
                ids_changed = False
        else:
            ids_changed = False

        # Persist if either ids list or hierarchies changed
        if overlay_changed or ids_changed or hierarchy_changed:
            _safe_backup_json(models_path)
            models[sel_idx] = model_obj
            write_json(models_path, models)
            if ids_changed:
                print(f"Appended {len(model_obj['businessLogicIds']) - before_len} rule id(s) to model {selected_model_id} in models.json")
            # Only print overlay change if true (never true now)
            if overlay_changed:
                print(f"Updated model {selected_model_id} Top-Level overlay.")
            if hierarchy_changed:
                print(f"Updated model {selected_model_id} hierarchies list.")
        else:
            print("No changes to models.json (businessLogicIds and hierarchies unchanged).")

        # Done with MIM-specific path; skip the original mixed add/update logic
        return
    else:
        # Original behavior for 'top' and 'comp'
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
        for new_rule in new_rules:
            rn = (new_rule.get("rule_name") or new_rule.get("name") or "").strip()
            kind_new = _norm_kind(new_rule.get("Kind") or new_rule.get("kind"))
            key_name_kind = f"{rn}||{kind_new}" if rn else None
            ex = existing_by_name.get(key_name_kind) if key_name_kind else None
            if ex and _has_dmn_material_change(ex["rule"], new_rule):
                preserved_id = ex["rule"].get("id")
                preserved_archived = ex["rule"].get("archived", False)
                new_rule["id"] = preserved_id or new_rule.get("id") or str(uuid.uuid4())
                new_rule["archived"] = preserved_archived
                business_rules[ex["idx"]] = new_rule
                if rn:
                    existing_by_name[f"{rn}||{kind_new}"] = {"idx": ex["idx"], "rule": new_rule}
                print(f"[INFO] Updated rule by name with DMN changes (incl. allowedValues): '{rn}'")
                continue
            new_norm = _normalize_rule_for_compare(new_rule)
            fp_new = _content_fingerprint(new_rule)
            rid_new = (new_rule.get("id") or "").strip()
            if rid_new:
                # If PCPT returned an id, treat existence by id instead of by content
                exists = any(((r.get("id") or "").strip() == rid_new) for r in business_rules if isinstance(r, dict))
            else:
                # Fallback to content-based duplicate detection when there is no id
                exists = any(_normalize_rule_for_compare(r) == new_norm for r in business_rules if isinstance(r, dict))
            if exists:
                # If this rule exists by id, attempt to merge in any new links instead of creating a duplicate.
                merged_links = False
                if rid_new:
                    ex_by_id = existing_by_id.get(rid_new)
                    if ex_by_id:
                        added_links = _merge_links_in_place_generic(ex_by_id["rule"], new_rule, existing_by_id)
                        if added_links:
                            business_rules[ex_by_id["idx"]] = ex_by_id["rule"]
                            print(f"[INFO] Updated rule by id with {added_links} new link(s): '{rn or '(unnamed)'}'")
                            merged_links = True
                            simple_links_updated = True
                if merged_links:
                    # We treated this as a link-only update; do not record as a skipped duplicate.
                    continue

                # Otherwise, treat as a pure duplicate and skip adding a new rule.
                matches = existing_by_fp.get(fp_new) or []
                if not matches and rn:
                    for idx2, r in enumerate(business_rules):
                        if not isinstance(r, dict):
                            continue
                        rn2 = (r.get("rule_name") or r.get("name") or "").strip()
                        if rn2 != rn:
                            continue
                        kind2 = _norm_kind(r.get("Kind") or r.get("kind"))
                        if kind2 != kind_new:
                            continue
                        matches.append({"idx": idx2, "id": r.get("id"), "name": rn})
                        break
                skipped_details.append({"new_name": rn or "(unnamed)", "fingerprint": fp_new, "matches": matches})
                if matches:
                    first = matches[0]
                    eprint(f"[INFO] Skipping duplicate by content: '{rn}' → existing id={first.get('id')} (fp={fp_new})")
                else:
                    eprint(f"[INFO] Skipping duplicate by content: '{rn}' (fp={fp_new})")
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

    if added_ids or simple_links_updated or (is_mim_mode and updated_top_links > 0):
        # Strip internal flags from all rules before persisting
        for rr in business_rules:
            if isinstance(rr, dict):
                rr.pop("__source_id_blank", None)
                rr.pop("__source_orig_id", None)
        _safe_backup_json(br_path)
        write_json(br_path, business_rules)
        if added_ids:
            print(f"Added {len(added_ids)} rule(s) to business_rules.json")
        if is_mim_mode and updated_top_links > 0:
            print(f"Updated links on Top-Level decisions (+{updated_top_links}).")
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

    # Add newly created ids and ensure updated existing ids are present in the model
    # 1) Start with newly created ids
    to_append_created = list(added_ids)

    # 2) Ensure step should only PRESERVE ids already in this model, not cross-add between models
    model_obj = models[sel_idx]
    ids = model_obj.get("businessLogicIds")
    if not isinstance(ids, list):
        ids = []

    filtered_ensure: list[str] = []
    for rid in ensure_model_ids:
        if rid in ids and rid not in to_append_created:
            filtered_ensure.append(rid)

    # Build final to_append (created + filtered ensure)
    to_append = list(to_append_created) + filtered_ensure

    if to_append:
        # Build a quick id→(name_cf, kind_norm) map from business_rules for (name,kind)-based collision handling
        id_to_name_kind = {}
        for r in business_rules:
            if isinstance(r, dict):
                rid0 = (r.get("id") or "").strip()
                rn0 = (r.get("rule_name") or r.get("name") or "").strip()
                if rid0 and rn0:
                    kind0 = _norm_kind(r.get("Kind") or r.get("kind"))
                    id_to_name_kind[rid0] = (rn0.casefold(), kind0)

        # Before appending any newly created id, remove any existing ids in this model
        # that have the same (name,kind), so we keep exactly one per model per name+kind.
        for new_id in to_append_created:
            if not new_id:
                continue
            new_key = id_to_name_kind.get(new_id)
            if not new_key:
                continue
            ids = [existing_id for existing_id in ids
                   if id_to_name_kind.get(existing_id) != new_key or existing_id == new_id]

        # Append unique ids
        before_len = len(ids)
        for rid in to_append:
            if rid and rid not in ids:
                ids.append(rid)

        if len(ids) != before_len:
            _safe_backup_json(models_path)
            model_obj["businessLogicIds"] = ids
            models[sel_idx] = model_obj
            write_json(models_path, models)
            print(f"Appended {len(ids) - before_len} rule id(s) to model {selected_model_id} in models.json")
        else:
            print("No changes to models.json (all relevant rule ids already present).")
    else:
        print("No changes to models.json (no new or updated rule ids).")

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

def _normalize_rule_for_compare(rule: dict) -> dict:
    # Exclude volatile keys for duplicate detection
    exclude = {"id", "timestamp", "archived"}
    return {k: v for k, v in rule.items() if k not in exclude}

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

def step_header(step_no, title, focus=None):
    """
    Print a prominent step header with optional focus context.
    - step_no: int | str | None (e.g., 5, "5A", or None to omit numbering)
    - title: short title for the step
    - focus: optional dict or list of strings to highlight current model/rules/hierarchies
    """
    bar = f"{ANSI_BLUE}{'―'*72}{ANSI_RESET}" if sys.stdout.isatty() else "—"*72
    print("\n" + bar)
    if step_no is None:
        print(f"{ANSI_BOLD}{title}{ANSI_RESET}")
    else:
        print(f"{ANSI_BOLD}STEP {step_no}:{ANSI_RESET} {title}")
    if focus:
        if isinstance(focus, dict):
            for k, v in focus.items():
                print(f"  • {ANSI_CYAN}{k}:{ANSI_RESET} {v}")
        elif isinstance(focus, (list, tuple)):
            for item in focus:
                print(f"  • {item}")
        else:
            print(f"  • {focus}")
    print(bar)

# ---------------------------
# Helper: Build temp source dir from model files
# ---------------------------
from typing import Any, Dict
def build_temp_source_from_model(model_info: Dict[str, Any], spec_info: Dict[str, Any]) -> Path:
    """
    Build a deterministic temp source directory containing all source/doc files referenced by model's rules.
    Returns the absolute Path to the temp directory.
    """
    temp_dir = TMP_DIR / "pcpt_source_from_model"
    # Remove if exists, then recreate
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    rules_path = Path(model_info["rules_out_path"])
    rules = load_json(rules_path)
    if not isinstance(rules, list):
        eprint(f"[WARN] build_temp_source_from_model: rules_out_path does not contain a list: {rules_path}")
        rules = []

    # Collect candidate file paths from rules
    candidate_paths = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        for k, v in rule.items():
            # Only include "code_file" and keys ending in "_file" except "doc_file"
            if k == "code_file" and isinstance(v, str):
                candidate_paths.append(v)
            elif k.endswith("_file") and k != "doc_file" and isinstance(v, str):
                candidate_paths.append(v)
    # Deduplicate, preserve order
    seen = set()
    deduped = []
    for p in candidate_paths:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    candidate_paths = deduped

    # Bases for resolution
    repo_root = REPO_ROOT
    spec_dir = Path(spec_info["spec_dir"])
    root_dir_str = spec_info.get("root_directory") or ""
    root_dir = Path(root_dir_str).expanduser()
    if not root_dir.is_absolute():
        root_dir = (spec_dir / root_dir).resolve()
    bases = [repo_root, spec_dir, root_dir]

    def _resolve(p: str) -> Optional[Path]:
        path_obj = Path(p).expanduser()
        if path_obj.is_absolute() and path_obj.exists():
            return path_obj
        for base in bases:
            candidate = (base / p).expanduser()
            if candidate.exists():
                return candidate
        eprint(f"[WARN] build_temp_source_from_model: Could not resolve file: {p}")
        return None

    resolved_files = []
    for p in candidate_paths:
        resolved = _resolve(p)
        if resolved is not None:
            resolved_files.append((p, resolved))

    copied = 0
    for orig_p, file_path in resolved_files:
        # Find first base that is a parent
        dest_path = None
        for base in bases:
            try:
                rel = file_path.relative_to(base)
                dest_path = temp_dir / rel
                break
            except ValueError:
                continue
        if dest_path is None:
            dest_path = temp_dir / file_path.name
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(file_path, dest_path)
            copied += 1
        except Exception as ex:
            eprint(f"[WARN] build_temp_source_from_model: Failed to copy {file_path} to {dest_path}: {ex}")

    print(f"[TRACE] build_temp_source_from_model: Discovered {len(candidate_paths)} file(s), copied {copied} to temp source: {temp_dir.resolve()}")
    if copied == 0:
        eprint(f"[WARN] build_temp_source_from_model: No files copied to temp source for model.")
    return temp_dir.resolve()

def _resolve_template_path(mode: str) -> Path:
    """
    mode: 'top' / 'selected-top'            -> single composed decision templates
          'next'                            -> multi composed decision template (can emit multiple rules)
          'mim' / 'mim-minimal'            -> meet-in-the-middle decision template
    """
    m = (mode or "top").strip().lower()
    # Allow callers to pass either the logical mode ('selected-top') or the template base name
    if m in ("selected-top", HIERARCHY_TEMPLATE_SELECTED_TOP):
        return HIERARCHY_TEMPLATE_SELECTED_TOP_DIR
    if m == "comp":
        return HIERARCHY_TEMPLATE_COMP_DIR
    if m in ("mim", "mim-minimal"):
        return HIERARCHY_TEMPLATE_MIM_DIR
    if m == "top":
        return HIERARCHY_TEMPLATE_TOP_DIR

    # Unknown mode → error
    raise ValueError(f"Unknown template mode: '{mode}' (normalized: '{m}')")

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