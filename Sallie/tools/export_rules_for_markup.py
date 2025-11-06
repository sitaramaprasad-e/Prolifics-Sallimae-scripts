#!/usr/bin/env python3
"""
Export rules for files under a given source path from business_rules.json.

Examples
--------
# Basic (prints to stdout, plain text like your sample)
python export_rules_for_path.py code/sf

# Specify a different rules file and write Markdown to a file
python export_rules_for_path.py code/sf \
  --rules-file /path/to/business_rules.json \
  --format md \
  --output rules_in_sf.md
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import PurePosixPath, Path
from typing import Any, Dict, List, Iterable, Tuple
import os

TRACE_ENABLED = False
TRACE_REMAINING = 0
REASONS_COUNT: Dict[str, int] = defaultdict(int)

# ---- Rule ID helpers --------------------------------------------------------
from typing import Optional

def _strip_brackets(s: str) -> str:
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("(") and s.endswith(")")):
        return s[1:-1]
    return s

def normalize_rule_id(v: Any) -> Optional[str]:
    """Return a normalized string id or None if unusable (trim, strip braces, lowercase)."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s = _strip_brackets(s)
    return s.lower()

def extract_rule_id(rule: Dict[str, Any]) -> Optional[str]:
    """Extract a rule id, prioritizing 'id' (business_rules.json), then common fallbacks."""
    for key in ("id", "rule_id", "ruleId", "uuid"):
        if key in rule:
            rid = normalize_rule_id(rule.get(key))
            if rid:
                return rid
    meta = rule.get("meta") or rule.get("metadata")
    if isinstance(meta, dict):
        for key in ("id", "rule_id", "ruleId", "uuid"):
            if key in meta:
                rid = normalize_rule_id(meta.get(key))
                if rid:
                    return rid
    return None

def _tprint(msg: str) -> None:
    global TRACE_REMAINING
    if TRACE_ENABLED and TRACE_REMAINING > 0:
        print(msg)
        TRACE_REMAINING -= 1

def _count(reason: str) -> None:
    REASONS_COUNT[reason] += 1

# Constants for default argument values
DEFAULT_FORMAT = "json"
DEFAULT_OUTPUT = "./.tmp/rules-for-markup/exported-rules.json"
DEFAULT_RULES_FILE = "~/.model/business_rules.json"
DEFAULT_RUNS_FILE = "~/.model/runs.json"

def normalize_posix(p: str) -> str:
    """Normalize to POSIX-style path without leading './'."""
    # Treat provided paths as POSIX-ish (business_rules.json uses forward slashes)
    pp = PurePosixPath(p)
    # PurePosixPath('.') -> '.': avoid that
    s = str(pp)
    if s.startswith("./"):
        s = s[2:]
    return s

def path_is_under(source_prefix: str, target: str, case_insensitive: bool) -> bool:
    """Return True if target path is under source prefix (prefix on parts)."""
    sp = PurePosixPath(source_prefix)
    tp = PurePosixPath(target)
    if case_insensitive:
        sp_parts = [part.lower() for part in sp.parts]
        tp_parts = [part.lower() for part in tp.parts]
    else:
        sp_parts = list(sp.parts)
        tp_parts = list(tp.parts)
    return len(tp_parts) >= len(sp_parts) and tp_parts[:len(sp_parts)] == sp_parts

def load_rules(rules_file: Path) -> List[Dict[str, Any]]:
    try:
        with rules_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("business_rules.json must contain a top-level JSON array.")
        return data
    except FileNotFoundError:
        print(f"ERROR: rules file not found: {rules_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: failed to parse JSON ({rules_file}): {e}", file=sys.stderr)
        sys.exit(1)

def load_runs(runs_file: Path) -> List[Dict[str, Any]]:
    try:
        with runs_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("runs.json must contain a top-level JSON array.")
        return data
    except FileNotFoundError:
        print(f"[warn] runs file not found: {runs_file}")
        return []
    except json.JSONDecodeError as e:
        print(f"[warn] failed to parse JSON ({runs_file}): {e}")
        return []

def normalize_fs_path(p: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(p)))

# Helper to format a run id for trace lines
def format_run_id(run: Dict[str, Any]) -> str:
    """Human-friendly identifier for a run (used in trace output)."""
    b = run.get("build")
    ts = run.get("timestamp")
    if b is not None and ts:
        return f"build={b} ts={ts}"
    if b is not None:
        return f"build={b}"
    if ts:
        return f"ts={ts}"
    return "(no-id)"

def collect_rule_ids_for_root(runs: List[Dict[str, Any]], root_path: str) -> set[str]:
    target = normalize_fs_path(root_path).lower()
    _tprint(f"[trace] root target: {target}")
    allowed: set[str] = set()
    for run in runs:
        rd = run.get("root_dir") or run.get("root_path")
        run_id_str = format_run_id(run)
        if not rd:
            _tprint(f"[trace] run skipped (no root_dir) (run={run_id_str})")
            _count("StageA:run missing root_dir")
            continue
        normalized = normalize_fs_path(rd).lower()
        if normalized == target:
            rids = (run.get("rule_ids") or [])
            _tprint(f"[trace] run matched root_dir: {rd} (run={run_id_str}, rule_ids={len(rids)})")
            # list all rule ids for this run (so you can manually compare)
            for rid in rids:
                _tprint(f"[trace]   run-linked rule_id: {rid}")
                if rid is not None:
                    allowed.add(str(rid).strip().lower())
        else:
            _tprint(f"[trace] run skipped (root_dir mismatch): {rd} (run={run_id_str})")
    if TRACE_ENABLED:
        sample = ", ".join(list(allowed)[:5])
        _tprint(f"[trace] allowed_ids sample (first 5): {sample}")
    return allowed

def dedupe_in_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def group_rules_by_file(
    rules: List[Dict[str, Any]],
    source_prefix: str,
    case_insensitive: bool,
    include_archived: bool = False,
) -> "OrderedDict[str, List[Dict[str, str]]]":
    """
    Group rules by code_file for files under source_prefix.
    Maintain first-seen order of files and rules.
    Each rule record includes rule_name, code_file, and code_function.
    """
    grouped: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
    considered_total = 0
    kept_total = 0
    for r in rules:
        considered_total += 1
        if not include_archived and r.get("archived") is True:
            _tprint(f"[trace] reject (Stage B) rule_id={extract_rule_id(r)} — archived and include_archived=False")
            _count("StageB:archived excluded")
            continue
        code_file = r.get("code_file")
        rule_name = r.get("rule_name")
        code_function = r.get("code_function")
        if not code_file or not rule_name:
            missing = "code_file" if not code_file else "rule_name"
            _tprint(f"[trace] reject (Stage B) rule_id={extract_rule_id(r)} — missing {missing}")
            _count(f"StageB:missing {missing}")
            continue
        cf_norm = normalize_posix(code_file)
        if path_is_under(source_prefix, cf_norm, case_insensitive):
            kept_total += 1
            _tprint(f"[trace] keep (Stage B: path under source) rule_id={extract_rule_id(r)} file={cf_norm}")
            if cf_norm not in grouped:
                grouped[cf_norm] = []
            grouped[cf_norm].append({
                "rule_name": rule_name,
                "code_function": code_function,
                "archived": r.get("archived"),
                "rule_id": extract_rule_id(r),
            })
        else:
            _tprint(f"[trace] reject (Stage B) rule_id={extract_rule_id(r)} file={cf_norm} — not under {source_prefix}")
            _count("StageB:not under source_prefix")
    if TRACE_ENABLED:
        print(f"[trace] Stage B considered={considered_total} kept={kept_total}")

    # de-duplicate rule records per file by rule_name, preserving order,
    # but prefer non-archived over archived when both exist.
    def _is_archived_flag(v):
        if v is True:
            return True
        if isinstance(v, str) and v.lower() in ("true", "1", "yes", "y"):
            return True
        if v == 1:
            return True
        return False
    for cf in list(grouped.keys()):
        index_by_rule = {}
        deduped = []
        for item in grouped[cf]:
            rn = item["rule_name"]
            is_arch = _is_archived_flag(item.get("archived"))
            if rn not in index_by_rule:
                index_by_rule[rn] = len(deduped)
                deduped.append(item)
            else:
                idx = index_by_rule[rn]
                existing = deduped[idx]
                existing_arch = _is_archived_flag(existing.get("archived"))
                # If the existing is archived and the new one is not, replace.
                if existing_arch and not is_arch:
                    _tprint(f"[trace] prefer non-archived over archived for rule_name='{rn}' in file={cf}")
                    deduped[idx] = item
                    # index remains the same; mapping already points to idx
                # Otherwise keep the first (non-archived already wins or both archived)
        grouped[cf] = deduped

    # Strip internal-only fields from output (do not expose archived/code_file in the result)
    for cf in grouped:
        for item in grouped[cf]:
            if "archived" in item:
                del item["archived"]
            if not TRACE_ENABLED and "rule_id" in item:
                del item["rule_id"]

    return grouped

def format_plain(grouped: "OrderedDict[str, List[Dict[str, str]]]") -> str:
    lines: List[str] = []
    for code_file, rule_items in grouped.items():
        lines.append(f"{code_file}:")
        for item in rule_items:
            rn = item["rule_name"]
            cf = item.get("code_function") or ""
            lines.append(f"{rn} ({cf})" if cf else rn)
        lines.append("")  # blank line between files
    # Remove trailing blank if present
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)

def format_md(grouped: "OrderedDict[str, List[Dict[str, str]]]") -> str:
    lines: List[str] = []
    for code_file, rule_items in grouped.items():
        lines.append(f"### `{code_file}`")
        for item in rule_items:
            rn = item["rule_name"]
            cf = item.get("code_function") or ""
            if cf:
                lines.append(f"- **{rn}** — `{cf}`")
            else:
                lines.append(f"- {rn}")
        lines.append("")
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)

def format_json(grouped: "OrderedDict[str, List[Dict[str, str]]]") -> str:
    # Convert OrderedDict to normal dict for JSON output (ordering preserved in Python 3.7+)
    return json.dumps(grouped, indent=2, ensure_ascii=False)

def load_rule_categories(rc_path: Path) -> Dict[str, Any]:
    try:
        with rc_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[warn] rule_categories.json not found: {rc_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"[warn] failed to parse JSON ({rc_path}): {e}")
        return {}

def compute_non_business_category_ids(rc: Dict[str, Any]) -> set[str]:
    groups = rc.get("ruleCategoryGroups", []) or []
    cat_list = rc.get("ruleCategories", []) or []
    # businessRelevant missing => treated as True
    group_biz = {g.get("id"): (g.get("businessRelevant") is not False) for g in groups if g.get("id")}
    non_biz_group_ids = {gid for gid, is_biz in group_biz.items() if is_biz is False}
    non_biz_cat_ids = {c.get("id") for c in cat_list if c.get("groupId") in non_biz_group_ids and c.get("id")}
    return non_biz_cat_ids

def extract_rule_categories(rule: Dict[str, Any]) -> List[str]:
    # NOTE: This function only returns raw category IDs embedded on the rule and may not resolve names.
    # Prefer single id fields first
    cid = rule.get("categoryId") or rule.get("category_id")
    if cid:
        return [str(cid)]
    cats = rule.get("categories")
    if isinstance(cats, list):
        # normalize to strings
        return [str(x) for x in cats if x is not None]
    return []

def main():
    parser = argparse.ArgumentParser(
        description="Export business rules grouped by code_file for a given source path."
    )
    parser.add_argument(
        "source_path",
        help="Source path prefix to filter files (e.g., 'code/sf', 'code/stored_proc')."
    )
    parser.add_argument(
        "--rules-file",
        default=DEFAULT_RULES_FILE,
        help=f"Path to business_rules.json (default: {DEFAULT_RULES_FILE}; '~' will be expanded)"
    )
    parser.add_argument(
        "--runs-file",
        default=DEFAULT_RUNS_FILE,
        help=f"Path to runs.json (default: {DEFAULT_RUNS_FILE}; '~' will be expanded)"
    )
    parser.add_argument(
        "--format",
        choices=["plain", "md", "json"],
        default=DEFAULT_FORMAT,
        help=f"Output format (default: {DEFAULT_FORMAT})."
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Write output to this file (default: {DEFAULT_OUTPUT})."
    )
    parser.add_argument(
        "--case-insensitive",
        action="store_true",
        help="Match paths case-insensitively."
    )
    parser.add_argument(
        "--include-archived",
        action="store_true",
        help="Include archived rules (by default they are skipped)."
    )
    parser.add_argument(
        "--include-all-rules",
        action="store_true",
        help="Include all rules regardless of business relevance (overrides rule_categories.json filtering)."
    )
    parser.add_argument(
        "--root-path",
        default=None,
        help="Absolute project root path to filter rules by runs.json root_dir; if omitted, no run-based filtering is applied."
    )
    parser.add_argument(
        "--no-strict-root",
        action="store_true",
        help="Disable strict root behavior. By default, when --root-path is provided and no matching runs/rule_ids are found, the result will be empty. Use this flag to fall back to exporting all rules."
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=True,
        help="Enable verbose per-item tracing of why each rule/run was accepted or rejected (ON by default)."
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=200,
        help="Maximum number of per-item trace lines to emit (default: 200)."
    )
    parser.add_argument(
        "--no-trace",
        action="store_true",
        help="Disable verbose tracing (tracing is ON by default)."
    )

    args = parser.parse_args()
    global TRACE_ENABLED, TRACE_REMAINING
    # Tracing is ON by default; --no-trace disables it
    TRACE_ENABLED = False if args.no_trace else bool(args.trace)
    TRACE_REMAINING = int(args.trace_limit)

    source_prefix = normalize_posix(args.source_path)
    rules_file = Path(args.rules_file).expanduser()
    print(f"[info] Using rules file: {rules_file}")
    print(f"[info] Source prefix: {source_prefix} (case_insensitive={args.case_insensitive})")
    print(f"[info] Trace: {'ON' if TRACE_ENABLED else 'OFF'} (limit={TRACE_REMAINING})")

    rules = load_rules(rules_file)
    print(f"[info] Loaded {len(rules)} rules")

    # Prepare rule category lookups (for printing names) even if we don't filter
    rc_path = (Path(args.rules_file).expanduser().parent) / "rule_categories.json"
    rc_lookup: Dict[str, Any] = {}
    rc_groups_by_id: Dict[str, Dict[str, Any]] = {}
    rc_cats_by_id: Dict[str, Dict[str, Any]] = {}
    if rc_path.exists():
        rc_lookup = load_rule_categories(rc_path)
        _groups = rc_lookup.get("ruleCategoryGroups", []) or []
        _cats = rc_lookup.get("ruleCategories", []) or []
        rc_groups_by_id = {g.get("id"): g for g in _groups if g.get("id")}
        rc_cats_by_id = {c.get("id"): c for c in _cats if c.get("id")}
        # Build reverse lookup: category name (lowercased) -> id
        rc_cats_by_name: Dict[str, str] = {}
        for _cid, _cmeta in rc_cats_by_id.items():
            _cname = (_cmeta.get("name") or "").strip()
            if _cname:
                rc_cats_by_name[_cname.lower()] = _cid
    else:
        print(f"[info] No rule_categories.json next to business_rules.json ({rc_path}); category names will not be printed")
        rc_cats_by_name: Dict[str, str] = {}

    def _format_rule_categories(rule: Dict[str, Any]) -> List[str]:
        ids = _extract_rule_category_ids(rule)
        out: List[str] = []
        for cid in ids:
            cmeta = rc_cats_by_id.get(cid)
            cname = cmeta.get("name") if cmeta else None
            if cname:
                out.append(f"{cname} ({cid})")
            else:
                out.append(str(cid))
        return out

    def _extract_rule_category_ids(rule: Dict[str, Any]) -> List[str]:
        """
        Returns normalized category IDs for a rule.
        Order of precedence:
          1) ID fields: categoryId, category_id
          2) ID list: categories (list of ids)
          3) Name fields: rule_category, category, categoryName (single string)
          4) Name list: category_names (list of names)
        Name-based fields are resolved via rule_categories.json (rc_cats_by_name).
        """
        out: List[str] = []
        # ID fields first
        cid = rule.get("categoryId") or rule.get("category_id")
        if cid:
            out.append(str(cid))
        cats = rule.get("categories")
        if isinstance(cats, list):
            out.extend(str(x) for x in cats if x is not None)
        # Name fields
        name_single = rule.get("rule_category") or rule.get("category") or rule.get("categoryName")
        if isinstance(name_single, str) and name_single.strip():
            nid = rc_cats_by_name.get(name_single.strip().lower())
            if nid:
                out.append(nid)
        name_list = rule.get("category_names")
        if isinstance(name_list, list):
            for nm in name_list:
                if isinstance(nm, str) and nm.strip():
                    nid = rc_cats_by_name.get(nm.strip().lower())
                    if nid:
                        out.append(nid)
        # De-duplicate preserving order
        return dedupe_in_order(out)

    # --- Business relevance filter (rule_categories.json) ---
    if args.include_all_rules:
        print("[info] --include-all-rules supplied: skipping business relevance filter")
    else:
        # rc_path already computed above
        if rc_path.exists():
            rc = load_rule_categories(rc_path)
            non_biz_cat_ids = compute_non_business_category_ids(rc)
            if non_biz_cat_ids:
                # Report non-business groups and their affected categories
                groups_list = list(rc_groups_by_id.values())
                cats_list = list(rc_cats_by_id.values())
                # Compute non-business groups directly for printing
                non_biz_group_ids = {g.get("id") for g in groups_list if g.get("id") and g.get("businessRelevant") is False}
                if non_biz_group_ids:
                    print("[info] Non-business groups (businessRelevant=false):")
                    # stable order by name then id
                    for g in sorted((g for g in groups_list if g.get("id") in non_biz_group_ids),
                                    key=lambda x: (str(x.get("name") or ""), str(x.get("id") or ""))):
                        print(f"- Group: {g.get('name') or '(unknown group)'} ({g.get('id')})")
                    # Categories affected under each non-business group
                    print("[info] Categories that become non-business due to those groups:")
                    for gid in sorted(non_biz_group_ids):
                        gname = next((g.get("name") for g in groups_list if g.get("id") == gid), "(unknown group)")
                        print(f"- Group: {gname} ({gid})")
                        affected = [c for c in cats_list if c.get("groupId") == gid]
                        if affected:
                            for c in sorted(affected, key=lambda x: (str(x.get('name') or ""), str(x.get('id') or ""))):
                                print(f"  - Category: {c.get('name') or '(unknown category)'} ({c.get('id')})")
                        else:
                            print("  (no categories found under this group)")
                groups_by_id = {g.get("id"): g for g in groups_list if g.get("id")}
                cats_by_id = {c.get("id"): c for c in cats_list if c.get("id")}

                # Structure to collect excluded rules under group->category
                # excluded_index = {group_id: {"group": {...}, "categories": {cat_id: {"cat": {...}, "rules": [rule, ...]}}}}
                excluded_index: Dict[str, Dict[str, Any]] = {}

                before = len(rules)
                kept = []
                excluded = 0
                for r in rules:
                    cat_ids = _extract_rule_category_ids(r)
                    offending = [c for c in cat_ids if c in non_biz_cat_ids]
                    # Exclude if any category is in the non-business set
                    if offending:
                        excluded += 1
                        # Index under each offending category for reporting
                        for cid in offending:
                            cmeta = cats_by_id.get(cid, {})
                            gid = cmeta.get("groupId")
                            if not gid:
                                # If category metadata missing, we can't map to group; still record under a placeholder key
                                gid = "__unknown_group__"
                            # Ensure group bucket
                            if gid not in excluded_index:
                                excluded_index[gid] = {
                                    "group": groups_by_id.get(gid, {"id": gid, "name": "(unknown group)", "businessRelevant": False}),
                                    "categories": {}
                                }
                            # Ensure category bucket
                            cat_bucket = excluded_index[gid]["categories"].setdefault(
                                cid,
                                {"cat": cmeta if cmeta else {"id": cid, "name": "(unknown category)", "groupId": gid}, "rules": []}
                            )
                            cat_bucket["rules"].append(r)
                        continue
                    kept.append(r)
                rules = kept
                after = len(rules)
                print(f"[info] Business relevance filter applied: excluded={excluded}, before={before}, after={after}")
                # Show a small sample of excluded category IDs
                sample = list(sorted(non_biz_cat_ids))[:10]
                print(f"[info] Non-business categories (sample): {', '.join(sample)}")

                # Detailed breakdown: groups -> categories -> rules excluded
                if excluded_index:
                    print("[info] Non-business relevant breakdown:")
                    # Sort groups by name/id for stable output
                    for gid in sorted(excluded_index.keys(), key=lambda x: (str(excluded_index[x]["group"].get("name") or ""), str(x))):
                        g = excluded_index[gid]["group"]
                        g_name = g.get("name") or "(unknown group)"
                        print(f"- Group: {g_name} ({gid}) businessRelevant=false")
                        cats = excluded_index[gid]["categories"]
                        for cid in sorted(cats.keys(), key=lambda c: (str(cats[c]["cat"].get("name") or ""), str(c))):
                            c = cats[cid]["cat"]
                            c_name = c.get("name") or "(unknown category)"
                            print(f"  - Category: {c_name} ({cid})")
                            rules_list = cats[cid]["rules"]
                            # stable order by rule_name then id
                            def _rule_key(rr: Dict[str, Any]):
                                return (str(rr.get('rule_name') or ""), str(extract_rule_id(rr) or ""))
                            for rr in sorted(rules_list, key=_rule_key):
                                rid = extract_rule_id(rr) or "(no-id)"
                                rname = rr.get("rule_name") or "(unnamed rule)"
                                rfile = rr.get("code_file") or ""
                                cat_label = f"{c_name} ({cid})"
                                if rfile:
                                    print(f"    - Rule: {rname} [{rid}] — {rfile} — Category: {cat_label}")
                                else:
                                    print(f"    - Rule: {rname} [{rid}] — Category: {cat_label}")
            else:
                print("[info] rule_categories.json present but no non-business groups detected; no rules filtered")
        else:
            print(f"[info] No rule_categories.json next to business_rules.json ({rc_path}); proceeding without business relevance filter")
    # --------------------------------------------------------

    # Write out rule ids before any run-based filtering (for manual comparison)
    if TRACE_ENABLED:
        _tprint("[trace] rules file inventory (all rule_ids before run filter):")
        for r in rules:
            rid = extract_rule_id(r)
            cats_formatted = _format_rule_categories(r)
            cats_str = f" cats=[{', '.join(cats_formatted)}]" if cats_formatted else " cats=[]"
            _tprint(f"[trace]   rule_id={rid} name={r.get('rule_name')} file={r.get('code_file')}{cats_str}")

    # Keep a copy for tracing
    original_rules = rules[:]

    # Optional run-based filtering by root path
    if args.root_path:
        runs_file = Path(args.runs_file).expanduser()
        runs = load_runs(runs_file)
        # Diagnostics: how many runs and what roots exist
        print(f"[info] Scanned {len(runs)} runs from {runs_file}")
        distinct_roots = []
        seen_roots = set()
        for r in runs:
            rd = r.get("root_dir") or r.get("root_path")
            if not rd:
                continue
            norm_rd = normalize_fs_path(str(rd)).lower()
            if norm_rd not in seen_roots:
                seen_roots.add(norm_rd)
                distinct_roots.append(rd)
        if distinct_roots:
            sample_roots = ", ".join(distinct_roots[:3])
            more = "" if len(distinct_roots) <= 3 else f" (+{len(distinct_roots)-3} more)"
            print(f"[info] runs.json distinct roots (sample): {sample_roots}{more}")
        allowed_ids = collect_rule_ids_for_root(runs, args.root_path)
        # Show normalized root target
        print(f"[info] Root-path filter target: {normalize_fs_path(args.root_path)}")
        if allowed_ids:
            before = len(rules)
            filtered_rules = []
            for r in rules:
                rid = extract_rule_id(r)
                if rid and rid in allowed_ids:
                    _tprint(f"[trace] keep (Stage A: root match) rule_id={rid}")
                    filtered_rules.append(r)
                else:
                    why = "no rule_id" if not rid else "rule_id not in runs for root"
                    _tprint(f"[trace] reject (Stage A) rule_id={rid} — {why}")
                    _count(f"StageA:{why}")
            rules = filtered_rules
            print(f"[info] Filtered by root-path: {args.root_path}")
            print(f"[info] runs.json: {runs_file}")
            print(f"[info] Allowed rule_ids: {len(allowed_ids)}; rules kept: {len(rules)} (from {before})")
            # Print a small sample of allowed ids to help troubleshoot
            sample_ids = ", ".join(list(sorted(allowed_ids))[:10])
            print(f"[info] Sample of allowed rule_ids: {sample_ids if sample_ids else '(none)'}")
        else:
            # No matching allowed IDs found for the provided root
            msg = (
                f"[warn] No matching runs/rule_ids found for root-path: {args.root_path}. "
                "If you expected matches, check that runs.json has rule_ids for the desired root and that the paths match after realpath/expanduser."
            )
            print(msg)
            # Strict mode is the default when --root-path is provided, unless --no-strict-root is passed
            strict_mode = not args.no_strict_root
            if strict_mode:
                print("[info] Strict root (default) active: producing an empty result due to no matches. Use --no-strict-root to disable.")
                rules = []
            else:
                print("[info] Proceeding without root filter (--no-strict-root was specified).")
    else:
        original_rules = rules[:]

    grouped = group_rules_by_file(rules, source_prefix, args.case_insensitive, args.include_archived)
    print(f"[info] Found {len(grouped)} files under {source_prefix}")
    total_rules = sum(len(rules) for rules in grouped.values())
    print(f"[info] Exported {total_rules} rules across {len(grouped)} files")
    # Show a small sample of files and their rule counts to aid troubleshooting
    if grouped:
        preview_items = []
        for i, (cf, items) in enumerate(grouped.items()):
            if i >= 5:
                break
            preview_items.append(f"{cf} ({len(items)} rules)")
        print(f"[info] Files preview: {', '.join(preview_items)}")

    if args.format == "plain":
        text = format_plain(grouped)
    elif args.format == "md":
        text = format_md(grouped)
    else:
        text = format_json(grouped)

    if TRACE_ENABLED:
        # Print a compact reasons tally to help diagnose why rules were dropped
        if REASONS_COUNT:
            print("[trace] rejection summary:")
            for reason, cnt in sorted(REASONS_COUNT.items(), key=lambda x: (-x[1], x[0])):
                print(f"[trace]   {reason}: {cnt}")
        else:
            print("[trace] no rejections recorded (after filters)")

    if args.output:
        print(f"[info] Writing output to {args.output}")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
    else:
        print(f"[info] Printing output to stdout")
        print(text)

if __name__ == "__main__":
    main()