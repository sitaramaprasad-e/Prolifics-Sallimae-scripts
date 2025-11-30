#!/usr/bin/env python3
"""
Export logics for files under a given source path from business_rules.json.

Examples
--------
# Basic (prints to stdout, plain text like your sample)
python export_logics_for_path.py code/sf

# Specify a different logics file and write Markdown to a file
python export_logics_for_path.py code/sf \
  --logics-file /path/to/business_rules.json \
  --format md \
  --output logics_in_sf.md
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
# Collect a small sample of path mismatches for clear diagnostics
MISMATCH_SAMPLES: List[Dict[str, str]] = []
MAX_MISMATCH_SAMPLES = 5

# ---- Logic ID helpers --------------------------------------------------------
from typing import Optional

def _strip_brackets(s: str) -> str:
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("(") and s.endswith(")")):
        return s[1:-1]
    return s

def normalize_logic_id(v: Any) -> Optional[str]:
    """Return a normalized string id or None if unusable (trim, strip braces, lowercase)."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s = _strip_brackets(s)
    return s.lower()

def extract_logic_id(logic: Dict[str, Any]) -> Optional[str]:
    """Extract a logic id, prioritizing 'id' (business_rules.json), then common fallbacks."""
    for key in ("id",):
        if key in logic:
            rid = normalize_logic_id(logic.get(key))
            if rid:
                return rid
    meta = logic.get("meta") or logic.get("metadata")
    if isinstance(meta, dict):
        for key in ("id",):
            if key in meta:
                rid = normalize_logic_id(meta.get(key))
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
DEFAULT_OUTPUT = "./.tmp/logics-for-markup/exported-logics.json"
DEFAULT_LOGICS_FILE = "~/.model/business_rules.json"

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

def load_logics(logics_file: Path) -> List[Dict[str, Any]]:
    """Load logics from business_rules.json.

    Supports both legacy flat-list structure and the newer wrapper structure:
      - [ ... ]
      - { "version": N, "logics": [ ... ] }
      - { "version": N, "rules": [ ... ] }  # legacy key
    """
    try:
        with logics_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: rules file not found: {logics_file}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: failed to parse JSON ({logics_file}): {e}", file=sys.stderr)
        sys.exit(1)

    # Legacy: bare list of logics
    if isinstance(data, list):
        return data

    # Newer: wrapper object with version + logics/rules
    if isinstance(data, dict):
        logics = data.get("logics")
        if isinstance(logics, list):
            return logics
        # Backwards compatibility with any older "rules" key
        rules = data.get("rules")
        if isinstance(rules, list):
            return rules
        print(
            f"ERROR: {logics_file} has dict structure but no 'logics' or 'rules' list; "
            "expected keys 'logics' or 'rules' to contain an array of logics.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"ERROR: {logics_file} has unexpected top-level type {type(data).__name__}; "
        "expected a list or an object with 'logics'/'rules'.",
        file=sys.stderr,
    )
    sys.exit(1)


def dedupe_in_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for s in items:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out

def group_logics_by_file(
    logics: List[Dict[str, Any]],
    source_prefix: str,
    case_insensitive: bool,
    include_archived: bool = False,
) -> "OrderedDict[str, List[Dict[str, str]]]":
    """
    Group logics by code_file for files under source_prefix.
    Maintain first-seen order of files and logics.
    Each logic record includes name, code_file, and code_function.
    """
    grouped: "OrderedDict[str, List[Dict[str, str]]]" = OrderedDict()
    considered_total = 0
    kept_total = 0
    for r in logics:
        considered_total += 1
        name = r.get("name") or "(unnamed logic)"
        code_file = r.get("code_file")
        code_function = r.get("code_function")

        # Archived filter
        if not include_archived and r.get("archived") is True:
            _tprint(
                f"[trace] SKIP: logic='{name}' file='{code_file or '(no code_file)'}' "
                f"archived (include_archived=False)"
            )
            _count("archived_excluded")
            continue

        # Basic required fields
        if not code_file or not name:
            missing = "code_file" if not code_file else "name"
            _tprint(
                f"[trace] SKIP: logic='{name}' file='{code_file or '(no code_file)'}' "
                f"missing {missing}"
            )
            _count("missing_field")
            continue

        cf_norm = normalize_posix(code_file)
        if path_is_under(source_prefix, cf_norm, case_insensitive):
            kept_total += 1
            _tprint(
                f"[trace] MATCH: logic='{name}' file='{cf_norm}' "
                f"under source_prefix '{source_prefix}'"
            )
            if cf_norm not in grouped:
                grouped[cf_norm] = []
            # Build a display name that is just the original logic name; we add short codes later.
            rid = extract_logic_id(r)
            code_suffix = None
            if rid and len(rid) >= 3:
                code_suffix = rid[-3:].upper()
            # Keep the exported name as the original logic name; we add short codes later.
            display_name = name
            grouped[cf_norm].append({
                "id": rid,
                "code": code_suffix,
                "name": display_name,
                "name_base": name,
                "code_function": code_function,
                "archived": r.get("archived"),
            })
            _count("matched")
        else:
            # Record up to MAX_MISMATCH_SAMPLES clear examples for troubleshooting
            if len(MISMATCH_SAMPLES) < MAX_MISMATCH_SAMPLES:
                MISMATCH_SAMPLES.append({
                    "rule_id": str(extract_logic_id(r) or "(no-id)"),
                    "name": str(name or "(unnamed)"),
                    "code_file": str(cf_norm),
                    "original_code_file": str(code_file),
                })
            _tprint(
                f"[trace] SKIP: logic='{name}' file='{cf_norm}' "
                f"not under source_prefix '{source_prefix}'"
            )
            _count("not_under_source_prefix")
    if TRACE_ENABLED:
        matched = REASONS_COUNT.get("matched", 0)
        skipped_archived = REASONS_COUNT.get("archived_excluded", 0)
        skipped_missing = REASONS_COUNT.get("missing_field", 0)
        skipped_path = REASONS_COUNT.get("not_under_source_prefix", 0)
        print(
            "[trace] selection summary: "
            f"considered={considered_total}, matched={matched}, "
            f"skipped_archived={skipped_archived}, "
            f"skipped_missing={skipped_missing}, "
            f"skipped_path_mismatch={skipped_path}"
        )

    # de-duplicate logic records per file by name, preserving order,
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
        index_by_logic = {}
        deduped = []
        for item in grouped[cf]:
            base_name = item.get("name_base") or item["name"]
            is_arch = _is_archived_flag(item.get("archived"))
            if base_name not in index_by_logic:
                index_by_logic[base_name] = len(deduped)
                deduped.append(item)
            else:
                idx = index_by_logic[base_name]
                existing = deduped[idx]
                existing_arch = _is_archived_flag(existing.get("archived"))
                # If the existing is archived and the new one is not, replace.
                if existing_arch and not is_arch:
                    _tprint(
                        f"[trace] prefer non-archived over archived for name='{base_name}' in file={cf}"
                    )
                    deduped[idx] = item
                # Otherwise keep the first (non-archived already wins or both archived)
        grouped[cf] = deduped

    # Strip internal-only fields from output (do not expose internal flags in the result)
    for cf in grouped:
        for item in grouped[cf]:
            if "archived" in item:
                del item["archived"]
            if "name_base" in item:
                del item["name_base"]

    return grouped

def format_plain(grouped: "OrderedDict[str, List[Dict[str, str]]]") -> str:
    lines: List[str] = []
    for code_file, logic_items in grouped.items():
        lines.append(f"{code_file}:")
        for item in logic_items:
            rn = item["name"]
            cf = item.get("code_function") or ""
            lines.append(f"{rn} ({cf})" if cf else rn)
        lines.append("")  # blank line between files
    # Remove trailing blank if present
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)

def format_md(grouped: "OrderedDict[str, List[Dict[str, str]]]") -> str:
    lines: List[str] = []
    for code_file, logic_items in grouped.items():
        lines.append(f"### `{code_file}`")
        for item in logic_items:
            rn = item["name"]
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

def load_categories(rc_path: Path) -> Dict[str, Any]:
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
    # New canonical names: categoryGroups / categories.
    # Fall back to ruleCategoryGroups / ruleCategories for backwards compatibility.
    groups = (rc.get("categoryGroups") or rc.get("ruleCategoryGroups") or []) or []
    cat_list = (rc.get("categories") or rc.get("ruleCategories") or []) or []
    # businessRelevant missing => treated as True
    group_biz = {g.get("id"): (g.get("businessRelevant") is not False) for g in groups if g.get("id")}
    non_biz_group_ids = {gid for gid, is_biz in group_biz.items() if is_biz is False}
    non_biz_cat_ids = {c.get("id") for c in cat_list if c.get("groupId") in non_biz_group_ids and c.get("id")}
    return non_biz_cat_ids

def extract_categories(logic: Dict[str, Any]) -> List[str]:
    # NOTE: This function only returns raw category IDs embedded on the logic and may not resolve names.
    # Prefer single id fields first
    cid = logic.get("categoryId") or logic.get("category_id")
    if cid:
        return [str(cid)]
    cats = logic.get("categories")
    if isinstance(cats, list):
        # normalize to strings
        return [str(x) for x in cats if x is not None]
    return []

def main():
    parser = argparse.ArgumentParser(
        description="Export logics grouped by code_file for a given source path."
    )
    parser.add_argument(
        "source_path",
        help="Source path prefix to filter files (e.g., 'code/sf', 'code/stored_proc')."
    )
    parser.add_argument(
        "--logics-file",
        default=DEFAULT_LOGICS_FILE,
        help=f"Path to business_rules.json (default: {DEFAULT_LOGICS_FILE}; '~' will be expanded)"
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
        "--include-archived",
        action="store_true",
        help="Include archived logics (by default they are skipped)."
    )
    parser.add_argument(
        "--include-all-logics",
        action="store_true",
        help="Include all logics regardless of business relevance (overrides rule_categories.json filtering)."
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        default=True,
        help="Enable verbose per-item tracing of why each logic was accepted or rejected (ON by default)."
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
    logics_file = Path(args.logics_file).expanduser()

    print("[info] ================= Export logics for markup: input summary =================")
    print(f"[info] Source path          : {source_prefix}")
    print(f"[info] Logics file           : {logics_file}")
    print(f"[info] Output file          : {args.output or '(stdout)'}")
    print(f"[info] Format               : {args.format}")
    print(f"[info] Include archived     : {args.include_archived}")
    print(f"[info] Include all logics    : {args.include_all_logics}  (skip business relevance filter if True)")
    print(f"[info] Trace                : {'ON' if TRACE_ENABLED else 'OFF'} (limit={TRACE_REMAINING})")
    print("[info] ========================================================================")

    logics = load_logics(logics_file)
    print(
        f"[info] Loaded {len(logics)} logics from {logics_file} "
        f"(before business relevance, archived, and source-path filters)"
    )

    # Prepare category lookups (for printing names) even if we don't filter
    rc_path = (Path(args.logics_file).expanduser().parent) / "rule_categories.json"
    rc_lookup: Dict[str, Any] = {}
    rc_groups_by_id: Dict[str, Dict[str, Any]] = {}
    rc_cats_by_id: Dict[str, Dict[str, Any]] = {}
    if rc_path.exists():
        rc_lookup = load_categories(rc_path)
        # New canonical names: categoryGroups / categories. Fall back to
        # ruleCategoryGroups / ruleCategories for older files.
        _groups = (rc_lookup.get("categoryGroups") or rc_lookup.get("ruleCategoryGroups") or []) or []
        _cats = (rc_lookup.get("categories") or rc_lookup.get("ruleCategories") or []) or []
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

    def _extract_category_ids(logic: Dict[str, Any]) -> List[str]:
        """
        Returns normalized category IDs for a logic.
        Order of precedence:
          1) ID fields: categoryId, category_id
          2) ID list: categories (list of ids)
          3) Name fields: category, categoryName (single string)
          4) Name list: category_names (list of names)
        Name-based fields are resolved via rule_categories.json (rc_cats_by_name).
        """
        out: List[str] = []
        # ID fields first
        cid = logic.get("categoryId") or logic.get("category_id")
        if cid:
            out.append(str(cid))
        cats = logic.get("categories")
        if isinstance(cats, list):
            out.extend(str(x) for x in cats if x is not None)
        # Name fields
        name_single = logic.get("category") or logic.get("category") or logic.get("categoryName")
        if isinstance(name_single, str) and name_single.strip():
            nid = rc_cats_by_name.get(name_single.strip().lower())
            if nid:
                out.append(nid)
        name_list = logic.get("category_names")
        if isinstance(name_list, list):
            for nm in name_list:
                if isinstance(nm, str) and nm.strip():
                    nid = rc_cats_by_name.get(nm.strip().lower())
                    if nid:
                        out.append(nid)
        # De-duplicate preserving order
        return dedupe_in_order(out)

    # --- Business relevance filter (rule_categories.json) ---
    if args.include_all_logics:
        print("[info] --include-all-logics supplied: skipping business relevance filter")
    else:
        # rc_path already computed above
        if rc_path.exists():
            rc = load_categories(rc_path)
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

                # Structure to collect excluded logics under group->category
                # excluded_index = {group_id: {"group": {...}, "categories": {cat_id: {"cat": {...}, "logics": [logic, ...]}}}}
                excluded_index: Dict[str, Dict[str, Any]] = {}

                before = len(logics)
                kept = []
                excluded = 0
                for r in logics:
                    cat_ids = _extract_category_ids(r)
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
                                {"cat": cmeta if cmeta else {"id": cid, "name": "(unknown category)", "groupId": gid}, "logics": []}
                            )
                            cat_bucket["logics"].append(r)
                        continue
                    kept.append(r)
                logics = kept
                after = len(logics)
                print(f"[info] Business relevance filter applied: excluded={excluded}, before={before}, after={after}")
                # Show a small sample of excluded category IDs
                sample = list(sorted(non_biz_cat_ids))[:10]
                print(f"[info] Non-business categories (sample): {', '.join(sample)}")

                # Detailed breakdown: groups -> categories -> logics excluded
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
                            categories_list = cats[cid]["logics"]
                            # stable order by name then id
                            def _logic_key(rr: Dict[str, Any]):
                                return (str(rr.get('name') or ""), str(extract_logic_id(rr) or ""))
                            for rr in sorted(categories_list, key=_logic_key):
                                rid = extract_logic_id(rr) or "(no-id)"
                                rname = rr.get("name") or "(unnamed logic)"
                                rfile = rr.get("code_file") or ""
                                cat_label = f"{c_name} ({cid})"
                                if rfile:
                                    print(f"    - Logic: {rname} [{rid}] — {rfile} — Category: {cat_label}")
                                else:
                                    print(f"    - Logic: {rname} [{rid}] — Category: {cat_label}")
            else:
                print("[info] rule_categories.json present but no non-business groups detected; no logics filtered")
        else:
            print(f"[info] No rule_categories.json next to business_rules.json ({rc_path}); proceeding without business relevance filter")
    # --------------------------------------------------------

    grouped = group_logics_by_file(logics, source_prefix, False, args.include_archived)
    print(f"[info] Found {len(grouped)} files under {source_prefix}")
    total_logics = sum(len(logics) for logics in grouped.values())
    print(f"[info] Exported {total_logics} logics across {len(grouped)} files")
    # Show a small sample of files and their logic counts to aid troubleshooting
    if grouped:
        preview_items = []
        for i, (cf, items) in enumerate(grouped.items()):
            if i >= 5:
                break
            preview_items.append(f"{cf} ({len(items)} logics)")
        print(f"[info] Files preview: {', '.join(preview_items)}")

    if args.format == "plain":
        text = format_plain(grouped)
    elif args.format == "md":
        text = format_md(grouped)
    else:
        text = format_json(grouped)

    if TRACE_ENABLED:
        matched = REASONS_COUNT.get("matched", 0)
        skipped_archived = REASONS_COUNT.get("archived_excluded", 0)
        skipped_missing = REASONS_COUNT.get("missing_field", 0)
        skipped_path = REASONS_COUNT.get("not_under_source_prefix", 0)
        print("[trace] final selection summary:")
        print(f"[trace]   matched           : {matched}")
        print(f"[trace]   skipped_archived  : {skipped_archived}")
        print(f"[trace]   skipped_missing   : {skipped_missing}")
        print(f"[trace]   skipped_path_mismatch: {skipped_path}")
        if matched == 0 and (skipped_archived or skipped_missing or skipped_path):
            print("[trace]   note: no logics matched; consider broadening source_path or relaxing filters")

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