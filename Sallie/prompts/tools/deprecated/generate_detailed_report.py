#!/usr/bin/env python3
# save as generate_rules_report.py

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
# add near the top with other imports
import os

# add right below other module-level helpers / constants
DEFAULT_JSON_PATH = os.path.expanduser("~/.model/business_rules.json")

# ---------- markdown helpers ----------

def md_escape(text: Optional[str]) -> str:
    if text is None:
        return "—"
    # Escape table pipes and backticks minimally
    return str(text).replace("|", r"\|").replace("`", r"\`")

def code_fence(lang: str, content: Optional[str]) -> str:
    content = "" if content is None else content.rstrip()
    return f"```{lang}\n{content}\n```" if content else f"```{lang}\n```\n"

def heading(level: int, text: str, anchor: Optional[str] = None) -> str:
    h = "#" * level + " " + text
    return h if not anchor else f"{h} <a id='{anchor}'></a>"

def bullet(label: str, value: Any) -> str:
    v = "—" if value in (None, "", []) else value
    return f"- **{label}:** {v}"

def lines_range(lines: Optional[List[int]]) -> str:
    if not lines:
        return "—"
    if len(lines) == 1:
        return f"Line {lines[0]}"
    return f"Lines {min(lines)}–{max(lines)}"

def anchorize(text: str) -> str:
    # simple, stable anchor
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")

# ---------- domain helpers ----------

CATEGORY_ORDER = {
    "Validation": 1,
    "Referential Integrity": 2,
    "Transformation": 3,
    "Workflow": 4,
}

def category_rank(cat: Optional[str]) -> int:
    if not cat:
        return 999
    return CATEGORY_ORDER.get(cat, 500)

def summarize_counts(rules: List[Dict[str, Any]]) -> Dict[str, Counter]:
    by_cat = Counter((r.get("rule_category") or "Uncategorized") for r in rules)
    by_area = Counter((r.get("business_area") or "—") for r in rules)
    return {"category": by_cat, "business_area": by_area}

def render_counts_table(counter: Counter, header_left: str) -> str:
    rows = [f"| {header_left} | Count |",
            "|---|---|"]
    for key, cnt in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        rows.append(f"| {md_escape(key)} | {cnt} |")
    return "\n".join(rows)

def render_kv_block(kvs: List[tuple]) -> str:
    return "\n".join(bullet(k, v) for k, v in kvs)

def render_dmn_ios(title: str, ios: Optional[List[Dict[str, Any]]]) -> str:
    if not ios:
        return f"**{title}:** —"
    rows = ["| Name | Type |", "|---|---|"]
    for io in ios:
        rows.append(f"| {md_escape(io.get('name'))} | {md_escape(io.get('type'))} |")
    return "\n".join(rows)

def render_rule_section(idx: int, r: Dict[str, Any]) -> str:
    name = r.get("rule_name") or f"Rule {idx}"
    anchor = anchorize(f"{idx}-{name}")
    parts = []

    parts.append(heading(2, f"{idx}. {name}", anchor))
    parts.append("")
    # Quick facts
    parts.append(render_kv_block([
        ("Category", r.get("rule_category") or "—"),
        ("Business Area", r.get("business_area") or "—"),
        ("Owner", r.get("owner") or "—"),
        ("Doc Rule ID", r.get("doc_rule_id") if r.get("doc_rule_id") is not None else "—"),
        ("ID", r.get("id") or "—"),
        ("Timestamp", r.get("timestamp") or "—"),
    ]))
    parts.append("")

    # Purpose & Spec
    if r.get("rule_purpose"):
        parts.append("**Purpose**")
        parts.append(md_escape(r.get("rule_purpose")))
        parts.append("")
    if r.get("rule_spec"):
        parts.append("**Specification**")
        parts.append(md_escape(r.get("rule_spec")))
        parts.append("")

    # Example
    if r.get("example"):
        parts.append("**Example**")
        parts.append(md_escape(r.get("example")))
        parts.append("")

    # Source code reference
    parts.append("**Source**")
    parts.append(render_kv_block([
        ("File", r.get("code_file") or "—"),
        ("Lines", lines_range(r.get("code_lines")))
    ]))
    parts.append("")
    # Code block
    cb = r.get("code_block")
    if cb:
        # heuristic: detect sql-like or default to text
        lang = "sql" if any(kw in cb.upper() for kw in ("SELECT", "UPDATE", "CASE", "FROM", "WHERE", "JOIN", "SET")) else ""
        parts.append(code_fence(lang, cb))
        parts.append("")

    # DMN section
    has_dmn = any(r.get(k) for k in ("dmn_hit_policy", "dmn_inputs", "dmn_outputs", "dmn_table", "dmn_expression"))
    if has_dmn:
        parts.append("**DMN:**")
        parts.append("")
        if r.get("dmn_hit_policy"):
            parts.append(f"Hit Policy: {r.get('dmn_hit_policy')}")
            parts.append("")
        # Inputs
        if r.get("dmn_inputs"):
            parts.append("Inputs:")
            for i in r["dmn_inputs"]:
                parts.append(f"- `{i.get('name')}:{i.get('type')}`")
            parts.append("")
        else:
            parts.append("Inputs: —")
            parts.append("")
        # Outputs
        if r.get("dmn_outputs"):
            parts.append("Outputs:")
            for o in r["dmn_outputs"]:
                parts.append(f"- `{o.get('name')}:{o.get('type')}`")
            parts.append("")
        else:
            parts.append("Outputs: —")
            parts.append("")
        # DMN Table
        if r.get("dmn_table"):
            parts.append(r.get("dmn_table").strip())
            parts.append("")
        elif r.get("dmn_expression"):
            parts.append("Expression:")
            parts.append(code_fence("feel", r.get("dmn_expression")))
            parts.append("")

    # Categorization explanation
    if r.get("category_explanation"):
        parts.append("<details><summary><strong>Why this category?</strong></summary>\n\n" +
                     md_escape(r.get("category_explanation")) + "\n\n</details>")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"

def render_toc(rules: List[Dict[str, Any]]) -> str:
    items = []
    for i, r in enumerate(rules, 1):
        name = r.get("rule_name") or f"Rule {i}"
        anchor = anchorize(f"{i}-{name}")
        items.append(f"- [{i}. {name}](#{anchor})")
    return "\n".join(items)

def generate_report(data: List[Dict[str, Any]], title: str) -> str:
    # Sort by category rank then rule_name
    rules = sorted(
        data,
        key=lambda r: (category_rank(r.get("rule_category")), (r.get("rule_name") or "").lower())
    )

    counts = summarize_counts(rules)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    out = []
    out.append(heading(1, title))
    out.append("")
    out.append(f"_Generated: {now}_")
    out.append("")
    out.append(f"**Total rules:** {len(rules)}")
    out.append("")
    out.append("### Summary by Category")
    out.append(render_counts_table(counts["category"], "Category"))
    out.append("")
    out.append("### Summary by Business Area")
    out.append(render_counts_table(counts["business_area"], "Business Area"))
    out.append("")
    out.append("### Table of Contents")
    out.append(render_toc(rules))
    out.append("\n---\n")

    for i, r in enumerate(rules, 1):
        out.append(render_rule_section(i, r))

    return "\n".join(out).rstrip() + "\n"

# ---------- cli ----------

def load_json_any(source: str) -> List[Dict[str, Any]]:
    """
    Attempt to parse source as: (1) path to JSON file, else (2) raw JSON string.
    """
    try:
        # Try as file path
        with open(source, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Try as raw JSON content
        return json.loads(source)

def prompt_with_default(prompt_text: str, default_value: Optional[str]) -> str:
    dv = "" if default_value is None else str(default_value)
    entered = input(f"{prompt_text} [{dv}]: ").strip()
    return entered if entered else dv

def read_json_from_user_choice() -> List[Dict[str, Any]]:
    """
    Prompt user to either provide a file path or paste raw JSON.
    Defaults to DEFAULT_JSON_PATH if path is blank and file exists.
    """
    while True:
        choice = input("Input mode: (F)ile path or (P)aste raw JSON? [F]: ").strip().lower()
        if choice not in ("f", "p", ""):
            print("Please enter F or P.")
            continue

        if choice in ("", "f"):
            path = prompt_with_default("Path to JSON file (or leave blank for default)", DEFAULT_JSON_PATH)
            if not path:
                print("A file path is required when using File mode.")
                continue
            expanded = os.path.expanduser(path)
            if not os.path.exists(expanded):
                print(f"File not found: {expanded}")
                continue
            try:
                return load_json_any(expanded)
            except Exception as e:
                print(f"Failed to load JSON from file: {e}")
                continue

        # Paste mode
        print("Paste JSON now. Press Ctrl-D (Linux/macOS) or Ctrl-Z then Enter (Windows) to finish.")
        try:
            raw = sys.stdin.read() if not sys.stdin.isatty() else sys.stdin.read()
        except Exception:
            raw = ""
        if not raw.strip():
            print("No JSON received. Try again.")
            continue
        try:
            data = json.loads(raw)
            return data
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
            print("Let's try again.")
            continue

def main():
    print("=== Business Rules → Markdown Report ===")
    # Title
    title = prompt_with_default("Report title", "Business Rules Report")

    # Input data (file or pasted JSON)
    data = read_json_from_user_choice()
    if not isinstance(data, list):
        raise SystemExit("Input JSON must be a list of rule objects.")

    # Output path (defaults to rules_report.md)
    out_path = prompt_with_default("Output .md path (empty to print to stdout)", "rules_report.md")

    report = generate_report(data, title)

    if out_path:
        out_path_expanded = os.path.expanduser(out_path)
        # Ensure parent dir exists if a directory was provided
        parent = os.path.dirname(out_path_expanded)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)
        with open(out_path_expanded, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote {out_path_expanded}")
    else:
        sys.stdout.write(report)

if __name__ == "__main__":
    main()