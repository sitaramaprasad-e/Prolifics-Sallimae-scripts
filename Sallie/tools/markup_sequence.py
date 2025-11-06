#!/usr/bin/env python3
"""
markup_sequence.py
-------------------------------------------------
Implements the end-to-end "sequence → markup → sequence" flow.

Steps (default values shown):
  1) Create standard Sequence Diagram
     pcpt.sh sequence --output docs/sf --domain-hints sf.hints --visualize code/sf

  2) Copy existing sequence report to temp folder
     mkdir -p .tmp/rules-for-markup
     cp docs/sf/sequence_report/sequence_report.txt .tmp/rules-for-markup/sequence_report.txt

  3) Export list of current business rules for code
     python tools/export_rules_for_markup.py code/sf

  4) Markup sequence description
     pcpt.sh run-custom-prompt --input-file .tmp/rules-for-markup/sequence_report.txt --input-file2 .tmp/rules-for-markup/exported-rules.json --output docs/sf code/sf markup-sequence.templ

  5) Regenerate sequence diagram
     pcpt.sh sequence --output docs/sf --visualize docs/sf/markup-sequence/markup-sequence.md

You can override defaults via CLI flags; run with -h for help.
"""
from __future__ import annotations


import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import List, Optional
import json
from typing import Any, Dict

# Resolve directory of this script (so we can locate sibling tools reliably)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# ----------------------------
# Constants / Defaults
# ----------------------------

DEFAULT_PROMPT_NAME = "markup-sequence.templ"

TMP_ROOT = ".tmp/rules-for-markup"
TMP_SEQUENCE_TXT = os.path.join(TMP_ROOT, "sequence_report.txt")
TMP_EXPORTED_RULES = os.path.join(TMP_ROOT, "exported-rules.json")
MARKUP_OUT_DIRNAME = "markup-sequence"
MARKUP_OUT_FILENAME = "markup-sequence.md"

LOG_DIR = os.path.expanduser("~/.pcpt/log/markup_sequence")


# ----------------------------
# Utilities
# ----------------------------
def _log(msg: str, header: bool = False) -> None:
  """Print timestamped log messages, with optional header formatting."""
  os.makedirs(LOG_DIR, exist_ok=True)
  ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
  if header:
    print(f"\n\033[95m[{ts}Z] ╔══════════════════════════════════════════════════════════╗\033[0m")
    print(f"\033[95m[{ts}Z] ║ {msg}\033[0m")
    print(f"\033[95m[{ts}Z] ╚══════════════════════════════════════════════════════════╝\033[0m")
  else:
    print(f"\033[90m[{ts}Z]\033[0m {msg}")

def _ensure_dir(path: str) -> None:
  os.makedirs(path, exist_ok=True)

def _run(cmd: List[str], cwd: Optional[str] = None, check: bool = True) -> None:
  """Run a command, show it in bold, and stream its output."""
  _log(f"$ \033[1m{' '.join(cmd)}\033[0m")
  try:
    process = subprocess.Popen(cmd, cwd=cwd)
    process.wait()
    if check and process.returncode != 0:
      raise subprocess.CalledProcessError(process.returncode, cmd)
  except subprocess.CalledProcessError as e:
    _log(f"❌ Command failed with exit code {e.returncode}: {' '.join(cmd)}", header=True)
    raise

def _copy(src: str, dst: str) -> None:
  _ensure_dir(os.path.dirname(dst))
  if not os.path.exists(src):
    raise FileNotFoundError(f"Expected file not found: {src}")
  shutil.copyfile(src, dst)
  _log(f"Copied: {src} -> {dst}")

def _load_json(path: str) -> Any:
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)

def _write_json(path: str, data: Any) -> None:
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

def _maybe_find_rule_categories(root_path: str, output_dir: str) -> Optional[str]:
  cand1 = os.path.join(root_path, "rule_categories.json")
  if os.path.exists(cand1):
    _log(f"Found rule_categories.json at: {cand1}")
    return cand1
  cand2 = os.path.join(output_dir, "rule_categories.json")
  if os.path.exists(cand2):
    _log(f"Found rule_categories.json at: {cand2}")
    return cand2
  _log("rule_categories.json not found in root or output directory.")
  return None


# ----------------------------
# PCPT helpers (inspired by categorize_rules.py)
# ----------------------------
def pcpt_sequence(output_dir: str, visualize: str, domain_hints: Optional[str] = None, filter_file: Optional[str] = None) -> None:
  """
  Run: pcpt.sh sequence --output <output_dir> [--domain-hints <domain_hints>] [--filter <filter_file>] --visualize <visualize>
  """
  cmd = ["pcpt.sh", "sequence", "--output", output_dir]
  if domain_hints:
    cmd.extend(["--domain-hints", domain_hints])
  if filter_file:
    cmd.extend(["--filter", filter_file])
  cmd.extend(["--visualize", visualize])
  _run(cmd)

def pcpt_run_custom_prompt(
    input_file: str,
    input_file2: str,
    output_dir: str,
    code_dir: str,
    prompt_name: str,
    filter_file: Optional[str] = None,
) -> None:
  """
  Run: pcpt.sh run-custom-prompt --input-file <input_file> --input-file2 <input_file2> --output <output_dir> [--filter <filter_file>] <code_dir> <prompt_name>
  """
  cmd = [
    "pcpt.sh",
    "run-custom-prompt",
    "--input-file",
    input_file,
    "--input-file2",
    input_file2,
    "--output",
    output_dir,
  ]
  if filter_file:
    cmd.extend(["--filter", filter_file])
  cmd.extend([
    code_dir,
    prompt_name,
  ])
  _run(cmd)


# ----------------------------
# Main flow
# ----------------------------
def main() -> None:
  parser = argparse.ArgumentParser(description="Generate, markup, and regenerate a sequence diagram.")
  parser.add_argument("code_dir", help="Path to code directory to visualize (required)")
  parser.add_argument("output_dir", help="Output directory for docs (required)")
  parser.add_argument("--domain-hints", default=None, help="Domain hints file (optional). If not provided, domain hints are not used.")
  parser.add_argument("--prompt-name", default=DEFAULT_PROMPT_NAME, help="Custom prompt template name (default: markup-sequence.templ)")
  parser.add_argument("--filter", default=None, help="Filter file (optional). If provided, it will be passed to PCPT.")
  parser.add_argument("--skip-initial-sequence", action="store_true", help="Skip the initial sequence generation step.")
  parser.add_argument(
      "--include-all-rules",
      action="store_true",
      help="Include all rules in markup. By default, rules whose category group has businessRelevant=false (from rule_categories.json) are excluded."
  )
  args = parser.parse_args()

  code_dir = args.code_dir
  output_dir = args.output_dir
  domain_hints = args.domain_hints if args.domain_hints and args.domain_hints.strip() else None
  prompt_name = args.prompt_name
  filter_file = args.filter if args.filter and args.filter.strip() else None

  # Paths used by the flow
  seq_report_src = os.path.join(output_dir, "sequence_report", "sequence_report.txt")
  markup_md = os.path.join(output_dir, MARKUP_OUT_DIRNAME, MARKUP_OUT_FILENAME)

  _log("Step 1: Create standard Sequence Diagram", header=True)
  if not args.skip_initial_sequence:
    if domain_hints:
      pcpt_sequence(output_dir=output_dir, visualize=code_dir, domain_hints=domain_hints, filter_file=filter_file)
    else:
      pcpt_sequence(output_dir=output_dir, visualize=code_dir, filter_file=filter_file)
  else:
    _log("Skipped initial sequence generation as requested.")

  _log("Step 2: Copy existing sequence report to temp folder", header=True)
  _ensure_dir(TMP_ROOT)
  _copy(seq_report_src, TMP_SEQUENCE_TXT)

  _log("Step 3: Export list of current business rules for code", header=True)
  # Compute root path of current run (absolute, resolved)
  root_path = os.path.realpath(os.path.abspath(os.getcwd()))
  _log(f"Detected root path: {root_path}")

  # Build absolute path to the sibling exporter script so this works from any CWD
  export_script = os.path.join(SCRIPT_DIR, "export_rules_for_markup.py")
  _log(f"Using export script: {export_script}")

  export_cmd = [
      sys.executable,
      export_script,
      code_dir,
      "--root-path",
      root_path,
      "--trace",
      "--trace-limit",
      "500",
  ]
  # pass-through switches to the export script
  if args.include_all_rules:
    export_cmd.append("--include-all-rules")
  _run(export_cmd)

  # Sanity check for the exported rules file (some environments might write elsewhere)
  if not os.path.exists(TMP_EXPORTED_RULES):
    raise FileNotFoundError(
      f"Expected exported rules at {TMP_EXPORTED_RULES} not found. "
      f"Ensure tools/export_rules_for_markup.py writes to that path."
    )

  _log("Applying business relevance filter (rule_categories.json)", header=True)
  if args.include_all_rules:
    _log("--include-all-rules supplied. Skipping business relevance filtering.")
  else:
    rc_path = _maybe_find_rule_categories(root_path, output_dir)
    if not rc_path:
      _log("No rule_categories.json found in root or output directory. Proceeding without filtering.")
    else:
      try:
        rc = _load_json(rc_path)
        groups = {g.get("id"): (g.get("businessRelevant") is not False) for g in rc.get("ruleCategoryGroups", [])}
        # groups[gid] == True means business relevant, False means NOT business relevant
        non_biz_group_ids = {gid for gid, is_biz in groups.items() if is_biz is False}
        non_biz_cat_ids = {c.get("id") for c in rc.get("ruleCategories", []) if c.get("groupId") in non_biz_group_ids}
        before_count = after_count = 0

        data = _load_json(TMP_EXPORTED_RULES)
        excluded = 0

        def rule_categories(rule: Dict[str, Any]):
          cat = rule.get("categoryId") or rule.get("category_id")
          if cat:
            return [cat]
          cats = rule.get("categories")
          if isinstance(cats, list):
            return cats
          return []

        def is_non_biz(rule: Dict[str, Any]) -> bool:
          cats = rule_categories(rule)
          return any(c in non_biz_cat_ids for c in cats)

        if isinstance(data, dict) and isinstance(data.get("rules"), list):
          rules = data["rules"]
          before_count = len(rules)
          kept = [r for r in rules if not is_non_biz(r)]
          after_count = len(kept)
          excluded = before_count - after_count
          data["rules"] = kept
          _write_json(TMP_EXPORTED_RULES, data)
        elif isinstance(data, list):
          before_count = len(data)
          kept = [r for r in data if not is_non_biz(r)]
          after_count = len(kept)
          excluded = before_count - after_count
          _write_json(TMP_EXPORTED_RULES, kept)
        else:
          _log("Unrecognized exported rules shape; skipping filtering.")

        if before_count:
          _log(f"Filtered non-business-relevant rules: excluded={excluded}, before={before_count}, after={after_count}")
          if non_biz_cat_ids:
            preview = list(non_biz_cat_ids)[:10]
            _log(f"Non-business category IDs (sample): {preview}")
      except Exception as e:
        _log(f"Failed to apply business relevance filter: {e}. Proceeding without filtering.")

  _log("Step 4: Markup sequence description", header=True)
  pcpt_run_custom_prompt(
    input_file=TMP_SEQUENCE_TXT,
    input_file2=TMP_EXPORTED_RULES,
    output_dir=output_dir,
    code_dir=code_dir,
    prompt_name=prompt_name,
    filter_file=filter_file,
  )

  _log("Step 5: Regenerate sequence diagram from markup", header=True)
  if not os.path.exists(markup_md):
    raise FileNotFoundError(
      f"Expected markup file not found: {markup_md}. "
      f"Verify the custom prompt produced it under {os.path.join(output_dir, MARKUP_OUT_DIRNAME)}"
    )
  # Only pass domain_hints if it was provided
  if domain_hints:
    pcpt_sequence(output_dir=output_dir, visualize=markup_md, domain_hints=domain_hints)
  else:
    pcpt_sequence(output_dir=output_dir, visualize=markup_md)

  _log("✅ Done. Sequence regenerated using markup.")
  _log(f"Markup file: {markup_md}")
  _log(f"Sequence report used: {TMP_SEQUENCE_TXT}")
  _log(f"Exported rules: {TMP_EXPORTED_RULES}")

if __name__ == "__main__":
  main()