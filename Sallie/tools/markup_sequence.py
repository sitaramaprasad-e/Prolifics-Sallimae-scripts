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


# ----------------------------
# PCPT helpers (inspired by categorize_rules.py)
# ----------------------------
def pcpt_sequence(output_dir: str, visualize: str, domain_hints: Optional[str] = None) -> None:
  """
  Run: pcpt.sh sequence --output <output_dir> [--domain-hints <domain_hints>] --visualize <visualize>
  """
  cmd = ["pcpt.sh", "sequence", "--output", output_dir]
  if domain_hints:
    cmd.extend(["--domain-hints", domain_hints])
  cmd.extend(["--visualize", visualize])
  _run(cmd)

def pcpt_run_custom_prompt(
    input_file: str,
    input_file2: str,
    output_dir: str,
    code_dir: str,
    prompt_name: str,
) -> None:
  """
  Run: pcpt.sh run-custom-prompt --input-file <input_file> --input-file2 <input_file2> --output <output_dir> <code_dir> <prompt_name>
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
    code_dir,
    prompt_name,
  ]
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
  parser.add_argument("--skip-initial-sequence", action="store_true", help="Skip the initial sequence generation step.")
  args = parser.parse_args()

  code_dir = args.code_dir
  output_dir = args.output_dir
  domain_hints = args.domain_hints if args.domain_hints and args.domain_hints.strip() else None
  prompt_name = args.prompt_name

  # Paths used by the flow
  seq_report_src = os.path.join(output_dir, "sequence_report", "sequence_report.txt")
  markup_md = os.path.join(output_dir, MARKUP_OUT_DIRNAME, MARKUP_OUT_FILENAME)

  _log("Step 1: Create standard Sequence Diagram", header=True)
  if not args.skip_initial_sequence:
    if domain_hints:
      pcpt_sequence(output_dir=output_dir, visualize=code_dir, domain_hints=domain_hints)
    else:
      pcpt_sequence(output_dir=output_dir, visualize=code_dir)
  else:
    _log("Skipped initial sequence generation as requested.")

  _log("Step 2: Copy existing sequence report to temp folder", header=True)
  _ensure_dir(TMP_ROOT)
  _copy(seq_report_src, TMP_SEQUENCE_TXT)

  _log("Step 3: Export list of current business rules for code", header=True)
  # Compute root path of current run (absolute, resolved)
  root_path = os.path.realpath(os.path.abspath(os.getcwd()))
  _log(f"Detected root path: {root_path}")

  export_cmd = [
      sys.executable,
      "tools/export_rules_for_markup.py",
      code_dir,
      "--root-path",
      root_path,
      "--trace",
      "--trace-limit",
      "500",
  ]
  _run(export_cmd)

  # Sanity check for the exported rules file (some environments might write elsewhere)
  if not os.path.exists(TMP_EXPORTED_RULES):
    raise FileNotFoundError(
      f"Expected exported rules at {TMP_EXPORTED_RULES} not found. "
      f"Ensure tools/export_rules_for_markup.py writes to that path."
    )

  _log("Step 4: Markup sequence description", header=True)
  pcpt_run_custom_prompt(
    input_file=TMP_SEQUENCE_TXT,
    input_file2=TMP_EXPORTED_RULES,
    output_dir=output_dir,
    code_dir=code_dir,
    prompt_name=prompt_name,
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