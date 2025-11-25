#!/usr/bin/env python3
"""
markup_domain.py
-------------------------------------------------
Implements the end-to-end "domain → markup → domain" flow.
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
import re

# Resolve directory of this script (so we can locate sibling tools reliably)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# ----------------------------
# Spec constants
# ----------------------------
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
SPEC_DIR = os.path.join(SCRIPT_DIR, "spec")

# ----------------------------
# Constants / Defaults
# ----------------------------

DEFAULT_PROMPT_NAME = "markup-domain.templ"

TMP_ROOT = ".tmp/rules-for-markup"
TMP_DOMAIN_TXT = os.path.join(TMP_ROOT, "domain_model_report.txt")
TMP_EXPORTED_RULES = os.path.join(TMP_ROOT, "exported-rules.json")
MARKUP_OUT_DIRNAME = "markup-domain"
MARKUP_OUT_FILENAME = "markup-domain.md"

LOG_DIR = os.path.expanduser("~/.pcpt/log/markup_domain")


# ----------------------------
# Utilities
# ----------------------------

def _derive_short_code_from_uuid(uuid_str: str) -> str:
    """
    Derive the 3-hex short rule code from a UUID string by stripping dashes
    and taking the last 3 characters, uppercased.
    Example:
      'e8972032-1c87-4670-a9e0-2329a15cd7c0' -> '7C0'
    """
    cleaned = uuid_str.replace("-", "")
    if len(cleaned) < 3:
        return cleaned.upper()
    return cleaned[-3:].upper()


def _find_logicstep_id_for_rule(
    steps: List[Dict[str, Any]],
    short_code: str,
    rule_name: str,
) -> Optional[str]:
    """
    Given a 3-hex short rule code from the domain model and a rule name,
    find the corresponding LogicStep id by matching BOTH:
      - last 3 hex chars of the UUID-derived id, and
      - the rule name (case-insensitive exact match).
    Returns the LogicStep id (UUID) if found, else None.
    """
    target_code = (short_code or "").strip().upper()
    target_name = (rule_name or "").strip().lower()
    if not target_code or not target_name:
        return None

    for step in steps:
        step_id = str(step.get("id") or "").strip()
        name = str(step.get("name") or "").strip()
        if not step_id or not name:
            continue

        code = _derive_short_code_from_uuid(step_id)
        if code == target_code and name.lower() == target_name:
            return step_id

    return None

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


# ----------------------------
# PlantUML Rules Format Normalization
# ----------------------------
def _normalize_rules_format(path: str) -> None:
  """
  Normalize PlantUML rules/decisions style in note blocks to bullet format.
  """
  with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

  out_lines = []
  in_note = False
  in_rules = False
  for idx, line in enumerate(lines):
    stripped = line.strip()
    # Entering note block?
    if not in_note and stripped.startswith("note"):
      in_note = True
      in_rules = False
      out_lines.append(line)
      continue
    # Exiting note block?
    if in_note and stripped == "end note":
      in_note = False
      in_rules = False
      out_lines.append(line)
      continue
    # Are we at the start of a Rules section?
    if in_note and not in_rules and stripped.startswith("Rules"):
      in_rules = True
      out_lines.append(line)
      continue
    # If in a Rules section, process lines until end note or blank or non-rule
    if in_note and in_rules:
      # End rules section if we see a blank line or something that looks like a new section
      if not stripped or (not re.match(r'^([A-Za-z0-9]+\s*·)', stripped) and not stripped.startswith("Rules")):
        in_rules = False
        out_lines.append(line)
        continue
      # Try to match rule lines
      m = re.match(r'^(\s*)([A-Za-z0-9]+)\s*·\s*(.*?)(,)?\s*$', line)
      if m:
        indent, rule_id, rule_name, _ = m.groups()
        out_lines.append(f"{indent}- {rule_id}: {rule_name}\n")
        continue
      else:
        out_lines.append(line)
        continue
    # Default: just copy line
    out_lines.append(line)

  with open(path, "w", encoding="utf-8") as f:
    f.writelines(out_lines)



# ----------------------------
# Interactive Spec selection helpers
# ----------------------------
from typing import Any, Dict

def _prompt_choice(title: str, options: List[str], default_index: int = 1) -> int:
  """
  Print a numbered menu with title and options. Prompt user to choose 1-N [default].
  Returns a 1-based index of the chosen option.
  """
  if not options:
    raise ValueError(f"No options to choose from for: {title}")
  print(f"\n{title}")
  for i, opt in enumerate(options, 1):
    print(f"  {i}. {opt}")
  while True:
    try:
      raw = input(f"Choose 1-{len(options)} [{default_index}]: ").strip()
    except EOFError:
      raw = ""
    if not raw:
      return default_index
    if raw.isdigit():
      idx = int(raw)
      if 1 <= idx <= len(options):
        return idx
    print("Invalid selection, please try again.")


def _resolve_candidate_path(candidate: Optional[str], bases: List[str]) -> Optional[str]:
  """
  Try to resolve a possibly relative path string against a list of base directories.
  Returns a string path (absolute) if found, else falls back to first base.
  If candidate is None/empty -> returns None.
  """
  if not candidate:
    return None
  candidate = os.path.expanduser(candidate)
  if os.path.isabs(candidate) and os.path.exists(candidate):
    return os.path.realpath(candidate)
  for base in bases:
    p = os.path.realpath(os.path.join(base, candidate))
    if os.path.exists(p) or os.path.isdir(os.path.dirname(p)):
      return p
  # fallback: join to first base
  return os.path.realpath(os.path.join(bases[0], candidate))


def _select_spec_pair() -> Dict[str, Any]:
  """
  Interactive spec/pair selection assuming a spec JSON shaped like:
  {
    "root-directory": "...",
    "model-home": "...",
    "path-pairs": [
      {
        "source-path": "...",
        "output-path": "...",
        "domain-hints": "...",
        "filter": "...",
        "mode": "...",
        "team": "...",
        "component": "...",
        "default_step": "..."
      },
      ...
    ]
  }
  Returns dict with keys: spec_path, spec_dir, root_directory, model_home, model_dir, pair
  """
  # List all JSON files under SPEC_DIR, preferring sources_*.json.
  try:
    all_spec_files = [f for f in os.listdir(SPEC_DIR) if f.endswith(".json")]
  except FileNotFoundError:
    raise FileNotFoundError(f"Spec directory not found: {SPEC_DIR}")

  if not all_spec_files:
    raise FileNotFoundError(f"No spec files found under {SPEC_DIR}.")

  # Prefer sources_*.json; fall back to all JSON files if none match
  sources_files = [f for f in all_spec_files if f.startswith("sources_")]
  spec_files = sources_files if sources_files else all_spec_files

  # Present spec choices.
  labels = [os.path.basename(f) for f in spec_files]
  idx = _prompt_choice("Select a spec file:", labels, default_index=1)
  chosen_spec_fname = spec_files[idx - 1]
  chosen_spec_path = os.path.realpath(os.path.join(SPEC_DIR, chosen_spec_fname))

  data = _load_json(chosen_spec_path)
  if not isinstance(data, dict):
    raise ValueError(f"Spec file {chosen_spec_path} must contain a JSON object at the top level.")

  # Root directory and model home
  root_directory = data.get("root-directory") or data.get("root_directory") or ""
  model_home_raw = data.get("model-home") or data.get("model_home") or "~"
  model_home_expanded = os.path.expanduser(model_home_raw)
  model_dir = os.path.realpath(os.path.join(model_home_expanded, ".model"))

  # Path pairs
  path_pairs = data.get("path-pairs") or data.get("path_pairs") or []
  if not isinstance(path_pairs, list) or not path_pairs:
    raise ValueError(f"No 'path-pairs' found in {chosen_spec_path}.")

  # Build labels for each path pair.
  pair_labels: List[str] = []
  valid_pairs: List[Dict[str, Any]] = []
  for pair in path_pairs:
    if not isinstance(pair, dict):
      continue
    team = pair.get("team") or ""
    component = pair.get("component") or ""
    src = pair.get("source-path") or pair.get("source_path") or "?"
    out = pair.get("output-path") or pair.get("output_path") or "?"
    meta = " / ".join([p for p in (team, component) if p])
    if meta:
      label = f"{src} → {out} [{meta}]"
    else:
      label = f"{src} → {out}"
    pair_labels.append(label)
    valid_pairs.append(pair)

  if not pair_labels:
    raise ValueError(f"'path-pairs' in {chosen_spec_path} did not contain any valid pair objects.")

  pair_idx = _prompt_choice("Select a source/output pair:", pair_labels, default_index=1)
  chosen_pair = valid_pairs[pair_idx - 1]

  return {
    "spec_path": chosen_spec_path,
    "spec_dir": os.path.dirname(chosen_spec_path),
    "root_directory": root_directory or "",
    "model_home": model_home_expanded,
    "model_dir": model_dir,
    "pair": chosen_pair,
  }


# ----------------------------
# PCPT helpers (inspired by categorize_rules.py)
# ----------------------------
def pcpt_sequence(output_dir: str, visualize: str, domain_hints: Optional[str] = None, filter_file: Optional[str] = None) -> None:
  """
  Run: pcpt.sh domain-model --output <output_dir> [--domain-hints <domain_hints>] [--filter <filter_file>] --visualize <visualize>
  """
  cmd = ["pcpt.sh", "domain-model", "--output", output_dir]
  if domain_hints:
    dh_value = os.path.basename(domain_hints) if os.path.isabs(domain_hints) else domain_hints
    cmd.extend(["--domain-hints", dh_value])
  if filter_file:
    ff_value = os.path.basename(filter_file) if os.path.isabs(filter_file) else filter_file
    cmd.extend(["--filter", ff_value])
  cmd.extend(["--visualize", visualize])
  _run(cmd)

def pcpt_run_custom_prompt(
    input_file: str,
    input_file2: str,
    output_dir: str,
    code_dir: str,
    prompt_name: str,
    domain_hints: Optional[str] = None,
    filter_file: Optional[str] = None,
) -> None:
  """
  Run: pcpt.sh run-custom-prompt --input-file <input_file> --input-file2 <input_file2> --output <output_dir> [--domain-hints <domain_hints>] [--filter <filter_file>] <code_dir> <prompt_name>
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
  if domain_hints:
    dh_value = os.path.basename(domain_hints) if os.path.isabs(domain_hints) else domain_hints
    cmd.extend(["--domain-hints", dh_value])
  if filter_file:
    ff_value = os.path.basename(filter_file) if os.path.isabs(filter_file) else filter_file
    cmd.extend(["--filter", ff_value])
  cmd.extend([
    code_dir,
    prompt_name,
  ])
  _run(cmd)


# ----------------------------
# Main flow
# ----------------------------
def main() -> None:
  parser = argparse.ArgumentParser(description="Generate, markup, and regenerate a domain diagram.")
  parser.add_argument("code_dir", nargs="?", default=None, help="Path to code directory to visualize. If omitted, you will be prompted via a spec file.")
  parser.add_argument("output_dir", nargs="?", default=None, help="Output directory for docs. If omitted, you will be prompted via a spec file.")
  parser.add_argument("--prompt-name", default=DEFAULT_PROMPT_NAME, help="Custom prompt template name (default: markup-domain.templ)")
  parser.add_argument("--filter", default=None, help="Filter file (optional). If provided, it will be passed to PCPT.")
  parser.add_argument("--domain-hints", default=None, help="Domain hints file (optional). If provided, it will be passed to PCPT.")
  parser.add_argument("--generate-initial-domain", action="store_true", help="Generate the initial domain diagram (default is to skip).")
  parser.add_argument(
      "--include-all-rules",
      action="store_true",
      help="Include all rules in markup by disabling business relevance filtering in export_rules_for_markup.py."
  )
  args = parser.parse_args()

  code_dir = args.code_dir.strip() if isinstance(args.code_dir, str) and args.code_dir.strip() else None
  output_dir = args.output_dir.strip() if isinstance(args.output_dir, str) and args.output_dir.strip() else None
  prompt_name = args.prompt_name
  filter_file = args.filter.strip() if isinstance(args.filter, str) and args.filter.strip() else None
  domain_hints = args.domain_hints.strip() if isinstance(args.domain_hints, str) and args.domain_hints.strip() else None

  root_path: Optional[str] = None
  pair_source: Optional[str] = None
  model_dir: Optional[str] = None

  if not code_dir or not output_dir:
    _log("No code_dir/output_dir provided. Entering spec-driven mode.", header=True)
    spec_info = _select_spec_pair()
    pair = spec_info["pair"]
    spec_dir = spec_info["spec_dir"]
    root_raw = (spec_info.get("root_directory") or "").strip()
    model_dir = spec_info.get("model_dir")

    # Compute root_path: prefer explicit root_directory from spec; otherwise fall back to repo root.
    if root_raw:
      root_path_candidate = os.path.expanduser(root_raw)
      if not os.path.isabs(root_path_candidate):
        root_path_candidate = os.path.join(spec_dir, root_path_candidate)
      root_path = os.path.realpath(root_path_candidate)
    else:
      root_path = os.path.realpath(REPO_ROOT)
    _log(f"Using root path from spec: {root_path}")

    # Set MODEL_HOME/.model from spec, if available
    if model_dir:
      os.environ["MODEL_HOME"] = model_dir
      _log(f"Using MODEL_HOME from spec: {model_dir}")

    bases = [root_path, REPO_ROOT, spec_dir]

    # Resolve code_dir and output_dir from pair (support both code_dir/source_path and output_dir/output_path keys, including dashed keys)
    code_candidate = (
      pair.get("code_dir")
      or pair.get("source_path")
      or pair.get("source-path")
    )
    output_candidate = (
      pair.get("output_dir")
      or pair.get("output_path")
      or pair.get("output-path")
    )

    # Remember the source path from the spec pair for export_rules_for_markup
    pair_source = pair.get("source-path") or pair.get("source_path") or None

    resolved_code = _resolve_candidate_path(code_candidate, bases) if code_candidate else None
    resolved_output = _resolve_candidate_path(output_candidate, bases) if output_candidate else None

    if not resolved_code or not resolved_output:
      raise ValueError(f"Could not resolve code_dir/output_dir from spec pair. code={code_candidate}, output={output_candidate}")

    code_dir = resolved_code
    output_dir = resolved_output

    # Optionally derive filter and domain_hints from pair if not already supplied on the CLI
    if not filter_file:
      filter_candidate = (
        pair.get("filter")
        or pair.get("filter_path")
        or pair.get("filter-path")
      )
      filter_file = _resolve_candidate_path(filter_candidate, bases) if filter_candidate else None

    if not domain_hints:
      domain_hints_candidate = (
        pair.get("domain-hints")
        or pair.get("domain_hints")
      )
      domain_hints = _resolve_candidate_path(domain_hints_candidate, bases) if domain_hints_candidate else None

    _log(f"Selected code_dir: {code_dir}")
    _log(f"Selected output_dir: {output_dir}")
    if filter_file:
      _log(f"Using filter from spec: {filter_file}")
    if domain_hints:
      _log(f"Using domain hints from spec: {domain_hints}")
  else:
    # CLI-provided mode: preserve previous behaviour for root_path
    root_path = os.path.realpath(os.path.abspath(os.getcwd()))
    _log(f"Detected root path from CWD: {root_path}")

  # Paths used by the flow
  domain_report_src = os.path.join(output_dir, "domain_model_report", "domain_model_report.txt")
  markup_md = os.path.join(output_dir, MARKUP_OUT_DIRNAME, MARKUP_OUT_FILENAME)

  _log("Step 1: Create standard Domain Diagram", header=True)
  if args.generate_initial_domain:
      pcpt_sequence(output_dir=output_dir, visualize=code_dir, domain_hints=domain_hints, filter_file=filter_file)
  else:
      _log("Skipped initial domain generation (default). Use --generate-initial-domain to enable.")

  _log("Step 2: Copy existing sequence report to temp folder", header=True)
  _ensure_dir(TMP_ROOT)
  _copy(domain_report_src, TMP_DOMAIN_TXT)
  _log("Normalizing rules format in temporary domain report (if needed)...")
  _normalize_rules_format(TMP_DOMAIN_TXT)

  _log("Step 3: Export list of current business rules for code", header=True)
  if not root_path:
    # Safety fallback; this should not normally happen.
    root_path = os.path.realpath(os.path.abspath(os.getcwd()))
    _log(f"root_path was not set earlier; defaulting to CWD: {root_path}")

  # Derive rules file path from model_dir (if provided via spec)
  rules_file_arg: Optional[str] = None
  if model_dir:
    rules_file_arg = os.path.join(model_dir, "business_rules.json")
    _log(f"Using rules file from model_dir: {rules_file_arg}")
  else:
    _log("No model_dir from spec; export_rules_for_markup will rely on its own default for rules file.")

  # Decide source_path for export_rules_for_markup: prefer spec pair source-path, else fall back to code_dir
  source_path_arg = pair_source if pair_source else code_dir
  _log(f"Using source_path for export_rules_for_markup: {source_path_arg}")

  # Build absolute path to the exporter script so this works from any CWD.
  # Prefer a helpers/ subfolder if present, but fall back to the legacy location
  helpers_export = os.path.join(SCRIPT_DIR, "helpers", "export_rules_for_markup.py")

  if os.path.exists(helpers_export):
    export_script = helpers_export
  else:
    raise FileNotFoundError(
      f"Could not locate export_rules_for_markup.py in {helpers_export}"
    )

  _log(f"Using export script: {export_script}")

  export_cmd = [
      sys.executable,
      export_script,
      source_path_arg,
      "--format",
      "json",
      "--trace",
      "--trace-limit",
      "500",
  ]
  if rules_file_arg:
    export_cmd.extend(["--rules-file", rules_file_arg])
  if args.include_all_rules:
    export_cmd.append("--include-all-rules")

  _run(export_cmd)

  # Sanity check for the exported rules file (some environments might write elsewhere)
  if not os.path.exists(TMP_EXPORTED_RULES):
    raise FileNotFoundError(
      f"Expected exported rules at {TMP_EXPORTED_RULES} not found. "
      f"Ensure tools/export_rules_for_markup.py writes to that path."
    )

  _log("Step 4: Markup domain description", header=True)

  _log("Step 4: Markup domain description", header=True)
  pcpt_run_custom_prompt(
    input_file=TMP_DOMAIN_TXT,
    input_file2=TMP_EXPORTED_RULES,
    output_dir=output_dir,
    code_dir=code_dir,
    prompt_name=prompt_name,
    domain_hints=domain_hints,
    filter_file=filter_file,
  )

  _log("Step 5: Regenerate domain diagram from markup", header=True)
  if not os.path.exists(markup_md):
    raise FileNotFoundError(
      f"Expected markup file not found: {markup_md}. "
      f"Verify the custom prompt produced it under {os.path.join(output_dir, MARKUP_OUT_DIRNAME)}"
    )

  pcpt_sequence(output_dir=output_dir, visualize=markup_md, domain_hints=domain_hints)

  _log("✅ Done. Domain regenerated using markup.")
  _log(f"Markup file: {markup_md}")
  _log(f"Domain report used: {TMP_DOMAIN_TXT}")
  _log(f"Exported rules: {TMP_EXPORTED_RULES}")

if __name__ == "__main__":
  main()