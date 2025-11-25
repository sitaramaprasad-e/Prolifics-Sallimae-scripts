#!/usr/bin/env python3
# I want you to implement the spec below to make a sequence of calls to scripts to 
# 1 Read the tools/spec folder and give me a choice to select one of the sources_*.json files
# 2 Read the contents of the selected file to give a set of values like below example
# {
#   "root-directory": "/Users/greghodgkinson/Documents/git.nosync/sallie-mae-example",
#   "model-home": "~/",
#   "path-pairs": [
#     {
#       "source-path": "code/stored_proc",
#       "output-path": "docs/stored_proc",
#       "domain-hints": "stored_proc.hints",
#       "filter": "stored_proc.filter",
#       "mode": "multi",
#       "default_step": "i"
#     },
#     {
#       "source-path": "code/sf",
#       "output-path": "docs/sf",
#       "domain-hints": "sf.hints",
#       "filter": "sf.filter",
#       "mode": "multi"
#       "default_step": "i"
#     }
#   ]
# }
# 3 Pre-flight check that all filter files exist under ~/.pcpt/filters and all domain-hints files exist under ~/.pcpt/hints, and that model home exists
# 4 iterate through each path-pair and do the following
# 4.1 Domain model run using function call pcpt_domain_model(output_dir_arg=<output-path>, domain_hints=<domain-hints>, visualize=True, filter_path=<filter>, source_path=<source-path>,) which can be imported as "from tools.call_pcpt import pcpt_domain_model" 
# 4.2 Business logic run using pcpt_business_logic(output_dir_arg=<output-path>, domain_path=<output-path>/domain_model_report/domain_model_report.txt, domain_hints=<domain-hints>, filter_path=<filter>, source_path=<source-path>,) which can be imported as "from tools.call_pcpt import pcpt_business_logic" 
# 4.3 Ingest the business logic outputs using run_ingest("<output_path>/business_logic_report/business_logic_report.md") which can be imported as "from tools.ingest_rules import run_ingest"
# 4.4 Run the categorize command to categorize the ingested business logic using subprocess.run([sys.executable, "tools/categorise_rules.py"],input="/path/to/model/home\n",   # e.g. "/Users/greg"text=True,check=True,) which needs import "import sys, subprocess"
# 4.5 Provide summary for this path-pair
# 5 Declare overall victory and provide summary
# Note - you must elegantly and concisely write to output along the way so we know what is going on. Including streaming all output from called programs where needed


"""
pcpt_pipeline.py

Implements a pipeline that:
 1) Lists spec files under tools/spec and lets the user choose a source_*.json
 2) Reads the selected spec and extracts:
      - root-directory (repo root)
      - model-home (base home; the script that consumes it appends '/.model')
      - path-pairs: [{ source-path, output-path, domain-hints, filter, mode }]
 3) Pre-flight checks:
      - model-home exists
      - ~/.pcpt/filters/<filter> exists for each path-pair that declares filter
      - ~/.pcpt/domain-hints/<domain-hints> exists for each path-pair that declares domain-hints
 4) For each path-pair, runs in order:
      4.1 Domain model via pcpt_domain_model(...)
      4.2 Business logic via pcpt_business_logic(...)
      4.3 Ingest via run_ingest(<output-path>/business_logic_report/business_logic_report.md)
      4.4 Categorize via subprocess calling tools/categorise_rules.py, piping model-home as stdin
      4.5 Summarize result for this path-pair
 5) Prints overall summary

All external commands stream their output to stdout/stderr (no capture),
so you see progress as it happens.
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
import shutil
from typing import List, Dict, Any, Optional

# Local imports
from tools.helpers.call_pcpt import pcpt_domain_model, pcpt_business_logic, pcpt_run_custom_prompt

# =========================
# Console helpers
# =========================

def _hdr(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def _subhdr(title: str) -> None:
    print("\n-- " + title)


def _fail(msg: str) -> None:
    print(f"❌ {msg}")


def _ok(msg: str) -> None:
    print(f"✅ {msg}")


# =========================
# Spec discovery & load
# =========================

def _spec_dir() -> str:
    # This script lives in <repo>/tools; specs are under <repo>/tools/spec
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec")


def _find_spec_files() -> List[str]:
    d = _spec_dir()
    if not os.path.isdir(d):
        return []
    return [
        os.path.join(d, f)
        for f in sorted(os.listdir(d))
        if f.endswith(".json") and (f.startswith("source_") or f.startswith("sources_")) and os.path.isfile(os.path.join(d, f))
    ]


def _choose_spec(specs: List[str]) -> Optional[str]:
    if not specs:
        _fail("No spec files found under tools/spec (expected 'source_*.json' or 'sources_*.json').")
        return None
    _hdr("Select a spec file")
    for i, path in enumerate(specs, 1):
        print(f"  {i}. {os.path.basename(path)}")
    print("Choose a spec by number (or Ctrl+C to cancel).")
    while True:
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Please enter a valid number.")
            continue
        idx = int(choice)
        if 1 <= idx <= len(specs):
            return specs[idx - 1]
        print("Out of range. Try again.")


def _load_spec(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# Interactive selections
# =========================

def _prompt_pairs_to_run(spec: Dict[str, Any]) -> List[int]:
    pairs = spec.get("path-pairs", [])
    if not pairs:
        return []
    print("\nAvailable path-pairs:")
    for i, p in enumerate(pairs, 1):
        print(f"  {i}. {p.get('source-path','?')} -> {p.get('output-path','?')} (team={p.get('team','-')}, component={p.get('component','-')})")
    print("\nPress Enter to run ALL, or enter comma-separated indexes (e.g., 1,3)")
    s = input("Select pairs: ").strip()
    if not s:
        return list(range(1, len(pairs)+1))
    selected: List[int] = []
    for token in s.split(','):
        token = token.strip()
        if not token.isdigit():
            continue
        idx = int(token)
        if 1 <= idx <= len(pairs):
            selected.append(idx)
    return sorted(set(selected))

_STEP_CHOICES = {
    'domain': 1,
    'business': 2,
    'ingest': 3,
    'categorise': 4,
}

_DEF_STEP_FOR_PROMPT = "domain"

# Alias mapping for step selection
_STEP_ALIASES = {
    'd': 'domain',
    'b': 'business',
    'i': 'ingest',
    'c': 'categorise',
    '1': 'domain',
    '2': 'business',
    '3': 'ingest',
    '4': 'categorise',
}

# Helper to map token to step number (alias, name, or number)
def _step_num_from_token(token: str) -> int:
    """Map a token (alias like 'd' or name like 'domain' or number '1') to a step number.
    Falls back to the default step if unrecognized."""
    if not token:
        return _STEP_CHOICES[_DEF_STEP_FOR_PROMPT]
    t = token.strip().lower()
    mapped = _STEP_ALIASES.get(t, t)
    return _STEP_CHOICES.get(mapped, _STEP_CHOICES[_DEF_STEP_FOR_PROMPT])


def _prompt_start_steps_for_pairs(spec: Dict[str, Any], selected_indexes: List[int]) -> Dict[int, int]:
    """Return mapping of 1-based pair index -> step number to start from.
    Steps: 1=domain, 2=business, 3=ingest, 4=categorise.

    Behavior:
    - Ask whether to use each pair's `default_step` from the spec.
    - If 'yes', offer to apply defaults to all selected pairs.
    - If not applying to all, allow per-pair confirmation or override.
    - If 'no', fall back to manual per-pair step selection.
    """
    result: Dict[int, int] = {}
    pairs = spec.get("path-pairs", [])
    if not pairs or not selected_indexes:
        return result

    print("\nChoose starting steps for the selected pairs.")
    print("Defaults:")
    defaults_map = {}
    for idx in selected_indexes:
        pair = pairs[idx - 1]
        ds_raw = str(pair.get("default_step", "")).strip()
        ds_norm = _STEP_ALIASES.get(ds_raw.lower(), ds_raw.lower()) if ds_raw else "domain"
        ds_label = ds_norm if ds_norm in _STEP_CHOICES else "domain"
        src = pair.get("source-path", "?")
        outp = pair.get("output-path", "?")
        defaults_map[idx] = (ds_raw, ds_label)
        pretty = f"{ds_raw} ({ds_label})" if ds_raw else "domain"
        print(f"  [{idx}] {src} -> {outp} • default: {pretty}")
    print("Options: [d]omain, [b]usiness, [i]ngest, [c]ategorise (or 1/2/3/4)")

    # 1) Offer to use defaults for ALL pairs first
    use_all_defaults = input("Use defaults for ALL selected pairs? [Y/n]: ").strip().lower()
    use_all_defaults = (use_all_defaults == "" or use_all_defaults.startswith("y"))

    if use_all_defaults:
        for idx in selected_indexes:
            ds = str(defaults_map[idx][0]).strip()
            result[idx] = _step_num_from_token(ds)
        return result

    # 2) Per-pair selection: Enter = default for that pair; otherwise pick d/b/i/c or 1/2/3/4
    for idx in selected_indexes:
        pair = pairs[idx - 1]
        src = pair.get('source-path', '?')
        outp = pair.get('output-path', '?')
        ds_raw, ds_label = defaults_map[idx]
        pretty = f"{ds_raw} ({ds_label})" if ds_raw else "domain"
        prompt = f"[{idx}] {src} -> {outp} start step [d/b/i/c or Enter=default {pretty}]: "
        resp = input(prompt).strip().lower()
        if resp == "":
            # Use the per-pair default (falls back to 'domain' if none)
            result[idx] = _step_num_from_token(ds_raw)
        else:
            result[idx] = _step_num_from_token(resp)
    return result


# =========================
# Pre-flight checks
# =========================

def _expand_home(p: str) -> str:
    return os.path.expanduser(p)


def _preflight(spec: Dict[str, Any]) -> bool:
    ok = True
    # Check that root-directory exists
    root_dir = os.path.expanduser(spec.get("root-directory", ""))
    if not os.path.isdir(root_dir):
        _fail(f"Root directory not found: {root_dir}")
        ok = False
    else:
        _ok(f"Root directory found: {root_dir}")
    model_home = _expand_home(spec.get("model-home", "~"))
    if not os.path.isdir(model_home):
        _fail(f"Model home not found: {model_home}")
        ok = False
    else:
        _ok(f"Model home found: {model_home}")

    filters_dir = os.path.expanduser("~/.pcpt/filters")
    hints_dir = os.path.expanduser("~/.pcpt/hints")

    pairs = spec.get("path-pairs", [])
    for pair in pairs:
        filt = pair.get("filter")
        if filt:
            fpath = os.path.join(filters_dir, filt)
            if not os.path.isfile(fpath):
                _fail(f"Missing filter file: {fpath}")
                ok = False
            else:
                _ok(f"Filter ready: {fpath}")
        dh = pair.get("domain-hints")
        if dh:
            hpath = os.path.join(hints_dir, dh)
            if not os.path.isfile(hpath):
                _fail(f"Missing domain-hints file: {hpath}")
                ok = False
            else:
                _ok(f"Domain-hints ready: {hpath}")

        # Check source-path and output-path under root
        src_rel = pair.get("source-path")
        out_rel = pair.get("output-path")
        if src_rel:
            src_path = os.path.join(root_dir, src_rel)
            if not os.path.isdir(src_path):
                _fail(f"Source path not found: {src_path}")
                ok = False
            else:
                _ok(f"Source path ok: {src_path}")
        if out_rel:
            out_path = os.path.join(root_dir, out_rel)
            out_parent = os.path.dirname(out_path)
            if not os.path.isdir(out_parent):
                _fail(f"Output directory base not found: {out_parent}")
                ok = False
            else:
                _ok(f"Output directory base ok: {out_parent}")

    return ok


# =========================
# Pipeline core
# =========================


def run_pipeline(
    spec: Dict[str, Any],
    selected_indexes: List[int],
    start_steps: Dict[int, int],
    skip_cat: bool = False,
    kg_only: bool = False,
    no_kg: bool = False,
) -> Dict[str, Any]:
    # Resolve and adopt repo root as CWD; keep everything else relative to it
    root = os.path.abspath(os.path.expanduser(spec.get("root-directory", os.getcwd())))
    os.chdir(root)
    model_home = _expand_home(spec.get("model-home", "~"))
    pairs = spec.get("path-pairs", [])

    _hdr("Starting PCPT Pipeline")
    print(f"Root directory: {root}")
    print(f"Model home: {model_home}")
    print(f"Path-pairs: {len(pairs)}")

    results: List[Dict[str, Any]] = []

    for idx, pair in enumerate(pairs, 1):
        if selected_indexes and idx not in selected_indexes:
            continue
        start_step = start_steps.get(idx, 1)  # default domain

        src_rel = pair.get("source-path")
        out_rel = pair.get("output-path")
        hints = pair.get("domain-hints")
        filt = pair.get("filter")
        mode = pair.get("mode")
        update_existing = bool(pair.get("update_existing", False))

        # Use spec-provided paths as-is; with CWD at repo root these resolve predictably
        source_path = src_rel
        output_path = out_rel
        domain_hints = hints  # passed as filename; wrapper just forwards
        filter_path = filt    # passed as filename; wrapper just forwards

        _hdr(f"[{idx}/{len(pairs)}] {src_rel} → {out_rel} (mode={mode})")

        pair_summary = {
            "source": source_path,
            "output": output_path,
            "domain_hints": domain_hints,
            "filter": filter_path,
            "update_existing": update_existing,
            "steps": [],
            "ok": True,
        }

        # 4.1 Domain model
        if start_step <= 1:
            _subhdr("4.1 Domain Model")
            try:
                pcpt_domain_model(
                    output_dir_arg=output_path,
                    domain_hints=domain_hints,
                    visualize=True,
                    filter_path=filter_path,
                    source_path=source_path,
                    mode=mode,
                )
                _ok("domain-model completed")
                pair_summary["steps"].append("domain-model: ok")
            except Exception as e:
                _fail(f"domain-model failed: {e}")
                pair_summary["steps"].append(f"domain-model: fail ({e})")
                pair_summary["ok"] = False
        else:
            pair_summary["steps"].append("domain-model: skipped")

        # 4.2 Business logic
        if start_step <= 2:
            if update_existing:
                _subhdr("4.2 Business Logic (update-existing)")
            else:
                _subhdr("4.2 Business Logic")
            try:
                domain_report_path = os.path.join(output_path, "domain_model_report", "domain_model_report.txt")
                business_report_path = os.path.join(output_path, "business_logic_report", "business_logic_report.md")

                if update_existing:
                    # Only use the update flow if an existing report is present; otherwise warn and fall back to normal behavior.
                    if os.path.isfile(business_report_path):
                        # Take an in-place backup before running the update
                        backup_path = business_report_path + ".bak"
                        shutil.copy2(business_report_path, backup_path)
                        _ok(f"Backed up existing business logic report to {backup_path}")

                        # --- Copy domain and business reports into a per-pair temp dir (not output_path) ---
                        # Use a shared .tmp folder at the repo root (CWD) for update inputs
                        tmp_update_dir = os.path.join(".tmp")
                        os.makedirs(tmp_update_dir, exist_ok=True)
                        tmp_domain_path = os.path.join(tmp_update_dir, f"domain_model_report_{idx}.txt")
                        tmp_business_path = os.path.join(tmp_update_dir, f"business_logic_report_{idx}.md")
                        shutil.copy2(domain_report_path, tmp_domain_path)
                        shutil.copy2(business_report_path, tmp_business_path)

                        # Prepare arguments for custom prompt; align index/total usage with standard business-logic behavior
                        prompt_template = "update-business-rules-report.templ"
                        prompt_base = os.path.splitext(prompt_template)[0]  # "update-business-rules-report"
                        run_kwargs = dict(
                            source_path=str(source_path),
                            custom_prompt_template=prompt_template,
                            input_file=str(tmp_domain_path),
                            input_file2=str(tmp_business_path),
                            output_dir_arg=str(output_path),
                            domain_hints=domain_hints,
                            filter_path=filter_path,
                            mode=mode,
                        )
                        # For multi-mode runs, provide index/total for progress; for single-mode, omit them
                        if mode == "multi":
                            run_kwargs["total"] = len(pairs)
                            run_kwargs["index"] = idx
                        pcpt_run_custom_prompt(**run_kwargs)

                        # After the custom prompt runs, copy its output over the original report
                        updated_report_path = os.path.join(output_path, prompt_base, f"{prompt_base}.md")
                        if os.path.isfile(updated_report_path):
                            shutil.copy2(updated_report_path, business_report_path)
                            _ok(f"Updated business logic report from {updated_report_path}")
                        else:
                            raise FileNotFoundError(f"Expected updated business report at {updated_report_path} was not found")
                    else:
                        _fail(f"No existing business logic report found at {business_report_path}; falling back to normal business-logic generation.")
                        # Fall back to the normal business logic flow below
                        pcpt_business_logic(
                            output_dir_arg=output_path,
                            domain_path=domain_report_path,
                            domain_hints=domain_hints,
                            filter_path=filter_path,
                            source_path=source_path,
                            mode=mode,
                        )
                else:
                    # Normal business logic flow (no update)
                    pcpt_business_logic(
                        output_dir_arg=output_path,
                        domain_path=domain_report_path,
                        domain_hints=domain_hints,
                        filter_path=filter_path,
                        source_path=source_path,
                        mode=mode,
                    )

                _ok("business-logic completed")
                pair_summary["steps"].append("business-logic: ok")
            except Exception as e:
                _fail(f"business-logic failed: {e}")
                pair_summary["steps"].append(f"business-logic: fail ({e})")
                pair_summary["ok"] = False
        else:
            pair_summary["steps"].append("business-logic: skipped")

        # 4.3 Ingest Business Logic
        if start_step <= 3:
            _subhdr("4.3 Ingest Business Logic")
            try:
                ingest_md = os.path.join(output_path, "business_logic_report", "business_logic_report.md")
                cmd_ingest = [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), "ingest_rules.py"),
                    ingest_md,
                    "--force",
                ]
                # Pass root-dir and source-path through so ingest_rules can propagate them to CodeFile nodes
                if root:
                    cmd_ingest.extend(["--root-dir", root])
                if source_path:
                    cmd_ingest.extend(["--source-path", source_path])
                # Forward KG-related flags to ingest_rules.py if requested
                if kg_only:
                    cmd_ingest.append("--KG-ONLY")
                if no_kg:
                    cmd_ingest.append("--NO-KG")
                # Feed model_home, team, and component to ingest_rules.py (matches its prompt order)
                stdin_values = f"{model_home}\n{pair.get('team', '')}\n{pair.get('component', '')}\n"
                subprocess.run(
                    cmd_ingest,
                    input=stdin_values,
                    text=True,
                    check=True,
                )
                _ok("ingest completed")
                pair_summary["steps"].append("ingest: ok")
            except Exception as e:
                _fail(f"ingest failed: {e}")
                pair_summary["steps"].append(f"ingest: fail ({e})")
                pair_summary["ok"] = False
        else:
            pair_summary["steps"].append("ingest: skipped")

        # 4.4 Categorize
        if start_step <= 4 and not skip_cat:
            _subhdr("4.4 Categorize")
            try:
                cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "categorise_rules.py")]
                # Ensure we run from repo root so relative paths resolve as the tools normally expect
                # (most of your scripts assume CWD = repo root)
                subprocess.run(
                    cmd,
                    input=(model_home + "\n"),
                    text=True,
                    check=True,
                )
                _ok("categorize completed")
                pair_summary["steps"].append("categorize: ok")
            except subprocess.CalledProcessError as e:
                _fail(f"categorize failed (exit {e.returncode})")
                pair_summary["steps"].append(f"categorize: fail (exit {e.returncode})")
                pair_summary["ok"] = False
            except Exception as e:
                _fail(f"categorize failed: {e}")
                pair_summary["steps"].append(f"categorize: fail ({e})")
                pair_summary["ok"] = False
        else:
            pair_summary["steps"].append("categorize: skipped" + (" (--skip-cat)" if skip_cat else ""))

        results.append(pair_summary)

    return {"root": root, "model_home": model_home, "results": results}


# =========================
# Entry point
# =========================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run PCPT pipeline using a spec file under tools/spec")
    parser.add_argument("--spec", help="Path to a spec JSON (if omitted, you will be prompted to choose under tools/spec)")
    parser.add_argument("--skip-cat", action="store_true", help="Skip the categorize step")
    parser.add_argument(
        "--KG-ONLY",
        "--kg-only",
        dest="kg_only",
        action="store_true",
        help="Forward KG-only mode to ingest_rules.py (reuse existing rules and only emit KG export)",
    )
    parser.add_argument(
        "--NO-KG",
        "--no-kg",
        dest="no_kg",
        action="store_true",
        help="Forward NO-KG mode to ingest_rules.py (disable Neo4j/graph export for this run)",
    )
    args = parser.parse_args()

    if args.spec:
        spec_path = args.spec
    else:
        specs = _find_spec_files()
        if not specs:
            _fail(f"No spec files found in {_spec_dir()}.")
            sys.exit(1)
        spec_path = _choose_spec(specs)
        if not spec_path:
            sys.exit(1)

    _hdr("Loading spec")
    spec = _load_spec(spec_path)

    _hdr("Pre-flight checks")
    if not _preflight(spec):
        _fail("Pre-flight checks failed. Aborting.")
        sys.exit(1)

    selected = _prompt_pairs_to_run(spec)
    if not selected:
        _fail("No pairs selected. Aborting.")
        sys.exit(1)
    start_steps = _prompt_start_steps_for_pairs(spec, selected)

    summary = run_pipeline(
        spec,
        selected,
        start_steps,
        skip_cat=args.skip_cat,
        kg_only=args.kg_only,
        no_kg=args.no_kg,
    )

    # Overall summary
    _hdr("Pipeline Summary")
    total = len(summary["results"])
    passed = sum(1 for s in summary["results"] if s.get("ok"))
    print(f"Root:   {summary['root']}")
    print(f"Home:   {summary['model_home']}")
    print(f"Pairs:  {total} (ok: {passed}, failed: {total - passed})")

    for s in summary["results"]:
        status = "OK" if s.get("ok") else "FAIL"
        print(f"\n[{status}] {s['source']} -> {s['output']}")
        for step in s["steps"]:
            print(f"   - {step}")

    if passed != total:
        sys.exit(2)


if __name__ == "__main__":
    main()