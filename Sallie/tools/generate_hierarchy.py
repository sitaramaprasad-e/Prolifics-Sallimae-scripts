#!/usr/bin/env python3
"""
Generate decisions or hierarchies by selecting a model and composing its logic
via PCPT. Supports a single active compose mode: 'mim-minimal' (meet‑in‑the‑middle, minimal).

All other modes ('top', 'comp', 'selected-top', 'mim', and 'chain') are **deprecated and disabled**; they cannot be selected or used.

Current behaviour specification:

0) Prompt for model-home directory (default: ~/.model). This folder must contain
   models.json and business_rules.json.

1) Load models.json, allow the user to select a model, and write a cleaned copy
   (no logicIds) into .tmp/generate_hierarchy/selected_model.json.

2) Export only the rules referenced by the selected model’s logicIds into
   .tmp/generate_hierarchy/logics_for_model.json. Fields not required by PCPT are
   stripped. IDs are preserved.

3) Require the user to select a sources_*.json spec. MODEL FILES mode is also
   supported, where all referenced code/doc files from the selected model are
   copied into a deterministic temp source directory.

4) If using the spec’s paths, allow selection of a path‑pair (source → output).
   In MODEL FILES mode, skip the path‑pair entirely.

5) Prepare a PCPT‑specific copy of logics_for_model.json where link 'from_logic_id'
   UUIDs are converted to 'from_logic' names so PCPT can reason over names.

6) Determine the correct prompt template based on compose_mode:
       mim-minimal  → suggest-decisions-mim-minimal.templ
   (All other modes are deprecated and not available.)

7) If --skip-generate is NOT used, call pcpt_run_custom_prompt with:
       - source directory/File
       - rules_for_pcpt.json
       - selected_model.json
       - chosen template
       - output directory
   If --skip-generate is used, skip the PCPT call and proceed directly to ingest.

8) Ingest and merge PCPT output into business_rules.json and models.json:
      • In top/next mode:
            - Add or update rules (DMN-aware diff)
            - Merge unique links
            - Append created/updated ids into model.logicIds
      • In MIM mode:
            - Strict CREATE vs UPDATE semantics
            - Do not overwrite Kind for existing decisions
            - Merge only links on existing decisions
            - Create new decisions only when PCPT marks them as new
            - Update model.hierarchies (not overlays)

9) Always back up JSON files before modifying them. Summaries of updates,
   skips, or merges are printed to the console.

This script is the orchestrator; per‑mode logic is isolated in run_simple_compose
(top/next) and run_mim_compose (mim).
"""
ACTIVE_MODES = ["mim-minimal"]
DEPRECATED_MODES = ["top", "comp", "selected-top", "mim", "chain"]

import json
import sys
import uuid
import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from helpers.deprecated.hierarchy_simple import run_simple_compose
from helpers.hierarchy_mim import run_mim_compose
from helpers.deprecated.hierarchy_chain import run_chain_compose

from helpers.hierarchy_common import step_header, load_json, ensure_dir, eprint, write_json, TMP_DIR, choose_from_list, SPEC_DIR, prompt_with_default

# ---------------------------
# Core steps
# ---------------------------

def step_select_model(model_home: Path, keep_ids: bool = True, compose_mode: Optional[str] = None) -> Dict[str, Any]:
    step_header(1, "Load models and select one", {"Model home": str(model_home)})
    models_path = model_home / "models.json"
    logics_path = model_home / "business_rules.json"

    if not models_path.exists():
        eprint(f"ERROR: models.json not found at {models_path}")
        sys.exit(1)
    if not logics_path.exists():
        eprint(f"ERROR: business_rules.json not found at {logics_path}")
        sys.exit(1)

    models_data = load_json(models_path)
    if isinstance(models_data, list):
        models = models_data
    elif isinstance(models_data, dict) and isinstance(models_data.get("models"), list):
        models = models_data["models"]
    else:
        eprint("ERROR: models.json must be either a list of models or an object with a 'models' array.")
        sys.exit(1)

    if not models:
        eprint("ERROR: models.json should be a non-empty list of models.")
        sys.exit(1)

    menu = []
    for m in models:
        name = m.get("name", "(unnamed)")
        mid = m.get("id", "")
        logic_ids = m.get("logicIds") or []
        logic_count = len(logic_ids) if isinstance(logic_ids, list) else 0
        hierarchies = m.get("hierarchies") or []
        hierarchy_count = len(hierarchies) if isinstance(hierarchies, list) else 0
        menu.append(f"{name} (logic: {logic_count}, hier: {hierarchy_count})  –  {mid}")
    sel_idx = choose_from_list("Select a model:", menu, default_index=1)
    selected_model = models[sel_idx - 1]

    step_header(2, "Model selected", {
        "Model": f"{selected_model.get('name','(unnamed)')}",
        "Model ID": f"{selected_model.get('id','')}"
    })

    # Temp write selected model
    ensure_dir(TMP_DIR)
    selected_model_path = TMP_DIR / "selected_model.json"
    # Remove logicIds from temp model file (not needed downstream)
    model_copy = dict(selected_model)
    model_copy.pop("logicIds", None)
    write_json(selected_model_path, model_copy)
    print(f"→ Wrote selected model to {selected_model_path}")

    # For 'selected-top' mode, require at least one hierarchy with a top decision
    if (compose_mode or "").strip().lower() == "selected-top":
        hierarchies = selected_model.get("hierarchies") or []
        has_top = False
        for h in hierarchies:
            if not isinstance(h, dict):
                continue
            # Prefer explicit topId
            if h.get("topId"):
                has_top = True
                break
        if not has_top:
            eprint(
                "ERROR: In 'selected-top' mode the selected model "
                f"'{selected_model.get('name', '(unnamed)')}' must have at least one "
                "hierarchy with a top decision)."
            )
            sys.exit(1)

    # Cross-reference logics
    step_header(3, "Export logics belonging to the selected model", {
        "Model": f"{selected_model.get('name','(unnamed)')}",
        "Model ID": f"{selected_model.get('id','')}"
    })
    logics_data = load_json(logics_path)
    if isinstance(logics_data, list):
        logics_all = logics_data
    elif isinstance(logics_data, dict) and isinstance(logics_data.get("logics"), list):
        logics_all = logics_data["logics"]
    else:
        eprint("ERROR: business_rules.json must be either a list of logics or an object with a 'logics' array.")
        sys.exit(1)

    wanted_ids = selected_model.get("logicIds") or []
    wanted_set = set(wanted_ids)
    # Keep order according to model's list
    id_to_logic = {r.get("id"): r for r in logics_all if isinstance(r, dict) and r.get("id")}
    filtered_logics: List[Dict[str, Any]] = [id_to_logic[rid] for rid in wanted_ids if rid in id_to_logic]
    if not filtered_logics:
        eprint(f"ERROR: No logics found in the selected model '{selected_model.get('name', '(unnamed)')}'.")
        sys.exit(1)

    missing = [rid for rid in wanted_ids if rid not in id_to_logic]
    if missing:
        eprint(f"WARNING: {len(missing)} logic ids listed in the model were not found in business_rules.json")

    # Strip fields not needed in composed decision temp output
    cleaned_logics = []
    for r in filtered_logics:
        rc = dict(r)
        drop_keys = ["timestamp", "business_area", "archived"]
        for k in drop_keys:
            rc.pop(k, None)
        cleaned_logics.append(rc)

    logics_out_path = TMP_DIR / "logics_for_model.json"
    write_json(logics_out_path, cleaned_logics)
    print(f"→ Wrote {len(cleaned_logics)} logics to {logics_out_path}")

    try:
        names_preview = [ (r.get("name") or r.get("name") or "(unnamed)") for r in cleaned_logics ][:5]
        if len(cleaned_logics) > 0:
            step_header(4, "Logics prepared for compose", {
                "Count": str(len(cleaned_logics)),
                "Preview": ", ".join(names_preview) + (" …" if len(cleaned_logics) > 5 else "")
            })
    except Exception:
        pass

    return {
        "selected_model": selected_model,
        "selected_model_path": str(selected_model_path),
        "logics_out_path": str(logics_out_path),
    }


def step_select_sources_spec() -> Dict[str, Any]:
    step_header(5, "Select a sources spec file", "tools/spec/sources_*.json")
    # Always require a sources spec; skip the "MODEL FILES (no sources spec)" branch entirely.
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

    # Prompt for source mode BEFORE any path-pair selection
    source_mode_menu = [
        "Use source path from spec",
        "Build from MODEL FILES for selected model"
    ]
    source_mode_sel = choose_from_list("Select a source for PCPT:", source_mode_menu, default_index=2)
    source_mode = "spec" if source_mode_sel == 1 else "model_files"

    if source_mode == "model_files":
        # Skip path-pair selection entirely, but still need to return a minimal pair for labels
        step_header(6, "MODEL FILES mode (skip path‑pair)", {"Spec": chosen.name})
        chosen_pair = {"source-path": "(model_files)", "output-path": ""}
        return {
            "spec_path": str(chosen),
            "spec_dir": str(chosen.parent),
            "root_directory": root_dir,
            "model_home": model_home_from_spec,
            "pair": chosen_pair,
            "source_mode": source_mode,
        }
    else:
        # Proceed with path-pair selection as before
        step_header(6, "Select a path‑pair from the spec", {"Spec": chosen.name})
        pair_menu = []
        for pair in path_pairs:
            src = pair.get("source-path", "")
            outp = pair.get("output-path", "")
            team = pair.get("team", "")
            comp = pair.get("component", "")
            pair_menu.append(f"{src} → {outp}  [{team} / {comp}]")
        pair_idx = choose_from_list("Select a pair:", pair_menu, default_index=1)
        chosen_pair = path_pairs[pair_idx - 1]

        step_header(7, "Path‑pair selected", {
            "Source → Output": f"{chosen_pair.get('source-path','')} → {chosen_pair.get('output-path','')}",
            "Team / Component": f"{chosen_pair.get('team','')} / {chosen_pair.get('component','')}"
        })

        return {
            "spec_path": str(chosen),
            "spec_dir": str(chosen.parent),
            "root_directory": root_dir,
            "model_home": model_home_from_spec,
            "pair": chosen_pair,
            "source_mode": source_mode,
        }


def step_run_pcpt(
    model_info: Dict[str, Any],
    spec_info: Dict[str, Any],
    model_home_prompted: Path,
    compose_mode: str,
    skip_generate: bool,
    keep_going: bool = False,
) -> None:
    mode_lower = (compose_mode or "").strip().lower()
    if mode_lower in ("mim", "mim-minimal"):
        return run_mim_compose(
            model_info,
            spec_info,
            model_home_prompted,
            compose_mode,
            skip_generate,
            keep_going=keep_going,
        )
    elif mode_lower == "chain":
        return run_chain_compose(
            model_info,
            spec_info,
            model_home_prompted,
            compose_mode,
            skip_generate,
            keep_going=keep_going,
        )
    else:
        return run_simple_compose(
            model_info,
            spec_info,
            model_home_prompted,
            compose_mode,
            skip_generate,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a composed decision report (single or multi).")
    parser.add_argument(
        "--mode",
        choices=ACTIVE_MODES + DEPRECATED_MODES,
        help=(
            "Select template behavior: "
            "'mim-minimal' uses suggest-decisions-mim-minimal.templ. "
            "All other modes ('top', 'comp', 'selected-top', 'mim', 'chain') are deprecated and cannot be used."
        ),
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=(
            "In MIM mode, process all hierarchies without prompting between each one. "
            "Has no effect in 'top' or 'next' modes."
        ),
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
    keep_going = args.keep_going
    if compose_mode in DEPRECATED_MODES:
        eprint(f"ERROR: compose mode '{compose_mode}' is deprecated and cannot be used. "
               f"Please choose one of: {', '.join(ACTIVE_MODES)}.")
        sys.exit(1)
    if not compose_mode:
        choice = choose_from_list(
            "Select compose mode:",
            [
                "top          – DEPRECATED (cannot be used)",
                "comp         – DEPRECATED (cannot be used)",
                "mim          – DEPRECATED (cannot be used)",
                "mim-minimal  – meet-in-the-middle (minimal hierarchy)",
                "selected-top – DEPRECATED (cannot be used)",
                "chain        – DEPRECATED (cannot be used)",
            ],
            default_index=4,
        )
        if choice == 1:
            compose_mode = "top"
        elif choice == 2:
            compose_mode = "comp"
        elif choice == 3:
            compose_mode = "mim"
        elif choice == 4:
            compose_mode = "mim-minimal"
        elif choice == 5:
            compose_mode = "selected-top"
        else:
            compose_mode = "chain"
        if compose_mode in DEPRECATED_MODES:
            eprint(
                f"ERROR: compose mode '{compose_mode}' is deprecated and cannot be used. "
                f"Please choose one of: {', '.join(ACTIVE_MODES)}."
            )
            sys.exit(1)

    step_header("0", "Generate Hierarchy", {"Compose mode": compose_mode})
    print("=== Generate Hierarchy ===")

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

    # Steps 1-2: Select model and export its logics
    model_info = step_select_model(
        model_home,
        keep_ids=(compose_mode in ("mim", "mim-minimal")),
        compose_mode=compose_mode,
    )

    # Step 5-6: Run pcpt and finish
    step_run_pcpt(model_info, spec_info, model_home, compose_mode, skip_generate, keep_going)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(130)