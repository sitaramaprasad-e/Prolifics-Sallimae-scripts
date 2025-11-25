import sys
from pathlib import Path
from typing import Any, Dict

from helpers.hierarchy_common import step_header, ANSI_YELLOW, ANSI_RESET, load_json, build_temp_source_from_model,ensure_dir, resolve_optional_path, pcpt_run_custom_prompt, prepare_rules_file_for_pcpt, merge_generated_rules_into_model_home, eprint, REPO_ROOT, _resolve_template_path, _normalize_rule_for_compare, hashlib, json

# --- Helper: content fingerprint for rules ---
def _content_fingerprint(rule: dict) -> str:
    try:
        norm = _normalize_rule_for_compare(rule)
        payload = json.dumps(norm, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]
    except Exception:
        return ""
    
def run_simple_compose(
    model_info: Dict[str, Any],
    spec_info: Dict[str, Any],
    model_home_prompted: Path,
    compose_mode: str,
    skip_generate: bool,
) -> None:
    """Handle 'top', 'selected-top', and 'comp' modes with shared, simple flow.
    Extracted to keep MIM logic isolated.
    """
    print("\nSTEP 8: Run PCPT (run-custom-prompt)" if not skip_generate else "\nSTEP 8: Skip generate – using existing report for ingest")
    step_header(8, "Run or Ingest Composed Decision", {
        "Compose mode": compose_mode,
        "Generate": "Skipped (ingest only)" if skip_generate else "Run PCPT"
    })

    # Resolve root-directory from spec; if relative, interpret relative to spec file directory
    spec_dir = Path(spec_info["spec_dir"])
    root_dir_str = spec_info.get("root_directory") or ""
    root_dir = Path(root_dir_str).expanduser()
    if not root_dir.is_absolute():
        root_dir = (spec_dir / root_dir).resolve()

    pair = spec_info["pair"]
    src_rel = pair.get("source-path", "")
    out_rel = pair.get("output-path", "")
    src_label = src_rel if (spec_info.get("source_mode") or "spec") == "spec" else "(model files)"
    filt_rel = ""  # pair.get("filter")
    pcpt_mode = "multi"  # force multi mode; ignore spec-provided mode

    # Optional domain-hints file from spec pair
    domain_hints_rel = pair.get("domain-hints") or pair.get("domain_hints") or ""
    domain_hints_path = resolve_optional_path(
        domain_hints_rel,
        base_candidates=[spec_dir, REPO_ROOT, root_dir],
    )

    # Determine source_mode
    source_mode = (spec_info.get("source_mode") or "spec")
    if source_mode == "model_files":
        print(f"\n{ANSI_YELLOW}--- MODEL FILES source mode active ---{ANSI_RESET}")
        temp_source = build_temp_source_from_model(model_info, spec_info)
        source_path = temp_source.resolve()
        output_path = (temp_source.parent / f"{temp_source.name}.out").resolve()
        ensure_dir(output_path)
        out_label = str(output_path)
        print(f"→ Source: MODEL FILES → {source_path}")
    else:
        source_path = (root_dir / src_rel).resolve()
        output_path = (root_dir / out_rel).resolve()
        out_label = out_rel
        print(f"→ Source: {src_label}")

    # Try to resolve filter in likely places: alongside spec file, under repo root, then under root-dir
    filter_path = resolve_optional_path(
        filt_rel,
        base_candidates=[spec_dir, REPO_ROOT, root_dir],
    )

    # Inputs for the prompt (temp files created earlier)
    rules_file = model_info["rules_out_path"]
    model_file = model_info["selected_model_path"]
    # Prepare a copy of the rules file for PCPT with from_step names instead of ids
    rules_file_for_pcpt = prepare_rules_file_for_pcpt(Path(rules_file))

    # For 'selected-top' mode, annotate the selected model JSON with the explicit top-level decision
    if compose_mode == "selected-top":
        try:
            sel_model_path = Path(model_file)
            if sel_model_path.exists():
                sel_model = load_json(sel_model_path)
                top_decision_id = None
                # Try to get topDecisionId from the first hierarchy (or any hierarchy that has it)
                for h in sel_model.get("hierarchies", []):
                    if isinstance(h, dict) and h.get("topDecisionId"):
                        top_decision_id = h["topDecisionId"]
                        break
                if top_decision_id:
                    # Locate business_rules.json in the model home to resolve the name
                    rules_home_path = Path(model_home_prompted) / "business_rules.json"
                    top_name = ""
                    if rules_home_path.exists():
                        rules_payload = load_json(rules_home_path)
                        rules_list = []
                        if isinstance(rules_payload, list):
                            rules_list = rules_payload
                        elif isinstance(rules_payload, dict):
                            # Support either a flat dict of id->rule or a dict with 'rules' key
                            if "rules" in rules_payload and isinstance(rules_payload["rules"], list):
                                rules_list = rules_payload["rules"]
                            else:
                                # If dict of id -> rule, take values
                                rules_list = list(rules_payload.values())
                        for rule in rules_list:
                            if not isinstance(rule, dict):
                                continue
                            rid = rule.get("id") or rule.get("rule_id")
                            if rid == top_decision_id:
                                top_name = rule.get("name") or rule.get("rule_name") or ""
                                break
                    # Append the annotation line if we have an id (always) and optionally a name
                    annotation_name = top_name if top_name else "(name-not-found)"
                    try:
                        with sel_model_path.open("a", encoding="utf-8") as f:
                            f.write(f"\nThe Top Level Decision Is: {top_decision_id} {annotation_name}\n")
                    except Exception as inner_ex:
                        eprint(f"[WARN] Failed to append top-level decision annotation to selected model: {inner_ex}")
        except Exception as ex:
            eprint(f"[WARN] Failed to resolve top-level decision for selected-top mode: {ex}")

    # Use shared template resolver; map only the special selected-top case
    if compose_mode == "selected-top":
        template_mode = "suggest-decisions-selected-top"
    else:
        template_mode = compose_mode

    template_path = _resolve_template_path(template_mode)
    if not template_path.exists():
        eprint(f"ERROR: Template not found: {template_path}")
        sys.exit(1)

    template_base = template_path.stem

    # Streaming: pcpt_run_custom_prompt uses subprocess.run with check=True (streams to console).
    step_header(9, "Compose decision with PCPT", {
        "Template": template_path.name,
        "Source": src_label,
        "Output": out_label
    })
    print(f"→ Output: {out_label}")
    if filt_rel:
        print(f"→ Filter: {filt_rel}")
    if pcpt_mode:
        print(f"→ Mode:   {pcpt_mode}")
    if domain_hints_path:
        print(f"→ Domain hints: {domain_hints_path}")
    print(f"→ Input 1 (rules for PCPT): {rules_file_for_pcpt} (from {rules_file})")
    print(f"→ Input 2 (model): {model_file}")
    print(f"→ Template: {template_path.name}")
    print(f"→ Compose Mode: {compose_mode}")

    if not skip_generate:
        pcpt_run_custom_prompt(
            source_path=str(source_path),
            custom_prompt_template=template_path.name,
            input_file=str(rules_file_for_pcpt),
            input_file2=str(model_file),
            output_dir_arg=str(output_path),
            domain_hints=str(domain_hints_path) if domain_hints_path else None,
            filter_path=filter_path,
            mode=pcpt_mode,
        )
    else:
        print("[INFO] --skip-generate supplied: not calling pcpt; proceeding to merge from existing report.")

    # Merge the generated rule(s) back into business_rules.json and models.json
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
            step_header(10, "Merge back to model home", {
                "Model home": str(model_home_prompted),
                "Model ID": str(sel_model_id),
                "Output path": str(output_path)
            })
            print("\nMerge back to model home")
            print(f"→ Model home: {model_home_prompted}")
            print(f"→ Output path: {output_path}")
            merge_generated_rules_into_model_home(
                Path(model_home_prompted),
                Path(output_path),
                sel_model_id,
                template_base,
                hierarchy_meta=None,
            )
    except Exception as ex:
        eprint(f"[WARN] Merge step failed: {ex}")

    print("\nDone ✔")
    print("Composed decision report generated and merged.")