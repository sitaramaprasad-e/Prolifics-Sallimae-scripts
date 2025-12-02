import json
import sys
import shutil
from pathlib import Path
from typing import Any, Dict, List, Set


from helpers.hierarchy_common import (
    step_header, ANSI_YELLOW, ANSI_RESET, load_json, build_temp_source_from_model, ensure_dir, resolve_optional_path,
    pcpt_run_custom_prompt, prepare_logics_file_for_pcpt, merge_generated_logics_into_model_home, eprint, REPO_ROOT, _resolve_template_path,
    SUGGEST_HIERARCHY_TEMPLATE_DIR, SUGGEST_HIERARCHY_MINIMAL_TEMPLATE_DIR, TMP_DIR, _load_logics_from_report, write_json, _safe_backup_json,
    attach_from_logic_ids_on_links,
)

def _scrub_links_to_hierarchy_scope(logics: List[dict], allowed_ids: Set[str], allowed_names_cf: Set[str]) -> List[dict]:
    """
    Remove links that point to producers outside the current hierarchy scope.
    Rules:
      - Keep only links whose from_logic_id resolves to one of allowed_ids.
      - If only a name is present (from_logic) and it matches allowed_names_cf, resolve to id; else drop.
      - Drop any link with empty/unresolvable from_logic_id.
      - Always persist id-only (remove from_logic).
      - Deduplicate by (from_logic_id, from_output, to_input, kind).
    """
    if not isinstance(logics, list) or not logics:
        return logics or []

    # Build in-scope name -> id map
    name_to_id: Dict[str, str] = {}
    for r in logics:
        if not isinstance(r, dict):
            continue
        rid = (r.get("id") or "").strip()
        rn  = (r.get("name") or r.get("name") or "").strip()
        if rid and rn:
            name_to_id[rn.casefold()] = rid

    def _looks_like_uuid(s: str) -> bool:
        s = (s or "").strip()
        return len(s) == 36 and s.count("-") == 4

    for r in logics:
        if not isinstance(r, dict):
            continue
        links = r.get("links") or []
        if not isinstance(links, list) or not links:
            r["links"] = []
            continue

        cleaned: List[dict] = []
        seen = set()
        for l in links:
            if not isinstance(l, dict):
                continue
            fs = (l.get("from_logic") or "").strip()
            fsid = (l.get("from_logic_id") or "").strip()

            # Resolve names -> ids
            if not fsid and fs:
                if _looks_like_uuid(fs):
                    fsid = fs
                else:
                    fsid = name_to_id.get(fs.casefold(), "")

            # If fsid looks like a name, try resolve via map
            if fsid and not _looks_like_uuid(fsid):
                fsid = name_to_id.get(fsid.casefold(), "")

            # Must resolve to an in-scope id
            if not fsid or fsid not in allowed_ids:
                # If we only got a name and it is in-scope by name, try one last resolve
                if fs and fs.casefold() in allowed_names_cf:
                    fsid = name_to_id.get(fs.casefold(), "")
                    if not fsid or fsid not in allowed_ids:
                        continue
                else:
                    continue

            key = (
                fsid,
                (l.get("from_output") or "").strip(),
                (l.get("to_input") or "").strip(),
                (l.get("kind") or "").strip(),
            )
            if key in seen:
                continue
            seen.add(key)

            l = dict(l)
            l["from_logic_id"] = fsid
            l.pop("from_logic", None)
            cleaned.append(l)

        r["links"] = cleaned

    return logics



# ---------------------------
# Helper: Extract Top-Level decision suggestions from report
# ---------------------------
from typing import Tuple, Set


# ---------------------------
# Helper: Deduplicate Decision Names
# ---------------------------
def _dedupe_decision_names(model_home: Path) -> None:
    """Ensure that rule names are unique across all logics.

    Behaviour:
      * Treat names that only differ by trailing numeric suffixes like
        " (2)", " (3)", etc. as part of the same decision family.
      * For each base name, the first occurrence becomes the plain
        base (no suffix), subsequent ones get " (2)", " (3)", ...
        picking the lowest unused number each time.
      * Avoid ever generating names like "Logic (2) (2)" – we always
        normalise back to the base name and a single numeric suffix.
    Operates on business_rules.json.
    """
    try:
        rules_path = (model_home / "business_rules.json").resolve()
        if not rules_path.exists():
            return

        payload = load_json(rules_path)
        logics = None
        root = payload

        if isinstance(payload, dict) and isinstance(payload.get("logics"), list):
            logics = payload["logics"]
        elif isinstance(payload, list):
            logics = payload
        else:
            eprint("[WARN] business_rules.json has unexpected shape for dedupe; skipping name deduplication.")
            return

        if not isinstance(logics, list):
            eprint("[WARN] business_rules.json 'logics' is not a list; skipping name deduplication.")
            return

        # Helper: split name into (base, suffix_num) where suffix_num == 0 means
        # "no numeric suffix". We treat things like "Foo (2)" as base "Foo" with
        # suffix_num == 2.
        import re

        name_re = re.compile(r"^(.*?)(?:\s*\((\d+)\))?$")

        def _split_name(n: str) -> tuple[str, int]:
            n = (n or "").strip()
            if not n:
                return "", 0
            m = name_re.match(n)
            if not m:
                return n, 0
            base = (m.group(1) or "").strip()
            num_str = m.group(2)
            if not num_str:
                return base, 0
            try:
                return base, int(num_str)
            except ValueError:
                return base, 0

        # First pass: capture original parsed names so we don't have to
        # re-parse during renaming.
        parsed: List[tuple[str, int]] = []
        for r in logics:
            if not isinstance(r, dict):
                parsed.append(("", 0))
                continue
            name = (r.get("name") or r.get("name") or "").strip()
            base, num = _split_name(name)
            parsed.append((base, num))

        # Second pass: assign unique names per base.
        used_numbers_by_base: Dict[str, Set[int]] = {}
        changed = 0

        for r, (base, _orig_num) in zip(logics, parsed):
            if not isinstance(r, dict):
                continue
            if not base:
                # Skip nameless rules
                continue

            current_name = (r.get("name") or r.get("name") or "").strip()
            used = used_numbers_by_base.setdefault(base, set())

            # Decide which suffix to use. 0 means the plain base name.
            if 0 not in used:
                suffix_num = 0
            else:
                # Find the smallest k >= 2 that is not yet used.
                suffix_num = 2
                while suffix_num in used:
                    suffix_num += 1

            used.add(suffix_num)

            if suffix_num == 0:
                new_name = base
            else:
                new_name = f"{base} ({suffix_num})"

            if new_name != current_name:
                r["name"] = new_name
                changed += 1

        if changed:
            _safe_backup_json(rules_path)
            if isinstance(root, dict):
                root["logics"] = logics
                write_json(rules_path, root)
            else:
                write_json(rules_path, logics)
            print(f"[INFO] Deduplicated {changed} decision name(s) in business_rules.json")
    except Exception as ex:
        eprint(f"[WARN] Failed to deduplicate decision names in business_rules.json: {ex}")

def _extract_top_suggestions_from_report(report_path: Path) -> Tuple[Set[str], Set[str]]:
    """Parse a suggestion report for MIM pre-step and extract Top-Level decision ids/names.

    Supports two formats:
      1) New hierarchy JSON:
         {
           "hierarchies": [
             { "top": {"id": "...", "name": "..."}, ... }, ...
           ]
         }
      2) Legacy rules-style JSON or markdown with embedded JSON that can be handled
         by _load_logics_from_report(), where each object may include id/name/name.
    Returns (ids, names) as sets.
    """
    raw = report_path.read_text(encoding="utf-8").strip()

    # Try direct JSON first (whole doc)
    def _try_full_json(text: str):
        try:
            return json.loads(text)
        except Exception:
            return None

    obj = _try_full_json(raw)

    # If not full JSON, try fenced code blocks ```json ... ``` or ``` ... ```
    if obj is None:
        import re
        for block in re.findall(r"```(?:json|JSON)?\s*([\s\S]*?)```", raw):
            cand = _try_full_json(block.strip())
            if cand is not None:
                obj = cand
                break

    ids: Set[str] = set()
    names: Set[str] = set()

    # Case 1: New hierarchy structure
    if isinstance(obj, dict) and isinstance(obj.get("hierarchies"), list):
        for h in obj["hierarchies"]:
            if not isinstance(h, dict):
                continue
            td = h.get("top") or {}
            tid = (td.get("id") or "").strip()
            tname = (td.get("name") or td.get("name") or "").strip()
            if tid:
                ids.add(tid)
            if tname:
                names.add(tname)
        return ids, names

    # Case 2: Fall back to legacy rules parsing
    try:
        rules = _load_logics_from_report(report_path)
    except Exception:
        rules = []
    for r in rules:
        rid = (r.get("id") or r.get("uuid") or r.get("rule_id") or "").strip()
        rn = (r.get("name") or r.get("name") or "").strip()
        if rid:
            ids.add(rid)
        if rn:
            names.add(rn)
    return ids, names

def run_mim_compose(
    model_info: Dict[str, Any],
    spec_info: Dict[str, Any],
    model_home_prompted: Path,
    compose_mode: str,
    skip_generate: bool,
    keep_going: bool = False,
) -> None:
    """Handle 'mim' (meet-in-the-middle) mode with hierarchy discovery and per-hierarchy loop.
    Isolated to allow surgical improvements without impacting 'top'/'next'.
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
    pcpt_mode = "multi"

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

    filter_path = resolve_optional_path(
        filt_rel,
        base_candidates=[spec_dir, REPO_ROOT, root_dir],
    )

    logics_file = model_info["logics_out_path"]
    model_file = model_info["selected_model_path"]
    # Prepare a PCPT-specific copy of the rules file for the Top-Level suggestion pre-step
    logics_file_for_pcpt_suggest = prepare_logics_file_for_pcpt(Path(logics_file))

    template_path = _resolve_template_path("mim")
    if not template_path.exists():
        eprint(f"ERROR: Template not found: {template_path}")
        sys.exit(1)
    template_base = template_path.stem

    # Choose which hierarchy suggestion template to use:
    # - Standard 'mim' mode uses the full suggest-hierarchy template.
    # - New 'mim-minimal' mode uses the minimal variant instead.
    mode_lower = (compose_mode or "").strip().lower()
    if mode_lower == "mim-minimal":
        suggest_template = SUGGEST_HIERARCHY_MINIMAL_TEMPLATE_DIR
    else:
        suggest_template = SUGGEST_HIERARCHY_TEMPLATE_DIR

    # Now also looking to see if there is a selected model topId to annotate
    try:
        sel_model_path = Path(model_file)
        if sel_model_path.exists():
            sel_model = load_json(sel_model_path)
            top_id = None
            # Try to get topId from the first hierarchy (or any hierarchy that has it)
            for h in sel_model.get("hierarchies", []):
                if isinstance(h, dict) and h.get("topId"):
                    top_id = h["topId"]
                    break
            if top_id:
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
                        if "logics" in rules_payload and isinstance(rules_payload["logics"], list):
                            rules_list = rules_payload["logics"]
                        else:
                            # If dict of id -> rule, take values
                            rules_list = list(rules_payload.values())
                    for rule in rules_list:
                        if not isinstance(rule, dict):
                            continue
                        rid = rule.get("id") or rule.get("rule_id")
                        if rid == top_id:
                            top_name = rule.get("name") or rule.get("name") or ""
                            break
                # Append the annotation line if we have an id (always) and optionally a name
                annotation_name = top_name if top_name else "(name-not-found)"
                try:
                    with sel_model_path.open("a", encoding="utf-8") as f:
                        f.write(f"\nThe Top Level Decision Is: {top_id} {annotation_name}\n")
                except Exception as inner_ex:
                    eprint(f"[WARN] Failed to append top-level decision annotation to selected model: {inner_ex}")
    except Exception as ex:
        eprint(f"[WARN] Failed to resolve top-level decision for selected-top mode: {ex}")

    # MIM pre‑step: discover top-level decisions
    if not skip_generate:
        step_header(9, "MIM pre‑step: Discover Top‑Level decisions", {
            "Template": suggest_template.name,
            "Output dir": str(output_path)
        })
        print("[MIM] Pre-step: Suggest Top-Level decisions")
        if not suggest_template.exists():
            eprint(f"ERROR: Template not found: {suggest_template}")
            sys.exit(1)
        pcpt_run_custom_prompt(
            source_path=str(source_path),
            custom_prompt_template=suggest_template.name,
            input_file=str(logics_file_for_pcpt_suggest),
            input_file2=str(model_file),
            output_dir_arg=str(output_path),
            domain_hints=str(domain_hints_path) if domain_hints_path else None,
            filter_path=filter_path,
            mode=pcpt_mode,
        )

    # Find suggestion report and optionally mark rules as Top-Level
    suggest_base = suggest_template.stem
    suggest_candidates = [
        output_path / suggest_base / f"{suggest_base}.json",
        output_path / f"{suggest_base}.json",
        output_path / suggest_base / f"{suggest_base}.md",
        output_path / f"{suggest_base}.md",
    ]
    suggest_report = next((p for p in suggest_candidates if p.exists()), None)

    suggest_report_path = None
    if suggest_report:
        ensure_dir(TMP_DIR)
        dest_path = TMP_DIR / suggest_report.name
        try:
            shutil.copy2(suggest_report, dest_path)
            suggest_report_path = str(dest_path)
            print(f"[MIM] Using suggested hierarchy as Input 2 (copied into TMP_DIR): {suggest_report_path}")
        except Exception as ex:
            suggest_report_path = str(suggest_report)
            eprint(f"[WARN] MIM: Failed to copy hierarchy report into TMP_DIR: {ex}. Using original path.")

        try:
            select_ids, select_names = _extract_top_suggestions_from_report(Path(suggest_report_path))
        except Exception as ex:
            eprint(f"[WARN] MIM: Failed to parse suggestion report for Top-Level discovery: {ex}; proceeding without Top-Level overrides.")
            select_ids, select_names = set(), set()
        if select_ids or select_names:
            try:
                logics_data = load_json(Path(logics_file))
                changed = 0
                for rr in logics_data:
                    if not isinstance(rr, dict):
                        continue
                    rid0 = (rr.get("id") or "").strip()
                    #rn0 = (rr.get("name") or rr.get("name") or "").strip()
                    if rid0 and rid0 in select_ids:
                        rr["kind"] = "Decision (Top-Level)"
                        rr.pop("Kind", None)
                        changed += 1
                if changed:
                    write_json(Path(logics_file), logics_data)
                    print(f"[MIM] Marked {changed} rule(s) as Top-Level in logics_for_model.json before main prompt.")
                else:
                    print("[MIM] No matching rules found to mark as Top-Level; proceeding as-is.")
            except Exception as ex:
                eprint(f"[WARN] MIM: Failed to update logics_for_model.json with Top-Level kinds: {ex}")

    # If we have a hierarchy doc, process one hierarchy at a time
    if suggest_report_path:
        def _load_hierarchy_doc(path_str: str) -> dict:
            p = Path(path_str)
            txt = p.read_text(encoding="utf-8")
            if p.suffix.lower() == ".json":
                return json.loads(txt)
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(txt[start:end+1])
            raise ValueError(f"Unsupported hierarchy doc format: {p}")

        # Resolve selected model id once for the loop
        selected_model = model_info.get("selected_model") or {}
        sel_model_id = selected_model.get("id")
        if not sel_model_id:
            sel_model_path = Path(model_info.get("selected_model_path", ""))
            if sel_model_path.exists():
                try:
                    sel_model = load_json(sel_model_path)
                    if isinstance(sel_model, dict):
                        sel_model_id = sel_model.get("id")
                except Exception:
                    sel_model_id = None

        try:
            doc = _load_hierarchy_doc(suggest_report_path)
        except Exception as ex:
            eprint(f"[WARN] MIM: Failed to load hierarchy doc '{suggest_report_path}': {ex}. Falling back to single-pass.")
            doc = {}

        hierarchies = doc.get("hierarchies") or []

        # One-time clear of existing hierarchies for this model at the start of a MIM run
        if isinstance(hierarchies, list) and hierarchies and sel_model_id:
            try:
                models_path = (model_home_prompted / "models.json").resolve()
                models_root: Any = None
                models_version = None
                models_payload = load_json(models_path) if models_path.exists() else []
                models = []
                if isinstance(models_payload, dict) and isinstance(models_payload.get("models"), list):
                    models_root = models_payload
                    models_version = models_payload.get("version")
                    models = models_payload.get("models", [])
                elif isinstance(models_payload, list):
                    models = models_payload
                else:
                    eprint("[WARN] models.json has unexpected shape when clearing hierarchies; skipping clear.")
                    return

                if isinstance(models, list):
                    sel_idx = None
                    for idx, m in enumerate(models):
                        if isinstance(m, dict) and m.get("id") == sel_model_id:
                            sel_idx = idx
                            break
                    if sel_idx is not None:
                        # Optimistic concurrency check for models.json before clearing hierarchies
                        if models_path.exists() and models_version is not None:
                            try:
                                cur_raw = load_json(models_path)
                                if isinstance(cur_raw, dict) and isinstance(cur_raw.get("models"), list):
                                    cur_version = cur_raw.get("version")
                                    if cur_version is not None and cur_version != models_version:
                                        eprint(
                                            f"[ERROR] models.json version changed on disk "
                                            f"(expected {models_version}, found {cur_version}); "
                                            f"aborting hierarchy clear to avoid overwriting concurrent changes."
                                        )
                                        return
                            except Exception as inner_ex:
                                eprint(f"[WARN] Could not re-read models.json for concurrency check during hierarchy clear: {inner_ex}")

                        _safe_backup_json(models_path)
                        model_obj = models[sel_idx]
                        model_obj["hierarchies"] = []
                        models[sel_idx] = model_obj

                        # Persist using rooted structure {"version": N, "models": [...]}, incrementing version on success
                        if isinstance(models_root, dict):
                            current_version = models_root.get("version")
                            if current_version is None:
                                current_version = models_version
                            new_version = (current_version or 0) + 1
                            models_root["models"] = models
                            models_root["version"] = new_version
                            write_json(models_path, models_root)
                        else:
                            new_version = (models_version or 0) + 1
                            write_json(models_path, {"version": new_version, "models": models})

                        print(f"[INFO] Cleared existing hierarchies for model {sel_model_id} at start of MIM run")
            except Exception as ex:
                eprint(f"[WARN] Could not clear existing hierarchies for model {sel_model_id}: {ex}")

        # Call validation before merging
        if isinstance(hierarchies, list) and hierarchies:
            total = len(hierarchies)
            for i, hier in enumerate(hierarchies):
                tmp_one = TMP_DIR / f"hierarchy_{i+1}.json"
                try:
                    write_json(tmp_one, {"hierarchies": [hier]})
                except Exception as ex:
                    eprint(f"[WARN] MIM: Could not write temp hierarchy file {tmp_one}: {ex}. Skipping this hierarchy.")
                    continue

                # Collect ids/names from this hierarchy
                def _collect_ids_names_from_hierarchy(h: dict) -> tuple[set[str], set[str]]:
                    ids, names = set(), set()
                    td = h.get("top") or {}
                    if isinstance(td, dict):
                        tid = (td.get("id") or "").strip()
                        tname = (td.get("name") or td.get("name") or "").strip()
                        if tid: ids.add(tid)
                        if tname: names.add(tname)
                    for d in (h.get("lower") or []):
                        if not isinstance(d, dict):
                            continue
                        did = (d.get("id") or "").strip()
                        dname = (d.get("name") or d.get("name") or "").strip()
                        if did: ids.add(did)
                        if dname: names.add(dname)
                    return ids, names

                try:
                    hier_ids, hier_names = _collect_ids_names_from_hierarchy(hier)
                    all_rules = load_json(Path(logics_file))
                    per_hier_rules = []
                    for rr in all_rules:
                        if not isinstance(rr, dict):
                            continue
                        rid = (rr.get("id") or "").strip()
                        #rn  = (rr.get("name") or rr.get("name") or "").strip()
                        if rid and rid in hier_ids:
                            per_hier_rules.append(rr)
                    if not per_hier_rules:
                        eprint(f"[WARN] MIM: No matching logics found in logics_for_model.json for hierarchy {i+1}; proceeding with empty subset.")
                    logics_file_for_iter = str(TMP_DIR / f"logics_for_model_h{i+1}.json")

                    # Scrub links so only in-hierarchy producer links remain
                    _allowed_ids = { (r.get("id") or "").strip() for r in per_hier_rules if isinstance(r, dict) and r.get("id") }
                    _allowed_names_cf = { (r.get("name") or r.get("name") or "").strip().casefold()
                                        for r in per_hier_rules if isinstance(r, dict) and (r.get("name") or r.get("name")) }
                    per_hier_rules = _scrub_links_to_hierarchy_scope(per_hier_rules, _allowed_ids, _allowed_names_cf)

                    write_json(Path(logics_file_for_iter), per_hier_rules)
                    print(f"[TRACE] Hierarchy {i+1}: prepared {len(per_hier_rules)} rule(s) for input.")
                except Exception as ex:
                    eprint(f"[WARN] MIM: Failed to filter rules for hierarchy {i+1}: {ex}. Using full rules file.")
                    logics_file_for_iter = str(logics_file)

                # Prepare a PCPT-specific copy of the per-hierarchy rules file (from_logic names instead of ids)
                logics_file_for_pcpt_h = prepare_logics_file_for_pcpt(Path(logics_file_for_iter))

                input2_file = str(tmp_one)
                input2_label = "hierarchy"
                step_header(10 + i, "Process hierarchy", {
                    "Hierarchy": hier.get("name") or "(unnamed)",
                    "Index": f"{i+1}/{total}"
                })
                print(f"[MIM] Processing hierarchy {i+1}/{total}: {hier.get('name') or '(unnamed)'}")
                print(f"→ Source: {src_label}")
                print(f"→ Output: {out_label}")
                if filt_rel:
                    print(f"→ Filter: {filt_rel}")
                if pcpt_mode:
                    print(f"→ Mode:   {pcpt_mode}")
                print(f"→ Input 1 (logics): {logics_file_for_iter}")
                print(f"→ Input 2 ({input2_label}): {input2_file}")
                print(f"→ Template: {template_path.name}")
                print(f"→ Compose Mode: {compose_mode}")

                if not skip_generate:
                    pcpt_run_custom_prompt(
                        source_path=str(source_path),
                        custom_prompt_template=template_path.name,
                        input_file=str(logics_file_for_pcpt_h),
                        input_file2=str(input2_file),
                        output_dir_arg=str(output_path),
                        domain_hints=str(domain_hints_path) if domain_hints_path else None,
                        filter_path=filter_path,
                        mode=pcpt_mode,
                        total=total,
                        index=i+1,
                    )
                else:
                    print("[INFO] --skip-generate supplied: not calling pcpt; proceeding to merge from existing report.")

                # Merge per-hierarchy
                try:
                    if not sel_model_id:
                        eprint("[WARN] Could not determine selected model id for merge; skipping merge for this hierarchy.")
                    else:
                        step_header(11 + i, "Merge back to model home", {
                            "Model home": str(model_home_prompted),
                            "Model ID": str(sel_model_id),
                            "Output path": str(output_path)
                        })
                        print("\nMerge back to model home (per-hierarchy)")
                        print(f"→ Model home: {model_home_prompted}")
                        print(f"→ Output path: {output_path}")
                        # Build hierarchy_meta for this top decision.
                        # Note: In MIM mode, the top-level decision is always treated as NEW.
                        # We therefore do NOT attempt to resolve a missing id by matching
                        # against existing logics by name – the id for the top-level will
                        # be assigned based on the newly generated logic during merge.
                        # We *do* always propagate the original hierarchy top.name so that
                        # even if PCPT renames the decision in its suggestions, the
                        # hierarchy metadata keeps the original top-level label.
                        td = (hier.get("top") or {}) if isinstance(hier, dict) else {}
                        td_id = (td.get("id") or "").strip()
                        td_name = (td.get("name") or td.get("name") or "").strip()
                        h_name = (hier.get("name") or "").strip()
                        h_desc = (hier.get("flow_description") or "").strip()
                        hier_meta = {"by_id": {}, "by_name": {}}
                        payload = {"hierarchy_name": h_name, "hierarchy_description": h_desc}
                        if td_id:
                            hier_meta["by_id"][td_id] = payload
                        if td_name:
                            hier_meta["by_name"][td_name] = payload
                        merge_generated_logics_into_model_home(
                            model_home=model_home_prompted,
                            output_path=output_path,
                            selected_model_id=sel_model_id,
                            template_base=template_base,     # <- ensures is_mim_mode=True in the callee
                            restrict_ids=hier_ids,           # <- scope to this hierarchy
                            restrict_names=hier_names,
                            hierarchy_meta=hier_meta,        # <- includes top decision (by id OR by name)
                        )
                except Exception as ex:
                    eprint(f"[WARN] Merge failed for hierarchy {i+1}/{total}: {ex}")

                if keep_going:
                    print(f"\n{ANSI_YELLOW}--- Completed hierarchy {i+1}/{total} ---{ANSI_RESET}")
                else:
                    print(f"\n{ANSI_YELLOW}--- Waiting before next hierarchy ({i+1}/{total}) ---{ANSI_RESET}")
                    input("Press Enter to continue when ready, or Ctrl+C to stop...\n")

            _dedupe_decision_names(model_home_prompted)
            print("\nDone ✔")
            print("Composed decision report generated and merged (per‑hierarchy).")
            return

    # Fallback: single combined pass if no hierarchy suggestions were found
    step_header(9, "Compose decision with PCPT", {
        "Template": template_path.name,
        "Source": src_label,
        "Output": out_label
    })
    print(f"→ Source: {src_label}")
    print(f"→ Output: {out_label}")
    if filt_rel:
        print(f"→ Filter: {filt_rel}")
    if pcpt_mode:
        print(f"→ Mode:   {pcpt_mode}")
    if domain_hints_path:
        print(f"→ Domain hints: {domain_hints_path}")
    print(f"→ Input 1 (rules for PCPT): {logics_file_for_pcpt_suggest} (from {logics_file})")
    print(f"→ Input 2 (hierarchy): {suggest_report_path or model_file}")
    print(f"→ Template: {template_path.name}")
    print(f"→ Compose Mode: {compose_mode}")

    if not skip_generate:
        pcpt_run_custom_prompt(
            source_path=str(source_path),
            custom_prompt_template=template_path.name,
            input_file=str(logics_file_for_pcpt_suggest),
            input_file2=str(suggest_report_path or model_file),
            output_dir_arg=str(output_path),
            domain_hints=str(domain_hints_path) if domain_hints_path else None,
            filter_path=filter_path,
            mode=pcpt_mode,
        )
    else:
        print("[INFO] --skip-generate supplied: not calling pcpt; proceeding to merge from existing report.")

    # Merge back (no per-hierarchy restriction)
    try:
        selected_model = model_info.get("selected_model") or {}
        sel_model_id = selected_model.get("id")
        if not sel_model_id:
            sel_model_path = Path(model_info.get("selected_model_path", ""))
            if sel_model_path and sel_model_path.exists():
                sel_model = load_json(sel_model_path)
                sel_model_id = sel_model.get("id") if isinstance(sel_model, dict) else None
        if not sel_model_id:
            eprint("[WARN] Could not determine selected model id for merge; skipping Step 6.")
        else:
            # Build combined_meta for all hierarchies if suggest_report_path exists
            combined_meta = None
            if suggest_report_path:
                try:
                    doc = json.loads(Path(suggest_report_path).read_text(encoding="utf-8"))
                except Exception:
                    doc = {}
                if isinstance(doc, dict) and isinstance(doc.get("hierarchies"), list):
                    by_id, by_name = {}, {}
                    for hh in doc.get("hierarchies"):
                        if not isinstance(hh, dict):
                            continue
                        td = hh.get("top") or {}
                        td_id = (td.get("id") or "").strip()
                        td_name = (td.get("name") or td.get("name") or "").strip()
                        h_name = (hh.get("name") or "").strip()
                        h_desc = (hh.get("flow_description") or "").strip()
                        payload = {"hierarchy_name": h_name, "hierarchy_description": h_desc}
                        if td_id:
                            by_id[td_id] = payload
                        if td_name:
                            by_name[td_name] = payload
                    combined_meta = {"by_id": by_id, "by_name": by_name}
            step_header(10, "Merge back to model home", {
                "Model home": str(model_home_prompted),
                "Model ID": str(sel_model_id),
                "Output path": str(output_path)
            })
            print("\nMerge back to model home")
            print(f"→ Model home: {model_home_prompted}")
            print(f"→ Output path: {output_path}")
            merge_generated_logics_into_model_home(
                Path(model_home_prompted),
                Path(output_path),
                sel_model_id,
                template_base,
                hierarchy_meta=combined_meta,
            )
    except Exception as ex:
        eprint(f"[WARN] Merge step failed: {ex}")

    print("\nDone ✔")
    print("Composed decision report generated and merged.")