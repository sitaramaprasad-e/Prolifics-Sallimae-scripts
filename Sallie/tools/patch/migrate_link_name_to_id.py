#!/usr/bin/env python3
"""
Replace link 'from_step' names with 'from_step_id' (rule/decision IDs) in business_rules.json.

Behavior:
- Prompts for a model home directory (default: ~/.model). Expects business_rules.json inside it.
- Builds a name -> [ids] index from all rules' (rule_name, id).
- For each rule.links entry containing 'from_step', resolves the ID and rewrites to 'from_step_id'.
- If any link cannot be resolved or is ambiguous (duplicate names with different IDs), the script fails
  without writing changes unless --allow-partial is provided.
- When writing, creates a timestamped ZIP backup of the original JSON and writes a sidecar changes report.

Usage:
  python3 link_name_to_id.py               # interactive prompt (strict mode)
  python3 link_name_to_id.py --model-home /path/to/model  # non-interactive
  python3 link_name_to_id.py --allow-partial              # proceed even if some unresolved
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Any, Set


def ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def zip_backup(path: Path) -> Path:
    zpath = path.with_suffix(f".{ts()}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(path, arcname=path.name)
    return zpath


def prompt_model_home(default: str) -> Path:
    try:
        raw = input(f"Model home directory [default {default}]: ").strip()
    except EOFError:
        raw = ""
    if not raw:
        raw = default
    return Path(os.path.expanduser(raw)).resolve()


def build_name_index(rules: List[dict]) -> Dict[str, List[str]]:
    index: Dict[str, List[str]] = {}
    for r in rules:
        name = r.get("rule_name")
        rid = r.get("id")
        if not name or not rid:
            continue
        index.setdefault(name, [])
        if rid not in index[name]:
            index[name].append(rid)
    return index

def build_id_set(rules: List[dict]) -> Set[str]:
    ids: Set[str] = set()
    for r in rules:
        rid = r.get("id")
        if isinstance(rid, str) and rid:
            ids.add(rid)
    return ids


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Replace link from_step (names) with from_step_id (ids) in business_rules.json")
    parser.add_argument("--model-home", dest="model_home", default=None, help="Path to model home directory containing business_rules.json (default: prompt, ~/.model)")
    parser.add_argument("--allow-partial", action="store_true", help="Write changes even if some links could not be resolved (unresolved are left as-is)")
    parser.add_argument("--clean-up-dangling-links", action="store_true",
                        help="Remove links whose source rule no longer exists (likely archived/deleted decisions); writes a cleanup report")
    args = parser.parse_args(argv)

    default_home = str(Path("~/.model"))
    model_home = Path(os.path.expanduser(args.model_home)).resolve() if args.model_home else prompt_model_home(default_home)

    rules_path = model_home / "business_rules.json"
    if not rules_path.exists():
        print(f"ERROR: business_rules.json not found at {rules_path}")
        return 1

    try:
        raw = rules_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as e:
        print(f"ERROR: Failed to read/parse {rules_path}: {e}")
        return 1

    if not isinstance(data, list):
        print("ERROR: Top-level JSON is not a list of rules.")
        return 1

    try:
        z = zip_backup(rules_path)
        print(f"Backup created: {z}")
    except Exception as e:
        print(f"ERROR: Failed to create backup zip: {e}")
        return 1

    name_index = build_name_index(data)

    id_set = build_id_set(data)

    # Optional: cleanup pass that removes links pointing to non-existent rules
    if args.clean_up_dangling_links:
        print("\nRunning dangling link cleanup...")
        removed_links: List[Dict[str, Any]] = []
        total_before = 0
        total_after = 0

        for rule in data:
            links = rule.get("links")
            if not isinstance(links, list) or not links:
                continue
            total_before += len(links)
            kept = []
            for link in links:
                if not isinstance(link, dict):
                    removed_links.append({
                        "rule_id": rule.get("id"),
                        "rule_name": rule.get("rule_name"),
                        "link": link,
                        "reason": "malformed_link"
                    })
                    continue

                from_id = link.get("from_step_id")
                from_name = link.get("from_step")

                if isinstance(from_id, str) and from_id:
                    if from_id in id_set:
                        kept.append(link)
                    else:
                        removed_links.append({
                            "rule_id": rule.get("id"),
                            "rule_name": rule.get("rule_name"),
                            "link": link,
                            "reason": "dangling_from_step_id_not_in_model"
                        })
                else:
                    if isinstance(from_name, str) and name_index.get(from_name):
                        kept.append(link)
                    else:
                        removed_links.append({
                            "rule_id": rule.get("id"),
                            "rule_name": rule.get("rule_name"),
                            "link": link,
                            "reason": "dangling_from_step_name_not_in_model"
                        })

            rule["links"] = kept
            total_after += len(kept)

        count_removed = len(removed_links)
        print(f"Removed {count_removed} dangling links out of {total_before} total links.")
        print("These are typically links to rules/decisions that were archived and later deleted — safe to clean up.")

        # Write updated JSON
        try:
            rules_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"ERROR: Failed to write updated JSON after cleanup: {e}")
            return 1

        # Sidecar report
        sidecar = rules_path.with_suffix(".link_cleanup.changes.json")
        report = {
            "when": ts(),
            "file": str(rules_path),
            "removed_links": removed_links,
            "removed_count": count_removed,
            "total_before": total_before,
            "total_after": total_after,
        }
        try:
            sidecar.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"Wrote cleanup report: {sidecar}")
        except Exception as e:
            print(f"WARNING: Failed to write cleanup report: {e}")

        print("Cleanup complete. Exiting.")
        return 0

    total_links = 0
    changed_links = 0
    unresolved: List[Dict[str, Any]] = []
    ambiguous: List[Dict[str, Any]] = []

    for rule in data:
        links = rule.get("links")
        if not isinstance(links, list):
            continue
        for link in links:
            if not isinstance(link, dict):
                continue
            if "from_step" not in link:
                continue

            total_links += 1
            from_name = link.get("from_step")
            id_candidates = name_index.get(from_name, [])

            if not id_candidates:
                unresolved.append({
                    "rule_id": rule.get("id"),
                    "rule_name": rule.get("rule_name"),
                    "from_step_name": from_name,
                    "link": link,
                    "reason": "no matching rule_name found"
                })
                continue

            if len(id_candidates) > 1:
                ambiguous.append({
                    "rule_id": rule.get("id"),
                    "rule_name": rule.get("rule_name"),
                    "from_step_name": from_name,
                    "link": link,
                    "candidates": id_candidates,
                    "reason": "multiple ids for from_step name"
                })
                continue

            # Success: exactly one id
            from_id = id_candidates[0]
            link["from_step_id"] = from_id
            del link["from_step"]
            changed_links += 1

    strict_fail = (len(unresolved) + len(ambiguous)) > 0 and not args.allow_partial

    print("--- Summary ---")
    print(f"File: {rules_path}")
    print(f"Total links scanned: {total_links}")
    print(f"Links updated (name -> id): {changed_links}")
    print(f"Unresolved (no matching rule): {len(unresolved)}")
    print(f"Ambiguous (duplicate names): {len(ambiguous)}")

    if unresolved:
        print("\nUnresolved details:")
        for i, item in enumerate(unresolved, 1):
            l = item.get("link", {}) or {}
            print(f"  [{i}] rule_id={item.get('rule_id')} rule_name={item.get('rule_name')}")
            print(f"      from_step_name={item.get('from_step_name')}")
            print(f"      link.kind={l.get('kind')} from_output={l.get('from_output')} to_input={l.get('to_input')}")
    if ambiguous:
        print("\nAmbiguous details:")
        for i, item in enumerate(ambiguous, 1):
            l = item.get("link", {}) or {}
            print(f"  [{i}] rule_id={item.get('rule_id')} rule_name={item.get('rule_name')}")
            print(f"      from_step_name={item.get('from_step_name')}")
            print(f"      candidates={', '.join(item.get('candidates', []))}")
            print(f"      link.kind={l.get('kind')} from_output={l.get('from_output')} to_input={l.get('to_input')}")

    if strict_fail:
        print("\nHint: These unresolved or ambiguous links are often due to rules that were archived and then deleted.")
        print("You can safely remove them using the --clean-up-dangling-links option.")
        print("ERROR: Unresolved/ambiguous links detected. Run again with --allow-partial to write partial changes, or fix names/duplicates.")
        # Write a diagnostics sidecar even on failure for visibility
        sidecar = rules_path.with_suffix(".link_name_to_id.diagnostics.json")
        report = {
            "when": ts(),
            "file": str(rules_path),
            "total_links": total_links,
            "changed_links": changed_links,
            "unresolved": unresolved,
            "ambiguous": ambiguous,
        }
        sidecar.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote diagnostics: {sidecar}")
        return 2

    # If nothing changed and no issues, exit cleanly
    if changed_links == 0 and len(unresolved) == 0 and len(ambiguous) == 0:
        print("No from_step links found to update.")
        return 0

    # Backup already created above, so comment out this to avoid double backup
    # try:
    #     z = zip_backup(rules_path)
    #     print(f"Backup created: {z}")
    # except Exception as e:
    #     print(f"ERROR: Failed to create backup zip: {e}")
    #     return 1

    try:
        rules_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"ERROR: Failed to write updated JSON: {e}")
        return 1

    # Sidecar report
    sidecar = rules_path.with_suffix(".link_name_to_id.changes.json")
    report = {
        "when": ts(),
        "file": str(rules_path),
        "total_links": total_links,
        "changed_links": changed_links,
        "unresolved": unresolved,
        "ambiguous": ambiguous,
        "allow_partial": args.allow_partial,
    }
    try:
        sidecar.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote sidecar: {sidecar}")
    except Exception as e:
        print(f"WARNING: Failed to write sidecar report: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))