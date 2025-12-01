

#!/usr/bin/env python3
"""Resolve duplicate logic names in business_rules.json.

This script:
  1. Locates a business_rules.json file under a model home (default: ~/.model).
  2. Scans all logics for duplicate `name` values.
  3. Prints a report of duplicate groups (including counts and IDs) and totals.
  4. Optionally renames the duplicates so that:
       - the first logic with a given name is left unchanged
       - subsequent logics are renamed to "<name> (2)", "<name> (3)", etc.

The script is intentionally self-contained but follows the general pattern of
other tools in this folder (model-home awareness, safety checks, clear output).
"""

import datetime
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Tuple


DEFAULT_MODEL_HOME = os.path.expanduser("~/.model")
BUSINESS_RULES_FILENAME = "business_rules.json"


def info(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr)




def load_business_rules(path: str) -> Tuple[Any, List[Dict[str, Any]]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"business rules file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both the newer wrapped structure {"version": n, "logics": [...]} and
    # the older structure that is just a list of logics.
    if isinstance(data, dict) and "logics" in data and isinstance(data["logics"], list):
        logics = data["logics"]
    elif isinstance(data, list):
        logics = data
    else:
        raise ValueError(
            "Unexpected business_rules.json structure: expected a dict with 'logics' "
            "or a list of logic objects."
        )

    return data, logics


def find_duplicate_names(logics: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Return mapping name -> list of indices where that name appears (only for dupes)."""
    name_to_indices: Dict[str, List[int]] = {}

    for idx, logic in enumerate(logics):
        name = logic.get("name")
        if not name:
            # Ignore nameless logics for this check
            continue
        name_to_indices.setdefault(name, []).append(idx)

    dupes = {name: idxs for name, idxs in name_to_indices.items() if len(idxs) > 1}
    return dupes


def print_duplicate_report(logics: List[Dict[str, Any]], dupes: Dict[str, List[int]]) -> None:
    total_logics = len(logics)
    total_groups = len(dupes)
    total_dupe_logics = sum(len(idxs) for idxs in dupes.values())

    print()
    print("=== Duplicate Logic Name Report ===")
    print(f"Total logics: {total_logics}")
    print(f"Duplicate name groups: {total_groups}")
    print(f"Total logics in duplicate groups: {total_dupe_logics}")
    print()

    if not dupes:
        print("No duplicate logic names found. Nothing to do.")
        return

    for name, idxs in sorted(dupes.items(), key=lambda item: item[0].lower()):
        print(f"- Name: {name!r} (count={len(idxs)})")
        for i, idx in enumerate(idxs, start=1):
            logic = logics[idx]
            lid = logic.get("id", "<no-id>")
            kind = logic.get("kind", "<no-kind>")
            print(f"    {i}. id={lid}, kind={kind}")
        print()


def prompt_yes_no(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        info("--yes supplied; proceeding without interactive confirmation.")
        return True

    try:
        answer = input(f"{prompt} [y/N]: ").strip().lower()
    except EOFError:
        # Non-interactive environment; be safe and treat as 'no'.
        warn("No TTY available for confirmation; assuming 'no'. Use --yes to override.")
        return False

    return answer in {"y", "yes"}


def backup_file(path: str) -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = f"{path}.dupe_backup-{timestamp}"
    shutil.copy2(path, backup_path)
    info(f"Created backup: {backup_path}")
    return backup_path


def resolve_duplicates(logics: List[Dict[str, Any]], dupes: Dict[str, List[int]]) -> int:
    """Rename duplicate logic names in-place.

    Returns the number of logic records whose name was changed.
    """
    renamed_count = 0

    for name, idxs in dupes.items():
        if len(idxs) <= 1:
            continue

        # Keep the first occurrence as-is, rename subsequent ones.
        for suffix, idx in enumerate(idxs[1:], start=2):
            logic = logics[idx]
            old_name = logic.get("name")
            new_name = f"{name} ({suffix})"
            if old_name == new_name:
                # Already has the expected suffix; skip.
                continue
            logic["name"] = new_name
            renamed_count += 1

    return renamed_count


def save_business_rules(path: str, original_data: Any, logics: List[Dict[str, Any]]) -> None:
    # Preserve top-level structure where possible.
    if isinstance(original_data, dict) and "logics" in original_data:
        original_data["logics"] = logics
        data_to_write = original_data
    else:
        data_to_write = logics

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_write, f, indent=2, ensure_ascii=False)
        f.write("\n")

    info(f"Updated business rules file: {path}")


def main(argv: List[str]) -> int:
    # Prompt for model home (defaults to ~/.model)
    default_model_home = DEFAULT_MODEL_HOME
    user_input = input(f"Model home directory? [{default_model_home}]: ").strip()
    model_home = user_input or default_model_home

    # Default business rules path based on model home
    rules_path = os.path.join(model_home, BUSINESS_RULES_FILENAME)

    # Simple CLI parsing:
    # - First non-flag arg (if any) is treated as an explicit business_rules.json path
    # - Flags:
    #     --yes / -y : assume yes to prompts
    assume_yes = False
    explicit_path = None

    for arg in argv[1:]:
        if arg in ("--yes", "-y"):
            assume_yes = True
        elif not arg.startswith("-") and explicit_path is None:
            explicit_path = arg
        else:
            # Unknown flag or extra positional args are ignored (for now)
            continue

    if explicit_path is not None:
        rules_path = os.path.abspath(os.path.expanduser(explicit_path))

    print(f"Using business_rules.json: {rules_path}")

    if not os.path.isfile(rules_path):
        error(f"business rules file not found: {rules_path}")
        return 1

    try:
        original_data, logics = load_business_rules(rules_path)
    except (FileNotFoundError, ValueError) as exc:
        error(str(exc))
        return 1

    dupes = find_duplicate_names(logics)
    print_duplicate_report(logics, dupes)

    if not dupes:
        # Nothing to do
        return 0

    total_affected = sum(len(idxs) - 1 for idxs in dupes.values())
    print(f"Logics that would be renamed if you proceed: {total_affected}")
    print("Renaming pattern: first instance keeps its current name; subsequent "
          "instances become '<name> (2)', '<name> (3)', etc.")
    print()

    if not prompt_yes_no("Proceed to resolve duplicate names?", assume_yes):
        info("Aborting without making any changes.")
        return 0

    backup_file(rules_path)
    renamed_count = resolve_duplicates(logics, dupes)

    if renamed_count == 0:
        info("No logic names needed to be changed after all. Original file left intact.")
        return 0

    save_business_rules(rules_path, original_data, logics)
    info(f"Renamed {renamed_count} logic(s) to resolve duplicate names.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))