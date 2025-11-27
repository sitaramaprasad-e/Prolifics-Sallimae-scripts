import os
import sys
import json
from pathlib import Path
from datetime import datetime
import shutil


def prompt_with_default(message: str, default: str) -> str:
    """Prompt the user with a default value."""
    resp = input(f"{message} [{default}]: ").strip()
    return resp or default


def load_models(models_path: Path):
    if not models_path.exists():
        raise FileNotFoundError(f"models.json not found at {models_path}")
    with models_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def print_help() -> None:
    print("=== restoreModelHierarchyFromBackup.py help ===")
    print("Usage: python restoreModelHierarchyFromBackup.py [options]")
    print("Options:")
    print("  -h, --h, --help   Show this help message and exit.")
    print()
    print("This tool:")
    print("  * Prompts for a jsons directory (default ~/.model)")
    print("  * Loads the current models.json and lets you choose a model and one of its hierarchies")
    print("  * Lets you scroll through models.json.bak-* snapshots (newest first)")
    print("  * For the selected snapshot, restores:")
    print("      - the selected hierarchy, and")
    print("      - businessLogicIds from that snapshot, merged into the model")
    print("    Business logic IDs from the snapshot that do not exist in business_rules.json")
    print("    are NOT merged back in.")
    print()
    print("Interactive commands while browsing backups:")
    print("  [n]ext     Move to the next (older) backup")
    print("  [p]revious Move to the previous (newer) backup")
    print("  [r]estore  Restore the currently shown hierarchy/business logic")
    print("  [q]uit     Exit without making changes")
    print()


def load_business_rule_names(json_dir: Path) -> dict[str, str]:
    """
    Load business_rules.json (if present) and return a mapping of id -> name/heading.
    """
    path = json_dir / "business_rules.json"
    if not path.exists():
        print(f"Warning: business_rules.json not found at {path}; "
              "businessLogicIds will be shown without names.")
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: failed to load business_rules.json: {e}")
        return {}

    if not isinstance(data, list):
        print("Warning: business_rules.json is not a list; "
              "businessLogicIds will be shown without names.")
        return {}

    id_to_name: dict[str, str] = {}
    for rule in data:
        if not isinstance(rule, dict):
            continue
        rid = rule.get("id")
        if not rid:
            continue
        # Prefer 'heading' then 'name', fall back to empty
        name = rule.get("heading") or rule.get("name")
        if not name:
            continue
        id_to_name[rid] = name

    return id_to_name


def load_business_rule_archived_flags(json_dir: Path) -> dict[str, bool]:
    """
    Load business_rules.json (if present) and return a mapping of id -> archived flag.
    """
    path = json_dir / "business_rules.json"
    if not path.exists():
        # Names loader already prints a warning; stay quiet here.
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        # If it fails, just skip archived flags.
        return {}

    if not isinstance(data, list):
        return {}

    flags: dict[str, bool] = {}
    for rule in data:
        if not isinstance(rule, dict):
            continue
        rid = rule.get("id")
        if not rid:
            continue
        flags[rid] = bool(rule.get("archived", False))
    return flags


def choose_index(count: int, label: str) -> int:
    while True:
        raw = input(f"Select {label} (1-{count}): ").strip()
        try:
            idx = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if 1 <= idx <= count:
            return idx - 1  # convert to 0-based
        print(f"Please enter a number between 1 and {count}.")


def choose_model(models):
    print("\nAvailable models in current models.json:")
    for i, m in enumerate(models):
        name = m.get("name", "<unnamed>")
        mid = m.get("id", "<no-id>")
        hier_count = len(m.get("hierarchies", []))
        print(f"  [{i+1}] {name} (id={mid}, hierarchies={hier_count})")
    return choose_index(len(models), "model")


def choose_hierarchy(model: dict, id_to_name: dict[str, str]) -> int:
    hierarchies = model.get("hierarchies") or []
    if not hierarchies:
        raise ValueError("Selected model has no hierarchies.")
    print(f"\nHierarchies for model '{model.get('name', '<unnamed>')}'")
    for i, h in enumerate(hierarchies):
        name = h.get("name", "<unnamed>")
        top = h.get("topDecisionId", "<no-topDecisionId>")
        use_graph = h.get("useGraph")
        top_name = id_to_name.get(top) if top and top != "<no-topDecisionId>" else None
        if top_name:
            top_display = f"{top} ({top_name})"
        else:
            top_display = top
        extra = f", useGraph={use_graph}" if use_graph is not None else ""
        print(f"  [{i+1}] {name} (topDecisionId={top_display}{extra})")
    return choose_index(len(hierarchies), "hierarchy")


def list_backup_files(models_dir: Path):
    backups = [
        p for p in models_dir.iterdir()
        if p.is_file() and p.name.startswith("models.json.bak-")
    ]
    backups.sort(key=lambda p: p.name, reverse=True)
    return backups


def parse_backup_timestamp(backup_name: str) -> str:
    # Expect suffix after last '-' to be timestamp like 20251118211732
    try:
        ts_str = backup_name.split("models.json.bak-")[-1]
        dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "<unknown>"


def find_model_in_backup(models_backup, target_model: dict):
    target_id = target_model.get("id")
    target_name = target_model.get("name")
    # Prefer match on id
    if target_id:
        for m in models_backup:
            if m.get("id") == target_id:
                return m
    # Fallback to name
    if target_name:
        for m in models_backup:
            if m.get("name") == target_name:
                return m
    return None


def find_matching_hierarchy(
    backup_model: dict,
    current_model: dict,
    current_hierarchy_index: int,
) -> dict | None:
    backup_hiers = backup_model.get("hierarchies") or []
    if not backup_hiers:
        return None

    current_hierarchies = current_model.get("hierarchies") or []
    if not current_hierarchies:
        return None

    current_h = current_hierarchies[current_hierarchy_index]
    current_top = current_h.get("topDecisionId")
    current_name = current_h.get("name")

    # 1) Match by topDecisionId
    if current_top:
        for h in backup_hiers:
            if h.get("topDecisionId") == current_top:
                return h

    # 2) Match by name
    if current_name:
        for h in backup_hiers:
            if h.get("name") == current_name:
                return h

    # 3) Fallback: same index position
    if current_hierarchy_index < len(backup_hiers):
        return backup_hiers[current_hierarchy_index]

    return None


def print_business_logic_ids_with_names(backup_model: dict, current_model: dict, id_to_name: dict[str, str], archived_flags: dict[str, bool]) -> None:
    backup_ids = backup_model.get("businessLogicIds") or []
    current_ids = current_model.get("businessLogicIds") or []
    if not backup_ids and not current_ids:
        print("\nNo businessLogicIds found for this model (backup or current).")
        return
    if not backup_ids:
        print("\nNo businessLogicIds found for this model in the backup snapshot.")
    else:
        print(f"\nBusiness logic for this model (from backup): {len(backup_ids)}")

    backup_ids_set = set(backup_ids)
    current_ids_set = set(current_ids)

    # First, list all businessLogicIds from the backup, marking those not present in the current model
    for bid in backup_ids:
        name = id_to_name.get(bid)
        archived = archived_flags.get(bid, False)
        annotations: list[str] = []

        if not name:
            annotations.append("WARNING: no matching rule in business_rules.json")
        elif archived:
            annotations.append("WARNING: rule is archived")

        if bid not in current_ids_set:
            annotations.append("INFO: not present in current model")

        if not name:
            base = f"  - {bid} (name not found)"
        else:
            base = f"  - {bid} ({name})"

        if annotations:
            print(f"{base} [{' | '.join(annotations)}]")
        else:
            print(base)

    # Then, list any businessLogicIds that are in the current model but missing in this backup snapshot
    missing_in_backup = current_ids_set - backup_ids_set
    if missing_in_backup:
        print(
            f"\nBusinessLogicIds present in current model but missing from this backup: {len(missing_in_backup)} "
            f"(these will be retained in the current model after restore)"
        )
        for bid in missing_in_backup:
            name = id_to_name.get(bid)
            archived = archived_flags.get(bid, False)
            annotations: list[str] = []

            if not name:
                annotations.append("WARNING: no matching rule in business_rules.json")
            elif archived:
                annotations.append("WARNING: rule is archived")

            base_name = name if name else "name not found"
            base = f"  - {bid} ({base_name})"
            if annotations:
                print(f"{base} [{' | '.join(annotations)}]")
            else:
                print(base)


def make_backup(models_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = models_path.with_name(f"models.json.bak-{ts}")
    shutil.copy2(models_path, backup_path)
    print(f"\nCreated backup: {backup_path}")
    return backup_path


def restore_hierarchy(
    models_path: Path,
    models_current,
    model_index: int,
    hierarchy_index: int | None,
    new_hierarchy: dict,
    backup_model: dict,
    valid_business_logic_ids: set[str],
):
    # Make a backup before modifying
    make_backup(models_path)

    model = models_current[model_index]
    hierarchies = model.get("hierarchies") or []
    if hierarchy_index is not None:
        if not hierarchies:
            raise ValueError("Target model has no hierarchies to replace.")
    # Merge businessLogicIds from backup model into current model (union, no dupes, only valid ids)
    current_ids = model.get("businessLogicIds") or []
    backup_ids = backup_model.get("businessLogicIds") or []
    merged_ids = list(current_ids)
    for bid in backup_ids:
        # Only merge business logic that still exists in business_rules.json
        if bid not in merged_ids and bid in valid_business_logic_ids:
            merged_ids.append(bid)
    model["businessLogicIds"] = merged_ids

    if hierarchy_index is not None:
        print(
            f"\nReplacing hierarchy index {hierarchy_index} "
            f"in model '{model.get('name', '<unnamed>')}' "
            f"(id={model.get('id', '<no-id>')})."
        )
        hierarchies[hierarchy_index] = new_hierarchy
    else:
        print(
            f"\nAppending new hierarchy to model '{model.get('name', '<unnamed>')}' "
            f"(id={model.get('id', '<no-id>')})."
        )
        hierarchies.append(new_hierarchy)
    model["hierarchies"] = hierarchies
    models_current[model_index] = model

    # Write updated models.json
    with models_path.open("w", encoding="utf-8") as f:
        json.dump(models_current, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Updated {models_path} with restored hierarchy.")


def main():
    if any(arg in ("-h", "--h", "--help") for arg in sys.argv[1:]):
        print_help()
        return

    default_dir = str(Path("~/.model").expanduser())
    print("=== Restore a hierarchy from models.json backups ===")
    print("NOTE: Restore will update both the selected hierarchy and the model's businessLogicIds.")
    print("      Business logic IDs from the backup that do not exist in business_rules.json")
    print("      will NOT be merged back in.\n")

    json_dir_str = prompt_with_default(
        "Enter path to jsons directory containing models.json",
        default_dir,
    )
    json_dir = Path(json_dir_str).expanduser()
    models_path = json_dir / "models.json"

    business_rule_names = load_business_rule_names(json_dir)
    business_rule_archived = load_business_rule_archived_flags(json_dir)
    valid_business_logic_ids: set[str] = set(business_rule_names.keys())

    try:
        models_current = load_models(models_path)
    except Exception as e:
        print(f"Error loading models.json: {e}")
        return

    if not isinstance(models_current, list):
        print("models.json is not a list of models; aborting.")
        return

    # Select model from current file
    model_index = choose_model(models_current)
    model_current = models_current[model_index]

    # Optionally select a hierarchy from the current model
    print("\nDo you want to select a hierarchy from the current model to match against backups?")
    print("  [y]es  - match and replace an existing hierarchy")
    print("  [n]o   - choose hierarchies directly from backups (restored as new entries)")
    choose_current = input("Select option [y/N]: ").strip().lower() or "n"
    if choose_current == "y":
        hierarchy_index: int | None = choose_hierarchy(model_current, business_rule_names)
    else:
        hierarchy_index = None

    # Discover backup files
    backups = list_backup_files(json_dir)
    if not backups:
        print("\nNo models.json.bak-* files found in this directory.")
        return

    print(
        f"\nFound {len(backups)} backup files. "
        "You can scroll through them starting from the most recent."
    )

    current_backup_pos = 0
    while 0 <= current_backup_pos < len(backups):
        backup_path = backups[current_backup_pos]
        ts_display = parse_backup_timestamp(backup_path.name)
        print(
            f"\n=== Backup {current_backup_pos + 1} of {len(backups)} ===\n"
            f"File: {backup_path.name}\n"
            f"Timestamp: {ts_display}"
        )

        try:
            with backup_path.open("r", encoding="utf-8") as f:
                models_backup = json.load(f)
        except Exception as e:
            print(f"  Error loading this backup: {e}")
            cmd = input("Command: [n]ext (older), [p]revious (newer), [q]uit: ").strip().lower() or "n"
            if cmd == "n":
                current_backup_pos += 1
                continue
            if cmd == "p":
                if current_backup_pos == 0:
                    print("Already at newest backup; no newer backups. Press [n] to see older backups.")
                    continue
                current_backup_pos -= 1
                continue
            if cmd == "q":
                print("Aborting without changes.")
                return
            continue

        if not isinstance(models_backup, list):
            print("  Backup file does not contain a list of models; skipping.")
            cmd = input("Command: [n]ext (older), [p]revious (newer), [q]uit: ").strip().lower() or "n"
            if cmd == "n":
                current_backup_pos += 1
            elif cmd == "p":
                if current_backup_pos == 0:
                    print("Already at newest backup; no newer backups. Press [n] to see older backups.")
                    continue
                current_backup_pos -= 1
            elif cmd == "q":
                print("Aborting without changes.")
                return
            continue

        backup_model = find_model_in_backup(models_backup, model_current)
        if backup_model is None:
            print(
                "  No matching model (by id or name) found in this backup; skipping."
            )
            cmd = input("Command: [n]ext (older), [p]revious (newer), [q]uit: ").strip().lower() or "n"
            if cmd == "n":
                current_backup_pos += 1
            elif cmd == "p":
                if current_backup_pos == 0:
                    print("Already at newest backup; no newer backups. Press [n] to see older backups.")
                    continue
                current_backup_pos -= 1
            elif cmd == "q":
                print("Aborting without changes.")
                return
            continue

        if hierarchy_index is not None:
            backup_hierarchy = find_matching_hierarchy(
                backup_model,
                model_current,
                hierarchy_index,
            )
            if backup_hierarchy is None:
                print(
                    "  Matching hierarchy not found in this backup's model; "
                    "skipping."
                )
                cmd = input("Command: [n]ext (older), [p]revious (newer), [q]uit: ").strip().lower() or "n"
                if cmd == "n":
                    current_backup_pos += 1
                elif cmd == "p":
                    if current_backup_pos == 0:
                        print("Already at newest backup; no newer backups. Press [n] to see older backups.")
                        continue
                    current_backup_pos -= 1
                elif cmd == "q":
                    print("Aborting without changes.")
                    return
                continue

            # Show the businessLogicIds for this backup model, with names from business_rules.json,
            # and mark differences vs the current model.
            print_business_logic_ids_with_names(backup_model, model_current, business_rule_names, business_rule_archived)

            # Cross-reference topDecisionId and print its name if available, with warnings
            top_id = backup_hierarchy.get("topDecisionId")
            if top_id:
                top_name = business_rule_names.get(top_id)
                top_archived = business_rule_archived.get(top_id, False)
                if not top_name:
                    print(f"\nTop decision: {top_id} (name not found) [WARNING: no matching rule in business_rules.json]")
                elif top_archived:
                    print(f"\nTop decision: {top_id} ({top_name}) [WARNING: rule is archived]")
                else:
                    print(f"\nTop decision: {top_id} ({top_name})")
            else:
                print("\nTop decision: <none>")

            print("\n--- Hierarchy candidate from this backup ---")
            print(json.dumps(backup_hierarchy, indent=2, ensure_ascii=False))
        else:
            # No current hierarchy selected: show all hierarchies in this backup model
            backup_hierarchy = None

            # Show the businessLogicIds for this backup model, with names from business_rules.json,
            # and mark differences vs the current model.
            print_business_logic_ids_with_names(backup_model, model_current, business_rule_names, business_rule_archived)

            backup_hiers = backup_model.get("hierarchies") or []
            if not backup_hiers:
                print("\nNo hierarchies found in this backup model.")
            else:
                print(f"\nHierarchies in this backup model: {len(backup_hiers)}")
                for idx, h in enumerate(backup_hiers, start=1):
                    h_name = h.get("name", "<unnamed>")
                    top = h.get("topDecisionId")
                    top_name = business_rule_names.get(top) if top else None
                    top_archived = business_rule_archived.get(top, False) if top else False
                    annotations: list[str] = []
                    if top:
                        if not top_name:
                            annotations.append("WARNING: no matching rule in business_rules.json")
                        elif top_archived:
                            annotations.append("WARNING: rule is archived")
                        if top_name:
                            top_display = f"{top} ({top_name})"
                        else:
                            top_display = top
                    else:
                        top_display = "<no-topDecisionId>"
                    ann_str = f" [{' | '.join(annotations)}]" if annotations else ""
                    print(f"  [{idx}] {h_name} (topDecisionId={top_display}){ann_str}")

        cmd = input(
            "\nCommand: [r]estore this version, [n]ext (older), [p]revious (newer), [q]uit: "
        ).strip().lower() or "n"

        if cmd == "r":
            # Determine which hierarchy to restore from this backup
            selected_backup_hierarchy = backup_hierarchy
            if hierarchy_index is None:
                backup_hiers = backup_model.get("hierarchies") or []
                if not backup_hiers:
                    print("No hierarchies available in this backup to restore.")
                    continue
                while True:
                    choice = input(f"Select hierarchy to restore from this backup (1-{len(backup_hiers)}): ").strip()
                    try:
                        choice_idx = int(choice)
                    except ValueError:
                        print("Please enter a valid integer.")
                        continue
                    if 1 <= choice_idx <= len(backup_hiers):
                        selected_backup_hierarchy = backup_hiers[choice_idx - 1]
                        break
                    print(f"Please enter a number between 1 and {len(backup_hiers)}.")

            if selected_backup_hierarchy is None:
                print("No hierarchy selected to restore; skipping.")
                continue

            confirm = input(
                "Are you sure you want to restore this hierarchy into the "
                "current models.json? [y/N]: "
            ).strip().lower()
            if confirm == "y":
                restore_hierarchy(
                    models_path,
                    models_current,
                    model_index,
                    hierarchy_index,
                    selected_backup_hierarchy,
                    backup_model,
                    valid_business_logic_ids,
                )
                print("Done.")
                return
            else:
                print("Restore cancelled; continuing to browse backups.")
        elif cmd == "n":
            current_backup_pos += 1
        elif cmd == "p":
            if current_backup_pos == 0:
                print("Already at newest backup; no newer backups. Press [n] to see older backups.")
                continue
            current_backup_pos -= 1
        elif cmd == "q":
            print("Aborting without changes.")
            return
        else:
            print("Unrecognized command; continuing with next backup.")
            current_backup_pos += 1

    print("\nReached the end of the backup list without restoring. No changes made.")


if __name__ == "__main__":
    main()