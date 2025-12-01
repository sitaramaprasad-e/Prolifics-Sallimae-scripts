import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Any, Dict, List
import copy


def print_help() -> None:
    print("=== compare_logics_to_current.py help ===")
    print("Usage: python compare_logics_to_current.py [options]")
    print("Options:")
    print("  -h, --h, --help   Show this help message and exit.")
    print()
    print("This tool compares the *logics* in the current business_rules.json")
    print("to logics in each business_rules.json.bak-* snapshot in the same directory.")
    print()
    print("For each backup snapshot, it shows:")
    print("  * Removed logics (present in this backup, not in current)")
    print("  * New logics (present in current, not in this backup) – optional")
    print()
    print("Interactive commands while browsing backups:")
    print("  [n]ext     Move to the next (older) backup")
    print("  [p]revious Move to the previous (newer) backup")
    print("  [r]estore  Restore a removed logic from this backup into current")
    print("  [q]uit     Exit the tool")
    print()


def prompt_with_default(message: str, default: str) -> str:
    resp = input(f"{message} [{default}]: ").strip()
    return resp or default


def load_business_rules(path: Path) -> List[Dict[str, Any]]:
    """Load business_rules.json or a backup, returning a list of logic objects.

    Supports:
      * Legacy shape: [ {...}, {...} ]
      * New shape:    { "version": N, "logics": [ {...}, {...} ] }
    """
    if not path.exists():
        raise FileNotFoundError(f"business_rules.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and isinstance(data.get("logics"), list):
        return data["logics"]
    if isinstance(data, list):
        return data

    raise ValueError(
        f"{path} is not in a supported format (expected a list or an object with a 'logics' array)."
    )


def list_backup_files(rules_dir: Path):
    backups = [
        p for p in rules_dir.iterdir()
        if p.is_file() and p.name.startswith("business_rules.json.bak-")
    ]
    # Newest first (lexical sort by timestamp suffix)
    backups.sort(key=lambda p: p.name, reverse=True)
    return backups


def parse_backup_timestamp(backup_name: str) -> str:
    try:
        ts_str = backup_name.split("business_rules.json.bak-")[-1]
        dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "<unknown>"


def build_rule_map(rules: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    rule_map: Dict[str, Dict[str, Any]] = {}
    for r in rules:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        if isinstance(rid, str) and rid:
            rule_map[rid] = r
    return rule_map


def migrate_logic_for_current(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Lightly migrate a logic object from an old backup into current schema.

    This mirrors the important bits from migrate_jsons.py for a single rule:
      * doc_rule_id      -> doc_logic_id (if present)
      * from_step_id     -> from_logic_id (on links)
      * from_step        -> from_logic
      * rule_category    -> category
      * rule_name        -> name
      * rule_purpose     -> purpose
      * rule_spec        -> spec
      * Drop transient fields: doc_match_score, doc_logic_id, dmn_expression,
        __modeloverlay / __modelOverlay
    """
    new_rule = copy.deepcopy(rule)

    # Top-level field renames
    rules_field_renames = {
        "doc_rule_id": "doc_logic_id",
        "from_step": "from_logic",
        "rule_category": "category",
        "rule_name": "name",
        "rule_purpose": "purpose",
        "rule_spec": "spec",
    }

    for old_key, new_key in rules_field_renames.items():
        if old_key in new_rule:
            value = new_rule.get(old_key)
            # Only overwrite if target missing or "empty-ish"
            if new_key not in new_rule or new_rule[new_key] in (None, "", []):
                new_rule[new_key] = value
            new_rule.pop(old_key, None)

    # Nested links: from_step_id -> from_logic_id
    links = new_rule.get("links")
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            if "from_step_id" in link:
                value = link.get("from_step_id")
                if "from_logic_id" not in link or link["from_logic_id"] in (None, "", []):
                    link["from_logic_id"] = value
                link.pop("from_step_id", None)

    # Drop transient / scoring fields
    for obsolete_key in ("doc_match_score", "doc_logic_id", "dmn_expression", "__modeloverlay", "__modelOverlay"):
        if obsolete_key in new_rule:
            new_rule.pop(obsolete_key, None)

    return new_rule


def diff_logics_for_backup(
    current_rules,
    backup_rules,
    show_new: bool,
    max_rules_to_show: int = 100,
) -> None:
    """Print concise logic-level diffs between backup and current."""
    current_map = build_rule_map(current_rules)
    backup_map = build_rule_map(backup_rules)

    current_ids = set(current_map.keys())
    backup_ids = set(backup_map.keys())

    removed_ids = sorted(backup_ids - current_ids)
    new_ids = sorted(current_ids - backup_ids)

    if not removed_ids and (not show_new or not new_ids):
        print("No logic differences vs current for this snapshot.")
        return

    print(f"Logics removed since this backup (in backup, not in current): {len(removed_ids)}")
    ids_to_show_removed = removed_ids[:max_rules_to_show]
    for idx, rid in enumerate(ids_to_show_removed, start=1):
        rule = backup_map.get(rid) or {}
        name = rule.get("name") or rule.get("rule_name") or "<unnamed>"
        kind = rule.get("kind") or "<unknown-kind>"
        archived = rule.get("archived")
        archived_display = "archived=true" if archived else "archived=false"
        print(f"  [-{idx}] {name} (id={rid}, kind={kind}, {archived_display})")
    if len(removed_ids) > max_rules_to_show:
        print(f"  ... and {len(removed_ids) - max_rules_to_show} more removed logic(s)")

    if show_new:
        print(f"\nLogics added since this backup (in current, not in this backup): {len(new_ids)}")
        ids_to_show_new = new_ids[:max_rules_to_show]
        for idx, rid in enumerate(ids_to_show_new, start=1):
            rule = current_map.get(rid) or {}
            name = rule.get("name") or "<unnamed>"
            kind = rule.get("kind") or "<unknown-kind>"
            archived = rule.get("archived")
            archived_display = "archived=true" if archived else "archived=false"
            print(f"  [+{idx}] {name} (id={rid}, kind={kind}, {archived_display})")
        if len(new_ids) > max_rules_to_show:
            print(f"  ... and {len(new_ids) - max_rules_to_show} more new logic(s)")


def restore_logic_from_backup_interactive(
    current_root,
    current_rules,
    backup_rules,
    logics_path: Path,
    original_mtime: float,
) -> float:
    """Allow restoring a removed logic (present in backup, missing in current).

    Returns updated mtime (or original_mtime if no change).
    """
    current_map = build_rule_map(current_rules)
    backup_map = build_rule_map(backup_rules)

    current_ids = set(current_map.keys())
    backup_ids = set(backup_map.keys())

    removed_ids = sorted(backup_ids - current_ids)
    if not removed_ids:
        print("No removed logics in this backup that can be restored (they all still exist in current).")
        return original_mtime

    print("\nRemoved logics that can be restored from this backup:")
    candidates: List[str] = []
    for idx, rid in enumerate(removed_ids, start=1):
        rule = backup_map.get(rid) or {}
        name = rule.get("name") or rule.get("rule_name") or "<unnamed>"
        kind = rule.get("kind") or "<unknown-kind>"
        archived = rule.get("archived")
        archived_display = "archived=true" if archived else "archived=false"
        print(f"  [{idx}] {name} (id={rid}, kind={kind}, {archived_display})")
        candidates.append(rid)

    selection = prompt_with_default(
        "Select logic number to restore (or 0 to cancel)",
        "0",
    ).strip()

    try:
        sel_idx = int(selection)
    except ValueError:
        sel_idx = 0

    if sel_idx <= 0 or sel_idx > len(candidates):
        print("Restore cancelled.")
        return original_mtime

    selected_id = candidates[sel_idx - 1]
    backup_rule = backup_map.get(selected_id)

    if not backup_rule:
        print("Selected logic not found in backup map; nothing to restore.")
        return original_mtime

    # Optimistic concurrency: ensure file has not changed since we loaded it
    try:
        current_mtime = logics_path.stat().st_mtime
    except Exception as e:
        print(f"ERROR: Could not verify business_rules.json mtime ({e}); refusing to restore.")
        return original_mtime

    if original_mtime and current_mtime != original_mtime:
        print(
            "ERROR: business_rules.json has changed on disk since this tool was started.\n"
            "Refusing to write changes. Please re-run the tool on the latest file."
        )
        return original_mtime

    # Prepare migrated copy of the rule
    restored_rule = migrate_logic_for_current(backup_rule)
    rid = restored_rule.get("id")
    if not isinstance(rid, str) or not rid.strip():
        print(
            "ERROR: Cannot restore logic because it lacks a valid 'id' after migration; "
            "leaving business_rules.json unchanged."
        )
        return original_mtime

    if any(isinstance(r, dict) and r.get("id") == rid for r in current_rules):
        print("Logic with this id already exists in current; nothing to restore.")
        return original_mtime

    current_rules.append(restored_rule)

    # Persist updated root
    try:
        with logics_path.open("w", encoding="utf-8") as f:
            json.dump(current_root, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        print(f"ERROR: Failed to write updated business_rules.json: {e}")
        return original_mtime

    try:
        new_mtime = logics_path.stat().st_mtime
    except Exception:
        new_mtime = original_mtime

    name = restored_rule.get("name") or restored_rule.get("rule_name") or "<unnamed>"
    print(
        f"Restored logic '{name}' (id={rid}) "
        f"from backup into current business_rules.json."
    )
    return new_mtime


def main():
    if any(arg in ("-h", "--h", "--help") for arg in sys.argv[1:]):
        print_help()
        return

    default_dir = str(Path("~/.model").expanduser())
    print("=== Compare logics in business_rules.json to backups ===")
    rules_dir_str = prompt_with_default(
        "Enter path to jsons directory containing business_rules.json",
        default_dir,
    )
    rules_dir = Path(rules_dir_str).expanduser()
    logics_path = rules_dir / "business_rules.json"

    # Load current root and normalize to list of rules
    try:
        with logics_path.open("r", encoding="utf-8") as f:
            current_root = json.load(f)
    except Exception as e:
        print(f"Error loading business_rules.json: {e}")
        return

    if isinstance(current_root, dict) and isinstance(current_root.get("logics"), list):
        current_rules = current_root["logics"]
    elif isinstance(current_root, list):
        current_rules = current_root
    else:
        print("business_rules.json is not in a supported format (expected a list or an object with a 'logics' array).")
        return

    try:
        original_mtime = logics_path.stat().st_mtime
    except Exception as e:
        print(f"Warning: could not read mtime for optimistic concurrency ({e}). Continuing without it.")
        original_mtime = 0.0

    backups = list_backup_files(rules_dir)
    if not backups:
        print("No business_rules.json.bak-* files found in this directory.")
        return

    print(f"\nFound {len(backups)} backup files. Starting from the most recent.")

    mode_choice = prompt_with_default(
        "Show [r]emoved logics only, or [b]oth removed and new logics? (r/b)",
        "r",
    ).lower()
    show_new = mode_choice.startswith("b")

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
            backup_rules = load_business_rules(backup_path)
        except Exception as e:
            print(f"  Error loading this backup: {e}")
            cmd = input("Command: [n]ext, [p]revious, [q]uit: ").strip().lower() or "n"
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
                print("Exiting.")
                return
            continue

        # Print logic-level diff
        diff_logics_for_backup(current_rules, backup_rules, show_new)

        cmd = input(
            "\nCommand: [n]ext (older), [p]revious (newer), [r]estore logic, [q]uit: "
        ).strip().lower() or "n"

        if cmd == "n":
            current_backup_pos += 1
        elif cmd == "p":
            if current_backup_pos == 0:
                print("Already at newest backup; no newer backups. Press [n] to see older backups.")
                continue
            current_backup_pos -= 1
        elif cmd == "r":
            original_mtime = restore_logic_from_backup_interactive(
                current_root=current_root,
                current_rules=current_rules,
                backup_rules=backup_rules,
                logics_path=logics_path,
                original_mtime=original_mtime,
            )
            # Stay on this backup after restore attempt
            continue
        elif cmd == "q":
            print("Exiting.")
            return
        else:
            print("Unrecognized command; moving to next backup.")
            current_backup_pos += 1

    print("\nReached the end of the backup list. Done.")


if __name__ == "__main__":
    main()