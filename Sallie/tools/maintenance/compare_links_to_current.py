import json
import sys
from pathlib import Path
from datetime import datetime

from typing import Optional


def print_help() -> None:
    print("=== compareLinksToCurrent.py help ===")
    print("Usage: python compareLinksToCurrent.py [options]")
    print("Options:")
    print("  -h, --h, --help   Show this help message and exit.")
    print()
    print("This tool compares the links in the current business_rules.json")
    print("to links in each business_rules.json.bak-* snapshot in the same directory.")
    print()
    print("For each backup snapshot, it shows per logic:")
    print("  * New links (present in current, not in this backup)")
    print("  * Removed links (present in this backup, not in current)")
    print()
    print("Output is kept concise:")
    print("  * Only logics with differences are listed")
    print("  * Links are summarized on a single line")
    print("  * When there are many changed rules, only the first N are shown")
    print()
    print("Interactive commands while browsing backups:")
    print("  [n]ext     Move to the next (older) backup")
    print("  [p]revious Move to the previous (newer) backup")
    print("  [q]uit     Exit the tool")
    print()


def prompt_with_default(message: str, default: str) -> str:
    resp = input(f"{message} [{default}]: ").strip()
    return resp or default


def load_business_rules(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"business_rules.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both legacy list shape and rooted {"version", "logics"} structure
    if isinstance(data, dict) and isinstance(data.get("logics"), list):
        return data["logics"]
    if isinstance(data, list):
        return data

    raise ValueError(
        "business_rules.json is not in a supported format (expected a list or an object with a 'logics' array)."
    )


def list_backup_files(rules_dir: Path):
    backups = [
        p for p in rules_dir.iterdir()
        if p.is_file() and p.name.startswith("business_rules.json.bak-")
    ]
    backups.sort(key=lambda p: p.name, reverse=True)
    return backups


def parse_backup_timestamp(backup_name: str) -> str:
    try:
        ts_str = backup_name.split("business_rules.json.bak-")[-1]
        dt = datetime.strptime(ts_str, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "<unknown>"


def canonical_link(link: dict) -> tuple:
    """
    Represent a link in a canonical tuple form for comparison.
    We include all four key fields that define a link relationship.
    """
    return (
        link.get("from_output"),
        link.get("to_input"),
        link.get("kind"),
        link.get("from_logic_id"),
    )


def summarize_link(link: dict, from_name=None) -> str:
    """
    Produce a concise single-line string describing a link, optionally
    including the name of the rule referenced by from_logic_id.
    """
    base = (
        f"{link.get('from_output')} -> {link.get('to_input')} "
        f"(kind={link.get('kind')}, from={link.get('from_logic_id')}"
    )
    if from_name:
        base += f" [{from_name}]"
    base += ")"
    return base


def build_rule_map(rules: list) -> dict:
    """
    Build a mapping of logic_id -> logic_object for quick lookup.
    """
    rule_map = {}
    for r in rules:
        if not isinstance(r, dict):
            continue
        rid = r.get("id")
        if rid:
            rule_map[rid] = r
    return rule_map


def diff_links_for_backup(
    current_rules: list,
    backup_rules: list,
    show_new: bool,
    target_logic_id: Optional[str] = None,
    max_rules_to_show: int = 50,
    max_links_per_section: int = 5,
) -> None:
    """
    Compare links in backup_rules vs current_rules and print concise differences.
    """
    current_map = build_rule_map(current_rules)
    backup_map = build_rule_map(backup_rules)

    current_ids = set(current_map.keys())
    backup_ids = set(backup_map.keys())
    all_ids = sorted(current_ids | backup_ids)

    # Build a combined id -> name map for cross-referencing from_logic_id
    id_to_name: dict[str, str] = {}
    for rule in current_map.values():
        if not isinstance(rule, dict):
            continue
        rid = rule.get("id")
        name = rule.get("name")
        if rid and name and rid not in id_to_name:
            id_to_name[rid] = name
    for rule in backup_map.values():
        if not isinstance(rule, dict):
            continue
        rid = rule.get("id")
        name = rule.get("name")
        if rid and name and rid not in id_to_name:
            id_to_name[rid] = name

    changed_rules = []

    for rid in all_ids:
        if target_logic_id and rid != target_logic_id:
            continue

        current_rule = current_map.get(rid)
        backup_rule = backup_map.get(rid)

        current_links = current_rule.get("links") if current_rule else None
        backup_links = backup_rule.get("links") if backup_rule else None

        current_links = current_links or []
        backup_links = backup_links or []

        current_link_tuples = {canonical_link(l) for l in current_links}
        backup_link_tuples = {canonical_link(l) for l in backup_links}

        new_link_tuples = current_link_tuples - backup_link_tuples
        removed_link_tuples = backup_link_tuples - current_link_tuples

        if show_new:
            # Include rules with any change (new or removed links)
            if not new_link_tuples and not removed_link_tuples:
                continue
        else:
            # Removed-only mode: only include rules that lost links
            if not removed_link_tuples:
                continue

        changed_rules.append(
            (rid, current_rule, backup_rule, new_link_tuples, removed_link_tuples)
        )

    if not changed_rules:
        print("No link differences vs current for this snapshot.")
        return

    print(f"Logics with link differences vs current: {len(changed_rules)}")
    if len(changed_rules) > max_rules_to_show:
        print(f"(Showing first {max_rules_to_show} changed rules only)")
        changed_rules = changed_rules[:max_rules_to_show]

    for idx, (rid, current_rule, backup_rule, new_link_tuples, removed_link_tuples) in enumerate(changed_rules, start=1):
        name = None
        kind = None
        archived = None

        # Prefer current rule metadata, fall back to backup
        if current_rule:
            name = current_rule.get("name")
            kind = current_rule.get("kind")
            archived = current_rule.get("archived")
        elif backup_rule:
            name = backup_rule.get("name")
            kind = backup_rule.get("kind")
            archived = backup_rule.get("archived")

        name_display = name or "<unnamed>"
        kind_display = kind or "<unknown-kind>"
        archived_display = "archived=true" if archived else "archived=false"

        print(f"\n[{idx}] {name_display} (id={rid}, kind={kind_display}, {archived_display})")

        # Build reverse lookup to reconstruct link dicts from tuples
        def links_from_tuples(rule_obj, tuples):
            if not rule_obj:
                return []
            links = rule_obj.get("links") or []
            tuple_set = set(tuples)
            result = []
            for l in links:
                if canonical_link(l) in tuple_set:
                    result.append(l)
            return result

        new_links = links_from_tuples(current_rule, new_link_tuples)
        removed_links = links_from_tuples(backup_rule, removed_link_tuples)

        if show_new and new_links:
            print(f"    + {len(new_links)} new link(s) in current (not in this backup)")
            for l in new_links[:max_links_per_section]:
                from_id = l.get("from_logic_id")
                from_name = id_to_name.get(from_id) if from_id else None
                if from_id and from_id not in current_map:
                    print(f"        [WARN] from_logic_id {from_id} not found in current json")
                print(f"      + {summarize_link(l, from_name)}")
            if len(new_links) > max_links_per_section:
                print(f"      ... and {len(new_links) - max_links_per_section} more new link(s)")

        if removed_links:
            print(f"    - {len(removed_links)} removed link(s) (in this backup, not in current)")
            for l in removed_links[:max_links_per_section]:
                from_id = l.get("from_logic_id")
                from_name = id_to_name.get(from_id) if from_id else None
                if from_id and from_id not in current_map:
                    print(f"        [WARN] from_logic_id {from_id} not found in current json")
                print(f"      - {summarize_link(l, from_name)}")
            if len(removed_links) > max_links_per_section:
                print(f"      ... and {len(removed_links) - max_links_per_section} more removed link(s)")


# === Restore helper for removed links ===
def restore_link_from_backup_interactive(
    current_rules: list,
    backup_rules: list,
    logic_filter_id: Optional[str],
    logics_path: Path,
    original_mtime: float,
    current_root,
) -> float:
    """
    Allow the user to restore a single removed link from this backup into
    the current business_rules.json, subject to:
      1) Both ends existing in current (owning rule and from_logic_id rule).
      2) Translating old semantics (from_step_id -> from_logic_id).
      3) Respecting optimistic concurrency on business_rules.json.
    Returns the (possibly updated) mtime to use for subsequent operations.
    """
    current_map = build_rule_map(current_rules)
    backup_map = build_rule_map(backup_rules)

    current_ids = set(current_map.keys())
    backup_ids = set(backup_map.keys())
    all_ids = sorted(current_ids | backup_ids)

    # Helper: reconstruct links for a rule from a set of canonical tuples
    def links_from_tuples(rule_obj, tuples):
        if not rule_obj:
            return []
        links = rule_obj.get("links") or []
        tuple_set = set(tuples)
        result = []
        for l in links:
            if canonical_link(l) in tuple_set:
                result.append(l)
        return result

    # First, find all rules that have removed links
    candidates = []
    for rid in all_ids:
        if logic_filter_id and rid != logic_filter_id:
            continue

        current_rule = current_map.get(rid)
        backup_rule = backup_map.get(rid)

        # We can only restore into a rule that still exists in current and has backup info
        if not current_rule or not backup_rule:
            continue

        current_links = current_rule.get("links") or []
        backup_links = backup_rule.get("links") or []

        current_link_tuples = {canonical_link(l) for l in current_links}
        backup_link_tuples = {canonical_link(l) for l in backup_links}

        removed_link_tuples = backup_link_tuples - current_link_tuples
        if not removed_link_tuples:
            continue

        removed_links = links_from_tuples(backup_rule, removed_link_tuples)
        if not removed_links:
            continue

        candidates.append((rid, current_rule, removed_links))

    if not candidates:
        print("No removed links in this backup that can be considered for restore (given current filter).")
        return original_mtime

    print("\nRemoved links that can potentially be restored from this backup:")
    for idx, (rid, rule, removed_links) in enumerate(candidates, start=1):
        name = rule.get("name") or "<unnamed>"
        print(f"  [{idx}] {name} (id={rid}) – {len(removed_links)} removed link(s)")

    selection = prompt_with_default(
        "Select logic number to inspect/restore from (or 0 to cancel)",
        "0",
    ).strip()

    try:
        sel_idx = int(selection)
    except ValueError:
        sel_idx = 0

    if sel_idx <= 0 or sel_idx > len(candidates):
        print("Restore cancelled.")
        return original_mtime

    selected_rid, selected_rule, removed_links = candidates[sel_idx - 1]
    selected_name = selected_rule.get("name") or "<unnamed>"

    # Build map again to ensure we have up-to-date ends
    current_map = build_rule_map(current_rules)

    print(f"\nRemoved links for logic: {selected_name} (id={selected_rid})")
    restorable_indices = []
    for idx, link in enumerate(removed_links, start=1):
        from_id = link.get("from_logic_id") or link.get("from_step_id")
        from_rule = current_map.get(from_id) if from_id else None
        has_source = from_rule is not None
        has_target = selected_rid in current_map
        can_restore = has_source and has_target

        status = "OK " if can_restore else "SKIP"
        from_name = from_rule.get("name") if from_rule else None
        print(f"  [{idx}] {status} {summarize_link(link, from_name)}")

        if can_restore:
            restorable_indices.append(idx)

    if not restorable_indices:
        print("None of the removed links have both ends present in current; nothing to restore.")
        return original_mtime

    link_sel = prompt_with_default(
        "Select link number to restore (or 0 to cancel)",
        "0",
    ).strip()

    try:
        link_idx = int(link_sel)
    except ValueError:
        link_idx = 0

    if link_idx <= 0 or link_idx not in restorable_indices:
        print("Restore cancelled.")
        return original_mtime

    link_to_restore = removed_links[link_idx - 1]
    from_id = link_to_restore.get("from_logic_id") or link_to_restore.get("from_step_id")
    from_rule = current_map.get(from_id) if from_id else None

    if not from_rule or selected_rid not in current_map:
        print("Ends no longer valid in current; cannot restore this link.")
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

    # Build the new link, translating any legacy from_step_id -> from_logic_id
    # and ensuring we never lose the source id in the process.
    src = link_to_restore.get("from_logic_id") or link_to_restore.get("from_step_id")
    if not isinstance(src, str) or not src.strip():
        print(
            "ERROR: Cannot restore link because it lacks a valid from_logic_id/from_step_id; "
            "leaving business_rules.json unchanged."
        )
        return original_mtime

    new_link = dict(link_to_restore)
    # Always set from_logic_id to the resolved source id
    new_link["from_logic_id"] = src
    # Only now do we drop any legacy from_step_id field
    new_link.pop("from_step_id", None)

    # Sanity check: the restored link must have all required fields, none blank.
    required_keys = ("from_output", "to_input", "kind", "from_logic_id")
    missing = [k for k in required_keys if k not in new_link]
    blanks = []
    for k in required_keys:
        if k not in new_link:
            continue
        val = new_link.get(k)
        if not isinstance(val, str) or not val.strip():
            blanks.append(k)

    if missing or blanks:
        print(
            "ERROR: Refusing to restore link because it would be malformed.\n"
            f"  Missing required fields: {', '.join(missing) if missing else 'none'}\n"
            f"  Blank/non-string required fields: {', '.join(blanks) if blanks else 'none'}"
        )
        return original_mtime

    # Avoid adding duplicate if somehow present
    existing_links = selected_rule.get("links") or []
    if any(canonical_link(l) == canonical_link(new_link) for l in existing_links):
        print("Link already exists in current; nothing to restore.")
        return original_mtime

    selected_rule.setdefault("links", []).append(new_link)

    # Persist the updated root to disk
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

    print(
        f"Restored link for logic '{selected_name}' (id={selected_rid}) "
        f"from backup into current business_rules.json."
    )
    return new_mtime


def main():
    if any(arg in ("-h", "--h", "--help") for arg in sys.argv[1:]):
        print_help()
        return

    default_dir = str(Path("~/.model").expanduser())
    print("=== Compare links in business_rules.json to backups ===")
    rules_dir_str = prompt_with_default(
        "Enter path to jsons directory containing business_rules.json",
        default_dir,
    )
    rules_dir = Path(rules_dir_str).expanduser()
    logics_path = rules_dir / "business_rules.json"

    try:
        with logics_path.open("r", encoding="utf-8") as f:
            current_root = json.load(f)
    except Exception as e:
        print(f"Error loading business_rules.json: {e}")
        return

    # Normalize current payload to a list of logic objects and keep root for writing back
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

    # Optional filter: focus on a single logic by name prefix
    current_map = build_rule_map(current_rules)
    logic_filter_id: Optional[str] = None

    prefix = prompt_with_default(
        "Filter to a single logic by name prefix (press Enter for all)",
        "",
    ).strip()

    if prefix:
        lowered = prefix.lower()
        candidates = [
            (rid, rule)
            for rid, rule in current_map.items()
            if isinstance(rule, dict)
            and isinstance(rule.get("name"), str)
            and rule.get("name").lower().startswith(lowered)
        ]

        if not candidates:
            print(f"No logics found with name starting with '{prefix}'. Showing all.")
        else:
            candidates.sort(key=lambda item: (item[1].get('name') or '').lower())
            print("\nMatching logics:")
            for idx, (rid, rule) in enumerate(candidates, start=1):
                name = rule.get("name") or "<unnamed>"
                kind = rule.get("kind") or "<unknown-kind>"
                print(f"  [{idx}] {name} (id={rid}, kind={kind})")

            selection = prompt_with_default(
                "Select logic number (or 0 for all)",
                "1",
            ).strip()

            try:
                sel_idx = int(selection)
            except ValueError:
                sel_idx = 1

            if sel_idx <= 0 or sel_idx > len(candidates):
                print("Showing all logics.")
            else:
                logic_filter_id = candidates[sel_idx - 1][0]
                chosen = candidates[sel_idx - 1][1]
                chosen_name = chosen.get("name") or "<unnamed>"
                print(f"Filtering diffs to logic: {chosen_name} (id={logic_filter_id})")

    backups = list_backup_files(rules_dir)
    if not backups:
        print("No business_rules.json.bak-* files found in this directory.")
        return

    print(f"\nFound {len(backups)} backup files. Starting from the most recent.")
    current_backup_pos = 0

    mode_choice = prompt_with_default(
        "Show [r]emoved links only, or [b]oth removed and new links? (r/b)",
        "r",
    ).lower()
    show_new = mode_choice.startswith("b")

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
                raw_backup = json.load(f)
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

        # Normalize backup payload to a list of logic objects
        if isinstance(raw_backup, dict) and isinstance(raw_backup.get("logics"), list):
            backup_rules = raw_backup["logics"]
        elif isinstance(raw_backup, list):
            backup_rules = raw_backup
        else:
            print("  Backup file does not contain a list of logics or an object with a 'logics' array; skipping.")
            cmd = input("Command: [n]ext, [p]revious, [q]uit: ").strip().lower() or "n"
            if cmd == "n":
                current_backup_pos += 1
            elif cmd == "p":
                if current_backup_pos == 0:
                    print("Already at newest backup; no newer backups. Press [n] to see older backups.")
                    continue
                current_backup_pos -= 1
            elif cmd == "q":
                print("Exiting.")
                return
            continue

        # Print concise diff of links vs current for this snapshot
        diff_links_for_backup(current_rules, backup_rules, show_new, target_logic_id=logic_filter_id)

        cmd = input(
            "\nCommand: [n]ext (older), [p]revious (newer), [r]estore link, [q]uit: "
        ).strip().lower() or "n"

        if cmd == "n":
            current_backup_pos += 1
        elif cmd == "p":
            if current_backup_pos == 0:
                print("Already at newest backup; no newer backups. Press [n] to see older backups.")
                continue
            current_backup_pos -= 1
        elif cmd == "r":
            original_mtime = restore_link_from_backup_interactive(
                current_rules,
                backup_rules,
                logic_filter_id,
                logics_path,
                original_mtime,
                current_root,
            )
            # Stay on the same backup after attempting restore
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
