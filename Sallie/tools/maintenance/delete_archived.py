

#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime
import zipfile

def main():
    default_home = Path(os.path.expanduser("~/.model"))
    user_input = input(f"Enter model home directory [{default_home}]: ").strip()
    model_home = Path(user_input) if user_input else default_home

    business_rules_path = model_home / "business_rules.json"
    if not business_rules_path.exists():
        print(f"[ERROR] business_rules.json not found at {business_rules_path}")
        return

    # Create a timestamped ZIP backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_backup_path = business_rules_path.parent / f"business_rules_backup_{timestamp}.zip"
    try:
        with zipfile.ZipFile(zip_backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(business_rules_path, arcname="business_rules.json")
        print(f"[INFO] ZIP backup created: {zip_backup_path}")
    except Exception as e:
        print(f"[WARN] Could not create ZIP backup: {e}")

    # Load and filter rules
    with open(business_rules_path, "r", encoding="utf-8") as f:
        try:
            rules = json.load(f)
            if not isinstance(rules, list):
                print(f"[ERROR] business_rules.json format invalid: expected a list.")
                return
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse JSON: {e}")
            return

    before_count = len(rules)
    # Remove archived rules (existing behavior)
    filtered = [r for r in rules if not r.get("archived", False)]
    removed_count = before_count - len(filtered)

    # --- Dangling link cleanup ---
    # Build a set of valid rule IDs from the remaining rules
    valid_ids = set()
    for r in filtered:
        rid = r.get("id")
        if isinstance(rid, str) and rid.strip():
            valid_ids.add(rid.strip())

    def _clean_links(rule_obj):
        links = rule_obj.get("links")
        if not isinstance(links, list):
            return 0
        keep = []
        removed = 0
        for l in links:
            if not isinstance(l, dict):
                # Non-dict entries are invalid; drop them
                removed += 1
                continue
            fsid = (l.get("from_step_id") or "").strip()
            # Drop links where from_step_id is missing or not a valid id remaining in the file
            if not fsid or fsid not in valid_ids:
                removed += 1
                continue
            keep.append(l)
        rule_obj["links"] = keep
        return removed

    total_links_removed = 0
    for r in filtered:
        try:
            total_links_removed += _clean_links(r)
        except Exception:
            # Be robust; if links are malformed, skip cleaning that rule
            pass

    # Write updated rules including cleaned links
    with open(business_rules_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Removed {removed_count} archived rule(s).")
    print(f"[INFO] Removed {total_links_removed} dangling link(s) where from_step_id no longer matches an existing rule id.")
    print(f"[INFO] {len(filtered)} rule(s) remain in {business_rules_path}")

if __name__ == "__main__":
    main()