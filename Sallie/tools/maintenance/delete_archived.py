#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime
import zipfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--report-only", action="store_true")
args = parser.parse_args()
report_only = args.report_only

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

    archived_rules = [r for r in rules if r.get("archived", False)]

    before_count = len(rules)
    # Remove archived rules (existing behavior)
    filtered = [r for r in rules if not r.get("archived", False)]
    removed_count = before_count - len(filtered)

    # Precompute IDs for archived rules (those we are deleting)
    archived_ids = set()
    for r in archived_rules:
        rid = r.get("id")
        if isinstance(rid, str) and rid.strip():
            archived_ids.add(rid.strip())

    # --- Dangling link cleanup ---
    # Build a set of ALL rule IDs (including archived).
    all_ids = set()
    for r in rules:
        rid = r.get("id")
        if isinstance(rid, str) and rid.strip():
            all_ids.add(rid.strip())

    def _clean_links(rule_obj, all_ids, archived_ids, dry_run=False):
        links = rule_obj.get("links")
        if not isinstance(links, list):
            return [], 0
        removed_links = []
        for l in links:
            if not isinstance(l, dict):
                removed_links.append(l)
                continue
            fsid = (l.get("from_step_id") or "").strip()
            # Always treat links whose source ID does not exist in all_ids as dangling
            if not fsid or fsid not in all_ids:
                removed_links.append(l)
                continue
            # When we are actually deleting archived rules (not report-only),
            # also remove links whose source points at an archived rule ID,
            # because those rules will no longer exist after this script runs.
            if not dry_run and fsid in archived_ids:
                removed_links.append(l)
        if not dry_run:
            rule_obj["links"] = [l for l in links if l not in removed_links]
        return removed_links, len(removed_links)

    total_links_removed = 0
    for r in filtered:
        try:
            removed_links, count = _clean_links(r, all_ids, archived_ids, dry_run=report_only)
            total_links_removed += count
            if report_only and count > 0:
                if "id" in r:
                    print(f"[REPORT] Rule {r['id']} has {count} dangling link(s):")
                    for dl in removed_links:
                        print(f"         from_step_id={dl.get('from_step_id')} to_input={dl.get('to_input')}")
        except Exception:
            # Be robust; if links are malformed, skip cleaning that rule
            pass

    if report_only:
        print("[INFO] --report-only mode: no changes written.")
        print(f"[INFO] {len(archived_rules)} rule(s) would be deleted:")
        for r in archived_rules:
            print(f"       {r.get('id')}  {r.get('rule_name')}")
        print(f"[INFO] {total_links_removed} link(s) would be cleaned up where the from_step_id no longer matches an existing rule id or would refer to a rule being deleted.")
        return

    # Write updated rules including cleaned links
    with open(business_rules_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Deleted {removed_count} archived rule(s).")
    print(f"[INFO] Removed {total_links_removed} dangling link(s) where from_step_id no longer matches an existing rule id.")
    print(f"[INFO] {len(filtered)} rule(s) remain in {business_rules_path}")

if __name__ == "__main__":
    main()