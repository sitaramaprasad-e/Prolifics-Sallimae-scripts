#!/usr/bin/env python3
import json
import os
from pathlib import Path
from datetime import datetime
import zipfile
from collections import Counter
import argparse


def main():
    parser = argparse.ArgumentParser(description="Archive or report duplicate rule names.")
    parser.add_argument("--report-only", action="store_true", help="Only report duplicate names; do not archive.")
    args = parser.parse_args()

    default_home = Path(os.path.expanduser("~/.model"))
    # No prompt when using argparse; always use default or env overrides
    model_home = default_home

    business_rules_path = model_home / "business_rules.json"
    if not business_rules_path.exists():
        print(f"[ERROR] business_rules.json not found at {business_rules_path}")
        return

    if args.report_only:
        print("[INFO] Running in report-only mode. No changes will be written.")
    # Create a timestamped ZIP backup (same style as delete_archived.py)
    if not args.report_only:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_backup_path = business_rules_path.parent / f"business_rules_backup_{timestamp}.zip"
        try:
            with zipfile.ZipFile(zip_backup_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(business_rules_path, arcname="business_rules.json")
            print(f"[INFO] ZIP backup created: {zip_backup_path}")
        except Exception as e:
            print(f"[WARN] Could not create ZIP backup: {e}")

    # Load rules
    try:
        with open(business_rules_path, "r", encoding="utf-8") as f:
            rules = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        return

    if not isinstance(rules, list):
        print("[ERROR] business_rules.json format invalid: expected a list.")
        return

    if len(rules) == 0:
        print("[INFO] No rules found. Nothing to archive.")
        return

    # Find duplicate rule names (exact match)
    names = [r.get("rule_name") for r in rules if isinstance(r, dict) and isinstance(r.get("rule_name"), str)]
    counts = Counter(names)
    duplicate_names = {name for name, cnt in counts.items() if cnt > 1}

    if not duplicate_names:
        print("[INFO] No duplicate rule names found. Nothing archived.")
        return

    # Archive all rules whose name appears more than once
    total_matched = 0
    newly_archived = 0

    for r in rules:
        if not isinstance(r, dict):
            continue
        rn = r.get("rule_name")
        if rn in duplicate_names:
            total_matched += 1
            if not args.report_only:
                if not r.get("archived", False):
                    r["archived"] = True
                    newly_archived += 1

    if not args.report_only:
        with open(business_rules_path, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Duplicate rule names detected: {len(duplicate_names)}")
    print(f"[INFO] Rules matched by duplicate names: {total_matched}")
    print(f"[INFO] Newly archived rules: {newly_archived}")
    if args.report_only:
        print("[INFO] No rules were modified due to --report-only mode.")
    print(f"[INFO] Updated file: {business_rules_path}")

    # Optionally, list the duplicate names (brief)
    try:
        sample = sorted(list(duplicate_names))
        preview = ", ".join(sample[:10])
        more = "" if len(sample) <= 10 else f" (+{len(sample)-10} more)"
        print(f"[INFO] Duplicate names sample: {preview}{more}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
