#!/usr/bin/env python3
import json
import os
from pathlib import Path
from collections import Counter
import argparse


def main():
    parser = argparse.ArgumentParser(description="Archive or report duplicate logic names.")
    parser.add_argument("--report-only", action="store_true", help="Only report duplicate names; do not archive.")
    args = parser.parse_args()

    print("[DEPRECATED] This script is deprecated and now operates in report-only mode. No changes will be made.")

    default_home = Path(os.path.expanduser("~/.model"))
    # No prompt when using argparse; always use default or env overrides
    model_home = default_home

    logics_path = model_home / "business_rules.json"
    if not logics_path.exists():
        print(f"[ERROR] business_rules.json not found at {logics_path}")
        return

    if args.report_only:
        print("[INFO] Running in report-only mode. No changes will be written.")

    # Load logic (support both legacy list and rooted {"version", "logics"} structure)
    try:
        with open(logics_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        return

    if isinstance(raw, dict) and isinstance(raw.get("logics"), list):
        logics = raw["logics"]
    elif isinstance(raw, list):
        logics = raw
    else:
        print("[ERROR] business_rules.json format invalid: expected a list or an object with a 'logics' array.")
        return

    if len(logics) == 0:
        print("[INFO] No logics found. Nothing to archive.")
        return

    # Find duplicate logic names (exact match)
    names = [r.get("name") for r in logics if isinstance(r, dict) and isinstance(r.get("name"), str)]
    counts = Counter(names)
    duplicate_names = {name for name, cnt in counts.items() if cnt > 1}

    if not duplicate_names:
        print("[INFO] No duplicate logic names found. Nothing archived.")
        return

    # For each duplicate name, keep the newest occurrence (last in the file)
    # and count only the older versions.
    name_to_indices = {}
    for idx, r in enumerate(logics):
        if not isinstance(r, dict):
            continue
        rn = r.get("name")
        if rn in duplicate_names:
            name_to_indices.setdefault(rn, []).append(idx)

    total_groups = len(duplicate_names)
    total_older_versions = 0

    for rn, idx_list in name_to_indices.items():
        idx_list = sorted(idx_list)
        if not idx_list:
            continue
        older_idxs = idx_list[:-1]
        total_older_versions += len(older_idxs)

    # Detailed listing of each duplicate group
    print("[INFO] Detailed duplicate groups (short id = last 3 hex chars of UUID):")
    for rn in sorted(name_to_indices.keys()):
        print(f"  - Logic name: {rn}")
        idx_list = sorted(name_to_indices[rn])
        for idx in idx_list:
            logic = logics[idx] if isinstance(logics[idx], dict) else {}
            raw_id = (
                logic.get("id")
                or "<no-id>"
            )
            short_id = "???"
            if isinstance(raw_id, str):
                hex_only = "".join(ch for ch in raw_id if ch.lower() in "0123456789abcdef")
                if len(hex_only) >= 3:
                    short_id = hex_only[-3:]
            ts = logic.get("timestamp") or logic.get("modified_at") or logic.get("updated_at") or "<no-timestamp>"
            print(f"      {short_id}\u00b7 {raw_id}  [ts: {ts}]")

    print(f"[INFO] Duplicate logic names detected (groups): {total_groups}")
    print(f"[INFO] Older versions identified (would have been archived): {total_older_versions}")
    print("[INFO] No changes were made due to deprecation.")

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
