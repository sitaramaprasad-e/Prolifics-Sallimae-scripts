

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
    filtered = [r for r in rules if not r.get("archived", False)]
    removed_count = before_count - len(filtered)

    # Write updated rules
    with open(business_rules_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Removed {removed_count} archived rule(s).")
    print(f"[INFO] {len(filtered)} rule(s) remain in {business_rules_path}")

if __name__ == "__main__":
    main()