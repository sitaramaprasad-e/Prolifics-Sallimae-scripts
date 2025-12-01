

#!/usr/bin/env python3
"""Remove invalid categories from business_rules.json.

This tool scans business_rules.json for logic entries whose
`category` field does **not** correspond to a valid (leaf) category
in rule_categories.json. It will:

1. List all offending logics (id, name, owner, invalid category).
2. Prompt you to optionally strip those invalid categories by
   setting `logic["category"] = ""` for each.
3. Save a timestamped backup of business_rules.json before writing
   any changes.

Run from the repo root, e.g.:

    python -m tools.patch.remove_invalid_categories

or

    python tools/patch/remove_invalid_categories.py
"""

import json
import os
import sys
import datetime
from typing import Any, Dict, List, Set

# Try to re-use shared config/constants from categorise_logic if available
try:
    # tools/categorise_logic.py should define these
    from tools.categorise_logic import (  # type: ignore
        CATEGORIES_JSON as _CATEGORIES_JSON,
        BUSINESS_RULES_JSON as _BUSINESS_RULES_JSON,
        _is_leaf_category as shared_is_leaf_category,
    )
except Exception:  # pragma: no cover - defensive fallback
    _CATEGORIES_JSON = None
    _BUSINESS_RULES_JSON = None
    shared_is_leaf_category = None


# ----------------------------
# Paths with sensible defaults
# ----------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Default locations (used if we can't import from categorise_logic)
DEFAULT_BUSINESS_RULES_JSON = os.path.join(REPO_ROOT, "business_rules.json")
DEFAULT_CATEGORIES_JSON = os.path.join(REPO_ROOT, "rule_categories.json")

BUSINESS_RULES_JSON = _BUSINESS_RULES_JSON or DEFAULT_BUSINESS_RULES_JSON
CATEGORIES_JSON = _CATEGORIES_JSON or DEFAULT_CATEGORIES_JSON


# ----------------------------
# Utility helpers
# ----------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def make_backup_path(path: str) -> str:
    base_dir = os.path.dirname(path)
    base_name = os.path.basename(path)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(base_dir, f"{base_name}.backup-{ts}")


# ----------------------------
# Category helpers
# ----------------------------

if shared_is_leaf_category is not None:
    # Re-use the implementation from categorise_logic so we stay in sync
    def is_leaf_category(cat: Dict[str, Any]) -> bool:
        return shared_is_leaf_category(cat)
else:

    def is_leaf_category(cat: Dict[str, Any]) -> bool:
        """Fallback heuristic for leaf vs group categories.

        This mirrors the logic used in tools/categorise_logic._is_leaf_category.
        """

        if not isinstance(cat, dict):
            return False

        type_val = str(cat.get("type", "")).strip().lower()
        kind_val = str(cat.get("kind", "")).strip().lower()

        if type_val in {"group", "category_group", "grouping"}:
            return False
        if kind_val in {"group", "category_group", "grouping"}:
            return False
        if bool(cat.get("isGroup")) or bool(cat.get("is_group")):
            return False

        # If this category has nested children/categories, treat it as a group
        if isinstance(cat.get("children"), list) and cat["children"]:
            return False
        if isinstance(cat.get("categories"), list) and cat["categories"]:
            return False

        # Otherwise, we treat it as a leaf
        return True


def collect_valid_category_names(categories_doc: Any) -> Set[str]:
    """Return the set of valid *leaf* category names from rule_categories.json."""

    valid: Set[str] = set()

    if not isinstance(categories_doc, dict):
        return valid

    cats = categories_doc.get("categories")
    if not isinstance(cats, list):
        return valid

    for cat in cats:
        if not isinstance(cat, dict):
            continue
        if not is_leaf_category(cat):
            continue
        name = cat.get("name")
        if isinstance(name, str) and name.strip():
            valid.add(name.strip())

    return valid


# ----------------------------
# Core logic
# ----------------------------


def find_invalid_categories(business_rules: Any, valid_names: Set[str]) -> List[Dict[str, Any]]:
    """Return a list of logic entries whose `category` value is invalid.

    Each item in the returned list is the actual logic dict (so we can mutate
    it later if the user chooses to fix them).
    """

    invalid: List[Dict[str, Any]] = []

    if not isinstance(business_rules, dict):
        return invalid

    logics = business_rules.get("logics")
    if not isinstance(logics, list):
        return invalid

    for logic in logics:
        if not isinstance(logic, dict):
            continue
        cat_val = logic.get("category")
        if not isinstance(cat_val, str):
            continue
        category = cat_val.strip()
        if category == "":
            continue

        if category not in valid_names:
            invalid.append(logic)

    return invalid


def print_invalid_report(invalid_logics: List[Dict[str, Any]]) -> None:
    if not invalid_logics:
        print("No invalid categories found. ✅")
        return

    print("Found the following logics with invalid categories:\n")
    for logic in invalid_logics:
        logic_id = logic.get("id", "<no-id>")
        name = logic.get("name", "<no-name>")
        owner = logic.get("owner", "") or "<no-owner>"
        category = (logic.get("category") or "").strip() or "<empty>"
        print(f"- id={logic_id} | name={name} | owner={owner} | category={category}")

    print("\nTotal logics with invalid categories:", len(invalid_logics))


def strip_invalid_categories(invalid_logics: List[Dict[str, Any]]) -> None:
    for logic in invalid_logics:
        # Set category to empty string as requested
        logic["category"] = ""


# ----------------------------
# Main entrypoint
# ----------------------------


def main(argv: List[str]) -> int:
    # Allow optional explicit paths via CLI, but keep default behaviour simple.
    br_path = BUSINESS_RULES_JSON
    cat_path = CATEGORIES_JSON

    # Prompt for model home (defaults to ~/.model)
    default_model_home = os.path.expanduser("~/.model")
    user_input = input(f"Model home directory? [{default_model_home}]: ").strip()
    model_home = user_input or default_model_home

    # Recalculate paths based on model_home
    br_path = os.path.join(model_home, "business_rules.json")
    cat_path = os.path.join(model_home, "rule_categories.json")

    if len(argv) >= 2:
        br_path = argv[1]
    if len(argv) >= 3:
        cat_path = argv[2]

    print(f"Using business_rules.json: {br_path}")
    print(f"Using rule_categories.json: {cat_path}")

    if not os.path.isfile(br_path):
        print(f"Error: business rules file not found: {br_path}", file=sys.stderr)
        return 1
    if not os.path.isfile(cat_path):
        print(f"Error: categories file not found: {cat_path}", file=sys.stderr)
        return 1

    business_rules = load_json(br_path)
    categories_doc = load_json(cat_path)

    valid_names = collect_valid_category_names(categories_doc)
    print(f"Valid leaf categories detected: {len(valid_names)}")

    invalid_logics = find_invalid_categories(business_rules, valid_names)
    print_invalid_report(invalid_logics)

    if not invalid_logics:
        return 0

    ans = input("\nDo you want to strip these invalid categories (set category to \"\")? [y/N]: ").strip().lower()
    if ans not in {"y", "yes"}:
        print("No changes made.")
        return 0

    # Backup before mutating on disk
    backup_path = make_backup_path(br_path)
    dump_json(backup_path, business_rules)
    print(f"Backup of business_rules.json written to: {backup_path}")

    # Apply fix in-memory and write back
    strip_invalid_categories(invalid_logics)
    dump_json(br_path, business_rules)
    print(f"Updated business rules written to: {br_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv))