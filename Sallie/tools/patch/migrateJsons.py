import argparse
import copy
import difflib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


@dataclass
class MigrationResult:
    file_path: Path
    changed: bool
    backed_up_to: Optional[Path]
    error: Optional[str] = None
    summary: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate JSON files from an old schema to a new schema, "
            "taking timestamped backups and updating in-place."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=None,
        help="Root directory containing JSON files to migrate (default: prompt, ~/.model)",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for JSON files (default: %(default)s)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories when searching for JSON files",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory where backups will be written. "
            "If omitted, backups are placed next to the original file."
        ),
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview changes without writing or taking backups",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (can be specified multiple times)",
    )
    return parser.parse_args()


def find_json_files(root: Path, pattern: str, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    # Pretty-print to keep the files readable and stable
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def decode_multi_json(text: str) -> list[Any]:
    """Decode one or more JSON documents from a string.

    Supports files that contain a single JSON value or multiple JSON values
    concatenated together (e.g. `{...}{...}` or `{...}\n{...}`). Returns a
    list of parsed documents in order.
    """

    decoder = json.JSONDecoder()
    idx = 0
    length = len(text)
    docs: list[Any] = []

    while idx < length:
        # Skip whitespace between JSON documents
        while idx < length and text[idx].isspace():
            idx += 1
        if idx >= length:
            break

        obj, end = decoder.raw_decode(text, idx)
        docs.append(obj)
        idx = end

    return docs


def migrate_document(data: Any) -> Tuple[Any, bool, Optional[str]]:
    """Apply schema migration to the JSON document.

    Currently supports the following transformations (idempotent):
    * ruleCategoryGroups  -> categoryGroups (rule_categories.json)
    * ruleCategories      -> categories (rule_categories.json)
    * examplarRuleIds / exemplarRuleIds -> exemplarLogicIds (on category objects)
    * doc_rule_id        -> doc_logic_id (business_rules.json)
    * from_step_id       -> from_logic_id (business_rules.json)
    * from_step          -> from_logic (business_rules.json)
    * rule_category      -> category (business_rules.json)
    * rule_name          -> name (business_rules.json)
    * rule_purpose       -> purpose (business_rules.json)
    * rule_spec          -> spec (business_rules.json)
    * rule_id           -> logic_id (supporting-decisions-suggestions.json)
    * businessLogicIds   -> logicIds (models.json)
    * topDecisionId      -> topId (models.json hierarchies)
    """

    changed = False
    summary_parts: list[str] = []
    # Deep copy so we can safely mutate nested structures
    new_data = copy.deepcopy(data)

    def _rename_rule_id_to_logic_id(obj: Any) -> int:
        """Recursively rename rule_id -> logic_id in any dict-like structure.

        This is primarily intended for supporting-decisions-suggestions.json where
        suggestion objects contain rule_id, but is safe and idempotent more broadly.
        Returns the number of objects updated.
        """

        updated = 0

        if isinstance(obj, dict):
            # If this dict looks like a suggestion, it may have rule_id/logic_id
            if "rule_id" in obj:
                value = obj.get("rule_id")
                # Only overwrite logic_id if it is missing or empty-like
                if "logic_id" not in obj or obj["logic_id"] in (None, "", []):
                    obj["logic_id"] = value
                obj.pop("rule_id", None)
                updated += 1

            # Recurse into all values
            for v in obj.values():
                updated += _rename_rule_id_to_logic_id(v)

        elif isinstance(obj, list):
            for item in obj:
                updated += _rename_rule_id_to_logic_id(item)

        return updated

    # -----------------------------
    # Object-shaped documents
    # -----------------------------
    top_groups_renamed = False
    top_categories_renamed = False
    exemplar_updates = 0
    suggestions_renamed = 0
    models_logic_ids_renamed = 0
    hierarchy_top_ids_renamed = 0

    if isinstance(new_data, dict):
        # --- Top-level key renames ---
        # ruleCategoryGroups -> categoryGroups
        if "ruleCategoryGroups" in new_data:
            # Only rename if the new key is not already present
            if "categoryGroups" not in new_data:
                new_data["categoryGroups"] = new_data.pop("ruleCategoryGroups")
                top_groups_renamed = True
                changed = True
            else:
                # If both exist, drop the old one to avoid ambiguity
                new_data.pop("ruleCategoryGroups", None)
                top_groups_renamed = True
                changed = True

        # ruleCategories -> categories
        if "ruleCategories" in new_data:
            if "categories" not in new_data:
                new_data["categories"] = new_data.pop("ruleCategories")
                top_categories_renamed = True
                changed = True
            else:
                new_data.pop("ruleCategories", None)
                top_categories_renamed = True
                changed = True

        # --- Nested field rename inside categories ---
        categories_key = None
        if "categories" in new_data and isinstance(new_data["categories"], list):
            categories_key = "categories"
        elif "ruleCategories" in new_data and isinstance(new_data["ruleCategories"], list):
            # In case this function runs before the top-level rename
            categories_key = "ruleCategories"

        if categories_key is not None:
            for cat in new_data.get(categories_key, []):
                if not isinstance(cat, dict):
                    continue

                # Some files use the misspelled key `examplarRuleIds`,
                # but the desired mapping is to `exemplarLogicIds`.
                old_keys = [
                    k for k in ("examplarRuleIds", "exemplarRuleIds") if k in cat
                ]
                if not old_keys:
                    continue

                # Pick the first old key present and move its value
                old_key = old_keys[0]
                value = cat.get(old_key)

                # Only update if the new key is not already there or is empty
                if "exemplarLogicIds" not in cat or cat["exemplarLogicIds"] in (None, []):
                    cat["exemplarLogicIds"] = value
                # Drop all legacy keys
                for k in old_keys:
                    cat.pop(k, None)

                exemplar_updates += 1
                changed = True

    # Apply a recursive rule_id -> logic_id pass across the whole document.
    # This ensures we catch all suggestion shapes, even if they are nested in
    # unexpected ways. It is idempotent and safe to run on non-suggestion docs.
    extra_suggestions_renamed = _rename_rule_id_to_logic_id(new_data)
    if extra_suggestions_renamed:
        suggestions_renamed += extra_suggestions_renamed
        changed = True

    # -----------------------------
    # List-shaped documents (e.g. business_rules.json)
    # -----------------------------
    rules_field_renames = {
        "doc_rule_id": "doc_logic_id",
        "from_step_id": "from_logic_id",
        "from_step": "from_logic",
        "rule_category": "category",
        "rule_name": "name",
        "rule_purpose": "purpose",
        "rule_spec": "spec",
    }
    renamed_rules = 0

    if isinstance(new_data, list):
        for rule in new_data:
            if not isinstance(rule, dict):
                continue

            rule_changed = False
            for old_key, new_key in rules_field_renames.items():
                if old_key not in rule:
                    continue

                value = rule.get(old_key)

                # Only overwrite the new key if it is missing or empty-like
                if new_key not in rule or rule[new_key] in (None, "", []):
                    rule[new_key] = value

                # Remove the legacy key
                rule.pop(old_key, None)

                rule_changed = True
                changed = True

            if rule_changed:
                renamed_rules += 1

        # Second pass: handle models.json style objects
        for model in new_data:
            if not isinstance(model, dict):
                continue

            # businessLogicIds -> logicIds on the model itself
            if "businessLogicIds" in model:
                value = model.get("businessLogicIds")
                if "logicIds" not in model or model["logicIds"] in (None, [], ""):
                    model["logicIds"] = value
                model.pop("businessLogicIds", None)
                models_logic_ids_renamed += 1
                changed = True

            # Within hierarchies, topDecisionId -> topId
            hierarchies = model.get("hierarchies")
            if isinstance(hierarchies, list):
                for hierarchy in hierarchies:
                    if not isinstance(hierarchy, dict):
                        continue
                    if "topDecisionId" not in hierarchy:
                        continue

                    value = hierarchy.get("topDecisionId")
                    if "topId" not in hierarchy or hierarchy["topId"] in (None, ""):
                        hierarchy["topId"] = value
                    hierarchy.pop("topDecisionId", None)

                    hierarchy_top_ids_renamed += 1
                    changed = True

    # -----------------------------
    # Build summary
    # -----------------------------
    if top_groups_renamed:
        summary_parts.append("renamed ruleCategoryGroups -> categoryGroups")
    if top_categories_renamed:
        summary_parts.append("renamed ruleCategories -> categories")
    if exemplar_updates:
        summary_parts.append(
            f"updated exemplar ids on {exemplar_updates} categor{'y' if exemplar_updates == 1 else 'ies'}"
        )
    if renamed_rules:
        summary_parts.append(
            f"renamed fields on {renamed_rules} rule{'s' if renamed_rules != 1 else ''} "
            "(doc_rule_id→doc_logic_id, from_step_id→from_logic_id, from_step→from_logic, "
            "rule_category→category, rule_name→name, rule_purpose→purpose, rule_spec→spec)"
        )
    if suggestions_renamed:
        summary_parts.append(
            f"renamed rule_id→logic_id on {suggestions_renamed} suggestion" +
            ("s" if suggestions_renamed != 1 else "")
        )
    if models_logic_ids_renamed:
        summary_parts.append(
            f"renamed businessLogicIds→logicIds on {models_logic_ids_renamed} model" +
            ("s" if models_logic_ids_renamed != 1 else "")
        )
    if hierarchy_top_ids_renamed:
        summary_parts.append(
            f"renamed topDecisionId→topId on {hierarchy_top_ids_renamed} hierarchy" +
            ("ies" if hierarchy_top_ids_renamed != 1 else "")
        )

    summary = "; ".join(summary_parts) if summary_parts else None
    return new_data, changed, summary


def migrate_file(
    path: Path,
    backup_root: Optional[Path],
    timestamp: str,
    preview: bool = False,
    verbose: int = 0,
) -> MigrationResult:
    try:
        original_text = path.read_text(encoding="utf-8")
        docs = decode_multi_json(original_text)
        if not docs:
            raise ValueError("No JSON documents found")
    except Exception as exc:  # noqa: BLE001
        return MigrationResult(file_path=path, changed=False, backed_up_to=None, error=str(exc))

    changed = False
    summaries: list[str] = []
    new_docs: list[Any] = []

    for doc in docs:
        new_doc, doc_changed, doc_summary = migrate_document(doc)
        new_docs.append(new_doc)
        if doc_changed:
            changed = True
        if doc_summary:
            summaries.append(doc_summary)

    summary: Optional[str] = "; ".join(summaries) if summaries else None

    if not changed:
        if verbose >= 2:
            print(f"[SKIP] {path} (no changes)")
        return MigrationResult(file_path=path, changed=False, backed_up_to=None, summary=summary)

    # Prepare the pretty-printed new text (same format as save_json) for one
    # or more documents. For multiple docs we write them back concatenated with
    # a blank line between each, which is safe and deterministic.
    new_text_parts: list[str] = []
    for doc in new_docs:
        new_text_parts.append(json.dumps(doc, indent=2, ensure_ascii=False))
    new_text = "\n".join(new_text_parts) + "\n"

    if preview:
        print(f"[PREVIEW] {path}")
        if summary:
            print(f"          {summary}")

        diff = difflib.unified_diff(
            original_text.splitlines(),
            new_text.splitlines(),
            fromfile=str(path),
            tofile=f"{path} (migrated)",
            lineterm="",
            n=2,
        )
        for line in diff:
            print(line)

        return MigrationResult(file_path=path, changed=True, backed_up_to=None, summary=summary)

    backup_path = make_backup(path, backup_root, timestamp)
    path.write_text(new_text, encoding="utf-8")

    extra = f" [{summary}]" if summary else ""
    print(f"[OK]   {path} (backed up to {backup_path}){extra}")

    return MigrationResult(file_path=path, changed=True, backed_up_to=backup_path, summary=summary)


def main() -> int:
    args = parse_args()

    # Prompt for root if not supplied
    if args.root is None:
        default_root = Path.home() / ".model"
        user_input = input(f"Enter path to JSON directory [{default_root}]: ").strip()
        if user_input:
            root = Path(user_input).expanduser()
        else:
            root = default_root
        args.root = root

    root: Path = args.root
    pattern: str = args.pattern
    recursive: bool = args.recursive
    backup_root: Optional[Path] = args.backup_dir
    preview: bool = args.preview
    verbose: int = args.verbose

    if not root.exists() or not root.is_dir():
        print(f"ERROR: Root directory does not exist or is not a directory: {root}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    json_files = list(find_json_files(root, pattern, recursive))
    print(
        f"Scanning {root} for pattern '{pattern}' (recursive={recursive}) - "
        f"found {len(json_files)} file(s)"
    )

    changed_count = 0
    error_count = 0

    for path in json_files:
        result = migrate_file(
            path=path,
            backup_root=backup_root,
            timestamp=timestamp,
            preview=preview,
            verbose=verbose,
        )

        if result.error:
            error_count += 1
            print(f"[ERR]  {path}: {result.error}")
        elif result.changed:
            changed_count += 1

    mode = "PREVIEW" if preview else "WRITE"
    print(
        f"Done. Mode={mode}, changed={changed_count}, "
        f"errors={error_count}, total={len(json_files)}."
    )

    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
