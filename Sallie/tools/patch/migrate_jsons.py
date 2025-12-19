import argparse
import copy
import difflib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from importlib.machinery import SourceFileLoader
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


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
    parser.add_argument(
        "--prune-missing-kg-links",
        action="store_true",
        help=(
            "For business_rules.json only: query the KG and remove JSON links that are marked "
            "in_graph=true (or missing in_graph, treated as true) but whose corresponding :SUPPORTS "
            "relationship is missing in the KG."
        ),
    )
    parser.add_argument(
        "--graph-url",
        default=None,
        help=(
            "Base URL for Rules Portal graph API. Can be a host (e.g. http://localhost:443) or already include /api/graph. "
            "If omitted, uses RULES_PORTAL_BASE_URL or falls back to http://localhost:443." 
        ),
    )
    parser.add_argument(
        "--prune-only-supports",
        action="store_true",
        help=(
            "When pruning, only consider JSON links whose kind is SUPPORTS/SUPPORT. By default, prunes any link with in_graph=true, "
            "because the KG stores all in-graph links as :SUPPORTS regardless of JSON kind (depends_on/invokes_bkm/etc.)."
        ),
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


# Helper to make a timestamped backup of a file.
def make_backup(path: Path, backup_root: Optional[Path], timestamp: str) -> Path:
    """Create a timestamped backup of `path`.

    If backup_root is provided, replicate the subdirectory structure under it.
    Otherwise, place the backup next to the original file.
    """
    if backup_root:
        rel = path.relative_to(path.anchor) if path.is_absolute() else path
        backup_dir = backup_root.joinpath(rel).parent
    else:
        backup_dir = path.parent

    backup_dir.mkdir(parents=True, exist_ok=True)

    backup_path = backup_dir / f"{path.name}.{timestamp}.bak"
    backup_path.write_bytes(path.read_bytes())
    return backup_path


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


# ---- KG-driven prune helpers ----
def _load_check_integrity_module() -> ModuleType:
    """Load tools/maintenance/check_integrity.py as a module at runtime.

    We reuse its KG querying and link coverage logic to avoid duplicating it.
    """
    here = Path(__file__).resolve()
    tools_dir = here.parents[1]  # <repo>/tools
    ci_path = tools_dir / "maintenance" / "check_integrity.py"
    if not ci_path.exists():
        raise FileNotFoundError(f"check_integrity.py not found at {ci_path}")

    mod_name = "_rules_portal_check_integrity"
    loader = SourceFileLoader(mod_name, str(ci_path))
    mod = loader.load_module(mod_name)  # type: ignore[attr-defined]
    return mod


def _effective_from_id(link: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of a link's source logic id across legacy shapes."""
    for k in ("from_logic_id", "fromLogicId", "from_step_id"):
        v = link.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _is_in_graph_trueish(link: Dict[str, Any]) -> bool:
    """Treat missing in_graph as True; only explicit False disables KG expectations."""
    return link.get("in_graph") is not False


def _kind_is_supports(link: Dict[str, Any]) -> bool:
    k = link.get("kind")
    if isinstance(k, str):
        k2 = k.strip().upper()
        return k2 in {"SUPPORTS", "SUPPORT"}
    if k is None:
        return False
    return str(k).strip().upper() in {"SUPPORTS", "SUPPORT"}


def _extract_logics_from_business_rules_doc(doc: Any) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Return (logics_list, logics_by_id) from a business_rules doc supporting both shapes."""
    if isinstance(doc, list):
        logics = doc
    elif isinstance(doc, dict) and isinstance(doc.get("logics"), list):
        logics = doc["logics"]
    else:
        raise ValueError(
            f"business_rules.json is not in a supported format: expected list or {{version, logics}} object, got {type(doc).__name__}"
        )

    logics_by_id: Dict[str, Dict[str, Any]] = {}
    for logic in logics:
        if not isinstance(logic, dict):
            continue
        lid = logic.get("id")
        if isinstance(lid, str) and lid:
            logics_by_id[lid] = logic

    return logics, logics_by_id


def _prune_links_for_missing_edges(
    logics_by_id: Dict[str, Dict[str, Any]],
    missing_edges: List[Tuple[str, str]],
    prune_only_supports: bool,
) -> Tuple[int, int]:
    """Remove JSON link objects from target logic when (fromId,toId) is missing in KG.

    Returns: (removed_links_count, affected_logics_count)
    """
    missing_set: Set[Tuple[str, str]] = set((u, v) for (u, v) in missing_edges if u and v)
    removed = 0
    affected = 0

    for to_id, logic in list(logics_by_id.items()):
        if not isinstance(logic, dict):
            continue
        links = logic.get("links")
        if not isinstance(links, list) or not links:
            continue

        before = len(links)
        new_links: List[Any] = []

        for link in links:
            if not isinstance(link, dict):
                new_links.append(link)
                continue

            # Only prune links that claim to be present in the KG.
            if not _is_in_graph_trueish(link):
                new_links.append(link)
                continue

            if prune_only_supports and not _kind_is_supports(link):
                new_links.append(link)
                continue

            from_id = _effective_from_id(link)
            if not from_id:
                new_links.append(link)
                continue

            if (from_id, to_id) in missing_set:
                removed += 1
                continue

            new_links.append(link)

        if len(new_links) != before:
            logic["links"] = new_links
            affected += 1

    return removed, affected


def prune_business_rules_links_missing_in_kg_in_memory(
    business_rules_doc: Any,
    graph_url: str,
    prune_only_supports: bool,
    verbose: int = 0,
) -> Tuple[Any, bool, str]:
    """Prune business_rules doc in-memory based on KG missing link list.

    Reuses check_integrity._load_kg_summary() to compute which in-graph JSON links
    are missing as KG :SUPPORTS relationships.

    Returns: (new_doc, changed, summary)
    """
    ci = _load_check_integrity_module()

    # Build logics_by_id from the doc we are migrating (do not re-read from disk).
    logics, logics_by_id = _extract_logics_from_business_rules_doc(business_rules_doc)

    kg_summary = ci._load_kg_summary(graph_url, logics_by_id)  # type: ignore[attr-defined]
    if kg_summary is None:
        return business_rules_doc, False, "KG summary unavailable; no pruning performed"

    # Prefer the broader list: all in-graph links regardless of JSON kind.
    missing_edges = getattr(kg_summary, "json_in_graph_links_missing_in_kg_list", [])
    if prune_only_supports:
        missing_edges = getattr(kg_summary, "json_supports_links_missing_in_kg_list", [])

    if not missing_edges:
        return business_rules_doc, False, "no missing KG links to prune"

    removed, affected = _prune_links_for_missing_edges(logics_by_id, missing_edges, prune_only_supports)
    if removed == 0:
        return business_rules_doc, False, "missing edges were found but no matching JSON links were removed"

    summary = f"KG prune: removed {removed} link(s) across {affected} logic(s)"
    if verbose >= 1:
        mode = "SUPPORTS-only" if prune_only_supports else "in_graph"
        print(f"[KG-PRUNE] {summary} (mode={mode})")

    return business_rules_doc, True, summary


def migrate_document(data: Any) -> Tuple[Any, bool, Optional[str]]:
    """Apply schema migration to the JSON document.

    Currently supports the following transformations (idempotent):
    * ruleCategoryGroups  -> categoryGroups (rule_categories.json)
    * ruleCategories      -> categories (rule_categories.json)
    * examplarRuleIds / exemplarRuleIds -> exemplarLogicIds (on category objects)
    * doc_rule_id        -> doc_logic_id (business_rules.json)
    * from_step_id       -> from_logic_id (business_rules.json, including nested `links` objects)
    * from_step          -> from_logic (business_rules.json)
    * rule_category      -> category (business_rules.json)
    * rule_name          -> name (business_rules.json)
    * rule_purpose       -> purpose (business_rules.json)
    * rule_spec          -> spec (business_rules.json)
    * rule_id           -> logic_id (supporting-decisions-suggestions.json)
    * businessLogicIds   -> logicIds (models.json)
    * topDecisionId      -> topId (models.json hierarchies)
    * rule_ids          -> logic_ids (runs.json)
    * models.json and business_rules.json are now wrapped under a versioned root object (version=1, models/logics array)
    * Optional: prune JSON links missing from KG (:SUPPORTS) in business_rules.json (when --prune-missing-kg-links is used)
    * Always remove derived link fields from_logic_name/to_logic_name from business_rules.json link objects
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
    runs_logic_ids_renamed = 0

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
            # Only rename if the new key is not already present
            if "categories" not in new_data:
                new_data["categories"] = new_data.pop("ruleCategories")
                top_categories_renamed = True
                changed = True
            else:
                # If both exist, drop the old one to avoid ambiguity
                new_data.pop("ruleCategories", None)
                top_categories_renamed = True
                changed = True

        # --- Nested field rename inside categories ---
        categories_key = None
        if "categories" in new_data and isinstance(new_data["categories"], list):
            categories_key = "categories"
        elif "ruleCategories" in new_data and isinstance(new_data["ruleCategories"], list):
            # In case this function runs before the top-level rename or on legacy shape
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

    # Also handle new business_rules.json root: {"version": ..., "logics": [ ... ]}
    if isinstance(new_data, dict) and isinstance(new_data.get("logics"), list):
        for rule in new_data["logics"]:
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

            # --- Nested links: from_step_id -> from_logic_id ---
            links = rule.get("links")
            if isinstance(links, list):
                for link in links:
                    if not isinstance(link, dict):
                        continue

                    link_changed = False

                    # from_step_id -> from_logic_id
                    if "from_step_id" in link:
                        value = link.get("from_step_id")
                        # Only overwrite from_logic_id if it is missing or empty-like
                        if "from_logic_id" not in link or link["from_logic_id"] in (None, "", []):
                            link["from_logic_id"] = value
                        link.pop("from_step_id", None)
                        link_changed = True

                    # Always strip derived/enriched name fields from links
                    if "from_logic_name" in link:
                        link.pop("from_logic_name", None)
                        link_changed = True
                    if "to_logic_name" in link:
                        link.pop("to_logic_name", None)
                        link_changed = True

                    if link_changed:
                        rule_changed = True
                        changed = True

            # --- Cleanup: drop transient document-scoring and expression fields ---
            for obsolete_key in ("doc_match_score", "doc_logic_id", "dmn_expression", "__modeloverlay"):
                if obsolete_key in rule:
                    rule.pop(obsolete_key, None)
                    rule_changed = True
                    changed = True

            if rule_changed:
                renamed_rules += 1

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

            # --- Nested links: from_step_id -> from_logic_id ---
            links = rule.get("links")
            if isinstance(links, list):
                for link in links:
                    if not isinstance(link, dict):
                        continue

                    link_changed = False

                    # from_step_id -> from_logic_id
                    if "from_step_id" in link:
                        value = link.get("from_step_id")
                        # Only overwrite from_logic_id if it is missing or empty-like
                        if "from_logic_id" not in link or link["from_logic_id"] in (None, "", []):
                            link["from_logic_id"] = value
                        link.pop("from_step_id", None)
                        link_changed = True

                    # Always strip derived/enriched name fields from links
                    if "from_logic_name" in link:
                        link.pop("from_logic_name", None)
                        link_changed = True
                    if "to_logic_name" in link:
                        link.pop("to_logic_name", None)
                        link_changed = True

                    if link_changed:
                        rule_changed = True
                        changed = True

            # --- runs.json: rule_ids -> logic_ids ---
            if "rule_ids" in rule:
                value = rule.get("rule_ids")
                # Only overwrite logic_ids if it is missing or empty-like
                if "logic_ids" not in rule or rule["logic_ids"] in (None, [], ""):
                    # Rebuild the dict so that logic_ids appears where rule_ids used to be
                    new_rule: dict[str, Any] = {}
                    for key in list(rule.keys()):
                        if key == "rule_ids":
                            new_rule["logic_ids"] = value
                        else:
                            new_rule[key] = rule[key]
                    rule.clear()
                    rule.update(new_rule)
                else:
                    # logic_ids already present and not empty-like; just drop rule_ids
                    rule.pop("rule_ids", None)
                runs_logic_ids_renamed += 1
                changed = True
                rule_changed = True

            # --- Cleanup: drop transient document-scoring and expression fields ---
            for obsolete_key in ("doc_match_score", "doc_logic_id", "dmn_expression", "__modelOverlay"):
                if obsolete_key in rule:
                    rule.pop(obsolete_key, None)
                    rule_changed = True
                    changed = True

            if rule_changed:
                renamed_rules += 1

        # Second pass: handle models.json style objects
        for model in new_data:
            if not isinstance(model, dict):
                continue

            # businessLogicIds -> logicIds on the model itself, preserving key order
            if "businessLogicIds" in model:
                value = model.get("businessLogicIds")
                if "logicIds" not in model or model["logicIds"] in (None, [], ""):
                    # Rebuild the dict so that logicIds appears where businessLogicIds used to be
                    new_model: dict[str, Any] = {}
                    for key in list(model.keys()):
                        if key == "businessLogicIds":
                            new_model["logicIds"] = value
                        else:
                            new_model[key] = model[key]
                    model.clear()
                    model.update(new_model)
                else:
                    # logicIds already present; just drop the legacy key without reordering
                    model.pop("businessLogicIds", None)
                models_logic_ids_renamed += 1
                changed = True

            # Within hierarchies, topDecisionId -> topId, preserving key order
            hierarchies = model.get("hierarchies")
            if isinstance(hierarchies, list):
                for hierarchy in hierarchies:
                    if not isinstance(hierarchy, dict):
                        continue
                    if "topDecisionId" not in hierarchy:
                        continue

                    old_value = hierarchy.get("topDecisionId")
                    if "topId" not in hierarchy or hierarchy["topId"] in (None, ""):
                        # Rebuild the dict so that topId appears where topDecisionId used to be
                        new_hierarchy: dict[str, Any] = {}
                        for key in list(hierarchy.keys()):
                            if key == "topDecisionId":
                                new_hierarchy["topId"] = old_value
                            else:
                                new_hierarchy[key] = hierarchy[key]
                        hierarchy.clear()
                        hierarchy.update(new_hierarchy)
                    else:
                        # topId already present; just drop the legacy key
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
    if runs_logic_ids_renamed:
        summary_parts.append(
            f"renamed rule_ids→logic_ids on {runs_logic_ids_renamed} run" +
            ("s" if runs_logic_ids_renamed != 1 else "")
        )

    summary = "; ".join(summary_parts) if summary_parts else None
    return new_data, changed, summary


def migrate_file(
    path: Path,
    backup_root: Optional[Path],
    timestamp: str,
    preview: bool = False,
    verbose: int = 0,
    prune_missing_kg_links: bool = False,
    graph_url: Optional[str] = None,
    prune_only_supports: bool = False,
) -> MigrationResult:
    try:
        original_text = path.read_text(encoding="utf-8")
        docs = decode_multi_json(original_text)
        if not docs:
            raise ValueError("No JSON documents found")
    except Exception as exc:  # noqa: BLE001
        return MigrationResult(file_path=path, changed=False, backed_up_to=None, error=str(exc))

    changed = False
    # Optional KG-driven pruning for business_rules.json before running schema migrations.
    if prune_missing_kg_links and path.name == "business_rules.json" and len(docs) == 1:
        base = graph_url or os.getenv("RULES_PORTAL_BASE_URL") or "http://localhost:443"
        # check_integrity normalizes host-only vs /api/graph, but it expects graph_base_url.
        try:
            pruned_doc, pruned_changed, prune_summary = prune_business_rules_links_missing_in_kg_in_memory(
                business_rules_doc=copy.deepcopy(docs[0]),
                graph_url=base,
                prune_only_supports=prune_only_supports,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001
            return MigrationResult(file_path=path, changed=False, backed_up_to=None, error=f"KG prune failed: {exc}")

        if pruned_changed:
            docs = [pruned_doc]
            changed = True
            # We treat this as a real change so it will be written (or diffed) alongside schema migrations.
            if verbose >= 1:
                print(f"[KG-PRUNE] {path}: {prune_summary}")
            summaries = [prune_summary]
        else:
            summaries = []
    else:
        summaries: list[str] = []

    # Track the original version for versioned, single-doc files
    original_version: Optional[int] = None
    if path.name in ("models.json", "business_rules.json") and len(docs) == 1 and isinstance(docs[0], dict):
        root_doc = docs[0]
        v = root_doc.get("version")
        if isinstance(v, int):
            original_version = v

    new_docs: list[Any] = []

    for doc in docs:
        new_doc, doc_changed, doc_summary = migrate_document(doc)
        new_docs.append(new_doc)
        if doc_changed:
            changed = True
        if doc_summary:
            summaries.append(doc_summary)

    # Additional wrapping for models.json and business_rules.json to support
    # the new root-level shape with a version and nested array.
    #
    # models.json:
    #   [ { ... }, { ... } ]
    # becomes
    #   { "version": 1, "models": [ { ... }, { ... } ] }
    #
    # business_rules.json:
    #   [ { ... }, { ... } ]
    # becomes
    #   { "version": 1, "logics": [ { ... }, { ... } ] }
    #
    # For already-migrated files that have {"models": [...]} or
    # {"logics": [...]}, we ensure a version field exists (defaulting to 1)
    # but otherwise leave the structure unchanged. This logic is idempotent.
    extra_summaries: list[str] = []

    # Handle models.json
    if path.name == "models.json" and len(new_docs) == 1:
        doc = new_docs[0]
        # Old shape: top-level list of models
        if isinstance(doc, list):
            new_docs[0] = {"version": 1, "models": doc}
            changed = True
            extra_summaries.append("wrapped models list under {version, models[]} (version=1)")
        # New shape but missing version: add version=1 at the top
        elif isinstance(doc, dict) and "models" in doc and "version" not in doc:
            old_doc = doc
            wrapped: dict[str, Any] = {"version": 1}
            for k, v in old_doc.items():
                if k == "version":
                    continue
                wrapped[k] = v
            new_docs[0] = wrapped
            changed = True
            extra_summaries.append("added version=1 to models root")

    # Handle business_rules.json
    if path.name == "business_rules.json" and len(new_docs) == 1:
        doc = new_docs[0]
        # Old shape: top-level list of business rules / logic objects
        if isinstance(doc, list):
            new_docs[0] = {"version": 1, "logics": doc}
            changed = True
            extra_summaries.append("wrapped business rules list under {version, logics[]} (version=1)")
        # New shape but missing version: add version=1 at the top
        elif isinstance(doc, dict) and "logics" in doc and "version" not in doc:
            old_doc = doc
            wrapped: dict[str, Any] = {"version": 1}
            for k, v in old_doc.items():
                if k == "version":
                    continue
                wrapped[k] = v
            new_docs[0] = wrapped
            changed = True
            extra_summaries.append("added version=1 to business rules root")

    if extra_summaries:
        summaries.extend(extra_summaries)

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

    # Optimistic concurrency check for versioned single-doc roots
    if original_version is not None:
        try:
            current_text = path.read_text(encoding="utf-8")
            current_docs = decode_multi_json(current_text)
            if len(current_docs) == 1 and isinstance(current_docs[0], dict):
                current_root = current_docs[0]
                cur_v = current_root.get("version")
                if isinstance(cur_v, int) and cur_v != original_version:
                    return MigrationResult(
                        file_path=path,
                        changed=False,
                        backed_up_to=None,
                        error=(
                            f"Version conflict for {path.name}: expected version {original_version}, "
                            f"found {cur_v}; aborting to avoid overwriting concurrent changes."
                        ),
                    )
        except Exception as exc:  # noqa: BLE001
            return MigrationResult(
                file_path=path,
                changed=False,
                backed_up_to=None,
                error=f"Failed optimistic concurrency check: {exc}",
            )

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
    if args.prune_missing_kg_links:
        mode = "SUPPORTS-only" if args.prune_only_supports else "in_graph"
        graph = args.graph_url or os.getenv("RULES_PORTAL_BASE_URL") or "http://localhost:443"
        print(f"KG pruning enabled for business_rules.json (mode={mode}, graph_url={graph})")

    changed_count = 0
    error_count = 0

    for path in json_files:
        result = migrate_file(
            path=path,
            backup_root=backup_root,
            timestamp=timestamp,
            preview=preview,
            verbose=verbose,
            prune_missing_kg_links=args.prune_missing_kg_links,
            graph_url=args.graph_url,
            prune_only_supports=args.prune_only_supports,
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
