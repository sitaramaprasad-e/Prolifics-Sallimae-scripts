import argparse
import copy
import difflib
import json
import os
import sys
import io
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from importlib.machinery import SourceFileLoader
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse


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
            "in_graph=true (missing or false in_graph is treated as out-of-graph) but whose corresponding :SUPPORTS "
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
        "--spec",
        default=None,
        help=(
            "Optional path to a spec JSON file (same selection mechanism as check_integrity). "
            "When --prune-missing-kg-links is enabled and neither --graph-url nor RULES_PORTAL_BASE_URL is provided, "
            "the tool can use the spec to determine the Rules Portal base URL. If omitted, you will be prompted to select a spec (or skip)."
        ),
    )
    parser.add_argument(
        "--prune-only-supports",
        action="store_true",
        help=(
            "When pruning, only consider JSON links whose kind is SUPPORTS/SUPPORT. By default, prunes any link with in_graph=true, "
            "because the KG stores all in-graph links as :SUPPORTS regardless of JSON kind."
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

# --- Helper to match check_integrity's default graph base logic ---
def _default_graph_url_from_check_integrity(ci: ModuleType) -> str:
    """Return the default graph URL, matching check_integrity behavior."""
    # check_integrity defines DEFAULT_GRAPH_BASE_URL (already includes /api/graph)
    v = getattr(ci, "DEFAULT_GRAPH_BASE_URL", None)
    if isinstance(v, str) and v.strip():
        return v.strip()
    # Fallback to the historic local default
    return "http://localhost:443/api/graph"


# --- Spec helpers for KG base selection (from check_integrity UX) ---
def _extract_graph_url_from_spec(spec: Any) -> Optional[str]:
    """Best-effort extraction of a Rules Portal / graph base URL from a spec dict.

    We intentionally keep this flexible because spec formats vary across repos.
    check_integrity will normalize whether /api/graph is appended.
    """
    if not isinstance(spec, dict):
        return None

    # Fast path: common explicit keys
    for k in (
        "RULES_PORTAL_BASE_URL",
        "rules_portal_base_url",
        "rulesPortalBaseUrl",
        "rules_portal_url",
        "rulesPortalUrl",
        "graph_base_url",
        "graphBaseUrl",
        "graph_url",
        "graphUrl",
        "base_url",
        "baseUrl",
        "url",
    ):
        v = spec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Nested containers commonly used in specs
    for container_key in (
        "endpoints",
        "endpoint",
        "config",
        "settings",
        "rules_portal",
        "rulesPortal",
        "graph",
        "kg",
    ):
        c = spec.get(container_key)
        if isinstance(c, dict):
            for k in (
                "RULES_PORTAL_BASE_URL",
                "rules_portal_base_url",
                "rulesPortalBaseUrl",
                "graph_base_url",
                "graphBaseUrl",
                "graph_url",
                "graphUrl",
                "url",
                "base_url",
                "baseUrl",
            ):
                v = c.get(k)
                if isinstance(v, str) and v.strip():
                    return v.strip()

    # Sources list heuristic
    sources = spec.get("sources")
    if isinstance(sources, list):
        for src in sources:
            if not isinstance(src, dict):
                continue
            name = str(src.get("name") or src.get("type") or "").strip().lower()
            if name and any(tok in name for tok in ("graph", "kg", "neo4j", "rules", "portal")):
                for k in (
                    "rules_portal_base_url",
                    "rulesPortalBaseUrl",
                    "graph_base_url",
                    "graphBaseUrl",
                    "graph_url",
                    "graphUrl",
                    "base_url",
                    "baseUrl",
                    "url",
                ):
                    v = src.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()

    # Last resort: recursively scan for any URL-looking string under URL-ish keys.
    want_keys = {
        "rules_portal_base_url",
        "rulesportalbaseurl",
        "rules_portal_url",
        "rulesportalurl",
        "graph_base_url",
        "graphbaseurl",
        "graph_url",
        "graphurl",
        "base_url",
        "baseurl",
        "url",
    }

    def _scan(obj: Any) -> Optional[str]:
        if isinstance(obj, dict):
            for kk, vv in obj.items():
                kkl = str(kk).strip().lower()
                if kkl in want_keys and isinstance(vv, str) and vv.strip():
                    return vv.strip()
            # also accept any http(s) URL-looking string anywhere (fallback)
            for vv in obj.values():
                if isinstance(vv, str) and vv.strip().startswith(("http://", "https://")):
                    return vv.strip()
            for vv in obj.values():
                found = _scan(vv)
                if found:
                    return found
        elif isinstance(obj, list):
            for it in obj:
                found = _scan(it)
                if found:
                    return found
        return None

    return _scan(spec)


def _maybe_get_graph_url_from_spec(ci: ModuleType, spec_path: Optional[str]) -> Optional[str]:
    """Use check_integrity’s spec selection UX to obtain a Rules Portal base URL."""
    # Non-interactive: explicit --spec path
    if spec_path:
        p = str(spec_path).strip()
        if not p:
            return None
        try:
            if hasattr(ci, "_load_spec"):
                spec = ci._load_spec(p)  # type: ignore[attr-defined]
            else:
                spec = None
        except Exception:
            spec = None
        v = _extract_graph_url_from_spec(spec)
        if v:
            return v
        # Fallback: if check_integrity exposes any direct extraction helper, try it.
        for fn_name in (
            "_extract_rules_portal_base_url_from_spec",
            "_get_rules_portal_base_url_from_spec",
            "_rules_portal_base_url_from_spec",
        ):
            if hasattr(ci, fn_name):
                try:
                    fn = getattr(ci, fn_name)
                    vv = fn(spec)
                    if isinstance(vv, str) and vv.strip():
                        return vv.strip()
                except Exception:
                    pass
        return None

    # Interactive selection using check_integrity helpers (same prompt behavior)
    if hasattr(ci, "_find_spec_files") and hasattr(ci, "_choose_spec") and hasattr(ci, "_load_spec"):
        try:
            specs = ci._find_spec_files()  # type: ignore[attr-defined]
            chosen = ci._choose_spec(specs)  # type: ignore[attr-defined]
            if not chosen:
                return None
            spec = ci._load_spec(chosen)  # type: ignore[attr-defined]
            v = _extract_graph_url_from_spec(spec)
            if v:
                return v
            # Fallback: if check_integrity exposes any direct extraction helper, try it.
            for fn_name in (
                "_extract_rules_portal_base_url_from_spec",
                "_get_rules_portal_base_url_from_spec",
                "_rules_portal_base_url_from_spec",
            ):
                if hasattr(ci, fn_name):
                    try:
                        fn = getattr(ci, fn_name)
                        vv = fn(spec)
                        if isinstance(vv, str) and vv.strip():
                            return vv.strip()
                    except Exception:
                        pass
            return None
        except Exception:
            return None

    return None


def _effective_from_id(link: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of a link's source logic id across legacy shapes."""
    for k in ("from_logic_id", "fromLogicId", "from_step_id"):
        v = link.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _is_in_graph_trueish(link: Dict[str, Any]) -> bool:
    """Only explicit in_graph=true means the link is expected in the KG.

    Missing in_graph is treated as False (do NOT expect a KG edge).
    """
    return link.get("in_graph") is True


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

    Only prune links that explicitly claim to be present in the KG (in_graph=true).
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



# --- Helper for KG base normalization, matching check_integrity.py logic ---
def _normalize_graph_base_candidates(graph_url: str) -> List[str]:
    """Return candidate base URLs to try for KG access.

    We want to behave like check_integrity in production:
    - accept bare host:port (e.g. localhost:443)
    - accept base without /api/graph (check_integrity appends it)
    - if scheme is missing, assume https for :443, otherwise http
    - if user provided http://...:443, also try https://...:443

    NOTE: check_integrity._load_kg_summary() itself appends /api/graph when needed.
    """
    raw = (graph_url or "").strip()
    if not raw:
        return []

    candidates: List[str] = []

    # If no scheme provided, assume https for :443, otherwise http.
    if "://" not in raw:
        scheme = "https" if ":443" in raw else "http"
        raw = f"{scheme}://{raw}"

    parsed = urlparse(raw)

    # If parse fails to produce netloc, just try the raw.
    if not parsed.netloc:
        candidates.append(raw.rstrip("/"))
        return candidates

    base = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", "")).rstrip("/")
    candidates.append(base)

    # If user gave http on port 443, also try https.
    if parsed.scheme == "http" and parsed.netloc.endswith(":443"):
        https_base = urlunparse(("https", parsed.netloc, parsed.path, "", "", "")).rstrip("/")
        if https_base not in candidates:
            candidates.append(https_base)

    return candidates


def prune_business_rules_links_missing_in_kg_in_memory(
    business_rules_doc: Any,
    graph_url: str,
    prune_only_supports: bool,
    verbose: int = 0,
    ci: Optional[ModuleType] = None,
) -> Tuple[Any, bool, str]:
    """Prune business_rules doc in-memory based on KG missing link list.

    Reuses check_integrity._load_kg_summary() to compute which in-graph JSON links
    are missing as KG :SUPPORTS relationships.

    Returns: (new_doc, changed, summary)
    """
    ci = ci or _load_check_integrity_module()

    # Build logics_by_id from the doc we are migrating (do not re-read from disk).
    logics, logics_by_id = _extract_logics_from_business_rules_doc(business_rules_doc)

    # Try to derive the same base URL that check_integrity would use, if available.
    base_in = (graph_url or "").strip()
    derived_base: Optional[str] = None
    try:
        if hasattr(ci, "_get_graph_base_url"):
            # check_integrity normalizes RULES_PORTAL_BASE_URL and appends /api/graph
            # NOTE: it expects a base like http(s)://host:port or host:port
            derived_base = ci._get_graph_base_url(base_in)  # type: ignore[attr-defined]
    except Exception:
        derived_base = None

    candidates: List[str] = []
    if derived_base:
        # Prefer the derived base first
        candidates.append(str(derived_base).rstrip("/"))

    # Also try normalized variants of what we were given
    for cand in _normalize_graph_base_candidates(base_in):
        if cand not in candidates:
            candidates.append(cand)

    # If nothing produced candidates, we can't proceed
    if not candidates:
        return business_rules_doc, False, "KG summary unavailable; no pruning performed (no graph_url candidates)"

    captured_msgs: List[str] = []
    kg_summary = None
    tried: List[str] = []
    for cand in candidates:
        tried.append(cand)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                kg_summary = ci._load_kg_summary(cand, logics_by_id)  # type: ignore[attr-defined]
        finally:
            out = buf.getvalue() or ""
            if out.strip():
                captured_msgs.append(out)
        if kg_summary is not None:
            graph_url = cand
            break

    # If check_integrity printed the old SUPPORTS-kind warning, only surface it when prune_only_supports is enabled.
    if prune_only_supports and captured_msgs:
        joined = "\n".join(captured_msgs)
        if "none were detected as SUPPORTS" in joined or "no links were detected as SUPPORTS" in joined:
            # Print a clearer warning aligned with this tool's semantics.
            print(
                "WARNING: --prune-only-supports is enabled but no SUPPORTS-kind links were found in JSON; "
                "pruning may remove nothing. (Note: in default mode we prune all in_graph links regardless of kind.)"
            )

    if verbose >= 2 and captured_msgs:
        # Re-emit any captured check_integrity output for debugging.
        for msg in captured_msgs:
            for line in (msg or "").splitlines():
                print(f"[CI] {line}")

    if kg_summary is None:
        return business_rules_doc, False, f"KG summary unavailable; no pruning performed (tried: {', '.join(tried)})"

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
        # Match check_integrity precedence: CLI arg > RULES_PORTAL_BASE_URL env > DEFAULT_GRAPH_BASE_URL
        ci = _load_check_integrity_module()
        base = graph_url or os.getenv("RULES_PORTAL_BASE_URL") or _default_graph_url_from_check_integrity(ci)
        try:
            pruned_doc, pruned_changed, prune_summary = prune_business_rules_links_missing_in_kg_in_memory(
                business_rules_doc=copy.deepcopy(docs[0]),
                graph_url=base,
                prune_only_supports=prune_only_supports,
                verbose=verbose,
                ci=ci,
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
    resolved_graph_url: Optional[str] = None
    if args.prune_missing_kg_links:
        mode = "SUPPORTS-only" if args.prune_only_supports else "in_graph"
        ci = _load_check_integrity_module()

        env_graph = os.getenv("RULES_PORTAL_BASE_URL")
        spec_graph: Optional[str] = None
        if not args.graph_url and not env_graph:
            spec_graph = _maybe_get_graph_url_from_spec(ci, args.spec)

        graph = args.graph_url or env_graph or spec_graph or _default_graph_url_from_check_integrity(ci)
        resolved_graph_url = graph

        cands = _normalize_graph_base_candidates(graph)
        shown = cands[0] if cands else graph
        extra = f" (will try: {', '.join(cands)})" if len(cands) > 1 else ""
        src = "cli" if args.graph_url else ("env" if env_graph else ("spec" if spec_graph else "default"))
        print(f"KG pruning enabled for business_rules.json (mode={mode}, graph_url={shown}, source={src}){extra}")

    changed_count = 0
    error_count = 0

    # Only scan for SUPPORTS links warning if prune_only_supports is set
    if args.prune_only_supports:
        # For each business_rules.json, check if any SUPPORTS-kind links exist
        for path in json_files:
            if path.name == "business_rules.json":
                try:
                    docs = decode_multi_json(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                supports_count = 0
                kind_counts = {}
                for doc in docs:
                    # Support both list and dict roots
                    logics = []
                    if isinstance(doc, list):
                        logics = doc
                    elif isinstance(doc, dict) and isinstance(doc.get("logics"), list):
                        logics = doc["logics"]
                    for logic in logics:
                        if not isinstance(logic, dict):
                            continue
                        links = logic.get("links")
                        if not isinstance(links, list):
                            continue
                        for link in links:
                            if not isinstance(link, dict):
                                continue
                            kind = link.get("kind")
                            kind_str = str(kind).upper() if kind is not None else ""
                            kind_counts[kind_str] = kind_counts.get(kind_str, 0) + 1
                            if kind_str in {"SUPPORTS", "SUPPORT"}:
                                supports_count += 1
                if supports_count == 0:
                    # Show top link kinds
                    top_kinds = sorted(kind_counts.items(), key=lambda x: -x[1])
                    top_kinds_str = ", ".join(f"{k or '(missing)'}={v}" for k, v in top_kinds[:3])
                    print(
                        f"WARNING: --prune-only-supports is enabled but no SUPPORTS-kind links were found in JSON. Top link kinds observed: {top_kinds_str}"
                    )

    for path in json_files:
        result = migrate_file(
            path=path,
            backup_root=backup_root,
            timestamp=timestamp,
            preview=preview,
            verbose=verbose,
            prune_missing_kg_links=args.prune_missing_kg_links,
            graph_url=resolved_graph_url,
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
