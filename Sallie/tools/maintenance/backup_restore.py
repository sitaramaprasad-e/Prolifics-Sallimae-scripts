#!/usr/bin/env python3

# Restrict JSON backups to a strict allow-list
ALLOWED_MODEL_JSON_FILES = {
    "business_rules.json",
    "business_areas.json",
    "runs.json",
    "sources.json",
    "teams.json",
    "models.json",
    "components.json",
    "rule_categories.json",
    "audit.json",
    "top-level-suggestions.json",
    "supporting-decisions-suggestions.json",
}

import os
import sys
import json
import shutil
import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import requests  # type: ignore
from requests.exceptions import SSLError  # type: ignore

# ----------------------------
# Logging (matches your style)
# ----------------------------
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def _trace(self, message, *args, **kws):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kws)

logging.Logger.trace = _trace

def _setup_logger(verbose: bool = True):
    level = TRACE_LEVEL_NUM if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

LOG = logging.getLogger("backup_restore")

# ----------------------------
# Helpers
# ----------------------------
def _now_ts_local() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")

def _human_timestamp_from_created_at(created_at: str) -> str:
    """Convert created_at like YYYYMMDD-HHMMSS to human readable YYYY-MM-DD HH:MM:SS."""
    s = (created_at or "").strip()
    if len(s) >= 15 and s[:8].isdigit() and s[9:15].isdigit():
        y = s[0:4]
        mo = s[4:6]
        d = s[6:8]
        hh = s[9:11]
        mm = s[11:13]
        ss = s[13:15]
        return f"{y}-{mo}-{d} {hh}:{mm}:{ss}"
    return ""

def _prompt_with_default(prompt_text: str, default_val: str) -> str:
    try:
        entered = input(f"{prompt_text} (default='{default_val}'): ").strip()
    except EOFError:
        entered = ""
    return entered if entered else default_val

def _confirm(prompt_text: str, default_no: bool = True) -> bool:
    default = "n" if default_no else "y"
    try:
        resp = input(f"{prompt_text} [y/n] (default='{default}'): ").strip().lower()
    except EOFError:
        resp = ""
    if not resp:
        resp = default
    return resp.startswith("y")

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(n)
    for u in units:
        if v < 1024.0:
            return f"{v:.1f}{u}"
        v /= 1024.0
    return f"{v:.1f}PB"

def _list_dir_tree(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root)
            out.append(rel)
    return sorted(out)

# --- Helper: directory stats for file count and total bytes ---
def _dir_stats(root: str) -> Tuple[int, int]:
    """Return (file_count, total_bytes) for all regular files under root."""
    cnt = 0
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            try:
                if os.path.isfile(full):
                    cnt += 1
                    total += os.path.getsize(full)
            except Exception:
                continue
    return cnt, total

# ----------------------------
# KG Export/Import strategies (HTTP JSON folder-based)
# ----------------------------
@dataclass
class KgConfig:
    http_base_url: str
    http_run_path: str = "/api/graph/run"
    http_status_path: str = "/api/graph/status"
    verify_tls: bool = False


def _kg_url(cfg: KgConfig, path: str) -> str:
    return cfg.http_base_url.rstrip("/") + path


# ----------------------------
# KG HTTP helpers: fallback to http:// on SSL error
# ----------------------------

def _with_http_fallback(url: str) -> str:
    if url.startswith("https://"):
        return "http://" + url[len("https://"):]
    return url


def _kg_request(method: str, url: str, *, verify: bool, timeout: int, json_body: Optional[Dict[str, Any]] = None) -> requests.Response:
    """Perform a request, retrying with http:// if https:// fails due to SSL issues."""
    try:
        if method == "GET":
            return requests.get(url, verify=verify, timeout=timeout)
        if method == "POST":
            return requests.post(url, json=json_body, verify=verify, timeout=timeout)
        raise ValueError(f"Unsupported method: {method}")
    except SSLError as e:
        alt = _with_http_fallback(url)
        if alt != url:
            LOG.warning("[kg] SSL error calling %s (%s). Retrying as %s", url, e, alt)
            if method == "GET":
                return requests.get(alt, verify=verify, timeout=timeout)
            if method == "POST":
                return requests.post(alt, json=json_body, verify=verify, timeout=timeout)
        raise


def _kg_check_status(cfg: KgConfig) -> Dict[str, Any]:
    url = _kg_url(cfg, cfg.http_status_path)
    LOG.info("[kg] Status: GET %s", url)
    r = _kg_request("GET", url, verify=cfg.verify_tls, timeout=30)
    # If we fell back to http, keep using it for subsequent calls in this run
    if url.startswith("https://") and r.url.startswith("http://"):
        LOG.warning("[kg] Using HTTP base URL for this run: %s", r.url.rsplit(cfg.http_status_path, 1)[0])
        cfg.http_base_url = r.url.rsplit(cfg.http_status_path, 1)[0]
    if r.status_code != 200:
        raise RuntimeError(f"KG status failed: {r.status_code} {r.text[:200]} (url={r.url})")
    try:
        return r.json()
    except Exception:
        return {"raw": r.text[:500]}


def _kg_run(cfg: KgConfig, cypher: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """Runs Cypher through the graph service. Tries a couple of common payload shapes."""
    url = _kg_url(cfg, cfg.http_run_path)
    payloads = [
        {"query": cypher, "params": params or {}},
        {"cypher": cypher, "params": params or {}},
        {"query": cypher, "parameters": params or {}},
    ]
    last_err = None
    for p in payloads:
        LOG.trace("[kg] POST %s payload_keys=%s", url, list(p.keys()))
        try:
            r = _kg_request("POST", url, json_body=p, verify=cfg.verify_tls, timeout=300)
            if url.startswith("https://") and r.url.startswith("http://"):
                # Persist scheme downgrade for subsequent calls
                cfg.http_base_url = r.url.rsplit(cfg.http_run_path, 1)[0]
            if r.status_code == 200:
                try:
                    return r.json()
                except Exception:
                    return {"raw": r.text}
            last_err = RuntimeError(f"KG run failed: {r.status_code} {r.text[:200]}")
        except Exception as e:
            last_err = e
    raise RuntimeError(str(last_err))


def _extract_rows(result: Any) -> List[Any]:
    """Best-effort extraction of rows from varied server response shapes."""
    if result is None:
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        for k in ("rows", "data", "records", "result", "results"):
            v = result.get(k)
            if isinstance(v, list):
                return v
        # neo4j-driver style
        if "records" in result and isinstance(result["records"], list):
            return result["records"]
    return []

# ----------------------------
# Neo4j JSON normalization / safety
# ----------------------------


def _maybe_neo4j_int(v: Any) -> Optional[int]:
    """Convert neo4j integer map shapes like {low: X, high: Y} into a Python int."""
    if not isinstance(v, dict):
        return None
    if "low" in v and "high" in v:
        try:
            low = int(v.get("low", 0))
            high = int(v.get("high", 0))
            # neo4j int encoding: signed 64-bit split into 32-bit low/high
            val = (high << 32) + (low & 0xFFFFFFFF)
            # convert to signed
            if val >= (1 << 63):
                val -= (1 << 64)
            return int(val)
        except Exception:
            return None
    return None

# --- Neo4j id normalization helper ---
def _normalize_neo_id(v: Any) -> Any:
    """Ensure neo_id is a primitive (int/str), not a Neo4j {low,high} map."""
    ni = _maybe_neo4j_int(v)
    if ni is not None:
        return ni
    # already primitive
    if isinstance(v, (int, str)):
        return v
    # last resort: stringify
    return str(v)


def _sanitize_json_value(v: Any) -> Any:
    """Recursively sanitize values to JSON primitives; converts neo4j int maps to ints."""
    neo_int = _maybe_neo4j_int(v)
    if neo_int is not None:
        return neo_int
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, list):
        return [_sanitize_json_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _sanitize_json_value(val) for k, val in v.items()}
    return str(v)


def _neo4j_safe_props(props: Any) -> Dict[str, Any]:
    """Neo4j property values must be primitives or arrays. Drop nested maps/objects."""
    if not isinstance(props, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in props.items():
        sv = _sanitize_json_value(v)
        if isinstance(sv, (str, int, float, bool)) or sv is None:
            out[str(k)] = sv
        elif isinstance(sv, list) and all(isinstance(x, (str, int, float, bool)) or x is None for x in sv):
            out[str(k)] = sv
        # else: drop it
    return out


def _cypher_escape_label_or_type(name: str) -> str:
    """Escape a label/type for use inside backticks."""
    return "`" + (name or "").replace("`", "``") + "`"


def _kg_export_json(cfg: KgConfig, out_dir: str) -> Dict[str, Any]:
    """Exports nodes + relationships to JSON files under out_dir/kg/."""
    os.makedirs(out_dir, exist_ok=True)
    kg_dir = os.path.join(out_dir, "kg")
    os.makedirs(kg_dir, exist_ok=True)

    _kg_check_status(cfg)

    LOG.info("[kg] Exporting nodes via /api/graph/run")
    nodes_q = """
    MATCH (n)
    RETURN id(n) AS neo_id, labels(n) AS labels, properties(n) AS props
    ORDER BY id(n)
    """.strip()
    nodes_res = _kg_run(cfg, nodes_q)
    nodes_rows = _extract_rows(nodes_res)

    # Normalize common row shapes: dict with keys, or list/tuple
    nodes: List[Dict[str, Any]] = []
    for row in nodes_rows:
        if isinstance(row, dict) and ("neo_id" in row or "neo_id" in row.get("row", {})):
            rr = row.get("row") if isinstance(row.get("row"), dict) else row
            nodes.append({
                "neo_id": rr.get("neo_id"),
                "labels": rr.get("labels") or [],
                "props": _sanitize_json_value(rr.get("props") or {}),
            })
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            nodes.append({"neo_id": row[0], "labels": row[1] or [], "props": _sanitize_json_value(row[2] or {})})
        else:
            # last resort: keep raw
            nodes.append({"raw": row})

    LOG.info("[kg] Exporting relationships via /api/graph/run")
    rels_q = """
    MATCH (a)-[r]->(b)
    RETURN id(r) AS neo_id, type(r) AS type, id(a) AS from_neo_id, id(b) AS to_neo_id, properties(r) AS props
    ORDER BY id(r)
    """.strip()
    rels_res = _kg_run(cfg, rels_q)
    rels_rows = _extract_rows(rels_res)

    rels: List[Dict[str, Any]] = []
    for row in rels_rows:
        if isinstance(row, dict) and ("neo_id" in row or "neo_id" in row.get("row", {})):
            rr = row.get("row") if isinstance(row.get("row"), dict) else row
            rels.append({
                "neo_id": rr.get("neo_id"),
                "type": rr.get("type"),
                "from_neo_id": rr.get("from_neo_id"),
                "to_neo_id": rr.get("to_neo_id"),
                "props": _sanitize_json_value(rr.get("props") or {}),
            })
        elif isinstance(row, (list, tuple)) and len(row) >= 5:
            rels.append({"neo_id": row[0], "type": row[1], "from_neo_id": row[2], "to_neo_id": row[3], "props": _sanitize_json_value(row[4] or {})})
        else:
            rels.append({"raw": row})

    nodes_path = os.path.join(kg_dir, "nodes.json")
    rels_path = os.path.join(kg_dir, "relationships.json")
    _safe_write_json(nodes_path, {"nodes": nodes})
    _safe_write_json(rels_path, {"relationships": rels})

    return {
        "strategy": "http-run-json",
        "status_url": _kg_url(cfg, cfg.http_status_path),
        "run_url": _kg_url(cfg, cfg.http_run_path),
        "nodes": len(nodes),
        "relationships": len(rels),
        "files": {
            "nodes": "kg/nodes.json",
            "relationships": "kg/relationships.json",
        },
    }


def _kg_restore_json(cfg: KgConfig, backup_root: str, batch_size: int = 500) -> Dict[str, Any]:
    """Restores graph from backup_root/kg/*.json. Fully replaces existing graph."""
    _kg_check_status(cfg)

    kg_dir = os.path.join(backup_root, "kg")
    nodes_path = os.path.join(kg_dir, "nodes.json")
    rels_path = os.path.join(kg_dir, "relationships.json")
    if not os.path.exists(nodes_path) or not os.path.exists(rels_path):
        raise RuntimeError("Backup is missing kg/nodes.json or kg/relationships.json")

    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes_obj = json.load(f)
    with open(rels_path, "r", encoding="utf-8") as f:
        rels_obj = json.load(f)

    nodes = nodes_obj.get("nodes", [])
    rels = rels_obj.get("relationships", [])

    LOG.info("[kg] Clearing existing graph (DETACH DELETE)…")
    _kg_run(cfg, "MATCH (n) DETACH DELETE n")

    # Create nodes with safe props and a marker for later linking
    marker = "__backup_neo_id"

    LOG.info("[kg] Restoring %d nodes…", len(nodes))
    created = 0
    for i in range(0, len(nodes), batch_size):
        chunk = nodes[i:i + batch_size]
        rows = []
        for row in chunk:
            if not isinstance(row, dict):
                continue
            rows.append({
                "neo_id": _normalize_neo_id(row.get("neo_id")),
                "labels": row.get("labels") or [],
                "props": _neo4j_safe_props(row.get("props") or {}),
            })

        cypher = """
        UNWIND $rows AS row
        CREATE (n)
        SET n += row.props
        SET n.__backup_neo_id = row.neo_id
        RETURN count(*) AS c
        """.strip()
        _kg_run(cfg, cypher, {"rows": rows})
        created += len(rows)

    LOG.info("[kg] Nodes restored (without labels yet): %d", created)

    # Apply labels without APOC by grouping per label
    label_to_ids: Dict[str, List[Any]] = {}
    for row in nodes:
        if not isinstance(row, dict):
            continue
        nid = _normalize_neo_id(row.get("neo_id"))
        for lab in (row.get("labels") or []):
            if not isinstance(lab, str) or not lab:
                continue
            label_to_ids.setdefault(lab, []).append(nid)

    if label_to_ids:
        LOG.info("[kg] Applying %d distinct labels…", len(label_to_ids))
    for lab, ids in label_to_ids.items():
        if not ids:
            continue
        lab_esc = _cypher_escape_label_or_type(lab)
        for j in range(0, len(ids), batch_size * 2):
            chunk_ids = ids[j:j + (batch_size * 2)]
            cypher = f"""
            UNWIND $ids AS id
            MATCH (n {{{marker}: id}})
            SET n:{lab_esc}
            RETURN count(*) AS c
            """.strip()
            _kg_run(cfg, cypher, {"ids": chunk_ids})

    # Restore relationships without APOC by grouping per relationship type
    LOG.info("[kg] Restoring %d relationships…", len(rels))
    type_to_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rels:
        if not isinstance(row, dict):
            continue
        rtype = row.get("type")
        if not isinstance(rtype, str) or not rtype:
            continue
        type_to_rows.setdefault(rtype, []).append({
            "from_neo_id": _normalize_neo_id(row.get("from_neo_id")),
            "to_neo_id": _normalize_neo_id(row.get("to_neo_id")),
            "props": _neo4j_safe_props(row.get("props") or {}),
        })

    rel_created = 0
    for rtype, rows in type_to_rows.items():
        if not rows:
            continue
        rtype_esc = _cypher_escape_label_or_type(rtype)
        for j in range(0, len(rows), batch_size):
            chunk = rows[j:j + batch_size]
            cypher = f"""
            UNWIND $rows AS row
            MATCH (a {{{marker}: row.from_neo_id}})
            MATCH (b {{{marker}: row.to_neo_id}})
            MERGE (a)-[r:{rtype_esc}]->(b)
            SET r += row.props
            RETURN count(*) AS c
            """.strip()
            _kg_run(cfg, cypher, {"rows": chunk})
            rel_created += len(chunk)

    LOG.info("[kg] Relationships restored: %d", rel_created)

    LOG.info("[kg] Cleaning marker property %s…", marker)
    _kg_run(cfg, f"MATCH (n) REMOVE n.{marker}")

    return {"strategy": "http-run-json", "nodes": created, "relationships": rel_created}

# ----------------------------
# Backup model artifacts
# ----------------------------
def _discover_model_artifacts(model_home: str) -> Dict[str, Any]:
    model_dir = os.path.join(model_home, ".model")
    if not os.path.isdir(model_dir):
        raise RuntimeError(f"Model directory not found: {model_dir}")

    # JSON files in .model (strict allow-list)
    json_files = []
    for fn in os.listdir(model_dir):
        if fn in ALLOWED_MODEL_JSON_FILES:
            json_files.append(os.path.join(model_dir, fn))

    missing = sorted(ALLOWED_MODEL_JSON_FILES - {os.path.basename(p) for p in json_files})
    if missing:
        LOG.warning(
            "[warn] Some expected model JSON files are missing and will not be backed up: %s",
            ", ".join(missing),
        )

    json_files.sort()

    # Optional sibling folders (based on your earlier packaging convention)
    opt_dirs = []
    for d in ("prompts", "hints", "filters"):
        p = os.path.join(model_home, d)
        if os.path.isdir(p):
            opt_dirs.append(p)

    # Optional tmp files directory under .model (must be backed up/restored if present)
    tmp_files_dir = os.path.join(model_dir, ".tmp", "files")
    tmp_files_dir_included = tmp_files_dir if os.path.isdir(tmp_files_dir) else None

    return {
        "model_dir": model_dir,
        "json_files": json_files,
        "optional_dirs": opt_dirs,
        "tmp_files_dir": tmp_files_dir_included,
    }

def _copy_artifacts_to_staging(model_home: str, staging_root: str) -> Dict[str, Any]:
    meta = _discover_model_artifacts(model_home)
    model_dir = meta["model_dir"]
    json_files: List[str] = meta["json_files"]
    opt_dirs: List[str] = meta["optional_dirs"]
    tmp_files_dir = meta.get("tmp_files_dir")

    included = []
    bytes_total = 0

    # Copy .model/*.json
    dest_model_dir = os.path.join(staging_root, ".model")
    os.makedirs(dest_model_dir, exist_ok=True)

    for src in json_files:
        rel = os.path.relpath(src, model_home)
        dest = os.path.join(staging_root, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)
        sz = os.path.getsize(dest)
        bytes_total += sz
        included.append({"path": rel, "bytes": sz, "sha256": _sha256_file(dest)})

    # Copy optional dirs (prompts/hints/filters)
    for d in opt_dirs:
        rel_dir = os.path.relpath(d, model_home)
        dest_dir = os.path.join(staging_root, rel_dir)
        shutil.copytree(d, dest_dir)
        for rel in _list_dir_tree(dest_dir):
            full = os.path.join(dest_dir, rel)
            if os.path.isfile(full):
                sz = os.path.getsize(full)
                bytes_total += sz
                included.append({"path": os.path.join(rel_dir, rel), "bytes": sz, "sha256": _sha256_file(full)})

    # Copy .model/.tmp/files (always include)
    if tmp_files_dir and os.path.isdir(tmp_files_dir):
        tmp_cnt, tmp_bytes = _dir_stats(tmp_files_dir)
        LOG.info("[info] Including .model/.tmp/files • files=%d • size=%s", tmp_cnt, _human_bytes(tmp_bytes))

        rel_dir = os.path.relpath(tmp_files_dir, model_home)  # ".model/.tmp/files"
        dest_dir = os.path.join(staging_root, rel_dir)

        # Ensure parent exists, then copy
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        if os.path.isdir(dest_dir):
            shutil.rmtree(dest_dir, ignore_errors=True)

        shutil.copytree(tmp_files_dir, dest_dir)

        for rel in _list_dir_tree(dest_dir):
            full = os.path.join(dest_dir, rel)
            if os.path.isfile(full):
                sz = os.path.getsize(full)
                bytes_total += sz
                included.append({
                    "path": os.path.join(rel_dir, rel).replace("\\", "/"),
                    "bytes": sz,
                    "sha256": _sha256_file(full),
                })

    return {"included_files": included, "bytes_total": bytes_total}

# ----------------------------
# Backup/Restore orchestration
# ----------------------------
def _backups_dir(model_home: str) -> str:
    return os.path.join(model_home, ".model", "backups")


# Helper for backup sort: prefer manifest created_at, fallback to mtime, tie-break by basename
def _backup_sort_key(path: str) -> Tuple[int, str]:
    """Return a sort key where larger means newer. Uses manifest created_at if available, else mtime."""
    man_path = os.path.join(path, "manifest.json")
    ts_val = 0
    try:
        with open(man_path, "r", encoding="utf-8") as f:
            man = json.load(f)
            created = (man.get("created_at") or "").strip()
            # expected format YYYYMMDD-HHMMSS
            if len(created) >= 15 and created[:8].isdigit() and created[9:15].isdigit():
                ts_val = int(created.replace("-", ""))
    except Exception:
        ts_val = 0

    if ts_val == 0:
        try:
            ts_val = int(os.path.getmtime(path) * 1000)
        except Exception:
            ts_val = 0

    # include basename for stable tie-break
    return (ts_val, os.path.basename(path))

def _list_backups(model_home: str) -> List[str]:
    bdir = _backups_dir(model_home)
    if not os.path.isdir(bdir):
        return []
    items = []
    for fn in os.listdir(bdir):
        full = os.path.join(bdir, fn)
        if os.path.isdir(full) and fn[:8].isdigit():
            items.append(full)
    items.sort(key=_backup_sort_key, reverse=True)
    return items

def _show_backup_summary(backup_dir: str) -> None:
    man_path = os.path.join(backup_dir, "manifest.json")
    man = None
    try:
        with open(man_path, "r", encoding="utf-8") as f:
            man = json.load(f)
    except Exception:
        man = None
    print("")
    print("=== Backup details ===")
    print(f"Backup: {backup_dir}")
    if not man:
        print("manifest.json: (missing or unreadable)")
        return
    print(f"Created : {man.get('created_at')}")
    print(f"Model   : {man.get('model_home')}")
    kgmeta = man.get("kg", {}).get("export_meta", {})
    print(f"KG meta : {kgmeta}")
    files = man.get("files", [])
    total_bytes = man.get("total_bytes", 0)
    print(f"Files   : {len(files)} • Total {_human_bytes(int(total_bytes or 0))}")
    # Show top few files
    for f in files[:12]:
        print(f"  - {f.get('path')} ({_human_bytes(int(f.get('bytes') or 0))})")
    if len(files) > 12:
        print(f"  … +{len(files) - 12} more")

def _create_backup(model_home: str, kg_cfg: KgConfig) -> str:
    bdir = _backups_dir(model_home)
    os.makedirs(bdir, exist_ok=True)

    ts = _now_ts_local()
    backup_root = os.path.join(bdir, ts)
    os.makedirs(backup_root, exist_ok=True)

    LOG.info("=== [step 1/3] Collect model artifacts ===")
    artifacts_meta = _copy_artifacts_to_staging(model_home, backup_root)
    LOG.info("[info] Collected %d files • %s",
             len(artifacts_meta["included_files"]), _human_bytes(artifacts_meta["bytes_total"]))

    LOG.info("=== [step 2/3] Export Knowledge Graph ===")
    kg_export_meta = {}
    try:
        kg_export_meta = _kg_export_json(kg_cfg, backup_root)
        LOG.info("[info] KG export OK • %s", kg_export_meta)
    except Exception as e:
        LOG.error("[error] KG export failed: %s", e)
        raise

    LOG.info("=== [step 3/3] Write manifest ===")
    files = artifacts_meta["included_files"]
    # Add KG JSON file info to file list
    kg_dir = os.path.join(backup_root, "kg")
    for kg_file in ("nodes.json", "relationships.json"):
        kgf = os.path.join(kg_dir, kg_file)
        if os.path.exists(kgf):
            files.append({
                "path": os.path.relpath(kgf, backup_root).replace("\\", "/"),
                "bytes": os.path.getsize(kgf),
                "sha256": _sha256_file(kgf),
            })
    total_bytes = sum(int(f.get("bytes") or 0) for f in files)
    manifest = {
        "created_at": ts,
        "model_home": os.path.expanduser(model_home),
        "files": sorted(files, key=lambda x: x.get("path") or ""),
        "total_bytes": total_bytes,
        "kg": {
            "export_meta": kg_export_meta,
        },
    }
    _safe_write_json(os.path.join(backup_root, "manifest.json"), manifest)
    LOG.info("[info] Manifest written (files=%d total=%s)", len(files), _human_bytes(total_bytes))

    LOG.info("[success] Backup created: %s", backup_root)
    print(f"\n[success] Backup created: {backup_root}")
    print(f"  - JSON files: {len([f for f in files if f['path'].startswith('.model/')])}")
    opt_list = [d for d in ("prompts", "hints", "filters") if os.path.isdir(os.path.join(backup_root, d))]
    if os.path.isdir(os.path.join(backup_root, ".model", ".tmp", "files")):
        opt_list.append(".model/.tmp/files")
    print(f"  - Optional dirs: {', '.join(opt_list)}")
    print(f"  - KG: nodes={kg_export_meta.get('nodes', 0)}, relationships={kg_export_meta.get('relationships', 0)}")
    return backup_root

def _restore_backup(model_home: str, kg_cfg: KgConfig, backup_root: str) -> None:
    if not os.path.isdir(backup_root):
        raise RuntimeError(f"Backup folder not found: {backup_root}")

    LOG.info("=== Inspecting backup before restore ===")
    _show_backup_summary(backup_root)

    if not _confirm("Proceed to RESTORE this backup? This will REPLACE existing JSON and KG.", default_no=True):
        LOG.info("[info] Restore cancelled.")
        return

    man_path = os.path.join(backup_root, "manifest.json")
    if not os.path.exists(man_path):
        raise RuntimeError("manifest.json missing in backup — refusing to restore.")
    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    LOG.info("=== [step 1/3] Replace model JSON artifacts ===")
    # Replace .model/*.json from backup
    backup_model_dir = os.path.join(backup_root, ".model")
    live_model_dir = os.path.join(model_home, ".model")
    if not os.path.isdir(backup_model_dir):
        raise RuntimeError("Backup missing .model directory")
    # Remove only allowed live .model/*.json (replace means replace, but only allow-list)
    removed = 0
    for fn in ALLOWED_MODEL_JSON_FILES:
        p = os.path.join(live_model_dir, fn)
        if os.path.exists(p):
            try:
                os.remove(p)
                removed += 1
            except Exception as e:
                LOG.warning("[warn] Failed removing %s: %s", fn, e)
    # Copy only allowed backup json into live
    restored = 0
    for fn in ALLOWED_MODEL_JSON_FILES:
        src = os.path.join(backup_model_dir, fn)
        if not os.path.exists(src):
            continue
        dst = os.path.join(live_model_dir, fn)
        shutil.copy2(src, dst)
        restored += 1
    LOG.info("[info] JSON replace complete • removed=%d restored=%d", removed, restored)
    # Optional dirs (prompts/hints/filters) - replace if present in backup
    for d in ("prompts", "hints", "filters"):
        src_dir = os.path.join(backup_root, d)
        dst_dir = os.path.join(model_home, d)
        if os.path.isdir(src_dir):
            if os.path.isdir(dst_dir):
                shutil.rmtree(dst_dir, ignore_errors=True)
            shutil.copytree(src_dir, dst_dir)
            LOG.info("[info] Replaced dir: %s", d)

    # .model/.tmp/files (replace if present in backup)
    backup_tmp_files_dir = os.path.join(backup_root, ".model", ".tmp", "files")
    live_tmp_files_dir = os.path.join(live_model_dir, ".tmp", "files")
    if os.path.isdir(backup_tmp_files_dir):
        # Ensure live parent exists
        os.makedirs(os.path.dirname(live_tmp_files_dir), exist_ok=True)
        if os.path.isdir(live_tmp_files_dir):
            shutil.rmtree(live_tmp_files_dir, ignore_errors=True)
        shutil.copytree(backup_tmp_files_dir, live_tmp_files_dir)
        LOG.info("[info] Replaced dir: .model/.tmp/files")

    LOG.info("=== [step 2/3] Restore Knowledge Graph (full replace) ===")
    kg_dir = os.path.join(backup_root, "kg")
    if not os.path.isdir(kg_dir):
        raise RuntimeError("Backup missing kg/ directory")
    nodes_path = os.path.join(kg_dir, "nodes.json")
    rels_path = os.path.join(kg_dir, "relationships.json")
    if not (os.path.exists(nodes_path) and os.path.exists(rels_path)):
        raise RuntimeError("Backup missing kg/nodes.json or kg/relationships.json")
    # Show KG counts
    try:
        with open(nodes_path, "r", encoding="utf-8") as f:
            nodes_obj = json.load(f)
        with open(rels_path, "r", encoding="utf-8") as f:
            rels_obj = json.load(f)
        LOG.info("[info] KG backup contains nodes=%d, relationships=%d",
                 len(nodes_obj.get("nodes", [])), len(rels_obj.get("relationships", [])))
    except Exception:
        pass
    meta = _kg_restore_json(kg_cfg, backup_root)
    LOG.info("[success] KG restore complete • %s", meta)

    LOG.info("=== [step 3/3] Restore finished ===")
    LOG.info("[success] Restore finished. JSON + KG replaced from backup: %s", backup_root)

def _print_existing_backups(backups: List[str]) -> None:
    if not backups:
        print("No backups found.")
        return
    print("=== Existing backups ===")
    for i, p in enumerate(backups, start=1):
        man_path = os.path.join(p, "manifest.json")
        ts = "unknown"
        size = ""
        try:
            with open(man_path, "r", encoding="utf-8") as f:
                man = json.load(f)
                ts = man.get("created_at") or "unknown"
                size = _human_bytes(int(man.get("total_bytes", 0) or 0))
        except Exception:
            pass
        human_ts = _human_timestamp_from_created_at(ts)
        if human_ts:
            print(f"  {i}. {os.path.basename(p)} • {ts} ({human_ts}) • {size}")
        else:
            print(f"  {i}. {os.path.basename(p)} • {ts} • {size}")

def _select_backup_interactively(backups: List[str]) -> Optional[str]:
    if not backups:
        print("No backups found.")
        return None
    print("")
    _print_existing_backups(backups)
    try:
        resp = input("Choose a backup number to restore (or press ENTER to cancel): ").strip()
    except EOFError:
        resp = ""
    if not resp:
        return None
    try:
        idx = int(resp)
    except Exception:
        return None
    if idx < 1 or idx > len(backups):
        return None
    return backups[idx - 1]

def main():
    _setup_logger(verbose=True)

    # Defaults
    default_model_home = os.path.expanduser("~")
    model_home = os.path.expanduser(_prompt_with_default("Enter model home path", default_model_home))

    # List backups FIRST (requirement)
    backups = _list_backups(model_home)
    print("")
    print("=== Backup/Restore ===")
    print(f"Backups directory: {_backups_dir(model_home)}")
    if backups:
        print(f"Found {len(backups)} backup(s).")
    _print_existing_backups(backups)

    print("")
    print("Choose an action:")
    print("  1) Restore an existing backup")
    print("  2) Take a new backup")
    try:
        action = input("Enter number: ").strip()
    except EOFError:
        action = ""

    if action not in {"1", "2"}:
        LOG.info("[info] No action selected. Exiting.")
        return 0

    # Prompt KG details (HTTP only)
    default_base = os.environ.get("PCPT_GRAPH_BASE", "https://localhost:443")
    default_run = os.environ.get("PCPT_GRAPH_RUN", "/api/graph/run")
    default_status = os.environ.get("PCPT_GRAPH_STATUS", "/api/graph/status")
    verify_tls = (os.environ.get("PCPT_GRAPH_VERIFY", "false").lower() in {"1", "true", "yes"})
    verify_str_default = "true" if verify_tls else "false"
    http_base_url = _prompt_with_default("KG HTTP base url", default_base)
    http_run_path = _prompt_with_default("KG HTTP run path", default_run)
    http_status_path = _prompt_with_default("KG HTTP status path", default_status)
    verify_str = _prompt_with_default("Verify TLS certs for HTTP KG endpoints? (true/false)", verify_str_default).lower()
    verify_tls = verify_str in {"1", "true", "yes"}

    kg_cfg = KgConfig(
        http_base_url=http_base_url,
        http_run_path=http_run_path,
        http_status_path=http_status_path,
        verify_tls=verify_tls,
    )

    if action == "2":
        # New backup
        backup_path = _create_backup(model_home, kg_cfg)
        print("")
        _show_backup_summary(backup_path)
        return 0

    # Restore
    sel = _select_backup_interactively(backups)
    if not sel:
        LOG.info("[info] No backup selected; exiting.")
        return 0
    _restore_backup(model_home, kg_cfg, sel)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[info] cancelled")
        sys.exit(130)