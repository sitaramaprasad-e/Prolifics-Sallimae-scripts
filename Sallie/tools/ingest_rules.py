import re
import json
import sys
import os
import uuid
import glob
import hashlib

# ===== Lightweight TRACE logger (inline; no external files) =====
import logging

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

LOG = logging.getLogger("ingest_rules")

from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any

# ===== Execution & Artifact linking (new) =====

LOG_DIR = os.environ.get("PCPT_LOG_DIR", os.path.expanduser("~/.pcpt/log"))

def _prompt_model_home() -> str:
    try:
        resp = input("Enter model home path (default='~/'):").strip()
    except EOFError:
        # Non-interactive (e.g., piped/cron) – fall back to default
        resp = ""
    if not resp:
        resp = "~"
    return os.path.expanduser(resp)

MODEL_HOME = _prompt_model_home()

# Enable TRACE logging by default and announce script start
_setup_logger(verbose=True)
LOG.info("[info] ingest_rules starting • MODEL_HOME=%s", MODEL_HOME)

# ===== PCPT Header parsing for model/sources.json & model/runs.json (new) =====
PCPT_PREFIX  = "[PCPTLOG:]"
# Robust markers that ignore exact chevron counts; match by phrase
HEADER_BEGIN_RX = re.compile(r"HEADER\s+BEGIN")
HEADER_END_RX   = re.compile(r"HEADER\s+END")
# RESPONSE block markers for logs (case-insensitive)
RESP_BEGIN_RX = re.compile(r"RESPONSE\s+BEGIN", re.IGNORECASE)
RESP_END_RX   = re.compile(r"RESPONSE\s+END", re.IGNORECASE)
KV_LINE = re.compile(rf"^{re.escape(PCPT_PREFIX)}\s+(?P<k>[a-zA-Z0-9_]+)=(?P<v>.*)$")
SOURCE_JSON = f"{MODEL_HOME}/.model/sources.json"
RUNS_JSON   = f"{MODEL_HOME}/.model/runs.json"
MIN_BUILD_NUM = 2510020935  # ignore headers from builds before this
TEAMS_JSON = f"{MODEL_HOME}/.model/teams.json"
COMPONENTS_JSON = f"{MODEL_HOME}/.model/components.json"
PROMPT_PREFIX_FILTER = "business_rules_report"

from datetime import timezone

_AUDIT = []  # list of dicts

def _audit_add(
    kind: str,
    path: str,
    source: str,
    decision: str,
    reason: str,
    tests: Optional[Dict[str, Any]] = None,
    derived: Optional[Dict[str, Any]] = None,
):
    _AUDIT.append({
        "kind": kind,
        "path": path,
        "source": source,
        "decision": decision,
        "reason": reason,
        "tests": tests or {},
        "derived": derived or {},
    })

def _audit_write(prefix: str) -> str:
    try:
        ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(os.getcwd(), ".audit")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{prefix}_audit_{ts}.json")
        meta = {
            "script": "ingest_rules.py",
            "model_home": MODEL_HOME,
            "source_json": SOURCE_JSON,
            "runs_json": RUNS_JSON,
            "input_file": input_file if 'input_file' in globals() else "",
            "timestamp": ts,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "entries": _AUDIT}, f, indent=2)
        return out_path
    except Exception as e:
        LOG.trace("[trace] audit write failed: %s", e)
        return ""

# Log parsing patterns
RE_OUTPUT_REPORT = re.compile(r"^\s*Output report:\s*(?P<path>.+)\s*$", re.IGNORECASE)
RE_OUTPUT_ALSO  = re.compile(r"^\s*Output file created at:\s*(?P<path>.+)\s*$", re.IGNORECASE)
RE_FILE_HEADER  = re.compile(r"^\s*File:\s*(?P<path>.+?)\s*$")
RE_FENCE_START  = re.compile(r"^\s*```")
RE_REPORT_SAVED = re.compile(r"^\s*(Report saved|Saved report|Saving report):\s*(?P<path>.+)\s*$", re.IGNORECASE)
RE_OUTPUT_GENERIC = re.compile(r"^\s*(Output|Wrote|Saved):\s*(?P<path>.+)\s*$", re.IGNORECASE)

# Accept input file path from command line argument
if len(sys.argv) < 2:
    print("Usage: python ingest_rules.py <input_file> [--force | --force-load]")
    sys.exit(1)

input_file = sys.argv[1]

LOG.info("[info] Input report   : %s", os.path.abspath(input_file))

# Optional switch to force load (even if same name + timestamp already exists)
FORCE_LOAD = any(arg in ("--force", "--force-load") for arg in sys.argv[2:])
if FORCE_LOAD:
    LOG.info("[info] Force-load enabled: ingesting even if (rule_name,timestamp) already exists")

# Optional mode: ingest rules from previous iterations of the same report (via PCPT logs)
ALL_RUNS = any(arg == "--all-runs" for arg in sys.argv[2:]) or \
           (os.environ.get("PCPT_INGEST_ALL_RUNS", "").lower() in {"1", "true", "yes"})
if ALL_RUNS:
    LOG.info("[info] All-runs mode: will also ingest rules from previous iterations via PCPT logs")

file_mtime = os.path.getmtime(input_file)
file_timestamp = datetime.utcfromtimestamp(file_mtime).replace(microsecond=0).isoformat() + "Z"

# Helper to parse ISO timestamps safely
def _parse_ts_safe(ts: str):
    try:
        t = str(ts or "").strip()
        if not t:
            return None
        if t.endswith('Z'):
            t = t[:-1]
        return datetime.fromisoformat(t)
    except Exception:
        return None

# Gather prior responses for the same output file using PCPT logs
def _gather_prior_responses_for_same_output(out_path: str) -> List[Tuple[str, str]]:
    """Return list of (timestamp, response_text) for previous runs based on a content anchor:
    1) Anchor to the single run whose RESPONSE contains the full current report text (content match).
    2) Then, from ALL runs, collect prior responses whose output_file tail matches the current report AND whose (root_dir, source_path, output_path) equal the anchor run's triple.
    Results are sorted oldest-first. Returns an empty list if none are found.
    """
    try:
        logs = _discover_pcpt_header_files()
        _, runs = _build_sources_and_runs_from_logs(logs)
    except Exception:
        return []

    def _tail2_key(p: str) -> str:
        try:
            n = os.path.normpath(str(p or "").strip())
            parts = [seg for seg in n.replace("\\", "/").split("/") if seg]
            if not parts:
                return ""
            if len(parts) >= 2:
                return "/".join(parts[-2:]).casefold()
            return parts[-1].casefold()
        except Exception:
            return ""

    target_key = _tail2_key(out_path)
    if not target_key:
        return []
    # Compute absolute path of current output and log exact match targets
    try:
        abs_target = os.path.abspath(out_path)
    except Exception:
        abs_target = ""
    LOG.info("[match] Current output • abs='%s' • tail='%s'", abs_target, target_key)

    # Establish anchor (root_dir, source_path, output_path) from a run that matches the *current report* by content.
    anchor: Tuple[Optional[str], Optional[str], Optional[str]] = (None, None, None)
    anchor_run: Optional[dict] = None

    # Normalize current report text for content-based matching
    doc_norm = ""
    try:
        with open(out_path, "r", encoding="utf-8", errors="ignore") as df:
            doc_norm = _normalize_text(df.read())
    except Exception:
        doc_norm = ""

    # (1) Anchor by content match only (runs are newest-first already)
    if doc_norm:
        for r in runs:
            resp = _normalize_text(r.get("response_text") or "")
            out_file_hdr = r.get("output_file")
            if not out_file_hdr:
                continue
            if resp and (doc_norm in resp) and (_tail2_key(out_file_hdr) == target_key):
                anchor = (r.get("root_dir"), r.get("source_path"), r.get("output_path"))
                anchor_run = r
                LOG.info("[match] Report matched to run via content • ts=%s • build=%s • provider=%s • model=%s • log=%s",
                         r.get("timestamp"), r.get("build"), r.get("provider"), r.get("model"), r.get("log_file"))
                break

    a_root, a_src, a_outp = anchor
    if not (a_root and a_src and a_outp):
        LOG.info("[match] No content anchor found — the current report's text did not appear in any run's RESPONSE. Skipping prior-runs ingestion.")
        return []

    # Print the exact targets and the run we matched on for the report
    LOG.info("[match] Targets • root_dir='%s' • source_path='%s' • output_path='%s' • output_file.tail='%s'", a_root, a_src, a_outp, target_key)
    LOG.info("[match] Using anchor run • ts=%s • build=%s • provider=%s • model=%s • root_dir='%s' • source_path='%s' • output_path='%s' • output_file='%s' • log=%s",
             anchor_run.get("timestamp"), anchor_run.get("build"), anchor_run.get("provider"), anchor_run.get("model"),
             anchor_run.get("root_dir"), anchor_run.get("source_path"), anchor_run.get("output_path"),
             anchor_run.get("output_file"), anchor_run.get("log_file"))

    # Start by ingesting the run for the current report (the anchor itself)
    anchor_ts = str(anchor_run.get("timestamp") or "")
    anchor_resp = (anchor_run.get("response_text") or "").strip()
    items: List[Tuple[str, str]] = []
    if anchor_resp:
        items.append((anchor_ts, anchor_resp))
    tail_hits = 0
    full_hits = 0
    tail_only_mismatches = 0
    tail_mismatch_samples: List[Dict[str, str]] = []
    # Will collect earlier matches here
    earlier_items: List[Tuple[str, str]] = []
    for r in runs:
        ts = r.get("timestamp") or ""
        resp = (r.get("response_text") or "").strip()
        if not resp:
            continue

        out_file_hdr = r.get("output_file")
        if not out_file_hdr:
            continue

        is_tail = (_tail2_key(out_file_hdr) == target_key)
        if is_tail:
            tail_hits += 1
            # Per-run concise trace of what we're matching against (only when tail matches to limit noise)
            eq_root = (r.get("root_dir") == a_root)
            eq_src  = (r.get("source_path") == a_src)
            eq_outp = (r.get("output_path") == a_outp)
            LOG.trace("[check] tail OK • run.tail='%s' == target.tail='%s' • root('%s' == '%s')=%s • src('%s' == '%s')=%s • out('%s' == '%s')=%s • log=%s",
                      _tail2_key(out_file_hdr), target_key,
                      r.get("root_dir"), a_root, eq_root,
                      r.get("source_path"), a_src, eq_src,
                      r.get("output_path"), a_outp, eq_outp,
                      r.get("log_file"))
        full_match = (is_tail
                    and r.get("root_dir") == a_root
                    and r.get("source_path") == a_src
                    and r.get("output_path") == a_outp)
        if full_match:
            LOG.trace("[check] FULL MATCH • tail='%s' • root='%s' • src='%s' • out='%s' • log=%s",
                      target_key, a_root, a_src, a_outp, r.get("log_file"))
            full_hits += 1
            # Only consider runs earlier than the anchor
            if (_parse_ts_safe(ts) or datetime.min) < (_parse_ts_safe(anchor_ts) or datetime.max):
                earlier_items.append((ts, resp))
        elif is_tail:
            tail_only_mismatches += 1
            if len(tail_mismatch_samples) < 5:
                tail_mismatch_samples.append({
                    "run_root": str(r.get("root_dir") or ""),
                    "run_source": str(r.get("source_path") or ""),
                    "run_output": str(r.get("output_path") or ""),
                    "run_outfile": str(out_file_hdr or ""),
                    "log": str(r.get("log_file") or "")
                })

    # Sort earlier items oldest-first, then append after the anchor
    earlier_items.sort(key=lambda t: _parse_ts_safe(t[0]) or datetime.min)
    items.extend(earlier_items)
    LOG.info("[info] Earlier matching runs found: %d (added after anchor)", len(earlier_items))
    LOG.info("[info] Prior iterations summary • criteria tail='%s' • root='%s' • src='%s' • out='%s' • considered=%d • tail-matches=%d • full-matches=%d • tail-only-mismatches=%d",
             target_key, a_root, a_src, a_outp, len(runs), tail_hits, full_hits, tail_only_mismatches)
    for sm in tail_mismatch_samples:
        LOG.info("[mismatch] tail matched but root/src/out differ • run_root='%s' • run_src='%s' • run_out='%s' • run_outfile='%s' • log='%s'",
                 sm["run_root"], sm["run_source"], sm["run_output"], sm["run_outfile"], sm["log"])
    for ts, _ in items:
        _audit_add(kind="run", path=str(out_path), source="pcpt_logs", decision="considered", reason="prior iteration", tests={"timestamp": ts}, derived={})
    return items

# Helper for normalized deduplication key
def _dedupe_key(rule_name, timestamp):
    """Build a stable key from rule_name and timestamp with normalization."""
    if rule_name is None or timestamp is None:
        return None
    rn = str(rule_name).strip()
    ts = str(timestamp).strip()
    # Normalize timestamp to drop microseconds
    if '.' in ts:
        ts = ts.split('.')[0] + 'Z'
    return (rn, ts)

def _now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# Helper: prompt with default value
def _prompt_with_default(prompt_text: str, default_val: str) -> str:
    try:
        entered = input(f"{prompt_text} (default='{default_val}'): ").strip()
    except EOFError:
        entered = ""
    return entered if entered else default_val


def _save_json_file(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

# Helper: append a unique non-empty string to a JSON list file
def _append_unique_value(list_path: str, value: str) -> None:
    """Append `value` into a JSON list inside a keyed object:
       teams.json -> { "teams": [...] }
       components.json -> { "components": [...] }
       Backward-compatible with legacy bare arrays.
    """
    val = (value or "").strip()
    if not val:
        return

    os.makedirs(os.path.dirname(list_path), exist_ok=True)

    filename = os.path.basename(list_path)
    key = "teams" if filename == "teams.json" else ("components" if filename == "components.json" else None)

    data_list = []
    try:
        if os.path.exists(list_path) and os.path.getsize(list_path) > 0:
            with open(list_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            if isinstance(loaded, dict) and key and isinstance(loaded.get(key), list):
                data_list = [str(x) for x in loaded[key] if isinstance(x, (str, int, float))]
            elif isinstance(loaded, list):
                # legacy bare array – accept, but we'll write back the keyed object
                data_list = [str(x) for x in loaded if isinstance(x, (str, int, float))]
            else:
                # best effort: first list value inside the dict
                if isinstance(loaded, dict):
                    for v in loaded.values():
                        if isinstance(v, list):
                            data_list = [str(x) for x in v if isinstance(x, (str, int, float))]
                            break
    except Exception:
        data_list = []

    if val not in data_list:
        data_list.append(val)

    with open(list_path, "w", encoding="utf-8") as f:
        if key:
            json.dump({key: data_list}, f, indent=2, ensure_ascii=False)
        else:
            # fallback for unknown registries; keep original behavior
            json.dump(data_list, f, indent=2, ensure_ascii=False)

# ===== PCPT Header parsing helpers for model/sources.json & model/runs.json (new) =====
def _coerce_header_value(v: str):
    s = (v or "").strip()
    # Try JSON first (arrays/objects/strings/numbers)
    try:
        return json.loads(s)
    except Exception:
        pass
    # Strip surrounding quotes if present
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s

# Text normalizer for tolerant comparisons
def _normalize_text(s: str) -> str:
    """Normalize text for tolerant comparisons: normalize newlines, strip trailing spaces, collapse multiple blank lines."""
    if s is None:
        return ""
    # Normalize newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing spaces on each line
    s = "\n".join([ln.rstrip() for ln in s.split("\n")])
    # Collapse 3+ blank lines to 2
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

# Helper: Parse DMN Inputs/Outputs line with optional Allowed Values
def _parse_dmn_io_decl(line: str) -> dict:
    """
    Parse a single DMN Inputs/Outputs line of the form:
      - Name : Type [Allowed Values: v1, v2, "v 3"]
    Returns a dict with keys: name, type, and optional allowedValues (list).
    Backwards compatible with lines that omit type or allowed values.
    """
    s = (line or "").strip()
    # Strip bullet markers if present
    s = s.lstrip("-*").strip()
    # Try full pattern with Allowed Values
    m = re.match(
        r'^(?P<name>[^:]+?)\s*:\s*(?P<type>[^\[\]]+?)(?:\s*\[\s*Allowed\s*Values\s*:\s*(?P<values>.+?)\s*\])?\s*$',
        s,
        flags=re.IGNORECASE
    )
    if m:
        name = m.group("name").strip().strip('`')
        typ = (m.group("type") or "").strip().strip('`')
        out = {"name": name, "type": typ}
        vals = m.group("values")
        if vals:
            # Split by comma, strip whitespace and surrounding quotes
            parts = []
            for tok in vals.split(","):
                t = tok.strip()
                # remove surrounding single/double quotes if present
                if (len(t) >= 2) and ((t[0] == t[-1]) and t[0] in {'"', "'"}):
                    t = t[1:-1]
                if t:
                    parts.append(t)
            if parts:
                out["allowedValues"] = parts
        return out
    # Fallbacks for legacy forms:
    # 1) "Name : Type"
    if ":" in s:
        name, typ = s.split(":", 1)
        return {"name": name.strip().strip('`'), "type": typ.strip().strip('`')}
    # 2) Only a field name
    field = s.strip().strip('`')
    return {"name": field, "type": ""}


# Helper: detect material change in DMN-relevant fields (hit policy, inputs, outputs, table)
def _has_material_change(existing: dict, new_fields: dict) -> bool:
    """
    Determine whether DMN-relevant fields have materially changed.
    Compares hit policy, inputs, outputs (including allowedValues), and table text.
    Returns True if different; False if effectively the same or if existing is missing.
    """
    if not isinstance(existing, dict):
        return True
    def _norm_io(lst):
        out = []
        for x in (lst or []):
            if not isinstance(x, dict):
                continue
            out.append({
                "name": str(x.get("name", "")),
                "type": str(x.get("type", "")),
                # Normalize allowedValues to a list of strings
                "allowedValues": [str(v) for v in (x.get("allowedValues") or [])]
            })
        # Keep deterministic order
        return out
    # Build comparable snapshots
    ex_hp   = str(existing.get("dmn_hit_policy", "") or "")
    ex_in   = _norm_io(existing.get("dmn_inputs"))
    ex_out  = _norm_io(existing.get("dmn_outputs"))
    ex_tbl  = str(existing.get("dmn_table", "") or "").strip()

    new_hp  = str(new_fields.get("dmn_hit_policy", "") or "")
    new_in  = _norm_io(new_fields.get("dmn_inputs"))
    new_out = _norm_io(new_fields.get("dmn_outputs"))
    new_tbl = str(new_fields.get("dmn_table", "") or "").strip()

    if ex_hp != new_hp:
        return True
    if ex_in != new_in:
        return True
    if ex_out != new_out:
        return True
    if ex_tbl != new_tbl:
        return True
    return False

def _parse_pcpt_header_block(lines):
    data = {}
    for line in lines:
        m = KV_LINE.match(line.rstrip("\n"))
        if not m:
            continue
        k, v = m.group("k"), m.group("v")
        data[k] = _coerce_header_value(v)
    return data

def _iter_pcpt_runs(text: str):
    """Yield dicts that merge header key/values and include `response_text` captured between
    'RESPONSE BEGIN' and 'RESPONSE END' that follow the header block in the same file."""
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        if HEADER_BEGIN_RX.search(lines[i]):
            i += 1
            header_block = []
            while i < n and not HEADER_END_RX.search(lines[i]):
                if lines[i].strip():
                    header_block.append(lines[i])
                i += 1
            # Skip HEADER END
            if i < n and HEADER_END_RX.search(lines[i]):
                i += 1
            header = _parse_pcpt_header_block(header_block)
            # Now attempt to find the next RESPONSE block after this header
            response_text = ""
            # Scan forward to next RESPONSE BEGIN
            while i < n and not RESP_BEGIN_RX.search(lines[i]):
                # Stop if we encounter another HEADER BEGIN before a response (some logs may omit)
                if HEADER_BEGIN_RX.search(lines[i]):
                    break
                i += 1
            if i < n and RESP_BEGIN_RX.search(lines[i]):
                i += 1
                resp_lines = []
                while i < n and not RESP_END_RX.search(lines[i]):
                    resp_lines.append(lines[i])
                    i += 1
                # Skip RESP END
                if i < n and RESP_END_RX.search(lines[i]):
                    i += 1
                response_text = "\n".join(resp_lines)
            # Yield a combined record
            rec = dict(header)
            rec["response_text"] = response_text
            yield rec
            continue
        i += 1

def _build_sources_and_runs_from_logs(log_paths):
    sources = {}  # root_dir -> set(source_paths)
    runs = []
    skipped_no_build = 0
    skipped_old_build = 0
    skipped_no_prompt = 0
    skipped_prefix = 0
    for log_path in log_paths:
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        for rec in _iter_pcpt_runs(text):
            # Skip headers from older builds, and skip if build is unknown
            build_raw = rec.get("build")
            build_num = None
            if build_raw is not None:
                try:
                    build_num = int(str(build_raw).strip())
                except Exception:
                    build_num = None
            if build_num is None:
                skipped_no_build += 1
                continue
            if build_num < MIN_BUILD_NUM:
                skipped_old_build += 1
                continue
            root_dir = rec.get("root_dir")
            source_path = rec.get("source_path")
            output_path = rec.get("output_path")
            input_files = rec.get("input_files") or []
            output_file = rec.get("output_file")
            prompt = rec.get("prompt") or rec.get("prompt_template")
            # Filter: only keep runs whose prompt starts with the required prefix
            _p_txt = (str(prompt).strip() if prompt is not None else "")
            if not _p_txt:
                skipped_no_prompt += 1
                continue
            if not _p_txt.startswith(PROMPT_PREFIX_FILTER):
                skipped_prefix += 1
                continue
            if root_dir and source_path:
                sources.setdefault(root_dir, set()).add(str(source_path))
            _audit_add("run", path=str(rec.get("output_file") or ""), source=str(log_path), decision="accepted", reason="header ok", tests={"build": build_num, "prompt_startswith": True}, derived={"timestamp": rec.get("timestamp")})
            runs.append({
                "timestamp": rec.get("timestamp"),
                "build": rec.get("build"),
                "mode": rec.get("mode"),
                "provider": rec.get("provider"),
                "model": rec.get("model"),
                "prompt": prompt,
                "source_path": source_path,
                "output_path": output_path,
                "input_files": input_files,
                "output_file": output_file,
                "root_dir": root_dir,
                "log_file": str(log_path),
                "response_text": rec.get("response_text") or "",
                "total": rec.get("total"),
                "index": rec.get("index"),
            })
    # Order runs newest-first (by header timestamp; fallback to log file mtime)
    def _parse_ts(ts: str):
        try:
            t = str(ts or "").strip()
            if not t:
                return None
            if t.endswith('Z'):
                t = t[:-1]
            return datetime.fromisoformat(t)
        except Exception:
            return None

    def _log_mtime(p: str):
        try:
            return os.path.getmtime(p)
        except Exception:
            return 0.0

    runs.sort(key=lambda r: (
        _parse_ts(r.get("timestamp")) or datetime.min,
        _log_mtime(r.get("log_file") or "")
    ), reverse=True)

    sources_out = [
        {"root_dir": rd, "source_paths": sorted(list(paths))}
        for rd, paths in sorted(sources.items(), key=lambda x: x[0])
    ]
    LOG.info("[info] Looking for runs with: build ≥ %s and prompt startswith '%s'", MIN_BUILD_NUM, PROMPT_PREFIX_FILTER)
    LOG.info("[info] Accepted runs: %d (skipped: no/invalid build=%d, old build=%d, no prompt=%d, prefix mismatch=%d)",
             len(runs), skipped_no_build, skipped_old_build, skipped_no_prompt, skipped_prefix)
    return sources_out, runs

def _discover_pcpt_header_files() -> list:
    """Return files that likely contain PCPT headers.
    We search LOG_DIR, the current working directory, and the directory of the input file
    (plus its parent) for common text extensions. We quickly pre-check content for the markers
    to avoid scanning big binaries.
    """
    exts = (".log", ".txt", ".out", ".md")
    candidates = set()

    # 1) LOG_DIR
    for pat in ("**/*.log", "**/*.txt", "**/*.out", "**/*.md"):
        candidates.update(glob.glob(os.path.join(LOG_DIR, pat), recursive=True))

    # 2) CWD
    def _walk_add(base: str):
        if not base or not os.path.isdir(base):
            return
        for root, _, files in os.walk(base):
            for name in files:
                if name.endswith(exts):
                    candidates.add(os.path.join(root, name))

    _walk_add(os.getcwd())

    # 3) input file dir and its parent (to catch reports under repo root)
    try:
        in_dir = os.path.dirname(os.path.abspath(input_file))
        _walk_add(in_dir)
        _walk_add(os.path.dirname(in_dir))
    except Exception:
        pass

    files = []
    for p in candidates:
        # Quick pre-check: look for markers near the top
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                head = f.read(8192)
            if PCPT_PREFIX in head and "HEADER" in head:
                files.append((p, os.path.getmtime(p)))
        except Exception:
            continue
    # Sort by file mtime (newest first)
    files.sort(key=lambda t: t[1], reverse=True)
    LOG.info("[info] PCPT header candidate files discovered: %d", len(files))
    return [path for path, _ in files]



def _extract_rule_names(doc_text: str) -> List[str]:
    t = _normalize_text(doc_text or "")
    names: List[str] = []
    # Split on headings and collect level-2+ headings as rule names
    parts = re.split(r"(?m)^\s{0,3}#{1,6}\s+", t.strip())
    for sec in parts[1:]:
        first = (sec.splitlines() or [""])[0]
        nm = _heading_text(first)
        if nm and nm.lower() not in {"rule name", "rule-name", "name"}:
            names.append(nm)
    # de-dupe while preserving order
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

def write_model_sources_and_runs(rule_ids_for_output: Optional[List[str]] = None,
                                 output_file_path: Optional[str] = None,
                                 link_all: bool = False):
    """Scan logs/reports for PCPT headers and emit .model/sources.json and .model/runs.json.
    If `rule_ids_for_output` and `output_file_path` are provided, attach the list of rule IDs
    based on content matching between the current output document and a run's RESPONSE content:
      • When `link_all` is False (default): attach to the single most recent matching run.
      • When `link_all` is True: attach to **all** matching runs.
    In single-file mode, supports partial-content fallback using rule names.
    """
    LOG.info("[info] Building sources/runs from logs…")
    logs = _discover_pcpt_header_files()
    if os.environ.get("PCPT_DEBUG"):
        LOG.info("[info] PCPT header candidate files: %d", len(logs))
        for p in logs[:15]:
            LOG.trace("[trace] candidate: %s", p)
    sources_out, runs = _build_sources_and_runs_from_logs(logs)
    LOG.info("[info] Runs discovered : %d", len(runs))
    LOG.info("[info] Roots discovered: %d", len(sources_out))
    # Prepare normalized text of the current output document (if provided)
    doc_norm = ""
    rule_names_from_doc: List[str] = []
    if output_file_path and os.path.exists(output_file_path):
        try:
            with open(output_file_path, "r", encoding="utf-8", errors="ignore") as df:
                raw = df.read()
            doc_norm = _normalize_text(raw)
            rule_names_from_doc = _extract_rule_names(raw)
        except Exception:
            doc_norm = ""
            rule_names_from_doc = []
    # Preserve previously stored rule_ids from existing runs.json before we add new links
    try:
        _existing_runs = []
        if os.path.exists(RUNS_JSON) and os.path.getsize(RUNS_JSON) > 0:
            with open(RUNS_JSON, "r", encoding="utf-8") as rf:
                _existing_runs = json.load(rf) or []
    except Exception:
        _existing_runs = []

    def _run_key(rec: dict) -> Tuple[str, str]:
        # Use (timestamp, log_file) as a stable identity; both are emitted by header parsing
        return (str(rec.get("timestamp") or ""), str(rec.get("log_file") or ""))

    _existing_map: Dict[Tuple[str, str], dict] = { _run_key(r): r for r in _existing_runs if isinstance(r, dict) }

    # Helper to preserve previous rule_ids after linking, but only if still empty
    def _preserve_previous_links_when_empty():
        for rec in runs:
            k = _run_key(rec)
            prev = _existing_map.get(k)
            if prev and isinstance(prev, dict):
                prev_ids = prev.get("rule_ids")
                if prev_ids and not rec.get("rule_ids"):
                    try:
                        rec["rule_ids"] = list(prev_ids)
                    except Exception:
                        rec["rule_ids"] = prev_ids
                    LOG.info("[info] preserved prior rule_ids for run ts=%s log=%s (count=%d)", rec.get("timestamp"), rec.get("log_file"), len(prev_ids))
                    _audit_add("run", path=str(rec.get("log_file")), source="runs.json(prev)", decision="accepted", reason="existing links preserved (empty current)", tests={"preserved": True}, derived={"count": len(prev_ids)})

    # Optionally attach rule IDs for the current run based on the produced report path
    if rule_ids_for_output and output_file_path:
        try:
            ids_list = list(rule_ids_for_output)
        except Exception:
            ids_list = None
        if ids_list:
            matched_idxs = []
            # 1) Prefer content-based match: response contains full document text
            if doc_norm:
                for idx, rec in enumerate(runs):
                    resp = _normalize_text(rec.get("response_text") or "")
                    if resp and doc_norm and doc_norm in resp:
                        matched_idxs.append(idx)
                        if not link_all:
                            break  # stop on first match when linking only the newest/first
            # --- partial match fallback for single-file mode ---
            if not matched_idxs and rule_names_from_doc:
                # Count rule-name matches per run, but only for single-file mode runs
                scored = []  # (idx, score, total, index)
                for idx, rec in enumerate(runs):
                    mode_val = str(rec.get("mode") or "").strip().lower()
                    if mode_val != "single":
                        continue
                    resp = _normalize_text(rec.get("response_text") or "")
                    if not resp:
                        continue
                    score = 0
                    for nm in rule_names_from_doc:
                        if (f"## {nm}" in resp) or (nm in resp):
                            score += 1
                    if score > 0:
                        scored.append((idx, score, str(rec.get("total")), str(rec.get("index"))))

                if scored:
                    # Group by total to ensure all parts belong to the same series
                    series: Dict[str, list] = {}
                    for tup in scored:
                        idx, s, tot, ind = tup
                        key = (tot or "")
                        series.setdefault(key, []).append(tup)

                    # Choose the best series: most unique indices; tie-break by newest timestamp
                    def _series_key(items: list):
                        uniq_idx = {x[3] for x in items}  # index values as strings
                        newest = max((_parse_ts_safe(runs[x[0]].get("timestamp")) or datetime.min) for x in items)
                        # Prefer series with a real total (non-empty) as an additional tiebreaker
                        has_total = 1 if (items and (items[0][2] or "")) else 0
                        return (len(uniq_idx), has_total, newest)

                    best_total = None
                    best_items = []
                    for tot, items in series.items():
                        if not items:
                            continue
                        if best_total is None:
                            best_total, best_items = tot, items
                        else:
                            cur_key = _series_key(items)
                            best_key = _series_key(best_items)
                            if cur_key > best_key:
                                best_total, best_items = tot, items

                    # Deduplicate by index within the chosen series (keep newest per index)
                    by_index: Dict[str, Tuple[int, int, str, str]] = {}
                    for tup in best_items:
                        idx, s, tot, ind = tup
                        ts = _parse_ts_safe(runs[idx].get("timestamp")) or datetime.min
                        prev = by_index.get(ind)
                        if (prev is None) or (ts > (_parse_ts_safe(runs[prev[0]].get("timestamp")) or datetime.min)):
                            by_index[ind] = tup

                    chosen = list(by_index.values())
                    # Order chosen by timestamp desc to make selection predictable
                    chosen.sort(key=lambda x: _parse_ts_safe(runs[x[0]].get("timestamp")) or datetime.min, reverse=True)

                    LOG.info(
                        "[match] Partial-match (single-file) • total=%s • parts=%d (unique indices) • linking=partial per-rule",
                        best_total, len(chosen)
                    )

                    # --- Begin: new per-rule partial-match logic ---
                    # 1. Load rule_id → rule_name mapping for ids_list from business_rules.json
                    id_to_name = {}
                    try:
                        rules_path = f"{MODEL_HOME}/.model/business_rules.json"
                        with open(rules_path, "r", encoding="utf-8") as rf:
                            rule_objs = json.load(rf)
                        for rule in rule_objs:
                            rid = rule.get("id")
                            nm = rule.get("rule_name")
                            if rid in ids_list and nm:
                                id_to_name[rid] = nm
                    except Exception as e:
                        LOG.info("[warn] Could not load business_rules.json for per-rule linking: %s", e)
                        # fallback: skip per-rule linking if mapping fails
                        id_to_name = {rid: rid for rid in ids_list}

                    # 2. For each run in chosen, collect the subset of ids_list whose rule_name is found in response_text
                    found_ids_per_run = []
                    found_rule_ids = set()
                    for tup in chosen:
                        idx = tup[0]
                        rec = runs[idx]
                        resp = _normalize_text(rec.get("response_text") or "")
                        matched_rule_ids = []
                        for rid in ids_list:
                            rule_name = id_to_name.get(rid)
                            if not rule_name:
                                continue
                            if (f"## {rule_name}" in resp) or (rule_name in resp):
                                matched_rule_ids.append(rid)
                        if matched_rule_ids:
                            found_rule_ids.update(matched_rule_ids)
                            # Attach only these rule IDs to this run, de-duped
                            existing = rec.get("rule_ids") or []
                            combined = list(existing) + [rid for rid in matched_rule_ids if rid not in set(existing)]
                            runs[idx]["rule_ids"] = combined
                            LOG.info("[info] link rules → run: idx=%d ts=%s log=%s ids=%d (partial per-rule match)", idx, rec.get("timestamp"), rec.get("log_file"), len(matched_rule_ids))
                            _audit_add("link", path=str(output_file_path), source=str(rec.get("log_file")), decision="accepted", reason="partial per-rule match", tests={"matched": True}, derived={"rule_ids": matched_rule_ids})
                        found_ids_per_run.append((idx, matched_rule_ids))
                    # Save and return (skip matched_idxs logic below)
                    _preserve_previous_links_when_empty()
                    os.makedirs(os.path.dirname(SOURCE_JSON), exist_ok=True)
                    _save_json_file(SOURCE_JSON, sources_out)
                    _save_json_file(RUNS_JSON, runs)
                    return
                    # --- End: new per-rule partial-match logic ---
            if matched_idxs:
                def _parse_ts(ts: str):
                    try:
                        t = str(ts or "").strip()
                        if not t:
                            return None
                        if t.endswith('Z'):
                            t = t[:-1]
                        return datetime.fromisoformat(t)
                    except Exception:
                        return None

                target_indices = matched_idxs if link_all else None
                if not link_all:
                    # Pick the most recent by timestamp
                    newest_idx = matched_idxs[0]
                    newest_dt = _parse_ts(runs[newest_idx].get("timestamp"))
                    for idx in matched_idxs[1:]:
                        dt = _parse_ts(runs[idx].get("timestamp"))
                        if newest_dt is None and dt is not None:
                            newest_idx, newest_dt = idx, dt
                        elif dt is not None and newest_dt is not None and dt > newest_dt:
                            newest_idx, newest_dt = idx, dt
                        elif dt is None and newest_dt is None and idx > newest_idx:
                            newest_idx = idx
                    target_indices = [newest_idx]

                # Determine if we are in partial mode (no doc_norm in any selected response)
                partial_mode = False
                if doc_norm:
                    partial_mode = not any((_normalize_text(runs[i].get("response_text") or "") and (doc_norm in _normalize_text(runs[i].get("response_text") or ""))) for i in (matched_idxs or []))
                # Attach rule IDs to each selected run (either newest or all)
                for idx in target_indices:
                    existing = runs[idx].get("rule_ids") or []
                    try:
                        combined = list(existing) + [rid for rid in ids_list if rid not in set(existing)]
                    except Exception:
                        combined = ids_list
                    LOG.info("[info] link rules → run: idx=%d ts=%s log=%s ids=%d", idx, runs[idx].get("timestamp"), runs[idx].get("log_file"), len(ids_list))
                    _audit_add("link", path=str(output_file_path), source=str(runs[idx].get("log_file")), decision="accepted", reason=("partial content match" if partial_mode else "content match"), tests={"matched": True}, derived={"rule_ids": ids_list})
                    runs[idx]["rule_ids"] = combined
    _preserve_previous_links_when_empty()
    os.makedirs(os.path.dirname(SOURCE_JSON), exist_ok=True)
    _save_json_file(SOURCE_JSON, sources_out)
    _save_json_file(RUNS_JSON, runs)

def _all_logs() -> list:
    patterns = ["**/*.log", "**/*.txt", "**/*.out"]
    results = []
    for pat in patterns:
        results.extend(glob.glob(os.path.join(LOG_DIR, pat), recursive=True))
    return sorted(results)


def _heading_text(line: str) -> str:
    """Extract clean heading text from a markdown heading line.
    Removes leading/trailing '#' and surrounding whitespace/markers.
    """
    s = (line or "").strip()
    # remove leading hashes and spaces
    s = re.sub(r"^\s*#{1,6}\s*", "", s)
    # remove trailing hashes and spaces
    s = re.sub(r"\s*#{1,6}\s*$", "", s)
    s = s.strip(" *-\t")
    # If the heading still includes a leading label like "Rule Name:", strip it.
    s = re.sub(r"^(?:Rule\s+Name|Name):\s*", "", s, flags=re.IGNORECASE)
    return s


output_file = f"{MODEL_HOME}/.model/business_rules.json"

# Read existing rules if output file already exists and is not empty
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    with open(output_file, "r", encoding="utf-8") as f:
        try:
            existing_rules = json.load(f)
        except json.JSONDecodeError:
            print(f"Warning: {output_file} is not valid JSON. Starting fresh.")
            existing_rules = []
else:
    existing_rules = []

existing_by_name = {
    r["rule_name"]: r
    for r in existing_rules
    if "rule_name" in r and "timestamp" in r
}

# Build set of seen (rule_name, timestamp) pairs from existing rules (normalized)
seen = set()
for rule in existing_rules:
    k = _dedupe_key(rule.get("rule_name"), rule.get("timestamp"))
    if k:
        seen.add(k)


# Defaults for team/owner and component must be empty strings (per requirement)
default_owner = ""
default_component = ""

# Prompt for values (requirement: prompt for team and component)
# Note: We map 'team' input to the existing 'owner' attribute to remain schema-compatible.
_ingest_owner = _prompt_with_default("Enter team/owner to set on all rules", default_owner)
_ingest_component = _prompt_with_default("Enter component to set on all rules", default_component)
# Update registries: add team/component if not already present (skip empty)
_append_unique_value(TEAMS_JSON, _ingest_owner)
_append_unique_value(COMPONENTS_JSON, _ingest_component)

updated_count = 0
new_count = 0
considered_count = 0
new_rules = []

def _normalize_rule_doc_text(t: str) -> str:
    # Normalize various "Rule Name" heading formats to a consistent "## " heading
    t = re.sub(r"#{2,6}\s*\d+\.\s*\*\*Rule Name:\*\*\s*", "## ", t)   # e.g., "### 1. **Rule Name:**"
    t = re.sub(r"#{2,6}\s*\*\*Rule Name:\*\*\s*", "## ", t)            # e.g., "### **Rule Name:**"
    t = re.sub(r"#{2,6}\s*\d+\.\s*Rule Name:\s*", "## ", t)            # e.g., "### 1. Rule Name:"
    t = re.sub(r"#{2,6}\s*Rule Name:\s*", "## ", t)                    # e.g., "### Rule Name:"

    # Newer "Name:" heading variants → also normalize to "## "
    t = re.sub(r"#{2,6}\s*\d+\.\s*\*\*Name:\*\*\s*", "## ", t)         # e.g., "### 1. **Name:**"
    t = re.sub(r"#{2,6}\s*\*\*Name:\*\*\s*", "## ", t)                 # e.g., "### **Name:**"
    t = re.sub(r"#{2,6}\s*\d+\.\s*Name:\s*", "## ", t)                 # e.g., "### 1. Name:"
    t = re.sub(r"#{2,6}\s*Name:\s*", "## ", t)                         # e.g., "### Name:"

    t = re.sub(r"\n---+\n", "\n", t)                                   # Remove separators

    # Also handle non-heading inline forms
    t = re.sub(r"(?m)^\s*\*\*Rule Name:\*\*\s*", "## ", t)
    t = re.sub(r"(?m)^\s*Rule Name:\s*", "## ", t)
    t = re.sub(r"(?m)^\s*\*\*Name:\*\*\s*", "## ", t)
    t = re.sub(r"(?m)^\s*Name:\s*", "## ", t)

    return t

def _add_rules_from_text(doc_text: str, section_timestamp: str, allowed_names: Optional[set] = None):
    global updated_count, new_count, considered_count
    t = _normalize_rule_doc_text(doc_text or "")
    rule_sections = re.split(r"(?m)^\s{0,3}#{1,6}\s+", t.strip())[1:]
    for section in rule_sections:
        considered_count += 1
        try:
            lines = section.strip().splitlines()
            rule_name = _heading_text(lines[0])
            if not rule_name or rule_name.lower() in {"rule name", "rule-name"}:
                rn_match = re.search(r"\*\*Rule Name:\*\*\s*(.+)", section)
                if rn_match:
                    rule_name = rn_match.group(1).strip()
            
            if (not rule_name or rule_name.lower() in {"rule name", "rule-name"}):
                rn2 = re.search(r"\*\*Name:\*\*\s*(.+)", section)
                if rn2:
                    rule_name = rn2.group(1).strip()

            # New optional Kind field (Decision | BKM)
            kind = ""
            m_kind = re.search(r"\*\*Kind:\*\*\s*([A-Za-z]+)", section)
            if m_kind:
                kind = m_kind.group(1).strip()
            
            # If an allowlist is provided (used for prior-runs mode), skip names not in the set
            if allowed_names is not None and rule_name not in allowed_names:
                continue
            # Purpose: accept "**Rule Purpose:**" or "**Purpose:**"
            purpose_match = re.search(
                r"\*\*(?:Rule\s+)?Purpose:\*\*\s*\n?(.*?)(?=\n\*\*(?:Rule\s+)?Spec|\n\*\*Specification|\n\*\*Code Block|\n\*\*Example|$)",
                section, re.DOTALL | re.IGNORECASE
            )
            rule_purpose = purpose_match.group(1).strip() if purpose_match else ""

            # Spec: accept "**Rule Spec:**" or "**Spec:**" or "**Specification:**"
            spec_marker = re.search(r"\*\*(?:Rule\s+)?Spec:\*\*|\*\*Specification:\*\*", section, re.IGNORECASE)
            if spec_marker:
                start = spec_marker.end()
                next_marker = re.search(r"\n\*\*(Code Block|Example):\*\*|\n(?:\*{0,2}\s*)?DMN\s*:\s*(?:\*{0,2})?", section[start:], re.DOTALL | re.IGNORECASE)
                end = next_marker.start() + start if next_marker else len(section)
                rule_spec = section[start:end].strip()
            else:
                rule_spec = ""
            code_match = re.search(r"```[a-zA-Z]*\n(.*?)```", section, re.DOTALL)
            code_block = code_match.group(1).strip() if code_match else ""
            example_match = re.search(r"\*\*Example:\*\*\s*\n?(.*?)(?=\n(?:\*{0,2}\s*)?DMN\s*:\s*(?:\*{0,2})?|\n## |\Z)", section, re.DOTALL | re.IGNORECASE)
            example = example_match.group(1).strip() if example_match else ""
            if example:
                example = re.split(r"\n(?:\*{0,2}\s*)?DMN\s*:\s*(?:\*{0,2})?", example, flags=re.IGNORECASE)[0].strip()
            dmn_hit_policy = ""
            dmn_inputs, dmn_outputs = [], []
            dmn_table = ""
            dmn_match = re.search(r"(?:^|\n)(?:\*{0,2}\s*)?DMN\s*:\s*\n?(.*?)(?=\n## |\Z)", section, re.DOTALL | re.IGNORECASE)
            if dmn_match:
                raw_dmn = dmn_match.group(1).strip()
                m_code = re.search(r"```.*?\n(.*?)```", raw_dmn, re.DOTALL)
                dmn_body = m_code.group(1).strip() if m_code else raw_dmn
                dmn_body = re.sub(r"`+", "", dmn_body)
                dmn_body = re.sub(r"\*\*", "", dmn_body)
                m_hp = re.search(r"Hit\s*Policy\s*:\s*([A-Za-z_]+)", dmn_body, re.IGNORECASE)
                if m_hp:
                    dmn_hit_policy = m_hp.group(1).strip()
                m_inputs = re.search(r"Inputs\s*:\s*\n(?P<block>(?:\s*[-*]\s*.*(?:\n|$))+)", dmn_body, re.IGNORECASE)
                if m_inputs:
                    for ln in m_inputs.group("block").splitlines():
                        ln = ln.strip()
                        if not (ln.startswith("-") or ln.startswith("*")):
                            continue
                        parsed = _parse_dmn_io_decl(ln)
                        # Only include allowedValues if present (backwards compatible)
                        dmn_inputs.append(parsed)
                m_outputs = re.search(r"Outputs\s*:\s*\n(?P<block>(?:\s*[-*]\s*.*(?:\n|$))+)", dmn_body, re.IGNORECASE)
                if m_outputs:
                    for ln in m_outputs.group("block").splitlines():
                        ln = ln.strip()
                        if not (ln.startswith("-") or ln.startswith("*")):
                            continue
                        parsed = _parse_dmn_io_decl(ln)
                        dmn_outputs.append(parsed)
                lines2 = [ln.rstrip() for ln in dmn_body.splitlines()]
                table_lines, in_table = [], False
                for ln in lines2:
                    if ("|" in ln) or ("+" in ln) or re.search(r"-{2,}", ln):
                        table_lines.append(ln.rstrip())
                        in_table = True
                    else:
                        if in_table:
                            break
                dmn_table = "\n".join(table_lines).strip()
            # Dedup key uses the per-section timestamp now
            k = _dedupe_key(rule_name, section_timestamp)
            if k in seen and not FORCE_LOAD:
                _audit_add("rule", path="section", source="input_doc", decision="rejected", reason="duplicate or older", tests={"key": str(k), "force": FORCE_LOAD}, derived={})
                continue
            existing = existing_by_name.get(rule_name)
            if existing:
                old_ts = existing.get("timestamp")
                # Build new snapshot of DMN-related fields for material change detection
                _new_fields = {
                    "dmn_hit_policy": dmn_hit_policy,
                    "dmn_inputs": dmn_inputs,
                    "dmn_outputs": dmn_outputs,
                    "dmn_table": dmn_table,
                }
                is_material_change = _has_material_change(existing, _new_fields)
                if not FORCE_LOAD and old_ts and old_ts >= section_timestamp and not is_material_change:
                    _audit_add("rule", path="section", source="input_doc", decision="rejected", reason="duplicate or older", tests={"key": str(k), "force": FORCE_LOAD, "material_change": False}, derived={})
                    continue
                if not FORCE_LOAD and old_ts and old_ts >= section_timestamp and is_material_change:
                    LOG.info("[info] material change detected for '%s' — updating despite same/older timestamp (allowedValues or DMN changed)", rule_name)
                rule_id = existing.get("id") or str(uuid.uuid4())
                updated_count += 1
            else:
                rule_id = str(uuid.uuid4())
                new_count += 1
            seen.add(k)
            # code_file
            code_file = ""
            code_lines = None
            m_codefile_inline = re.search(r"\*\*Code\s*Block:\*\*\s*`?([^`\n]+)`?", section, re.IGNORECASE)
            if m_codefile_inline:
                code_file = m_codefile_inline.group(1).strip()
            else:
                m_codefile_fileline = re.search(r"(?mi)^\s*File:\s*`?([^`\n]+)`?", section)
                if m_codefile_fileline:
                    code_file = m_codefile_fileline.group(1).strip()
            if code_file:
                code_file = code_file.replace("`", "").strip()
            m_codelines = re.search(r"\bLine(?:s)?\s*:??\s*(\d+)(?:\s*[\-\u2013\u2014]\s*(\d+))?", section, re.IGNORECASE)
            if m_codelines:
                try:
                    start_line = int(m_codelines.group(1))
                    end_line = m_codelines.group(2)
                    if end_line is not None:
                        end_line = int(end_line)
                    else:
                        end_line = start_line
                    code_lines = [start_line, end_line]
                except Exception:
                    code_lines = None
            code_function = ""
            m_codefunc = re.search(r"(?mi)^\s*Function\b\s*[:\-\u2013\u2014]*\s*`?([^`\n]+)`?", section)
            if m_codefunc:
                code_function = m_codefunc.group(1).strip()
                code_function = re.sub(r"^[\s:;\-\u2013\u2014]+", "", code_function).strip()
            # New: parse **Links:** block (optional)
            links = []
            m_links = re.search(r"(?mi)^\s*\*\*Links:\*\*\s*\n(?P<body>.*?)(?=\n## |\Z)", section, re.DOTALL)
            if m_links:
                body = m_links.group("body").strip()
                if body.lower() == "none":
                    links = []
                else:
                    for ln in [x.strip() for x in body.splitlines() if x.strip()]:
                        # Accept two forms (case-insensitive):
                        #  A) Implicit to_step (preferred):
                        #     <from_step>.<from_output> -> <to_input> [kind=...]
                        #     (to_step is inferred as the current rule_name)
                        #  B) Legacy explicit to_step:
                        #     <from_step>.<from_output> -> <to_step>.<to_input> [kind=...]
                        # We do NOT store to_step; it is implied by the owning rule. The entire RHS
                        # is treated as to_input. If RHS starts with "<rule_name>." we strip that prefix.
                        mm = re.match(r"^(.+?)\.(.+?)\s*->\s*(.+?)\s*\[kind=([^\]]+)\]\s*$", ln, flags=re.IGNORECASE)
                        if mm:
                            _from_step = mm.group(1).strip()
                            _from_output = mm.group(2).strip()
                            rhs = mm.group(3).strip()
                            _kind = mm.group(4).strip()
                            links.append({
                                "from_step": _from_step,
                                "from_output": _from_output,
                                "to_input": rhs,               # full RHS preserved (may contain dots)
                                "kind": _kind
                            })
                        else:
                            # keep raw line for diagnostics if it doesn't match
                            links.append({"raw": ln})
            rule_rec = {
                "rule_name": rule_name,
                "rule_purpose": rule_purpose,
                "rule_spec": rule_spec,
                "code_block": code_block,
                "code_file": code_file,
                "code_lines": code_lines,
                "code_function": code_function,
                "example": example,
                "dmn_hit_policy": dmn_hit_policy,
                "dmn_inputs": dmn_inputs,
                "dmn_outputs": dmn_outputs,
                "dmn_table": dmn_table,
                "timestamp": section_timestamp,
                "id": rule_id,
                "owner": _ingest_owner,
                "component": _ingest_component,
                "kind": kind,
                "links": links,
            }
            if existing and existing.get("archived") is not None:
                # If rule was archived and we are updating it, unarchive automatically
                if existing.get("archived") is True:
                    rule_rec["archived"] = False
                    LOG.info("[info] Unarchived rule on update: %s", rule_name)
                else:
                    rule_rec["archived"] = existing["archived"]
            if FORCE_LOAD and existing:
                for src_key in ("rule_category", "category", "category_id", "categoryId"):
                    val = existing.get(src_key)
                    if val not in (None, "", []):
                        rule_rec["rule_category"] = val
                        break
            _audit_add(
                "rule",
                path=rule_rec.get("code_file") or "",
                source="input_doc",
                decision="accepted",
                reason=("updated" if existing else "new"),
                tests={"has_code_block": bool(code_block)},
                derived={"rule_id": rule_id, "rule_name": rule_name, "kind": kind, "links": len(links)}
            )
            LOG.info("[info] ✓ %s rule: %s", ("updated" if existing else "new"), rule_name)
            new_rules.append(rule_rec)
        except Exception as e:
            print(f"Failed to parse a rule section:\n{section[:100]}...\nError: {e}")

 # First ingest the current file content (collect current rule names)
LOG.info("[step 1/4] Parsing current report to extract rules…")
with open(input_file, "r", encoding="utf-8") as f:
    _add_rules_from_text(f.read(), file_timestamp)

# Then, if requested, ingest prior iterations **only for rules that already exist**
if ALL_RUNS:
    LOG.info("[step 2/4] Searching logs for prior iterations (same output file, same root/src/out)…")
    prior_items = _gather_prior_responses_for_same_output(input_file)
    if prior_items:
        print(f"[info] All-runs: found {len(prior_items)} prior iteration(s) for {input_file}")
    # Allow only names that already exist (either previously in the model, or just parsed from the current file)
    _existing_names = set(existing_by_name.keys())
    _current_names  = {r.get("rule_name") for r in new_rules if r.get("rule_name")}
    _allowed_names  = _existing_names.union(_current_names)
    for ts, resp_text in prior_items:
        _add_rules_from_text(resp_text, ts or file_timestamp, allowed_names=_allowed_names)

final_rules = {r["rule_name"]: r for r in existing_rules}
for r in new_rules:
    final_rules[r["rule_name"]] = r  # overwrite with latest

LOG.info("[step 3/4] Persisting rules to model…")
# Persist rules first to ensure output file exists (helps first-run linkage)
final_rules_list = list(final_rules.values())
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as out_file:
    json.dump(final_rules_list, out_file, indent=2)

LOG.info("[info] Extracted %d rules: %d new, %d updated. Total now: %d. Saved to %s. Considered %d section(s).",
         len(new_rules), new_count, updated_count, len(final_rules_list), output_file, considered_count)
sidecar_path = _audit_write(prefix="rules_ingest")
if sidecar_path:
    LOG.info("[info] Wrote audit sidecar: %s", sidecar_path)

# Build auxiliary model indices from PCPT headers and link this run to its rule IDs
_current_run_rule_ids = [r.get("id") for r in new_rules if r.get("id")]
LOG.info("[step 4/4] Rebuilding sources/runs index and linking this run…")
write_model_sources_and_runs(
    rule_ids_for_output=_current_run_rule_ids,
    output_file_path=input_file,
    link_all=False
)