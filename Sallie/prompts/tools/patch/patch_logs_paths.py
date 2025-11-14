# Specification
# 1) prompt for json spec file (default to tools/spec/patch_logs_paths.json)
# 2) prompt for root dir (default to value in spec file and allow change)
# 3) prompt for logs dir (default to ~/.pcpt/log and allow change)
# 4) Preflight checks a) verify files and dirs exist b) verify all paths specified in spec file exist
# 5) create a zip file backup of contents of logs dir with timestamp and place it in the logs dir
# 6) create a zip file backup of contents of the root dir with timestamp and place it in the roots dir
# PHASE ONE) Updating reports in docs
# 7) For each path-pair entry in the json spec  file which is a spec path and an output path, do the following
# 8) - get a list of all path/files under root dir + spec path
# 9) - for each business_logic_report.md file, find all the code block entries (e.g. **Code Block:** code/sf/flow/Approve_New_Product.flow) and find the closest matching path/file from your list in step 8 (matching from right char to left char), and replace the path/file in the entry with source path + the matching piece
# 10) - write out to a .updated report file, and also create a json that summarises changes
# 11) declare success and give a summary of what done
# Here is a reference example spec file (note that the source and output paths are relative under the root dir)...
# {
#  "root-directory": "/Users/greghodgkinson/Documents/git.nosync/sm-dir-layout-greg",
#  "path-pairs": [
#    {
#      "source-path": "code/fdrit-0103",
#      "output-path": "docs/fdrit-0103"
#    },
#    {
#      "source-path": "code/sf/force-app/main/default",
#      "output-path": "docs/sf/force-app/main/default"
#    }
#  ]
#} 
# PHASE TWO) Updating logs
# 12) For each path-pair entry in the json spec  file which is a spec path and an output path, do the following
# 13) - For each .updated report in the output path, take the previous report it was based on, and look for the most recent log file in the logs directory that has the exact contents of the report in its response section (as shown in example below). For that log file we will do the following...
# 14) -- update root_dir to the new root dir
# 15) -- update source_path to the new source path
# 16) -- update output_path to the new output path
# 17) -- update the response section, replacing the previous report contents, with the contents of the new .updated report
# 18) -- write out a .updated log file instead of replacing the original
# 19) declare succes, and give a summary of what done
# Reference example of log file header is below...
# [PCPTLOG:]     ###################################################### PCPT LOG ######################################################
# [PCPTLOG:]     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< HEADER BEGIN >>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<
# [PCPTLOG:]     mode=multi
# [PCPTLOG:]     provider=langchain
# [PCPTLOG:]     model=global.anthropic.claude-sonnet-4-20250514-v1:0
# [PCPTLOG:]     build=2510291836
# [PCPTLOG:]     timestamp=2025-10-30_07-07-45
# [PCPTLOG:]     root_dir=/Users/greghodgkinson/Documents/git.nosync/sallie-mae-example
# [PCPTLOG:]     source_path=code/stored_proc
# [PCPTLOG:]     output_path=docs/stored_proc
# [PCPTLOG:]     input_files=["Process_Staging_To_ODS_Claims.txt"]
# [PCPTLOG:]     prompt_template=business_rules_report_with_domain.templ
# [PCPTLOG:]     output_file=business_logic_report/business_logic_report.md
# [PCPTLOG:]     <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< HEADER END >>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<
# [PCPTLOG:]     ======================================== PROMPT BEGIN ========================================
# INPUT 1: SOURCE CODE
# Reference example of log file response section (begin and end shown so you can see how demarcated)...
# | < 18                | -                   | Minor     | 0        |
# | >= 18 and < 65      | "Gold","Platinum"   | Adult+    | 0.1      |
# | >= 18 and < 65      | -                   | Adult     | 0        |
# | >= 65               | -                   | Senior    | 0.15     |
# 1. The code is stored procedure code.
# [PCPTLOG:]     ======================================== PROMPT END ==========================================
# [PCPTLOG:]     ======================================== RESPONSE BEGIN ========================================
# ### Rule Name: Claim Date Validation
# 
# **Rule Purpose:** Ensures that all claims have valid claim dates before processing into the operational data store.
# 
# **Rule Spec:** If a claim's ClaimDate_Raw cannot be converted to a valid DATE format, then the claim header record is marked as invalid and excluded from further processing.
# 
# **Code Block:** code/stored_proc/Process_Staging_To_ODS_Claims.txt
# Lines: 67-70
# ...
# **Example:** If a claim header has TotalClaimChargeAmount = 150.00 and the calculated line total is 150.00, no update occurs. If the calculated total is 175.00, the update is performed.
# 
# **DMN:**
# Hit Policy: FIRST
# 
# Inputs:
# - currentTotal:number
# - calculatedTotal:number
# 
# Outputs:
# - updateTotal:boolean
# 
# | currentTotal condition | calculatedTotal condition | totals are different | updateTotal |
# |------------------------|---------------------------|---------------------|-------------|
# | -                      | -                         | true                | true        |
# | -                      | -                         | false               | false       |
# [PCPTLOG:]     ======================================== RESPONSE END ==========================================
# [PCPTLOG:]     ======================================== AUDIT BEGIN ========================================
# ||| Tokens Used: 14,687 = Prompt Tokens: 8,990 + Generated Tokens: 5,697
# ||| Cumulative Tokens Used: 14,687, Prompt Tokens: 8,990, Generated Tokens: 5,697, Time Elapsed: 1.48 minutes, Tokens Per Minute (TPM): 9,954.14
# [PCPTLOG:]     ======================================== AUDIT END ==========================================
# PHASE TWO part b)
# 20) For each of the log update files created, get the following values from the old file it replaces: root dir, source path, output path, prompt template
# 21) - apart from the old file we updated, look for other log files that have the same root dir, source path, output path and prompt template. These are previous ingestions that need to also be updated.
# 22) -- For each of these matching files, create an update file copy with the new root dir, source path, output path and prompt template
# 23) -- Update the paths in the RESPONSE block in the Code Block lines by replaceing the old source path part of the path with the new source path
# PHASE 3)
# 24) Prompt to promote update log and report files to replace old ones, and then remove temporary update files
# 25) Declare success and print summary
# Implementation of this spec now goes here below....

import os
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Set
from math import log2

# ----------------------------
# Backup modes
# ----------------------------
BACKUP_FAST = "fast"      # ZIP_STORED (no compression) — fastest
BACKUP_NORMAL = "normal"  # ZIP_DEFLATED (compressed) — smaller, slower
BACKUP_SKIP = "skip"      # no backup


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    if n <= 0:
        return "0 B"
    idx = min(int(log2(n) / 10), len(units) - 1)
    return f"{n / (1024 ** idx):.1f} {units[idx]}"


def _dir_stats(path: Path, *, exclude_backup_zips: bool = False, exclude_dirs: Optional[Set[str]] = None) -> Tuple[int, int]:
    """Return (file_count, total_bytes) for a directory tree."""
    files = 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            if exclude_dirs:
                try:
                    rel_parts = p.relative_to(path).parts
                except Exception:
                    rel_parts = p.parts
                if any(part in exclude_dirs for part in rel_parts):
                    continue
            if exclude_backup_zips and p.suffix.lower() == ".zip" and p.name.startswith("backup-"):
                continue
            files += 1
            try:
                total += p.stat().st_size
            except Exception:
                pass
    return files, total

# ----------------------------
# Helpers: prompts & validation
# ----------------------------

def _prompt_with_default(prompt: str, default: str) -> str:
    entered = input(f"{prompt} [{default}]: ").strip()
    return entered or default


def _confirm(prompt: str, default: bool = True) -> bool:
    """Ask user to confirm an action. Returns True if confirmed."""
    yn = "Y/n" if default else "y/N"
    print(prompt)
    ans = input(f"[{yn}] : ").strip().lower()
    if not ans:
        return default
    return ans in {"y", "yes"}


def _ensure_dir(path: Path, name: str) -> None:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"{name} does not exist or is not a directory: {path}")


def _ensure_file(path: Path, name: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{name} does not exist or is not a file: {path}")


# ----------------------------
# Backup utilities
# ----------------------------

def _zip_directory(src_dir: Path, out_dir: Path, label: str, *, mode: str = BACKUP_NORMAL, progress_every: int = 1000, exclude_dirs: Optional[Set[str]] = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    zip_path = out_dir / f"backup-{label}-{ts}.zip"
    compression = zipfile.ZIP_STORED if mode == BACKUP_FAST else zipfile.ZIP_DEFLATED
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=compression, allowZip64=True) as zf:
        for p in src_dir.rglob("*"):
            if p.is_file():
                # Skip this zip file itself and any prior backup zips
                if p == zip_path:
                    continue
                if p.suffix.lower() == ".zip" and p.name.startswith("backup-"):
                    continue
                if exclude_dirs:
                    try:
                        rel_parts = p.relative_to(src_dir).parts
                    except Exception:
                        rel_parts = p.parts
                    if any(part in exclude_dirs for part in rel_parts):
                        continue
                zf.write(p, p.relative_to(src_dir))
                count += 1
                if progress_every and count % progress_every == 0:
                    print(f"[progress] {label}: added {count} files…")
    print(f"[info] {label} backup complete. Files added: {count}")
    return zip_path
# ----------------------------
# Backup mode chooser
# ----------------------------
DEFAULT_FAST_FILE_THRESHOLD = 2000
DEFAULT_FAST_SIZE_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1 GB

def _choose_backup_mode(dir_path: Path, files: int, total_bytes: int, label: str) -> str:
    big = files >= DEFAULT_FAST_FILE_THRESHOLD or total_bytes >= DEFAULT_FAST_SIZE_THRESHOLD
    default_choice = BACKUP_FAST if big else BACKUP_NORMAL
    prompt_lines = [
        f"Create ZIP backup of {label} directory?",
        f"  • Path : {dir_path}",
        f"  • Files: {files}",
        f"  • Size : {_human_bytes(total_bytes)}",
        f"  • Output: {dir_path}/backup-{label}-<timestamp>.zip",
        "  • Excludes    : backup-*.zip",
    ]
    if label == "root":
        prompt_lines.append("  • Excludes    : venv/")
    prompt_lines += [
        "Choose mode: [F]ast (no compression), [N]ormal (compressed), [S]kip",
        f"Recommendation: {'Fast' if default_choice==BACKUP_FAST else 'Normal'} (based on size/count)",
    ]
    ans = input("\n".join(prompt_lines) + " [F/n/s]: ").strip().lower()
    if not ans:
        return default_choice
    if ans.startswith("s"):
        return BACKUP_SKIP
    if ans.startswith("f"):
        return BACKUP_FAST
    if ans.startswith("n"):
        return BACKUP_NORMAL
    # Fallback to default on unrecognized input
    return default_choice


# ----------------------------
# Path normalization helpers (for bundle-style folders)
# ----------------------------

def _normalize_parts(parts: List[str]) -> List[str]:
    """Normalize path components for better suffix matching.
    - Treat `*.lwc` directories as their base name (strip `.lwc`).
    - Treat `lwc_bundle` as `lwc` (common alias in reports).
    """
    out: List[str] = []
    for p in parts:
        q = p
        if q.endswith('.lwc'):
            q = q[:-4]  # strip .lwc from bundle folder names
        if q == 'lwc_bundle':
            q = 'lwc'
        out.append(q)
    return out

# ----------------------------
# Path matching logic
# ----------------------------

def _common_suffix_components(a_parts: List[str], b_parts: List[str]) -> int:
    """Return the number of matching path components from the right (basename-first)."""
    count = 0
    for i in range(1, min(len(a_parts), len(b_parts)) + 1):
        if a_parts[-i] == b_parts[-i]:
            count += 1
        else:
            break
    return count


def _best_suffix_match(target_path_text: str, candidate_rel_paths: List[Path]) -> Optional[Path]:
    # Normalize target text into path components (POSIX style)
    target_parts_raw = [p for p in Path(target_path_text).as_posix().split('/') if p]
    if not target_parts_raw:
        return None

    target_basename = target_parts_raw[-1].lower()

    # 0) QUICK PASS: prefer a direct basename match with the **fewest components**
    #    This avoids picking very deep paths like stacks/python-raw/src/... when a
    #    shallower candidate exists (e.g., code/FDRIT-0103/<file>). Case-insensitive.
    shallow_best: Optional[Tuple[int, Path]] = None  # (depth, path)
    for cand in candidate_rel_paths:
        cand_parts = [p for p in cand.as_posix().split('/') if p]
        if not cand_parts:
            continue
        if cand_parts[-1].lower() == target_basename:
            depth = len(cand_parts)
            if shallow_best is None or depth < shallow_best[0]:
                shallow_best = (depth, cand)
    if shallow_best is not None:
        return shallow_best[1]

    # Build target variants for matching
    variants: List[List[str]] = []

    # Variant A: straight normalization (default)
    norm_default = _normalize_parts(target_parts_raw)
    variants.append(norm_default)

    # Variant B: collapse .../<comp>.lwc/<comp>.* to .../<comp>.lwc
    if len(target_parts_raw) >= 2:
        last = target_parts_raw[-1]
        prev = target_parts_raw[-2]
        if prev.lower().endswith('.lwc'):
            base_lwc = prev[:-4]  # strip .lwc
            if base_lwc and last.lower().startswith(base_lwc.lower() + '.'):
                collapsed = target_parts_raw[:-2] + [prev]
                variants.append(_normalize_parts(collapsed))

    # 1) SUFFIX SCORING: choose longest common suffix; on ties, prefer **shallower** path
    best_suffix_len = 0
    best_total_len = 10**9  # large so smaller wins
    best_path: Optional[Path] = None

    for cand in candidate_rel_paths:
        cand_parts = _normalize_parts([p for p in cand.as_posix().split('/') if p])
        total_len = len(cand_parts)
        for v in variants:
            suffix_len = _common_suffix_components(v, cand_parts)
            if suffix_len > 0:
                if (suffix_len > best_suffix_len) or (
                    suffix_len == best_suffix_len and total_len < best_total_len
                ):
                    best_suffix_len = suffix_len
                    best_total_len = total_len
                    best_path = cand

    return best_path


# ----------------------------
# Report rewriting
# ----------------------------


CODE_BLOCK_RE = re.compile(r"(^\*\*Code\s+Block:\*\*\s*)([^\n\r]+)", re.MULTILINE)
# Variant: some reports put the file path on the next line as "File: <path>"
CODE_FILE_RE = re.compile(r"(^[ \t]*File:\s*)([^\n\r]+)", re.MULTILINE)

# Extract code-block paths from a markdown string (both same-line and next-line forms)
def _extract_codeblock_paths(md_text: str) -> List[str]:
    paths: List[str] = []
    # Same-line form: **Code Block:** <path>
    paths.extend(m.group(2).strip() for m in CODE_BLOCK_RE.finditer(md_text))
    # Next-line form: File: <path>
    paths.extend(m.group(2).strip() for m in CODE_FILE_RE.finditer(md_text))
    return paths

# Helper to load the adjacent JSON changelog if present
def _load_report_changelog(updated_report_path: Path) -> Optional[dict]:
    json_path = updated_report_path.with_name(updated_report_path.name + ".json")
    if json_path.exists():
        try:
            return json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

# Helper to extract old header values from a matched log text
def _extract_current_header_values(log_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    def _val(regex: re.Pattern) -> Optional[str]:
        m = regex.search(log_text)
        if m:
            line = m.group(0)
            # split on '=' once from the right side of the header key
            try:
                return line.split('=', 1)[1].strip()
            except Exception:
                return None
        return None
    return _val(HDR_ROOT_RE), _val(HDR_SRC_RE), _val(HDR_OUT_RE)


def _rewrite_report(report_path: Path, source_rel_files: List[Path], *, root_dir: Optional[Path] = None, src_rel: Optional[Path] = None, report_index: Optional[int] = None, report_total: Optional[int] = None) -> Tuple[int, int]:
    """Rewrite a single business_logic_report.md with per-change confirmation and rich context.
    Returns (num_found, num_rewritten).
    """
    original_text = report_path.read_text(encoding="utf-8")

    num_found = 0
    num_rewritten = 0

    # Collect matches first
    matches = list(CODE_BLOCK_RE.finditer(original_text))
    # Also capture `File: <path>` entries
    matches_file = list(CODE_FILE_RE.finditer(original_text))
    # Merge both kinds and sort by start position so we rewrite in document order
    matches = sorted(matches + matches_file, key=lambda m: m.start())
    if not matches:
        return 0, 0

    rel_report = report_path.as_posix()
    if root_dir:
        try:
            rel_report = report_path.relative_to(root_dir).as_posix()
        except Exception:
            pass

    # Build new text with selective replacements
    new_chunks = []
    last_idx = 0
    accepted_changes: List[Tuple[str, str]] = []

    total_blocks = len(matches)
    header_prefix = f"[report {report_index}/{report_total}] " if (report_index and report_total) else ""

    for idx, m in enumerate(matches, start=1):
        prefix = m.group(1)
        orig_path_text = m.group(2).strip()
        num_found += 1

        best = _best_suffix_match(orig_path_text, source_rel_files)
        # Write unchanged segment before the match
        new_chunks.append(original_text[last_idx:m.start()])

        # Default replacement is original
        replacement_text = orig_path_text
        if best is not None:
            candidate = best.as_posix()
            # Compute a quick match score (suffix components matched / candidate components) using normalized parts
            target_parts = _normalize_parts([p for p in Path(orig_path_text).as_posix().split('/') if p])
            cand_parts = _normalize_parts([p for p in best.as_posix().split('/') if p])
            score = _common_suffix_components(target_parts, cand_parts)
            # Heuristic: if the original includes '<x>.lwc/<x>.js', note that we applied bundle collapse
            parts_raw = [p for p in Path(orig_path_text).as_posix().split('/') if p]
            bundle_collapse = (
                len(parts_raw) >= 2
                and parts_raw[-2].lower().endswith('.lwc')
                and parts_raw[-2][:-4]
                and parts_raw[-1].lower().startswith(parts_raw[-2][:-4].lower() + '.')
            )
            if bundle_collapse:
                # If best points at .../<comp>.lwc, propose .../<comp>.lwc.<member-filename>
                best_str = best.as_posix()
                if best_str.lower().endswith('.lwc'):
                    member_filename = parts_raw[-1]
                    candidate = f"{best_str}.{member_filename}"
            normalized_hint = " (normalized .lwc bundle)" if any(part.endswith('.lwc') for part in Path(orig_path_text).as_posix().split('/')) else ""
            detail = "\n".join([
                f"{header_prefix}Change code block path in {rel_report} (block {idx}/{total_blocks})",
                f"  • Source root : {src_rel.as_posix() if src_rel else '<multiple>'}",
                f"  • From        : {orig_path_text}{normalized_hint}{' (collapsed .lwc/<name>.*)' if bundle_collapse else ''}",
                f"  • To          : {candidate}",
                f"  • Match score : {score} / {len(cand_parts)} components",
            ])
            if _confirm(detail, default=True):
                replacement_text = candidate
                num_rewritten += 1
                accepted_changes.append((orig_path_text, candidate))
            else:
                print("[skip] Change declined; leaving original path.")
        else:
            print("[warn] No suffix match:", orig_path_text)

        # Reconstruct the matched segment with chosen replacement
        new_chunks.append(prefix + replacement_text)
        last_idx = m.end()

    # Append remaining tail
    new_chunks.append(original_text[last_idx:])

    new_text = "".join(new_chunks)

    if new_text != original_text:
        # We do not overwrite original; write to a side-by-side .updated file
        out_path = report_path.with_name(report_path.name + ".updated")
        rel_out = out_path.as_posix()
        if root_dir:
            try:
                rel_out = out_path.relative_to(root_dir).as_posix()
            except Exception:
                pass

        summary_lines = [
            f"Write changes to new file? {rel_out}",
            f"  • Based on      : {rel_report}",
            f"  • Blocks found  : {num_found}",
            f"  • Blocks updated: {num_rewritten}",
        ]
        if accepted_changes:
            preview = "\n".join([f"    - {a}  →  {b}" for a, b in accepted_changes])
            summary_lines.append("  • Changes:")
            summary_lines.append(preview)
        if _confirm("\n".join(summary_lines), default=True):
            # 1) Write the updated markdown file (side-by-side)
            out_path.write_text(new_text, encoding="utf-8")
            print(f"[info] Wrote updated report: {rel_out}")

            # 2) Write a JSON change log next to the updated file
            changelog = {
                "report": rel_out,
                "based_on": rel_report,
                "blocks_found": num_found,
                "blocks_updated": num_rewritten,
                "changes": [{"from": a, "to": b} for (a, b) in accepted_changes],
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
            json_path = out_path.with_name(out_path.name + ".json")  # e.g., business_logic_report.md.updated.json
            try:
                json_path.write_text(json.dumps(changelog, indent=2), encoding="utf-8")
                rel_json = json_path.as_posix()
                if root_dir:
                    try:
                        rel_json = json_path.relative_to(root_dir).as_posix()
                    except Exception:
                        pass
                print(f"[info] Wrote change log: {rel_json}")
            except Exception as je:
                print(f"[warn] Failed to write change log JSON: {je}")
        else:
            print(f"[skip] Did not write changes to {rel_out}")
            num_rewritten = 0

    return num_found, num_rewritten



# ----------------------------
# PHASE TWO: Update logs based on .updated reports
# ----------------------------

RESP_BEGIN_RE = re.compile(r"^\[PCPTLOG:\]\s+=+\s+RESPONSE BEGIN\s+=+\s*$", re.MULTILINE)
RESP_END_RE   = re.compile(r"^\[PCPTLOG:\]\s+=+\s+RESPONSE END\s+=+\s*$", re.MULTILINE)

HDR_ROOT_RE   = re.compile(r"^(\[PCPTLOG:\]\s+root_dir=).*$", re.MULTILINE)
HDR_SRC_RE    = re.compile(r"^(\[PCPTLOG:\]\s+source_path=).*$", re.MULTILINE)
HDR_OUT_RE    = re.compile(r"^(\[PCPTLOG:\]\s+output_path=).*$", re.MULTILINE)
HDR_PROMPT_RE = re.compile(r"^(\[PCPTLOG:\]\s+prompt_template=).*$", re.MULTILINE)
def _extract_prompt_template(log_text: str) -> Optional[str]:
    m = HDR_PROMPT_RE.search(log_text)
    if not m:
        return None
    line = m.group(0)
    try:
        return line.split('=', 1)[1].strip()
    except Exception:
        return None


def _update_prompt_template(text: str, new_prompt: Optional[str]) -> str:
    if not new_prompt:
        return text
    return HDR_PROMPT_RE.sub(r"\1" + new_prompt, text)


def _find_logs_by_header(logs_dir: Path, *, root: str, src: str, out: str, prompt: Optional[str]) -> List[Path]:
    matches: List[Path] = []
    for p in logs_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        r = HDR_ROOT_RE.search(txt)
        s = HDR_SRC_RE.search(txt)
        o = HDR_OUT_RE.search(txt)
        t = HDR_PROMPT_RE.search(txt)
        if not (r and s and o):
            continue
        rv = r.group(0).split('=', 1)[1].strip()
        sv = s.group(0).split('=', 1)[1].strip()
        ov = o.group(0).split('=', 1)[1].strip()
        tv = t.group(0).split('=', 1)[1].strip() if t else None
        if rv == root and sv == src and ov == out and (prompt is None or tv == prompt):
            matches.append(p)
    # Sort oldest->newest to make preview deterministic
    matches.sort(key=lambda p: p.stat().st_mtime)
    return matches


# Apply from->to replacements only inside **Code Block:** lines of RESPONSE section
def _apply_codeblock_changes_to_response(response_md: str, changes: List[dict]) -> Tuple[str, int]:
    """Apply from->to replacements only inside **Code Block:** lines. Returns (new_response, replacements).
    """
    # Python trick: use dict to hold mutable counter inside nested func
    repl_counter = {"n": 0}
    def _sub_line_count(m: re.Match) -> str:
        prefix = m.group(1)
        path = m.group(2)
        new_path = path
        for ch in changes:
            frm = ch.get("from")
            to  = ch.get("to")
            if not frm or not to:
                continue
            if new_path == frm:
                new_path = to
                repl_counter["n"] += 1
        return prefix + new_path
    new_md = CODE_BLOCK_RE.sub(_sub_line_count, response_md)
    # Also apply to `File: <path>` form
    def _sub_line_count_file(m: re.Match) -> str:
        prefix = m.group(1)
        path = m.group(2)
        new_path = path
        for ch in changes:
            frm = ch.get("from"); to = ch.get("to")
            if not frm or not to:
                continue
            if new_path == frm:
                new_path = to
                repl_counter["n"] += 1
        return prefix + new_path
    new_md = CODE_FILE_RE.sub(_sub_line_count_file, new_md)
    return new_md, repl_counter["n"]

# Rewrite **Code Block:** paths by replacing old source-path prefix with new source-path
# Returns (new_response_markdown, replacements_made)
# Rewrite **Code Block:** paths by replacing old source-path prefix with new source-path,
# or insert new source-path when only a bare filename is present. Handles both forms
# (same line and next-line 'File:') and preserves backticks.

def _rewrite_codeblock_paths_by_src(response_md: str, old_src: str, new_src: str) -> Tuple[str, int]:
    # Rewrite **Code Block:** paths by replacing old source-path prefix with new source-path,
    # insert new source-path for bare filenames, and normalize partial old-src suffixes
    # (e.g., path starts with tail of old_src such as 'stored_proc/...'). Works for both
    # same line and next-line 'File:' forms and preserves backticks.
    old_src_posix = Path(old_src).as_posix().rstrip('/')
    new_src_posix = Path(new_src).as_posix().rstrip('/')

    # Build suffixes of old_src to catch partial prefixes like 'stored_proc/...'
    old_parts = [p for p in old_src_posix.split('/') if p]
    old_suffixes = []
    for i in range(len(old_parts)):
        suffix = '/'.join(old_parts[i:])  # progressively shorter heads removed
        if suffix:  # keep non-empty suffixes
            old_suffixes.append(suffix)
    # Ensure unique and prefer longer suffix first for matching
    old_suffixes = sorted(set(old_suffixes), key=lambda s: (-len(s), s))

    counter = {"n": 0}

    def _transform_raw_path(raw: str) -> str:
        # Preserve backticks if they wrap the path
        had_ticks = raw.startswith("`") and raw.endswith("`") and len(raw) >= 2
        inner = raw[1:-1] if had_ticks else raw
        inner = inner.strip()

        # Case A: replace exact old_src prefix → new_src
        if inner == old_src_posix or inner.startswith(old_src_posix + "/"):
            new_inner = new_src_posix + inner[len(old_src_posix):]
            counter["n"] += 1
            return f"`{new_inner}`" if had_ticks else new_inner

        # Case B: bare filename (no '/') → insert new_src/
        if "/" not in inner and inner and not inner.startswith('.'):
            new_inner = new_src_posix + "/" + inner
            counter["n"] += 1
            return f"`{new_inner}`" if had_ticks else new_inner

        # Case C: partial old-src suffix present at start → strip that suffix and prepend new_src/
        # e.g., inner='stored_proc/Process.sql', old_src='code/stored_proc' ⇒ new_src+'/Process.sql'
        for suf in old_suffixes:
            if inner == suf:
                new_inner = new_src_posix
                counter["n"] += 1
                return f"`{new_inner}`" if had_ticks else new_inner
            if inner.startswith(suf + "/"):
                new_inner = new_src_posix + inner[len(suf):]
                counter["n"] += 1
                return f"`{new_inner}`" if had_ticks else new_inner

        # No change
        return raw

    def _sub_same_line(m: re.Match) -> str:
        prefix = m.group(1)
        raw = m.group(2).strip()
        return prefix + _transform_raw_path(raw)

    def _sub_next_line(m: re.Match) -> str:
        prefix = m.group(1)
        raw = m.group(2).strip()
        return prefix + _transform_raw_path(raw)

    new_md = CODE_BLOCK_RE.sub(_sub_same_line, response_md)
    new_md = CODE_FILE_RE.sub(_sub_next_line, new_md)
    return new_md, counter["n"]


def _find_response_section(text: str) -> Optional[Tuple[int, int, str]]:
    """Return (start_idx, end_idx, content) for the RESPONSE section within the PCPT log text.
    The indices are the boundaries of the content between the BEGIN and END markers (exclusive).
    """
    m_begin = RESP_BEGIN_RE.search(text)
    if not m_begin:
        return None
    m_end = RESP_END_RE.search(text, m_begin.end())
    if not m_end:
        return None
    start = m_begin.end()  # content starts after BEGIN line and newline
    end = m_end.start()    # content ends right before END line
    return start, end, text[start:end]


def _replace_response_section(text: str, new_content: str) -> str:
    loc = _find_response_section(text)
    if not loc:
        raise ValueError("Could not locate RESPONSE section in the log file.")
    start, end, _old = loc
    return text[:start] + "\n" + new_content.rstrip("\n") + "\n" + text[end:]


def _update_header_fields(text: str, *, new_root: str, new_src: str, new_out: str) -> str:
    text = HDR_ROOT_RE.sub(r"\1" + new_root, text)
    text = HDR_SRC_RE.sub(r"\1" + new_src, text)
    text = HDR_OUT_RE.sub(r"\1" + new_out, text)
    return text


def _most_recent_matching_log(logs_dir: Path, original_report_md: str) -> Optional[Path]:
    """Search logs_dir recursively for the most recent file whose RESPONSE content exactly equals original_report_md."""
    best_path: Optional[Path] = None
    best_mtime: float = -1.0
    for p in logs_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        loc = _find_response_section(txt)
        if not loc:
            continue
        _s, _e, resp = loc
        # Exact content match
        if resp.strip() == original_report_md.strip():
            mt = p.stat().st_mtime
            if mt > best_mtime:
                best_mtime = mt
                best_path = p
    return best_path


def _write_updated_log(orig_log_path: Path, new_text: str) -> Path:
    out_path = orig_log_path.with_name(orig_log_path.name + ".updated")
    out_path.write_text(new_text, encoding="utf-8")
    return out_path


def _process_phase_two(root_dir: Path, logs_dir: Path, path_pairs: List[dict]) -> Tuple[int, int, int]:
    """For each path-pair, for each .updated report: find most recent matching log and write a .updated log with header and response updated.
    Returns (reports_seen, logs_matched, logs_updated).
    """
    reports_seen = 0
    logs_matched = 0
    logs_updated = 0

    for i, pair in enumerate(path_pairs, start=1):
        src_rel = Path(pair["source-path"]).as_posix()
        out_rel = Path(pair["output-path"]).as_posix()
        out_dir = (root_dir / out_rel)

        # All .updated reports in output path
        updated_reports = list(out_dir.rglob("business_logic_report.md.updated"))
        if not updated_reports:
            print(f"[info] Phase 2: Pair {i} has no .updated reports under {out_rel}")
            continue

        print(f"[info] Phase 2: Pair {i}: source-path={src_rel} output-path={out_rel} | updated reports: {len(updated_reports)}")

        for rpt_upd in updated_reports:
            reports_seen += 1
            rpt_orig = rpt_upd.with_name("business_logic_report.md")
            try:
                original_md = rpt_orig.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[warn] Cannot read base report for {rpt_upd}: {e}")
                continue
            try:
                updated_md = rpt_upd.read_text(encoding="utf-8")
            except Exception as e:
                print(f"[warn] Cannot read updated report {rpt_upd}: {e}")
                continue

            # Find most recent log that contains original_md in RESPONSE
            match = _most_recent_matching_log(logs_dir, original_md)
            if not match:
                print(f"[warn] No matching log found for updated report: {rpt_upd.relative_to(root_dir).as_posix()}")
                continue
            logs_matched += 1

            # Build a richer, decision-ready preview
            try:
                log_text_for_hdr = match.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                log_text_for_hdr = ""
            old_root, old_src, old_out = _extract_current_header_values(log_text_for_hdr)

            # Path for the matched log relative to root (if possible)
            try:
                rel_match = match.relative_to(root_dir).as_posix()
            except Exception:
                rel_match = match.as_posix()

            # Build preview of Code Block changes with counts and cap
            changelog = _load_report_changelog(rpt_upd)
            changes_preview: List[str] = []
            MAX_PREVIEW = 12

            def _dedupe_preserve_order(pairs: List[tuple]) -> List[tuple]:
                seen = set()
                out = []
                for a, b in pairs:
                    key = (a, b)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(key)
                return out

            if changelog and isinstance(changelog.get("changes"), list):
                pairs = [(ch.get("from"), ch.get("to")) for ch in changelog["changes"] if ch.get("from") and ch.get("to")]
                pairs = _dedupe_preserve_order(pairs)
                total = len(pairs)
                head = pairs[:MAX_PREVIEW]
                changes_preview.append(f"  • Code block changes (showing {len(head)} of {total}):")
                for frm, to in head:
                    changes_preview.append(f"    - {frm}  →  {to}")
                if total > len(head):
                    changes_preview.append(f"    … and {total - len(head)} more")
            else:
                # Fallback: show a small sample from original vs updated
                orig_blocks = _extract_codeblock_paths(original_md)
                upd_blocks  = _extract_codeblock_paths(updated_md)
                changes_preview.append("  • Code block changes: (no changelog; sample)")
                if orig_blocks:
                    changes_preview.append("    • Original (first 3):")
                    for pth in orig_blocks[:3]:
                        changes_preview.append(f"      - {pth}")
                if upd_blocks:
                    changes_preview.append("    • Updated (first 3):")
                    for pth in upd_blocks[:3]:
                        changes_preview.append(f"      - {pth}")

            rel_rpt_upd = rpt_upd.relative_to(root_dir).as_posix() if root_dir in rpt_upd.parents else rpt_upd.as_posix()
            header_lines = [
                f"Update log? {rel_match}",
                f"  • root_dir      : {old_root or '<unknown>'}  →  {root_dir.as_posix()}",
                f"  • source_path   : {old_src or '<unknown>'}  →  {src_rel}",
                f"  • output_path   : {old_out or '<unknown>'}  →  {out_rel}",
                f"  • RESPONSE from : base report (current)",
                f"  • RESPONSE to   : {rel_rpt_upd}",
                f"  • Code block changes:",
            ] + changes_preview

            preview = "\n".join(header_lines)
            if not _confirm(preview, default=True):
                print("[skip] Log update declined by user.")
                continue

            # Perform replacements
            try:
                txt = match.read_text(encoding="utf-8", errors="ignore")
                txt2 = _update_header_fields(txt, new_root=str(root_dir), new_src=src_rel, new_out=out_rel)
                txt3 = _replace_response_section(txt2, updated_md)
            except Exception as e:
                print(f"[warn] Failed to update log {rel_match}: {e}")
                continue

            # Write .updated next to original
            out_log = _write_updated_log(match, txt3)
            rel_out = out_log.relative_to(root_dir).as_posix() if root_dir in out_log.parents else out_log.as_posix()
            print(f"[info] Wrote updated log: {rel_out}")
            logs_updated += 1

            # ----------------------------
            # PHASE 2b — propagate updates to prior matching ingestions
            # ----------------------------
            # Gather header quartet from the *original* matched log before changes
            try:
                old_root, old_src, old_out = _extract_current_header_values(log_text_for_hdr)
                old_prompt = _extract_prompt_template(log_text_for_hdr)
            except Exception:
                old_root = old_src = old_out = None
                old_prompt = None

            if not (old_root and old_src and old_out):
                continue  # cannot proceed without the original quartet

            # Find other logs with the same old quartet (exclude the one we just updated)
            siblings = [p for p in _find_logs_by_header(logs_dir, root=old_root, src=old_src, out=old_out, prompt=old_prompt) if p != match]
            if not siblings:
                continue

            # Load changelog changes (from phase 1) to apply to RESPONSE Code Block lines
            changes = changelog.get("changes") if changelog else None
            if not isinstance(changes, list):
                changes = []

            print(f"[info] Phase 2b: Found {len(siblings)} prior ingestion log(s) to update for header quartet.")

            for sib in siblings:
                try:
                    sib_text = sib.read_text(encoding="utf-8", errors="ignore")
                except Exception as e:
                    print(f"[warn] Cannot read sibling log {sib}: {e}")
                    continue

                # Update headers to new values
                sib_new = _update_header_fields(sib_text, new_root=str(root_dir), new_src=src_rel, new_out=out_rel)
                # Keep prompt_template the same as in the sibling unless we have a new value we want to enforce
                new_prompt = _extract_prompt_template(txt2) or _extract_prompt_template(sib_text)
                sib_new = _update_prompt_template(sib_new, new_prompt)

                # Apply Code Block path changes inside the RESPONSE section by replacing old src prefix → new src
                loc = _find_response_section(sib_new)
                if not loc:
                    print(f"[warn] Sibling log missing RESPONSE section: {sib}")
                    continue
                s_i, s_j, s_resp = loc
                s_resp_new, rep_count = _rewrite_codeblock_paths_by_src(s_resp, old_src, src_rel)

                # Prepare a sample of transformed Code Block paths for the preview (up to 12, with counts)
                sample_lines = []
                shown = 0
                MAX_PREVIEW = 12
                orig_blocks = _extract_codeblock_paths(s_resp)
                old_src_posix = Path(old_src).as_posix().rstrip('/')
                new_src_posix = Path(src_rel).as_posix().rstrip('/')
                old_parts = [p for p in old_src_posix.split('/') if p]
                old_suffixes = sorted({"/".join(old_parts[i:]) for i in range(len(old_parts)) if "/".join(old_parts[i:])}, key=lambda s: (-len(s), s))

                total_candidates = len(orig_blocks)
                for pth_raw in orig_blocks:
                    had_ticks = pth_raw.startswith('`') and pth_raw.endswith('`') and len(pth_raw) >= 2
                    pth = pth_raw[1:-1] if had_ticks else pth_raw

                    new_disp = None
                    # Case A: full old_src prefix
                    if pth == old_src_posix or pth.startswith(old_src_posix + "/"):
                        new_inner = new_src_posix + pth[len(old_src_posix):]
                        new_disp = f"`{new_inner}`" if had_ticks else new_inner
                    # Case B: bare filename
                    elif "/" not in pth and pth and not pth.startswith('.'):
                        new_inner = new_src_posix + "/" + pth
                        new_disp = f"`{new_inner}`" if had_ticks else new_inner
                    else:
                        # Case C: partial old-src suffix at start
                        for suf in old_suffixes:
                            if pth == suf:
                                new_inner = new_src_posix
                                new_disp = f"`{new_inner}`" if had_ticks else new_inner
                                break
                            if pth.startswith(suf + "/"):
                                new_inner = new_src_posix + pth[len(suf):]
                                new_disp = f"`{new_inner}`" if had_ticks else new_inner
                                break

                    if new_disp:
                        sample_lines.append(f"    - {pth_raw}  →  {new_disp}")
                        shown += 1
                        if shown >= MAX_PREVIEW:
                            break

                # Build preview
                try:
                    rel_sib = sib.relative_to(root_dir).as_posix()
                except Exception:
                    rel_sib = sib.as_posix()
                preview_lines = [
                    f"Update prior ingestion log? {rel_sib}",
                    f"  • root_dir    : {old_root}  →  {root_dir.as_posix()}",
                    f"  • source_path : {old_src}   →  {src_rel}",
                    f"  • output_path : {old_out}   →  {out_rel}",
                    f"  • prompt_templ: {old_prompt or '<unchanged>'}  →  {new_prompt or (old_prompt or '<unchanged>')}",
                    f"  • Code Block rewrite rule: replace prefix '{Path(old_src).as_posix()}' → '{Path(src_rel).as_posix()}'",
                    f"  • Code Block paths updated: {rep_count}",
                ]
                if sample_lines:
                    preview_lines.append(f"  • Samples (showing {shown} of {total_candidates}):")
                    preview_lines.extend(sample_lines)

                if not _confirm("\n".join(preview_lines), default=True):
                    print("[skip] Prior ingestion log update declined by user.")
                    continue

                # Write sibling .updated log
                sib_updated = sib_new[:s_i] + "\n" + s_resp_new.rstrip("\n") + "\n" + sib_new[s_j:]
                out_sib = _write_updated_log(sib, sib_updated)
                try:
                    rel_out_sib = out_sib.relative_to(root_dir).as_posix()
                except Exception:
                    rel_out_sib = out_sib.as_posix()
                print(f"[info] Wrote updated prior ingestion log: {rel_out_sib}")
                logs_updated += 1

    return reports_seen, logs_matched, logs_updated



# ----------------------------
# PHASE THREE: Apply .updated files (promote and clean up)
# ----------------------------

def _promote_updated_file(updated_path: Path, *, root_dir: Optional[Path] = None) -> bool:
    """Replace the original file with its `.updated` version and remove the `.updated` file.
    Returns True if promotion happened, False if declined or failed.
    """
    if not updated_path.name.endswith('.updated'):
        return False
    original_path = updated_path.with_name(updated_path.name[:-8])  # strip .updated

    # Build a decision-ready preview
    try:
        rel_updated = updated_path.relative_to(root_dir).as_posix() if root_dir and root_dir in updated_path.parents else updated_path.as_posix()
    except Exception:
        rel_updated = updated_path.as_posix()
    try:
        rel_original = original_path.relative_to(root_dir).as_posix() if root_dir and root_dir in original_path.parents else original_path.as_posix()
    except Exception:
        rel_original = original_path.as_posix()

    size_upd = updated_path.stat().st_size if updated_path.exists() else 0
    size_orig = original_path.stat().st_size if original_path.exists() else 0

    prompt = "\n".join([
        f"Promote updated file?",
        f"  • From (.updated): {rel_updated}  [{_human_bytes(size_upd)}]",
        f"  • To (original) : {rel_original}  [{_human_bytes(size_orig)}]",
        "  • Action        : replace original with .updated and remove .updated",
    ])
    if not _confirm(prompt, default=True):
        print("[skip] Promotion declined by user.")
        return False

    try:
        # Ensure destination directory exists
        original_path.parent.mkdir(parents=True, exist_ok=True)
        # Replace original with updated (this also removes the .updated path)
        updated_path.replace(original_path)
        print(f"[info] Promoted: {rel_updated}  →  {rel_original}")

        # Best-effort cleanup of the report sidecar changelog JSON (Phase 1 artifact)
        # This file is named like: <filename>.updated.json next to the promoted file
        sidecar_json = updated_path.with_name(updated_path.name + ".json")
        if sidecar_json.exists():
            try:
                sidecar_json.unlink()
                # Show relative path if possible
                try:
                    rel_sidecar = sidecar_json.relative_to(root_dir).as_posix() if root_dir and root_dir in sidecar_json.parents else sidecar_json.as_posix()
                except Exception:
                    rel_sidecar = sidecar_json.as_posix()
                print(f"[info] Removed sidecar JSON: {rel_sidecar}")
            except Exception as je:
                print(f"[warn] Failed to remove sidecar JSON {sidecar_json}: {je}")

        return True
    except Exception as e:
        print(f"[warn] Failed to promote {rel_updated}: {e}")
        return False


def _process_phase_three(root_dir: Path, logs_dir: Path, path_pairs: List[dict]) -> Tuple[int, int]:
    """Promote .updated files for reports and logs, then delete .updated files.
    Returns (reports_promoted, logs_promoted).
    """
    reports_promoted = 0
    logs_promoted = 0

    # 1) Reports/docs side: for each path pair, promote any \*.updated files under output-path
    for i, pair in enumerate(path_pairs, start=1):
        out_rel = Path(pair["output-path"]).as_posix()
        out_dir = (root_dir / out_rel)
        updated_reports = list(out_dir.rglob("*.updated"))
        if updated_reports:
            print(f"[info] Phase 3: Pair {i} has {len(updated_reports)} .updated file(s) under {out_rel}")
        for upd in updated_reports:
            if _promote_updated_file(upd, root_dir=root_dir):
                reports_promoted += 1

    # Optional bulk cleanup of any remaining report sidecar JSONs (*.updated.json) under output-paths
    leftover_sidecars = []
    for i, pair in enumerate(path_pairs, start=1):
        out_rel = Path(pair["output-path"]).as_posix()
        out_dir = (root_dir / out_rel)
        leftover_sidecars.extend(list(out_dir.rglob("*.updated.json")))

    if leftover_sidecars:
        try:
            # Present a concise, decision-ready prompt
            sample = []
            for p in leftover_sidecars[:8]:
                try:
                    sample.append(p.relative_to(root_dir).as_posix())
                except Exception:
                    sample.append(p.as_posix())
            prompt_lines = [
                "Remove leftover report sidecar JSON files?",
                f"  • Count : {len(leftover_sidecars)}",
                "  • Sample:",
            ] + [f"    - {s}" for s in sample]
            if _confirm("\n".join(prompt_lines), default=True):
                removed = 0
                for p in leftover_sidecars:
                    try:
                        p.unlink()
                        removed += 1
                    except Exception:
                        pass
                print(f"[info] Removed {removed} sidecar JSON file(s).")
            else:
                print("[skip] Leftover sidecar JSON files retained.")
        except Exception as ce:
            print(f"[warn] Sidecar JSON cleanup skipped due to error: {ce}")

    # 2) Logs side: promote any \*.updated files in logs_dir
    updated_logs = list(logs_dir.rglob("*.updated"))
    if updated_logs:
        print(f"[info] Phase 3: Logs have {len(updated_logs)} .updated file(s) to promote")
    for upd in updated_logs:
        if _promote_updated_file(upd, root_dir=root_dir):
            logs_promoted += 1

    return reports_promoted, logs_promoted

# ----------------------------
# Main flow (implements the spec)
# ----------------------------

def main() -> None:
    print("[info] patch_logs_paths starting…")

    # 1) prompt for json spec file (default to spec/patch_logs_paths.json)
    default_spec = str(Path("tools/spec/patch_logs_paths.json"))
    spec_file = Path(_prompt_with_default("Spec file (JSON)", default_spec)).expanduser().resolve()

    # Ensure spec exists before proceeding
    _ensure_file(spec_file, "Spec file")

    # Load spec JSON early so we can default the root directory from it
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    # Diagnostics: show which spec file and its top-level keys
    print(f"[info] Using spec file: {spec_file}")
    try:
        print(f"[info] Spec top-level keys: {list(spec.keys())}")
    except Exception:
        print("[warn] Spec is not a JSON object; type=", type(spec))

    # Derive the default root from spec if available; else fall back to CWD
    spec_root = spec.get("root-directory")
    default_root = str(Path(spec_root).expanduser().resolve()) if spec_root else str(Path.cwd())

    # 2) prompt for root dir (default to value from spec if present)
    root_dir = Path(_prompt_with_default("Root directory", default_root)).expanduser().resolve()

    # 3) prompt for logs dir (default to ~/.pcpt/logs and allow change)
    default_logs = str(Path.home() / ".pcpt" / "log")
    logs_dir = Path(_prompt_with_default("Logs directory", default_logs)).expanduser().resolve()

    # 4) Preflight checks: existence
    _ensure_dir(root_dir, "Root directory")
    _ensure_dir(logs_dir, "Logs directory")

    # Robust lookup for path-pairs (tolerate minor variants & stray whitespace)
    path_pairs = None
    if isinstance(spec, dict):
        # direct lookups first
        path_pairs = spec.get("path-pairs") or spec.get("path_pairs")
        if path_pairs is None:
            # attempt fuzzy match across keys (normalize underscores/hyphens/spaces, lowercase)
            def _norm(s: str) -> str:
                return s.replace("_", "-").replace(" ", "").strip().lower()
            wanted = _norm("path-pairs")
            for k, v in spec.items():
                if _norm(str(k)) == wanted:
                    path_pairs = v
                    break
    if not path_pairs:
        raise ValueError("Spec missing 'path-pairs'.")

    # 4b) verify all paths specified in spec file exist under the root
    for i, pair in enumerate(path_pairs, start=1):
        src_rel = Path(pair["source-path"]).as_posix()
        out_rel = Path(pair["output-path"]).as_posix()
        src_dir = (root_dir / src_rel).resolve()
        out_dir = (root_dir / out_rel).resolve()
        if root_dir not in src_dir.parents and src_dir != root_dir:
            raise ValueError(f"source-path#{i} escapes root: {src_dir}")
        if root_dir not in out_dir.parents and out_dir != root_dir:
            raise ValueError(f"output-path#{i} escapes root: {out_dir}")
        _ensure_dir(src_dir, f"source-path#{i}")
        _ensure_dir(out_dir, f"output-path#{i}")

    # 5) backup logs dir to a zip placed in the logs dir (prompt first, with context & mode)
    logs_files, logs_bytes = _dir_stats(logs_dir, exclude_backup_zips=True)
    if logs_files >= DEFAULT_FAST_FILE_THRESHOLD or logs_bytes >= DEFAULT_FAST_SIZE_THRESHOLD:
        print("[note] Large logs directory detected; defaulting to FAST backup for speed.")
    logs_mode = _choose_backup_mode(logs_dir, logs_files, logs_bytes, "log")
    if logs_mode != BACKUP_SKIP:
        logs_zip = _zip_directory(logs_dir, logs_dir, "log", mode=logs_mode, progress_every=1000)
        print(f"[info] Backup (logs) -> {logs_zip}")
    else:
        print("[skip] Logs backup skipped by user.")

    # 6) backup root dir to a zip placed in the root dir (prompt first, with context & mode)
    root_files, root_bytes = _dir_stats(root_dir, exclude_backup_zips=True, exclude_dirs={"venv"})
    if root_files >= DEFAULT_FAST_FILE_THRESHOLD or root_bytes >= DEFAULT_FAST_SIZE_THRESHOLD:
        print("[note] Large root directory detected; defaulting to FAST backup for speed.")
    root_mode = _choose_backup_mode(root_dir, root_files, root_bytes, "root")
    if root_mode != BACKUP_SKIP:
        root_zip = _zip_directory(root_dir, root_dir, "root", mode=root_mode, progress_every=2000, exclude_dirs={"venv"})
        print(f"[info] Backup (root) -> {root_zip}")
    else:
        print("[skip] Root backup skipped by user.")

    total_reports_scanned = 0
    total_entries_found = 0
    total_entries_rewritten = 0

    # 7,8,9) Process each path-pair
    for i, pair in enumerate(path_pairs, start=1):
        src_rel = Path(pair["source-path"])  # relative under root
        out_rel = Path(pair["output-path"])  # relative under root
        src_dir = (root_dir / src_rel)
        out_dir = (root_dir / out_rel)

        print(f"[info] Pair {i}: source-path={src_rel.as_posix()} output-path={out_rel.as_posix()}")

        # 8) Build a list of all files under root/source-path (store paths relative to root)
        source_rel_files: List[Path] = [p.relative_to(root_dir) for p in src_dir.rglob('*') if p.is_file()]
        print(f"[info]   source files discovered: {len(source_rel_files)}")

        # Find business_logic_report.md files under output-path
        reports = list(out_dir.rglob('business_logic_report.md'))
        print(f"[info]   reports discovered: {len(reports)}")

        total_reports = len(reports)
        for idx, rpt in enumerate(reports, start=1):
            found, rewritten = _rewrite_report(
                rpt,
                source_rel_files,
                root_dir=root_dir,
                src_rel=src_rel,
                report_index=idx,
                report_total=total_reports,
            )
            total_reports_scanned += 1
            total_entries_found += found
            total_entries_rewritten += rewritten
            print(f"[info]     processed report: {rpt.relative_to(root_dir).as_posix()} | code-blocks: {found} | rewritten: {rewritten}")

    # 10) Phase One summary
    print("[success] Phase One complete: patched code block paths.")
    print(f"[summary] reports scanned: {total_reports_scanned}")
    print(f"[summary] code-block entries found: {total_entries_found}")
    print(f"[summary] code-block entries rewritten: {total_entries_rewritten}")

    # ----------------------------
    # PHASE TWO — update logs
    # ----------------------------
    print("[info] Starting Phase Two: updating logs from .updated reports…")
    p2_reports_seen, p2_logs_matched, p2_logs_updated = _process_phase_two(root_dir, logs_dir, path_pairs)

    print("[success] Phase Two complete.")
    print(f"[summary] phase2 reports considered: {p2_reports_seen}")
    print(f"[summary] phase2 matching logs found: {p2_logs_matched}")
    print(f"[summary] phase2 logs updated: {p2_logs_updated}")

    # ----------------------------
    # PHASE THREE — promote .updated files and clean up
    # ----------------------------
    print("[info] Starting Phase Three: promoting .updated files…")
    p3_reports_promoted, p3_logs_promoted = _process_phase_three(root_dir, logs_dir, path_pairs)
    print("[success] Phase Three complete.")
    print(f"[summary] phase3 reports promoted: {p3_reports_promoted}")
    print(f"[summary] phase3 logs promoted   : {p3_logs_promoted}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[error] {e}")
        raise
