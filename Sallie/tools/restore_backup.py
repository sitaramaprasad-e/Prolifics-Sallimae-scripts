#!/usr/bin/env python3
"""
Restore a project backup in-place.

Spec:
1) Prompt for zip — default to most recent backup-root-<timestamp>.zip
2) Do an in-place unzip, overwriting existing files.
3) Announce success.

Extras (safe defaults):
- Supports non-interactive mode via --zip and --yes.
- Shows candidate backups with size and timestamp.
- Prints clear step headers and counts (Greg’s preference for trace/info).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
import zipfile
import json

# ----------------------------
# Helpers
# ----------------------------

# Prompt helpers

def _prompt_with_default(prompt: str, default: str) -> str:
    entered = input(f"{prompt} [{default}]: ").strip()
    return entered or default


def _confirm(prompt: str, default: bool = False) -> bool:
    yn = "Y/n" if default else "y/N"
    print(prompt)
    ans = input(f"[{yn}] : ").strip().lower()
    if not ans:
        return default
    return ans in ("y", "yes")


def human_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num < 1024.0:
            return f"{num:,.0f} {unit}" if unit == "B" else f"{num:,.1f} {unit}"
        num /= 1024.0
    return f"{num:,.1f} PB"


def default_spec_path() -> Path:
    """Return default path to tools/spec/patch_logs_paths.json relative to this script."""
    # This script typically lives at <repo>/tools/restore_backup.py
    # The spec lives at <repo>/tools/spec/patch_logs_paths.json
    return Path(__file__).resolve().parent / "spec" / "patch_logs_paths.json"


def load_restore_root_from_spec(spec_path: Path) -> Path:
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to read/parse spec JSON: {spec_path} ({e})")
    root_dir = spec.get("root-directory") or spec.get("root_directory")
    if not root_dir:
        raise KeyError("Spec is missing 'root-directory'.")
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Spec root-directory does not exist: {root}")
    return root


def find_candidate_backups(base: Path, kind: str) -> list[Path]:
    """Return sorted list of backup zips for a given kind ('root' or 'log') under base."""
    if kind == "root":
        patterns = [
            base / "backup-root-*.zip",
            base / "backups" / "backup-root-*.zip",
        ]
    elif kind == "log":
        patterns = [
            base / "backup-log-*.zip",
            base / "backups" / "backup-log-*.zip",
        ]
    else:
        patterns = []
    candidates: list[Path] = []
    for p in patterns:
        candidates.extend(sorted(p.parent.glob(p.name)))
    # sort newest first by mtime
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)


def describe_zip(zp: Path) -> tuple[int, int]:
    """Return (file_count, total_uncompressed_size)."""
    with zipfile.ZipFile(zp, "r") as zf:
        infos = zf.infolist()
        total = sum(i.file_size for i in infos)
        count = sum(1 for i in infos if not i.is_dir())
        return count, total


def choose_backup_interactively(candidates: list[Path], base_dir: Path) -> Path:
    if not candidates:
        print("[error] No backup zip files found matching 'backup-root-*.zip' in repo root or ./backups", file=sys.stderr)
        sys.exit(1)

    default = candidates[0]

    print("[info] Found backup candidates:")
    for idx, p in enumerate(candidates[:10], start=1):
        count, total = describe_zip(p)
        dt = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        marker = "(default)" if p == default else ""
        # Print path relative to the base_dir where they live
        try:
            rel = p.relative_to(base_dir)
        except Exception:
            rel = p
        print(f"  {idx}. {rel} — {count} files, {human_size(total)}, {dt} {marker}")
    if len(candidates) > 10:
        print(f"  ... and {len(candidates) - 10} more")

    prompt = f"[prompt] Enter a number (1..{min(len(candidates),10)}) or path to a .zip, or press Enter for default: "
    choice = input(prompt).strip()

    if choice == "":
        return default

    # number selection
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= min(len(candidates), 10):
            return candidates[idx - 1]
        else:
            print(f"[error] Selection out of range.")
            sys.exit(2)

    # path selection
    path = Path(choice).expanduser().resolve()
    if not path.exists() or path.suffix.lower() != ".zip":
        print(f"[error] Provided path is not a .zip: {path}")
        sys.exit(3)
    return path


def restore_zip_to_root(zip_path: Path, root: Path) -> tuple[int, int]:
    """Extract zip into root, overwriting existing files. Returns (extracted_files, skipped_dirs)."""
    extracted = 0
    skipped_dirs = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            # zipfile.extract will overwrite by default for files
            if info.is_dir():
                skipped_dirs += 1
                continue
            target = root / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            zf.extract(info, root)
            extracted += 1
    return extracted, skipped_dirs


# ----------------------------
# Main
# ----------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Restore a project backup in-place from backup-root-<timestamp>.zip")
    parser.add_argument("--zip", dest="zip_path", help="Path to backup zip to restore. If omitted, you will be prompted.")
    parser.add_argument("--yes", "-y", action="store_true", help="Do not ask for confirmation before restoring.")
    parser.add_argument("--spec", dest="spec_path", help="Path to tools/spec/patch_logs_paths.json (to locate root-directory)")
    args = parser.parse_args(argv)

    # Determine spec file path (default next to this script under tools/spec/patch_logs_paths.json)
    spec_path = Path(args.spec_path).expanduser().resolve() if args.spec_path else default_spec_path()
    print(f"[info] Using spec file: {spec_path}")
    try:
        root = load_restore_root_from_spec(spec_path)
    except Exception as e:
        print(f"[error] {e}")
        return 2
    print(f"[info] Restore target root (from spec): {root}")

    # Prompt for restore targets (allows overrides)
    root = Path(_prompt_with_default("Restore ROOT directory to", str(root))).expanduser().resolve()
    logs_default = Path.home() / ".pcpt" / "log"
    logs_dir = Path(_prompt_with_default("Restore LOGS directory to", str(logs_default))).expanduser().resolve()

    # ---- ROOT restore ----
    if args.zip_path:
        zip_root = Path(args.zip_path).expanduser().resolve()
        if not zip_root.exists() or zip_root.suffix.lower() != ".zip":
            print(f"[error] Not a valid .zip: {zip_root}")
            return 3
    else:
        print("[1/5] Discovering ROOT backup candidates…")
        root_candidates = find_candidate_backups(root, "root")
        zip_root = choose_backup_interactively(root_candidates, root)

    print(f"[2/5] Selected ROOT backup: {zip_root}")
    try:
        file_count, total_size = describe_zip(zip_root)
    except zipfile.BadZipFile:
        print(f"[error] Corrupt or unreadable zip: {zip_root}")
        return 4

    print(f"[info] ROOT backup contains {file_count} files (~{human_size(total_size)} uncompressed).")
    print(f"[info] It will be restored into: {root}\n[warn] Existing files will be overwritten.")

    if not args.yes and not _confirm("Proceed with ROOT restore?", default=False):
        print("[info] ROOT restore cancelled.")
    else:
        print("[3/5] Restoring ROOT files…")
        extracted, skipped_dirs = restore_zip_to_root(zip_root, root)
        print(f"[done] ROOT: Restored {extracted} files (skipped {skipped_dirs} directories) from {zip_root.name}.")

    # ---- LOGS restore ----
    print("[4/5] Discovering LOG backup candidates…")
    log_candidates = find_candidate_backups(logs_dir, "log")
    if not log_candidates:
        print(f"[info] No log backups found under {logs_dir}. Skipping log restore.")
        return 0

    zip_log = choose_backup_interactively(log_candidates, logs_dir)
    print(f"[5/5] Selected LOG backup: {zip_log}")
    try:
        file_count_log, total_size_log = describe_zip(zip_log)
    except zipfile.BadZipFile:
        print(f"[error] Corrupt or unreadable zip: {zip_log}")
        return 4

    print(f"[info] LOG backup contains {file_count_log} files (~{human_size(total_size_log)} uncompressed).")
    print(f"[info] It will be restored into: {logs_dir}\n[warn] Existing files will be overwritten.")

    if not args.yes and not _confirm("Proceed with LOG restore?", default=False):
        print("[info] LOG restore cancelled.")
        return 0

    print("[5/5] Restoring LOG files…")
    extracted_log, skipped_dirs_log = restore_zip_to_root(zip_log, logs_dir)
    print(f"[done] LOG: Restored {extracted_log} files (skipped {skipped_dirs_log} directories) from {zip_log.name}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
