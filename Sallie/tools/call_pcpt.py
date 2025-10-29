

#!/usr/bin/env python3
"""
call_pcpt.py

Reusable helpers to invoke `pcpt.sh run-custom-prompt` and manage
its output files, without leaking any CLI details to caller scripts.

These functions are intentionally generic and accept the output
directory, output file, and prompt name as parameters so that
callers (e.g., categorize_rules.py) retain full control over where
artifacts go without duplicating the invocation details.

Also includes a helper for calling `pcpt.sh sequence` with optional
`--domain-hints` support.

No side effects at import-time.
"""
from __future__ import annotations

import json
import os
import subprocess
import glob
from typing import Optional, Tuple


def _derive_output_parts(output_dir_arg: str, output_file_arg: str) -> Tuple[str, str, str]:
    """
    Returns (output_parent_dir, base_name, ext) derived from the caller's args.

    Example:
      output_dir_arg="docs"
      output_file_arg="categorise-rule/categorise-rule.md"
      -> ("docs/categorise-rule", "categorise-rule", ".md")
    """
    output_parent_dir = os.path.join(output_dir_arg, os.path.dirname(output_file_arg)) if os.path.dirname(output_file_arg) else output_dir_arg
    base_name, ext = os.path.splitext(os.path.basename(output_file_arg))
    return (output_parent_dir, base_name, ext)


def build_output_path(output_dir_arg: str, output_file_arg: str, index: Optional[int] = None, total: Optional[int] = None) -> str:
    """
    Matches pcpt run-custom-prompt filename rules used by the existing scripts:
    - Without index/total: <OUTPUT_PARENT_DIR>/<BASE_NAME>.md
    - With index/total:    <OUTPUT_PARENT_DIR>/<BASE_NAME>-XofY-.md
      (note the trailing '-' before the extension is intentional)
    """
    output_parent_dir, base_name, ext = _derive_output_parts(output_dir_arg, output_file_arg)
    os.makedirs(output_parent_dir, exist_ok=True)
    if index is not None and total is not None:
        return os.path.join(output_parent_dir, f"{base_name}-{index}of{total}-{ext}")
    return os.path.join(output_parent_dir, f"{base_name}{ext}")


def clean_previous_outputs(output_dir_arg: str, output_file_arg: str) -> None:
    """
    Remove prior outputs that match the unsuffixed and suffixed patterns.
    This ensures each run starts clean. Silent on errors.
    """
    output_parent_dir, base_name, ext = _derive_output_parts(output_dir_arg, output_file_arg)
    patterns = [
        os.path.join(output_parent_dir, f"{base_name}{ext}"),
        os.path.join(output_parent_dir, f"{base_name}-*of*-{ext}"),
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                pass



def _ensure_output_dirs(output_dir_arg: str, output_file_arg: str) -> str:
    """
    Ensure the output directory (including nested subfolders) exists.
    Returns the parent output directory path.
    """
    output_parent_dir, _, _ = _derive_output_parts(output_dir_arg, output_file_arg)
    os.makedirs(output_parent_dir, exist_ok=True)
    return output_parent_dir


# Helper for calling pcpt.sh sequence with optional --domain-hints
def pcpt_sequence(
    output_dir_arg: str,
    visualize_arg: str,
    domain_hints: Optional[str] = None,
) -> None:
    """
    Wrapper around:
      pcpt.sh sequence --output <output_dir_arg> [--domain-hints <domain_hints>] --visualize <visualize_arg>
    If domain_hints is None or empty, the flag is omitted (backwards compatible).
    """
    cmd = ["pcpt.sh", "sequence", "--output", output_dir_arg]
    if domain_hints:
        cmd.extend(["--domain-hints", domain_hints])
    cmd.extend(["--visualize", visualize_arg])
    subprocess.run(cmd, check=True)


def run_pcpt_for_rule(
    dynamic_rule_file: str,
    categories_path: str,
    output_dir_arg: str,
    output_file_arg: str,
    prompt_name: str,
    index: Optional[int] = None,
    total: Optional[int] = None,
):
    """
    Wrapper around:
      pcpt.sh run-custom-prompt \
        --input-file <categories_path> \
        --input-file2 <dynamic_rule_file> \
        --output <output_dir_arg> \
        [--index X --total Y] \
        <dynamic_rule_file> <prompt_name>

    Returns parsed JSON from the expected output file, or None on failure.
    """
    expected_output = build_output_path(output_dir_arg, output_file_arg, index=index, total=total)

    # Remove any stale output before we run
    if os.path.exists(expected_output):
        try:
            os.remove(expected_output)
        except OSError:
            pass

    # Ensure output directory exists
    _ensure_output_dirs(output_dir_arg, output_file_arg)

    cmd = [
        "pcpt.sh",
        "run-custom-prompt",
        "--input-file",
        categories_path,
        "--input-file2",
        dynamic_rule_file,
        "--output",
        output_dir_arg,
    ]
    if index is not None and total is not None:
        cmd.extend(["--index", str(index), "--total", str(total)])
    cmd.extend([dynamic_rule_file, prompt_name])

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        # Defer to caller for logging; mirror previous behavior of returning None
        print(f"❌ pcpt.sh failed: {e}")
        return None

    if not os.path.exists(expected_output):
        print("⚠️ Expected output not found at:", expected_output)
        return None

    try:
        with open(expected_output, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to parse JSON from {expected_output}: {e}")
        return None