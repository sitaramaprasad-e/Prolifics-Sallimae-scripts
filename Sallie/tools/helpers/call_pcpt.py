#!/usr/bin/env python3
"""
call_pcpt.py

Reusable helpers to invoke `pcpt.sh run-custom-prompt` and manage
its output files, without leaking any CLI details to caller scripts.

These functions are intentionally generic and accept the output
directory, output file, and prompt name as parameters so that
callers (e.g., categorize_logics.py) retain full control over where
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
      output_file_arg="categorise-logic/categorise-logic.md"
      -> ("docs/categorise-logic", "categorise-logic", ".md")
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
    visualize_arg: Optional[str] = None,
    domain_hints: Optional[str] = None,
    visualize: bool = False,
    filter_path: Optional[str] = None,
    source_path: Optional[str] = None,
    index: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    """
    Wrapper around:
      pcpt.sh sequence \
        --output <output_dir_arg> \
        [--domain-hints <domain_hints>] \
        [--visualize] \
        [--filter <filter_path>] \
        <source_path>

    Backward compatible behavior:
      - If `visualize_arg` is provided (any non-None value), `--visualize` is added.
      - New boolean `visualize` also controls `--visualize` when True.
      - `filter_path` and `source_path` are optional and omitted when not provided.
    """
    cmd = ["pcpt.sh", "sequence", "--output", output_dir_arg]
    if domain_hints:
        cmd.extend(["--domain-hints", domain_hints])

    # Honor both legacy visualize_arg and new boolean visualize switch
    if visualize or (visualize_arg is not None and str(visualize_arg).strip() != ""):
        cmd.append("--visualize")

    if filter_path:
        cmd.extend(["--filter", filter_path])

    if total is not None:
        cmd.append(f"--total={total}")
    if index is not None:
        cmd.append(f"--index={index}")

    if source_path:
        cmd.append(source_path)

    subprocess.run(cmd, check=True)

# Helper for calling pcpt.sh domain-model with optional --domain-hints
def pcpt_domain_model(
    output_dir_arg: str,
    domain_hints: Optional[str] = None,
    visualize: bool = False,
    filter_path: Optional[str] = None,
    mode: Optional[str] = None,
    source_path: str = None,
    index: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    """
    Wrapper around:
      pcpt.sh domain-model \
        --output <output_dir_arg> \
        [--domain-hints <domain_hints>] \
        [--visualize] \
        [--filter <filter_path>] \
        [--mode {multi,single}] \
        <source_path>
    If any optional values are None/False, their flags are omitted (backwards compatible).
    """
    cmd = ["pcpt.sh", "domain-model", "--output", output_dir_arg]
    if domain_hints:
        cmd.extend(["--domain-hints", domain_hints])
    if visualize:
        cmd.append("--visualize")
    if filter_path:
        cmd.extend(["--filter", filter_path])
    if mode:
        cmd.extend(["--mode", mode])

    if total is not None:
        cmd.append(f"--total={total}")
    if index is not None:
        cmd.append(f"--index={index}")

    if source_path:
        cmd.append(source_path)
    subprocess.run(cmd, check=True)

# Helper for calling pcpt.sh business-logic with optional --domain-hints

def pcpt_business_logic(
    output_dir_arg: str,
    domain_path: str,
    domain_hints: Optional[str] = None,
    filter_path: Optional[str] = None,
    mode: Optional[str] = None,
    source_path: str = None,
    index: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    """
    Wrapper around:
      pcpt.sh business-logic --output <output_dir_arg> --domain <domain_path> [--domain-hints <domain_hints>] [--filter <filter_path>] [--mode {multi,single}] <source_path>
    If domain_hints is None or empty, the flag is omitted (backwards compatible).
    """
    cmd = ["pcpt.sh", "business-logic", "--output", output_dir_arg]
    cmd.extend(["--domain", domain_path])
    if domain_hints:
        cmd.extend(["--domain-hints", domain_hints])
    if filter_path:
        cmd.extend(["--filter", filter_path])
    if mode:
        cmd.extend(["--mode", mode])

    if total is not None:
        cmd.append(f"--total={total}")
    if index is not None:
        cmd.append(f"--index={index}")

    if source_path:
        cmd.append(source_path)
    subprocess.run(cmd, check=True)

# Helper for calling pcpt.sh run-custom-prompt (generic wrapper)
def pcpt_run_custom_prompt(
    source_path: str,
    custom_prompt_template: str,
    domain_hints: Optional[str] = None,
    input_file: Optional[str] = None,
    input_file2: Optional[str] = None,
    echo_only: bool = False,
    output_dir_arg: Optional[str] = None,
    filter_path: Optional[str] = None,
    mode: Optional[str] = None,  # 'multi' (default) or 'single'
    index: Optional[int] = None,
    total: Optional[int] = None,
) -> None:
    """
    Wrapper around:
      pcpt.sh run-custom-prompt \
        [--input-file <input_file>] \
        [--input-file2 <input_file2>] \
        [--echo-only] \
        [--output <output_dir_arg>] \
        [--domain-hints <domain_hints>] \
        [--filter <filter_path>] \
        [--mode {multi,single}] \
        [--index X --total Y] \
        <source_path> <custom_prompt_template>

    Notes / Backward compatibility:
      - Any optional flag is omitted when its corresponding argument is None/False.
      - `mode` should be 'multi' or 'single' when provided.
      - `index` and `total` must be provided together (mirrors CLI requirement).
    """
    cmd = ["pcpt.sh", "run-custom-prompt"]

    if input_file:
        cmd.extend(["--input-file", input_file])
    if input_file2:
        cmd.extend(["--input-file2", input_file2])
    if echo_only:
        cmd.append("--echo-only")
    if output_dir_arg:
        cmd.extend(["--output", output_dir_arg])
    if domain_hints:
        cmd.extend(["--domain-hints", domain_hints])
    if filter_path:
        cmd.extend(["--filter", filter_path])
    if mode:
        cmd.extend(["--mode", mode])

    # Replace pair-based addition with new single-line flags
    if total is not None:
        cmd.append(f"--total={total}")
    if index is not None:
        cmd.append(f"--index={index}")

    # Positional arguments
    cmd.extend([source_path, custom_prompt_template])

    subprocess.run(cmd, check=True)

def run_pcpt_for_logic(
    dynamic_logic_file: str,
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
        --input-file2 <dynamic_logic_file> \
        --output <output_dir_arg> \
        [--index X --total Y] \
        <dynamic_logic_file> <prompt_name>

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
        dynamic_logic_file,
        "--output",
        output_dir_arg,
    ]
    # Replace pair-based addition with new single-line flags
    if total is not None:
        cmd.append(f"--total={total}")
    if index is not None:
        cmd.append(f"--index={index}")
    cmd.extend([dynamic_logic_file, prompt_name])

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