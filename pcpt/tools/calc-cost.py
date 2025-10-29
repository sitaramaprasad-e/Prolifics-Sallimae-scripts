#!/usr/bin/env python3
import os
import re
import sys
import argparse
import configparser
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Optional

# --- Defaults & Constants ---
DEFAULT_LOG_DIR = os.path.expanduser("~/.pcpt/log")
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.pcpt/config/pcpt.config")

TOKEN_PATTERN = r"Tokens Used:\s*([\d,]+)\s*=\s*Prompt Tokens:\s*([\d,]+)\s*\+\s*Generated Tokens:\s*([\d,]+)"
LOG_TIMESTAMP_PATTERN = r"log_(\d{4}-\d{2}-\d{2})_(\d{2})-\d{2}-\d{2}\.txt"

# Token pricing per 1,000 tokens (USD)
PRICING = {
    "gpt-4":   {"prompt": 0.03,  "completion": 0.06},
    "gpt-4.1": {"prompt": 0.01,  "completion": 0.03},
    "gpt-4o":  {"prompt": 0.005, "completion": 0.015},
    "gpt-3.5": {"prompt": 0.001, "completion": 0.002},
    # Add more here as you need, e.g.:
    # "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
}

# --- Helpers ---

def debug(msg: str):
    # flip to True if you want extra logging
    if False:
        print(f"[DEBUG] {msg}", file=sys.stderr)

def normalize_model(model_raw: Optional[str]) -> Optional[str]:
    """Map various model names/aliases to PRICING keys."""
    if not model_raw:
        return None
    m = model_raw.lower().strip()

    # Common normalizations
    if m.startswith("gpt-4.1"):
        return "gpt-4.1"
    if m.startswith("gpt-4o"):
        return "gpt-4o"
    if m.startswith("gpt-4"):
        return "gpt-4"
    if m.startswith("gpt-3.5"):
        return "gpt-3.5"
    return m

def get_model_name(config_path: str) -> Optional[str]:
    """
    Try, in order:
      1) Environment variables: OPENAI_MODEL / AZUREAI_MODEL / MODEL_NAME
      2) Config file at config_path: DEFAULT and any sections, keys tried:
         openai_model / azureai_model / model / default_model
    Returns None if nothing is found.
    """
    # Env overrides
    env_model = os.getenv("OPENAI_MODEL") or os.getenv("AZUREAI_MODEL") or os.getenv("MODEL_NAME")
    if env_model:
        return env_model.strip()

    # Config file
    cfg = configparser.ConfigParser()
    read_files = cfg.read(config_path)
    if not read_files:
        debug(f"No config read from {config_path}")
        return None

    # DEFAULT section
    for key in ("openai_model", "azureai_model", "model", "default_model"):
        val = cfg["DEFAULT"].get(key) if "DEFAULT" in cfg else None
        if val:
            return val.strip()

    # Any other sections
    for section in cfg.sections():
        for key in ("openai_model", "azureai_model", "model", "default_model"):
            val = cfg[section].get(key, fallback=None)
            if val:
                return val.strip()

    return None

def parse_tokens_from_log(log_path: Path) -> tuple[Optional[int], Optional[int]]:
    """
    Returns (prompt_tokens, completion_tokens) or (None, None) if not found.
    Looks from the end of the file upward to find the most recent line with token info.
    """
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"⚠️  Skipping {log_path.name}: Failed to read ({e})")
        return None, None

    for line in reversed(lines):
        match = re.search(TOKEN_PATTERN, line)
        if match:
            try:
                total, prompt, completion = [int(val.replace(",", "")) for val in match.groups()]
                # we trust prompt/completion; 'total' is not needed
                return prompt, completion
            except ValueError:
                # malformed numbers
                return None, None
    return None, None

def extract_datetime_from_filename(filename: str) -> tuple[Optional[str], Optional[int]]:
    """
    Extracts (date_str 'YYYY-MM-DD', hour int 0-23) from filenames like:
    log_2025-08-21_10-05-12.txt
    """
    match = re.search(LOG_TIMESTAMP_PATTERN, filename)
    if match:
        date_str, hour_str = match.groups()
        return date_str, int(hour_str)
    return None, None

def calculate_cost(prompt_tokens: int, completion_tokens: int, model_key: str) -> float:
    if model_key not in PRICING:
        raise ValueError(f"Unknown pricing for model: {model_key}")
    rate = PRICING[model_key]
    return ((prompt_tokens / 1000.0) * rate["prompt"]) + ((completion_tokens / 1000.0) * rate["completion"])

def iter_log_files(log_dir: Path):
    if not log_dir.exists():
        print(f"⚠️  Log directory does not exist: {log_dir}")
        return []
    return sorted(log_dir.glob("log_*.txt"))

# --- CLI ---

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Estimate token costs from PCPT logs.")
    p.add_argument(
        "--model",
        help="Override model name (e.g., gpt-4o). Takes precedence over env/config.",
        default=None,
    )
    p.add_argument(
        "--log-dir",
        help=f"Directory containing log_*.txt (default: {DEFAULT_LOG_DIR})",
        default=DEFAULT_LOG_DIR,
    )
    p.add_argument(
        "--config",
        help=f"Config file path to read model from (default: {DEFAULT_CONFIG_PATH})",
        default=DEFAULT_CONFIG_PATH,
    )
    return p

# --- Main ---

def main():
    args = build_arg_parser().parse_args()

    # Determine model
    raw_model = args.model or get_model_name(args.config)
    model = normalize_model(raw_model)

    if model not in PRICING:
        if raw_model is None:
            print("[WARN] No model configured (env, config, or --model). Defaulting to 'gpt-4o'")
        else:
            print(f"[WARN] Could not determine pricing for model {repr(raw_model)}. Defaulting to 'gpt-4o'")
        model = "gpt-4o"

    print(f"📘 Using model: {model}")

    log_dir = Path(args.log_dir).expanduser()
    log_files = list(iter_log_files(log_dir))
    print(f"🔍 Found {len(log_files)} log files in {log_dir}")

    if not log_files:
        print("Nothing to do.")
        return

    per_day: dict[str, float] = defaultdict(float)
    per_hour: dict[tuple[str, int], float] = defaultdict(float)
    total_cost = 0.0
    counted = 0

    for log_file in log_files:
        date_str, hour = extract_datetime_from_filename(log_file.name)
        if date_str is None or hour is None:
            print(f"⚠️  Skipping {log_file.name}: No date/hour info")
            continue

        prompt, completion = parse_tokens_from_log(log_file)
        if prompt is None or completion is None:
            print(f"⚠️  Skipping {log_file.name}: No token data found")
            continue

        try:
            cost = calculate_cost(prompt, completion, model)
        except ValueError as e:
            print(f"⚠️  Skipping {log_file.name}: {e}")
            continue

        total_cost += cost
        per_day[date_str] += cost
        per_hour[(date_str, hour)] += cost
        counted += 1

        print(f"{log_file.name}: Prompt={prompt}, Completion={completion}, Cost=${cost:.4f}")

    # Summaries
    print("\n📅 Cost per Day:")
    for date in sorted(per_day):
        print(f"  {date}: ${per_day[date]:.4f}")

    print("\n🕒 Cost per Hour:")
    for (date, hour) in sorted(per_hour):
        print(f"  {date} {hour:02d}:00 - {hour:02d}:59 → ${per_hour[(date, hour)]:.4f}")

    print(f"\n🧮 Files counted: {counted}/{len(log_files)}")
    print(f"💰 Total Estimated Cost: ${total_cost:.4f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)