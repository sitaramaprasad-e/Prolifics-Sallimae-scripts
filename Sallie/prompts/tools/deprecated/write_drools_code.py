# tools/write_drools_code.py

import os
import json
import re

def _prompt_model_home() -> str:
    try:
        resp = input("Enter model home path (default='~/'):").strip()
    except EOFError:
        # Non-interactive (e.g., piped/cron) – fall back to default
        resp = ""
    if not resp:
        resp = "~/"
    return resp

MODEL_HOME = _prompt_model_home()

# Paths
input_file = f"~MODEL_HOME/.model/drools_rules.json"
output_dir = "code/drools"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load rules
with open(input_file, "r", encoding="utf-8") as f:
    rules = json.load(f)

# Utility to sanitize file names
def sanitize_filename(name):
    name = name.strip().lower()
    name = re.sub(r"[^\w\d\- ]+", "", name)  # remove special chars except dash/space
    name = re.sub(r"\s+", "_", name)  # replace spaces with underscore
    return name

# Track filenames to avoid overwriting if duplicates
filename_counts = {}

for rule in rules:
    rule_name = rule.get("rule_name", "unnamed_rule")
    drools_code = rule.get("drools_code", "").strip()

    if not drools_code:
        continue  # skip if no code

    base_filename = sanitize_filename(rule_name)
    count = filename_counts.get(base_filename, 0)
    filename_counts[base_filename] = count + 1

    # Add numeric suffix if necessary
    filename = f"{base_filename}.drl" if count == 0 else f"{base_filename}_{count}.drl"

    file_path = os.path.join(output_dir, filename)

    with open(file_path, "w", encoding="utf-8") as out_file:
        out_file.write(drools_code)

print(f"Wrote {len(filename_counts)} Drools rule files to {output_dir}")