import os
import json
import subprocess
from hashlib import md5

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

# CONFIGURATION
INPUT_JSON = f"~MODEL_HOME/.model/correlated_business_rules.json"
OUTPUT_JSON = f"~MODEL_HOME/.model/drools_rules.json"
PROMPT_NAME = "generate-drools-rule.templ"
TMP_FOLDER = ".tmp/drools_generation"
CUSTOM_PROMPT_DIR = os.path.join(TMP_FOLDER, "custom_prompt")
OUTPUT_MD = os.path.join(CUSTOM_PROMPT_DIR, PROMPT_NAME.replace(".templ", ".md"))

# Ensure required folders exist
os.makedirs(CUSTOM_PROMPT_DIR, exist_ok=True)

# Load rule definitions
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    all_rules = json.load(f)

# De-duplicate
seen = set()
unique_rules = []
for rule in all_rules:
    key = (rule["rule_id"], md5(rule["code_block"].encode()).hexdigest())
    if key not in seen:
        seen.add(key)
        unique_rules.append(rule)

# Process rules
results = []
for i, rule in enumerate(unique_rules):
    input_text = f"""### Rule Name:
{rule['rule_name']}

### Rule Spec:
{rule['rule_spec']}

### Code Block:
{rule['code_block']}
"""

    input_file = os.path.join(CUSTOM_PROMPT_DIR, f"rule_{i}.input")
    output_file = os.path.join(CUSTOM_PROMPT_DIR, f"generate-drools-rule.md")

    # Write input for prompt
    with open(input_file, "w", encoding="utf-8") as f:
        f.write(input_text)

    # Run pcpt.sh
    try:
        subprocess.run([
            "pcpt.sh", "run-custom-prompt",
            "--output", TMP_FOLDER,
            input_file,
            PROMPT_NAME
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ pcpt.sh failed for rule: {rule['rule_name']} → {e}")
        continue

    # Read generated Drools code
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            drools_code = f.read()
    else:
        print(f"⚠️ Output missing for rule: {rule['rule_name']}")
        continue

    # Store result
    results.append({
        "rule_name": rule["rule_name"],
        "rule_category": rule["rule_category"],
        "business_area": rule["business_area"],
        "owner": rule["owner"],
        "drools_code": drools_code.strip()
    })

# Save final output
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"✅ Drools rules generated: {len(results)} written to {OUTPUT_JSON}")