from __future__ import annotations
KEEP_EXISTING = False
#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import time
import pty
import select
import shlex
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROMPTS_DIR = os.path.join(os.getcwd(), "prompts")

# ----------------------------
# Constants / Defaults
# ----------------------------
TMP_DIR = ".tmp/suggest_models"
RULES_EXPORT_FILE = os.path.join(TMP_DIR, "rules_export.json")
MODELS_INPUT_FILE = os.path.join(TMP_DIR, "models_input.json")

# ----------------------------
# Small helpers
# ----------------------------

def _utcnow() -> str:
    return (
        datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, data: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --- Lenient Markdown-aware JSON loader ---
def _strip_code_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if lines and lines[0].strip().startswith("```"):
        # find closing fence
        for i in range(1, len(lines)):
            if lines[i].strip().startswith("```"):
                return "\n".join(lines[1:i])
    return text


def _extract_json_substring(text: str) -> str | None:
    start_idx = None
    for i, ch in enumerate(text):
        if ch == '{' or ch == '[':
            start_idx = i
            opening = ch
            closing = '}' if ch == '{' else ']'
            break
    if start_idx is None:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(start_idx, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
            elif ch == opening:
                depth += 1
            elif ch == closing:
                depth -= 1
                if depth == 0:
                    return text[start_idx:j+1]
    return None


def _load_json_lenient(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()
    # Try direct JSON first
    try:
        return json.loads(raw)
    except Exception:
        pass
    # Strip markdown code fences
    stripped = _strip_code_fences(raw)
    try:
        return json.loads(stripped)
    except Exception:
        pass
    # Extract the first JSON object/array
    segment = _extract_json_substring(stripped)
    if segment is not None:
        return json.loads(segment)
    # Fall back to empty dict quietly
    return {}


def _log(msg: str) -> None:
    ts = _utcnow()
    line = f"[{ts}] {msg}"
    print(line)


# ─────────────────────────────
# Enhanced step logging & pretty print
# ─────────────────────────────
_STEP_NUM = 0

def _hr(char: str = "─", width: int = 60) -> str:
    return char * width


def _step(title: str, subtitle: str | None = None) -> None:
    global _STEP_NUM
    _STEP_NUM += 1
    print()
    print(_hr("═"))
    print(f"STEP {_STEP_NUM}: {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(_hr("─"))


def _prompt_model_home() -> str:
    try:
        resp = input("Enter model home path (default='~'): ").strip()
    except EOFError:
        resp = ""
    if not resp:
        resp = "~"
    return os.path.abspath(os.path.expanduser(resp))


def _list_prompts() -> list[str]:
    if not os.path.isdir(PROMPTS_DIR):
        return []
    items = []
    for f in os.listdir(PROMPTS_DIR):
        if not f.endswith(".templ"):
            continue
        if f.startswith("suggest-models"):
            items.append(f)
    return sorted(items)


def _select_prompt() -> str:
    prompts = _list_prompts()
    # Ensure default exists in the list as first item
    default_name = "suggest-models-func-groups.templ"
    if default_name in prompts:
        # Put default first
        prompts = [default_name] + [p for p in prompts if p != default_name]
    elif prompts:
        # leave as-is
        pass
    else:
        raise FileNotFoundError(f"No prompt files found in {PROMPTS_DIR}")

    print("\nAvailable prompts:")
    for idx, name in enumerate(prompts, start=1):
        dmark = " (default)" if name == default_name else ""
        print(f"  {idx}. {name}{dmark}")

    try:
        choice = input(f"Choose prompt [1]: ").strip()
    except EOFError:
        choice = ""

    selected = None
    if not choice:
        selected = prompts[0]
    else:
        if choice.isdigit():
            i = int(choice)
            if 1 <= i <= len(prompts):
                selected = prompts[i-1]
        # allow direct name entry
        if not selected and choice in prompts:
            selected = choice
    if not selected:
        selected = prompts[0]

    return selected


def _derive_output_from_prompt(prompt_filename: str) -> tuple[str, str]:
    base = os.path.splitext(prompt_filename)[0]  # e.g., suggest_models2
    kebab = base.replace("_", "-")            # e.g., suggest-models2
    pcpt_output_dir = "docs"                    # always pass just 'docs' to PCPT
    expected_md = os.path.join("docs", kebab, f"{kebab}.md")
    return pcpt_output_dir, expected_md



# Helper to detect archived rules
def _is_archived(rule: Dict[str, Any]) -> bool:
    val = rule.get("archived")
    if val is None:
        val = rule.get("is_archived")
    # Normalize strings like "true", "yes", "1"
    if isinstance(val, str):
        val = val.strip().lower() in {"true", "yes", "y", "1"}

    # Also support schemas where a generic 'flag' field denotes archived status
    flag_val = rule.get("flag")
    if isinstance(flag_val, str) and flag_val.strip().lower() == "archived":
        return True

    return bool(val)

def _export_rules(business_rules_path: str) -> int:
    data = _load_json(business_rules_path)
    if not isinstance(data, list):
        raise ValueError(f"{business_rules_path} must be a JSON array of rules")

    def pick(rule: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": rule.get("id"),
            "rule_name": rule.get("rule_name"),
            "rule_spec": rule.get("rule_spec"),
            "rule_category": rule.get("rule_category"),
            "owner": rule.get("owner"),
            "component": rule.get("component"),
            "code_file": rule.get("code_file"),
            "code_function": rule.get("code_function"),
            "dmn_inputs": rule.get("dmn_inputs"),
            "dmn_outputs": rule.get("dmn_outputs"),
        }

    # Filter out archived rules
    active_rules = [r for r in data if isinstance(r, dict) and not _is_archived(r)]
    archived_count = len([r for r in data if isinstance(r, dict) and _is_archived(r)])

    exported = [pick(r) for r in active_rules]
    _dump_json(RULES_EXPORT_FILE, {"rules": exported})
    _log(f"Exported {len(exported)} active rules (skipped {archived_count} archived)")
    return len(exported)


def _prepare_models_input(models_json_path: str) -> Dict[str, Any]:
    # If not keeping existing, always return empty models list
    if not KEEP_EXISTING:
        doc = {"models": []}
        _dump_json(MODELS_INPUT_FILE, doc)
        return doc

    # Otherwise load existing if present
    if os.path.exists(models_json_path):
        try:
            doc = _load_json(models_json_path)
        except Exception:
            doc = {"models": []}
    else:
        doc = {"models": []}
    if not isinstance(doc, dict):
        doc = {"models": []}
    if "models" not in doc or not isinstance(doc["models"], list):
        doc["models"] = []
    _dump_json(MODELS_INPUT_FILE, doc)
    return doc


def _run_pcpt(code_dir: str, input_file: str, output_dir: str, prompt_name: str) -> None:
    """Call pcpt.sh with a pseudo‑TTY so child processes flush output immediately.

    This greatly improves streaming for tools that buffer when stdout is not a TTY
    (e.g., Python default I/O, Docker/Podman wrappers, rich console libs).
    """
    cmd = [
        "pcpt.sh",
        "run-custom-prompt",
        "--input-file",
        input_file,
        "--output",
        output_dir,
        code_dir,
        prompt_name,
    ]

    # Pretty log with proper shell quoting
    _log("Running: " + " ".join(shlex.quote(c) for c in cmd))

    env = os.environ.copy()
    # Nudge child interpreters to be unbuffered/UTF-8
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    # Allocate a PTY so downstream tools behave as if attached to a real terminal
    master_fd, slave_fd = pty.openpty()

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            env=env,
            close_fds=True,
        )
        # We don't write to the child; close our dup of the slave end
        os.close(slave_fd)

        # Pump bytes from the PTY master to our stdout in near‑real time
        rc: Optional[int] = None
        while True:
            # Read with a tiny timeout to keep the UI snappy without busy‑waiting
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in r:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    chunk = b""
                if not chunk:
                    # EOF from child PTY
                    pass
                else:
                    # Write raw bytes to preserve ANSI art / carriage returns
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()

            rc = proc.poll()
            if rc is not None:
                # Drain any final bytes that arrived between poll and loop break
                while True:
                    r2, _, _ = select.select([master_fd], [], [], 0)
                    if master_fd not in r2:
                        break
                    try:
                        chunk2 = os.read(master_fd, 4096)
                    except OSError:
                        break
                    if not chunk2:
                        break
                    sys.stdout.buffer.write(chunk2)
                    sys.stdout.buffer.flush()
                break

        if rc != 0:
            raise RuntimeError(f"PCPT run failed with exit code {rc}")

    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass


def _normalize_models(payload: Any) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        if "suggestedModels" in payload and isinstance(payload["suggestedModels"], dict):
            inner = payload["suggestedModels"].get("models")
            if isinstance(inner, list):
                return [m for m in inner if isinstance(m, dict)]
        if "models" in payload and isinstance(payload["models"], list):
            return [m for m in payload["models"] if isinstance(m, dict)]
    if isinstance(payload, list):
        return [m for m in payload if isinstance(m, dict)]
    return []


def _merge_models(base_models: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {m.get("id"): i for i, m in enumerate(base_models) if isinstance(m, dict) and m.get("id")}
    by_name = {m.get("name"): i for i, m in enumerate(base_models) if isinstance(m, dict) and m.get("name")}

    for m in incoming:
        if not isinstance(m, dict):
            continue
        mid = m.get("id")
        mname = m.get("name")
        if mid and mid in by_id:
            base_models[by_id[mid]] = m
        elif mname and mname in by_name:
            base_models[by_name[mname]] = m
        else:
            base_models.append(m)
    return base_models


def main() -> None:
    _start = time.perf_counter()
    print("\n" + _hr("═"))
    print("Suggest Models • PCPT Orchestrator")
    print(_hr("═"))

    global KEEP_EXISTING
    if "--keep-existing" in sys.argv:
        KEEP_EXISTING = True
        print("[flag] Keeping existing models when suggesting new ones")
    else:
        KEEP_EXISTING = False
        print("[flag] NOT keeping existing models (default)")

    # Prompt for MODEL_HOME
    model_home = _prompt_model_home()
    model_dir = os.path.join(model_home, ".model")
    business_rules_path = os.path.join(model_dir, "business_rules.json")
    models_json_path = os.path.join(model_dir, "models.json")

    if not os.path.exists(business_rules_path):
        raise FileNotFoundError(f"Missing {business_rules_path}")

    _ensure_dir(TMP_DIR)

    _step("Setup", "Resolving directories and inputs")
    print(f"Model home         : {model_home}")
    print(f"Business rules JSON: {business_rules_path}")
    print(f"Existing models JSON: {models_json_path}")
    print(f"Prompts directory  : {PROMPTS_DIR}")

    _step("Export rules", f"→ {RULES_EXPORT_FILE}")
    rule_count = _export_rules(business_rules_path)
    print(f"Export complete. File     : {RULES_EXPORT_FILE}")

    _step("Prepare models input", f"→ {MODELS_INPUT_FILE}")
    existing_models_doc = _prepare_models_input(models_json_path)
    existing_count = len(existing_models_doc.get("models", []))
    if KEEP_EXISTING:
        print(f"Existing models included : {existing_count}")
    else:
        print(f"Existing models ignored  : {existing_count}")

    # Choose prompt and derive expected output locations
    prompt_name = _select_prompt()
    pcpt_output_dir, expected_md = _derive_output_from_prompt(prompt_name)

    _step("Select prompt", "Interactive selection complete")
    print(f"Chosen prompt       : {prompt_name}")
    print(f"Expected output     : {expected_md}")
    print(f"PCPT --output       : {pcpt_output_dir}")

    # Ensure base docs dir exists (PCPT will create the subfolder named after the prompt base)
    _ensure_dir(pcpt_output_dir)

    # Remove any stale expected files to avoid confusion
    for stale in (expected_md,):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:
                pass

    _step("Run PCPT", "Executing custom prompt")
    print("Command:")
    print("  pcpt.sh run-custom-prompt \\")
    print(f"    --input-file {MODELS_INPUT_FILE} \\")
    print(f"    --output {pcpt_output_dir} \\")
    print(f"    {RULES_EXPORT_FILE} \\")
    print(f"    {prompt_name}")
    _run_pcpt(RULES_EXPORT_FILE, MODELS_INPUT_FILE, pcpt_output_dir, prompt_name)

    # Determine which expected file was produced (MD only)
    produced_path = expected_md if os.path.exists(expected_md) else None
    if not produced_path:
        raise FileNotFoundError(
            f"Expected PCPT output not found. Checked: {expected_md}. Check prompt '{prompt_name}'."
        )

    _step("Read PCPT output", f"← {produced_path}")
    pcpt_payload = _load_json_lenient(produced_path)

    has_models_key = isinstance(pcpt_payload, dict) and isinstance(pcpt_payload.get("models"), list)
    if not has_models_key:
        return

    # We have models; normalize and persist
    suggested = _normalize_models(pcpt_payload)
    if not suggested:
        return

    _step("Merge models", f"Incoming: {len(suggested)}")
    merged = _merge_models(existing_models_doc.get("models", []), suggested)
    final_doc = {"models": merged}
    _dump_json(models_json_path, final_doc)

    print("\n════════ Suggest Models Summary ════════")
    print(f"Rules exported:           {rule_count}")
    print(f"Models suggested:         {len(suggested)}")
    print(f"Models file updated:      {models_json_path}")
    print(f"PCPT output (report):     {produced_path}")

    _elapsed = time.perf_counter() - _start
    print("\n" + _hr("═"))
    print(f"Done in {_elapsed:.2f}s • Success")
    print(_hr("═"))


if __name__ == "__main__":
    main()