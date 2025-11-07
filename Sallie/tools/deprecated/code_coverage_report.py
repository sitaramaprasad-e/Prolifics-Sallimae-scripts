import os
import re
from datetime import datetime
from collections import defaultdict

LOG_DIR = os.path.expanduser("~/.pcpt/log")
CODE_DIR = "code"
REPORT_PATH = "code_coverage_report.md"

def is_today_log(filename):
    # Support old: log_YYYY-MM-DD_HH-MM-SS.txt
    # Support new: log-YYYY-MM-DD_HH-MM-SS-build-<build>-<provider>-<model>.txt
    today = datetime.now().strftime("%Y-%m-%d")
    m = re.match(r"log[_-](\d{4}-\d{2}-\d{2})_", filename)
    return bool(m and m.group(1) == today)

def get_timestamp_from_filename(filename):
    # Old format: log_YYYY-MM-DD_HH-MM-SS.txt
    m_old = re.match(r"log_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})\.txt", filename)
    if m_old:
        return f"{m_old.group(1)} {m_old.group(2)}:{m_old.group(3)}:{m_old.group(4)}"

    # New format: log-YYYY-MM-DD_HH-MM-SS-build-<build>-<provider>-<model>.txt
    m_new = re.match(r"log-(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})-build-.*\.txt", filename)
    if m_new:
        return f"{m_new.group(1)} {m_new.group(2)}:{m_new.group(3)}:{m_new.group(4)}"

    return "?"

def classify_activity(instructions):
    text = instructions.lower()
    patterns = [
        ("Analyze - Domain Model", lambda t: "generate a semantically descriptive domain model" in t),
        ("Analyze - Use Cases", lambda t: (
            "generate a list of functional use cases" in t or
            "generate use cases" in t or
            "generate list of use cases" in t
        )),
        ("Analyze - Sequence Diagram", lambda t: "generate a sequence diagram" in t),
        ("Visualize - Domain Model", lambda t: "render it in plantuml" in t and "domain" in t),
        ("Visualize - Use Case", lambda t: "render it in plantuml" in t and "use case" in t),
        ("Visualize - Sequence Diagram", lambda t: "render it in plantuml" in t and "sequence" in t),
        ("Transform - To DMN", lambda t: "transforming rules descriptions into formal dmn" in t),
        ("Generate - Code (Drools)", lambda t: "implementing drools rules" in t),
        ("Analyze - Business Rules", lambda t: "finding business rules in code" in t),
    ]
    for label, condition in patterns:
        if condition(text):
            return label
    return "Other"

def extract_activity_info(log_path, log_file):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    instructions_match = re.search(r"INSTRUCTIONS\s*\n+([^\n][\s\S]*?)(?:\n\s*\n|\nHints|\Z)", content, re.IGNORECASE)
    instructions = instructions_match.group(1).strip().replace("\n", " ") if instructions_match else ""
    activity = classify_activity(instructions)

    token_match = re.search(r"\|\|\|\s*Tokens Used:\s*(\d+)", content)
    tokens_used = token_match.group(1) if token_match else "?"

    file_matches = re.findall(r"File:\s+([^\n\r]+)", content)
    stripped_files = sorted({
        f.replace("/source_path/", "").replace("/source_path", "").strip()
        for f in file_matches if f.strip()
    })

    # Timestamp: prefer filename; if unknown, fall back to header line `timestamp=YYYY-MM-DD_HH-MM-SS`
    timestamp = get_timestamp_from_filename(log_file)
    if timestamp == "?":
        m_hdr = re.search(r"timestamp=(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", content)
        if m_hdr:
            timestamp = f"{m_hdr.group(1)} {m_hdr.group(2)}:{m_hdr.group(3)}:{m_hdr.group(4)}"

    # Display log filename without leading prefix (supports old/new)
    display_log = re.sub(r"^log[_-]", "", log_file)

    return {
        "timestamp": timestamp,
        "activity": activity,
        "tokens_used": tokens_used,
        "files": stripped_files,
        "log_file": display_log,
    }

def scan_all_code_files():
    all_files = []
    for root, _, files in os.walk(CODE_DIR):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), CODE_DIR)
            all_files.append(rel_path.replace("\\", "/"))
    return sorted(set(all_files))

def generate_audit_report():
    try:
        today_logs = sorted([f for f in os.listdir(LOG_DIR) if f.endswith(".txt") and is_today_log(f)])
    except FileNotFoundError:
        print(f"[ERROR] Log directory '{LOG_DIR}' not found.")
        return

    rows = []
    file_activity_counts = defaultdict(lambda: defaultdict(int))
    file_activity_last_seen = defaultdict(lambda: defaultdict(str))
    covered_files_by_activity = defaultdict(set)

    for log_file in today_logs:
        try:
            log_path = os.path.join(LOG_DIR, log_file)
            activity_info = extract_activity_info(log_path, log_file)
            rows.append(activity_info)
            timestamp = activity_info["timestamp"]
            activity = activity_info["activity"]
            for file in activity_info["files"]:
                if file.strip():
                    file_activity_counts[file][activity] += 1
                    file_activity_last_seen[file][activity] = max(
                        file_activity_last_seen[file][activity], timestamp
                    )
                    covered_files_by_activity[activity].add(file)
        except Exception as e:
            print(f"[WARN] Could not parse {log_file}: {e}")

    all_code_files = scan_all_code_files()
    total_code_files = len(all_code_files)

    with open(REPORT_PATH, "w", encoding="utf-8") as out:
        out.write("# 📋 Code Coverage Audit Report (Today Only)\n")
        out.write(f"_Generated on {datetime.now().strftime('%Y-%m-%d')} — This report summarizes PCPT activities from today’s logs._\n\n")

        # ======= CODE COVERAGE % BY ACTIVITY =======
        out.write("## Code Coverage Percentages by Activity\n\n")
        out.write("| Activity                   | Files Covered | Total Files | % Coverage |\n")
        out.write("|----------------------------|----------------|--------------|------------|\n")

        for activity in sorted(covered_files_by_activity.keys()):
            covered = len(covered_files_by_activity[activity])
            percentage = (covered / total_code_files) * 100 if total_code_files else 0.0
            out.write(f"| {activity:<26} | {covered:<14} | {total_code_files:<12} | {percentage:>8.1f}% |\n")

        # ======= CODE COVERAGE SUMMARY (PER FILE) =======
        out.write("\n## Code Coverage Summary\n\n")
        out.write("| File Name                  | Activity                   | Last Seen        | # Occurrences |\n")
        out.write("|----------------------------|----------------------------|------------------|----------------|\n")

        if not file_activity_counts:
            out.write("| _No files found_           | -                          | -                | -              |\n")
        else:
            for file in sorted(file_activity_counts.keys()):
                if not file.strip():
                    continue
                activities = sorted(file_activity_counts[file].items(), key=lambda x: x[0])
                for i, (activity, count) in enumerate(activities):
                    last_seen = file_activity_last_seen[file][activity]
                    cleaned_file = file.strip("`")
                    file_cell = cleaned_file if i == 0 else ""
                    out.write(f"| {file_cell:<26} | {activity:<26} | {last_seen:<16} | {count:<14} |\n")

        # ======= FULL ACTIVITY LOG =======
        out.write("\n## Activity Log\n\n")
        out.write("| Timestamp           | Activity                     | Tokens Used | Files Covered | Log File |\n")
        out.write("|---------------------|------------------------------|-------------|----------------|----------|\n")

        if not rows:
            out.write("| _No activity found_ | - | - | - | - |\n")
        else:
            for row in rows:
                files = "<br>".join(f.strip("`") for f in row["files"]) if row["files"] else ""
                out.write(f"| {row['timestamp']} | {row['activity']:<28} | {row['tokens_used']} | {files} | {row['log_file']} |\n")

    print(f"[INFO] Audit report written to {REPORT_PATH}")

if __name__ == "__main__":
    generate_audit_report()