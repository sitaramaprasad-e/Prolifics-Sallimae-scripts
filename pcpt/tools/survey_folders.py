import os
import argparse
from pathlib import Path
from collections import defaultdict

def load_filter(filter_name):
    if not filter_name:
        return None

    filter_path = os.path.expanduser(f"~/.pcpt/filters/{filter_name}")
    if not os.path.exists(filter_path):
        print(f"⚠️  Filter file not found: {filter_path}. Proceeding without filter.")
        return None

    with open(filter_path, 'r') as f:
        extensions = {
            line.strip().upper() if line.strip().startswith('.') else '.' + line.strip().upper()
            for line in f if line.strip()
        }
    return extensions

def scan_folder(path, allowed_extensions=None, indent=0):
    folder_summary = defaultdict(int)
    entries = sorted(os.listdir(path))
    prefix = '    ' * indent + '📁 ' + os.path.basename(path) + '/'
    print(prefix)

    for entry in entries:
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            sub_summary = scan_folder(full_path, allowed_extensions, indent + 1)
            for ext, count in sub_summary.items():
                folder_summary[ext] += count
        else:
            ext = Path(entry).suffix.upper()
            if allowed_extensions and ext not in allowed_extensions:
                continue
            folder_summary[ext] += 1

    if folder_summary:
        for ext, count in sorted(folder_summary.items()):
            print('    ' * (indent + 1) + f'- {ext}: {count} file(s)')

    return folder_summary

def main():
    parser = argparse.ArgumentParser(description="Scan folders and summarize file types.")
    parser.add_argument("directory", help="Root directory to scan")
    parser.add_argument("--filter", type=str, help="Filter name from ~/.pcpt/filters (e.g., blazor.filter)", default=None)

    args = parser.parse_args()

    root_dir = os.path.abspath(args.directory)
    if not os.path.exists(root_dir):
        print(f"❌ Directory not found: {root_dir}")
        return

    allowed_extensions = load_filter(args.filter)
    print(f"🔍 Scanning: {root_dir}")
    if allowed_extensions:
        print(f"🧰 Filter applied: {', '.join(sorted(allowed_extensions))}")
    else:
        print("🛠️  No filter applied — all file types included.")

    scan_folder(root_dir, allowed_extensions)

if __name__ == "__main__":
    main()