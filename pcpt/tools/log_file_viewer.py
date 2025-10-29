#!/usr/bin/env python3

import os

LOG_DIR = "/Users/greghodgkinson/.pcpt/log"
LINES_PER_PAGE = 500

def get_sorted_log_files():
    files = [f for f in os.listdir(LOG_DIR) if os.path.isfile(os.path.join(LOG_DIR, f))]
    return sorted(files, key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)), reverse=True)

def read_lines(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()

def show_page(lines, page_num, file_name):
    total_pages = (len(lines) + LINES_PER_PAGE - 1) // LINES_PER_PAGE
    start = page_num * LINES_PER_PAGE
    end = start + LINES_PER_PAGE
    header = f"--- FILE: {file_name} | PAGE: {page_num + 1} / {total_pages} ---"
    print(f"\n{header}\n")
    print("".join(lines[start:end]))
    print(f"\n{header}\n")  # Footer

def main():
    files = get_sorted_log_files()
    if not files:
        print("No log files found.")
        return

    file_index = 0
    page_num = 0

    while True:
        file_path = os.path.join(LOG_DIR, files[file_index])
        lines = read_lines(file_path)
        show_page(lines, page_num, files[file_index])

        cmd = input("[n]ext page | [p]revious page | [f]orward (newer file) | [b]ack (older file) | [q]uit: ").strip().lower()

        if cmd == 'q':
            break
        elif cmd == 'n':
            if (page_num + 1) * LINES_PER_PAGE < len(lines):
                page_num += 1
            else:
                print("End of file.")
        elif cmd == 'p':
            if page_num > 0:
                page_num -= 1
            else:
                print("Already at start of file.")
        elif cmd == 'f':
            if file_index > 0:
                file_index -= 1
                page_num = 0
            else:
                print("No newer files.")
        elif cmd == 'b':
            if file_index + 1 < len(files):
                file_index += 1
                page_num = 0
            else:
                print("No older files.")
        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()