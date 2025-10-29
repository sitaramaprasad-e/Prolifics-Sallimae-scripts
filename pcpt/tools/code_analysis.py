import os
import collections
import re
import sys
import argparse

def load_filter_extensions(filter_name):
    if not filter_name:
        return None
    filter_path = os.path.expanduser(f"~/.pcpt/filters/{filter_name}")
    if os.path.exists(filter_path):
        with open(filter_path, "r") as f:
            return set(line.strip().upper() for line in f if line.strip())
    else:
        print(f"⚠️  Warning: Filter file {filter_path} not found. Proceeding without filtering.")
        return None

def should_include_file(file, filter_extensions):
    if not filter_extensions:
        return True
    ext = os.path.splitext(file)[1].upper()
    return ext in filter_extensions

def analyze_directory(directory, weight_factor, input_cost_per_million, output_cost_per_million, filter_extensions):
    file_counts = collections.defaultdict(int)
    line_counts = collections.defaultdict(int)
    complexity_counts = collections.defaultdict(int)
    token_counts = collections.defaultdict(int)
    char_counts = collections.defaultdict(int)
    complexity_ratios = collections.defaultdict(float)
    effort_estimates = collections.defaultdict(float)

    complexity_keywords = {"IF", "DO", "SELECT", "WHEN", "ELSE", "FOR", "WHILE", "SUBR", "PROC", "CALLP"}

    for root, _, files in os.walk(directory):
        for file in files:
            if not should_include_file(file, filter_extensions):
                continue

            file_ext = os.path.splitext(file)[1].upper()
            file_path = os.path.join(root, file)

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    file_counts[file_ext] += 1
                    total_lines = len(lines)
                    line_counts[file_ext] += total_lines

                    complexity = sum(1 for line in lines if any(kw in line for kw in complexity_keywords))
                    complexity_counts[file_ext] += complexity

                    estimated_tokens = sum(len(line.split()) for line in lines)
                    token_counts[file_ext] += estimated_tokens

                    char_count = sum(len(line) for line in lines)
                    char_counts[file_ext] += char_count

                    complexity_ratios[file_ext] = complexity_counts[file_ext] / line_counts[file_ext] if line_counts[file_ext] > 0 else 0

                    effort_estimates[file_ext] = (line_counts[file_ext] * complexity_ratios[file_ext] * weight_factor) / 100
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

    return file_counts, line_counts, complexity_counts, token_counts, char_counts, complexity_ratios, effort_estimates

def estimate_llm_cost(total_tokens, input_cost_per_million, output_cost_per_million):
    scale_factor = total_tokens / 1872
    reports = {
        "Analyze": {"input": 5287, "output": 791},
        "Component Model": {"input": 5961, "output": 604},
        "UX Model": {"input": 6053, "output": 439},
        "Domain Model": {"input": 6197, "output": 773},
        "Use Cases and Scenarios": {"input": 6719, "output": 1731},
        "Code Review": {"input": 5280, "output": 1041},
        "Business Rules": {"input": 5320, "output": 1728},
    }

    fudge_factor_for_cost = 0.91566265

    total_cost = 0
    cost_estimates = {}
    for report, tokens in reports.items():
        est_input_tokens = tokens["input"] * scale_factor / fudge_factor_for_cost
        est_output_tokens = tokens["output"] * scale_factor / fudge_factor_for_cost
        total_tokens = est_input_tokens + est_output_tokens
        cost = (est_input_tokens / 1_000_000) * input_cost_per_million + (est_output_tokens / 1_000_000) * output_cost_per_million
        cost_estimates[report] = {"input_tokens": est_input_tokens, "output_tokens": est_output_tokens, "total_tokens": total_tokens, "cost": cost}
        total_cost += cost
    return cost_estimates, total_cost

def print_results(file_counts, line_counts, complexity_counts, token_counts, char_counts, complexity_ratios, effort_estimates, input_cost_per_million, output_cost_per_million):
    total_tokens = sum(token_counts.values())
    llm_costs, total_cost = estimate_llm_cost(total_tokens, input_cost_per_million, output_cost_per_million)

    print("\nEstimate of LLM Processing Costs:")
    print("---------------------------------")
    print(f"Total Tokens Based On Files Contents: {total_tokens:,}")
    for report_type, data in llm_costs.items():
        print(f"{report_type} Report Estimated Running Cost: Input Tokens = {data['input_tokens']:.0f}, Output Tokens = {data['output_tokens']:.0f}, Total Tokens = {data['total_tokens']:.0f}, Cost = ${data['cost']:.2f}")
    print("\nEstimated Total Cost of Running All Reports: ${:.2f}".format(total_cost))

    print("\PCPT Code File Statistics:")
    print("--------------------------")
    print(f"{'File Type':<10}{'Count':<10}{'Lines':<15}{'Complexity':<15}{'Tokens (Est)':<15}{'Characters':<15}{'Complexity Ratio':<20}{'Effort (Hours)':<15}")
    print("-" * 120)

    total_files = total_lines = total_complexity = total_tokens = total_chars = total_effort = 0

    for file_type in sorted(file_counts.keys()):
        total_files += file_counts[file_type]
        total_lines += line_counts[file_type]
        total_complexity += complexity_counts[file_type]
        total_tokens += token_counts[file_type]
        total_chars += char_counts[file_type]
        total_effort += effort_estimates[file_type]

        print(f"{file_type:<10}{file_counts[file_type]:<10,}{line_counts[file_type]:<15,}{complexity_counts[file_type]:<15,}{token_counts[file_type]:<15,}{char_counts[file_type]:<15,}{complexity_ratios[file_type]:<20.5f}{effort_estimates[file_type]:<15,.2f}")

    total_complexity_ratio = total_complexity / total_lines if total_lines > 0 else 0
    print("-" * 120)
    print(f"{'TOTAL':<10}{total_files:<10,}{total_lines:<15,}{total_complexity:<15,}{total_tokens:<15,}{total_chars:<15,}{total_complexity_ratio:<20.5f}{total_effort:<15,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=None, help="Directory to analyze")
    parser.add_argument("--filter", type=str, default=None, help="Name of filter file in ~/.pcpt/filters")

    args = parser.parse_args()

    directory = args.directory or input("Enter AS400 code directory to analyze: ")
    filter_extensions = load_filter_extensions(args.filter)

    print("\nEffort Calculation: This tool estimates effort based on a custom multiplier.")
    print("Enter a multiplier (number of hours required per 100 lines of code, based on complexity ratio). Default is 30 if left blank:")

    while True:
        try:
            weight_factor_input = input("Enter weight factor: ")
            weight_factor = float(weight_factor_input) if weight_factor_input.strip() else 30.0
            if weight_factor <= 0:
                print("Please enter a positive number.")
            else:
                break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    input_cost_per_million = 2.50
    output_cost_per_million = 10.00

    file_counts, line_counts, complexity_counts, token_counts, char_counts, complexity_ratios, effort_estimates = analyze_directory(
        directory, weight_factor, input_cost_per_million, output_cost_per_million, filter_extensions
    )
    print_results(file_counts, line_counts, complexity_counts, token_counts, char_counts, complexity_ratios, effort_estimates, input_cost_per_million, output_cost_per_million)