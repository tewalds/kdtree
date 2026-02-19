#!/usr/bin/env python3
import subprocess
import json
import os
import sys
import re

BASELINE_FILE = "bench_results.json"

def strip_ansi(text):
    return re.sub(r'\033\[[0-9;]*m', '', text)

def format_cell(text, width, align="right"):
    visible_len = len(strip_ansi(text))
    padding = " " * (width - visible_len)
    if align == "left":
        return text + padding
    return padding + text

def get_diff(curr, base):
    if base is None or base == 0: return "+0.0%"
    diff = (curr - base) / base * 100
    color = "\033[91m" if diff > 5 else "\033[92m" if diff < -5 else ""
    reset = "\033[0m" if color else ""
    return f"{color}{diff:+.1f}%{reset}"

def run_bench(path):
    print(f"Running {path}...", end="", flush=True)
    if path.endswith(".cpp"):
        bin_path = path.replace(".cpp", "_bin")
        subprocess.run(["clang++", "-std=c++20", "-O3", "-I.", path, "-o", bin_path], check=True, capture_output=True)
        cmd = ["./" + bin_path]
    else:
        cmd = ["python3", path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(" Done.")
    if result.returncode != 0:
        print(f"Error running {path}:\n{result.stderr}")
        return []
    try:
        return json.loads(result.stdout)
    except:
        print(f"Failed to parse JSON from {path}")
        return []

def main():
    update_baseline = "--update" in sys.argv
    bench_dir = "benchmarks"

    files = [os.path.join(bench_dir, f) for f in os.listdir(bench_dir)
             if f.endswith(".py") or f.endswith(".cpp")]
    files.sort()

    all_results = []
    for f in files:
        all_results.extend(run_bench(f))

    # Sort results by Test, then Impl, then N
    all_results.sort(key=lambda x: (x["test"], x["implementation"], x["n"]))

    baseline = []
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            baseline = json.load(f)

    def find_base(test, impl, n):
        for b in baseline:
            if b["test"] == test and b["implementation"] == impl and b["n"] == n:
                return b.get("time_ms")
        return None

    # Compute dynamic column widths
    max_test_len = max([len(res["test"]) for res in all_results] + [4]) # 4 for "Test"
    test_w = max_test_len + 2

    max_impl_len = max([len(res["implementation"]) for res in all_results] + [4]) # 4 for "Impl"
    impl_w = max_impl_len + 2

    table_width = test_w + impl_w + 10 + 10 + 8 + 10 + 18

    print("\n" + "="*table_width)
    header = f"| {format_cell('Test', test_w, 'left')} | {format_cell('Impl', impl_w, 'left')} | {'N':10} | {'Time (ms)':10} | {'Diff':8} | {'Iters':10} |"
    print(header)
    print("-" * table_width)

    for res in all_results:
        test = res["test"]
        impl = res["implementation"]
        n = res["n"]
        time_ms = res["time_ms"]
        iters = res.get("iters", 1)

        base_time = find_base(test, impl, n)
        diff_str = get_diff(time_ms, base_time)

        row = f"| {format_cell(test, test_w, 'left')} | {format_cell(impl, impl_w, 'left')} | {n:10} | {time_ms:10.2f} | {format_cell(diff_str, 8)} | {format_cell(str(iters), 10)} |"
        print(row)

    if update_baseline:
        with open(BASELINE_FILE, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nBaseline updated in {BASELINE_FILE}")

if __name__ == "__main__":
    main()
