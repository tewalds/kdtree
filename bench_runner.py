#!/usr/bin/env python3

import json
import os
import re
import subprocess
import sys

# Standard benchmark sizes
SIZES = [1000, 10000, 100000, 1000000, 10000000]
BASELINE_FILE = "bench_results.json"

def run_command(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\nError running {' '.join(cmd)}:")
        print(result.stderr)
        sys.exit(1)
    return result.stdout

def build_bench():
    print("Building C++ benchmark...", end="", flush=True)
    run_command(["clang++", "-std=c++20", "-O3", "-I.", "bench.cpp", "-o", "bench"])
    print(" Done.")

def get_results(script_or_bin, sizes, label):
    print(f"Running {label}...", end="", flush=True)
    if script_or_bin.endswith(".py"):
        output = run_command(["python3", script_or_bin] + [str(s) for s in sizes])
    else:
        output = run_command(["./" + script_or_bin] + [str(s) for s in sizes])
    print(" Done.")
    return json.loads(output)

def strip_ansi(text):
    return re.sub(r'\033\[[0-9;]*m', '', text)

def format_cell(text, width, align="right"):
    visible_len = len(strip_ansi(text))
    padding = " " * (width - visible_len)
    if align == "left":
        return text + padding
    return padding + text

def format_row(n, build_curr, build_base, query_curr, query_base, label):
    def get_diff(curr, base):
        if base == 0: return "+0.0%"
        diff = (curr - base) / base * 100
        color = "\033[91m" if diff > 5 else "\033[92m" if diff < -5 else ""
        reset = "\033[0m" if color else ""
        return f"{color}{diff:+.1f}%{reset}"

    cols = [
        format_cell(str(n), 10),
        format_cell(f"{build_curr:.2f}", 10),
        format_cell(get_diff(build_curr, build_base), 8),
        format_cell(f"{query_curr:.2f}", 10),
        format_cell(get_diff(query_curr, query_base), 8),
        format_cell(label, 12, "left")
    ]
    return "| " + " | ".join(cols) + " |"

def main():
    update_baseline = "--update" in sys.argv
    custom_sizes = [int(arg) for arg in sys.argv[1:] if arg.isdigit()]
    sizes = custom_sizes if custom_sizes else SIZES

    build_bench()
    cpp_results = get_results("bench", sizes, "C++ Core")
    py_results = get_results("bench.py", sizes, "Python Bindings")
    scipy_results = get_results("bench_scipy.py", sizes, "SciPy")

    baseline = {}
    if os.path.exists(BASELINE_FILE):
        with open(BASELINE_FILE, "r") as f:
            baseline = json.load(f)

    print("\n" + "="*100)
    print(f"| {'N':^10} | {'Build (ms)':^10} | {'B. Diff':^8} | {'Query (us)':^10} | {'Q. Diff':^8} | {'Implementation':^12} |")
    print("-" * 100)

    new_baseline = {"cpp": cpp_results, "py": py_results, "scipy": scipy_results}

    for i, n in enumerate(sizes):
        def find_base(base_results, n_val):
            if not base_results: return None
            for r in base_results.get("results", []):
                if r["n"] == n_val: return r
            return None

        # C++ Comparison
        c_curr = cpp_results["results"][i]
        c_base = find_base(baseline.get("cpp"), n)
        cb_base = c_base["build_ms"] if c_base else c_curr["build_ms"]
        cq_base = c_base["query_us"] if c_base else c_curr["query_us"]
        print(format_row(n, c_curr["build_ms"], cb_base, c_curr["query_us"], cq_base, "This (C++)"))

        # Python Bindings Comparison
        p_curr = py_results["results"][i]
        p_base = find_base(baseline.get("py"), n)
        pb_base = p_base["build_ms"] if p_base else p_curr["build_ms"]
        pq_base = p_base["query_us"] if p_base else p_curr["query_us"]
        print(format_row(n, p_curr["build_ms"], pb_base, p_curr["query_us"], pq_base, "This (Py)"))

        # SciPy Current
        s_curr = scipy_results["results"][i]
        print(format_row(n, s_curr["build_ms"], s_curr["build_ms"], s_curr["query_us"], s_curr["query_us"], "SciPy"))
        print("-" * 100)

    if update_baseline:
        with open(BASELINE_FILE, "w") as f:
            json.dump(new_baseline, f, indent=2)
        print(f"\nBaseline updated in {BASELINE_FILE}")
    elif not os.path.exists(BASELINE_FILE):
        print(f"\nNo baseline found. Run with --update to save these results as baseline.")

if __name__ == "__main__":
    main()
