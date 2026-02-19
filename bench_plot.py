#!/usr/bin/env python3
import json
import os
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

BASELINE_FILE = "bench_results.json"
PLOTS_DIR = "plots"

def main():
    if not os.path.exists(BASELINE_FILE):
        print(f"Error: {BASELINE_FILE} not found. Run bench_runner.py first.")
        return

    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    with open(BASELINE_FILE, "r") as f:
        data = json.load(f)

    # Group data by test
    tests = defaultdict(lambda: defaultdict(list))
    for entry in data:
        test_name = entry["test"]
        impl = entry["implementation"]
        n = entry["n"]
        time = entry["time_ms"]
        tests[test_name][impl].append((n, time))

    available_tests = sorted(tests.keys())

    # If no arg, plot everything. Otherwise, plot matches.
    query = sys.argv[1].lower() if len(sys.argv) > 1 else None

    if query:
        matches = [t for t in available_tests if query in t.lower()]
    else:
        matches = available_tests

    if not matches:
        print(f"No tests found matching '{query or 'all'}'")
        return

    print(f"Generating {len(matches)} plots in {PLOTS_DIR}/...")

    for test_name in matches:
        plt.figure(figsize=(10, 6))
        impls = tests[test_name]

        for impl, points in impls.items():
            # Sort by N for plotting
            points.sort()
            ns = [p[0] for p in points]
            times = [p[1] for p in points]
            plt.plot(ns, times, marker='o', label=impl)

        plt.title(f"Scaling: {test_name}")
        plt.xlabel("N (Number of points)")
        plt.ylabel("Time (ms)")
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.legend()

        safe_name = test_name.replace(':', '').replace(' ', '_').replace('(', '').replace(')', '').lower()
        filename = os.path.join(PLOTS_DIR, f"bench_{safe_name}.png")
        plt.savefig(filename)
        plt.close() # Free memory
        print(f"  - {test_name}")

    print("\nDone.")

if __name__ == "__main__":
    main()
