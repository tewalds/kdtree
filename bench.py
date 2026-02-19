#!/usr/bin/env python3

import json
import sys
import time

import kdtree
import numpy as np

def run_bench(n):
    # Same seed as consistency
    np.random.seed(42)
    data = np.random.uniform(0, 1000, size=(n, 2))

    # Measure Build
    start_build = time.perf_counter()
    tree = kdtree.KDTreed(data, np.arange(n, dtype=np.int64))
    end_build = time.perf_counter()
    build_ms = (end_build - start_build) * 1000

    # Measure Queries
    num_queries = 1000
    queries = np.random.uniform(0, 1000, size=(num_queries, 2))

    start_query = time.perf_counter()
    for q in queries:
        res = tree.find_closest(q.tolist())
    end_query = time.perf_counter()
    query_us = ((end_query - start_query) * 1_000_000) / num_queries

    return {
        "n": n,
        "build_ms": build_ms,
        "query_us": query_us
    }

if __name__ == "__main__":
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    if len(sys.argv) > 1:
        sizes = [int(x) for x in sys.argv[1:]]

    results = []
    for n in sizes:
        results.append(run_bench(n))

    print(json.dumps({"results": results}, indent=2))
